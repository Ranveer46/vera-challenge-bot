"""Thin async client for the Groq chat-completions API (OpenAI-compatible)."""

from __future__ import annotations

import os
import json
import logging
import time
from pathlib import Path

import httpx

logger = logging.getLogger("vera.groq")


def _load_dotenv_if_present() -> None:
    """Minimal local-dev convenience: if GROQ_API_KEY isn't already in the
    environment, read KEY=VALUE pairs from a .env file next to this module.
    Never overwrites a real env var (e.g. one set in Render's dashboard).
    Not used in production -- there GROQ_API_KEY is a real platform env var."""
    if os.environ.get("GROQ_API_KEY"):
        return
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as e:
        logger.warning("could not read .env: %s", e)


_load_dotenv_if_present()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = os.environ.get("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")


class GroqError(Exception):
    pass


def api_key() -> str | None:
    return os.environ.get("GROQ_API_KEY") or None


REASONING_MODELS_PREFIX = ("openai/gpt-oss",)

# Free-tier Groq keys have a per-model tokens-per-minute cap (observed: 8000 TPM
# for both openai/gpt-oss-120b and openai/gpt-oss-20b). Each is capped
# independently, so round-robining between two known-good models roughly
# doubles effective throughput under load. A small client-side tracker below
# avoids firing calls we already know will 429, converting straight to the
# deterministic fallback instead of wasting a round trip against the budget.
_MODEL_POOL = [DEFAULT_MODEL, FALLBACK_MODEL]
_usage_window: dict[str, list[tuple[float, int]]] = {m: [] for m in _MODEL_POOL}
TPM_SAFETY_LIMIT = 7200  # stay under the observed 8000 cap


def _record_usage(model: str, tokens: int) -> None:
    now = time.time()
    bucket = _usage_window.setdefault(model, [])
    bucket.append((now, tokens))
    cutoff = now - 60
    _usage_window[model] = [(t, n) for t, n in bucket if t >= cutoff]


def _estimated_recent_usage(model: str) -> int:
    now = time.time()
    cutoff = now - 60
    return sum(n for t, n in _usage_window.get(model, []) if t >= cutoff)


def pick_model(estimated_tokens: int = 2500) -> str | None:
    """Prefer the primary (higher-quality) model; only spill over to the secondary
    when the primary is genuinely near its per-minute token budget. Blind round-robin
    was rejected here: the secondary model follows multi-step behavioral instructions
    (e.g. off-topic redirects) noticeably less reliably, so it should be overflow
    capacity, not an equal partner. Returns None if both are near their limit
    (caller should fall back to the deterministic template)."""
    for candidate in _MODEL_POOL:
        if _estimated_recent_usage(candidate) + estimated_tokens <= TPM_SAFETY_LIMIT:
            return candidate
    return None


async def complete(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 750,
    timeout: float = 15.0,
    client: httpx.AsyncClient | None = None,
    reasoning_effort: str = "low",
    estimated_prompt_tokens: int = 1500,
) -> str:
    """Call Groq chat completions. Raises GroqError on failure.

    Note: openai/gpt-oss-* models on Groq are reasoning models that spend part
    of `max_tokens` on a hidden chain-of-thought before the actual answer. We
    pin reasoning_effort="low" to keep that bounded. If neither pooled model
    has estimated headroom under the per-minute token cap, raises GroqError
    immediately so the caller can use its deterministic fallback rather than
    waiting out a guaranteed 429.
    """
    key = api_key()
    if not key:
        raise GroqError("GROQ_API_KEY not set")

    if model is None:
        chosen_model = pick_model(estimated_prompt_tokens + max_tokens)
        if chosen_model is None:
            raise GroqError("all pooled models near TPM limit; skipping call to avoid a guaranteed 429")
    else:
        chosen_model = model

    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if chosen_model.startswith(REASONING_MODELS_PREFIX):
        payload["reasoning_effort"] = reasoning_effort

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        resp = await client.post(GROQ_URL, json=payload, headers=headers)
        if resp.status_code == 429:
            # one retry against the OTHER pooled model, if it has headroom
            other = next((m for m in _MODEL_POOL if m != chosen_model), None)
            if other and model is None and _estimated_recent_usage(other) + estimated_prompt_tokens + max_tokens <= TPM_SAFETY_LIMIT:
                payload["model"] = other
                if other.startswith(REASONING_MODELS_PREFIX):
                    payload["reasoning_effort"] = reasoning_effort
                chosen_model = other
                resp = await client.post(GROQ_URL, json=payload, headers=headers)
        if resp.status_code == 400 and model is None:
            payload["model"] = FALLBACK_MODEL
            if FALLBACK_MODEL.startswith(REASONING_MODELS_PREFIX):
                payload["reasoning_effort"] = reasoning_effort
            chosen_model = FALLBACK_MODEL
            resp = await client.post(GROQ_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        _record_usage(chosen_model, usage.get("total_tokens", estimated_prompt_tokens + max_tokens))
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        finish_reason = data["choices"][0].get("finish_reason")
        if not content.strip() and finish_reason == "length":
            raise GroqError("truncated before content was emitted (finish_reason=length); increase max_tokens")
        return content
    except httpx.TimeoutException as e:
        raise GroqError(f"timeout: {e}") from e
    except httpx.HTTPStatusError as e:
        raise GroqError(f"http {e.response.status_code}: {e.response.text[:300]}") from e
    except GroqError:
        raise
    except Exception as e:  # noqa: BLE001
        raise GroqError(str(e)) from e
    finally:
        if owns_client:
            await client.aclose()


def extract_json(text: str) -> dict | None:
    """Pull the first top-level {...} block out of an LLM response and parse it."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # try trimming trailing commas, a common small-model mistake
        cleaned = candidate.replace(",\n}", "\n}").replace(", }", " }")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("failed to parse JSON from LLM output: %s", candidate[:200])
            return None
