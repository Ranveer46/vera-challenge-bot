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

# Free-tier Groq keys have a per-model tokens-per-minute cap, and it is NOT the
# same for every model (observed on this account: 8000 TPM for the two
# openai/gpt-oss-* models and for qwen/qwen3.8-27b, but 70000 TPM for
# groq/compound-mini). Each model's budget is independent, so pooling several
# gives a much larger combined ceiling than any single model alone.
#
# Pool order is a deliberate quality/determinism preference, not round-robin:
# 1. openai/gpt-oss-120b -- primary. Best-understood, plain chat model (no
#    autonomous tool use), extensively validated in this codebase.
# 2. qwen/qwen3.8-27b -- confirmed equally reliable on our hardest instruction-
#    following case (off-topic decline+redirect) and produces clean JSON with
#    no reasoning leakage into content.
# 3. groq/compound-mini -- large overflow capacity (70000 TPM). Also validated
#    on both the real composer prompt and the off-topic case, but it's an
#    agentic model that CAN invoke tools autonomously, which is a small
#    determinism/latency risk we'd rather not take as the default path.
# 4. openai/gpt-oss-20b -- last-resort overflow before the deterministic
#    template; demonstrated weaker multi-step instruction-following earlier.
#
# Explicitly excluded after testing: qwen/qwen3.6-27b (leaks <think>...</think>
# into the content field, breaking JSON parsing), allam-2-7b (defaults to
# Arabic responses even for English/Hindi-English prompts), and
# openai/gpt-oss-safeguard-20b (a moderation-tuned variant, not validated for
# this use case).
DEFAULT_MODEL_POOL = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "groq/compound-mini",
    "openai/gpt-oss-20b",
]
_POOL_ENV = os.environ.get("GROQ_MODEL_POOL")
_MODEL_POOL = [m.strip() for m in _POOL_ENV.split(",") if m.strip()] if _POOL_ENV else DEFAULT_MODEL_POOL

MODEL_TPM_LIMITS: dict[str, int] = {
    "openai/gpt-oss-120b": 8000,
    "openai/gpt-oss-20b": 8000,
    "qwen/qwen3.8-27b": 8000,
    "groq/compound-mini": 70000,
}
DEFAULT_TPM_LIMIT = 8000
TPM_SAFETY_FRACTION = 0.9  # stay under this fraction of each model's own cap

_usage_window: dict[str, list[tuple[float, int]]] = {m: [] for m in _MODEL_POOL}


def _model_tpm_limit(model: str) -> int:
    return int(MODEL_TPM_LIMITS.get(model, DEFAULT_TPM_LIMIT) * TPM_SAFETY_FRACTION)


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


def _has_headroom(model: str, estimated_tokens: int) -> bool:
    return _estimated_recent_usage(model) + estimated_tokens <= _model_tpm_limit(model)


def pick_model(estimated_tokens: int = 2500, exclude: str | None = None) -> str | None:
    """Walk the pool in preference order, returning the first model with estimated
    headroom under ITS OWN per-minute token budget. Returns None if every pooled
    model is near its limit (caller should fall back to the deterministic template)."""
    for candidate in _MODEL_POOL:
        if candidate == exclude:
            continue
        if _has_headroom(candidate, estimated_tokens):
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
    pin reasoning_effort="low" to keep that bounded. If no pooled model has
    estimated headroom under its own per-minute token cap, raises GroqError
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
        if resp.status_code == 429 and model is None:
            # retry against the next pooled model with headroom, not just one fixed "other"
            other = pick_model(estimated_prompt_tokens + max_tokens, exclude=chosen_model)
            if other:
                payload["model"] = other
                payload.pop("reasoning_effort", None)
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
