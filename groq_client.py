"""Thin async client for the Groq chat-completions API (OpenAI-compatible)."""

from __future__ import annotations

import asyncio
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


class GroqError(Exception):
    pass


def api_key() -> str | None:
    return os.environ.get("GROQ_API_KEY") or None


REASONING_MODELS_PREFIX = ("openai/gpt-oss",)

# Free-tier Groq keys have a per-model tokens-per-minute cap (observed: 8000 TPM
# for each model below). Pooling several INDEPENDENT models gives a larger
# combined ceiling than any single model alone -- "independent" is the key
# word, verified the hard way: groq/compound-mini advertises a 70000 TPM budget
# of its own, but it's an agentic model that internally calls other sub-models
# (its real 429 responses named `openai/gpt-oss-120b` and
# `llama-3.3-70b-versatile` as the actually-exhausted resource) -- so its
# capacity silently collapses to near-zero exactly when the primary model is
# already under load, i.e. exactly when we'd need overflow most. Excluded.
#
# Pool order is a deliberate quality/determinism preference, not round-robin.
# All four below are validated on both the real composer prompt and our
# hardest instruction-following case (off-topic decline+redirect):
# 1. openai/gpt-oss-120b -- primary, best-understood plain chat model.
# 2. qwen/qwen3.8-27b -- equally reliable, clean JSON, no reasoning leakage.
# 3. openai/gpt-oss-safeguard-20b -- moderation-tuned but produces correct,
#    on-voice business messages in testing; independent standalone model.
# 4. openai/gpt-oss-20b -- last resort before the deterministic template;
#    demonstrated weaker multi-step instruction-following earlier.
#
# Explicitly excluded after testing: groq/compound-mini and groq/compound (not
# independent capacity, see above), qwen/qwen3.6-27b (leaks <think>...</think>
# into the content field, breaking JSON parsing), and allam-2-7b (defaults to
# Arabic responses even for English/Hindi-English prompts).
#
# GROQ_MODEL_POOL (comma-separated) is the one config knob that controls both
# the pool AND the primary/display model (DEFAULT_MODEL = pool[0]) -- there is
# no separate GROQ_MODEL override to keep out of sync with it.
DEFAULT_MODEL_POOL = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-safeguard-20b",
    "openai/gpt-oss-20b",
]
_POOL_ENV = os.environ.get("GROQ_MODEL_POOL")
_MODEL_POOL = [m.strip() for m in _POOL_ENV.split(",") if m.strip()] if _POOL_ENV else DEFAULT_MODEL_POOL
DEFAULT_MODEL = _MODEL_POOL[0]

MODEL_TPM_LIMITS: dict[str, int] = {
    "openai/gpt-oss-120b": 8000,
    "openai/gpt-oss-20b": 8000,
    "openai/gpt-oss-safeguard-20b": 8000,
    "qwen/qwen3.8-27b": 8000,
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


async def _post_once(client: httpx.AsyncClient, headers: dict, chosen_model: str,
                      system: str, user: str, temperature: float, max_tokens: int,
                      reasoning_effort: str) -> tuple[str | None, str | None]:
    """Fire one real HTTP call. Returns (content, None) on success or (None, error_message)."""
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
    try:
        resp = await client.post(GROQ_URL, json=payload, headers=headers)
        if resp.status_code in (429, 400):
            return None, f"http {resp.status_code} on {chosen_model}: {resp.text[:200]}"
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        finish_reason = data["choices"][0].get("finish_reason")
        if not content.strip() and finish_reason == "length":
            return None, f"truncated (finish_reason=length) on {chosen_model}"
        return content, None
    except httpx.TimeoutException as e:
        return None, f"timeout on {chosen_model}: {e}"
    except httpx.HTTPStatusError as e:
        return None, f"http {e.response.status_code} on {chosen_model}: {e.response.text[:200]}"
    except Exception as e:  # noqa: BLE001
        return None, f"{chosen_model}: {e}"


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
    pin reasoning_effort="low" to keep that bounded.

    When `model` is None (the normal path), this walks the pool: for each
    candidate, it RESERVES the estimated token cost immediately (before firing
    the request), not after the response returns -- otherwise several
    concurrent calls in the same tick (composing multiple triggers at once)
    would all see "headroom" on the same model and race onto it, blowing past
    its real per-request burst capacity even though the minute-level budget
    looked fine. If every pooled model looks exhausted by our own estimate, we
    take one short real backoff (Groq's server-side burst bucket often
    recovers within a couple seconds even when our rolling-window estimate
    hasn't) and retry the primary model directly. Only after that raises
    GroqError, so the caller can fall back to the deterministic template
    rather than waiting out a guaranteed failure.
    """
    key = api_key()
    if not key:
        raise GroqError("GROQ_API_KEY not set")

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    total_estimate = estimated_prompt_tokens + max_tokens

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        if model is not None:
            content, err = await _post_once(client, headers, model, system, user, temperature, max_tokens, reasoning_effort)
            if content is not None:
                _record_usage(model, total_estimate)
                return content
            raise GroqError(err or "unknown error")

        tried: set[str] = set()
        attempts: list[str] = []

        for candidate in _MODEL_POOL:
            if not _has_headroom(candidate, total_estimate):
                attempts.append(f"{candidate}: skipped (estimated {_estimated_recent_usage(candidate)}/{_model_tpm_limit(candidate)} TPM)")
                continue
            _record_usage(candidate, total_estimate)  # reserve BEFORE firing
            tried.add(candidate)
            content, err = await _post_once(client, headers, candidate, system, user, temperature, max_tokens, reasoning_effort)
            if content is not None:
                return content
            attempts.append(f"{candidate}: {err}")

        # Every pooled model was either estimated-exhausted or genuinely failed.
        # One short real backoff, then retry whichever model has the MOST estimated
        # headroom (not blindly the primary) -- Groq's server-side burst bucket often
        # recovers within a couple seconds even when our rolling-window estimate hasn't.
        if _MODEL_POOL:
            await asyncio.sleep(1.5)
            best = min(_MODEL_POOL, key=lambda m: _estimated_recent_usage(m) / _model_tpm_limit(m))
            _record_usage(best, total_estimate)
            content, err = await _post_once(client, headers, best, system, user, temperature, max_tokens, reasoning_effort)
            if content is not None:
                return content
            attempts.append(f"{best} (post-backoff): {err}")

        raise GroqError(" | ".join(attempts) if attempts else "no pooled models configured")
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
