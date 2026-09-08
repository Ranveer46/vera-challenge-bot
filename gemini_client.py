"""Thin async client for the Gemini generateContent API.

Kept deliberately independent of groq_client.py: Gemini is tried as the
first-choice composer model (when configured), with a bounded timeout, and
ANY failure here (timeout, connection error, non-200, empty content) raises
GeminiError so the caller can fall through to the already-proven Groq pool
without any special-casing. Gemini is pure upside if it works, zero risk if
it doesn't -- composer.py never blocks on it longer than its own timeout.
"""

from __future__ import annotations

import json
import logging
import os

import httpx

logger = logging.getLogger("vera.gemini")

GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")


class GeminiError(Exception):
    pass


def api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or None


def enabled() -> bool:
    return bool(api_key())


async def complete(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    timeout: float = 12.0,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Call Gemini generateContent. Raises GeminiError on any failure.

    thinkingConfig.thinkingBudget=0 disables Gemini 3.x's default internal
    "thinking" pass -- without it, the model can silently spend its entire
    max_tokens budget on hidden reasoning and return empty content (the same
    class of bug we hit with Groq's reasoning models, different vendor).
    """
    key = api_key()
    if not key:
        raise GeminiError("GEMINI_API_KEY not set")

    chosen_model = model or DEFAULT_MODEL
    url = GEMINI_URL_TMPL.format(model=chosen_model, key=key)
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            raise GeminiError(f"http {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise GeminiError(f"no candidates in response: {json.dumps(data)[:300]}")
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        finish_reason = candidates[0].get("finishReason")
        if not text.strip():
            raise GeminiError(f"empty content (finishReason={finish_reason})")
        return text
    except httpx.TimeoutException as e:
        raise GeminiError(f"timeout: {e}") from e
    except httpx.HTTPError as e:
        raise GeminiError(f"connection error: {e}") from e
    except GeminiError:
        raise
    except Exception as e:  # noqa: BLE001
        raise GeminiError(str(e)) from e
    finally:
        if owns_client:
            await client.aclose()
