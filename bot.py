"""Vera challenge bot -- FastAPI server implementing the 5 endpoints defined in
challenge-testing-brief.md, backed by the composer in composer.py and the
reply logic in conversation_handlers.py.

Run:
    uvicorn bot:app --host 0.0.0.0 --port 8080

Env:
    GROQ_API_KEY        required for real LLM composition (falls back to
                        deterministic templates if unset, so the bot still
                        runs and responds correctly-shaped JSON without it)
    GROQ_MODEL_POOL     optional, comma-separated model list; default is a validated
                        4-model pool (see groq_client.py)
    TEAM_NAME, TEAM_MEMBERS, CONTACT_EMAIL, BOT_VERSION  optional, for /v1/metadata
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Optional

import composer
import conversation_handlers as ch
import groq_client
from store import STORE, now_iso

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vera.bot")

app = FastAPI(title="Vera Challenge Bot")

START_TIME = time.time()
TICK_DEADLINE_SECONDS = 22.0
TICK_MAX_TRIGGERS_PROCESSED = 10
TICK_MAX_CONCURRENT_LLM_CALLS = 5


@app.on_event("startup")
async def _startup() -> None:
    app.state.http_client = httpx.AsyncClient(timeout=12.0)
    logger.info("Vera bot starting. GROQ_API_KEY set: %s", bool(groq_client.api_key()))


@app.on_event("shutdown")
async def _shutdown() -> None:
    await app.state.http_client.aclose()


# ---------------------------------------------------------------------------
# GET /v1/healthz
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": STORE.counts(),
    }


# ---------------------------------------------------------------------------
# GET /v1/metadata
# ---------------------------------------------------------------------------

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.environ.get("TEAM_NAME", "Solo Builder"),
        "team_members": os.environ.get("TEAM_MEMBERS", "Nikhil").split(","),
        "model": f"groq/{groq_client.DEFAULT_MODEL} (+ {len(groq_client._MODEL_POOL) - 1} pooled overflow models)",
        "approach": (
            "4-context composer (category/merchant/trigger/customer) dispatched by trigger.kind to a "
            "kind-specific framing prompt, Groq LLM at temperature=0, post-LLM validation (URL strip, "
            "CTA-shape check, taboo-vocab strip, anti-repetition retry) with a deterministic template "
            "fallback if the LLM is unavailable or times out. Composition draws from a preference-ordered "
            "pool of Groq models with independent per-minute token budgets (a large-capacity model included "
            "as overflow), so sustained load degrades gracefully through several real-LLM tiers before "
            "ever reaching the template fallback. Reply handling uses fast regex/streak heuristics for "
            "auto-reply detection, intent-transition routing, and hostile/off-topic exits before falling "
            "through to an LLM-composed continuation."
        ),
        "contact_email": os.environ.get("CONTACT_EMAIL", "nikhil19092005@gmail.com"),
        "version": os.environ.get("BOT_VERSION", "1.0.0"),
        "submitted_at": os.environ.get("SUBMITTED_AT", now_iso()),
    }


# ---------------------------------------------------------------------------
# POST /v1/context
# ---------------------------------------------------------------------------

class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: Optional[str] = None


VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in VALID_SCOPES:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {sorted(VALID_SCOPES)}"},
        )
    accepted, version_info = STORE.put_context(body.scope, body.context_id, body.version, body.payload)
    if not accepted:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version", "current_version": version_info},
        )
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": now_iso(),
    }


# ---------------------------------------------------------------------------
# POST /v1/tick
# ---------------------------------------------------------------------------

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


def _split_body_for_template(body: str, name: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", body.strip())
    if len(sentences) <= 1:
        return [name, body]
    cta_line = sentences[-1]
    hook = " ".join(sentences[:-1])
    return [name, hook, cta_line]


async def _compose_action_for_trigger(trigger_id: str, client: httpx.AsyncClient, sem: asyncio.Semaphore) -> Optional[dict]:
    trigger = STORE.get_context("trigger", trigger_id)
    if not trigger:
        return None

    merchant_id = trigger.get("merchant_id")
    customer_id = trigger.get("customer_id")
    if not merchant_id:
        return None

    suppression_key = trigger.get("suppression_key", f"{trigger.get('kind')}:{merchant_id}")
    if STORE.is_suppression_key_sent(suppression_key):
        return None
    if STORE.is_merchant_suppressed(merchant_id):
        return None

    merchant = STORE.get_context("merchant", merchant_id)
    if not merchant:
        return None
    category = STORE.get_category_for_merchant(merchant)
    if not category:
        return None
    customer = STORE.get_context("customer", customer_id) if customer_id else None

    async with sem:
        result = await composer.compose_message(category, merchant, trigger, customer, client=client)

    if not result.get("body"):
        return None

    conversation_id = f"conv_{merchant_id}_{trigger.get('id', trigger_id)}"
    if STORE.get_conversation(conversation_id) is not None:
        # already an in-flight conversation for this exact trigger; don't double-initiate
        return None

    if customer:
        template_name_subject = customer.get("identity", {}).get("name", "there")
    else:
        identity = merchant.get("identity", {})
        template_name_subject = identity.get("owner_first_name") or identity.get("name", "there")
    template_params = _split_body_for_template(result["body"], template_name_subject)
    kind_prefix = "merchant" if result["send_as"] == "merchant_on_behalf" else "vera"
    template_name = f"{kind_prefix}_{trigger.get('kind', 'generic')}_v1"

    conv = STORE.get_or_create_conversation(
        conversation_id,
        merchant_id=merchant_id,
        customer_id=customer_id,
        trigger_id=trigger.get("id", trigger_id),
        send_as=result["send_as"],
    )
    conv.record_bot_send(result["body"])
    STORE.mark_suppression_key_sent(suppression_key)

    return {
        "conversation_id": conversation_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "send_as": result["send_as"],
        "trigger_id": trigger.get("id", trigger_id),
        "template_name": template_name,
        "template_params": template_params,
        "body": result["body"],
        "cta": result["cta"],
        "suppression_key": result["suppression_key"],
        "rationale": result["rationale"],
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    if not body.available_triggers:
        return {"actions": []}

    triggers_with_urgency = []
    for tid in body.available_triggers:
        t = STORE.get_context("trigger", tid)
        triggers_with_urgency.append((t.get("urgency", 1) if t else 0, tid))
    triggers_with_urgency.sort(key=lambda x: -x[0])
    ordered_ids = [tid for _, tid in triggers_with_urgency][:TICK_MAX_TRIGGERS_PROCESSED]

    client: httpx.AsyncClient = app.state.http_client
    sem = asyncio.Semaphore(TICK_MAX_CONCURRENT_LLM_CALLS)

    tasks = [asyncio.create_task(_compose_action_for_trigger(tid, client, sem)) for tid in ordered_ids]
    done, pending = await asyncio.wait(tasks, timeout=TICK_DEADLINE_SECONDS)
    for p in pending:
        p.cancel()

    actions = []
    seen_merchants: set[str] = set()
    # preserve original urgency ordering among completed tasks
    for task, tid in zip(tasks, ordered_ids):
        if task not in done:
            continue
        try:
            action = task.result()
        except Exception as e:  # noqa: BLE001
            logger.warning("tick: composing action for %s failed: %s", tid, e)
            continue
        if not action:
            continue
        if action["merchant_id"] in seen_merchants:
            continue  # at most one send per merchant per tick -- avoid spamming
        seen_merchants.add(action["merchant_id"])
        actions.append(action)

    return {"actions": actions}


# ---------------------------------------------------------------------------
# POST /v1/reply
# ---------------------------------------------------------------------------

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: Optional[str] = None
    turn_number: Optional[int] = None


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv = STORE.get_or_create_conversation(
        body.conversation_id, merchant_id=body.merchant_id, customer_id=body.customer_id
    )
    if body.merchant_id and not conv.merchant_id:
        conv.merchant_id = body.merchant_id
    if body.customer_id and not conv.customer_id:
        conv.customer_id = body.customer_id

    merchant = STORE.get_context("merchant", conv.merchant_id) if conv.merchant_id else None
    category = STORE.get_category_for_merchant(merchant) if merchant else None
    trigger = STORE.get_context("trigger", conv.trigger_id) if conv.trigger_id else None
    customer = STORE.get_context("customer", conv.customer_id) if conv.customer_id else None

    client: httpx.AsyncClient = app.state.http_client

    try:
        result = await asyncio.wait_for(
            ch.handle_reply(conv, body.message, merchant, category, trigger, customer, client=client),
            timeout=25.0,
        )
    except asyncio.TimeoutError:
        result = {"action": "wait", "wait_seconds": 300, "rationale": "Composition took too long this turn; backing off briefly."}

    return result


# ---------------------------------------------------------------------------
# POST /v1/teardown (optional, per testing-brief section 11)
# ---------------------------------------------------------------------------

@app.post("/v1/teardown")
async def teardown():
    STORE.wipe()
    return {"status": "wiped"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bot:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
