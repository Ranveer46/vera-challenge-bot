"""Multi-turn reply handling for /v1/reply.

Implements the "open challenges" from challenge-brief.md section 12:
auto-reply detection, intent-transition routing, and graceful exit --
plus hostile / off-topic handling exercised by the phase-4 replay tests
in challenge-testing-brief.md.

Classification is done with fast deterministic heuristics (regex/streak
counters) BEFORE any LLM call, because these are exactly the behaviors the
judge scores as pass/fail on conversation *flow*, not just message quality
-- they must be reliable, not a coin flip from a sampled LLM response.
"""

from __future__ import annotations

import json
import logging
import re

import gemini_client
import groq_client
from store import ConversationState

logger = logging.getLogger("vera.conversation")

AUTO_REPLY_PATTERNS = [
    r"thank you for (contacting|reaching)",
    r"we (will|shall|'ll) (get back|respond|revert)",
    r"team will (respond|get back|revert)",
    r"this is an automated",
    r"automated (assistant|reply|response|message)",
    r"currently (unavailable|closed|busy)",
    r"busy right now",
    r"out of office",
    r"will respond shortly",
]
AUTO_REPLY_RE = re.compile("|".join(AUTO_REPLY_PATTERNS), re.IGNORECASE)

INTENT_COMMIT_PATTERNS = [
    r"let'?s do it",
    r"lets do it",
    r"go ahead",
    r"\bproceed\b",
    r"\bconfirm(ed)?\b",
    r"sounds good",
    r"\bi'?m in\b",
    r"sign me up",
    r"i want to join",
    r"want to join",
    r"\byes\b.{0,15}(want|like|do it|join|start)",
    r"ok(ay)?[,.]?\s*(let'?s|lets)?\s*(do it|go|start|proceed)?",
    r"what'?s next",
    r"whats next",
]
INTENT_COMMIT_RE = re.compile("|".join(INTENT_COMMIT_PATTERNS), re.IGNORECASE)

QUALIFYING_PHRASES_RE = re.compile(
    r"would you say|do you (usually|often|typically)|can you tell me|what if|how about|"
    r"just to (understand|plan|clarify)|before we (start|proceed)",
    re.IGNORECASE,
)

HOSTILE_PATTERNS = [
    r"\bstop (messaging|contacting|texting)\b",
    r"\bspam\b",
    r"\buseless\b",
    r"\bannoying\b",
    r"leave me alone",
    r"don'?t (message|contact|text) me",
    r"\bstupid\b",
    r"\bharass",
    r"f+u+c+k",
    r"\bpiss off\b",
    r"\bshut up\b",
]
HOSTILE_RE = re.compile("|".join(HOSTILE_PATTERNS), re.IGNORECASE)

OFFTOPIC_DOMAIN_RE = re.compile(
    r"\bgst\b|\btax(es)?\b|\bloan\b|\binsurance\b|\bvisa\b|\blegal advice\b|\bpersonal problem\b|"
    r"\bare you (human|a bot|real)\b|\bwho (made|built) you\b",
    re.IGNORECASE,
)

NOT_INTERESTED_RE = re.compile(r"not interested|no thanks|maybe later|not now|don'?t need", re.IGNORECASE)


def classify(message: str, conv: ConversationState) -> str:
    """Returns one of: auto_reply, hostile, intent_commit, off_topic, not_interested, normal."""
    text = message.strip()

    if HOSTILE_RE.search(text):
        return "hostile"

    is_verbatim_repeat = bool(conv.turns) and any(
        t["from"] == "merchant" and t["message"].strip().lower() == text.lower() for t in conv.turns
    )
    if AUTO_REPLY_RE.search(text) or is_verbatim_repeat:
        return "auto_reply"

    if NOT_INTERESTED_RE.search(text):
        return "not_interested"

    if INTENT_COMMIT_RE.search(text) and not QUALIFYING_PHRASES_RE.search(text):
        return "intent_commit"

    if OFFTOPIC_DOMAIN_RE.search(text):
        return "off_topic"

    return "normal"


def handle_auto_reply(conv: ConversationState) -> dict:
    conv.auto_reply_streak += 1
    streak = conv.auto_reply_streak

    if streak == 1:
        body = "Looks like an auto-reply 😊 When the owner sees this, a quick reply here works."
        conv.record_bot_send(body)
        return {"action": "send", "body": body, "cta": "binary_yes_no",
                "rationale": "Detected likely WhatsApp Business auto-reply (canned phrasing/verbatim repeat). "
                             "One explicit nudge to flag it for the human owner before backing off."}
    if streak == 2:
        return {"action": "wait", "wait_seconds": 86400,
                "rationale": "Same auto-reply twice in a row -- owner not at phone. Waiting 24h before retry "
                             "rather than burning more turns."}
    conv.ended = True
    return {"action": "end",
            "rationale": "Auto-reply 3+ times with zero real engagement signal. Closing the conversation "
                         "instead of continuing to burn turns on a canned response."}


def handle_hostile(conv: ConversationState) -> dict:
    conv.ended = True
    return {"action": "end",
            "rationale": "Merchant signaled explicit frustration/opt-out. Closing immediately without further "
                         "engagement; this merchant should be suppressed for future proactive sends."}


def handle_not_interested(conv: ConversationState) -> dict:
    conv.ended = True
    return {"action": "end",
            "rationale": "Merchant declined. Graceful exit -- no further nudges on this conversation."}


def handle_intent_commit(conv: ConversationState) -> dict:
    """Caller still needs to fill in `body` via LLM/fallback -- this just flips the mode
    so the composer prompt knows to switch from qualifying to action."""
    conv.mode = "action"
    return {}


async def compose_reply(
    conv: ConversationState,
    merchant_message: str,
    merchant: dict | None,
    category: dict | None,
    trigger: dict | None,
    customer: dict | None,
    *,
    label: str = "normal",
    client=None,
) -> dict:
    """Builds the free-form LLM reply for 'normal', 'intent_commit', and 'off_topic' cases.

    `label` must be the classification computed by the caller BEFORE `conv.record_inbound()`
    was called -- classifying again here, after the message is already in conv.turns, would
    make the verbatim-repeat check match the message against itself (see handle_reply)."""
    history = conv.history_text()

    identity = (merchant or {}).get("identity", {})
    name = identity.get("owner_first_name") or identity.get("name") or "there"
    hindi_pref = "hi" in (identity.get("languages") or [])
    if customer:
        hindi_pref = hindi_pref or "hi" in (customer.get("identity", {}).get("language_pref") or "").lower()

    mode_instruction = ""
    if conv.mode == "action":
        mode_instruction = (
            "CRITICAL: the merchant just gave EXPLICIT commitment/agreement (e.g. 'let's do it', 'go ahead'). "
            "Do NOT ask another qualifying question. Respond as if you are already executing: name a concrete "
            "next step, a scope/number if derivable from context, and end with an explicit action-style CTA "
            "using a word like 'confirm', 'proceed', 'sending', or 'done' (e.g. 'Reply CONFIRM to proceed')."
        )
    elif label == "off_topic":
        mode_instruction = (
            "The merchant asked something outside your scope (e.g. tax/legal/personal/unrelated). You are "
            "Vera, a marketing/engagement assistant -- you have NO ability to actually help with this, so do "
            "NOT attempt it, do not offer steps, do not ask clarifying questions about it. In one short clause "
            "decline (e.g. 'that's outside what I can help with -- best to check with your CA/a professional'), "
            "then pivot back to the live topic between you and the merchant (use conversation history / the "
            "trigger if present; if neither is available, pivot to a generic marketing-help offer instead)."
        )
    else:
        mode_instruction = (
            "Continue the conversation naturally: acknowledge what the merchant just said, honor any explicit "
            "ask, and move the conversation one concrete step forward with a single CTA."
        )

    system = (
        "You are Vera, magicpin's merchant-AI assistant, mid-conversation over WhatsApp. Reply to the "
        "merchant's/customer's latest message. Never invent facts not present in the context you're given. "
        f"{'Write natural Hindi-English code-mix in ROMAN SCRIPT ONLY (transliterated, never Devanagari characters).' if hindi_pref else 'Write clean English.'} "
        "No preamble, no re-introducing yourself. Exactly one CTA, no URLs. Never expose raw internal "
        "field names/slugs with underscores -- paraphrase them in plain words. "
        "Respond with ONLY JSON: {\"body\": \"...\", \"cta\": \"open_ended|binary_yes_no|binary_confirm_cancel|"
        "multi_choice_slot|none\", \"rationale\": \"...\"}"
    )

    voice = (category or {}).get("voice", {}) or {}
    cat_slim = {"slug": (category or {}).get("slug"), "tone": voice.get("tone"), "vocab_taboo": voice.get("vocab_taboo")}
    perf = (merchant or {}).get("performance", {})
    merch_slim = {
        "active_offers": [o.get("title") for o in (merchant or {}).get("offers", []) if o.get("status") == "active"],
        "signals": (merchant or {}).get("signals", [])[:4],
        "performance": {"views": perf.get("views"), "calls": perf.get("calls"), "ctr": perf.get("ctr")} if perf else {},
    }
    trig_slim = {"kind": (trigger or {}).get("kind"), "payload": (trigger or {}).get("payload")} if trigger else None
    cust_slim = {
        "relationship": (customer or {}).get("relationship"),
        "state": (customer or {}).get("state"),
        "preferences": (customer or {}).get("preferences"),
    } if customer else None

    user = (
        f"CONVERSATION SO FAR:\n{history}\n\nLATEST MESSAGE FROM {'CUSTOMER' if customer else 'MERCHANT'}: "
        f"{merchant_message}\n\n{mode_instruction}\n\n"
        f"Name to use: {name}\n"
        f"Category (may be empty): {json.dumps(cat_slim, ensure_ascii=False)}\n"
        f"Merchant (may be empty): {json.dumps(merch_slim, ensure_ascii=False)}\n"
        f"Original trigger (may be empty): {json.dumps(trig_slim, ensure_ascii=False)}\n"
        f"Customer (may be empty): {json.dumps(cust_slim, ensure_ascii=False)}\n"
        f"Already sent, do not repeat: {conv.sent_bodies[-2:]}"
    )

    if not groq_client.api_key() and not gemini_client.enabled():
        body = f"Got it, {name} -- noted. What would you like me to do next?"
        return {"action": "send", "body": body, "cta": "open_ended",
                "rationale": "Fallback (no LLM configured): generic acknowledgment + open next-step ask."}

    try:
        est_tokens = (len(system) + len(user)) // 4
        raw = None
        if gemini_client.enabled():
            try:
                raw = await gemini_client.complete(system, user, max_tokens=800, timeout=11.0)
            except gemini_client.GeminiError as e:
                logger.info("Gemini unavailable this reply, falling back to Groq pool: %s", e)
        if raw is None:
            raw = await groq_client.complete(system, user, client=client, estimated_prompt_tokens=est_tokens)
        parsed = groq_client.extract_json(raw) or {}
        body = (parsed.get("body") or "").strip()
        cta = parsed.get("cta") if parsed.get("cta") in {
            "open_ended", "binary_yes_no", "binary_confirm_cancel", "multi_choice_slot", "none"
        } else "open_ended"
        rationale = parsed.get("rationale") or "Continued conversation based on latest merchant message."
        if not body:
            raise groq_client.GroqError("empty body")
        body = re.sub(r"https?://\S+|www\.\S+", "", body).strip()
        return {"action": "send", "body": body, "cta": cta, "rationale": rationale}
    except groq_client.GroqError as e:
        logger.warning("reply LLM failed, using fallback: %s", e)
        body = f"Got it, {name} -- noted. What would you like me to do next?"
        return {"action": "send", "body": body, "cta": "open_ended",
                "rationale": f"Fallback after LLM error ({e}): generic acknowledgment + open next-step ask."}


async def handle_reply(
    conv: ConversationState,
    merchant_message: str,
    merchant: dict | None,
    category: dict | None,
    trigger: dict | None,
    customer: dict | None,
    *,
    client=None,
) -> dict:
    """Top-level dispatcher used by bot.py's /v1/reply handler."""
    if conv.ended:
        return {"action": "end", "rationale": "Conversation already ended; not re-engaging."}

    # classify against history BEFORE recording this message, so the verbatim-repeat
    # check compares against prior turns, not against the message matching itself.
    label = classify(merchant_message, conv)
    conv.record_inbound(merchant_message)

    if label == "auto_reply":
        return handle_auto_reply(conv)
    if label == "hostile":
        return handle_hostile(conv)
    if label == "not_interested":
        return handle_not_interested(conv)

    conv.auto_reply_streak = 0  # any real reply resets the streak
    if label == "intent_commit":
        handle_intent_commit(conv)

    result = await compose_reply(conv, merchant_message, merchant, category, trigger, customer, label=label, client=client)
    if result.get("action") == "send" and result.get("body"):
        conv.record_bot_send(result["body"])
    return result
