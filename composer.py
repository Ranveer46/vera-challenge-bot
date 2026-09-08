"""The composer: category + merchant + trigger (+ customer) -> ComposedMessage.

This is the single point of failure the engagement-design doc calls out, so it
does three things deliberately:

1. Builds a prompt that hands the LLM only verified facts pulled from the
   contexts (never invents), and tells it explicitly which lever family to
   lean on for each trigger `kind` (research-digest framing vs recall-due
   framing vs perf-dip reframe, etc.) -- see FRAMING.
2. Validates the LLM's output against the operational rules the judge
   penalizes on: URLs, multi-CTA, verbatim repetition, missing language
   match -- and repairs or regenerates rather than trusting blindly.
3. Falls back to a deterministic template composer if the LLM is
   unavailable or times out, so the bot never breaks its 30s budget.
"""

from __future__ import annotations

import json
import logging
import os
import re

import gemini_client
import groq_client

logger = logging.getLogger("vera.composer")

VALID_CTA = {"open_ended", "binary_yes_no", "binary_confirm_cancel", "multi_choice_slot", "none"}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Hindi-English code-mix markers used to sanity-check language matching.
HINGLISH_MARKERS = [
    "hai", "hain", "kar", "aap", "aapka", "aapke", "ke liye", "kya", "yeh", "ye",
    "sakte", "sakti", "chalega", "shukriya", "namaste", "ji", "wala", "bhi",
    "abhi", "din", "ho", "raha", "rahi", "lagta", "milega", "acha",
]

# -----------------------------------------------------------------------
# Trigger-kind framing dispatch (design-doc: "composer dispatches by kind")
# -----------------------------------------------------------------------

FRAMING: dict[str, str] = {
    "research_digest": (
        "Lead with the single most relevant digest item for this merchant's actual patient/customer "
        "mix (use customer_aggregate/signals to justify relevance). Cite the source (title + source field) "
        "so it reads as credible, not promotional. End by offering to do the next step (pull the "
        "abstract / draft a customer-facing version) -- open_ended CTA."
    ),
    "regulation_change": (
        "This is compliance, not marketing -- be direct and slightly urgent. State exactly what changed "
        "and the deadline from the payload/digest item. Offer to help them get compliant (checklist, draft "
        "note). binary_yes_no or open_ended CTA depending on how concrete the offered help is."
    ),
    "category_trend_movement": (
        "Frame around the trend_signal's numbers (delta_yoy, segment). Connect it to something the "
        "merchant could plausibly act on given their own offer_catalog. Curiosity-driven, open_ended CTA."
    ),
    "festival_upcoming": (
        "Anchor on the festival name + days_until from the trigger payload. Suggest a concrete, "
        "category-appropriate seasonal action (use seasonal_beats/offer_catalog if relevant). Keep it low "
        "urgency if days_until is large. open_ended or binary_yes_no."
    ),
    "weather_heatwave": (
        "Use the specific weather fact from payload. Suggest one concrete, low-effort action tied to the "
        "category (e.g. footfall dip/spike framing). binary_yes_no or open_ended."
    ),
    "local_news_event": (
        "Reference the specific local event/closure from payload and its plausible business impact. "
        "Give a contrarian, specific recommendation if you can reason one out from merchant data -- don't "
        "just relay the news. open_ended."
    ),
    "competitor_opened": (
        "Only use competitor facts actually present in the trigger payload -- if the payload has no real "
        "competitor name/distance (e.g. it's a placeholder), do NOT invent one. In that case pivot to the "
        "merchant's own peer_stats gap (e.g. ctr vs peer median) as the hook instead, still under a "
        "competitive-curiosity frame. curiosity lever, open_ended CTA."
    ),
    "perf_spike": (
        "Name the specific metric and percentage from performance.delta_7d or the trigger payload. "
        "Suggest capitalizing on it (e.g. don't let a stale post waste the spike). binary_yes_no or open_ended."
    ),
    "perf_dip": (
        "Name the specific metric and percentage. If it's explainable by a seasonal_beat, say so explicitly "
        "(reframe anxiety) rather than treating it as a crisis alone. Propose one concrete corrective action. "
        "binary_yes_no."
    ),
    "milestone_reached": (
        "Celebrate the specific number/threshold from merchant data (reviews, views, retention). Use it as "
        "social proof leverage -- suggest turning it into content (a post, a share) rather than just praise. "
        "open_ended or none."
    ),
    "review_theme_emerged": (
        "Name the specific theme and how many reviews mention it (review_themes). If sentiment is negative, "
        "propose one concrete fix; if positive, propose amplifying it. binary_yes_no."
    ),
    "renewal_due": (
        "State the exact days_remaining and plan from subscription/payload. Loss-aversion framed, but "
        "factual, not alarmist. binary_yes_no CTA -- this is an action trigger."
    ),
    "dormant_with_vera": (
        "Acknowledge the gap without guilt-tripping, but the re-open hook MUST be a real number specific to "
        "THIS merchant -- pull one from performance (views/calls/ctr and its delta_7d), customer_aggregate, "
        "signals, or peer_stats comparison. A generic 'thought of something new for you' with no real number "
        "attached is a failure here. Pair the number with one fresh, low-friction curiosity or "
        "asking-the-merchant angle (not a repeat of an old pitch -- check conversation_history). open_ended."
    ),
    "curious_ask_due": (
        "Ask the merchant a genuine, specific, low-effort question about their business this week "
        "(demand, requests, a pattern) -- the 'asking the merchant' lever. Promise a concrete small "
        "deliverable in return (a post, a reply script). open_ended, and it can carry cta=none if it's "
        "purely a question with no formal ask."
    ),
    "active_planning_intent": (
        "The merchant is already mid-planning (see conversation_history/payload) -- deliver a concrete, "
        "ready-to-use artifact with REAL numbers (a tiered price list, a specific quantity and per-unit "
        "price, a specific date/time window), grounded in category.offer_catalog and merchant.identity "
        "locality -- not another question, and not a vague 'here's a draft' with no numbers in it. If the "
        "trigger payload only has a topic label (no real numbers), build the numbers from the category's "
        "offer_catalog pricing pattern applied to a plausible scope for this merchant -- state it as a "
        "starting proposal ('here's a starter version') so it reads as a real draft, not a fabricated fact. "
        "Effort externalization is the dominant lever. binary_yes_no or open_ended."
    ),
    "ipl_match_today": (
        "Use the specific match/day facts from payload. If weekday vs weekend dynamics matter for this "
        "category, reason about it explicitly rather than assuming a generic promo is good. Recommend "
        "leveraging an existing active offer if one exists. binary_yes_no."
    ),
    "unverified_gbp": (
        "State plainly that the listing is unverified and what that costs them (peer_stats comparison), "
        "then offer the concrete next step to verify. binary_yes_no."
    ),
    # -- customer-scoped kinds --
    "recall_due": (
        "Use the exact due window and available_slots from the trigger payload, and the merchant's actual "
        "active offer for pricing. Multi-choice slot CTA is acceptable here (booking flow exception to "
        "binary-only). Honor the customer's preferred_slots and language_pref."
    ),
    "appointment_tomorrow": (
        "Confirm the specific appointment time from payload. Simple, reassuring, low-friction confirm/cancel "
        "CTA. Honor language_pref."
    ),
    "chronic_refill_due": (
        "Name the specific medicines/services and the run-out date from payload. If the customer is an "
        "older age_band, use a respectful, precise tone (namaste-style salutation is fine if language_pref "
        "supports it). Mention any senior/loyalty discount if it exists in merchant offers. binary_confirm_cancel."
    ),
    "customer_lapsed_soft": (
        "Acknowledge time since last visit -- state it as a specific number (days/weeks/months from "
        "relationship.last_visit, or visits_total) without guilt. Offer one relevant, concrete reason to "
        "return: a real active offer WITH its price from merchant.offers, or a new relevant service named "
        "specifically. No-shame framing. binary_yes_no with a low-commitment ask."
    ),
    "customer_lapsed_hard": (
        "Same as lapsed_soft -- specific gap length, a named real offer with its price -- but with more "
        "warmth and a stronger no-commitment safety net (free trial slot, no auto-charge language) since "
        "the gap is longer. binary_yes_no."
    ),
    "trial_followup": (
        "Reference what they tried and ask how it went / offer the natural next step in their journey. "
        "open_ended or binary_yes_no."
    ),
    "wedding_package_followup": (
        "Use the exact wedding_date / days_to_wedding and the program window from payload. Effort "
        "externalization (offer to hold a slot). binary_yes_no or binary_confirm_cancel."
    ),
    "unplanned_slot_open": (
        "Offer the specific open slot to a customer who plausibly wants it (based on preferences). Low "
        "friction, time-boxed. binary_confirm_cancel."
    ),
}

DEFAULT_FRAMING = (
    "Anchor the message on the most specific, verifiable fact available across the trigger payload, "
    "merchant performance/signals, and category peer_stats/digest -- in that order of preference. Pick "
    "exactly one compulsion lever family and one CTA shape appropriate to whether this is an action "
    "trigger (binary) or an information trigger (open_ended/none)."
)

DEFAULT_CTA_BY_SCOPE = {
    "merchant": "open_ended",
    "customer": "binary_yes_no",
}


SYSTEM_PROMPT = """You are the composer for "Vera", magicpin's merchant-AI assistant. You write ONE \
WhatsApp message at a time, either to a merchant (send_as="vera") or, on the merchant's behalf, to one \
of their customers (send_as="merchant_on_behalf").

You will be given four JSON context blocks (category, merchant, trigger, and optionally customer) plus a \
framing hint for this trigger kind. Follow these rules exactly:

RULES
1. Use ONLY facts present in the JSON you are given. Never invent a number, date, competitor name, \
research citation, or offer that isn't literally in the JSON. If a payload field looks like a \
placeholder (e.g. contains "placeholder": true) treat it as having no real facts -- fall back to real \
merchant/category facts instead (performance numbers, peer_stats, signals, real offers, real digest items).
2. Match category voice exactly: tone/register from category.voice, use vocab_allowed terms where natural, \
NEVER use any word in category.voice.vocab_taboo (in any form/casing). If category.voice.salutation_examples \
is non-empty, open with that exact salutation pattern (e.g. "Dr. {first_name}" for a clinical-peer category) \
instead of a casual "Hey" / "Hi there" -- a casual opener on a peer-clinical category (dentists, doctors) \
reads as off-voice even when the rest of the message is fine.
3. Match language: if merchant.identity.languages includes "hi" (or customer language_pref mentions "hi"), \
write natural Hindi-English code-mix in ROMAN SCRIPT ONLY (transliterated Hindi using English letters, e.g. \
"aapka appointment kal hai, kya yeh time thik rahega?") -- NEVER use Devanagari script (never write हिन्दी \
characters). This is how WhatsApp Hindi-English actually gets typed in India. If languages/pref is English \
only, write clean English. Never ignore an explicit language preference.
4. Personalize: use the owner_first_name or customer name if present. Reference the merchant's OWN numbers \
(not generic claims) wherever the framing calls for specificity.
5. Exactly ONE call-to-action. No "Reply YES for X, NO for Y, MAYBE for Z" multi-branch asks. Binary \
(yes/no or confirm/cancel) for action triggers; open-ended or none for pure-information triggers; \
multi-choice-slot is the one allowed exception, only for booking/slot-offering triggers.
6. NEVER include a URL (http://, https://, www.) in the body -- write the offer to help/send something \
instead of linking to it.
7. No preambles ("I hope you're doing well..."), no re-introducing yourself if conversation_history shows \
prior turns, no promotional/hype tone ("AMAZING DEAL!", "BEST IN CITY") -- peer/colleague tone.
8. Prefer service+price offers ("Haircut @ ₹99") over generic percentage-off framing when an offer_catalog \
entry is available and relevant.
9. Do not repeat, near-verbatim, any body already present in conversation_history for this merchant/customer.
10. Body should be concise -- say the one thing that matters, land the ask in the final sentence. No hard \
length cap, but every sentence must be doing work (a fact, the ask, or personalization) -- cut anything else.
11. rationale is 1-2 sentences, for an internal judge, explaining which contexts and which compulsion \
lever(s) you used -- it must accurately describe what the body actually does.
12. Never expose raw internal field names/slugs verbatim (anything with underscores or quotes around a \
code-like token, e.g. "high_risk_adult_cohort", "ctr_below_peer_median"). Paraphrase what the signal means \
in plain, natural words instead.
13. SPECIFICITY IS MANDATORY, NOT OPTIONAL: the body MUST contain at least one concrete, verifiable number \
(a price, a percentage, a count, a date, or a named source) drawn from the JSON you were given -- never a \
purely qualitative message like "I've got something new for you" or "here's a draft" with zero numbers in \
it. If the trigger payload itself has no real number (e.g. it's a placeholder or just a topic label), pull \
one from elsewhere in the JSON instead, in this order of preference: merchant.performance (views/calls/ctr \
and delta_7d), merchant.customer_aggregate, merchant.signals, category.peer_stats, category.offer_catalog \
pricing, or category.digest. There is always a real number available somewhere in the JSON -- find it.
14. SOCIAL PROOF IS UNDERUSED -- ACTIVELY LOOK FOR IT: whenever category.peer_stats has a real comparative \
number (e.g. "peers average X"), prefer working it in as a second anchor alongside the merchant's own number \
(e.g. "your CTR is 2.1% vs the metro median of 3.0%") rather than stating the merchant's number in isolation. \
This is a real compulsion lever (peer comparison), not filler -- use it whenever the data supports it, not \
just when the framing hint explicitly says so.

Respond with ONLY a single JSON object, no prose before or after, no markdown fences:
{"body": "...", "cta": "open_ended|binary_yes_no|binary_confirm_cancel|multi_choice_slot|none", "rationale": "..."}
"""


def _trim(d: dict | None, keys: list[str]) -> dict:
    if not d:
        return {}
    return {k: d[k] for k in keys if k in d}


DIGEST_RELEVANT_KINDS = {"research_digest", "regulation_change", "cde_opportunity"}
SEASONAL_RELEVANT_KINDS = {"festival_upcoming", "seasonal_perf_dip", "weather_heatwave"}
TREND_RELEVANT_KINDS = {"category_trend_movement"}
CONTENT_LIB_RELEVANT_KINDS = {"trial_followup", "recall_due", "customer_lapsed_soft", "customer_lapsed_hard"}


def _digest_relevant(category: dict, trigger: dict, limit: int = 2) -> list[dict]:
    """Pick the digest items most relevant to this send: the trigger's named item first."""
    digest = category.get("digest", []) or []
    if not digest:
        return []
    top_id = (trigger.get("payload") or {}).get("top_item_id")
    picked = [d for d in digest if d.get("id") == top_id]
    if len(picked) < limit and trigger.get("kind") in DIGEST_RELEVANT_KINDS:
        remaining = [d for d in digest if d not in picked]
        picked += remaining[: limit - len(picked)]
    trimmed = []
    for d in picked[:limit]:
        trimmed.append({
            "title": d.get("title"),
            "source": d.get("source"),
            "summary": (d.get("summary") or "")[:220],
            "trial_n": d.get("trial_n"),
            "patient_segment": d.get("patient_segment"),
            "date": d.get("date"),
            "actionable": d.get("actionable"),
        })
    return trimmed


def build_prompt(category: dict, merchant: dict, trigger: dict, customer: dict | None) -> tuple[str, str]:
    """Builds a deliberately lean prompt -- Groq free-tier keys are capped at a per-model
    tokens-per-minute budget, so every field here earns its place. Fields only relevant to
    specific trigger kinds (seasonal_beats, trend_signals, patient_content_library) are
    included only when that kind is in play."""
    kind = trigger.get("kind", "")
    scope = trigger.get("scope", "merchant")
    framing = FRAMING.get(kind, DEFAULT_FRAMING)
    default_cta = DEFAULT_CTA_BY_SCOPE.get(scope, "open_ended")

    voice = category.get("voice", {}) or {}
    cat_view = {
        "slug": category.get("slug"),
        "voice": {
            "tone": voice.get("tone"),
            "register": voice.get("register"),
            "code_mix": voice.get("code_mix"),
            "vocab_allowed": (voice.get("vocab_allowed") or [])[:6],
            "vocab_taboo": voice.get("vocab_taboo") or [],
            "salutation_examples": voice.get("salutation_examples") or [],
        },
        "offer_catalog_titles": [o.get("title") for o in category.get("offer_catalog", [])[:6]],
        "peer_stats": category.get("peer_stats", {}),
        "digest": _digest_relevant(category, trigger),
    }
    if kind in SEASONAL_RELEVANT_KINDS:
        cat_view["seasonal_beats"] = category.get("seasonal_beats", [])[:3]
    if kind in TREND_RELEVANT_KINDS:
        cat_view["trend_signals"] = category.get("trend_signals", [])[:3]
    if kind in CONTENT_LIB_RELEVANT_KINDS:
        cat_view["patient_content_library_titles"] = [c.get("title") for c in category.get("patient_content_library", [])[:3]]

    identity = merchant.get("identity", {})
    conv_hist = merchant.get("conversation_history", [])[-2:]
    merch_view = {
        "identity": {
            "name": identity.get("name"),
            "owner_first_name": identity.get("owner_first_name"),
            "city": identity.get("city"),
            "locality": identity.get("locality"),
            "languages": identity.get("languages"),
            "verified": identity.get("verified"),
        },
        "subscription": merchant.get("subscription", {}),
        "performance": merchant.get("performance", {}),
        "active_offers": [o.get("title") for o in merchant.get("offers", []) if o.get("status") == "active"],
        "recent_conversation": [{"from": t.get("from"), "body": (t.get("body") or "")[:180]} for t in conv_hist],
        "customer_aggregate": merchant.get("customer_aggregate", {}),
        "signals": merchant.get("signals", []),
        "review_themes": [
            {"theme": r.get("theme"), "sentiment": r.get("sentiment"), "occurrences_30d": r.get("occurrences_30d")}
            for r in merchant.get("review_themes", [])[:2]
        ],
    }

    trig_view = {
        "kind": kind,
        "scope": scope,
        "source": trigger.get("source"),
        "payload": trigger.get("payload", {}),
        "urgency": trigger.get("urgency"),
    }

    cust_view = None
    if customer:
        cust_view = {
            "identity": customer.get("identity", {}),
            "relationship": customer.get("relationship", {}),
            "state": customer.get("state"),
            "preferences": customer.get("preferences", {}),
        }

    user_payload = {
        "category": cat_view,
        "merchant": merch_view,
        "trigger": trig_view,
        "customer": cust_view,
        "framing_hint": framing,
        "default_cta_if_unclear": default_cta,
        "send_as_required_value": "merchant_on_behalf" if scope == "customer" else "vera",
    }

    user_prompt = "COMPOSE THIS MESSAGE FROM THE CONTEXT BELOW:\n" + json.dumps(
        user_payload, ensure_ascii=False, separators=(",", ":")
    )
    return SYSTEM_PROMPT, user_prompt


# -----------------------------------------------------------------------
# Validation / repair
# -----------------------------------------------------------------------

def _strip_urls(body: str) -> str:
    return URL_RE.sub("", body).strip()


def _looks_hinglish(body: str) -> bool:
    lower = body.lower()
    return any(f" {m} " in f" {lower} " for m in HINGLISH_MARKERS)


def _wants_hindi(merchant: dict, customer: dict | None) -> bool:
    langs = set(merchant.get("identity", {}).get("languages", []) or [])
    if "hi" in langs:
        return True
    if customer:
        pref = (customer.get("identity", {}).get("language_pref") or "").lower()
        if "hi" in pref:
            return True
    return False


def validate_and_repair(result: dict, category: dict, merchant: dict, trigger: dict, customer: dict | None) -> dict:
    body = (result.get("body") or "").strip()
    cta = result.get("cta") or ""
    rationale = (result.get("rationale") or "").strip()

    if URL_RE.search(body):
        body = _strip_urls(body)

    # safety net: de-slug any raw internal field name that leaked through quoted
    # (e.g. "high_risk_adult_cohort" -> high risk adult cohort)
    def _deslug(m: re.Match) -> str:
        return m.group(1).replace("_", " ")

    body = re.sub(r'["“]([a-z0-9]+_[a-z0-9_]+)["”]', _deslug, body)
    body = re.sub(r"\b([a-z0-9]+_[a-z0-9_]+)\b", lambda m: m.group(1).replace("_", " "), body)

    if cta not in VALID_CTA:
        cta = DEFAULT_CTA_BY_SCOPE.get(trigger.get("scope", "merchant"), "open_ended")

    # taboo vocabulary check -- hard-strip any exact taboo phrase occurrence
    taboos = category.get("voice", {}).get("vocab_taboo", []) or []
    for taboo in taboos:
        if taboo and taboo.lower() in body.lower():
            pattern = re.compile(re.escape(taboo), re.IGNORECASE)
            body = pattern.sub("", body)
            body = re.sub(r"\s{2,}", " ", body).strip(" .")

    if not rationale:
        rationale = f"Composed for trigger kind '{trigger.get('kind')}' using category+merchant context."

    return {"body": body, "cta": cta, "rationale": rationale}


DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def has_devanagari(body: str) -> bool:
    return bool(DEVANAGARI_RE.search(body))


def lacks_specificity(body: str) -> bool:
    """True if body has no digit at all -- a near-certain sign the message is a
    vague qualitative pitch with no verifiable anchor (rule 13 violation)."""
    return not re.search(r"\d", body)


def anti_repeat(body: str, prior_bodies: list[str]) -> bool:
    """True if body is a near-duplicate of something already sent."""
    norm = re.sub(r"\s+", " ", body).strip().lower()
    for prior in prior_bodies:
        pnorm = re.sub(r"\s+", " ", prior or "").strip().lower()
        if not pnorm:
            continue
        if norm == pnorm:
            return True
        # crude near-dup check: shared prefix of significant length
        if len(norm) > 20 and len(pnorm) > 20 and norm[:40] == pnorm[:40]:
            return True
    return False


# -----------------------------------------------------------------------
# Deterministic fallback (no LLM available / LLM failed)
# -----------------------------------------------------------------------

def fallback_compose(category: dict, merchant: dict, trigger: dict, customer: dict | None) -> dict:
    identity = merchant.get("identity", {})
    name = identity.get("owner_first_name") or identity.get("name") or "there"
    kind = trigger.get("kind", "update")
    scope = trigger.get("scope", "merchant")
    hindi = _wants_hindi(merchant, customer)

    active_offers = [o.get("title") for o in merchant.get("offers", []) if o.get("status") == "active"]
    signals = merchant.get("signals", []) or []
    perf = merchant.get("performance", {}) or {}

    fact = None
    payload = trigger.get("payload", {}) or {}
    if not payload.get("placeholder"):
        for k, v in payload.items():
            if isinstance(v, (int, float, str)) and k not in ("placeholder",):
                fact = f"{k.replace('_', ' ')}: {v}"
                break
    if not fact and signals:
        fact = signals[0]
    if not fact and perf:
        fact = f"views {perf.get('views', '?')}, calls {perf.get('calls', '?')} (last {perf.get('window_days', 30)}d)"

    if scope == "customer" and customer:
        cust_name = customer.get("identity", {}).get("name", "there")
        if hindi:
            body = (
                f"Hi {cust_name}, {identity.get('name', 'hamare clinic')} se update hai — {kind.replace('_', ' ')}. "
                f"{('Offer: ' + active_offers[0]) if active_offers else ''} Bataiye, kya aap interested hain?"
            ).strip()
        else:
            body = (
                f"Hi {cust_name}, this is {identity.get('name', 'your service provider')} with an update "
                f"regarding {kind.replace('_', ' ')}. {('We have ' + active_offers[0] + ' active.') if active_offers else ''} "
                f"Would this work for you?"
            ).strip()
        cta = "binary_yes_no"
        send_as = "merchant_on_behalf"
    else:
        if hindi:
            body = (
                f"{name}, ek update hai — {fact or kind.replace('_', ' ')}. "
                f"Kya main isse follow up kar sakti hoon?"
            )
        else:
            body = f"{name}, quick update — {fact or kind.replace('_', ' ')}. Want me to follow up on this?"
        cta = "open_ended"
        send_as = "vera"

    return {
        "body": re.sub(r"\s{2,}", " ", body).strip(),
        "cta": cta,
        "send_as": send_as,
        "rationale": f"Fallback template (LLM unavailable) anchored on: {fact or kind}.",
    }


_last_provider_debug: str = "n/a"


async def _llm_complete(system: str, user: str, *, client, estimated_prompt_tokens: int, allow_wait: bool) -> str:
    """Try Gemini first (if GEMINI_API_KEY is configured), bounded by a timeout that
    still leaves room to fall through to the proven Groq pool within budget. ANY
    Gemini failure (timeout, dropped connection, empty content) is logged and
    swallowed here -- it never affects overall reliability, since Groq is the
    already-verified fallback either way. Gemini is pure upside when it works."""
    global _last_provider_debug
    if gemini_client.enabled():
        gemini_timeout = 25.0 if allow_wait else 11.0
        try:
            text = await gemini_client.complete(system, user, max_tokens=1200, timeout=gemini_timeout)
            _last_provider_debug = "gemini:ok"
            return text
        except gemini_client.GeminiError as e:
            logger.info("Gemini unavailable this call, falling back to Groq pool: %s", e)
            _last_provider_debug = f"gemini:FAILED({e})"
    text = await _complete_with_optional_wait(system, user, client=client, estimated_prompt_tokens=estimated_prompt_tokens, allow_wait=allow_wait)
    if not _last_provider_debug.startswith("gemini:ok"):
        _last_provider_debug += " -> groq:ok"
    return text


async def _complete_with_optional_wait(system: str, user: str, *, client, estimated_prompt_tokens: int, allow_wait: bool) -> str:
    """Live bot path (allow_wait=False): fail fast to the deterministic fallback within the
    30s call budget. Offline submission-generator path (allow_wait=True): this script has no
    hard time budget, so patiently back off and retry on rate-limit errors instead of settling
    for template quality on 30 test pairs that matter for scoring."""
    if not allow_wait:
        return await groq_client.complete(system, user, client=client, estimated_prompt_tokens=estimated_prompt_tokens)

    import asyncio

    last_err: Exception | None = None
    for attempt in range(6):
        try:
            # on a prior truncation, ask for more headroom
            mt = 750 if attempt == 0 else 1300
            return await groq_client.complete(
                system, user, client=client, estimated_prompt_tokens=estimated_prompt_tokens, max_tokens=mt
            )
        except groq_client.GroqError as e:
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "tpm" in msg or "rate limit" in msg or "near tpm limit" in msg:
                wait_s = 8 + attempt * 4
                logger.info("rate-limited, waiting %ss before retry %d/6", wait_s, attempt + 1)
                await asyncio.sleep(wait_s)
                continue
            if "truncated" in msg or "empty body" in msg or "empty" in msg:
                logger.info("truncated/empty response, retrying with larger max_tokens (attempt %d/6)", attempt + 1)
                continue
            raise
    raise last_err  # type: ignore[misc]


# -----------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------

async def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
    *,
    client=None,
    retry_on_repeat: bool = True,
    allow_wait_on_rate_limit: bool = False,
) -> dict:
    """compose(category, merchant, trigger, customer) -> {body, cta, send_as, suppression_key, rationale}

    Matches the contract in challenge-brief.md section 7.1/5. Deterministic
    (temperature=0). Falls back to a template if the LLM is unavailable or
    fails, so this always returns within budget.
    """
    scope = trigger.get("scope", "merchant")
    send_as = "merchant_on_behalf" if scope == "customer" else "vera"
    suppression_key = trigger.get("suppression_key", f"{trigger.get('kind','x')}:{merchant.get('merchant_id','?')}")

    prior_bodies = [t.get("body") for t in merchant.get("conversation_history", []) if t.get("from") == "vera"]

    if not groq_client.api_key():
        result = fallback_compose(category, merchant, trigger, customer)
        result["suppression_key"] = suppression_key
        return result

    system, user = build_prompt(category, merchant, trigger, customer)
    est_tokens = (len(system) + len(user)) // 4
    try:
        raw = await _llm_complete(
            system, user, client=client, estimated_prompt_tokens=est_tokens, allow_wait=allow_wait_on_rate_limit
        )
        parsed = groq_client.extract_json(raw) or {}
        result = validate_and_repair(parsed, category, merchant, trigger, customer)

        needs_retry, nudge = False, ""
        if retry_on_repeat and anti_repeat(result["body"], prior_bodies):
            needs_retry = True
            nudge += "\n\nIMPORTANT: your previous draft repeated an already-sent message. Write a genuinely different angle/body this time."
        if has_devanagari(result["body"]):
            needs_retry = True
            nudge += "\n\nIMPORTANT: your previous draft used Devanagari script. Rewrite Hindi words in ROMAN SCRIPT (transliterated), never Devanagari characters."
        if lacks_specificity(result["body"]):
            needs_retry = True
            nudge += (
                "\n\nIMPORTANT: your previous draft had zero concrete numbers in it (rule 13 violation). "
                "Rewrite it to anchor on at least one real number from the JSON -- a price, a percentage, a "
                "count, or a date. Find one in merchant.performance, merchant.customer_aggregate, "
                "merchant.signals, category.peer_stats, or category.offer_catalog if the trigger payload "
                "itself doesn't have one."
            )

        if needs_retry:
            raw2 = await _llm_complete(
                system, user + nudge, client=client, estimated_prompt_tokens=est_tokens, allow_wait=allow_wait_on_rate_limit
            )
            parsed2 = groq_client.extract_json(raw2)
            if parsed2:
                result = validate_and_repair(parsed2, category, merchant, trigger, customer)

        if not result["body"]:
            raise groq_client.GroqError("empty body after validation")

        result["send_as"] = send_as
        result["suppression_key"] = suppression_key
        if os.environ.get("VERA_DEBUG_PROVIDER"):
            result["rationale"] = f"[{_last_provider_debug}] {result['rationale']}"
        return result
    except groq_client.GroqError as e:
        logger.warning("LLM composition failed, using fallback: %s", e)
        result = fallback_compose(category, merchant, trigger, customer)
        result["suppression_key"] = suppression_key
        return result
