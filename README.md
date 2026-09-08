# Vera Challenge Submission

## Approach

A single **composer** (`composer.py`) builds one prompt from the four context layers
(category, merchant, trigger, customer), dispatches to a **trigger-kind-specific framing
hint** (research-digest gets source-citation framing, recall-due gets slot-offering framing,
perf-dip gets seasonal reframe, competitor-opened falls back to the merchant's own peer-gap
if the payload has no real competitor data, etc.), and calls Groq (`openai/gpt-oss-120b`,
temperature=0) to produce `{body, cta, rationale}`. `send_as` and `suppression_key` are
computed in code, not by the LLM, so they're always correct.

The output then goes through **post-LLM validation** (`validate_and_repair`) before it's
trusted: strip any URL, reject invalid CTA shapes, de-slug any raw internal field name that
leaked into the body (e.g. `"high_risk_adult_cohort"` → "high risk adult cohort"), and retry
once (higher temperature) if the draft either near-repeats a message already in
`conversation_history` or slipped into Devanagari script instead of the Roman-script
Hindi-English mix every example in the brief actually uses.

If Groq is unavailable, times out, or the account's per-model rate limit is already spent,
`fallback_compose()` produces a deterministic, still-correctly-shaped template message from
whatever real facts are present (a signal, a performance number, a trigger payload field) —
never a crash, never an empty body, always within the 30s budget.

**HTTP layer** (`bot.py`) implements the 5 endpoints from `challenge-testing-brief.md`:
context push (versioned, idempotent), tick (ranks available triggers by urgency, composes
up to 10 in parallel bounded by a semaphore, caps at one send per merchant per tick,
respects suppression keys and a 22s soft deadline), reply (stateful per `conversation_id`),
healthz, and metadata.

**Multi-turn handling** (`conversation_handlers.py`) uses fast deterministic heuristics —
not a sampled LLM call — for the three behaviors the judge scores as pass/fail on
conversation *flow*: auto-reply detection (canned-phrase regex + verbatim-repeat check,
with a 3-strike escalation: nudge → wait 24h → end), intent-transition (explicit commitment
phrases flip the conversation into "action mode" so the next LLM turn is told explicitly not
to ask another qualifying question), and hostile/not-interested detection (immediate graceful
`end`, with the merchant suppressed for 30 days). Off-topic curveballs get one clause of
polite decline plus a redirect back to the live trigger topic. Everything else falls through
to an LLM-composed continuation of the conversation.

## Tradeoffs

- **Free-tier Groq keys cap each model at its own tokens-per-minute budget.** I trimmed the
  prompt aggressively (compact JSON, only kind-relevant category fields, 2-turn conversation
  history, offer titles instead of full offer objects) and pool four *independently hosted*
  models behind a single client-side usage tracker (`groq_client.py`): `openai/gpt-oss-120b`
  (primary) → `qwen/qwen3.8-27b` → `openai/gpt-oss-safeguard-20b` → `openai/gpt-oss-20b` (last
  resort) → deterministic template, ~32,000 combined TPM. Pool order is a quality preference,
  not round-robin -- all four are validated on the real composer prompt and on the hardest
  instruction-following case in this codebase (off-topic decline+redirect).
  `groq/compound-mini` was tried first as a large-capacity 5th tier (its own quota reports
  70,000 TPM) but rejected after its real 429 responses named `openai/gpt-oss-120b` and
  `llama-3.3-70b-versatile` as the actually-exhausted resource -- it's an agentic model that
  calls other sub-models internally, so its capacity silently collapses to near-zero exactly
  when the primary is already under load, i.e. exactly when overflow is needed most. Usage is
  *reserved* the moment a model is picked, before the request fires, so several triggers
  composed concurrently in one tick see each other's pending usage and spread across the pool
  instead of racing onto the same "available" model. Under an artificial worst-case burst (25
  triggers fired back-to-back with zero delay, far tighter than the real judge's 5-minutes-
  between-ticks cadence) this pool sustained 13/14 real LLM compositions; a paid/higher-tier
  key removes the ceiling entirely with no code change if still not enough.
- **URLs are never included**, even though the main brief says they're allowed when they add
  value — the testing brief's own failure-mode table (F.4) scores any URL as a hard fail
  (-3), so I optimized for the stricter, more operational rule.
- **`/v1/reply` for a `conversation_id` the bot never ticked** (e.g. a fresh replay-test
  scenario) is handled gracefully with whatever merchant/category context is in the store and
  no original trigger — heuristic classification (auto-reply/hostile/intent) doesn't need the
  trigger at all, only the free-form continuation does.
- **One send per merchant per tick** even though the spec allows more, to keep behavior
  conservative — restraint is explicitly rewarded, spam is explicitly penalized.
- Multi-turn state is in-memory only (per the testing brief's own guidance) — it does not
  survive a process restart, which is fine for a single bounded test window but would need
  Redis/a DB for a real multi-instance deployment.

## What additional context would have helped most

- **Real competitor data on `competitor_opened` triggers.** Most generated (non-seed)
  triggers of this kind carry a `{"placeholder": true}` payload with no actual competitor
  name/distance, so the composer has to fall back to the merchant's own peer-stat gap instead
  of the sharper "voyeur-curiosity" framing the design doc describes — a real payload would
  score meaningfully higher on trigger relevance for this kind.
- **A merchant-level `preferred_reply_language` observed from their own past messages**
  (rather than inferring purely from `identity.languages`) would remove ambiguity for
  merchants who list `["en","hi"]` but have only ever replied in English.
- **Explicit slot/availability data for non-recall booking kinds** (`appointment_tomorrow`,
  `unplanned_slot_open`) the way `recall_due` already provides `available_slots` — several
  generated triggers of these kinds also only carry placeholder payloads.

## Running locally

```bash
pip install -r requirements.txt
# create a local .env with GROQ_API_KEY=... (gitignored, never committed)
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Generate `submission.jsonl` for the 30 canonical test pairs:

```bash
python generate_submission.py --dataset-dir ../dataset/expanded --out submission.jsonl
```

Self-test against the provided judge simulator (from the repo root, with `dataset/` present):

```bash
export BOT_URL=http://localhost:8080
python judge_simulator.py
```

## Deployment

Deployed on Render (`render.yaml` included) as a standard Python web service:
`uvicorn bot:app --host 0.0.0.0 --port $PORT`. `GROQ_API_KEY` is set as a Render environment
variable, never committed.
