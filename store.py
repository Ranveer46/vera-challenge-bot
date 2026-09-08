"""In-memory, thread-safe state for the Vera challenge bot.

Holds every context the judge has pushed (versioned, idempotent), plus
per-conversation state used by the reply/tick handlers. Everything lives
in-process for the duration of the test window, per the testing brief.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    trigger_id: str | None = None
    send_as: str = "vera"
    mode: str = "pitch"  # "pitch" | "action"
    turns: list = field(default_factory=list)  # [{"from": "vera"/"merchant", "message": str, "ts": str}]
    sent_bodies: list = field(default_factory=list)
    auto_reply_streak: int = 0
    sends_without_reply: int = 0
    ended: bool = False
    wait_until_epoch: float = 0.0
    created_at: str = field(default_factory=now_iso)

    def record_bot_send(self, body: str) -> None:
        self.turns.append({"from": "vera", "message": body, "ts": now_iso()})
        if body:
            self.sent_bodies.append(body)
        self.sends_without_reply += 1

    def record_inbound(self, message: str) -> None:
        self.turns.append({"from": "merchant", "message": message, "ts": now_iso()})
        self.sends_without_reply = 0

    def history_text(self, max_turns: int = 8) -> str:
        lines = []
        for t in self.turns[-max_turns:]:
            lines.append(f"{t['from']}: {t['message']}")
        return "\n".join(lines)


class Store:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._contexts: dict[tuple[str, str], dict] = {}
        self._conversations: dict[str, ConversationState] = {}
        self._sent_suppression_keys: dict[str, float] = {}
        self._merchant_suppressed_until: dict[str, float] = {}
        self.started_at = time.time()

    # ---------- contexts ----------

    def put_context(self, scope: str, context_id: str, version: int, payload: dict):
        with self._lock:
            key = (scope, context_id)
            cur = self._contexts.get(key)
            if cur and cur["version"] >= version:
                return False, cur["version"]
            self._contexts[key] = {"version": version, "payload": payload}
            return True, version

    def get_context(self, scope: str, context_id: str) -> dict | None:
        with self._lock:
            entry = self._contexts.get((scope, context_id))
            return entry["payload"] if entry else None

    def counts(self) -> dict:
        with self._lock:
            counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
            for (scope, _cid) in self._contexts:
                counts[scope] = counts.get(scope, 0) + 1
            return counts

    def get_category_for_merchant(self, merchant: dict) -> dict | None:
        slug = merchant.get("category_slug") or merchant.get("identity", {}).get("category_slug")
        return self.get_context("category", slug) if slug else None

    # ---------- conversations ----------

    def get_or_create_conversation(self, conversation_id: str, **defaults) -> ConversationState:
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                conv = ConversationState(conversation_id=conversation_id, **defaults)
                self._conversations[conversation_id] = conv
            return conv

    def get_conversation(self, conversation_id: str) -> ConversationState | None:
        with self._lock:
            return self._conversations.get(conversation_id)

    # ---------- suppression ----------

    def is_suppression_key_sent(self, suppression_key: str) -> bool:
        with self._lock:
            return suppression_key in self._sent_suppression_keys

    def mark_suppression_key_sent(self, suppression_key: str) -> None:
        with self._lock:
            self._sent_suppression_keys[suppression_key] = time.time()

    def is_merchant_suppressed(self, merchant_id: str) -> bool:
        with self._lock:
            until = self._merchant_suppressed_until.get(merchant_id, 0)
            return time.time() < until

    def suppress_merchant(self, merchant_id: str, days: int = 30) -> None:
        with self._lock:
            self._merchant_suppressed_until[merchant_id] = time.time() + days * 86400

    # ---------- teardown ----------

    def wipe(self) -> None:
        with self._lock:
            self._contexts.clear()
            self._conversations.clear()
            self._sent_suppression_keys.clear()
            self._merchant_suppressed_until.clear()


STORE = Store()
