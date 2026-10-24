"""Feedback capture (thumbs up/down) and the system-level learning loop.

  thumbs up   -> stored; if it was a successful analysis, the ORIGINAL (unscoped) SQL is queued as a golden
                 CANDIDATE for analyst review (never indexed automatically). Customer-role turns contribute the
                 question and SQL only: their personal figures never enter the shared golden set.
  thumbs down -> stored with the (PII-scrubbed) comment; the comment can teach the user's preferences
                 ("too long") and the row feeds the eval set / failure review (`failures()`).
Everything is keyed by trace_id, so any rating can be traced back to the full turn in the logs.
"""
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.guards.output_scrubber import scrub_text
from src.memory import preferences as prefs_mod
from src.retrieval.curation import add_candidate
from src.sql.policy import subject_of

log = logging.getLogger(__name__)
ELIGIBLE_INTENTS = {"analysis", "report"}


@dataclass
class FeedbackResult:
    saved: bool
    message: str
    candidate: Optional[str] = None
    learned: List[str] = field(default_factory=list)


class FeedbackStore:
    def __init__(self, path=":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, subject TEXT, role TEXT, "
                "trace_id TEXT, thread_id TEXT, rating INTEGER, comment TEXT, question TEXT, answer TEXT, sql TEXT, "
                "intent TEXT, status TEXT, golden_ids TEXT, UNIQUE(subject, trace_id))")

    def upsert(self, **r: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO feedback(ts, subject, role, trace_id, thread_id, rating, comment, question, answer, sql, intent, status, golden_ids) "
                "VALUES (:ts,:subject,:role,:trace_id,:thread_id,:rating,:comment,:question,:answer,:sql,:intent,:status,:golden_ids) "
                "ON CONFLICT(subject, trace_id) DO UPDATE SET rating=excluded.rating, comment=excluded.comment, ts=excluded.ts",
                {**r, "ts": time.time(), "golden_ids": json.dumps(r.get("golden_ids") or [])})

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) n, COALESCE(SUM(rating>0),0) up, COALESCE(SUM(rating<0),0) down FROM feedback").fetchone()
        n = row["n"]
        return {"total": n, "up": row["up"], "down": row["down"], "down_rate": round(row["down"] / n, 3) if n else 0.0}

    def failures(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Thumbs-down turns: raw material for the eval set and the weekly failure review."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM feedback WHERE rating<0 ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(r), "golden_ids": json.loads(r["golden_ids"])} for r in rows]


def record_feedback(store: FeedbackStore, prefs: prefs_mod.PreferenceStore, golden_dir, state: Dict[str, Any],
                    rating: int, comment: str = "") -> FeedbackResult:
    if rating not in (1, -1):
        raise ValueError("rating must be 1 or -1")
    if not state.get("trace_id") or not state.get("question"):
        return FeedbackResult(False, "Nothing to rate yet: ask a question first.")
    principal = state.get("principal") or {}
    subject, role = subject_of(principal), principal.get("role")
    comment = scrub_text((comment or "").strip()[:500])[0]
    store.upsert(subject=subject, role=role, trace_id=state["trace_id"], thread_id=state.get("thread_id"),
                 rating=rating, comment=comment, question=scrub_text(state.get("question", ""))[0],
                 answer=scrub_text(state.get("answer", ""))[0], sql=state.get("sql"), intent=state.get("intent"),
                 status=state.get("status"), golden_ids=state.get("golden_ids"))
    candidate, learned = None, []
    if (rating > 0 and state.get("intent") in ELIGIBLE_INTENTS and state.get("status") == "done"
            and state.get("sql") and state.get("executed_sql")):
        path = add_candidate(
            golden_dir, question=state.get("resolved_question") or state["question"], sql=state["sql"],
            report=state.get("answer", "") if role == "executive" else "",   # customer figures stay out of the shared set
            source_trace_id=state["trace_id"], meta={"source_role": role, "golden_ids_used": state.get("golden_ids") or []})
        candidate = path.stem
    if rating < 0 and comment:
        sigs = prefs_mod.detect(comment)
        if sigs:
            prefs.observe(subject, sigs)
            learned = [prefs_mod.describe_signals([s]) for s in sigs]
    log.info("feedback", extra={"trace_id": state["trace_id"], "rating": rating, "has_comment": bool(comment),
                                "candidate": candidate, "learned": learned, "intent": state.get("intent")})
    if rating > 0:
        msg = ("Thanks! I've queued this query for analyst review." if candidate else "Thanks for the feedback!")
    else:
        msg = "Thanks, noted." + (f" I'll {', '.join(learned)} from now on." if learned else "")
    return FeedbackResult(True, msg, candidate, learned)
