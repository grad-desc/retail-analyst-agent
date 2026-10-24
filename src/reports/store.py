"""Saved-reports library (SQLite for the prototype; Firestore/Postgres in production).

Safety properties enforced HERE, in code, not in prompts:
  * every read/write is scoped by `owner`: other people's reports are physically unreachable
  * deletes are soft (recoverable inside an undo window) and every action lands in an append-only audit log
  * search is done in Python on the owner's own rows, so user text is never interpolated into SQL
"""
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    owner        TEXT NOT NULL,
    thread_id    TEXT,
    title        TEXT NOT NULL,
    question     TEXT NOT NULL,
    content      TEXT NOT NULL,
    created_at   REAL NOT NULL,
    deleted_at   REAL,
    delete_batch TEXT
);
CREATE INDEX IF NOT EXISTS idx_reports_owner ON reports(owner, deleted_at);
CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    owner      TEXT NOT NULL,
    action     TEXT NOT NULL,
    report_ids TEXT NOT NULL,
    detail     TEXT NOT NULL,
    trace_id   TEXT
);
"""


@dataclass(frozen=True)
class Report:
    id: int
    owner: str
    thread_id: Optional[str]
    title: str
    question: str
    content: str
    created_at: float
    deleted_at: Optional[float] = None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


class ReportStore:
    def __init__(self, path=":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    @staticmethod
    def _row(r: sqlite3.Row) -> Report:
        return Report(r["id"], r["owner"], r["thread_id"], r["title"], r["question"], r["content"],
                      r["created_at"], r["deleted_at"])

    # ------------------------------------------------------------------ writes
    def save(self, owner: str, thread_id: Optional[str], title: str, question: str, content: str,
             now: Optional[float] = None) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO reports(owner, thread_id, title, question, content, created_at) VALUES (?,?,?,?,?,?)",
                (owner, thread_id, title, question, content, now or time.time()))
            return int(cur.lastrowid)

    def soft_delete(self, owner: str, ids: Sequence[int], batch: str, now: Optional[float] = None) -> List[int]:
        """Marks the owner's still-live reports among `ids` as deleted. Returns the ids actually deleted."""
        ids = [int(i) for i in ids]
        if not ids:
            return []
        with self._lock, self._conn:
            live = [r["id"] for r in self._conn.execute(
                f"SELECT id FROM reports WHERE owner=? AND deleted_at IS NULL AND id IN ({_placeholders(len(ids))})",
                [owner, *ids])]
            if live:
                self._conn.execute(
                    f"UPDATE reports SET deleted_at=?, delete_batch=? WHERE owner=? AND id IN ({_placeholders(len(live))})",
                    [now or time.time(), batch, owner, *live])
            return sorted(live)

    def restore_last_batch(self, owner: str, window_s: float, now: Optional[float] = None) -> List[int]:
        """Undo the owner's most recent delete if it happened within the window."""
        now = now or time.time()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT delete_batch FROM reports WHERE owner=? AND deleted_at IS NOT NULL AND deleted_at>=? "
                "ORDER BY deleted_at DESC LIMIT 1", (owner, now - window_s)).fetchone()
            if not row:
                return []
            ids = [r["id"] for r in self._conn.execute(
                "SELECT id FROM reports WHERE owner=? AND delete_batch=?", (owner, row["delete_batch"]))]
            self._conn.execute("UPDATE reports SET deleted_at=NULL, delete_batch=NULL WHERE owner=? AND delete_batch=?",
                               (owner, row["delete_batch"]))
            return sorted(ids)

    def purge_expired(self, retention_s: float = 7 * 86400, now: Optional[float] = None) -> int:
        """Hard-delete reports soft-deleted longer ago than the retention period."""
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM reports WHERE deleted_at IS NOT NULL AND deleted_at<?",
                                     ((now or time.time()) - retention_s,))
            return cur.rowcount

    # ------------------------------------------------------------------- reads
    def list(self, owner: str, limit: int = 50) -> List[Report]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM reports WHERE owner=? AND deleted_at IS NULL ORDER BY created_at DESC, id DESC LIMIT ?",
                (owner, limit)).fetchall()
        return [self._row(r) for r in rows]

    def get(self, owner: str, report_id: int) -> Optional[Report]:
        with self._lock:
            r = self._conn.execute("SELECT * FROM reports WHERE owner=? AND id=? AND deleted_at IS NULL",
                                   (owner, int(report_id))).fetchone()
        return self._row(r) if r else None

    def find(self, owner: str, *, ids: Optional[Sequence[int]] = None, mentions: Optional[Sequence[str]] = None,
             thread_id: Optional[str] = None, match_all: bool = False) -> List[Report]:
        """Filters combine with AND; `mentions` is any-of (case-insensitive, over title+question+content).
        With no filter and match_all=False nothing is returned: a delete can never silently mean 'everything'."""
        if not (ids or mentions or thread_id or match_all):
            return []
        sql, args = "SELECT * FROM reports WHERE owner=? AND deleted_at IS NULL", [owner]
        if ids:
            sql += f" AND id IN ({_placeholders(len(ids))})"
            args += [int(i) for i in ids]
        if thread_id:
            sql += " AND thread_id=?"
            args.append(thread_id)
        with self._lock:
            rows = [self._row(r) for r in self._conn.execute(sql + " ORDER BY created_at DESC, id DESC", args)]
        if mentions:
            needles = [_norm(m) for m in mentions if _norm(m)]
            rows = [r for r in rows if any(n in _norm(f"{r.title} {r.question} {r.content}") for n in needles)]
        return rows

    # ------------------------------------------------------------------- audit
    def audit(self, owner: str, action: str, report_ids: Sequence[int], detail: Optional[dict] = None,
              trace_id: Optional[str] = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO audit_log(ts, owner, action, report_ids, detail, trace_id) VALUES (?,?,?,?,?,?)",
                (time.time(), owner, action, json.dumps(list(report_ids)), json.dumps(detail or {}), trace_id))

    def audit_log(self, owner: Optional[str] = None, limit: int = 100) -> List[dict]:
        with self._lock:
            q, a = "SELECT * FROM audit_log", []
            if owner:
                q, a = q + " WHERE owner=?", [owner]
            rows = self._conn.execute(q + " ORDER BY id DESC LIMIT ?", [*a, limit]).fetchall()
        return [{**dict(r), "report_ids": json.loads(r["report_ids"]), "detail": json.loads(r["detail"])} for r in rows]
