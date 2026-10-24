"""Pure helpers for the destructive-action flow: criteria parsing, preview text, confirmation parsing."""
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence, Tuple

UNDO_WINDOW_S = 600       # soft-deleted reports can be restored for 10 minutes
CONFIRM_TTL_S = 300       # a pending confirmation expires after 5 minutes
MAX_PREVIEW = 10

_YES = {"yes", "y", "yep", "yeah", "confirm", "confirmed", "yes delete", "yes delete them", "delete",
        "delete them", "go ahead", "do it", "proceed", "ok delete",
        "yes please", "yes go ahead", "yes proceed", "yes do it", "yes confirm", "yes please delete them",
        "please delete", "please delete them", "confirm delete"}
_NO = {"no", "n", "nope", "cancel", "stop", "abort", "never mind", "nevermind", "dont", "do not",
       "no dont delete", "no cancel"}


def parse_confirmation(text: str) -> Optional[bool]:
    """True / False for an explicit answer, None for anything else. Deliberately strict string
    equality (no LLM): an ambiguous reply is never treated as consent."""
    t = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", (text or "").lower())).strip()
    if t in _YES:
        return True
    if t in _NO:
        return False
    return None


@dataclass(frozen=True)
class Criteria:
    ids: Tuple[int, ...] = ()
    mentions: Tuple[str, ...] = ()
    scope: Optional[str] = None       # "conversation" | "all" | None

    @property
    def is_specific(self) -> bool:
        return bool(self.ids or self.mentions or self.scope in ("conversation", "all"))

    def describe(self) -> str:
        parts = []
        if self.ids:
            parts.append("with id " + ", ".join(f"#{i}" for i in self.ids))
        if self.mentions:
            parts.append("mentioning " + " or ".join(f'"{m}"' for m in self.mentions))
        if self.scope == "conversation":
            parts.append("created in this conversation")
        if self.scope == "all" and not parts:
            parts.append("(all of your saved reports)")
        return " ".join(parts)


def parse_criteria(parsed: dict) -> Criteria:
    """Sanitise LLM-extracted criteria. The LLM only *describes* what to match; owner scoping and the
    actual matching happen in code."""
    ids, seen = [], set()
    for x in (parsed.get("report_ids") or [])[:50]:
        if isinstance(x, bool):
            continue
        if isinstance(x, str) and x.strip().isdigit():
            x = int(x.strip())
        if isinstance(x, int) and x > 0 and x not in seen:
            seen.add(x)
            ids.append(x)
    raw = parsed.get("mentions") or []
    mentions = tuple(m.strip()[:100] for m in raw[:5] if isinstance(m, str) and len(m.strip()) >= 2)
    scope = parsed.get("scope") if parsed.get("scope") in ("conversation", "all") else None
    return Criteria(tuple(ids), mentions, scope)


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def format_listing(reports: Sequence) -> str:
    if not reports:
        return "You have no saved reports yet. Ask me for a report and I'll save it here."
    lines = [f"- **#{r.id}** · {r.title} · {_fmt_ts(r.created_at)}" for r in reports]
    return f"Your saved reports ({len(reports)}):\n\n" + "\n".join(lines)


def format_preview(reports: Sequence, criteria: Criteria) -> str:
    n = len(reports)
    shown = [f"- **#{r.id}** · {r.title} · {_fmt_ts(r.created_at)}" for r in reports[:MAX_PREVIEW]]
    more = f"\n- ...and {n - MAX_PREVIEW} more" if n > MAX_PREVIEW else ""
    return (f"I found **{n} report{'s' if n != 1 else ''}** {criteria.describe()}:\n\n" + "\n".join(shown) + more +
            f"\n\nThis will delete {'them' if n != 1 else 'it'}; you can say \"undo\" for the next "
            f"{UNDO_WINDOW_S // 60} minutes. Reply **yes** to confirm or **no** to cancel.")
