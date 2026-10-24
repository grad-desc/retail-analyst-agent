"""Per-user preference learning (format, depth, visuals): deterministic, cheap, inspectable.

Why no LLM extractor on every turn: it would add a model call (cost + latency) to each message. Signals
here come from explicit wording ("I prefer tables", "keep it short", "too long") via rules, so learning is
free, testable and explainable. A nuanced LLM extractor can run off the critical path in production.

Learning model: evidence-weighted with decay.
  * standing statement ("always", "from now on", "I prefer ...")  weight 3.0  -> takes effect at once
  * complaint in feedback ("too long")                             weight 1.0
  * one-off request inside a question ("as a table")               weight 0.6  -> honoured THIS turn, but
    only becomes a default after repeated use (0.6 alone is below the 1.0 threshold)
Each new observation decays older evidence for that dimension (x0.85), so preferences can change.
Users can see (/prefs) and wipe (/forget) everything that was learned.
"""
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

VALUES = {"format": ("table", "bullets", "prose"), "depth": ("brief", "standard", "deep"), "visuals": ("charts", "text")}
DECAY, THRESHOLD = 0.85, 1.0
W_STANDING, W_COMPLAINT, W_REQUEST = 3.0, 1.0, 0.6

INSTRUCTIONS = {
    ("format", "table"): "Present comparisons and rankings as a small table; keep surrounding prose minimal.",
    ("format", "bullets"): "Use short bullet points; avoid long paragraphs and avoid tables unless essential.",
    ("format", "prose"): "Write in short paragraphs; avoid bullet lists and tables.",
    ("depth", "brief"): "Be brief: lead with the answer, at most about 120 words, skip methodology.",
    ("depth", "standard"): "Use a normal level of detail.",
    ("depth", "deep"): "Go deeper: include drivers, comparisons, caveats and supporting numbers.",
    ("visuals", "charts"): "When comparing numbers, add a compact text bar chart (use the block character, scaled to the largest value).",
    ("visuals", "text"): "Do not include charts; use text and tables only.",
}
LABELS = {("format", "table"): "use tables", ("format", "bullets"): "use bullet points", ("format", "prose"): "write in paragraphs",
          ("depth", "brief"): "keep answers brief", ("depth", "standard"): "use a standard level of detail",
          ("depth", "deep"): "go into depth", ("visuals", "charts"): "include charts", ("visuals", "text"): "stay text-only"}

# complaints are matched first and removed from the text, so "too short" can never be read as "short"
_COMPLAINTS = [
    ("depth", "brief", r"\btoo (long|verbose|wordy|detailed)\b|\bshorter\b|\btoo much (text|detail)\b"),
    ("depth", "deep", r"\btoo (short|brief|shallow|thin|superficial)\b|\bnot enough detail\b|\bmore detail(s|ed)?\b|\bmore depth\b"),
]
_RULES = [
    ("visuals", "text", r"\b(no|without) (charts?|graphs?|plots?|visuals?)\b|\btext[- ]only\b"),
    # "table" is also a data term ("which tables do you have?"), so only presentation wording counts
    ("format", "table", r"\btabular\b|\b(as|in|into)( a| an)?( small| simple| summary| compact| comparison)? tables?\b"
                         r"|\b(prefer|like|want|use|using)( a)? tables?\b|\btables? (please|only|format)\b"
                         r"|\b(give|show) me (a|an) (\w+ )?table\b"),
    ("format", "bullets", r"\bbullets?( points?)?\b|\bbulleted\b"),
    ("format", "prose", r"\bparagraphs?\b|\bprose\b|\bnarrative\b|\bplain text\b|\bin sentences\b"),
    ("depth", "brief", r"\b(brief|concise|tl;?dr|one[- ]liner|high[- ]level|keep it short|keep (it|answers?) short|short answers?)\b"),
    ("depth", "deep", r"\bin[- ]depth\b|\bdeep(er)? dive\b|\bdetailed\b|\bthorough(ly)?\b|\bcomprehensive\b|\bdrill (down|into)\b"),
    ("visuals", "charts", r"\bcharts?\b|\bgraphs?\b|\bplots?\b|\bvisuali[sz]ations?\b"),
]
_STANDING = re.compile(r"\b(always|from now on|going forward|in future|by default|i prefer|i'd prefer|i would prefer|i like|i want you to|default to|stick to)\b", re.I)
_FORGET = re.compile(r"\b(forget|reset|clear|delete|erase)\b.{0,30}\b(preferences?|what you (learned|know|remember))\b", re.I)
_RECALL = re.compile(r"\bwhat (do you|have you).{0,25}(remember|learned|know).{0,20}(me|my|preferences?)|\bmy preferences\b", re.I)


@dataclass(frozen=True)
class Signal:
    key: str
    value: str
    kind: str          # "request" | "complaint"


def detect(text: str) -> List[Signal]:
    """Preference signals in a message. One signal per dimension (the last one mentioned wins)."""
    found: Dict[str, Signal] = {}
    work = text or ""
    for key, value, pat in _COMPLAINTS:
        if re.search(pat, work, re.I):
            found[key] = Signal(key, value, "complaint")
            work = re.sub(pat, " ", work, flags=re.I)
    for key, value, pat in _RULES:
        if key in found or not re.search(pat, work, re.I):
            continue                      # a complaint already decided this dimension
        if key == "visuals" and value == "charts" and found.get("visuals") is not None:
            continue                      # "no charts" already matched
        found[key] = Signal(key, value, "request")
    return list(found.values())


def is_standing(text: str) -> bool:
    return bool(_STANDING.search(text or ""))


def wants_forget(text: str) -> bool:
    return bool(_FORGET.search(text or ""))


def wants_recall(text: str) -> bool:
    return bool(_RECALL.search(text or ""))


def describe_signals(signals: Sequence[Signal]) -> str:
    return ", ".join(LABELS[(s.key, s.value)] for s in signals)


class PreferenceStore:
    def __init__(self, path=":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock, self._conn:
            self._conn.execute("CREATE TABLE IF NOT EXISTS prefs (subject TEXT, key TEXT, value TEXT, score REAL, "
                               "updated_at REAL, PRIMARY KEY (subject, key, value))")

    def observe(self, subject: str, signals: Sequence[Signal], standing: bool = False) -> None:
        with self._lock, self._conn:
            for s in signals:
                if s.value not in VALUES.get(s.key, ()):
                    continue
                w = W_STANDING if standing else (W_COMPLAINT if s.kind == "complaint" else W_REQUEST)
                self._conn.execute("UPDATE prefs SET score = score * ? WHERE subject=? AND key=?", (DECAY, subject, s.key))
                self._conn.execute(
                    "INSERT INTO prefs(subject, key, value, score, updated_at) VALUES (?,?,?,?,?) "
                    "ON CONFLICT(subject, key, value) DO UPDATE SET score = score + excluded.score, updated_at = excluded.updated_at",
                    (subject, s.key, s.value, w, time.time()))

    def current(self, subject: str) -> Dict[str, str]:
        with self._lock:
            rows = self._conn.execute("SELECT key, value, score FROM prefs WHERE subject=? ORDER BY score DESC, updated_at DESC",
                                      (subject,)).fetchall()
        out: Dict[str, str] = {}
        for key, value, score in rows:
            if key not in out and score >= THRESHOLD:
                out[key] = value
        return out

    def forget(self, subject: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM prefs WHERE subject=?", (subject,))

    def effective(self, subject: str, now: Sequence[Signal] = ()) -> Dict[str, str]:
        """Learned preferences, overridden by what the user asks for in the current message."""
        prefs = self.current(subject)
        for s in now:
            if s.value in VALUES.get(s.key, ()):
                prefs[s.key] = s.value
        return prefs

    def render(self, subject: str, now: Sequence[Signal] = ()) -> str:
        return "\n".join(f"- {INSTRUCTIONS[(k, v)]}" for k, v in self.effective(subject, now).items())

    def describe(self, subject: str) -> str:
        cur = self.current(subject)
        if not cur:
            return "I haven't learned any preferences yet. Tell me, for example, \"I prefer tables\" or \"keep it brief\"."
        return "Here is what I've learned: I " + ", ".join(LABELS[(k, v)] for k, v in cur.items()) + \
               ". Say \"forget my preferences\" to reset."
