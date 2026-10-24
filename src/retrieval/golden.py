"""Golden bucket: analyst-approved "trios" (question -> SQL -> report) used as few-shot context.

Prototype: YAML files on disk + in-memory TF-IDF retrieval (no API cost, no rate limits, fully offline).
Production: same interface backed by GCS + a hybrid dense/BM25 index (Vertex AI Vector Search or pgvector).

Trio content ends up inside LLM prompts, so it is treated as untrusted input at index time:
every trio must (1) have status "approved", (2) pass the SAME SQL validator the agent's own queries pass
(no PII columns, allowed tables, SELECT only), (3) contain no PII-looking text, and (4) contain no
prompt-injection phrases. Anything failing is quarantined with a reason, never silently indexed.
"""
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import yaml

from src.guards import input_guard
from src.guards.output_scrubber import scrub_text

log = logging.getLogger(__name__)

MAX_LEN = {"question": 500, "interpretation": 500, "sql": 4000, "report": 3000}
_ID = re.compile(r"[a-z0-9][a-z0-9_-]{2,63}")

_STOP = frozenset("""a an and are as at be by can could did do does for from get give has have how i in is it its me my
of on or our show tell that the their there these this those to us was we were what when where which who whom why will with
would you your please list find than less more over""".split())
_CANON = {
    "sales": "revenue", "sale": "revenue", "turnover": "revenue", "earnings": "revenue", "income": "revenue",
    "spending": "spend", "spent": "spend", "customers": "customer", "clients": "customer", "users": "customer",
    "buyers": "customer", "shoppers": "customer", "compared": "compare", "comparing": "compare",
    "comparison": "compare", "versus": "compare", "vs": "compare", "performing": "performance",
    "perform": "performance", "performed": "performance", "biggest": "top", "most": "top", "spender": "spend", "best": "top", "highest": "top",
    "largest": "top", "returned": "return", "returns": "return", "monthly": "month", "quarterly": "quarter",
    "q1": "quarter", "q2": "quarter", "q3": "quarter", "q4": "quarter", "yearly": "year", "annual": "year",
    "churn": "churn", "churned": "churn", "lost": "churn", "retention": "churn", "states": "state",
}


# Domain normalisation: a bare place name in a question means the geography dimension ("state").
# (Lexical retrieval limit; a dense-embedding backend would not need this.)
_STATE_NAMES = frozenset("""alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho
illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts michigan minnesota mississippi missouri montana
nebraska nevada ohio oklahoma oregon pennsylvania tennessee texas utah vermont virginia washington wisconsin wyoming""".split())
for _n in _STATE_NAMES:
    _CANON[_n] = "state"
for _w in ("underspending", "underspend", "underspent", "overspending"):
    _CANON[_w] = "spend"


def tokenize(text: str) -> List[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        w = _CANON.get(w, w)
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        w = _CANON.get(w, w)
        if w not in _STOP and len(w) > 1:
            out.append(w)
    return out


class TrioError(ValueError):
    """A trio was rejected; the message says why (shown to the curator)."""


@dataclass(frozen=True)
class Trio:
    id: str
    question: str
    sql: str
    report: str
    interpretation: str
    tags: Tuple[str, ...]
    path: str = ""


@dataclass(frozen=True)
class Hit:
    trio: Trio
    score: float


@dataclass(frozen=True)
class GoldenContext:
    sql_examples: str = ""
    report_examples: str = ""
    ids: Tuple[str, ...] = ()
    top_score: float = 0.0


def _default_validate(sql: str) -> None:
    from src.sql.policy import UserPolicy      # lazy: sqlglot only needed when indexing for real
    from src.sql.validator import validate_and_rewrite
    validate_and_rewrite(sql, UserPolicy.executive())


def parse_trio(data: dict, path: str = "", validate: Optional[Callable[[str], None]] = None) -> Trio:
    """Validate one trio dict; raises TrioError with the reason. Used by the loader AND by curation."""
    if not isinstance(data, dict):
        raise TrioError("not a mapping")
    if data.get("status") != "approved":
        raise TrioError(f"status is {data.get('status')!r}, only 'approved' trios are indexed")
    tid = str(data.get("id", ""))
    if not _ID.fullmatch(tid):
        raise TrioError("id must be 3-64 chars of a-z, 0-9, '-' or '_'")
    fields = {}
    for key in ("question", "sql"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise TrioError(f"missing {key}")
    for key, limit in MAX_LEN.items():
        val = data.get(key) or ""
        if not isinstance(val, str):
            raise TrioError(f"{key} must be text")
        if len(val) > limit:
            raise TrioError(f"{key} longer than {limit} chars")
        fields[key] = val.strip()
        # supply-chain defenses: content goes into prompts
        if key != "sql":
            if not scrub_text(val)[1].clean:
                raise TrioError(f"{key} contains PII-like text")
            hit = input_guard.check(val)
            if hit:
                raise TrioError(f"{key} contains an instruction-override phrase ({hit})")
    try:
        (validate or _default_validate)(fields["sql"])
    except Exception as e:  # noqa: BLE001 - any validator failure quarantines the trio
        raise TrioError(f"SQL rejected by validator: {e}") from e
    tags = tuple(str(t).lower() for t in (data.get("tags") or []) if isinstance(t, (str, int)))[:10]
    return Trio(tid, fields["question"], fields["sql"], fields["report"], fields["interpretation"], tags, path)


def _esc(text: str) -> str:
    return re.sub(r"</?\s*example[^>]*>", "", text, flags=re.I)   # example content can't close its own block


class GoldenIndex:
    def __init__(self, trios: Sequence[Trio], rejected: Sequence[Tuple[str, str]] = ()) -> None:
        self.trios: List[Trio] = list(trios)
        self.rejected: List[Tuple[str, str]] = list(rejected)
        docs = [self._doc_tokens(t) for t in self.trios]
        df = Counter(tok for d in docs for tok in set(d))
        n = len(docs)
        self._idf: Dict[str, float] = {tok: math.log((n + 1) / (c + 1)) + 1 for tok, c in df.items()}
        self._vecs = [self._vec(d) for d in docs]

    @staticmethod
    def _doc_tokens(t: Trio) -> List[str]:
        # question counts double; interpretation and tags add synonyms/intent
        return tokenize(t.question) * 2 + tokenize(t.interpretation) + tokenize(" ".join(t.tags))

    def _vec(self, tokens: Sequence[str]) -> Dict[str, float]:
        tf = Counter(t for t in tokens if t in self._idf)
        v = {t: c * self._idf[t] for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    @classmethod
    def load(cls, directory, validate: Optional[Callable[[str], None]] = None) -> "GoldenIndex":
        trios, rejected, seen = [], [], set()
        for path in sorted(Path(directory).glob("*.y*ml")):
            try:
                trio = parse_trio(yaml.safe_load(path.read_text(encoding="utf-8")), str(path), validate)
                if trio.id in seen:
                    raise TrioError(f"duplicate id {trio.id}")
                seen.add(trio.id)
                trios.append(trio)
            except (TrioError, yaml.YAMLError, OSError) as e:
                rejected.append((path.name, str(e)[:200]))
                log.warning("golden_trio_rejected", extra={"file": path.name, "reason": str(e)[:200]})
        log.info("golden_index_built", extra={"trios": len(trios), "rejected": len(rejected)})
        return cls(trios, rejected)

    def search(self, question: str, k: int = 3, min_score: float = 0.2) -> List[Hit]:
        q = self._vec(tokenize(question))
        scored = [Hit(t, sum(w * v.get(tok, 0.0) for tok, w in q.items())) for t, v in zip(self.trios, self._vecs)]
        return sorted((h for h in scored if h.score >= min_score), key=lambda h: -h.score)[:k]

    def similar(self, question: str, threshold: float = 0.8) -> Optional[Hit]:
        """Nearest existing trio if it is at least `threshold` similar (duplicate gate for curation)."""
        hits = self.search(question, k=1, min_score=threshold)
        return hits[0] if hits else None

    def examples_for(self, question: str, k: int = 3, min_score: float = 0.2) -> GoldenContext:
        hits = self.search(question, k, min_score)
        if not hits:
            return GoldenContext()
        sql_blocks, report_blocks = [], []
        for h in hits:
            t = h.trio
            note = f"Analyst interpretation: {_esc(t.interpretation)}\n" if t.interpretation else ""
            sql_blocks.append(f'<example id="{t.id}">\nQuestion: {_esc(t.question)}\n{note}SQL:\n{_esc(t.sql)}\n</example>')
            if t.report:
                report_blocks.append(f'<example id="{t.id}">\nQuestion: {_esc(t.question)}\nReport:\n{_esc(t.report)}\n</example>')
        return GoldenContext("\n".join(sql_blocks), "\n".join(report_blocks),
                             tuple(h.trio.id for h in hits), hits[0].score)
