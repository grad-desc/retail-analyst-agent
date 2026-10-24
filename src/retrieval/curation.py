"""Golden-bucket update loop: candidate -> human review -> gated promotion.

Nothing a user or the agent produces ever goes straight into the golden set:
  add_candidate  : the agent (e.g. after a thumbs-up) writes golden/candidates/<id>.yaml, status=candidate (NOT indexed)
  promote        : an analyst approves it; it must pass the same trio validation as seed data AND a
                   near-duplicate check, then moves to golden/trios/ with reviewer + timestamp
  reject         : moved to golden/rejected/ with the reason (kept for audit and as eval negatives)
"""
import hashlib
import re
import time
from pathlib import Path
from typing import Callable, Optional

import yaml

from src.guards.output_scrubber import scrub_text
from src.retrieval.golden import GoldenIndex, TrioError, parse_trio


class CurationError(Exception):
    pass


def _str_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|" if "\n" in data else None)


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(str, _str_presenter)


def dump_trio(data: dict) -> str:
    return yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)


def _slug(question: str) -> str:
    words = re.findall(r"[a-z0-9]+", question.lower())[:6]
    return "-".join(words) or "trio"


def add_candidate(golden_dir, *, question: str, sql: str, report: str, interpretation: str = "",
                  source_trace_id: Optional[str] = None, feedback: str = "thumbs_up",
                  meta: Optional[dict] = None) -> Path:
    cdir = Path(golden_dir) / "candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    question = scrub_text(question)[0]
    cid = f"cand-{hashlib.sha1((question + sql).encode()).hexdigest()[:10]}"
    data = {"id": cid, "status": "candidate", "question": question, "interpretation": interpretation,
            "tags": [], "sql": sql, "report": scrub_text(report)[0], "source_trace_id": source_trace_id,
            "feedback": feedback, "created_at": int(time.time()), **(meta or {})}
    path = cdir / f"{cid}.yaml"
    path.write_text(dump_trio(data), encoding="utf-8")
    return path


def promote(golden_dir, candidate_id: str, reviewer: str, *, validate: Optional[Callable[[str], None]] = None,
            index: Optional[GoldenIndex] = None, dup_threshold: float = 0.8,
            allow_duplicate: bool = False, new_id: Optional[str] = None) -> Path:
    gdir = Path(golden_dir)
    src = gdir / "candidates" / f"{candidate_id}.yaml"
    if not src.exists():
        raise CurationError(f"candidate {candidate_id} not found")
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    tid = new_id or f"{_slug(data.get('question', ''))}-{hashlib.sha1(candidate_id.encode()).hexdigest()[:4]}"
    data.update(id=tid, status="approved", approved_by=reviewer, approved_at=int(time.time()))
    try:
        trio = parse_trio(data, validate=validate)            # same gate as seed data
    except TrioError as e:
        raise CurationError(f"rejected by validation: {e}") from e
    index = index or GoldenIndex.load(gdir / "trios", validate)
    dup = index.similar(trio.question, dup_threshold)
    if dup and not allow_duplicate:
        raise CurationError(f"near-duplicate of existing trio '{dup.trio.id}' (similarity {dup.score:.2f})")
    dest = gdir / "trios" / f"{tid}.yaml"
    if dest.exists():
        raise CurationError(f"{dest.name} already exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(dump_trio({k: v for k, v in data.items() if k not in ("feedback",)}), encoding="utf-8")
    src.unlink()
    return dest


def reject(golden_dir, candidate_id: str, reviewer: str, reason: str) -> Path:
    gdir = Path(golden_dir)
    src = gdir / "candidates" / f"{candidate_id}.yaml"
    if not src.exists():
        raise CurationError(f"candidate {candidate_id} not found")
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    data.update(status="rejected", rejected_by=reviewer, reject_reason=reason[:300])
    dest = gdir / "rejected" / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(dump_trio(data), encoding="utf-8")
    src.unlink()
    return dest


def list_candidates(golden_dir) -> list:
    out = []
    for p in sorted((Path(golden_dir) / "candidates").glob("*.yaml")):
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        out.append({"id": d.get("id"), "question": d.get("question"), "feedback": d.get("feedback")})
    return out
