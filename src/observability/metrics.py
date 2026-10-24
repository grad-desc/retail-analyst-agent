"""Agent-level metrics and per-trace debugging, computed from the structured JSON log (logs/agent.jsonl).

In production the same events stream to BigQuery (log sink) and these become dashboard queries; here they are
computed in-process so the prototype needs no extra infrastructure.
"""
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ANALYSIS = {"analysis", "report"}
_WORD = re.compile(r"[a-z0-9]+")


def load_events(path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue                      # a torn line must never break reporting
    return out


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, math.ceil(q * len(s)) - 1)], 1)


def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 3) if den else None


def _tokens(text: str) -> set:
    return {w for w in _WORD.findall((text or "").lower()) if len(w) > 2}


def _is_rephrase(a: str, b: str, threshold: float = 0.6) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= threshold


def compute_metrics(events: Iterable[Dict[str, Any]], price_in_per_m: Optional[float] = None,
                    price_out_per_m: Optional[float] = None) -> Dict[str, Any]:
    events = list(events)
    by_trace: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in events:
        if e.get("trace_id"):
            by_trace[e["trace_id"]].append(e)
    turns = [e for e in events if e.get("event") == "turn_done"]
    n = len(turns)
    analysis = [t for t in turns if t.get("intent") in ANALYSIS]
    done_analysis = [t for t in analysis if t.get("status") == "done"]

    latencies = [sum(x.get("latency_ms", 0) for x in by_trace[t["trace_id"]] if x.get("event") == "node_done")
                 for t in turns]
    llm_calls = [e for e in events if e.get("event") == "llm_call"]
    tokens_in = sum(e.get("tokens_in", 0) for e in llm_calls)
    tokens_out = sum(e.get("tokens_out", 0) for e in llm_calls)
    cost = None
    if price_in_per_m is not None and price_out_per_m is not None:
        cost = round(tokens_in / 1e6 * price_in_per_m + tokens_out / 1e6 * price_out_per_m, 6)
    bq = [e for e in events if e.get("event") == "bq_query_done"]
    feedback = [e for e in events if e.get("event") == "feedback"]

    # UX: how often does a user immediately rephrase the previous question in the same thread?
    threads: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in analysis:
        if t.get("thread_id"):
            threads[t["thread_id"]].append(t)
    pairs = rephrased = 0
    for seq in threads.values():
        seq.sort(key=lambda x: x.get("ts", ""))
        for a, b in zip(seq, seq[1:]):
            pairs += 1
            rephrased += _is_rephrase(a.get("question", ""), b.get("question", ""))

    return {
        "turns": n,
        "success_rate": _rate(sum(t.get("status") == "done" for t in turns), n),
        "failure_rate": _rate(sum(t.get("status") == "failed" for t in turns), n),
        "guard_block_rate": _rate(sum(t.get("status") == "blocked" for t in turns), n),
        "failures_by_code": dict(Counter(t["failure"] for t in turns if t.get("failure"))),
        "intents": dict(Counter(t.get("intent") for t in turns if t.get("intent"))),
        "retry_rate": _rate(sum((t.get("attempts") or 0) > 1 for t in analysis), len(analysis)),
        "avg_sql_attempts": round(sum(t.get("attempts") or 0 for t in analysis) / len(analysis), 2) if analysis else None,
        "empty_result_rate": _rate(sum((t.get("rows") or 0) == 0 for t in done_analysis), len(done_analysis)),
        "golden_hit_rate": _rate(sum(bool(t.get("golden_ids")) for t in analysis), len(analysis)),
        "latency_ms_p50": _pct(latencies, 0.5), "latency_ms_p95": _pct(latencies, 0.95),
        "llm_calls_per_turn": round(len(llm_calls) / n, 2) if n else None,
        "llm_call_failures": sum(e.get("event") == "llm_call_failed" for e in events),
        "llm_fallback_rate": _rate(sum(bool(e.get("fallback")) for e in llm_calls), len(llm_calls)),
        "tokens_in": tokens_in, "tokens_out": tokens_out, "est_cost_usd": cost,
        "est_cost_per_turn_usd": round(cost / n, 6) if cost is not None and n else None,
        "bq_queries": len(bq), "bq_bytes_billed_total": sum(e.get("bytes_billed") or 0 for e in bq),
        "bq_truncated_results": sum(bool(e.get("truncated")) for e in bq),
        "security_blocks": dict(Counter(e.get("layer") for e in events if e.get("event") == "security_block")),
        # >0 means an upstream layer let PII through and the last-line output guard caught it: investigate
        "output_pii_redactions": sum(bool((t.get("guard") or {}).get("output_redactions")) for t in turns),
        "feedback_total": len(feedback), "thumbs_down_rate": _rate(sum(e.get("rating") == -1 for e in feedback), len(feedback)),
        "rephrase_rate": _rate(rephrased, pairs),
    }


_HIDE = {"ts", "level", "logger", "event", "trace_id"}


def trace_timeline(events: Iterable[Dict[str, Any]], trace_id: str) -> List[str]:
    """Human-readable story of one turn: router decision, each SQL attempt and why it failed, LLM calls, guards."""
    lines = []
    for e in sorted((x for x in events if x.get("trace_id") == trace_id), key=lambda x: x.get("ts", "")):
        details = " ".join(f"{k}={json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}"
                           for k, v in e.items() if k not in _HIDE and v not in (None, "", [], {}))
        lines.append(f"{e.get('ts', '')[11:23]}  {e.get('level', ''):<7} {e.get('event', '')}  {details}"[:600])
    return lines


def filter_since(events: Iterable[Dict[str, Any]], hours: Optional[float]) -> List[Dict[str, Any]]:
    from datetime import datetime, timedelta, timezone
    events = list(events)
    if not hours:
        return events
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    return [e for e in events if e.get("ts", "") >= cutoff]
