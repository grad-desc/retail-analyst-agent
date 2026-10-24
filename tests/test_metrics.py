from src.observability.metrics import compute_metrics, trace_timeline


def ev(event, trace="t1", ts="2026-01-01T10:00:00.000+00:00", **kw):
    return {"ts": ts, "level": "INFO", "event": event, "trace_id": trace, **kw}


def turn(trace, status="done", intent="analysis", attempts=1, rows=5, ts="2026-01-01T10:00:00.000+00:00", thread="th", q="q", **kw):
    return [ev("node_done", trace, ts, node="route", latency_ms=100), ev("node_done", trace, ts, node="write_report", latency_ms=900),
            ev("turn_done", trace, ts, status=status, intent=intent, attempts=attempts, rows=rows, thread_id=thread, question=q, **kw)]


def test_core_rates_and_latency():
    events = (turn("a") + turn("b", attempts=3) + turn("c", rows=0) + turn("d", status="failed", failure="sql_failed", attempts=3)
              + turn("e", status="blocked", intent="unsafe", attempts=0))
    m = compute_metrics(events)
    assert m["turns"] == 5 and m["success_rate"] == 0.6 and m["failure_rate"] == 0.2 and m["guard_block_rate"] == 0.2
    assert m["failures_by_code"] == {"sql_failed": 1} and m["retry_rate"] == 0.5          # 2 of 4 analysis turns retried
    assert m["empty_result_rate"] == 0.333 and m["latency_ms_p50"] == 1000 and m["latency_ms_p95"] == 1000


def test_llm_tokens_cost_fallback_and_failures():
    events = turn("a") + [ev("llm_call", model="m1", tokens_in=1000, tokens_out=500, fallback=False),
                          ev("llm_call", model="m2", tokens_in=1000, tokens_out=500, fallback=True),
                          ev("llm_call_failed", kind="retry")]
    m = compute_metrics(events, price_in_per_m=1.0, price_out_per_m=4.0)
    assert m["tokens_in"] == 2000 and m["tokens_out"] == 1000 and m["llm_fallback_rate"] == 0.5 and m["llm_call_failures"] == 1
    assert m["est_cost_usd"] == 0.006 and m["est_cost_per_turn_usd"] == 0.006 and m["llm_calls_per_turn"] == 2.0
    assert compute_metrics(events)["est_cost_usd"] is None                                  # never invents prices


def test_security_golden_pii_feedback_and_rephrase():
    events = (turn("a", golden_ids=["x"], q="revenue by category 2024", ts="2026-01-01T10:00:00.000+00:00")
              + turn("b", golden_ids=[], q="revenue by category for 2024", ts="2026-01-01T10:01:00.000+00:00")
              + turn("c", golden_ids=[], q="top brands by return rate", ts="2026-01-01T10:02:00.000+00:00",
                     guard={"output_redactions": {"email": 1}})
              + [ev("security_block", layer="sql_validator"), ev("security_block", layer="input_guard"),
                 ev("security_block", layer="sql_validator"), ev("feedback", rating=1), ev("feedback", rating=-1)])
    m = compute_metrics(events)
    assert m["golden_hit_rate"] == 0.333 and m["security_blocks"] == {"sql_validator": 2, "input_guard": 1}
    assert m["output_pii_redactions"] == 1 and m["thumbs_down_rate"] == 0.5
    assert m["rephrase_rate"] == 0.5                      # b rephrases a; c does not rephrase b


def test_empty_log_is_safe():
    m = compute_metrics([])
    assert m["turns"] == 0 and m["success_rate"] is None and m["latency_ms_p95"] is None


def test_trace_timeline_orders_events_and_hides_noise():
    events = [ev("turn_done", ts="2026-01-01T10:00:02.000+00:00", status="done", question="q", answer="a"),
              ev("sql_validated", ts="2026-01-01T10:00:01.000+00:00"), ev("other", trace="zzz"),
              ev("security_block", ts="2026-01-01T10:00:00.500+00:00", level="WARNING", layer="sql_validator", kind="pii_column")]
    lines = trace_timeline(events, "t1")
    assert [l.split()[2] for l in lines] == ["security_block", "sql_validated", "turn_done"]
    assert "kind=pii_column" in lines[0] and "trace_id" not in lines[0]
