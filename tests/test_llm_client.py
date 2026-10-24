import pytest

from src.llm.client import (BudgetExceeded, CircuitBreaker, LLMClient, LLMRequestError,
                            LLMUnavailable, TurnBudget)


class Err(Exception):
    def __init__(self, code, msg=""):
        super().__init__(msg or f"error {code}")
        self.code = code


def make(script, models=("pro", "flash"), **kw):
    """script: {model: [outcome, ...]}; outcome is str (text) or Exception. Records calls."""
    calls = []

    def backend(model, prompt, system, json_mode, temperature, timeout_s):
        calls.append(model)
        out = script[model].pop(0)
        if isinstance(out, Exception):
            raise out
        return (out, 10, 5)

    c = LLMClient(models, backend, sleep=lambda s: None, **kw)
    return c, calls


def test_retries_transient_then_succeeds_on_same_model():
    c, calls = make({"pro": [Err(503), Err(429), "ok"]})
    r = c.generate("hi")
    assert r.text == "ok" and r.model == "pro" and not r.fallback_used and calls == ["pro"] * 3


def test_quota_zero_skips_model_without_retrying():
    c, calls = make({"pro": [Err(429, "RESOURCE_EXHAUSTED limit: 0")], "flash": ["ok"]})
    r = c.generate("hi")
    assert calls == ["pro", "flash"] and r.fallback_used


def test_exhausted_retries_fall_back_to_next_model():
    c, calls = make({"pro": [Err(503)] * 3, "flash": ["ok"]})
    assert c.generate("hi").model == "flash" and calls == ["pro"] * 3 + ["flash"]


def test_fatal_error_not_retried():
    c, calls = make({"pro": [Err(403, "API key invalid")]})
    with pytest.raises(LLMRequestError):
        c.generate("hi")
    assert calls == ["pro"]


def test_all_models_down_raises_unavailable_with_detail():
    c, _ = make({"pro": [Err(503)] * 3, "flash": [Err(500)] * 3})
    with pytest.raises(LLMUnavailable) as e:
        c.generate("hi")
    assert "pro#" in e.value.detail


def test_empty_response_is_retried():
    c, calls = make({"pro": ["  ", "ok"]})
    assert c.generate("hi").text == "ok" and len(calls) == 2


def test_circuit_breaker_opens_and_stops_hitting_api():
    t = [0.0]
    br = CircuitBreaker(threshold=3, cooldown_s=60, max_cooldown_s=600, clock=lambda: t[0])
    c, calls = make({"pro": [Err(503)] * 3 + ["late"], "flash": ["a", "b"]}, breaker=br)

    c.generate("1")  # pro fails 3x -> breaker opens (cooldown=60)
    n = len(calls)
    c.generate("2")  # pro skipped entirely
    assert calls[n:] == ["flash"]

    # cooldown для pro = 60 * 2^(1-1) = 60s, значит через 61s можно пробовать
    t[0] = 61
    assert c.generate("3").model == "pro"


def test_turn_budget_caps_total_attempts():
    c, calls = make({"pro": [Err(503)] * 3, "flash": [Err(503)] * 3})
    with pytest.raises(BudgetExceeded):
        c.generate("hi", budget=TurnBudget(max_calls=4))
    assert len(calls) == 4


def test_skipped_model_does_not_consume_budget():
    c, calls = make({"pro": [Err(429, "RESOURCE_EXHAUSTED limit: 0")], "flash": ["ok"]})
    b = TurnBudget(max_calls=2)
    c.generate("hi", budget=b)
    assert calls == ["pro", "flash"] and b.used == 1      # only the call that really ran is charged


def test_httpx_connect_error_is_failover():
    """httpx.ConnectError is not a ConnectionError; it must still map to FAILOVER."""
    import httpx
    c, calls = make({
        "pro": [httpx.ConnectError("network unreachable")],
        "flash": [httpx.ConnectError("network unreachable")],
    })
    with pytest.raises(LLMUnavailable):
        c.generate("hi", budget=TurnBudget(max_calls=4))
    assert len(calls) == 2  # failover: 1 attempt per model
