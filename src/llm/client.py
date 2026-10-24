"""Resilient Gemini client: retry + backoff/jitter, model fallback chain, circuit breaker,
per-turn call budget. Errors are classified so each class gets the right mechanism:

  transient (429 rate, 5xx, timeout, empty reply) -> retry with backoff, then next model
  model unusable (404, 429 'limit: 0' / daily quota) -> skip to next model immediately
  fatal (400/401/403: bad key, bad request)          -> LLMRequestError, no retry (don't burn quota)
  everything exhausted                               -> LLMUnavailable (friendly message upstream)

The SDK is imported lazily and the call is injectable, so the logic is unit-testable offline.
"""
from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

# backend(model, prompt, system, json_mode, temperature, timeout_s) -> (text, tokens_in, tokens_out)
Backend = Callable[[str, str, Optional[str], bool, float, float], Tuple[str, int, int]]


class LLMError(Exception):
    pass


class LLMUnavailable(LLMError):
    """All models failed / circuit open. `detail` is for logs only, never shown to the user."""
    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail


class LLMRequestError(LLMError):
    """Non-retryable (bad API key, invalid request)."""


class BudgetExceeded(LLMError):
    """Per-turn LLM call budget is spent: stops runaway self-correction loops."""


class _EmptyResponse(Exception):
    code = None


@dataclass
class TurnBudget:
    max_calls: int = 8
    used: int = 0

    def spend(self) -> None:
        self.used += 1
        if self.used > self.max_calls:
            raise BudgetExceeded(f"LLM call budget ({self.max_calls}) exhausted for this turn")

    def refund(self) -> None:
        """Give back one unit for an attempt that cost nothing (the model refused before generating)."""
        self.used = max(0, self.used - 1)


@dataclass
class LLMResult:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    attempts: int          # API attempts spent on this generate() call
    fallback_used: bool    # answered by a non-primary model


class CircuitBreaker:
    """Per-model. Opens after `threshold` consecutive failures, or immediately on a *severe* one (a timeout).
    Half-opens after a cooldown that doubles every time a probe fails again (up to `max_cooldown_s`), so a
    model that stays down is probed rarely instead of costing a full timeout every minute."""

    def __init__(self, threshold: int = 3, cooldown_s: float = 30.0, max_cooldown_s: float = 600.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.threshold, self.cooldown_s, self.max_cooldown_s, self._clock = threshold, cooldown_s, max_cooldown_s, clock
        self._fails: Dict[str, int] = {}
        self._opened_at: Dict[str, float] = {}
        self._opens: Dict[str, int] = {}

    def _cooldown(self, model: str) -> float:
        return min(self.max_cooldown_s, self.cooldown_s * 2 ** max(0, self._opens.get(model, 1) - 1))

    def allow(self, model: str) -> bool:
        opened = self._opened_at.get(model)
        if opened is None:
            return True
        if self._clock() - opened >= self._cooldown(model):   # half-open: let one probe through
            del self._opened_at[model]
            self._fails[model] = self.threshold - 1           # one more failure re-opens it
            return True
        return False

    def success(self, model: str) -> None:
        for d in (self._fails, self._opened_at, self._opens):
            d.pop(model, None)

    def failure(self, model: str, severe: bool = False) -> None:
        self._fails[model] = self.threshold if severe else self._fails.get(model, 0) + 1
        if self._fails[model] >= self.threshold and model not in self._opened_at:
            self._opened_at[model] = self._clock()
            self._opens[model] = self._opens.get(model, 0) + 1


RETRY, SKIP_MODEL, FAILOVER, FATAL = "retry", "skip_model", "failover", "fatal"


def classify(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    msg = str(exc).lower()
    squashed = msg.replace("_", "").replace(" ", "")
    if code == 429:
        if "limit:0" in squashed or "perday" in squashed:
            return SKIP_MODEL
        return RETRY
    if code == 404:
        return SKIP_MODEL
    if code in (400, 401, 403):
        return FATAL
    if code == 504 or "deadline" in msg or "timed out" in msg or isinstance(exc, TimeoutError):
        return FAILOVER
    # Network errors: ConnectionError (builtin), httpx.ConnectError, OSError with network hints
    if isinstance(exc, ConnectionError):
        return FAILOVER
    if isinstance(exc, OSError) and ("network" in msg or "connection" in msg):
        return FAILOVER
    if type(exc).__name__ in ("ConnectError", "NetworkError", "RemoteProtocolError", "ReadTimeout", "ConnectTimeout"):
        return FAILOVER
    return RETRY


def _retry_hint_s(exc: Exception) -> Optional[float]:
    m = re.search(r"retry in ([\d.]+)s", str(exc), re.I)
    return float(m.group(1)) if m else None


def gemini_backend(api_key: str) -> Backend:
    from google import genai            # lazy: tests/CI don't need the SDK
    from google.genai import types

    client = genai.Client(api_key=api_key)

    def call(model, prompt, system, json_mode, temperature, timeout_s):
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),  # ms
        )
        r = client.models.generate_content(model=model, contents=prompt, config=cfg)
        u = r.usage_metadata
        t_in = getattr(u, "prompt_token_count", 0) or 0
        # thinking tokens are billed as output
        t_out = (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
        return (r.text or "", t_in, t_out)

    return call


class LLMClient:
    def __init__(self, models: Sequence[str], backend: Backend, *, timeout_s: float = 60,
                 max_attempts: int = 3, backoff_base_s: float = 1.5, backoff_cap_s: float = 20.0,
                 breaker: Optional[CircuitBreaker] = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not models:
            raise ValueError("at least one model required")
        self.models: List[str] = list(models)
        self._backend, self.timeout_s = backend, timeout_s
        self.max_attempts, self.backoff_base_s, self.backoff_cap_s = max_attempts, backoff_base_s, backoff_cap_s
        self.breaker = breaker or CircuitBreaker()
        self._sleep = sleep

    @classmethod
    def from_settings(cls, s) -> "LLMClient":
        return cls(s.model_chain, gemini_backend(s.gemini_api_key), timeout_s=s.llm_timeout_s,
                   max_attempts=s.llm_max_attempts)

    def with_model(self, model: str) -> "LLMClient":
        """Return a new client with the same backend but a different model.

        Useful for the LLM judge: same API key, same timeout, different model.
        """
        return LLMClient(
            [model],
            self._backend,
            timeout_s=self.timeout_s,
            max_attempts=self.max_attempts,
            backoff_base_s=self.backoff_base_s,
            backoff_cap_s=self.backoff_cap_s,
            breaker=self.breaker,
            sleep=self._sleep,
        )

    def _delay(self, attempt: int, exc: Exception) -> float:
        hint = _retry_hint_s(exc)
        base = min(self.backoff_cap_s, self.backoff_base_s * 2 ** (attempt - 1))
        return min(self.backoff_cap_s, hint) if hint else base * random.uniform(0.5, 1.0)

    def generate(self, prompt: str, *, system: Optional[str] = None, json_mode: bool = False,
                 temperature: float = 0.2, trace_id: Optional[str] = None,
                 budget: Optional[TurnBudget] = None) -> LLMResult:
        chain = [m for m in self.models if self.breaker.allow(m)]
        if not chain:
            raise LLMUnavailable("All models are temporarily unavailable (circuit open).",
                                 detail="circuit open for: " + ", ".join(self.models))
        attempts, errors = 0, []
        for model in chain:
            for attempt in range(1, self.max_attempts + 1):
                if budget:
                    budget.spend()
                attempts += 1
                t0 = time.perf_counter()
                try:
                    text, t_in, t_out = self._backend(model, prompt, system, json_mode,
                                                      temperature, self.timeout_s)
                    if not text.strip():
                        raise _EmptyResponse("empty response (possibly safety-blocked)")
                except Exception as e:  # noqa: BLE001 - classified below
                    kind = classify(e)
                    errors.append(f"{model}#{attempt}:{kind}:{type(e).__name__}:{str(e)[:120]}")
                    log.warning("llm_call_failed", extra={
                        "trace_id": trace_id, "model": model, "attempt": attempt, "kind": kind,
                        "error_type": type(e).__name__, "code": getattr(e, "code", None)})
                    if kind == FATAL:
                        raise LLMRequestError(str(e)[:300]) from e
                    if kind == SKIP_MODEL and budget:
                        budget.refund()   # quota 0 / unknown model: no tokens were spent, so no budget either
                    self.breaker.failure(model, severe=(kind == FAILOVER))   # a timeout trips the breaker at once
                    if (kind in (SKIP_MODEL, FAILOVER) or attempt == self.max_attempts
                            or not self.breaker.allow(model)):          # stop retrying a model the breaker just opened
                        break
                    self._sleep(self._delay(attempt, e))
                    continue
                latency = int((time.perf_counter() - t0) * 1000)
                self.breaker.success(model)
                log.info("llm_call", extra={"trace_id": trace_id, "model": model, "attempt": attempt,
                                            "latency_ms": latency, "tokens_in": t_in, "tokens_out": t_out,
                                            "fallback": model != self.models[0]})
                return LLMResult(text=text, model=model, tokens_in=t_in, tokens_out=t_out,
                                 latency_ms=latency, attempts=attempts,
                                 fallback_used=(model != self.models[0]))
        raise LLMUnavailable("The language model is not responding right now.",
                             detail=" | ".join(errors))