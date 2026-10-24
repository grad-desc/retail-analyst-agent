"""Settings: the shipped example config must give a real fallback chain, and the knobs must really be knobs."""
from pathlib import Path

import pytest

from src.config import Settings

KNOBS = ("GEMINI_FALLBACK_MODELS", "LLM_MAX_ATTEMPTS", "MAX_LLM_CALLS_PER_TURN", "GEMINI_JUDGE_MODEL")


def base_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)       # a developer's own .env must not change these tests


def test_default_fallback_is_a_different_model(monkeypatch):
    base_env(monkeypatch)
    s = Settings.from_env()
    assert len(s.model_chain) == 2 and s.model_chain[0] == "gemini-3.5-flash"
    assert s.llm_max_attempts == 2 and s.max_llm_calls_per_turn == 8


def test_identical_fallback_leaves_a_chain_of_one(monkeypatch):
    base_env(monkeypatch)
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash")
    assert Settings.from_env().model_chain == ("gemini-3.5-flash",)


def test_attempts_and_budget_are_configurable(monkeypatch):
    base_env(monkeypatch)
    monkeypatch.setenv("LLM_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("MAX_LLM_CALLS_PER_TURN", "5")
    s = Settings.from_env()
    assert s.llm_max_attempts == 3 and s.max_llm_calls_per_turn == 5


def test_env_example_ships_a_real_fallback():
    values = dict(line.split("=", 1) for line in (Path(__file__).resolve().parents[1] / ".env.example").read_text().splitlines()
                  if "=" in line and not line.lstrip().startswith("#"))
    chain = dict.fromkeys([values["GEMINI_MODEL_NAME"].strip(), *[m.strip() for m in values["GEMINI_FALLBACK_MODELS"].split(",") if m.strip()]])
    assert len(chain) >= 2
