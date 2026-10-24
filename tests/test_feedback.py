from pathlib import Path

import pytest
import yaml

from src.memory.feedback import FeedbackStore, record_feedback
from src.memory.preferences import PreferenceStore


def turn(role="executive", **kw):
    principal = {"role": role, "user_id": None if role == "executive" else 123,
                 "subject": "executive:dana" if role == "executive" else "customer:123"}
    base = {"trace_id": "tr1", "thread_id": "t", "question": "Top customers?", "resolved_question": "top customers by spend",
            "intent": "analysis", "status": "done", "sql": "SELECT user_id FROM order_items", "executed_sql": "SELECT ...scoped...",
            "answer": "Customer 7 leads with 1,200.", "golden_ids": ["top-customers-by-spend"], "principal": principal}
    base.update(kw)
    return base


def setup(tmp_path):
    return FeedbackStore(), PreferenceStore(), tmp_path


def test_thumbs_up_on_executive_turn_queues_a_golden_candidate_with_the_original_sql(tmp_path):
    fb, prefs, g = setup(tmp_path)
    res = record_feedback(fb, prefs, g, turn(), 1)
    assert res.saved and res.candidate and "analyst review" in res.message
    data = yaml.safe_load((g / "candidates" / f"{res.candidate}.yaml").read_text())
    assert data["status"] == "candidate" and data["sql"] == "SELECT user_id FROM order_items"   # NOT the scoped rewrite
    assert data["report"].startswith("Customer 7") and data["source_role"] == "executive"
    assert data["source_trace_id"] == "tr1" and data["golden_ids_used"] == ["top-customers-by-spend"]


def test_customer_turns_never_put_their_figures_into_the_shared_golden_set(tmp_path):
    fb, prefs, g = setup(tmp_path)
    res = record_feedback(fb, prefs, g, turn("customer", answer="You spent 1,200 on Jeans."), 1)
    data = yaml.safe_load((g / "candidates" / f"{res.candidate}.yaml").read_text())
    assert data["report"] == "" and data["source_role"] == "customer"
    assert "1,200" not in (g / "candidates" / f"{res.candidate}.yaml").read_text()


@pytest.mark.parametrize("override", [{"intent": "schema"}, {"status": "failed"}, {"executed_sql": None}, {"sql": None}])
def test_only_successful_analyses_become_candidates(tmp_path, override):
    fb, prefs, g = setup(tmp_path)
    res = record_feedback(fb, prefs, g, turn(**override), 1)
    assert res.saved and res.candidate is None and not (g / "candidates").exists()


def test_thumbs_down_with_comment_teaches_preferences_and_scrubs_pii(tmp_path):
    fb, prefs, g = setup(tmp_path)
    res = record_feedback(fb, prefs, g, turn(), -1, "way too long, ask jane@corp.com why")
    assert res.learned == ["keep answers brief"] and "brief" in res.message
    assert prefs.current("executive:dana") == {"depth": "brief"}
    row = fb.failures()[0]
    assert "jane@corp.com" not in row["comment"] and "[REDACTED_EMAIL]" in row["comment"]
    assert row["sql"] == "SELECT user_id FROM order_items" and row["golden_ids"] == ["top-customers-by-spend"]


def test_one_rating_per_turn_and_it_can_be_changed(tmp_path):
    fb, prefs, g = setup(tmp_path)
    record_feedback(fb, prefs, g, turn(), 1)
    record_feedback(fb, prefs, g, turn(), -1, "wrong numbers")
    assert fb.stats() == {"total": 1, "up": 0, "down": 1, "down_rate": 1.0}


def test_stats_and_failures_export(tmp_path):
    fb, prefs, g = setup(tmp_path)
    record_feedback(fb, prefs, g, turn(trace_id="a"), 1)
    record_feedback(fb, prefs, g, turn(trace_id="b"), -1, "numbers look off")
    record_feedback(fb, prefs, g, turn(trace_id="c"), 1)
    assert fb.stats()["down_rate"] == 0.333 and [f["trace_id"] for f in fb.failures()] == ["b"]


def test_rating_before_any_question_is_a_friendly_noop(tmp_path):
    fb, prefs, g = setup(tmp_path)
    res = record_feedback(fb, prefs, g, {}, 1)
    assert not res.saved and "ask a question first" in res.message


def test_invalid_rating_is_rejected(tmp_path):
    fb, prefs, g = setup(tmp_path)
    with pytest.raises(ValueError):
        record_feedback(fb, prefs, g, turn(), 5)
