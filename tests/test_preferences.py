import pytest

from src.memory.preferences import (PreferenceStore, describe_signals, detect, is_standing, wants_forget,
                                    wants_recall)

U, V = "customer:1", "customer:2"


def kv(text):
    return {(s.key, s.value, s.kind) for s in detect(text)}


@pytest.mark.parametrize("text,expected", [
    ("I prefer tables from now on", {("format", "table", "request")}),
    ("use bullet points please", {("format", "bullets", "request")}),
    ("write it as a narrative", {("format", "prose", "request")}),
    ("keep it short", {("depth", "brief", "request")}),
    ("give me a deep dive on Q1", {("depth", "deep", "request")}),
    ("graph it", {("visuals", "charts", "request")}),
    ("no charts, text only", {("visuals", "text", "request")}),
    ("That was too short", {("depth", "deep", "complaint")}),            # never misread as "short"
    ("way too long", {("depth", "brief", "complaint")}),
    ("too long, use bullet points", {("depth", "brief", "complaint"), ("format", "bullets", "request")}),
    ("Show monthly revenue for 2024", set()),
    ("Why did churn spike last month?", set()),
])
def test_signal_detection(text, expected):
    assert kv(text) == expected


def test_standing_statement_takes_effect_immediately():
    s = PreferenceStore()
    s.observe(U, detect("I prefer tables"), standing=True)
    assert s.current(U) == {"format": "table"}


def test_one_off_request_is_honoured_this_turn_but_not_learned():
    s = PreferenceStore()
    sig = detect("show revenue as a table")
    s.observe(U, sig)
    assert s.current(U) == {}                                        # a single inline request is not a habit
    assert s.effective(U, sig) == {"format": "table"}                # ...but it applies to the current answer
    assert "table" in s.render(U, sig)


def test_repeated_requests_become_a_default():
    s = PreferenceStore()
    for _ in range(2):
        s.observe(U, detect("as a table please"))
    assert s.current(U) == {"format": "table"}


def test_preferences_can_change_with_new_evidence():
    s = PreferenceStore()
    s.observe(U, detect("I prefer tables"), standing=True)
    s.observe(U, detect("I prefer bullet points"), standing=True)
    assert s.current(U) == {"format": "bullets"}


def test_complaint_teaches_depth():
    s = PreferenceStore()
    s.observe(U, detect("too long"))
    assert s.current(U) == {"depth": "brief"}


def test_subjects_are_isolated_and_forget_wipes_everything():
    s = PreferenceStore()
    s.observe(U, detect("I prefer tables"), standing=True)
    s.observe(V, detect("I prefer bullet points"), standing=True)
    assert s.current(U) == {"format": "table"} and s.current(V) == {"format": "bullets"}
    s.forget(U)
    assert s.current(U) == {} and s.current(V) == {"format": "bullets"}


def test_render_and_describe():
    s = PreferenceStore()
    assert s.render(U) == "" and "haven't learned" in s.describe(U)
    s.observe(U, detect("I prefer tables and keep answers brief"), standing=True)
    text = s.render(U)
    assert "table" in text and "brief" in text.lower()
    assert "use tables" in s.describe(U) and "forget" in s.describe(U)
    assert describe_signals(detect("no charts")) == "stay text-only"


def test_intent_helpers():
    assert is_standing("From now on use tables") and not is_standing("as a table")
    assert wants_forget("please forget my preferences") and wants_forget("reset what you learned about me")
    assert wants_recall("what do you remember about me?") and not wants_recall("show revenue")


@pytest.mark.parametrize("text", ["Which tables do you have?", "What tables can I query?", "show me the tables in the dataset",
                                  "which table has the orders?", "What columns are in the orders table?"])
def test_table_as_a_data_term_is_not_a_format_preference(text):
    assert kv(text) == set()


@pytest.mark.parametrize("text", ["put it in a table", "show it in tabular form", "give me a summary table",
                                  "use a table", "revenue by state as a table", "tables please"])
def test_presentation_wording_for_tables_is_still_detected(text):
    assert kv(text) == {("format", "table", "request")}


def test_asking_about_tables_twice_does_not_create_a_preference():
    s = PreferenceStore()
    for q in ("Which tables do you have?", "What tables are available?"):
        s.observe(U, detect(q))
    assert s.current(U) == {}
