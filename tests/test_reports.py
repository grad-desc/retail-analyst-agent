import pytest

from src.reports.delete_flow import Criteria, format_preview, parse_confirmation, parse_criteria
from src.reports.store import ReportStore

ME, OTHER = "customer:123", "customer:456"


def seeded():
    s = ReportStore(":memory:")
    ids = {
        "a": s.save(ME, "t1", "Q1 revenue for Client X", "q", "body"),
        "b": s.save(ME, "t1", "Weekly overview", "q", "We discussed  CLIENT   x here"),
        "c": s.save(ME, "t2", "Unrelated", "q", "nothing"),
        "d": s.save(OTHER, "t1", "Client X deep dive", "q", "body"),
    }
    return s, ids


def test_owner_isolation_on_every_operation():
    s, ids = seeded()
    assert {r.id for r in s.list(ME)} == {ids["a"], ids["b"], ids["c"]}
    assert s.get(ME, ids["d"]) is None
    assert s.find(ME, mentions=["client x"]) and all(r.owner == ME for r in s.find(ME, mentions=["client x"]))
    assert s.soft_delete(ME, [ids["d"]], "b1") == []              # someone else's id: nothing happens
    assert s.get(OTHER, ids["d"]) is not None


def test_mentions_are_case_and_whitespace_insensitive_and_search_title_question_content():
    s, ids = seeded()
    assert {r.id for r in s.find(ME, mentions=["Client X"])} == {ids["a"], ids["b"]}


def test_find_without_any_filter_returns_nothing_unless_match_all():
    s, _ = seeded()
    assert s.find(ME) == [] and len(s.find(ME, match_all=True)) == 3


def test_filters_combine_with_and():
    s, ids = seeded()
    assert [r.id for r in s.find(ME, mentions=["client x"], thread_id="t2")] == []
    assert {r.id for r in s.find(ME, thread_id="t1")} == {ids["a"], ids["b"]}


def test_user_text_is_never_interpolated_into_sql():
    s, _ = seeded()
    assert s.find(ME, mentions=["x'; DROP TABLE reports; --"]) == []
    assert len(s.list(ME)) == 3


def test_soft_delete_hides_and_undo_restores_within_window_only():
    s, ids = seeded()
    s.soft_delete(ME, [ids["a"], ids["b"]], "batch1", now=1000)
    assert [r.id for r in s.list(ME)] == [ids["c"]]
    assert s.restore_last_batch(ME, window_s=600, now=1700) == []          # 700s later: window closed
    assert s.restore_last_batch(ME, window_s=600, now=1500) == sorted([ids["a"], ids["b"]])
    assert len(s.list(ME)) == 3


def test_undo_restores_only_the_latest_batch():
    s, ids = seeded()
    s.soft_delete(ME, [ids["a"]], "old", now=1000)
    s.soft_delete(ME, [ids["c"]], "new", now=1100)
    assert s.restore_last_batch(ME, 600, now=1200) == [ids["c"]]
    assert ids["a"] not in {r.id for r in s.list(ME)}


def test_purge_hard_deletes_only_expired_soft_deletes():
    s, ids = seeded()
    s.soft_delete(ME, [ids["a"]], "b", now=1000)
    assert s.purge_expired(retention_s=100, now=1050) == 0
    assert s.purge_expired(retention_s=100, now=1200) == 1
    assert s.restore_last_batch(ME, 10**9, now=1300) == []


def test_audit_log_is_append_only_and_filtered_by_owner():
    s, ids = seeded()
    s.audit(ME, "delete_planned", [ids["a"]], {"criteria": "x"}, "tr1")
    s.audit(OTHER, "save", [ids["d"]])
    log = s.audit_log(ME)
    assert len(log) == 1 and log[0]["action"] == "delete_planned" and log[0]["trace_id"] == "tr1"


@pytest.mark.parametrize("text,expected", [
    ("yes", True), ("Yes!", True), ("  y ", True), ("yes, delete them", True), ("go ahead", True),
    ("no", False), ("No.", False), ("cancel", False), ("never mind", False),
    ("maybe", None), ("yes but also show revenue", None), ("", None), ("sure why not", None),
    ("what about Q2?", None),
])
def test_confirmation_parsing_is_strict(text, expected):
    assert parse_confirmation(text) is expected


def test_criteria_are_sanitised():
    c = parse_criteria({"report_ids": [3, "5", True, -1, 3, "x", 2.5], "mentions": [" Client X ", "", 7, "a"],
                        "scope": "everything"})
    assert c.ids == (3, 5) and c.mentions == ("Client X",) and c.scope is None and c.is_specific
    assert not parse_criteria({}).is_specific
    assert parse_criteria({"scope": "all"}).is_specific


def test_preview_lists_reports_and_truncates():
    s = ReportStore(":memory:")
    for i in range(13):
        s.save(ME, "t", f"R{i}", "q", "c")
    text = format_preview(s.list(ME), Criteria(scope="all"))
    assert "13 reports" in text and "and 3 more" in text and "**yes**" in text and "undo" in text
