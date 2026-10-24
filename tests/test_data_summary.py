"""Whole-result summaries for the report prompt and the future-period check (pandas only, no network)."""
from datetime import date

import pytest

pd = pytest.importorskip("pandas")

from src.data_summary import MAX_SUMMARY_CHARS, future_periods, summarize_result  # noqa: E402


def month_by_category(months=12, categories=26):
    rows = [{"month": f"2023-{m:02d}", "category": f"cat{c:02d}", "revenue": float(100 * m + c), "orders": m + c}
            for m in range(1, months + 1) for c in range(categories)]
    return pd.DataFrame(rows)


def test_small_result_is_passed_through_unchanged():
    df = pd.DataFrame({"category": ["Jeans", "Dresses"], "revenue": [10.5, 20.0]})
    assert summarize_result(df, max_rows=50) == df.to_csv(index=False)


def test_notes_come_first():
    df = pd.DataFrame({"a": [1]})
    assert summarize_result(df, max_rows=50, notes=["latest month is incomplete"]).startswith(
        "NOTE: latest month is incomplete\n")


def test_large_result_summaries_cover_every_row_not_just_the_sample():
    df = month_by_category()                              # 312 rows, only the first ~2 months fit in 50 rows
    out = summarize_result(df, max_rows=50)
    assert "first 50 of 312" in out and "Summaries over ALL 312 rows" in out
    assert "2023-12" in out                               # a month that is NOT in the first 50 rows
    assert f"{df['revenue'].sum():,.2f}" in out           # grand total over all rows
    assert "By month" in out and "By category" in out
    assert "from 2023-01 to 2023-12" in out


def test_rate_like_columns_are_averaged_not_summed():
    df = pd.DataFrame({"state": [f"s{i % 5}" for i in range(80)], "margin_rate": [0.5] * 80})
    out = summarize_result(df, max_rows=10)
    assert "mean of margin_rate" in out and "sum of margin_rate" not in out


def test_summary_block_stays_small_for_wide_results():
    df = month_by_category(months=12, categories=26)
    df["extra"] = [f"k{i % 40}" for i in range(len(df))]
    out = summarize_result(df, max_rows=50)
    head = df.head(50).to_csv(index=False)
    assert len(out) <= MAX_SUMMARY_CHARS + len(head) + 600


def test_ids_are_not_used_as_grouping_keys_or_summed():
    df = pd.DataFrame({"user_id": range(200), "spend": [1.0] * 200})
    out = summarize_result(df, max_rows=20)
    assert "By user_id" not in out and "of user_id" not in out
    assert "sum of spend over all rows: 200.00" in out


def test_future_month_is_flagged():
    df = pd.DataFrame({"month": pd.to_datetime(["2026-09-01", "2026-10-01", "2026-11-01"]), "churn": [0.9, 0.9, 1.0]})
    flagged = future_periods(df, date(2026, 10, 23))
    assert len(flagged) == 1 and flagged[0].startswith("month (latest 2026-11-01")


def test_current_and_past_months_are_not_flagged():
    df = pd.DataFrame({"month": ["2026-08", "2026-09", "2026-10"], "churn": [0.9, 0.9, 0.9]})
    assert future_periods(df, date(2026, 10, 23)) == []


def test_numeric_columns_are_ignored_except_year():
    df = pd.DataFrame({"avg_delivery_days": [3, 400], "orders": [1, 2]})
    assert future_periods(df, date(2026, 10, 23)) == []
    assert future_periods(pd.DataFrame({"year": [2025, 2027]}), date(2026, 10, 23)) == ["year (latest 2027)"]


def test_empty_frame_is_fine():
    assert future_periods(pd.DataFrame({"month": []}), date(2026, 10, 23)) == []
