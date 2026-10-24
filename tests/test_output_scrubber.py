import logging

import pytest

from src.guards.output_scrubber import RedactingFilter, scrub_dataframe, scrub_text


@pytest.mark.parametrize("raw,kind", [
    ("write to john.doe+shop@example.co.uk now", "email"),
    ("call +1 (415) 555-0132 or 415-555-0132", "phone"),
    ("tel +381641234567", "phone"),
    ("card 4111 1111 1111 1111", "card"),
    ("ssn 123-45-6789", "ssn"),
    ("iban DE89 3704 0044 0532 0130 00", "iban"),
    ("from 192.168.10.25", "ipv4"),
    ("lives at 45.2671, 19.8335", "coords"),
])
def test_text_redaction(raw, kind):
    clean, rep = scrub_text(raw)
    assert f"[REDACTED_{kind.upper()}]" in clean
    assert rep.redactions[kind] >= 1


@pytest.mark.parametrize("benign", [
    "Revenue 1234567.89 in 2024-01-15 10:30:45", "order 100234567 shipped", "Q1: 12.5% growth",
    "price 129.99, cost 61.25",
    "version 1.2.3.4.5", "ids 12345-67890",
])
def test_no_false_positives(benign):
    clean, rep = scrub_text(benign)
    assert clean == benign and rep.clean


def test_report_has_counts_not_values():
    _, rep = scrub_text("a@b.com and c@d.org")
    assert rep.as_log() == {"pii_redactions": {"email": 2}, "pii_dropped_columns": []}


def test_dataframe_drops_pii_columns_and_redacts_cells():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"email": ["a@b.com"], "Last_Name": ["X"], "customer_phone": ["1"],
                       "note": ["reach me at a@b.com"], "total": [10.5]})
    df.attrs["job_id"] = "j1"
    out, rep = scrub_dataframe(df)
    assert list(out.columns) == ["note", "total"]
    assert out["note"][0] == "reach me at [REDACTED_EMAIL]"
    assert out.attrs["job_id"] == "j1"
    assert "email" in df.columns            # input untouched
    assert set(rep.dropped_columns) == {"email", "Last_Name", "customer_phone"}


def test_logging_filter_redacts():
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "user %s asked", ("a@b.com",), None)
    RedactingFilter().filter(rec)
    assert "a@b.com" not in rec.getMessage()


@pytest.mark.parametrize("dtype", ["object", "string"])
def test_dataframe_cells_are_redacted_for_every_text_dtype(dtype):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"note": ["reach me at a@b.com", None, "call 415-555-0132"]}).astype({"note": dtype})
    out, rep = scrub_dataframe(df)
    assert list(out["note"])[0] == "reach me at [REDACTED_EMAIL]" and list(out["note"])[2] == "call [REDACTED_PHONE]"
    assert rep.redactions["email"] == 1 and rep.redactions["phone"] == 1
