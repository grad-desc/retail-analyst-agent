"""Output guard: last line of defence against PII reaching the user (or the logs).

Layers 1-2 (schema allowlist, SQL validator) stop PII at the source. This layer assumes they
failed or that PII arrived through *values* (free-text columns, LLM-written prose) and cleans:
  * DataFrames: drops PII-named columns, redacts PII-looking values in string cells
  * Text (final report / chat answer): redacts emails, phones, card numbers, SSNs, IBANs,
    IPs and precise coordinates
Deterministic, no LLM, no network.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.sql.schema import PII_COLUMNS

# Extra column names that never belong in an answer, beyond the schema-level PII set.
EXTRA_PII_COLUMN_HINTS = ("phone", "ssn", "passport", "iban", "credit_card", "card_number", "password")

_PATTERNS: Dict[str, re.Pattern] = {
    # order matters: specific patterns run first so phone (loosest) can't eat cards/IBANs
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}"),
    "card": re.compile(r"(?<![\d.])(?:\d[ \-]?){13,19}(?![\d.])"),  # Luhn-checked below
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,4})?\b"),
    "ssn": re.compile(r"(?<![\d\-])\d{3}-\d{2}-\d{4}(?![\d\-])"),
    "ipv4": re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"),
    "coords": re.compile(r"(?<![\d.])-?\d{1,3}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}(?![\d.])"),
    # separators are required for the national form so order ids / prices / dates don't match
    "phone": re.compile(
        r"(?<![\w.])(?:\+\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?|\d{2,4}[\s.\-])\d{3,4}[\s.\-]\d{3,4}(?![\w.])"
        r"|(?<![\w.])\+\d{10,15}(?!\w)"),
}


def _luhn_ok(s: str) -> bool:
    digits = [int(c) for c in s if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
    return total % 10 == 0


@dataclass
class ScrubReport:
    redactions: Counter = field(default_factory=Counter)   # {"email": 3, ...}
    dropped_columns: List[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.redactions and not self.dropped_columns

    def merge(self, other: "ScrubReport") -> "ScrubReport":
        self.redactions.update(other.redactions)
        self.dropped_columns.extend(other.dropped_columns)
        return self

    def as_log(self) -> dict:
        """Counts only: never the matched values."""
        return {"pii_redactions": dict(self.redactions), "pii_dropped_columns": self.dropped_columns}


def scrub_text(text: str) -> Tuple[str, ScrubReport]:
    report = ScrubReport()
    if not text:
        return text, report
    for kind, pat in _PATTERNS.items():
        def _sub(m: re.Match, kind=kind) -> str:
            if kind == "card" and not _luhn_ok(m.group(0)):
                return m.group(0)
            report.redactions[kind] += 1
            return f"[REDACTED_{kind.upper()}]"
        text = pat.sub(_sub, text)
    return text, report


def _is_pii_column(name: str) -> bool:
    n = str(name).lower()
    return n in PII_COLUMNS or any(h in n for h in EXTRA_PII_COLUMN_HINTS)


def scrub_dataframe(df):
    """Return (clean_copy, ScrubReport). Never mutates the input; keeps df.attrs."""
    report = ScrubReport()
    drop = [c for c in df.columns if _is_pii_column(c)]
    out = df.drop(columns=drop)
    report.dropped_columns = [str(c) for c in drop]
    out.attrs.update(df.attrs)
    for col in out.columns:
        dtype = out[col].dtype
        # text-like columns: object (pandas 2) and the dedicated string dtype (default in pandas 3)
        if getattr(dtype, "kind", None) == "O" or str(dtype) in ("string", "str"):
            def _cell(v):
                if not isinstance(v, str):
                    return v
                clean, r = scrub_text(v)
                report.redactions.update(r.redactions)
                return clean
            out[col] = out[col].map(_cell)
    return out, report


_STD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class RedactingFilter(logging.Filter):
    """Attach to handlers so PII never reaches the log sink: scrubs the message AND any
    string-valued extra fields (question, answer, error details...)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        clean, rep = scrub_text(msg)
        if not rep.clean:
            record.msg, record.args = clean, None
        for key, val in list(record.__dict__.items()):
            if key not in _STD_ATTRS and isinstance(val, str):
                record.__dict__[key] = scrub_text(val)[0]
        return True
