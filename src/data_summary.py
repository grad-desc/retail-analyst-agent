"""Compact, whole-result view of a query result for the report prompt, and a future-period check.

Why this exists: the report prompt used to receive only the first `max_prompt_rows` rows. For a result
that is wider than that (for example month x category over a year) the model saw a few months, concluded
that "the data only covers January and February" and saved a wrong report. Now, when a result does not fit,
the prompt gets the sample rows PLUS summaries computed over ALL rows (totals, ranges, totals by key), so
trends and totals stay correct while the prompt stays small.

Everything here is deterministic pandas code (no LLM, no network). Callers pass the already PII-scrubbed
frame, so group keys and sample rows are clean.
"""
from __future__ import annotations

import re
from datetime import date
from typing import List, Optional, Sequence

import pandas as pd

MAX_SUMMARY_CHARS = 6000        # hard cap for the summary block, keeps the report prompt small
MAX_KEY_COLS = 3                # at most this many "group by" columns are summarized
MAX_VALUE_COLS = 3              # at most this many numeric columns per summary table
MAX_KEY_CARDINALITY = 60        # a column with more distinct values is not a useful grouping key
MAX_ORDERED_KEYS = 36           # time-like keys are shown in order (the most recent ones if there are more)
MAX_RANKED_KEYS = 10            # other keys are ranked by the first numeric column

_TIME_NAME = re.compile(r"(month|date|day|week|quarter|year|period|_at$|time)", re.I)
_MEAN_NAME = re.compile(r"(rate|pct|percent|avg|average|ratio|margin|share|price|mean|per_)", re.I)
_ID_NAME = re.compile(r"(^id$|_id$|^id_)", re.I)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _is_time_like(name: str, s: pd.Series) -> bool:
    return (pd.api.types.is_datetime64_any_dtype(s) or str(s.dtype) in ("dbdate", "dbtime", "period[M]")
            or bool(_TIME_NAME.search(str(name))))


def _fmt(v, as_int: bool = False) -> str:
    if as_int:
        return f"{int(v):,}"
    try:
        return f"{float(v):,.2f}"
    except (TypeError, ValueError):
        return str(v)


def _value_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns
            if _is_numeric(df[c]) and not _ID_NAME.search(str(c)) and not _TIME_NAME.search(str(c))]
    return cols[:MAX_VALUE_COLS]


def _key_columns(df: pd.DataFrame) -> List[str]:
    n = len(df)
    time_keys, other_keys = [], []
    for c in df.columns:
        s = df[c]
        if _is_numeric(s) and not _TIME_NAME.search(str(c)):
            continue                                 # measures and ids are not grouping keys
        try:
            card = s.nunique(dropna=True)
        except TypeError:                            # unhashable cells: not a usable key
            continue
        if not 1 < card <= MAX_KEY_CARDINALITY or card >= n:
            continue
        (time_keys if _is_time_like(c, s) else other_keys).append((card, str(c)))
    ordered = [c for _, c in sorted(time_keys)] + [c for _, c in sorted(other_keys)]
    return ordered[:MAX_KEY_COLS]


def _agg_for(name: str) -> str:
    return "mean" if _MEAN_NAME.search(str(name)) else "sum"


def _overall_block(df: pd.DataFrame, values: Sequence[str]) -> str:
    lines = []
    for c in df.columns:
        if _is_time_like(c, df[c]) and not _is_numeric(df[c]):
            s = df[c].dropna().astype(str)
            if len(s):
                lines.append(f"- {c}: from {s.min()} to {s.max()}")
    for v in values:
        agg = _agg_for(v)
        total = getattr(df[v], agg)()
        lines.append(f"- {agg} of {v} over all rows: {_fmt(total, as_int=(agg == 'sum' and pd.api.types.is_integer_dtype(df[v])))}")
    return "\n".join(lines)


def _by_key_block(df: pd.DataFrame, key: str, values: Sequence[str]) -> str:
    keys = df[key].astype(str)
    time_like = _is_time_like(key, df[key])
    if values:
        aggs = {v: _agg_for(v) for v in values}
        table = pd.DataFrame({f"{aggs[v]} {v}": df[v] for v in values}).groupby(keys, dropna=False).agg(
            {f"{aggs[v]} {v}": aggs[v] for v in values})
        heads = list(table.columns)
        as_int = {h: pd.api.types.is_integer_dtype(df[v]) and aggs[v] == "sum" for h, v in zip(heads, values)}
    else:
        table = keys.value_counts().sort_index().to_frame("rows")
        heads, as_int = ["rows"], {"rows": True}
    more = ""
    if time_like:
        table = table.sort_index()
        if len(table) > MAX_ORDERED_KEYS:
            more = f"\n(+{len(table) - MAX_ORDERED_KEYS} earlier periods not shown)"
            table = table.tail(MAX_ORDERED_KEYS)
    else:
        table = table.sort_values(heads[0], ascending=False)
        if len(table) > MAX_RANKED_KEYS:
            more = f"\n(+{len(table) - MAX_RANKED_KEYS} more values not shown)"
            table = table.head(MAX_RANKED_KEYS)
    rows = [f"{k} | " + " | ".join(_fmt(r[h], as_int[h]) for h in heads) for k, r in table.iterrows()]
    return f"By {key} ({' | '.join(heads)}):\n" + "\n".join(rows) + more


def summarize_result(df: pd.DataFrame, *, max_rows: int, notes: Sequence[str] = ()) -> str:
    """Text for the report prompt's <data> block.

    Small result (<= max_rows): exactly the CSV, as before. Larger result: sample rows + summaries over ALL rows.
    `notes` (for example "latest period is incomplete") are put first so the model reads them first.
    """
    n = len(df)
    head_csv = df.head(max_rows).to_csv(index=False)
    note_block = "".join(f"NOTE: {x}\n" for x in notes)
    if n <= max_rows:
        return note_block + head_csv
    values, keys = _value_columns(df), _key_columns(df)
    blocks: List[str] = [_overall_block(df, values)]
    blocks += [_by_key_block(df, k, values) for k in keys]
    out = [f"Rows shown below: the first {max_rows} of {n} (a sample, NOT the whole result).", head_csv.rstrip(),
           f"Summaries over ALL {n} rows (use these for totals, ranges and trends):"]
    used = sum(len(x) for x in out) + len(note_block)
    for b in blocks:
        if not b:
            continue
        if used + len(b) > MAX_SUMMARY_CHARS + len(head_csv):
            out.append("(further summaries omitted for length)")
            break
        out.append(b)
        used += len(b)
    return note_block + "\n".join(out)


def future_periods(df: pd.DataFrame, today: date) -> List[str]:
    """Time-like columns whose latest value is after `today`, as 'column (latest YYYY-MM-DD)'.

    A result labelled with a month that has not happened yet is almost always a query artefact (for example
    'month = previous month + 1' for the current month, where nobody can have bought yet, which looks like 100%
    churn). Returns [] when nothing is in the future."""
    out: List[str] = []
    cutoff = pd.Timestamp(today, tz="UTC")
    for name in df.columns:
        s = df[name]
        if len(s) == 0 or not _is_time_like(str(name), s):
            continue
        if _is_numeric(s):
            if re.search(r"year", str(name), re.I) and s.dropna().size and s.max() > today.year:
                out.append(f"{name} (latest {int(s.max())})")
            continue
        try:
            t = pd.to_datetime(s.astype(str), errors="coerce", utc=True)
        except (TypeError, ValueError):
            continue
        latest: Optional[pd.Timestamp] = t.max()
        if pd.notna(latest) and latest > cutoff:
            out.append(f"{name} (latest {latest.date().isoformat()})")
    return out
