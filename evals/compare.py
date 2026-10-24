"""Execution-accuracy comparison: do two result sets contain the same data?

Order-insensitive for rows and columns, alias-insensitive (column names are ignored), numerically tolerant
(rounded to 3 decimals). This is how text-to-SQL systems are normally scored: the question is whether the
agent's query returned the right answer, not whether it was written like the reference.
"""
import math
from decimal import Decimal
from typing import Any, List, Tuple


def _norm(v: Any) -> Any:
    if v is None or str(v) in ("<NA>", "NaT", "nan", "None"):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, Decimal)):
        f = float(v)
        return None if math.isnan(f) else round(f, 3)
    return str(v)


def _columns(df) -> List[List[Any]]:
    return [[_norm(v) for v in df[c]] for c in df.columns]


def _signature(col: List[Any]) -> Tuple:
    return tuple(sorted(col, key=lambda x: (x is None, str(x))))


def _canonical_rows(cols: List[List[Any]]) -> List[Tuple]:
    ordered = sorted(cols, key=lambda c: str(_signature(c)))
    return sorted(zip(*ordered), key=str) if ordered else []


def results_match(reference, actual, allow_extra_columns: bool = False) -> Tuple[bool, str]:
    if len(reference) != len(actual):
        return False, f"row count differs: expected {len(reference)}, got {len(actual)}"
    ref_cols, act_cols = _columns(reference), _columns(actual)
    if not allow_extra_columns:
        if len(ref_cols) != len(act_cols):
            return False, f"column count differs: expected {len(ref_cols)}, got {len(act_cols)}"
        return (True, "") if _canonical_rows(ref_cols) == _canonical_rows(act_cols) else (False, "values differ")
    pool, matched = list(act_cols), []
    for rc in ref_cols:
        sig = _signature(rc)
        hit = next((i for i, ac in enumerate(pool) if _signature(ac) == sig), None)
        if hit is None:
            return False, "a reference column's values are missing from the result"
        matched.append(pool.pop(hit))
    return (True, "") if _canonical_rows(ref_cols) == _canonical_rows(matched) else (False, "values differ")
