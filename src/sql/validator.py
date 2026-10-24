"""AST-based SQL guard: SELECT-only, table allowlist, PII columns, RLS rewrite, LIMIT.

Raises SQLValidationError (with `.feedback` for the LLM self-correction loop and `.security`
for alerting). Everything is deterministic code; the LLM cannot talk its way around it.
"""
from __future__ import annotations

import logging
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from .errors import SQLValidationError, ValidatedSQL
from .rls import UserPolicy, apply_rls
from .schema import ALLOWED_TABLES, DATASET, PII_COLUMNS, PROJECT

log = logging.getLogger(__name__)

MAX_SQL_CHARS = 20_000

ROOT_TYPES = tuple(t for t in (getattr(exp, "Query", None), exp.Select, exp.Union) if t)

_FORBIDDEN_NODE_NAMES = (
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable", "Merge", "Command",
    "Set", "Use", "Grant", "TruncateTable", "Copy", "LoadData", "Export", "Declare", "Call",
    "Transaction", "Commit", "Rollback", "Into",
)
FORBIDDEN_NODES = tuple(getattr(exp, n) for n in _FORBIDDEN_NODE_NAMES if hasattr(exp, n))

FORBIDDEN_FUNCTION_PREFIXES = ("EXTERNAL_QUERY", "EXTERNAL_OBJECT_TRANSFORM", "ML.", "AI.",
                               "VECTOR_SEARCH", "SESSION_USER", "APPENDS", "CHANGES")
FORBIDDEN_SUBSTRINGS = ("information_schema", "external_query", "__tables__")


def validate_and_rewrite(sql: str, policy: UserPolicy, max_rows: int = 10_000) -> ValidatedSQL:
    if not sql or not sql.strip():
        raise SQLValidationError("syntax", "Empty SQL.")
    if len(sql) > MAX_SQL_CHARS:
        raise SQLValidationError("policy", "SQL is too long.")

    # 1. parse: exactly one statement
    try:
        statements = [s for s in sqlglot.parse(sql, read="bigquery") if s is not None]
    except (ParseError, TokenError) as e:
        raise SQLValidationError("syntax", f"SQL parse error: {str(e).splitlines()[0][:300]}") from e
    if len(statements) != 1:
        raise SQLValidationError("multi_statement", "Exactly one SELECT statement is allowed.",
                                 security=True)
    tree = statements[0]

    # 2. read-only
    if not isinstance(tree, ROOT_TYPES):
        raise SQLValidationError("forbidden_statement", "Only SELECT queries are allowed.",
                                 security=True)
    bad = next((n for n in tree.walk() if isinstance(n, FORBIDDEN_NODES)), None)
    if bad is not None:
        raise SQLValidationError("forbidden_statement",
                                 f"Statement type {type(bad).__name__} is not allowed.", security=True)

    # 3. functions / meta tables
    low = tree.sql(dialect="bigquery").lower()
    for s in FORBIDDEN_SUBSTRINGS:
        if s in low:
            raise SQLValidationError("forbidden_table", f"'{s}' is not accessible.", security=True)
    for fn in tree.find_all(exp.Func):
        name = (fn.name if isinstance(fn, exp.Anonymous) else fn.sql_name()).upper()
        if name.startswith(FORBIDDEN_FUNCTION_PREFIXES):
            raise SQLValidationError("forbidden_function", f"Function {name} is not allowed.",
                                     security=True)

    # 4. tables: allowlist; CTEs must not shadow real tables (would bypass RLS)
    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    shadow = cte_names & ALLOWED_TABLES
    if shadow:
        raise SQLValidationError("policy", f"CTE name(s) {sorted(shadow)} clash with table names; "
                                           "rename them.", security=True)
    real_tables = []
    for t in tree.find_all(exp.Table):
        if not isinstance(t.this, exp.Identifier):
            raise SQLValidationError("forbidden_table", "Table functions are not allowed.",
                                     security=True)
        name, db, cat = t.name.lower(), t.db.lower(), t.catalog.lower()
        if not db and not cat and name in cte_names:
            continue  # reference to a CTE defined in this query
        if (cat and cat != PROJECT) or (db and db != DATASET) or name not in ALLOWED_TABLES:
            raise SQLValidationError(
                "forbidden_table",
                f"Table '{t.sql(dialect='bigquery')}' is not available. "
                f"Allowed tables: {sorted(ALLOWED_TABLES)}.", security=True)
        if t.args.get("version"):
            raise SQLValidationError("policy", "Time travel is not allowed.", security=True)
        real_tables.append(t)

    # 5. PII columns (clear message; SAFE_COLUMNS projection is the actual barrier)
    for col in tree.find_all(exp.Column):
        if col.name.lower() in PII_COLUMNS:
            raise SQLValidationError("pii_column",
                                     f"Column '{col.name}' is restricted and cannot be used.",
                                     security=True)

    # 6. RLS + safe projection
    tables = sorted({t.name.lower() for t in real_tables})
    tree = apply_rls(tree, real_tables, policy)

    # 7. LIMIT
    limit_added = False
    if isinstance(tree, exp.Select) and not tree.args.get("limit"):
        tree = tree.limit(max_rows)
        limit_added = True

    out = tree.sql(dialect="bigquery")
    log.info("sql_validated", extra={"role": policy.role.value, "user_id": policy.user_id, "tables": tables})
    return ValidatedSQL(sql=out, tables=tables, limit_added=limit_added)
