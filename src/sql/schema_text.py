"""Schema description handed to the LLM. Built only from the allowlist (never exposes PII columns)."""
from .schema import SAFE_COLUMNS


def static_schema_text() -> str:
    return "\n".join(f"- {t}({', '.join(cols)})" for t, cols in SAFE_COLUMNS.items())


def build_schema_text(runner) -> str:
    """Typed schema from BigQuery (allowlist-filtered); falls back to the static one if BQ is down."""
    try:
        lines = []
        for table in SAFE_COLUMNS:
            cols = runner.get_safe_schema(table)
            lines.append(f"- {table}(" + ", ".join(f"{c['name']} {c['type']}" for c in cols) + ")")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 - resilience: the agent must start even if BQ is flaky
        return static_schema_text()
