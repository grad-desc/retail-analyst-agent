"""Exception types and result objects shared across the SQL layer (no heavy dependencies)."""
from dataclasses import dataclass, field
from typing import List


class SQLValidationError(Exception):
    """Query rejected by the AST guard."""

    def __init__(self, kind: str, message: str, security: bool = False) -> None:
        super().__init__(message)
        self.kind = kind
        self.security = security  # True -> security event for logs / guard-block metrics

    @property
    def feedback(self) -> str:
        """Text safe to hand back to the SQL-generator LLM."""
        return f"[{self.kind}] {self}"


@dataclass
class ValidatedSQL:
    sql: str
    tables: List[str] = field(default_factory=list)
    limit_added: bool = False


class SQLSyntaxError(Exception):
    """SQL error reported by BigQuery: text goes back to the LLM for self-correction."""


class QueryTooExpensive(Exception):
    """Over the byte cap: ask the LLM to narrow the query."""


class BQUnavailable(Exception):
    """BigQuery unreachable after retries: friendly message, no LLM tokens spent."""
