"""Graph state and dependency container."""
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

from src.memory.preferences import PreferenceStore
from src.prompts import PromptRegistry
from src.reports.delete_flow import CONFIRM_TTL_S
from src.reports.store import ReportStore


class AgentState(TypedDict, total=False):
    # --- inputs (per turn) ---
    question: str
    principal: Dict[str, Any]          # {"role": "customer" | "executive", "user_id": int | None}
    trace_id: str
    thread_id: str
    # --- persisted across turns ---
    history: Annotated[List[Dict[str, str]], operator.add]
    # --- turn-scoped (reset by the route node) ---
    intent: Optional[str]
    resolved_question: str
    status: str                        # ok | failed | blocked | done
    failure: Optional[str]
    sql: Optional[str]
    validated_sql: Optional[str]
    executed_sql: Optional[str]
    feedback: Optional[str]            # why the last SQL attempt failed (fed back to the LLM)
    attempts: int
    empty_retries: int
    future_retries: int                # retries because the result contained periods after today
    llm_calls: int
    row_count: int
    truncated: bool                    # capped at the row limit OR only a sample went into the prompt
    capped: bool                       # the warehouse result itself was cut at the row cap
    sampled: bool                      # more rows than fit in the prompt: sample + whole-result summaries sent
    data_csv: str                      # scrubbed rows for the report prompt
    table_md: str                      # scrubbed table for display
    answer: str
    guard: Dict[str, Any]              # guard outcomes (counts only, never PII values)
    pending_delete: Optional[Dict[str, Any]]   # plan awaiting explicit user confirmation
    saved_report_id: Optional[int]
    golden_ids: List[str]              # golden trios used as few-shot for this turn (observability)


def _default_validate(sql, policy):
    from src.sql.validator import validate_and_rewrite  # lazy: keeps sqlglot optional for unit tests
    return validate_and_rewrite(sql, policy)


@dataclass
class Deps:
    llm: Any                           # src.llm.client.LLMClient (or compatible)
    runner: Any                        # SafeBigQueryRunner (or compatible)
    prompts: PromptRegistry
    schema_text: str
    max_sql_attempts: int = 3          # 1 generation + up to 2 corrections
    max_llm_calls: int = 8             # hard per-turn budget (cost control)
    max_prompt_rows: int = 50
    reports: ReportStore = field(default_factory=lambda: ReportStore(":memory:"))
    confirm_ttl_s: float = CONFIRM_TTL_S
    validate: Callable = field(default=_default_validate)
    preferences: PreferenceStore = field(default_factory=lambda: PreferenceStore(":memory:"))
    golden: Any = None                 # src.retrieval.golden.GoldenIndex (or None: no few-shot)
