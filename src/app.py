"""Application bootstrap shared by the CLI commands (kept free of UI code so it is testable)."""
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.config import Settings
from src.graph import build_graph
from src.llm.client import LLMClient
from src.memory.feedback import FeedbackStore
from src.memory.preferences import PreferenceStore
from src.prompts import PromptRegistry
from src.reports.store import ReportStore
from src.retrieval.golden import GoldenIndex
from src.sql.policy import Role
from src.sql.schema import fq
from src.sql.schema_text import build_schema_text
from src.state import Deps

ROOT = Path(__file__).resolve().parent.parent


def principal_from_flags(user_id: Optional[int], role: str, name: str = "default") -> Dict[str, Any]:
    """Role must come from authenticated identity in production; the CLI flag stands in for it."""
    role = (role or "customer").lower()
    if role == Role.EXECUTIVE.value:
        if user_id is not None:
            raise ValueError("--user-id cannot be combined with --role executive.")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", name or ""):
            raise ValueError("--name may contain letters, digits, dot, dash and underscore (max 40).")
        return {"role": role, "user_id": None, "subject": f"executive:{name}"}
    if role == Role.CUSTOMER.value:
        if user_id is None:
            raise ValueError("Customer mode needs --user-id (or use --role executive).")
        if user_id < 0:
            raise ValueError("--user-id must be a non-negative integer.")
        return {"role": role, "user_id": int(user_id), "subject": f"customer:{int(user_id)}"}
    raise ValueError("--role must be 'customer' or 'executive'.")


def make_checkpointer(path: Path):
    """SQLite checkpointer: conversations survive restarts (resume with --thread)."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
    except ImportError:  # pragma: no cover
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


@dataclass
class App:
    graph: Any
    runner: Any
    preferences: PreferenceStore
    feedback: FeedbackStore
    golden_root: Path


def bootstrap(settings: Settings, *, prompts_dir: Path = ROOT / "prompts",
              checkpoint_path: Path = ROOT / "data" / "checkpoints.sqlite",
              memory_path: Path = ROOT / "data" / "memory.sqlite",
              golden_root: Path = ROOT / "golden") -> App:
    from src.sql.executor import SafeBigQueryRunner  # lazy: pulls in google-cloud-bigquery

    runner = SafeBigQueryRunner(project_id=settings.gcp_project_id, max_bytes=settings.bq_max_bytes,
                                location=settings.bq_location)
    reports = ReportStore(ROOT / "data" / "reports.sqlite")
    reports.purge_expired()   # hard-delete reports soft-deleted more than 7 days ago
    golden = GoldenIndex.load(golden_root / "trios")   # quarantines invalid trios, logs why
    preferences = PreferenceStore(memory_path)
    deps = Deps(llm=LLMClient.from_settings(settings), runner=runner, reports=reports, golden=golden,
                preferences=preferences,
                prompts=PromptRegistry(prompts_dir), schema_text=build_schema_text(runner),
                max_llm_calls=settings.max_llm_calls_per_turn)
    return App(build_graph(deps, make_checkpointer(checkpoint_path)), runner, preferences,
               FeedbackStore(memory_path), golden_root)


def customer_summary(runner, user_id: int) -> Tuple[bool, int]:
    """(exists, order_count). A system query, never LLM-generated."""
    uid = int(user_id)
    df = runner.execute_query(
        f"SELECT (SELECT COUNT(1) FROM {fq('users')} WHERE id = {uid}) AS u, "
        f"(SELECT COUNT(1) FROM {fq('orders')} WHERE user_id = {uid}) AS o")
    return bool(df["u"][0]), int(df["o"][0])
