"""CLI for the retail analysis agent.

  python -m src.cli chat --user-id 123          # customer: sees only their own orders/products
  python -m src.cli chat --role executive       # executive: company-wide rows (PII still blocked)
  python -m src.cli ask "Monthly revenue 2024" --role executive
  python -m src.cli candidates | approve <id> --reviewer you | reject <id> --reason "..." --reviewer you
"""
import time
import uuid
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.app import ROOT, bootstrap, customer_summary, principal_from_flags
from src.config import ConfigError, Settings
from src.graph import run_turn
from src.memory.feedback import record_feedback
from src.observability.logging_setup import setup_logging
from src.observability.metrics import compute_metrics, filter_since, load_events, trace_timeline
from src.retrieval.curation import CurationError, list_candidates, promote, reject as reject_candidate
from src.sql.policy import subject_of

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Retail data analysis assistant (BigQuery thelook_ecommerce + Gemini).")
console = Console()

USER_ID = typer.Option(None, "--user-id", "-u", help="Customer id (users.id). Limits data to their own orders.")
ROLE = typer.Option("customer", "--role", "-r", help="'customer' (needs --user-id) or 'executive'.")
NAME = typer.Option("default", "--name", help="Executive identity; owns the reports you save (customers are identified by --user-id).")
THREAD = typer.Option(None, "--thread", "-t", help="Resume a previous conversation thread.")

HELP = ("[bold]Commands[/bold]  /sql  last executed query   /data  last result table   /new  fresh conversation\n"
        "          /up  /down [comment]  rate the last answer   /prefs  what I learned about you   /forget  reset it\n"
        "          /help   /exit")


def _start(user_id: Optional[int], role: str, name: str = "default"):
    try:
        principal = principal_from_flags(user_id, role, name)
        settings = Settings.from_env()
    except (ValueError, ConfigError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2)
    log_file = setup_logging(ROOT / "logs")
    try:
        application = bootstrap(settings)
        who = f"executive '{name}' (company-wide access)"
        if principal["role"] == "customer":
            exists, orders = customer_summary(application.runner, principal["user_id"])
            if not exists:
                console.print(f"[red]No customer with id {principal['user_id']} in the dataset.[/red]")
                raise typer.Exit(2)
            who = f"customer #{principal['user_id']} ({orders} orders)"
    except typer.Exit:
        raise
    except Exception as e:  # noqa: BLE001 - startup problems get a hint, not a traceback
        console.print(f"[red]Startup failed:[/red] {type(e).__name__}: {str(e)[:300]}\n"
                      "Check .env (GEMINI_API_KEY, GCP_PROJECT_ID) and run "
                      "[bold]gcloud auth application-default login[/bold].")
        raise typer.Exit(1)
    return application, principal, who, log_file


def _show(state: dict, elapsed: float) -> None:
    status = state.get("status", "done")
    style = {"blocked": "yellow", "failed": "red", "awaiting_confirmation": "yellow"}.get(status)
    title = "Confirmation needed" if status == "awaiting_confirmation" else "Analyst"
    console.print(Panel(Markdown(state.get("answer", "")), border_style=style or "cyan", title=title, title_align="left"))
    parts = [status, f"trace {state.get('trace_id', '-')}", f"{elapsed:.1f}s", f"{state.get('llm_calls', 0)} LLM calls"]
    if state.get("attempts"):
        parts.append(f"{state['attempts']} SQL attempt(s)")
    if state.get("intent") in ("analysis", "report") and status == "done":
        parts.append("rate with /up or /down")
    console.print("[dim]" + " · ".join(parts) + "[/dim]\n")


def _thread(principal: dict) -> str:
    return f"{principal['role']}-{principal['user_id'] or 'all'}-{uuid.uuid4().hex[:8]}"


@app.command()
def chat(user_id: Optional[int] = USER_ID, role: str = ROLE, name: str = NAME, thread: Optional[str] = THREAD) -> None:
    """Interactive conversation."""
    application, principal, who, log_file = _start(user_id, role, name)
    subject = subject_of(principal)
    thread_id = thread or _thread(principal)
    console.print(Panel(f"Signed in as [bold]{who}[/bold]\nThread [dim]{thread_id}[/dim] · logs [dim]{log_file}[/dim]\n" + HELP,
                        title="Retail Insights Analyst", border_style="green"))
    last: dict = {}
    while True:
        try:
            q = console.input("[bold green]You ›[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            return
        if not q:
            continue
        cmd = q.lower()
        if cmd in ("/exit", "/quit"):
            return
        if cmd == "/help":
            console.print(HELP)
        elif cmd == "/new":
            thread_id = _thread(principal)
            console.print(f"New conversation: [dim]{thread_id}[/dim]")
        elif cmd == "/sql":
            console.print(f"[dim]{last.get('executed_sql') or 'No query has run yet.'}[/dim]")
        elif cmd == "/data":
            console.print(Markdown(last["table_md"]) if last.get("table_md") else "[dim]No result table yet.[/dim]")
        elif cmd == "/prefs":
            console.print(application.preferences.describe(subject))
        elif cmd == "/forget":
            application.preferences.forget(subject)
            console.print("Cleared everything I had learned about your preferences.")
        elif cmd == "/up" or cmd == "/down" or cmd.startswith("/down "):
            res = record_feedback(application.feedback, application.preferences, application.golden_root,
                                  last, 1 if cmd == "/up" else -1, q[5:].strip() if cmd.startswith("/down") else "")
            console.print(f"[dim]{res.message}[/dim]")
        else:
            t0 = time.perf_counter()
            try:
                with console.status("Analyzing..."):
                    last = run_turn(application.graph, q, principal, thread_id)
                _show(last, time.perf_counter() - t0)
            except KeyboardInterrupt:
                console.print("[yellow]Cancelled.[/yellow]")
            except Exception:  # noqa: BLE001 - the UI must never crash
                console.print("[red]Something went wrong on my side. Please try again.[/red]")


@app.command()
def ask(question: str = typer.Argument(..., help="Question to answer."), user_id: Optional[int] = USER_ID,
        role: str = ROLE, name: str = NAME, thread: Optional[str] = THREAD) -> None:
    """Ask a single question and exit (handy for demos and scripts)."""
    application, principal, _who, _log = _start(user_id, role, name)
    t0 = time.perf_counter()
    state = run_turn(application.graph, question, principal, thread or uuid.uuid4().hex[:8])
    _show(state, time.perf_counter() - t0)


# ------------------------------------------------------------------ observability
@app.command()
def metrics(since_hours: Optional[float] = typer.Option(None, "--since-hours", help="Only look at the last N hours."),
            as_json: bool = typer.Option(False, "--json"),
            price_in: Optional[float] = typer.Option(None, "--price-in", help="USD per 1M input tokens (enables cost estimates)."),
            price_out: Optional[float] = typer.Option(None, "--price-out", help="USD per 1M output tokens.")) -> None:
    """Agent-level health metrics computed from logs/agent.jsonl."""
    m = compute_metrics(filter_since(load_events(ROOT / "logs" / "agent.jsonl"), since_hours), price_in, price_out)
    if as_json:
        console.print_json(data=m)
        return
    if not m["turns"]:
        console.print("[dim]No completed turns in the log yet.[/dim]")
        return
    from rich.table import Table
    t = Table(title="Agent metrics", show_header=False)
    for k, v in m.items():
        t.add_row(k, str(v))
    console.print(t)


@app.command()
def trace(trace_id: str) -> None:
    """Replay one turn from the log: router decision, every SQL attempt and its feedback, LLM calls, guards."""
    lines = trace_timeline(load_events(ROOT / "logs" / "agent.jsonl"), trace_id)
    console.print("\n".join(lines) if lines else f"[red]No events for trace {trace_id}.[/red]", markup=False, highlight=False)


# ---------------------------------------------------------------- analyst curation (golden bucket)
@app.command()
def candidates() -> None:
    """List golden-bucket candidates waiting for analyst review."""
    rows = list_candidates(ROOT / "golden")
    if not rows:
        console.print("[dim]No candidates waiting for review.[/dim]")
    for r in rows:
        console.print(f"[bold]{r['id']}[/bold]  {r['question']}")


@app.command()
def approve(candidate_id: str, reviewer: str = typer.Option(..., "--reviewer", help="Your name or email (audit trail)."),
            allow_duplicate: bool = typer.Option(False, "--allow-duplicate")) -> None:
    """Promote a candidate into the golden bucket (SQL validation + duplicate check run first)."""
    try:
        dest = promote(ROOT / "golden", candidate_id, reviewer, allow_duplicate=allow_duplicate)
    except CurationError as e:
        console.print(f"[red]Not promoted:[/red] {e}")
        raise typer.Exit(1)
    console.print(f"Promoted to [bold]{dest.name}[/bold]. Edit the file to refine the interpretation/report, then restart.")


@app.command()
def reject(candidate_id: str, reviewer: str = typer.Option(..., "--reviewer"),
           reason: str = typer.Option(..., "--reason")) -> None:
    """Reject a candidate (kept under golden/rejected for audit and eval negatives)."""
    try:
        reject_candidate(ROOT / "golden", candidate_id, reviewer, reason)
    except CurationError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    console.print("Rejected.")


if __name__ == "__main__":
    app()
