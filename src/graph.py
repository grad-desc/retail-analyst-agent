"""LangGraph wiring. All logic lives in nodes.py; this file only declares the topology.

START -> route -+-> refuse -------------------------------------------+
                +-> schema_qa / preferences ----------------------------+
                +-> manage_reports -+-> (no delete planned) ------------+--> finalize -> END
                |                   +-> confirm_delete [PAUSE: human] -+
                +-> gen_sql -> validate -> fetch -> write_report -> save_report -+
                      ^           |          |                                    |
                      +-----------+----------+ (feedback, <= N attempts) -> give_up +
"""
import uuid
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.guards.output_scrubber import scrub_text
from src.nodes import make_nodes
from src.reports.delete_flow import parse_confirmation
from src.state import AgentState, Deps


def build_graph(deps: Deps, checkpointer=None):
    n = make_nodes(deps)
    g = StateGraph(AgentState)
    for name in ("route", "refuse", "schema_qa", "preferences", "manage_reports", "confirm_delete", "gen_sql", "validate",
                 "fetch", "write_report", "save_report", "give_up", "finalize"):
        g.add_node(name, n[name])

    g.add_edge(START, "route")
    g.add_conditional_edges("route", n["after_route"], {
        "gen_sql": "gen_sql", "schema_qa": "schema_qa", "manage_reports": "manage_reports",
        "preferences": "preferences", "refuse": "refuse", "finalize": "finalize"})
    g.add_conditional_edges("manage_reports", n["after_manage"], {
        "confirm_delete": "confirm_delete", "finalize": "finalize"})
    g.add_conditional_edges("gen_sql", n["after_gen"], {
        "validate": "validate", "gen_sql": "gen_sql", "give_up": "give_up", "finalize": "finalize"})
    g.add_conditional_edges("validate", n["after_validate"], {
        "fetch": "fetch", "gen_sql": "gen_sql", "give_up": "give_up", "finalize": "finalize"})
    g.add_conditional_edges("fetch", n["after_fetch"], {
        "write_report": "write_report", "gen_sql": "gen_sql", "give_up": "give_up", "finalize": "finalize"})
    g.add_edge("write_report", "save_report")
    for terminal in ("refuse", "schema_qa", "preferences", "confirm_delete", "save_report", "give_up"):
        g.add_edge(terminal, "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def pending_confirmation(graph, thread_id: str) -> Optional[Dict[str, Any]]:
    """The payload of an open confirmation on this thread, or None."""
    snap = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not getattr(snap, "next", None):
        return None
    for task in snap.tasks:
        for it in task.interrupts:
            return it.value
    return None


def _with_pending(graph, result: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
    out = dict(result)
    pending = pending_confirmation(graph, thread_id)
    if pending:   # the preview is built from stored titles: scrub it like any other output
        out.update(status="awaiting_confirmation", pending_confirmation=pending,
                   answer=scrub_text(out.get("answer", ""))[0])
    return out


def run_turn(graph, question: str, principal: Dict[str, Any], thread_id: str,
             trace_id: Optional[str] = None) -> Dict[str, Any]:
    """One conversational turn.

    If a destructive action is waiting for confirmation, this message is its answer:
      explicit yes -> executes the previewed plan; explicit no -> cancels;
      anything else -> cancels (ambiguity is never consent) and the message is then handled as a new question.
    The user just chats; there is no separate confirmation UI to break the flow.
    """
    cfg = {"configurable": {"thread_id": thread_id}}
    prefix = ""
    if pending_confirmation(graph, thread_id):
        decision = parse_confirmation(question)
        final = graph.invoke(Command(resume={"confirmed": decision is True}), cfg)
        if decision is not None:
            return _with_pending(graph, final, thread_id)
        prefix = "_I didn't get a clear yes, so nothing was deleted._\n\n"
    result = graph.invoke(
        {
            "question": question,
            "principal": principal,
            "thread_id": thread_id,
            "trace_id": trace_id or uuid.uuid4().hex[:12],
            "llm_calls": 0,  # ← reset per turn: llm_calls accumulates via the checkpointer otherwise
        },
        config=cfg,
    )
    out = _with_pending(graph, result, thread_id)
    if prefix:
        out["answer"] = prefix + out.get("answer", "")
    return out
