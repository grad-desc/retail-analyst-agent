"""Node logic for the agent graph. Plain functions over a state dict, so the flow control can be
unit-tested without LangGraph or network access.

Recovery strategy by failure class:
  invalid / blocked SQL, BigQuery SQL error, over-budget query -> feedback to the LLM (<= N attempts)
  empty result                                                  -> one retry with a hint, then report "no data"
  BigQuery down / LLM down / budget spent                       -> immediate friendly message (no more LLM spend)
"""
import functools
import logging
import re
import secrets
import time
from datetime import date, datetime, timezone
from typing import Any, Dict

from langgraph.types import interrupt

from src.data_summary import future_periods, summarize_result
from src.guards import input_guard
from src.guards.output_scrubber import scrub_dataframe, scrub_text
from src.jsonutil import parse_json
from src.llm.client import BudgetExceeded, LLMRequestError, LLMUnavailable, TurnBudget
from src.memory import preferences as prefs_mod
from src.reports import delete_flow
from src.sql.errors import BQUnavailable, QueryTooExpensive, SQLSyntaxError, SQLValidationError
from src.sql.policy import Role, UserPolicy, subject_of
from src.state import Deps

log = logging.getLogger(__name__)

INTENTS = {"analysis", "report", "schema", "manage_reports", "preferences", "pii_request", "unsafe", "off_topic"}
MAX_QUESTION_CHARS = 4000

FRIENDLY = {
    "llm_unavailable": "I'm having trouble reaching the language model right now. Please try again in a minute.",
    "bq_unavailable": "The data warehouse is temporarily unavailable. Please try again shortly.",
    "budget": "This question needed more processing than I can spend in one turn. Try narrowing it down.",
    "sql_failed": ("I couldn't build a working query for that after a few attempts. Try rephrasing, "
                   "or ask about a narrower slice (a period, a category or a brand)."),
    "blocked": "I can't help with that request. I can only answer questions that analyze the retail data.",
    "pii": ("I can't show personal data such as names, emails, phone numbers or addresses. I can analyze "
            "behavior in aggregate instead, for example spend by state, age group or traffic source."),
    "off_topic": ("I'm a retail data analysis assistant: I can answer questions about orders, products, "
                  "customers and revenue. What would you like to analyze?"),
    "internal": "Something went wrong on my side. Please try again.",
}
FUTURE_HINT = ("The result contains periods after today ({today}): {cols}. Those periods are incomplete or have not "
               "happened yet, so they are not valid findings (a month label one month ahead usually means a query "
               "bug, e.g. 100% churn because nobody can have bought yet). Exclude the current incomplete period and "
               "anything later, and follow any rule about incomplete periods in the analyst examples.")
EMPTY_HINT = ("The query ran but returned no rows. Re-check status values, date ranges, joins and filters. "
              "If they are already correct, keep the query as is.")
FORMAT_HINTS = {
    "default": "short headline, then supporting points; add a small table only if it helps.",
    "report": ("a structured report with these sections: Summary (2-3 sentences), Key findings (data-backed bullets), "
               "Risks or anomalies, and Recommended actions (3-5 concrete action items, each with a priority)."),
}
REPORT_ACTIONS = {"list", "show", "delete", "undo"}


owner_of = subject_of   # saved reports, preferences and feedback all belong to the same subject


def _today() -> date:
    """UTC calendar date; a module-level function so tests can pin it."""
    return datetime.now(timezone.utc).date()


def _title(question: str) -> str:
    t = re.sub(r"\s+", " ", question).strip()
    return scrub_text(t[:77] + "..." if len(t) > 80 else t)[0] or "Untitled report"


def policy_from(principal: Dict[str, Any]) -> UserPolicy:
    if principal.get("role") == Role.EXECUTIVE.value:
        return UserPolicy.executive()
    return UserPolicy.customer(int(principal["user_id"]))


def scope_note(principal: Dict[str, Any]) -> str:
    if principal.get("role") == Role.EXECUTIVE.value:
        return "You have company-wide access to the data."
    return ("Data is automatically limited to this customer's own orders and the products in them. "
            "Company-wide totals and other customers are not visible; say so if the question needs them.")


_parse_json = parse_json


def _fail(code: str, detail: str = "") -> Dict[str, Any]:
    log.warning("turn_failed", extra={"failure": code, "detail": detail[:300]})
    return {"status": "failed", "failure": code, "answer": FRIENDLY[code]}


def _hist(state, n: int = 6) -> str:
    items = state.get("history", [])[-n:]
    return "\n".join(f"{m['role'].capitalize()}: {m['content'][:600]}" for m in items) or "(none)"


def df_to_markdown(df, max_rows: int) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:,.2f}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    cols = [str(c) for c in df.columns]
    rows = [[fmt(v) for v in r] for r in df.head(max_rows).itertuples(index=False, name=None)]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def make_nodes(deps: Deps) -> Dict[str, Any]:
    def generate(state, prompt: str, **kw):
        budget = TurnBudget(deps.max_llm_calls, used=state.get("llm_calls", 0))
        res = deps.llm.generate(prompt, budget=budget, trace_id=state.get("trace_id"), **kw)
        return res, budget.used

    def guarded(name: str):
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(state):
                t0 = time.perf_counter()
                try:
                    out = fn(state)
                except LLMUnavailable as e:
                    out = _fail("llm_unavailable", e.detail)
                except BudgetExceeded as e:
                    out = _fail("budget", str(e))
                except BQUnavailable as e:
                    out = _fail("bq_unavailable", str(e))
                except LLMRequestError as e:
                    out = _fail("internal", f"llm request error: {e}")
                except Exception as e:  # noqa: BLE001 - never crash the UI
                    log.exception("node_crashed", extra={"node": name, "trace_id": state.get("trace_id")})
                    out = _fail("internal", type(e).__name__)
                log.info("node_done", extra={
                    "trace_id": state.get("trace_id"), "node": name,
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                    "status": out.get("status", state.get("status")),
                    "attempts": out.get("attempts", state.get("attempts"))})
                return out
            return wrapper
        return deco

    # ------------------------------------------------------------------ nodes
    @guarded("route")
    def route(state):
        q = (state.get("question") or "").strip()
        base = dict(status="ok", failure=None, intent=None, sql=None, validated_sql=None,
                    executed_sql=None, feedback=None, attempts=0, empty_retries=0, llm_calls=0,
                    row_count=0, truncated=False, capped=False, sampled=False, future_retries=0,
                    data_csv="", table_md="", answer="", guard={},
                    pending_delete=None, saved_report_id=None, golden_ids=[], resolved_question=q)
        if not q:
            return {**base, "intent": "off_topic"}
        if len(q) > MAX_QUESTION_CHARS:
            return {**base, "intent": "unsafe", "guard": {"input_block": "too_long"}}
        hit = input_guard.check(q)
        if hit:
            log.warning("security_block", extra={"trace_id": state.get("trace_id"), "layer": "input_guard",
                                                 "pattern": hit})
            return {**base, "intent": "unsafe", "guard": {"input_block": hit}}
        # `state` still holds the previous turn's llm_calls (persisted by the checkpointer): start this turn at 0
        res, used = generate({**state, "llm_calls": 0}, deps.prompts.render("router", question=q, history=_hist(state)),
                             json_mode=True, temperature=0.0)
        data = _parse_json(res.text) or {}
        intent = data.get("intent") if data.get("intent") in INTENTS else "analysis"  # downstream guards still apply
        resolved = str(data.get("standalone_question") or q)[:MAX_QUESTION_CHARS]
        if intent not in ("unsafe", "pii_request"):          # learn only from legitimate messages
            sigs = prefs_mod.detect(q)
            if sigs:
                deps.preferences.observe(owner_of(state["principal"]), sigs,
                                         standing=prefs_mod.is_standing(q) or intent == "preferences")
                log.info("preferences_updated", extra={"trace_id": state.get("trace_id"),
                                                       "signals": [f"{s.key}={s.value}" for s in sigs]})
        return {**base, "intent": intent, "resolved_question": resolved, "llm_calls": used}

    @guarded("refuse")
    def refuse(state):
        intent = state.get("intent")
        if intent == "pii_request":
            return {"status": "blocked", "answer": FRIENDLY["pii"], "guard": {**state.get("guard", {}), "pii_request": True}}
        if intent == "unsafe":
            return {"status": "blocked", "answer": FRIENDLY["blocked"]}
        return {"answer": FRIENDLY["off_topic"]}

    @guarded("schema_qa")
    def schema_qa(state):
        prompt = deps.prompts.render("schema_qa", schema=deps.schema_text,
                                     scope=scope_note(state["principal"]),
                                     question=state["resolved_question"])
        res, used = generate(state, prompt, temperature=0.3)
        return {"answer": res.text.strip(), "llm_calls": used}

    @guarded("preferences")
    def set_preferences(state):
        owner, q = owner_of(state["principal"]), state.get("question", "")
        if prefs_mod.wants_forget(q):
            deps.preferences.forget(owner)
            return {"answer": "Done: I've cleared everything I had learned about your preferences."}
        sigs = prefs_mod.detect(q)
        if sigs and not prefs_mod.wants_recall(q):
            return {"answer": f"Got it: I'll {prefs_mod.describe_signals(sigs)}. "
                              'Say "forget my preferences" any time to reset.'}
        return {"answer": deps.preferences.describe(owner)}

    @guarded("manage_reports")
    def manage_reports(state):
        """Phase 1: understand the request. list/show/undo run immediately; delete only PLANS
        (resolve + preview) and hands over to confirm_delete. Nothing is deleted in this node."""
        owner, trace, store = owner_of(state["principal"]), state.get("trace_id"), deps.reports
        res, used = generate(state, deps.prompts.render("reports_action", question=state["resolved_question"],
                                                        history=_hist(state)), json_mode=True, temperature=0.0)
        parsed = _parse_json(res.text) or {}
        action = parsed.get("action") if parsed.get("action") in REPORT_ACTIONS else "unclear"
        out: Dict[str, Any] = {"llm_calls": used}
        crit = delete_flow.parse_criteria(parsed)
        if action == "list":
            out["answer"] = delete_flow.format_listing(store.list(owner, limit=20))
        elif action == "show":
            rep = store.get(owner, crit.ids[0]) if crit.ids else None
            out["answer"] = (f"**{rep.title}** (report #{rep.id})\n\n{rep.content}" if rep else
                             'I couldn\'t find that report. Say "list my reports" to see what is saved.')
        elif action == "undo":
            ids = store.restore_last_batch(owner, delete_flow.UNDO_WINDOW_S)
            store.audit(owner, "undo", ids, {}, trace)
            out["answer"] = (f"Restored {len(ids)} report{'s' if len(ids) != 1 else ''}." if ids else
                             "Nothing to undo: no reports were deleted in the last "
                             f"{delete_flow.UNDO_WINDOW_S // 60} minutes.")
        elif action == "delete":
            if not crit.is_specific:
                out["answer"] = ("Which reports should I delete? You can give a report number, a keyword they "
                                 'mention, say "this conversation", or say "all" to remove everything.')
            else:
                thread = state.get("thread_id") if crit.scope == "conversation" else None
                matches = [] if (crit.scope == "conversation" and not thread) else store.find(
                    owner, ids=crit.ids or None, mentions=crit.mentions or None, thread_id=thread,
                    match_all=crit.scope == "all")
                if not matches:
                    out["answer"] = "No saved reports match that request, so there is nothing to delete."
                else:
                    plan = {"token": secrets.token_hex(4), "ids": [r.id for r in matches],
                            "expires_at": time.time() + deps.confirm_ttl_s, "criteria": crit.describe()}
                    store.audit(owner, "delete_planned", plan["ids"], {"criteria": plan["criteria"]}, trace)
                    log.info("confirmation_requested", extra={"trace_id": trace, "count": len(matches),
                                                              "token": plan["token"]})
                    out.update(pending_delete=plan, answer=delete_flow.format_preview(matches, crit))
        else:
            out["answer"] = ("I can list your saved reports, open one by number, delete reports (I'll always ask "
                             "you to confirm first), or undo a recent deletion.")
        return out

    @guarded("execute_delete")
    def execute_delete(state):
        """Phase 2 (after the human answered): acts ONLY on the ids that were previewed."""
        plan, owner, trace = state["pending_delete"], owner_of(state["principal"]), state.get("trace_id")
        decision, store = state["_decision"], deps.reports
        confirmed = isinstance(decision, dict) and decision.get("confirmed") is True   # strict: only literal True
        if not confirmed:
            store.audit(owner, "delete_cancelled", plan["ids"], {"token": plan["token"]}, trace)
            answer = "Okay, nothing was deleted."
        elif time.time() > plan["expires_at"]:
            store.audit(owner, "delete_expired", plan["ids"], {"token": plan["token"]}, trace)
            answer = "That confirmation expired, so nothing was deleted. Ask again if you still want to delete them."
        else:
            done = store.soft_delete(owner, plan["ids"], batch=plan["token"])
            store.audit(owner, "delete_confirmed", done, {"token": plan["token"], "planned": plan["ids"]}, trace)
            answer = (f"Deleted {len(done)} report{'s' if len(done) != 1 else ''}. "
                      f"Say \"undo\" within {delete_flow.UNDO_WINDOW_S // 60} minutes to restore "
                      f"{'them' if len(done) != 1 else 'it'}.")
        return {"pending_delete": None, "answer": answer}

    def confirm_delete(state):
        """Pauses the graph until the user answers. NOT wrapped by `guarded`: interrupt() works by
        raising, and a blanket except would swallow the pause. On resume this node re-runs from the
        top, so everything before interrupt() is side-effect free."""
        plan = state["pending_delete"]
        decision = interrupt({"type": "confirm_delete", "token": plan["token"], "count": len(plan["ids"]),
                              "expires_at": plan["expires_at"]})
        return execute_delete({**state, "_decision": decision})

    @guarded("save_report")
    def save_report(state):
        if state.get("intent") != "report" or state.get("status") != "ok":
            return {}
        owner, content = owner_of(state["principal"]), scrub_text(state.get("answer", ""))[0]
        title = _title(state["resolved_question"])
        rid = deps.reports.save(owner, state.get("thread_id"), title, scrub_text(state.get("question", ""))[0], content)
        deps.reports.audit(owner, "save", [rid], {"title": title}, state.get("trace_id"))
        return {"answer": f"{content}\n\n---\n*Saved as report #{rid}: {title}*", "saved_report_id": rid}

    @guarded("gen_sql")
    def gen_sql(state):
        attempts = state.get("attempts", 0) + 1
        principal = state["principal"]
        ctx = deps.golden.examples_for(state["resolved_question"]) if deps.golden else None
        if ctx and attempts == 1:
            log.info("golden_retrieved", extra={"trace_id": state.get("trace_id"), "ids": list(ctx.ids),
                                                "top_score": round(ctx.top_score, 3)})
        prompt = deps.prompts.render(
            "sql_gen", schema=deps.schema_text, scope=scope_note(principal), today=_today().isoformat(),
            examples=(ctx.sql_examples if ctx else "") or "(none)",
            history=_hist(state), question=state["resolved_question"],
            previous_sql=state.get("sql") or "(none)", feedback=state.get("feedback") or "(none)")
        res, used = generate(state, prompt, json_mode=True, temperature=0.0)
        sql = (_parse_json(res.text) or {}).get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return {"sql": None, "validated_sql": None, "attempts": attempts, "llm_calls": used,
                    "golden_ids": list(ctx.ids) if ctx else [],
                    "feedback": "Reply was not valid JSON with a non-empty 'sql' field."}
        return {"sql": sql.strip(), "validated_sql": None, "feedback": None, "attempts": attempts,
                "llm_calls": used, "golden_ids": list(ctx.ids) if ctx else []}

    @guarded("validate")
    def validate(state):
        try:
            v = deps.validate(state["sql"], policy_from(state["principal"]))
        except SQLValidationError as e:
            if e.security:
                log.warning("security_block", extra={"trace_id": state.get("trace_id"), "layer": "sql_validator",
                                                     "kind": e.kind, "role": state["principal"].get("role")})
            return {"feedback": e.feedback, "validated_sql": None,
                    "guard": {**state.get("guard", {}), "sql_blocked": state.get("guard", {}).get("sql_blocked", 0) + 1}}
        return {"validated_sql": v.sql, "feedback": None}

    @guarded("fetch")
    def fetch(state):
        """Execute AND scrub in one node: raw rows never leave this function."""
        sql = state["validated_sql"]
        try:
            df = deps.runner.execute_query(sql, trace_id=state.get("trace_id"))
        except (SQLSyntaxError, QueryTooExpensive) as e:
            return {"feedback": f"[bigquery] {e}"}
        # BQUnavailable propagates to the wrapper: friendly failure, zero extra LLM spend.
        if len(df) == 0 and state.get("empty_retries", 0) < 1 and state.get("attempts", 0) < deps.max_sql_attempts:
            return {"feedback": EMPTY_HINT, "empty_retries": state.get("empty_retries", 0) + 1}
        today = _today()
        future = future_periods(df, today) if len(df) else []
        if future and state.get("future_retries", 0) < 1 and state.get("attempts", 0) < deps.max_sql_attempts:
            return {"feedback": FUTURE_HINT.format(today=today.isoformat(), cols=", ".join(future)),
                    "future_retries": state.get("future_retries", 0) + 1}
        notes = ([f"{', '.join(future)} lie after today ({today.isoformat()}): that period is incomplete or in the "
                  "future. Do not present it as a finding or a spike."] if future else [])
        clean, rep = scrub_dataframe(df)
        guard = {**state.get("guard", {}), **rep.as_log()}
        capped, sampled = bool(clean.attrs.get("truncated")), len(clean) > deps.max_prompt_rows
        return {"feedback": None, "executed_sql": sql, "row_count": len(clean),
                "truncated": capped or sampled, "capped": capped, "sampled": sampled,
                "data_csv": summarize_result(clean, max_rows=deps.max_prompt_rows, notes=notes),
                "table_md": df_to_markdown(clean, deps.max_prompt_rows) if len(clean) else "",
                "guard": guard}

    @guarded("write_report")
    def write_report(state):
        ctx = deps.golden.examples_for(state["resolved_question"]) if deps.golden else None
        prompt = deps.prompts.render(
            "report", preferences=deps.preferences.render(owner_of(state["principal"]),
                                                          prefs_mod.detect(state.get("question", ""))) or "(none recorded)",
            examples=(ctx.report_examples if ctx else "") or "(none)",
            format_hint=FORMAT_HINTS["report" if state.get("intent") == "report" else "default"], question=state["resolved_question"],
            today=_today().isoformat(), row_count=state.get("row_count", 0),
            truncated_note=((", cut at the row cap, so totals may be incomplete" if state.get("capped") else "") +
                            (f", first {deps.max_prompt_rows} shown plus summaries over all returned rows"
                             if state.get("sampled") else "")),
            data=(state.get("data_csv") if state.get("row_count") else "") or "(no rows)", sql=state.get("executed_sql") or "(none)")
        res, used = generate(state, prompt, temperature=0.3)
        return {"answer": res.text.strip(), "llm_calls": used}

    @guarded("give_up")
    def give_up(state):
        return _fail("sql_failed", f"attempts={state.get('attempts')} last_feedback={state.get('feedback')}")

    def finalize(state):
        """Output guard: last-line PII scrub of the final text, then record the turn."""
        answer = state.get("answer") or FRIENDLY["internal"]
        clean, rep = scrub_text(answer)
        status = state.get("status", "ok")
        if status == "ok":
            status = "done"
        guard = {**state.get("guard", {})}
        if not rep.clean:
            guard["output_redactions"] = dict(rep.redactions)
            log.warning("output_redacted", extra={"trace_id": state.get("trace_id"), **rep.as_log()})
        log.info("turn_done", extra={
            "trace_id": state.get("trace_id"), "status": status, "failure": state.get("failure"),
            "intent": state.get("intent"), "role": state.get("principal", {}).get("role"),
            "attempts": state.get("attempts"), "llm_calls": state.get("llm_calls"),
            "thread_id": state.get("thread_id"), "rows": state.get("row_count"), "golden_ids": state.get("golden_ids"), "question": (state.get("question") or "")[:2000],
            "answer": clean[:4000], "sql": state.get("executed_sql"),
            "guard": guard, "prompt_versions": {p: deps.prompts.version(p) for p in
                                                ("persona", "router", "sql_gen", "report")}})
        return {"answer": clean, "status": status, "guard": guard,
                "history": [{"role": "user", "content": state.get("question", "")},
                            {"role": "assistant", "content": clean}]}

    # ---------------------------------------------------------------- routing
    def _terminal(state) -> bool:
        return state.get("status") == "failed"

    def _retry_or_give_up(state) -> str:
        return "gen_sql" if state.get("attempts", 0) < deps.max_sql_attempts else "give_up"

    def after_route(state) -> str:
        if _terminal(state):
            return "finalize"
        return {"analysis": "gen_sql", "report": "gen_sql", "schema": "schema_qa",
                "manage_reports": "manage_reports", "preferences": "preferences"}.get(state.get("intent"), "refuse")

    def after_gen(state) -> str:
        if _terminal(state):
            return "finalize"
        return "validate" if state.get("sql") else _retry_or_give_up(state)

    def after_validate(state) -> str:
        if _terminal(state):
            return "finalize"
        return _retry_or_give_up(state) if state.get("feedback") else "fetch"

    def after_manage(state) -> str:
        if _terminal(state):
            return "finalize"
        return "confirm_delete" if state.get("pending_delete") else "finalize"

    def after_fetch(state) -> str:
        if _terminal(state):
            return "finalize"
        return _retry_or_give_up(state) if state.get("feedback") else "write_report"

    return {"route": route, "refuse": refuse, "schema_qa": schema_qa, "manage_reports": manage_reports,
            "confirm_delete": confirm_delete, "save_report": save_report, "preferences": set_preferences, "after_manage": after_manage,
            "gen_sql": gen_sql, "validate": validate, "fetch": fetch, "write_report": write_report,
            "give_up": give_up, "finalize": finalize, "after_route": after_route, "after_gen": after_gen,
            "after_validate": after_validate, "after_fetch": after_fetch}
