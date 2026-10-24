"""End-to-end tests of save / list / confirm-before-delete / undo through the real graph topology."""
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from src.graph import build_graph, pending_confirmation, run_turn  # noqa: E402
from src.prompts import PromptRegistry  # noqa: E402
from src.reports.store import ReportStore  # noqa: E402
from src.state import Deps  # noqa: E402
from tests.test_graph import ANALYSIS, FakeLLM, FakeRunner, fake_validate, sql  # noqa: E402

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
ME = {"role": "customer", "user_id": 123}
ME_KEY, OTHER_KEY = "customer:123", "customer:456"
MANAGE = '{"intent": "manage_reports", "standalone_question": "x"}'


def action(**kw):
    import json
    return json.dumps({"action": "delete", "report_ids": [], "mentions": [], "scope": "none", **kw})


def setup(script, ttl=300, runner=None):
    store = ReportStore(":memory:")
    ids = {
        "a": store.save(ME_KEY, "t", "Q1 revenue for Client X", "q", "body"),
        "b": store.save(ME_KEY, "t", "Weekly overview", "q", "mentions client x in the body"),
        "c": store.save(ME_KEY, "other", "Unrelated", "q", "nothing"),
        "d": store.save(OTHER_KEY, "t", "Client X deep dive", "q", "body"),
    }
    llm = FakeLLM(script)
    deps = Deps(llm=llm, runner=runner or FakeRunner([]), prompts=PromptRegistry(PROMPTS), schema_text="s",
                validate=fake_validate, reports=store, confirm_ttl_s=ttl)
    return build_graph(deps, MemorySaver()), store, ids, llm


def live(store, owner=ME_KEY):
    return {r.id for r in store.list(owner)}


def test_delete_requires_confirmation_then_removes_only_the_previewed_own_reports():
    g, store, ids, llm = setup([MANAGE, action(mentions=["Client X"])])
    r1 = run_turn(g, "Delete all reports mentioning Client X", ME, "t")
    assert r1["status"] == "awaiting_confirmation" and "2 reports" in r1["answer"] and "yes" in r1["answer"]
    assert live(store) == {ids["a"], ids["b"], ids["c"]}                 # NOTHING deleted before the user says yes
    assert pending_confirmation(g, "t")["count"] == 2

    r2 = run_turn(g, "yes", ME, "t")
    assert r2["status"] == "done" and "Deleted 2 reports" in r2["answer"] and "undo" in r2["answer"]
    assert live(store) == {ids["c"]}
    assert live(store, OTHER_KEY) == {ids["d"]}                          # another owner's report untouched
    assert len(llm.prompts) == 2                                         # "yes" cost no LLM call
    actions = [x["action"] for x in store.audit_log(ME_KEY)]
    assert actions == ["delete_confirmed", "delete_planned"]
    assert pending_confirmation(g, "t") is None


def test_no_cancels():
    g, store, ids, _ = setup([MANAGE, action(mentions=["Client X"])])
    run_turn(g, "delete reports mentioning client x", ME, "t")
    r = run_turn(g, "no", ME, "t")
    assert "nothing was deleted" in r["answer"] and len(live(store)) == 3
    assert store.audit_log(ME_KEY)[0]["action"] == "delete_cancelled"


def test_ambiguous_reply_cancels_and_is_handled_as_a_new_question():
    g, store, ids, _ = setup([MANAGE, action(mentions=["Client X"]),
                              '{"intent": "schema", "standalone_question": "data?"}', "We have orders."])
    run_turn(g, "delete reports mentioning client x", ME, "t")
    r = run_turn(g, "what data do we have?", ME, "t")
    assert r["answer"].startswith("_I didn't get a clear yes") and "We have orders." in r["answer"]
    assert len(live(store)) == 3 and pending_confirmation(g, "t") is None


def test_confirmation_expires():
    g, store, _, _ = setup([MANAGE, action(mentions=["Client X"])], ttl=-1)
    run_turn(g, "delete reports mentioning client x", ME, "t")
    r = run_turn(g, "yes", ME, "t")
    assert "expired" in r["answer"] and len(live(store)) == 3
    assert store.audit_log(ME_KEY)[0]["action"] == "delete_expired"


def test_only_the_previewed_ids_are_deleted_even_if_new_matches_appear_before_yes():
    g, store, ids, _ = setup([MANAGE, action(mentions=["Client X"])])
    run_turn(g, "delete reports mentioning client x", ME, "t")
    late = store.save(ME_KEY, "t", "Client X addendum", "q", "body")     # created after the preview
    run_turn(g, "yes", ME, "t")
    assert live(store) == {ids["c"], late}


def test_delete_without_criteria_asks_instead_of_deleting():
    g, store, _, _ = setup([MANAGE, action()])
    r = run_turn(g, "delete my reports", ME, "t")
    assert r["status"] == "done" and "Which reports" in r["answer"] and len(live(store)) == 3
    assert pending_confirmation(g, "t") is None


def test_conversation_scope_deletes_only_this_threads_reports():
    g, store, ids, _ = setup([MANAGE, action(scope="conversation")])
    r = run_turn(g, "delete all the reports we made in this conversation", ME, "t")
    assert "2 reports" in r["answer"] and "created in this conversation" in r["answer"]
    run_turn(g, "yes", ME, "t")
    assert live(store) == {ids["c"]}                                     # report from thread "other" survives


def test_scope_all_is_explicit_and_still_owner_scoped():
    g, store, ids, _ = setup([MANAGE, action(scope="all")])
    assert "3 reports" in run_turn(g, "delete everything", ME, "t")["answer"]
    run_turn(g, "yes", ME, "t")
    assert live(store) == set() and live(store, OTHER_KEY) == {ids["d"]}


def test_no_match_means_no_confirmation_prompt():
    g, store, _, _ = setup([MANAGE, action(mentions=["Nobody"])])
    r = run_turn(g, "delete reports mentioning Nobody", ME, "t")
    assert r["status"] == "done" and "nothing to delete" in r["answer"]


def test_undo_restores_the_last_deletion():
    g, store, ids, _ = setup([MANAGE, action(mentions=["Client X"]), MANAGE, '{"action": "undo"}'])
    run_turn(g, "delete reports mentioning client x", ME, "t")
    run_turn(g, "yes", ME, "t")
    r = run_turn(g, "undo that", ME, "t")
    assert "Restored 2 reports" in r["answer"] and len(live(store)) == 3


def test_list_and_show_are_owner_scoped():
    g, store, ids, _ = setup([MANAGE, '{"action": "list"}', MANAGE, '{"action": "show", "report_ids": [%d]}' % ids_d()])
    assert "Q1 revenue for Client X" in run_turn(g, "list my reports", ME, "t")["answer"]
    assert "couldn't find" in run_turn(g, "show report 4", ME, "t")["answer"]   # id 4 belongs to someone else


def ids_d():
    return 4


def test_report_request_is_saved_with_pii_scrubbed():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"category": ["Jeans"], "revenue": [10.0]})
    g, store, ids, _ = setup(['{"intent": "report", "standalone_question": "Q1 report for Client Y"}',
                              sql("SELECT 1"), "Summary. Contact jane@corp.com for details."], runner=FakeRunner([df]))
    r = run_turn(g, "Create a Q1 report for Client Y", ME, "t")
    assert "Saved as report #5" in r["answer"] and r["saved_report_id"] == 5
    saved = store.get(ME_KEY, 5)
    assert "jane@corp.com" not in saved.content and "[REDACTED_EMAIL]" in saved.content
    assert store.find(ME_KEY, mentions=["client y"])[0].id == 5          # findable by "mentioning Client Y"


def test_golden_examples_reach_the_sql_and_report_prompts_and_are_traced():
    pd = pytest.importorskip("pandas")
    from src.retrieval.golden import GoldenIndex
    golden = GoldenIndex.load(Path(__file__).resolve().parent.parent / "golden" / "trios", validate=lambda s: None)
    llm = FakeLLM(['{"intent": "analysis", "standalone_question": "why did churn spike last month"}',
                   sql("SELECT 1"), "Churn rose."])
    deps = Deps(llm=llm, runner=FakeRunner([pd.DataFrame({"churn_rate": [0.2]})]), prompts=PromptRegistry(PROMPTS),
                schema_text="s", validate=fake_validate, golden=golden)
    out = run_turn(build_graph(deps, MemorySaver()), "why did churn spike last month?", ME, "t")
    assert out["golden_ids"][0] == "monthly-churn-rate"
    assert "WITH monthly AS" in llm.prompts[1] and 'example id="monthly-churn-rate"' in llm.prompts[1]
    assert "Headline:" in llm.prompts[2] and "WITH monthly AS" not in llm.prompts[2]


# ============================================================ human-in-the-loop (graph level)
def _cfg(thread="t"):
    return {"configurable": {"thread_id": thread}}


def test_delete_confirmation_pauses_and_resumes():
    """The graph really stops at the confirmation node, deletes nothing while paused, and the user's next
    message resumes THE SAME run (no new routing, same trace) instead of starting a new turn."""
    g, store, ids, llm = setup([MANAGE, action(mentions=["Client X"])])

    # 1. the delete request pauses the graph at the confirmation node
    first = run_turn(g, "Delete all reports mentioning Client X", ME, "t")
    snap = g.get_state(_cfg())
    assert tuple(snap.next) == ("confirm_delete",)                         # suspended, not finished
    payload = snap.tasks[0].interrupts[0].value
    assert payload["type"] == "confirm_delete" and payload["count"] == 2 and payload["token"]
    assert first["status"] == "awaiting_confirmation" and "2 reports" in first["answer"]
    assert not snap.values.get("history")                                  # the turn has not been finalized yet
    assert live(store) == {ids["a"], ids["b"], ids["c"]}                   # NOTHING is deleted while paused
    assert store.audit_log(ME_KEY)[0]["action"] == "delete_planned"        # only the plan was recorded

    # 2. the next message resumes the paused run
    llm_calls_before = len(llm.prompts)
    second = run_turn(g, "yes", ME, "t")
    assert len(llm.prompts) == llm_calls_before                            # resuming costs no LLM call, no re-routing
    assert tuple(g.get_state(_cfg()).next) == ()                           # the run completed
    assert second["trace_id"] == first["trace_id"]                         # same turn, continued
    assert second["question"] == "Delete all reports mentioning Client X"
    assert second["status"] == "done" and "Deleted 2 reports" in second["answer"]
    assert live(store) == {ids["c"]}
    assert [x["action"] for x in store.audit_log(ME_KEY)][:2] == ["delete_confirmed", "delete_planned"]
    assert g.get_state(_cfg()).values["history"][-1]["content"] == second["answer"]   # finalized after the resume


@pytest.mark.parametrize("resume_value", [{"confirmed": False}, {"confirmed": "yes"}, {"confirmed": 1}, "yes", True])
def test_only_a_literal_true_resume_deletes(resume_value):
    """Defense in depth below the chat layer: even a direct resume with a truthy-looking value must not delete."""
    from langgraph.types import Command
    g, store, ids, _ = setup([MANAGE, action(mentions=["Client X"])])
    run_turn(g, "Delete all reports mentioning Client X", ME, "t")
    g.invoke(Command(resume=resume_value), _cfg())
    assert live(store) == {ids["a"], ids["b"], ids["c"]}
    assert store.audit_log(ME_KEY)[0]["action"] == "delete_cancelled"


def test_pending_confirmation_survives_a_restart(tmp_path):
    """The pause is persisted: a brand-new process (new graph, new checkpointer connection, same files) can resume it."""
    pytest.importorskip("langgraph.checkpoint.sqlite")
    from src.app import make_checkpointer

    store = ReportStore(tmp_path / "reports.sqlite")
    ids = [store.save(ME_KEY, "t", f"Client X report {i}", "q", "body") for i in range(2)]

    def build(script):
        deps = Deps(llm=FakeLLM(script), runner=FakeRunner([]), prompts=PromptRegistry(PROMPTS), schema_text="s",
                    validate=fake_validate, reports=ReportStore(tmp_path / "reports.sqlite"))
        return build_graph(deps, make_checkpointer(tmp_path / "checkpoints.sqlite")), deps

    g1, _ = build([MANAGE, action(mentions=["Client X"])])
    assert run_turn(g1, "Delete all reports mentioning Client X", ME, "t")["status"] == "awaiting_confirmation"
    del g1                                                                  # "the process exits"

    g2, deps2 = build([])                                                   # the new process needs no LLM call to resume
    assert pending_confirmation(g2, "t")["count"] == 2
    out = run_turn(g2, "yes", ME, "t")
    assert "Deleted 2 reports" in out["answer"] and deps2.llm.prompts == []
    assert {r.id for r in ReportStore(tmp_path / "reports.sqlite").list(ME_KEY)} == set()
    assert ids  # both reports existed before the restart
