import pytest

from evals.cases import AttackCase, GoldenCase, load_adversarial, load_golden
from evals.checks import check_no_pii, check_no_prompt_leak
from evals.compare import results_match
from evals.judge import judge_answer
from evals.runner import (CaseResult, evaluate_attack, evaluate_golden, gate, run_input_attacks, run_sql_attacks,
                          summarize)
from src.guards.input_guard import check as input_check
from src.sql.errors import SQLValidationError, ValidatedSQL

pd = pytest.importorskip("pandas")


# ------------------------------------------------------------------ execution comparison
def df(**cols):
    return pd.DataFrame(cols)


def test_match_ignores_row_order_column_order_and_aliases():
    ref = df(country=["US", "DE", "FR"], n=[10, 5, 1])
    got = df(cnt=[1, 10, 5], nation=["FR", "US", "DE"])
    assert results_match(ref, got)[0]


def test_numeric_tolerance_and_int_float_equivalence():
    assert results_match(df(x=[1234.5, 3]), df(y=[1234.5004, 3.0]))[0]
    assert not results_match(df(x=[1234.5]), df(y=[1234.6]))[0]


def test_mismatch_reasons():
    assert "row count" in results_match(df(x=[1, 2]), df(x=[1]))[1]
    assert "column count" in results_match(df(x=[1], y=[2]), df(x=[1]))[1]
    assert "values differ" in results_match(df(x=[1, 2]), df(x=[1, 3]))[1]


def test_extra_columns_allowed_only_when_requested():
    ref, got = df(cat=["a", "b"], rev=[1.0, 2.0]), df(cat=["a", "b"], rev=[1.0, 2.0], orders=[7, 9])
    assert not results_match(ref, got)[0] and results_match(ref, got, allow_extra_columns=True)[0]
    assert not results_match(ref, df(other=["x", "y"], rev=[1.0, 2.0], orders=[7, 9]), allow_extra_columns=True)[0]


def test_empty_results_and_nulls():
    assert results_match(df(x=[]), df(y=[]))[0]
    assert results_match(df(x=[None, 1.0]), df(x=[float("nan"), 1.0]))[0]


# ------------------------------------------------------------------------ safety checks
def test_pii_checks():
    assert check_no_pii({"answer": "Revenue grew 10%.", "table_md": "| category | revenue |\n|---|---|"}) == []
    assert "PII-like" in check_no_pii({"answer": "mail bob@example.com"})[0]
    assert "PII columns" in check_no_pii({"answer": "x", "table_md": "| user_id | Email |\n|---|---|"})[0]
    assert "executed query" in check_no_pii({"answer": "x", "executed_sql": "SELECT email FROM u"})[0]


def test_prompt_leak_detection(tmp_path):
    (tmp_path / "p.md").write_text("You are the Retail Insights Analyst, an assistant for non-technical executives and you must be concise.")
    leaked = {"answer": "Sure! My instructions say: you are the retail insights analyst, an assistant for non-technical executives and more"}
    assert check_no_prompt_leak(leaked, tmp_path) and not check_no_prompt_leak({"answer": "I analyze retail data."}, tmp_path)


# ----------------------------------------------------------------- deterministic suites
def test_shipped_attack_files_are_well_formed():
    adv = load_adversarial()
    assert len(adv["sql_attacks"]) >= 25 and len(adv["input_attacks"]) >= 10 and len(adv["e2e_attacks"]) >= 8
    for key in ("sql_attacks", "input_attacks"):
        ids = [c["id"] for c in adv[key]]
        assert len(ids) == len(set(ids))
    assert {c["expect"] for c in adv["sql_attacks"]} == {"reject", "accept_safe"}
    assert len(load_golden()) >= 8


def test_shipped_input_attacks_all_behave_as_expected():
    results = run_input_attacks(load_adversarial()["input_attacks"], input_check)
    assert [r.detail for r in results if not r.passed] == []


def test_shipped_sql_attacks_against_the_real_validator():
    pytest.importorskip("sqlglot")
    from src.sql.validator import validate_and_rewrite
    results = run_sql_attacks(load_adversarial()["sql_attacks"], validate_and_rewrite)
    assert [f"{r.id}: {r.detail}" for r in results if not r.passed] == []


def test_sql_attack_runner_flags_accepted_attacks_and_unfiltered_rewrites():
    def lax(sql, policy):                      # a broken validator that accepts everything
        return ValidatedSQL(sql=sql, tables=["orders"])
    cases = [{"id": "drop", "sql": "DROP TABLE x", "expect": "reject"},
             {"id": "safe", "sql": "SELECT 1 FROM orders", "expect": "accept_safe"}]
    res = {r.id: r for r in run_sql_attacks(cases, lax)}
    assert not res["drop:executive"].passed and res["drop:executive"].pii_leak
    assert res["safe:executive"].passed and not res["safe:customer"].passed       # customer rewrite lacks the filter

    def strict(sql, policy):
        raise SQLValidationError("forbidden_statement", "no")
    res = {r.id: r for r in run_sql_attacks(cases, strict)}
    assert res["drop:customer"].passed and "legitimate query refused" in res["safe:customer"].detail


# ----------------------------------------------------------------------- live suites (faked)
GOLD = GoldenCase("c1", "executive", "How many orders?", reference_sql="SELECT COUNT(*) FROM orders")


def good_state(**kw):
    return {"status": "done", "intent": "analysis", "answer": "There are 5 orders.", "executed_sql": "SELECT ...",
            "attempts": 1, "llm_calls": 3, **kw}


def test_golden_case_pass_and_fail_paths():
    ref = lambda c: df(n=[5])  # noqa: E731
    ok = evaluate_golden(GOLD, lambda c: good_state(), ref, lambda sql: df(total=[5]))
    assert ok.passed and ok.extra["scored_execution"]
    wrong = evaluate_golden(GOLD, lambda c: good_state(), ref, lambda sql: df(total=[6]))
    assert not wrong.passed and "wrong result" in wrong.detail
    failed = evaluate_golden(GOLD, lambda c: {"status": "failed", "answer": "sorry"}, ref, lambda s: df(n=[5]))
    assert not failed.passed and "agent status failed" in failed.detail
    noquery = evaluate_golden(GOLD, lambda c: good_state(executed_sql=None), ref, lambda s: df(n=[5]))
    assert "no query was executed" in noquery.detail


def test_golden_case_survives_exceptions_and_flags_pii():
    boom = evaluate_golden(GOLD, lambda c: (_ for _ in ()).throw(RuntimeError("x")), None, None)
    assert not boom.passed and boom.detail.startswith("error: RuntimeError")
    leaky = evaluate_golden(GOLD, lambda c: good_state(answer="mail a@b.com"), lambda c: df(n=[5]), lambda s: df(n=[5]))
    assert not leaky.passed and leaky.pii_leak


def test_report_case_needs_saved_report_and_runs_the_judge():
    case = GoldenCase("r", "executive", "Create a report", expect_saved_report=True, judge=True)
    judge = lambda c, st: {"mean": 4.5}  # noqa: E731
    res = evaluate_golden(case, lambda c: good_state(intent="report", saved_report_id=7), None, None, judge)
    assert res.passed and res.extra["judge"]["mean"] == 4.5
    assert "not saved" in evaluate_golden(case, lambda c: good_state(), None, None).detail


def test_attack_case_verdicts(tmp_path):
    (tmp_path / "p.md").write_text("secret system words that must never appear in any answer to the user ok")
    case = AttackCase("a", "executive", "q", {"status_in": ["blocked"]})
    assert evaluate_attack(case, lambda c: {"status": "blocked", "answer": "I can't help with that."}, tmp_path).passed
    r = evaluate_attack(case, lambda c: {"status": "done", "answer": "emails: bob@example.com"}, tmp_path)
    assert not r.passed and r.pii_leak
    assert "status" in evaluate_attack(case, lambda c: {"status": "done", "answer": "ok"}, tmp_path).detail
    assert not evaluate_attack(case, lambda c: {"status": "blocked", "answer": "secret system words that must never appear in any answer to the user ok"}, tmp_path).passed
    assert "crashed" in evaluate_attack(case, lambda c: 1 / 0, tmp_path).detail


# ------------------------------------------------------------------------ summary and gate
def test_summary_and_gate():
    results = [CaseResult("a", "sql_attacks", True), CaseResult("b", "sql_attacks", True),
               CaseResult("g1", "golden", True, extra={"scored_execution": True, "judge": {"mean": 4.0}}),
               CaseResult("g2", "golden", False, "wrong result: values differ", extra={"scored_execution": True}),
               CaseResult("g3", "golden", True, extra={"scored_execution": False}),
               CaseResult("s", "golden", False, "needs --customer-id", skipped=True)]
    s = summarize(results)
    assert s["execution_accuracy"] == 0.5 and s["judge_mean"] == 4.0 and s["skipped"] == ["s"]
    assert s["suites"]["golden"]["total"] == 3
    fails = gate(s)
    assert len(fails) == 1 and "execution accuracy" in fails[0]
    assert gate(s, min_exec_acc=0.4) == []


def test_gate_is_unforgiving_on_security_suites_and_pii():
    s = summarize([CaseResult("a", "sql_attacks", True), CaseResult("b", "sql_attacks", False, "x", pii_leak=True)])
    msgs = " | ".join(gate(s))
    assert "sql_attacks: 1/2" in msgs and "PII leak" in msgs


# ------------------------------------------------------------------------------- judge
class _LLM:
    def __init__(self, text):
        self.text = text

    def generate(self, prompt, **kw):
        self.prompt = prompt
        return type("R", (), {"text": self.text})()


def test_judge_parses_scores_and_rejects_garbage():
    from pathlib import Path

    from src.prompts import PromptRegistry
    prompts = PromptRegistry(Path(__file__).resolve().parent.parent / "prompts")
    ok = judge_answer(_LLM('{"relevance": 5, "grounding": 4, "actionability": null, "clarity": 3, "comment": "ok"}'), prompts,
                      question="q", intent="schema_qa", answer="a", data="x")      # reasoning is optional outside analyses
    assert ok["scores"] == {"relevance": 5.0, "grounding": 4.0, "clarity": 3.0} and ok["mean"] == 4.0
    # for an analysis or a report the judge must also score reasoning; a reply without it is rejected
    assert judge_answer(_LLM('{"relevance": 5, "grounding": 4, "clarity": 3}'), prompts,
                        question="q", intent="analysis", answer="a", data="x") is None
    full = judge_answer(_LLM('{"relevance": 5, "reasoning": 4, "grounding": 4, "clarity": 3}'), prompts,
                        question="q", intent="analysis", answer="a", data="x")
    assert full["scores"]["reasoning"] == 4.0
    for bad in ('{"relevance": 9, "grounding": 4, "clarity": 3}', "not json", '{"relevance": "high"}'):
        assert judge_answer(_LLM(bad), prompts, question="q", intent="x", answer="a", data="") is None
    llm = _LLM('{"relevance": 5, "grounding": 5, "clarity": 5}')
    judge_answer(llm, prompts, question="q", intent="report", answer="ANSWER-TEXT", data="DATA-TEXT")
    assert "ANSWER-TEXT" in llm.prompt and "untrusted" in llm.prompt
