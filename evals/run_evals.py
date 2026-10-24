"""Pre-deployment evaluation gate.

  python -m evals.run_evals --suite static                      # no keys needed: validator + input guard (run in CI on every PR)
  python -m evals.run_evals --suite all --customer-id 123 --judge   # full run against live Gemini + BigQuery (pre-release / nightly)

Exit code 1 when the gate fails, so a pipeline can block the deployment.
Suites: static (sql + input attacks) | golden (execution accuracy + report quality) | attacks (live attacks) | all
"""
import argparse
import json
import sys
import time
import uuid

from evals.cases import HERE, load_adversarial, load_golden
from evals.runner import (CaseResult, evaluate_attack, evaluate_golden, gate, run_input_attacks, run_sql_attacks,
                          summarize)

ROOT = HERE.parent


def _live_context(customer_id, judge_enabled):
    from src.app import bootstrap, principal_from_flags
    from src.config import Settings
    from src.graph import run_turn
    from src.llm.client import LLMClient
    from src.observability.logging_setup import setup_logging
    from src.prompts import PromptRegistry
    from src.sql.policy import UserPolicy
    from src.sql.validator import validate_and_rewrite
    setup_logging(ROOT / "logs")
    settings = Settings.from_env()
    app = bootstrap(settings)

    def principal(role):
        return principal_from_flags(customer_id if role == "customer" else None, role, "eval")

    def agent(case):
        return run_turn(app.graph, case.question, principal(case.role), f"eval-{case.id}-{uuid.uuid4().hex[:6]}")

    def policy(role):
        return UserPolicy.customer(customer_id) if role == "customer" else UserPolicy.executive()

    def reference(case):
        return app.runner.execute_query(validate_and_rewrite(case.reference_sql, policy(case.role)).sql)

    judge = None
    if judge_enabled:
        from evals.judge import judge_answer
        prompts = PromptRegistry(ROOT / "prompts")
        base_llm = LLMClient.from_settings(settings)
        if settings.judge_model:
            judge_llm = base_llm.with_model(settings.judge_model)
        else:
            judge_llm = base_llm
        judge = lambda case, st: judge_answer(judge_llm, prompts, question=case.question, intent=st.get("intent"),  # noqa: E731
                                              answer=st.get("answer", ""), data=st.get("data_csv", ""))
    return agent, reference, app.runner.execute_query, judge


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["static", "golden", "attacks", "all"], default="static")
    ap.add_argument("--customer-id", type=int, help="A real users.id with orders; enables the customer (RLS) cases.")
    ap.add_argument("--judge", action="store_true", help="Score reports with the LLM judge (extra LLM calls).")
    ap.add_argument("--min-exec-acc", type=float, default=0.8)
    ap.add_argument("--min-judge", type=float, default=3.5)
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds between live cases (free-tier rate limits).")
    args = ap.parse_args(argv)

    adv, results = load_adversarial(), []
    if args.suite in ("static", "all"):
        from src.guards.input_guard import check
        from src.sql.validator import validate_and_rewrite
        results += run_sql_attacks(adv["sql_attacks"], validate_and_rewrite)
        results += run_input_attacks(adv["input_attacks"], check)
    if args.suite in ("golden", "attacks", "all"):
        agent, reference, execute, judge = _live_context(args.customer_id, args.judge)
        if args.suite in ("golden", "all"):
            for case in load_golden():
                if case.role == "customer" and args.customer_id is None:
                    results.append(CaseResult(case.id, "golden", False, "needs --customer-id", skipped=True))
                    continue
                results.append(evaluate_golden(case, agent, reference, execute, judge))
                time.sleep(args.sleep)
        if args.suite in ("attacks", "all"):
            for case in adv["e2e_attacks"]:
                results.append(evaluate_attack(case, agent, ROOT / "prompts"))
                time.sleep(args.sleep)

    summary = summarize(results)
    failures = gate(summary, args.min_exec_acc, args.min_judge)
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "gate_failures": failures,
                               "results": [r.__dict__ for r in results]}, indent=2, default=str), encoding="utf-8")

    print("\nSuite            passed/total  rate")
    for name, s in summary["suites"].items():
        print(f"{name:<16} {s['passed']:>3}/{s['total']:<8} {s['pass_rate']:.2f}")
    print(f"\nexecution accuracy: {summary['execution_accuracy']}   judge mean: {summary['judge_mean']}   PII leaks: {summary['pii_leaks']}")
    if summary["skipped"]:
        print("skipped:", ", ".join(summary["skipped"]))
    for f in summary["failed"]:
        print("  FAIL", f)
    print(f"\nresults: {out}")
    print("GATE: PASSED" if not failures else "GATE: FAILED\n  - " + "\n  - ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
