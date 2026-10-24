"""LLM-as-judge for report quality. Rubric dimensions are scored 1-5; safety is NOT judged by an LLM
(see checks.py: deterministic). Prefer a judge model different from the generator (GEMINI_JUDGE_MODEL)."""
from typing import Any, Dict, Optional

from src.jsonutil import parse_json

DIMENSIONS = ("relevance", "reasoning", "grounding", "actionability", "clarity")


def judge_answer(llm, prompts, *, question: str, intent: str, answer: str, data: str) -> Optional[Dict[str, Any]]:
    """Returns {"scores": {...}, "mean": float, "comment": str} or None if the judge failed or replied badly."""
    prompt = prompts.render("judge", question=question, intent=intent or "analysis", answer=answer, data=data[:6000] or "(no rows)")
    try:
        parsed = parse_json(llm.generate(prompt, json_mode=True, temperature=0.0).text)
    except Exception:  # noqa: BLE001 - a flaky judge must not crash the eval run
        return None
    if not parsed:
        return None
    scores = {}
    for d in DIMENSIONS:
        v = parsed.get(d)
        if v is None and d == "actionability":
            continue  # actionability is optional
        if v is None and d == "reasoning":
            # reasoning is optional for non-report (schema_qa, preferences)
            if intent != "report" and intent != "analysis":
                continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 1 <= v <= 5:
            return None
        scores[d] = float(v)
    if not scores:
        return None
    return {"scores": scores, "mean": round(sum(scores.values()) / len(scores), 2), "comment": str(parsed.get("comment", ""))[:300]}
