"""Cheap deterministic pre-filter for obvious prompt-injection attempts (no LLM call, no cost).
The LLM router is the semantic second layer; the SQL validator and output scrubber are the
hard layers, so this only has to catch the blatant cases."""
import re
from typing import Optional

_PATTERNS = {
    "override_instructions": r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|any|your)\b.{0,30}\b(instructions?|rules?|prompts?|guidelines?)\b",
    "reveal_prompt": r"\b(reveal|show|print|display|repeat|leak|output)\b.{0,30}\b(system|developer|hidden|initial)\b.{0,15}\b(prompt|message|instructions?)\b",
    "role_hijack": r"\b(you are now|from now on you|pretend (to be|you are)|act as (if|an? unrestricted))\b",
    "jailbreak": r"\b(jailbreak|dan mode|developer mode|do anything now)\b",
    # "executive" is a valid role, not an escalation; "switch to root cause" is not an escalation.
    # "switch me to" is an escalation; "switch to" alone is navigation.
    "role_escalation": r"\b(i am|i'm|treat me as|grant me|switch me to)\b.{0,20}\b(admin|administrator|root|superuser)\b",
    # "pii" alone is not a safety-disabling keyword (skipping PII columns is a valid request);
    # "filter" and "security" are too generic.
    "disable_safety": r"\b(disable|bypass|turn off|skip)\b.{0,25}\b(safety|guard|restriction|masking)\b",
}
_COMPILED = {k: re.compile(v, re.I | re.S) for k, v in _PATTERNS.items()}


def check(text: Optional[str]) -> Optional[str]:
    """Return the name of the matched pattern, or None if the text looks fine."""
    if not text:
        return None
    for name, pat in _COMPILED.items():
        if pat.search(text):
            return name
    return None