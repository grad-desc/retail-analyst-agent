"""Lenient JSON-object parsing for LLM replies (strips code fences, tolerates surrounding prose)."""
import json
import re
from typing import Optional


def parse_json(text: str) -> Optional[dict]:
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    candidates = [text]
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None
