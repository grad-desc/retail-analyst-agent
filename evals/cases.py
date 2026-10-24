from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class GoldenCase:
    id: str
    role: str
    question: str
    reference_sql: Optional[str] = None
    allow_extra_columns: bool = False
    expect_saved_report: bool = False
    judge: bool = False


@dataclass(frozen=True)
class AttackCase:
    id: str
    role: str
    question: str
    checks: Dict[str, Any]


def load_golden(path=HERE / "golden_set.yaml") -> List[GoldenCase]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [GoldenCase(**c) for c in data["cases"]]


def load_adversarial(path=HERE / "adversarial.yaml") -> Dict[str, list]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data["e2e_attacks"] = [AttackCase(**c) for c in data.get("e2e_attacks", [])]
    return data
