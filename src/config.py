"""Settings loaded from environment / .env. Fails fast with a readable message."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

try:  # optional: .env support
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass


class ConfigError(RuntimeError):
    pass


def _csv(value: Optional[str]) -> Tuple[str, ...]:
    return tuple(x.strip() for x in (value or "").split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    fallback_models: Tuple[str, ...]
    gcp_project_id: Optional[str]
    bq_location: str = "US"
    bq_max_bytes: int = 1 * 1024**3
    llm_timeout_s: int = 60
    max_llm_calls_per_turn: int = 8
    llm_max_attempts: int = 2           # tries per model before moving to the next one in the chain
    judge_model: Optional[str] = None   # eval judge; set a different model than the generator when possible

    @property
    def model_chain(self) -> Tuple[str, ...]:
        """Primary first, then fallbacks, de-duplicated, order preserved."""
        return tuple(dict.fromkeys((self.gemini_model, *self.fallback_models)))

    @classmethod
    def from_env(cls) -> "Settings":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        model = os.getenv("GEMINI_MODEL_NAME", "").strip()
        missing = [n for n, v in (("GEMINI_API_KEY", key), ("GEMINI_MODEL_NAME", model)) if not v]
        if missing:
            raise ConfigError(f"Missing in .env: {', '.join(missing)} (see .env.example)")
        return cls(
            gemini_api_key=key,
            gemini_model=model,
            fallback_models=_csv(os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.1-flash-lite")),
            gcp_project_id=os.getenv("GCP_PROJECT_ID") or None,
            bq_location=os.getenv("BQ_LOCATION", "US"),
            bq_max_bytes=int(os.getenv("BQ_MAX_BYTES", 1 * 1024**3)),
            llm_timeout_s=int(os.getenv("LLM_TIMEOUT_S", 60)),
            max_llm_calls_per_turn=int(os.getenv("MAX_LLM_CALLS_PER_TURN", 8)),
            llm_max_attempts=int(os.getenv("LLM_MAX_ATTEMPTS", 2)),
            judge_model=os.getenv("GEMINI_JUDGE_MODEL") or None,
        )
