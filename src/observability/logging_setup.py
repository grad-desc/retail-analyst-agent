"""Structured JSON logs (one event per line) with PII redaction.

Every record carries `trace_id`, so one grep reconstructs a whole turn: route decision, each SQL
attempt and its validator/BigQuery feedback, LLM calls (model, latency, tokens), guard outcomes,
and the final question/answer pair.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.guards.output_scrubber import _STD_ATTRS, RedactingFilter


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        out.update({k: v for k, v in record.__dict__.items() if k not in _STD_ATTRS and not k.startswith("_")})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)  # stack traces go to the log, never to the user
        return json.dumps(out, default=str, ensure_ascii=False)


def setup_logging(log_dir, level: int = logging.INFO) -> Path:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    file = path / "agent.jsonl"
    handler = logging.FileHandler(file, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactingFilter())
    root = logging.getLogger()
    for h in list(root.handlers):          # keep the CLI free of log noise
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)
    for noisy in ("google", "urllib3", "httpx", "grpc"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return file
