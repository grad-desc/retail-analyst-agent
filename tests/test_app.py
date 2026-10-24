import json
import logging

import pytest

from src.app import principal_from_flags
from src.observability.logging_setup import setup_logging


def test_principal_customer_and_executive():
    assert principal_from_flags(123, "customer") == {"role": "customer", "user_id": 123, "subject": "customer:123"}
    assert principal_from_flags(None, "executive", "dana") == {"role": "executive", "user_id": None, "subject": "executive:dana"}
    assert principal_from_flags(5, "CUSTOMER")["user_id"] == 5


@pytest.mark.parametrize("uid,role", [(None, "customer"), (1, "executive"), (-3, "customer"), (1, "root")])
def test_principal_rejects_invalid_combinations(uid, role):
    with pytest.raises(ValueError):
        principal_from_flags(uid, role)


def test_json_logs_carry_trace_id_and_never_contain_pii(tmp_path):
    f = setup_logging(tmp_path)
    log = logging.getLogger("t")
    log.info("turn_done", extra={"trace_id": "abc123", "question": "mail bob@example.com", "rows": 3})
    log.info("plain message for %s", "carol@example.com")
    for h in logging.getLogger().handlers:
        h.flush()
    lines = [json.loads(x) for x in f.read_text().splitlines()]
    assert lines[0]["trace_id"] == "abc123" and lines[0]["rows"] == 3 and lines[0]["event"] == "turn_done"
    assert "bob@example.com" not in f.read_text() and "carol@example.com" not in f.read_text()
    assert "[REDACTED_EMAIL]" in lines[0]["question"]
