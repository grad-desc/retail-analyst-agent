"""Deletion confirmation is strict string matching: explicit consent only, never a guess."""
import pytest

from src.reports.delete_flow import parse_confirmation


@pytest.mark.parametrize("text", ["yes", "Yes!", "yes please", "Yes, please delete them.", "Yes, go ahead!",
                                  "please delete", "confirm delete"])
def test_explicit_consent_confirms(text):
    assert parse_confirmation(text) is True


@pytest.mark.parametrize("text", ["no", "No, don't delete", "cancel", "never mind"])
def test_explicit_refusal_cancels(text):
    assert parse_confirmation(text) is False


@pytest.mark.parametrize("text", ["ok", "sure", "maybe", "yes but only the first one", "what data do we have?", ""])
def test_vague_or_conditional_replies_never_confirm(text):
    assert parse_confirmation(text) is None
