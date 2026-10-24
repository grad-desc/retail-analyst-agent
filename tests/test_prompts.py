from pathlib import Path

from src.prompts import PromptRegistry

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def test_user_text_cannot_inject_template_variables(tmp_path):
    (tmp_path / "persona.md").write_text("PERSONA")
    (tmp_path / "t.md").write_text("{{persona}} | {{question}} | {{secret}}")
    out = PromptRegistry(tmp_path).render("t", question="{{persona}} {{secret}}", secret="S3CRET")
    assert out == "PERSONA | {{persona}} {{secret}} | S3CRET"   # single pass: values are never re-scanned


def test_hot_reload_and_version_change(tmp_path):
    (tmp_path / "persona.md").write_text("formal")
    reg = PromptRegistry(tmp_path)
    v1, t1 = reg.version("persona"), reg.load("persona")
    p = tmp_path / "persona.md"
    p.write_text("playful tone")
    import os
    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 5_000_000))
    assert reg.load("persona") == "playful tone" and reg.version("persona") != v1 and t1 == "formal"


def test_all_shipped_prompts_render_without_leftover_variables():
    reg = PromptRegistry(PROMPTS)
    values = dict(history="h", question="q", schema="s", scope="sc", examples="e", previous_sql="p",
                  feedback="f", preferences="pr", format_hint="fh", row_count=1, truncated_note="",
                  data="d", sql="sql")
    for name in ("router", "schema_qa", "sql_gen", "report", "reports_action"):
        assert "{{" not in reg.render(name, **values), name


def test_today_is_always_available_to_prompts(tmp_path):
    (tmp_path / "persona.md").write_text("P")
    (tmp_path / "t.md").write_text("today={{today}}")
    out = PromptRegistry(tmp_path).render("t")
    assert out.startswith("today=20") and "{{" not in out
    assert PromptRegistry(tmp_path).render("t", today="2026-10-23") == "today=2026-10-23"
