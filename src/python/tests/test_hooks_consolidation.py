"""Test that prettify task has been fully consolidated into format."""

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read_pixi_toml() -> dict:
    with open(PROJECT_ROOT / "pixi.toml", "rb") as f:
        return tomllib.load(f)


def test_prettify_task_removed():
    """prettify should not exist as a pixi task — format replaces it."""
    config = read_pixi_toml()
    # pixi uses [tasks] section
    tasks = config.get("tasks", config.get("task", {}))
    assert "prettify" not in tasks, "prettify task must be removed; use format instead"


def test_black_dependency_removed():
    """black should not be a pixi dependency — ruff is the sole formatter."""
    config = read_pixi_toml()
    deps = config.get("dependencies", {})
    assert "black" not in deps, "black dependency must be removed; ruff is used instead"


def test_format_uses_ruff():
    """format task should use ruff format, not black."""
    config = read_pixi_toml()
    tasks = config.get("tasks", config.get("task", {}))
    format_cmd = tasks.get("format", "")
    assert "ruff format" in format_cmd, f"format task should use ruff format, got: {format_cmd}"


def test_format_check_uses_ruff():
    """format-check task should use ruff format --check, not black."""
    config = read_pixi_toml()
    tasks = config.get("tasks", config.get("task", {}))
    cmd = tasks.get("format-check", "")
    assert "ruff format --check" in cmd, f"format-check should use ruff, got: {cmd}"


def test_prettify_not_in_pre_commit_hook():
    """pre-commit hook must not reference prettify task."""
    hook = (PROJECT_ROOT / ".githooks" / "pre-commit").read_text()
    assert "prettify" not in hook, "pre-commit hook must not call prettify"


def test_prettify_not_in_pre_push_hook():
    """pre-push hook must not reference prettify task."""
    hook = (PROJECT_ROOT / ".githooks" / "pre-push").read_text()
    assert "prettify" not in hook, "pre-push hook must not call prettify"


def test_prettify_not_in_ci_workflow():
    """CI workflow must not reference prettify task."""
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "coverage-test.yml").read_text()
    assert "prettify" not in workflow, "CI workflow must not call prettify"
