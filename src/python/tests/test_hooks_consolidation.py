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


def test_fast_tests_job_excludes_slow():
    """The 'Fast Tests (not slow)' job must use test-cov, not plain `test`.

    Regression: test-pr.yml ran `pixi run test`, which does not exclude slow
    tests (pytest.ini addopts only adds `-m "not config"`). The job name claims
    to skip slow tests but actually ran them, diverging from pre-push/pre-commit
    hooks and the coverage job, which all use `pixi run test-cov`.
    """
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "test-pr.yml").read_text()
    fast_block = workflow.split("name: Fast Tests (not slow)")[1]
    fast_step = fast_block.split("run:")[1].splitlines()[0].strip()
    assert "pixi run test-cov" in fast_step, f"Fast Tests job must run test-cov, got: {fast_step}"


def test_correlation_validation_tests_marked_slow():
    """Correlation validation tests must be marked slow.

    Regression: test_correlation_validation.py runs 20 full-simulation tests
    (~23 min) but had no `slow` marker, so it executed in every fast suite run
    (CI fast jobs, pre-push, pre-commit), inflating runtime to ~40 min.
    """
    test_file = PROJECT_ROOT / "src" / "python" / "tests" / "test_correlation_validation.py"
    src = test_file.read_text()
    assert "pytestmark" in src, "module must declare pytestmark"
    assert "mark.slow" in src, "pytestmark must include the slow marker"


def test_pixi_test_task_excludes_slow():
    """The `test` task must exclude slow tests.

    Regression: `pixi run test` was plain `pytest src/python/tests`, which only
    inherited `-m "not config"` from pytest.ini addopts — so slow tests (incl.
    the 23-min correlation file) still ran, making the fast suite ~40 min.
    """
    config = read_pixi_toml()
    tasks = config.get("tasks", config.get("task", {}))
    test_cmd = tasks.get("test", "")
    assert "not slow" in test_cmd, f"test task must exclude slow tests, got: {test_cmd}"
