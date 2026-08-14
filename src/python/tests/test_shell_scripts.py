"""
Tests for shell scripts that automate config extraction and env file management.

These tests invoke the scripts via subprocess and verify their output.
"""

import subprocess
from pathlib import Path

SHELL_DIR = Path("src/shell")
PROJECT_ROOT = Path(".")


def _run_awk(file: Path, *input_files: str) -> str:
    """Run extract_env.awk against given input files and return stdout."""
    result = subprocess.run(
        ["awk", "-f", str(file)] + list(input_files),
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT.resolve()),
    )
    if result.returncode != 0:
        raise RuntimeError(f"awk failed: {result.stderr}")
    return result.stdout


# ── extract_env.awk ───────────────────────────────────────────────


class TestExtractEnvAwk:
    """Test the extract_env.awk script against sample config snippets."""

    def test_extract_get_env_value(self, tmp_path):
        """Extract KEY=DEFAULT from _get_env_value() call."""
        snippet = 'self.pss10_scale = self._get_env_value("PSS10_SCALE", float, 3.5)\n'
        f = tmp_path / "config_sample.py"
        f.write_text(snippet)

        output = _run_awk(SHELL_DIR / "extract_env.awk", str(f))
        assert "PSS10_SCALE=3.5" in output

    def test_extract_get_env_value_string_default(self, tmp_path):
        """Extract KEY from _get_env_value() with a string default."""
        snippet = 'self.log_level = self._get_env_value("LOG_LEVEL", str, "INFO")\n'
        f = tmp_path / "config_sample.py"
        f.write_text(snippet)

        output = _run_awk(SHELL_DIR / "extract_env.awk", str(f))
        assert "LOG_LEVEL=INFO" in output

    def test_extract_get_env_array(self, tmp_path):
        """Extract KEY=[...] from _get_env_array() call."""
        snippet = (
            "self.pss10_item_means = self._get_env_array(\n"
            '    "PSS10_ITEM_MEAN", float, [2.1, 1.8, 2.3], expected_length=3\n'
            ")\n"
        )
        f = tmp_path / "config_sample.py"
        f.write_text(snippet)

        output = _run_awk(SHELL_DIR / "extract_env.awk", str(f))
        assert "PSS10_ITEM_MEAN=[2.1, 1.8, 2.3]" in output

    def test_extract_env_float(self, tmp_path):
        """Extract KEY=default from _env_float() call."""
        snippet = (
            'resource_reward: float = field(default_factory=lambda: _env_float("ASSUMPTION_RESOURCE_REWARD", 0.10))\n'
        )
        f = tmp_path / "assumption_sample.py"
        f.write_text(snippet)

        output = _run_awk(SHELL_DIR / "extract_env.awk", str(f))
        assert "ASSUMPTION_RESOURCE_REWARD=0.10" in output

    def test_extract_env_float_int_default(self, tmp_path):
        """Extract KEY=int from _env_float() with int default."""
        snippet = (
            "pss10_estimation_base: int = field(\n"
            '    default_factory=lambda: int(_env_float("ASSUMPTION_PSS10_ESTIMATION_BASE", 10.0))\n'
            ")\n"
        )
        f = tmp_path / "assumption_sample.py"
        f.write_text(snippet)

        output = _run_awk(SHELL_DIR / "extract_env.awk", str(f))
        assert "ASSUMPTION_PSS10_ESTIMATION_BASE=10.0" in output

    def test_extract_multiple_files(self, tmp_path):
        """Extract from two files, producing combined output."""
        f1 = tmp_path / "a.py"
        f1.write_text('self.x = self._get_env_value("ALPHA", float, 0.5)\n')
        f2 = tmp_path / "b.py"
        f2.write_text('_env_float("BETA", 1.5)\n')

        output = _run_awk(SHELL_DIR / "extract_env.awk", str(f1), str(f2))
        assert "ALPHA=0.5" in output
        assert "BETA=1.5" in output

    def test_ignore_non_env_lines(self, tmp_path):
        """Lines without _get_env_value/_get_env_array/_env_float produce nothing."""
        snippet = "self._load_all_parameters()\nx = 42\n# comment\nself.foo = bar\n"
        f = tmp_path / "noise.py"
        f.write_text(snippet)
        output = _run_awk(SHELL_DIR / "extract_env.awk", str(f))
        assert output.strip() == ""

    def test_skip_function_definition_signature(self, tmp_path):
        """The `def _env_float(...)` signature must not be emitted as KEY=default.

        Regression: extract_env.awk matched `_env_float(` inside the function
        definition `def _env_float(key: str, default: float) -> float:` and
        emitted the junk line `key: str=default: float` into .env.example.
        """
        snippet = "def _env_float(key: str, default: float) -> float:\n    return 0.0\n"
        f = tmp_path / "assumption_def.py"
        f.write_text(snippet)

        output = _run_awk(SHELL_DIR / "extract_env.awk", str(f))
        assert "key: str" not in output
        assert output.strip() == ""


# ── parameterize_env.sh ────────────────────────────────────────────


class TestParameterizeEnv:
    """Test the parameterize_env.sh pipeline script."""

    def _run_param(self, tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
        """Run parameterize_env.sh in tmp_path with any extra args.
        The config files in tmp_path are passed as positional args after --force if present.
        """
        script = str((PROJECT_ROOT / SHELL_DIR / "parameterize_env.sh").resolve())
        cfg1 = str(tmp_path / "config.py")
        cfg2 = str(tmp_path / "assumption_config.py")
        cmd = ["bash", script] + list(args) + [cfg1, cfg2]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        return result

    def test_updates_env_example_with_new_defaults(self, tmp_path):
        """parameterize_env.sh updates .env.example with extracted defaults."""
        # Create minimal config stubs with known env vars
        config_py = tmp_path / "config.py"
        config_py.write_text('self.x = self._get_env_value("MY_VAR", float, 1.5)\n')
        assumption_py = tmp_path / "assumption_config.py"
        assumption_py.write_text('_env_float("ASSUMPTION_X", 2.5)\n')

        # Create .env.example with an outdated value and a comment
        env_example = tmp_path / ".env.example"
        env_example.write_text("# This is a comment\nMY_VAR=0.5\n")

        # Create .env with same outdated value
        env_file = tmp_path / ".env"
        env_file.write_text("MY_VAR=0.5\n")

        result = self._run_param(tmp_path)
        assert result.returncode == 0, f"script failed: {result.stderr}"

        # .env.example should have MY_VAR updated to 1.5, comment preserved
        updated_example = env_example.read_text()
        assert "# This is a comment" in updated_example
        assert "MY_VAR=1.5" in updated_example

    def test_env_preserves_existing_overrides(self, tmp_path):
        """Normal mode: .env preserves existing values, adds new keys."""
        config_py = tmp_path / "config.py"
        config_py.write_text(
            'self.x = self._get_env_value("MY_VAR", float, 1.5)\nself.y = self._get_env_value("NEW_VAR", int, 99)\n'
        )
        assumption_py = tmp_path / "assumption_config.py"
        assumption_py.write_text("")

        env_file = tmp_path / ".env"
        env_file.write_text("MY_VAR=0.5\n")
        env_example = tmp_path / ".env.example"
        env_example.write_text("MY_VAR=0.5\n")

        result = self._run_param(tmp_path)
        assert result.returncode == 0, f"script failed: {result.stderr}"

        # MY_VAR should keep its override (0.5), NEW_VAR should be added with default (99)
        env_content = env_file.read_text()
        assert "MY_VAR=0.5" in env_content
        assert "NEW_VAR=99" in env_content

    def test_force_overwrites_env_values(self, tmp_path):
        """--force overwrites .env values with extracted defaults."""
        config_py = tmp_path / "config.py"
        config_py.write_text('self.x = self._get_env_value("MY_VAR", float, 1.5)\n')
        assumption_py = tmp_path / "assumption_config.py"
        assumption_py.write_text("")

        env_file = tmp_path / ".env"
        env_file.write_text("MY_VAR=0.5\n")
        env_example = tmp_path / ".env.example"
        env_example.write_text("MY_VAR=0.5\n")

        result = self._run_param(tmp_path, "--force")
        assert result.returncode == 0, f"script failed: {result.stderr}"

        env_content = env_file.read_text()
        assert "MY_VAR=1.5" in env_content

    def test_handles_nonexistent_env(self, tmp_path):
        """Creates .env if it doesn't exist."""
        config_py = tmp_path / "config.py"
        config_py.write_text('self.x = self._get_env_value("MY_VAR", float, 1.5)\n')
        assumption_py = tmp_path / "assumption_config.py"
        assumption_py.write_text("")

        env_example = tmp_path / ".env.example"
        env_example.write_text("MY_VAR=0.5\n")
        # No .env file

        result = self._run_param(tmp_path)
        assert result.returncode == 0, f"script failed: {result.stderr}"

        env_file = tmp_path / ".env"
        assert env_file.exists()
        assert "MY_VAR=1.5" in env_file.read_text()
