"""Test that calibration results are persisted to config, env, and test files."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.python.calibration.persistence import (
    persist_calibration_results,
    _update_env_file,
    _update_config_default,
    _update_test_expected_means,
)


class TestPersistCalibrationResults:
    """Verify calibration persistence writes to all required files."""

    def test_update_env_file_preserves_other_vars(self):
        """_update_env_file replaces PSS10_ITEM_MEAN line, leaves others intact."""
        content = "A=1\nPSS10_ITEM_MEAN=[2.1, 1.8]\nB=2\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        try:
            new_means = [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]
            _update_env_file(tmp_path, new_means)

            result = Path(tmp_path).read_text()
            assert "A=1" in result
            assert "B=2" in result
            assert "PSS10_ITEM_MEAN=[3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]" in result
            assert "[2.1, 1.8]" not in result
        finally:
            os.unlink(tmp_path)

    def test_update_env_file_adds_if_missing(self):
        """_update_env_file appends PSS10_ITEM_MEAN if not present."""
        content = "A=1\nB=2\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        try:
            new_means = [1.0] * 10
            _update_env_file(tmp_path, new_means)

            result = Path(tmp_path).read_text()
            assert "PSS10_ITEM_MEAN=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]" in result
        finally:
            os.unlink(tmp_path)

    def test_update_config_default_replaces_array(self):
        """_update_config_default replaces the PSS10_ITEM_MEAN default array in config.py."""
        content = '''
        self.pss10_item_means = self._get_env_array(
            "PSS10_ITEM_MEAN", float, [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5], expected_length=10
        )
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        try:
            new_means = [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]
            _update_config_default(tmp_path, new_means)

            result = Path(tmp_path).read_text()
            assert "[3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]" in result
            assert "[2.1, 1.8, 2.3" not in result
        finally:
            os.unlink(tmp_path)

    def test_update_test_expected_means_replaces_all_occurrences(self):
        """_update_test_expected_means replaces all expected_means lists in test file."""
        content = '''expected_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
        assert config.pss10_item_means == expected_means, f"Expected {expected_means}, got {config.pss10_item_means}"
        expected_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        try:
            new_means = [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]
            old_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
            _update_test_expected_means(tmp_path, old_means, new_means)

            result = Path(tmp_path).read_text()
            assert result.count("[3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]") == 2
            assert "[2.1, 1.8" not in result
        finally:
            os.unlink(tmp_path)

    def test_persist_calibration_results_writes_all_files(self):
        """persist_calibration_results writes to env, config, and test files."""
        new_means = [2.5, 2.0, 2.5, 2.0, 2.5, 2.0, 2.5, 2.0, 2.5, 2.0]

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create dummy files
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text("A=1\nPSS10_ITEM_MEAN=[2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]\nB=2\n")

            config_path = Path(tmp_dir) / "config.py"
            config_path.write_text(
                'self.pss10_item_means = self._get_env_array(\n'
                '    "PSS10_ITEM_MEAN", float, [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5], expected_length=10\n'
                ')\n'
            )

            test_path = Path(tmp_dir) / "test_pss10_comprehensive.py"
            test_path.write_text(
                "expected_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]\n"
            )

            old_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]

            # Mock the project root paths
            with (
                patch("src.python.calibration.persistence.PROJECT_ROOT", Path(tmp_dir)),
                patch("src.python.calibration.persistence._ENV_PATH", env_path),
                patch("src.python.calibration.persistence._ENV_EXAMPLE_PATH", env_path),
                patch("src.python.calibration.persistence._CONFIG_PATH", config_path),
                patch("src.python.calibration.persistence._TEST_COMPREHENSIVE_PATH", test_path),
            ):
                persist_calibration_results(new_means, old_means)

            # Verify all files updated
            env_content = env_path.read_text()
            assert "2.5, 2.0, 2.5" in env_content
            assert "A=1" in env_content  # other lines preserved

            config_content = config_path.read_text()
            assert "2.5, 2.0, 2.5" in config_content

            test_content = test_path.read_text()
            assert "2.5, 2.0, 2.5" in test_content
