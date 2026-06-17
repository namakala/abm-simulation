"""Persistence helpers for writing calibrated PSS-10 item means to files."""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Paths relative to project root — resolved at call time so tests can patch.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ENV_PATH = PROJECT_ROOT / ".env"
_ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"
_CONFIG_PATH = PROJECT_ROOT / "src" / "python" / "config.py"
_TEST_COMPREHENSIVE_PATH = PROJECT_ROOT / "src" / "python" / "tests" / "test_pss10_comprehensive.py"


def persist_calibration_results(
    item_means: List[float],
    old_item_means: Optional[List[float]] = None,
) -> None:
    """Write calibrated PSS-10 item means to config, env, and test files.

    Updates:
    - ``.env`` and ``.env.example`` — PSS10_ITEM_MEAN line
    - ``src/python/config.py`` — default array in ``_get_env_array`` call
    - ``src/python/tests/test_pss10_comprehensive.py`` — all ``expected_means``

    Args:
        item_means: New calibrated item means (length 10).
        old_item_means: Previous values to replace in test file.
            Defaults to the original config default.
    """
    if old_item_means is None:
        old_item_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]

    _update_env_file(_ENV_PATH, item_means)
    _update_env_file(_ENV_EXAMPLE_PATH, item_means)
    _update_config_default(_CONFIG_PATH, item_means)
    _update_test_expected_means(_TEST_COMPREHENSIVE_PATH, old_item_means, item_means)

    logger.info(f"Persisted calibrated item means: {item_means}")


def _update_env_file(path: Path, item_means: List[float]) -> None:
    """Update or add PSS10_ITEM_MEAN line in an env file."""
    path = Path(path)
    means_str = "[" + ", ".join(str(v) for v in item_means) + "]"
    new_line = f"PSS10_ITEM_MEAN={means_str}"

    if path.exists():
        lines = path.read_text().splitlines()
        replaced = False
        for i, line in enumerate(lines):
            if line.startswith("PSS10_ITEM_MEAN="):
                lines[i] = new_line
                replaced = True
                break
        if not replaced:
            lines.append(new_line)
        path.write_text("\n".join(lines) + "\n")
    else:
        path.write_text(new_line + "\n")


def _update_config_default(path: Path, item_means: List[float]) -> None:
    """Replace the PSS10_ITEM_MEAN default array in config.py."""
    import re

    path = Path(path)
    content = path.read_text()
    means_str = "[" + ", ".join(str(v) for v in item_means) + "]"
    pattern = r'("PSS10_ITEM_MEAN",\s*float,\s*)\[[^\]]+\]'
    replacement = rf"\g<1>{means_str}"
    new_content = re.sub(pattern, replacement, content)
    path.write_text(new_content)


def _update_test_expected_means(path: Path, old_means: List[float], new_means: List[float]) -> None:
    """Replace all expected_means lists in the PSS-10 comprehensive test file."""
    path = Path(path)
    content = path.read_text()
    old_str = "[" + ", ".join(str(v) for v in old_means) + "]"
    new_str = "[" + ", ".join(str(v) for v in new_means) + "]"
    new_content = content.replace(old_str, new_str)
    path.write_text(new_content)


def run_verification_tests() -> bool:
    """Run the tests that depend on calibrated PSS-10 item means.

    Returns True if all pass.
    """
    import subprocess
    import sys

    test_files = [
        "src/python/tests/test_pss10_comprehensive.py",
        "src/python/tests/test_integrated_population_theory.py::TestPSS10Distribution",
    ]
    cmd = [sys.executable, "-m", "pytest", "--no-header", "--no-cov", "-q"] + test_files
    result = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(f"Verification test output:\n{result.stdout}")
    if result.returncode != 0:
        logger.warning(f"Verification tests failed:\n{result.stderr}")
    return result.returncode == 0
