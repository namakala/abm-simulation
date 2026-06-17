"""Test that the calibration entry point resolves its imports correctly."""

import subprocess
import sys
from pathlib import Path


def test_run_full_calibration_script_has_path_setup():
    """run_full_calibration.py has sys.path setup before the src import."""
    script = Path(__file__).resolve().parent.parent / "calibration" / "run_full_calibration.py"
    content = script.read_text()
    # sys.path setup must appear before the 'from src' import
    path_pos = content.find("sys.path.insert")
    import_pos = content.find("from src.python.calibration import")
    assert path_pos >= 0, "Missing sys.path setup in run_full_calibration.py"
    assert import_pos >= 0, "Missing src import in run_full_calibration.py"
    assert path_pos < import_pos, "sys.path setup must appear before the src import"


def test_import_resolves_with_project_root_on_path():
    """When project root is on sys.path, calibration imports succeed."""
    project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, '{project_root}'); "
            "from src.python.calibration import run_calibration; "
            "print('OK')",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.stdout.strip() == "OK", f"Import failed: {result.stderr}"
