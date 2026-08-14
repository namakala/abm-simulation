"""Tests for the CalibrationProgress progress reporter."""

import re
from unittest import mock
from src.python.calibration.progress import CalibrationProgress


class TestCalibrationProgress:
    """Test the CalibrationProgress progress reporter."""

    def test_init_stores_total_and_label(self):
        """CalibrationProgress stores total and label."""
        progress = CalibrationProgress(total=20, label="PSS-10 mean")
        assert progress.total == 20
        assert progress.label == "PSS-10 mean"

    def test_default_label(self):
        """Default label is 'Calibration'."""
        progress = CalibrationProgress(total=10)
        assert progress.label == "Calibration"

    def test_update_prints_timestamped_line(self, capfd):
        """update prints a line with timestamp, iteration, mean, and std on stderr."""
        progress = CalibrationProgress(total=20, label="Test")
        progress.update(iteration=3, mean=13.8, std=7.2)
        captured = capfd.readouterr()
        assert captured.out == ""
        line = captured.err.strip()
        assert re.match(
            r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] Test iteration 3/20 \| mean=13\.8, std=7\.2",
            line,
        ), f"Unexpected format: {line!r}"

    def test_update_with_converged(self, capfd):
        """update with converged=True includes convergence indicator."""
        progress = CalibrationProgress(total=20, label="Test")
        progress.update(iteration=5, mean=14.0, std=6.5, converged=True)
        captured = capfd.readouterr()
        assert "converged" in captured.err.lower()

    def test_multiple_updates_each_new_line(self, capfd):
        """Each update call produces a separate line (no overwriting)."""
        progress = CalibrationProgress(total=20, label="Test")
        progress.update(iteration=1, mean=10.0, std=5.0)
        progress.update(iteration=2, mean=11.0, std=5.5)
        progress.update(iteration=3, mean=12.0, std=6.0)
        captured = capfd.readouterr()
        lines = [line for line in captured.err.strip().split("\n") if line]
        assert len(lines) == 3

    def test_close_does_not_raise(self):
        """close completes without error."""
        progress = CalibrationProgress(total=20)
        progress.close()


class TestCalibrationProgressIntegration:
    """Test CalibrationProgress is used inside run_calibration."""

    def test_run_calibration_prints_progress(self, capfd):
        """run_calibration prints progress lines to stderr."""
        from src.python.calibration.calibrate_pss10_population import run_calibration

        # Mock measurement to avoid slow simulations, and persistence to avoid file access
        with (
            mock.patch(
                "src.python.calibration.calibrate_pss10_population._measure_across_seeds",
                side_effect=[
                    (20.0, 7.0),  # initial measurement: above target
                    (20.0, 7.0),  # iteration 1: still above target
                    (14.0, 6.8),  # iteration 2: hits target -> convergence
                ],
            ),
            mock.patch("src.python.calibration.calibrate_pss10_population.persist_calibration_results"),
            mock.patch(
                "src.python.calibration.calibrate_pss10_population.run_verification_tests",
                return_value=True,
            ),
        ):
            result = run_calibration(N=10, max_days=2, seeds=[42], max_iterations=5, learning_rate=0.5)

        captured = capfd.readouterr()
        lines = [line for line in captured.err.strip().split("\n") if line]
        assert len(lines) == 2, f"Expected 2 progress lines, got {len(lines)}"
        assert "iteration 1/5" in lines[0]
        assert "iteration 2/5" in lines[1]
        assert "converged" in lines[1].lower()
        assert result["converged"]
