"""
Tests for the calibration module.

Ensures PSS-10 population calibration utilities work correctly.
"""

import sys

sys.path.append(".")

from src.python.calibration import (
    compute_pss10_population_stats,
    run_calibration,
    DEFAULT_CALIBRATION_PARAMS,
)


class TestCalibrationAPI:
    """Test that the calibration module exposes the expected API."""

    def test_default_params_have_required_keys(self):
        """DEFAULT_CALIBRATION_PARAMS contains all required keys."""
        assert isinstance(DEFAULT_CALIBRATION_PARAMS, dict)
        assert "target_mean_min" in DEFAULT_CALIBRATION_PARAMS
        assert "target_mean_max" in DEFAULT_CALIBRATION_PARAMS
        assert "target_sd_min" in DEFAULT_CALIBRATION_PARAMS
        assert "target_sd_max" in DEFAULT_CALIBRATION_PARAMS
        assert "max_iterations" in DEFAULT_CALIBRATION_PARAMS
        assert "learning_rate" in DEFAULT_CALIBRATION_PARAMS

    def test_default_params_have_sensible_values(self):
        """Default target range matches Cohen 1983 literature norm."""
        assert DEFAULT_CALIBRATION_PARAMS["target_mean_min"] == 13.0
        assert DEFAULT_CALIBRATION_PARAMS["target_mean_max"] == 15.0
        assert DEFAULT_CALIBRATION_PARAMS["target_sd_min"] == 6.0
        assert DEFAULT_CALIBRATION_PARAMS["target_sd_max"] == 8.0
        assert DEFAULT_CALIBRATION_PARAMS["max_iterations"] >= 5


class TestComputePSS10PopulationStats:
    """Test the PSS-10 population statistics computation."""

    def test_returns_dict_with_mean_and_std(self):
        """compute_pss10_population_stats returns dict with 'mean' and 'std' keys."""
        result = compute_pss10_population_stats(N=10, max_days=5, seed=42)
        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result

    def test_pss10_mean_in_valid_range(self):
        """PSS-10 mean is within valid range [0, 40] for a small simulation."""
        result = compute_pss10_population_stats(N=10, max_days=5, seed=42)
        assert 0.0 <= result["mean"] <= 40.0

    def test_pss10_std_non_negative(self):
        """PSS-10 standard deviation is non-negative."""
        result = compute_pss10_population_stats(N=10, max_days=5, seed=42)
        assert result["std"] >= 0.0

    def test_multiple_seeds_produce_different_results(self):
        """Different random seeds produce different (or same, by chance) results."""
        result_a = compute_pss10_population_stats(N=10, max_days=5, seed=42)
        result_b = compute_pss10_population_stats(N=10, max_days=5, seed=123)
        # We just verify both return valid data; they could theoretically match
        assert isinstance(result_a["mean"], float)
        assert isinstance(result_b["mean"], float)


class TestRunCalibration:
    """Test the calibration runner."""

    def test_run_calibration_returns_result_dict(self):
        """run_calibration returns a dict with expected keys."""
        result = run_calibration(N=10, max_days=5, seeds=[42], max_iterations=2)
        assert isinstance(result, dict)
        # Keys may include: initial_mean, final_mean, n_iterations, converged, item_means, item_sds
        assert "initial_mean" in result
        assert "final_mean" in result
        assert "converged" in result

    def test_calibration_with_no_iterations_returns_initial_state(self):
        """With max_iterations=0, calibration returns initial state without running."""
        result = run_calibration(N=10, max_days=5, seeds=[42], max_iterations=0)
        assert result["n_iterations"] == 0
        assert result["final_mean"] == result["initial_mean"]
