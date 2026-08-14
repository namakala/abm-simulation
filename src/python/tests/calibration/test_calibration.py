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


class TestPSS10PopulationStatsEnvOverride:
    """Test that compute_pss10_population_stats respects custom item_means."""

    def get_default_mean(self):
        """Helper: get the default PSS-10 mean as a baseline."""
        return compute_pss10_population_stats(N=50, max_days=10, seed=42)["mean"]

    def test_low_item_means_reduce_mean(self):
        """Low item_means should produce a lower mean than defaults."""
        default = self.get_default_mean()
        low_result = compute_pss10_population_stats(N=50, max_days=10, seed=42, item_means=[0.5] * 10)
        assert low_result["mean"] < default, f"Low item means gave {low_result['mean']:.2f}, expected < {default:.2f}"

    def test_high_item_means_increase_mean(self):
        """High item_means should produce a higher mean than defaults."""
        default = self.get_default_mean()
        high_result = compute_pss10_population_stats(N=50, max_days=10, seed=42, item_means=[3.5] * 10)
        assert high_result["mean"] > default, (
            f"High item means gave {high_result['mean']:.2f}, expected > {default:.2f}"
        )

    def test_low_vs_high_item_means_ordered_correctly(self):
        """Low item_means should produce a strictly lower mean than high item_means."""
        low = compute_pss10_population_stats(N=50, max_days=10, seed=42, item_means=[0.5] * 10)["mean"]
        high = compute_pss10_population_stats(N=50, max_days=10, seed=42, item_means=[3.5] * 10)["mean"]
        assert low < high, f"Low item means mean ({low:.2f}) should be < high item means mean ({high:.2f})"
