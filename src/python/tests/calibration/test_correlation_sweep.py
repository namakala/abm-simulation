"""
Tests for the correlation calibration sweep module.

Validates that the sweep function correctly runs simulations,
computes correlation targets against CI bounds, and ranks
parameter combinations by fitness.
"""

import sys

sys.path.append(".")

from src.python.calibration.correlation_sweep import (
    CORRELATION_TARGETS,
    SWEEP_PARAMS,
    FIXED_OVERRIDES,
    compute_correlation_metrics,
    run_sweep,
    SweepResult,
)


class TestSweepConstants:
    """Sweep constants define the correct grid."""

    def test_correlation_targets_defined(self):
        """CORRELATION_TARGETS is a non-empty list of dicts with required keys."""
        assert isinstance(CORRELATION_TARGETS, list)
        assert len(CORRELATION_TARGETS) > 0
        for target in CORRELATION_TARGETS:
            assert "label" in target
            assert "var1" in target
            assert "var2" in target
            assert "ci_lower" in target
            assert "ci_upper" in target

    def test_sweep_params_defined(self):
        """SWEEP_PARAMS is a non-empty dict mapping env vars to lists of values."""
        assert isinstance(SWEEP_PARAMS, dict)
        assert len(SWEEP_PARAMS) > 0
        for key, values in SWEEP_PARAMS.items():
            assert isinstance(key, str)
            assert isinstance(values, list)
            assert len(values) >= 2

    def test_fixed_overrides_defined(self):
        """FIXED_OVERRIDES is a dict of env var -> value."""
        assert isinstance(FIXED_OVERRIDES, dict)
        assert len(FIXED_OVERRIDES) > 0
        for key, value in FIXED_OVERRIDES.items():
            assert isinstance(key, str)
            assert isinstance(value, (str, int, float))


class TestComputeCorrelationMetrics:
    """compute_correlation_metrics returns correct structure."""

    def test_returns_expected_keys(self):
        """Metrics includes pss10_mean, pss10_std, and all correlation labels."""
        metrics = compute_correlation_metrics(N=10, max_days=5, seed=42)
        assert isinstance(metrics, dict)
        assert "pss10_mean" in metrics
        assert "pss10_std" in metrics
        for target in CORRELATION_TARGETS:
            label = target["label"]
            assert label in metrics
            # Value should be a float or NaN (if computation fails)
            assert isinstance(metrics[label], float)

    def test_pss10_mean_range(self):
        """PSS-10 mean is in [0, 40]."""
        metrics = compute_correlation_metrics(N=10, max_days=5, seed=42)
        assert 0.0 <= metrics["pss10_mean"] <= 40.0


class TestRunSweep:
    """run_sweep produces correct output."""

    def test_run_sweep_single_combo(self):
        """run_sweep with a single parameter combination works."""
        param_sets = [{"PSS10_BIAS_SD": "2.0", "PSS10_STRESS_DAMPENING": "1.0"}]
        results = run_sweep(N=10, max_days=5, seed=42, param_sets=param_sets)
        assert isinstance(results, list)
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SweepResult)
        assert isinstance(result.params, dict)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.targets_satisfied, int)
        assert isinstance(result.total_targets, int)
        assert result.total_targets > 0

    def test_sweep_result_ranking(self):
        """Results are sorted by targets_satisfied descending."""
        param_sets = [
            {"PSS10_BIAS_SD": "2.0", "PSS10_STRESS_DAMPENING": "1.0"},
            {"PSS10_BIAS_SD": "4.0", "PSS10_STRESS_DAMPENING": "1.0"},
            {"PSS10_BIAS_SD": "1.0", "PSS10_STRESS_DAMPENING": "1.0"},
        ]
        results = run_sweep(N=10, max_days=5, seed=42, param_sets=param_sets)
        assert len(results) == 3
        # Check descending order
        for i in range(len(results) - 1):
            assert results[i].targets_satisfied >= results[i + 1].targets_satisfied
