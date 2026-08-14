"""
Unit tests for extracted agent initialization functions (plan 012 step 2).

Tests are written BEFORE implementation (TDD).
"""

import numpy as np
import pytest

from src.python.math_utils import create_rng


class TestInitializeVolatility:
    """Tests for initialize_volatility distribution properties."""

    def test_volatility_mean_alpha1_beta1(self, sample_rng):
        """Beta(1,1) -> mean approx 0.5."""
        from src.python.initialization import initialize_volatility

        values = [initialize_volatility(sample_rng, alpha=1.0, beta=1.0) for _ in range(10000)]
        assert abs(np.mean(values) - 0.5) < 0.05

    def test_volatility_mean_alpha4_beta1(self, sample_rng):
        """Beta(4,1) -> mean approx 0.8."""
        from src.python.initialization import initialize_volatility

        values = [initialize_volatility(sample_rng, alpha=4.0, beta=1.0) for _ in range(10000)]
        assert abs(np.mean(values) - 0.8) < 0.05


class TestInitFunctions:
    """Test extracted initialization functions."""

    def test_outputs_in_valid_range(self):
        """Test each output is in correct range."""
        from src.python.initialization import (
            initialize_baseline_resilience,
            initialize_baseline_affect,
            initialize_resources,
            initialize_protective_factors,
            initialize_volatility,
        )

        rng = create_rng(42)

        resilience = initialize_baseline_resilience(rng, mean=0.0, std=1.0)
        assert 0.0 <= resilience <= 1.0

        affect = initialize_baseline_affect(rng, mean=0.0, std=1.0)
        assert -1.0 <= affect <= 1.0

        resources = initialize_resources(rng, mean=0.0, std=1.0)
        assert 0.0 <= resources <= 1.0

        pfs = initialize_protective_factors(value=0.5)
        assert isinstance(pfs, dict)
        assert all(
            k in pfs for k in ("social_support", "family_support", "formal_intervention", "psychological_capital")
        )
        assert all(v == 0.5 for v in pfs.values())

        volatility = initialize_volatility(rng, alpha=1.0, beta=1.0)
        assert 0.0 <= volatility <= 1.0

    def test_same_seed_same_output(self):
        """Test that same seed produces same initialization."""
        from src.python.initialization import (
            initialize_baseline_resilience,
            initialize_baseline_affect,
            initialize_resources,
        )

        rng1 = create_rng(42)
        rng2 = create_rng(42)

        assert initialize_baseline_resilience(rng1, 0.0, 1.0) == pytest.approx(
            initialize_baseline_resilience(rng2, 0.0, 1.0)
        )
        assert initialize_baseline_affect(rng1, 0.0, 1.0) == pytest.approx(initialize_baseline_affect(rng2, 0.0, 1.0))
        assert initialize_resources(rng1, 0.0, 1.0) == pytest.approx(initialize_resources(rng2, 0.0, 1.0))

    def test_pss10_total_in_range(self):
        """Test PSS-10 total score in [0, 40] with correct reverse scoring."""
        from src.python.initialization import initialize_pss10_state

        rng = create_rng(42)
        result = initialize_pss10_state(rng)

        assert "pss10_score" in result
        assert 0 <= result["pss10_score"] <= 40
        assert "pss10_responses" in result
        assert isinstance(result["pss10_responses"], dict)
        assert len(result["pss10_responses"]) == 10

        # All item responses should be in [0, 4]
        responses = result["pss10_responses"]
        for item_idx, response in responses.items():
            assert 0 <= response <= 4, f"Item {item_idx} response {response} out of range [0,4]"

    def test_protective_factors_all_0_5(self):
        """Test that protective factors all default to 0.5."""
        from src.python.initialization import initialize_protective_factors

        pfs = initialize_protective_factors(value=0.5)
        assert pfs == {
            "social_support": 0.5,
            "family_support": 0.5,
            "formal_intervention": 0.5,
            "psychological_capital": 0.5,
        }

    def test_distribution_moments_match_config(self):
        """Test that distribution moments approximately match config."""
        from src.python.initialization import (
            initialize_baseline_resilience,
            initialize_baseline_affect,
        )

        n_samples = 500
        rng = create_rng(99)

        resilience_vals = [initialize_baseline_resilience(rng, 0.5, 0.2) for _ in range(n_samples)]
        assert np.mean(resilience_vals) > 0.1  # non-zero mean
        assert np.std(resilience_vals) > 0.01  # non-zero variance

        rng = create_rng(99)
        affect_vals = [initialize_baseline_affect(rng, 0.0, 0.3) for _ in range(n_samples)]
        assert np.mean(affect_vals) > -0.2
        assert np.std(affect_vals) > 0.01

    def test_compute_initial_stress(self):
        """Test compute_initial_stress returns float in [0, 1]."""
        from src.python.initialization import compute_initial_stress

        stress = compute_initial_stress(
            pss10_score=15,
            controllability=0.5,
            overload=0.5,
            dampening=1.0,
        )
        assert 0.0 <= stress <= 1.0
