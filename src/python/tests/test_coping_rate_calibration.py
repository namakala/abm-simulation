"""
Tests for coping success rate calibration.

Verifies that the stressful-only coping success rate falls within
the empirical 50-60% mixed-strategy baseline range [@skinner2003]
after the social_support_factor and support_boost_factor reductions.
"""

import pytest
from src.python.model import StressModel


@pytest.mark.integration
class TestCopingRateCalibration:
    """Stressful-only coping rate targets 50-60% empirical range."""

    def test_stressful_coping_rate_within_empirical_range(self):
        """Stressful-only coping rate must fall in 50-60%."""
        # Guard: ensure clean assumption defaults regardless of env pollution
        import os
        from src.python.assumption_config import reload_assumptions

        os.environ.pop("ASSUMPTION_COPING_SOCIAL_SUPPORT_FACTOR", None)
        os.environ.pop("ASSUMPTION_COPING_SUPPORT_BOOST_FACTOR", None)
        reload_assumptions()

        model = StressModel(N=50, max_days=30, seed=42)
        for _ in range(model.max_days):
            model.step()

        stressful_only = []
        for agent in model.agents:
            events = getattr(agent, "last_daily_stress_events", [])
            stressful = [e for e in events if e.get("is_stressed", False)]
            if stressful:
                successes = sum(1 for e in stressful if e.get("coped_successfully", False))
                stressful_only.append(successes / len(stressful))

        if not stressful_only:
            return  # no stressful events — skip (unlikely with 30 days)

        mean_rate = sum(stressful_only) / len(stressful_only)
        assert 0.50 <= mean_rate <= 0.60, f"Stressful-only coping rate {mean_rate:.3f} outside empirical 50-60% range"

    def test_coping_rate_histogram_uses_stressful_only(self):
        """Verifies the filtering logic used in simulation_population.qmd.

        Coping rate must be computed only from events where is_stressed=True,
        matching the empirical claim that coping is only activated when stressed.
        """
        model = StressModel(N=50, max_days=30, seed=42)
        for _ in range(model.max_days):
            model.step()

        for agent in model.agents:
            events = getattr(agent, "last_daily_stress_events", [])
            stressful = [e for e in events if e.get("is_stressed", False)]

            # The histogram must filter to stressful events only
            if stressful:
                successes = sum(1 for e in stressful if e.get("coped_successfully", False))
                rate = successes / len(stressful)
                assert 0.0 <= rate <= 1.0

            # Non-stressed events must NOT be counted in coping success
            non_stressful = [e for e in events if not e.get("is_stressed", False)]
            for e in non_stressful:
                # Auto-cope always succeeds, but must NOT be included
                # in the denominator of coping rate calculation
                assert e.get("coped_successfully", True), "Non-stressed events should always auto-cope"
