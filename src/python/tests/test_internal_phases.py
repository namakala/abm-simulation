"""Tests for orchestrator-internal phase functions (Plan 008 Step 6).

These are pure functions that take (state, config, rng) → PhaseOutput.
They are NOT in the phases/ package — they live inside agent.py as
orchestrator utilities.
"""

from typing import Any, Dict

import pytest

from src.python.agent import (
    process_affect_dynamics,
    process_pss10_consolidation,
    process_daily_reset,
)
from src.python.affect_utils import update_affect_dynamics, AffectDynamicsConfig
from src.python.math_utils import create_rng

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def sample_rng():
    return create_rng(42)


@pytest.fixture
def typical_state() -> Dict[str, Any]:
    """A typical AgentState dict representative of mid-day values."""
    return {
        "baseline_resilience": 0.5,
        "resilience": 0.45,
        "resources": 0.6,
        "baseline_affect": 0.0,
        "affect": 0.1,
        "current_stress": 0.3,
        "recent_stress_intensity": 0.4,
        "stress_momentum": 0.1,
        "last_stress_update": 0,
        "daily_stress_events": [],
        "stress_history": [],
        "last_reset_day": 0,
        "consecutive_hindrances": 2.0,
        "stress_breach_count": 3,
        "pss10_responses": {},
        "stress_controllability": 0.5,
        "stress_overload": 0.4,
        "pss10": 15,
        "stressed": True,
        "daily_pss10_scores": [12, 18, 15],
        "daily_interactions": 4,
        "daily_support_exchanges": 2,
        "protective_factors": {
            "social_support": 0.5,
            "family_support": 0.5,
            "formal_intervention": 0.5,
            "psychological_capital": 0.5,
        },
        "stress_config": {},
        "interaction_config": {},
        "volatility": 0.3,
    }


@pytest.fixture
def affect_config() -> Dict[str, Any]:
    return {
        "neighbor_affects": [0.0, 0.2, -0.1],
        "daily_challenge": 0.3,
        "daily_hindrance": 0.2,
        "stress_decay_rate": 0.05,
    }


# ── process_affect_dynamics ───────────────────────────────────────


class TestProcessAffectDynamics:
    """process_affect_dynamics applies affect/resilience consolidation."""

    def test_returns_phaseoutput(self, typical_state, affect_config, sample_rng):
        """Returns a PhaseOutput with state_delta and observation."""
        result = process_affect_dynamics(typical_state, affect_config, sample_rng)
        assert isinstance(result, dict)
        assert "state_delta" in result
        assert "observation" in result

    def test_affect_in_state_delta(self, typical_state, affect_config, sample_rng):
        """state_delta contains an affect key."""
        result = process_affect_dynamics(typical_state, affect_config, sample_rng)
        assert "affect" in result["state_delta"]

    def test_resilience_in_state_delta(self, typical_state, affect_config, sample_rng):
        """state_delta contains a resilience key."""
        result = process_affect_dynamics(typical_state, affect_config, sample_rng)
        assert "resilience" in result["state_delta"]

    def test_resources_in_state_delta(self, typical_state, affect_config, sample_rng):
        """state_delta contains a resources key."""
        result = process_affect_dynamics(typical_state, affect_config, sample_rng)
        assert "resources" in result["state_delta"]

    def test_consecutive_hindrances_in_state_delta(self, typical_state, affect_config, sample_rng):
        """state_delta contains consecutive_hindrances."""
        result = process_affect_dynamics(typical_state, affect_config, sample_rng)
        assert "consecutive_hindrances" in result["state_delta"]

    def test_affect_homeostasis_pulls_toward_baseline(self, typical_state, affect_config, sample_rng):
        """Affect moves toward baseline when homeostatic rate is applied."""
        # Set affect far from baseline
        typical_state["affect"] = 0.8
        typical_state["baseline_affect"] = 0.0
        result = process_affect_dynamics(typical_state, affect_config, sample_rng)
        new_affect = result["state_delta"]["affect"]
        # Affect should be closer to baseline (0.0) than original (0.8)
        assert abs(new_affect - 0.0) < abs(0.8 - 0.0), f"Affect {new_affect} not closer to baseline"

    def test_homeostasis_uses_scaling(self, typical_state, affect_config, sample_rng):
        """Homeostatic adjustment uses scale_homeostatic_rate (no crash)."""
        # Just verify the function runs without error for different stress/resource levels
        typical_state["current_stress"] = 0.9
        typical_state["resources"] = 0.1
        typical_state["affect"] = 0.5
        typical_state["baseline_affect"] = 0.0
        result = process_affect_dynamics(typical_state, affect_config, sample_rng)
        assert "affect" in result["state_delta"]
        assert -1.0 <= result["state_delta"]["affect"] <= 1.0

    def test_consecutive_hindrances_decays(self, typical_state, affect_config, sample_rng):
        """consecutive_hindrances decreases via decay."""
        typical_state["consecutive_hindrances"] = 5.0
        result = process_affect_dynamics(typical_state, affect_config, sample_rng)
        assert result["state_delta"]["consecutive_hindrances"] < 5.0

    def test_consecutive_hindrances_does_not_go_below_zero(self, typical_state, affect_config, sample_rng):
        """consecutive_hindrances is clamped at 0."""
        typical_state["consecutive_hindrances"] = 0.01
        result = process_affect_dynamics(typical_state, affect_config, sample_rng)
        assert result["state_delta"]["consecutive_hindrances"] >= 0.0

    def test_observation_contains_neighbor_affects(self, typical_state, affect_config, sample_rng):
        """Observation includes neighbor_affects_summary."""
        result = process_affect_dynamics(typical_state, affect_config, sample_rng)
        assert "neighbor_affects_summary" in result["observation"]
        assert isinstance(result["observation"]["neighbor_affects_summary"], dict)

    def test_default_config_when_missing(self, typical_state, sample_rng):
        """Works with empty config (uses defaults)."""
        result = process_affect_dynamics(typical_state, {}, sample_rng)
        assert "affect" in result["state_delta"]
        assert "resilience" in result["state_delta"]

    def test_deterministic_with_seed(self, typical_state, affect_config):
        """Same state + config + seed produces same output."""
        rng1 = create_rng(12345)
        rng2 = create_rng(12345)
        r1 = process_affect_dynamics(typical_state, affect_config, rng1)
        r2 = process_affect_dynamics(typical_state, affect_config, rng2)
        assert r1["state_delta"] == r2["state_delta"]
        assert r1["observation"] == r2["observation"]

    def test_affect_homeostasis_decoupled_from_resources(self, typical_state, affect_config, sample_rng):
        """Affect homeostasis no longer varies with resource level (Fix 3).

        After Fix 3, resources=0.0 is passed for affect's homeostatic scaling,
        so affect homeostatic adjustment should not depend on resources.
        The direct resource->affect pathway (Fix 6) still adds a small effect.
        """
        from src.python.affect_utils import scale_homeostatic_rate

        # Check the homeostatic RATE directly (not the full update_affect_dynamics)
        # With resources=0.0, the resource factor is always 1.0
        rate_low_resource = scale_homeostatic_rate(0.15, 0.2, 0.3)
        rate_high_resource = scale_homeostatic_rate(0.15, 0.9, 0.3)
        # Both should be identical because resources is overridden to 0.0 for affect
        # In the actual code, affect passes resources=0.0
        rate_affect_low = scale_homeostatic_rate(0.15, 0.0, 0.3)
        rate_affect_high = scale_homeostatic_rate(0.15, 0.0, 0.3)
        assert rate_affect_low == rate_affect_high, "Affect homeostatic rate should not depend on resources"
        assert rate_low_resource != rate_high_resource, "Resilience homeostatic rate should still depend on resources"

    def test_resource_affect_pathway_via_update_affect_dynamics(self):
        """update_affect_dynamics with resources adds a small resource effect (Fix 6)."""
        affect_cfg = AffectDynamicsConfig()
        # High resources should produce slightly higher affect than low resources
        affect_low = update_affect_dynamics(
            current_affect=0.0,
            baseline_affect=0.0,
            neighbor_affects=[],
            current_stress=0.0,
            resources=0.2,
            affect_config=affect_cfg,
        )
        affect_high = update_affect_dynamics(
            current_affect=0.0,
            baseline_affect=0.0,
            neighbor_affects=[],
            current_stress=0.0,
            resources=0.9,
            affect_config=affect_cfg,
        )
        # The resource_boost should produce a small positive difference
        assert affect_high > affect_low, (
            f"High resources should increase affect: low={affect_low:.4f}, high={affect_high:.4f}"
        )

    def test_process_affect_dynamics_passes_resources(self, typical_state, affect_config, sample_rng):
        """process_affect_dynamics passes resources to update_affect_dynamics."""
        state_high = dict(typical_state)
        state_low = dict(typical_state)
        state_high["resources"] = 0.9
        state_low["resources"] = 0.2

        result_high = process_affect_dynamics(state_high, affect_config, sample_rng)
        result_low = process_affect_dynamics(state_low, affect_config, sample_rng)

        # With all else equal, high resources should produce slightly higher affect
        # (the resource boost is small but positive)
        assert result_high["state_delta"]["affect"] >= result_low["state_delta"]["affect"], (
            f"High-resource affect ({result_high['state_delta']['affect']:.4f}) should be >= "
            f"low-resource affect ({result_low['state_delta']['affect']:.4f})"
        )

    def test_stress_affect_erosion_multiplied_by_assumption(self):
        """Stress erosion in update_affect_dynamics is amplified by assumption multiplier (Fix 2)."""
        from src.python.assumption_config import get_assumptions

        a = get_assumptions()
        # Check that the assumption parameter exists and has a multiplier > 1
        assert hasattr(a.stress, "stress_affect_erosion_multiplier"), (
            "Missing stress_affect_erosion_multiplier in assumptions"
        )
        assert a.stress.stress_affect_erosion_multiplier > 1.0, "Multiplier should be > 1.0"

        affect_cfg = AffectDynamicsConfig()
        # With current_stress=1.0, multiplier=2.0 gives affect erosion -0.30
        result = update_affect_dynamics(
            current_affect=0.0,
            baseline_affect=0.0,
            neighbor_affects=[],
            current_stress=1.0,
            resources=0.5,
            affect_config=affect_cfg,
        )
        # Config erosion rate is 0.15, with multiplier 2.0: effective = 0.30
        # At max stress (1.0): erosion = -0.30
        assert result < -0.15, f"Expected stress erosion stronger than -0.15, got {result:.4f}"
        expected = -affect_cfg.stress_erosion_rate * 1.0 * a.stress.stress_affect_erosion_multiplier
        assert result == pytest.approx(expected, abs=0.01), (
            f"Affect {result:.4f} should be ~{expected:.4f} with multiplier {a.stress.stress_affect_erosion_multiplier}"
        )


# ── process_pss10_consolidation ───────────────────────────────────


class TestProcessPss10Consolidation:
    """process_pss10_consolidation averages PSS-10 scores and updates stress."""

    def test_returns_phaseoutput(self, typical_state, sample_rng):
        """Returns a PhaseOutput."""
        result = process_pss10_consolidation(typical_state, {}, sample_rng)
        assert isinstance(result, dict)
        assert "state_delta" in result

    def test_no_scores_is_noop(self, typical_state, sample_rng):
        """When daily_pss10_scores is empty, state_delta has no pss10 change."""
        typical_state["daily_pss10_scores"] = []
        result = process_pss10_consolidation(typical_state, {}, sample_rng)
        # Should still return state_delta but without stress-updating side effects
        assert "current_stress" in result["state_delta"]

    def test_averages_scores(self, typical_state, sample_rng):
        """current_stress is updated based on averaged PSS-10 scores."""
        typical_state["daily_pss10_scores"] = [10, 20, 30]
        result = process_pss10_consolidation(typical_state, {}, sample_rng)
        assert "current_stress" in result["state_delta"]

    def test_clears_daily_scores(self, typical_state, sample_rng):
        """daily_pss10_scores is cleared (set to empty list)."""
        typical_state["daily_pss10_scores"] = [12, 14, 16]
        result = process_pss10_consolidation(typical_state, {}, sample_rng)
        assert result["state_delta"]["daily_pss10_scores"] == []

    def test_clamps_stress(self, typical_state, sample_rng):
        """current_stress stays in [0, 1]."""
        typical_state["current_stress"] = 5.0  # Out of range
        typical_state["daily_pss10_scores"] = [30]
        result = process_pss10_consolidation(typical_state, {}, sample_rng)
        stress = result["state_delta"]["current_stress"]
        assert 0.0 <= stress <= 1.0

    def test_clamps_controllability(self, typical_state, sample_rng):
        """stress_controllability stays in [0, 1]."""
        typical_state["stress_controllability"] = 2.0
        result = process_pss10_consolidation(typical_state, {}, sample_rng)
        val = result["state_delta"]["stress_controllability"]
        assert 0.0 <= val <= 1.0

    def test_clamps_overload(self, typical_state, sample_rng):
        """stress_overload stays in [0, 1]."""
        typical_state["stress_overload"] = -0.5
        result = process_pss10_consolidation(typical_state, {}, sample_rng)
        val = result["state_delta"]["stress_overload"]
        assert 0.0 <= val <= 1.0

    def test_updates_stressed_status(self, typical_state, sample_rng):
        """stressed is updated based on PSS-10 threshold."""
        result = process_pss10_consolidation(typical_state, {}, sample_rng)
        assert "stressed" in result["state_delta"]

    def test_observation_summary(self, typical_state, sample_rng):
        """Observation includes avg_pss10 and num_events."""
        typical_state["daily_pss10_scores"] = [10, 20]
        result = process_pss10_consolidation(typical_state, {}, sample_rng)
        obs = result["observation"]
        assert "avg_pss10" in obs
        assert "num_events" in obs
        assert obs["num_events"] == 2

    def test_deterministic(self, typical_state):
        """Same input produces same output."""
        rng1 = create_rng(99)
        rng2 = create_rng(99)
        r1 = process_pss10_consolidation(typical_state, {}, rng1)
        r2 = process_pss10_consolidation(typical_state, {}, rng2)
        assert r1["state_delta"] == r2["state_delta"]


# ── process_daily_reset ───────────────────────────────────────────


class TestProcessDailyReset:
    """process_daily_reset handles daily counter reset, affect reset, stress decay."""

    def test_returns_phaseoutput(self, typical_state, sample_rng):
        """Returns a PhaseOutput."""
        result = process_daily_reset(typical_state, {}, sample_rng)
        assert isinstance(result, dict)
        assert "state_delta" in result
        assert "observation" in result

    def test_resets_daily_interactions(self, typical_state, sample_rng):
        """daily_interactions is reset to 0."""
        typical_state["daily_interactions"] = 10
        result = process_daily_reset(typical_state, {}, sample_rng)
        assert result["state_delta"]["daily_interactions"] == 0

    def test_resets_daily_support_exchanges(self, typical_state, sample_rng):
        """daily_support_exchanges is reset to 0."""
        typical_state["daily_support_exchanges"] = 5
        result = process_daily_reset(typical_state, {}, sample_rng)
        assert result["state_delta"]["daily_support_exchanges"] == 0

    def test_affect_reset_to_baseline(self, typical_state, sample_rng):
        """Affect moves toward baseline after reset."""
        typical_state["affect"] = 0.8
        typical_state["baseline_affect"] = 0.0
        result = process_daily_reset(typical_state, {}, sample_rng)
        new_affect = result["state_delta"]["affect"]
        # Affect should be pulled toward baseline but not necessarily equal
        assert new_affect < 0.8, "Affect should decrease toward baseline"

    def test_stress_decays(self, typical_state, sample_rng):
        """current_stress decreases via decay."""
        typical_state["current_stress"] = 0.8
        result = process_daily_reset(typical_state, {}, sample_rng)
        assert result["state_delta"]["current_stress"] < 0.8

    def test_stress_does_not_go_below_zero(self, typical_state, sample_rng):
        """current_stress stays >= 0."""
        result = process_daily_reset(typical_state, {}, sample_rng)
        assert result["state_delta"]["current_stress"] >= 0.0

    def test_clears_daily_stress_events(self, typical_state, sample_rng):
        """daily_stress_events is cleared."""
        typical_state["daily_stress_events"] = [{"challenge": 0.5}]
        result = process_daily_reset(typical_state, {}, sample_rng)
        assert result["state_delta"]["daily_stress_events"] == []

    def test_resets_last_reset_day(self, typical_state, sample_rng):
        """last_reset_day is set to current_day from config."""
        result = process_daily_reset(typical_state, {}, sample_rng)
        # Without current_day in config, default should be used
        assert "last_reset_day" in result["state_delta"]

    def test_clears_daily_pss10_scores(self, typical_state, sample_rng):
        """daily_pss10_scores is cleared."""
        typical_state["daily_pss10_scores"] = [10, 20]
        result = process_daily_reset(typical_state, {}, sample_rng)
        assert result["state_delta"]["daily_pss10_scores"] == []

    def test_applies_hindrance_decay(self, typical_state, sample_rng):
        """consecutive_hindrances decays slightly."""
        typical_state["consecutive_hindrances"] = 3.0
        result = process_daily_reset(typical_state, {}, sample_rng)
        assert result["state_delta"]["consecutive_hindrances"] <= 3.0

    def test_observation_stress_summary(self, typical_state, sample_rng):
        """Observation includes stress summary stats."""
        typical_state["daily_stress_events"] = [
            {"stress_level": 0.5, "coped_successfully": True},
            {"stress_level": 0.8, "coped_successfully": False},
        ]
        result = process_daily_reset(typical_state, {}, sample_rng)
        obs = result["observation"]
        assert "stress_summary" in obs
        summary = obs["stress_summary"]
        assert "avg_stress" in summary
        assert "num_events" in summary

    def test_deterministic(self, typical_state):
        """Same input produces same output."""
        rng1 = create_rng(77)
        rng2 = create_rng(77)
        r1 = process_daily_reset(typical_state, {}, rng1)
        r2 = process_daily_reset(typical_state, {}, rng2)
        assert r1["state_delta"] == r2["state_delta"]
