"""Theory-based tests for stress buffering phase.

Verifies Baron & Kenny mediation properties:
- a-path: stress -> resources (negative)
- b-path: resources -> buffering (positive)
- c'-path: stress -> buffering|resources (negative)
- Indirect effect a*b != 0 confirms mediation
- Social support as parallel mediator
- PF boost: larger when R far below R0; never exceeds R0 - R_current
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from src.python.phases.interfaces import AgentState, PhaseOutput
from src.python.phases.stress_buffering import PHASE_FREQUENCY, run_phase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FACTORS = ["social_support", "family_support", "formal_intervention", "psychological_capital"]


def _make_state(**overrides: Any) -> AgentState:
    """Build a minimal valid AgentState for stress buffering."""
    defaults: Dict[str, Any] = dict(
        resilience=0.5,
        baseline_resilience=0.5,
        resources=0.5,
        current_stress=0.3,
        protective_factors={f: 0.5 for f in FACTORS},
    )
    defaults.update(overrides)
    return AgentState(**defaults)


def _make_config(**overrides: Any) -> Dict[str, Any]:
    """Build a minimal config for stress buffering."""
    defaults = dict(
        boost_rate=0.1,
        a_coefficient=-0.3,
        b_coefficient=0.5,
        c_prime_coefficient=-0.2,
        social_stress_path=-0.2,
        social_buffering_path=0.4,
    )
    defaults.update(overrides)
    return defaults


def _run(state: AgentState, config: Dict[str, Any], seed: int = 42) -> PhaseOutput:
    """Run phase with a seeded RNG and return output."""
    rng = np.random.default_rng(seed)
    return run_phase(state, config, rng)


# ---------------------------------------------------------------------------
# Phase contract
# ---------------------------------------------------------------------------


class TestPhaseContract:
    """Phase module exports the correct contract."""

    def test_frequency_is_daily(self):
        assert PHASE_FREQUENCY == "daily"

    def test_run_phase_is_callable(self):
        assert callable(run_phase)

    def test_run_phase_returns_phase_output(self):
        state = _make_state()
        config = _make_config()
        rng = np.random.default_rng(42)
        result = run_phase(state, config, rng)
        assert isinstance(result, dict)
        assert "state_delta" in result
        assert "observation" in result


# ---------------------------------------------------------------------------
# Mediation a-path: stress -> resources
# ---------------------------------------------------------------------------


class TestMediationAPath:
    """Theory: a-path (stress -> resources) is negative."""

    def test_higher_stress_lower_resources(self):
        """Higher stress -> more resource depletion (lower final resources)."""
        state_low_stress = _make_state(current_stress=0.1, resources=0.5)
        state_high_stress = _make_state(current_stress=0.9, resources=0.5)
        config = _make_config()
        out_low = _run(state_low_stress, config)
        out_high = _run(state_high_stress, config)
        assert out_high["state_delta"]["resources"] < out_low["state_delta"]["resources"]

    def test_a_path_coefficient_in_observation(self):
        """Observation includes a_coefficient."""
        config = _make_config(a_coefficient=-0.5)
        state = _make_state()
        out = _run(state, config)
        assert out["observation"]["a_coefficient"] == -0.5

    def test_resource_depletion_formula(self):
        """Resources decrease by a * stress."""
        state = _make_state(current_stress=0.4, resources=0.7)
        config = _make_config(a_coefficient=-0.3)
        out = _run(state, config)
        expected = max(0.0, min(1.0, 0.7 + (-0.3 * 0.4)))
        assert out["state_delta"]["resources"] == pytest.approx(expected)

    def test_resources_clamped_below_zero(self):
        """Resources never go below 0."""
        state = _make_state(current_stress=1.0, resources=0.05)
        config = _make_config(a_coefficient=-0.5)
        out = _run(state, config)
        assert out["state_delta"]["resources"] >= 0.0


# ---------------------------------------------------------------------------
# Mediation b-path: resources -> buffering
# ---------------------------------------------------------------------------


class TestMediationBPath:
    """Theory: b-path (resources -> buffering) is positive."""

    def test_higher_resources_stronger_buffering(self):
        """Higher resources -> stronger buffering (same stress)."""
        state_low = _make_state(resources=0.1, current_stress=0.3)
        state_high = _make_state(resources=0.9, current_stress=0.3)
        config = _make_config(c_prime_coefficient=0.0)  # isolate b-path
        out_low = _run(state_low, config)
        out_high = _run(state_high, config)
        assert out_high["observation"]["buffering_strength"] > out_low["observation"]["buffering_strength"]

    def test_b_path_coefficient_in_observation(self):
        """Observation includes b_coefficient."""
        config = _make_config(b_coefficient=0.7)
        out = _run(_make_state(), config)
        assert out["observation"]["b_coefficient"] == 0.7

    def test_buffering_increases_with_b_coefficient(self):
        """Higher b -> stronger buffering for same resources."""
        state = _make_state(resources=0.6, current_stress=0.2)
        out_low_b = _run(state, _make_config(b_coefficient=0.2, c_prime_coefficient=0.0))
        out_high_b = _run(state, _make_config(b_coefficient=0.8, c_prime_coefficient=0.0))
        assert out_high_b["observation"]["buffering_strength"] > out_low_b["observation"]["buffering_strength"]

    def test_zero_resources_minimal_buffering(self):
        """Zero resources -> buffering_strength = 0 (b * 0 = 0, no c' effect)."""
        state = _make_state(resources=0.0, current_stress=0.0)
        config = _make_config(b_coefficient=0.5, c_prime_coefficient=0.0)
        out = _run(state, config)
        assert out["observation"]["buffering_strength"] == 0.0


# ---------------------------------------------------------------------------
# Mediation c'-path: stress -> buffering | resources
# ---------------------------------------------------------------------------


class TestMediationCPath:
    """Theory: c'-path (stress -> buffering controlling for resources) is negative."""

    def test_higher_stress_lower_buffering_controlling_for_resources(self):
        """Same resources, higher stress -> lower buffering (c'-path negative).

        Uses non-zero b_coefficient so buffering stays positive and the
        c'-path reduction is detectable (not lost by max(0, ...) clamp).
        """
        state_low_stress = _make_state(resources=0.5, current_stress=0.1)
        state_high_stress = _make_state(resources=0.5, current_stress=0.9)
        config = _make_config(b_coefficient=0.5)  # c' effect visible on positive buffering
        out_low = _run(state_low_stress, config)
        out_high = _run(state_high_stress, config)
        assert out_high["observation"]["buffering_strength"] < out_low["observation"]["buffering_strength"]

    def test_c_prime_in_observation(self):
        """Observation includes c_prime_coefficient."""
        config = _make_config(c_prime_coefficient=-0.4)
        out = _run(_make_state(), config)
        assert out["observation"]["c_prime_coefficient"] == -0.4

    def test_more_negative_c_prime_reduces_buffering(self):
        """More negative c' -> lower buffering for same stress and resources."""
        state = _make_state(resources=0.5, current_stress=0.5)
        out_mild = _run(state, _make_config(c_prime_coefficient=-0.1, b_coefficient=0.5))
        out_strong = _run(state, _make_config(c_prime_coefficient=-0.9, b_coefficient=0.5))
        assert out_strong["observation"]["buffering_strength"] < out_mild["observation"]["buffering_strength"]


# ---------------------------------------------------------------------------
# Indirect effect a*b
# ---------------------------------------------------------------------------


class TestIndirectEffect:
    """Theory: indirect effect a*b != 0 confirms mediation (Baron & Kenny)."""

    def test_indirect_effect_nonzero(self):
        """a*b != 0 when both a and b are non-zero."""
        config = _make_config(a_coefficient=-0.3, b_coefficient=0.5)
        out = _run(_make_state(), config)
        assert out["observation"]["indirect_effect"] == pytest.approx(-0.15)

    def test_indirect_effect_in_observation(self):
        """Observation includes indirect_effect."""
        out = _run(_make_state(), _make_config())
        assert "indirect_effect" in out["observation"]

    def test_indirect_effect_zero_when_a_zero(self):
        """a = 0 -> indirect effect = 0."""
        config = _make_config(a_coefficient=0.0, b_coefficient=0.5)
        out = _run(_make_state(), config)
        assert out["observation"]["indirect_effect"] == 0.0

    def test_indirect_effect_zero_when_b_zero(self):
        """b = 0 -> indirect effect = 0."""
        config = _make_config(a_coefficient=-0.3, b_coefficient=0.0)
        out = _run(_make_state(), config)
        assert out["observation"]["indirect_effect"] == 0.0


# ---------------------------------------------------------------------------
# Social support as parallel mediator
# ---------------------------------------------------------------------------


class TestSocialSupportMediation:
    """Theory: social_support as a parallel mediator of buffering."""

    def test_social_support_mediation_in_observation(self):
        """Observation includes social_support_mediation."""
        out = _run(_make_state(), _make_config())
        assert "social_support_mediation" in out["observation"]

    def test_higher_social_support_stronger_mediation(self):
        """Higher social_support efficacy -> stronger social_support_mediation."""
        state_low_ss = _make_state(
            protective_factors={
                "social_support": 0.1,
                "family_support": 0.5,
                "formal_intervention": 0.5,
                "psychological_capital": 0.5,
            }
        )
        state_high_ss = _make_state(
            protective_factors={
                "social_support": 0.9,
                "family_support": 0.5,
                "formal_intervention": 0.5,
                "psychological_capital": 0.5,
            }
        )
        config = _make_config(social_buffering_path=0.4)
        out_low = _run(state_low_ss, config)
        out_high = _run(state_high_ss, config)
        assert out_high["observation"]["social_support_mediation"] > out_low["observation"]["social_support_mediation"]

    def test_social_mediation_formula(self):
        """social_support_mediation = social_support_efficacy * social_buffering_path."""
        state = _make_state(
            protective_factors={
                "social_support": 0.7,
                "family_support": 0.5,
                "formal_intervention": 0.5,
                "psychological_capital": 0.5,
            }
        )
        config = _make_config(social_buffering_path=0.4)
        out = _run(state, config)
        expected = 0.7 * 0.4
        assert out["observation"]["social_support_mediation"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# PF boost to resilience
# ---------------------------------------------------------------------------


class TestPFBoost:
    """Theory: protective factors boost resilience toward baseline."""

    def test_boost_when_resilience_below_baseline(self):
        """resilience < baseline -> pf_boost > 0 -> resilience increases."""
        state = _make_state(resilience=0.3, baseline_resilience=0.7)
        config = _make_config(boost_rate=0.1)
        out = _run(state, config)
        assert out["state_delta"]["resilience"] > 0.3

    def test_no_boost_when_at_or_above_baseline(self):
        """resilience >= baseline -> no boost."""
        state = _make_state(resilience=0.7, baseline_resilience=0.5)
        config = _make_config(boost_rate=0.1)
        out = _run(state, config)
        assert out["state_delta"]["resilience"] == pytest.approx(0.7)

    def test_larger_gap_larger_boost(self):
        """Larger baseline - current gap -> larger pf_boost."""
        state_small_gap = _make_state(resilience=0.4, baseline_resilience=0.5)
        state_large_gap = _make_state(resilience=0.1, baseline_resilience=0.5)
        config = _make_config(boost_rate=0.1)
        out_small = _run(state_small_gap, config)
        out_large = _run(state_large_gap, config)
        boost_small = out_small["state_delta"]["resilience"] - 0.4
        boost_large = out_large["state_delta"]["resilience"] - 0.1
        assert boost_large > boost_small

    def test_boost_never_exceeds_gap(self):
        """pf_boost never exceeds (baseline - current) even with high boost_rate."""
        state = _make_state(resilience=0.3, baseline_resilience=0.5)
        config = _make_config(boost_rate=10.0)
        out = _run(state, config)
        boost = out["state_delta"]["resilience"] - 0.3
        gap = 0.5 - 0.3
        assert boost <= gap + 1e-10

    def test_higher_pf_efficacy_higher_boost(self):
        """Higher PF efficacy -> larger boost (same gap)."""
        state_low_pf = _make_state(
            resilience=0.3,
            baseline_resilience=0.7,
            protective_factors={f: 0.1 for f in FACTORS},
        )
        state_high_pf = _make_state(
            resilience=0.3,
            baseline_resilience=0.7,
            protective_factors={f: 0.9 for f in FACTORS},
        )
        config = _make_config(boost_rate=0.1)
        out_low = _run(state_low_pf, config)
        out_high = _run(state_high_pf, config)
        boost_low = out_low["state_delta"]["resilience"] - 0.3
        boost_high = out_high["state_delta"]["resilience"] - 0.3
        assert boost_high > boost_low

    def test_pf_boost_in_observation(self):
        """Observation includes pf_boost value."""
        state = _make_state(resilience=0.3, baseline_resilience=0.7)
        out = _run(state, _make_config(boost_rate=0.1))
        assert "pf_boost" in out["observation"]
        assert out["observation"]["pf_boost"] > 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and resilience."""

    def test_no_stress_no_resource_change(self):
        """current_stress = 0 -> no stress-induced resource change."""
        state = _make_state(current_stress=0.0, resources=0.5)
        out = _run(state, _make_config())
        assert out["state_delta"]["resources"] == pytest.approx(0.5)

    def test_zero_pf_no_boost(self):
        """All protective factors at 0 -> no pf_boost."""
        state = _make_state(
            resilience=0.3,
            baseline_resilience=0.7,
            protective_factors={f: 0.0 for f in FACTORS},
        )
        out = _run(state, _make_config(boost_rate=0.1))
        assert out["observation"]["pf_boost"] == 0.0
        assert out["state_delta"]["resilience"] == pytest.approx(0.3)

    def test_buffering_strength_non_negative(self):
        """buffering_strength clamped to >= 0."""
        state = _make_state(resources=0.0, current_stress=1.0)
        config = _make_config(b_coefficient=0.5, c_prime_coefficient=-2.0)
        out = _run(state, config)
        assert out["observation"]["buffering_strength"] >= 0.0

    def test_resilience_clamped_to_one(self):
        """Resilience never exceeds 1.0."""
        state = _make_state(
            resilience=0.95,
            baseline_resilience=1.0,
            protective_factors={f: 1.0 for f in FACTORS},
        )
        config = _make_config(boost_rate=1.0)
        out = _run(state, config)
        assert out["state_delta"]["resilience"] <= 1.0

    def test_deterministic_with_seed(self):
        """Same state + config + seed -> identical output."""
        state = _make_state()
        config = _make_config()
        out1 = _run(state, config, seed=42)
        out2 = _run(state, config, seed=42)
        assert out1["state_delta"] == out2["state_delta"]
        assert out1["observation"] == out2["observation"]

    def test_no_overlap_with_plan003_keys(self):
        """State delta does NOT use 'delta_resilience' or 'DeltaR' as keys.

        Plan 003 uses coping-induced DeltaR. Plan 006 uses direct 'resilience' key.
        """
        state = _make_state(resilience=0.3, baseline_resilience=0.7)
        out = _run(state, _make_config(boost_rate=0.1))
        delta_keys = set(out["state_delta"].keys())
        # Plan 006 uses 'resilience' (full value), not 'DeltaR' (delta value)
        assert "resilience" in delta_keys
        assert "DeltaR" not in delta_keys
