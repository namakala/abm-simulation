"""Theory-based tests for resource allocation phase.

Verifies the core theoretical properties of resource allocation logic
before checking specific numerical values.

Tests cover:
- Softmax monotonicity: higher e_f -> higher w_f
- Temperature: high T -> uniform w_f; low T -> winner-take-most
- Diminishing returns: higher e_f -> smaller Delta e_f per unit r_f
- Regeneration: R near 0 -> high R'; R = 1 -> R' = 0; positive A boosts R'
- Conservation: sum(r_f) = available R; R always in [0,1]
- No interaction variables referenced
- Affect multiplier = 1 + 0.5 * max(0, affect); resilience multiplier = 1 + 0.3 * resilience
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from src.python.phases.interfaces import AgentState, PhaseOutput
from src.python.phases.resource_allocation import PHASE_FREQUENCY, run_phase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FACTORS = ["social_support", "family_support", "formal_intervention", "psychological_capital"]


def _make_state(**overrides: Any) -> AgentState:
    """Build a minimal valid AgentState for resource allocation."""
    defaults: Dict[str, Any] = dict(
        resources=0.5,
        affect=0.0,
        resilience=0.5,
        protective_factors={
            "social_support": 0.5,
            "family_support": 0.5,
            "formal_intervention": 0.5,
            "psychological_capital": 0.5,
        },
    )
    defaults.update(overrides)
    return AgentState(**defaults)


def _make_config(**overrides: Any) -> Dict[str, Any]:
    """Build a minimal config for resource allocation."""
    defaults = dict(
        base_regeneration=0.1,
        softmax_temperature=1.0,
        protective_improvement_rate=0.1,
    )
    defaults.update(overrides)
    return defaults


def _run(state: AgentState, config: Dict[str, Any], seed: int = 42) -> PhaseOutput:
    """Run phase with a seeded RNG and return output."""
    rng = np.random.default_rng(seed)
    return run_phase(state, config, rng)


# ---------------------------------------------------------------------------
# Module-level structural tests
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
# Resource Regeneration theory
# ---------------------------------------------------------------------------


class TestResourceRegeneration:
    """Theory: regeneration behavior independent of allocation."""

    def test_low_resources_yield_high_regeneration(self):
        """R near 0 → large positive R' (> base_regeneration * affect_mult * resil_mult)."""
        state = _make_state(resources=0.01, affect=0.0, resilience=0.5)
        config = _make_config(base_regeneration=0.1)
        out = _run(state, config)
        regen = out["observation"]["regeneration_amount"]
        # R' ≈ 0.1 * (1 - 0.01) * (1 + 0.5*0) * (1 + 0.3*0.5) ≈ 0.1*0.99*1.0*1.15 ≈ 0.114
        assert regen > 0.05, f"Expected significant regeneration at low R, got {regen}"

    def test_full_resources_yield_no_regeneration(self):
        """R = 1 → R' ≈ 0."""
        state = _make_state(resources=1.0, affect=0.0, resilience=0.5)
        config = _make_config(base_regeneration=0.1)
        out = _run(state, config)
        regen = out["observation"]["regeneration_amount"]
        assert regen < 1e-10, f"Expected ~0 regeneration at R=1, got {regen}"

    def test_regeneration_increases_with_base_rate(self):
        """Higher base_regeneration → larger R'."""
        out_low = _run(_make_state(resources=0.3, affect=0.0, resilience=0.5), _make_config(base_regeneration=0.05))
        out_high = _run(_make_state(resources=0.3, affect=0.0, resilience=0.5), _make_config(base_regeneration=0.20))
        assert out_high["observation"]["regeneration_amount"] > out_low["observation"]["regeneration_amount"]

    def test_positive_affect_boosts_regeneration(self):
        """Positive affect (A > 0) → larger R' compared to A = 0."""
        out_neutral = _run(_make_state(resources=0.3, affect=0.0, resilience=0.5), _make_config(base_regeneration=0.1))
        out_positive = _run(_make_state(resources=0.3, affect=0.5, resilience=0.5), _make_config(base_regeneration=0.1))
        assert out_positive["observation"]["regeneration_amount"] > out_neutral["observation"]["regeneration_amount"]

    def test_negative_affect_does_not_hurt_regeneration(self):
        """Negative affect (A ≤ 0) → multiplier = 1.0 (no penalty)."""
        out_negative = _run(
            _make_state(resources=0.3, affect=-0.5, resilience=0.5), _make_config(base_regeneration=0.1)
        )
        out_zero = _run(_make_state(resources=0.3, affect=0.0, resilience=0.5), _make_config(base_regeneration=0.1))
        # Negative affect should NOT reduce regeneration below neutral
        assert (
            abs(out_negative["observation"]["regeneration_amount"] - out_zero["observation"]["regeneration_amount"])
            < 1e-10
        )

    def test_resilience_boosts_regeneration(self):
        """Higher resilience → larger R' (same resources, affect)."""
        out_low_res = _run(_make_state(resources=0.3, affect=0.0, resilience=0.1), _make_config(base_regeneration=0.1))
        out_high_res = _run(_make_state(resources=0.3, affect=0.0, resilience=0.9), _make_config(base_regeneration=0.1))
        assert out_high_res["observation"]["regeneration_amount"] > out_low_res["observation"]["regeneration_amount"]

    def test_regeneration_formula_match(self):
        """R' = base_regeneration * (1 - R) * (1 + 0.5 * max(0, A)) * (1 + 0.3 * resilience)."""
        R, A, resil = 0.25, 0.4, 0.6
        state = _make_state(resources=R, affect=A, resilience=resil)
        config = _make_config(base_regeneration=0.1)
        out = _run(state, config)
        expected = 0.1 * (1.0 - R) * (1.0 + 0.5 * max(0.0, A)) * (1.0 + 0.3 * resil)
        assert out["observation"]["regeneration_amount"] == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Softmax allocation theory
# ---------------------------------------------------------------------------


class TestSoftmaxAllocation:
    """Theory: softmax-based PF allocation."""

    def test_monotonic_weight(self):
        """Higher e_f → higher w_f (monotonic)."""
        pfs = {
            "social_support": 0.1,
            "family_support": 0.9,
            "formal_intervention": 0.5,
            "psychological_capital": 0.5,
        }
        state = _make_state(resources=1.0, protective_factors=pfs)
        config = _make_config(softmax_temperature=1.0)
        out = _run(state, config)
        weights = out["observation"]["allocation_weights"]
        # family_support (0.9) should get higher weight than social_support (0.1)
        assert weights["family_support"] > weights["social_support"]

    def test_equal_efficacy_equal_weight(self):
        """All e_f equal → all w_f equal (for high enough precision)."""
        pfs = {f: 0.5 for f in FACTORS}
        state = _make_state(resources=1.0, protective_factors=pfs)
        config = _make_config(softmax_temperature=1.0)
        out = _run(state, config)
        weights = out["observation"]["allocation_weights"]
        w_vals = list(weights.values())
        for w in w_vals:
            assert w == pytest.approx(0.25, abs=1e-10)

    def test_high_temperature_uniform(self):
        """T → large → w_f ≈ uniform."""
        pfs = {"social_support": 0.9, "family_support": 0.1, "formal_intervention": 0.5, "psychological_capital": 0.3}
        state = _make_state(resources=1.0, protective_factors=pfs)
        config = _make_config(softmax_temperature=100.0)
        out = _run(state, config)
        weights = out["observation"]["allocation_weights"]
        for w in weights.values():
            assert w == pytest.approx(0.25, abs=0.02)

    def test_low_temperature_winner_take_most(self):
        """T → 0 → near one-hot for max efficacy."""
        pfs = {"social_support": 0.9, "family_support": 0.1, "formal_intervention": 0.5, "psychological_capital": 0.3}
        state = _make_state(resources=1.0, protective_factors=pfs)
        config = _make_config(softmax_temperature=0.01)
        out = _run(state, config)
        weights = out["observation"]["allocation_weights"]
        # social_support (0.9) should dominate
        assert weights["social_support"] > 0.99

    def test_resources_conservation(self):
        """sum(r_f) = available R (the total allocated to PFs)."""
        state = _make_state(resources=0.7)
        config = _make_config(softmax_temperature=1.0)
        out = _run(state, config)
        # Resources after allocation = resources_before + regen - sum(allocations)
        allocated = out["observation"]["allocated_resources"]
        total_allocated = sum(allocated.values())
        regen = out["observation"]["regeneration_amount"]
        expected_resources = min(1.0, 0.7 + regen - total_allocated)
        assert out["state_delta"]["resources"] == pytest.approx(expected_resources, rel=1e-10)
        assert 0.0 <= out["state_delta"]["resources"] <= 1.0

    def test_no_resources_no_allocation(self):
        """R = 0 → no allocation (r_f = 0 for all f)."""
        state = _make_state(resources=0.0, affect=0.0, resilience=0.5)
        config = _make_config(base_regeneration=0.0, softmax_temperature=1.0)
        out = _run(state, config)
        allocated = out["observation"]["allocated_resources"]
        for f in FACTORS:
            assert allocated[f] == 0.0

    def test_weight_sum_is_one(self):
        """sum(w_f) = 1 (allocation weights are a probability distribution)."""
        pfs = {f: np.random.uniform(0.1, 0.9) for f in FACTORS}
        state = _make_state(resources=0.8, protective_factors=pfs)
        config = _make_config(softmax_temperature=1.0)
        out = _run(state, config)
        weights = out["observation"]["allocation_weights"]
        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-10)

    def test_allocation_deterministic_with_seed(self):
        """Same state + config + seed → same allocation."""
        state = _make_state(resources=0.7)
        config = _make_config()
        out1 = _run(state, config, seed=42)
        out2 = _run(state, config, seed=42)
        assert out1["observation"]["allocation_weights"] == out2["observation"]["allocation_weights"]


# ---------------------------------------------------------------------------
# PF Efficacy Update (diminishing returns)
# ---------------------------------------------------------------------------


class TestProtectiveFactorUpdates:
    """Theory: PF efficacy updates with diminishing returns."""

    def test_higher_efficacy_smaller_increase(self):
        """Higher e_f → smaller Delta e_f per unit r_f (diminishing returns)."""
        # Same allocation amount, different starting efficacies
        pfs_low = {
            "social_support": 0.1,
            "family_support": 0.5,
            "formal_intervention": 0.5,
            "psychological_capital": 0.5,
        }
        pfs_high = {
            "social_support": 0.9,
            "family_support": 0.5,
            "formal_intervention": 0.5,
            "psychological_capital": 0.5,
        }
        state_low = _make_state(resources=1.0, protective_factors=pfs_low, affect=0.0, resilience=0.5)
        state_high = _make_state(resources=1.0, protective_factors=pfs_high, affect=0.0, resilience=0.5)
        config = _make_config()
        out_low = _run(state_low, config)
        out_high = _run(state_high, config)
        # Same total allocation for social_support across both states
        delta_low = out_low["state_delta"]["protective_factors"]["social_support"] - 0.1
        delta_high = out_high["state_delta"]["protective_factors"]["social_support"] - 0.9
        assert delta_low > delta_high

    def test_efficacy_never_exceeds_one(self):
        """e_f ∈ [0, 1] after update."""
        pfs = {f: 0.99 for f in FACTORS}
        state = _make_state(resources=1.0, protective_factors=pfs, affect=0.0, resilience=0.9)
        config = _make_config(protective_improvement_rate=0.5)
        out = _run(state, config)
        updated = out["state_delta"]["protective_factors"]
        for f in FACTORS:
            assert updated[f] <= 1.0 + 1e-10

    def test_zero_allocation_no_change(self):
        """r_f = 0 → e_f unchanged."""
        state = _make_state(resources=0.0, affect=0.0, resilience=0.5)
        config = _make_config(base_regeneration=0.0, softmax_temperature=1.0)
        out = _run(state, config)
        updated = out["state_delta"]["protective_factors"]
        for f in FACTORS:
            assert updated[f] == pytest.approx(0.5, abs=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and resilience."""

    def test_resources_clamped(self):
        """Final resources always in [0, 1]."""
        # With large regeneration, resources should not exceed 1
        state = _make_state(resources=0.9, affect=1.0, resilience=1.0)
        config = _make_config(base_regeneration=0.5, softmax_temperature=1.0)
        out = _run(state, config)
        assert 0.0 <= out["state_delta"]["resources"] <= 1.0

    def test_zero_resources_at_boundary(self):
        """When R=0 and regen=0, resources stay 0."""
        state = _make_state(resources=0.0, affect=-0.5, resilience=0.0)
        config = _make_config(base_regeneration=0.0, softmax_temperature=1.0)
        out = _run(state, config)
        assert out["state_delta"]["resources"] == 0.0

    def test_no_interaction_variables_in_state(self):
        """Phase state does NOT require interaction fields. We verify by passing minimal state."""
        minimal = AgentState(
            resources=0.5,
            affect=0.0,
            resilience=0.5,
            protective_factors={f: 0.5 for f in FACTORS},
        )
        config = _make_config()
        rng = np.random.default_rng(42)
        out = run_phase(minimal, config, rng)
        assert "state_delta" in out
        assert "observation" in out

    def test_protective_factors_all_zero_handling(self):
        """All e_f = 0 → softmax divides evenly, allocation still valid."""
        pfs = {f: 0.0 for f in FACTORS}
        state = _make_state(resources=0.5, protective_factors=pfs)
        config = _make_config(softmax_temperature=1.0, base_regeneration=0.0)
        out = _run(state, config)
        weights = out["observation"]["allocation_weights"]
        for w in weights.values():
            assert w == pytest.approx(0.25, abs=1e-10)
        allocated = out["observation"]["allocated_resources"]
        for f in FACTORS:
            assert allocated[f] == pytest.approx(0.25 * 0.5, abs=1e-10)
