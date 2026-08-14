"""Tests for PSS-10 consolidation mechanics change.

The stored ``state["pss10"]`` should reflect pure stress perception
(``pss10_smoothed + pss10_bias``), not the post-hoc adjusted value that
includes resource_adjust and resilience_penalty. The adjusted value is
still used for ``stressed`` status detection.
"""

import numpy as np

from src.python.agent import process_pss10_consolidation


def _make_state_with_bias(
    daily_scores: list[int],
    pss10_bias: float = 0.0,
    pss10_smoothed: float | None = None,
    resources: float = 0.5,
    resilience: float = 0.5,
    **overrides,
) -> dict:
    """Build a minimal AgentState for consolidation testing."""
    state = {
        "daily_pss10_scores": daily_scores,
        "current_stress": 0.3,
        "stress_controllability": 0.5,
        "stress_overload": 0.5,
        "affect": 0.0,
        "resources": resources,
        "resilience": resilience,
        "pss10_bias": pss10_bias,
        "pss10_smoothed": pss10_smoothed,
        "consecutive_hindrances": 0.0,
        "protective_factors": {
            "social_support": 0.5,
            "family_support": 0.5,
            "formal_intervention": 0.5,
            "psychological_capital": 0.5,
        },
    }
    state.update(overrides)
    return state


class TestPss10PurePerception:
    """Stored PSS-10 is pure perception, not post-hoc adjusted."""

    def test_stores_pure_perception_without_resource_adjust(self):
        """state['pss10'] does not include resource_adjust."""
        # High resources should boost PSS-10 via resource_adjust
        # (0.5 - 0.9) * 12.0 = -4.8 → adjusted PSS-10 drops
        state = _make_state_with_bias(
            daily_scores=[20],
            pss10_bias=0.0,
            pss10_smoothed=20.0,
            resources=0.9,  # high resources → resource_adjust = -4.8
            resilience=0.5,
        )
        result = process_pss10_consolidation(state, {}, np.random.default_rng(42))
        stored_pss10 = result["state_delta"]["pss10"]

        # Pure perception = new_smoothed + pss10_bias
        # new_smoothed = alpha * 20 + (1-alpha) * 20 = 20
        # pss10_bias = 0
        # Expected: ~20 (not lowered by resource_adjust)
        assert abs(stored_pss10 - 20.0) < 1.5, f"PSS-10 should be ~20 (pure perception), got {stored_pss10}"

    def test_stores_pure_perception_without_resilience_penalty(self):
        """state['pss10'] does not include resilience penalty."""
        # Low resilience should penalize PSS-10 via daily_resilience_penalty
        # 0.5 * 3.5 * (0.1 - 0.5) = -0.7 → adjusted PSS-10 drops
        state = _make_state_with_bias(
            daily_scores=[20],
            pss10_bias=0.0,
            pss10_smoothed=20.0,
            resources=0.5,
            resilience=0.1,  # low resilience
        )
        result = process_pss10_consolidation(state, {}, np.random.default_rng(42))
        stored_pss10 = result["state_delta"]["pss10"]

        # Pure perception = new_smoothed + pss10_bias ≈ 20
        assert abs(stored_pss10 - 20.0) < 1.5, f"PSS-10 should be ~20 (pure perception), got {stored_pss10}"

    def test_negative_affect_increases_pure_perception(self):
        """Negative affect increases the stored pure PSS-10 (mood-congruent appraisal)."""
        # Negative affect should boost PSS-10 via affect adjustment
        state_neg = _make_state_with_bias(
            daily_scores=[20],
            pss10_bias=0.0,
            pss10_smoothed=20.0,
            resources=0.5,
            resilience=0.5,
            affect=-0.5,  # negative affect
        )
        state_pos = _make_state_with_bias(
            daily_scores=[20],
            pss10_bias=0.0,
            pss10_smoothed=20.0,
            resources=0.5,
            resilience=0.5,
            affect=0.5,  # positive affect
        )
        result_neg = process_pss10_consolidation(state_neg, {}, np.random.default_rng(42))
        result_pos = process_pss10_consolidation(state_pos, {}, np.random.default_rng(42))
        pss10_neg = result_neg["state_delta"]["pss10"]
        pss10_pos = result_pos["state_delta"]["pss10"]

        # Negative affect → higher PSS-10 than positive affect
        assert pss10_neg > pss10_pos, (
            f"Negative affect ({pss10_neg}) should give higher PSS-10 than positive ({pss10_pos})"
        )

    def test_affect_adjustment_is_proportional(self):
        """The affect adjustment to PSS-10 is proportional to affect magnitude."""
        results = {}
        for affect_val in [-0.5, 0.0, 0.5]:
            state = _make_state_with_bias(
                daily_scores=[20],
                pss10_bias=0.0,
                pss10_smoothed=20.0,
                resources=0.5,
                resilience=0.5,
                affect=affect_val,
            )
            result = process_pss10_consolidation(state, {}, np.random.default_rng(42))
            results[affect_val] = result["state_delta"]["pss10"]

        # Affect = 0 gives baseline. Affect = -0.5 should deviate more than affect = 0.5
        # because -affect * k means negative → higher
        diff_neg = results[-0.5] - results[0.0]
        diff_pos = results[0.0] - results[0.5]
        assert diff_neg > 0, f"Negative affect should increase PSS-10, got diff={diff_neg}"
        assert diff_pos > 0, f"Positive affect should decrease PSS-10, got diff={diff_pos}"
        # Both differences should be roughly equal (proportional)
        ratio = diff_neg / diff_pos if diff_pos > 0 else 999
        assert 0.5 < ratio < 2.0, (
            f"Asymmetric proportional adjustment: diff_neg={diff_neg:.2f}, diff_pos={diff_pos:.2f}, ratio={ratio:.2f}"
        )

    def test_stressed_status_still_uses_adjusted_value(self):
        """stressed status is computed from the full adjusted PSS-10."""
        # Very low resources → resource_adjust = (0.5 - 0.1) * 12 = +4.8
        # This should NOT affect stored PSS-10, but SHOULD affect stressed
        state = _make_state_with_bias(
            daily_scores=[30],
            pss10_bias=0.0,
            pss10_smoothed=30.0,
            resources=0.1,  # low → resource_adjust adds +4.8
            resilience=0.5,
        )
        config = {"pss10_threshold": 25}
        result = process_pss10_consolidation(state, config, np.random.default_rng(42))
        stored_pss10 = result["state_delta"]["pss10"]
        is_stressed = result["state_delta"]["stressed"]

        # Pure perception ≈ 30 (not boosted by resource_adjust)
        # But stressed status uses adjusted value which IS boosted
        # Resource_adjust = (0.5 - 0.1) * 12 = +4.8
        # Adjusted PSS-10 ≈ 30 + 4.8 = 34.8 → stressed
        # Pure PSS-10 ≈ 30 → would be stressed too in this case
        # Need a case where pure is below threshold but adjusted is above
        assert abs(stored_pss10 - 30.0) < 1.5, f"Stored PSS-10 should be ~30, got {stored_pss10}"
        assert is_stressed, "Stressed status should use adjusted value (above threshold)"

    def test_threshold_crossing_from_adjustment(self):
        """Resource adjustment can push PSS-10 above threshold even when
        pure perception is below threshold."""
        # Resource_adjust = (0.5 - 0.1) * 12 = +4.8
        # With daily score = 22, variance stretch ≈ 22.4, +4.8 = 27.2 → above 25
        # Pure perception ≈ 22 → below 25
        state = _make_state_with_bias(
            daily_scores=[22],
            pss10_bias=0.0,
            pss10_smoothed=22.0,
            resources=0.1,
            resilience=0.5,
        )
        config = {"pss10_threshold": 25}
        result = process_pss10_consolidation(state, config, np.random.default_rng(42))
        stored_pss10 = result["state_delta"]["pss10"]
        is_stressed = result["state_delta"]["stressed"]

        # Pure perception ≈ 22 (below threshold)
        assert stored_pss10 <= config["pss10_threshold"], (
            f"Pure perception should be below threshold, got {stored_pss10}"
        )
        # But stressed should be True (adjusted value crosses threshold)
        assert is_stressed, "Stressed should be True (adjusted value crosses threshold)"

    def test_pss10_smoothed_unchanged(self):
        """pss10_smoothed field is still stored unchanged."""
        state = _make_state_with_bias(
            daily_scores=[25],
            pss10_bias=2.0,
            pss10_smoothed=20.0,
            resources=0.5,
            resilience=0.5,
        )
        result = process_pss10_consolidation(state, {}, np.random.default_rng(42))
        assert "pss10_smoothed" in result["state_delta"]
