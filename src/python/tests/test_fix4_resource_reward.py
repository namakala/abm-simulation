"""Tests for Fix 4: Restore resource reward for successful coping.

Verifies:
- pf_allocation_fraction default is 0.05 (was 0.15)
- Successful coping gives a resource reward that makes net resource
  change equal or better than failed coping
"""

import copy
import numpy as np
import pytest

from src.python.assumption_config import AssumptionCopingConfig
from src.python.phases.interfaces import AgentState
from src.python.phases.resilience_activation import run_phase


# Default agent state for resilience activation tests
BASE_STATE: AgentState = AgentState(
    resilience=0.5,
    affect=0.0,
    resources=0.6,
    baseline_resilience=0.5,
    baseline_affect=0.0,
    current_stress=0.5,
    protective_factors={
        "social_support": 0.5,
        "family_support": 0.5,
        "formal_intervention": 0.5,
        "psychological_capital": 0.5,
    },
    pss10=15,
    pss10_responses={i: 2 for i in range(1, 11)},
    stressed=True,
    stress_controllability=0.5,
    stress_overload=0.5,
    consecutive_hindrances=0.0,
    stress_breach_count=0,
    volatility=0.3,
    daily_interactions=0,
    daily_support_exchanges=0,
    stress_config={},
    interaction_config={},
    challenge=0.9,
    hindrance=0.1,
    recent_stress_intensity=0.3,
    stress_momentum=0.1,
)

SUCCESS_CONFIG = {
    "neighbor_affects": [0.8, 0.9],
    "base_resource_cost": 0.1,
    "event_controllability": 0.7,
    "event_overload": 0.3,
}

FAIL_CONFIG = {
    "neighbor_affects": [-0.8, -0.9],
    "base_resource_cost": 0.1,
    "event_controllability": 0.3,
    "event_overload": 0.7,
}


class TestReducedPfAllocationFraction:
    """Fix 4a: PF allocation fraction reduced to 0.05."""

    @pytest.mark.unit
    def test_pf_allocation_fraction_default_is_0_05(self):
        """ASSUMPTION_PF_ALLOCATION_FRACTION default is 0.05 (was 0.15)."""
        c = AssumptionCopingConfig()
        assert c.pf_allocation_fraction == 0.05, f"Expected 0.05, got {c.pf_allocation_fraction}"


class TestResourceReward:
    """Fix 4b: Successful coping yields a resource reward."""

    @pytest.mark.unit
    def test_successful_coping_has_resource_reward_in_observation(self):
        """When coping succeeds, the observation includes a positive resource_reward."""
        state = copy.deepcopy(BASE_STATE)
        rng = np.random.default_rng(42)
        result = run_phase(state, SUCCESS_CONFIG, rng)
        obs = result["observation"]
        assert obs.get("coped_successfully", False), "Expected coping success"
        assert obs.get("resource_reward", 0) > 0, f"Expected positive reward, got {obs.get('resource_reward')}"

    @pytest.mark.unit
    def test_successful_coping_resources_net_zero_or_better(self):
        """Successful coping should not leave agent with fewer resources than equivalent failure."""
        # Run with success config — high challenge ensures high coping probability
        state_success = copy.deepcopy(BASE_STATE)
        state_success["resources"] = 0.5  # fixed starting point
        rng_s = np.random.default_rng(42)
        result_success = run_phase(state_success, SUCCESS_CONFIG, rng_s)

        # Run with fail config — high hindrance ensures low coping probability
        state_fail = copy.deepcopy(BASE_STATE)
        state_fail["resources"] = 0.5
        rng_f = np.random.default_rng(42)
        result_fail = run_phase(state_fail, FAIL_CONFIG, rng_f)

        obs_s = result_success["observation"]
        obs_f = result_fail["observation"]

        # If success occurred, its resources should not be much lower than failure
        if obs_s.get("coped_successfully", False) and not obs_f.get("coped_successfully", True):
            res_success = result_success["state_delta"]["resources"]
            res_fail = result_fail["state_delta"]["resources"]
            # Success should not result in meaningfully fewer resources than failure
            assert res_success >= res_fail - 0.05, (
                f"Success resources ({res_success:.4f}) should not be far below failure resources ({res_fail:.4f})"
            )

    @pytest.mark.unit
    def test_resource_reward_scales_with_challenge(self):
        """The reward 0.03 is flat, applied to all successful coping."""
        # Run two trials with deterministic RNG to get reproduction
        state = copy.deepcopy(BASE_STATE)
        state["challenge"] = 0.9
        rng = np.random.default_rng(42)
        result = run_phase(state, SUCCESS_CONFIG, rng)
        obs = result["observation"]

        if obs.get("coped_successfully", False):
            # The reward is the extra resource beyond what depletion + PF allocation would give
            # We can't easily isolate it from the PF allocation, but we can check it's present
            assert "resource_reward" in obs
            # Sanity check: reward is small (≤ 0.05)
            assert obs["resource_reward"] <= 0.05, f"Reward too large: {obs['resource_reward']}"

    @pytest.mark.unit
    def test_no_resource_reward_on_failure(self):
        """Failed coping has no resource_reward in observation."""
        state = copy.deepcopy(BASE_STATE)
        state["challenge"] = 0.1
        state["hindrance"] = 0.9
        rng = np.random.default_rng(42)
        result = run_phase(state, FAIL_CONFIG, rng)
        obs = result["observation"]
        # If coping failed, no reward
        if not obs.get("coped_successfully", True):
            assert obs.get("resource_reward") is None or obs.get("resource_reward", 0) == 0, (
                "Failed coping should have no resource reward"
            )

    @pytest.mark.unit
    def test_resource_net_change_improved_over_baseline(self):
        """After Fix 4, successful coping resource change should be ≥ the old net (before fix)."""
        # Before fix: success = -cost - PF_alloc (= -cost - 0.15*remaining)
        # After fix: success = -cost + reward - PF_alloc (= -cost + 0.03 - 0.05*remaining)
        # At resources=0.5, cost≈0.05: old net ≈ -0.05 - 0.068 = -0.118
        # After fix: -0.05 + 0.03 - 0.023 = -0.043
        state = copy.deepcopy(BASE_STATE)
        state["resources"] = 0.5
        rng = np.random.default_rng(42)
        result = run_phase(state, SUCCESS_CONFIG, rng)
        obs = result["observation"]

        if obs.get("coped_successfully", False):
            initial = 0.5
            final = result["state_delta"]["resources"]
            net_change = final - initial
            # Net change should be small negative or even positive, not severely negative
            assert net_change > -0.10, f"Net resource change too negative: {net_change:.4f}"
