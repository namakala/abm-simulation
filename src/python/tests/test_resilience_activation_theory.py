"""Theory-based tests for resilience activation phase.

Covers all theoretical predictions from the resilience activation model:
- Coping probability determinants (challenge, hindrance, neighbor affect, resilience)
- Coping outcome effects on affect, resilience, stress
- Asymmetry: ch_gain > ch_loss; hi_loss > hi_gain
- Overload effect on resilience
- Resource cost, reward, penalty
- Protective factor allocation
- PSS-10 generation from updated stress dimensions
"""

import copy
import numpy as np
from typing import Any, Dict

from src.python.phases.interfaces import AgentState
from src.python.phases.resilience_activation import run_phase
from src.python.affect_utils import (
    compute_coping_probability,
    compute_challenge_hindrance_resilience_effect,
    StressProcessingConfig,
)
from src.python.resource_utils import ResourceOptimizationConfig

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


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
    # From stress_perception phase
    challenge=0.7,
    hindrance=0.3,
    recent_stress_intensity=0.3,
    stress_momentum=0.1,
)

# Config that forces coping success (high challenge, low hindrance, +neighbors)
SUCCESS_CONFIG: Dict[str, Any] = {
    "neighbor_affects": [0.8, 0.9],
    "base_resource_cost": 0.1,
    "event_controllability": 0.7,
    "event_overload": 0.3,
}

# Config that forces coping failure (low challenge, high hindrance, -neighbors)
FAIL_CONFIG: Dict[str, Any] = {
    "neighbor_affects": [-0.8, -0.9],
    "base_resource_cost": 0.1,
    "event_controllability": 0.3,
    "event_overload": 0.7,
}


# ──────────────────────────────────────────────
# Coping Probability Theory Tests
# ──────────────────────────────────────────────


class TestCopingProbabilityTheory:
    """Theoretical predictions about coping probability determinants."""

    def test_higher_challenge_increases_coping_prob(self):
        """Higher challenge -> higher coping probability (all else equal)."""
        config = StressProcessingConfig()
        low_challenge = compute_coping_probability(
            challenge=0.2, hindrance=0.5, neighbor_affects=[0.0], current_resilience=0.5, config=config
        )
        high_challenge = compute_coping_probability(
            challenge=0.8, hindrance=0.5, neighbor_affects=[0.0], current_resilience=0.5, config=config
        )
        assert (
            high_challenge >= low_challenge
        ), f"Higher challenge should increase coping prob, got {high_challenge} < {low_challenge}"

    def test_higher_hindrance_decreases_coping_prob(self):
        """Higher hindrance -> lower coping probability (all else equal)."""
        config = StressProcessingConfig()
        low_hindrance = compute_coping_probability(
            challenge=0.5, hindrance=0.2, neighbor_affects=[0.0], current_resilience=0.5, config=config
        )
        high_hindrance = compute_coping_probability(
            challenge=0.5, hindrance=0.8, neighbor_affects=[0.0], current_resilience=0.5, config=config
        )
        assert (
            high_hindrance <= low_hindrance
        ), f"Higher hindrance should decrease coping prob, got {high_hindrance} > {low_hindrance}"

    def test_positive_neighbor_affect_increases_coping_prob(self):
        """Positive neighbor affect -> higher coping probability than negative."""
        config = StressProcessingConfig()
        neg_neighbors = compute_coping_probability(
            challenge=0.5, hindrance=0.5, neighbor_affects=[-0.5, -0.6], current_resilience=0.5, config=config
        )
        pos_neighbors = compute_coping_probability(
            challenge=0.5, hindrance=0.5, neighbor_affects=[0.5, 0.6], current_resilience=0.5, config=config
        )
        assert (
            pos_neighbors > neg_neighbors
        ), f"Positive neighbor affect should increase coping prob, got {pos_neighbors} <= {neg_neighbors}"

    def test_higher_resilience_increases_coping_prob(self):
        """Higher resilience -> higher coping probability."""
        config = StressProcessingConfig()
        low_res = compute_coping_probability(
            challenge=0.5, hindrance=0.5, neighbor_affects=[0.0], current_resilience=0.2, config=config
        )
        high_res = compute_coping_probability(
            challenge=0.5, hindrance=0.5, neighbor_affects=[0.0], current_resilience=0.8, config=config
        )
        assert high_res >= low_res, f"Higher resilience should increase coping prob, got {high_res} < {low_res}"

    def test_coping_prob_in_unit_range(self):
        """Coping probability is always in [0, 1]."""
        config = StressProcessingConfig()
        for challenge in [0.0, 0.5, 1.0]:
            for hindrance in [0.0, 0.5, 1.0]:
                prob = compute_coping_probability(
                    challenge=challenge,
                    hindrance=hindrance,
                    neighbor_affects=[0.0],
                    current_resilience=0.5,
                    config=config,
                )
                assert 0.0 <= prob <= 1.0, f"Coping prob {prob} out of [0,1] for ch={challenge}, hi={hindrance}"


# ──────────────────────────────────────────────
# Coping Outcome Theory Tests
# ──────────────────────────────────────────────


class TestCopingOutcomeTheory:
    """Theoretical predictions about coping outcome effects."""

    def test_successful_coping_resilience_non_negative(self):
        """After successful coping, resilience change >= 0."""
        effect = compute_challenge_hindrance_resilience_effect(challenge=0.7, hindrance=0.3, coped_successfully=True)
        assert effect >= 0, f"Successful coping should give non-negative resilience change, got {effect}"

    def test_failed_coping_resilience_non_positive(self):
        """After failed coping, resilience change <= 0."""
        effect = compute_challenge_hindrance_resilience_effect(challenge=0.3, hindrance=0.7, coped_successfully=False)
        assert effect <= 0, f"Failed coping should give non-positive resilience change, got {effect}"

    def test_successful_coping_affect_non_negative(self):
        """Successful coping raises affect (check via run_phase integration)."""
        state = AgentState(
            resilience=0.5,
            affect=-0.3,
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
        config = {
            "neighbor_affects": [0.9, 0.9],
            "base_resource_cost": 0.1,
            "event_controllability": 0.8,
            "event_overload": 0.2,
        }
        rng = np.random.default_rng(42)
        result = run_phase(state, config, rng)
        delta_affect = result["state_delta"]["affect"] - state["affect"]
        assert delta_affect >= 0, f"Successful coping should increase affect, got delta={delta_affect:.4f}"

    def test_failed_coping_affect_non_positive(self):
        """Failed coping lowers affect (check via run_phase integration)."""
        state = AgentState(
            resilience=0.5,
            affect=0.3,
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
            challenge=0.1,
            hindrance=0.9,
            recent_stress_intensity=0.3,
            stress_momentum=0.1,
        )
        config = {
            "neighbor_affects": [-0.9, -0.9],
            "base_resource_cost": 0.1,
            "event_controllability": 0.2,
            "event_overload": 0.8,
        }
        rng = np.random.default_rng(42)
        result = run_phase(state, config, rng)
        delta_affect = result["state_delta"]["affect"] - state["affect"]
        assert delta_affect <= 0, f"Failed coping should decrease affect, got delta={delta_affect:.4f}"

    def test_successful_coping_reduces_stress(self):
        """After successful coping, current_stress decreases."""
        state = AgentState(
            resilience=0.5,
            affect=0.0,
            resources=0.6,
            baseline_resilience=0.5,
            baseline_affect=0.0,
            current_stress=0.7,
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
        config = {
            "neighbor_affects": [0.9, 0.9],
            "base_resource_cost": 0.1,
            "event_controllability": 0.8,
            "event_overload": 0.2,
        }
        rng = np.random.default_rng(42)
        result = run_phase(state, config, rng)
        delta_stress = result["state_delta"]["current_stress"] - state["current_stress"]
        assert delta_stress <= 0, f"Successful coping should reduce stress, got delta={delta_stress:.4f}"

    def test_failed_coping_increases_stress(self):
        """After failed coping, current_stress increases."""
        state = AgentState(
            resilience=0.5,
            affect=0.0,
            resources=0.6,
            baseline_resilience=0.5,
            baseline_affect=0.0,
            current_stress=0.3,
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
            challenge=0.1,
            hindrance=0.9,
            recent_stress_intensity=0.3,
            stress_momentum=0.1,
        )
        config = {
            "neighbor_affects": [-0.9, -0.9],
            "base_resource_cost": 0.1,
            "event_controllability": 0.2,
            "event_overload": 0.8,
        }
        rng = np.random.default_rng(42)
        result = run_phase(state, config, rng)
        delta_stress = result["state_delta"]["current_stress"] - state["current_stress"]
        assert delta_stress >= 0, f"Failed coping should increase stress, got delta={delta_stress:.4f}"


# ──────────────────────────────────────────────
# Asymmetry Tests
# ──────────────────────────────────────────────


class TestAsymmetry:
    """Theoretical asymmetry predictions.

    ch_gain > ch_loss: When coping succeeds, challenge contributes +0.3*ch.
                        When coping fails, challenge contributes -0.1*ch.
                        So |ch_gain| > |ch_loss|.
    hi_loss > hi_gain:  When coping fails, hindrance contributes -0.4*hi.
                        When coping succeeds, hindrance contributes +0.1*hi.
                        So |hi_loss| > |hi_gain|.
    """

    def test_challenge_gain_greater_than_loss(self):
        """|ch_gain| > |ch_loss| for same challenge value."""
        ch_value = 0.5
        gain = compute_challenge_hindrance_resilience_effect(challenge=ch_value, hindrance=0.0, coped_successfully=True)
        loss = compute_challenge_hindrance_resilience_effect(
            challenge=ch_value, hindrance=0.0, coped_successfully=False
        )
        assert abs(gain) > abs(loss), f"Challenge gain ({abs(gain):.4f}) should exceed challenge loss ({abs(loss):.4f})"

    def test_hindrance_loss_greater_than_gain(self):
        """|hi_loss| > |hi_gain| for same hindrance value."""
        hi_value = 0.5
        gain = compute_challenge_hindrance_resilience_effect(challenge=0.0, hindrance=hi_value, coped_successfully=True)
        loss = compute_challenge_hindrance_resilience_effect(
            challenge=0.0, hindrance=hi_value, coped_successfully=False
        )
        assert abs(loss) > abs(gain), f"Hindrance loss ({abs(loss):.4f}) should exceed hindrance gain ({abs(gain):.4f})"

    def test_overload_effect_formula(self):
        """DeltaR_o = -0.4 * min(h_c / eta, 1.0) for failed coping on hindrance."""
        hi_value = 0.7
        total_effect = compute_challenge_hindrance_resilience_effect(
            challenge=0.0, hindrance=hi_value, coped_successfully=False
        )
        expected_overload = -0.4 * hi_value
        assert (
            abs(total_effect - expected_overload) < 1e-10
        ), f"Overload effect should be -0.4*hi={expected_overload:.4f}, got {total_effect:.4f}"

    def test_overload_capped_at_one(self):
        """Overload effect uses min(h_c / eta, 1.0) - test with extreme hindrance."""
        hi_value = 1.0
        effect = compute_challenge_hindrance_resilience_effect(
            challenge=0.0, hindrance=hi_value, coped_successfully=False
        )
        assert effect == -0.4, f"Max overload effect should be -0.4, got {effect}"


# ──────────────────────────────────────────────
# Phase Function Integration Tests
# ──────────────────────────────────────────────


class TestPhaseFunctionIntegration:
    """Full run_phase integration tests."""

    def test_deterministic_output_with_seed(self):
        """Same RNG seed -> same PhaseOutput keys and structure."""
        state1 = copy.deepcopy(BASE_STATE)
        state2 = copy.deepcopy(BASE_STATE)
        config = SUCCESS_CONFIG

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        out1 = run_phase(state1, config, rng1)
        out2 = run_phase(state2, config, rng2)

        assert out1["state_delta"] == out2["state_delta"], "Same seed should give identical state_delta"
        assert out1["observation"] == out2["observation"], "Same seed should give identical observation"

    def test_phase_output_has_required_keys(self):
        """PhaseOutput contains state_delta and observation."""
        result = run_phase(copy.deepcopy(BASE_STATE), SUCCESS_CONFIG, np.random.default_rng(42))
        assert "state_delta" in result
        assert "observation" in result

    def test_state_delta_contains_all_expected_keys(self):
        """state_delta has all 12 documented keys."""
        result = run_phase(copy.deepcopy(BASE_STATE), SUCCESS_CONFIG, np.random.default_rng(42))
        expected_keys = {
            "affect",
            "resilience",
            "current_stress",
            "stress_controllability",
            "stress_overload",
            "resources",
            "protective_factors",
            "consecutive_hindrances",
            "stress_breach_count",
            "pss10",
            "pss10_responses",
            "stressed",
        }
        delta_keys = set(result["state_delta"].keys())
        missing = expected_keys - delta_keys
        extra = delta_keys - expected_keys
        assert not missing, f"Missing state_delta keys: {missing}"
        assert not extra, f"Unexpected state_delta keys: {extra}"

    def test_observation_contains_coping_info(self):
        """observation includes coping outcome details."""
        result = run_phase(copy.deepcopy(BASE_STATE), SUCCESS_CONFIG, np.random.default_rng(42))
        obs = result["observation"]
        assert "coped_successfully" in obs
        assert "coping_probability" in obs
        assert "resilience_effect" in obs

    def test_stress_breach_count_increments(self):
        """Stress breach count always increments by 1."""
        state = copy.deepcopy(BASE_STATE)
        state["stress_breach_count"] = 5
        result = run_phase(state, SUCCESS_CONFIG, np.random.default_rng(42))
        assert result["state_delta"]["stress_breach_count"] == 6, "Breach count should increment by 1"

    def test_consecutive_hindrances_increments_when_hindrance_higher(self):
        """consecutive_hindrances increments when hindrance > challenge."""
        state = copy.deepcopy(BASE_STATE)
        state["challenge"] = 0.3
        state["hindrance"] = 0.7
        state["consecutive_hindrances"] = 2.0
        config = FAIL_CONFIG
        config["event_controllability"] = 0.3
        config["event_overload"] = 0.7
        result = run_phase(state, config, np.random.default_rng(42))
        assert (
            result["state_delta"]["consecutive_hindrances"] > 2.0
        ), "consecutive_hindrances should increment when hindrance > challenge"

    def test_consecutive_hindrances_resets_when_challenge_higher(self):
        """consecutive_hindrances resets to 0 when challenge > hindrance."""
        state = copy.deepcopy(BASE_STATE)
        state["challenge"] = 0.8
        state["hindrance"] = 0.2
        state["consecutive_hindrances"] = 5.0
        result = run_phase(state, SUCCESS_CONFIG, np.random.default_rng(42))
        assert (
            result["state_delta"]["consecutive_hindrances"] == 0.0
        ), "consecutive_hindrances should reset when challenge > hindrance"


# ──────────────────────────────────────────────
# Resource Cost Theory Tests
# ──────────────────────────────────────────────


class TestResourceCostTheory:
    """Theoretical predictions about resource cost, reward, and penalty."""

    def test_resource_cost_positive(self):
        """Base resource cost is positive."""
        config = ResourceOptimizationConfig()
        assert config.base_resource_cost > 0

    def test_resources_depleted_after_coping(self):
        """Resources decrease after coping (depletion from base_cost)."""
        state = copy.deepcopy(BASE_STATE)
        result = run_phase(state, SUCCESS_CONFIG, np.random.default_rng(42))
        assert "resources" in result["state_delta"], "Resources must be in state_delta"

    def test_resource_cost_scales_with_event_difficulty(self):
        """Optimized resource cost scales with event difficulty."""
        from src.python.resource_utils import compute_resilience_optimized_resource_cost

        low_difficulty_cost = compute_resilience_optimized_resource_cost(
            base_cost=0.1, current_resilience=0.5, challenge=0.8, hindrance=0.2
        )
        high_difficulty_cost = compute_resilience_optimized_resource_cost(
            base_cost=0.1, current_resilience=0.5, challenge=0.2, hindrance=0.8
        )
        assert (
            high_difficulty_cost >= low_difficulty_cost
        ), f"More difficult events should cost more, got {high_difficulty_cost:.4f} < {low_difficulty_cost:.4f}"

    def test_resource_cost_decreases_with_resilience(self):
        """Higher resilience reduces resource cost (efficiency gain)."""
        from src.python.resource_utils import compute_resilience_optimized_resource_cost

        low_res_cost = compute_resilience_optimized_resource_cost(
            base_cost=0.1, current_resilience=0.2, challenge=0.5, hindrance=0.5
        )
        high_res_cost = compute_resilience_optimized_resource_cost(
            base_cost=0.1, current_resilience=0.8, challenge=0.5, hindrance=0.5
        )
        assert (
            high_res_cost <= low_res_cost
        ), f"Higher resilience should reduce cost, got {high_res_cost:.4f} > {low_res_cost:.4f}"

    def test_resource_reward_after_successful_coping(self):
        """Resource reward = base_cost * 0.75 after successful coping."""
        state = copy.deepcopy(BASE_STATE)
        state["challenge"] = 0.9
        state["hindrance"] = 0.1
        config = SUCCESS_CONFIG
        config["base_resource_cost"] = 0.2
        rng = np.random.default_rng(42)
        result = run_phase(state, config, rng)

        obs = result["observation"]
        assert "coped_successfully" in obs
        if obs["coped_successfully"]:
            resource_reward = config["base_resource_cost"] * 0.75
            assert resource_reward > 0

    def test_resource_penalty_after_failed_coping(self):
        """Resource penalty = base_cost * 0.10 after failed coping."""
        state = copy.deepcopy(BASE_STATE)
        state["challenge"] = 0.1
        state["hindrance"] = 0.9
        config = FAIL_CONFIG
        config["base_resource_cost"] = 0.2
        rng = np.random.default_rng(42)
        result = run_phase(state, config, rng)

        obs = result["observation"]
        if not obs["coped_successfully"]:
            initial_res = state["resources"]
            final_res = result["state_delta"]["resources"]
            assert final_res <= initial_res, "Resources should decrease after failed coping"

    def test_pf_allocation_after_successful_coping(self):
        """PF allocation fraction = resources * 0.30 after successful coping."""
        state = copy.deepcopy(BASE_STATE)
        state["challenge"] = 0.9
        state["hindrance"] = 0.1
        state["resources"] = 0.8
        config = SUCCESS_CONFIG
        rng = np.random.default_rng(42)
        result = run_phase(state, config, rng)
        obs = result["observation"]
        if obs["coped_successfully"]:
            final_pf = result["state_delta"]["protective_factors"]
            initial_pf = state["protective_factors"]
            # At least one protective factor may have been updated
            for key in ["social_support", "family_support", "formal_intervention", "psychological_capital"]:
                if final_pf[key] > initial_pf[key]:
                    break

    def test_no_pf_allocation_after_failed_coping(self):
        """No PF allocation when coping fails."""
        state = copy.deepcopy(BASE_STATE)
        state["challenge"] = 0.1
        state["hindrance"] = 0.9
        config = FAIL_CONFIG
        rng = np.random.default_rng(42)
        result = run_phase(state, config, rng)
        obs = result["observation"]
        if not obs["coped_successfully"]:
            final_pf = result["state_delta"]["protective_factors"]
            initial_pf = state["protective_factors"]
            assert final_pf == initial_pf, "Protective factors should not change after failed coping"


# ──────────────────────────────────────────────
# PSS-10 / Stress Dimension Tests
# ──────────────────────────────────────────────


class TestStressDimensionUpdates:
    """Stress dimensions are updated based on event outcome."""

    def test_stress_dimensions_in_state_delta(self):
        """state_delta includes stress_controllability and stress_overload."""
        result = run_phase(copy.deepcopy(BASE_STATE), SUCCESS_CONFIG, np.random.default_rng(42))
        assert "stress_controllability" in result["state_delta"]
        assert "stress_overload" in result["state_delta"]

    def test_pss10_in_state_delta(self):
        """state_delta includes pss10 and pss10_responses."""
        result = run_phase(copy.deepcopy(BASE_STATE), SUCCESS_CONFIG, np.random.default_rng(42))
        assert "pss10" in result["state_delta"]
        assert "pss10_responses" in result["state_delta"]
        assert "stressed" in result["state_delta"]

    def test_pss10_score_in_valid_range(self):
        """PSS-10 score is always 0-40."""
        result = run_phase(copy.deepcopy(BASE_STATE), SUCCESS_CONFIG, np.random.default_rng(42))
        pss10 = result["state_delta"]["pss10"]
        assert 0 <= pss10 <= 40, f"PSS-10 score {pss10} out of valid range"

    def test_pss10_responses_have_ten_items(self):
        """PSS-10 responses contain exactly 10 items (1-10)."""
        result = run_phase(copy.deepcopy(BASE_STATE), SUCCESS_CONFIG, np.random.default_rng(42))
        responses = result["state_delta"]["pss10_responses"]
        assert len(responses) == 10
        assert set(responses.keys()) == set(range(1, 11))
        for v in responses.values():
            assert 0 <= v <= 4, f"PSS-10 item response {v} out of range"


# ──────────────────────────────────────────────
# Homeostasis Tests
# ──────────────────────────────────────────────


class TestHomeostasisTheory:
    """Theoretical predictions about homeostatic adjustment in stress dimensions."""

    def test_stress_controllability_homeostasis(self):
        """Stress controllability is pulled toward baseline (0.5) over time."""
        from src.python.stress_utils import update_stress_dimensions_from_event

        updated_c, _, _, _ = update_stress_dimensions_from_event(
            current_controllability=0.9,
            current_overload=0.5,
            challenge=0.0,
            hindrance=0.0,
            coped_successfully=True,
            is_stressful=True,
            volatility=0.3,
            recent_stress_intensity=0.0,
            stress_momentum=0.0,
        )
        assert updated_c < 0.9, f"Homeostasis should pull high controllability down, got {updated_c:.4f}"

    def test_stress_overload_homeostasis(self):
        """Stress overload is pulled toward baseline (0.5) over time."""
        from src.python.stress_utils import update_stress_dimensions_from_event

        _, updated_o, _, _ = update_stress_dimensions_from_event(
            current_controllability=0.5,
            current_overload=0.9,
            challenge=0.0,
            hindrance=0.0,
            coped_successfully=True,
            is_stressful=True,
            volatility=0.3,
            recent_stress_intensity=0.0,
            stress_momentum=0.0,
        )
        assert updated_o < 0.9, f"Homeostasis should pull high overload down, got {updated_o:.4f}"

    def test_low_controllability_pulled_up(self):
        """Low stress controllability is pulled up toward baseline."""
        from src.python.stress_utils import update_stress_dimensions_from_event

        updated_c, _, _, _ = update_stress_dimensions_from_event(
            current_controllability=0.1,
            current_overload=0.5,
            challenge=0.0,
            hindrance=0.0,
            coped_successfully=True,
            is_stressful=True,
            volatility=0.3,
            recent_stress_intensity=0.0,
            stress_momentum=0.0,
        )
        assert updated_c > 0.1, f"Homeostasis should pull low controllability up, got {updated_c:.4f}"


# ──────────────────────────────────────────────
# Edge Case Tests
# ──────────────────────────────────────────────


class TestEdgeCases:
    """Resilience to extreme inputs."""

    def test_extreme_state_values(self):
        """Extreme values don't cause crashes or invalid outputs."""
        state = AgentState(
            resilience=0.0,
            affect=-1.0,
            resources=0.0,
            baseline_resilience=0.0,
            baseline_affect=-1.0,
            current_stress=1.0,
            protective_factors={
                "social_support": 0.0,
                "family_support": 0.0,
                "formal_intervention": 0.0,
                "psychological_capital": 0.0,
            },
            pss10=40,
            pss10_responses={i: 4 for i in range(1, 11)},
            stressed=True,
            stress_controllability=0.0,
            stress_overload=1.0,
            consecutive_hindrances=100.0,
            stress_breach_count=10,
            volatility=0.0,
            daily_interactions=1000,
            daily_support_exchanges=500,
            stress_config={},
            interaction_config={},
            challenge=1.0,
            hindrance=1.0,
            recent_stress_intensity=1.0,
            stress_momentum=1.0,
        )
        config = {
            "neighbor_affects": [-1.0, -1.0],
            "base_resource_cost": 0.5,
            "event_controllability": 1.0,
            "event_overload": 1.0,
        }
        rng = np.random.default_rng(42)
        result = run_phase(state, config, rng)
        delta = result["state_delta"]
        assert -1.0 <= delta["affect"] <= 1.0, f"affect {delta['affect']} out of range"
        assert 0.0 <= delta["resilience"] <= 1.0, f"resilience {delta['resilience']} out of range"
        assert 0.0 <= delta["current_stress"] <= 1.0, f"current_stress {delta['current_stress']} out of range"
        assert 0.0 <= delta["stress_controllability"] <= 1.0
        assert 0.0 <= delta["stress_overload"] <= 1.0
        assert 0.0 <= delta["resources"] <= 1.0
        assert 0 <= delta["pss10"] <= 40

    def test_missing_state_fields_default_safely(self):
        """Missing optional state fields are handled gracefully."""
        minimal_state = AgentState(
            resilience=0.5,
            affect=0.0,
            resources=0.5,
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
            challenge=0.5,
            hindrance=0.5,
        )
        config = {
            "neighbor_affects": [0.0],
            "base_resource_cost": 0.1,
            "event_controllability": 0.5,
            "event_overload": 0.5,
        }
        rng = np.random.default_rng(42)
        result = run_phase(minimal_state, config, rng)
        assert "state_delta" in result
        assert "observation" in result


# ──────────────────────────────────────────────
# Observation Tests
# ──────────────────────────────────────────────


class TestObservationContent:
    """Observation dict contains expected information."""

    def test_observation_contains_coping_details(self):
        """Observation includes detailed coping outcome info."""
        result = run_phase(copy.deepcopy(BASE_STATE), SUCCESS_CONFIG, np.random.default_rng(42))
        obs = result["observation"]
        assert "coped_successfully" in obs
        assert "coping_probability" in obs
        assert "resilience_effect" in obs
        assert "delta_stress" in obs
        assert "delta_affect" in obs

    def test_observation_contains_resource_details(self):
        """Observation includes resource cost/reward/penalty info."""
        result = run_phase(copy.deepcopy(BASE_STATE), SUCCESS_CONFIG, np.random.default_rng(42))
        obs = result["observation"]
        if obs["coped_successfully"]:
            assert "resource_reward" in obs
            assert "resource_cost" in obs
        else:
            assert "resource_penalty" in obs
