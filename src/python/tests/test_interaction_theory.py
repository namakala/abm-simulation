"""Theory-grounded tests for the interaction phase.

Tests cover:
- Affect convergence: |self_A - partner_A| decreases
- Negativity bias: negative influence 1.5x stronger than positive
- No PF efficacy modification in resource-exchange scenarios
- Resource state machine: all 5 scenarios
- Support detection: threshold-based
- Edge cases: clamping, identical affect
"""

import pytest
import numpy as np

from src.python.phases.interaction import process_interaction
from src.python.phases.interfaces import AgentState

# ──────────────────────────────────────────────
# Constants matching planned hardcoded defaults
# ──────────────────────────────────────────────

INFLUENCE_RATE = 0.05
RESILIENCE_INFLUENCE = 0.05
SUPPORT_THRESHOLD = 0.05
BOOST = 0.10
COST = 0.05
SMALL_BOOST = 0.02


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def interaction_config():
    """Default interaction phase configuration."""
    return {
        "influence_rate": INFLUENCE_RATE,
        "resilience_influence": RESILIENCE_INFLUENCE,
        "support_threshold": SUPPORT_THRESHOLD,
        "boost": BOOST,
        "cost": COST,
        "small_boost": SMALL_BOOST,
    }


@pytest.fixture
def sample_rng():
    """Seeded RNG for reproducible tests."""
    return np.random.default_rng(42)


def make_state(**overrides) -> AgentState:
    """Create an AgentState with sensible defaults for interaction testing."""
    defaults = {
        "affect": 0.0,
        "resilience": 0.5,
        "resources": 0.6,
        "baseline_affect": 0.0,
        "baseline_resilience": 0.5,
        "protective_factors": {
            "social_support": 0.5,
            "family_support": 0.5,
            "formal_intervention": 0.5,
            "psychological_capital": 0.5,
        },
        "stressed": False,
        "pss10": 10,
        "current_stress": 0.3,
        "daily_interactions": 0,
        "daily_support_exchanges": 0,
        "stress_controllability": 0.5,
        "stress_overload": 0.5,
        "consecutive_hindrances": 0.0,
        "volatility": 0.3,
        "stress_config": {},
        "interaction_config": {},
    }
    defaults.update(overrides)
    return AgentState(**defaults)


# ──────────────────────────────────────────────
# Affect convergence
# ──────────────────────────────────────────────


class TestAffectConvergence:
    """Affect values move toward each other during interaction."""

    def test_self_affect_moves_toward_partner(self, interaction_config, sample_rng):
        """Self affect moves toward partner affect (positive pull)."""
        self_state = make_state(affect=0.0)
        partner_state = make_state(affect=0.8)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        # Self affect should increase (moves toward partner's 0.8)
        assert self_delta["state_delta"]["affect"] > 0

    def test_absolute_difference_decreases(self, interaction_config, sample_rng):
        """|self_A - partner_A| decreases after interaction."""
        self_state = make_state(affect=0.0)
        partner_state = make_state(affect=0.8)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        new_self = 0.0 + self_delta["state_delta"]["affect"]
        new_partner = 0.8 + partner_delta["state_delta"]["affect"]
        new_diff = abs(new_self - new_partner)

        assert new_diff < abs(0.0 - 0.8)

    def test_both_agents_converge_from_opposite_sides(self, interaction_config, sample_rng):
        """Both agents move toward each other's affect from opposite sides."""
        self_state = make_state(affect=-0.3)
        partner_state = make_state(affect=0.6)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        # Self should increase, partner should decrease
        assert self_delta["state_delta"]["affect"] > 0
        assert partner_delta["state_delta"]["affect"] < 0

    def test_resilience_also_moves_toward_partner(self, interaction_config, sample_rng):
        """Resilience is influenced by partner affect."""
        self_state = make_state(affect=0.3, resilience=0.5)
        partner_state = make_state(affect=0.0, resilience=0.5)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        # self affect is 0.3, so partner resilience gets a positive push
        # partner affect is 0.0, so self resilience gets no push
        assert partner_delta["state_delta"]["resilience"] > 0
        assert self_delta["state_delta"]["resilience"] == pytest.approx(0.0, abs=1e-10)


# ──────────────────────────────────────────────
# Negativity bias
# ──────────────────────────────────────────────


class TestNegativityBias:
    """Negative influence is 1.5x stronger than positive influence."""

    def test_negative_stronger_than_positive(self, interaction_config, sample_rng):
        """Same absolute partner affect: negative produces 1.5x stronger change."""
        # Partner with positive affect
        self_state_pos = make_state(affect=0.0)
        partner_pos = make_state(affect=0.5)
        self_delta_pos, _ = process_interaction(self_state_pos, partner_pos, interaction_config, sample_rng)

        # Partner with negative affect
        self_state_neg = make_state(affect=0.0)
        partner_neg = make_state(affect=-0.5)
        self_delta_neg, _ = process_interaction(self_state_neg, partner_neg, interaction_config, sample_rng)

        pos_change = self_delta_pos["state_delta"]["affect"]
        neg_change = self_delta_neg["state_delta"]["affect"]

        # Negative change should be 1.5x the absolute positive change
        assert abs(neg_change) == pytest.approx(abs(pos_change) * 1.5, rel=1e-10)

    def test_negativity_bias_only_for_affect_not_resilience(self, interaction_config, sample_rng):
        """Negativity bias applies to affect, not resilience."""
        self_state = make_state(affect=-0.5)
        partner_state = make_state(affect=0.5)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        # partner gets influenced by self's affect (-0.5)
        # partner_affect_change = 0.05 * (-0.5) = -0.025, with bias: -0.0375
        expected_partner_affect_change = INFLUENCE_RATE * (-0.5) * 1.5
        assert partner_delta["state_delta"]["affect"] == pytest.approx(expected_partner_affect_change, rel=1e-10)

        # partner_resilience_change = 0.05 * (-0.5) = -0.025 (no bias)
        expected_partner_resilience_change = RESILIENCE_INFLUENCE * (-0.5)
        assert partner_delta["state_delta"]["resilience"] == pytest.approx(
            expected_partner_resilience_change, rel=1e-10
        )


# ──────────────────────────────────────────────
# No PF in resource-exchange scenarios
# ──────────────────────────────────────────────


class TestNoPFEfficacyModification:
    """No protective_factors keys in resource-exchange state deltas."""

    def test_no_pf_in_stressed_stressed_delta(self, interaction_config, sample_rng):
        """(T,T,T) state_delta has no protective_factors."""
        self_state = make_state(stressed=True, affect=0.5)
        partner_state = make_state(stressed=True, affect=0.5)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        assert "protective_factors" not in self_delta["state_delta"]
        assert "protective_factors" not in partner_delta["state_delta"]

    def test_no_pf_in_stressed_not_stressed_delta(self, interaction_config, sample_rng):
        """(T,F,_) state_delta has no protective_factors."""
        self_state = make_state(stressed=True, affect=0.5)
        partner_state = make_state(stressed=False, affect=0.5)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        assert "protective_factors" not in self_delta["state_delta"]
        assert "protective_factors" not in partner_delta["state_delta"]


# ──────────────────────────────────────────────
# Resource state machine — all 5 scenarios
# ──────────────────────────────────────────────


class TestResourceStateMachine:
    """All 5 scenarios of the resource exchange state machine."""

    def test_both_stressed_support_boost(self, interaction_config, sample_rng):
        """(T,T,T) -> both resources +boost."""
        self_state = make_state(stressed=True, affect=0.5)
        partner_state = make_state(stressed=True, affect=0.5)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        assert self_delta["state_delta"]["resources"] == pytest.approx(BOOST, rel=1e-10)
        assert partner_delta["state_delta"]["resources"] == pytest.approx(BOOST, rel=1e-10)

    def test_both_stressed_no_support_cost(self, interaction_config, sample_rng):
        """(T,T,F) -> both resources -cost."""
        self_state = make_state(stressed=True, affect=0.0)
        partner_state = make_state(stressed=True, affect=0.0)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        assert self_delta["state_delta"]["resources"] == pytest.approx(-COST, rel=1e-10)
        assert partner_delta["state_delta"]["resources"] == pytest.approx(-COST, rel=1e-10)

    def test_self_stressed_partner_not_support_boost(self, interaction_config, sample_rng):
        """(T,F,T) -> self +boost, partner resources unchanged."""
        self_state = make_state(stressed=True, affect=0.5)
        partner_state = make_state(stressed=False, affect=0.5)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        # Self resources +boost (support detected)
        assert self_delta["state_delta"]["resources"] == pytest.approx(BOOST, rel=1e-10)
        # Partner resources unchanged (not in delta dict, or 0.0)
        assert partner_delta["state_delta"].get("resources", 0.0) == pytest.approx(0.0, abs=1e-10)

    def test_self_stressed_partner_not_no_support_cost(self, interaction_config, sample_rng):
        """(T,F,F) -> self -cost, partner resources unchanged."""
        self_state = make_state(stressed=True, affect=0.0)
        partner_state = make_state(stressed=False, affect=0.0)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        # Self resources -cost (no support)
        assert self_delta["state_delta"]["resources"] == pytest.approx(-COST, rel=1e-10)
        # Partner resources unchanged
        assert partner_delta["state_delta"].get("resources", 0.0) == pytest.approx(0.0, abs=1e-10)

    def test_both_not_stressed_pf_small_boost(self, interaction_config, sample_rng):
        """(F,F,_) -> both PF social_support +small_boost, resources unchanged."""
        self_state = make_state(stressed=False, affect=0.5)
        partner_state = make_state(stressed=False, affect=0.5)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        # Resources unchanged
        assert self_delta["state_delta"].get("resources", 0.0) == pytest.approx(0.0, abs=1e-10)
        assert partner_delta["state_delta"].get("resources", 0.0) == pytest.approx(0.0, abs=1e-10)

        # PF social_support got the small boost
        assert "protective_factors" in self_delta["state_delta"]
        assert "protective_factors" in partner_delta["state_delta"]
        assert self_delta["state_delta"]["protective_factors"]["social_support"] == pytest.approx(
            SMALL_BOOST, rel=1e-10
        )
        assert partner_delta["state_delta"]["protective_factors"]["social_support"] == pytest.approx(
            SMALL_BOOST, rel=1e-10
        )

    def test_both_not_stressed_other_pf_unchanged(self, interaction_config, sample_rng):
        """(F,F,_) -> only social_support PF changes, other PFs unchanged."""
        self_state = make_state(stressed=False, affect=0.5)
        partner_state = make_state(stressed=False, affect=0.5)

        self_delta, _ = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        pf_delta = self_delta["state_delta"]["protective_factors"]
        assert "social_support" in pf_delta
        # Other PF keys should NOT be in the delta (only social_support changed)
        assert "family_support" not in pf_delta
        assert "formal_intervention" not in pf_delta
        assert "psychological_capital" not in pf_delta


# ──────────────────────────────────────────────
# Support detection
# ──────────────────────────────────────────────


class TestSupportDetection:
    """Support detection from convergence magnitude."""

    def test_large_convergence_triggers_support(self, interaction_config, sample_rng):
        """> threshold total convergence -> support_occurred=True."""
        self_state = make_state(affect=0.0)
        partner_state = make_state(affect=1.0)  # Max positive affect

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        # Total convergence = affect changes + resilience changes.
        # With influence_rate=0.05, affect changes = 0.05*1.0 + 0.05*0.0 = 0.05
        # Resilience changes = 0.05*1.0 + 0.05*0.0 = 0.05
        # Total = 0.10 > 0.05
        assert self_delta["observation"]["support_occurred"] is True

    def test_small_convergence_no_support(self, interaction_config, sample_rng):
        """<= threshold total convergence -> support_occurred=False."""
        self_state = make_state(affect=0.0)
        partner_state = make_state(affect=0.01)  # Tiny affect

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        # Total convergence ≈ 4 * 0.05 * 0.01 = 0.002 < 0.05
        assert self_delta["observation"]["support_occurred"] is False

    def test_support_flag_in_both_observations(self, interaction_config, sample_rng):
        """Both agent observations contain the same support_occurred flag."""
        self_state = make_state(stressed=True, affect=0.5)
        partner_state = make_state(stressed=True, affect=0.5)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        assert "support_occurred" in self_delta["observation"]
        assert "support_occurred" in partner_delta["observation"]
        assert self_delta["observation"]["support_occurred"] == partner_delta["observation"]["support_occurred"]


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for interaction processing."""

    def test_identical_affect_no_diff_change(self, interaction_config, sample_rng):
        """Identical affect: difference stays zero, both move same direction."""
        self_state = make_state(affect=0.3)
        partner_state = make_state(affect=0.3)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        new_self = 0.3 + self_delta["state_delta"]["affect"]
        new_partner = 0.3 + partner_delta["state_delta"]["affect"]

        assert new_self == pytest.approx(new_partner, rel=1e-10)

    def test_clamp_affect_to_valid_range(self, interaction_config, sample_rng):
        """Affect stays within [-1, 1]."""
        self_state = make_state(affect=0.95)
        partner_state = make_state(affect=1.0)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        new_self_affect = 0.95 + self_delta["state_delta"]["affect"]
        new_partner_affect = 1.0 + partner_delta["state_delta"]["affect"]

        assert -1.0 <= new_self_affect <= 1.0
        assert -1.0 <= new_partner_affect <= 1.0

    def test_clamp_resilience_to_valid_range(self, interaction_config, sample_rng):
        """Resilience stays within [0, 1]."""
        self_state = make_state(resilience=0.02, affect=1.0)
        partner_state = make_state(resilience=0.02, affect=1.0)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        new_self_resilience = 0.02 + self_delta["state_delta"]["resilience"]
        new_partner_resilience = 0.02 + partner_delta["state_delta"]["resilience"]

        assert 0.0 <= new_self_resilience <= 1.0
        assert 0.0 <= new_partner_resilience <= 1.0

    def test_delta_keys_always_present(self, interaction_config, sample_rng):
        """Affect and resilience always present in both deltas."""
        self_state = make_state(affect=0.0)
        partner_state = make_state(affect=0.0)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        assert "affect" in self_delta["state_delta"]
        assert "resilience" in self_delta["state_delta"]
        assert "affect" in partner_delta["state_delta"]
        assert "resilience" in partner_delta["state_delta"]


class TestPhaseOutputContract:
    """PhaseOutput shape compliance."""

    def test_both_returns_have_state_delta_and_observation(self, interaction_config, sample_rng):
        """Both returned PhaseOutputs have state_delta and observation."""
        self_state = make_state(affect=0.5)
        partner_state = make_state(affect=0.5)

        self_delta, partner_delta = process_interaction(self_state, partner_state, interaction_config, sample_rng)

        for output in (self_delta, partner_delta):
            assert "state_delta" in output
            assert "observation" in output
            assert isinstance(output["state_delta"], dict)
            assert isinstance(output["observation"], dict)
