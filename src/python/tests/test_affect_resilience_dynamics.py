"""
Comprehensive unit tests for affect and resilience dynamics functions.

This file tests the new enhanced affect and resilience dynamics functions:
- compute_peer_influence() - peer influence on affect
- compute_event_appraisal_effect() - challenge/hindrance → affect mapping
- compute_homeostasis_effect() - baseline reversion behavior
- compute_cumulative_overload() - consecutive hindrance tracking
- update_affect_dynamics() - complete affect update integration
- update_resilience_dynamics() - complete resilience update integration

Tests cover edge cases, typical scenarios, configuration variations,
mathematical correctness, and proper normalization.
"""

import pytest
import numpy as np
from src.python.affect_utils import (
    compute_peer_influence,
    compute_event_appraisal_effect,
    compute_homeostasis_effect,
    compute_cumulative_overload,
    update_affect_dynamics,
    update_resilience_dynamics,
    AffectDynamicsConfig,
    ResilienceDynamicsConfig,
    clamp
)


class TestPeerInfluence:
    """Test peer influence computation on affect."""

    def test_peer_influence_empty_neighbors(self):
        """Test peer influence with no neighbors."""
        config = AffectDynamicsConfig(peer_influence_rate=0.1, influencing_neighbors=5)
        self_affect = 0.0

        influence = compute_peer_influence(self_affect, [], config)

        assert influence == 0.0

    def test_peer_influence_single_positive_neighbor(self):
        """Test peer influence with single positive neighbor."""
        config = AffectDynamicsConfig(peer_influence_rate=0.1, influencing_neighbors=5)
        self_affect = 0.0
        neighbor_affects = [0.8]

        influence = compute_peer_influence(self_affect, neighbor_affects, config)

        # Should be positive since neighbor affect > self affect
        expected = 0.1 * (0.8 - 0.0)  # rate * (neighbor - self)
        assert influence == expected

    def test_peer_influence_single_negative_neighbor(self):
        """Test peer influence with single negative neighbor."""
        config = AffectDynamicsConfig(peer_influence_rate=0.2, influencing_neighbors=5)
        self_affect = 0.5
        neighbor_affects = [-0.3]

        influence = compute_peer_influence(self_affect, neighbor_affects, config)

        # Should be negative since neighbor affect < self affect
        expected = 0.2 * (-0.3 - 0.5)  # rate * (neighbor - self)
        assert influence == expected

    def test_peer_influence_multiple_neighbors(self):
        """Test peer influence with multiple neighbors."""
        config = AffectDynamicsConfig(peer_influence_rate=0.1, influencing_neighbors=3)
        self_affect = 0.0
        neighbor_affects = [0.8, -0.4, 0.2, 0.6]  # Only first 3 should be considered

        influence = compute_peer_influence(self_affect, neighbor_affects, config)

        # Average of first 3 neighbors: (0.8 + (-0.4) + 0.2) / 3 = 0.2
        # Then: rate * (avg_neighbor - self) = 0.1 * (0.2 - 0.0) = 0.02
        expected = 0.1 * ((0.8 - 0.4 + 0.2) / 3)
        assert abs(influence - expected) < 1e-10

    def test_peer_influence_neighbor_limit(self):
        """Test that neighbor limit is respected."""
        config = AffectDynamicsConfig(peer_influence_rate=0.1, influencing_neighbors=2)
        self_affect = 0.0
        neighbor_affects = [0.5, 0.8, 0.3]  # Should only use first 2

        influence = compute_peer_influence(self_affect, neighbor_affects, config)

        # Average of first 2: (0.5 + 0.8) / 2 = 0.65
        expected = 0.1 * (0.65 - 0.0)
        assert abs(influence - expected) < 1e-10

    def test_peer_influence_extreme_values(self):
        """Test peer influence with extreme affect values."""
        config = AffectDynamicsConfig(peer_influence_rate=0.1, influencing_neighbors=5)

        # Test with very high neighbor affect
        influence_high = compute_peer_influence(-1.0, [1.0], config)
        expected_high = 0.1 * (1.0 - (-1.0))
        assert abs(influence_high - expected_high) < 1e-10

        # Test with very low neighbor affect
        influence_low = compute_peer_influence(1.0, [-1.0], config)
        expected_low = 0.1 * (-1.0 - 1.0)
        assert abs(influence_low - expected_low) < 1e-10

    def test_peer_influence_configuration_variations(self):
        """Test peer influence with different configuration parameters."""
        self_affect = 0.0
        neighbor_affects = [0.5]

        # Test different influence rates
        config_low = AffectDynamicsConfig(peer_influence_rate=0.05, influencing_neighbors=5)
        config_high = AffectDynamicsConfig(peer_influence_rate=0.2, influencing_neighbors=5)

        influence_low = compute_peer_influence(self_affect, neighbor_affects, config_low)
        influence_high = compute_peer_influence(self_affect, neighbor_affects, config_high)

        # Higher rate should produce stronger influence
        assert influence_high > influence_low
        assert abs(influence_low - 0.05 * 0.5) < 1e-10
        assert abs(influence_high - 0.2 * 0.5) < 1e-10


class TestEventAppraisalEffect:
    """Test event appraisal effect on affect."""

    def test_appraisal_effect_challenge_only(self):
        """Test appraisal effect with challenge only."""
        config = AffectDynamicsConfig(event_appraisal_rate=0.1)
        challenge = 0.8
        hindrance = 0.0
        current_affect = 0.0

        effect = compute_event_appraisal_effect(challenge, hindrance, current_affect, config)

        # Challenge should improve affect when current affect is low
        challenge_effect = 0.1 * 0.8 * (1.0 - 0.0)  # rate * challenge * (1 - current)
        hindrance_effect = -0.1 * 0.0 * (0.0 + 1.0)  # rate * hindrance * (current + 1)
        expected = challenge_effect + hindrance_effect

        assert abs(effect - expected) < 1e-10

    def test_appraisal_effect_hindrance_only(self):
        """Test appraisal effect with hindrance only."""
        config = AffectDynamicsConfig(event_appraisal_rate=0.15)
        challenge = 0.0
        hindrance = 0.7
        current_affect = 0.0

        effect = compute_event_appraisal_effect(challenge, hindrance, current_affect, config)

        # Hindrance should worsen affect
        challenge_effect = 0.15 * 0.0 * (1.0 - 0.0)
        hindrance_effect = -0.15 * 0.7 * (0.0 + 1.0)
        expected = challenge_effect + hindrance_effect

        assert abs(effect - expected) < 1e-10

    def test_appraisal_effect_balanced(self):
        """Test appraisal effect with balanced challenge and hindrance."""
        config = AffectDynamicsConfig(event_appraisal_rate=0.1)
        challenge = 0.6
        hindrance = 0.4
        current_affect = 0.0

        effect = compute_event_appraisal_effect(challenge, hindrance, current_affect, config)

        challenge_effect = 0.1 * 0.6 * (1.0 - 0.0)
        hindrance_effect = -0.1 * 0.4 * (0.0 + 1.0)
        expected = challenge_effect + hindrance_effect

        assert abs(effect - expected) < 1e-10

    def test_appraisal_effect_high_current_affect(self):
        """Test appraisal effect when current affect is high."""
        config = AffectDynamicsConfig(event_appraisal_rate=0.1)
        challenge = 0.8
        hindrance = 0.2
        current_affect = 0.8  # High positive affect

        effect = compute_event_appraisal_effect(challenge, hindrance, current_affect, config)

        # Challenge effect should be weaker due to high current affect
        # Hindrance effect should be stronger due to high current affect
        challenge_effect = 0.1 * 0.8 * (1.0 - 0.8)  # (1 - current) = 0.2
        hindrance_effect = -0.1 * 0.2 * (0.8 + 1.0)  # (current + 1) = 1.8
        expected = challenge_effect + hindrance_effect

        assert abs(effect - expected) < 1e-10

    def test_appraisal_effect_negative_current_affect(self):
        """Test appraisal effect when current affect is negative."""
        config = AffectDynamicsConfig(event_appraisal_rate=0.1)
        challenge = 0.3
        hindrance = 0.7
        current_affect = -0.6  # Negative affect

        effect = compute_event_appraisal_effect(challenge, hindrance, current_affect, config)

        # Challenge effect should be stronger due to low current affect
        # Hindrance effect should be weaker due to low current affect
        challenge_effect = 0.1 * 0.3 * (1.0 - (-0.6))  # (1 - current) = 1.6
        hindrance_effect = -0.1 * 0.7 * (-0.6 + 1.0)   # (current + 1) = 0.4
        expected = challenge_effect + hindrance_effect

        assert abs(effect - expected) < 1e-10

    def test_appraisal_effect_extreme_values(self):
        """Test appraisal effect with extreme challenge/hindrance values."""
        config = AffectDynamicsConfig(event_appraisal_rate=0.1)

        # Test with maximum challenge, minimum hindrance
        effect_max_challenge = compute_event_appraisal_effect(1.0, 0.0, 0.0, config)
        expected_max_challenge = 0.1 * 1.0 * (1.0 - 0.0) + (-0.1 * 0.0 * (0.0 + 1.0))
        assert abs(effect_max_challenge - expected_max_challenge) < 1e-10

        # Test with maximum hindrance, minimum challenge
        effect_max_hindrance = compute_event_appraisal_effect(0.0, 1.0, 0.0, config)
        expected_max_hindrance = 0.1 * 0.0 * (1.0 - 0.0) + (-0.1 * 1.0 * (0.0 + 1.0))
        assert abs(effect_max_hindrance - expected_max_hindrance) < 1e-10

    def test_appraisal_effect_configuration_variations(self):
        """Test appraisal effect with different configuration parameters."""
        challenge = 0.8  # Strong challenge
        hindrance = 0.0  # No hindrance
        current_affect = 0.0

        # Test different appraisal rates
        config_low = AffectDynamicsConfig(event_appraisal_rate=0.05)
        config_high = AffectDynamicsConfig(event_appraisal_rate=0.2)

        effect_low = compute_event_appraisal_effect(challenge, hindrance, current_affect, config_low)
        effect_high = compute_event_appraisal_effect(challenge, hindrance, current_affect, config_high)

        # Higher rate should produce stronger positive effects (challenge dominant)
        assert effect_high > effect_low > 0


class TestHomeostasisEffect:
    """Test homeostasis effect on affect."""

    def test_homeostasis_effect_above_baseline(self):
        """Test homeostasis when current affect is above baseline."""
        config = AffectDynamicsConfig(homeostasis_rate=0.1)
        current_affect = 0.5
        baseline_affect = 0.0

        effect = compute_homeostasis_effect(current_affect, baseline_affect, config)

        # Should push affect downward toward baseline
        distance = 0.0 - 0.5  # baseline - current = -0.5
        expected = -0.1 * abs(distance)  # Negative because distance < 0

        assert abs(effect - expected) < 1e-10

    def test_homeostasis_effect_below_baseline(self):
        """Test homeostasis when current affect is below baseline."""
        config = AffectDynamicsConfig(homeostasis_rate=0.15)
        current_affect = -0.4
        baseline_affect = 0.2

        effect = compute_homeostasis_effect(current_affect, baseline_affect, config)

        # Should push affect upward toward baseline
        distance = 0.2 - (-0.4)  # baseline - current = 0.6
        expected = 0.15 * abs(distance)  # Positive because distance > 0

        assert abs(effect - expected) < 1e-10

    def test_homeostasis_effect_at_baseline(self):
        """Test homeostasis when current affect equals baseline."""
        config = AffectDynamicsConfig(homeostasis_rate=0.1)
        current_affect = 0.3
        baseline_affect = 0.3

        effect = compute_homeostasis_effect(current_affect, baseline_affect, config)

        # No effect when at baseline
        assert effect == 0.0

    def test_homeostasis_effect_extreme_values(self):
        """Test homeostasis with extreme affect values."""
        config = AffectDynamicsConfig(homeostasis_rate=0.1)

        # Test with very high current affect
        effect_high = compute_homeostasis_effect(1.0, 0.0, config)
        expected_high = -0.1 * abs(0.0 - 1.0)  # Push down from 1.0 to 0.0
        assert abs(effect_high - expected_high) < 1e-10

        # Test with very low current affect
        effect_low = compute_homeostasis_effect(-1.0, 0.0, config)
        expected_low = 0.1 * abs(0.0 - (-1.0))  # Push up from -1.0 to 0.0
        assert abs(effect_low - expected_low) < 1e-10

    def test_homeostasis_effect_configuration_variations(self):
        """Test homeostasis with different configuration parameters."""
        current_affect = 0.5
        baseline_affect = 0.0

        # Test different homeostasis rates
        config_low = AffectDynamicsConfig(homeostasis_rate=0.05)
        config_high = AffectDynamicsConfig(homeostasis_rate=0.2)

        effect_low = compute_homeostasis_effect(current_affect, baseline_affect, config_low)
        effect_high = compute_homeostasis_effect(current_affect, baseline_affect, config_high)

        # Higher rate should produce stronger effect
        assert abs(effect_high) > abs(effect_low)
        assert abs(effect_low - (-0.05 * 0.5)) < 1e-10  # Negative (pushing down)
        assert abs(effect_high - (-0.2 * 0.5)) < 1e-10   # Negative (pushing down)


class TestCumulativeOverload:
    """Test cumulative overload effect on resilience."""

    def test_overload_effect_below_threshold(self):
        """Test overload effect when below threshold."""
        config = ResilienceDynamicsConfig(overload_threshold=3, influencing_hindrance=3)
        consecutive_hindrances = 2  # Below threshold

        effect = compute_cumulative_overload(consecutive_hindrances, config)

        # No effect below threshold
        assert effect == 0.0

    def test_overload_effect_at_threshold(self):
        """Test overload effect exactly at threshold."""
        config = ResilienceDynamicsConfig(overload_threshold=3, influencing_hindrance=3)
        consecutive_hindrances = 3  # At threshold

        effect = compute_cumulative_overload(consecutive_hindrances, config)

        # Effect at threshold: min(3/3, 2.0) = 1.0, then -0.2 * 1.0 = -0.2
        expected = -0.2 * min(3/3, 2.0)
        assert abs(effect - expected) < 1e-10

    def test_overload_effect_above_threshold(self):
        """Test overload effect above threshold."""
        config = ResilienceDynamicsConfig(overload_threshold=2, influencing_hindrance=4)
        consecutive_hindrances = 5  # Above threshold

        effect = compute_cumulative_overload(consecutive_hindrances, config)

        # Effect calculation: min(5/4, 2.0) = 1.25, then -0.2 * 1.25 = -0.25
        overload_intensity = min(5/4, 2.0)
        expected = -0.2 * overload_intensity
        assert abs(effect - expected) < 1e-10

    def test_overload_effect_maximum_intensity(self):
        """Test overload effect at maximum intensity."""
        config = ResilienceDynamicsConfig(overload_threshold=1, influencing_hindrance=2)
        consecutive_hindrances = 10  # Way above threshold

        effect = compute_cumulative_overload(consecutive_hindrances, config)

        # Should cap at maximum intensity of 2.0
        overload_intensity = min(10/2, 2.0)  # = 2.0
        expected = -0.2 * overload_intensity
        assert abs(effect - expected) < 1e-10

    def test_overload_effect_configuration_variations(self):
        """Test overload effect with different configuration parameters."""
        consecutive_hindrances = 5

        # Test different thresholds - both should be above their respective thresholds
        config_threshold_2 = ResilienceDynamicsConfig(overload_threshold=2, influencing_hindrance=3)
        config_threshold_4 = ResilienceDynamicsConfig(overload_threshold=4, influencing_hindrance=3)

        effect_2 = compute_cumulative_overload(consecutive_hindrances, config_threshold_2)
        effect_4 = compute_cumulative_overload(consecutive_hindrances, config_threshold_4)

        # For threshold 2: min(5/3, 2.0) = 1.666, effect = -0.2 * 1.666 = -0.333
        # For threshold 4: min(5/3, 2.0) = 1.666, effect = -0.2 * 1.666 = -0.333
        # They should be equal since the calculation gives the same result
        assert abs(effect_2 - effect_4) < 1e-10


class TestAffectDynamicsIntegration:
    """Test complete affect dynamics integration."""

    def test_affect_dynamics_basic_update(self):
        """Test basic affect dynamics update."""
        config = AffectDynamicsConfig()
        current_affect = 0.0
        baseline_affect = 0.0
        neighbor_affects = [0.5, -0.2]
        challenge = 0.3
        hindrance = 0.1

        new_affect = update_affect_dynamics(
            current_affect, baseline_affect, neighbor_affects,
            challenge, hindrance, config
        )

        # Should be in valid range
        assert -1.0 <= new_affect <= 1.0

        # With positive neighbor influence and challenge, should tend positive
        # (exact value depends on specific calculations)

    def test_affect_dynamics_extreme_inputs(self):
        """Test affect dynamics with extreme input values."""
        affect_config = AffectDynamicsConfig()

        # Test with extreme neighbor affects
        new_affect = update_affect_dynamics(
            current_affect=0.0,
            baseline_affect=0.0,
            neighbor_affects=[1.0, -1.0, 0.8],
            challenge=1.0,
            hindrance=1.0,
            affect_config=affect_config
        )

        # Should still be clamped to valid range
        assert -1.0 <= new_affect <= 1.0

    def test_affect_dynamics_empty_neighbors(self):
        """Test affect dynamics with no neighbors."""
        config = AffectDynamicsConfig()
        current_affect = 0.0
        baseline_affect = 0.0
        neighbor_affects = []
        challenge = 0.0
        hindrance = 0.0

        new_affect = update_affect_dynamics(
            current_affect, baseline_affect, neighbor_affects,
            challenge, hindrance, config
        )

        # Should be in valid range
        assert -1.0 <= new_affect <= 1.0

    def test_affect_dynamics_homeostasis_dominance(self):
        """Test that homeostasis pulls extreme affects toward baseline."""
        affect_config = AffectDynamicsConfig(homeostasis_rate=0.2)  # Strong homeostasis

        # Start with extreme positive affect
        new_affect = update_affect_dynamics(
            current_affect=1.0,
            baseline_affect=0.0,
            neighbor_affects=[],  # No peer influence
            challenge=0.0,
            hindrance=0.0,
            affect_config=affect_config
        )

        # Should be pulled back toward baseline (less than 1.0)
        assert new_affect < 1.0

        # Start with extreme negative affect
        new_affect = update_affect_dynamics(
            current_affect=-1.0,
            baseline_affect=0.0,
            neighbor_affects=[],
            challenge=0.0,
            hindrance=0.0,
            affect_config=affect_config
        )

        # Should be pulled back toward baseline (greater than -1.0)
        assert new_affect > -1.0

    def test_affect_dynamics_configuration_impact(self):
        """Test how different configurations affect the outcome."""
        current_affect = 0.0
        baseline_affect = 0.0
        neighbor_affects = [0.5]
        challenge = 0.3
        hindrance = 0.1

        # High influence configuration
        config_high = AffectDynamicsConfig(
            peer_influence_rate=0.3,
            event_appraisal_rate=0.3,
            homeostasis_rate=0.1
        )

        # Low influence configuration
        config_low = AffectDynamicsConfig(
            peer_influence_rate=0.05,
            event_appraisal_rate=0.05,
            homeostasis_rate=0.01
        )

        affect_high = update_affect_dynamics(
            current_affect, baseline_affect, neighbor_affects,
            challenge, hindrance, config_high
        )

        affect_low = update_affect_dynamics(
            current_affect, baseline_affect, neighbor_affects,
            challenge, hindrance, config_low
        )

        # Higher configuration values should produce larger magnitude changes
        assert abs(affect_high) > abs(affect_low)


class TestResilienceDynamicsIntegration:
    """Test complete resilience dynamics integration."""

    def test_resilience_dynamics_basic_update(self):
        """Test basic resilience dynamics update."""
        config = ResilienceDynamicsConfig()
        current_resilience = 0.5
        coped_successfully = True
        received_social_support = True
        consecutive_hindrances = 0

        new_resilience = update_resilience_dynamics(
            current_resilience, coped_successfully, received_social_support,
            consecutive_hindrances, config
        )

        # Should be in valid range
        assert 0.0 <= new_resilience <= 1.0

        # With positive factors, should tend to increase
        assert new_resilience >= current_resilience

    def test_resilience_dynamics_coping_success(self):
        """Test resilience dynamics with successful coping."""
        config = ResilienceDynamicsConfig(coping_success_rate=0.15)
        current_resilience = 0.5
        coped_successfully = True
        received_social_support = False
        consecutive_hindrances = 0

        new_resilience = update_resilience_dynamics(
            current_resilience, coped_successfully, received_social_support,
            consecutive_hindrances, config
        )

        # Should increase by coping success rate
        expected_increase = 0.15
        assert abs(new_resilience - (current_resilience + expected_increase)) < 1e-10

    def test_resilience_dynamics_social_support(self):
        """Test resilience dynamics with social support."""
        config = ResilienceDynamicsConfig(social_support_rate=0.12)
        current_resilience = 0.5
        coped_successfully = False
        received_social_support = True
        consecutive_hindrances = 0

        new_resilience = update_resilience_dynamics(
            current_resilience, coped_successfully, received_social_support,
            consecutive_hindrances, config
        )

        # Should increase by social support rate
        expected_increase = 0.12
        assert abs(new_resilience - (current_resilience + expected_increase)) < 1e-10

    def test_resilience_dynamics_overload_effect(self):
        """Test resilience dynamics with overload effect."""
        config = ResilienceDynamicsConfig(overload_threshold=2, influencing_hindrance=3)
        current_resilience = 0.8
        coped_successfully = False
        received_social_support = False
        consecutive_hindrances = 5  # Above threshold

        new_resilience = update_resilience_dynamics(
            current_resilience, coped_successfully, received_social_support,
            consecutive_hindrances, config
        )

        # Should decrease due to overload
        expected_overload = -0.2 * min(5/3, 2.0)  # -0.2 * 1.666... ≈ -0.333
        expected_resilience = current_resilience + expected_overload
        assert abs(new_resilience - expected_resilience) < 1e-10

    def test_resilience_dynamics_combined_effects(self):
        """Test resilience dynamics with multiple effects."""
        config = ResilienceDynamicsConfig(
            coping_success_rate=0.1,
            social_support_rate=0.08,
            overload_threshold=2,
            influencing_hindrance=3
        )
        current_resilience = 0.5
        coped_successfully = True
        received_social_support = True
        consecutive_hindrances = 4  # Above threshold

        new_resilience = update_resilience_dynamics(
            current_resilience, coped_successfully, received_social_support,
            consecutive_hindrances, config
        )

        # Should combine positive and negative effects
        overload_effect = -0.2 * min(4/3, 2.0)  # -0.2 * 1.333... ≈ -0.267
        total_expected = current_resilience + 0.1 + 0.08 + overload_effect

        assert abs(new_resilience - total_expected) < 1e-10

    def test_resilience_dynamics_extreme_values(self):
        """Test resilience dynamics with extreme input values."""
        resilience_config = ResilienceDynamicsConfig()

        # Test with very low resilience
        new_resilience = update_resilience_dynamics(
            current_resilience=0.0,
            coped_successfully=True,
            received_social_support=True,
            consecutive_hindrances=0,
            resilience_config=resilience_config
        )

        # Should still be in valid range
        assert 0.0 <= new_resilience <= 1.0

        # Test with very high resilience
        new_resilience = update_resilience_dynamics(
            current_resilience=1.0,
            coped_successfully=False,
            received_social_support=False,
            consecutive_hindrances=10,  # High overload
            resilience_config=resilience_config
        )

        # Should still be in valid range
        assert 0.0 <= new_resilience <= 1.0

    def test_resilience_dynamics_no_effects(self):
        """Test resilience dynamics with no positive or negative effects."""
        config = ResilienceDynamicsConfig()
        current_resilience = 0.5
        coped_successfully = False
        received_social_support = False
        consecutive_hindrances = 0  # Below threshold

        new_resilience = update_resilience_dynamics(
            current_resilience, coped_successfully, received_social_support,
            consecutive_hindrances, config
        )

        # Should remain unchanged
        assert abs(new_resilience - current_resilience) < 1e-10

    def test_resilience_dynamics_clamping_at_boundaries(self):
        """Test resilience dynamics clamping at boundaries."""
        resilience_config = ResilienceDynamicsConfig(coping_success_rate=0.3, social_support_rate=0.2)

        # Test clamping at upper boundary
        new_resilience = update_resilience_dynamics(
            current_resilience=0.9,
            coped_successfully=True,
            received_social_support=True,
            consecutive_hindrances=0,
            resilience_config=resilience_config
        )

        # Should be clamped at 1.0
        assert new_resilience == 1.0

        # Test clamping at lower boundary with overload
        resilience_config_overload = ResilienceDynamicsConfig(overload_threshold=1, influencing_hindrance=2)
        new_resilience = update_resilience_dynamics(
            current_resilience=0.1,
            coped_successfully=False,
            received_social_support=False,
            consecutive_hindrances=5,  # High overload
            resilience_config=resilience_config_overload
        )

        # Should be clamped at 0.0
        assert new_resilience == 0.0


class TestMathematicalCorrectness:
    """Test mathematical correctness of all calculations."""

    def test_peer_influence_mathematical_properties(self):
        """Test mathematical properties of peer influence."""
        config = AffectDynamicsConfig(peer_influence_rate=0.1, influencing_neighbors=5)

        # Test that influence is proportional to difference
        affect_diff = 0.3 - (-0.1)  # 0.4
        influence = compute_peer_influence(0.0, [0.3], config)
        expected = 0.1 * (0.3 - 0.0)
        assert abs(influence - expected) < 1e-10

        # Test that identical affects produce zero influence
        influence_zero = compute_peer_influence(0.5, [0.5, 0.5], config)
        assert abs(influence_zero) < 1e-10

    def test_appraisal_effect_mathematical_properties(self):
        """Test mathematical properties of appraisal effect."""
        config = AffectDynamicsConfig(event_appraisal_rate=0.1)

        # Test that challenge and hindrance effects are opposites in some sense
        challenge_effect = compute_event_appraisal_effect(0.5, 0.0, 0.0, config)
        hindrance_effect = compute_event_appraisal_effect(0.0, 0.5, 0.0, config)

        # Challenge should be positive, hindrance should be negative
        assert challenge_effect > 0
        assert hindrance_effect < 0

    def test_homeostasis_mathematical_properties(self):
        """Test mathematical properties of homeostasis."""
        config = AffectDynamicsConfig(homeostasis_rate=0.1)

        # Test that effect strength increases with distance from baseline
        effect_close = compute_homeostasis_effect(0.1, 0.0, config)
        effect_far = compute_homeostasis_effect(0.5, 0.0, config)

        # Farther distance should produce stronger effect
        assert abs(effect_far) > abs(effect_close)

    def test_overload_mathematical_properties(self):
        """Test mathematical properties of overload effect."""
        config = ResilienceDynamicsConfig(overload_threshold=3, influencing_hindrance=4)

        # Test that effect magnitude increases with consecutive hindrances
        effect_4 = compute_cumulative_overload(4, config)
        effect_6 = compute_cumulative_overload(6, config)

        # Higher consecutive hindrances should produce stronger negative effect
        assert effect_6 < effect_4  # More negative

    def test_clamping_mathematical_properties(self):
        """Test mathematical properties of clamping."""
        # Test that clamp is idempotent
        assert clamp(clamp(0.5)) == clamp(0.5)

        # Test that clamp preserves order
        a, b = 0.3, 0.7
        assert (clamp(a) <= clamp(b)) == (a <= b)

        # Test boundary conditions
        assert clamp(-2.0) == 0.0
        assert clamp(2.0) == 1.0
        assert clamp(0.5) == 0.5


class TestConfigurationIntegration:
    """Test integration with configuration system."""

    def test_affect_dynamics_config_defaults(self):
        """Test that affect dynamics config uses proper defaults."""
        config = AffectDynamicsConfig()

        # Should use values from configuration system
        assert 0.0 <= config.peer_influence_rate <= 1.0
        assert 0.0 <= config.event_appraisal_rate <= 1.0
        assert 0.0 <= config.homeostasis_rate <= 1.0
        assert config.influencing_neighbors > 0

    def test_resilience_dynamics_config_defaults(self):
        """Test that resilience dynamics config uses proper defaults."""
        config = ResilienceDynamicsConfig()

        # Should use values from configuration system
        assert 0.0 <= config.coping_success_rate <= 1.0
        assert 0.0 <= config.social_support_rate <= 1.0
        assert config.overload_threshold > 0
        assert config.influencing_hindrance > 0

    def test_custom_configuration_values(self):
        """Test that custom configuration values are used correctly."""
        config = AffectDynamicsConfig(
            peer_influence_rate=0.25,
            event_appraisal_rate=0.18,
            homeostasis_rate=0.08,
            influencing_neighbors=3
        )

        # Test that custom values are preserved
        assert config.peer_influence_rate == 0.25
        assert config.event_appraisal_rate == 0.18
        assert config.homeostasis_rate == 0.08
        assert config.influencing_neighbors == 3

        # Test that custom config affects computation
        self_affect = 0.0
        neighbor_affects = [0.5]

        influence = compute_peer_influence(self_affect, neighbor_affects, config)
        expected = 0.25 * (0.5 - 0.0)  # Custom rate
        assert abs(influence - expected) < 1e-10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_functions_with_none_config(self):
        """Test that all functions handle None config parameter."""
        # These should not raise exceptions
        compute_peer_influence(0.0, [0.5])
        compute_event_appraisal_effect(0.3, 0.2, 0.0)
        compute_homeostasis_effect(0.5, 0.0)
        compute_cumulative_overload(5)
        update_affect_dynamics(0.0, 0.0, [0.5], 0.3, 0.2)
        update_resilience_dynamics(0.5, True, True, 5)

    def test_functions_with_extreme_parameter_values(self):
        """Test functions with extreme but valid parameter values."""
        # Test with very small positive values
        config = AffectDynamicsConfig(peer_influence_rate=1e-10)
        influence = compute_peer_influence(0.0, [1.0], config)
        assert abs(influence) < 1e-9

        # Test with very large but valid values
        config = AffectDynamicsConfig(peer_influence_rate=0.999)
        influence = compute_peer_influence(0.0, [1.0], config)
        assert abs(influence - 0.999) < 1e-10

    def test_functions_with_boundary_parameter_values(self):
        """Test functions at exact boundary values."""
        # Test at exact zero boundaries
        config = AffectDynamicsConfig(peer_influence_rate=0.0)
        influence = compute_peer_influence(0.0, [1.0], config)
        assert influence == 0.0

        # Test at exact one boundaries
        config = AffectDynamicsConfig(event_appraisal_rate=1.0)
        effect = compute_event_appraisal_effect(1.0, 0.0, 0.0, config)
        expected = 1.0 * 1.0 * (1.0 - 0.0)  # Should be 1.0
        assert abs(effect - expected) < 1e-10


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_typical_mental_health_scenario(self):
        """Test a typical scenario of mental health dynamics."""
        # Scenario: Agent experiences moderate stress but copes well and has social support
        config_affect = AffectDynamicsConfig()
        config_resilience = ResilienceDynamicsConfig()

        # Initial state
        current_affect = -0.2  # Slightly negative
        current_resilience = 0.6
        baseline_affect = 0.0

        # Moderate stress event (more challenge than hindrance)
        challenge = 0.6
        hindrance = 0.3

        # Positive social environment
        neighbor_affects = [0.3, 0.1, 0.4]

        # Update affect
        new_affect = update_affect_dynamics(
            current_affect, baseline_affect, neighbor_affects,
            challenge, hindrance, config_affect
        )

        # Update resilience (successful coping + social support)
        new_resilience = update_resilience_dynamics(
            current_resilience, True, True, 0, config_resilience
        )

        # Both should improve
        assert new_affect > current_affect
        assert new_resilience > current_resilience

        # Should remain in valid ranges
        assert -1.0 <= new_affect <= 1.0
        assert 0.0 <= new_resilience <= 1.0

    def test_chronic_stress_scenario(self):
        """Test scenario with chronic stress and poor coping."""
        affect_config = AffectDynamicsConfig()
        resilience_config = ResilienceDynamicsConfig()

        # Initial state (already stressed)
        current_affect = -0.6
        current_resilience = 0.3
        baseline_affect = 0.0

        # High hindrance, low challenge (frustrating situation)
        challenge = 0.2
        hindrance = 0.8

        # Negative social environment
        neighbor_affects = [-0.4, -0.2, -0.6]

        # Update affect
        new_affect = update_affect_dynamics(
            current_affect, baseline_affect, neighbor_affects,
            challenge, hindrance, affect_config
        )

        # Update resilience (failed coping + no social support + high overload)
        new_resilience = update_resilience_dynamics(
            current_resilience, False, False, 5, resilience_config
        )

        # Resilience should worsen due to overload
        assert new_resilience < current_resilience

        # Both should remain in valid ranges
        assert -1.0 <= new_affect <= 1.0
        assert 0.0 <= new_resilience <= 1.0

        # The affect change depends on multiple factors, so we just check bounds

    def test_recovery_scenario(self):
        """Test scenario of recovery from stress."""
        config_affect = AffectDynamicsConfig(homeostasis_rate=0.2)  # Strong homeostasis
        config_resilience = ResilienceDynamicsConfig()

        # Initial state (recovering)
        current_affect = -0.8
        current_resilience = 0.2
        baseline_affect = 0.0

        # No new stress
        challenge = 0.0
        hindrance = 0.0

        # Supportive social environment
        neighbor_affects = [0.4, 0.2, 0.6]

        # Update affect (homeostasis should pull toward baseline)
        new_affect = update_affect_dynamics(
            current_affect, baseline_affect, neighbor_affects,
            challenge, hindrance, config_affect
        )

        # Update resilience (successful coping + social support)
        new_resilience = update_resilience_dynamics(
            current_resilience, True, True, 0, config_resilience
        )

        # Affect should improve toward baseline
        assert new_affect > current_affect
        assert new_resilience > current_resilience

        # Should remain in valid ranges
        assert -1.0 <= new_affect <= 1.0
        assert 0.0 <= new_resilience <= 1.0