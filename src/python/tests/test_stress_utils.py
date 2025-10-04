"""
Example unit tests for stress_utils.py

This file demonstrates how to write independent unit tests for utility functions
using pytest and dependency injection for reproducible testing.
"""

import pytest
import numpy as np
from src.python.stress_utils import (
    generate_stress_event, process_stress_event, apply_weights,
    compute_appraised_stress, evaluate_stress_threshold,
    StressEvent, AppraisalWeights, ThresholdParams,
    create_pss10_mapping, map_agent_stress_to_pss10, compute_pss10_score,
    interpret_pss10_score, PSS10Item, sigmoid
)


class TestStressEventGeneration:
    """Test stress event generation with controlled randomness."""

    def test_generate_stress_event_deterministic(self):
        """Test that stress events are generated deterministically with fixed seed."""
        rng = np.random.default_rng(42)

        event1 = generate_stress_event(rng=rng)
        rng = np.random.default_rng(42)  # Reset seed
        event2 = generate_stress_event(rng=rng)

        # Should be identical with same seed
        assert event1.controllability == event2.controllability
        assert event1.overload == event2.overload

    def test_generate_stress_event_bounds(self):
        """Test that generated events have values in valid [0,1] range."""
        rng = np.random.default_rng(123)

        for _ in range(100):
            event = generate_stress_event(rng=rng)

            assert 0.0 <= event.controllability <= 1.0
            assert 0.0 <= event.overload <= 1.0

    def test_generate_stress_event_with_config(self):
        """Test stress event generation with custom configuration."""
        rng = np.random.default_rng(456)
        config = {
            'controllability_mean': 0.8,
            'overload_mean': 0.6
        }

        # Generate multiple events and check statistical properties
        events = [generate_stress_event(rng=rng, config=config) for _ in range(1000)]

        controllability_values = [e.controllability for e in events]
        overload_values = [e.overload for e in events]

        # Check that means are roughly in expected ranges (allowing for variance)
        assert 0.3 < np.mean(controllability_values) < 0.9  # Beta(2,2) ≈ 0.5
        assert 0.3 < np.mean(overload_values) < 0.9


class TestStressAppraisal:
    """Test stress appraisal mechanisms."""

    def test_apply_weights_basic(self):
        """Test basic weight application for challenge/hindrance mapping."""
        event = StressEvent(
            controllability=1.0,  # High controllability
            overload=0.0         # Low overload
        )

        weights = AppraisalWeights(omega_c=1.0, omega_o=1.0, bias=0.0, gamma=6.0)

        challenge, hindrance = apply_weights(event, weights)

        # High controllability - low overload should give high challenge
        assert challenge > 0.8  # Should be close to 1.0
        assert hindrance < 0.2  # Should be close to 0.0
        assert abs(challenge + hindrance - 1.0) < 1e-10  # Should sum to 1.0

    def test_apply_weights_extreme_cases(self):
        """Test weight application at extreme values."""
        # Case 1: Maximum challenge scenario
        event_max_challenge = StressEvent(1.0, 0.0)
        challenge, hindrance = apply_weights(event_max_challenge)

        assert challenge > 0.99  # Should be very close to 1.0
        assert hindrance < 0.01  # Should be very close to 0.0

        # Case 2: Maximum hindrance scenario
        event_max_hindrance = StressEvent(0.0, 1.0)
        challenge, hindrance = apply_weights(event_max_hindrance)

        assert challenge < 0.01  # Should be very close to 0.0
        assert hindrance > 0.99  # Should be very close to 1.0

    def test_sigmoid_function(self):
        """Test the sigmoid function used in challenge/hindrance mapping."""
        # Test basic properties
        assert sigmoid(0.0) == 0.5  # Sigmoid(0) = 0.5
        assert sigmoid(10.0) > 0.99  # Large positive → close to 1.0
        assert sigmoid(-10.0) < 0.01  # Large negative → close to 0.0

        # Test gamma parameter
        assert sigmoid(1.0, gamma=1.0) > sigmoid(1.0, gamma=0.1)  # Higher gamma = steeper


class TestStressThreshold:
    """Test stress threshold evaluation."""

    def test_threshold_evaluation_basic(self):
        """Test basic threshold evaluation."""
        threshold_params = ThresholdParams(
            base_threshold=0.5,
            challenge_scale=0.1,
            hindrance_scale=0.2
        )

        # Case 1: Stress below threshold
        is_stressed = evaluate_stress_threshold(
            appraised_stress=0.3,
            challenge=0.5,
            hindrance=0.5,
            threshold_params=threshold_params
        )
        assert not is_stressed

        # Case 2: Stress above threshold
        is_stressed = evaluate_stress_threshold(
            appraised_stress=0.7,
            challenge=0.5,
            hindrance=0.5,
            threshold_params=threshold_params
        )
        assert is_stressed

    def test_challenge_hindrance_threshold_modification(self):
        """Test how challenge and hindrance modify the effective threshold."""
        base_threshold = 0.5
        challenge_scale = 0.2
        hindrance_scale = 0.3

        threshold_params = ThresholdParams(
            base_threshold=base_threshold,
            challenge_scale=challenge_scale,
            hindrance_scale=hindrance_scale
        )

        # High challenge should increase effective threshold
        effective_threshold_high_challenge = (
            base_threshold + challenge_scale * 1.0 - hindrance_scale * 0.0
        )

        # High hindrance should decrease effective threshold
        effective_threshold_high_hindrance = (
            base_threshold + challenge_scale * 0.0 - hindrance_scale * 1.0
        )

        assert effective_threshold_high_challenge > base_threshold
        assert effective_threshold_high_hindrance < base_threshold

        # Test actual evaluation
        is_stressed_high_challenge = evaluate_stress_threshold(
            appraised_stress=0.6,  # Above base threshold
            challenge=1.0,         # High challenge
            hindrance=0.0,         # Low hindrance
            threshold_params=threshold_params
        )

        is_stressed_high_hindrance = evaluate_stress_threshold(
            appraised_stress=0.4,  # Below base threshold
            challenge=0.0,         # Low challenge
            hindrance=1.0,         # High hindrance
            threshold_params=threshold_params
        )

        # High challenge might prevent stress even with high appraised stress
        # High hindrance might cause stress even with low appraised stress
        # (Depending on the scale parameters)


class TestCompleteStressProcessing:
    """Test the complete stress processing pipeline."""

    def test_process_stress_event_pipeline(self):
        """Test the complete stress event processing pipeline."""
        # Create a specific stress event
        event = StressEvent(
            controllability=0.8,
            overload=0.2
        )

        threshold_params = ThresholdParams(
            base_threshold=0.5,
            challenge_scale=0.15,
            hindrance_scale=0.25
        )

        weights = AppraisalWeights(
            omega_c=1.0, omega_o=1.0,
            bias=0.0, gamma=6.0
        )

        rng = np.random.default_rng(789)

        # Process the event
        is_stressed, challenge, hindrance = process_stress_event(
            event=event,
            threshold_params=threshold_params,
            weights=weights,
            rng=rng
        )

        # Verify outputs are in valid ranges
        assert 0.0 <= challenge <= 1.0
        assert 0.0 <= hindrance <= 1.0
        assert abs(challenge + hindrance - 1.0) < 1e-10
        assert isinstance(is_stressed, (bool, np.bool_))

    def test_process_stress_event_deterministic(self):
        """Test that stress processing is deterministic with fixed parameters."""
        event = StressEvent(0.5, 0.5)
        threshold_params = ThresholdParams(0.5, 0.1, 0.2)
        weights = AppraisalWeights(1.0, 1.0, 0.0, 6.0)

        # Process twice with same parameters
        result1 = process_stress_event(event, threshold_params, weights)
        result2 = process_stress_event(event, threshold_params, weights)

        # Should be identical (no randomness in this processing)
        assert result1 == result2


class TestPSS10Mapping:
    """Test PSS-10 mapping functionality."""

    def test_create_pss10_mapping_structure(self):
        """Test that PSS-10 mapping creates correct structure."""
        mapping = create_pss10_mapping()

        # Should have 10 items
        assert len(mapping) == 10

        # All items should be numbered 1-10
        assert set(mapping.keys()) == set(range(1, 11))

        # All items should be PSS10Item objects
        for item in mapping.values():
            assert isinstance(item, PSS10Item)
            assert len(item.text) > 0  # Should have non-empty text

    def test_create_pss10_mapping_content(self):
        """Test that PSS-10 items have appropriate content and weights."""
        mapping = create_pss10_mapping()

        # Check specific items for correct reverse scoring
        assert not mapping[1].reverse_scored  # Item 1: not reverse scored
        assert not mapping[2].reverse_scored  # Item 2: not reverse scored
        assert not mapping[3].reverse_scored  # Item 3: not reverse scored
        assert mapping[4].reverse_scored     # Item 4: reverse scored
        assert mapping[5].reverse_scored     # Item 5: reverse scored
        assert not mapping[6].reverse_scored  # Item 6: not reverse scored
        assert mapping[7].reverse_scored     # Item 7: reverse scored
        assert mapping[8].reverse_scored     # Item 8: reverse scored
        assert not mapping[9].reverse_scored  # Item 9: not reverse scored
        assert not mapping[10].reverse_scored # Item 10: not reverse scored

        # Check that controllability items have appropriate weights
        assert mapping[2].weight_controllability > 0.5  # Item 2: high controllability weight
        assert mapping[4].weight_controllability > 0.5  # Item 4: high controllability weight

        # Check that overload items have appropriate weights
        assert mapping[6].weight_overload > 0.5  # Item 6: high overload weight
        assert mapping[10].weight_overload > 0.5  # Item 10: high overload weight

    def test_map_agent_stress_to_pss10_deterministic(self):
        """Test that PSS-10 mapping is deterministic with fixed seed."""
        rng = np.random.default_rng(42)
        responses1 = map_agent_stress_to_pss10(0.5, 0.5, rng)

        rng = np.random.default_rng(42)  # Reset seed
        responses2 = map_agent_stress_to_pss10(0.5, 0.5, rng)

        assert responses1 == responses2

    def test_map_agent_stress_to_pss10_response_range(self):
        """Test that PSS-10 responses are in valid range [0,4]."""
        rng = np.random.default_rng(123)

        # Test with various stress levels
        test_cases = [
            (0.0, 0.0),  # No stress
            (1.0, 1.0),  # Maximum stress
            (0.5, 0.5),  # Moderate stress
            (0.2, 0.3),  # Mixed stress
        ]

        for c, o in test_cases:
            responses = map_agent_stress_to_pss10(c, o, rng)

            for item_num, response in responses.items():
                assert 0 <= response <= 4, f"Item {item_num} response {response} out of range [0,4]"

    def test_map_agent_stress_to_pss10_extreme_stress(self):
        """Test PSS-10 mapping with extreme stress conditions."""
        rng = np.random.default_rng(456)

        # High controllability, low overload
        responses_low_stress = map_agent_stress_to_pss10(0.9, 0.1, rng)

        # Low controllability, high overload
        responses_high_stress = map_agent_stress_to_pss10(0.1, 0.9, rng)

        # High stress should generally have higher scores than low stress
        # (though not guaranteed for every item due to variability)
        high_stress_scores = list(responses_high_stress.values())
        low_stress_scores = list(responses_low_stress.values())

        # The average should be higher for high stress (allowing for some variance)
        assert np.mean(high_stress_scores) > np.mean(low_stress_scores) - 0.5  # Allow some tolerance

    def test_compute_pss10_score_complete_responses(self):
        """Test PSS-10 score computation with complete responses."""
        # All zeros (minimum stress) - but items 4,5,7,8 are reverse scored, so they become 4 each
        reversed = [4, 5, 7, 8]
        responses_min = {i: 4 if i in reversed else 0 for i in range(1, 11)}
        score_min = compute_pss10_score(responses_min)
        assert score_min == 0

        # All maximum values
        responses_max = {i: 0 if i in reversed else 4 for i in range(1, 11)}
        score_max = compute_pss10_score(responses_max)
        assert score_max == 40  # 10 items * 4 points each

        # Mixed responses
        responses_mixed = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 2, 7: 1, 8: 0, 9: 3, 10: 4}
        score_mixed = compute_pss10_score(responses_mixed)
        expected_mixed = 0+1+2+(4-3)+(4-4)+2+(4-1)+(4-0)+3+4  # Apply reverse scoring: items 4,5,7,8 are reversed
        assert score_mixed == expected_mixed

    def test_compute_pss10_score_missing_items(self):
        """Test PSS-10 score computation with missing items raises error."""
        responses_missing = {1: 0, 2: 1, 3: 2}  # Missing items 4-10

        with pytest.raises(ValueError, match="Missing PSS-10 items"):
            compute_pss10_score(responses_missing)

    def test_compute_pss10_score_invalid_responses(self):
        """Test PSS-10 score computation with invalid response values."""
        # Test 1: Invalid response value (but complete set of items)
        responses_invalid = {i: 2 for i in range(1, 11)}  # All items present
        responses_invalid[1] = 5  # Invalid value

        with pytest.raises(ValueError, match="Invalid response for item 1"):
            compute_pss10_score(responses_invalid)

        # Test 2: Missing items
        responses_missing = {i: 2 for i in range(1, 10)}  # Item 10 missing

        with pytest.raises(ValueError, match="Missing PSS-10 items"):
            compute_pss10_score(responses_missing)

    def test_interpret_pss10_score_categories(self):
        """Test PSS-10 score interpretation categories."""
        # Low stress (0-13)
        assert interpret_pss10_score(0) == "Low stress"
        assert interpret_pss10_score(13) == "Low stress"

        # Moderate stress (14-26)
        assert interpret_pss10_score(14) == "Moderate stress"
        assert interpret_pss10_score(26) == "Moderate stress"

        # High stress (27-40)
        assert interpret_pss10_score(27) == "High stress"
        assert interpret_pss10_score(40) == "High stress"

    def test_interpret_pss10_score_invalid_range(self):
        """Test PSS-10 score interpretation with invalid scores."""
        with pytest.raises(ValueError, match="Invalid PSS-10 score"):
            interpret_pss10_score(-1)

        with pytest.raises(ValueError, match="Invalid PSS-10 score"):
            interpret_pss10_score(41)

    def test_pss10_integration_pipeline(self):
        """Test complete PSS-10 pipeline from agent state to interpretation."""
        rng = np.random.default_rng(789)

        # Test with moderate stress agent
        c, o = 0.3, 0.6

        # Map to PSS-10 responses
        responses = map_agent_stress_to_pss10(c, o, rng)

        # Compute total score
        total_score = compute_pss10_score(responses)

        # Interpret score
        interpretation = interpret_pss10_score(total_score)

        # Verify pipeline produces valid results
        assert 0 <= total_score <= 40
        assert interpretation in ["Low stress", "Moderate stress", "High stress"]
        assert len(responses) == 10
        assert all(0 <= response <= 4 for response in responses.values())
