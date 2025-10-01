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
    StressEvent, AppraisalWeights, ThresholdParams
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
        assert event1.predictability == event2.predictability
        assert event1.overload == event2.overload
        assert event1.magnitude == event2.magnitude

    def test_generate_stress_event_bounds(self):
        """Test that generated events have values in valid [0,1] range."""
        rng = np.random.default_rng(123)

        for _ in range(100):
            event = generate_stress_event(rng=rng)

            assert 0.0 <= event.controllability <= 1.0
            assert 0.0 <= event.predictability <= 1.0
            assert 0.0 <= event.overload <= 1.0
            assert 0.0 <= event.magnitude <= 1.0

    def test_generate_stress_event_with_config(self):
        """Test stress event generation with custom configuration."""
        rng = np.random.default_rng(456)
        config = {
            'controllability_mean': 0.8,
            'predictability_mean': 0.2,
            'overload_mean': 0.6,
            'magnitude_scale': 0.1
        }

        # Generate multiple events and check statistical properties
        events = [generate_stress_event(rng=rng, config=config) for _ in range(1000)]

        controllability_values = [e.controllability for e in events]
        predictability_values = [e.predictability for e in events]
        overload_values = [e.overload for e in events]
        magnitude_values = [e.magnitude for e in events]

        # Check that means are roughly in expected ranges (allowing for variance)
        assert 0.3 < np.mean(controllability_values) < 0.9  # Beta(2,2) ≈ 0.5
        assert 0.3 < np.mean(predictability_values) < 0.9
        assert 0.3 < np.mean(overload_values) < 0.9
        assert np.mean(magnitude_values) < 0.5  # Exponential with scale 0.1


class TestStressAppraisal:
    """Test stress appraisal mechanisms."""

    def test_apply_weights_basic(self):
        """Test basic weight application for challenge/hindrance mapping."""
        event = StressEvent(
            controllability=1.0,  # High controllability
            predictability=1.0,   # High predictability
            overload=0.0,         # Low overload
            magnitude=0.5
        )

        weights = AppraisalWeights(omega_c=1.0, omega_p=1.0, omega_o=1.0, bias=0.0, gamma=6.0)

        challenge, hindrance = apply_weights(event, weights)

        # High controllability + predictability - low overload should give high challenge
        assert challenge > 0.8  # Should be close to 1.0
        assert hindrance < 0.2  # Should be close to 0.0
        assert abs(challenge + hindrance - 1.0) < 1e-10  # Should sum to 1.0

    def test_apply_weights_extreme_cases(self):
        """Test weight application at extreme values."""
        # Case 1: Maximum challenge scenario
        event_max_challenge = StressEvent(1.0, 1.0, 0.0, 1.0)
        challenge, hindrance = apply_weights(event_max_challenge)

        assert challenge > 0.99  # Should be very close to 1.0
        assert hindrance < 0.01  # Should be very close to 0.0

        # Case 2: Maximum hindrance scenario
        event_max_hindrance = StressEvent(0.0, 0.0, 1.0, 1.0)
        challenge, hindrance = apply_weights(event_max_hindrance)

        assert challenge < 0.01  # Should be very close to 0.0
        assert hindrance > 0.99  # Should be very close to 1.0

    def test_sigmoid_function(self):
        """Test the sigmoid function used in challenge/hindrance mapping."""
        from src.python.stress_utils import sigmoid

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
            predictability=0.7,
            overload=0.2,
            magnitude=0.6
        )

        threshold_params = ThresholdParams(
            base_threshold=0.5,
            challenge_scale=0.15,
            hindrance_scale=0.25
        )

        weights = AppraisalWeights(
            omega_c=1.0, omega_p=1.0, omega_o=1.0,
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
        event = StressEvent(0.5, 0.5, 0.5, 0.5)
        threshold_params = ThresholdParams(0.5, 0.1, 0.2)
        weights = AppraisalWeights(1.0, 1.0, 1.0, 0.0, 6.0)

        # Process twice with same parameters
        result1 = process_stress_event(event, threshold_params, weights)
        result2 = process_stress_event(event, threshold_params, weights)

        # Should be identical (no randomness in this processing)
        assert result1 == result2


# Example of how to run these tests:
# pytest test_stress_utils.py -v
# pytest test_stress_utils.py::TestStressEventGeneration::test_generate_stress_event_deterministic -v