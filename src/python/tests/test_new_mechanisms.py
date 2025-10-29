#!/usr/bin/env python3
"""
Test script to verify the new stress processing mechanisms work correctly.
"""

import numpy as np
import sys
import os

from affect_utils import (
    compute_coping_probability,
    compute_challenge_hindrance_resilience_effect,
    compute_daily_affect_reset,
    compute_stress_decay,
    determine_coping_outcome_and_psychological_impact,
    StressProcessingConfig
)

def test_new_mechanisms():
    """Test the new stress processing mechanisms."""
    print("Testing new stress processing mechanisms...")

    # Test coping probability mechanism
    print("\n1. Testing coping probability mechanism:")
    challenge = 0.8  # High challenge
    hindrance = 0.2  # Low hindrance
    neighbor_affects = [0.5, 0.3, 0.7]  # Positive neighbors

    coping_prob = compute_coping_probability(challenge, hindrance, neighbor_affects)
    print(f"   Challenge: {challenge}, Hindrance: {hindrance}")
    print(f"   Neighbor affects: {neighbor_affects}")
    print(f"   Coping probability: {coping_prob:.3f}")
    assert coping_prob > 0.5, "High challenge should increase coping probability"

    # Test with negative neighbors
    negative_neighbors = [-0.5, -0.3, -0.7]
    coping_prob_neg = compute_coping_probability(challenge, hindrance, negative_neighbors)
    print(f"   Negative neighbors coping probability: {coping_prob_neg:.3f}")
    assert coping_prob_neg < coping_prob, "Negative neighbors should decrease coping probability"

    # Test challenge/hindrance resilience effects
    print("\n2. Testing challenge/hindrance resilience effects:")

    # Success case
    resilience_success = compute_challenge_hindrance_resilience_effect(
        challenge=0.8, hindrance=0.2, coped_successfully=True
    )
    print(f"   Success case - Challenge: {challenge}, Hindrance: {hindrance}")
    print(f"   Resilience effect: {resilience_success:.3f}")
    assert resilience_success > 0, "Success should increase resilience"

    # Failure case
    resilience_failure = compute_challenge_hindrance_resilience_effect(
        challenge=0.8, hindrance=0.2, coped_successfully=False
    )
    print(f"   Failure case - Challenge: {challenge}, Hindrance: {hindrance}")
    print(f"   Resilience effect: {resilience_failure:.3f}")
    assert resilience_failure < 0, "Failure should decrease resilience"

    # Test daily affect reset
    print("\n3. Testing daily affect reset:")
    current_affect = 0.8
    baseline_affect = 0.0

    reset_affect = compute_daily_affect_reset(current_affect, baseline_affect)
    print(f"   Current affect: {current_affect}, Baseline: {baseline_affect}")
    print(f"   Reset affect: {reset_affect:.3f}")
    assert reset_affect < current_affect, "Should move toward baseline"

    # Test stress decay
    print("\n4. Testing stress decay:")
    current_stress = 0.8

    decayed_stress = compute_stress_decay(current_stress)
    print(f"   Current stress: {current_stress}")
    print(f"   Decayed stress: {decayed_stress:.3f}")
    assert decayed_stress < current_stress, "Stress should decay over time"

    # Test complete stress event processing
    print("\n5. Testing complete stress event processing:")
    current_affect = 0.0
    current_resilience = 0.5
    current_stress = 0.3
    challenge = 0.7
    hindrance = 0.3
    neighbor_affects = [0.4, 0.6]

    new_affect, new_resilience, new_stress, coped_successfully = determine_coping_outcome_and_psychological_impact(
        current_affect, current_resilience, current_stress,
        challenge, hindrance, neighbor_affects
    )

    print(f"   Challenge: {challenge}, Hindrance: {hindrance}")
    print(f"   Coping successful: {coped_successfully}")
    print(f"   Affect: {current_affect:.3f} -> {new_affect:.3f}")
    print(f"   Resilience: {current_resilience:.3f} -> {new_resilience:.3f}")
    print(f"   Stress: {current_stress:.3f} -> {new_stress:.3f}")

    # Verify ranges
    assert -1.0 <= new_affect <= 1.0, "Affect should be in valid range"
    assert 0.0 <= new_resilience <= 1.0, "Resilience should be in valid range"
    assert 0.0 <= new_stress <= 1.0, "Stress should be in valid range"

    print("\n✅ All new mechanisms are working correctly!")

if __name__ == "__main__":
    test_new_mechanisms()