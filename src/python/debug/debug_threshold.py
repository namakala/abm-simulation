#!/usr/bin/env python3
"""
Debug script to check threshold evaluation function.
"""

import sys
sys.path.append('.')
from src.python.config import get_config
from src.python.stress_utils import StressEvent, ThresholdParams, evaluate_stress_threshold

# Get configuration
config = get_config()

# Create test data
stress_event = StressEvent(0.5, 0.5, 0.5, 0.5)
threshold_params = ThresholdParams()

print("Debugging stress threshold...\n")
print(f"Stress event: {stress_event}")
print(f"Threshold params: {threshold_params}")

# Test threshold evaluation
appraised_stress = 0.6
challenge = 0.3
hindrance = 0.7

print(f"Appraised stress: {appraised_stress}")
print(f"Challenge: {challenge}")
print(f"Hindrance: {hindrance}")

result = evaluate_stress_threshold(appraised_stress, challenge, hindrance, threshold_params)
print(f"Result: {result}")
print(f"Result type: {type(result)}")

# Calculate effective threshold manually
effective_threshold = (threshold_params.base_threshold +
                      threshold_params.challenge_scale * challenge -
                      threshold_params.hindrance_scale * hindrance)
effective_threshold = max(0.0, min(1.0, effective_threshold))

print(f"Effective threshold: {effective_threshold}")
print(f"Comparison: {appraised_stress} > {effective_threshold} = {appraised_stress > effective_threshold}\n")