#!/usr/bin/env python3
"""
Simple test to isolate threshold evaluation issue.
"""

import sys
sys.path.append('.')

from src.python.config import get_config
from src.python.stress_utils import generate_stress_event, apply_weights, evaluate_stress_threshold, ThresholdParams, AppraisalWeights
from src.python.math_utils import create_rng

# Get configuration
config = get_config()

# Create test data exactly as in the integration test
rng = create_rng(42)
stress_event = generate_stress_event(rng)
appraisal_weights = AppraisalWeights()
threshold_params = ThresholdParams()

print("Conducting a simple threshold test...\n")
print(f"Stress event: {stress_event}")
print(f"Appraisal weights: {appraisal_weights}")
print(f"Threshold params: {threshold_params}")

# Apply weights
challenge, hindrance = apply_weights(stress_event, appraisal_weights)
print(f"Challenge: {challenge}, Hindrance: {hindrance}")

# Test threshold evaluation
appraised_stress = 0.6
is_stressed = evaluate_stress_threshold(appraised_stress, challenge, hindrance, threshold_params)
print(f"Is stressed: {is_stressed}")
print(f"Type: {type(is_stressed)}")

# Manual calculation
effective_threshold = (threshold_params.base_threshold +
                      threshold_params.challenge_scale * challenge -
                      threshold_params.hindrance_scale * hindrance)
effective_threshold = max(0.0, min(1.0, effective_threshold))
print(f"Effective threshold: {effective_threshold}")
print(f"Manual comparison: {appraised_stress} > {effective_threshold} = {appraised_stress > effective_threshold}\n")