#!/usr/bin/env python3
"""
Test script to verify the model properly integrates with new stress processing mechanisms.
"""

import numpy as np
import sys
import os

from src.python.model import StressModel
from src.python.config import get_config

def test_model_integration():
    """Test that the model works with new stress processing mechanisms."""
    print("Testing model integration with new stress processing mechanisms...")

    # Create a small model for testing
    model = StressModel(N=10, max_days=3, seed=42)

    print(f"Created model with {len(model.agents)} agents")

    # Run a few steps
    for day in range(3):
        print(f"\n--- Day {day} ---")
        model.step()

        # Get population summary
        summary = model.get_population_summary()

        print(f"Avg affect: {summary['avg_affect']:.3f}")
        print(f"Avg resilience: {summary['avg_resilience']:.3f}")
        print(f"Avg stress: {summary['avg_stress']:.3f}")
        print(f"Avg challenge: {summary['avg_challenge']:.3f}")
        print(f"Avg hindrance: {summary['avg_hindrance']:.3f}")
        print(f"Coping success rate: {summary['coping_success_rate']:.3f}")
        print(f"Stress events: {summary.get('stress_events', 'N/A')}")

        # Verify that new metrics are being collected
        assert 'avg_stress' in summary, "avg_stress metric missing"
        assert 'avg_challenge' in summary, "avg_challenge metric missing"
        assert 'avg_hindrance' in summary, "avg_hindrance metric missing"
        assert 'coping_success_rate' in summary, "coping_success_rate metric missing"

        # Verify metrics are in valid ranges
        assert -1.0 <= summary['avg_affect'] <= 1.0, "Affect out of range"
        assert 0.0 <= summary['avg_resilience'] <= 1.0, "Resilience out of range"
        assert 0.0 <= summary['avg_stress'] <= 1.0, "Stress out of range"
        assert 0.0 <= summary['avg_challenge'] <= 1.0, "Challenge out of range"
        assert 0.0 <= summary['avg_hindrance'] <= 1.0, "Hindrance out of range"
        assert 0.0 <= summary['coping_success_rate'] <= 1.0, "Coping rate out of range"

    # Test time series data collection
    time_series = model.get_time_series_data()
    print(f"\nTime series data shape: {time_series.shape}")

    # Verify new columns are in time series
    expected_columns = [
        'avg_stress', 'avg_challenge', 'avg_hindrance',
        'coping_success_rate', 'avg_consecutive_hindrances', 'challenge_hindrance_ratio'
    ]

    for col in expected_columns:
        assert col in time_series.columns, f"Missing column: {col}"

    print(f"✅ All expected columns present: {list(time_series.columns)}")
    print("✅ Model integration test passed!")

if __name__ == "__main__":
    test_model_integration()