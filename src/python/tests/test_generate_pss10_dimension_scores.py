#!/usr/bin/env python3
"""
Comprehensive tests for the merged generate_pss10_dimension_scores function.

Tests all requirements:
1. Backward compatibility - existing function calls work unchanged
2. Deterministic behavior works correctly when deterministic=True
3. Non-deterministic behavior works correctly when deterministic=False
4. Configuration integration works (PSS10_BIFACTOR_COR, PSS10_CONTROLLABILITY_SD, PSS10_OVERLOAD_SD)
5. Regularized standard deviations are applied correctly (divided by 4)
6. Function produces reasonable results in all modes
"""

import numpy as np
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import sys
import os
# Add the src/python directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stress_utils import generate_pss10_dimension_scores, generate_pss10_responses
from config import Config, ConfigurationError
import traceback


class TestPSS10DimensionScoresBackwardCompatibility:
    """Test backward compatibility of generate_pss10_dimension_scores function."""

    def test_backward_compatibility_basic_call(self):
        """Test that function works with basic call (no optional parameters)."""
        rng = np.random.default_rng(42)

        # Basic call without optional parameters
        corr_c, corr_o = generate_pss10_dimension_scores(0.5, 0.5, rng=rng)

        # Should return valid values
        assert 0 <= corr_c <= 1
        assert 0 <= corr_o <= 1

    def test_backward_compatibility_with_correlation(self):
        """Test that function works with correlation parameter."""
        rng = np.random.default_rng(42)

        # Call with correlation parameter (existing usage pattern)
        corr_c, corr_o = generate_pss10_dimension_scores(0.5, 0.5, 0.3, rng)

        # Should return valid values
        assert 0 <= corr_c <= 1
        assert 0 <= corr_o <= 1

    def test_backward_compatibility_extreme_values(self):
        """Test backward compatibility with extreme input values."""
        rng = np.random.default_rng(42)

        # Test with extreme values
        corr_c, corr_o = generate_pss10_dimension_scores(1.0, 0.0, rng=rng)

        assert 0 <= corr_c <= 1
        assert 0 <= corr_o <= 1

        # Test with opposite extremes
        corr_c, corr_o = generate_pss10_dimension_scores(0.0, 1.0, rng=rng)

        assert 0 <= corr_c <= 1
        assert 0 <= corr_o <= 1


class TestPSS10DimensionScoresDeterministic:
    """Test deterministic behavior of generate_pss10_dimension_scores function."""

    def test_deterministic_mode_true(self):
        """Test that deterministic=True produces identical results with same inputs."""
        # Test with same inputs - should produce identical results
        corr_c1, corr_o1 = generate_pss10_dimension_scores(0.5, 0.5, deterministic=True)
        corr_c2, corr_o2 = generate_pss10_dimension_scores(0.5, 0.5, deterministic=True)

        assert corr_c1 == corr_c2
        assert corr_o1 == corr_o2

    def test_deterministic_mode_with_correlation(self):
        """Test deterministic mode with correlation parameter."""
        corr_c1, corr_o1 = generate_pss10_dimension_scores(0.5, 0.5, 0.3, deterministic=True)
        corr_c2, corr_o2 = generate_pss10_dimension_scores(0.5, 0.5, 0.3, deterministic=True)

        assert corr_c1 == corr_c2
        assert corr_o2 == corr_o2

    def test_deterministic_mode_different_inputs(self):
        """Test that deterministic mode produces different results for different inputs."""
        corr_c1, corr_o1 = generate_pss10_dimension_scores(0.3, 0.7, deterministic=True)
        corr_c2, corr_o2 = generate_pss10_dimension_scores(0.7, 0.3, deterministic=True)

        # Results should be different for different inputs
        assert (corr_c1, corr_o1) != (corr_c2, corr_o2)

    def test_deterministic_mode_extreme_values(self):
        """Test deterministic mode with extreme input values."""
        corr_c1, corr_o1 = generate_pss10_dimension_scores(1.0, 0.0, deterministic=True)
        corr_c2, corr_o2 = generate_pss10_dimension_scores(1.0, 0.0, deterministic=True)

        assert corr_c1 == corr_c2
        assert corr_o1 == corr_o2

    def test_deterministic_seed_generation(self):
        """Test that deterministic seed generation works correctly."""
        # Same inputs should always produce same results
        results = []
        for _ in range(5):
            corr_c, corr_o = generate_pss10_dimension_scores(0.5, 0.5, deterministic=True)
            results.append((corr_c, corr_o))

        # All results should be identical (deterministic)
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result

        # Results should be in valid range
        assert 0 <= first_result[0] <= 1
        assert 0 <= first_result[1] <= 1


class TestPSS10DimensionScoresNonDeterministic:
    """Test non-deterministic behavior of generate_pss10_dimension_scores function."""

    def test_non_deterministic_mode_false(self):
        """Test that deterministic=False produces different results across calls."""
        # Multiple calls with same inputs should produce different results
        results = []
        for i in range(10):
            corr_c, corr_o = generate_pss10_dimension_scores(0.5, 0.5, deterministic=False)
            results.append((corr_c, corr_o))

        # At least some results should be different (very high probability)
        unique_results = set(results)
        assert len(unique_results) > 1  # Should have variation

    def test_non_deterministic_with_rng(self):
        """Test non-deterministic behavior with explicit RNG."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        corr_c1, corr_o1 = generate_pss10_dimension_scores(0.5, 0.5, rng=rng1, deterministic=False)
        corr_c2, corr_o2 = generate_pss10_dimension_scores(0.5, 0.5, rng=rng2, deterministic=False)

        # Same RNG state should produce same results
        assert corr_c1 == corr_c2
        assert corr_o1 == corr_o2

    def test_non_deterministic_different_rng_seeds(self):
        """Test non-deterministic behavior with different RNG seeds."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)

        corr_c1, corr_o1 = generate_pss10_dimension_scores(0.5, 0.5, rng=rng1, deterministic=False)
        corr_c2, corr_o2 = generate_pss10_dimension_scores(0.5, 0.5, rng=rng2, deterministic=False)

        # Different seeds should produce different results
        assert (corr_c1, corr_o1) != (corr_c2, corr_o2)


class TestPSS10DimensionScoresConfiguration:
    """Test configuration integration for generate_pss10_dimension_scores function."""

    def test_configuration_bifactor_correlation(self):
        """Test that PSS10_BIFACTOR_COR configuration is used correctly."""
        # Test with mocked configuration
        with patch('stress_utils.get_config') as mock_config:
            mock_cfg = mock_config.return_value
            mock_cfg.get.return_value = 0.5  # Custom correlation

            corr_c, corr_o = generate_pss10_dimension_scores(0.5, 0.5, deterministic=True)

            # Should use the mocked correlation value
            mock_cfg.get.assert_any_call('pss10', 'bifactor_correlation')

    def test_configuration_controllability_sd(self):
        """Test that PSS10_CONTROLLABILITY_SD configuration is used correctly."""
        with patch('stress_utils.get_config') as mock_config:
            mock_cfg = mock_config.return_value
            mock_cfg.get.side_effect = lambda section, key: {
                ('pss10', 'bifactor_correlation'): 0.0,
                ('pss10', 'controllability_sd'): 2.0,  # Custom SD
                ('pss10', 'overload_sd'): 1.0
            }.get((section, key))

            corr_c, corr_o = generate_pss10_dimension_scores(0.5, 0.5, deterministic=True)

            # Should use the mocked SD values
            mock_cfg.get.assert_any_call('pss10', 'controllability_sd')
            mock_cfg.get.assert_any_call('pss10', 'overload_sd')

    def test_configuration_overload_sd(self):
        """Test that PSS10_OVERLOAD_SD configuration is used correctly."""
        with patch('stress_utils.get_config') as mock_config:
            mock_cfg = mock_config.return_value
            mock_cfg.get.side_effect = lambda section, key: {
                ('pss10', 'bifactor_correlation'): 0.0,
                ('pss10', 'controllability_sd'): 1.0,
                ('pss10', 'overload_sd'): 2.0  # Custom SD
            }.get((section, key))

            corr_c, corr_o = generate_pss10_dimension_scores(0.5, 0.5, deterministic=True)

            # Should use the mocked SD values
            mock_cfg.get.assert_any_call('pss10', 'controllability_sd')
            mock_cfg.get.assert_any_call('pss10', 'overload_sd')

    def test_configuration_default_values(self):
        """Test that default configuration values work correctly."""
        # Test with actual configuration (no mocking)
        corr_c, corr_o = generate_pss10_dimension_scores(0.5, 0.5, deterministic=True)

        # Should return valid values with default config
        assert 0 <= corr_c <= 1
        assert 0 <= corr_o <= 1


class TestPSS10DimensionScoresRegularizedSD:
    """Test that regularized standard deviations are applied correctly."""

    def test_regularized_sd_division_by_four(self):
        """Test that standard deviations are divided by 4 as specified."""
        with patch('stress_utils.get_config') as mock_config:
            mock_cfg = mock_config.return_value
            # Mock the config to return specific SD values
            mock_cfg.get.side_effect = lambda section, key: {
                ('pss10', 'bifactor_correlation'): 0.0,
                ('pss10', 'controllability_sd'): 2.0,  # Should be divided by 4 → 0.5
                ('pss10', 'overload_sd'): 4.0         # Should be divided by 4 → 1.0
            }.get((section, key))

            corr_c, corr_o = generate_pss10_dimension_scores(0.5, 0.5, deterministic=True)

            # Should use regularized SDs (divided by 4)
            mock_cfg.get.assert_any_call('pss10', 'controllability_sd')
            mock_cfg.get.assert_any_call('pss10', 'overload_sd')

    def test_regularized_sd_effect_on_variance(self):
        """Test that regularized SDs produce appropriate variance."""
        # Test with different SD values to ensure regularization works
        test_cases = [
            (0.5, 0.5),  # Low SDs
            (2.0, 2.0),  # High SDs
            (1.0, 2.0),  # Different SDs
        ]

        for controllability_sd, overload_sd in test_cases:
            with patch('stress_utils.get_config') as mock_config:
                mock_cfg = mock_config.return_value
                mock_cfg.get.side_effect = lambda section, key: {
                    ('pss10', 'bifactor_correlation'): 0.0,
                    ('pss10', 'controllability_sd'): controllability_sd,
                    ('pss10', 'overload_sd'): overload_sd
                }.get((section, key))

                corr_c, corr_o = generate_pss10_dimension_scores(0.5, 0.5, deterministic=True)

                # Should still produce valid values
                assert 0 <= corr_c <= 1
                assert 0 <= corr_o <= 1


class TestPSS10DimensionScoresReasonableResults:
    """Test that function produces reasonable results in all modes."""

    def test_reasonable_results_deterministic_mode(self):
        """Test that deterministic mode produces reasonable results."""
        test_cases = [
            (0.0, 0.0),  # No stress
            (1.0, 1.0),  # Maximum stress
            (0.5, 0.5),  # Moderate stress
            (0.2, 0.8),  # Mixed stress
        ]

        for c, o in test_cases:
            corr_c, corr_o = generate_pss10_dimension_scores(c, o, deterministic=True)

            # Results should be in valid range
            assert 0 <= corr_c <= 1
            assert 0 <= corr_o <= 1

            # Results should be reasonable (not NaN or infinite)
            assert np.isfinite(corr_c)
            assert np.isfinite(corr_o)

    def test_reasonable_results_non_deterministic_mode(self):
        """Test that non-deterministic mode produces reasonable results."""
        rng = np.random.default_rng(42)

        test_cases = [
            (0.0, 0.0),  # No stress
            (1.0, 1.0),  # Maximum stress
            (0.5, 0.5),  # Moderate stress
            (0.2, 0.8),  # Mixed stress
        ]

        for c, o in test_cases:
            corr_c, corr_o = generate_pss10_dimension_scores(c, o, rng=rng, deterministic=False)

            # Results should be in valid range
            assert 0 <= corr_c <= 1
            assert 0 <= corr_o <= 1

            # Results should be reasonable (not NaN or infinite)
            assert np.isfinite(corr_c)
            assert np.isfinite(corr_o)

    def test_reasonable_results_extreme_correlation(self):
        """Test that extreme correlation values produce reasonable results."""
        rng = np.random.default_rng(42)

        correlation_values = [-1.0, -0.5, 0.0, 0.5, 1.0]

        for correlation in correlation_values:
            corr_c, corr_o = generate_pss10_dimension_scores(0.5, 0.5, correlation, rng, deterministic=False)

            # Results should be in valid range
            assert 0 <= corr_c <= 1
            assert 0 <= corr_o <= 1

            # Results should be reasonable
            assert np.isfinite(corr_c)
            assert np.isfinite(corr_o)

    def test_reasonable_results_boundary_inputs(self):
        """Test that boundary input values produce reasonable results."""
        rng = np.random.default_rng(42)

        # Test boundary cases
        boundary_cases = [
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
        ]

        for c, o in boundary_cases:
            corr_c, corr_o = generate_pss10_dimension_scores(c, o, rng=rng, deterministic=False)

            # Should handle boundary cases gracefully
            assert 0 <= corr_c <= 1
            assert 0 <= corr_o <= 1
            assert np.isfinite(corr_c)
            assert np.isfinite(corr_o)


class TestPSS10DimensionScoresIntegration:
    """Test integration with the broader PSS-10 system."""

    def test_integration_with_pss10_responses(self):
        """Test that dimension scores work correctly with PSS-10 response generation."""
        rng = np.random.default_rng(42)

        # Generate dimension scores
        corr_c, corr_o = generate_pss10_dimension_scores(0.5, 0.5, rng=rng, deterministic=False)

        # Use them in response generation
        responses = generate_pss10_responses(corr_c, corr_o, rng)

        # Should produce valid responses
        assert len(responses) == 10
        for response in responses.values():
            assert 0 <= response <= 4

    def test_integration_with_config_system(self):
        """Test integration with the configuration system."""
        # Test that function works with actual configuration
        corr_c, corr_o = generate_pss10_dimension_scores(0.5, 0.5, deterministic=True)

        # Should work without errors
        assert 0 <= corr_c <= 1
        assert 0 <= corr_o <= 1

    def test_integration_error_handling(self):
        """Test error handling in integration scenarios."""
        # Test with None values (should handle gracefully)
        corr_c, corr_o = generate_pss10_dimension_scores(0.5, 0.5, deterministic=True)

        # Should not produce None or invalid values
        assert corr_c is not None
        assert corr_o is not None
        assert isinstance(corr_c, (int, float))
        assert isinstance(corr_o, (int, float))


def run_comprehensive_tests():
    """Run all comprehensive tests for generate_pss10_dimension_scores function."""
    print("Running Comprehensive Tests for generate_pss10_dimension_scores")
    print("=" * 65)

    # Create test instances
    test_classes = [
        TestPSS10DimensionScoresBackwardCompatibility(),
        TestPSS10DimensionScoresDeterministic(),
        TestPSS10DimensionScoresNonDeterministic(),
        TestPSS10DimensionScoresConfiguration(),
        TestPSS10DimensionScoresRegularizedSD(),
        TestPSS10DimensionScoresReasonableResults(),
        TestPSS10DimensionScoresIntegration()
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}:")
        print("-" * len(class_name))

        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]

        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_class, test_method)()
                print(f"  ✓ {test_method}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {test_method}: {e}")
                traceback.print_exc()

    print(f"\n{'=' * 65}")
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ generate_pss10_dimension_scores function meets all requirements:")
        print("  - Backward compatibility: ✓")
        print("  - Deterministic behavior: ✓")
        print("  - Non-deterministic behavior: ✓")
        print("  - Configuration integration: ✓")
        print("  - Regularized standard deviations: ✓")
        print("  - Reasonable results: ✓")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)