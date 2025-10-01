"""
Comprehensive unit tests for math_utils.py

This file provides complete test coverage for all mathematical utility functions
with edge cases, boundary conditions, and reproducible testing.
"""

import pytest
import numpy as np
from src.python.math_utils import (
    create_rng, clamp, normalize_to_range, sigmoid, softmax, sample_poisson,
    sample_beta, sample_exponential, compute_running_average, compute_percentile,
    compute_z_score, logistic_function, linear_interpolation, calculate_entropy,
    normalize_probabilities, RNGConfig
)


class TestRandomNumberGeneration:
    """Test random number generation utilities."""

    def test_create_rng_with_seed(self):
        """Test RNG creation with specific seed for reproducibility."""
        rng1 = create_rng(42)
        rng2 = create_rng(42)

        # Should produce identical sequences
        assert rng1.random() == rng2.random()
        assert rng1.random() == rng2.random()

    def test_create_rng_without_seed(self):
        """Test RNG creation without seed."""
        rng = create_rng()
        assert rng is not None
        assert hasattr(rng, 'random')

    def test_rng_config_dataclass(self):
        """Test RNGConfig dataclass."""
        config = RNGConfig(seed=123)
        assert config.seed == 123
        assert config.generator is None

        # Test with generator
        rng = create_rng(456)
        config_with_rng = RNGConfig(seed=456, generator=rng)
        assert config_with_rng.generator is rng


class TestClampingAndNormalization:
    """Test clamping and normalization functions."""

    def test_clamp_basic(self):
        """Test basic clamping functionality."""
        assert clamp(0.5) == 0.5  # Within default range
        assert clamp(-0.1) == 0.0  # Below minimum
        assert clamp(1.5) == 1.0   # Above maximum

    def test_clamp_custom_bounds(self):
        """Test clamping with custom bounds."""
        assert clamp(5, 0, 10) == 5
        assert clamp(-5, 0, 10) == 0
        assert clamp(15, 0, 10) == 10

    def test_clamp_preserves_type(self):
        """Test that clamp preserves input type."""
        assert isinstance(clamp(0.5), float)
        # Note: clamp always returns Python float, not numpy types
        assert isinstance(clamp(np.float64(0.5)), float)

    def test_normalize_to_range_basic(self):
        """Test basic range normalization."""
        # Normalize from [0, 10] to [0, 1]
        result = normalize_to_range(5.0, 0, 10, 0, 1)
        assert abs(result - 0.5) < 1e-10

    def test_normalize_to_range_custom_ranges(self):
        """Test normalization with custom ranges."""
        # Normalize from [-5, 5] to [0, 100]
        result = normalize_to_range(0.0, -5, 5, 0, 100)
        assert abs(result - 50.0) < 1e-10

    def test_normalize_to_range_identical_bounds(self):
        """Test normalization when old_min == old_max."""
        result = normalize_to_range(5.0, 5, 5, 0, 1)
        assert result == 0  # Should return new_min

    def test_normalize_to_range_edge_cases(self):
        """Test normalization edge cases."""
        # Test boundary values
        result_min = normalize_to_range(0.0, 0, 10, 0, 1)
        result_max = normalize_to_range(10.0, 0, 10, 0, 1)

        assert abs(result_min - 0.0) < 1e-10
        assert abs(result_max - 1.0) < 1e-10


class TestActivationFunctions:
    """Test activation and mathematical functions."""

    def test_sigmoid_basic(self):
        """Test sigmoid function basic properties."""
        assert sigmoid(0.0) == 0.5  # Sigmoid(0) = 0.5
        assert sigmoid(10.0) > 0.99  # Large positive → close to 1.0
        assert sigmoid(-10.0) < 0.01  # Large negative → close to 0.0

    def test_sigmoid_with_gamma(self):
        """Test sigmoid with custom gamma parameter."""
        # Higher gamma should be steeper
        result_low = sigmoid(1.0, gamma=1.0)
        result_high = sigmoid(1.0, gamma=10.0)

        assert result_high > result_low  # Higher gamma = steeper = higher output

    def test_sigmoid_extreme_values(self):
        """Test sigmoid with extreme values."""
        assert 0.0 <= sigmoid(1000.0) <= 1.0
        assert 0.0 <= sigmoid(-1000.0) <= 1.0

    def test_softmax_basic(self):
        """Test softmax function basic properties."""
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)

        # Should sum to 1
        assert abs(np.sum(result) - 1.0) < 1e-10
        # All values should be positive
        assert np.all(result > 0)
        # Should be in ascending order (higher input = higher probability)
        assert result[0] < result[1] < result[2]

    def test_softmax_with_temperature(self):
        """Test softmax with temperature parameter."""
        x = np.array([1.0, 2.0, 3.0])

        # High temperature should make distribution more uniform
        result_low_temp = softmax(x, temperature=0.1)
        result_high_temp = softmax(x, temperature=10.0)

        # Low temperature should be more peaked
        assert np.var(result_low_temp) > np.var(result_high_temp)

    def test_softmax_temperature_zero(self):
        """Test softmax with temperature = 0 (deterministic)."""
        x = np.array([1.0, 3.0, 2.0])
        result = softmax(x, temperature=0.0)

        # Should return one-hot of maximum value
        max_idx = np.argmax(x)
        expected = np.zeros_like(x)
        expected[max_idx] = 1.0

        np.testing.assert_array_equal(result, expected)

    def test_softmax_single_value(self):
        """Test softmax with single value."""
        x = np.array([2.0])
        result = softmax(x)
        assert result[0] == 1.0

    def test_softmax_identical_values(self):
        """Test softmax with identical values."""
        x = np.array([2.0, 2.0, 2.0])
        result = softmax(x)

        # All should be equal
        expected = np.array([1/3, 1/3, 1/3])
        np.testing.assert_array_almost_equal(result, expected)


class TestSamplingFunctions:
    """Test statistical sampling functions."""

    def test_sample_poisson_basic(self):
        """Test Poisson sampling basic functionality."""
        rng = create_rng(42)

        # Test with different lambda values
        sample1 = sample_poisson(1.0, rng)
        sample2 = sample_poisson(5.0, rng)

        assert isinstance(sample1, (int, np.integer))
        assert isinstance(sample2, (int, np.integer))
        assert sample1 >= 0  # Should never be negative
        assert sample2 >= 0

    def test_sample_poisson_with_min_value(self):
        """Test Poisson sampling with minimum value constraint."""
        rng = create_rng(42)

        # Test that minimum value is enforced
        samples = [sample_poisson(0.1, rng, min_value=2) for _ in range(100)]
        assert all(s >= 2 for s in samples)

    def test_sample_poisson_deterministic(self):
        """Test Poisson sampling determinism with fixed RNG."""
        rng1 = create_rng(42)
        rng2 = create_rng(42)

        sample1 = sample_poisson(2.0, rng1, min_value=0)
        sample2 = sample_poisson(2.0, rng2, min_value=0)

        assert sample1 == sample2

    def test_sample_beta_basic(self):
        """Test Beta distribution sampling."""
        rng = create_rng(42)

        sample = sample_beta(2.0, 3.0, rng)

        assert 0.0 <= sample <= 1.0
        assert isinstance(sample, (float, np.floating))

    def test_sample_beta_deterministic(self):
        """Test Beta sampling determinism."""
        rng1 = create_rng(42)
        rng2 = create_rng(42)

        sample1 = sample_beta(2.0, 3.0, rng1)
        sample2 = sample_beta(2.0, 3.0, rng2)

        assert sample1 == sample2

    def test_sample_exponential_basic(self):
        """Test exponential distribution sampling."""
        rng = create_rng(42)

        sample = sample_exponential(1.0, rng)

        assert sample >= 0.0
        assert isinstance(sample, (float, np.floating))

    def test_sample_exponential_with_max_value(self):
        """Test exponential sampling with maximum value constraint."""
        rng = create_rng(42)

        samples = [sample_exponential(1.0, rng, max_value=2.0) for _ in range(100)]
        assert all(s <= 2.0 for s in samples)

    def test_sample_exponential_deterministic(self):
        """Test exponential sampling determinism."""
        rng1 = create_rng(42)
        rng2 = create_rng(42)

        sample1 = sample_exponential(1.0, rng1, max_value=10.0)
        sample2 = sample_exponential(1.0, rng2, max_value=10.0)

        assert sample1 == sample2


class TestStatisticalFunctions:
    """Test statistical computation functions."""

    def test_compute_running_average_count_based(self):
        """Test count-based running average."""
        # Test with multiple values
        avg1 = compute_running_average(0.0, 10.0, 1)  # First value
        assert avg1 == 10.0

        avg2 = compute_running_average(avg1, 20.0, 2)  # Second value
        assert abs(avg2 - 15.0) < 1e-10

        avg3 = compute_running_average(avg2, 30.0, 3)  # Third value
        assert abs(avg3 - 20.0) < 1e-10

    def test_compute_running_average_exponential(self):
        """Test exponential moving average."""
        alpha = 0.2

        # First value should be weighted combination of current_avg and new_value
        avg1 = compute_running_average(0.0, 10.0, 1, alpha)
        expected1 = alpha * 10.0 + (1 - alpha) * 0.0  # 2.0
        assert abs(avg1 - expected1) < 1e-10

        avg2 = compute_running_average(avg1, 20.0, 2, alpha)
        expected2 = alpha * 20.0 + (1 - alpha) * avg1  # 0.2 * 20.0 + 0.8 * 2.0 = 5.6
        assert abs(avg2 - expected2) < 1e-10

        # Test with different alpha value
        avg3 = compute_running_average(avg2, 30.0, 3, alpha)
        expected3 = alpha * 30.0 + (1 - alpha) * avg2
        assert abs(avg3 - expected3) < 1e-10

        # Test with alpha = 1.0 (should return current value)
        avg4 = compute_running_average(5.0, 10.0, 1, 1.0)
        assert avg4 == 10.0

    def test_compute_percentile_basic(self):
        """Test percentile computation."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        median = compute_percentile(values, 50)
        assert abs(median - 5.5) < 1e-10

        q1 = compute_percentile(values, 25)
        assert abs(q1 - 3.25) < 1e-10

    def test_compute_percentile_empty_list(self):
        """Test percentile computation with empty list."""
        with pytest.raises(ValueError, match="Cannot compute percentile of empty list"):
            compute_percentile([], 50)

    def test_compute_percentile_edge_cases(self):
        """Test percentile edge cases."""
        values = [5.0]

        # Single value should return itself
        assert compute_percentile(values, 50) == 5.0
        assert compute_percentile(values, 0) == 5.0
        assert compute_percentile(values, 100) == 5.0

    def test_compute_z_score_basic(self):
        """Test z-score computation."""
        value = 10.0
        mean = 7.0
        std = 2.0

        z_score = compute_z_score(value, mean, std)
        expected = (value - mean) / std
        assert abs(z_score - expected) < 1e-10

    def test_compute_z_score_zero_std(self):
        """Test z-score with zero standard deviation."""
        z_score = compute_z_score(5.0, 5.0, 0.0)
        assert z_score == 0.0

    def test_compute_z_score_with_ddof(self):
        """Test z-score computation respects ddof parameter."""
        # This is mainly tested indirectly since the function uses numpy
        # but we can verify it doesn't crash
        z_score = compute_z_score(10.0, 7.0, 2.0, ddof=1)
        assert isinstance(z_score, (int, float))


class TestMathematicalFunctions:
    """Test general mathematical functions."""

    def test_logistic_function_basic(self):
        """Test logistic function basic properties."""
        # Test default parameters
        result = logistic_function(0.0)
        assert abs(result - 0.5) < 1e-10

        # Test with custom parameters
        result = logistic_function(1.0, L=2.0, k=1.0, x0=0.0)
        expected = 2.0 / (1 + np.exp(-1.0))
        assert abs(result - expected) < 1e-10

    def test_logistic_function_extreme_values(self):
        """Test logistic function with extreme values."""
        assert logistic_function(1000.0) > 0.99
        assert logistic_function(-1000.0) < 0.01

    def test_linear_interpolation_basic(self):
        """Test linear interpolation."""
        # Interpolate between (0, 0) and (10, 10) at x=5
        result = linear_interpolation(5.0, 0, 0, 10, 10)
        assert abs(result - 5.0) < 1e-10

    def test_linear_interpolation_identical_x(self):
        """Test linear interpolation with identical x values."""
        result = linear_interpolation(5.0, 5, 10, 5, 20)
        assert result == 10  # Should return y1

    def test_calculate_entropy_basic(self):
        """Test entropy calculation."""
        # Uniform distribution should have high entropy
        uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
        entropy_uniform = calculate_entropy(uniform_probs)

        # Concentrated distribution should have low entropy
        concentrated_probs = np.array([1.0, 0.0, 0.0, 0.0])
        entropy_concentrated = calculate_entropy(concentrated_probs)

        assert entropy_uniform > entropy_concentrated

    def test_calculate_entropy_empty_array(self):
        """Test entropy calculation with empty array."""
        entropy = calculate_entropy(np.array([]))
        assert entropy == 0.0

    def test_calculate_entropy_zero_probabilities(self):
        """Test entropy calculation with zero probabilities."""
        probs = np.array([0.5, 0.0, 0.5])
        entropy = calculate_entropy(probs)
        assert entropy >= 0.0  # Should handle zeros gracefully

    def test_normalize_probabilities_softmax(self):
        """Test probability normalization with softmax method."""
        values = np.array([1.0, 2.0, 3.0])
        probs = normalize_probabilities(values, method='softmax')

        assert abs(np.sum(probs) - 1.0) < 1e-10
        assert np.all(probs > 0)

    def test_normalize_probabilities_clip(self):
        """Test probability normalization with clip method."""
        values = np.array([0.5, -0.1, 1.5])
        probs = normalize_probabilities(values, method='clip')

        assert abs(np.sum(probs) - 1.0) < 1e-10
        assert np.all(probs >= 0)

    def test_normalize_probabilities_linear(self):
        """Test probability normalization with linear method."""
        values = np.array([1.0, 3.0, 2.0])
        probs = normalize_probabilities(values, method='linear')

        assert abs(np.sum(probs) - 1.0) < 1e-10
        assert np.all(probs >= 0)

    def test_normalize_probabilities_identical_values(self):
        """Test probability normalization with identical values."""
        values = np.array([2.0, 2.0, 2.0])
        probs = normalize_probabilities(values, method='linear')

        expected = np.array([1/3, 1/3, 1/3])
        np.testing.assert_array_almost_equal(probs, expected)

    def test_normalize_probabilities_unknown_method(self):
        """Test probability normalization with unknown method."""
        values = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_probabilities(values, method='unknown')


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_all_functions_handle_none_inputs_gracefully(self):
        """Test that functions handle None inputs appropriately."""
        # Most functions should handle None rng parameters
        # by creating their own RNG

        # Test functions that accept rng parameter
        result = sample_poisson(1.0, None)
        assert isinstance(result, (int, np.integer))

        result = sample_beta(1.0, 1.0, None)
        assert 0.0 <= result <= 1.0

        result = sample_exponential(1.0, None)
        assert result >= 0.0

    def test_functions_with_array_inputs(self):
        """Test functions handle various array input types."""
        # Test with different numpy array types
        values_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        values_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Softmax should work with both
        result_f64 = softmax(values_f64)
        result_f32 = softmax(values_f32)

        assert result_f64.dtype == np.float64
        assert result_f32.dtype == np.float32

    def test_numerical_stability(self):
        """Test numerical stability of functions."""
        # Test with very large numbers
        large_values = np.array([1000.0, 2000.0, 3000.0])
        result = softmax(large_values)
        assert np.all(np.isfinite(result))
        assert abs(np.sum(result) - 1.0) < 1e-10

        # Test with very small numbers
        small_values = np.array([0.001, 0.002, 0.003])
        result = softmax(small_values)
        assert np.all(np.isfinite(result))
        assert abs(np.sum(result) - 1.0) < 1e-10


# Example of how to run these tests:
# pytest src/python/tests/test_math_utils.py -v
# pytest src/python/tests/test_math_utils.py::TestClampingAndNormalization::test_clamp_basic -v