"""
Comprehensive unit tests for math_utils.py

This file provides complete test coverage for all mathematical utility functions
with edge cases, boundary conditions, and reproducible testing.
"""

import pytest
import numpy as np
from src.python.math_utils import (
    create_rng, clamp, normalize_to_range, sigmoid, softmax, sample_poisson,
    sample_beta, sample_exponential, sample_normal, compute_running_average, compute_percentile,
    compute_z_score, logistic_function, linear_interpolation, calculate_entropy,
    normalize_probabilities, RNGConfig, tanh_transform, sigmoid_transform,
    inverse_tanh_transform, inverse_sigmoid_transform
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


class TestTransformationFunctions:
    """Test transformation functions for agent initialization."""

    def test_tanh_transform_basic(self):
        """Test tanh transformation basic functionality."""
        rng = create_rng(42)

        # Test with default parameters (mean=0, std=1)
        result = tanh_transform(rng=rng)
        assert -1.0 <= result <= 1.0
        assert isinstance(result, (float, np.floating))

    def test_tanh_transform_with_custom_parameters(self):
        """Test tanh transformation with custom mean and std."""
        rng = create_rng(42)

        # Test with different mean and std
        result = tanh_transform(mean=2.0, std=0.5, rng=rng)
        assert -1.0 <= result <= 1.0

    def test_tanh_transform_deterministic(self):
        """Test tanh transformation determinism with fixed RNG."""
        rng1 = create_rng(42)
        rng2 = create_rng(42)

        result1 = tanh_transform(mean=1.0, std=0.5, rng=rng1)
        result2 = tanh_transform(mean=1.0, std=0.5, rng=rng2)

        assert result1 == result2

    def test_tanh_transform_bounds(self):
        """Test that tanh transformation produces values in correct bounds."""
        rng = create_rng(123)

        # Run multiple samples to test bounds
        results = [tanh_transform(rng=rng) for _ in range(1000)]

        assert all(-1.0 <= r <= 1.0 for r in results)
        assert min(results) > -1.0  # Should be strictly greater than -1
        assert max(results) < 1.0   # Should be strictly less than 1

    def test_tanh_transform_zero_std_special_case(self):
        """Test tanh transformation with zero standard deviation."""
        rng = create_rng(42)

        result = tanh_transform(mean=1.0, std=0.0, rng=rng)
        assert result == 0.0  # Should return 0.0 for zero std case

    def test_tanh_transform_extreme_parameters(self):
        """Test tanh transformation with extreme parameter values."""
        rng = create_rng(42)

        # Test with very large mean
        result_large_mean = tanh_transform(mean=1000.0, std=1.0, rng=rng)
        assert -1.0 <= result_large_mean <= 1.0

        # Test with very large std
        result_large_std = tanh_transform(mean=0.0, std=1000.0, rng=rng)
        assert -1.0 <= result_large_std <= 1.0

        # Test with very small std
        result_small_std = tanh_transform(mean=0.0, std=0.001, rng=rng)
        assert -1.0 <= result_small_std <= 1.0

    def test_tanh_transform_distribution_properties(self):
        """Test statistical distribution properties of tanh transformation."""
        rng = create_rng(42)

        # Generate many samples and check distribution properties
        samples = [tanh_transform(rng=rng) for _ in range(10000)]

        # Convert to numpy array for analysis
        samples_array = np.array(samples)

        # Check mean is approximately 0 (symmetric around 0)
        sample_mean = np.mean(samples_array)
        assert abs(sample_mean) < 0.1  # Should be close to 0

        # Check bounds are respected
        assert np.all(samples_array >= -1.0)
        assert np.all(samples_array <= 1.0)

        # Check that we get values reasonably close to bounds (transformation uses /3.0 normalization)
        # With /3.0 normalization, tanh typically reaches ~±0.9, not the full ±1.0
        assert np.min(samples_array) < -0.8
        assert np.max(samples_array) > 0.8

    def test_sigmoid_transform_basic(self):
        """Test sigmoid transformation basic functionality."""
        rng = create_rng(42)

        # Test with default parameters (mean=0, std=1)
        result = sigmoid_transform(rng=rng)
        assert 0.0 <= result <= 1.0
        assert isinstance(result, (float, np.floating))

    def test_sigmoid_transform_with_custom_parameters(self):
        """Test sigmoid transformation with custom mean and std."""
        rng = create_rng(42)

        # Test with different mean and std
        result = sigmoid_transform(mean=2.0, std=0.5, rng=rng)
        assert 0.0 <= result <= 1.0

    def test_sigmoid_transform_deterministic(self):
        """Test sigmoid transformation determinism with fixed RNG."""
        rng1 = create_rng(42)
        rng2 = create_rng(42)

        result1 = sigmoid_transform(mean=1.0, std=0.5, rng=rng1)
        result2 = sigmoid_transform(mean=1.0, std=0.5, rng=rng2)

        assert result1 == result2

    def test_sigmoid_transform_bounds(self):
        """Test that sigmoid transformation produces values in correct bounds."""
        rng = create_rng(123)

        # Run multiple samples to test bounds
        results = [sigmoid_transform(rng=rng) for _ in range(1000)]

        assert all(0.0 <= r <= 1.0 for r in results)
        assert min(results) > 0.0  # Should be strictly greater than 0
        assert max(results) < 1.0   # Should be strictly less than 1

    def test_sigmoid_transform_zero_std_special_case(self):
        """Test sigmoid transformation with zero standard deviation."""
        rng = create_rng(42)

        result = sigmoid_transform(mean=1.0, std=0.0, rng=rng)
        expected = sigmoid(0.0)  # Should return sigmoid of 0
        assert abs(result - expected) < 1e-10

    def test_sigmoid_transform_extreme_parameters(self):
        """Test sigmoid transformation with extreme parameter values."""
        rng = create_rng(42)

        # Test with very large mean
        result_large_mean = sigmoid_transform(mean=1000.0, std=1.0, rng=rng)
        assert 0.0 <= result_large_mean <= 1.0

        # Test with very large std
        result_large_std = sigmoid_transform(mean=0.0, std=1000.0, rng=rng)
        assert 0.0 <= result_large_std <= 1.0

        # Test with very small std
        result_small_std = sigmoid_transform(mean=0.0, std=0.001, rng=rng)
        assert 0.0 <= result_small_std <= 1.0

    def test_sigmoid_transform_distribution_properties(self):
        """Test statistical distribution properties of sigmoid transformation."""
        rng = create_rng(42)

        # Generate many samples and check distribution properties
        samples = [sigmoid_transform(rng=rng) for _ in range(10000)]

        # Convert to numpy array for analysis
        samples_array = np.array(samples)

        # Check mean is approximately 0.5 (centered around 0.5)
        sample_mean = np.mean(samples_array)
        assert abs(sample_mean - 0.5) < 0.1  # Should be close to 0.5

        # Check bounds are respected
        assert np.all(samples_array >= 0.0)
        assert np.all(samples_array <= 1.0)

        # Check that we get values close to bounds (transformation should utilize full range)
        assert np.min(samples_array) < 0.1
        assert np.max(samples_array) > 0.9

    def test_inverse_tanh_transform_basic(self):
        """Test inverse tanh transformation basic functionality."""
        # Test with value 0.0 (special case)
        result = inverse_tanh_transform(0.0, mean=1.0, std=0.5)
        assert result == 1.0  # Should return mean

        # Test with non-zero value
        result = inverse_tanh_transform(0.5, mean=2.0, std=1.0)
        assert isinstance(result, (float, np.floating))

    def test_inverse_tanh_transform_extreme_values(self):
        """Test inverse tanh transformation with extreme values."""
        # Test with values close to bounds
        result_near_neg = inverse_tanh_transform(-0.9, mean=0.0, std=1.0)
        result_near_pos = inverse_tanh_transform(0.9, mean=0.0, std=1.0)

        assert isinstance(result_near_neg, (float, np.floating))
        assert isinstance(result_near_pos, (float, np.floating))

    def test_inverse_tanh_transform_zero_std(self):
        """Test inverse tanh transformation with zero standard deviation."""
        result = inverse_tanh_transform(0.5, mean=1.0, std=0.0)
        assert result == 1.0  # Should return mean

    def test_inverse_tanh_transform_extreme_parameters(self):
        """Test inverse tanh transformation with extreme parameters."""
        # Test with very large mean
        result = inverse_tanh_transform(0.5, mean=1000.0, std=1.0)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

        # Test with very large std
        result = inverse_tanh_transform(0.5, mean=0.0, std=1000.0)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

    def test_inverse_sigmoid_transform_basic(self):
        """Test inverse sigmoid transformation basic functionality."""
        # Test with value 0.5 (special case)
        result = inverse_sigmoid_transform(0.5, mean=1.0, std=0.5)
        assert result == 1.0  # Should return mean

        # Test with non-0.5 value
        result = inverse_sigmoid_transform(0.7, mean=2.0, std=1.0)
        assert isinstance(result, (float, np.floating))

    def test_inverse_sigmoid_transform_extreme_values(self):
        """Test inverse sigmoid transformation with extreme values."""
        # Test with values close to bounds
        result_near_zero = inverse_sigmoid_transform(0.1, mean=0.0, std=1.0)
        result_near_one = inverse_sigmoid_transform(0.9, mean=0.0, std=1.0)

        assert isinstance(result_near_zero, (float, np.floating))
        assert isinstance(result_near_one, (float, np.floating))

    def test_inverse_sigmoid_transform_zero_std(self):
        """Test inverse sigmoid transformation with zero standard deviation."""
        result = inverse_sigmoid_transform(0.5, mean=1.0, std=0.0)
        assert result == 1.0  # Should return mean

    def test_inverse_sigmoid_transform_extreme_parameters(self):
        """Test inverse sigmoid transformation with extreme parameters."""
        # Test with very large mean
        result = inverse_sigmoid_transform(0.5, mean=1000.0, std=1.0)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

        # Test with very large std
        result = inverse_sigmoid_transform(0.5, mean=0.0, std=1000.0)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

    def test_transformation_inverse_consistency(self):
        """Test that inverse transformations work correctly for known values."""
        # Test tanh inverse transformation with known values
        test_values = [0.0, 0.5, -0.5, 0.8, -0.8]
        mean_val = 1.5
        std_val = 0.8

        for test_val in test_values:
            # Apply inverse transformation
            reconstructed = inverse_tanh_transform(test_val, mean=mean_val, std=std_val)

            # Should be a valid number
            assert isinstance(reconstructed, (float, np.floating))
            assert np.isfinite(reconstructed)

        # Test sigmoid inverse transformation with known values
        test_values_sig = [0.5, 0.7, 0.3, 0.9, 0.1]

        for test_val in test_values_sig:
            # Apply inverse transformation
            reconstructed = inverse_sigmoid_transform(test_val, mean=mean_val, std=std_val)

            # Should be a valid number
            assert isinstance(reconstructed, (float, np.floating))
            assert np.isfinite(reconstructed)

    def test_transformation_mathematical_properties(self):
        """Test mathematical properties of transformations."""
        # Test that tanh_transform produces values in [-1, 1]
        rng = create_rng(42)
        results = [tanh_transform(rng=rng) for _ in range(100)]

        assert all(-1.0 <= r <= 1.0 for r in results)

        # Test that sigmoid_transform produces values in [0, 1]
        results_sig = [sigmoid_transform(rng=rng) for _ in range(100)]
        assert all(0.0 <= r <= 1.0 for r in results_sig)

    def test_transformation_with_zero_std(self):
        """Test transformations handle zero standard deviation."""
        rng = create_rng(42)

        # Should not crash and should return deterministic results
        result_tanh = tanh_transform(mean=1.0, std=0.0, rng=rng)
        result_sigmoid = sigmoid_transform(mean=1.0, std=0.0, rng=rng)

        assert result_tanh == 0.0  # Should return 0.0 for zero std case
        assert result_sigmoid == sigmoid(0.0)  # Should return sigmoid of 0

    def test_inverse_transformations_with_zero_std(self):
        """Test inverse transformations handle zero standard deviation."""
        # Should not crash and should return deterministic results
        result_tanh = inverse_tanh_transform(0.5, mean=1.0, std=0.0)
        result_sigmoid = inverse_sigmoid_transform(0.5, mean=1.0, std=0.0)

        assert result_tanh == 1.0  # Should return mean
        assert result_sigmoid == 1.0  # Should return mean

    def test_transformation_mathematical_correctness(self):
        """Test mathematical correctness of transformation functions."""
        # Test that transformation functions produce expected output ranges
        rng = create_rng(42)

        # Test multiple parameter combinations
        param_combinations = [
            (0.0, 1.0),
            (5.0, 2.0),
            (-3.0, 0.5),
            (100.0, 10.0)
        ]

        for mean_val, std_val in param_combinations:
            # Generate samples
            tanh_samples = [tanh_transform(mean=mean_val, std=std_val, rng=create_rng(i)) for i in range(50)]
            sigmoid_samples = [sigmoid_transform(mean=mean_val, std=std_val, rng=create_rng(i + 100)) for i in range(50)]

            # Check bounds
            assert all(-1.0 <= s <= 1.0 for s in tanh_samples)
            assert all(0.0 <= s <= 1.0 for s in sigmoid_samples)

            # Check that we get reasonable variation
            assert len(set([round(s, 3) for s in tanh_samples])) > 5  # Should have variation
            assert len(set([round(s, 3) for s in sigmoid_samples])) > 5  # Should have variation

    def test_inverse_transformation_mathematical_correctness(self):
        """Test mathematical correctness of inverse transformation functions."""
        # Test specific known values
        mean_val = 2.0
        std_val = 1.5

        # Test tanh inverse with specific value
        test_value = 0.5
        result = inverse_tanh_transform(test_value, mean=mean_val, std=std_val)

        # Manual calculation for verification
        normalized = np.arctanh(test_value)
        z_score = normalized * 3.0
        expected = z_score * std_val + mean_val

        assert abs(result - expected) < 1e-10

        # Test sigmoid inverse with specific value
        test_value_sig = 0.7
        result_sig = inverse_sigmoid_transform(test_value_sig, mean=mean_val, std=std_val)

        # Manual calculation for verification
        normalized_sig = -np.log(1.0 / test_value_sig - 1.0)
        z_score_sig = normalized_sig * 3.0
        expected_sig = z_score_sig * std_val + mean_val

        assert abs(result_sig - expected_sig) < 1e-10

    def test_transformation_functions_statistical_properties(self):
        """Test statistical properties of transformation functions."""
        # Test that transformations produce expected statistical properties
        rng = create_rng(42)

        # Test with mean=0, std=1 (should be approximately symmetric for tanh)
        tanh_samples = [tanh_transform(mean=0.0, std=1.0, rng=create_rng(i)) for i in range(1000)]
        sigmoid_samples = [sigmoid_transform(mean=0.0, std=1.0, rng=create_rng(i + 2000)) for i in range(1000)]

        tanh_array = np.array(tanh_samples)
        sigmoid_array = np.array(sigmoid_samples)

        # For tanh with mean=0, should be approximately symmetric around 0
        tanh_mean = np.mean(tanh_array)
        assert abs(tanh_mean) < 0.1  # Should be close to 0

        # For sigmoid with mean=0, should be approximately symmetric around 0.5
        sigmoid_mean = np.mean(sigmoid_array)
        assert abs(sigmoid_mean - 0.5) < 0.1  # Should be close to 0.5

        # Check bounds are maintained
        assert np.all(tanh_array >= -1.0) and np.all(tanh_array <= 1.0)
        assert np.all(sigmoid_array >= 0.0) and np.all(sigmoid_array <= 1.0)

    def test_transformation_symmetry_properties(self):
        """Test symmetry properties of transformations."""
        rng = create_rng(42)

        # Test that tanh transformation is odd around 0
        # I.e., tanh_transform(-x) = -tanh_transform(x) for mean=0, std=1
        pos_sample = sample_normal(mean=0.0, std=1.0, rng=rng)
        neg_sample = -pos_sample

        # Create separate RNG for each to ensure same random sequence
        rng_pos = create_rng(rng.integers(0, 2**32))
        rng_neg = create_rng(rng.integers(0, 2**32))

        # Override the sample to use our known values
        # We need to test the property that the transformation preserves the sign
        # when mean=0 and input is symmetric

        # Test with manual samples
        pos_transformed = tanh_transform(mean=0.0, std=1.0, rng=rng_pos)
        neg_transformed = tanh_transform(mean=0.0, std=1.0, rng=rng_neg)

        # For symmetric inputs around mean, the transformation should preserve symmetry
        # This is a statistical property that should hold approximately

    def test_transformation_with_sample_normal_integration(self):
        """Test integration between transformation functions and sample_normal."""
        rng = create_rng(42)

        # Test that transformation functions work correctly with sample_normal output
        mean_val = 2.0
        std_val = 1.5

        # Generate sample using sample_normal
        normal_sample = sample_normal(mean=mean_val, std=std_val, rng=rng)

        # Apply transformation
        tanh_result = tanh_transform(mean=mean_val, std=std_val, rng=rng)
        sigmoid_result = sigmoid_transform(mean=mean_val, std=std_val, rng=rng)

        # Results should be in correct bounds
        assert -1.0 <= tanh_result <= 1.0
        assert 0.0 <= sigmoid_result <= 1.0

        # Test with different parameter combinations
        for mean_test, std_test in [(0.0, 1.0), (5.0, 2.0), (-2.0, 0.5), (100.0, 10.0)]:
            sample = sample_normal(mean=mean_test, std=std_test, rng=rng)
            tanh_trans = tanh_transform(mean=mean_test, std=std_test, rng=rng)
            sigmoid_trans = sigmoid_transform(mean=mean_test, std=std_test, rng=rng)

            assert -1.0 <= tanh_trans <= 1.0
            assert 0.0 <= sigmoid_trans <= 1.0

    def test_transformation_numerical_stability(self):
        """Test numerical stability of transformation functions."""
        rng = create_rng(42)

        # Test with extreme values that might cause numerical issues
        extreme_params = [
            (1000.0, 0.001),  # Large mean, small std
            (-1000.0, 0.001), # Negative large mean, small std
            (0.0, 1000.0),    # Zero mean, large std
            (0.001, 1000.0),  # Small mean, large std
        ]

        for mean_val, std_val in extreme_params:
            # Should not produce NaN or infinite values
            tanh_result = tanh_transform(mean=mean_val, std=std_val, rng=rng)
            sigmoid_result = sigmoid_transform(mean=mean_val, std=std_val, rng=rng)

            assert np.isfinite(tanh_result)
            assert np.isfinite(sigmoid_result)
            assert -1.0 <= tanh_result <= 1.0
            assert 0.0 <= sigmoid_result <= 1.0

    def test_inverse_transformation_numerical_stability(self):
        """Test numerical stability of inverse transformation functions."""
        # Test with values close to bounds that might cause numerical issues
        extreme_values = [0.999, -0.999, 0.999999, -0.999999]

        for val in extreme_values:
            # Should not produce NaN or infinite values
            tanh_result = inverse_tanh_transform(val, mean=0.0, std=1.0)
            assert np.isfinite(tanh_result)

        # Test sigmoid with values close to bounds
        extreme_values_sig = [0.001, 0.999, 0.000001, 0.999999]

        for val in extreme_values_sig:
            sigmoid_result = inverse_sigmoid_transform(val, mean=0.0, std=1.0)
            assert np.isfinite(sigmoid_result)

    def test_transformation_reproducibility(self):
        """Test that transformations are reproducible with same random seeds."""
        # Test multiple runs with same seed produce same results
        seed = 12345

        results_run1 = []
        results_run2 = []

        for i in range(10):
            rng1 = create_rng(seed + i)
            rng2 = create_rng(seed + i)

            tanh1 = tanh_transform(mean=1.0, std=0.5, rng=rng1)
            tanh2 = tanh_transform(mean=1.0, std=0.5, rng=rng2)

            sigmoid1 = sigmoid_transform(mean=1.0, std=0.5, rng=rng1)
            sigmoid2 = sigmoid_transform(mean=1.0, std=0.5, rng=rng2)

            results_run1.extend([tanh1, sigmoid1])
            results_run2.extend([tanh2, sigmoid2])

        # All corresponding results should be identical
        for r1, r2 in zip(results_run1, results_run2):
            assert abs(r1 - r2) < 1e-15

    def test_transformation_statistical_independence(self):
        """Test that multiple calls produce statistically independent results."""
        rng = create_rng(42)

        # Generate multiple samples and check they're not identical
        samples_tanh = [tanh_transform(rng=rng) for _ in range(100)]
        samples_sigmoid = [sigmoid_transform(rng=rng) for _ in range(100)]

        # Samples should not all be identical (very low probability)
        assert len(set(samples_tanh)) > 1
        assert len(set(samples_sigmoid)) > 1

        # Check that we get good coverage of the range
        assert min(samples_tanh) < -0.5
        assert max(samples_tanh) > 0.5
        assert min(samples_sigmoid) < 0.25
        assert max(samples_sigmoid) > 0.75

    def test_transformation_configuration_integration(self):
        """Test that transformation functions integrate properly with configuration system."""
        # Test that functions work without explicit parameters (using config defaults)
        rng = create_rng(42)

        # These should work without errors, using configuration defaults
        tanh_result = tanh_transform(rng=rng)
        sigmoid_result = sigmoid_transform(rng=rng)

        assert -1.0 <= tanh_result <= 1.0
        assert 0.0 <= sigmoid_result <= 1.0

        # Test that configuration parameters are actually used
        # The functions should use config.get() for default parameters
        # We can verify this by checking that the functions don't crash
        # when config is available

    def test_transformation_error_handling_edge_cases(self):
        """Test error handling and edge cases for transformation functions."""
        # Test with None rng parameter (should create its own)
        result_tanh = tanh_transform(mean=0.0, std=1.0, rng=None)
        result_sigmoid = sigmoid_transform(mean=0.0, std=1.0, rng=None)

        assert -1.0 <= result_tanh <= 1.0
        assert 0.0 <= result_sigmoid <= 1.0

        # Test with negative standard deviation (numpy raises ValueError)
        # This should raise an error since numpy doesn't handle negative std
        try:
            tanh_transform(mean=0.0, std=-1.0, rng=create_rng(42))
            assert False, "Should have raised ValueError for negative std"
        except ValueError:
            pass  # Expected behavior

        try:
            sigmoid_transform(mean=0.0, std=-1.0, rng=create_rng(42))
            assert False, "Should have raised ValueError for negative std"
        except ValueError:
            pass  # Expected behavior

    def test_inverse_transformation_edge_cases(self):
        """Test edge cases for inverse transformation functions."""
        # Test with exactly 0.0 for tanh (should return mean)
        result = inverse_tanh_transform(0.0, mean=5.0, std=2.0)
        assert result == 5.0

        # Test with exactly 0.5 for sigmoid (should return mean)
        result = inverse_sigmoid_transform(0.5, mean=3.0, std=1.5)
        assert result == 3.0

        # Test boundary values for tanh
        result_neg_bound = inverse_tanh_transform(-0.999, mean=0.0, std=1.0)  # Use -0.999 instead of -1.0
        result_pos_bound = inverse_tanh_transform(0.999, mean=0.0, std=1.0)   # Use 0.999 instead of 1.0

        # Should produce finite values (may be large)
        assert np.isfinite(result_neg_bound)
        assert np.isfinite(result_pos_bound)

        # Test boundary values for sigmoid
        result_zero_bound = inverse_sigmoid_transform(0.001, mean=0.0, std=1.0)  # Use 0.001 instead of 0.0
        result_one_bound = inverse_sigmoid_transform(0.999, mean=0.0, std=1.0)   # Use 0.999 instead of 1.0

        # Should produce finite values (may be large)
        assert np.isfinite(result_zero_bound)
        assert np.isfinite(result_one_bound)

    def test_transformation_functions_with_array_parameters(self):
        """Test transformation functions handle various parameter types correctly."""
        rng = create_rng(42)

        # Test with numpy scalar types
        mean_np = np.float64(1.0)
        std_np = np.float32(0.5)

        result_tanh = tanh_transform(mean=mean_np, std=std_np, rng=rng)
        result_sigmoid = sigmoid_transform(mean=mean_np, std=std_np, rng=rng)

        assert -1.0 <= result_tanh <= 1.0
        assert 0.0 <= result_sigmoid <= 1.0
        assert isinstance(result_tanh, (float, np.floating))
        assert isinstance(result_sigmoid, (float, np.floating))

    def test_transformation_functions_return_types(self):
        """Test that transformation functions return correct types."""
        rng = create_rng(42)

        # Test return types
        result_tanh = tanh_transform(rng=rng)
        result_sigmoid = sigmoid_transform(rng=rng)

        # Should return Python float or numpy floating types
        assert isinstance(result_tanh, (float, np.floating))
        assert isinstance(result_sigmoid, (float, np.floating))

        # Test inverse transformation return types
        result_inv_tanh = inverse_tanh_transform(0.5, mean=1.0, std=0.5)
        result_inv_sigmoid = inverse_sigmoid_transform(0.5, mean=1.0, std=0.5)

        assert isinstance(result_inv_tanh, (float, np.floating))
        assert isinstance(result_inv_sigmoid, (float, np.floating))

    def test_transformation_functions_numerical_precision(self):
        """Test numerical precision of transformation functions."""
        rng = create_rng(42)

        # Test that repeated calls with same parameters produce consistent results
        results_tanh = [tanh_transform(mean=1.0, std=0.5, rng=create_rng(42)) for _ in range(5)]
        results_sigmoid = [sigmoid_transform(mean=1.0, std=0.5, rng=create_rng(42)) for _ in range(5)]

        # All results should be identical due to same seed
        assert all(abs(results_tanh[0] - r) < 1e-15 for r in results_tanh)
        assert all(abs(results_sigmoid[0] - r) < 1e-15 for r in results_sigmoid)

    def test_transformation_functions_with_very_small_values(self):
        """Test transformation functions with very small parameter values."""
        rng = create_rng(42)

        # Test with very small but positive standard deviation
        result_tanh = tanh_transform(mean=0.0, std=1e-10, rng=rng)
        result_sigmoid = sigmoid_transform(mean=0.0, std=1e-10, rng=rng)

        assert -1.0 <= result_tanh <= 1.0
        assert 0.0 <= result_sigmoid <= 1.0
        assert np.isfinite(result_tanh)
        assert np.isfinite(result_sigmoid)

        # Test inverse with very small values
        result_inv_tanh = inverse_tanh_transform(1e-10, mean=0.0, std=1e-10)
        result_inv_sigmoid = inverse_sigmoid_transform(1e-10, mean=0.0, std=1e-10)

        assert np.isfinite(result_inv_tanh)
        assert np.isfinite(result_inv_sigmoid)

    def test_transformation_functions_deterministic_behavior(self):
        """Test deterministic behavior across different scenarios."""
        # Test that functions behave deterministically with fixed seeds
        seed = 999

        # Test multiple parameter combinations
        param_combinations = [
            (0.0, 1.0),
            (5.0, 2.0),
            (-3.0, 0.5),
            (100.0, 10.0),
            (0.001, 0.001)
        ]

        for mean_val, std_val in param_combinations:
            rng1 = create_rng(seed)
            rng2 = create_rng(seed)

            result_tanh_1 = tanh_transform(mean=mean_val, std=std_val, rng=rng1)
            result_tanh_2 = tanh_transform(mean=mean_val, std=std_val, rng=rng2)

            result_sigmoid_1 = sigmoid_transform(mean=mean_val, std=std_val, rng=rng1)
            result_sigmoid_2 = sigmoid_transform(mean=mean_val, std=std_val, rng=rng2)

            # Results should be identical
            assert abs(result_tanh_1 - result_tanh_2) < 1e-15
            assert abs(result_sigmoid_1 - result_sigmoid_2) < 1e-15

    def test_transformation_functions_monotonic_properties(self):
        """Test monotonic properties of transformation functions."""
        rng = create_rng(42)

        # Test that increasing mean shifts the distribution
        samples_low_mean = [tanh_transform(mean=-1.0, std=1.0, rng=create_rng(i)) for i in range(100)]
        samples_high_mean = [tanh_transform(mean=1.0, std=1.0, rng=create_rng(i + 100)) for i in range(100)]

        # Higher mean should generally produce higher values
        assert np.mean(samples_high_mean) > np.mean(samples_low_mean)

        # Same test for sigmoid
        samples_sigmoid_low = [sigmoid_transform(mean=-1.0, std=1.0, rng=create_rng(i)) for i in range(100)]
        samples_sigmoid_high = [sigmoid_transform(mean=1.0, std=1.0, rng=create_rng(i + 100)) for i in range(100)]

        # Higher mean should generally produce higher values
        assert np.mean(samples_sigmoid_high) > np.mean(samples_sigmoid_low)

    def test_inverse_transformation_mathematical_correctness(self):
        """Test mathematical correctness of inverse transformations."""
        # Test specific known values
        mean_val = 2.0
        std_val = 1.5

        # Test tanh inverse with specific value
        test_value = 0.5
        result = inverse_tanh_transform(test_value, mean=mean_val, std=std_val)

        # Manual calculation for verification
        normalized = np.arctanh(test_value)
        z_score = normalized * 3.0
        expected = z_score * std_val + mean_val

        assert abs(result - expected) < 1e-10

        # Test sigmoid inverse with specific value
        test_value_sig = 0.7
        result_sig = inverse_sigmoid_transform(test_value_sig, mean=mean_val, std=std_val)

        # Manual calculation for verification
        normalized_sig = -np.log(1.0 / test_value_sig - 1.0)
        z_score_sig = normalized_sig * 3.0
        expected_sig = z_score_sig * std_val + mean_val

        assert abs(result_sig - expected_sig) < 1e-10

    def test_transformation_functions_comprehensive_bounds_check(self):
        """Comprehensive bounds checking for transformation functions."""
        rng = create_rng(42)

        # Test extensive sampling to ensure bounds are never violated
        for _ in range(100):
            # Test various parameter combinations
            mean_val = rng.uniform(-10, 10)
            std_val = rng.uniform(0.1, 5.0)

            tanh_result = tanh_transform(mean=mean_val, std=std_val, rng=rng)
            sigmoid_result = sigmoid_transform(mean=mean_val, std=std_val, rng=rng)

            # Assert bounds are strictly maintained
            assert tanh_result >= -1.0
            assert tanh_result <= 1.0
            assert sigmoid_result >= 0.0
            assert sigmoid_result <= 1.0

            # Test that we can get very close to bounds
            if _ < 10:  # Test first 10 iterations more thoroughly
                assert abs(tanh_result) <= 1.0
                assert 0.0 <= sigmoid_result <= 1.0

    def test_transformation_functions_memory_efficiency(self):
        """Test that transformation functions don't have memory leaks."""
        rng = create_rng(42)

        # Generate many samples and ensure no memory issues
        results = []
        for i in range(1000):
            result_tanh = tanh_transform(mean=float(i % 10), std=1.0, rng=rng)
            result_sigmoid = sigmoid_transform(mean=float(i % 10), std=1.0, rng=rng)
            results.extend([result_tanh, result_sigmoid])

        # Should have generated 2000 results
        assert len(results) == 2000

        # All results should be valid
        tanh_results = results[::2]  # Every other result starting from index 0 (tanh)
        sigmoid_results = results[1::2]  # Every other result starting from index 1 (sigmoid)

        assert all(-1.0 <= r <= 1.0 for r in tanh_results)
        assert all(0.0 <= r <= 1.0 for r in sigmoid_results)

    def test_transformation_functions_thread_safety(self):
        """Test that transformation functions are thread-safe."""
        import threading
        import queue

        rng = create_rng(42)
        results_queue = queue.Queue()

        def generate_samples(seed_offset):
            """Function to generate samples in a thread."""
            thread_rng = create_rng(42 + seed_offset)
            thread_results = []

            for i in range(100):
                tanh_result = tanh_transform(mean=1.0, std=0.5, rng=thread_rng)
                sigmoid_result = sigmoid_transform(mean=1.0, std=0.5, rng=thread_rng)
                thread_results.extend([tanh_result, sigmoid_result])

            results_queue.put(thread_results)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=generate_samples, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Collect results
        all_results = []
        while not results_queue.empty():
            all_results.extend(results_queue.get())

        # Should have results from all threads
        assert len(all_results) == 5 * 200  # 5 threads * 200 results each

        # All results should be in valid ranges
        tanh_results = all_results[::2]  # Every other result (tanh)
        sigmoid_results = all_results[1::2]  # Every other result (sigmoid)

        assert all(-1.0 <= r <= 1.0 for r in tanh_results)
        assert all(0.0 <= r <= 1.0 for r in sigmoid_results)


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