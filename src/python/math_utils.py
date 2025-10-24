"""
Mathematical utility functions for agent-based mental health simulation.

This module contains stateless mathematical functions for:
- Normalization and clamping operations
- Random number generation with dependency injection
- Statistical distributions and sampling
- All functions are pure and support dependency injection for testability
"""

import numpy as np
from typing import Optional, List, Union, Tuple
from dataclasses import dataclass
from src.python.config import get_config

# Load configuration
config = get_config()


@dataclass
class RNGConfig:
    """Configuration for random number generation."""
    seed: Optional[int] = None
    generator: Optional[np.random.Generator] = None


def create_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a random number generator with optional seed for reproducibility.

    Args:
        seed: Random seed for reproducible results

    Returns:
        Configured random number generator
    """
    return np.random.default_rng(seed)


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Clamp a value to specified bounds.

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value in [min_val, max_val]
    """
    return max(min_val, min(value, max_val))


def normalize_to_range(
    value: float,
    old_min: float,
    old_max: float,
    new_min: float = 0.0,
    new_max: float = 1.0
) -> float:
    """
    Normalize a value from one range to another.

    Args:
        value: Value to normalize
        old_min: Original range minimum
        old_max: Original range maximum
        new_min: Target range minimum
        new_max: Target range maximum

    Returns:
        Normalized value in [new_min, new_max]
    """
    if old_max == old_min:
        return new_min

    normalized = (value - old_min) / (old_max - old_min)
    scaled = normalized * (new_max - new_min) + new_min

    return clamp(scaled, new_min, new_max)


def sigmoid(x: float, gamma: float = None) -> float:
    """
    Sigmoid activation function.

    Args:
        x: Input value
        gamma: Steepness parameter

    Returns:
        Sigmoid output in [0,1]
    """
    if gamma is None:
        gamma = config.get('appraisal', 'gamma')  # Use existing gamma parameter
    return 1.0 / (1.0 + np.exp(-gamma * x))


def softmax(x: np.ndarray, temperature: float = None) -> np.ndarray:
    """
    Softmax function with temperature control.

    Args:
        x: Input array
        temperature: Temperature parameter (higher = more uniform, lower = more peaked)

    Returns:
        Softmax probabilities
    """
    if temperature is None:
        temperature = config.get('utility', 'softmax_temperature')
    if temperature == 0:
        # Handle temperature = 0 case (returns one-hot of max)
        max_idx = np.argmax(x)
        result = np.zeros_like(x)
        result[max_idx] = 1.0
        return result

    # Standard softmax with temperature
    x_scaled = x / temperature
    x_exp = np.exp(x_scaled - np.max(x_scaled))  # Numerical stability
    return x_exp / np.sum(x_exp)


def sample_poisson(
    lam: float,
    rng: Optional[np.random.Generator] = None,
    min_value: int = None
) -> int:
    """
    Sample from Poisson distribution with minimum value constraint.

    Args:
        lam: Poisson rate parameter
        rng: Random number generator
        min_value: Minimum allowed value

    Returns:
        Sampled value ≥ min_value
    """
    if rng is None:
        rng = np.random.default_rng()
    if min_value is None:
        min_value = 0  # Default minimum value

    sample = rng.poisson(lam)
    return max(sample, min_value)


def sample_beta(
    alpha: float,
    beta: float,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Sample from Beta distribution in [0,1].

    Args:
        alpha: Beta distribution alpha parameter
        beta: Beta distribution beta parameter
        rng: Random number generator

    Returns:
        Sample in [0,1]
    """
    if rng is None:
        rng = np.random.default_rng()

    return rng.beta(alpha, beta)


def sample_exponential(
    scale: float,
    rng: Optional[np.random.Generator] = None,
    max_value: float = None
) -> float:
    """
    Sample from exponential distribution capped at max_value.

    Args:
        scale: Exponential distribution scale parameter
        rng: Random number generator
        max_value: Maximum allowed value

    Returns:
        Sample in [0, max_value]
    """
    if rng is None:
        rng = np.random.default_rng()
    if max_value is None:
        max_value = 10.0  # Default maximum value

    sample = rng.exponential(scale)
    return min(sample, max_value)


def sample_normal(
    mean: float = 0.0,
    std: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    min_value: float = None,
    max_value: float = None
) -> float:
    """
    Sample from normal distribution with optional clamping.

    Args:
        mean: Normal distribution mean parameter
        std: Normal distribution standard deviation parameter
        rng: Random number generator
        min_value: Minimum allowed value (optional clamping)
        max_value: Maximum allowed value (optional clamping)

    Returns:
        Sample from normal distribution, clamped to [min_value, max_value] if specified
    """
    if rng is None:
        rng = np.random.default_rng()

    sample = rng.normal(mean, std)

    # Apply clamping if bounds are specified
    if min_value is not None or max_value is not None:
        # Handle None values properly to avoid using 0.0 as -inf
        clamp_min = min_value if min_value is not None else -np.inf
        clamp_max = max_value if max_value is not None else np.inf
        sample = clamp(sample, clamp_min, clamp_max)

    return sample


def compute_running_average(
    current_avg: float,
    new_value: float,
    count: int,
    alpha: Optional[float] = None
) -> float:
    """
    Compute running average using either count-based or exponential moving average.

    Args:
        current_avg: Current average value
        new_value: New value to incorporate
        count: Number of values seen so far
        alpha: Smoothing factor for exponential moving average (if provided)

    Returns:
        Updated average
    """
    if alpha is not None:
        # Exponential moving average
        return alpha * new_value + (1 - alpha) * current_avg
    else:
        # Standard running average
        return ((count - 1) * current_avg + new_value) / count


def compute_percentile(
    values: List[float],
    percentile: float,
    interpolation: str = 'linear'
) -> float:
    """
    Compute percentile of a list of values.

    Args:
        values: List of numeric values
        percentile: Percentile to compute (0-100)
        interpolation: Interpolation method for non-integer ranks

    Returns:
        Percentile value
    """
    if not values:
        raise ValueError("Cannot compute percentile of empty list")

    sorted_values = np.sort(values)
    return np.percentile(sorted_values, percentile, method=interpolation)


def compute_z_score(
    value: float,
    mean: float,
    std: float,
    ddof: int = 1
) -> float:
    """
    Compute z-score (standard score).

    Args:
        value: Value to standardize
        mean: Population mean
        std: Population standard deviation
        ddof: Delta degrees of freedom for sample standard deviation

    Returns:
        Z-score
    """
    if std == 0:
        return 0.0

    return (value - mean) / std


def logistic_function(
    x: float,
    L: float = 1.0,
    k: float = 1.0,
    x0: float = 0.0
) -> float:
    """
    General logistic function.

    Args:
        x: Input value
        L: Maximum value
        k: Steepness parameter
        x0: Midpoint parameter

    Returns:
        Logistic function output
    """
    return L / (1 + np.exp(-k * (x - x0)))


def linear_interpolation(
    x: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> float:
    """
    Linear interpolation between two points.

    Args:
        x: X value to interpolate at
        x1: First X value
        y1: First Y value
        x2: Second X value
        y2: Second Y value

    Returns:
        Interpolated Y value
    """
    if x1 == x2:
        return y1

    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def calculate_entropy(probabilities: np.ndarray) -> float:
    """
    Calculate Shannon entropy of a probability distribution.

    Args:
        probabilities: Array of probabilities that sum to 1

    Returns:
        Entropy value in nats
    """
    if len(probabilities) == 0:
        return 0.0

    # Filter out zero probabilities to avoid log(0)
    probs = probabilities[probabilities > 0]

    if len(probs) == 0:
        return 0.0

    return -np.sum(probs * np.log(probs))


def tanh_transform(
    mean: float = 0.0,
    std: float = 1.0,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Transform normal distribution sample to [-1,1] bounds using tanh.

    The transformation pipeline:
    1. Sample from normal distribution N(mean, std)
    2. Transform to Z-scale ~ N(0,1) by subtracting mean and dividing by std
    3. Normalize by dividing by 3 to confine to approximately [-1,1]
    4. Apply tanh() for [-1,1] bounds

    Args:
        mean: Normal distribution mean parameter
        std: Normal distribution standard deviation parameter
        rng: Random number generator

    Returns:
        Transformed value in [-1,1]
    """
    if rng is None:
        rng = np.random.default_rng()

    # Handle zero standard deviation case
    if std == 0:
        return 0.0  # Return 0 for zero std case

    # Sample from normal distribution
    sample = rng.normal(mean, std)

    # Transform to Z-scale and normalize
    z_score = (sample - mean) / std
    normalized = z_score / 3.0  # Divide by 3 to confine to ~[-1,1]

    # Apply tanh transformation for [-1,1] bounds
    result = np.tanh(normalized)

    return result


def sigmoid_transform(
    mean: float = 0.0,
    std: float = 1.0,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Transform normal distribution sample to [0,1] bounds using sigmoid.

    The transformation pipeline:
    1. Sample from normal distribution N(mean, std)
    2. Transform to Z-scale ~ N(0,1) by subtracting mean and dividing by std
    3. Normalize by dividing by 3 to confine to approximately [-1,1]
    4. Apply sigmoid() for [0,1] bounds

    Args:
        mean: Normal distribution mean parameter
        std: Normal distribution standard deviation parameter
        rng: Random number generator

    Returns:
        Transformed value in [0,1]
    """
    if rng is None:
        rng = np.random.default_rng()

    # Handle zero standard deviation case
    if std == 0:
        return sigmoid(0.0)  # Return sigmoid of 0 for zero std case

    # Sample from normal distribution
    sample = rng.normal(mean, std)

    # Transform to Z-scale and normalize
    z_score = (sample - mean) / std
    normalized = z_score / 3.0  # Divide by 3 to confine to ~[-1,1]

    # Apply sigmoid transformation for [0,1] bounds
    result = sigmoid(normalized)

    return result


def inverse_tanh_transform(
    value: float,
    mean: float = 0.0,
    std: float = 1.0
) -> float:
    """
    Inverse transformation of tanh_transform.

    Args:
        value: Value in [-1,1] to transform back
        mean: Original normal distribution mean parameter
        std: Original normal distribution standard deviation parameter

    Returns:
        Value in original scale
    """
    if value == 0.0:
        return mean  # Handle special case: 0.0 input → mean output

    # Handle extreme values
    modifier = 1e-10
    if value == -1:
        value = value + modifier
    elif value == 1:
        value = value - modifier

    # Apply inverse tanh (artanh)
    normalized = np.arctanh(value)

    # Scale back from normalized range
    z_score = normalized * 3.0

    # Transform back to original scale
    original = z_score * std + mean

    return original


def inverse_sigmoid_transform(
    value: float,
    mean: float = 0.0,
    std: float = 1.0
) -> float:
    """
    Inverse transformation of sigmoid_transform.

    Args:
        value: Value in [0,1] to transform back
        mean: Original normal distribution mean parameter
        std: Original normal distribution standard deviation parameter

    Returns:
        Value in original scale
    """
    if value == 0.5:
        return mean  # Handle special case: 0.5 input → mean output

    # Handle extreme values
    modifier = 1e-10
    if value == 0:
        value = modifier
    elif value == 1:
        value = 1 - modifier

    # Apply inverse sigmoid (logit)
    normalized = -np.log(1.0 / value - 1.0)

    # Scale back from normalized range
    z_score = normalized * 3.0

    # Transform back to original scale
    original = z_score * std + mean

    return original


def normalize_probabilities(
    values: np.ndarray,
    temperature: float = 1.0,
    method: str = 'softmax'
) -> np.ndarray:
    """
    Normalize values to probabilities using specified method.

    Args:
        values: Input values to normalize
        temperature: Temperature parameter for softmax
        method: Normalization method ('softmax', 'clip', or 'linear')

    Returns:
        Normalized probabilities
    """
    if method == 'softmax':
        return softmax(values, temperature)
    elif method == 'clip':
        # Simple clipping to [0,1] and renormalization
        clipped = np.clip(values, 0, 1)
        total = np.sum(clipped)
        if total == 0:
            return np.ones_like(values) / len(values)
        return clipped / total
    elif method == 'linear':
        # Linear normalization to [0,1]
        min_val, max_val = np.min(values), np.max(values)
        if min_val == max_val:
            return np.ones_like(values) / len(values)
        normalized = (values - min_val) / (max_val - min_val)
        return normalized / np.sum(normalized)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
