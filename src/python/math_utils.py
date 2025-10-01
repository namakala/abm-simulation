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
    return float(np.clip(value, min_val, max_val))


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


def sigmoid(x: float, gamma: float = 1.0) -> float:
    """
    Sigmoid activation function.

    Args:
        x: Input value
        gamma: Steepness parameter

    Returns:
        Sigmoid output in [0,1]
    """
    return 1.0 / (1.0 + np.exp(-gamma * x))


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Softmax function with temperature control.

    Args:
        x: Input array
        temperature: Temperature parameter (higher = more uniform, lower = more peaked)

    Returns:
        Softmax probabilities
    """
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
    min_value: int = 1
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
    max_value: float = 1.0
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

    sample = rng.exponential(scale)
    return min(sample, max_value)


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