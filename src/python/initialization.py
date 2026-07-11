"""
Agent initialization utilities extracted from agent.py.

Provides pure, testable functions for all initial state computations.
Every function is deterministic given the same RNG seed.
"""

from typing import Any, Dict

import numpy as np

from src.python.math_utils import sigmoid_transform, tanh_transform
from src.python.stress_utils import compute_stress_from_dimensions, initialize_pss10_from_items


def initialize_baseline_resilience(
    rng: np.random.Generator,
    mean: float,
    std: float,
) -> float:
    """Sample baseline resilience from sigmoid-transformed normal.

    Args:
        rng: Seeded random number generator.
        mean: Mean of the underlying normal distribution.
        std: Standard deviation of the underlying normal distribution.

    Returns:
        Resilience value clamped to [0, 1].
    """
    return sigmoid_transform(mean=mean, std=std, rng=rng)


def initialize_baseline_affect(
    rng: np.random.Generator,
    mean: float,
    std: float,
) -> float:
    """Sample baseline affect from tanh-transformed normal.

    Args:
        rng: Seeded random number generator.
        mean: Mean of the underlying normal distribution.
        std: Standard deviation of the underlying normal distribution.

    Returns:
        Affect value clamped to [-1, 1].
    """
    return tanh_transform(mean=mean, std=std, rng=rng)


def initialize_resources(
    rng: np.random.Generator,
    mean: float,
    std: float,
) -> float:
    """Sample initial resources from sigmoid-transformed normal.

    Args:
        rng: Seeded random number generator.
        mean: Mean of the underlying normal distribution.
        std: Standard deviation of the underlying normal distribution.

    Returns:
        Resource value clamped to [0, 1].
    """
    return sigmoid_transform(mean=mean, std=std, rng=rng)


def initialize_protective_factors(value: float = 0.5) -> Dict[str, float]:
    """Create default protective factors dict with uniform initial values.

    Args:
        value: Initial value for each factor (default 0.5).

    Returns:
        Dict with keys social_support, family_support, formal_intervention,
        psychological_capital, all set to *value*.
    """
    return {
        "social_support": value,
        "family_support": value,
        "formal_intervention": value,
        "psychological_capital": value,
    }


def initialize_volatility(
    rng: np.random.Generator,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Sample agent volatility from Beta distribution.

    Args:
        rng: Seeded random number generator.
        alpha: Beta distribution alpha parameter (default 1.0).
        beta: Beta distribution beta parameter (default 1.0).

    Returns:
        Volatility value in [0, 1].
    """
    return float(rng.beta(alpha, beta))


def initialize_pss10_state(
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Generate full PSS-10 state: responses, dimensions, total score.

    Delegates to ``initialize_pss10_from_items`` from stress_utils.

    Args:
        rng: Seeded random number generator.

    Returns:
        Dict with keys: pss10_responses (dict[int, int]),
        stress_controllability (float), stress_overload (float),
        pss10_score (int), stressed (bool).
    """
    return initialize_pss10_from_items(rng=rng)


def compute_initial_stress(
    pss10_score: int,
    controllability: float,
    overload: float,
    dampening: float = 1.0,
) -> float:
    """Compute initial stress level from PSS-10 dimensions.

    Args:
        pss10_score: Initial PSS-10 total score (0-40).
        controllability: Stress controllability dimension in [0, 1].
        overload: Stress overload dimension in [0, 1].
        dampening: Scaling factor for stress level.

    Returns:
        Initial stress level in [0, 1].
    """
    return compute_stress_from_dimensions(
        stress_controllability=controllability,
        stress_overload=overload,
        dampening=dampening,
    )
