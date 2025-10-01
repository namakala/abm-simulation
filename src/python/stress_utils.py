"""
Stress-related utility functions for agent-based mental health simulation.

This module contains stateless functions for:
- Stress event generation and appraisal
- Challenge/hindrance mapping
- Threshold evaluation for stress responses
- All functions are pure and support dependency injection for testability
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from src.python.config import get_config

# Load configuration
config = get_config()


@dataclass
class StressEvent:
    """Represents a stress event with controllability, predictability, and overload."""
    controllability: float # c ∈ [0,1]
    predictability: float  # p ∈ [0,1]
    overload: float        # o ∈ [0,1]
    magnitude: float       # s ∈ [0,1]


@dataclass
class AppraisalWeights:
    """Weights for the apply-weight function in stress appraisal."""
    omega_c: float = field(default_factory=lambda: get_config().get('appraisal', 'omega_c')) # Weight for controllability
    omega_p: float = field(default_factory=lambda: get_config().get('appraisal', 'omega_p')) # Weight for predictability
    omega_o: float = field(default_factory=lambda: get_config().get('appraisal', 'omega_o')) # Weight for overload
    bias: float    = field(default_factory=lambda: get_config().get('appraisal', 'bias')) # Bias term
    gamma: float   = field(default_factory=lambda: get_config().get('appraisal', 'gamma')) # Sigmoid steepness


@dataclass
class ThresholdParams:
    """Parameters for stress threshold evaluation."""
    base_threshold: float  = field(default_factory=lambda: get_config().get('threshold', 'base_threshold'))
    challenge_scale: float = field(default_factory=lambda: get_config().get('threshold', 'challenge_scale'))
    hindrance_scale: float = field(default_factory=lambda: get_config().get('threshold', 'hindrance_scale'))


def generate_stress_event(
    rng: Optional[np.random.Generator] = None,
    config: Optional[Dict[str, Any]] = None
) -> StressEvent:
    """
    Generate a random stress event with controllability, predictability, and overload.

    Args:
        rng: Random number generator for reproducible testing
        config: Configuration parameters for event generation

    Returns:
        StressEvent with normalized attributes in [0,1]
    """
    # Get fresh config instance to avoid global config issues
    cfg = get_config()

    if rng is None:
        rng = np.random.default_rng()

    if config is None:
        config = {
            'controllability_mean': cfg.get('stress', 'controllability_mean'),
            'predictability_mean': cfg.get('stress', 'predictability_mean'),
            'overload_mean': cfg.get('stress', 'overload_mean'),
            'magnitude_scale': cfg.get('stress', 'magnitude_scale')
        }

    # Generate event attributes using beta distribution for bounded [0,1] values
    alpha = cfg.get('stress', 'beta_alpha')
    beta  = cfg.get('stress', 'beta_beta')
    controllability = rng.beta(alpha, beta)
    predictability = rng.beta(alpha, beta)
    overload = rng.beta(alpha, beta)
    magnitude = min(rng.exponential(config['magnitude_scale']), 1.0)  # Cap at 1.0

    return StressEvent(
        controllability=controllability,
        predictability=predictability,
        overload=overload,
        magnitude=magnitude
    )


def sigmoid(x: float, gamma: float = 6.0) -> float:
    """
    Sigmoid function for challenge/hindrance mapping.

    Args:
        x: Input value
        gamma: Steepness parameter

    Returns:
        Sigmoid output in [0,1]
    """
    return 1.0 / (1.0 + np.exp(-gamma * x))


def apply_weights(
    event: StressEvent,
    weights: Optional[AppraisalWeights] = None
) -> Tuple[float, float]:
    """
    Apply weights to stress event attributes to compute challenge and hindrance.

    Args:
        event: Stress event with c, p, o attributes
        weights: Weight parameters for the mapping function

    Returns:
        Tuple of (challenge, hindrance) values in [0,1]
    """
    if weights is None:
        weights = AppraisalWeights()

    # Compute weighted sum: z = ωc*c + ωp*p - ωo*o + b
    z = (weights.omega_c * event.controllability +
         weights.omega_p * event.predictability -
         weights.omega_o * event.overload +
         weights.bias)

    # Apply sigmoid to get challenge
    challenge = sigmoid(z, weights.gamma)
    hindrance = 1.0 - challenge

    return challenge, hindrance


def compute_appraised_stress(
    event: StressEvent,
    challenge: float,
    hindrance: float,
    config: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute overall appraised stress load from event and challenge/hindrance.

    Args:
        event: Stress event
        challenge: Challenge component from appraisal
        hindrance: Hindrance component from appraisal
        config: Configuration for stress computation

    Returns:
        Appraised stress load L ∈ [0,1]
    """
    if config is None:
        # Get fresh config instance to avoid global config issues
        cfg = get_config()
        config = {
            'alpha_challenge': cfg.get('stress_params', 'alpha_challenge'),
            'alpha_hindrance': cfg.get('stress_params', 'alpha_hindrance'),
            'delta': cfg.get('stress_params', 'delta')
        }

    # Method 1: Weighted combination
    # L = s * (α_ch * (1-challenge) + α_hd * hindrance)

    # Method 2: Polarity-based (more interpretable)
    polarity_effect = config['delta'] * (hindrance - challenge)
    stress_load = event.magnitude * (1.0 + polarity_effect)

    return min(stress_load, 1.0)  # Cap at 1.0


def evaluate_stress_threshold(
    appraised_stress: float,
    challenge: float,
    hindrance: float,
    threshold_params: Optional[ThresholdParams] = None
) -> bool:
    """
    Evaluate whether an agent becomes stressed based on appraised stress and threshold.

    Args:
        appraised_stress: Computed stress load L
        challenge: Challenge component
        hindrance: Hindrance component
        threshold_params: Threshold evaluation parameters

    Returns:
        True if agent becomes stressed, False otherwise
    """
    if threshold_params is None:
        # Get fresh config instance to avoid global config issues
        cfg = get_config()
        threshold_params = ThresholdParams(
            base_threshold=cfg.get('threshold', 'base_threshold'),
            challenge_scale=cfg.get('threshold', 'challenge_scale'),
            hindrance_scale=cfg.get('threshold', 'hindrance_scale')
        )

    # Effective threshold: T_eff = T_base + λ_C*challenge - λ_H*hindrance
    effective_threshold = (threshold_params.base_threshold +
                          threshold_params.challenge_scale * challenge -
                          threshold_params.hindrance_scale * hindrance)

    # Clamp effective threshold to [0,1]
    effective_threshold = max(0.0, min(1.0, effective_threshold))

    return appraised_stress > effective_threshold


def process_stress_event(
    event: StressEvent,
    threshold_params: Optional[ThresholdParams] = None,
    weights: Optional[AppraisalWeights] = None,
    rng: Optional[np.random.Generator] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, float, float]:
    """
    Complete stress event processing pipeline.

    Args:
        event: Stress event to process
        threshold_params: Parameters for threshold evaluation
        weights: Parameters for challenge/hindrance mapping
        rng: Random number generator (for testing)
        config: Configuration parameters

    Returns:
        Tuple of (is_stressed, challenge, hindrance)
    """
    # Apply weights to get challenge/hindrance
    challenge, hindrance = apply_weights(event, weights)

    # Compute appraised stress
    appraised_stress = compute_appraised_stress(event, challenge, hindrance, config)

    # Evaluate threshold
    is_stressed = evaluate_stress_threshold(
        appraised_stress, challenge, hindrance, threshold_params
    )

    return is_stressed, challenge, hindrance