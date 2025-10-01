"""
Stress-related utility functions for agent-based mental health simulation.

This module contains stateless functions for:
- Stress event generation and appraisal
- Challenge/hindrance mapping
- Threshold evaluation for stress responses
- PSS-10 mapping functionality
- All functions are pure and support dependency injection for testability
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
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


@dataclass
class PSS10Item:
    """Represents a PSS-10 questionnaire item with response mapping."""
    text: str
    reverse_scored: bool = False
    weight_controllability: float = 0.0
    weight_predictability: float = 0.0
    weight_overload: float = 0.0
    weight_distress: float = 0.0


def create_pss10_mapping() -> Dict[int, PSS10Item]:
    """
    Create PSS-10 item mapping with theoretical relationships to stress components.

    Returns:
        Dictionary mapping item numbers (1-10) to PSS10Item objects
    """
    return {
        1: PSS10Item(
            text="In the last month, how often have you been upset because of something that happened unexpectedly?",
            reverse_scored=False,
            weight_predictability=0.8,  # High predictability weight
            weight_distress=0.6
        ),
        2: PSS10Item(
            text="In the last month, how often have you felt that you were unable to control the important things in your life?",
            reverse_scored=False,
            weight_controllability=0.9,  # High controllability weight
            weight_distress=0.7
        ),
        3: PSS10Item(
            text="In the last month, how often have you felt nervous and 'stressed'?",
            reverse_scored=False,
            weight_overload=0.6,
            weight_distress=0.8
        ),
        4: PSS10Item(
            text="In the last month, how often have you felt confident about your ability to handle your personal problems?",
            reverse_scored=True,  # Reverse scored
            weight_controllability=0.7,
            weight_distress=0.5
        ),
        5: PSS10Item(
            text="In the last month, how often have you felt that things were going your way?",
            reverse_scored=True,  # Reverse scored
            weight_controllability=0.6,
            weight_predictability=0.5,
            weight_distress=0.4
        ),
        6: PSS10Item(
            text="In the last month, how often have you found that you could not cope with all the things that you had to do?",
            reverse_scored=False,
            weight_overload=0.9,  # High overload weight
            weight_distress=0.8
        ),
        7: PSS10Item(
            text="In the last month, how often have you been able to control irritations in your life?",
            reverse_scored=True,  # Reverse scored
            weight_controllability=0.8,
            weight_distress=0.6
        ),
        8: PSS10Item(
            text="In the last month, how often have you felt that you were on top of things?",
            reverse_scored=True,  # Reverse scored
            weight_controllability=0.5,
            weight_predictability=0.4,
            weight_overload=0.3,
            weight_distress=0.5
        ),
        9: PSS10Item(
            text="In the last month, how often have you been angered because of things that were outside of your control?",
            reverse_scored=False,
            weight_controllability=0.8,
            weight_distress=0.7
        ),
        10: PSS10Item(
            text="In the last month, how often have you felt difficulties were piling up so high that you could not overcome them?",
            reverse_scored=False,
            weight_overload=0.9,  # High overload weight
            weight_distress=0.9
        )
    }


def map_agent_stress_to_pss10(
    controllability: float,
    predictability: float,
    overload: float,
    distress: float,
    rng: Optional[np.random.Generator] = None
) -> Dict[int, int]:
    """
    Map agent stress state to PSS-10 item responses.

    Args:
        controllability: Agent's controllability level ∈ [0,1]
        predictability: Agent's predictability level ∈ [0,1]
        overload: Agent's overload level ∈ [0,1]
        distress: Agent's current distress level ∈ [0,1]
        rng: Random number generator for response variability

    Returns:
        Dictionary mapping item numbers (1-10) to response values (0-4)
    """
    if rng is None:
        rng = np.random.default_rng()

    pss10_items = create_pss10_mapping()
    responses = {}

    for item_num, item in pss10_items.items():
        # Calculate base score from weighted components
        base_score = (
            item.weight_controllability * (1.0 - controllability) +  # Low controllability = high stress
            item.weight_predictability * (1.0 - predictability) +    # Low predictability = high stress
            item.weight_overload * overload +                        # High overload = high stress
            item.weight_distress * distress                          # High distress = high stress
        ) / max(sum([item.weight_controllability, item.weight_predictability,
                   item.weight_overload, item.weight_distress]), 1e-10)

        # Add response variability (±0.2) and measurement error
        variability = rng.normal(0, 0.1)
        measurement_error = rng.normal(0, 0.05)

        # Apply reverse scoring if needed
        if item.reverse_scored:
            score = 1.0 - base_score + variability + measurement_error
        else:
            score = base_score + variability + measurement_error

        # Clamp to [0,1] and scale to PSS-10 response range (0-4)
        score = max(0.0, min(1.0, score))
        response_value = int(round(score * 4.0))

        responses[item_num] = response_value

    return responses


def compute_pss10_score(responses: Dict[int, int]) -> int:
    """
    Compute total PSS-10 score from item responses.

    Args:
        responses: Dictionary mapping item numbers to response values (0-4)

    Returns:
        Total PSS-10 score (0-40)

    Raises:
        ValueError: If responses don't contain all required items or contain invalid values
    """
    required_items = set(range(1, 11))

    if set(responses.keys()) != required_items:
        missing = required_items - set(responses.keys())
        raise ValueError(f"Missing PSS-10 items: {missing}")

    for item_num, response in responses.items():
        if not (0 <= response <= 4):
            raise ValueError(f"Invalid response for item {item_num}: {response} (must be 0-4)")

    # Apply reverse scoring for items 4, 5, 7, 8
    reverse_items = {4, 5, 7, 8}
    total_score = 0

    for item_num in range(1, 11):
        response = responses[item_num]
        if item_num in reverse_items:
            # Reverse score: 0→4, 1→3, 2→2, 3→1, 4→0
            total_score += (4 - response)
        else:
            total_score += response

    return total_score


def interpret_pss10_score(score: int) -> str:
    """
    Interpret PSS-10 score according to standard categories.

    Args:
        score: Total PSS-10 score (0-40)

    Returns:
        Interpretation string

    Raises:
        ValueError: If score is outside valid range
    """
    if not (0 <= score <= 40):
        raise ValueError(f"Invalid PSS-10 score: {score} (must be 0-40)")

    if score <= 13:
        return "Low stress"
    elif score <= 26:
        return "Moderate stress"
    else:
        return "High stress"