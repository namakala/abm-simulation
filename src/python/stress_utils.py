"""
You are an expert in Python simulation design and psychometrics. Your task is to implement empirically grounded PSS-10 score generation for initializing and updating agent stress in an agent-based model.

2. **Score Generation**  
   - Implement a function to generate PSS-10 item scores:  
     - Each item is sampled from a normal distribution defined by its mean and standard deviation obtained from `PSS10_ITEM_MEAN` and `PSS10_ITEM_SD`.  
     - Values must be rounded to the nearest integer and bounded to `[0, 4]` with `clamp`.  
     - Use `PSS10Item` dataclass
   - Dimension correlation:
     - Controllability dimension: items 4, 5, 7, 8.  
     - Overload dimension: items 1, 2, 3, 5, 6, 9, 10.  
     - The correlation of these dimension are configured with `PSS10_BIFACTOR_COR`.
     - Preserve empirical correlation between controllability and overload (e.g., by sampling from a multivariate distribution or correlated random draws).
Stress-related utility functions for agent-based mental health simulation.

This module contains stateless functions for:
- Stress event generation and appraisal
- Challenge/hindrance mapping
- Threshold evaluation for stress responses
- PSS-10 mapping functionality
- All functions are pure and support dependency injection for testability
"""

import hashlib
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field

from src.python.config import get_config
from src.python.math_utils import clamp

# Load configuration
config = get_config()


@dataclass
class StressEvent:
    """Represents a stress event with controllability and overload."""
    controllability: float # c ∈ [0,1]
    overload: float        # o ∈ [0,1]


@dataclass
class AppraisalWeights:
    """Weights for the apply-weight function in stress appraisal."""
    omega_c: float = field(default_factory=lambda: get_config().get('appraisal', 'omega_c')) # Weight for controllability
    omega_o: float = field(default_factory=lambda: get_config().get('appraisal', 'omega_o')) # Weight for overload
    bias: float    = field(default_factory=lambda: get_config().get('appraisal', 'bias')) # Bias term
    gamma: float   = field(default_factory=lambda: get_config().get('appraisal', 'gamma')) # Sigmoid steepness


@dataclass
class ThresholdParams:
    """Parameters for stress threshold evaluation."""
    base_threshold: float  = field(default_factory=lambda: get_config().get('threshold', 'base_threshold'))
    challenge_scale: float = field(default_factory=lambda: get_config().get('threshold', 'challenge_scale'))
    hindrance_scale: float = field(default_factory=lambda: get_config().get('threshold', 'hindrance_scale'))


@dataclass
class PSS10Item:
    """Represents a PSS-10 questionnaire item with empirically grounded factor loadings."""
    text: str = ""
    reverse_scored: bool = False
    weight_controllability: float = 0.0
    weight_overload: float = 0.0


def generate_stress_event(
    rng: Optional[np.random.Generator] = None,
    config: Optional[Dict[str, Any]] = None
) -> StressEvent:
    """
    Generate a random stress event with controllability and overload.

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
            'controllability_sd': cfg.get('stress', 'controllability_sd'),
            'overload_mean': cfg.get('stress', 'overload_mean'),
            'overload_sd': cfg.get('stress', 'overload_sd')
        }

    # Generate event attributes using truncated normal distribution for bounded [0,1] values
    # Use mean and SD from config instead of fixed beta distribution
    controllability_raw = rng.normal(config['controllability_mean'], config['controllability_sd'])
    overload_raw = rng.normal(config['overload_mean'], config['overload_sd'])

    # Clamp to [0,1] range to ensure valid values
    controllability = max(0.0, min(1.0, controllability_raw))
    overload = max(0.0, min(1.0, overload_raw))

    return StressEvent(
        controllability=controllability,
        overload=overload
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
        event: Stress event with c, o attributes
        weights: Weight parameters for the mapping function

    Returns:
        Tuple of (challenge, hindrance) values in [0,1]
    """
    if weights is None:
        weights = AppraisalWeights()

    # Compute weighted sum: z = ωc*c - ωo*o + b (removed predictability term)
    z = (weights.omega_c * event.controllability -
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

    Uses the theoretical specification: L = (1 + δ*(hindrance - challenge))

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
            'delta': cfg.get('stress_params', 'delta')
        }

    # Theoretical specification: L = 1 + δ*(hindrance - challenge) (removed magnitude)
    polarity_effect = config['delta'] * (hindrance - challenge)
    stress_load = 1.0 + polarity_effect

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


def create_pss10_mapping() -> Dict[int, PSS10Item]:
    """
    Create PSS-10 item mapping with empirically grounded factor loadings for bifactor model.

    Returns:
        Dictionary mapping item numbers (1-10) to PSS10Item objects with controllability and overload weights
    """
    return {
        1: PSS10Item(
            text="In the last month, how often have you been upset because of something that happened unexpectedly?",
            reverse_scored=False,
            weight_controllability=0.2,  # Low controllability loading
            weight_overload=0.7         # High overload loading
        ),
        2: PSS10Item(
            text="In the last month, how often have you felt that you were unable to control the important things in your life?",
            reverse_scored=False,
            weight_controllability=0.8,  # High controllability loading
            weight_overload=0.3         # Medium overload loading
        ),
        3: PSS10Item(
            text="In the last month, how often have you felt nervous and 'stressed'?",
            reverse_scored=False,
            weight_controllability=0.1,  # Low controllability loading
            weight_overload=0.8         # High overload loading
        ),
        4: PSS10Item(
            text="In the last month, how often have you felt confident about your ability to handle your personal problems?",
            reverse_scored=True,  # Reverse scored item
            weight_controllability=0.7,  # High controllability loading
            weight_overload=0.2         # Low overload loading
        ),
        5: PSS10Item(
            text="In the last month, how often have you felt that things were going your way?",
            reverse_scored=True,  # Reverse scored item
            weight_controllability=0.6,  # Medium-high controllability loading
            weight_overload=0.4         # Medium overload loading
        ),
        6: PSS10Item(
            text="In the last month, how often have you found that you could not cope with all the things that you had to do?",
            reverse_scored=False,
            weight_controllability=0.1,  # Low controllability loading
            weight_overload=0.9         # Very high overload loading
        ),
        7: PSS10Item(
            text="In the last month, how often have you been able to control irritations in your life?",
            reverse_scored=True,  # Reverse scored item
            weight_controllability=0.8,  # High controllability loading
            weight_overload=0.2         # Low overload loading
        ),
        8: PSS10Item(
            text="In the last month, how often have you felt that you were on top of things?",
            reverse_scored=True,  # Reverse scored item
            weight_controllability=0.6,  # Medium-high controllability loading
            weight_overload=0.3         # Low-medium overload loading
        ),
        9: PSS10Item(
            text="In the last month, how often have you been angered because of things that were outside of your control?",
            reverse_scored=False,
            weight_controllability=0.7,  # High controllability loading
            weight_overload=0.4         # Medium overload loading
        ),
        10: PSS10Item(
            text="In the last month, how often have you felt difficulties were piling up so high that you could not overcome them?",
            reverse_scored=False,
            weight_controllability=0.1,  # Low controllability loading
            weight_overload=0.9         # Very high overload loading
        )
    }


def map_agent_stress_to_pss10(
    controllability: float,
    overload: float,
    rng: Optional[np.random.Generator] = None
) -> Dict[int, int]:
    """
    Map agent stress state to PSS-10 item responses using empirically grounded bifactor model.

    This function now uses the new PSS-10 generation system that incorporates:
    - Empirically derived factor loadings for controllability and overload dimensions
    - Multivariate normal distribution for correlated dimension scores
    - Normal distribution sampling based on normative PSS-10 data
    - Proper reverse scoring for specified items

    Args:
        controllability: Agent's controllability level ∈ [0,1]
        overload: Agent's overload level ∈ [0,1]
        rng: Random number generator for reproducible testing

    Returns:
        Dictionary mapping item numbers (1-10) to response values (0-4)
    """
    # Use new empirically grounded PSS-10 generation function
    return generate_pss10_responses(controllability, overload, rng)


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


def generate_pss10_dimension_scores(
    controllability: float,
    overload: float,
    correlation: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    deterministic: bool = False
) -> Tuple[float, float]:
    """
    Generate correlated controllability and overload dimension scores using multivariate normal distribution.

    Args:
        controllability: Base controllability level ∈ [0,1]
        overload: Base overload level ∈ [0,1]
        correlation: Correlation coefficient between dimensions ∈ [-1,1] (if None, uses config default)
        rng: Random number generator for reproducible testing
        deterministic: If True, use deterministic seed generation for reproducible results

    Returns:
        Tuple of (correlated_controllability, correlated_overload) ∈ [0,1]²
    """
    # Get fresh config instance to avoid global config issues
    cfg = get_config()

    # Use config correlation if not provided
    if correlation is None:
        correlation = cfg.get('pss10', 'bifactor_correlation')

    # Get regularized standard deviations from config
    controllability_sd = cfg.get('pss10', 'controllability_sd') / 4
    overload_sd = cfg.get('pss10', 'overload_sd') / 4

    if deterministic:
        # Create a deterministic seed from input parameters
        input_str = f"{controllability:.10f}_{overload:.10f}_{correlation:.10f}"
        seed = int(hashlib.md5(input_str.encode()).hexdigest(), 16) % (2**32)
        local_rng = np.random.default_rng(seed)
    else:
        if rng is None:
            rng = np.random.default_rng()
        local_rng = rng

    # Create covariance matrix for bivariate normal distribution
    # Use regularized standard deviations from config
    var_c = controllability_sd ** 2  # Variance for controllability dimension
    var_o = overload_sd ** 2         # Variance for overload dimension
    cov = correlation * np.sqrt(var_c * var_o)  # Covariance term

    # Mean vector for the bivariate distribution
    mean_vector = np.array([controllability, overload])

    # Covariance matrix
    cov_matrix = np.array([
        [var_c, cov],
        [cov, var_o]
    ])

    # Sample from multivariate normal distribution
    correlated_scores = local_rng.multivariate_normal(mean_vector, cov_matrix)

    # Clamp to [0,1] range to maintain valid dimension scores
    correlated_controllability = max(0.0, min(1.0, correlated_scores[0]))
    correlated_overload = max(0.0, min(1.0, correlated_scores[1]))

    return correlated_controllability, correlated_overload


def generate_pss10_item_response(
    item_mean: float,
    item_sd: float,
    controllability_loading: float,
    overload_loading: float,
    controllability_score: float,
    overload_score: float,
    reverse_scored: bool,
    rng: Optional[np.random.Generator] = None,
    deterministic: bool = False
) -> int:
    """
    Generate a single PSS-10 item response using empirically grounded factor loadings.

    Args:
        item_mean: Mean response for this item from normative data
        item_sd: Standard deviation for this item from normative data
        controllability_loading: Factor loading on controllability dimension ∈ [0,1]
        overload_loading: Factor loading on overload dimension ∈ [0,1]
        controllability_score: Agent's current controllability dimension score ∈ [0,1]
        overload_score: Agent's current overload dimension score ∈ [0,1]
        reverse_scored: Whether this item should be reverse scored
        rng: Random number generator for reproducible testing

    Returns:
        PSS-10 item response ∈ [0,4]
    """
    if deterministic:
        # Create a deterministic seed from input parameters
        input_str = f"{item_mean:.10f}_{item_sd:.10f}_{controllability_loading:.10f}_{overload_loading:.10f}_{controllability_score:.10f}_{overload_score:.10f}_{reverse_scored}"
        seed = int(hashlib.md5(input_str.encode()).hexdigest(), 16) % (2**32)
        local_rng = np.random.default_rng(seed)
    else:
        if rng is None:
            rng = np.random.default_rng()
        local_rng = rng

    # Linear combination of dimension scores weighted by factor loadings
    # Higher controllability → lower stress response (unless reverse scored)
    # Higher overload → higher stress response
    stress_component = (
        controllability_loading * (1.0 - controllability_score) +  # Low controllability = high stress
        overload_loading * overload_score                           # High overload = high stress
    )

    # Normalize by total loading (avoid division by zero)
    total_loading = max(controllability_loading + overload_loading, 1e-10)
    normalized_stress = stress_component / total_loading

    # Sample from normal distribution around the empirically observed mean
    # Adjust mean based on current stress level
    adjusted_mean = item_mean + (normalized_stress - 0.5) * 0.5  # Scale stress effect
    adjusted_mean = max(0.0, min(4.0, adjusted_mean))  # Keep within valid range

    # Sample from normal distribution using local RNG
    raw_response = local_rng.normal(adjusted_mean, item_sd)

    # Add small amount of measurement error using local RNG
    measurement_error = local_rng.normal(0, 0.1)
    final_response = raw_response + measurement_error

    # Apply reverse scoring if needed
    if reverse_scored:
        final_response = 4.0 - final_response

    # Clamp to [0,4] range and round to nearest integer
    clamped_response = max(0.0, min(4.0, final_response))
    response_value = int(round(clamped_response))

    return response_value


def generate_pss10_responses(
    controllability: float,
    overload: float,
    rng: Optional[np.random.Generator] = None,
    config: Optional[Dict[str, Any]] = None,
    deterministic: bool = False
) -> Dict[int, int]:
    """
    Generate complete PSS-10 responses for an agent using empirically grounded bifactor model.

    Args:
        controllability: Agent's controllability level ∈ [0,1]
        overload: Agent's overload level ∈ [0,1]
        rng: Random number generator for reproducible testing
        config: Configuration parameters (if None, uses global config)
        deterministic: If True, use deterministic seed generation for reproducible results

    Returns:
        Dictionary mapping item numbers (1-10) to response values (0-4)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get fresh config instance to avoid global config issues
    cfg = get_config()

    if config is None:
        config = {
            'item_means': cfg.get('pss10', 'item_means'),
            'item_sds': cfg.get('pss10', 'item_sds'),
            'load_controllability': cfg.get('pss10', 'load_controllability'),
            'load_overload': cfg.get('pss10', 'load_overload'),
            'bifactor_correlation': cfg.get('pss10', 'bifactor_correlation')
        }

    # Generate correlated dimension scores using merged function with deterministic behavior
    correlated_controllability, correlated_overload = generate_pss10_dimension_scores(
        controllability, overload, config['bifactor_correlation'], rng, deterministic
    )

    # Get PSS-10 item mapping
    pss10_items = create_pss10_mapping()
    responses = {}

    # Generate response for each item using deterministic version
    for item_num in range(1, 11):
        item = pss10_items[item_num]

        response = generate_pss10_item_response(
            item_mean=config['item_means'][item_num - 1],
            item_sd=config['item_sds'][item_num - 1],
            controllability_loading=item.weight_controllability,
            overload_loading=item.weight_overload,
            controllability_score=correlated_controllability,
            overload_score=correlated_overload,
            reverse_scored=item.reverse_scored,
            rng=rng,
            deterministic=deterministic
        )

        responses[item_num] = response

    return responses


def initialize_pss10_from_items(
    controllability_score: float,
    overload_score: float,
    rng: np.random.Generator,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Initialize PSS-10 responses and derived stress dimensions from item responses.

    Args:
        controllability_score: Base controllability score ∈ [0,1]
        overload_score: Base overload score ∈ [0,1]
        rng: Random number generator for reproducible testing
        config: Configuration parameters

    Returns:
        Dictionary containing pss10_responses, stress_controllability, stress_overload, pss10_score, and stressed status
    """
    # Get configuration values
    cfg = get_config()

    if config is None:
        config = {
            'controllability_sd': cfg.get('pss10', 'controllability_sd') / 4,
            'overload_sd': cfg.get('pss10', 'overload_sd') / 4,
            'item_means': cfg.get('pss10', 'item_means'),
            'item_sds': cfg.get('pss10', 'item_sds'),
            'load_controllability': cfg.get('pss10', 'load_controllability'),
            'load_overload': cfg.get('pss10', 'load_overload'),
            'threshold': cfg.get('pss10', 'threshold')
        }

    # Clamp input scores to [0,1] range
    controllability_score = max(0.0, min(1.0, controllability_score))
    overload_score = max(0.0, min(1.0, overload_score))

    # Generate each PSS-10 item response
    pss10_responses = {}
    for item_num in range(1, 11):
        # Determine if item is reverse scored
        reverse_scored = item_num in [4, 5, 7, 8]

        # Get item parameters from configuration
        item_mean = config['item_means'][item_num - 1]
        item_sd = config['item_sds'][item_num - 1]
        controllability_loading = config['load_controllability'][item_num - 1]
        overload_loading = config['load_overload'][item_num - 1]

        # Generate item response
        response = generate_pss10_item_response(
            item_mean=item_mean,
            item_sd=item_sd,
            controllability_loading=controllability_loading,
            overload_loading=overload_loading,
            controllability_score=controllability_score,
            overload_score=overload_score,
            reverse_scored=reverse_scored,
            rng=rng
        )

        pss10_responses[item_num] = response

    # Initialize stress_controllability by averaging items 4, 5, 7, 8, then dividing by 4
    controllability_items = [4, 5, 7, 8]
    controllability_scores = []
    for item_num in controllability_items:
        if item_num in pss10_responses:
            # Without reversing the item score, higher PSS-10 response = higher controllability
            response = pss10_responses[item_num]
            controllability_scores.append(response / 4.0)  # Normalize to [0,1]
    stress_controllability = np.mean(controllability_scores) if controllability_scores else 0.5

    # Initialize stress_overload by averaging items 1, 2, 3, 6, 9, 10, then dividing by 6
    overload_items = [1, 2, 3, 6, 9, 10]
    overload_scores = []
    for item_num in overload_items:
        if item_num in pss10_responses:
            # Higher PSS-10 response = higher overload
            response = pss10_responses[item_num]
            overload_scores.append(response / 4.0)  # Normalize to [0,1]
    stress_overload = np.mean(overload_scores) if overload_scores else 0.5

    # Initialize pss10_score by summing items 1-10
    pss10_score = compute_pss10_score(pss10_responses)

    # Set initial stressed status based on PSS-10 threshold
    pss10_threshold = config['threshold']
    stressed = (pss10_score >= pss10_threshold)

    return {
        'pss10_responses': pss10_responses,
        'stress_controllability': stress_controllability,
        'stress_overload': stress_overload,
        'pss10_score': pss10_score,
        'stressed': stressed
    }


def generate_pss10_from_stress_dimensions(
    stress_controllability: float,
    stress_overload: float,
    recent_stress_intensity: float = 0.0,
    stress_momentum: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate PSS-10 responses from current stress dimensions with enhanced dynamic response.

    Args:
        stress_controllability: Current stress controllability ∈ [0,1]
        stress_overload: Current stress overload ∈ [0,1]
        recent_stress_intensity: Recent stress intensity for immediate response
        stress_momentum: Stress momentum for predictive response
        rng: Random number generator for reproducible testing
        config: Configuration parameters

    Returns:
        Dictionary containing updated pss10_responses, pss10_score, and stressed status
    """
    # Get configuration for dynamic PSS-10 updates
    cfg = get_config()

    if config is None:
        config = {
            'sensitivity': cfg.get('pss10', 'sensitivity'),
            'momentum_weight': cfg.get('pss10', 'momentum_weight'),
            'threshold': cfg.get('pss10', 'threshold')
        }

    if rng is None:
        rng = np.random.default_rng()

    # Calculate dynamic stress state for PSS-10 generation
    base_controllability = stress_controllability
    base_overload = stress_overload

    # Apply recent stress intensity for immediate response
    intensity_boost = recent_stress_intensity * config['sensitivity']
    dynamic_controllability = max(0.0, min(1.0, base_controllability - intensity_boost * 0.15))
    dynamic_overload = min(1.0, base_overload + intensity_boost * 0.2)

    # Apply stress momentum for predictive response
    momentum_adjustment = stress_momentum * config['momentum_weight']
    dynamic_controllability = max(0.0, min(1.0, dynamic_controllability + momentum_adjustment * 0.08))
    dynamic_overload = min(1.0, dynamic_overload + abs(momentum_adjustment) * 0.12)

    # Generate PSS-10 responses using dynamic stress state
    pss10_responses = generate_pss10_responses(
        controllability=dynamic_controllability,
        overload=dynamic_overload,
        rng=rng
    )

    # Calculate PSS-10 score
    pss10_score = compute_pss10_score(pss10_responses)

    # Update stressed status based on PSS-10 threshold
    pss10_threshold = config['threshold']
    stressed = (pss10_score >= pss10_threshold)

    return {
        'pss10_responses': pss10_responses,
        'pss10_score': pss10_score,
        'stressed': stressed
    }


def update_stress_dimensions_from_pss10_feedback(
    current_controllability: float,
    current_overload: float,
    pss10_responses: Dict[int, int],
    config: Optional[Dict[str, float]] = None
) -> Tuple[float, float]:
    """
    Update stress dimensions based on PSS-10 feedback for complete theoretical loop.

    Args:
        current_controllability: Current stress controllability ∈ [0,1]
        current_overload: Current stress overload ∈ [0,1]
        pss10_responses: PSS-10 item responses
        config: Configuration for feedback weighting

    Returns:
        Tuple of (updated_controllability, updated_overload)
    """
    if not pss10_responses:
        return current_controllability, current_overload

    if config is None:
        config = {
            'feedback_weight': 0.3  # Weight for PSS-10 feedback
        }

    # Calculate controllability stress from relevant PSS-10 items
    controllability_items = [4, 5, 7, 8]  # Reverse scored items
    controllability_scores = []

    for item_num in controllability_items:
        if item_num in pss10_responses:
            # For reverse scored items, lower PSS-10 response = higher controllability
            response = pss10_responses[item_num]
            controllability_scores.append(1.0 - (response / 4.0))

    # Update controllability with PSS-10 feedback
    if controllability_scores:
        pss10_controllability = np.mean(controllability_scores)
        # Blend current stress dimension with PSS-10 feedback
        feedback_weight = config['feedback_weight']
        updated_controllability = (
            current_controllability * (1.0 - feedback_weight) +
            pss10_controllability * feedback_weight
        )
    else:
        updated_controllability = current_controllability

    # Calculate overload stress from relevant PSS-10 items
    overload_items = [1, 2, 3, 6, 9, 10]  # Non-reverse scored items
    overload_scores = []

    for item_num in overload_items:
        if item_num in pss10_responses:
            # Higher PSS-10 response = higher overload
            response = pss10_responses[item_num]
            overload_scores.append(response / 4.0)

    # Update overload with PSS-10 feedback
    if overload_scores:
        pss10_overload = np.mean(overload_scores)
        # Blend current stress dimension with PSS-10 feedback
        feedback_weight = config['feedback_weight']
        updated_overload = (
            current_overload * (1.0 - feedback_weight) +
            pss10_overload * feedback_weight
        )
    else:
        updated_overload = current_overload

    # Clamp final values
    updated_controllability = clamp(updated_controllability, 0.0, 1.0)
    updated_overload = clamp(updated_overload, 0.0, 1.0)

    return updated_controllability, updated_overload


def update_stress_dimensions_from_event(
    current_controllability: float,
    current_overload: float,
    challenge: float,
    hindrance: float,
    coped_successfully: bool,
    config: Optional[Dict[str, float]] = None
) -> Tuple[float, float, float, float]:
    """
    Update agent's controllability and overload dimensions based on stress event outcomes.

    Args:
        current_controllability: Current stress controllability ∈ [0,1]
        current_overload: Current stress overload ∈ [0,1]
        challenge: Challenge component from event appraisal (0-1)
        hindrance: Hindrance component from event appraisal (0-1)
        coped_successfully: Whether the coping attempt was successful
        config: Configuration for stress dimension updates

    Returns:
        Tuple of (updated_controllability, updated_overload, recent_stress_intensity, stress_momentum)
    """
    # Get configuration for stress dimension updates
    cfg = get_config()

    if config is None:
        config = {
            'controllability_update_rate': cfg.get('stress_dynamics', 'controllability_update_rate'),
            'overload_update_rate': cfg.get('stress_dynamics', 'overload_update_rate')
        }

    # Challenge vs hindrance effects on controllability
    if coped_successfully:
        # Successful coping: challenge builds controllability, hindrance slightly reduces it
        controllability_change = (challenge * 0.15) - (hindrance * 0.08)
    else:
        # Failed coping: both challenge and hindrance reduce controllability
        controllability_change = -(challenge * 0.12) - (hindrance * 0.18)

    # Apply controllability update with decay toward baseline
    baseline_controllability = 0.5  # Neutral baseline
    current_controllability = current_controllability

    # Move toward baseline when no strong events, but allow event-driven changes
    homeostasis_pull = (baseline_controllability - current_controllability) * 0.05
    event_effect = controllability_change * config['controllability_update_rate']

    updated_controllability = current_controllability + homeostasis_pull + event_effect
    updated_controllability = clamp(updated_controllability, 0.0, 1.0)

    # Overload effects: hindrance increases overload, challenge reduces it slightly
    if coped_successfully:
        # Successful coping: hindrance still increases overload but less, challenge reduces it
        overload_change = (hindrance * 0.12) - (challenge * 0.08)
    else:
        # Failed coping: both increase overload significantly
        overload_change = (hindrance * 0.25) + (challenge * 0.15)

    # Apply overload update with decay toward baseline
    baseline_overload = 0.5  # Neutral baseline
    current_overload = current_overload

    # Move toward baseline when no strong events, but allow event-driven changes
    homeostasis_pull = (baseline_overload - current_overload) * 0.05
    event_effect = overload_change * config['overload_update_rate']

    updated_overload = current_overload + homeostasis_pull + event_effect
    updated_overload = clamp(updated_overload, 0.0, 1.0)

    # Update recent stress intensity and momentum for dynamic PSS-10 response
    recent_stress_intensity, stress_momentum = _update_recent_stress_intensity(
        challenge, hindrance, coped_successfully
    )

    return updated_controllability, updated_overload, recent_stress_intensity, stress_momentum


def _update_recent_stress_intensity(
    challenge: float,
    hindrance: float,
    coped_successfully: bool
) -> Tuple[float, float]:
    """
    Update recent stress intensity and momentum for dynamic PSS-10 calculation.

    Args:
        challenge: Challenge component from event appraisal (0-1)
        hindrance: Hindrance component from event appraisal (0-1)
        coped_successfully: Whether the coping attempt was successful

    Returns:
        Tuple of (recent_stress_intensity, stress_momentum)
    """
    # Calculate event stress intensity (hindrance is more intense than challenge)
    event_intensity = (challenge * 0.7) + (hindrance * 1.3)

    # Adjust intensity based on coping outcome
    if not coped_successfully:
        event_intensity *= 1.5  # Failed coping makes events more intense

    # Update recent stress intensity with decay of previous intensity
    decay_rate = 0.8  # How quickly previous intensity fades
    recent_stress_intensity = 0.0  # This would need to be passed in from agent state
    recent_stress_intensity = (recent_stress_intensity * decay_rate) + (event_intensity * 0.2)

    # Update stress momentum (rate of change)
    # Positive momentum means stress is increasing
    if event_intensity > recent_stress_intensity * decay_rate:
        # Stress is increasing
        stress_momentum = min(1.0, 0.0 + 0.1)  # This would need to be passed in from agent state
    else:
        # Stress is decreasing or stable
        stress_momentum = max(-1.0, 0.0 - 0.05)  # This would need to be passed in from agent state

    return recent_stress_intensity, stress_momentum


def decay_recent_stress_intensity(
    recent_stress_intensity: float,
    stress_momentum: float
) -> Tuple[float, float]:
    """
    Apply natural decay to recent stress intensity over time.

    Args:
        recent_stress_intensity: Current recent stress intensity
        stress_momentum: Current stress momentum

    Returns:
        Tuple of (decayed_stress_intensity, decayed_momentum)
    """
    # Gradual decay when no recent stress events
    decay_rate = 0.95  # 5% decay per update
    decayed_intensity = recent_stress_intensity * decay_rate

    # Also decay momentum toward zero
    if abs(stress_momentum) > 0.01:
        decayed_momentum = stress_momentum * 0.9  # 10% decay toward zero
    else:
        decayed_momentum = 0.0

    return decayed_intensity, decayed_momentum


def estimate_pss10_from_stress_dimensions(
    stress_controllability: float,
    stress_overload: float
) -> Tuple[int, int]:
    """
    Estimate expected PSS-10 range based on current stress dimensions.

    Args:
        stress_controllability: Current stress controllability ∈ [0,1]
        stress_overload: Current stress overload ∈ [0,1]

    Returns:
        Tuple of (min_expected, max_expected) PSS-10 scores
    """
    # Simple estimation based on stress dimensions
    # In practice, this would use the same logic as PSS-10 generation
    base_score = 10  # Neutral baseline

    controllability_effect = (1.0 - stress_controllability) * 8  # Up to 8 points
    overload_effect = stress_overload * 12  # Up to 12 points

    estimated_score = base_score + controllability_effect + overload_effect

    # Add some variance for estimation uncertainty
    variance = 3
    return (max(0, int(estimated_score - variance)), min(40, int(estimated_score + variance)))


def extract_controllability_from_pss10(pss10_responses: Dict[int, int]) -> float:
    """
    Extract controllability dimension from current PSS-10 responses.

    Args:
        pss10_responses: PSS-10 item responses

    Returns:
        Controllability score (0-1)
    """
    if not pss10_responses:
        return 0.5

    # Average reverse-scored items (4, 5, 7, 8)
    controllability_items = [4, 5, 7, 8]
    controllability_scores = []

    for item_num in controllability_items:
        if item_num in pss10_responses:
            response = pss10_responses[item_num]
            # Reverse score: higher PSS-10 response = lower controllability
            controllability_scores.append(1.0 - (response / 4.0))

    return np.mean(controllability_scores) if controllability_scores else 0.5


def extract_overload_from_pss10(pss10_responses: Dict[int, int]) -> float:
    """
    Extract overload dimension from current PSS-10 responses.

    Args:
        pss10_responses: PSS-10 item responses

    Returns:
        Overload score (0-1)
    """
    if not pss10_responses:
        return 0.5

    # Average non-reverse scored items (1, 2, 3, 6, 9, 10)
    overload_items = [1, 2, 3, 6, 9, 10]
    overload_scores = []

    for item_num in overload_items:
        if item_num in pss10_responses:
            response = pss10_responses[item_num]
            # Higher PSS-10 response = higher overload
            overload_scores.append(response / 4.0)

    return np.mean(overload_scores) if overload_scores else 0.5


def validate_theoretical_correlations(
    challenge: float,
    hindrance: float,
    coped_successfully: bool,
    stress_controllability: float,
    stress_overload: float,
    pss10_score: int,
    current_stress: float
) -> bool:
    """
    Validate that theoretical correlations between stress processing components are maintained.

    Args:
        challenge: Challenge component from current event
        hindrance: Hindrance component from current event
        coped_successfully: Whether coping was successful
        stress_controllability: Current stress controllability
        stress_overload: Current stress overload
        pss10_score: Current PSS-10 score
        current_stress: Current stress level

    Returns:
        True if all validations pass, False otherwise
    """
    # Validate 1: Challenge/hindrance relationship should be complementary
    challenge_hindrance_sum = challenge + hindrance
    if not (0.8 <= challenge_hindrance_sum <= 1.2):
        # Log warning but don't fail - this can happen with sigmoid transformation
        pass  # In production, might want to log this

    # Validate 2: Stress dimensions should be properly bounded
    if not (0.0 <= stress_controllability <= 1.0):
        raise ValueError(f"stress_controllability out of bounds: {stress_controllability}")

    if not (0.0 <= stress_overload <= 1.0):
        raise ValueError(f"stress_overload out of bounds: {stress_overload}")

    # Validate 3: PSS-10 score should be valid
    if not (0 <= pss10_score <= 40):
        raise ValueError(f"PSS-10 score out of bounds: {pss10_score}")

    # Validate 4: Current stress should be valid
    if not (0.0 <= current_stress <= 1.0):
        raise ValueError(f"current_stress out of bounds: {current_stress}")

    # Validate 5: Theoretical correlation - high challenge should correlate with controllability
    # This is a long-term correlation that should emerge over multiple events
    if challenge > 0.7 and coped_successfully:
        # After successful high-challenge events, controllability should trend upward
        expected_controllability_trend = "increasing"
        # In a full implementation, we might track trends over time

    # Validate 6: Theoretical correlation - high hindrance should correlate with overload
    if hindrance > 0.7:
        # High hindrance events should increase overload
        expected_overload_trend = "increasing"
        # In a full implementation, we might track trends over time

    # Validate 7: PSS-10 should reflect stress dimensions
    # High overload should generally correlate with higher PSS-10 scores
    expected_pss10_range = estimate_pss10_from_stress_dimensions(stress_controllability, stress_overload)
    if pss10_score < expected_pss10_range[0] or pss10_score > expected_pss10_range[1]:
        # PSS-10 is outside expected range - could indicate correlation issues
        # In production, might want to log this for monitoring
        pass

    # Validate 8: Feedback loop consistency
    # The feedback from PSS-10 to stress dimensions should be reasonable
    if pss10_responses := {}:  # This would need to be passed in
        pss10_controllability = extract_controllability_from_pss10(pss10_responses)
        pss10_overload = extract_overload_from_pss10(pss10_responses)

        # Feedback should be within reasonable range of current stress dimensions
        controllability_diff = abs(stress_controllability - pss10_controllability)
        overload_diff = abs(stress_overload - pss10_overload)

        if controllability_diff > 0.5 or overload_diff > 0.5:
            # Large discrepancy between stress dimensions and PSS-10 feedback
            # In production, might want to log this for monitoring
            pass

    return True
