"""
Affect dynamics and social interaction utilities for agent-based mental health simulation.

This module contains stateless functions for:
- Affect changes due to social interactions
- Stress impact on affect
- Recovery mechanisms
- Social influence calculations
- All functions are pure and support dependency injection for testability
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from src.python.config import get_config
from src.python.math_utils import clamp, softmax

# Load configuration
config = get_config()


@dataclass
class InteractionConfig:
    """Configuration parameters for social interactions."""
    influence_rate: float = field(default_factory=lambda: get_config().get('interaction', 'influence_rate'))
    resilience_influence: float = field(default_factory=lambda: get_config().get('interaction', 'resilience_influence'))
    max_neighbors: int = field(default_factory=lambda: get_config().get('interaction', 'max_neighbors'))


@dataclass
class ProtectiveFactors:
    """Agent's protective factors for resource allocation."""
    social_support: float = field(default_factory=lambda: get_config().get('protective', 'social_support'))
    family_support: float = field(default_factory=lambda: get_config().get('protective', 'family_support'))
    formal_intervention: float = field(default_factory=lambda: get_config().get('protective', 'formal_intervention'))
    psychological_capital: float = field(default_factory=lambda: get_config().get('protective', 'psychological_capital'))


@dataclass
class ResourceParams:
    """Parameters for resource dynamics."""
    base_regeneration: float = field(default_factory=lambda: get_config().get('resource', 'base_regeneration'))
    allocation_cost: float = field(default_factory=lambda: get_config().get('resource', 'allocation_cost'))
    cost_exponent: float = field(default_factory=lambda: get_config().get('resource', 'cost_exponent'))


def compute_social_influence(
    self_affect: float,
    partner_affect: float,
    config: Optional[InteractionConfig] = None
) -> float:
    """
    Compute affect change due to social interaction.

    Args:
        self_affect: Agent's current affect
        partner_affect: Interaction partner's affect
        config: Interaction configuration parameters

    Returns:
        Affect change (positive = improvement, negative = deterioration)
    """
    if config is None:
        config = InteractionConfig()

    # Influence is proportional to partner's affect and base influence rate
    # Positive partner affect pulls self upward, negative pulls downward
    influence = config.influence_rate * np.sign(partner_affect)

    return influence


def compute_mutual_influence(
    self_affect: float,
    partner_affect: float,
    config: Optional[InteractionConfig] = None
) -> tuple[float, float]:
    """
    Compute mutual affect changes for both agents in an interaction.

    Args:
        self_affect: First agent's affect
        partner_affect: Second agent's affect
        config: Interaction configuration

    Returns:
        Tuple of (self_affect_change, partner_affect_change)
    """
    if config is None:
        config = InteractionConfig()

    # Each agent influences the other based on their affect
    self_influence = compute_social_influence(partner_affect, self_affect, config)
    partner_influence = compute_social_influence(self_affect, partner_affect, config)

    return self_influence, partner_influence


def compute_resilience_influence(
    partner_affect: float,
    config: Optional[InteractionConfig] = None
) -> float:
    """
    Compute resilience change due to partner's affect.

    Args:
        partner_affect: Interaction partner's affect
        config: Interaction configuration

    Returns:
        Resilience change
    """
    if config is None:
        config = InteractionConfig()

    # Positive affect from partner improves resilience
    return config.resilience_influence * partner_affect


def process_interaction(
    self_affect: float,
    partner_affect: float,
    self_resilience: float,
    partner_resilience: float,
    config: Optional[InteractionConfig] = None
) -> tuple[float, float, float, float]:
    """
    Process a complete social interaction between two agents.

    Args:
        self_affect: First agent's affect
        partner_affect: Second agent's affect
        self_resilience: First agent's resilience
        partner_resilience: Second agent's resilience
        config: Interaction configuration

    Returns:
        Tuple of (new_self_affect, new_partner_affect,
                 new_self_resilience, new_partner_resilience)
    """
    if config is None:
        config = InteractionConfig()

    # Compute mutual influences
    affect_influence_self, affect_influence_partner = compute_mutual_influence(
        self_affect, partner_affect, config
    )

    # Apply affect changes
    new_self_affect = self_affect + affect_influence_self
    new_partner_affect = partner_affect + affect_influence_partner

    # Compute resilience changes based on partner's affect
    resilience_influence_self = compute_resilience_influence(partner_affect, config)
    resilience_influence_partner = compute_resilience_influence(self_affect, config)

    new_self_resilience = self_resilience + resilience_influence_self
    new_partner_resilience = partner_resilience + resilience_influence_partner

    # Clamp all values to valid ranges
    new_self_affect = clamp(new_self_affect, -1.0, 1.0)
    new_partner_affect = clamp(new_partner_affect, -1.0, 1.0)
    new_self_resilience = clamp(new_self_resilience, 0.0, 1.0)
    new_partner_resilience = clamp(new_partner_resilience, 0.0, 1.0)

    return new_self_affect, new_partner_affect, new_self_resilience, new_partner_resilience


def compute_stress_impact_on_affect(
    current_affect: float,
    is_stressed: bool,
    coped_successfully: bool,
    config: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute affect change due to stress event outcome.

    Args:
        current_affect: Agent's current affect
        is_stressed: Whether the agent experienced stress
        coped_successfully: Whether coping was successful
        config: Configuration for affect changes

    Returns:
        Affect change
    """
    if config is None:
        # Get fresh config instance to avoid global config issues
        cfg = get_config()
        config = {
            'coping_improvement': cfg.get('agent', 'coping_success_rate') * 0.2,  # Scale based on success rate
            'coping_deterioration': cfg.get('agent', 'coping_success_rate') * 0.4,  # Scale based on success rate
            'no_stress_effect': 0.0 # No change if not stressed
        }

    if not is_stressed:
        return config['no_stress_effect']

    if coped_successfully:
        return config['coping_improvement']
    else:
        return -config['coping_deterioration']


def compute_stress_impact_on_resilience(
    current_resilience: float,
    is_stressed: bool,
    coped_successfully: bool,
    config: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute resilience change due to stress event outcome.

    Args:
        current_resilience: Agent's current resilience
        is_stressed: Whether the agent experienced stress
        coped_successfully: Whether coping was successful
        config: Configuration for resilience changes

    Returns:
        Resilience change
    """
    if config is None:
        # Get fresh config instance to avoid global config issues
        cfg = get_config()
        config = {
            'coping_improvement': cfg.get('agent', 'coping_success_rate') * 0.1,  # Scale based on success rate
            'coping_deterioration': cfg.get('agent', 'coping_success_rate') * 0.2,  # Scale based on success rate
            'no_stress_effect': 0.0 # No change if not stressed
        }

    if not is_stressed:
        return config['no_stress_effect']

    if coped_successfully:
        return config['coping_improvement']
    else:
        return -config['coping_deterioration']


def allocate_protective_resources(
    available_resources: float,
    protective_factors: Optional[ProtectiveFactors] = None,
    rng: Optional[np.random.Generator] = None,
    config: Optional[ResourceParams] = None
) -> Dict[str, float]:
    """
    Allocate resources across protective factors using softmax decision making.

    Args:
        available_resources: Total resources available for allocation
        protective_factors: Current efficacy levels of protective factors
        rng: Random number generator for stochastic decisions
        config: Resource allocation parameters

    Returns:
        Dictionary mapping factor names to allocated resources
    """
    cfg = get_config()

    if protective_factors is None:
        protective_factors = ProtectiveFactors()

    if config is None:
        config = ResourceParams()

    if rng is None:
        rng = np.random.default_rng()

    # Create allocation weights based on efficacy
    factors = ['social_support', 'family_support', 'formal_intervention', 'psychological_capital']
    efficacies = [
        protective_factors.social_support,
        protective_factors.family_support,
        protective_factors.formal_intervention,
        protective_factors.psychological_capital
    ]

    # Softmax decision making with temperature from config
    temperature = cfg.get('utility', 'softmax_temperature')
    logits = np.array(efficacies) / temperature
    softmax_weights = np.exp(logits) / np.sum(np.exp(logits))

    # Allocate resources proportionally
    allocations = {
        factor: available_resources * weight
        for factor, weight in zip(factors, softmax_weights)
    }

    return allocations


def compute_resource_regeneration(
    current_resources: float,
    config: Optional[ResourceParams] = None
) -> float:
    """
    Compute passive resource regeneration.

    Args:
        current_resources: Current resource level
        config: Resource parameters

    Returns:
        Resource regeneration amount
    """
    if config is None:
        config = ResourceParams()

    # Linear regeneration: γ_R * (1 - current_resources)
    # This ensures resources tend toward 1.0 over time
    regeneration = config.base_regeneration * (1.0 - current_resources)

    return regeneration


def compute_allocation_cost(
    allocated_amount: float,
    config: Optional[ResourceParams] = None
) -> float:
    """
    Compute cost of allocating resources (convex cost function).

    Args:
        allocated_amount: Amount of resources allocated
        config: Resource parameters

    Returns:
        Cost of allocation
    """
    if config is None:
        config = ResourceParams()

    # Convex cost function: κ * allocated^γ_c
    cost = config.allocation_cost * (allocated_amount ** config.cost_exponent)

    return cost


# ==============================================
# ENHANCED AFFECT AND RESILIENCE DYNAMICS
# ==============================================

@dataclass
class AffectDynamicsConfig:
    """Configuration parameters for affect dynamics."""
    peer_influence_rate: float = field(default_factory=lambda: get_config().get('affect_dynamics', 'peer_influence_rate'))
    event_appraisal_rate: float = field(default_factory=lambda: get_config().get('affect_dynamics', 'event_appraisal_rate'))
    homeostatic_rate: float = field(default_factory=lambda: get_config().get('affect_dynamics', 'homeostatic_rate'))
    influencing_neighbors: int = field(default_factory=lambda: get_config().get('influence', 'influencing_neighbors'))


@dataclass
class ResilienceDynamicsConfig:
    """Configuration parameters for resilience dynamics."""
    coping_success_rate: float = field(default_factory=lambda: get_config().get('resilience_dynamics', 'coping_success_rate'))
    social_support_rate: float = field(default_factory=lambda: get_config().get('resilience_dynamics', 'social_support_rate'))
    overload_threshold: int = field(default_factory=lambda: get_config().get('resilience_dynamics', 'overload_threshold'))
    influencing_hindrance: int = field(default_factory=lambda: get_config().get('influence', 'influencing_hindrance'))


def compute_peer_influence(
    self_affect: float,
    neighbor_affects: List[float],
    config: Optional[AffectDynamicsConfig] = None
) -> float:
    """
    Compute aggregated influence from multiple neighbors on an agent's affect.

    Args:
        self_affect: Agent's current affect
        neighbor_affects: List of neighbor affect values
        config: Affect dynamics configuration

    Returns:
        Net affect change from peer influence
    """
    if config is None:
        config = AffectDynamicsConfig()

    if not neighbor_affects:
        return 0.0

    # Limit to specified number of influencing neighbors
    n_neighbors = min(len(neighbor_affects), config.influencing_neighbors)
    selected_affects = neighbor_affects[:n_neighbors]

    # Compute influence from each neighbor
    influences = []
    for neighbor_affect in selected_affects:
        # Positive neighbor affect pulls self upward, negative pulls downward
        raw_influence = config.peer_influence_rate * (neighbor_affect - self_affect)
        influences.append(raw_influence)

    # Average the influences and apply homeostasis consideration
    avg_influence = np.mean(influences)

    return avg_influence


def compute_event_appraisal_effect(
    challenge: float,
    hindrance: float,
    current_affect: float,
    config: Optional[AffectDynamicsConfig] = None
) -> float:
    """
    Compute affect change based on challenge/hindrance appraisal of events.

    Args:
        challenge: Challenge component from event appraisal (0-1)
        hindrance: Hindrance component from event appraisal (0-1)
        current_affect: Agent's current affect
        config: Affect dynamics configuration

    Returns:
        Affect change from event appraisal
    """
    if config is None:
        config = AffectDynamicsConfig()

    # Challenge tends to improve affect (motivating), hindrance tends to worsen it
    # The effect is stronger when current affect is more extreme
    challenge_effect = config.event_appraisal_rate * challenge * (1.0 - current_affect)
    hindrance_effect = -config.event_appraisal_rate * hindrance * (current_affect + 1.0)

    total_effect = challenge_effect + hindrance_effect

    return total_effect


def compute_homeostasis_effect(
    current_affect: float,
    baseline_affect: float = 0.0,
    config: Optional[AffectDynamicsConfig] = None
) -> float:
    """
    Compute tendency of affect to return to baseline (homeostasis).

    Args:
        current_affect: Agent's current affect
        baseline_affect: Agent's baseline affect level
        config: Affect dynamics configuration

    Returns:
        Affect change toward baseline
    """
    if config is None:
        config = AffectDynamicsConfig()

    # Homeostasis pulls affect toward baseline
    # Strength increases with distance from baseline
    distance_from_baseline = baseline_affect - current_affect
    homeostasis_strength = config.homeostatic_rate * abs(distance_from_baseline)

    # Direction toward baseline
    if distance_from_baseline > 0:
        # Current affect is below baseline, push up
        return homeostasis_strength
    else:
        # Current affect is above baseline, push down
        return -homeostasis_strength


def compute_homeostatic_adjustment(
    initial_value: float,
    final_value: float,
    homeostatic_rate: Optional[float] = None,
    value_type: str = 'affect'
) -> float:
    """
    Apply homeostatic adjustment to pull values back toward initial state.

    This function implements a homeostatic mechanism that adjusts values toward
    their initial state at the beginning of each day, simulating natural
    tendencies to return to baseline levels.

    Args:
        initial_value: Value at the start of the day (baseline)
        final_value: Value at the end of the day after all actions
        homeostatic_rate: Rate of homeostatic adjustment (0-1).
                         If None, uses config default.
        value_type: Type of value being adjusted ('affect' or 'resilience')

    Returns:
        Homeostatically adjusted value

    Raises:
        ValueError: If value_type is not 'affect' or 'resilience'
    """
    if homeostatic_rate is None:
        homeostatic_rate = get_config().get('affect_dynamics', 'homeostatic_rate')

    # Validate value_type
    if value_type not in ['affect', 'resilience']:
        raise ValueError(f"value_type must be 'affect' or 'resilience', got '{value_type}'")

    # Calculate distance from initial value
    distance = homeostatic_rate * abs(final_value - initial_value)

    # Determine adjustment direction
    if final_value > initial_value:
        # Final value is above initial, adjust downward
        adjusted_value = final_value - distance
    elif final_value < initial_value:
        # Final value is below initial, adjust upward
        adjusted_value = final_value + distance
    else:
        # Values are equal, no adjustment needed
        adjusted_value = final_value

    # Apply appropriate normalization based on value type
    if value_type == 'affect':
        # Affect values normalized to [-1, 1]
        adjusted_value = clamp(adjusted_value, -1.0, 1.0)
    else:  # resilience
        # Resilience values normalized to [0, 1]
        adjusted_value = clamp(adjusted_value, 0.0, 1.0)

    return adjusted_value


def compute_cumulative_overload(
    consecutive_hindrances: int,
    config: Optional[ResilienceDynamicsConfig] = None
) -> float:
    """
    Compute resilience impact from cumulative hindrance events.

    Args:
        consecutive_hindrances: Number of consecutive hindrance events
        config: Resilience dynamics configuration

    Returns:
        Resilience change from overload effect
    """
    if config is None:
        config = ResilienceDynamicsConfig()

    # Overload effect only occurs after threshold is reached
    if consecutive_hindrances < config.overload_threshold:
        return 0.0

    # Overload effect increases with more consecutive hindrances
    overload_intensity = min(consecutive_hindrances / config.influencing_hindrance, 2.0)

    # Overload reduces resilience significantly
    return -0.2 * overload_intensity


def update_affect_dynamics(
    current_affect: float,
    baseline_affect: float,
    neighbor_affects: List[float],
    challenge: float = 0.0,
    hindrance: float = 0.0,
    affect_config: Optional[AffectDynamicsConfig] = None
) -> float:
    """
    Update agent's affect based on peer influence, event appraisal, and homeostasis.

    Args:
        current_affect: Agent's current affect
        baseline_affect: Agent's baseline affect level
        neighbor_affects: List of neighbor affect values
        challenge: Challenge component from recent events
        hindrance: Hindrance component from recent events
        affect_config: Affect dynamics configuration

    Returns:
        New affect value
    """
    if affect_config is None:
        affect_config = AffectDynamicsConfig()

    # Compute individual effect components
    peer_effect = compute_peer_influence(current_affect, neighbor_affects, affect_config)
    appraisal_effect = compute_event_appraisal_effect(challenge, hindrance, current_affect, affect_config)
    homeostasis_effect = compute_homeostasis_effect(current_affect, baseline_affect, affect_config)

    # Combine all effects
    total_effect = peer_effect + appraisal_effect + homeostasis_effect

    # Apply the change
    new_affect = current_affect + total_effect

    # Clamp to valid range
    return clamp(new_affect, -1.0, 1.0)


def update_resilience_dynamics(
    current_resilience: float,
    coped_successfully: bool = False,
    received_social_support: bool = False,
    consecutive_hindrances: int = 0,
    resilience_config: Optional[ResilienceDynamicsConfig] = None
) -> float:
    """
    Update agent's resilience based on coping, social support, and overload effects.

    Args:
        current_resilience: Agent's current resilience
        coped_successfully: Whether agent successfully coped with recent stress
        received_social_support: Whether agent received social support
        consecutive_hindrances: Number of consecutive hindrance events
        resilience_config: Resilience dynamics configuration

    Returns:
        New resilience value
    """
    if resilience_config is None:
        resilience_config = ResilienceDynamicsConfig()

    # Compute individual effect components
    coping_effect = 0.0
    if coped_successfully:
        coping_effect = resilience_config.coping_success_rate

    social_support_effect = 0.0
    if received_social_support:
        social_support_effect = resilience_config.social_support_rate

    overload_effect = compute_cumulative_overload(consecutive_hindrances, resilience_config)

    # Combine all effects
    total_effect = coping_effect + social_support_effect + overload_effect

    # Apply the change
    new_resilience = current_resilience + total_effect

    # Clamp to valid range
    return clamp(new_resilience, 0.0, 1.0)