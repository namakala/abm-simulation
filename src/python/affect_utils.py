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


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Clamp a value to specified bounds.

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


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