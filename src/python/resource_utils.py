"""
Resource management utilities for agent-based mental health simulation.

This module contains stateless functions for:
- Resource regeneration and allocation
- Protective factor management
- Social resource exchange
- Resilience-based resource optimization
- All functions are pure and support dependency injection for testability
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

from src.python.config import get_config
from src.python.math_utils import clamp

# Load configuration
config = get_config()


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


@dataclass
class ResourceOptimizationConfig:
    """Configuration parameters for resilience-based resource optimization."""
    base_resource_cost: float = field(default_factory=lambda: get_config().get('agent', 'resource_cost'))
    resilience_efficiency_factor: float = 0.15  # 15% efficiency gain from resilience
    minimum_resource_threshold: float = 0.05  # Minimum resources needed for allocation
    coping_difficulty_scale: float = 0.5  # Scale for event difficulty effects
    stressed_resource_floor: float = 0.1  # Minimum resources maintained for stressed agents
    preservation_threshold: float = 0.1  # Resources to preserve for basic needs before allocation
    efficiency_return_factor: float = 0.05  # Efficiency return on protective factor investments


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

    # Linear regeneration to ensure resources tend toward 1.0 over time
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


def compute_resilience_optimized_resource_cost(
    base_cost: float,
    current_resilience: float,
    challenge: float,
    hindrance: float,
    config: Optional[ResourceOptimizationConfig] = None
) -> float:
    """
    Compute resource cost for coping that adapts based on resilience level.

    Higher resilience provides efficiency gains, reducing the effective cost.
    Challenge events are less costly than hindrance events for high-resilience agents.

    Args:
        base_cost: Base resource cost for coping attempt
        current_resilience: Agent's current resilience level (0-1)
        challenge: Challenge component of the stressor (0-1)
        hindrance: Hindrance component of the stressor (0-1)
        config: Resource optimization configuration

    Returns:
        Optimized resource cost considering resilience efficiency
    """
    if config is None:
        config = ResourceOptimizationConfig()

    # Base cost influenced by event difficulty (hindrance is more costly)
    event_difficulty = (challenge * 0.7 + hindrance * 1.3)  # Hindrance is 30% more difficult
    difficulty_multiplier = 1.0 + (event_difficulty * config.coping_difficulty_scale)

    # Resilience provides efficiency gains
    # Higher resilience = lower effective cost (more efficient resource use)
    resilience_efficiency = 1.0 - (current_resilience * config.resilience_efficiency_factor)

    # Challenge events benefit more from resilience (resilience helps with motivation)
    # Hindrance events benefit less from resilience (hindrance is more about obstacles)
    challenge_resilience_bonus = challenge * current_resilience * 0.2
    hindrance_resilience_bonus = hindrance * current_resilience * 0.1

    resilience_bonus = challenge_resilience_bonus + hindrance_resilience_bonus

    # Calculate final cost
    optimized_cost = base_cost * difficulty_multiplier * max(0.3, resilience_efficiency - resilience_bonus)

    return optimized_cost


def compute_resource_efficiency_gain(
    current_resilience: float,
    baseline_resilience: float,
    config: Optional[ResourceOptimizationConfig] = None
) -> float:
    """
    Compute efficiency gain from resilience for resource utilization.

    Agents with higher resilience relative to baseline use resources more efficiently.
    This represents learned coping strategies and psychological resource management.

    Args:
        current_resilience: Agent's current resilience level (0-1)
        baseline_resilience: Agent's baseline resilience level (0-1)
        config: Resource optimization configuration

    Returns:
        Efficiency multiplier (0.5-1.5 range, where >1 means more efficient)
    """
    if config is None:
        config = ResourceOptimizationConfig()

    # Resilience above baseline provides efficiency gains
    resilience_surplus = current_resilience - baseline_resilience

    if resilience_surplus <= 0:
        # No efficiency gain when resilience is at or below baseline
        return 1.0

    # Efficiency gain scales with resilience surplus
    # Maximum 50% efficiency improvement at very high resilience surplus
    max_efficiency_gain = 0.5
    efficiency_gain = min(resilience_surplus * config.resilience_efficiency_factor, max_efficiency_gain)

    # Return efficiency multiplier (1.0 + gain)
    return 1.0 + efficiency_gain


def allocate_resilience_optimized_resources(
    available_resources: float,
    current_resilience: float,
    baseline_resilience: float,
    protective_factors: Optional[ProtectiveFactors] = None,
    rng: Optional[np.random.Generator] = None,
    config: Optional[ResourceOptimizationConfig] = None
) -> Dict[str, float]:
    """
    Allocate resources with resilience-based optimization and preservation thresholds.

    Higher resilience provides:
    1. More efficient resource utilization (lower effective costs)
    2. Better allocation decisions (improved softmax weighting)
    3. Enhanced protective factor development
    4. Resource preservation for basic needs
    5. Efficiency returns on protective factor investments

    Args:
        available_resources: Total resources available for allocation
        current_resilience: Agent's current resilience level (0-1)
        baseline_resilience: Agent's baseline resilience level (0-1)
        protective_factors: Current efficacy levels of protective factors
        rng: Random number generator for stochastic decisions
        config: Resource optimization configuration

    Returns:
        Dictionary mapping factor names to allocated resources
    """
    if config is None:
        config = ResourceOptimizationConfig()

    if protective_factors is None:
        protective_factors = ProtectiveFactors()

    if rng is None:
        rng = np.random.default_rng()

    # Check if agent has minimum resources for allocation
    if available_resources < config.minimum_resource_threshold:
        return {
            'social_support': 0.0,
            'family_support': 0.0,
            'formal_intervention': 0.0,
            'psychological_capital': 0.0
        }

    # Preserve resources for basic needs before allocation
    preservable_resources = max(0.0, available_resources - config.preservation_threshold)
    if preservable_resources <= 0:
        return {
            'social_support': 0.0,
            'family_support': 0.0,
            'formal_intervention': 0.0,
            'psychological_capital': 0.0
        }

    # Compute resilience-based efficiency gain
    efficiency_gain = compute_resource_efficiency_gain(current_resilience, baseline_resilience, config)

    # Apply efficiency gain to preservable resources (more resilience = more effective resources)
    effective_resources = preservable_resources * efficiency_gain

    # Create allocation weights based on efficacy and resilience
    factors = ['social_support', 'family_support', 'formal_intervention', 'psychological_capital']
    efficacies = [
        protective_factors.social_support,
        protective_factors.family_support,
        protective_factors.formal_intervention,
        protective_factors.psychological_capital
    ]

    # Resilience improves allocation decisions by reducing temperature (more focused allocation)
    base_temperature = config.get('utility', 'softmax_temperature') if hasattr(config, 'get') else 1.0
    resilience_focus = current_resilience * 0.5  # Higher resilience = more focused allocation
    temperature = max(0.1, base_temperature - resilience_focus)

    # Add resilience bonus to allocation weights
    resilience_bonuses = [current_resilience * 0.5] * len(factors)  # 50% resilience bonus to all factors
    adjusted_efficacies = [efficacy + bonus for efficacy, bonus in zip(efficacies, resilience_bonuses)]

    # Softmax decision making with resilience-adjusted temperature
    logits = np.array(adjusted_efficacies) / temperature
    softmax_weights = np.exp(logits) / np.sum(np.exp(logits))

    # Allocate resources proportionally
    allocations = {
        factor: effective_resources * weight
        for factor, weight in zip(factors, softmax_weights)
    }

    return allocations


def compute_resource_depletion_with_resilience(
    current_resources: float,
    cost: float,
    current_resilience: float,
    coping_successful: bool,
    is_stressed: bool,
    config: Optional[ResourceOptimizationConfig] = None
) -> float:
    """
    Compute resource depletion considering resilience-based optimization.

    Args:
        current_resources: Agent's current resources before depletion
        cost: Base cost of coping attempt
        current_resilience: Agent's current resilience level
        coping_successful: Whether the coping attempt was successful
        is_stressed: Whether the agent is currently stressed
        config: Resource optimization configuration

    Returns:
        Remaining resources after depletion
    """
    if config is None:
        config = ResourceOptimizationConfig()

    # Apply resilience-based cost optimization
    optimized_cost = cost * (1.0 - current_resilience * config.resilience_efficiency_factor)

    # Failed coping attempts cost more (inefficient resource use)
    if not coping_successful:
        optimized_cost *= 1.1  # 10% penalty for failed coping

    # Ensure minimum cost even with very high resilience
    optimized_cost = max(cost * 0.1, optimized_cost)

    # Deplete resources
    remaining_resources = max(0.0, current_resources - optimized_cost)

    # For stressed agents, maintain minimum resource floor to preserve correlation patterns
    if is_stressed:
        remaining_resources = max(remaining_resources, config.stressed_resource_floor)

    return remaining_resources


def process_social_resource_exchange(
    self_resources: float,
    partner_resources: float,
    self_resilience: float,
    partner_resilience: float,
    social_support_boost: float = 1.0,
    config: Optional[Dict[str, float]] = None
) -> Tuple[float, float, float, float]:
    """
    Process social resource exchange between agents with resilience optimization.

    Args:
        self_resources: Self agent's current resources
        partner_resources: Partner agent's current resources
        self_resilience: Self agent's resilience level
        partner_resilience: Partner agent's resilience level
        social_support_boost: Boost factor from social support efficacy
        config: Configuration for resource exchange

    Returns:
        Tuple of (self_resource_transfer, partner_resource_transfer,
                  new_self_resources, new_partner_resources)
    """
    # Get configuration for resource exchange
    cfg = get_config()

    if config is None:
        config = {
            'base_exchange_rate': cfg.get('resource', 'social_exchange_rate'),
            'exchange_threshold': cfg.get('resource', 'exchange_threshold'),
            'max_exchange_ratio': cfg.get('resource', 'max_exchange_ratio'),
            'minimum_resource_threshold_for_sharing': 0.2,  # Minimum resources needed before sharing
            'exchange_amount_reduction_factor': 0.5  # Reduce exchange amounts to minimize correlation impact
        }

    # Calculate resource difference (positive if partner has more resources)
    resource_diff = partner_resources - self_resources

    # Only exchange if there's a meaningful resource difference
    if abs(resource_diff) < config['exchange_threshold']:
        return 0.0, 0.0, self_resources, partner_resources

    # Determine exchange direction and resilience factors
    if resource_diff > 0:
        # Partner has more resources - they can provide support
        giver, receiver = partner_resources, self_resources
        giver_resilience, receiver_resilience = partner_resilience, self_resilience
    else:
        # Self has more resources - we can provide support
        giver, receiver = self_resources, partner_resources
        giver_resilience, receiver_resilience = self_resilience, partner_resilience

    # Check minimum resource threshold for sharing
    if giver < config['minimum_resource_threshold_for_sharing']:
        return 0.0, 0.0, self_resources, partner_resources

    # Resilience-optimized exchange calculation
    # Higher giver resilience = more generous sharing
    # Higher receiver resilience = more efficient resource utilization
    giver_resilience_bonus = giver_resilience * 0.2
    receiver_efficiency_bonus = receiver_resilience * 0.15

    # Calculate exchange amount with resilience optimization
    max_transferable = giver * config['max_exchange_ratio']
    willingness_factor = _calculate_resilience_optimized_willingness(giver_resilience)

    # Base exchange amount with resilience enhancement
    base_amount = min(config['base_exchange_rate'] * abs(resource_diff), max_transferable) * willingness_factor

    # Apply receiver efficiency bonus (more resilient receivers use resources better)
    optimized_amount = base_amount * (1.0 + receiver_efficiency_bonus)

    # Apply social support boost
    final_exchange_amount = optimized_amount * social_support_boost

    # Reduce exchange amount to minimize correlation impact
    final_exchange_amount *= config['exchange_amount_reduction_factor']

    # Apply exchange only if giver has sufficient resources
    if final_exchange_amount > 0 and giver >= final_exchange_amount:
        # Transfer resources
        if resource_diff > 0:
            # Partner gave resources to self
            new_self_resources = self_resources + final_exchange_amount
            new_partner_resources = partner_resources  # No loss for giver (giver benefits)
            return 0.0, final_exchange_amount, new_self_resources, new_partner_resources
        else:
            # Self gave resources to partner
            new_self_resources = self_resources  # No loss for giver (giver benefits)
            new_partner_resources = partner_resources + final_exchange_amount
            return final_exchange_amount, 0.0, new_self_resources, new_partner_resources

    return 0.0, 0.0, self_resources, partner_resources


def _calculate_resilience_optimized_willingness(giver_resilience: float) -> float:
    """
    Calculate resource sharing willingness with resilience optimization.

    Args:
        giver_resilience: Resilience level of the giver

    Returns:
        Float indicating willingness to share (0.0 to 1.0)
    """
    # Enhanced willingness calculation considering resilience
    base_resilience_factor = giver_resilience

    # Combine factors with enhanced weights
    willingness = base_resilience_factor * 0.6

    return min(1.0, willingness)


def update_protective_factors_with_allocation(
    protective_factors: Dict[str, float],
    allocations: Dict[str, float],
    current_resilience: float,
    config: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Update protective factor levels based on resource allocations with efficiency returns.

    Args:
        protective_factors: Current protective factor efficacy levels
        allocations: Resources allocated to each factor
        current_resilience: Agent's current resilience level
        config: Configuration for protective factor updates

    Returns:
        Updated protective factor levels
    """
    cfg = get_config()

    if config is None:
        config = {
            'improvement_rate': cfg.get('resource', 'protective_improvement_rate'),
            'efficiency_return_factor': 0.05  # Efficiency return on investments
        }

    updated_factors = protective_factors.copy()

    # Update each protective factor based on allocation
    for factor, allocation in allocations.items():
        if allocation > 0 and factor in updated_factors:
            current_efficacy = updated_factors[factor]

            # Resilience provides additional improvement rate bonus
            resilience_bonus = current_resilience * 0.2  # 20% bonus from resilience

            # Investment return is higher when current efficacy is lower and resilience is higher
            improvement_rate = config['improvement_rate']
            investment_effectiveness = 1.0 - current_efficacy  # Higher return when efficacy is low

            # Apply resilience-based efficiency gain
            efficiency_gain = 1.0 + resilience_bonus

            # Add efficiency returns: investments yield additional benefits over time
            efficiency_return = allocation * config.get('efficiency_return_factor', 0.05)

            efficacy_increase = (allocation * improvement_rate * investment_effectiveness * efficiency_gain) + efficiency_return
            updated_factors[factor] = min(1.0, current_efficacy + efficacy_increase)

    return updated_factors


def get_resilience_boost_from_protective_factors(
    protective_factors: Dict[str, float],
    baseline_resilience: float,
    current_resilience: float,
    config: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate resilience boost from active protective factors.

    Args:
        protective_factors: Current protective factor efficacy levels
        baseline_resilience: Agent's baseline resilience level
        current_resilience: Agent's current resilience level
        config: Configuration for resilience boost calculation

    Returns:
        Float indicating resilience boost from protective factors
    """
    cfg = get_config()

    if config is None:
        config = {
            'boost_rate': cfg.get('resilience_dynamics', 'boost_rate')
        }

    # Only apply boost when resilience is low
    current_need = baseline_resilience - current_resilience

    if current_need < 0:
        return 0.0

    total_boost = 0.0

    # Each protective factor provides boost based on efficacy and current resilience need
    for factor, efficacy in protective_factors.items():
        if efficacy > 0:
            # Boost is higher when resilience is low (more needed)
            total_boost += efficacy * current_need * config['boost_rate']

    return total_boost


def allocate_protective_factors(
    available_resources: float,
    current_resilience: float,
    baseline_resilience: float,
    protective_factors: Dict[str, float],
    rng: Optional[np.random.Generator] = None,
    config: Optional[ResourceOptimizationConfig] = None
) -> Dict[str, float]:
    """
    Allocate available resources across protective factors using resilience-optimized dynamics.

    Args:
        available_resources: Total resources available for allocation
        current_resilience: Agent's current resilience level (0-1)
        baseline_resilience: Agent's baseline resilience level (0-1)
        protective_factors: Current efficacy levels of protective factors
        rng: Random number generator for stochastic decisions
        config: Resource optimization configuration

    Returns:
        Dictionary mapping factor names to allocated resources
    """
    if config is None:
        config = ResourceOptimizationConfig()

    if rng is None:
        rng = np.random.default_rng()

    # Use resilience-optimized resource allocation
    allocations = allocate_resilience_optimized_resources(
        available_resources=available_resources,
        current_resilience=current_resilience,
        baseline_resilience=baseline_resilience,
        protective_factors=ProtectiveFactors(**protective_factors),
        rng=rng,
        config=config
    )

    return allocations


def update_protective_factors_efficacy(
    protective_factors: Dict[str, float],
    allocations: Dict[str, float],
    current_resilience: float,
    stress_state: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Update protective factor efficacy levels based on resource allocations.

    Args:
        protective_factors: Current protective factor efficacy levels
        allocations: Resources allocated to each factor
        current_resilience: Agent's current resilience level
        stress_state: Optional stress state for stress-based optimization
        config: Configuration for protective factor updates

    Returns:
        Updated protective factor levels
    """
    cfg = get_config()

    if config is None:
        config = {
            'improvement_rate': cfg.get('resource', 'protective_improvement_rate')
        }

    updated_factors = protective_factors.copy()

    # Apply stress-based optimization if stress state provided
    stress_efficiency = 1.0
    controllability_bonus = 0.0

    if stress_state:
        # High overload reduces allocation efficiency
        overload_penalty = stress_state.get('stress_overload', 0.0) * 0.1
        stress_efficiency = 1.0 - overload_penalty

        # Low controllability increases allocation urgency
        controllability_bonus = (1.0 - stress_state.get('stress_controllability', 0.5)) * 0.05

    # Update each protective factor based on allocation
    for factor, allocation in allocations.items():
        if allocation > 0 and factor in updated_factors:
            current_efficacy = updated_factors[factor]

            # Stress state influences improvement effectiveness
            stress_effectiveness = 1.0 + (stress_state.get('current_stress', 0.0) * 0.1) if stress_state else 1.0

            improvement_rate = config['improvement_rate']
            investment_effectiveness = 1.0 - current_efficacy

            efficacy_increase = (allocation * improvement_rate * investment_effectiveness *
                               stress_effectiveness)
            updated_factors[factor] = min(1.0, current_efficacy + efficacy_increase)

    return updated_factors


def calculate_recent_social_benefit(daily_support_exchanges: int) -> float:
    """
    Calculate recent social support benefit for resilience optimization.

    Args:
        daily_support_exchanges: Number of daily support exchanges

    Returns:
        Float indicating recent social support level (0-1)
    """
    if daily_support_exchanges <= 0:
        return 0.0

    # Weight by number of support exchanges (more exchanges = more benefit)
    # Cap at reasonable level to avoid excessive boosting
    recent_benefit = min(1.0, daily_support_exchanges * 0.2)

    return recent_benefit


def allocate_protective_factors_with_social_boost(
    available_resources: float,
    current_resilience: float,
    baseline_resilience: float,
    protective_factors: Dict[str, float],
    social_benefit: float,
    rng: Optional[np.random.Generator] = None,
    config: Optional[ResourceOptimizationConfig] = None
) -> Dict[str, float]:
    """
    Allocate protective factors with social support enhancement and preservation thresholds.

    Args:
        available_resources: Total resources available for allocation
        current_resilience: Agent's current resilience level (0-1)
        baseline_resilience: Agent's baseline resilience level (0-1)
        protective_factors: Current efficacy levels of protective factors
        social_benefit: Level of recent social support received (0-1)
        rng: Random number generator for stochastic decisions
        config: Resource optimization configuration

    Returns:
        Dictionary mapping factor names to allocated resources
    """
    if config is None:
        config = ResourceOptimizationConfig()

    if rng is None:
        rng = np.random.default_rng()

    # Preserve resources for basic needs before allocation
    preservable_resources = max(0.0, available_resources - config.preservation_threshold)
    if preservable_resources <= 0:
        return {
            'social_support': 0.0,
            'family_support': 0.0,
            'formal_intervention': 0.0,
            'psychological_capital': 0.0
        }

    # Social support increases available resources for allocation
    social_resource_boost = social_benefit * 0.1  # 10% boost per social benefit unit
    available_for_allocation = preservable_resources * 0.1 + social_resource_boost

    if available_for_allocation > 0 and preservable_resources > 0:
        # Use resilience-optimized allocation function with social enhancement
        allocations = allocate_resilience_optimized_resources(
            available_resources=available_for_allocation,
            current_resilience=current_resilience,
            baseline_resilience=baseline_resilience,
            protective_factors=ProtectiveFactors(**protective_factors),
            rng=rng,
            config=config
        )

        # Apply social support boost to social_support allocation specifically
        if 'social_support' in allocations and allocations['social_support'] > 0:
            allocations['social_support'] *= (1.0 + social_benefit * 0.3)

        # Normalize allocations
        total_allocated = sum(allocations.values())
        if total_allocated > 0:
            allocations = {k: v / total_allocated for k, v in allocations.items()}

        return allocations

    return {
        'social_support': 0.0,
        'family_support': 0.0,
        'formal_intervention': 0.0,
        'psychological_capital': 0.0
    }
