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
    Process a complete social interaction between two agents with positive/negative effects.

    Positive neighbor affect provides greater benefit, negative affect causes greater harm.
    This reflects realistic social dynamics where negative interactions are more impactful.

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

    # Compute mutual influences with asymmetric positive/negative effects
    affect_influence_self, affect_influence_partner = compute_mutual_influence(
        self_affect, partner_affect, config
    )

    # Apply asymmetric weighting: negative affects have stronger impact
    if affect_influence_self < 0:
        affect_influence_self *= 1.5  # Negative influence is 50% stronger
    if affect_influence_partner < 0:
        affect_influence_partner *= 1.5  # Negative influence is 50% stronger

    # Apply affect changes
    new_self_affect = self_affect + affect_influence_self
    new_partner_affect = partner_affect + affect_influence_partner

    # Compute resilience changes based on partner's affect with threshold effects
    resilience_influence_self = compute_resilience_influence(partner_affect, config)
    resilience_influence_partner = compute_resilience_influence(self_affect, config)

    # Apply resilience influence only if partner affect exceeds threshold
    stress_config = StressProcessingConfig()
    if abs(partner_affect) > stress_config.affect_threshold:
        new_self_resilience = self_resilience + resilience_influence_self
        new_partner_resilience = partner_resilience + resilience_influence_partner
    else:
        new_self_resilience = self_resilience
        new_partner_resilience = partner_resilience

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
    challenge: float = 0.5,
    hindrance: float = 0.5,
    config: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute affect change due to stress event outcome using new mechanism.

    Args:
        current_affect: Agent's current affect
        is_stressed: Whether the agent experienced stress
        coped_successfully: Whether coping was successful
        challenge: Challenge component from event appraisal (0-1)
        hindrance: Hindrance component from event appraisal (0-1)
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

    # Check if using default challenge/hindrance values (backward compatibility)
    if challenge == 0.5 and hindrance == 0.5:
        # Fall back to old mechanism for backward compatibility
        if coped_successfully:
            return config['coping_improvement']
        else:
            return -config['coping_deterioration']
    else:
        # Use new mechanism with challenge/hindrance effects
        stress_config = StressProcessingConfig()

        if coped_successfully:
            # Success: challenge provides positive affect boost
            affect_change = stress_config.challenge_bonus * challenge
        else:
            # Failure: hindrance provides negative affect impact
            affect_change = -stress_config.hindrance_penalty * hindrance

        return affect_change


def compute_stress_impact_on_resilience(
    current_resilience: float,
    is_stressed: bool,
    coped_successfully: bool,
    challenge: float = 0.5,
    hindrance: float = 0.5,
    config: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute resilience change due to stress event outcome using new mechanism.

    Args:
        current_resilience: Agent's current resilience
        is_stressed: Whether the agent experienced stress
        coped_successfully: Whether coping was successful
        challenge: Challenge component from event appraisal (0-1)
        hindrance: Hindrance component from event appraisal (0-1)
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

    # Check if using default challenge/hindrance values (backward compatibility)
    if challenge == 0.5 and hindrance == 0.5:
        # Fall back to old mechanism for backward compatibility
        if coped_successfully:
            return config['coping_improvement']
        else:
            return -config['coping_deterioration']
    else:
        # Use new mechanism with challenge/hindrance effects
        resilience_effect = compute_challenge_hindrance_resilience_effect(
            challenge, hindrance, coped_successfully
        )

        return resilience_effect


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
# NEW STRESS PROCESSING MECHANISMS
# ==============================================

@dataclass
class StressProcessingConfig:
    """Configuration parameters for new stress processing mechanisms."""
    stress_threshold: float = field(default_factory=lambda: get_config().get('threshold', 'stress_threshold'))
    affect_threshold: float = field(default_factory=lambda: get_config().get('threshold', 'affect_threshold'))
    base_coping_probability: float = field(default_factory=lambda: get_config().get('coping', 'base_probability'))
    social_influence_factor: float = field(default_factory=lambda: get_config().get('coping', 'social_influence'))
    challenge_bonus: float = field(default_factory=lambda: get_config().get('coping', 'challenge_bonus'))
    hindrance_penalty: float = field(default_factory=lambda: get_config().get('coping', 'hindrance_penalty'))
    daily_decay_rate: float = field(default_factory=lambda: get_config().get('affect_dynamics', 'homeostatic_rate'))
    stress_decay_rate: float = field(default_factory=lambda: get_config().get('resilience_dynamics', 'homeostatic_rate'))


def compute_coping_probability(
    challenge: float,
    hindrance: float,
    neighbor_affects: List[float],
    config: Optional[StressProcessingConfig] = None
) -> float:
    """
    Compute coping success probability based on challenge/hindrance and social influence.

    Challenge increases coping probability, hindrance decreases it.
    Positive neighbor affects increase probability, negative decrease it.

    Args:
        challenge: Challenge component from event appraisal (0-1)
        hindrance: Hindrance component from event appraisal (0-1)
        neighbor_affects: List of neighbor affect values
        config: Stress processing configuration

    Returns:
        Coping success probability (0-1)
    """
    if config is None:
        config = StressProcessingConfig()

    # Base probability from configuration
    base_prob = config.base_coping_probability

    # Challenge/hindrance effects
    challenge_effect = config.challenge_bonus * challenge
    hindrance_effect = -config.hindrance_penalty * hindrance

    # Social influence from neighbors
    social_effect = 0.0
    if neighbor_affects:
        avg_neighbor_affect = np.mean(neighbor_affects)
        social_effect = config.social_influence_factor * avg_neighbor_affect

    # Combine all effects
    total_effect = challenge_effect + hindrance_effect + social_effect

    # Apply effects to base probability
    coping_prob = base_prob + total_effect

    # Clamp to valid range
    return clamp(coping_prob, 0.0, 1.0)


def compute_challenge_hindrance_resilience_effect(
    challenge: float,
    hindrance: float,
    coped_successfully: bool,
    config: Optional[StressProcessingConfig] = None
) -> float:
    """
    Compute resilience change based on challenge/hindrance and coping outcome.

    When coping fails:
    - Hindrance greatly reduces resilience (-0.3 to -0.5)
    - Challenge slightly reduces resilience (-0.05 to -0.1)

    When coping succeeds:
    - Hindrance slightly increases resilience (+0.05 to +0.1)
    - Challenge greatly increases resilience (+0.2 to +0.4)

    Args:
        challenge: Challenge component from event appraisal (0-1)
        hindrance: Hindrance component from event appraisal (0-1)
        coped_successfully: Whether coping was successful
        config: Stress processing configuration

    Returns:
        Resilience change
    """
    if config is None:
        config = StressProcessingConfig()

    if coped_successfully:
        # Success case: hindrance slightly helps, challenge greatly helps
        hindrance_effect = 0.1 * hindrance  # Small positive effect
        challenge_effect = 0.3 * challenge  # Large positive effect
    else:
        # Failure case: hindrance greatly hurts, challenge slightly hurts
        hindrance_effect = -0.4 * hindrance  # Large negative effect
        challenge_effect = -0.1 * challenge  # Small negative effect

    total_effect = hindrance_effect + challenge_effect
    return total_effect


def compute_daily_affect_reset(
    current_affect: float,
    baseline_affect: float,
    config: Optional[StressProcessingConfig] = None
) -> float:
    """
    Reset affect toward baseline at the end of each day.

    Args:
        current_affect: Agent's current affect
        baseline_affect: Agent's baseline affect level
        config: Stress processing configuration

    Returns:
        Reset affect value
    """
    if config is None:
        config = StressProcessingConfig()

    # Calculate distance from baseline
    distance = baseline_affect - current_affect

    # Apply decay toward baseline
    reset_amount = config.daily_decay_rate * distance

    # Apply reset
    new_affect = current_affect + reset_amount

    # Clamp to valid range
    return clamp(new_affect, -1.0, 1.0)


def compute_stress_decay(
    current_stress: float,
    config: Optional[StressProcessingConfig] = None
) -> float:
    """
    Apply natural decay to stress levels over time.

    Args:
        current_stress: Agent's current stress level
        config: Stress processing configuration

    Returns:
        Decayed stress value
    """
    if config is None:
        config = StressProcessingConfig()

    # Exponential decay toward zero
    decayed_stress = current_stress * (1.0 - config.stress_decay_rate)

    # Clamp to valid range
    return clamp(decayed_stress, 0.0, 1.0)


@dataclass
class ResourceOptimizationConfig:
    """Configuration parameters for resilience-based resource optimization."""
    base_resource_cost: float = field(default_factory=lambda: get_config().get('agent', 'resource_cost'))
    resilience_efficiency_factor: float = 0.3  # 30% efficiency gain from resilience
    minimum_resource_threshold: float = 0.05  # Minimum resources needed for allocation
    coping_difficulty_scale: float = 0.5  # Scale for event difficulty effects


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


def get_neighbor_affects(agent, model) -> List[float]:
    """
    Get affect values of neighboring agents for social influence calculations.

    Args:
        agent: Agent instance
        model: Mesa model instance

    Returns:
        List of neighbor affect values
    """
    # Check if agent has a valid position
    if agent.pos is None:
        return []

    try:
        neighbors = list(
            model.grid.get_neighbors(
                agent.pos, include_center=False
            )
        )
        return [neighbor.affect for neighbor in neighbors if hasattr(neighbor, 'affect')]
    except Exception:
        # Return empty list if there are any issues with neighbor lookup
        return []


def integrate_social_resilience_optimization(
    current_resilience: float,
    daily_interactions: int,
    daily_support_exchanges: int,
    resources: float,
    baseline_resilience: float,
    protective_factors: Dict[str, float],
    rng: Optional[np.random.Generator] = None,
    config: Optional[ResourceOptimizationConfig] = None
) -> float:
    """
    Integrate social resource exchange with resilience optimization mechanisms.

    Args:
        current_resilience: Agent's current resilience level
        daily_interactions: Number of daily interactions
        daily_support_exchanges: Number of daily support exchanges
        resources: Agent's current resources
        baseline_resilience: Agent's baseline resilience level
        protective_factors: Current protective factor efficacy levels
        rng: Random number generator for stochastic decisions
        config: Resource optimization configuration

    Returns:
        Updated resilience level after social optimization
    """
    if config is None:
        config = ResourceOptimizationConfig()

    if rng is None:
        rng = np.random.default_rng()

    # Check if agent received social support recently (within last few interactions)
    recent_social_benefit = calculate_recent_social_benefit(daily_support_exchanges)

    if recent_social_benefit > 0:
        # Social support enhances resilience optimization
        # Boost resilience temporarily for better resource allocation
        social_resilience_boost = recent_social_benefit * 0.1

        # Store original resilience for this calculation
        original_resilience = current_resilience

        # Temporarily boost resilience for optimization calculations
        boosted_resilience = min(1.0, current_resilience + social_resilience_boost)

        # Re-allocate protective factors with social support boost if resources available
        if resources > 0.1:  # Only if agent has resources to allocate
            # This would call the resource allocation function with boosted resilience
            # For now, just return the boosted resilience
            pass

        # Return boosted resilience
        return boosted_resilience

    return current_resilience


def calculate_recent_social_benefit(daily_support_exchanges: int) -> float:
    """
    Calculate recent social support benefit for resilience optimization.

    Args:
        daily_support_exchanges: Number of daily support exchanges

    Returns:
        Float indicating recent social support level (0-1)
    """
    # Use daily support exchanges as a proxy for recent social benefit
    # More recent exchanges have higher weight
    recent_benefit = 0.0

    if daily_support_exchanges > 0:
        # Weight by number of support exchanges (more exchanges = more benefit)
        # Cap at reasonable level to avoid excessive boosting
        recent_benefit = min(1.0, daily_support_exchanges * 0.2)

    return recent_benefit


def allocate_resilience_optimized_resources(
    available_resources: float,
    current_resilience: float,
    baseline_resilience: float,
    protective_factors: Optional[ProtectiveFactors] = None,
    rng: Optional[np.random.Generator] = None,
    config: Optional[ResourceOptimizationConfig] = None
) -> Dict[str, float]:
    """
    Allocate resources with resilience-based optimization.

    Higher resilience provides:
    1. More efficient resource utilization (lower effective costs)
    2. Better allocation decisions (improved softmax weighting)
    3. Enhanced protective factor development

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

    # Compute resilience-based efficiency gain
    efficiency_gain = compute_resource_efficiency_gain(current_resilience, baseline_resilience, config)

    # Apply efficiency gain to available resources (more resilience = more effective resources)
    effective_resources = available_resources * efficiency_gain

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
    resilience_bonuses = [current_resilience * 0.2] * len(factors)  # 20% resilience bonus to all factors
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
    config: Optional[ResourceOptimizationConfig] = None
) -> float:
    """
    Compute resource depletion considering resilience-based optimization.

    Args:
        current_resources: Agent's current resources before depletion
        cost: Base cost of coping attempt
        current_resilience: Agent's current resilience level
        coping_successful: Whether the coping attempt was successful
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
        optimized_cost *= 1.3  # 30% penalty for failed coping

    # Ensure minimum cost even with very high resilience
    optimized_cost = max(cost * 0.3, optimized_cost)

    # Deplete resources
    remaining_resources = max(0.0, current_resources - optimized_cost)

    return remaining_resources


def process_stress_event_with_new_mechanism(
    current_affect: float,
    current_resilience: float,
    current_stress: float,
    challenge: float,
    hindrance: float,
    neighbor_affects: List[float],
    config: Optional[StressProcessingConfig] = None
) -> tuple[float, float, float, bool]:
    """
    Process stress event using new mechanism with challenge/hindrance effects.

    Args:
        current_affect: Agent's current affect
        current_resilience: Agent's current resilience
        current_stress: Agent's current stress level
        challenge: Challenge component from event appraisal (0-1)
        hindrance: Hindrance component from event appraisal (0-1)
        neighbor_affects: List of neighbor affect values
        config: Stress processing configuration

    Returns:
        Tuple of (new_affect, new_resilience, new_stress, coped_successfully)
    """
    if config is None:
        config = StressProcessingConfig()

    # Compute coping probability based on challenge/hindrance and social influence
    coping_prob = compute_coping_probability(challenge, hindrance, neighbor_affects, config)

    # Determine if coping was successful
    coped_successfully = np.random.random() < coping_prob

    # Compute resilience effect based on challenge/hindrance and coping outcome
    resilience_effect = compute_challenge_hindrance_resilience_effect(
        challenge, hindrance, coped_successfully, config
    )

    # Update resilience
    new_resilience = current_resilience + resilience_effect
    new_resilience = clamp(new_resilience, 0.0, 1.0)

    # Update stress based on coping outcome
    if coped_successfully:
        # Successful coping reduces stress
        stress_reduction = 0.2 * (1.0 + challenge)  # Challenge helps reduce stress more
        new_stress = current_stress - stress_reduction
    else:
        # Failed coping increases stress
        stress_increase = 0.3 * (1.0 + hindrance)  # Hindrance increases stress more
        new_stress = current_stress + stress_increase

    new_stress = clamp(new_stress, 0.0, 1.0)

    # Update affect based on stress outcome
    if coped_successfully:
        affect_change = 0.1 * challenge  # Challenge provides positive affect boost
    else:
        affect_change = -0.2 * hindrance  # Hindrance provides negative affect impact

    new_affect = current_affect + affect_change
    new_affect = clamp(new_affect, -1.0, 1.0)

    return new_affect, new_resilience, new_stress, coped_successfully


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

    Based on theoretical model where challenge tends to improve affect (motivating)
    and hindrance tends to worsen it (demotivating).

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
    # Effect is proportional to the challenge/hindrance intensity and current affect state
    challenge_effect = config.event_appraisal_rate * challenge * (1.0 - current_affect)
    hindrance_effect = -config.event_appraisal_rate * hindrance * max(0.1, current_affect + 1.0)

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


def scale_homeostatic_rate(
    base_rate: float,
    resources: float,
    stress: float
) -> float:
    """
    Scale homeostatic rate based on resource availability and stress level.
    Higher resources lead to weaker homeostasis, which allows over time
    adaptation. Lower resource lead to stronger homeostasis, which maintains
    stability. In contrast, higher stress leads to stronger homeostasis, and
    lowe stress lead to weaker homeostasis.
    """

    resource_factor = 1.0 - (resources * 0.7) # Range: [0.3, 0.7]
    stress_factor   = 1.0 + (stress * 0.5)    # Range: [1.0, 1.5]
    scaled_rate     = base_rate * resource_factor * stress_factor

    return min(base_rate * 2, scaled_rate) # Cap at 2x
