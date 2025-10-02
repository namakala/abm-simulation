"""
Agent-based model for mental health simulation with modular, testable utilities.

This module implements an agent that experiences stress events and social interactions,
using utility functions for all domain-specific behaviors to ensure modularity and testability.
"""

import random
import numpy as np
import mesa

# Import utility modules
from src.python.stress_utils import (
    generate_stress_event, process_stress_event,
    StressEvent, AppraisalWeights, ThresholdParams
)

from src.python.affect_utils import (
    process_interaction, compute_stress_impact_on_affect,
    compute_stress_impact_on_resilience, clamp, InteractionConfig,
    update_affect_dynamics, update_resilience_dynamics,
    AffectDynamicsConfig, ResilienceDynamicsConfig
)

from src.python.math_utils import sample_poisson, create_rng
from src.python.config import get_config

# Load configuration
config = get_config()


class Person(mesa.Agent):
    """
    A person who experiences social interactions and stressful events.

    This agent implementation uses utility functions for all domain-specific behaviors,
    keeping the class focused on simulation orchestration and state management.

    State variables:
    - resilience: Current resilience level ∈ [0,1]
    - affect: Current affect level ∈ [-1,1]
    - resources: Available psychological/physical resources ∈ [0,1]
    - protective_factors: Current levels of protective mechanisms
    """

    def __init__(self, model, config=None):
        """
        Initialize agent with configuration parameters.

        Args:
            model: Mesa model instance
            config: Optional configuration dictionary for agent parameters
        """
        super().__init__(model)

        # Set default configuration
        if config is None:
            # Use the global config object directly
            cfg = get_config()
            config = {
                'initial_resilience': cfg.get('agent', 'initial_resilience'),
                'initial_affect': cfg.get('agent', 'initial_affect'),
                'initial_resources': cfg.get('agent', 'initial_resources'),
                'stress_probability': cfg.get('agent', 'stress_probability'),
                'coping_success_rate': cfg.get('agent', 'coping_success_rate'),
                'subevents_per_day': cfg.get('agent', 'subevents_per_day')
            }

        # Initialize state variables
        self.resilience = config['initial_resilience']
        self.affect = config['initial_affect']
        self.resources = config['initial_resources']

        # Initialize baseline affect for homeostasis
        self.baseline_affect = config['initial_affect']

        # Initialize protective factors
        self.protective_factors = {
            'social_support': 0.5,
            'family_support': 0.5,
            'formal_intervention': 0.5,
            'psychological_capital': 0.5
        }

        # Initialize consecutive hindrances tracking for overload effects
        self.consecutive_hindrances = 0

        # Configuration for utility functions
        self.stress_config = {
            'stress_probability': config['stress_probability'],
            'coping_success_rate': config['coping_success_rate']
        }

        self.interaction_config = InteractionConfig()

        # Random number generator for reproducible testing
        # Note: Mesa Agent base class has 'rng' property, so we use '_rng'
        self._rng = create_rng(getattr(model, 'seed', None))

    def step(self):
        """
        Execute one day of simulation with integrated stress and affect dynamics.

        Coordinates stress events, social interactions, and baseline dynamics
        to create realistic mental health trajectories.
        """
        # Get configuration for dynamics
        affect_config = AffectDynamicsConfig()
        resilience_config = ResilienceDynamicsConfig()

        # Get neighbor affects for social influence throughout the day
        neighbor_affects = self._get_neighbor_affects()

        # Initialize daily tracking variables
        daily_challenge = 0.0
        daily_hindrance = 0.0
        stress_events_count = 0
        social_interactions_count = 0

        # Determine number of subevents using utility function
        n_subevents = sample_poisson(
            lam=config.get('agent', 'subevents_per_day'),
            rng=self._rng,
            min_value=1
        )

        # Generate random sequence of actions
        actions = [
            self._rng.choice(["interact", "stress"])
            for _ in range(n_subevents)
        ]

        # Shuffle for random order
        self._rng.shuffle(actions)

        # Execute actions and accumulate daily challenge/hindrance
        for action in actions:
            if action == "interact":
                self.interact()
                social_interactions_count += 1
            elif action == "stress":
                challenge, hindrance = self.stressful_event()
                daily_challenge += challenge
                daily_hindrance += hindrance
                stress_events_count += 1

        # Normalize daily challenge/hindrance by number of events
        if stress_events_count > 0:
            daily_challenge /= stress_events_count
            daily_hindrance /= stress_events_count

        # Apply integrated affect dynamics (homeostasis + peer influence + event appraisal)
        self.affect = update_affect_dynamics(
            current_affect=self.affect,
            baseline_affect=self.baseline_affect,
            neighbor_affects=neighbor_affects,
            challenge=daily_challenge,
            hindrance=daily_hindrance,
            affect_config=affect_config
        )

        # Apply integrated resilience dynamics (coping + social support + overload effects)
        # Check if agent received social support during interactions
        received_social_support = social_interactions_count > 0 and self._rng.random() < 0.3

        self.resilience = update_resilience_dynamics(
            current_resilience=self.resilience,
            coped_successfully=stress_events_count > 0,  # Simplified: coped if had stress events
            received_social_support=received_social_support,
            consecutive_hindrances=getattr(self, 'consecutive_hindrances', 0),
            resilience_config=resilience_config
        )

        # Add boost from protective factors
        protective_boost = self._get_resilience_boost_from_protective_factors()
        self.resilience = min(1.0, self.resilience + protective_boost)

        # Apply enhanced resource regeneration with affect influence
        from .affect_utils import compute_resource_regeneration, ResourceParams
        regen_params = ResourceParams(
            base_regeneration=config.get('resource', 'base_regeneration')
        )

        # Affect influences resource regeneration (positive affect helps recovery)
        affect_multiplier = 1.0 + 0.2 * max(0.0, self.affect)  # Positive affect boosts regeneration
        base_regeneration = compute_resource_regeneration(self.resources, regen_params)
        self.resources += base_regeneration * affect_multiplier

        # Decay consecutive hindrances over time if no new hindrance events
        if hasattr(self, 'consecutive_hindrances') and self.consecutive_hindrances > 0:
            # Slowly decay consecutive hindrances when no new hindrance events occur
            decay_rate = 0.1
            self.consecutive_hindrances = max(0, self.consecutive_hindrances - decay_rate)

        # Clamp all values to valid ranges
        self.resilience = clamp(self.resilience, 0.0, 1.0)
        self.affect = clamp(self.affect, -1.0, 1.0)
        self.resources = clamp(self.resources, 0.0, 1.0)

    def interact(self):
        """
        Interact with a random neighbor using utility functions.

        Delegates all domain logic to utility functions for modularity and testability.
        """
        # Get neighbors using Mesa's grid
        neighbors = list(
            self.model.grid.get_neighbors(
                self.pos, include_center=False
            )
        )

        if not neighbors:
            return

        # Select random interaction partner
        partner = self._rng.choice(neighbors)

        # Use utility function for interaction processing
        new_self_affect, new_partner_affect, new_self_resilience, new_partner_resilience = (
            process_interaction(
                self_affect=self.affect,
                partner_affect=partner.affect,
                self_resilience=self.resilience,
                partner_resilience=partner.resilience,
                config=self.interaction_config
            )
        )

        # Update state with results from utility function
        self.affect = clamp(new_self_affect, -1.0, 1.0)
        partner.affect = clamp(new_partner_affect, -1.0, 1.0)
        self.resilience = clamp(new_self_resilience, 0.0, 1.0)
        partner.resilience = clamp(new_partner_resilience, 0.0, 1.0)

    def stressful_event(self):
        """
        Process a stressful event using enhanced challenge/hindrance appraisal.

        Returns challenge and hindrance values for integration with daily dynamics.
        Uses dynamic threshold adjustment based on challenge/hindrance as specified
        in the theoretical model: T_eff = T_base + λ_C*challenge - λ_H*hindrance

        Returns:
            Tuple of (challenge, hindrance) values for the event
        """
        # Generate stress event using utility function
        event = generate_stress_event(rng=self._rng)

        # Get configuration parameters for threshold adjustment
        cfg = get_config()
        base_threshold = cfg.get('threshold', 'base_threshold')
        challenge_scale = cfg.get('threshold', 'challenge_scale')
        hindrance_scale = cfg.get('threshold', 'hindrance_scale')

        # Compute challenge and hindrance using appraisal weights
        weights = AppraisalWeights(
            omega_c=cfg.get('appraisal', 'omega_c'),
            omega_p=cfg.get('appraisal', 'omega_p'),
            omega_o=cfg.get('appraisal', 'omega_o'),
            bias=cfg.get('appraisal', 'bias'),
            gamma=cfg.get('appraisal', 'gamma')
        )

        # Process stress event to get challenge/hindrance values
        is_stressed, challenge, hindrance = process_stress_event(
            event=event,
            threshold_params=ThresholdParams(
                base_threshold=base_threshold,
                challenge_scale=challenge_scale,
                hindrance_scale=hindrance_scale
            ),
            weights=weights,
            rng=self._rng
        )

        if not is_stressed:
            return 0.0, 0.0  # No challenge/hindrance if not stressed

        # Determine coping outcome using utility function
        coped_successfully = self._rng.random() < self.resilience

        # Use resources for coping if successful
        if coped_successfully:
            resource_cost = cfg.get('agent', 'resource_cost')
            self.resources = max(0.0, self.resources - resource_cost)

        # Track consecutive hindrances for overload effects
        if hindrance > challenge:  # More hindrance than challenge
            self.consecutive_hindrances = getattr(self, 'consecutive_hindrances', 0) + 1
        else:
            self.consecutive_hindrances = 0  # Reset if not predominantly hindrance

        # Track stress breach count for network adaptation
        self.stress_breach_count = getattr(self, 'stress_breach_count', 0) + 1

        # Allocate resources to protective factors for next time step
        self._allocate_protective_factors()

        # Adapt network based on stress patterns
        self._adapt_network()

        return challenge, hindrance

    def _allocate_protective_factors(self):
        """
        Allocate available resources across protective factors using enhanced dynamics.

        Uses current stress state and resilience to determine optimal allocation.
        """
        from .affect_utils import allocate_protective_resources, ProtectiveFactors, ResourceParams

        # Create protective factors object with current efficacy levels
        protective_factors = ProtectiveFactors(
            social_support=self.protective_factors['social_support'],
            family_support=self.protective_factors['family_support'],
            formal_intervention=self.protective_factors['formal_intervention'],
            psychological_capital=self.protective_factors['psychological_capital']
        )

        # Adjust allocation based on current stress and resilience state
        available_for_allocation = self.resources * 0.3  # Allocate 30% of resources to protective factors

        if available_for_allocation > 0:
            # Allocate resources using enhanced utility function
            allocations = allocate_protective_resources(
                available_resources=available_for_allocation,
                protective_factors=protective_factors,
                rng=self._rng
            )

            # Update protective factor levels based on allocations
            total_allocated = sum(allocations.values())
            if total_allocated > 0:
                # Update efficacy based on resource investment and current needs
                for factor, allocation in allocations.items():
                    if allocation > 0:
                        # Current efficacy influences how effectively resources are used
                        current_efficacy = self.protective_factors[factor]
                        # Investment return is higher when current efficacy is lower (more room for improvement)
                        improvement_rate = 0.5  # Fixed improvement rate for now
                        investment_effectiveness = 1.0 - current_efficacy  # Higher return when efficacy is low

                        efficacy_increase = allocation * improvement_rate * investment_effectiveness
                        self.protective_factors[factor] = min(1.0, current_efficacy + efficacy_increase)

                        # Use some resources for the allocation
                        self.resources -= allocation

    def _get_resilience_boost_from_protective_factors(self):
        """
        Calculate resilience boost from active protective factors.

        Returns:
            Float indicating resilience boost from protective factors
        """
        total_boost = 0.0

        # Each protective factor provides boost based on efficacy and current resilience need
        for factor, efficacy in self.protective_factors.items():
            if efficacy > 0:
                # Boost is higher when resilience is low (more needed)
                need_multiplier = max(0.1, 1.0 - self.resilience)
                boost_rate = 0.1  # Fixed boost rate for now
                total_boost += efficacy * need_multiplier * boost_rate

        return total_boost

    def _get_neighbor_affects(self):
        """
        Get affect values of neighboring agents for social influence calculations.

        Returns:
            List of neighbor affect values
        """
        neighbors = list(
            self.model.grid.get_neighbors(
                self.pos, include_center=False
            )
        )

        return [neighbor.affect for neighbor in neighbors if hasattr(neighbor, 'affect')]

    def _adapt_network(self):
        """
        Adapt network connections based on stress patterns and social support effectiveness.

        Implements the theoretical network adaptation mechanisms:
        - Rewiring when stress threshold is breached repeatedly
        - Strengthening ties with effective support providers
        - Homophily based on similar stress levels
        """
        cfg = get_config()

        # Check if agent should consider network adaptation
        stress_breach_count = getattr(self, 'stress_breach_count', 0)
        adaptation_threshold = 3  # Fixed adaptation threshold for now

        if stress_breach_count < adaptation_threshold:
            return

        # Get current neighbors
        current_neighbors = list(
            self.model.grid.get_neighbors(
                self.pos, include_center=False
            )
        )

        if not current_neighbors:
            return

        # Calculate adaptation metrics
        rewire_prob = 0.01  # Fixed rewire probability for now
        homophily_strength = 0.7  # Fixed homophily strength for now

        # Check each neighbor for potential rewiring
        for neighbor in current_neighbors:
            if self._rng.random() < rewire_prob:
                # Calculate similarity with neighbor (for homophily)
                affect_similarity = 1.0 - abs(self.affect - neighbor.affect)
                resilience_similarity = 1.0 - abs(self.resilience - neighbor.resilience)
                overall_similarity = (affect_similarity + resilience_similarity) / 2.0

                # Calculate support effectiveness (track if implemented)
                support_effectiveness = getattr(self, '_get_support_effectiveness', lambda n: 0.5)(neighbor)

                # Decide whether to keep or rewire connection
                keep_connection_prob = (overall_similarity * homophily_strength +
                                     support_effectiveness * (1.0 - homophily_strength))

                if self._rng.random() > keep_connection_prob:
                    # Rewire: find a new potential connection
                    self._rewire_to_similar_agent(current_neighbors)

        # Reset stress breach count after adaptation
        self.stress_breach_count = 0

    def _rewire_to_similar_agent(self, exclude_agents):
        """
        Find and connect to a new agent with similar stress characteristics.

        Args:
            exclude_agents: List of agents to exclude from potential connections
        """
        # Get all agents except current neighbors and self
        all_agents = [agent for agent in self.model.agents if agent != self]
        available_agents = [agent for agent in all_agents if agent not in exclude_agents]

        if not available_agents:
            return

        # Find agents with similar affect/resilience levels
        similarities = []
        for agent in available_agents:
            affect_similarity = 1.0 - abs(self.affect - agent.affect)
            resilience_similarity = 1.0 - abs(self.resilience - agent.resilience)
            overall_similarity = (affect_similarity + resilience_similarity) / 2.0
            similarities.append((agent, overall_similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Try to connect to most similar agent
        for similar_agent, _ in similarities[:3]:  # Try top 3 most similar
            # Check if connection is possible (simplified - would need actual grid management)
            # This is a placeholder for the actual network rewiring logic
            # In a real implementation, this would involve:
            # 1. Checking if the target position has space
            # 2. Updating the grid structure
            # 3. Updating both agents' neighbor relationships
            pass

    def _get_support_effectiveness(self, neighbor):
        """
        Get the effectiveness of a neighbor as a support provider.

        Args:
            neighbor: Neighbor agent to evaluate

        Returns:
            Float indicating support effectiveness (0.0 to 1.0)
        """
        # Placeholder implementation - in practice this would track:
        # - Historical success of support provided
        # - Response times to support requests
        # - Quality of support based on neighbor's own resilience/affect

        # For now, return a value based on neighbor's current state
        neighbor_fitness = (neighbor.resilience + (1.0 + neighbor.affect) / 2.0) / 2.0
        return min(1.0, neighbor_fitness + 0.2)  # Add some base effectiveness
