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
    compute_stress_impact_on_resilience, clamp, InteractionConfig
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

        # Initialize protective factors (from memory bank architecture)
        self.protective_factors = {
            'social_support': 0.5,
            'family_support': 0.5,
            'formal_intervention': 0.5,
            'psychological_capital': 0.5
        }

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
        Execute one day of simulation.

        Performs random sequence of social interactions and stress events,
        using utility functions for all domain-specific behaviors.
        """
        # Determine number of subevents using utility function
        n_subevents = sample_poisson(
            lam=config.get('agent', 'subevents_per_day'),  # subevents per day from config
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

        # Execute actions using utility functions
        for action in actions:
            if action == "interact":
                self.interact()
            elif action == "stress":
                self.stressful_event()

        # Apply resource regeneration using utility function
        from .affect_utils import compute_resource_regeneration, ResourceParams
        regen_params = ResourceParams(
            base_regeneration=config.get('resource', 'base_regeneration')
        )

        self.resources += compute_resource_regeneration(self.resources, regen_params)

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
        self.affect = new_self_affect
        partner.affect = new_partner_affect
        self.resilience = new_self_resilience
        partner.resilience = new_partner_resilience

    def stressful_event(self):
        """
        Process a stressful event using utility functions.

        Delegates stress generation, appraisal, and outcome computation to utilities.
        """
        # Generate stress event using utility function
        event = generate_stress_event(rng=self._rng)

        # Set up stress processing parameters (from memory bank)
        threshold_params = ThresholdParams(
            base_threshold=0.5,
            challenge_scale=0.15,
            hindrance_scale=0.25
        )

        weights = AppraisalWeights(
            omega_c=1.0, omega_p=1.0, omega_o=1.0,
            bias=0.0, gamma=6.0
        )

        # Process stress event using utility function
        is_stressed, challenge, hindrance = process_stress_event(
            event=event,
            threshold_params=threshold_params,
            weights=weights,
            rng=self._rng
        )

        if not is_stressed:
            return

        # Determine coping outcome using utility function
        coped_successfully = self.rng.random() < self.resilience

        # Compute affect and resilience changes using utility functions
        affect_change = compute_stress_impact_on_affect(
            current_affect=self.affect,
            is_stressed=True,
            coped_successfully=coped_successfully
        )

        resilience_change = compute_stress_impact_on_resilience(
            current_resilience=self.resilience,
            is_stressed=True,
            coped_successfully=coped_successfully
        )

        # Apply changes
        self.affect += affect_change
        self.resilience += resilience_change

        # Use resources for coping if successful
        if coped_successfully:
            resource_cost = config.get('agent', 'resource_cost')
            self.resources = max(0.0, self.resources - resource_cost)
