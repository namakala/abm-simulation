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
    StressEvent, AppraisalWeights, ThresholdParams,
    generate_pss10_responses, compute_pss10_score, generate_pss10_item_response
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
    - daily_interactions: Count of daily social interactions ∈ [0,∞)
    - daily_support_exchanges: Count of daily support exchanges ∈ [0,∞)
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

        # Initialize baseline resilience for homeostasis
        self.baseline_resilience = config['initial_resilience']

        # Initialize protective factors
        self.protective_factors = {
            'social_support': 0.5,
            'family_support': 0.5,
            'formal_intervention': 0.5,
            'psychological_capital': 0.5
        }

        # Track hindrances as float to preserve data consistency when decaying
        self.consecutive_hindrances = 0.0

        # Initialize new stress tracking state variables
        self.current_stress = 0.0  # Current stress level ∈ [0,1]
        self.daily_stress_events = []  # Track stress events within current day
        self.stress_history = []  # Historical stress levels for analysis
        self.last_reset_day = 0  # Track when last daily reset occurred

        # Initialize daily interaction tracking attributes
        self.daily_interactions = 0  # Count of daily social interactions
        self.daily_support_exchanges = 0  # Count of daily support exchanges

        # Initialize PSS-10 state variables
        self.pss10_responses = {}  # Individual PSS-10 item responses
        self.stress_controllability = 0.5  # Controllability stress level ∈ [0,1]
        self.stress_overload = 0.5  # Overload stress level ∈ [0,1]
        self.pss10 = 0  # Total PSS-10 score (0-40)
        self.stressed = False  # Stress classification based on PSS-10 threshold

        # Configuration for utility functions
        self.stress_config = {
            'stress_probability': config['stress_probability'],
            'coping_success_rate': config['coping_success_rate']
        }

        self.interaction_config = InteractionConfig()

        # Random number generator for reproducible testing
        # Note: Mesa Agent base class has 'rng' property, so we use '_rng'
        self._rng = create_rng(getattr(model, 'seed', None))

        # Initialize PSS-10 scores using generate_pss10_item_response
        self._initialize_pss10_from_items()

    def _initialize_pss10_from_items(self):
        """
        Initialize PSS-10 responses using generate_pss10_item_response for each item.

        Uses configuration parameters and generates individual item responses
        with proper reverse scoring for items 4, 5, 7, 8.
        """
        # Get configuration values
        cfg = get_config()

        # Generate controllability and overload scores from normal distributions
        controllability_score = self._rng.normal(
            cfg.get('stress', 'controllability_mean'),
            cfg.get('pss10', 'controllability_sd') / 4
        )
        overload_score = self._rng.normal(
            cfg.get('stress', 'overload_mean'),
            cfg.get('pss10', 'overload_sd') / 4
        )

        # Clamp to [0,1] range
        controllability_score = max(0.0, min(1.0, controllability_score))
        overload_score = max(0.0, min(1.0, overload_score))

        # Generate each PSS-10 item response
        for item_num in range(1, 11):
            # Determine if item is reverse scored
            reverse_scored = item_num in [4, 5, 7, 8]

            # Get item parameters from configuration
            item_mean = cfg.get('pss10', 'item_means')[item_num - 1]
            item_sd = cfg.get('pss10', 'item_sds')[item_num - 1]
            controllability_loading = cfg.get('pss10', 'load_controllability')[item_num - 1]
            overload_loading = cfg.get('pss10', 'load_overload')[item_num - 1]

            # Generate item response
            response = generate_pss10_item_response(
                item_mean=item_mean,
                item_sd=item_sd,
                controllability_loading=controllability_loading,
                overload_loading=overload_loading,
                controllability_score=controllability_score,
                overload_score=overload_score,
                reverse_scored=reverse_scored,
                rng=self._rng
            )

            self.pss10_responses[item_num] = response

        # Initialize stress_controllability by averaging items 4, 5, 7, 8, then dividing by 4
        controllability_items = [4, 5, 7, 8]
        controllability_scores = []
        for item_num in controllability_items:
            if item_num in self.pss10_responses:
                # Without reversing the item score, higher PSS-10 response = higher controllability
                response = self.pss10_responses[item_num]
                controllability_scores.append(response / 4.0)  # Normalize to [0,1]
        self.stress_controllability = np.mean(controllability_scores) if controllability_scores else 0.5

        # Initialize stress_overload by averaging items 1, 2, 3, 6, 9, 10, then dividing by 6
        overload_items = [1, 2, 3, 6, 9, 10]
        overload_scores = []
        for item_num in overload_items:
            if item_num in self.pss10_responses:
                # Higher PSS-10 response = higher overload
                response = self.pss10_responses[item_num]
                overload_scores.append(response / 4.0)  # Normalize to [0,1]
        self.stress_overload = np.mean(overload_scores) if overload_scores else 0.5

        # Initialize pss10_score by summing items 1-10
        self.pss10 = compute_pss10_score(self.pss10_responses)

        # Set initial stressed status based on PSS-10 threshold
        cfg = get_config()
        pss10_threshold = cfg.get('pss10', 'threshold')
        self.stressed = (self.pss10 >= pss10_threshold)

    def step(self):
        """
        Execute one day of simulation with integrated stress and affect dynamics.

        Coordinates stress events, social interactions, and baseline dynamics
        to create realistic mental health trajectories.
        """
        # Get configuration for dynamics
        affect_config = AffectDynamicsConfig()
        resilience_config = ResilienceDynamicsConfig()

        # Store initial affect and resilience values at the beginning of each day
        initial_affect = self.affect
        initial_resilience = self.resilience

        # Get neighbor affects for social influence throughout the day
        neighbor_affects = self._get_neighbor_affects()

        # Initialize daily tracking variables
        daily_challenge = 0.0
        daily_hindrance = 0.0
        stress_events_count = 0

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
        # Each action represents a subevent within the day, simulating realistic timing
        for action in actions:
            if action == "interact":
                # Track social interaction and check for meaningful support exchange
                # The interact() method returns detailed information about the interaction outcome
                interaction_result = self.interact()
                self.daily_interactions += 1

                # Check if this was a meaningful support exchange
                # A support exchange occurs when interaction results in positive affect change
                # and/or resilience improvement for either agent (threshold = 0.05)
                # This tracks when social connections provide genuine emotional or psychological support
                if interaction_result and interaction_result.get('support_exchange', False):
                    self.daily_support_exchanges += 1

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

        # Apply integrated resilience dynamics using new stress processing mechanisms
        # The new mechanism handles coping success determination within each stress event
        # and incorporates social interaction effects on coping probability

        # Check if agent received social support during interactions (for resilience boost)
        received_social_support = self.daily_interactions > 0 and self._rng.random() < 0.3

        # Use new resilience dynamics that work with the updated stress processing
        # The coping success is now determined within each stress event using social influence
        self.resilience = update_resilience_dynamics(
            current_resilience=self.resilience,
            coped_successfully=False,  # Will be handled per event in new mechanism
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
            decay_rate = config.get('dynamics', 'stress_decay_rate')
            self.consecutive_hindrances = max(0, self.consecutive_hindrances - decay_rate)

        # Apply homeostatic adjustment to both affect and resilience
        # This pulls values back toward their FIXED baseline (natural equilibrium point)
        from .affect_utils import compute_homeostatic_adjustment

        # Get homeostatic rates from configuration
        cfg = get_config()
        affect_homeostatic_rate = cfg.get('affect_dynamics', 'homeostatic_rate')
        resilience_homeostatic_rate = cfg.get('resilience_dynamics', 'homeostatic_rate')

        # Apply homeostatic adjustment to affect using FIXED baseline
        self.affect = compute_homeostatic_adjustment(
            initial_value=self.baseline_affect,  # Use fixed baseline, not daily initial value
            final_value=self.affect,
            homeostatic_rate=affect_homeostatic_rate,
            value_type='affect'
        )

        # Apply homeostatic adjustment to resilience using FIXED baseline
        self.resilience = compute_homeostatic_adjustment(
            initial_value=self.baseline_resilience,  # Use fixed baseline, not daily initial value
            final_value=self.resilience,
            homeostatic_rate=resilience_homeostatic_rate,
            value_type='resilience'
        )

        # NOTE: baseline_affect and baseline_resilience remain FIXED (not updated daily)
        # This ensures homeostasis pulls toward the agent's natural equilibrium point

        # Update PSS-10 scores based on current stress levels
        self.compute_pss10_score()

        # Clamp all values to valid ranges
        self.resilience = clamp(self.resilience, 0.0, 1.0)
        self.affect = clamp(self.affect, -1.0, 1.0)
        self.resources = clamp(self.resources, 0.0, 1.0)
        self.current_stress = clamp(self.current_stress, 0.0, 1.0)
        self.stress_controllability = clamp(self.stress_controllability, 0.0, 1.0)
        self.stress_overload = clamp(self.stress_overload, 0.0, 1.0)

    def interact(self):
        """
        Interact with a random neighbor using utility functions.

        Delegates all domain logic to utility functions for modularity and testability.

        Returns:
            Dictionary with interaction results including support exchange detection
        """
        # Get neighbors using Mesa's grid
        neighbors = list(
            self.model.grid.get_neighbors(
                self.pos, include_center=False
            )
        )

        if not neighbors:
            return {'support_exchange': False, 'affect_change': 0.0, 'resilience_change': 0.0}

        # Store original values for change calculation
        original_self_affect = self.affect
        original_self_resilience = self.resilience

        # Select random interaction partner
        partner = self._rng.choice(neighbors)

        # Store original partner values for change calculation
        original_partner_affect = partner.affect
        original_partner_resilience = partner.resilience

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

        # Calculate changes for support exchange detection
        # Track how much each agent's state improved (or declined) during interaction
        self_affect_change = self.affect - original_self_affect
        self_resilience_change = self.resilience - original_self_resilience
        partner_affect_change = partner.affect - original_partner_affect
        partner_resilience_change = partner.resilience - original_partner_resilience

        # Detect support exchange: when at least one agent benefits significantly
        # Support exchange occurs when there's meaningful positive change in affect or resilience
        # This captures when social interaction provides genuine emotional or psychological benefit
        # Threshold of 0.05 ensures we only count meaningful improvements, not minor fluctuations
        support_threshold = 0.05  # Minimum change to count as support
        support_exchange = (
            self_affect_change > support_threshold or
            self_resilience_change > support_threshold or
            partner_affect_change > support_threshold or
            partner_resilience_change > support_threshold
        )

        return {
            'support_exchange': support_exchange,
            'affect_change': self_affect_change,
            'resilience_change': self_resilience_change,
            'partner_affect_change': partner_affect_change,
            'partner_resilience_change': partner_resilience_change
        }

    def stressful_event(self):
        """
        Process a stressful event using the complete stress processing pipeline:
        stress exposure → perception → appraisal → coping → resilience/affect changes

        Returns challenge and hindrance values for integration with daily dynamics.
        Uses new stress processing mechanism with social interaction effects on coping.

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

        # Track ALL stress events, not just stressful ones
        self.daily_stress_events.append({
            'challenge': challenge,
            'hindrance': hindrance,
            'is_stressed': is_stressed,
            'stress_level': 0.0,  # Will be updated below if actually stressed
            'coped_successfully': False  # Will be updated below if actually stressed
        })

        if not is_stressed:
            return challenge, hindrance  # No stress processing if not stressed

        # Get neighbor affects for social influence on coping
        neighbor_affects = self._get_neighbor_affects()

        # Use new stress processing mechanism with social interaction effects
        from .affect_utils import process_stress_event_with_new_mechanism, StressProcessingConfig

        stress_config = StressProcessingConfig()
        new_affect, new_resilience, new_stress, coped_successfully = process_stress_event_with_new_mechanism(
            current_affect=self.affect,
            current_resilience=self.resilience,
            current_stress=self.current_stress,
            challenge=challenge,
            hindrance=hindrance,
            neighbor_affects=neighbor_affects,
            config=stress_config
        )

        # Update agent state with new values
        self.affect = new_affect
        self.resilience = new_resilience
        self.current_stress = new_stress

        # Update the tracked event with actual stress processing results
        if self.daily_stress_events:
            self.daily_stress_events[-1].update({
                'stress_level': new_stress,
                'coped_successfully': coped_successfully
            })

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

        # Allocate resources to protective factors only if stressed and coping
        if is_stressed and coped_successfully:
            self._allocate_protective_factors()

        # Adapt network based on stress patterns
        self._adapt_network()

        # Update PSS-10 scores when stress event occurs
        if is_stressed:
            self.compute_pss10_score()

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
                        improvement_rate = config.get('agent_parameters', 'protective_improvement_rate')
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
                boost_rate = config.get('agent_parameters', 'resilience_boost_rate')
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

    def _initialize_pss10_scores(self):
        """
        Initialize PSS-10 scores and map to stress levels during agent creation.

        Generates initial PSS-10 responses using default stress levels (0.5, 0.5)
        and maps them to controllability and overload stress dimensions.
        """
        # Generate initial PSS-10 responses using default stress levels
        initial_responses = generate_pss10_responses(
            controllability=0.5,  # Default initial controllability
            overload=0.5,         # Default initial overload
            rng=self._rng
        )

        # Initialize pss10_responses by generating PSS10 item scores
        self.pss10_responses = initial_responses

        # Initialize stress_controllability by averaging items 4, 5, 7, 8, then dividing by 4
        controllability_items = [4, 5, 7, 8]
        controllability_scores = []
        for item_num in controllability_items:
            if item_num in self.pss10_responses:
                # For reverse scored items, lower PSS-10 response = higher controllability
                response = self.pss10_responses[item_num]
                controllability_scores.append(1.0 - (response / 4.0))  # Normalize to [0,1]
        self.stress_controllability = np.mean(controllability_scores) if controllability_scores else 0.5

        # Initialize stress_overload by averaging items 1, 2, 3, 6, 9, 10, then dividing by 6
        overload_items = [1, 2, 3, 6, 9, 10]
        overload_scores = []
        for item_num in overload_items:
            if item_num in self.pss10_responses:
                # Higher PSS-10 response = higher overload
                response = self.pss10_responses[item_num]
                overload_scores.append(response / 4.0)  # Normalize to [0,1]
        self.stress_overload = np.mean(overload_scores) if overload_scores else 0.5

        # Initialize pss10_score by summing items 1-10
        self.pss10 = compute_pss10_score(initial_responses)

        # Set initial stressed status based on PSS-10 threshold
        cfg = get_config()
        pss10_threshold = cfg.get('pss10', 'threshold')
        self.stressed = (self.pss10 >= pss10_threshold)

    def _update_stress_levels_from_pss10(self):
        """
        Update stress levels based on current PSS-10 responses.

        Maps PSS-10 dimension scores back to controllability and overload stress levels.
        This creates a feedback loop where stress affects PSS-10, which then affects future stress.
        """
        if not self.pss10_responses:
            return

        # Calculate controllability stress from relevant PSS-10 items
        # Items 4, 5, 7, 8 are controllability-focused (reverse scored)
        controllability_items = [4, 5, 7, 8]
        controllability_scores = []

        for item_num in controllability_items:
            if item_num in self.pss10_responses:
                # For reverse scored items, lower PSS-10 response = higher controllability
                response = self.pss10_responses[item_num]
                controllability_scores.append(1.0 - (response / 4.0))  # Normalize to [0,1]

        self.stress_controllability = np.mean(controllability_scores) if controllability_scores else 0.5

        # Calculate overload stress from relevant PSS-10 items
        # Items 1, 2, 3, 6, 9, 10 are overload-focused
        overload_items = [1, 2, 3, 6, 9, 10]
        overload_scores = []

        for item_num in overload_items:
            if item_num in self.pss10_responses:
                # Higher PSS-10 response = higher overload
                response = self.pss10_responses[item_num]
                overload_scores.append(response / 4.0)  # Normalize to [0,1]

        self.stress_overload = np.mean(overload_scores) if overload_scores else 0.5

    def compute_pss10_score(self):
        """
        Recompute and update PSS-10 scores based on current stress levels.

        This function should be called at the end of each iteration step to:
        1. Generate new PSS-10 responses from current stress_controllability and stress_overload
        2. Update pss10_responses and pss10 score
        3. Update stress levels based on new PSS-10 responses
        """
        # Use generate_pss10_responses with controllability and overload from the end of the current iteration
        new_responses = generate_pss10_responses(
            controllability=self.stress_controllability,
            overload=self.stress_overload,
            rng=self._rng
        )

        # Update PSS-10 state
        self.pss10_responses = new_responses

        # Use generate_pss10_responses to obtain the pss10_score
        self.pss10 = compute_pss10_score(new_responses)

        # Update stress levels based on new PSS-10 responses
        self._update_stress_levels_from_pss10()

        # Update stressed status based on PSS-10 threshold
        cfg = get_config()
        pss10_threshold = cfg.get('pss10', 'threshold')
        self.stressed = (self.pss10 >= pss10_threshold)

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
        adaptation_threshold = config.get('agent_parameters', 'network_adaptation_threshold')

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
        rewire_prob = config.get('agent_parameters', 'network_rewire_probability')
        homophily_strength = config.get('agent_parameters', 'network_homophily_strength')

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

    def _daily_reset(self, current_day):
        """
        Perform daily reset of stress tracking variables and apply stress decay.

        This method implements the daily affect reset to baseline and stress decay
        mechanisms as specified in the new stress processing flow. It also handles
        the reset of daily interaction and support exchange counters to ensure
        accurate daily tracking for DataCollector analysis.

        Args:
            current_day: Current simulation day for tracking reset timing
        """
        from .affect_utils import compute_daily_affect_reset, compute_stress_decay, StressProcessingConfig

        # Update last reset day
        self.last_reset_day = current_day

        # Reset daily interaction tracking counters
        # These counters track social interactions and meaningful support exchanges per day
        # Resetting ensures DataCollector gets accurate daily totals, not cumulative counts
        previous_interactions = self.daily_interactions
        previous_support_exchanges = self.daily_support_exchanges

        self.daily_interactions = 0
        self.daily_support_exchanges = 0

        # Validate that counters were properly reset
        # This ensures data integrity for daily tracking and analysis
        if self.daily_interactions != 0 or self.daily_support_exchanges != 0:
            raise ValueError(
                f"Failed to reset daily counters for agent {self.unique_id}. "
                f"Expected: interactions=0, support_exchanges=0. "
                f"Got: interactions={self.daily_interactions}, support_exchanges={self.daily_support_exchanges}"
            )

        # Apply daily affect reset to baseline
        stress_config = StressProcessingConfig()
        self.affect = compute_daily_affect_reset(
            current_affect=self.affect,
            baseline_affect=self.baseline_affect,
            config=stress_config
        )

        # Apply stress decay over time
        self.current_stress = compute_stress_decay(
            current_stress=self.current_stress,
            config=stress_config
        )

        # Store daily stress summary in history for analysis
        if self.daily_stress_events:
            daily_summary = {
                'day': current_day,
                'avg_stress': np.mean([event['stress_level'] for event in self.daily_stress_events]),
                'max_stress': max([event['stress_level'] for event in self.daily_stress_events]),
                'num_events': len(self.daily_stress_events),
                'coping_success_rate': np.mean([event['coped_successfully'] for event in self.daily_stress_events])
            }
            self.stress_history.append(daily_summary)

        # Reset daily stress events for new day
        self.daily_stress_events = []

        # Apply gradual decay to consecutive hindrances over days
        if hasattr(self, 'consecutive_hindrances') and self.consecutive_hindrances > 0:
            # Slowly decay consecutive hindrances over days when no new hindrance events
            daily_decay_rate = 0.05  # Small daily decay rate
            self.consecutive_hindrances = max(0, self.consecutive_hindrances - daily_decay_rate)

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
