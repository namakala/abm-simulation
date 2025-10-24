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
    generate_pss10_responses, compute_pss10_score, generate_pss10_item_response,
    initialize_pss10_from_items, generate_pss10_from_stress_dimensions,
    update_stress_dimensions_from_pss10_feedback, update_stress_dimensions_from_event,
    decay_recent_stress_intensity, validate_theoretical_correlations,
    estimate_pss10_from_stress_dimensions, extract_controllability_from_pss10,
    extract_overload_from_pss10, compute_stress_from_pss10
)

from src.python.affect_utils import (
    process_interaction, compute_stress_impact_on_affect,
    compute_stress_impact_on_resilience, clamp, InteractionConfig,
    update_affect_dynamics, update_resilience_dynamics,
    AffectDynamicsConfig, ResilienceDynamicsConfig,
    compute_resource_regeneration, ResourceParams,
    compute_homeostatic_adjustment, scale_homeostatic_rate,
    process_stress_event_with_new_mechanism, StressProcessingConfig,
    compute_daily_affect_reset, compute_stress_decay,
    get_neighbor_affects, integrate_social_resilience_optimization
)

from src.python.resource_utils import (
    ProtectiveFactors, ResourceOptimizationConfig,
    compute_resilience_optimized_resource_cost, compute_resource_efficiency_gain,
    allocate_resilience_optimized_resources, compute_resource_depletion_with_resilience,
    process_social_resource_exchange, update_protective_factors_with_allocation,
    get_resilience_boost_from_protective_factors, allocate_protective_factors,
    calculate_recent_social_benefit, allocate_protective_factors_with_social_boost
)

from src.python.math_utils import sample_poisson, create_rng, sample_normal, tanh_transform, sigmoid_transform
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
                'initial_resilience_mean': cfg.get('agent', 'initial_resilience_mean'),
                'initial_resilience_sd': cfg.get('agent', 'initial_resilience_sd'),
                'initial_affect_mean': cfg.get('agent', 'initial_affect_mean'),
                'initial_affect_sd': cfg.get('agent', 'initial_affect_sd'),
                'initial_resources_mean': cfg.get('agent', 'initial_resources_mean'),
                'initial_resources_sd': cfg.get('agent', 'initial_resources_sd'),
                'stress_probability': cfg.get('agent', 'stress_probability'),
                'coping_success_rate': cfg.get('agent', 'coping_success_rate'),
                'subevents_per_day': cfg.get('agent', 'subevents_per_day')
            }

        # Random number generator for reproducible testing
        # Note: Mesa Agent base class has 'rng' property, so we use '_rng'
        self._rng = create_rng(getattr(model, 'seed', None))

        # Initialize state variables using new transformation pipeline
        # Use sigmoid_transform for [0,1] bounds (resilience, baseline_resilience, resources)
        self.baseline_resilience = sigmoid_transform(
            mean=config['initial_resilience_mean'],
            std=config['initial_resilience_sd'],
            rng=self._rng
        )
        self.resilience = self.baseline_resilience
        self.resources = sigmoid_transform(
            mean=config['initial_resources_mean'],
            std=config['initial_resources_sd'],
            rng=self._rng
        )

        # Use tanh_transform for [-1,1] bounds (affect, baseline_affect)
        self.baseline_affect = tanh_transform(
            mean=config['initial_affect_mean'],
            std=config['initial_affect_sd'],
            rng=self._rng
        )
        self.affect = self.baseline_affect

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

        # Enhanced stress state tracking for dynamic PSS-10 updates
        self.recent_stress_intensity = 0.0  # Tracks recent stress for immediate PSS-10 response
        self.stress_momentum = 0.0  # Tracks rate of stress change for predictive updates
        self.last_stress_update = 0  # Track timing of stress updates for decay calculations

        # Initialize daily interaction tracking attributes
        self.daily_interactions = 0  # Count of daily social interactions
        self.daily_support_exchanges = 0  # Count of daily support exchanges

        # Initialize PSS-10 state variables
        self.pss10_responses = {}  # Individual PSS-10 item responses
        self.stress_controllability = 0.5  # Controllability stress level ∈ [0,1]
        self.stress_overload = 0.5  # Overload stress level ∈ [0,1]
        self.pss10 = 0  # Total PSS-10 score (0-40)
        self.stressed = False  # Stress classification based on PSS-10 threshold
        self.daily_pss10_scores = []  # List to collect PSS-10 scores for the current day

        # Configuration for utility functions
        self.stress_config = {
            'stress_probability': config['stress_probability'],
            'coping_success_rate': config['coping_success_rate']
        }

        self.interaction_config = InteractionConfig()

        # Initialize PSS-10 scores using utility function
        self._initialize_pss10_scores()

        # Step 3: Initialize stress level based on the initialized PSS-10 score
        self._initialize_stress_from_pss10()

    def _initialize_pss10_scores(self):
        """
        Initialize PSS-10 scores and map to stress levels during agent creation.

        Uses utility function to generate initial PSS-10 responses and dimensions.
        """
        # Generate initial controllability and overload scores
        cfg = get_config()
        controllability_score = sigmoid_transform(
            mean=cfg.get('stress', 'controllability_mean'),
            std=cfg.get('pss10', 'controllability_sd'),
            rng=self._rng
        )
        overload_score = sigmoid_transform(
            mean=cfg.get('stress', 'overload_mean'),
            std=cfg.get('pss10', 'overload_sd'),
            rng=self._rng
        )

        # Use utility function to initialize PSS-10
        pss10_data = initialize_pss10_from_items(
            controllability_score=controllability_score,
            overload_score=overload_score,
            rng=self._rng
        )

        # Update agent state with PSS-10 data
        self.pss10_responses = pss10_data['pss10_responses']
        self.stress_controllability = pss10_data['stress_controllability']
        self.stress_overload = pss10_data['stress_overload']
        self.pss10 = pss10_data['pss10_score']
        self.stressed = pss10_data['stressed']

    def step(self):
        """
        Execute one day of simulation with integrated stress and affect dynamics.

        Coordinates stress events, social interactions, and baseline dynamics
        to create realistic mental health trajectories.
        """
        # Get configuration for dynamics
        affect_config = getattr(self, 'affect_config', None) or AffectDynamicsConfig()
        resilience_config = ResilienceDynamicsConfig()

        # Store initial affect and resilience values at the beginning of each day
        initial_affect = self.affect
        initial_resilience = self.resilience

        # Get neighbor affects for social influence throughout the day
        neighbor_affects = get_neighbor_affects(self, self.model)

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

        print(f"DEBUG: Total stressful_event calls: {stress_events_count}")

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
        protective_boost = get_resilience_boost_from_protective_factors(
            protective_factors=self.protective_factors,
            baseline_resilience=self.baseline_resilience,
            current_resilience=self.resilience
        )
        self.resilience = min(1.0, self.resilience + protective_boost)

        # Apply enhanced resource regeneration with affect influence
        regen_params = ResourceParams(
            base_regeneration=config.get('resource', 'base_regeneration')
        )

        # Affect influences resource regeneration (positive affect helps recovery)
        affect_multiplier = 1.0 + 0.2 * max(0.0, self.affect)  # Positive affect boosts regeneration
        base_regeneration = compute_resource_regeneration(self.resources, regen_params)
        self.resources += base_regeneration * affect_multiplier

        # Integrate social support with resilience optimization
        self.resilience = integrate_social_resilience_optimization(
            current_resilience=self.resilience,
            daily_interactions=self.daily_interactions,
            daily_support_exchanges=self.daily_support_exchanges,
            resources=self.resources,
            baseline_resilience=self.baseline_resilience,
            protective_factors=self.protective_factors,
            rng=self._rng
        )

        # Decay consecutive hindrances over time if no new hindrance events
        if hasattr(self, 'consecutive_hindrances') and self.consecutive_hindrances > 0:
            # Slowly decay consecutive hindrances when no new hindrance events occur
            decay_rate = config.get('dynamics', 'stress_decay_rate')
            self.consecutive_hindrances = max(0, self.consecutive_hindrances - decay_rate)

        # Apply homeostatic adjustment to both affect and resilience
        # This pulls values back toward their FIXED baseline (natural equilibrium point)

        # Get homeostatic rates from configuration
        cfg = get_config()
        affect_homeostatic_rate = cfg.get('affect_dynamics', 'homeostatic_rate')
        resilience_homeostatic_rate = cfg.get('resilience_dynamics', 'homeostatic_rate')
        pss10_threshold = cfg.get('pss10', 'threshold')

        # Scale the homeostatic rate based on resources and stress
        scaled_affect_homeostatic_rate = scale_homeostatic_rate(
            affect_homeostatic_rate,
            self.resources,
            self.current_stress
        )

        scaled_resilience_homeostatic_rate = scale_homeostatic_rate(
            resilience_homeostatic_rate,
            self.resources,
            self.current_stress
        )

        # Apply homeostatic adjustment to affect using FIXED baseline
        self.affect = compute_homeostatic_adjustment(
            initial_value=self.baseline_affect,  # Use fixed baseline, not daily initial value
            final_value=self.affect,
            homeostatic_rate=scaled_affect_homeostatic_rate,
            value_type='affect'
        )

        # Apply homeostatic adjustment to resilience using FIXED baseline
        self.resilience = compute_homeostatic_adjustment(
            initial_value=self.baseline_resilience,  # Use fixed baseline, not daily initial value
            final_value=self.resilience,
            homeostatic_rate=scaled_resilience_homeostatic_rate,
            value_type='resilience'
        )

        # NOTE: baseline_affect and baseline_resilience remain FIXED (not updated daily)
        # This ensures homeostasis pulls toward the agent's natural equilibrium point

        # Consolidate daily PSS-10 scores
        if self.daily_pss10_scores:
            avg_score = np.mean(self.daily_pss10_scores)
            rounded_score = round(avg_score)
            self.pss10 = rounded_score

            # Step 7: Use the daily PSS-10 score to initialize stress level for next day
            self._update_stress_from_daily_pss10(rounded_score)

        # Clear daily scores for next day
        self.daily_pss10_scores = []

        # Clamp values that are not handled by transformation pipeline
        # Note: resilience, affect, and resources are now handled by transformation pipeline
        self.current_stress = clamp(self.current_stress, 0.0, 1.0)
        self.stress_controllability = clamp(self.stress_controllability, 0.0, 1.0)
        self.stress_overload = clamp(self.stress_overload, 0.0, 1.0)
        self.stressed = (self.pss10 >= pss10_threshold)

    def interact(self):
        """
        Interact with a random neighbor using utility functions with social resource exchange.

        Delegates all domain logic to utility functions for modularity and testability.
        Implements social resource exchange mechanism to fix correlation issues by allowing
        agents to share resources during meaningful support interactions.

        Returns:
            Dictionary with interaction results including support exchange detection and resource transfers
        """
        # Check if agent has a valid position
        if self.pos is None:
            return {
                'support_exchange': False,
                'affect_change': 0.0,
                'resilience_change': 0.0,
                'resource_transfer': 0.0,
                'received_resources': 0.0
            }

        # Get neighbors using Mesa's grid
        try:
            neighbors = list(
                self.model.grid.get_neighbors(
                    self.pos, include_center=False
                )
            )
        except Exception:
            # Return empty result if there are issues with neighbor lookup
            return {
                'support_exchange': False,
                'affect_change': 0.0,
                'resilience_change': 0.0,
                'resource_transfer': 0.0,
                'received_resources': 0.0
            }

        if not neighbors:
            return {
                'support_exchange': False,
                'affect_change': 0.0,
                'resilience_change': 0.0,
                'resource_transfer': 0.0,
                'received_resources': 0.0
            }

        # Store original values for change calculation
        original_self_affect = self.affect
        original_self_resilience = self.resilience
        original_self_resources = self.resources

        # Select random interaction partner
        partner = self._rng.choice(neighbors)

        # Store original partner values for change calculation
        original_partner_affect = partner.affect
        original_partner_resilience = partner.resilience
        original_partner_resources = partner.resources

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

        # Implement social resource exchange mechanism with resilience optimization
        social_benefit = calculate_recent_social_benefit(self.daily_support_exchanges)
        _, _, new_self_resources, new_partner_resources = process_social_resource_exchange(
            self_resources=original_self_resources,
            partner_resources=original_partner_resources,
            self_resilience=self.resilience,
            partner_resilience=partner.resilience,
            social_support_boost=1.0 + (self.protective_factors['social_support'] * 0.1)
        )

        # Calculate resource changes for return values
        resource_transfer = abs(new_self_resources - original_self_resources)
        received_resources = new_self_resources - original_self_resources

        # Update agent resources
        self.resources = new_self_resources
        partner.resources = new_partner_resources

        # Update resources after exchange
        self.resources = clamp(self.resources, 0.0, 1.0)
        partner.resources = clamp(partner.resources, 0.0, 1.0)

        # Detect support exchange: when at least one agent benefits significantly
        # Support exchange occurs when there's meaningful positive change in affect, resilience, or resources
        # This captures when social interaction provides genuine emotional, psychological, or material benefit
        # Threshold of 0.05 ensures we only count meaningful improvements, not minor fluctuations
        support_threshold = 0.05  # Minimum change to count as support
        support_exchange = (
            self_affect_change > support_threshold or
            self_resilience_change > support_threshold or
            resource_transfer > support_threshold or  # Resource giving as support
            partner_affect_change > support_threshold or
            partner_resilience_change > support_threshold or
            received_resources > support_threshold  # Resource receiving as support
        )

        return {
            'support_exchange': support_exchange,
            'affect_change': self_affect_change,
            'resilience_change': self_resilience_change,
            'resource_transfer': resource_transfer,
            'received_resources': received_resources,
            'partner_affect_change': partner_affect_change,
            'partner_resilience_change': partner_resilience_change
        }

    def stressful_event(self):
        """
        Process a stressful event using the complete stress processing pipeline:
        Stress Event → current_stress → stress_dimensions → PSS-10 → stress_dimensions (feedback)

        Returns challenge and hindrance values for integration with daily dynamics.
        Implements complete theoretical loop ensuring all correlations are achieved.

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

        # STEP 1: Track ALL stress events for complete processing loop
        self.daily_stress_events.append({
            'challenge': challenge,
            'hindrance': hindrance,
            'is_stressed': is_stressed,
            'stress_level': 0.0,
            'coped_successfully': False,
            'event_controllability': event.controllability,
            'event_overload': event.overload
        })

        if not is_stressed:
            # Even non-stressful events provide learning opportunities for stress dimensions
            (
                self.stress_controllability,
                self.stress_overload,
                self.recent_stress_intensity,
                self.stress_momentum
            ) = update_stress_dimensions_from_event(
                current_controllability=self.stress_controllability,
                current_overload=self.stress_overload,
                challenge=challenge,
                hindrance=hindrance,
                coped_successfully=True,  # No coping needed for non-stressful events
                is_stressful=False
            )
            return challenge, hindrance

        # STEP 2: Get neighbor affects for social influence on coping
        neighbor_affects = get_neighbor_affects(self, self.model)

        # STEP 3: Use enhanced stress processing mechanism with complete feedback loop
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

        # STEP 4: Update core agent state
        self.affect = new_affect
        self.resilience = new_resilience
        self.current_stress = new_stress

        # STEP 5: Update stress dimensions based on event outcome (feedback loop)
        (
            self.stress_controllability,
            self.stress_overload,
            self.recent_stress_intensity,
            self.stress_momentum
        ) = update_stress_dimensions_from_event(
            current_controllability=self.stress_controllability,
            current_overload=self.stress_overload,
            challenge=challenge,
            hindrance=hindrance,
            coped_successfully=coped_successfully,
            is_stressful=True
        )

        # STEP 6: Generate PSS-10 from updated stress dimensions
        pss10_data = generate_pss10_from_stress_dimensions(
            stress_controllability=self.stress_controllability,
            stress_overload=self.stress_overload,
            recent_stress_intensity=self.recent_stress_intensity,
            stress_momentum=self.stress_momentum,
            rng=self._rng
        )
        self.pss10_responses = pss10_data['pss10_responses']
        self.pss10 = pss10_data['pss10_score']
        self.stressed = pss10_data['stressed']

        # Collect PSS-10 score for daily consolidation
        self.daily_pss10_scores.append(self.pss10)

        # STEP 7: Update stress dimensions from PSS-10 feedback (complete loop)
        self.stress_controllability, self.stress_overload = update_stress_dimensions_from_pss10_feedback(
            current_controllability=self.stress_controllability,
            current_overload=self.stress_overload,
            pss10_responses=self.pss10_responses
        )

        # STEP 8: Validate theoretical correlations are maintained
        validate_theoretical_correlations(
            challenge=challenge,
            hindrance=hindrance,
            coped_successfully=coped_successfully,
            stress_controllability=self.stress_controllability,
            stress_overload=self.stress_overload,
            pss10_score=self.pss10,
            current_stress=self.current_stress
        )

        # Update the tracked event with complete processing results
        if self.daily_stress_events:
            self.daily_stress_events[-1].update({
                'stress_level': new_stress,
                'coped_successfully': coped_successfully,
                'final_stress_controllability': self.stress_controllability,
                'final_stress_overload': self.stress_overload,
                'pss10_score': self.pss10
            })

        # STEP 8: Use resources for coping attempt with complete stress state
        base_resource_cost = cfg.get('agent', 'resource_cost')
        resource_config = ResourceOptimizationConfig()

        # Compute resilience-optimized resource cost using complete stress state
        optimized_cost = compute_resilience_optimized_resource_cost(
            base_cost=base_resource_cost,
            current_resilience=self.resilience,
            challenge=challenge,
            hindrance=hindrance,
            config=resource_config
        )

        # Apply resource depletion during coping attempt
        self.resources = compute_resource_depletion_with_resilience(
            current_resources=self.resources,
            cost=optimized_cost,
            current_resilience=self.resilience,
            coping_successful=coped_successfully,
            config=resource_config
        )

        # STEP 9: Track consecutive hindrances for overload effects
        if hindrance > challenge:
            self.consecutive_hindrances = getattr(self, 'consecutive_hindrances', 0.0) + 1.0
        else:
            self.consecutive_hindrances = 0.0

        # STEP 10: Track stress breach count for network adaptation
        self.stress_breach_count = getattr(self, 'stress_breach_count', 0) + 1

        # STEP 11: Allocate resources to protective factors with complete stress integration
        if is_stressed and coped_successfully:
            # Use utility function for protective factor allocation
            allocations = allocate_protective_factors(
                available_resources=self.resources * 0.3,
                current_resilience=self.resilience,
                baseline_resilience=self.baseline_resilience,
                protective_factors=self.protective_factors,
                rng=self._rng
            )

            # Update protective factors with allocations
            self.protective_factors = update_protective_factors_with_allocation(
                protective_factors=self.protective_factors,
                allocations=allocations,
                current_resilience=self.resilience
            )

            # Deduct allocated resources
            total_allocated = sum(allocations.values())
            self.resources -= total_allocated

        return challenge, hindrance

    def _initialize_stress_from_pss10(self):
        """
        Initialize current_stress level based on PSS-10 score using utility function.

        This implements Step 3 of the PSS-10 workflow: using the initialized PSS-10 score
        to set the initial current_stress level for the agent.
        """
        self.current_stress = compute_stress_from_pss10(
            pss10_score=self.pss10,
            stress_controllability=self.stress_controllability,
            stress_overload=self.stress_overload
        )

    def _update_stress_from_daily_pss10(self, daily_pss10_score):
        """
        Update current_stress level based on daily consolidated PSS-10 score using utility function.

        This implements Step 7 of the PSS-10 workflow: using the daily PSS-10 score
        to set the stress level for the next day, creating a feedback loop.

        Args:
            daily_pss10_score: The consolidated daily PSS-10 score (0-40)
        """
        # Compute new stress level using utility function
        new_stress_level = compute_stress_from_pss10(
            pss10_score=daily_pss10_score,
            stress_controllability=self.stress_controllability,
            stress_overload=self.stress_overload
        )

        # Apply exponential smoothing to create more realistic stress transitions
        # This prevents stress from changing too abruptly between days
        smoothing_factor = 0.7  # Weight for new stress level (0.3 weight for previous)
        self.current_stress = (smoothing_factor * new_stress_level +
                              (1.0 - smoothing_factor) * self.current_stress)

        # Ensure stress level is in valid range
        self.current_stress = clamp(self.current_stress, 0.0, 1.0)

    def _update_stress_dimensions_from_event(self, challenge, hindrance, coped_successfully):
        """
        Update agent's controllability and overload dimensions based on stress event outcomes.

        This creates a direct feedback loop between stress events and PSS-10 dimensions:
        - Challenge events build controllability when coping succeeds
        - Hindrance events reduce controllability and increase overload
        - Successful coping improves both dimensions
        - Failed coping worsens both dimensions

        Args:
            challenge: Challenge component from event appraisal (0-1)
            hindrance: Hindrance component from event appraisal (0-1)
            coped_successfully: Whether the coping attempt was successful
        """
        # Get configuration for stress dimension updates
        cfg = get_config()

        # Base update rates from configuration
        controllability_update_rate = cfg.get('stress_dynamics', 'controllability_update_rate')
        overload_update_rate = cfg.get('stress_dynamics', 'overload_update_rate')

        # Challenge vs hindrance effects on controllability
        if coped_successfully:
            # Successful coping: challenge builds controllability, hindrance slightly reduces it
            controllability_change = (challenge * 0.15) - (hindrance * 0.08)
        else:
            # Failed coping: both challenge and hindrance reduce controllability
            controllability_change = -(challenge * 0.12) - (hindrance * 0.18)

        # Apply controllability update with decay toward baseline
        baseline_controllability = 0.5  # Neutral baseline
        current_controllability = self.stress_controllability

        # Move toward baseline when no strong events, but allow event-driven changes
        homeostasis_pull = (baseline_controllability - current_controllability) * 0.05
        event_effect = controllability_change * controllability_update_rate

        self.stress_controllability += homeostasis_pull + event_effect
        self.stress_controllability = clamp(self.stress_controllability, 0.0, 1.0)

        # Overload effects: hindrance increases overload, challenge reduces it slightly
        if coped_successfully:
            # Successful coping: hindrance still increases overload but less, challenge reduces it
            overload_change = (hindrance * 0.12) - (challenge * 0.08)
        else:
            # Failed coping: both increase overload significantly
            overload_change = (hindrance * 0.25) + (challenge * 0.15)

        # Apply overload update with decay toward baseline
        baseline_overload = 0.5  # Neutral baseline
        current_overload = self.stress_overload

        # Move toward baseline when no strong events, but allow event-driven changes
        homeostasis_pull = (baseline_overload - current_overload) * 0.05
        event_effect = overload_change * overload_update_rate

        self.stress_overload += homeostasis_pull + event_effect
        self.stress_overload = clamp(self.stress_overload, 0.0, 1.0)

        # Update recent stress intensity and momentum for dynamic PSS-10 response
        self._update_recent_stress_intensity(challenge, hindrance, coped_successfully)

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
            self.consecutive_hindrances = max(0.0, self.consecutive_hindrances - daily_decay_rate)

        # Reset daily PSS-10 scores for new day
        self.daily_pss10_scores = []
