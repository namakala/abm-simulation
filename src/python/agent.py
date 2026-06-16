"""
Agent-based model for mental health simulation with modular, testable utilities.

This module implements an agent that experiences stress events and social interactions,
using utility functions for all domain-specific behaviors to ensure modularity and testability.
"""

import numpy as np
import mesa

# Import utility modules
from src.python.stress_utils import (
    generate_stress_event,
    process_stress_event,
    AppraisalWeights,
    ThresholdParams,
    initialize_pss10_from_items,
    update_stress_dimensions_from_event,
    compute_stress_from_pss10,
)

from src.python.affect_utils import (
    clamp,
    InteractionConfig,
    update_affect_dynamics,
    update_resilience_dynamics,
    AffectDynamicsConfig,
    ResilienceDynamicsConfig,
    compute_resource_regeneration,
    ResourceParams,
    compute_homeostatic_adjustment,
    scale_homeostatic_rate,
    StressProcessingConfig,
    compute_daily_affect_reset,
    compute_stress_decay,
    get_neighbor_affects,
    integrate_social_resilience_optimization,
)

from src.python.resource_utils import (
    get_resilience_boost_from_protective_factors,
)

from src.python.math_utils import sample_poisson, create_rng, tanh_transform, sigmoid_transform
from src.python.config import get_config
from src.python.assumption_config import get_assumptions

# Phase functions (Plan 008 orchestrator)
from src.python.phases import (
    run_stress_perception,
    run_resilience_activation,
    run_resource_allocation,
    run_stress_buffering,
)
from src.python.phases.interaction import process_interaction as phase_process_interaction

# Load configuration
config = get_config()


# ── Orchestrator-internal phase functions (Plan 008) ──────────────


def process_affect_dynamics(
    state: dict,
    config: dict,
    rng: np.random.Generator,
) -> dict:
    """Apply daily affect dynamics: homeostasis, peer influence, resource regen.

    Consolidates affect/resilience/resource updates that happen once per day
    after the subevent loop.  Extracted from ``Person.step()``.

    Applies in order:
    1. Affect dynamics (homeostasis + peer influence + event appraisal)
    2. Resilience dynamics + protective factor boost
    3. Resource regeneration (affected by affect and resilience)
    4. Social resilience optimisation
    5. Consecutive hindrance decay
    6. Homeostatic adjustment for affect and resilience

    Args:
        state: Current AgentState dict (not modified).
        config: Config dict with optional keys:
            ``neighbor_affects``, ``daily_challenge``, ``daily_hindrance``,
            ``stress_decay_rate``, ``affect_config``, ``resilience_config``.
        rng: Seeded random number generator.

    Returns:
        PhaseOutput-like dict with ``state_delta`` and ``observation``.
    """
    # ── Read inputs from state ────────────────────────────────────
    affect = state["affect"]
    baseline_affect = state["baseline_affect"]
    resilience = state["resilience"]
    baseline_resilience = state["baseline_resilience"]
    resources = state["resources"]
    current_stress = state["current_stress"]
    daily_interactions = state.get("daily_interactions", 0)
    daily_support_exchanges = state.get("daily_support_exchanges", 0)
    protective_factors = dict(state.get("protective_factors", {}))
    consecutive_hindrances = state.get("consecutive_hindrances", 0.0)

    # ── Read config ───────────────────────────────────────────────
    neighbor_affects = config.get("neighbor_affects", [])
    daily_challenge = config.get("daily_challenge", 0.0)
    daily_hindrance = config.get("daily_hindrance", 0.0)
    stress_decay_rate = config.get("stress_decay_rate", 0.05)
    affect_cfg = config.get("affect_config", None) or AffectDynamicsConfig()
    resilience_cfg = config.get("resilience_config", None) or ResilienceDynamicsConfig()

    # ── 1. Affect dynamics ────────────────────────────────────────
    new_affect = update_affect_dynamics(
        current_affect=affect,
        baseline_affect=baseline_affect,
        neighbor_affects=neighbor_affects,
        challenge=daily_challenge,
        hindrance=daily_hindrance,
        affect_config=affect_cfg,
    )

    # ── 2. Resilience dynamics + PF boost ─────────────────────────
    received_social_support = daily_interactions > 0 and rng.random() < 0.3 if hasattr(rng, "random") else False

    new_resilience = update_resilience_dynamics(
        current_resilience=resilience,
        coped_successfully=False,  # handled per event in new mechanism
        received_social_support=received_social_support,
        consecutive_hindrances=consecutive_hindrances,
        resilience_config=resilience_cfg,
    )

    protective_boost = get_resilience_boost_from_protective_factors(
        protective_factors=protective_factors,
        baseline_resilience=baseline_resilience,
        current_resilience=new_resilience,
    )
    new_resilience = min(1.0, new_resilience + protective_boost)

    # ── 3. Resource regeneration ──────────────────────────────────
    cfg = get_config()
    regen_params = ResourceParams(base_regeneration=cfg.get("resource", "base_regeneration"))
    affect_mult = 1.0 + 0.5 * max(0.0, new_affect)
    resil_mult = 1.0 + 0.3 * new_resilience
    base_regeneration = compute_resource_regeneration(resources, regen_params)
    new_resources = resources + base_regeneration * affect_mult * resil_mult
    new_resources = min(1.0, max(0.0, new_resources))

    # ── 4. Social resilience optimisation ─────────────────────────
    new_resilience = integrate_social_resilience_optimization(
        current_resilience=new_resilience,
        daily_interactions=daily_interactions,
        daily_support_exchanges=daily_support_exchanges,
        resources=new_resources,
        baseline_resilience=baseline_resilience,
        protective_factors=protective_factors,
        rng=rng,
    )

    # ── 5. Consecutive hindrance decay ────────────────────────────
    new_consecutive_hindrances = consecutive_hindrances
    if consecutive_hindrances > 0:
        new_consecutive_hindrances = max(0.0, consecutive_hindrances - stress_decay_rate)

    # ── 6. Homeostatic adjustment ─────────────────────────────────
    affect_homeostatic_rate = get_assumptions().stress.affect_homeostatic_rate
    resilience_homeostatic_rate = get_assumptions().stress.resilience_homeostatic_rate

    scaled_affect_rate = scale_homeostatic_rate(affect_homeostatic_rate, new_resources, current_stress)
    scaled_resilience_rate = scale_homeostatic_rate(resilience_homeostatic_rate, new_resources, current_stress)

    new_affect = compute_homeostatic_adjustment(
        initial_value=baseline_affect,
        final_value=new_affect,
        homeostatic_rate=scaled_affect_rate,
        value_type="affect",
    )

    new_resilience = compute_homeostatic_adjustment(
        initial_value=baseline_resilience,
        final_value=new_resilience,
        homeostatic_rate=scaled_resilience_rate,
        value_type="resilience",
    )

    # ── Build PhaseOutput ─────────────────────────────────────────
    state_delta = {
        "affect": new_affect,
        "resilience": new_resilience,
        "resources": new_resources,
        "consecutive_hindrances": new_consecutive_hindrances,
    }

    observation = {
        "neighbor_affects_summary": {
            "count": len(neighbor_affects),
            "mean": float(np.mean(neighbor_affects)) if neighbor_affects else 0.0,
        },
        "protective_boost": protective_boost,
        "regeneration": base_regeneration * affect_mult * resil_mult,
    }

    return {"state_delta": state_delta, "observation": observation}


def process_pss10_consolidation(
    state: dict,
    config: dict,
    rng: np.random.Generator,
) -> dict:
    """Consolidate daily PSS-10 scores and update stress level.

    Extracted from ``Person.step()`` lines 363-380.

    Averages the day's PSS-10 scores, updates current_stress via
    exponential smoothing, clamps all values, and updates ``stressed``
    status based on the PSS-10 threshold.

    Args:
        state: Current AgentState dict (not modified).
        config: Config dict (``pss10_threshold`` can be overridden).
        rng: Seeded random number generator (unused, for protocol compat).

    Returns:
        PhaseOutput-like dict with ``state_delta`` and ``observation``.
    """
    # ── Read state ────────────────────────────────────────────────
    daily_pss10_scores = state.get("daily_pss10_scores", [])
    current_stress = state.get("current_stress", 0.0)
    stress_controllability = state.get("stress_controllability", 0.5)
    stress_overload = state.get("stress_overload", 0.5)
    pss10 = state.get("pss10", 0)

    # ── Read config ───────────────────────────────────────────────
    cfg = get_config()
    pss10_threshold = config.get("pss10_threshold", cfg.get("pss10", "threshold"))

    # ── Consolidate ───────────────────────────────────────────────
    if daily_pss10_scores:
        # Compute new stress from PSS-10 dimensions
        new_stress_level = compute_stress_from_pss10(
            stress_controllability=stress_controllability,
            stress_overload=stress_overload,
        )
        smoothing_factor = 0.7
        current_stress = smoothing_factor * new_stress_level + (1.0 - smoothing_factor) * current_stress

    # ── Clamp values ──────────────────────────────────────────────
    current_stress = clamp(current_stress, 0.0, 1.0)
    stress_controllability = clamp(stress_controllability, 0.0, 1.0)
    stress_overload = clamp(stress_overload, 0.0, 1.0)

    # ── Consolidate PSS-10 score ─────────────────────────────────
    if daily_pss10_scores:
        consolidated_pss10 = int(round(float(np.mean(daily_pss10_scores))))
    else:
        consolidated_pss10 = pss10  # keep existing

    # ── Apply exponential smoothing across days (Plan 011) ────────
    from src.python.stress_utils import smooth_pss10_across_days
    from src.python.assumption_config import get_assumptions

    prev_smoothed = state.get("pss10_smoothed", None)
    alpha = get_assumptions().stress.pss10_smoothing_alpha
    new_smoothed = smooth_pss10_across_days(consolidated_pss10, prev_smoothed, alpha)
    final_pss10 = int(round(new_smoothed))

    # ── Update stressed status ────────────────────────────────────
    stressed = final_pss10 >= pss10_threshold

    # ── Build PhaseOutput ─────────────────────────────────────────
    state_delta = {
        "pss10": final_pss10,
        "pss10_smoothed": new_smoothed,
        "current_stress": current_stress,
        "stress_controllability": stress_controllability,
        "stress_overload": stress_overload,
        "stressed": stressed,
        "daily_pss10_scores": [],  # cleared for next day
    }

    observation = {
        "avg_pss10": float(np.mean(daily_pss10_scores)) if daily_pss10_scores else 0.0,
        "num_events": len(daily_pss10_scores),
    }

    return {"state_delta": state_delta, "observation": observation}


def process_daily_reset(
    state: dict,
    config: dict,
    rng: np.random.Generator,
) -> dict:
    """Perform daily reset: counters, affect reset, stress decay, event clear.

    Extracted from ``Person._daily_reset()``.  Runs once per day after
    the consolidation phases.

    Args:
        state: Current AgentState dict (not modified).
        config: Config dict with optional key ``current_day`` (int).
        rng: Seeded random number generator.

    Returns:
        PhaseOutput-like dict with ``state_delta`` and ``observation``.
    """
    # ── Read state ────────────────────────────────────────────────
    affect = state.get("affect", 0.0)
    baseline_affect = state.get("baseline_affect", 0.0)
    current_stress = state.get("current_stress", 0.0)
    daily_stress_events = list(state.get("daily_stress_events", []))
    consecutive_hindrances = state.get("consecutive_hindrances", 0.0)

    # ── Read config ───────────────────────────────────────────────
    current_day = config.get("current_day", 0)

    # ── 1. Affect reset toward baseline ───────────────────────────
    stress_config = StressProcessingConfig()
    new_affect = compute_daily_affect_reset(
        current_affect=affect,
        baseline_affect=baseline_affect,
        config=stress_config,
    )

    # ── 2. Stress decay ───────────────────────────────────────────
    new_stress = compute_stress_decay(
        current_stress=current_stress,
        config=stress_config,
    )

    # ── 3. Stress summary observation ─────────────────────────────
    stress_summary = {}
    if daily_stress_events:
        stress_levels = [e.get("stress_level", 0.0) for e in daily_stress_events]
        coping_success = [e.get("coped_successfully", False) for e in daily_stress_events]
        stress_summary = {
            "avg_stress": float(np.mean(stress_levels)),
            "max_stress": float(max(stress_levels)),
            "num_events": len(daily_stress_events),
            "coping_success_rate": float(np.mean(coping_success)) if coping_success else 0.0,
        }

    # ── 4. Hindrance daily decay ──────────────────────────────────
    daily_decay_rate = 0.05
    new_consecutive_hindrances = max(0.0, consecutive_hindrances - daily_decay_rate)

    # ── Build PhaseOutput ─────────────────────────────────────────
    state_delta = {
        "daily_interactions": 0,
        "daily_support_exchanges": 0,
        "daily_stress_events": [],  # cleared for new day
        "last_daily_interactions": state.get("daily_interactions", 0),
        "last_daily_support_exchanges": state.get("daily_support_exchanges", 0),
        "last_daily_stress_events": list(state.get("daily_stress_events", [])),
        "affect": new_affect,
        "current_stress": new_stress,
        "stress_history": [],  # storage moved to model level
        "last_reset_day": current_day,
        "daily_pss10_scores": [],  # cleared for new day
        "consecutive_hindrances": new_consecutive_hindrances,
    }

    observation = {
        "stress_summary": stress_summary,
    }

    return {"state_delta": state_delta, "observation": observation}


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
                "initial_resilience_mean": cfg.get("agent", "initial_resilience_mean"),
                "initial_resilience_sd": cfg.get("agent", "initial_resilience_sd"),
                "initial_affect_mean": cfg.get("agent", "initial_affect_mean"),
                "initial_affect_sd": cfg.get("agent", "initial_affect_sd"),
                "initial_resources_mean": cfg.get("agent", "initial_resources_mean"),
                "initial_resources_sd": cfg.get("agent", "initial_resources_sd"),
                "stress_probability": cfg.get("agent", "stress_probability"),
                "coping_success_rate": cfg.get("agent", "coping_success_rate"),
                "subevents_per_day": cfg.get("agent", "subevents_per_day"),
            }

        # Random number generator for reproducible testing
        # Use model seed combined with unique_id for per-agent determinism
        # Note: Mesa Agent base class has 'rng' property, so we use '_rng'
        model_seed = getattr(model, "seed", None)
        if model_seed is not None and self.unique_id is not None:
            agent_seed = model_seed + self.unique_id
        else:
            agent_seed = model_seed
        self._rng = create_rng(agent_seed)

        # Initialize state variables using new transformation pipeline
        # Use sigmoid_transform for [0,1] bounds (resilience, baseline_resilience, resources)
        self.baseline_resilience = sigmoid_transform(
            mean=config["initial_resilience_mean"], std=config["initial_resilience_sd"], rng=self._rng
        )
        self.resilience = self.baseline_resilience
        self.resources = sigmoid_transform(
            mean=config["initial_resources_mean"], std=config["initial_resources_sd"], rng=self._rng
        )

        # Use tanh_transform for [-1,1] bounds (affect, baseline_affect)
        self.baseline_affect = tanh_transform(
            mean=config["initial_affect_mean"], std=config["initial_affect_sd"], rng=self._rng
        )
        self.affect = self.baseline_affect

        # Initialize protective factors
        self.protective_factors = {
            "social_support": 0.5,
            "family_support": 0.5,
            "formal_intervention": 0.5,
            "psychological_capital": 0.5,
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
        self.pss10 = 0  # Total PSS-10 score (0-40) — smoothed across days
        self.pss10_smoothed = 0.0  # Float smoothed value, carried across days
        self.stressed = False  # Stress classification based on PSS-10 threshold
        self.daily_pss10_scores = []  # List to collect PSS-10 scores for the current day

        # Configuration for utility functions
        self.stress_config = {
            "stress_probability": config["stress_probability"],
            "coping_success_rate": config["coping_success_rate"],
        }

        self.interaction_config = InteractionConfig()

        # Initialize PSS-10 scores using utility function
        self._initialize_pss10_scores()

        # Step 3: Initialize stress level based on the initialized PSS-10 score
        self._initialize_stress_from_pss10()

        # Initialize agent-specific volatility from Beta(1,1) distribution
        self.volatility = self._rng.beta(1, 1)

        # Track stress breach count for network adaptation
        self.stress_breach_count = 0

        # Within-day support boost from recent support exchanges
        self.support_boost = 0.0

        # Track whether network adaptation has been applied
        self._adapted_network = False

    def _initialize_pss10_scores(self):
        """
        Initialize PSS-10 scores and map to stress levels during agent creation.

        Uses utility function to generate initial PSS-10 responses and dimensions.
        """
        # Generate initial controllability and overload scores
        cfg = get_config()
        controllability_score = sigmoid_transform(
            mean=cfg.get("stress", "controllability_mean"), std=cfg.get("pss10", "controllability_sd"), rng=self._rng
        )
        overload_score = sigmoid_transform(
            mean=cfg.get("stress", "overload_mean"), std=cfg.get("pss10", "overload_sd"), rng=self._rng
        )

        # Use utility function to initialize PSS-10
        pss10_data = initialize_pss10_from_items(
            controllability_score=controllability_score, overload_score=overload_score, rng=self._rng
        )

        # Update agent state with PSS-10 data
        self.pss10_responses = pss10_data["pss10_responses"]
        self.stress_controllability = pss10_data["stress_controllability"]
        self.stress_overload = pss10_data["stress_overload"]
        self.pss10 = pss10_data["pss10_score"]
        self.stressed = pss10_data["stressed"]

    def step(self):
        """
        Execute one day of simulation using the two-loop orchestrator pattern.

        Two loops:
        1. Subevent loop: shuffled "stress" / "interact" actions using event-driven phases
        2. Daily consolidation loop: affect dynamics, resource allocation, stress buffering,
           PSS-10 consolidation, daily reset

        State flows through phases via ``_build_agent_state`` → ``_apply_delta`` → ``_write_back_state``.
        """
        # ── 1. Build agent state ────────────────────────────────────
        state = self._build_agent_state()

        # ── 2. Get shared config values ─────────────────────────────
        neighbor_affects = get_neighbor_affects(self, self.model)
        cfg = get_config()
        base_resource_cost = cfg.get("agent", "resource_cost")
        pss10_threshold = cfg.get("pss10", "threshold")
        subevents_per_day = cfg.get("agent", "subevents_per_day")

        # ── 3. Subevent loop (event-driven phases) ──────────────────
        n_subevents = sample_poisson(lam=subevents_per_day, rng=self._rng, min_value=1)
        actions = [self._rng.choice(["interact", "stress"]) for _ in range(n_subevents)]
        self._rng.shuffle(actions)

        daily_challenge_total = 0.0
        daily_hindrance_total = 0.0
        stress_event_count = 0

        for action in actions:
            # Decay support_boost at each subevent (10% per subevent)
            current_boost = state.get("support_boost", 0.0)
            state["support_boost"] = current_boost * 0.9

            if action == "stress":
                # ── Stress perception phase ─────────────────────────
                perception_config = {
                    "omega_c": cfg.get("appraisal", "omega_c"),
                    "omega_o": cfg.get("appraisal", "omega_o"),
                    "bias": cfg.get("appraisal", "bias"),
                    "gamma": cfg.get("appraisal", "gamma"),
                    "delta": 0.2,  # stress_perception default
                    "base_threshold": cfg.get("threshold", "base_threshold"),
                    "challenge_scale": cfg.get("threshold", "challenge_scale"),
                    "hindrance_scale": cfg.get("threshold", "hindrance_scale"),
                }
                perception_result = run_stress_perception(state, perception_config, self._rng)
                state = self._apply_delta(state, perception_result["state_delta"])

                # Accumulate challenge/hindrance for daily dynamics
                challenge = perception_result["state_delta"].get("challenge", 0.0)
                hindrance = perception_result["state_delta"].get("hindrance", 0.0)
                daily_challenge_total += challenge
                daily_hindrance_total += hindrance
                stress_event_count += 1

                # Track stress event for model-level reporting
                coped_successfully = True  # Non-stressed → auto-cope

                # ── Resilience activation phase (only if stressed) ──
                if state.get("is_stressed", False):
                    activation_config = {
                        "neighbor_affects": neighbor_affects,
                        "base_resource_cost": base_resource_cost,
                    }
                    activation_result = run_resilience_activation(state, activation_config, self._rng)
                    state = self._apply_delta(state, activation_result["state_delta"])
                    coped_successfully = activation_result["observation"].get("coped_successfully", False)

                    # Accumulate PSS-10 using partial moving average (Plan 011)
                    from src.python.stress_utils import append_daily_pss10_score

                    current_pss10 = state.get("pss10", 0)
                    if current_pss10 > 0:
                        daily_scores = list(state.get("daily_pss10_scores", []))
                        daily_scores = append_daily_pss10_score(daily_scores, current_pss10)
                        state["daily_pss10_scores"] = daily_scores

                # Append to daily stress events for model-level aggregation
                daily_events = list(state.get("daily_stress_events", []))
                daily_events.append(
                    {
                        "challenge": challenge,
                        "hindrance": hindrance,
                        "is_stressed": state.get("is_stressed", False),
                        "stress_level": state.get("current_stress", 0.0),
                        "coped_successfully": coped_successfully,
                    }
                )
                state["daily_stress_events"] = daily_events

            elif action == "interact":
                # ── Interaction phase ───────────────────────────────
                interaction_config = {
                    "influence_rate": cfg.get("interaction", "influence_rate"),
                    "resilience_influence": cfg.get("interaction", "resilience_influence"),
                }

                # Find a random neighbor for interaction
                if self.pos is not None:
                    try:
                        neighbors = list(self.model.grid.get_neighbors(self.pos, include_center=False))
                    except Exception:
                        neighbors = []
                else:
                    neighbors = []

                if neighbors:
                    partner = self._rng.choice(neighbors)
                    partner_state = partner._build_agent_state()

                    self_output, partner_output = phase_process_interaction(
                        state, partner_state, interaction_config, self._rng
                    )
                    # Interaction phase returns CHANGE (delta) values
                    for key, value in self_output["state_delta"].items():
                        if key in state:
                            if isinstance(state[key], dict) and isinstance(value, dict):
                                # Merge dict fields (e.g. protective_factors)
                                state[key].update(value)
                            elif isinstance(state[key], (int, float)) and isinstance(value, (int, float)):
                                # Add numeric changes
                                state[key] = state[key] + value
                            else:
                                state[key] = value
                        else:
                            state[key] = value

                    # Apply partner delta and write back
                    for key, value in partner_output["state_delta"].items():
                        if key in partner_state:
                            if isinstance(partner_state[key], dict) and isinstance(value, dict):
                                partner_state[key].update(value)
                            elif isinstance(partner_state[key], (int, float)) and isinstance(value, (int, float)):
                                partner_state[key] = partner_state[key] + value
                            else:
                                partner_state[key] = value
                        else:
                            partner_state[key] = value
                    partner._write_back_state(partner_state)

                    # Track interaction
                    state["daily_interactions"] = state.get("daily_interactions", 0) + 1
                    if self_output["observation"].get("support_occurred", False):
                        state["daily_support_exchanges"] = state.get("daily_support_exchanges", 0) + 1
                        # Accumulate within-day support boost (Plan 011)
                        current = state.get("support_boost", 0.0)
                        state["support_boost"] = min(1.0, current + 0.10)

        # Normalize daily challenge/hindrance
        if stress_event_count > 0:
            daily_challenge_total /= stress_event_count
            daily_hindrance_total /= stress_event_count

        # ── 4. Daily consolidation loop ─────────────────────────────
        # 4a. Affect dynamics (internal phase)
        affect_config = {
            "neighbor_affects": neighbor_affects,
            "daily_challenge": daily_challenge_total,
            "daily_hindrance": daily_hindrance_total,
            "stress_decay_rate": cfg.get("dynamics", "stress_decay_rate"),
        }
        affect_result = process_affect_dynamics(state, affect_config, self._rng)
        state = self._apply_delta(state, affect_result["state_delta"])

        # 4b. Resource allocation (phase module)
        resource_config = {
            "base_regeneration": cfg.get("resource", "base_regeneration"),
            "preservable_allocation_fraction": cfg.get("assumptions", "preservable_allocation_fraction"),
            "softmax_temperature": cfg.get("utility", "softmax_temperature"),
            "protective_improvement_rate": cfg.get("resource", "protective_improvement_rate"),
        }
        resource_result = run_resource_allocation(state, resource_config, self._rng)
        state = self._apply_delta(state, resource_result["state_delta"])

        # Add small noise to resources for cross-sectional variation
        current_resources = state.get("resources", 0.5)
        noise = self._rng.normal(0, 0.02) * current_resources
        state["resources"] = max(0.0, min(1.0, current_resources + noise))

        # 4c. Stress buffering (phase module)
        buffering_config = {}
        buffering_result = run_stress_buffering(state, buffering_config, self._rng)
        state = self._apply_delta(state, buffering_result["state_delta"])

        # 4d. PSS-10 consolidation (internal phase)
        pss10_config = {"pss10_threshold": pss10_threshold}
        pss10_result = process_pss10_consolidation(state, pss10_config, self._rng)
        state = self._apply_delta(state, pss10_result["state_delta"])

        # 4e. Daily reset (internal phase)
        reset_config = {"current_day": getattr(self.model, "day", 0)}
        reset_result = process_daily_reset(state, reset_config, self._rng)
        state = self._apply_delta(state, reset_result["state_delta"])

        # ── 5. Write back state ─────────────────────────────────────
        self._write_back_state(state)

    def interact(self):
        """
        Interact with a random neighbor using the phase-based process_interaction.

        Delegates to ``phase_process_interaction`` (Plan 008) and translates the
        ``PhaseOutput`` into the legacy return-dict format with change values.

        Returns:
            Dictionary with interaction results:
            - support_exchange: bool
            - affect_change: float (self affect change)
            - resilience_change: float (self resilience change)
            - resource_transfer: float
            - received_resources: float
            - partner_affect_change: float
            - partner_resilience_change: float
        """
        # Check if agent has a valid position
        if self.pos is None:
            return {
                "support_exchange": False,
                "affect_change": 0.0,
                "resilience_change": 0.0,
                "resource_transfer": 0.0,
                "received_resources": 0.0,
            }

        # Get neighbors
        try:
            neighbors = list(self.model.grid.get_neighbors(self.pos, include_center=False))
        except Exception:
            return {
                "support_exchange": False,
                "affect_change": 0.0,
                "resilience_change": 0.0,
                "resource_transfer": 0.0,
                "received_resources": 0.0,
            }

        if not neighbors:
            return {
                "support_exchange": False,
                "affect_change": 0.0,
                "resilience_change": 0.0,
                "resource_transfer": 0.0,
                "received_resources": 0.0,
            }

        # Record original state for change calculation
        original_self_affect = self.affect
        original_self_resilience = self.resilience
        original_self_resources = self.resources

        # Select random partner
        partner = self._rng.choice(neighbors)
        original_partner_affect = partner.affect
        original_partner_resilience = partner.resilience

        # Build states and call phase function
        self_state = self._build_agent_state()
        partner_state = partner._build_agent_state()

        self_output, partner_output = phase_process_interaction(
            self_state,
            partner_state,
            self.interaction_config.__dict__
            if hasattr(self.interaction_config, "__dict__")
            else dict(self.interaction_config),
            self._rng,
        )

        # Apply self delta (phase returns CHANGE values)
        self.affect = clamp(self.affect + self_output["state_delta"].get("affect", 0.0), -1.0, 1.0)
        self.resilience = clamp(self.resilience + self_output["state_delta"].get("resilience", 0.0), 0.0, 1.0)
        resource_delta = self_output["state_delta"].get("resources", 0.0)
        self.resources = clamp(self.resources + resource_delta, 0.0, 1.0)

        # Apply partner delta
        partner.affect = clamp(partner.affect + partner_output["state_delta"].get("affect", 0.0), -1.0, 1.0)
        partner.resilience = clamp(partner.resilience + partner_output["state_delta"].get("resilience", 0.0), 0.0, 1.0)
        partner.resources = clamp(partner.resources + partner_output["state_delta"].get("resources", 0.0), 0.0, 1.0)

        # Calculate changes
        self_affect_change = self.affect - original_self_affect
        self_resilience_change = self.resilience - original_self_resilience
        partner_affect_change = partner.affect - original_partner_affect
        partner_resilience_change = partner.resilience - original_partner_resilience
        received_resources = self.resources - original_self_resources
        resource_transfer = abs(received_resources)

        # Detect support exchange (threshold = 0.05)
        support_threshold = 0.05
        support_occurred = self_output["observation"].get("support_occurred", False)
        support_exchange = (
            support_occurred
            or self_affect_change > support_threshold
            or self_resilience_change > support_threshold
            or resource_transfer > support_threshold
            or partner_affect_change > support_threshold
            or partner_resilience_change > support_threshold
            or received_resources > support_threshold
        )

        # Increment daily interaction counter
        self.daily_interactions += 1

        return {
            "support_exchange": support_exchange,
            "affect_change": self_affect_change,
            "resilience_change": self_resilience_change,
            "resource_transfer": resource_transfer,
            "received_resources": received_resources,
            "partner_affect_change": partner_affect_change,
            "partner_resilience_change": partner_resilience_change,
        }

    def stressful_event(self):
        """
        Process a stressful event (Plan 008 delegation wrapper).

        Deprecated: prefer using Person.step() which handles the full phase
        pipeline.  This method is kept for backward compatibility with tests.
        Calls generate_stress_event (pachable) then delegates to the phase
        pipeline for resilience activation.

        Returns:
            Tuple of (challenge, hindrance) values for the event
        """
        # Generate event (uses module-level import that tests can patch)
        event = generate_stress_event(rng=self._rng)

        # Get configuration
        cfg = get_config()
        weights = AppraisalWeights(
            omega_c=cfg.get("appraisal", "omega_c"),
            omega_o=cfg.get("appraisal", "omega_o"),
            bias=cfg.get("appraisal", "bias"),
            gamma=cfg.get("appraisal", "gamma"),
        )
        threshold_params = ThresholdParams(
            base_threshold=cfg.get("threshold", "base_threshold"),
            challenge_scale=cfg.get("threshold", "challenge_scale"),
            hindrance_scale=cfg.get("threshold", "hindrance_scale"),
        )

        # Appraise event and check threshold
        is_stressed, challenge, hindrance = process_stress_event(event, threshold_params, weights, rng=self._rng)

        # Build state and apply stress perception delta
        state = self._build_agent_state()
        state["challenge"] = challenge
        state["hindrance"] = hindrance
        state["is_stressed"] = is_stressed
        state["event_controllability"] = event.controllability
        state["event_overload"] = event.overload

        if not is_stressed:
            # Update stress dimensions for non-stressful event
            (
                state["stress_controllability"],
                state["stress_overload"],
                state["recent_stress_intensity"],
                state["stress_momentum"],
            ) = update_stress_dimensions_from_event(
                current_controllability=state["stress_controllability"],
                current_overload=state["stress_overload"],
                challenge=challenge,
                hindrance=hindrance,
                coped_successfully=True,
                is_stressful=False,
                volatility=state["volatility"],
                recent_stress_intensity=state["recent_stress_intensity"],
                stress_momentum=state["stress_momentum"],
            )
            self._write_back_state(state)
            return challenge, hindrance

        # Stressed: run resilience activation via phase function
        neighbor_affects = get_neighbor_affects(self, self.model)
        activation_config = {
            "neighbor_affects": neighbor_affects,
            "base_resource_cost": cfg.get("agent", "resource_cost"),
        }
        activation_result = run_resilience_activation(state, activation_config, self._rng)
        state = self._apply_delta(state, activation_result["state_delta"])

        # Ensure daily_pss10_scores reflects the latest event
        pss10 = state.get("pss10", 0)
        if pss10 > 0:
            # Flush to daily_pss10_scores via state
            scores = list(state.get("daily_pss10_scores", []))
            scores.append(pss10)
            state["daily_pss10_scores"] = scores

        # Write back and track stress event
        self._write_back_state(state)
        self.daily_stress_events.append(
            {
                "challenge": challenge,
                "hindrance": hindrance,
                "is_stressed": is_stressed,
                "stress_level": state.get("current_stress", 0.0),
                "coped_successfully": activation_result["observation"].get("coped_successfully", False),
                "event_controllability": event.controllability,
                "event_overload": event.overload,
            }
        )

        return challenge, hindrance

    def _initialize_stress_from_pss10(self):
        """
        Initialize current_stress level based on PSS-10 score using utility function.

        This implements Step 3 of the PSS-10 workflow: using the initialized PSS-10 score
        to set the initial current_stress level for the agent.
        """
        self.current_stress = compute_stress_from_pss10(
            stress_controllability=self.stress_controllability, stress_overload=self.stress_overload
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
            stress_controllability=self.stress_controllability, stress_overload=self.stress_overload
        )

        # Apply exponential smoothing to create more realistic stress transitions
        # This prevents stress from changing too abruptly between days
        smoothing_factor = 0.7  # Weight for new stress level (0.3 weight for previous)
        self.current_stress = smoothing_factor * new_stress_level + (1.0 - smoothing_factor) * self.current_stress

        # Ensure stress level is in valid range
        self.current_stress = clamp(self.current_stress, 0.0, 1.0)

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
            current_affect=self.affect, baseline_affect=self.baseline_affect, config=stress_config
        )

        # Apply stress decay over time
        self.current_stress = compute_stress_decay(current_stress=self.current_stress, config=stress_config)

        # Store daily stress summary in history for analysis
        if self.daily_stress_events:
            daily_summary = {
                "day": current_day,
                "avg_stress": np.mean([event["stress_level"] for event in self.daily_stress_events]),
                "max_stress": max([event["stress_level"] for event in self.daily_stress_events]),
                "num_events": len(self.daily_stress_events),
                "coping_success_rate": np.mean([event["coped_successfully"] for event in self.daily_stress_events]),
            }
            self.stress_history.append(daily_summary)

        # Reset daily stress events for new day
        self.daily_stress_events = []

        # Apply gradual decay to consecutive hindrances over days
        if hasattr(self, "consecutive_hindrances") and self.consecutive_hindrances > 0:
            # Slowly decay consecutive hindrances over days when no new hindrance events
            daily_decay_rate = 0.05  # Small daily decay rate
            self.consecutive_hindrances = max(0.0, self.consecutive_hindrances - daily_decay_rate)

        # Reset daily PSS-10 scores for new day
        self.daily_pss10_scores = []

    # ── Orchestrator helpers (Plan 008) ─────────────────────────────

    def _build_agent_state(self) -> dict:
        """Build an AgentState dict from current ``self.*`` attributes.

        Mutable containers (dicts, lists) are shallow-copied to prevent
        accidental aliasing between the state dict and agent attributes.

        Returns:
            dict with all AgentState fields populated.
        """
        import copy

        state: dict = {
            # Core state
            "baseline_resilience": self.baseline_resilience,
            "resilience": self.resilience,
            "resources": self.resources,
            "baseline_affect": self.baseline_affect,
            "affect": self.affect,
            # Protective factors
            "protective_factors": copy.copy(self.protective_factors),
            # Stress tracking
            "current_stress": self.current_stress,
            "recent_stress_intensity": self.recent_stress_intensity,
            "stress_momentum": self.stress_momentum,
            "last_stress_update": self.last_stress_update,
            "daily_stress_events": list(self.daily_stress_events),
            "stress_history": list(self.stress_history),
            "last_reset_day": self.last_reset_day,
            "consecutive_hindrances": self.consecutive_hindrances,
            "stress_breach_count": self.stress_breach_count,
            # PSS-10 state
            "pss10_responses": copy.copy(self.pss10_responses),
            "stress_controllability": self.stress_controllability,
            "stress_overload": self.stress_overload,
            "pss10": self.pss10,
            "pss10_smoothed": self.pss10_smoothed,
            "stressed": self.stressed,
            "daily_pss10_scores": list(self.daily_pss10_scores),
            # Daily counters
            "daily_interactions": self.daily_interactions,
            "daily_support_exchanges": self.daily_support_exchanges,
            # Configuration (read-only)
            "stress_config": dict(self.stress_config),
            "interaction_config": copy.copy(self.interaction_config.__dict__)
            if hasattr(self.interaction_config, "__dict__")
            else dict(self.interaction_config),
            # Fixed traits
            "volatility": self.volatility,
            # Within-day support boost
            "support_boost": self.support_boost,
        }
        return state

    def _apply_delta(self, state: dict, delta: dict) -> dict:
        """Merge ``delta`` into a copy of ``state``.

        For dict-valued keys (e.g. ``protective_factors``), performs a
        shallow merge rather than replacement.

        Args:
            state: Current AgentState dict (not mutated).
            delta: State-delta dict from a phase function.

        Returns:
            New AgentState dict with delta applied.
        """
        import copy

        new_state = copy.copy(state)
        for key, value in delta.items():
            if key in new_state and isinstance(new_state[key], dict) and isinstance(value, dict):
                # Merge dict fields (e.g. protective_factors)
                merged = copy.copy(new_state[key])
                merged.update(value)
                new_state[key] = merged
            else:
                new_state[key] = value
        return new_state

    def _write_back_state(self, state: dict) -> None:
        """Write AgentState values back to ``self.*`` attributes.

        Only writes fields that correspond to actual Person attributes.
        Transient keys (challenge, hindrance, is_stressed,
        event_controllability, event_overload) are silently skipped.
        Bounded keys (resilience, affect, resources) are clamped.

        Args:
            state: AgentState dict containing values to write back.
        """
        # Mapping from state keys to self.* attribute names (identity for most)
        # Excludes transient keys that should not be written to self
        writable_keys = {
            "resilience",
            "affect",
            "resources",
            "current_stress",
            "recent_stress_intensity",
            "stress_momentum",
            "last_stress_update",
            "daily_stress_events",
            "stress_history",
            "last_reset_day",
            "consecutive_hindrances",
            "stress_breach_count",
            "pss10_responses",
            "stress_controllability",
            "stress_overload",
            "pss10",
            "pss10_smoothed",
            "stressed",
            "daily_pss10_scores",
            "daily_interactions",
            "daily_support_exchanges",
            "last_daily_interactions",
            "last_daily_support_exchanges",
            "last_daily_stress_events",
            "protective_factors",
            "support_boost",
        }

        # Clamp bounded keys
        if "resilience" in state:
            state["resilience"] = max(0.0, min(1.0, state["resilience"]))
        if "affect" in state:
            state["affect"] = max(-1.0, min(1.0, state["affect"]))
        if "resources" in state:
            state["resources"] = max(0.0, min(1.0, state["resources"]))

        for key in writable_keys:
            if key in state:
                setattr(self, key, state[key])
