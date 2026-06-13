"""Resilience activation phase (event-driven).

Determines coping outcome and updates resilience after a stress event.
Implements the full coping pipeline: neighbor influence → coping probability
→ coping outcome → affect/resilience/stress changes → stress dimension update
→ PSS-10 generation → resource cost → PF allocation → hindrance tracking.
"""

from typing import Any, Dict

from numpy.random import Generator

from src.python.phases.interfaces import AgentState, PhaseOutput, PhaseFrequency
from src.python.affect_utils import (
    StressProcessingConfig,
    compute_coping_probability,
    compute_challenge_hindrance_resilience_effect,
    determine_coping_outcome_and_psychological_impact,
)
from src.python.resource_utils import (
    ResourceOptimizationConfig,
    compute_resilience_optimized_resource_cost,
    compute_resource_depletion_with_resilience,
    allocate_protective_factors,
    update_protective_factors_with_allocation,
)
from src.python.stress_utils import (
    update_stress_dimensions_from_event,
    generate_pss10_from_stress_dimensions,
    update_stress_dimensions_from_pss10_feedback,
    validate_theoretical_correlations,
)
from src.python.math_utils import clamp

PHASE_FREQUENCY: PhaseFrequency = "event_driven"

# Hardcoded constants (Plan 007 will externalise these)
_RESOURCE_REWARD_MULTIPLIER = 0.75
_RESOURCE_PENALTY_MULTIPLIER = 0.10
_PF_ALLOCATION_FRACTION = 0.30


def run_phase(
    state: AgentState,
    config: Dict[str, Any],
    rng: Generator,
) -> PhaseOutput:
    """Activate resilience mechanisms in response to a stress event.

    Pure function: all inputs via ``state`` and ``config``; all outputs via
    ``PhaseOutput``.  No side effects.

    Args:
        state: Current agent state (must include ``challenge``, ``hindrance``).
        config: Phase-specific configuration.  Expected keys:
            ``neighbor_affects`` (list[float]),
            ``base_resource_cost`` (float),
            ``event_controllability`` (float),
            ``event_overload`` (float).
        rng: Seeded random number generator for reproducible outcomes.

    Returns:
        PhaseOutput with:
        - state_delta: new values for 12 keys (affect, resilience,
          current_stress, stress_controllability, stress_overload, resources,
          protective_factors, consecutive_hindrances, stress_breach_count,
          pss10, pss10_responses, stressed).
        - observation: coped_successfully, coping_probability,
          resilience_effect, delta_stress, delta_affect, resource_cost,
          resource_reward (if success) / resource_penalty (if failure).
    """
    # ── Unpack state ────────────────────────────────────────────────
    current_affect = state["affect"]
    current_resilience = state["resilience"]
    current_stress = state["current_stress"]
    challenge = state["challenge"]
    hindrance = state["hindrance"]
    baseline_resilience = state.get("baseline_resilience", 0.5)

    stress_controllability = state.get("stress_controllability", 0.5)
    stress_overload = state.get("stress_overload", 0.5)
    volatility = state.get("volatility", 0.3)
    recent_stress_intensity = state.get("recent_stress_intensity", 0.0)
    stress_momentum = state.get("stress_momentum", 0.0)
    current_resources = state.get("resources", 0.5)
    protective_factors = state.get(
        "protective_factors",
        {
            "social_support": 0.5,
            "family_support": 0.5,
            "formal_intervention": 0.5,
            "psychological_capital": 0.5,
        },
    )
    consecutive_hindrances = state.get("consecutive_hindrances", 0.0)

    # ── Unpack config ───────────────────────────────────────────────
    neighbor_affects = config.get("neighbor_affects", [])
    base_resource_cost = config.get("base_resource_cost", 0.1)

    # ── STEP 1: Coping outcome ──────────────────────────────────────
    stress_config = StressProcessingConfig()
    new_affect, new_resilience, new_stress, coped_successfully = determine_coping_outcome_and_psychological_impact(
        current_affect=current_affect,
        current_resilience=current_resilience,
        current_stress=current_stress,
        challenge=challenge,
        hindrance=hindrance,
        neighbor_affects=neighbor_affects,
        rng=rng,
        config=stress_config,
    )

    # Compute values needed for observation
    coping_probability = compute_coping_probability(
        challenge=challenge,
        hindrance=hindrance,
        neighbor_affects=neighbor_affects,
        current_resilience=current_resilience,
        config=stress_config,
    )
    resilience_effect = compute_challenge_hindrance_resilience_effect(
        challenge=challenge, hindrance=hindrance, coped_successfully=coped_successfully
    )

    # ── STEP 2: Update stress dimensions from event ─────────────────
    updated_controllability, updated_overload, updated_intensity, updated_momentum = (
        update_stress_dimensions_from_event(
            current_controllability=stress_controllability,
            current_overload=stress_overload,
            challenge=challenge,
            hindrance=hindrance,
            coped_successfully=coped_successfully,
            is_stressful=True,
            volatility=volatility,
            recent_stress_intensity=recent_stress_intensity,
            stress_momentum=stress_momentum,
        )
    )

    # ── STEP 3: Generate PSS-10 from updated stress dimensions ──────
    pss10_data = generate_pss10_from_stress_dimensions(
        stress_controllability=updated_controllability,
        stress_overload=updated_overload,
        recent_stress_intensity=updated_intensity,
        stress_momentum=updated_momentum,
        affect=new_affect,
        resources=current_resources,
        rng=rng,
    )
    new_pss10_responses = pss10_data["pss10_responses"]
    new_pss10 = pss10_data["pss10_score"]
    new_stressed = pss10_data["stressed"]

    # ── STEP 4: Update stress dimensions from PSS-10 feedback ───────
    final_controllability, final_overload = update_stress_dimensions_from_pss10_feedback(
        current_controllability=updated_controllability,
        current_overload=updated_overload,
        pss10_responses=new_pss10_responses,
        current_resources=current_resources,
    )

    # ── STEP 5: Validate theoretical correlations ───────────────────
    validate_theoretical_correlations(
        challenge=challenge,
        hindrance=hindrance,
        coped_successfully=coped_successfully,
        stress_controllability=final_controllability,
        stress_overload=final_overload,
        pss10_score=new_pss10,
        current_stress=new_stress,
        pss10_responses=new_pss10_responses,
    )

    # ── STEP 6: Resource cost and depletion ─────────────────────────
    resource_config = ResourceOptimizationConfig()
    optimized_cost = compute_resilience_optimized_resource_cost(
        base_cost=base_resource_cost,
        current_resilience=new_resilience,
        challenge=challenge,
        hindrance=hindrance,
        config=resource_config,
    )
    new_resources = compute_resource_depletion_with_resilience(
        current_resources=current_resources,
        cost=optimized_cost,
        current_resilience=new_resilience,
        coping_successful=coped_successfully,
        is_stressed=True,
        config=resource_config,
    )

    # ── STEP 7: Track consecutive hindrances ────────────────────────
    new_consecutive_hindrances = consecutive_hindrances + 1.0 if hindrance > challenge else 0.0

    # ── STEP 8: Increment stress breach count ───────────────────────
    new_stress_breach_count = state.get("stress_breach_count", 0) + 1

    # ── STEP 9: Resource reward / penalty + PF allocation ───────────
    new_protective_factors = dict(protective_factors)
    resource_reward: float | None = None
    resource_penalty: float | None = None

    if coped_successfully:
        resource_reward = base_resource_cost * _RESOURCE_REWARD_MULTIPLIER
        new_resources = clamp(new_resources + resource_reward, 0.0, 1.0)

        allocations = allocate_protective_factors(
            available_resources=new_resources * _PF_ALLOCATION_FRACTION,
            current_resilience=new_resilience,
            baseline_resilience=baseline_resilience,
            protective_factors=protective_factors,
            rng=rng,
        )

        new_protective_factors = update_protective_factors_with_allocation(
            protective_factors=protective_factors,
            allocations=allocations,
            current_resilience=new_resilience,
        )

        total_allocated = sum(allocations.values())
        new_resources = clamp(new_resources - total_allocated, 0.0, 1.0)
    else:
        resource_penalty = base_resource_cost * _RESOURCE_PENALTY_MULTIPLIER
        new_resources = clamp(new_resources - resource_penalty, 0.0, 1.0)

    # ── Build state_delta ───────────────────────────────────────────
    state_delta: Dict[str, Any] = {
        "affect": new_affect,
        "resilience": new_resilience,
        "current_stress": new_stress,
        "stress_controllability": final_controllability,
        "stress_overload": final_overload,
        "resources": new_resources,
        "protective_factors": new_protective_factors,
        "consecutive_hindrances": new_consecutive_hindrances,
        "stress_breach_count": new_stress_breach_count,
        "pss10": new_pss10,
        "pss10_responses": new_pss10_responses,
        "stressed": new_stressed,
    }

    # ── Build observation ───────────────────────────────────────────
    observation: Dict[str, Any] = {
        "coped_successfully": coped_successfully,
        "coping_probability": coping_probability,
        "resilience_effect": resilience_effect,
        "delta_stress": new_stress - current_stress,
        "delta_affect": new_affect - current_affect,
        "resource_cost": optimized_cost,
    }
    if resource_reward is not None:
        observation["resource_reward"] = resource_reward
    if resource_penalty is not None:
        observation["resource_penalty"] = resource_penalty

    return PhaseOutput(state_delta=state_delta, observation=observation)
