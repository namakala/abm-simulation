"""Stress perception phase (event-driven).

Appraises incoming stressors and updates stress dimensions.

Called per stress event (not once per day).  Non-stressed events return
early after updating stress dimensions with is_stressful=False (minimal
no-op update).
"""

from typing import Any, Dict

from numpy.random import Generator

from src.python.phases.interfaces import AgentState, PhaseOutput, PhaseFrequency
from src.python.stress_utils import (
    generate_stress_event,
    apply_weights,
    compute_appraised_stress,
    evaluate_stress_threshold,
    update_stress_dimensions_from_event,
    AppraisalWeights,
    ThresholdParams,
)

PHASE_FREQUENCY: PhaseFrequency = "event_driven"


def run_phase(
    state: AgentState,
    config: Dict[str, Any],
    rng: Generator,
) -> PhaseOutput:
    """Process a stress perception event.

    Generates a stress event, appraises it (challenge/hindrance), evaluates
    whether it exceeds the stress threshold, and updates stress dimensions.

    Args:
        state: Current agent state.  Requires fields:
            stress_controllability, stress_overload, volatility,
            recent_stress_intensity, stress_momentum.
        config: Phase configuration with keys:
            omega_c, omega_o, bias, gamma, delta,
            base_threshold, challenge_scale, hindrance_scale.
        rng: Seeded random number generator.

    Returns:
        PhaseOutput with:
            state_delta: challenge, hindrance, is_stressed, event attrs,
                         updated stress dimensions.
            observation: event attrs, appraisal values, threshold values.
    """
    # ── 1. Generate stress event ───────────────────────────────────
    event = generate_stress_event(rng)

    # ── 2. Build appraisal weights from config ─────────────────────
    weights = AppraisalWeights(
        omega_c=config.get("omega_c", 1.0),
        omega_o=config.get("omega_o", 1.0),
        bias=config.get("bias", 0.0),
        gamma=config.get("gamma", 6.0),
    )

    # ── 3. Compute challenge / hindrance ───────────────────────────
    challenge, hindrance = apply_weights(event, weights)

    # ── 4. Compute appraised stress load ───────────────────────────
    appraised_stress = compute_appraised_stress(event, challenge, hindrance, {"delta": config.get("delta", 0.2)})

    # ── 5. Evaluate stress threshold ───────────────────────────────
    threshold_params = ThresholdParams(
        base_threshold=config.get("base_threshold", 0.5),
        challenge_scale=config.get("challenge_scale", 0.15),
        hindrance_scale=config.get("hindrance_scale", 0.25),
    )
    is_stressed = evaluate_stress_threshold(appraised_stress, challenge, hindrance, threshold_params)

    # Compute effective threshold for observation
    effective_threshold = (
        threshold_params.base_threshold
        + threshold_params.challenge_scale * challenge
        - threshold_params.hindrance_scale * hindrance
    )
    effective_threshold = max(0.0, min(1.0, float(effective_threshold)))

    # ── 6. Update stress dimensions ─────────────────────────────────
    volatility = state.get("volatility", 0.5)
    current_controllability = state.get("stress_controllability", 0.5)
    current_overload = state.get("stress_overload", 0.5)
    recent_stress_intensity = state.get("recent_stress_intensity", 0.0)
    stress_momentum = state.get("stress_momentum", 0.0)

    updated_controllability, updated_overload, new_intensity, new_momentum = update_stress_dimensions_from_event(
        current_controllability=current_controllability,
        current_overload=current_overload,
        challenge=challenge,
        hindrance=hindrance,
        coped_successfully=True,
        is_stressful=is_stressed,
        volatility=volatility,
        recent_stress_intensity=recent_stress_intensity,
        stress_momentum=stress_momentum,
    )

    # ── 7. Build PhaseOutput ────────────────────────────────────────
    state_delta: Dict[str, Any] = {
        "challenge": challenge,
        "hindrance": hindrance,
        "is_stressed": is_stressed,
        "event_controllability": event.controllability,
        "event_overload": event.overload,
        "stress_controllability": updated_controllability,
        "stress_overload": updated_overload,
        "recent_stress_intensity": new_intensity,
        "stress_momentum": new_momentum,
    }

    observation: Dict[str, Any] = {
        "event_controllability": event.controllability,
        "event_overload": event.overload,
        "challenge": challenge,
        "hindrance": hindrance,
        "appraised_stress": appraised_stress,
        "effective_threshold": effective_threshold,
        "is_stressed": is_stressed,
    }

    return PhaseOutput(state_delta=state_delta, observation=observation)
