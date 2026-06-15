"""Resource allocation phase (daily).

Regenerates resources and allocates to protective factors via softmax.

Theory-based properties:
- Resource regeneration: R near 0 → high R'; R = 1 → R' ≈ 0
- Regeneration multipliers: affect 0.50×, resilience 0.30× (hardcoded, Plan 007)
- Softmax allocation: higher e_f → higher w_f (monotonic)
- Temperature control: high T → uniform; low T → winner-take-most
- Diminishing returns: higher e_f → smaller Δe_f per unit r_f
- Conservation: sum(r_f) = available R; R ∈ [0, 1]
- No interaction variables referenced
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from numpy.random import Generator

from src.python.phases.interfaces import AgentState, PhaseOutput, PhaseFrequency
from src.python.assumption_config import get_assumptions

PHASE_FREQUENCY: PhaseFrequency = "daily"

# Factor names in canonical order — must match AgentState.protective_factors keys
_FACTORS = ["social_support", "family_support", "formal_intervention", "psychological_capital"]

# Assumption-parameterized constants (Plan 007)
_assumptions = get_assumptions()
_AFFECT_MULT_COEFFICIENT = _assumptions.resource.affect_regeneration_multiplier
_RESILIENCE_MULT_COEFFICIENT = _assumptions.resource.resilience_regeneration_multiplier
_EFFICIENCY_RETURN_FACTOR = _assumptions.resource.efficiency_return_factor
_RESILIENCE_BONUS_FACTOR = _assumptions.resource.challenge_resilience_bonus_factor


def _compute_regeneration(
    resources: float,
    affect: float,
    resilience: float,
    base_regeneration: float,
) -> float:
    """Compute resource regeneration amount.

    Formula::

        R' = base_regeneration × (1 - R) × (1 + 0.5 × max(0, A)) × (1 + 0.3 × resilience)

    Args:
        resources: Current resource level in [0, 1].
        affect: Current affect level in [-1, 1].
        resilience: Current resilience level in [0, 1].
        base_regeneration: Base regeneration rate.

    Returns:
        Regeneration amount (non-negative).
    """
    affect_mult = 1.0 + _AFFECT_MULT_COEFFICIENT * max(0.0, affect)
    resil_mult = 1.0 + _RESILIENCE_MULT_COEFFICIENT * resilience
    return base_regeneration * (1.0 - resources) * affect_mult * resil_mult


def _softmax_weights(efficacies: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax with temperature.

    Args:
        efficacies: Array of efficacy values.
        temperature: Temperature parameter (higher = more uniform).

    Returns:
        Normalised probability vector.
    """
    if temperature == 0:
        # One-hot: the max efficacy gets all weight
        idx = int(np.argmax(efficacies))
        w = np.zeros_like(efficacies)
        w[idx] = 1.0
        return w
    logits = efficacies / temperature
    # Numerical stability: subtract max
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def _allocate_resources(
    available: float,
    efficacies: Dict[str, float],
    temperature: float,
) -> Dict[str, float]:
    """Allocate ``available`` resources across protective factors via softmax.

    Args:
        available: Total resources to allocate.
        efficacies: Current PF efficacy levels.
        temperature: Softmax temperature.

    Returns:
        Mapping from factor name to allocated resource amount.
    """
    eff_array = np.array([efficacies[f] for f in _FACTORS])
    weights = _softmax_weights(eff_array, temperature)
    return {f: float(available * w) for f, w in zip(_FACTORS, weights)}


def _update_efficacies(
    efficacies: Dict[str, float],
    allocations: Dict[str, float],
    resilience: float,
    improvement_rate: float,
) -> Dict[str, float]:
    """Update PF efficacy levels with diminishing returns.

    Formula::

        Δe_f = r_f × improvement_rate × (1 - e_f) × (1 + 0.2 × resilience) + r_f × 0.05
        e_f' = min(1.0, e_f + Δe_f)

    Args:
        efficacies: Current PF efficacy levels.
        allocations: Resources allocated to each PF.
        resilience: Current resilience (boosts efficiency gain).
        improvement_rate: Base improvement rate.

    Returns:
        Updated PF efficacy levels.
    """
    updated = dict(efficacies)
    efficiency_gain = 1.0 + _RESILIENCE_BONUS_FACTOR * resilience
    for f in _FACTORS:
        r_f = allocations.get(f, 0.0)
        if r_f > 0:
            current = updated[f]
            investment_effectiveness = 1.0 - current
            efficiency_return = r_f * _EFFICIENCY_RETURN_FACTOR
            delta = (r_f * improvement_rate * investment_effectiveness * efficiency_gain) + efficiency_return
            updated[f] = min(1.0, current + delta)
    return updated


def run_phase(
    state: AgentState,
    config: Dict[str, Any],
    rng: Generator,
) -> PhaseOutput:
    """Regenerate resources → softmax allocation → PF efficacy updates → depletion.

    Pure function: all inputs via ``state`` / ``config``; all outputs via
    ``PhaseOutput``.

    Args:
        state: Current agent state.  Required fields:
            ``resources``, ``affect``, ``resilience``, ``protective_factors``.
        config: Phase configuration.  Expected keys:
            ``base_regeneration`` (float, default 0.1),
            ``softmax_temperature`` (float, default 1.0),
            ``protective_improvement_rate`` (float, default 0.1).
        rng: Seeded random number generator (not used in this phase).

    Returns:
        PhaseOutput with:
        - state_delta: ``resources`` (updated), ``protective_factors`` (updated).
        - observation: ``regeneration_amount``, ``allocation_weights``,
          ``allocated_resources``, ``efficacies_before``, ``efficacies_after``.
    """
    # Unpack state
    resources = state.get("resources", 0.5)
    affect = state.get("affect", 0.0)
    resilience = state.get("resilience", 0.5)
    efficacies_before: Dict[str, float] = dict(
        state.get(
            "protective_factors",
            {f: 0.5 for f in _FACTORS},
        )
    )

    # Unpack config
    base_regeneration = config.get("base_regeneration", 0.1)
    temperature = config.get("softmax_temperature", 1.0)
    improvement_rate = config.get("protective_improvement_rate", 0.1)
    # Fraction of regenerated resources to preserve (not allocate)
    preservable_fraction = config.get("preservable_allocation_fraction", 0.1)

    # ── 1. Resource regeneration ────────────────────────────────────
    regeneration = _compute_regeneration(resources, affect, resilience, base_regeneration)

    # Total resources available after regeneration
    available_for_allocation = resources + regeneration

    # Preserve a fraction of resources (prevent depletion)
    preserved = available_for_allocation * preservable_fraction
    spendable = available_for_allocation - preserved

    # ── 2. Softmax allocation ───────────────────────────────────────
    allocations = _allocate_resources(spendable, efficacies_before, temperature)
    total_allocated = sum(allocations.values())

    # ── 3. PF efficacy updates (diminishing returns) ────────────────
    efficacies_after = _update_efficacies(efficacies_before, allocations, resilience, improvement_rate)

    # ── 4. Remaining resources (preserved + unspent) ────────────────
    new_resources = min(1.0, max(0.0, preserved + (spendable - total_allocated)))

    # ── Build PhaseOutput ───────────────────────────────────────────
    state_delta: Dict[str, Any] = {
        "resources": new_resources,
        "protective_factors": efficacies_after,
    }

    observation: Dict[str, Any] = {
        "regeneration_amount": regeneration,
        "allocation_weights": {f: float(allocations[f] / spendable) if spendable > 0 else 0.25 for f in _FACTORS},
        "spendable_fraction": float(spendable / available_for_allocation) if available_for_allocation > 0 else 0.0,
        "allocated_resources": {f: float(allocations[f]) for f in _FACTORS},
        "efficacies_before": dict(efficacies_before),
        "efficacies_after": dict(efficacies_after),
    }

    return PhaseOutput(state_delta=state_delta, observation=observation)
