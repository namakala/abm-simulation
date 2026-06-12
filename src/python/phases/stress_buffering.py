"""Stress buffering phase (daily).

Applies protective-factor resilience boost and resource mediation of stress buffering.

Theory (Baron & Kenny mediation):
- a-path: stress -> resources (negative: stress depletes resources)
- b-path: resources -> buffering (positive: resources enhance coping)
- c'-path: stress -> buffering | resources (negative: direct residual stress effect)
- Indirect effect a * b != 0 confirms mediation
- Social support as parallel mediator

Extracted from agent.py:288-293 and resource_utils.py get_resilience_boost_from_protective_factors.

Runs once after resource allocation (Plan 004).
Does NOT include affect dynamics, PSS-10 consolidation, or daily reset.
"""

from __future__ import annotations

from typing import Any, Dict

from numpy.random import Generator

from src.python.phases.interfaces import AgentState, PhaseOutput, PhaseFrequency

PHASE_FREQUENCY: PhaseFrequency = "daily"

# Hardcoded constants (Plan 007 externalises these)
_DEFAULT_BOOST_RATE = 0.1
_DEFAULT_A_COEFFICIENT = -0.3  # stress -> resources
_DEFAULT_B_COEFFICIENT = 0.5  # resources -> buffering
_DEFAULT_C_PRIME_COEFFICIENT = -0.2  # stress -> buffering | resources
_DEFAULT_SOCIAL_STRESS_PATH = -0.2
_DEFAULT_SOCIAL_BUFFERING_PATH = 0.4

# Factor names in canonical order
_FACTORS = ["social_support", "family_support", "formal_intervention", "psychological_capital"]


def run_phase(
    state: AgentState,
    config: Dict[str, Any],
    rng: Generator,
) -> PhaseOutput:
    """Apply PF resilience boost and resource mediation of stress buffering.

    Two mechanisms:

    1. **PF boost**: protective factors boost resilience toward baseline
       when current resilience is below baseline. The boost is proportional
       to each factor's efficacy, the gap ``(baseline - current)``, and
       the ``boost_rate``. It never exceeds the gap.

    2. **Resource mediation**: stress depletes resources (a-path), resources
       enhance buffering (b-path), and stress has a residual negative direct
       effect on buffering controlling for resources (c'-path).

    Args:
        state: Current agent state. Required fields:
            ``resilience``, ``baseline_resilience``, ``protective_factors``,
            ``current_stress``, ``resources``.
        config: Phase configuration. Expected keys:
            ``boost_rate`` (float, default 0.1),
            ``a_coefficient`` (float, default -0.3),
            ``b_coefficient`` (float, default 0.5),
            ``c_prime_coefficient`` (float, default -0.2),
            ``social_stress_path`` (float, default -0.2),
            ``social_buffering_path`` (float, default 0.4).
        rng: Seeded random number generator (not used in this phase).

    Returns:
        PhaseOutput with:
        - state_delta: ``resilience`` (boosted toward baseline),
          ``resources`` (depleted by stress via a-path).
        - observation: ``pf_boost``, ``buffering_strength``,
          ``a_coefficient``, ``b_coefficient``, ``c_prime_coefficient``,
          ``indirect_effect``, ``social_support_mediation``.
    """
    # ── Unpack state ────────────────────────────────────────────────
    protective_factors: Dict[str, float] = dict(state.get("protective_factors", {f: 0.5 for f in _FACTORS}))
    baseline_resilience = state.get("baseline_resilience", 0.5)
    current_resilience = state.get("resilience", 0.5)
    current_stress = state.get("current_stress", 0.0)
    resources = state.get("resources", 0.5)

    # ── Unpack config ───────────────────────────────────────────────
    boost_rate = config.get("boost_rate", _DEFAULT_BOOST_RATE)
    a_coefficient = config.get("a_coefficient", _DEFAULT_A_COEFFICIENT)
    b_coefficient = config.get("b_coefficient", _DEFAULT_B_COEFFICIENT)
    c_prime_coefficient = config.get("c_prime_coefficient", _DEFAULT_C_PRIME_COEFFICIENT)
    social_buffering_path = config.get("social_buffering_path", _DEFAULT_SOCIAL_BUFFERING_PATH)

    # ── Mechanism 1: PF boost to resilience ─────────────────────────
    # Boost is larger when resilience is far below baseline.
    # Never exceeds (baseline - current) — can't overshoot baseline.
    resilience_need = max(0.0, baseline_resilience - current_resilience)

    pf_boost = 0.0
    for factor in _FACTORS:
        efficacy = protective_factors.get(factor, 0.0)
        if efficacy > 0.0:
            pf_boost += efficacy * resilience_need * boost_rate

    pf_boost = min(pf_boost, resilience_need)
    new_resilience = min(1.0, current_resilience + pf_boost)

    # ── Mechanism 2: Resource mediation of stress buffering ─────────
    # a-path: stress -> resources (negative coefficient → depletion)
    resource_depletion = a_coefficient * current_stress
    new_resources = max(0.0, min(1.0, resources + resource_depletion))

    # b-path + c'-path: buffering from resources and residual stress effect
    buffering_strength = max(
        0.0,
        b_coefficient * new_resources + c_prime_coefficient * current_stress,
    )

    # Indirect effect (a * b) — Baron & Kenny mediation
    indirect_effect = a_coefficient * b_coefficient

    # Social support parallel mediation
    social_support_efficacy = protective_factors.get("social_support", 0.0)
    social_support_mediation = social_support_efficacy * social_buffering_path

    # ── Build PhaseOutput ───────────────────────────────────────────
    state_delta: Dict[str, Any] = {
        "resilience": new_resilience,
        "resources": new_resources,
    }

    observation: Dict[str, Any] = {
        "pf_boost": pf_boost,
        "buffering_strength": buffering_strength,
        "a_coefficient": a_coefficient,
        "b_coefficient": b_coefficient,
        "c_prime_coefficient": c_prime_coefficient,
        "indirect_effect": indirect_effect,
        "social_support_mediation": social_support_mediation,
    }

    return PhaseOutput(state_delta=state_delta, observation=observation)
