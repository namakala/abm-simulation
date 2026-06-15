"""
Assumption Parameterization: Externalized config for all ~84 hardcoded constants.

Provides a hierarchical dataclass tree matching the specification in
docs/plans/007-assumption-parameterization.md. Every constant becomes an
ASSUMPTION_* env var reference, loaded at runtime with defaults that
preserve current behavioral equivalence.

Config hierarchy:
  AssumptionConfig
    ├── coping   (AssumptionCopingConfig)    — 15 fields
    ├── stress   (AssumptionStressConfig)    — 21 fields
    ├── resource (AssumptionResourceConfig)  — 17 fields
    ├── social   (AssumptionSocialConfig)    —  3 fields
    └── buffering (AssumptionBufferingConfig) —  6 fields

Total: 62 mechanism-level constants, plus 12 phase-level constants in
the phase modules themselves. ~74 total when combined.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# =========================================================================
# Helper: load a float env var with a default
# =========================================================================


def _env_float(key: str, default: float) -> float:
    """Load a float from an env var, falling back to *default*."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        return default


# =========================================================================
# Sub-group configs
# =========================================================================


@dataclass(frozen=True)
class AssumptionCopingConfig:
    """Coping-mechanism constants (15 fields).

    Controls resource rewards/penalties, affect/resilience changes from
    coping outcomes, protective factor allocation fraction, and
    challenge/hindrance resilience effects.
    """

    resource_reward: float = field(default_factory=lambda: _env_float("ASSUMPTION_RESOURCE_REWARD", 0.75))
    resource_penalty: float = field(default_factory=lambda: _env_float("ASSUMPTION_RESOURCE_PENALTY", 0.10))
    pf_allocation_fraction: float = field(default_factory=lambda: _env_float("ASSUMPTION_PF_ALLOCATION_FRACTION", 0.30))
    affect_improvement_scale: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_AFFECT_IMPROVEMENT_SCALE", 0.2)
    )
    affect_deterioration_scale: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_AFFECT_DETERIORATION_SCALE", 0.4)
    )
    resilience_improvement_scale: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_RESILIENCE_IMPROVEMENT_SCALE", 0.1)
    )
    resilience_deterioration_scale: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_RESILIENCE_DETERIORATION_SCALE", 0.2)
    )
    challenge_success_resilience: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_CHALLENGE_SUCCESS_RESILIENCE", 0.3)
    )
    challenge_failure_resilience: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_CHALLENGE_FAILURE_RESILIENCE", -0.1)
    )
    hindrance_success_resilience: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_HINDRANCE_SUCCESS_RESILIENCE", 0.1)
    )
    hindrance_failure_resilience: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_HINDRANCE_FAILURE_RESILIENCE", -0.4)
    )
    success_stress_reduction: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_SUCCESS_STRESS_REDUCTION", 0.2)
    )
    failure_stress_increase: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_FAILURE_STRESS_INCREASE", 0.3)
    )
    success_affect_change: float = field(default_factory=lambda: _env_float("ASSUMPTION_SUCCESS_AFFECT_CHANGE", 0.1))
    failure_affect_change: float = field(default_factory=lambda: _env_float("ASSUMPTION_FAILURE_AFFECT_CHANGE", -0.2))


@dataclass(frozen=True)
class AssumptionStressConfig:
    """Stress-dynamics constants (21 fields).

    Controls controllability/overload weights for challenge/hindrance,
    homeostasis rates, event intensity computation, momentum dynamics,
    and PSS-10 estimation parameters.
    """

    controllability_challenge_weight: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_CONTROLLABILITY_CHALLENGE_WEIGHT", 0.10)
    )
    controllability_hindrance_weight: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_CONTROLLABILITY_HINDRANCE_WEIGHT", 0.05)
    )
    overload_challenge_weight: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_OVERLOAD_CHALLENGE_WEIGHT", 0.05)
    )
    overload_hindrance_weight: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_OVERLOAD_HINDRANCE_WEIGHT", 0.10)
    )
    baseline_controllability: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_BASELINE_CONTROLLABILITY", 0.5)
    )
    baseline_overload: float = field(default_factory=lambda: _env_float("ASSUMPTION_BASELINE_OVERLOAD", 0.5))
    controllability_homeostasis_rate: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_CONTROLLABILITY_HOMEOSTASIS_RATE", 0.05)
    )
    overload_homeostasis_rate: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_OVERLOAD_HOMEOSTASIS_RATE", 0.05)
    )
    event_intensity_challenge_weight: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_EVENT_INTENSITY_CHALLENGE_WEIGHT", 0.7)
    )
    event_intensity_hindrance_weight: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_EVENT_INTENSITY_HINDRANCE_WEIGHT", 1.3)
    )
    failed_coping_intensity_multiplier: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_FAILED_COPING_INTENSITY_MULTIPLIER", 1.5)
    )
    stress_intensity_decay_rate: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_STRESS_INTENSITY_DECAY_RATE", 0.8)
    )
    new_intensity_weight: float = field(default_factory=lambda: _env_float("ASSUMPTION_NEW_INTENSITY_WEIGHT", 0.2))
    momentum_increase_rate: float = field(default_factory=lambda: _env_float("ASSUMPTION_MOMENTUM_INCREASE_RATE", 0.1))
    momentum_decrease_rate: float = field(default_factory=lambda: _env_float("ASSUMPTION_MOMENTUM_DECREASE_RATE", 0.05))
    momentum_zero_threshold: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_MOMENTUM_ZERO_THRESHOLD", 0.01)
    )
    momentum_decay_factor: float = field(default_factory=lambda: _env_float("ASSUMPTION_MOMENTUM_DECAY_FACTOR", 0.9))
    pss10_estimation_base: int = field(
        default_factory=lambda: int(_env_float("ASSUMPTION_PSS10_ESTIMATION_BASE", 10.0))
    )
    pss10_controllability_max_effect: int = field(
        default_factory=lambda: int(_env_float("ASSUMPTION_PSS10_CONTROLLABILITY_MAX_EFFECT", 8.0))
    )
    pss10_overload_max_effect: int = field(
        default_factory=lambda: int(_env_float("ASSUMPTION_PSS10_OVERLOAD_MAX_EFFECT", 12.0))
    )
    pss10_estimation_variance: int = field(
        default_factory=lambda: int(_env_float("ASSUMPTION_PSS10_ESTIMATION_VARIANCE", 3.0))
    )


@dataclass(frozen=True)
class AssumptionResourceConfig:
    """Resource-dynamics constants (17 fields).

    Controls resilience efficiency, resource thresholds, cost floors,
    coping penalties, efficiency gains, social and stress effects on
    resource dynamics, and regeneration multipliers.
    """

    resilience_efficiency_factor: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_RESILIENCE_EFFICIENCY_FACTOR", 0.3)
    )
    min_resource_threshold: float = field(default_factory=lambda: _env_float("ASSUMPTION_MIN_RESOURCE_THRESHOLD", 0.05))
    coping_difficulty_scale: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_COPING_DIFFICULTY_SCALE", 0.5)
    )
    min_cost_floor: float = field(default_factory=lambda: _env_float("ASSUMPTION_MIN_COST_FLOOR", 0.3))
    failed_coping_cost_penalty: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_FAILED_COPING_COST_PENALTY", 1.3)
    )
    max_efficiency_gain: float = field(default_factory=lambda: _env_float("ASSUMPTION_MAX_EFFICIENCY_GAIN", 0.5))
    social_resilience_boost_factor: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_SOCIAL_RESILIENCE_BOOST_FACTOR", 0.1)
    )
    support_exchange_benefit_weight: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_SUPPORT_EXCHANGE_BENEFIT_WEIGHT", 0.2)
    )
    challenge_resilience_bonus_factor: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_CHALLENGE_RESILIENCE_BONUS_FACTOR", 0.2)
    )
    hindrance_resilience_bonus_factor: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_HINDRANCE_RESILIENCE_BONUS_FACTOR", 0.1)
    )
    overload_allocation_penalty_rate: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_OVERLOAD_ALLOCATION_PENALTY_RATE", 0.1)
    )
    stress_improvement_effectiveness: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_STRESS_IMPROVEMENT_EFFECTIVENESS", 0.1)
    )
    social_resource_boost_factor: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_SOCIAL_RESOURCE_BOOST_FACTOR", 0.1)
    )
    preservable_allocation_fraction: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_PRESERVABLE_ALLOCATION_FRACTION", 0.5)
    )
    social_support_allocation_boost: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_SOCIAL_SUPPORT_ALLOCATION_BOOST", 0.3)
    )
    # Social exchange parameters
    min_resource_threshold_for_sharing: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_MIN_RESOURCE_FOR_SHARING", 0.2)
    )
    exchange_amount_reduction_factor: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_EXCHANGE_AMOUNT_REDUCTION", 0.5)
    )
    affect_regeneration_multiplier: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_AFFECT_REGENERATION_MULTIPLIER", 0.2)
    )
    resilience_regeneration_multiplier: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_RESILIENCE_REGENERATION_MULTIPLIER", 0.3)
    )
    # Phase-level constants (beyond the original spec 62)
    efficiency_return_factor: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_EFFICIENCY_RETURN_FACTOR", 0.05)
    )
    # Resource allocation optimisation factors
    resilience_focus_factor: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_RESILIENCE_FOCUS_FACTOR", 0.5)
    )
    resilience_allocation_bonus: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_RESILIENCE_ALLOCATION_BONUS", 0.2)
    )
    giver_resilience_sharing_factor: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_GIVER_RESILIENCE_SHARING_FACTOR", 0.2)
    )
    receiver_efficiency_bonus_factor: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_RECEIVER_EFFICIENCY_BONUS_FACTOR", 0.15)
    )
    buffering_boost_rate: float = field(default_factory=lambda: _env_float("ASSUMPTION_BUFFERING_BOOST_RATE", 0.1))
    buffering_a_coefficient: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_BUFFERING_A_COEFFICIENT", -0.03)
    )
    buffering_b_coefficient: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_BUFFERING_B_COEFFICIENT", 0.5)
    )
    buffering_c_prime_coefficient: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_BUFFERING_C_PRIME_COEFFICIENT", -0.2)
    )


@dataclass(frozen=True)
class AssumptionSocialConfig:
    """Social-interaction constants (3 fields).

    Controls support exchange threshold, social support probability,
    and exchange boost factor.
    """

    support_exchange_threshold: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_SUPPORT_EXCHANGE_THRESHOLD", 0.05)
    )
    social_support_probability: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_SOCIAL_SUPPORT_PROBABILITY", 0.3)
    )
    social_support_exchange_boost: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_SOCIAL_SUPPORT_EXCHANGE_BOOST", 0.1)
    )


@dataclass(frozen=True)
class AssumptionBufferingConfig:
    """Stress-buffering constants (6 fields).

    Controls resilience thresholds, volatility prior parameters,
    and initial protective factor values.
    """

    resilience_low_threshold: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_RESILIENCE_LOW_THRESHOLD", 0.3)
    )
    resilience_high_threshold: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_RESILIENCE_HIGH_THRESHOLD", 0.7)
    )
    volatility_beta_alpha: float = field(default_factory=lambda: _env_float("ASSUMPTION_VOLATILITY_BETA_ALPHA", 1.0))
    volatility_beta_beta: float = field(default_factory=lambda: _env_float("ASSUMPTION_VOLATILITY_BETA_BETA", 1.0))
    initial_protective_factor_values: float = field(
        default_factory=lambda: _env_float("ASSUMPTION_INITIAL_PROTECTIVE_FACTOR_VALUES", 0.5)
    )


# =========================================================================
# Top-level config container
# =========================================================================


@dataclass(frozen=True)
class AssumptionConfig:
    """All assumption-parameterized constants, grouped by mechanism.

    Usage::

        from src.python.assumption_config import get_assumptions

        assumptions = get_assumptions()
        reward = assumptions.coping.resource_reward
        cost_floor = assumptions.resource.min_cost_floor
    """

    coping: AssumptionCopingConfig = field(default_factory=AssumptionCopingConfig)
    stress: AssumptionStressConfig = field(default_factory=AssumptionStressConfig)
    resource: AssumptionResourceConfig = field(default_factory=AssumptionResourceConfig)
    social: AssumptionSocialConfig = field(default_factory=AssumptionSocialConfig)
    buffering: AssumptionBufferingConfig = field(default_factory=AssumptionBufferingConfig)


# =========================================================================
# Global singleton-like accessor (lightweight, freezable dataclass)
# =========================================================================

_assumptions: Optional[AssumptionConfig] = None


def get_assumptions() -> AssumptionConfig:
    """Get the (cached) AssumptionConfig singleton, reloading env each call.

    Unlike ``get_config()``, this does NOT cache across reloads — the
    frozen dataclass is safe to share.  To force re-read from env::

        from src.python.assumption_config import get_assumptions, reload_assumptions
        reload_assumptions()
    """
    global _assumptions
    if _assumptions is None:
        _assumptions = AssumptionConfig()
    return _assumptions


def reload_assumptions() -> AssumptionConfig:
    """Force re-create AssumptionConfig from current env vars."""
    global _assumptions
    _assumptions = AssumptionConfig()
    return _assumptions
