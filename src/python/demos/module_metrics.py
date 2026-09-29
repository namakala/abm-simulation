"""Shared metric schemas and extraction functions for all 5 modules.

Provides:
- TypedDict schemas for each module's metrics
- Extraction functions from PhaseOutput / AgentState
- Population-level aggregation helpers
- Isolation input grid generators

Used by simulation_individual.qmd, simulation_population.qmd, module_isolation.qmd.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
from numpy.random import Generator, PCG64

from src.python.phases.interfaces import AgentState, PhaseOutput


## Metric Schemas (one TypedDict per module)


class PerceptionMetrics(TypedDict):
    """Metrics extracted from the stress perception phase."""

    event_controllability: float
    event_overload: float
    challenge: float
    hindrance: float
    appraised_stress: float
    effective_threshold: float
    is_stressed: bool
    post_stress_controllability: float
    post_stress_overload: float
    perception_sensitivity: float


class ActivationMetrics(TypedDict):
    """Metrics extracted from the resilience activation phase."""

    coping_probability: float
    coped_successfully: bool
    resilience_effect: float
    delta_stress: float
    delta_affect: float
    resource_cost: float
    optimized_cost: float
    consecutive_hindrances: float
    post_coping_pss10: int
    neighbor_affect_modulation: float
    pf_allocation: Dict[str, float]


class AllocationMetrics(TypedDict):
    """Metrics extracted from the resource allocation phase."""

    regeneration_amount: float
    resources_before: float
    resources_after: float
    allocation_weights: Dict[str, float]
    pf_efficacies_before: Dict[str, float]
    pf_efficacies_after: Dict[str, float]
    total_allocated: float
    net_resource_change: float
    pf_growth_rate: Dict[str, float]


class BufferingMetrics(TypedDict):
    """Metrics extracted from the stress buffering phase."""

    current_stress_input: float
    resources_before: float
    resources_after: float
    resilience_after: float
    buffering_strength: float
    pf_boost: float
    stress_depletion: float
    mediated_effect: float


class DailyCycleMetrics(TypedDict):
    """Metrics extracted from the daily cycle integration."""

    pre_reset_affect: float
    post_reset_affect: float
    affect_reset_magnitude: float
    pre_decay_stress: float
    post_decay_stress: float
    stress_decay_amount: float
    daily_stress_event_count: int
    daily_interaction_count: int
    daily_support_count: int
    social_influence_amount: float
    homeostatic_amount: float
    resource_regen_amount: float
    final_daily_pss10: int


class InitializationMetrics(TypedDict):
    """Metrics extracted from the initialization block (baseline snapshot)."""

    baseline_resilience: float
    baseline_affect: float
    resources: float
    volatility: float
    pss10: int
    stress_controllability: float
    stress_overload: float
    protective_factors: Dict[str, float]


class InteractionMetrics(TypedDict):
    """Metrics extracted from the interaction phase (change values)."""

    affect_change: float
    resilience_change: float
    resource_change: float
    support_occurred: bool
    influence_magnitude: float


## Extraction Functions


def extract_perception_metrics(
    perception_output: Optional[PhaseOutput],
    pre_state: AgentState,
) -> Optional[PerceptionMetrics]:
    """Extract PerceptionMetrics from a stress perception PhaseOutput.

    Args:
        perception_output: PhaseOutput from run_stress_perception, or None.
        pre_state: AgentState before the perception phase (for baseline comparison).

    Returns:
        PerceptionMetrics dict, or None if output is None.
    """
    if perception_output is None:
        return None

    obs = perception_output["observation"]
    delta = perception_output["state_delta"]

    pre_controllability = pre_state.get("stress_controllability", 0.5)
    pre_overload = pre_state.get("stress_overload", 0.5)
    post_controllability = delta.get("stress_controllability", pre_controllability)
    post_overload = delta.get("stress_overload", pre_overload)

    sensitivity = 0.0
    if pre_controllability != 0:
        sensitivity = abs(post_controllability - pre_controllability) / pre_controllability

    return PerceptionMetrics(
        event_controllability=obs.get("event_controllability", 0.0),
        event_overload=obs.get("event_overload", 0.0),
        challenge=obs.get("challenge", 0.0),
        hindrance=obs.get("hindrance", 0.0),
        appraised_stress=obs.get("appraised_stress", 0.0),
        effective_threshold=obs.get("effective_threshold", 0.5),
        is_stressed=bool(obs.get("is_stressed", False)),
        post_stress_controllability=post_controllability,
        post_stress_overload=post_overload,
        perception_sensitivity=sensitivity,
    )


def extract_activation_metrics(
    activation_output: Optional[PhaseOutput],
    pre_state: AgentState,
) -> Optional[ActivationMetrics]:
    """Extract ActivationMetrics from a resilience activation PhaseOutput.

    Args:
        activation_output: PhaseOutput from run_resilience_activation, or None.
        pre_state: AgentState before the activation phase.

    Returns:
        ActivationMetrics dict, or None if output is None.
    """
    if activation_output is None:
        return None

    obs = activation_output["observation"]
    delta = activation_output["state_delta"]

    pre_affect = pre_state.get("affect", 0.0)
    pre_stress = pre_state.get("current_stress", 0.0)
    post_affect = delta.get("affect", pre_affect)
    post_stress = delta.get("current_stress", pre_stress)

    neighbor_affects = obs.get("neighbor_affect_modulation", 0.0)
    if not isinstance(neighbor_affects, (int, float)):
        neighbor_affects = 0.0

    pf_allocation = {}
    # Attempt to extract PF allocation from observation or state_delta
    if "pf_allocation" in obs:
        pf_allocation = obs["pf_allocation"]
    elif "protective_factors" in delta:
        pf_allocation = {"total_change": 0.0}

    return ActivationMetrics(
        coping_probability=obs.get("coping_probability", 0.0),
        coped_successfully=bool(obs.get("coped_successfully", False)),
        resilience_effect=obs.get("resilience_effect", 0.0),
        delta_stress=post_stress - pre_stress,
        delta_affect=post_affect - pre_affect,
        resource_cost=obs.get("resource_cost", 0.0),
        optimized_cost=obs.get("optimized_cost", obs.get("resource_cost", 0.0)),
        consecutive_hindrances=delta.get("consecutive_hindrances", 0.0),
        post_coping_pss10=delta.get("pss10", 0),
        neighbor_affect_modulation=neighbor_affects,
        pf_allocation=pf_allocation,
    )


def extract_allocation_metrics(
    allocation_output: Optional[PhaseOutput],
    pre_state: AgentState,
) -> Optional[AllocationMetrics]:
    """Extract AllocationMetrics from a resource allocation PhaseOutput.

    Args:
        allocation_output: PhaseOutput from run_resource_allocation, or None.
        pre_state: AgentState before the allocation phase.

    Returns:
        AllocationMetrics dict, or None if output is None.
    """
    if allocation_output is None:
        return None

    obs = allocation_output["observation"]
    delta = allocation_output["state_delta"]

    pre_resources = pre_state.get("resources", 0.5)
    post_resources = delta.get("resources", pre_resources)
    pre_pf = pre_state.get("protective_factors", {})
    post_pf = obs.get("efficacies_after", pre_pf)

    total_allocated = 0.0
    if "allocation_weights" in obs:
        total_allocated = sum(obs["allocation_weights"].values())

    pf_growth_rate: Dict[str, float] = {}
    for factor in pre_pf:
        before = pre_pf.get(factor, 0.0)
        after = post_pf.get(factor, 0.0)
        if before > 0:
            pf_growth_rate[factor] = (after - before) / before
        else:
            pf_growth_rate[factor] = after

    return AllocationMetrics(
        regeneration_amount=obs.get("regeneration_amount", 0.0),
        resources_before=pre_resources,
        resources_after=post_resources,
        allocation_weights=obs.get("allocation_weights", {}),
        pf_efficacies_before=pre_pf,
        pf_efficacies_after=post_pf,
        total_allocated=total_allocated,
        net_resource_change=post_resources - pre_resources,
        pf_growth_rate=pf_growth_rate,
    )


def extract_buffering_metrics(
    buffering_output: Optional[PhaseOutput],
    pre_state: AgentState,
) -> Optional[BufferingMetrics]:
    """Extract BufferingMetrics from a stress buffering PhaseOutput.

    Args:
        buffering_output: PhaseOutput from run_stress_buffering, or None.
        pre_state: AgentState before the buffering phase.

    Returns:
        BufferingMetrics dict, or None if output is None.
    """
    if buffering_output is None:
        return None

    obs = buffering_output["observation"]
    delta = buffering_output["state_delta"]

    pre_resources = pre_state.get("resources", 0.5)
    pre_resilience = pre_state.get("resilience", 0.5)
    post_resources = delta.get("resources", pre_resources)
    post_resilience = delta.get("resilience", pre_resilience)

    stress_depletion = post_resources - pre_resources  # negative = depletion
    buffering_strength = obs.get("buffering_strength", 0.0)
    mediated_effect = buffering_strength * post_resources if post_resources > 0 else 0.0

    return BufferingMetrics(
        current_stress_input=pre_state.get("current_stress", 0.0),
        resources_before=pre_resources,
        resources_after=post_resources,
        resilience_after=post_resilience,
        buffering_strength=buffering_strength,
        pf_boost=obs.get("pf_boost", 0.0),
        stress_depletion=stress_depletion,
        mediated_effect=mediated_effect,
    )


def extract_daily_cycle_metrics(
    pre_state: AgentState,
    post_state: AgentState,
    reset_output: Optional[PhaseOutput],
    affect_dynamics_output: Optional[PhaseOutput],
    pss10_output: Optional[PhaseOutput],
) -> Optional[DailyCycleMetrics]:
    """Extract DailyCycleMetrics from daily cycle sub-phases.

    Args:
        pre_state: AgentState before the daily cycle (start of day).
        post_state: AgentState after the daily cycle (end of day).
        reset_output: PhaseOutput from process_daily_reset.
        affect_dynamics_output: PhaseOutput from process_affect_dynamics.
        pss10_output: PhaseOutput from process_pss10_consolidation.

    Returns:
        DailyCycleMetrics dict, or None if any required output is None.
    """
    if reset_output is None:
        return None

    pre_affect = pre_state.get("affect", 0.0)
    pre_stress = pre_state.get("current_stress", 0.0)
    baseline_affect = pre_state.get("baseline_affect", 0.0)

    post_affect = post_state.get("affect", pre_affect)
    post_stress = post_state.get("current_stress", pre_stress)

    # Affect reset: how much did we move toward baseline?
    pre_deviation = abs(pre_affect - baseline_affect)
    post_deviation = abs(post_affect - baseline_affect)
    affect_reset_magnitude = max(0.0, pre_deviation - post_deviation)

    stress_decay_amount = max(0.0, pre_stress - post_stress)

    # Social influence and homeostatic amounts from affect dynamics
    social_influence = 0.0
    homeostatic = 0.0
    if affect_dynamics_output is not None:
        ad_obs = affect_dynamics_output["observation"]
        social_influence = ad_obs.get("social_influence", 0.0)
        homeostatic = ad_obs.get("homeostatic_amount", 0.0)

    # Resource regeneration from reset observation
    resource_regen = 0.0
    if reset_output is not None:
        reset_obs = reset_output["observation"]
        stress_summary = reset_obs.get("stress_summary", {})
        if isinstance(stress_summary, dict):
            resource_regen = stress_summary.get("resource_regen", 0.0)

    final_pss10 = 15
    if pss10_output is not None:
        final_pss10 = pss10_output["state_delta"].get("pss10", final_pss10)
    final_pss10 = post_state.get("pss10", final_pss10)

    return DailyCycleMetrics(
        pre_reset_affect=pre_affect,
        post_reset_affect=post_affect,
        affect_reset_magnitude=affect_reset_magnitude,
        pre_decay_stress=pre_stress,
        post_decay_stress=post_stress,
        stress_decay_amount=stress_decay_amount,
        daily_stress_event_count=len(pre_state.get("daily_stress_events", [])),
        daily_interaction_count=pre_state.get("daily_interactions", 0),
        daily_support_count=pre_state.get("daily_support_exchanges", 0),
        social_influence_amount=social_influence,
        homeostatic_amount=homeostatic,
        resource_regen_amount=resource_regen,
        final_daily_pss10=int(final_pss10),
    )


def extract_initialization_metrics(state: AgentState) -> InitializationMetrics:
    """Extract InitializationMetrics from a baseline AgentState snapshot.

    The initialization block has no phase output; the pre-simulation agent
    state itself is the source of truth.

    Args:
        state: AgentState as built at simulation start (no dynamics applied).

    Returns:
        InitializationMetrics dict.
    """
    return InitializationMetrics(
        baseline_resilience=state.get("baseline_resilience", 0.5),
        baseline_affect=state.get("baseline_affect", 0.0),
        resources=state.get("resources", 0.5),
        volatility=state.get("volatility", 0.5),
        pss10=int(state.get("pss10", 0)),
        stress_controllability=state.get("stress_controllability", 0.5),
        stress_overload=state.get("stress_overload", 0.5),
        protective_factors=dict(state.get("protective_factors", {})),
    )


def extract_interaction_metrics(
    interaction_output: Optional[PhaseOutput],
    pre_state: AgentState,
) -> Optional[InteractionMetrics]:
    """Extract InteractionMetrics from an interaction PhaseOutput.

    Interaction deltas are change values (additive), so metrics read them
    directly rather than differencing against pre_state.

    Args:
        interaction_output: PhaseOutput from process_interaction (self side).
        pre_state: AgentState before the interaction (used for baseline
            comparison when deltas are absolute).

    Returns:
        InteractionMetrics dict, or None if output is None.
    """
    if interaction_output is None:
        return None

    obs = interaction_output["observation"]
    delta = interaction_output["state_delta"]

    affect_change = delta.get("affect", 0.0)
    resilience_change = delta.get("resilience", 0.0)
    resource_change = delta.get("resources", 0.0)
    support_occurred = bool(obs.get("support_occurred", False))
    influence_magnitude = abs(affect_change) + abs(resilience_change)

    return InteractionMetrics(
        affect_change=affect_change,
        resilience_change=resilience_change,
        resource_change=resource_change,
        support_occurred=support_occurred,
        influence_magnitude=influence_magnitude,
    )


## Aggregation Functions


def _compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, min, max for a list of floats."""
    if not values:
        return {}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def aggregate_perception(metrics: List[PerceptionMetrics]) -> Dict[str, Any]:
    """Aggregate PerceptionMetrics across a population.

    Args:
        metrics: List of PerceptionMetrics from multiple agents/events.

    Returns:
        Dict with keys like challenge_mean, hindrance_std, stress_prevalence, etc.
    """
    if not metrics:
        return {}

    challenges = [m["challenge"] for m in metrics]
    hindrances = [m["hindrance"] for m in metrics]
    stressed = [m["is_stressed"] for m in metrics]

    result: Dict[str, Any] = {}
    result.update({f"challenge_{k}": v for k, v in _compute_stats(challenges).items()})
    result.update({f"hindrance_{k}": v for k, v in _compute_stats(hindrances).items()})
    result["stress_prevalence"] = float(np.mean(stressed))
    return result


def aggregate_activation(metrics: List[ActivationMetrics]) -> Dict[str, Any]:
    """Aggregate ActivationMetrics across a population.

    Args:
        metrics: List of ActivationMetrics from multiple agents/events.

    Returns:
        Dict with coping success rate, mean resilience effect, etc.
    """
    if not metrics:
        return {}

    successes = [m["coped_successfully"] for m in metrics]
    resilience_effects = [m["resilience_effect"] for m in metrics]
    resource_costs = [m["resource_cost"] for m in metrics]

    result: Dict[str, Any] = {}
    result["coping_success_rate"] = float(np.mean(successes))
    result.update({f"resilience_effect_{k}": v for k, v in _compute_stats(resilience_effects).items()})
    result.update({f"resource_cost_{k}": v for k, v in _compute_stats(resource_costs).items()})
    return result


def aggregate_allocation(metrics: List[AllocationMetrics]) -> Dict[str, Any]:
    """Aggregate AllocationMetrics across a population.

    Args:
        metrics: List of AllocationMetrics from multiple agents.

    Returns:
        Dict with mean regeneration, mean net resource change, etc.
    """
    if not metrics:
        return {}

    regenerations = [m["regeneration_amount"] for m in metrics]
    net_changes = [m["net_resource_change"] for m in metrics]

    result: Dict[str, Any] = {}
    result.update({f"regeneration_{k}": v for k, v in _compute_stats(regenerations).items()})
    result.update({f"net_resource_change_{k}": v for k, v in _compute_stats(net_changes).items()})

    # Average allocation weights across all agents
    all_weights: Dict[str, List[float]] = {}
    for m in metrics:
        for factor, weight in m["allocation_weights"].items():
            all_weights.setdefault(factor, []).append(weight)
    result["avg_allocation_weights"] = {f: float(np.mean(w)) for f, w in all_weights.items()}

    return result


def aggregate_buffering(metrics: List[BufferingMetrics]) -> Dict[str, Any]:
    """Aggregate BufferingMetrics across a population.

    Args:
        metrics: List of BufferingMetrics from multiple agents.

    Returns:
        Dict with mean buffering strength, mean stress depletion, etc.
    """
    if not metrics:
        return {}

    strengths = [m["buffering_strength"] for m in metrics]
    depletions = [m["stress_depletion"] for m in metrics]
    boosts = [m["pf_boost"] for m in metrics]

    result: Dict[str, Any] = {}
    result.update({f"buffering_strength_{k}": v for k, v in _compute_stats(strengths).items()})
    result.update({f"stress_depletion_{k}": v for k, v in _compute_stats(depletions).items()})
    result.update({f"pf_boost_{k}": v for k, v in _compute_stats(boosts).items()})
    return result


def aggregate_daily_cycle(metrics: List[DailyCycleMetrics]) -> Dict[str, Any]:
    """Aggregate DailyCycleMetrics across a population.

    Args:
        metrics: List of DailyCycleMetrics from multiple agents.

    Returns:
        Dict with mean reset magnitude, mean stress decay, etc.
    """
    if not metrics:
        return {}

    reset_mags = [m["affect_reset_magnitude"] for m in metrics]
    stress_decays = [m["stress_decay_amount"] for m in metrics]
    event_counts = [m["daily_stress_event_count"] for m in metrics]
    pss10_scores = [m["final_daily_pss10"] for m in metrics]

    result: Dict[str, Any] = {}
    result.update({f"affect_reset_magnitude_{k}": v for k, v in _compute_stats(reset_mags).items()})
    result.update({f"stress_decay_{k}": v for k, v in _compute_stats(stress_decays).items()})
    result.update({f"daily_stress_events_{k}": v for k, v in _compute_stats(list(map(float, event_counts))).items()})
    result.update({f"pss10_{k}": v for k, v in _compute_stats(list(map(float, pss10_scores))).items()})
    return result


def aggregate_initialization(metrics: List[InitializationMetrics]) -> Dict[str, Any]:
    """Aggregate InitializationMetrics across a population.

    Args:
        metrics: List of InitializationMetrics from multiple agents.

    Returns:
        Dict with population means for each baseline trait.
    """
    if not metrics:
        return {}

    result: Dict[str, Any] = {}
    result["resilience_mean"] = float(np.mean([m["baseline_resilience"] for m in metrics]))
    result["affect_mean"] = float(np.mean([m["baseline_affect"] for m in metrics]))
    result["resources_mean"] = float(np.mean([m["resources"] for m in metrics]))
    result["pss10_mean"] = float(np.mean([m["pss10"] for m in metrics]))
    result["volatility_mean"] = float(np.mean([m["volatility"] for m in metrics]))
    return result


def aggregate_interaction(metrics: List[InteractionMetrics]) -> Dict[str, Any]:
    """Aggregate InteractionMetrics across a population.

    Args:
        metrics: List of InteractionMetrics from multiple interactions.

    Returns:
        Dict with support exchange rate and mean change magnitudes.
    """
    if not metrics:
        return {}

    supports = [m["support_occurred"] for m in metrics]
    affect_changes = [m["affect_change"] for m in metrics]
    resilience_changes = [m["resilience_change"] for m in metrics]
    resource_changes = [m["resource_change"] for m in metrics]
    influences = [m["influence_magnitude"] for m in metrics]

    result: Dict[str, Any] = {}
    result["support_exchange_rate"] = float(np.mean(supports))
    result.update({f"affect_change_{k}": v for k, v in _compute_stats(affect_changes).items()})
    result.update({f"resilience_change_{k}": v for k, v in _compute_stats(resilience_changes).items()})
    result.update({f"resource_change_{k}": v for k, v in _compute_stats(resource_changes).items()})
    result.update({f"influence_magnitude_{k}": v for k, v in _compute_stats(influences).items()})
    return result


## Isolation Input Generator


def compute_isolation_inputs(
    module_name: str,
    variation_grid: Dict[str, Any],
    fixed_state: AgentState,
) -> Sequence[Tuple[AgentState, Dict[str, Any], Generator]]:
    """Generate controlled input sets for a module's isolation demo.

    Each tuple contains (state, config, rng) suitable for passing to the
    corresponding ``run_phase()`` function.

    Args:
        module_name: One of "perception", "activation", "allocation",
                     "buffering", "daily_cycle".
        variation_grid: Dict controlling sweep resolution. Expected key
                        "default" with sub-key "resolution" (int, default 20).
        fixed_state: Baseline AgentState shared across all grid points.

    Returns:
        Sequence of (state, config, rng) tuples, each representing one
        point in the isolated parameter sweep.

    Raises:
        ValueError: If module_name is not recognized.
    """
    resolution = variation_grid.get("default", {}).get("resolution", 20)
    results: List[Tuple[AgentState, Dict[str, Any], Generator]] = []
    seed_base = 42

    if module_name == "perception":
        for i, omega_c in enumerate(np.linspace(0, 1, resolution)):
            for j, omega_o in enumerate(np.linspace(0, 1, resolution)):
                state = dict(fixed_state)
                state.setdefault("stress_controllability", 0.5)
                state.setdefault("stress_overload", 0.5)
                state.setdefault("volatility", 0.5)
                state.setdefault("recent_stress_intensity", 0.0)
                state.setdefault("stress_momentum", 0.0)
                state.setdefault("resilience", 0.5)
                config = {
                    "omega_c": omega_c,
                    "omega_o": omega_o,
                    "bias": 0.0,
                    "gamma": 6.0,
                    "delta": 0.2,
                    "base_threshold": 0.5,
                    "challenge_scale": 0.15,
                    "hindrance_scale": 0.25,
                }
                rng = Generator(PCG64(seed_base + i * resolution + j))
                results.append((state, config, rng))

    elif module_name == "activation":
        neighbor_levels = [-0.5, 0.0, 0.5]
        for k, neighbor_affect in enumerate(neighbor_levels):
            for i, challenge in enumerate(np.linspace(0, 1, resolution)):
                for j, hindrance in enumerate(np.linspace(0, 1, resolution)):
                    state = dict(fixed_state)
                    state.setdefault("affect", 0.0)
                    state.setdefault("resilience", 0.5)
                    state.setdefault("current_stress", 0.3)
                    state["challenge"] = challenge
                    state["hindrance"] = hindrance
                    state.setdefault(
                        "protective_factors",
                        {
                            "social_support": 0.5,
                            "family_support": 0.5,
                            "formal_intervention": 0.5,
                            "psychological_capital": 0.5,
                        },
                    )
                    config = {
                        "neighbor_affects": [neighbor_affect],
                        "base_resource_cost": 0.1,
                    }
                    seed = seed_base + k * resolution * resolution + i * resolution + j
                    rng = Generator(PCG64(seed))
                    results.append((state, config, rng))

    elif module_name == "allocation":
        for i, resources in enumerate(np.linspace(0, 1, resolution)):
            for j, temperature in enumerate(np.linspace(0.5, 5.0, resolution)):
                state = dict(fixed_state)
                state["resources"] = resources
                state.setdefault("affect", 0.0)
                state.setdefault("resilience", 0.5)
                state.setdefault(
                    "protective_factors",
                    {
                        "social_support": 0.5,
                        "family_support": 0.5,
                        "formal_intervention": 0.5,
                        "psychological_capital": 0.5,
                    },
                )
                config = {
                    "base_regeneration": 0.1,
                    "softmax_temperature": temperature,
                    "protective_improvement_rate": 0.1,
                }
                seed = seed_base + i * resolution + j
                rng = Generator(PCG64(seed))
                results.append((state, config, rng))

    elif module_name == "buffering":
        for i, current_stress in enumerate(np.linspace(0, 1, resolution)):
            state = dict(fixed_state)
            state["current_stress"] = current_stress
            state.setdefault("resilience", 0.5)
            state.setdefault("resources", 0.5)
            state.setdefault(
                "protective_factors",
                {
                    "social_support": 0.5,
                    "family_support": 0.5,
                    "formal_intervention": 0.5,
                    "psychological_capital": 0.5,
                },
            )
            config = {}
            rng = Generator(PCG64(seed_base + i))
            results.append((state, config, rng))

    elif module_name == "daily_cycle":
        for affect in np.linspace(-1, 1, resolution):
            for current_stress in np.linspace(0, 1, resolution // 2):
                state = dict(fixed_state)
                state["affect"] = affect
                state["current_stress"] = current_stress
                state.setdefault("baseline_affect", 0.0)
                config = {}
                rng = Generator(PCG64(42))
                results.append((state, config, rng))

    else:
        raise ValueError(f"Unknown module: {module_name}")

    return results
