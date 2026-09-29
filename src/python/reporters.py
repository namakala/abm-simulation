"""
DataCollector reporter functions for the ABM stress simulation.

All reporters are named functions with explicit model/agent parameters,
extracted from the lambda-based definitions in model.py:_initialize_datacollector().
"""

from typing import Any

import numpy as np


## Agent-level reporters


def report_pss10(agent: Any) -> int:
    """Agent's current PSS-10 score."""
    return agent.pss10


def report_resilience(agent: Any) -> float:
    """Agent's current resilience level."""
    return agent.resilience


def report_affect(agent: Any) -> float:
    """Agent's current affect level."""
    return agent.affect


def report_resources(agent: Any) -> float:
    """Agent's current resources level."""
    return agent.resources


def report_current_stress(agent: Any) -> float:
    """Agent's current stress level."""
    return getattr(agent, "current_stress", 0.0)


def report_stress_controllability(agent: Any) -> float:
    """Agent's perceived stress controllability."""
    return getattr(agent, "stress_controllability", 0.5)


def report_stress_overload(agent: Any) -> float:
    """Agent's perceived stress overload."""
    return getattr(agent, "stress_overload", 0.5)


def report_consecutive_hindrances(agent: Any) -> int:
    """Agent's consecutive hindrance event count."""
    return getattr(agent, "consecutive_hindrances", 0)


def report_coping_success(agent: Any) -> float:
    """Agent's coping success rate for stressful events only."""
    events = getattr(agent, "last_daily_stress_events", [])
    stress_events = [e for e in events if e.get("is_stressed", False)]
    if not stress_events:
        return 0.0
    successes = sum(1 for e in stress_events if e.get("coped_successfully", False))
    return successes / len(stress_events)


def report_challenge_appraisal(agent: Any) -> float:
    """Agent's average challenge appraisal for stressed events."""
    events = getattr(agent, "last_daily_stress_events", [])
    stress_events = [e for e in events if e.get("is_stressed", False)]
    if not stress_events:
        return 0.0
    return float(np.mean([e.get("challenge", 0.0) for e in stress_events]))


def report_hindrance_appraisal(agent: Any) -> float:
    """Agent's average hindrance appraisal for stressed events."""
    events = getattr(agent, "last_daily_stress_events", [])
    stress_events = [e for e in events if e.get("is_stressed", False)]
    if not stress_events:
        return 0.0
    return float(np.mean([e.get("hindrance", 0.0) for e in stress_events]))


def report_interaction_frequency(agent: Any) -> int:
    """Agent's daily interaction count."""
    return getattr(agent, "last_daily_interactions", 0)


def report_stressed(agent: Any) -> bool:
    """Whether agent is classified as stressed."""
    return getattr(agent, "stressed", False)


def report_support_boost(agent: Any) -> float:
    """Agent's within-day support boost from exchanges."""
    return getattr(agent, "support_boost", 0.0)


## Model-level reporters


def model_report_avg_pss10(model: Any) -> float:
    """Population average PSS-10 score."""
    if not model.agents:
        return 0.0
    values = [a.pss10 for a in model.agents if a.pss10 is not None]
    if not values:
        return 0.0
    return float(np.mean(values))


def model_report_avg_resilience(model: Any) -> float:
    """Population average resilience."""
    if not model.agents:
        return 0.0
    return float(np.mean([a.resilience for a in model.agents]))


def model_report_avg_affect(model: Any) -> float:
    """Population average affect."""
    if not model.agents:
        return 0.0
    return float(np.mean([a.affect for a in model.agents]))


def model_report_coping_success_rate(model: Any) -> float:
    """Population coping success rate."""
    return model.get_success_rate()


def model_report_avg_resources(model: Any) -> float:
    """Population average resources."""
    if not model.agents:
        return 0.0
    return float(np.mean([a.resources for a in model.agents]))


def model_report_avg_stress(model: Any) -> float:
    """Population average current stress."""
    if not model.agents:
        return 0.0
    return float(np.mean([getattr(a, "current_stress", 0.0) for a in model.agents]))


def model_report_social_support_rate(model: Any) -> float:
    """Cumulative social support rate."""
    return model._calculate_social_support_rate()


def model_report_daily_social_support_rate(model: Any) -> float:
    """Per-step social support rate."""
    return model._calculate_daily_social_support_rate()


def model_report_stress_events(model: Any) -> int:
    """Total stress events per day."""
    return sum(len(getattr(a, "last_daily_stress_events", [])) for a in model.agents)


def model_report_network_density(model: Any) -> float:
    """Network connectivity measure."""
    return model._calculate_network_density()


def model_report_stress_prevalence(model: Any) -> float:
    """Proportion of agents classified as stressed."""
    if not model.agents:
        return 0.0
    return sum(1 for a in model.agents if getattr(a, "stressed", False)) / len(model.agents)


def model_report_low_resilience(model: Any) -> int:
    """Count of agents with low resilience (< 0.3)."""
    return sum(1 for a in model.agents if a.resilience < 0.3)


def model_report_high_resilience(model: Any) -> int:
    """Count of agents with high resilience (> 0.7)."""
    return sum(1 for a in model.agents if a.resilience > 0.7)


def model_report_avg_challenge(model: Any) -> float:
    """Average challenge appraisal across all events."""
    return model._get_avg_challenge()


def model_report_avg_hindrance(model: Any) -> float:
    """Average hindrance appraisal across all events."""
    return model._get_avg_hindrance()


def model_report_challenge_hindrance_ratio(model: Any) -> float:
    """Balance between challenge and hindrance."""
    return model._get_challenge_hindrance_ratio()


def model_report_avg_consecutive_hindrances(model: Any) -> float:
    """Average consecutive hindrance events."""
    return model._get_avg_consecutive_hindrances()


def model_report_total_stress_events(model: Any) -> int:
    """Total stress events (alias for stress_events)."""
    return model_report_stress_events(model)


def model_report_successful_coping(model: Any) -> int:
    """Total successful coping instances."""
    return sum(
        sum(1 for e in getattr(a, "last_daily_stress_events", []) if e.get("coped_successfully", False))
        for a in model.agents
    )


def model_report_social_interactions(model: Any) -> int:
    """Total social interactions per step."""
    return sum(getattr(a, "last_daily_interactions", 0) for a in model.agents)


def model_report_support_exchanges(model: Any) -> int:
    """Total support exchanges per step."""
    return sum(getattr(a, "last_daily_support_exchanges", 0) for a in model.agents)


def model_report_total_interactions(model: Any) -> int:
    """Cumulative social interactions."""
    return getattr(model, "total_interactions", 0)


def model_report_social_support_exchanges(model: Any) -> int:
    """Cumulative support exchanges."""
    return getattr(model, "social_support_exchanges", 0)


def model_report_daily_coping_support_corr(model: Any) -> float:
    """Per-step correlation between coping success and support exchanges."""
    return model._compute_daily_coping_support_corr()


def _phase_observation(agent: Any, phase_key: str, obs_key: str) -> float:
    """Extract one float value from an agent's last phase observation.

    Returns ``float("nan")`` when the agent has no recorded phase output
    (or the output is not a proper dict, e.g. uninitialized agents).

    Args:
        agent: Agent instance.
        phase_key: Key under ``_last_phase_outputs``, e.g. "resource_allocation".
        obs_key: Observation field to read, e.g. "regeneration_amount".

    Returns:
        The observed float value or NaN.
    """
    outputs = getattr(agent, "_last_phase_outputs", None)
    if not isinstance(outputs, dict):
        return float("nan")
    phase_out = outputs.get(phase_key)
    if not isinstance(phase_out, dict):
        return float("nan")
    obs = phase_out.get("observation")
    if not isinstance(obs, dict):
        return float("nan")
    value = obs.get(obs_key, float("nan"))
    return float(value) if isinstance(value, (int, float)) else float("nan")


def model_report_avg_regeneration(model: Any) -> float:
    """Population mean of the daily resource regeneration amount.

    Reads the last allocation phase output observed by each agent.
    Returns ``float("nan")`` when no agent recorded an observation.
    """
    values = [_phase_observation(agent, "resource_allocation", "regeneration_amount") for agent in model.agents]
    return float(np.nanmean(values)) if values else float("nan")


def model_report_avg_buffering_strength(model: Any) -> float:
    """Population mean of the daily stress buffering strength.

    Reads the last buffering phase output observed by each agent.
    Returns ``float("nan")`` when no agent recorded an observation.
    """
    values = [_phase_observation(agent, "stress_buffering", "buffering_strength") for agent in model.agents]
    return float(np.nanmean(values)) if values else float("nan")


## Lookup dicts for DataCollector construction

AGENT_REPORTERS = {
    "pss10": report_pss10,
    "resilience": report_resilience,
    "affect": report_affect,
    "resources": report_resources,
    "current_stress": report_current_stress,
    "stress_controllability": report_stress_controllability,
    "stress_overload": report_stress_overload,
    "consecutive_hindrances": report_consecutive_hindrances,
    "coping_success": report_coping_success,
    "challenge_appraisal": report_challenge_appraisal,
    "hindrance_appraisal": report_hindrance_appraisal,
    "interaction_frequency": report_interaction_frequency,
    "stressed": report_stressed,
    "support_boost": report_support_boost,
}

MODEL_REPORTERS = {
    "avg_pss10": model_report_avg_pss10,
    "avg_resilience": model_report_avg_resilience,
    "avg_affect": model_report_avg_affect,
    "coping_success_rate": model_report_coping_success_rate,
    "avg_resources": model_report_avg_resources,
    "avg_stress": model_report_avg_stress,
    "social_support_rate": model_report_social_support_rate,
    "daily_social_support_rate": model_report_daily_social_support_rate,
    "stress_events": model_report_stress_events,
    "network_density": model_report_network_density,
    "stress_prevalence": model_report_stress_prevalence,
    "low_resilience": model_report_low_resilience,
    "high_resilience": model_report_high_resilience,
    "avg_challenge": model_report_avg_challenge,
    "avg_hindrance": model_report_avg_hindrance,
    "challenge_hindrance_ratio": model_report_challenge_hindrance_ratio,
    "avg_consecutive_hindrances": model_report_avg_consecutive_hindrances,
    "total_stress_events": model_report_total_stress_events,
    "successful_coping": model_report_successful_coping,
    "social_interactions": model_report_social_interactions,
    "support_exchanges": model_report_support_exchanges,
    "total_interactions": model_report_total_interactions,
    "social_support_exchanges": model_report_social_support_exchanges,
    "daily_coping_support_corr": model_report_daily_coping_support_corr,
    "avg_regeneration": model_report_avg_regeneration,
    "avg_buffering_strength": model_report_avg_buffering_strength,
}
