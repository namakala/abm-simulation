"""
Correlation calibration sweep for the ABM simulation.

Runs a grid of parameter combinations, computes all empirical correlation
targets from the integrated population test suite, and ranks combinations
by how many targets are satisfied.

Usage:
    python src/python/calibration/correlation_sweep.py

Output:
    Prints a ranked table of parameter sets by targets_satisfied count.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.python.model import StressModel
from src.python.config import reload_config
from src.python.assumption_config import reload_assumptions

## Correlation targets (from test_integrated_population_theory.py)

CORRELATION_TARGETS: List[Dict] = [
    {
        "label": "PSS-10 vs Resilience",
        "var1": "pss10",
        "var2": "resilience",
        "ci_lower": -0.55,
        "ci_upper": -0.40,
    },
    {
        "label": "PSS-10 vs Stress",
        "var1": "pss10",
        "var2": "current_stress",
        "ci_lower": 0.40,
        "ci_upper": 0.60,
    },
    {
        "label": "PSS-10 vs Affect",
        "var1": "pss10",
        "var2": "affect",
        "ci_lower": -0.30,
        "ci_upper": -0.18,
    },
    {
        "label": "PSS-10 vs Resources",
        "var1": "pss10",
        "var2": "resources",
        "ci_lower": -0.22,
        "ci_upper": -0.08,
    },
    {
        "label": "Resources vs Stress",
        "var1": "resources",
        "var2": "current_stress",
        "ci_lower": -0.25,
        "ci_upper": -0.10,
    },
    {
        "label": "Coping vs Challenge",
        "var1": "coping_success",
        "var2": "challenge_appraisal",
        "ci_lower": 0.242,
        "ci_upper": 0.449,
    },
    {
        "label": "Coping vs Hindrance",
        "var1": "coping_success",
        "var2": "hindrance_appraisal",
        "ci_lower": -0.248,
        "ci_upper": -0.151,
    },
    {
        "label": "Interaction vs Resilience",
        "var1": "interaction_frequency",
        "var2": "resilience",
        "ci_lower": 0.158,
        "ci_upper": 0.336,
    },
    {
        "label": "Resilience vs Resources",
        "var1": "resilience",
        "var2": "resources",
        "ci_lower": 0.20,
        "ci_upper": 0.63,
    },
    {
        "label": "Affect vs Resources",
        "var1": "affect",
        "var2": "resources",
        "ci_lower": 0.15,
        "ci_upper": 0.30,
    },
]

## Sweep parameters (env var -> list of values to try)

SWEEP_PARAMS: Dict[str, List[str]] = {
    "PSS10_RESILIENCE_COUPLING": ["2.0", "4.0", "6.0"],
    "PSS10_BIAS_SD": ["1.0", "2.0", "3.0"],
    "PSS10_STRESS_DAMPENING": ["0.7", "1.0"],
    "STRESS_DELTA": ["0.3", "0.4", "0.5"],
    "ASSUMPTION_AFFECT_HOMEOSTATIC_RATE": ["0.05", "0.08", "0.12"],
}

## Fixed secondary overrides (applied to all combos)

FIXED_OVERRIDES: Dict[str, str] = {
    "ASSUMPTION_AFFECT_DETERIORATION_SCALE": "0.5",
    "ASSUMPTION_AFFECT_REGENERATION_MULTIPLIER": "0.03",
    "ASSUMPTION_FAILED_COPING_COST_PENALTY": "0.8",
    "ASSUMPTION_RESOURCE_PENALTY": "0.1",
    "ASSUMPTION_RESILIENCE_IMPROVEMENT_SCALE": "0.08",
    "ASSUMPTION_COPING_SOCIAL_SUPPORT_FACTOR": "0.30",
    "ASSUMPTION_COPING_SUPPORT_BOOST_FACTOR": "0.20",
}

## Default simulation parameters

SIM_N = 200
SIM_D = 100
SIM_SEED = 42

## Data structures


@dataclass
class SweepResult:
    """Outcome of one parameter combination in the sweep."""

    params: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    targets_satisfied: int = 0
    total_targets: int = 0
    pss10_mean: float = 0.0
    pss10_std: float = 0.0


## Helpers


def _compute_pearsonr(series_a, series_b):
    """Compute Pearson r, handling edge cases."""
    valid = ~(np.isnan(series_a) | np.isnan(series_b))
    if valid.sum() < 3:
        return np.nan
    r_val, _ = stats.pearsonr(series_a[valid], series_b[valid])
    return r_val


def _get_final_agent_data(model):
    """Extract final-step agent data from model."""
    agent_data = model.get_agent_time_series_data()
    if agent_data.empty:
        return agent_data
    final_step = agent_data["Step"].max()
    return agent_data[agent_data["Step"] == final_step]


def _run_simulation(N: int, D: int, seed: int):
    """Run a simulation to completion."""
    model = StressModel(N=N, max_days=D, seed=seed)
    while model.running:
        model.step()
    return model


def _apply_env_overrides(overrides: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Apply env overrides, return previous values for restore."""
    saved = {}
    for key, value in overrides.items():
        saved[key] = os.environ.get(key)
        os.environ[key] = value
    return saved


def _restore_env(saved: Dict[str, Optional[str]]) -> None:
    """Restore env from saved state."""
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _with_env(overrides: Dict[str, str], func, *args, **kwargs):
    """Run a function with temporary env overrides, then restore."""
    saved = _apply_env_overrides(overrides)
    reload_config()
    reload_assumptions()
    try:
        return func(*args, **kwargs)
    finally:
        _restore_env(saved)
        reload_config()
        reload_assumptions()


## Core computation


def compute_correlation_metrics(N: int = 200, max_days: int = 100, seed: int = 42) -> Dict[str, float]:
    """Run a simulation and compute all correlation targets.

    Args:
        N: Number of agents.
        max_days: Simulation duration.
        seed: Random seed.

    Returns:
        Dict mapping target labels to computed r values,
        plus "pss10_mean" and "pss10_std".
    """
    model = _run_simulation(N=N, D=max_days, seed=seed)
    agent_data = _get_final_agent_data(model)
    metrics: Dict[str, float] = {}

    # PSS-10 distribution metrics
    pss10_vals = agent_data["pss10"].dropna().values
    metrics["pss10_mean"] = float(np.mean(pss10_vals)) if len(pss10_vals) > 0 else 0.0
    metrics["pss10_std"] = float(np.std(pss10_vals)) if len(pss10_vals) > 0 else 0.0

    # Correlation targets
    col_map = {
        "pss10": "pss10",
        "resilience": "resilience",
        "affect": "affect",
        "resources": "resources",
        "current_stress": "current_stress",
        "coping_success": "coping_success",
        "challenge_appraisal": "challenge_appraisal",
        "hindrance_appraisal": "hindrance_appraisal",
        "interaction_frequency": "interaction_frequency",
    }

    for target in CORRELATION_TARGETS:
        label = target["label"]
        col_a = col_map.get(target["var1"])
        col_b = col_map.get(target["var2"])
        if col_a is None or col_b is None:
            metrics[label] = float("nan")
            continue
        if col_a not in agent_data.columns or col_b not in agent_data.columns:
            metrics[label] = float("nan")
            continue
        r_val = _compute_pearsonr(agent_data[col_a], agent_data[col_b])
        metrics[label] = r_val

    return metrics


def _check_targets(metrics: Dict[str, float]) -> int:
    """Count how many correlation targets are satisfied."""
    satisfied = 0
    for target in CORRELATION_TARGETS:
        label = target["label"]
        r_val = metrics.get(label, float("nan"))
        if not np.isnan(r_val) and target["ci_lower"] <= r_val <= target["ci_upper"]:
            satisfied += 1
    return satisfied


def _build_param_sets(
    sweep_params: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, str]]:
    """Build the Cartesian product of sweep parameters."""
    if sweep_params is None:
        sweep_params = SWEEP_PARAMS
    keys = list(sweep_params.keys())
    value_lists = list(sweep_params.values())

    import itertools

    param_sets = []
    for combo in itertools.product(*value_lists):
        param_set = dict(zip(keys, combo))
        param_sets.append(param_set)
    return param_sets


def run_sweep(
    N: int = SIM_N,
    max_days: int = SIM_D,
    seed: int = SIM_SEED,
    param_sets: Optional[List[Dict[str, str]]] = None,
    sweep_params: Optional[Dict[str, List[str]]] = None,
) -> List[SweepResult]:
    """Run the calibration sweep over parameter combinations.

    Args:
        N: Number of agents per simulation.
        max_days: Simulation duration.
        seed: Random seed.
        param_sets: Explicit list of parameter dicts (if None, builds grid from SWEEP_PARAMS).
        sweep_params: Sweep grid definition (used only if param_sets is None).

    Returns:
        List of SweepResult, sorted by targets_satisfied descending.
    """
    if param_sets is None:
        param_sets = _build_param_sets(sweep_params)

    total_targets = len(CORRELATION_TARGETS)
    results: List[SweepResult] = []

    for params in param_sets:
        # Merge sweep params with fixed overrides
        env = {}
        env.update(FIXED_OVERRIDES)
        env.update(params)

        def _run_and_measure():
            metrics = compute_correlation_metrics(N=N, max_days=max_days, seed=seed)
            satisfied = _check_targets(metrics)
            return metrics, satisfied

        metrics, satisfied = _with_env(env, _run_and_measure)

        results.append(
            SweepResult(
                params=dict(params),
                metrics=metrics,
                targets_satisfied=satisfied,
                total_targets=total_targets,
                pss10_mean=metrics.get("pss10_mean", 0.0),
                pss10_std=metrics.get("pss10_std", 0.0),
            )
        )

    # Sort by targets_satisfied descending
    results.sort(key=lambda r: r.targets_satisfied, reverse=True)
    return results


def print_results(results: List[SweepResult]) -> None:
    """Print a ranked table of sweep results."""
    print(f"\n{'=' * 90}")
    print("  CORRELATION CALIBRATION SWEEP RESULTS")
    print(f"  Targets: {len(CORRELATION_TARGETS)} correlation pairs")
    print(f"{'=' * 90}\n")

    print(f"{'Rank':<6} {'Satisfied':<10} {'PSS-10 μ':<10} {'PSS-10 σ':<10} {'Params':<50}")
    print(f"{'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 50}")

    for rank, result in enumerate(results, start=1):
        params_str = ", ".join(f"{k}={v}" for k, v in sorted(result.params.items()))
        print(
            f"{rank:<6} {result.targets_satisfied}/{result.total_targets:<6} "
            f"{result.pss10_mean:<10.2f} {result.pss10_std:<10.2f} "
            f"{params_str:<50}"
        )

    # Print best result details
    if results:
        best = results[0]
        print(f"\n{'─' * 90}")
        print(f"  BEST PARAMETER SET (#{best.targets_satisfied}/{best.total_targets} targets satisfied)")
        print(f"{'─' * 90}\n")
        print(f"  PSS-10 mean: {best.pss10_mean:.2f} (target 13-15)")
        print(f"  PSS-10 SD:   {best.pss10_std:.2f} (target 6-8)")
        print("\n  Parameters:")
        for k, v in sorted(best.params.items()):
            print(f"    {k} = {v}")
        print("\n  Correlation breakdown:")
        for target in CORRELATION_TARGETS:
            label = target["label"]
            r_val = best.metrics.get(label, float("nan"))
            in_ci = target["ci_lower"] <= r_val <= target["ci_upper"] if not np.isnan(r_val) else False
            icon = "✓" if in_ci else "✗"
            print(f"    {icon} {label:<30} r={r_val:.4f}  CI=[{target['ci_lower']:.3f}, {target['ci_upper']:.3f}]")


## Entry point


def main():
    """Run the full sweep and print results."""
    print("Building parameter grid...")
    all_param_sets = _build_param_sets()
    print(f"  {len(all_param_sets)} combinations to evaluate")

    print(f"Running sweep (N={SIM_N}, D={SIM_D}, seed={SIM_SEED})...")
    results = run_sweep(param_sets=all_param_sets)

    print_results(results)
    return results


if __name__ == "__main__":
    main()
