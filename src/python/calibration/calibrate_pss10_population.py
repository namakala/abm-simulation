"""
PSS-10 population mean calibration for the ABM simulation.

Adjusts PSS-10 item parameters so the population mean falls within
the literature norm of 13-15 (Cohen, Kamarck & Mermelstein 1983).
"""

import logging
from pathlib import Path

import numpy as np
from typing import Dict, List, Optional

from src.python.model import StressModel

logger = logging.getLogger(__name__)

# ── Default calibration parameters ────────────────────────────────

DEFAULT_CALIBRATION_PARAMS: Dict[str, float] = {
    "target_mean_min": 13.0,
    "target_mean_max": 15.0,
    "target_sd_min": 6.0,
    "target_sd_max": 8.0,
    "max_iterations": 20,
    "learning_rate": 0.5,
}

# ── Public API ────────────────────────────────────────────────────


def compute_pss10_population_stats(
    N: int = 200,
    max_days: int = 100,
    seed: int = 42,
    item_means: Optional[List[float]] = None,
    item_sds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Run a simulation and compute PSS-10 population mean and standard deviation.

    Parameters
    ----------
    N : int
        Number of agents.
    max_days : int
        Number of simulation days.
    seed : int
        Random seed for reproducibility.
    item_means : list of float, optional
        Override PSS-10 item means (length 10).
    item_sds : list of float, optional
        Override PSS-10 item standard deviations (length 10).

    Returns
    -------
    dict with keys 'mean' and 'std' — the PSS-10 population statistics.
    """
    env_overrides = _build_env_overrides(item_means, item_sds)
    original_env = _apply_env_overrides(env_overrides)

    try:
        model = StressModel(N=N, max_days=max_days, seed=seed)
        while model.running:
            model.step()

        agent_data = model.get_agent_time_series_data()
        if agent_data.empty:
            return {"mean": 0.0, "std": 0.0}

        final_step = agent_data["Step"].max()
        final_epoch = agent_data[agent_data["Step"] == final_step]

        pss10_vals = final_epoch["pss10"].dropna().values
        if len(pss10_vals) == 0:
            return {"mean": 0.0, "std": 0.0}

        return {"mean": float(np.mean(pss10_vals)), "std": float(np.std(pss10_vals))}
    finally:
        _restore_env(original_env)


def run_calibration(
    N: int = 200,
    max_days: int = 100,
    seeds: Optional[List[int]] = None,
    max_iterations: int = 20,
    learning_rate: float = 0.5,
    target_mean_min: float = 13.0,
    target_mean_max: float = 15.0,
    target_sd_min: float = 6.0,
    target_sd_max: float = 8.0,
) -> Dict:
    """
    Run calibration to adjust PSS-10 item parameters to hit the target range.

    Uses a simple gradient-descent-style adjustment on PSS-10 item means.

    Parameters
    ----------
    N : int
        Number of agents per simulation.
    max_days : int
        Simulation duration.
    seeds : list of int, optional
        Seeds to average over (default: [42, 123, 456, 789, 101112]).
    max_iterations : int
        Maximum calibration iterations.
    learning_rate : float
        Adjustment step size for item means.
    target_mean_min, target_mean_max : float
        Target PSS-10 mean range.
    target_sd_min, target_sd_max : float
        Target PSS-10 SD range.

    Returns
    -------
    dict with initial_mean, final_mean, n_iterations, converged, item_means, item_sds
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 101112]

    # Get baseline item means from config
    from src.python.config import get_config

    cfg = get_config()
    item_means = list(cfg.get("pss10", "item_means"))
    item_sds = list(cfg.get("pss10", "item_sds"))

    # Initial measurement
    initial_mean, initial_std = _measure_across_seeds(N, max_days, seeds, item_means, item_sds)

    result = {
        "initial_mean": initial_mean,
        "initial_std": initial_std,
        "final_mean": initial_mean,
        "final_std": initial_std,
        "n_iterations": 0,
        "converged": False,
        "item_means": list(item_means),
        "item_sds": list(item_sds),
    }

    if max_iterations == 0:
        return result

    current_means = list(item_means)
    current_sds = list(item_sds)

    for iteration in range(1, max_iterations + 1):
        mean_val, std_val = _measure_across_seeds(N, max_days, seeds, current_means, current_sds)

        result["n_iterations"] = iteration
        result["final_mean"] = mean_val
        result["final_std"] = std_val
        result["item_means"] = list(current_means)
        result["item_sds"] = list(current_sds)

        mean_ok = target_mean_min <= mean_val <= target_mean_max
        sd_ok = target_sd_min <= std_val <= target_sd_max

        if mean_ok and sd_ok:
            result["converged"] = True
            logger.info(f"Calibration converged at iteration {iteration}: mean={mean_val:.2f}, std={std_val:.2f}")

            # Persist calibrated values to files
            original_means = list(item_means)
            persist_calibration_results(current_means, original_means)

            # Verify by running dependent tests
            logger.info("Running verification tests...")
            tests_pass = run_verification_tests()
            if tests_pass:
                logger.info("Verification tests passed.")
                break
            else:
                logger.warning("Verification tests failed — continuing calibration.")
                result["converged"] = False
                # Continue to next iteration

        # Adjust item means: shift all items proportionally to error
        target_mid = (target_mean_min + target_mean_max) / 2.0
        error = target_mid - mean_val
        adjustment = error * learning_rate / 10.0  # spread across 10 items

        current_means = [max(0.0, min(4.0, m + adjustment)) for m in current_means]

    return result


# ── Internal helpers ──────────────────────────────────────────────


def _measure_across_seeds(
    N: int,
    max_days: int,
    seeds: List[int],
    item_means: List[float],
    item_sds: List[float],
) -> tuple:
    """Run simulations across multiple seeds and return average PSS-10 mean and std."""
    all_means = []
    all_stds = []

    for seed in seeds:
        stats = compute_pss10_population_stats(
            N=N, max_days=max_days, seed=seed, item_means=item_means, item_sds=item_sds
        )
        all_means.append(stats["mean"])
        all_stds.append(stats["std"])

    return float(np.mean(all_means)), float(np.mean(all_stds))


def _build_env_overrides(
    item_means: Optional[List[float]] = None,
    item_sds: Optional[List[float]] = None,
) -> Dict[str, str]:
    """Build environment variable overrides dict."""
    overrides = {}
    if item_means is not None:
        overrides["PSS10_ITEM_MEAN"] = "[" + ", ".join(str(v) for v in item_means) + "]"
    if item_sds is not None:
        overrides["PSS10_ITEM_SD"] = "[" + ", ".join(str(v) for v in item_sds) + "]"
    return overrides


def _apply_env_overrides(overrides: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Apply env overrides, return previous values for restore."""
    import os

    saved = {}
    for key, value in overrides.items():
        saved[key] = os.environ.get(key)
        os.environ[key] = value
    return saved


def _restore_env(saved: Dict[str, Optional[str]]) -> None:
    """Restore environment variables from saved state."""
    import os

    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# ── Persistence (delegates to persistence module) ───────────────

from src.python.calibration.persistence import (
    persist_calibration_results,
    run_verification_tests,
    _update_env_file,
    _update_config_default,
    _update_test_expected_means,
)
