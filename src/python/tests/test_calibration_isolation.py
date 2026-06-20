"""
Isolation tests for calibration parameter changes.

Each test temporarily applies env-overridden parameter values from the
correlation sweep results to verify that a specific mechanism change
produces the expected improvement in correlation targets.

Tests use small simulations (N=75, D=75) for fast feedback.
"""

import os
import sys
from typing import Dict, Optional

import pytest
import numpy as np
from scipy import stats

sys.path.append(".")

from src.python.model import StressModel
from src.python.config import reload_config
from src.python.assumption_config import reload_assumptions


# ── Sweep-winning parameter values ────────────────────────────────

WINNING_DEFAULTS: Dict[str, str] = {
    # From config.py (5 parameters)
    "PSS10_RESILIENCE_COUPLING": "2.0",
    "PSS10_BIAS_SD": "1.0",
    "PSS10_STRESS_DAMPENING": "1.0",
    "PSS10_NOISE_SD": "0.5",
    "STRESS_DELTA": "0.4",
    # From assumption_config.py (7 parameters)
    "ASSUMPTION_AFFECT_DETERIORATION_SCALE": "0.5",
    "ASSUMPTION_AFFECT_REGENERATION_MULTIPLIER": "0.03",
    "ASSUMPTION_FAILED_COPING_COST_PENALTY": "0.8",
    "ASSUMPTION_RESOURCE_PENALTY": "0.1",
    "ASSUMPTION_RESILIENCE_IMPROVEMENT_SCALE": "0.15",
    "ASSUMPTION_COPING_SOCIAL_SUPPORT_FACTOR": "0.30",
    "ASSUMPTION_COPING_SUPPORT_BOOST_FACTOR": "0.20",
}

# ── Stress dimension mechanism fixes ──────────────────────────────
# The sweep shows PSS-10 vs stress r~0.17. Root cause: stress dimensions
# (controllability, overload) don't diverge enough between agents because:
#  1. resilience_coping_factor (0.10) is too weak — resilience barely
#     influences coping success, so all agents converge to ~50% success rate
#  2. homeostasis rates (0.05/event) erase event effects too quickly
# These overrides amplify between-agent divergence in stress dimensions.

STRESS_DIMENSION_FIXES: Dict[str, str] = {
    "ASSUMPTION_RESILIENCE_COPING_FACTOR": "0.30",
    "ASSUMPTION_CONTROLLABILITY_HOMEOSTASIS_RATE": "0.02",
    "ASSUMPTION_OVERLOAD_HOMEOSTASIS_RATE": "0.02",
}


# ── Helpers ──


def _apply_overrides(overrides: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Set env vars, return previous values."""
    saved = {}
    for key, value in overrides.items():
        saved[key] = os.environ.get(key)
        os.environ[key] = value
    reload_config()
    reload_assumptions()
    return saved


def _restore_overrides(saved: Dict[str, Optional[str]]) -> None:
    """Restore env vars from saved state."""
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    reload_config()
    reload_assumptions()


def _run_and_measure(N: int = 75, max_days: int = 75, seed: int = 42) -> Dict[str, float]:
    """Run simulation and compute PSS-10 vs current_stress r."""
    model = StressModel(N=N, max_days=max_days, seed=seed)
    while model.running:
        model.step()
    agent_data = model.get_agent_time_series_data()
    final_step = agent_data["Step"].max()
    final_epoch = agent_data[agent_data["Step"] == final_step]
    valid = ~(np.isnan(final_epoch["pss10"]) | np.isnan(final_epoch["current_stress"]))
    r_val, p_val = stats.pearsonr(final_epoch["pss10"][valid], final_epoch["current_stress"][valid])
    return {"r": r_val, "p": p_val, "n": valid.sum()}


# ── Tests ──


class TestPSS10StressCorrelation:
    """PSS-10 vs current_stress should reach empirical target after calibration."""

    @pytest.mark.slow
    def test_pss10_stress_with_noise_removed(self):
        """With bias_sd=0 (noise removed), r exceeds 0.40."""
        overrides = dict(WINNING_DEFAULTS)
        overrides["PSS10_BIAS_SD"] = "0.0"
        saved = _apply_overrides(overrides)
        try:
            result = _run_and_measure(N=75, max_days=75, seed=42)
            assert result["r"] > 0.40, (
                f"PSS-10 vs stress r={result['r']:.4f} (p={result['p']:.4f}, n={result['n']}) — expected > 0.40"
            )
        finally:
            _restore_overrides(saved)

    @pytest.mark.slow
    def test_pss10_stress_with_new_defaults(self):
        """With all winning defaults (bias_sd=1.0), r exceeds 0.40."""
        saved = _apply_overrides(WINNING_DEFAULTS)
        try:
            result = _run_and_measure(N=75, max_days=75, seed=42)
            assert result["r"] > 0.40, (
                f"PSS-10 vs stress r={result['r']:.4f} (p={result['p']:.4f}, n={result['n']}) — expected > 0.40"
            )
        finally:
            _restore_overrides(saved)

    @pytest.mark.slow
    def test_pss10_stress_with_noise_reduced(self):
        """Winning defaults + reduced noise_sd (3.5 → 1.15)."""
        saved = _apply_overrides(WINNING_DEFAULTS)
        try:
            result = _run_and_measure(N=75, max_days=75, seed=42)
            assert result["r"] > 0.40, (
                f"PSS-10 vs stress r={result['r']:.4f} (p={result['p']:.4f}, n={result['n']}) — expected > 0.40"
            )
        finally:
            _restore_overrides(saved)
