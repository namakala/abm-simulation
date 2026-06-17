"""
Calibration tools for tuning the ABM simulation to empirical targets.

This package provides batch-tuning utilities to adjust PSS-10 item parameters
so the population mean falls within the literature norm of [13, 15].
"""

from src.python.calibration.calibrate_pss10_population import (
    run_calibration,
    compute_pss10_population_stats,
    DEFAULT_CALIBRATION_PARAMS,
    persist_calibration_results,
    run_verification_tests,
)

__all__ = [
    "run_calibration",
    "compute_pss10_population_stats",
    "DEFAULT_CALIBRATION_PARAMS",
    "persist_calibration_results",
    "run_verification_tests",
]
