#!/usr/bin/env python3
"""
Full calibration runner.

Adjusts PSS-10 item parameters so the population mean falls within
the literature norm of 13-15 (Cohen, Kamarck & Mermelstein 1983).

Usage:
    pixi run calibrate
"""

from src.python.calibration import run_calibration

if __name__ == "__main__":
    result = run_calibration(
        N=200,
        max_days=100,
        seeds=[42, 123, 456, 789, 101112],
        max_iterations=20,
        learning_rate=0.5,
    )

    print("=" * 50)
    print("Calibration complete")
    print(f"  Initial mean: {result['initial_mean']:.2f}")
    print(f"  Final mean:   {result['final_mean']:.2f}")
    print(f"  Converged:    {result['converged']}")
    print(f"  Iterations:   {result['n_iterations']}")
    if result["converged"]:
        print("  Calibration successful!")
    else:
        print("  Calibration did not converge within max_iterations")
