#!/usr/bin/env python3
"""
Parameter sweep script to validate correlation stability across different scenarios.

This script performs parameter sweeps on key model parameters to ensure that
theoretical correlations remain stable and within expected ranges across
different configurations. It uses Latin Hypercube Sampling (LHS) to efficiently
explore the parameter space.

Key parameters to sweep:
- omega_c: Controllability weight in appraisal function
- omega_o: Overload weight in appraisal function
- gamma: Sigmoid steepness parameter
- lambda_shock: Shock arrival rate
- alpha_soc: Social support efficacy

For each parameter set, runs a simulation and validates correlations.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# Add project root to path for imports
sys.path.append('.')

from src.python.model import StressModel
from src.python.config import get_config

# Try to import LHS sampler, fallback to random sampling
try:
    from pyDOE import lhs
    HAS_PYDOE = True
except ImportError:
    HAS_PYDOE = False
    print("Warning: pyDOE not available, using random sampling instead of LHS")


def latin_hypercube_sampling(n_samples, param_ranges):
    """Generate Latin Hypercube samples for parameter ranges."""
    if HAS_PYDOE:
        # Use proper LHS
        n_params = len(param_ranges)
        lhs_samples = lhs(n_params, samples=n_samples, criterion='maximin')

        samples = {}
        for i, (param_name, (min_val, max_val)) in enumerate(param_ranges.items()):
            samples[param_name] = min_val + lhs_samples[:, i] * (max_val - min_val)
    else:
        # Fallback to random uniform sampling
        samples = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            samples[param_name] = np.random.uniform(min_val, max_val, n_samples)

    return pd.DataFrame(samples)


def run_simulation_with_params(params, n_agents=30, max_days=30, seed=42):
    """Run simulation with given parameters and return final correlations."""
    # Set environment variables for this simulation
    original_env = {}
    for key, value in params.items():
        env_key = key.upper()
        original_env[env_key] = os.environ.get(env_key)
        os.environ[env_key] = str(value)

    try:
        # Initialize model (will pick up environment variables)
        model = StressModel(N=n_agents, max_days=max_days, seed=seed)

        # Run simulation
        while model.running:
            model.step()

        # Get final agent data
        agent_data = model.get_agent_time_series_data()
        final_epoch = agent_data[agent_data['Step'] == agent_data['Step'].max()]

        # Calculate key correlations
        correlations = {}
        key_pairs = [
            ('pss10', 'current_stress'),
            ('pss10', 'resilience'),
            ('pss10', 'affect'),
            ('pss10', 'resources'),
            ('resilience', 'affect'),
            ('resilience', 'resources'),
            ('affect', 'resources'),
            ('current_stress', 'affect'),
            ('current_stress', 'resources')
        ]

        for var1, var2 in key_pairs:
            if var1 in final_epoch.columns and var2 in final_epoch.columns:
                corr = final_epoch[var1].corr(final_epoch[var2])
                correlations[f'{var1}_{var2}'] = corr
            else:
                correlations[f'{var1}_{var2}'] = np.nan

        return correlations

    finally:
        # Restore original environment variables
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value



def validate_correlations(correlations):
    """Validate that correlations are within expected theoretical ranges."""
    expected_ranges = {
        'pss10_current_stress': (0.2, 0.9),       # Positive
        'pss10_resilience': (-1.0, 0.5),          # Negative to weak
        'pss10_affect': (-0.3, 0.3),              # Weak
        'pss10_resources': (-0.5, 0.5),           # Variable
        'resilience_affect': (-0.5, 0.5),         # Variable
        'resilience_resources': (-1.0, 1.0),      # Any correlation
        'affect_resources': (-0.2, 0.4),          # Weak
        'current_stress_affect': (-0.2, 0.2),     # Weak
        'current_stress_resources': (-0.8, 0.1)   # Negative to weak
    }

    violations = []
    for pair, (min_corr, max_corr) in expected_ranges.items():
        if pair in correlations:
            corr = correlations[pair]
            if not (min_corr <= corr <= max_corr):
                violations.append(f"{pair}: {corr:.3f} not in [{min_corr}, {max_corr}]")

    return violations


def main():
    """Main parameter sweep execution."""
    print("Parameter Sweep for Correlation Validation")
    print("=" * 50)

    # Define parameter ranges to sweep
    param_ranges = {
        'omega_c': (0.5, 2.0),      # Controllability weight
        'omega_o': (0.5, 2.0),      # Overload weight
        'gamma': (3.0, 10.0),       # Sigmoid steepness
        'lambda_shock': (0.01, 0.2), # Shock arrival rate
        'alpha_soc': (0.01, 0.2)    # Social support efficacy
    }

    # Number of parameter sets to test
    n_samples = 20

    print(f"Sampling {n_samples} parameter sets using {'LHS' if HAS_PYDOE else 'random uniform'} sampling")
    print(f"Parameters: {list(param_ranges.keys())}")
    print(f"Ranges: {param_ranges}")
    print()

    # Generate parameter samples
    param_df = latin_hypercube_sampling(n_samples, param_ranges)

    # Results storage
    results = []
    failed_configs = []

    # Run simulations for each parameter set
    for i, params in param_df.iterrows():
        print(f"Running simulation {i+1}/{n_samples} with params: {params.to_dict()}")

        try:
            # Run simulation
            correlations = run_simulation_with_params(params.to_dict())

            # Validate correlations
            violations = validate_correlations(correlations)

            # Store results
            result = params.to_dict()
            result.update(correlations)
            result['violations'] = len(violations)
            result['violation_details'] = '; '.join(violations) if violations else 'None'
            results.append(result)

            if violations:
                print(f"  ⚠️  Correlation violations: {violations}")
                failed_configs.append((i, params.to_dict(), violations))
            else:
                print("  ✓ All correlations within expected ranges")
        except Exception as e:
            print(f"  ✗ Simulation failed: {e}")
            failed_configs.append((i, params.to_dict(), [str(e)]))

    # Summary
    print("\n" + "=" * 50)
    print("PARAMETER SWEEP SUMMARY")
    print("=" * 50)

    successful_runs = len(results) - len(failed_configs)
    total_runs = len(results)

    print(f"Total parameter sets tested: {total_runs}")
    print(f"Successful runs (no violations): {successful_runs}")
    print(f"Failed runs (violations or errors): {len(failed_configs)}")

    if successful_runs == total_runs:
        print("\n🎉 ALL PARAMETER SWEEPS PASSED!")
        print("✅ Correlation stability validated across different scenarios")
        print("✅ Theoretical correlations remain robust under parameter variation")
    else:
        print(f"\n❌ {len(failed_configs)} parameter configurations failed validation")
        print("Failed configurations:")
        for i, params, violations in failed_configs[:5]:  # Show first 5
            print(f"  Config {i}: {params}")
            print(f"    Violations: {violations}")
        if len(failed_configs) > 5:
            print(f"  ... and {len(failed_configs) - 5} more")

    # Save detailed results
    results_df = pd.DataFrame(results)
    output_file = "data/demo/parameter_sweep_correlations.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    return successful_runs == total_runs


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)