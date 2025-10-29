#!/usr/bin/env python3
"""
Demo script for correlation analysis of mental health simulation variables.

This script runs a simulation and calculates correlations between key variables:
- avg_pss10 (Perceived Stress Scale-10)
- avg_resilience (Population resilience)
- avg_affect (Population affect)
- avg_stress (Population stress level)

The script demonstrates:
1. Running a simulation programmatically using the simulate.py interface
2. Extracting model data from the simulation results
3. Computing correlation matrix for the specified variables
4. Displaying results in a clear format

Usage:
    python src/python/demos/demo_correlation_analysis.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path for imports
sys.path.append('.')

from src.python.model import StressModel
from src.python.config import get_config


def run_simulation_and_get_data(days: int = 100, agents: int = 30, seed: int = 42):
    """
    Run simulation and return the model dataframe.

    Args:
        days: Number of simulation days
        agents: Number of agents in simulation
        seed: Random seed for reproducibility

    Returns:
        pandas.DataFrame: Model data with time series metrics
    """
    print(f"Running simulation with {agents} agents for {days} days (seed={seed})...")

    # Create and run the model
    model = StressModel(N=agents, max_days=days, seed=seed)

    # Run simulation
    while model.running:
        model.step()

    # Get the model dataframe
    model_data = model.get_time_series_data()

    print(f"Simulation completed. Data shape: {model_data.shape}")
    return model_data


def calculate_correlation_matrix(model_data: pd.DataFrame, variables: list):
    """
    Calculate correlation matrix for specified variables.

    Args:
        model_data: DataFrame containing model time series data
        variables: List of variable names to include in correlation analysis

    Returns:
        pandas.DataFrame: Correlation matrix
    """
    # Check if all variables exist in the data
    missing_vars = [var for var in variables if var not in model_data.columns]
    if missing_vars:
        print(f"Warning: Variables not found in data: {missing_vars}")
        # Filter to only available variables
        variables = [var for var in variables if var in model_data.columns]

    if not variables:
        raise ValueError("No valid variables found for correlation analysis")

    # Extract the variables and calculate correlation
    correlation_data = model_data[variables]
    correlation_matrix = correlation_data.corr()

    return correlation_matrix


def print_correlation_matrix(corr_matrix: pd.DataFrame):
    """
    Print correlation matrix in a formatted way.

    Args:
        corr_matrix: Correlation matrix DataFrame
    """
    print("\n" + "="*60)
    print("CORRELATION MATRIX")
    print("="*60)

    # Print the matrix with nice formatting
    print(corr_matrix.to_string(float_format='%.4f'))

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)

    # Provide interpretation of key correlations
    variables = corr_matrix.columns.tolist()

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i < j:  # Only print upper triangle
                corr = corr_matrix.loc[var1, var2]
                strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
                direction = "positive" if corr > 0 else "negative"
                print(f"{var1} ↔ {var2}: {strength} {direction} correlation ({corr:.4f})")


def main():
    """Main function to run the correlation analysis demo."""

    print("Mental Health ABM - Correlation Analysis Demo")
    print("=" * 50)

    # Simulation parameters (matching the requested command)
    days = 100
    agents = 30
    seed = 42  # Using default seed for reproducibility

    # Variables to analyze
    variables_of_interest = [
        'avg_pss10',
        'avg_resilience',
        'avg_affect',
        'avg_stress',
        'avg_resources',
        'social_support_rate',
        'coping_success_rate'
    ]

    try:
        # Run simulation
        model_data = run_simulation_and_get_data(days=days, agents=agents, seed=seed)

        # Check if we have the required data
        if model_data.empty:
            print("Error: No data returned from simulation")
            return 1

        print(f"\nAvailable variables in model data: {list(model_data.columns)}")

        # Calculate correlation matrix
        print(f"\nAnalyzing correlations for variables: {variables_of_interest}")
        corr_matrix = calculate_correlation_matrix(model_data, variables_of_interest)

        # Print results
        print_correlation_matrix(corr_matrix)

        # Additional statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)

        for var in variables_of_interest:
            if var in model_data.columns:
                values = model_data[var]
                print(f"{var}:")
                print(f"  Mean: {values.mean():.4f}")
                print(f"  Std:  {values.std():.4f}")
                print(f"  Min:  {values.min():.4f}")
                print(f"  Max:  {values.max():.4f}")
                print()

        print("Demo completed successfully!")
        return 0

    except Exception as e:
        print(f"Error running correlation analysis demo: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
