#!/usr/bin/env python3
"""
Demo script for correlation analysis of agent-level mental health simulation variables.

This script runs a simulation and calculates correlations between key agent variables
from the final epoch of the simulation:
- pss10 (Perceived Stress Scale-10)
- resilience (Individual resilience capacity)
- affect (Individual positive/negative affect balance)
- resources (Individual resource availability)
- current_stress (Current stress level)
- stress_controllability (Perceived controllability)
- stress_overload (Perceived overload)

The script demonstrates:
1. Running a simulation programmatically using the simulate.py interface
2. Extracting agent data from the final epoch of the simulation results
3. Computing correlation matrix for the specified agent variables
4. Computing summary statistics (mean, median, std, min, max) for each variable
5. Displaying results in a clear format

Usage:
    python src/python/demos/agent_correlation_analysis.py
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
    Run simulation and return the agent dataframe from the final epoch.

    Args:
        days: Number of simulation days
        agents: Number of agents in simulation
        seed: Random seed for reproducibility

    Returns:
        pandas.DataFrame: Agent data from the final epoch with individual metrics
    """
    print(f"Running simulation with {agents} agents for {days} days (seed={seed})...")

    # Create and run the model
    model = StressModel(N=agents, max_days=days, seed=seed)

    # Run simulation
    while model.running:
        model.step()

    # Get the agent dataframe and filter for final epoch
    agent_data = model.get_agent_time_series_data()
    if agent_data.empty:
        print("Warning: No agent data available")
        return pd.DataFrame()

    # Filter for the final epoch (maximum step)
    final_step = agent_data['Step'].max()
    final_epoch_data = agent_data[agent_data['Step'] == final_step].copy()

    print(f"Simulation completed. Final epoch data shape: {final_epoch_data.shape}")
    return final_epoch_data


def calculate_correlation_matrix(agent_data: pd.DataFrame, variables: list):
    """
    Calculate correlation matrix for specified agent variables.

    Args:
        agent_data: DataFrame containing agent data from final epoch
        variables: List of variable names to include in correlation analysis

    Returns:
        pandas.DataFrame: Correlation matrix
    """
    # Check if all variables exist in the data
    missing_vars = [var for var in variables if var not in agent_data.columns]
    if missing_vars:
        print(f"Warning: Variables not found in data: {missing_vars}")
        # Filter to only available variables
        variables = [var for var in variables if var in agent_data.columns]

    if not variables:
        raise ValueError("No valid variables found for correlation analysis")

    # Extract the variables and calculate correlation
    correlation_data = agent_data[variables]
    correlation_matrix = correlation_data.corr(method='pearson')

    return correlation_matrix


def print_correlation_matrix(corr_matrix: pd.DataFrame):
    """
    Print correlation matrix in a formatted way.

    Args:
        corr_matrix: Correlation matrix DataFrame
    """
    print("\n" + "="*60)
    print("AGENT-LEVEL CORRELATION MATRIX (Final Epoch)")
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


def print_summary_statistics(agent_data: pd.DataFrame, variables: list):
    """
    Print summary statistics for specified variables.

    Args:
        agent_data: DataFrame containing agent data from final epoch
        variables: List of variable names to analyze
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (Final Epoch)")
    print("="*60)

    for var in variables:
        if var in agent_data.columns:
            values = agent_data[var]
            print(f"{var}:")
            print(f"  Mean: {values.mean():.4f}")
            print(f"  Median: {values.median():.4f}")
            print(f"  Std:  {values.std():.4f}")
            print(f"  Min:  {values.min():.4f}")
            print(f"  Max:  {values.max():.4f}")
            print()


def main():
    """Main function to run the correlation analysis demo."""

    print("Mental Health ABM - Correlation Analysis Demo")
    print("=" * 50)

    # Simulation parameters (matching the requested command)
    days = 100
    agents = 30
    seed = 42  # Using default seed for reproducibility

    # Variables to analyze (agent-level from final epoch)
    variables_of_interest = [
        'pss10',
        'resilience',
        'affect',
        'resources',
        'current_stress',
        'stress_controllability',
        'stress_overload'
    ]

    try:
        # Run simulation
        agent_data = run_simulation_and_get_data(days=days, agents=agents, seed=seed)

        # Check if we have the required data
        if agent_data.empty:
            print("Error: No agent data returned from simulation")
            return 1

        print(f"\nAvailable variables in agent data: {list(agent_data.columns)}")

        # Calculate correlation matrix
        print(f"\nAnalyzing correlations for variables: {variables_of_interest}")
        corr_matrix = calculate_correlation_matrix(agent_data, variables_of_interest)

        # Print results
        print_correlation_matrix(corr_matrix)

        # Print summary statistics
        print_summary_statistics(agent_data, variables_of_interest)

        print("Demo completed successfully!")
        return 0

    except Exception as e:
        print(f"Error running agent correlation analysis demo: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
