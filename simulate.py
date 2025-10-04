#!/usr/bin/env python3
"""
Comprehensive Agent-Based Mental Health Simulation Runner

This script implements a complete simulation workflow for the mental health ABM,
including configuration management, data extraction, and result persistence.

Features:
- Configuration-driven simulation setup using .env file
- Comprehensive data collection using Mesa DataCollector
- Automatic directory creation and file I/O error handling
- Detailed logging for debugging and monitoring
- Reproducible results with proper random seeding
- Model and agent-level data export to CSV files

Usage:
   python simulate.py

Output:
   - data/raw/model.csv: Population-level time series data
   - data/raw/agent.csv: Individual agent time series data
   - Console logging with simulation progress and statistics
"""

## IMPORT MODULES

import os
import sys
import logging
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Import project modules
from src.python.model import StressModel
from src.python.config import get_config, ConfigurationError


## LOGGING CONFIGURATION

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
   """
   Set up logging configuration for the simulation.

   Args:
       log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

   Returns:
       Configured logger instance
   """
   # Create formatter
   formatter = logging.Formatter(
       '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       datefmt='%Y-%m-%d %H:%M:%S'
   )

   # Set up console handler
   console_handler = logging.StreamHandler(sys.stdout)
   console_handler.setFormatter(formatter)

   # Configure root logger
   logger = logging.getLogger('simulation')
   logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
   logger.addHandler(console_handler)

   return logger


## UTILITY FUNCTIONS

def ensure_directory_exists(directory_path: str) -> bool:
   """
   Ensure a directory exists, creating it if necessary.

   Args:
       directory_path: Path to the directory

   Returns:
       True if directory exists or was created successfully, False otherwise
   """
   try:
       path = Path(directory_path)
       path.mkdir(parents=True, exist_ok=True)
       return True
   except (OSError, PermissionError) as e:
       print(f"Error creating directory {directory_path}: {e}")
       return False


def set_random_seeds(seed: int) -> None:
   """
   Set random seeds for reproducible results across all libraries.

   Args:
       seed: Random seed value
   """
   random.seed(seed)
   np.random.seed(seed)
   # Note: Mesa uses Python's random module, so setting random.seed() is sufficient


def save_dataframe_with_error_handling(
   df: pd.DataFrame,
   filepath: str,
   logger: logging.Logger,
   description: str = "data"
) -> bool:
   """
   Save DataFrame to CSV with comprehensive error handling.

   Args:
       df: DataFrame to save
       filepath: Output file path
       logger: Logger instance for error reporting
       description: Description of the data being saved (for logging)

   Returns:
       True if save was successful, False otherwise
   """
   try:
       # Ensure parent directory exists
       parent_dir = Path(filepath).parent
       if not ensure_directory_exists(str(parent_dir)):
           logger.error(f"Failed to create directory for {filepath}")
           return False

       # Save DataFrame to CSV
       df.to_csv(filepath, index=False)
       logger.info(f"Successfully saved {description} to {filepath} "
                  f"(shape: {df.shape})")
       return True

   except Exception as e:
       logger.error(f"Error saving {description} to {filepath}: {e}")
       return False


## MAIN SIMULATION FUNCTION

def run_simulation(
   num_agents: int = 10,
   max_steps: int = 100,
   seed: Optional[int] = None,
   logger: Optional[logging.Logger] = None
) -> tuple[StressModel, pd.DataFrame, pd.DataFrame]:
   """
   Run the complete mental health simulation.

   Args:
       num_agents: Number of agents in the simulation
       max_steps: Maximum number of simulation steps (days)
       seed: Random seed for reproducibility (uses config default if None)
       logger: Logger instance (creates default if None)

   Returns:
       Tuple of (model, model_data, agent_data)

   Raises:
       ConfigurationError: If configuration loading fails
       RuntimeError: If simulation fails to complete
   """
   if logger is None:
       logger = setup_logging()

   try:
       # Load configuration
       logger.info("Loading configuration from .env file...")
       config = get_config()
       logger.info("Configuration loaded successfully")

       # Override configuration with provided parameters
       if seed is None:
           seed = config.get('simulation', 'seed')

       logger.info(f"Simulation parameters: agents={num_agents}, steps={max_steps}, seed={seed}")

       # Set random seeds for reproducibility
       set_random_seeds(seed)
       logger.info(f"Random seeds set to {seed}")

       # Initialize the simulation model
       logger.info("Initializing StressModel...")
       model = StressModel(N=num_agents, max_days=max_steps, seed=seed)

       logger.info(f"Model initialized with {len(model.agents)} agents")
       logger.info("Starting simulation...")

       # Run simulation loop
       step_count = 0
       while model.running and step_count < max_steps:
           model.step()
           step_count += 1

           # Log progress every 10 steps
           if step_count % 10 == 0:
               logger.info(f"Completed step {step_count}/{max_steps}")

       logger.info(f"Simulation completed after {model.day} days")

       # Extract data using DataCollector methods
       logger.info("Extracting model data...")
       model_data = model.get_time_series_data()

       logger.info("Extracting agent data...")
       agent_data = model.get_agent_time_series_data()

       logger.info(f"Data extraction complete - Model: {model_data.shape}, Agent: {agent_data.shape}")

       return model, model_data, agent_data

   except ConfigurationError as e:
       logger.error(f"Configuration error: {e}")
       raise
   except Exception as e:
       logger.error(f"Simulation error: {e}")
       raise RuntimeError(f"Simulation failed: {e}")


## MAIN EXECUTION

def main():
   """Main execution function."""
   # Set up logging
   logger = setup_logging()

   logger.info("="*60)
   logger.info("MENTAL HEALTH ABM SIMULATION STARTED")
   logger.info("="*60)

   try:
       # Run simulation with specified parameters
       model, model_data, agent_data = run_simulation(
           num_agents=1000,
           max_steps=1000,
           logger=logger
       )

       # Save results to CSV files
       logger.info("Saving results to CSV files...")

       # Define output paths
       raw_dir = "data/raw"
       model_csv_path = f"{raw_dir}/model.csv"
       agent_csv_path = f"{raw_dir}/agent.csv"

       # Save model data
       model_success = save_dataframe_with_error_handling(
           model_data, model_csv_path, logger, "model data"
       )

       # Save agent data
       agent_success = save_dataframe_with_error_handling(
           agent_data, agent_csv_path, logger, "agent data"
       )

       if model_success and agent_success:
           logger.info("All data saved successfully!")
           logger.info(f"Model data: {model_csv_path}")
           logger.info(f"Agent data: {agent_csv_path}")
       else:
           logger.error("Some data files failed to save")
           return 1

       # Print summary statistics
       logger.info("Simulation Summary:")
       logger.info(f"  - Agents: {len(model.agents)}")
       logger.info(f"  - Days simulated: {model.day}")
       logger.info(f"  - Model data points: {len(model_data)}")
       logger.info(f"  - Agent data points: {len(agent_data)}")

       # Print final population summary if available
       try:
           summary = model.get_population_summary()
           if summary:
               logger.info("Final Population State:")
               logger.info(f"  - Average PSS-10: {summary.get('avg_pss10', 'N/A')}")
               logger.info(f"  - Average resilience: {summary.get('avg_resilience', 'N/A')}")
               logger.info(f"  - Average affect: {summary.get('avg_affect', 'N/A')}")
               logger.info(f"  - Stress prevalence: {summary.get('stress_prevalence', 'N/A')}")
       except Exception as e:
           logger.warning(f"Could not generate population summary: {e}")

       logger.info("="*60)
       logger.info("SIMULATION COMPLETED SUCCESSFULLY")
       logger.info("="*60)

       return 0

   except Exception as e:
       logger.error(f"Simulation failed: {e}")
       logger.info("="*60)
       logger.error("SIMULATION FAILED")
       logger.info("="*60)
       return 1


if __name__ == "__main__":
   """Script entry point."""
   exit_code = main()
   sys.exit(exit_code)
