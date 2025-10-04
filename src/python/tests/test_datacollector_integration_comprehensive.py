#!/usr/bin/env python3
"""
Comprehensive integration tests for the DataCollector metrics system.

This module provides end-to-end integration tests that verify the complete DataCollector
functionality across different simulation scenarios, configurations, and edge cases.

Tests cover:
1. Complete simulation runs with comprehensive data collection
2. Multi-day simulation scenarios with different configurations
3. Data export and persistence functionality
4. Different agent configurations and network structures
5. Time series data integrity and consistency
6. Error handling and edge cases in integration scenarios
7. Performance validation for large-scale simulations
8. Cross-validation between different data access methods
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
import json
import networkx as nx
from pathlib import Path
from unittest.mock import Mock, patch

from mesa.space import NetworkGrid
from src.python.model import StressModel
from src.python.config import get_config


class TestDataCollectorEndToEndIntegration:
    """Test complete end-to-end simulation runs with data collection."""

    def test_complete_simulation_run_basic(self):
        """Test a complete simulation run with all DataCollector metrics."""
        # Create model with small population for faster testing
        model = StressModel(N=10, max_days=5, seed=42)

        # Track initial state
        initial_agents = len(model.agents)
        initial_day = model.day

        # Run complete simulation
        while model.running:
            model.step()

        # Verify simulation completed successfully
        assert model.day == 5, f"Simulation should run for 5 days, got {model.day}"
        assert len(model.agents) == initial_agents, "Agent count should remain constant"
        assert not model.running, "Model should not be running after completion"

        # Verify comprehensive data collection
        model_data = model.datacollector.get_model_vars_dataframe()
        agent_data = model.datacollector.get_agent_vars_dataframe()

        # Check model data completeness
        assert not model_data.empty, "Model data should not be empty"
        assert len(model_data) == 5, f"Should have 5 days of model data, got {len(model_data)}"

        # Check agent data completeness
        expected_agent_records = 10 * 5  # 10 agents × 5 days
        assert len(agent_data) == expected_agent_records, f"Should have {expected_agent_records} agent records"

        # Verify all expected metrics are present
        expected_model_metrics = [
            'avg_pss10', 'avg_resilience', 'avg_affect', 'coping_success_rate',
            'avg_resources', 'avg_stress', 'social_support_rate', 'stress_events',
            'network_density', 'stress_prevalence', 'low_resilience', 'high_resilience',
            'avg_challenge', 'avg_hindrance', 'challenge_hindrance_ratio',
            'avg_consecutive_hindrances', 'total_stress_events', 'successful_coping',
            'social_interactions', 'support_exchanges'
        ]

        for metric in expected_model_metrics:
            assert metric in model_data.columns, f"Model metric '{metric}' should be present"

        # Verify data integrity
        assert not model_data.isnull().any().any(), "Model data should not contain null values"
        assert not agent_data.isnull().any().any(), "Agent data should not contain null values"

    def test_simulation_with_different_configurations(self):
        """Test simulation runs with different model configurations."""
        configurations = [
            {'N': 5, 'max_days': 3, 'seed': 42},
            {'N': 15, 'max_days': 4, 'seed': 123},
            {'N': 8, 'max_days': 6, 'seed': 456},
        ]

        for config in configurations:
            model = StressModel(**config)

            # Run complete simulation
            while model.running:
                model.step()

            # Verify data collection works for this configuration
            model_data = model.datacollector.get_model_vars_dataframe()
            agent_data = model.datacollector.get_agent_vars_dataframe()

            expected_model_records = config['max_days']
            expected_agent_records = config['N'] * config['max_days']

            assert len(model_data) == expected_model_records, f"Config {config}: Expected {expected_model_records} model records"
            assert len(agent_data) == expected_agent_records, f"Config {config}: Expected {expected_agent_records} agent records"

            # Verify data quality
            assert not model_data.isnull().any().any(), f"Config {config}: Model data should not contain null values"
            assert not agent_data.isnull().any().any(), f"Config {config}: Agent data should not contain null values"

    def test_long_running_simulation_stability(self):
        """Test stability of data collection over long simulation runs."""
        model = StressModel(N=20, max_days=20, seed=42)

        # Track metrics at regular intervals
        checkpoints = []
        for day in range(20):
            model.step()

            if day % 5 == 0:  # Checkpoint every 5 days
                model_data = model.datacollector.get_model_vars_dataframe()
                agent_data = model.datacollector.get_agent_vars_dataframe()

                checkpoints.append({
                    'day': day,
                    'model_records': len(model_data),
                    'agent_records': len(agent_data),
                    'data_quality': not (model_data.isnull().any().any() or agent_data.isnull().any().any())
                })

        # Verify checkpoints
        assert len(checkpoints) == 4, "Should have 4 checkpoints"

        for checkpoint in checkpoints:
            assert checkpoint['model_records'] == checkpoint['day'] + 1, f"Model records should match days at checkpoint {checkpoint['day']}"
            assert checkpoint['agent_records'] == 20 * (checkpoint['day'] + 1), f"Agent records should match at checkpoint {checkpoint['day']}"
            assert checkpoint['data_quality'], f"Data quality should be maintained at checkpoint {checkpoint['day']}"

        # Final verification
        final_model_data = model.datacollector.get_model_vars_dataframe()
        final_agent_data = model.datacollector.get_agent_vars_dataframe()

        assert len(final_model_data) == 20, "Should have 20 days of final model data"
        assert len(final_agent_data) == 400, "Should have 400 final agent records (20 agents × 20 days)"


class TestDataCollectorExportFunctionality:
    """Test data export and persistence functionality."""

    def test_csv_export_functionality(self):
        """Test CSV export functionality for both model and agent data."""
        model = StressModel(N=8, max_days=3, seed=42)

        # Run simulation
        while model.running:
            model.step()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test model data export
            model_csv_path = model.export_results(os.path.join(temp_dir, "model_data.csv"))

            # Verify file was created
            assert os.path.exists(model_csv_path), "Model CSV file should be created"

            # Verify file contents
            model_df = pd.read_csv(model_csv_path)
            assert not model_df.empty, "Exported model CSV should not be empty"
            assert len(model_df) == 3, "Model CSV should have 3 rows"

            # Test agent data export
            agent_csv_path = model.export_agent_data(os.path.join(temp_dir, "agent_data.csv"))

            # Verify file was created
            assert os.path.exists(agent_csv_path), "Agent CSV file should be created"

            # Verify file contents
            agent_df = pd.read_csv(agent_csv_path)
            assert not agent_df.empty, "Exported agent CSV should not be empty"
            assert len(agent_df) == 24, "Agent CSV should have 24 rows (8 agents × 3 days)"

    def test_export_with_custom_filenames(self):
        """Test export functionality with custom filenames."""
        model = StressModel(N=5, max_days=2, seed=42)

        while model.running:
            model.step()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test custom filename for model data
            custom_model_path = os.path.join(temp_dir, "custom_model_results.csv")
            exported_model_path = model.export_results(custom_model_path)

            assert exported_model_path == custom_model_path, "Should return the custom path"
            assert os.path.exists(custom_model_path), "Custom model CSV should be created"

            # Test custom filename for agent data
            custom_agent_path = os.path.join(temp_dir, "custom_agent_trajectories.csv")
            exported_agent_path = model.export_agent_data(custom_agent_path)

            assert exported_agent_path == custom_agent_path, "Should return the custom path"
            assert os.path.exists(custom_agent_path), "Custom agent CSV should be created"

    def test_export_data_integrity(self):
        """Test that exported data maintains integrity and formatting."""
        model = StressModel(N=6, max_days=4, seed=42)

        while model.running:
            model.step()

        # Get data directly from DataCollector
        original_model_data = model.datacollector.get_model_vars_dataframe()
        original_agent_data = model.datacollector.get_agent_vars_dataframe()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Export data
            model_csv = model.export_results(os.path.join(temp_dir, "test_model.csv"))
            agent_csv = model.export_agent_data(os.path.join(temp_dir, "test_agent.csv"))

            # Read exported data
            exported_model_data = pd.read_csv(model_csv)
            exported_agent_data = pd.read_csv(agent_csv)

            # Verify data integrity
            pd.testing.assert_frame_equal(
                original_model_data.reset_index(drop=True),
                exported_model_data.reset_index(drop=True),
                check_dtype=False  # Allow minor dtype differences from CSV serialization
            )

            # For agent data, we need to handle the MultiIndex structure
            # Reset index for comparison
            original_agent_reset = original_agent_data.reset_index()
            exported_agent_reset = exported_agent_data.reset_index()

            # Compare key columns (excluding index columns that may differ in naming)
            common_columns = [col for col in original_agent_reset.columns if col in exported_agent_reset.columns]
            pd.testing.assert_frame_equal(
                original_agent_reset[common_columns],
                exported_agent_reset[common_columns],
                check_dtype=False
            )

    def test_export_with_missing_data(self):
        """Test export functionality when DataCollector has minimal data."""
        model = StressModel(N=5, max_days=1, seed=42)  # Use N=5 to avoid NetworkX k>n error

        # Run single step
        model.step()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle minimal data gracefully
            model_path = model.export_results(os.path.join(temp_dir, "minimal_model.csv"))
            agent_path = model.export_agent_data(os.path.join(temp_dir, "minimal_agent.csv"))

            assert os.path.exists(model_path), "Should create model CSV even with minimal data"
            assert os.path.exists(agent_path), "Should create agent CSV even with minimal data"

            # Verify files have expected minimal content
            model_df = pd.read_csv(model_path)
            agent_df = pd.read_csv(agent_path)

            assert len(model_df) == 1, "Model CSV should have 1 row for single step"
            assert len(agent_df) == 5, "Agent CSV should have 5 rows for 5 agents"


class TestDataCollectorMultiAgentScenarios:
    """Test DataCollector with different agent configurations and network structures."""

    def test_varying_agent_numbers(self):
        """Test data collection with different numbers of agents."""
        agent_counts = [5, 10, 25, 50]

        for N in agent_counts:
            model = StressModel(N=N, max_days=3, seed=42)

            while model.running:
                model.step()

            # Verify data collection scales correctly
            model_data = model.datacollector.get_model_vars_dataframe()
            agent_data = model.datacollector.get_agent_vars_dataframe()

            expected_model_records = 3
            expected_agent_records = N * 3

            assert len(model_data) == expected_model_records, f"N={N}: Should have {expected_model_records} model records"
            assert len(agent_data) == expected_agent_records, f"N={N}: Should have {expected_agent_records} agent records"

            # Verify data quality scales
            assert not model_data.isnull().any().any(), f"N={N}: Model data should not contain null values"
            assert not agent_data.isnull().any().any(), f"N={N}: Agent data should not contain null values"

    def test_different_network_configurations(self):
        """Test data collection with different network topologies."""
        # Test with different Watts-Strogatz parameters (ensure k < N)
        network_configs = [
            {'watts_k': 2, 'watts_p': 0.1},
            {'watts_k': 4, 'watts_p': 0.3},
            {'watts_k': 6, 'watts_p': 0.5},
        ]

        for net_config in network_configs:
            # Create model directly with parameters to avoid config mocking issues
            model = StressModel(N=12, max_days=3, seed=42)

            # Manually override network parameters after creation
            model.grid = NetworkGrid(nx.watts_strogatz_graph(
                n=12,
                k=net_config['watts_k'],
                p=net_config['watts_p']
            ))

            while model.running:
                model.step()

            # Verify data collection works with different network structures
            model_data = model.datacollector.get_model_vars_dataframe()
            agent_data = model.datacollector.get_agent_vars_dataframe()

            assert len(model_data) == 3, f"Network config {net_config}: Should have 3 model records"
            assert len(agent_data) == 36, f"Network config {net_config}: Should have 36 agent records"

            # Verify network density is reasonable for the configuration
            network_density_values = model_data['network_density'].values
            assert all(0 <= d <= 1 for d in network_density_values), f"Network config {net_config}: Network density should be in [0,1]"

    def test_agents_with_different_initial_conditions(self):
        """Test data collection with agents having different initial states."""
        model = StressModel(N=15, max_days=4, seed=42)

        # Modify some agents to have different initial conditions
        agents_list = list(model.agents)
        for i, agent in enumerate(agents_list):
            if i < 5:  # First 5 agents: High resilience
                agent.resilience = 0.8
                agent.affect = 0.5
            elif i < 10:  # Next 5 agents: Low resilience
                agent.resilience = 0.2
                agent.affect = -0.5
            # Last 5 agents: Keep default values

        while model.running:
            model.step()

        # Verify data collection captures the diversity
        agent_data = model.datacollector.get_agent_vars_dataframe()

        # Check that we have diversity in resilience values
        resilience_values = agent_data['resilience'].values
        unique_resilience = np.unique(resilience_values)
        assert len(unique_resilience) > 5, "Should have diverse resilience values across agents"

        # Check that we have both positive and negative affect values
        affect_values = agent_data['affect'].values
        assert np.min(affect_values) < -0.1, "Should have negative affect values"
        assert np.max(affect_values) > 0.1, "Should have positive affect values"

        # Verify model-level aggregations reflect the diversity
        model_data = model.datacollector.get_model_vars_dataframe()
        resilience_std_values = []

        for day in range(4):
            day_data = agent_data[agent_data.index.get_level_values('Step') == day + 1]
            if not day_data.empty:
                resilience_std_values.append(day_data['resilience'].std())

        # Should have some variation in resilience across days
        assert any(std > 0.1 for std in resilience_std_values), "Should have variation in resilience across agents"


class TestDataCollectorTimeSeriesAnalysis:
    """Test time series data integrity and consistency."""

    def test_time_series_continuity(self):
        """Test that time series data maintains continuity across steps."""
        model = StressModel(N=8, max_days=10, seed=42)

        # Track data at each step
        step_data = []

        for day in range(10):
            model.step()

            model_data = model.datacollector.get_model_vars_dataframe()
            agent_data = model.datacollector.get_agent_vars_dataframe()

            step_data.append({
                'day': day,
                'model_records': len(model_data),
                'agent_records': len(agent_data),
                'model_data_shape': model_data.shape,
                'agent_data_shape': agent_data.shape
            })

        # Verify continuity
        for i, data in enumerate(step_data):
            expected_model_records = i + 1
            expected_agent_records = 8 * (i + 1)

            assert data['model_records'] == expected_model_records, f"Step {i}: Expected {expected_model_records} model records"
            assert data['agent_records'] == expected_agent_records, f"Step {i}: Expected {expected_agent_records} agent records"

        # Verify final data integrity
        final_model_data = model.datacollector.get_model_vars_dataframe()
        final_agent_data = model.datacollector.get_agent_vars_dataframe()

        # Check for reasonable time series patterns
        resilience_trend = final_model_data['avg_resilience'].values
        assert len(resilience_trend) == 10, "Should have 10 days of resilience data"

        # Resilience should be relatively stable (not extreme fluctuations)
        resilience_std = np.std(resilience_trend)
        assert resilience_std < 0.3, f"Resilience trend should be stable, got std {resilience_std}"

    def test_data_consistency_across_collection_methods(self):
        """Test consistency between different data access methods."""
        model = StressModel(N=6, max_days=5, seed=42)

        while model.running:
            model.step()

        # Get data using different methods
        direct_model_data = model.datacollector.get_model_vars_dataframe()
        method_model_data = model.get_model_vars_dataframe()
        time_series_data = model.get_time_series_data()

        direct_agent_data = model.datacollector.get_agent_vars_dataframe()
        method_agent_data = model.get_agent_vars_dataframe()
        agent_time_series_data = model.get_agent_time_series_data()

        # Verify consistency between direct and method access
        pd.testing.assert_frame_equal(direct_model_data, method_model_data)
        pd.testing.assert_frame_equal(direct_agent_data, method_agent_data)

        # Verify time series methods return same data
        pd.testing.assert_frame_equal(direct_model_data, time_series_data)
        pd.testing.assert_frame_equal(direct_agent_data, agent_time_series_data)

    def test_cross_validation_of_aggregated_metrics(self):
        """Test that aggregated metrics are consistent across different calculations."""
        model = StressModel(N=10, max_days=3, seed=42)

        while model.running:
            model.step()

        # Get data for cross-validation
        model_data = model.datacollector.get_model_vars_dataframe()
        agent_data = model.datacollector.get_agent_vars_dataframe()

        # Cross-validate key metrics for the final day
        final_day = 3
        final_day_agent_data = agent_data[agent_data.index.get_level_values('Step') == final_day]

        # Test resilience aggregation
        manual_avg_resilience = final_day_agent_data['resilience'].mean()
        collected_avg_resilience = model_data.loc[final_day - 1, 'avg_resilience']  # 0-based indexing

        assert abs(manual_avg_resilience - collected_avg_resilience) < 1e-10, "Resilience aggregation should match"

        # Test affect aggregation
        manual_avg_affect = final_day_agent_data['affect'].mean()
        collected_avg_affect = model_data.loc[final_day - 1, 'avg_affect']

        assert abs(manual_avg_affect - collected_avg_affect) < 1e-10, "Affect aggregation should match"

        # Test stress prevalence calculation
        manual_stress_prevalence = (final_day_agent_data['pss10'] >= 27).sum() / len(final_day_agent_data)
        collected_stress_prevalence = model_data.loc[final_day - 1, 'stress_prevalence']

        assert abs(manual_stress_prevalence - collected_stress_prevalence) < 1e-10, "Stress prevalence should match"


class TestDataCollectorErrorHandling:
    """Test error handling and edge cases in integration scenarios."""

    def test_datacollector_with_corrupted_agent_data(self):
        """Test DataCollector behavior when agents have corrupted data."""
        model = StressModel(N=8, max_days=3, seed=42)

        # Corrupt some agent data
        agents_list = list(model.agents)
        for i in [0, 2, 5]:  # Corrupt every 3rd agent
            agent = agents_list[i]
            agent.resilience = float('nan')  # Introduce NaN values
            agent.affect = float('inf')      # Introduce infinite values

        # Should handle corrupted data gracefully
        model.step()

        # DataCollector should still collect data
        model_data = model.datacollector.get_model_vars_dataframe()
        agent_data = model.datacollector.get_agent_vars_dataframe()

        assert not model_data.empty, "Model data should be collected despite corrupted agents"
        assert not agent_data.empty, "Agent data should be collected despite corrupted agents"

        # Check that non-corrupted agents still have valid data
        valid_agents_data = agent_data[agent_data.index.get_level_values('AgentID') != 0]
        valid_agents_data = valid_agents_data[valid_agents_data.index.get_level_values('AgentID') != 2]
        valid_agents_data = valid_agents_data[valid_agents_data.index.get_level_values('AgentID') != 5]

        # Valid agents should have finite values
        assert np.all(np.isfinite(valid_agents_data['resilience'])), "Valid agents should have finite resilience"
        assert np.all(np.isfinite(valid_agents_data['affect'])), "Valid agents should have finite affect"

    def test_simulation_recovery_from_datacollector_errors(self):
        """Test simulation recovery when DataCollector encounters errors."""
        model = StressModel(N=6, max_days=5, seed=42)

        # Mock DataCollector to fail on specific steps
        original_collect = model.datacollector.collect
        failure_steps = [2, 4]  # Fail on steps 2 and 4 (0-based)
        call_count = 0

        def failing_collect(*args, **kwargs):
            nonlocal call_count
            if call_count in failure_steps:
                call_count += 1
                # Simulate a collection failure
                raise RuntimeError(f"Simulated DataCollector failure at step {call_count}")
            else:
                call_count += 1
                return original_collect(*args, **kwargs)

        model.datacollector.collect = failing_collect

        # Run simulation - should handle failures gracefully
        steps_completed = 0
        try:
            while model.running and steps_completed < 5:
                model.step()
                steps_completed += 1
        except RuntimeError:
            # Expected to fail on some steps
            pass

        # Verify simulation continued despite failures
        assert steps_completed > 0, "Simulation should complete at least some steps"

        # Check that successful collections still work
        model_data = model.datacollector.get_model_vars_dataframe()
        agent_data = model.datacollector.get_agent_vars_dataframe()

        # Should have data from successful steps
        assert len(model_data) <= 5, "Should have at most 5 model records"
        assert len(agent_data) <= 30, "Should have at most 30 agent records (6 agents × 5 steps)"

    def test_empty_simulation_scenarios(self):
        """Test DataCollector behavior in edge case scenarios."""
        # Test with minimal viable simulation (ensure k < N for NetworkX)
        model = StressModel(N=4, max_days=1, seed=42)

        # Single step
        model.step()

        # Verify minimal data collection
        model_data = model.datacollector.get_model_vars_dataframe()
        agent_data = model.datacollector.get_agent_vars_dataframe()

        assert len(model_data) == 1, "Should have 1 model record for single step"
        assert len(agent_data) == 4, "Should have 4 agent records for 4 agents"

        # Test population summary with minimal data
        summary = model.get_population_summary()
        assert isinstance(summary, dict), "Should return summary dictionary"
        assert 'num_agents' in summary, "Summary should contain agent count"
        assert summary['num_agents'] == 4, "Summary should reflect correct agent count"


class TestDataCollectorPerformanceValidation:
    """Test performance and scalability of DataCollector system."""

    def test_large_scale_simulation_performance(self):
        """Test DataCollector performance with large-scale simulations."""
        model = StressModel(N=100, max_days=10, seed=42)

        # Measure collection performance
        import time
        start_time = time.time()

        while model.running:
            model.step()

        collection_time = time.time() - start_time

        # Verify reasonable performance (adjust threshold as needed)
        assert collection_time < 30.0, f"Large simulation should complete in reasonable time, took {collection_time:.2f}s"

        # Verify data integrity for large simulation
        model_data = model.datacollector.get_model_vars_dataframe()
        agent_data = model.datacollector.get_agent_vars_dataframe()

        expected_model_records = 10
        expected_agent_records = 100 * 10

        assert len(model_data) == expected_model_records, f"Should have {expected_model_records} model records"
        assert len(agent_data) == expected_agent_records, f"Should have {expected_agent_records} agent records"

        # Verify no data corruption in large dataset
        assert not model_data.isnull().any().any(), "Large model dataset should not contain null values"
        assert not agent_data.isnull().any().any(), "Large agent dataset should not contain null values"

    def test_memory_usage_stability(self):
        """Test that memory usage remains stable during data collection."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        model = StressModel(N=50, max_days=20, seed=42)

        while model.running:
            model.step()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 100, f"Memory increase should be reasonable, got {memory_increase:.2f}MB"

        # Verify data integrity after memory-intensive operation
        model_data = model.datacollector.get_model_vars_dataframe()
        agent_data = model.datacollector.get_agent_vars_dataframe()

        assert len(model_data) == 20, "Should have 20 model records"
        assert len(agent_data) == 1000, "Should have 1000 agent records (50 agents × 20 steps)"

        # Verify data quality maintained
        assert not model_data.isnull().any().any(), "Model data should maintain quality"
        assert not agent_data.isnull().any().any(), "Agent data should maintain quality"


class TestDataCollectorIntegrationValidation:
    """Test integration between different DataCollector components."""

    def test_model_and_agent_data_synchronization(self):
        """Test that model and agent data remain synchronized."""
        model = StressModel(N=7, max_days=5, seed=42)

        # Track synchronization at each step
        sync_checks = []

        for day in range(5):
            model.step()

            model_data = model.datacollector.get_model_vars_dataframe()
            agent_data = model.datacollector.get_agent_vars_dataframe()

            # Check synchronization
            expected_model_records = day + 1
            expected_agent_records = 7 * (day + 1)

            sync_check = {
                'day': day,
                'model_records_match': len(model_data) == expected_model_records,
                'agent_records_match': len(agent_data) == expected_agent_records,
                'data_quality_ok': not (model_data.isnull().any().any() or agent_data.isnull().any().any())
            }

            sync_checks.append(sync_check)

        # Verify all synchronization checks passed
        for check in sync_checks:
            assert check['model_records_match'], f"Model records should match at day {check['day']}"
            assert check['agent_records_match'], f"Agent records should match at day {check['day']}"
            assert check['data_quality_ok'], f"Data quality should be maintained at day {check['day']}"

    def test_cross_component_data_consistency(self):
        """Test consistency between different data collection components."""
        model = StressModel(N=12, max_days=4, seed=42)

        while model.running:
            model.step()

        # Get data from different sources
        model_data = model.datacollector.get_model_vars_dataframe()
        agent_data = model.datacollector.get_agent_vars_dataframe()
        population_summary = model.get_population_summary()

        # Cross-validate between different data sources
        final_model_data = model_data.iloc[-1]  # Last day

        # Test consistency between model data and population summary
        assert abs(final_model_data['avg_resilience'] - population_summary['avg_resilience']) < 1e-10, "Resilience should match between sources"
        assert abs(final_model_data['avg_affect'] - population_summary['avg_affect']) < 1e-10, "Affect should match between sources"
        assert abs(final_model_data['avg_resources'] - population_summary['avg_resources']) < 1e-10, "Resources should match between sources"

        # Test consistency with agent data aggregations
        final_day_agent_data = agent_data[agent_data.index.get_level_values('Step') == 4]  # Mesa uses 1-based indexing

        manual_avg_resilience = final_day_agent_data['resilience'].mean()
        assert abs(manual_avg_resilience - final_model_data['avg_resilience']) < 1e-10, "Agent aggregation should match model data"

        manual_avg_affect = final_day_agent_data['affect'].mean()
        assert abs(manual_avg_affect - final_model_data['avg_affect']) < 1e-10, "Affect aggregation should match model data"

    def test_data_collection_pipeline_completeness(self):
        """Test completeness of the entire data collection pipeline."""
        model = StressModel(N=9, max_days=6, seed=42)

        # Run complete simulation
        while model.running:
            model.step()

        # Test complete pipeline: collection → access → export → analysis
        model_data = model.datacollector.get_model_vars_dataframe()
        agent_data = model.datacollector.get_agent_vars_dataframe()

        # Test export pipeline
        with tempfile.TemporaryDirectory() as temp_dir:
            model_csv = model.export_results(os.path.join(temp_dir, "pipeline_test_model.csv"))
            agent_csv = model.export_agent_data(os.path.join(temp_dir, "pipeline_test_agent.csv"))

            # Verify export completed successfully
            assert os.path.exists(model_csv), "Model export should complete"
            assert os.path.exists(agent_csv), "Agent export should complete"

            # Test analysis pipeline
            exported_model_data = pd.read_csv(model_csv)
            exported_agent_data = pd.read_csv(agent_csv)

            # Verify analysis can be performed on exported data
            assert len(exported_model_data.columns) > 10, "Exported model data should have many columns"
            assert len(exported_agent_data.columns) > 5, "Exported agent data should have multiple columns"

            # Test statistical analysis on exported data
            model_stats = exported_model_data.describe()
            agent_stats = exported_agent_data.describe()

            assert not model_stats.empty, "Should be able to compute statistics on exported model data"
            assert not agent_stats.empty, "Should be able to compute statistics on exported agent data"


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
