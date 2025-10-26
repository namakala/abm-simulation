#!/usr/bin/env python3
"""
Test script to verify the model properly integrates with new stress processing mechanisms.
"""

import numpy as np
import pytest
import pandas as pd
import networkx as nx
from unittest.mock import patch, MagicMock

from src.python.model import StressModel
from src.python.config import get_config

def test_model_integration():
    """Test that the model works with new stress processing mechanisms."""
    print("Testing model integration with new stress processing mechanisms...")

    # Create a small model for testing
    model = StressModel(N=10, max_days=3, seed=42)

    print(f"Created model with {len(model.agents)} agents")

    # Run a few steps
    for day in range(3):
        print(f"\n--- Day {day} ---")
        model.step()

        # Get population summary
        summary = model.get_population_summary()

        print(f"Avg affect: {summary['avg_affect']:.3f}")
        print(f"Avg resilience: {summary['avg_resilience']:.3f}")
        print(f"Avg stress: {summary['avg_stress']:.3f}")
        print(f"Avg challenge: {summary['avg_challenge']:.3f}")
        print(f"Avg hindrance: {summary['avg_hindrance']:.3f}")
        print(f"Coping success rate: {summary['coping_success_rate']:.3f}")
        print(f"Stress events: {summary.get('stress_events', 'N/A')}")

        # Verify that new metrics are being collected
        assert 'avg_stress' in summary, "avg_stress metric missing"
        assert 'avg_challenge' in summary, "avg_challenge metric missing"
        assert 'avg_hindrance' in summary, "avg_hindrance metric missing"
        assert 'coping_success_rate' in summary, "coping_success_rate metric missing"

        # Verify metrics are in valid ranges
        assert -1.0 <= summary['avg_affect'] <= 1.0, "Affect out of range"
        assert 0.0 <= summary['avg_resilience'] <= 1.0, "Resilience out of range"
        assert 0.0 <= summary['avg_stress'] <= 1.0, "Stress out of range"
        assert 0.0 <= summary['avg_challenge'] <= 1.0, "Challenge out of range"
        assert 0.0 <= summary['avg_hindrance'] <= 1.0, "Hindrance out of range"
        assert 0.0 <= summary['coping_success_rate'] <= 1.0, "Coping rate out of range"

    # Test time series data collection
    time_series = model.get_time_series_data()
    print(f"\nTime series data shape: {time_series.shape}")

    # Verify new columns are in time series
    expected_columns = [
        'avg_stress', 'avg_challenge', 'avg_hindrance',
        'coping_success_rate', 'avg_consecutive_hindrances', 'challenge_hindrance_ratio'
    ]

    for col in expected_columns:
        assert col in time_series.columns, f"Missing column: {col}"

    print(f"✅ All expected columns present: {list(time_series.columns)}")
    print("✅ Model integration test passed!")

def test_model_initialization_edge_cases():
    """Test model initialization with edge cases."""
    # Test with N=0 (should handle gracefully) - skip network creation
    with patch('src.python.model.nx.watts_strogatz_graph') as mock_graph:
        mock_graph.return_value = nx.Graph()  # Empty graph for N=0
        model = StressModel(N=0, max_days=1, seed=42)
        assert len(model.agents) == 0
        assert model.num_agents == 0

    # Test with max_days=0
    model = StressModel(N=5, max_days=0, seed=42)
    assert model.max_days == 0
    assert model.running == True  # Should still be running initially

    # Test with None parameters (should use config defaults)
    model = StressModel()
    assert model.num_agents > 0  # Should use config default
    assert model.max_days > 0

def test_datacollector_initialization():
    """Test DataCollector initialization."""
    model = StressModel(N=5, max_days=1, seed=42)
    assert hasattr(model, 'datacollector')
    assert model.datacollector is not None

    # Test model reporters
    model_reporters = model.datacollector.model_reporters
    expected_reporters = ['avg_pss10', 'avg_resilience', 'avg_affect', 'coping_success_rate']
    for reporter in expected_reporters:
        assert reporter in model_reporters

    # Test agent reporters
    agent_reporters = model.datacollector.agent_reporters
    expected_agent_reporters = ['pss10', 'resilience', 'affect', 'resources']
    for reporter in expected_agent_reporters:
        assert reporter in agent_reporters

def test_step_method_edge_cases():
    """Test step method with edge cases."""
    # Test with empty agents
    with patch('src.python.model.nx.watts_strogatz_graph') as mock_graph:
        mock_graph.return_value = nx.Graph()
        model = StressModel(N=0, max_days=5, seed=42)
        initial_day = model.day
        model.step()
        assert model.day == initial_day + 1

    # Test termination after max_days
    model = StressModel(N=5, max_days=2, seed=42)
    assert model.running == True
    model.step()
    assert model.running == True
    model.step()
    assert model.running == False

def test_get_avg_pss10_error_handling():
    """Test get_avg_pss10 with error handling."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Test with missing pss10 attribute
    for agent in model.agents:
        delattr(agent, 'pss10') if hasattr(agent, 'pss10') else None

    avg_pss10 = model.get_avg_pss10()
    assert avg_pss10 == 0.0  # Should use default

    # Test with None pss10
    for agent in model.agents:
        agent.pss10 = None
    avg_pss10 = model.get_avg_pss10()
    assert avg_pss10 == 0.0

    # Test with invalid pss10
    for agent in model.agents:
        agent.pss10 = "invalid"
    avg_pss10 = model.get_avg_pss10()
    assert avg_pss10 == 0.0

def test_get_avg_resilience_error_handling():
    """Test get_avg_resilience with error handling."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Test with missing resilience
    for agent in model.agents:
        delattr(agent, 'resilience') if hasattr(agent, 'resilience') else None

    avg_resilience = model.get_avg_resilience()
    assert avg_resilience == 0.0

    # Test with invalid resilience
    for agent in model.agents:
        agent.resilience = "invalid"
    avg_resilience = model.get_avg_resilience()
    assert avg_resilience == 0.0

def test_calculate_network_density_edge_cases():
    """Test _calculate_network_density with edge cases."""
    # Test with N=1 (no possible connections)
    with patch('src.python.model.nx.watts_strogatz_graph') as mock_graph:
        mock_graph.return_value = nx.Graph()
        model = StressModel(N=1, max_days=1, seed=42)
        density = model._calculate_network_density()
        assert density == 0.0

    # Test with N=0
    with patch('src.python.model.nx.watts_strogatz_graph') as mock_graph:
        mock_graph.return_value = nx.Graph()
        model = StressModel(N=0, max_days=1, seed=42)
        density = model._calculate_network_density()
        assert density == 0.0

def test_challenge_hindrance_ratio_edge_cases():
    """Test _get_challenge_hindrance_ratio with edge cases."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Test with no events
    ratio = model._get_challenge_hindrance_ratio()
    assert ratio == 0.0

    # Test with zero sum (challenge + hindrance = 0)
    # Mock events with zero challenge and hindrance
    for agent in model.agents:
        agent.daily_stress_events = [{'challenge': 0.0, 'hindrance': 0.0}]

    ratio = model._get_challenge_hindrance_ratio()
    assert ratio == 0.0

def test_calculate_social_support_rate_edge_cases():
    """Test _calculate_social_support_rate with edge cases."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Test with zero interactions
    model.total_interactions = 0
    model.social_support_exchanges = 0
    rate = model._calculate_social_support_rate()
    assert rate == 0.0

def test_population_summary_empty_agents():
    """Test get_population_summary with empty agents."""
    with patch('src.python.model.nx.watts_strogatz_graph') as mock_graph:
        mock_graph.return_value = nx.Graph()
        model = StressModel(N=0, max_days=1, seed=42)
        summary = model.get_population_summary()
        assert summary == {}

def test_population_summary_datacollector_none():
    """Test get_population_summary when datacollector is None."""
    model = StressModel(N=5, max_days=1, seed=42)
    model.datacollector = None
    summary = model.get_population_summary()
    assert summary == {}

def test_population_summary_empty_dataframe():
    """Test get_population_summary with empty DataCollector dataframe."""
    model = StressModel(N=5, max_days=1, seed=42)
    with patch.object(model.datacollector, 'get_model_vars_dataframe', return_value=pd.DataFrame()):
        summary = model.get_population_summary()
        assert summary == {}

def test_export_results_edge_cases():
    """Test export_results with edge cases."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Test with custom filename
    filename = model.export_results("test_export.csv")
    assert filename == "test_export.csv"

    # Test with empty data
    model.datacollector = None
    filename = model.export_results()
    assert filename.startswith("simulation_results_day_")

def test_export_agent_data_edge_cases():
    """Test export_agent_data with edge cases."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Test with custom filename
    filename = model.export_agent_data("test_agent.csv")
    assert filename == "test_agent.csv"

    # Test with empty data
    model.datacollector = None
    filename = model.export_agent_data()
    assert filename.startswith("agent_data_day_")

def test_get_success_rate_edge_cases():
    """Test get_success_rate with edge cases."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Test with no events
    for agent in model.agents:
        agent.daily_stress_events = []
    rate = model.get_success_rate()
    assert rate == 0.0

    # Test with invalid event structure
    for agent in model.agents:
        agent.daily_stress_events = [{'invalid': 'data'}]
    rate = model.get_success_rate()
    assert rate == 0.0

def test_network_adaptation_methods():
    """Test network adaptation methods."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Test _apply_network_adaptation
    adaptation_count = model._apply_network_adaptation()
    assert isinstance(adaptation_count, int)
    assert adaptation_count >= 0

    # Test get_network_adaptation_summary
    summary = model.get_network_adaptation_summary()
    assert 'agents_considering_adaptation' in summary
    assert 'adaptation_rate' in summary
    assert 0.0 <= summary['adaptation_rate'] <= 1.0

def test_get_agent_time_series_data_edge_cases():
    """Test get_agent_time_series_data with edge cases."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Test with None datacollector
    model.datacollector = None
    df = model.get_agent_time_series_data()
    assert df.empty

    # Test with empty datacollector
    model.datacollector = MagicMock()
    model.datacollector.get_agent_vars_dataframe.return_value = pd.DataFrame()
    df = model.get_agent_time_series_data()
    assert df.empty

def test_model_vars_dataframe_edge_cases():
    """Test get_model_vars_dataframe with edge cases."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Test with None datacollector
    model.datacollector = None
    df = model.get_model_vars_dataframe()
    assert df.empty

def test_agent_vars_dataframe_edge_cases():
    """Test get_agent_vars_dataframe with edge cases."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Test with None datacollector
    model.datacollector = None
    df = model.get_agent_vars_dataframe()
    assert df.empty

def test_get_avg_pss10_empty_values():
    """Test get_avg_pss10 when no valid pss10 values."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Set all pss10 to None
    for agent in model.agents:
        agent.pss10 = None

    avg_pss10 = model.get_avg_pss10()
    assert avg_pss10 == 0.0  # Should use default

def test_get_avg_resilience_empty_values():
    """Test get_avg_resilience when no valid resilience values."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Remove resilience attribute
    for agent in model.agents:
        delattr(agent, 'resilience')

    avg_resilience = model.get_avg_resilience()
    assert avg_resilience == 0.0

def test_get_avg_affect_empty_values():
    """Test get_avg_affect when no valid affect values."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Remove affect attribute
    for agent in model.agents:
        delattr(agent, 'affect')

    avg_affect = model.get_avg_affect()
    assert avg_affect == 0.0

def test_get_success_rate_zero_attempts():
    """Test get_success_rate when no attempts."""
    model = StressModel(N=5, max_days=1, seed=42)

    # No stress events
    for agent in model.agents:
        agent.daily_stress_events = []

    rate = model.get_success_rate()
    assert rate == 0.0

def test_network_density_zero_connections():
    """Test _calculate_network_density with zero connections."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Mock empty graph
    model.grid.G = nx.Graph()
    density = model._calculate_network_density()
    assert density == 0.0

def test_apply_network_adaptation_with_adapted_agents():
    """Test _apply_network_adaptation when agents have adapted."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Set some agents as adapted
    for agent in model.agents[:2]:
        agent._adapted_network = True

    adaptation_count = model._apply_network_adaptation()
    assert adaptation_count == 2

def test_population_summary_with_empty_agent_data():
    """Test get_population_summary with empty agent data."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Mock empty agent data
    with patch.object(model.datacollector, 'get_agent_vars_dataframe', return_value=pd.DataFrame()):
        summary = model.get_population_summary()
        # When agent_data is empty, it returns {} early
        assert summary == {}

def test_export_results_with_data():
    """Test export_results with data."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Run a step to generate data
    model.step()

    filename = model.export_results("test.csv")
    assert filename == "test.csv"

def test_get_agent_time_series_data_with_error():
    """Test get_agent_time_series_data when processing fails."""
    model = StressModel(N=5, max_days=1, seed=42)

    # Mock datacollector to return data, then mock reset_index to raise error
    with patch.object(model.datacollector, 'get_agent_vars_dataframe', return_value=pd.DataFrame({'test': [1,2]})):
        with patch('pandas.DataFrame.reset_index', side_effect=Exception("Test error")):
            df = model.get_agent_time_series_data()
            # Should return the original data on error
            assert not df.empty

if __name__ == "__main__":
    test_model_integration()
