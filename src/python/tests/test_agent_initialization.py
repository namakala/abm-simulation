"""
Comprehensive unit tests for agent initialization logic.

This file provides complete test coverage for the Person class __init__ method
and related initialization functionality, including PSS-10 setup and configuration integration.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from agent import Person
from config import get_config, Config
from math_utils import sample_normal, create_rng


class MockModel:
    """Mock Mesa model for testing."""

    def __init__(self, seed=None):
        self.seed = seed
        self.grid = Mock()
        self.grid.get_neighbors.return_value = []
        self.agents = Mock()
        self.register_agent = Mock()  # Required by Mesa Agent base class
        self.rng = np.random.default_rng(seed)  # Required by Mesa Agent base class


class TestAgentInitializationCore:
    """Test core agent initialization functionality."""

    def test_agent_initialization_normal_distributions(self):
        """Test that agent initialization uses normal distributions with correct parameters."""
        # Test with specific configuration parameters
        config = {
            'initial_resilience_mean': 0.7,
            'initial_resilience_sd': 0.1,
            'initial_affect_mean': 0.2,
            'initial_affect_sd': 0.15,
            'initial_resources_mean': 0.8,
            'initial_resources_sd': 0.05,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        }

        model = MockModel(seed=42)
        agent = Person(model, config)

        # Verify that values are reasonable for the given parameters
        # (exact values depend on the specific random seed)
        assert 0.0 <= agent.resilience <= 1.0
        assert -1.0 <= agent.affect <= 1.0
        assert 0.0 <= agent.resources <= 1.0

        # Test that baseline values are also initialized
        assert -1.0 <= agent.baseline_affect <= 1.0
        assert 0.0 <= agent.baseline_resilience <= 1.0

    def test_agent_initialization_clamping(self):
        """Test that agent initialization properly clamps values to valid ranges."""
        # Test with extreme mean values that should be clamped
        config = {
            'initial_resilience_mean': 2.0,  # Above valid range
            'initial_resilience_sd': 0.1,
            'initial_affect_mean': -3.0,     # Below valid range
            'initial_affect_sd': 0.1,
            'initial_resources_mean': -1.0,  # Below valid range
            'initial_resources_sd': 0.1,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        }

        model = MockModel(seed=42)
        agent = Person(model, config)

        # Values should be clamped to valid ranges despite extreme means
        assert 0.0 <= agent.resilience <= 1.0
        assert -1.0 <= agent.affect <= 1.0
        assert 0.0 <= agent.resources <= 1.0

        # Baseline values should also be clamped
        assert -1.0 <= agent.baseline_affect <= 1.0
        assert 0.0 <= agent.baseline_resilience <= 1.0

    def test_agent_initialization_variation(self):
        """Test that different agents get different initial values (realistic variation)."""
        config = {
            'initial_resilience_mean': 0.5,
            'initial_resilience_sd': 0.2,
            'initial_affect_mean': 0.0,
            'initial_affect_sd': 0.3,
            'initial_resources_mean': 0.6,
            'initial_resources_sd': 0.15,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        }

        # Create multiple agents with different model seeds
        agents = []
        for seed in [42, 43, 44, 45, 46]:
            model = MockModel(seed=seed)
            agent = Person(model, config)
            agents.append(agent)

        # Agents should have different initial values due to different seeds
        resilience_values = [agent.resilience for agent in agents]
        affect_values = [agent.affect for agent in agents]
        resources_values = [agent.resources for agent in agents]

        # At least some variation should exist (with high probability)
        assert len(set(np.round(resilience_values, 3))) > 1  # Round to avoid floating point issues
        assert len(set(np.round(affect_values, 3))) > 1
        assert len(set(np.round(resources_values, 3))) > 1

    def test_agent_initialization_reproducible(self):
        """Test reproducible initialization with same seed."""
        config = {
            'initial_resilience_mean': 0.6,
            'initial_resilience_sd': 0.1,
            'initial_affect_mean': 0.1,
            'initial_affect_sd': 0.2,
            'initial_resources_mean': 0.7,
            'initial_resources_sd': 0.1,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        }

        # Create two agents with identical configuration and model seed
        model1 = MockModel(seed=123)
        model2 = MockModel(seed=123)

        agent1 = Person(model1, config)
        agent2 = Person(model2, config)

        # Agents should have identical initial state
        assert agent1.resilience == agent2.resilience
        assert agent1.affect == agent2.affect
        assert agent1.resources == agent2.resources
        assert agent1.baseline_affect == agent2.baseline_affect
        assert agent1.baseline_resilience == agent2.baseline_resilience

    def test_agent_initialization_configuration_integration(self):
        """Test integration with configuration system."""
        # Test that agent uses configuration values correctly
        config = {
            'initial_resilience_mean': 0.8,
            'initial_resilience_sd': 0.05,
            'initial_affect_mean': -0.1,
            'initial_affect_sd': 0.1,
            'initial_resources_mean': 0.9,
            'initial_resources_sd': 0.02,
            'stress_probability': 0.3,
            'coping_success_rate': 0.7,
            'subevents_per_day': 5
        }

        model = MockModel(seed=42)
        agent = Person(model, config)

        # Verify configuration is stored correctly
        assert hasattr(agent, 'stress_config')
        assert agent.stress_config['stress_probability'] == 0.3
        assert agent.stress_config['coping_success_rate'] == 0.7

        # Verify interaction config is initialized
        assert hasattr(agent, 'interaction_config')
        assert isinstance(agent.interaction_config, Mock) or hasattr(agent.interaction_config, '__dict__')

    def test_agent_initialization_protective_factors(self):
        """Test that protective factors are initialized correctly."""
        model = MockModel(seed=42)
        agent = Person(model)

        # Check that all protective factors are initialized
        expected_factors = ['social_support', 'family_support', 'formal_intervention', 'psychological_capital']
        for factor in expected_factors:
            assert factor in agent.protective_factors
            assert agent.protective_factors[factor] == 0.5  # Default value

    def test_agent_initialization_tracking_variables(self):
        """Test that tracking variables are initialized correctly."""
        model = MockModel(seed=42)
        agent = Person(model)

        # Check stress tracking variables
        assert agent.current_stress == 0.0
        assert agent.daily_stress_events == []
        assert agent.stress_history == []
        assert agent.last_reset_day == 0

        # Check interaction tracking variables
        assert agent.daily_interactions == 0
        assert agent.daily_support_exchanges == 0

        # Check PSS-10 variables
        assert isinstance(agent.pss10_responses, dict)
        assert 0.0 <= agent.stress_controllability <= 1.0
        assert 0.0 <= agent.stress_overload <= 1.0
        assert isinstance(agent.pss10, (int, float))
        assert isinstance(agent.stressed, bool)

    def test_agent_initialization_rng_creation(self):
        """Test that RNG is created correctly for reproducible testing."""
        model = MockModel(seed=42)
        agent = Person(model)

        # Check that RNG is created and accessible
        assert hasattr(agent, '_rng')
        assert agent._rng is not None

        # Test that RNG produces reproducible results
        model2 = MockModel(seed=42)
        agent2 = Person(model2)

        # Both agents should have identical RNG behavior with same seed
        # Reset RNG states to ensure we're comparing equivalent states
        agent._rng.random()
        agent2._rng.random()
        assert agent._rng.random() == agent2._rng.random()


class TestPSS10Initialization:
    """Test PSS-10 initialization functionality."""

    def test_pss10_initialization_basic(self):
        """Test basic PSS-10 initialization."""
        model = MockModel(seed=42)
        agent = Person(model)

        # Check that PSS-10 responses are generated
        assert len(agent.pss10_responses) == 10  # Should have 10 items

        # Check that all items are valid PSS-10 responses (0-4)
        for item_num, response in agent.pss10_responses.items():
            assert 1 <= item_num <= 10
            assert 0 <= response <= 4

        # Check that total PSS-10 score is computed
        assert 0 <= agent.pss10 <= 40  # Valid PSS-10 range

    def test_pss10_initialization_stress_dimensions(self):
        """Test that PSS-10 initialization creates proper stress dimensions."""
        model = MockModel(seed=42)
        agent = Person(model)

        # Check that stress dimensions are within valid ranges
        assert 0.0 <= agent.stress_controllability <= 1.0
        assert 0.0 <= agent.stress_overload <= 1.0

        # Check that stressed status is boolean
        assert isinstance(agent.stressed, bool)

    def test_pss10_initialization_reproducible(self):
        """Test that PSS-10 initialization is reproducible with same seed."""
        config = {
            'initial_resilience_mean': 0.5,
            'initial_resilience_sd': 0.1,
            'initial_affect_mean': 0.0,
            'initial_affect_sd': 0.1,
            'initial_resources_mean': 0.5,
            'initial_resources_sd': 0.1,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        }

        model1 = MockModel(seed=123)
        model2 = MockModel(seed=123)

        agent1 = Person(model1, config)
        agent2 = Person(model2, config)

        # PSS-10 responses should be identical
        assert agent1.pss10_responses == agent2.pss10_responses
        assert agent1.pss10 == agent2.pss10
        assert agent1.stress_controllability == agent2.stress_controllability
        assert agent1.stress_overload == agent2.stress_overload
        assert agent1.stressed == agent2.stressed

    def test_pss10_initialization_threshold_logic(self):
        """Test PSS-10 threshold-based stress classification."""
        model = MockModel(seed=42)
        agent = Person(model)

        # Test that stressed status is based on PSS-10 threshold
        cfg = get_config()
        pss10_threshold = cfg.get('pss10', 'threshold')

        if agent.pss10 >= pss10_threshold:
            assert agent.stressed == True
        else:
            assert agent.stressed == False


class TestAgentInitializationEdgeCases:
    """Test edge cases and error handling in agent initialization."""

    def test_agent_initialization_extreme_means(self):
        """Test initialization with extreme mean values."""
        # Test with means at boundaries
        config = {
            'initial_resilience_mean': 1.0,  # Maximum
            'initial_resilience_sd': 0.01,
            'initial_affect_mean': -1.0,     # Minimum
            'initial_affect_sd': 0.01,
            'initial_resources_mean': 0.0,  # Minimum
            'initial_resources_sd': 0.01,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        }

        model = MockModel(seed=42)
        agent = Person(model, config)

        # Values should still be clamped appropriately
        assert 0.0 <= agent.resilience <= 1.0
        assert -1.0 <= agent.affect <= 1.0
        assert 0.0 <= agent.resources <= 1.0

    def test_agent_initialization_zero_standard_deviation(self):
        """Test initialization with zero standard deviation."""
        config = {
            'initial_resilience_mean': 0.5,
            'initial_resilience_sd': 0.0,  # No variation
            'initial_affect_mean': 0.0,
            'initial_affect_sd': 0.0,     # No variation
            'initial_resources_mean': 0.7,
            'initial_resources_sd': 0.0,   # No variation
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        }

        model = MockModel(seed=42)
        agent = Person(model, config)

        # With zero SD, values should be very close to means
        assert abs(agent.resilience - 0.5) < 0.01
        assert abs(agent.affect - 0.0) < 0.01
        assert abs(agent.resources - 0.7) < 0.01

    def test_agent_initialization_without_config(self):
        """Test initialization without explicit config (uses defaults)."""
        model = MockModel(seed=42)
        agent = Person(model)

        # Should initialize successfully with default values
        assert 0.0 <= agent.resilience <= 1.0
        assert -1.0 <= agent.affect <= 1.0
        assert 0.0 <= agent.resources <= 1.0

        # Should have all required attributes
        required_attrs = [
            'protective_factors', 'current_stress', 'daily_stress_events',
            'stress_history', 'daily_interactions', 'daily_support_exchanges',
            'pss10_responses', 'stress_controllability', 'stress_overload',
            'pss10', 'stressed', 'stress_config', 'interaction_config', '_rng'
        ]

        for attr in required_attrs:
            assert hasattr(agent, attr), f"Missing required attribute: {attr}"

    def test_agent_initialization_large_standard_deviation(self):
        """Test initialization with large standard deviation."""
        config = {
            'initial_resilience_mean': 0.5,
            'initial_resilience_sd': 10.0,  # Very large SD
            'initial_affect_mean': 0.0,
            'initial_affect_sd': 5.0,      # Very large SD
            'initial_resources_mean': 0.5,
            'initial_resources_sd': 8.0,    # Very large SD
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        }

        model = MockModel(seed=42)
        agent = Person(model, config)

        # Despite large SD, values should still be clamped to valid ranges
        assert 0.0 <= agent.resilience <= 1.0
        assert -1.0 <= agent.affect <= 1.0
        assert 0.0 <= agent.resources <= 1.0


class TestAgentInitializationIntegration:
    """Test integration between agent initialization and other systems."""

    def test_agent_initialization_with_configuration_system(self):
        """Test that agent initialization integrates properly with configuration system."""
        # Test that agent uses actual configuration values
        model = MockModel(seed=42)
        agent = Person(model)

        # Get the actual configuration
        cfg = get_config()

        # Agent should use configuration values for stress_config
        expected_stress_prob = cfg.get('agent', 'stress_probability')
        expected_coping_rate = cfg.get('agent', 'coping_success_rate')

        assert agent.stress_config['stress_probability'] == expected_stress_prob
        assert agent.stress_config['coping_success_rate'] == expected_coping_rate

    def test_agent_initialization_math_utils_integration(self):
        """Test integration with math_utils sampling functions."""
        # Mock the sample_normal function to verify it's called correctly
        with patch('agent.sample_normal') as mock_sample_normal:
            # Configure the mock to return specific values
            mock_sample_normal.side_effect = [0.7, 0.2, 0.8, 0.2, 0.7]  # resilience, affect, resources, baseline_affect, baseline_resilience

            config = {
                'initial_resilience_mean': 0.5,
                'initial_resilience_sd': 0.1,
                'initial_affect_mean': 0.0,
                'initial_affect_sd': 0.2,
                'initial_resources_mean': 0.6,
                'initial_resources_sd': 0.15,
                'stress_probability': 0.5,
                'coping_success_rate': 0.5,
                'subevents_per_day': 3
            }

            model = MockModel(seed=42)
            agent = Person(model, config)

            # Verify sample_normal was called with correct parameters
            expected_calls = [
                # (mean, std, rng, min_value, max_value) for each call
                (0.5, 0.1, agent._rng, 0.0, 1.0),  # resilience
                (0.0, 0.2, agent._rng, -1.0, 1.0), # affect
                (0.6, 0.15, agent._rng, 0.0, 1.0), # resources
                (0.0, 0.2, agent._rng, -1.0, 1.0), # baseline_affect
                (0.5, 0.1, agent._rng, 0.0, 1.0),  # baseline_resilience
            ]

            # Check that sample_normal was called 5 times with correct parameters
            assert mock_sample_normal.call_count == 5

            # Verify the calls match expected parameters
            for i, expected_call in enumerate(expected_calls):
                call_args = mock_sample_normal.call_args_list[i]
                assert call_args[1]['mean'] == expected_call[0]
                assert call_args[1]['std'] == expected_call[1]
                assert call_args[1]['min_value'] == expected_call[3]
                assert call_args[1]['max_value'] == expected_call[4]

    def test_agent_initialization_different_seeds_different_results(self):
        """Test that different seeds produce different initialization results."""
        config = {
            'initial_resilience_mean': 0.5,
            'initial_resilience_sd': 0.2,
            'initial_affect_mean': 0.0,
            'initial_affect_sd': 0.2,
            'initial_resources_mean': 0.5,
            'initial_resources_sd': 0.2,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        }

        # Create agents with different seeds
        agents_data = []
        for seed in [100, 200, 300]:
            model = MockModel(seed=seed)
            agent = Person(model, config)
            agents_data.append({
                'seed': seed,
                'resilience': agent.resilience,
                'affect': agent.affect,
                'resources': agent.resources,
                'pss10': agent.pss10
            })

        # With high probability, different seeds should produce different results
        # (This test might occasionally fail due to random chance, but should pass most of the time)
        resilience_values = [data['resilience'] for data in agents_data]
        affect_values = [data['affect'] for data in agents_data]

        # At least two different values should exist across the agents
        unique_resilience = set(np.round(resilience_values, 2))
        unique_affect = set(np.round(affect_values, 2))

        assert len(unique_resilience) > 1 or len(unique_affect) > 1


# Example of how to run these tests:
# pytest src/python/tests/test_agent_initialization.py -v
# pytest src/python/tests/test_agent_initialization.py::TestAgentInitializationCore::test_agent_initialization_normal_distributions -v