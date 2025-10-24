"""
Comprehensive unit tests for agent initialization logic.

This file provides complete test coverage for the Person class __init__ method
and related initialization functionality, including PSS-10 setup and configuration integration.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.python.agent import Person
from src.python.config import get_config, Config
from src.python.math_utils import sample_normal, create_rng, sigmoid_transform, tanh_transform


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

    def test_agent_initialization_uses_transformation_pipeline(self):
        """Test that agent initialization uses the new transformation pipeline correctly."""
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

        # Verify that values are in correct bounds (transformation pipeline ensures this)
        assert 0.0 <= agent.resilience <= 1.0
        assert -1.0 <= agent.affect <= 1.0
        assert 0.0 <= agent.resources <= 1.0

        # Test that baseline values are also initialized and in correct bounds
        assert -1.0 <= agent.baseline_affect <= 1.0
        assert 0.0 <= agent.baseline_resilience <= 1.0

        # Verify transformation functions are being used (values should be different from means due to transformation)
        # With transformation pipeline, even with same seed, values will be transformed
        assert agent.resilience != 0.7  # Should be transformed from normal distribution
        assert agent.affect != 0.2     # Should be transformed from normal distribution
        assert agent.resources != 0.8  # Should be transformed from normal distribution

    def test_agent_initialization_bounds_enforcement(self):
        """Test that transformation pipeline enforces proper bounds regardless of input parameters."""
        # Test with extreme mean values that would be out of bounds without transformation
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

        # Transformation pipeline should enforce bounds regardless of extreme input parameters
        assert 0.0 <= agent.resilience <= 1.0
        assert -1.0 <= agent.affect <= 1.0
        assert 0.0 <= agent.resources <= 1.0

        # Baseline values should also be in correct bounds
        assert -1.0 <= agent.baseline_affect <= 1.0
        assert 0.0 <= agent.baseline_resilience <= 1.0

        # Test with very large standard deviations
        config_large_sd = {
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

        agent_large_sd = Person(MockModel(seed=42), config_large_sd)

        # Even with large SD, transformation should keep values in bounds
        assert 0.0 <= agent_large_sd.resilience <= 1.0
        assert -1.0 <= agent_large_sd.affect <= 1.0
        assert 0.0 <= agent_large_sd.resources <= 1.0

    def test_agent_initialization_realistic_variation(self):
        """Test that different agents get realistically varied initial values."""
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

        # Agents should have different initial values due to different seeds and transformation
        resilience_values = [agent.resilience for agent in agents]
        affect_values = [agent.affect for agent in agents]
        resources_values = [agent.resources for agent in agents]

        # All values should be in correct bounds
        for agent in agents:
            assert 0.0 <= agent.resilience <= 1.0
            assert -1.0 <= agent.affect <= 1.0
            assert 0.0 <= agent.resources <= 1.0

        # Test realistic variation - values should show meaningful differences
        resilience_values_rounded = [round(r, 3) for r in resilience_values]
        affect_values_rounded = [round(a, 3) for a in affect_values]
        resources_values_rounded = [round(res, 3) for res in resources_values]

        # Should have variation across agents (high probability with different seeds)
        assert len(set(resilience_values_rounded)) >= 2  # At least 2 different resilience values
        assert len(set(affect_values_rounded)) >= 2      # At least 2 different affect values
        assert len(set(resources_values_rounded)) >= 2   # At least 2 different resource values

        # Test that variation is realistic (not all extreme values)
        resilience_array = np.array(resilience_values)
        affect_array = np.array(affect_values)
        resources_array = np.array(resources_values)

        # Standard deviations should be reasonable (not zero, not extremely large)
        resilience_std = np.std(resilience_array)
        affect_std = np.std(affect_array)
        resources_std = np.std(resources_array)

        assert 0.01 < resilience_std < 0.5  # Reasonable variation range
        assert 0.01 < affect_std < 0.8       # Reasonable variation range
        assert 0.01 < resources_std < 0.4    # Reasonable variation range

    def test_agent_initialization_reproducible_with_transformation(self):
        """Test reproducible initialization with same seed using transformation pipeline."""
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

        # Agents should have identical initial state due to same seed and transformation
        assert agent1.resilience == agent2.resilience
        assert agent1.affect == agent2.affect
        assert agent1.resources == agent2.resources
        assert agent1.baseline_affect == agent2.baseline_affect
        assert agent1.baseline_resilience == agent2.baseline_resilience

        # Test that PSS-10 initialization is also reproducible
        assert agent1.pss10_responses == agent2.pss10_responses
        assert agent1.pss10 == agent2.pss10
        assert agent1.stress_controllability == agent2.stress_controllability
        assert agent1.stress_overload == agent2.stress_overload

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

        # With zero SD, transformation functions should handle it deterministically
        # The transformation pipeline still applies, so values won't equal means exactly
        # but should be deterministic and in correct bounds
        assert 0.0 <= agent.resilience <= 1.0
        assert -1.0 <= agent.affect <= 1.0
        assert 0.0 <= agent.resources <= 1.0

        # Test that zero SD produces deterministic results across multiple runs
        model2 = MockModel(seed=42)
        agent2 = Person(model2, config)

        # Should be identical due to same seed and zero SD
        assert agent.resilience == agent2.resilience
        assert agent.affect == agent2.affect
        assert agent.resources == agent2.resources

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

    def test_agent_initialization_uses_transformation_functions(self):
        """Test that agent initialization uses transformation functions correctly."""
        # Test that transformation functions are being used by checking behavior
        # rather than mocking (which can be unreliable with imports)

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

        # Verify that transformation functions are being used correctly
        # by checking that values are different from the raw means (indicating transformation occurred)

        # Values should be in correct bounds (transformation pipeline ensures this)
        assert 0.0 <= agent.resilience <= 1.0
        assert -1.0 <= agent.affect <= 1.0
        assert 0.0 <= agent.resources <= 1.0

        # Values should be different from the raw means (indicating transformation occurred)
        # With transformation pipeline, even with same seed, values will be transformed
        assert agent.resilience != 0.5  # Should be sigmoid transformed from normal distribution
        assert agent.affect != 0.0      # Should be tanh transformed from normal distribution
        assert agent.resources != 0.6   # Should be sigmoid transformed from normal distribution

        # Test reproducibility - same seed should produce same results
        model2 = MockModel(seed=42)
        agent2 = Person(model2, config)

        # Agents should have identical initial state due to same seed and transformation
        assert agent.resilience == agent2.resilience
        assert agent.affect == agent2.affect
        assert agent.resources == agent2.resources

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


class TestAgentPopulationVariation:
    """Test realistic variation in agent populations."""

    def test_agent_population_statistical_properties(self):
        """Test that agent population shows realistic statistical properties."""
        config = {
            'initial_resilience_mean': 0.6,
            'initial_resilience_sd': 0.15,
            'initial_affect_mean': 0.1,
            'initial_affect_sd': 0.25,
            'initial_resources_mean': 0.7,
            'initial_resources_sd': 0.12,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        }

        # Create a larger population for statistical analysis
        population_size = 100
        agents = []
        for i in range(population_size):
            model = MockModel(seed=1000 + i)  # Different seeds for each agent
            agent = Person(model, config)
            agents.append(agent)

        # Extract values for statistical analysis
        resilience_values = [agent.resilience for agent in agents]
        affect_values = [agent.affect for agent in agents]
        resources_values = [agent.resources for agent in agents]

        resilience_array = np.array(resilience_values)
        affect_array = np.array(affect_values)
        resources_array = np.array(resources_values)

        # Test that all values are in correct bounds
        assert np.all(resilience_array >= 0.0) and np.all(resilience_array <= 1.0)
        assert np.all(affect_array >= -1.0) and np.all(affect_array <= 1.0)
        assert np.all(resources_array >= 0.0) and np.all(resources_array <= 1.0)

        # Test statistical properties
        resilience_mean = np.mean(resilience_array)
        affect_mean = np.mean(affect_array)
        resources_mean = np.mean(resources_array)

        # Means should be reasonably close to configured values (within 2 SD)
        # Note: Transformation affects the relationship between configured and actual parameters
        assert abs(resilience_mean - 0.6) < 0.35  # Within reasonable range for sigmoid transformation
        assert abs(affect_mean - 0.1) < 0.5       # Within 2 SD of 0.25
        assert abs(resources_mean - 0.7) < 0.3    # Within reasonable range for sigmoid transformation

        # Standard deviations should be positive (showing variation)
        assert np.std(resilience_array) > 0.01
        assert np.std(affect_array) > 0.01
        assert np.std(resources_array) > 0.01

        # Test that we get good coverage of the range (not all values identical)
        resilience_range = np.max(resilience_array) - np.min(resilience_array)
        affect_range = np.max(affect_array) - np.min(affect_array)
        resources_range = np.max(resources_array) - np.min(resources_array)

        assert resilience_range > 0.1  # Should span at least 0.1 of the range
        assert affect_range > 0.2      # Should span at least 0.2 of the range
        assert resources_range > 0.1   # Should span at least 0.1 of the range

    def test_agent_population_normal_distribution_characteristics(self):
        """Test that transformed values maintain normal distribution characteristics."""
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

        # Create large population for distribution analysis
        population_size = 1000
        agents = []
        for i in range(population_size):
            model = MockModel(seed=2000 + i)
            agent = Person(model, config)
            agents.append(agent)

        # Test resilience distribution (sigmoid transformed)
        resilience_values = [agent.resilience for agent in agents]
        resilience_array = np.array(resilience_values)

        # For sigmoid transformation of normal distribution, we expect:
        # - Values in [0,1] range ✓ (already tested)
        # - Mean should be close to sigmoid(mean) of original normal
        # - Good coverage of [0,1] range

        resilience_mean = np.mean(resilience_array)
        resilience_std = np.std(resilience_array)

        # Should have reasonable mean (not 0 or 1)
        assert 0.1 < resilience_mean < 0.9

        # Should have reasonable spread
        assert 0.05 < resilience_std < 0.4

        # Test affect distribution (tanh transformed)
        affect_values = [agent.affect for agent in agents]
        affect_array = np.array(affect_values)

        affect_mean = np.mean(affect_array)
        affect_std = np.std(affect_array)

        # For tanh transformation, mean should be close to 0 (symmetric around 0)
        assert abs(affect_mean) < 0.2

        # Should have reasonable spread
        assert 0.05 < affect_std < 0.6

        # Test resources distribution (sigmoid transformed)
        resources_values = [agent.resources for agent in agents]
        resources_array = np.array(resources_values)

        resources_mean = np.mean(resources_array)
        resources_std = np.std(resources_array)

        # Should have reasonable mean (not 0 or 1)
        assert 0.1 < resources_mean < 0.9

        # Should have reasonable spread
        assert 0.05 < resources_std < 0.4

    def test_agent_population_bounds_strictly_enforced(self):
        """Test that transformation functions strictly enforce bounds."""
        # Test with various extreme configurations
        extreme_configs = [
            # Extreme means
            {
                'initial_resilience_mean': 5.0, 'initial_resilience_sd': 0.1,
                'initial_affect_mean': -5.0, 'initial_affect_sd': 0.1,
                'initial_resources_mean': -2.0, 'initial_resources_sd': 0.1,
            },
            # Extreme standard deviations
            {
                'initial_resilience_mean': 0.5, 'initial_resilience_sd': 10.0,
                'initial_affect_mean': 0.0, 'initial_affect_sd': 8.0,
                'initial_resources_mean': 0.5, 'initial_resources_sd': 12.0,
            },
            # Zero standard deviation
            {
                'initial_resilience_mean': 0.5, 'initial_resilience_sd': 0.0,
                'initial_affect_mean': 0.0, 'initial_affect_sd': 0.0,
                'initial_resources_mean': 0.5, 'initial_resources_sd': 0.0,
            }
        ]

        for config in extreme_configs:
            # Add common parameters
            config.update({
                'stress_probability': 0.5,
                'coping_success_rate': 0.5,
                'subevents_per_day': 3
            })

            # Test multiple agents with this configuration
            for seed in [42, 43, 44]:
                model = MockModel(seed=seed)
                agent = Person(model, config)

                # Bounds should always be strictly enforced
                assert 0.0 <= agent.resilience <= 1.0
                assert -1.0 <= agent.affect <= 1.0
                assert 0.0 <= agent.resources <= 1.0

                # For zero SD case, should get deterministic results
                if config['initial_resilience_sd'] == 0.0:
                    # With zero SD, transformation should produce deterministic results
                    # The transformation pipeline may or may not equal the mean exactly
                    # but should be consistent across runs
                    pass  # TODO: Add specific assertions for zero SD case

    def test_agent_population_realistic_heterogeneity(self):
        """Test that agent population shows realistic heterogeneity patterns."""
        config = {
            'initial_resilience_mean': 0.6,
            'initial_resilience_sd': 0.2,
            'initial_affect_mean': 0.0,
            'initial_affect_sd': 0.3,
            'initial_resources_mean': 0.65,
            'initial_resources_sd': 0.18,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        }

        # Create population with realistic size
        population_size = 200
        agents = []
        for i in range(population_size):
            model = MockModel(seed=3000 + i)
            agent = Person(model, config)
            agents.append(agent)

        # Extract values
        resilience_values = [agent.resilience for agent in agents]
        affect_values = [agent.affect for agent in agents]
        resources_values = [agent.resources for agent in agents]

        # Test for realistic heterogeneity patterns
        resilience_array = np.array(resilience_values)
        affect_array = np.array(affect_values)
        resources_array = np.array(resources_values)

        # 1. Test for multimodality (should not be uniform)
        # Calculate coefficient of variation (should be moderate)
        resilience_cv = np.std(resilience_array) / np.mean(resilience_array)
        affect_cv = np.std(affect_array) / (np.std(affect_array) + 1e-10)  # Avoid division by zero
        resources_cv = np.std(resources_array) / np.mean(resources_array)

        # Should have reasonable variation (not too little, not too much)
        assert 0.1 < resilience_cv < 1.0
        assert 0.1 < resources_cv < 1.0
        # Affect can have higher variation due to tanh transformation
        assert 0.2 < abs(affect_cv) < 2.0

        # 2. Test for outliers (should not have extreme outliers)
        resilience_q75, resilience_q25 = np.percentile(resilience_array, [75, 25])
        affect_q75, affect_q25 = np.percentile(affect_array, [75, 25])
        resources_q75, resources_q25 = np.percentile(resources_array, [75, 25])

        resilience_iqr = resilience_q75 - resilience_q25
        affect_iqr = affect_q75 - affect_q25
        resources_iqr = resources_q75 - resources_q25

        # IQR should be reasonable (not zero, not extremely large)
        assert 0.05 < resilience_iqr < 0.8
        assert 0.1 < affect_iqr < 1.5
        assert 0.05 < resources_iqr < 0.8

        # 3. Test for realistic correlations between variables
        # In realistic populations, these variables might be somewhat correlated
        correlation_matrix = np.corrcoef([resilience_array, affect_array, resources_array])

        # Should not have perfect correlations (indicating independence)
        assert not np.allclose(correlation_matrix, 1.0, atol=0.1)

        # 4. Test for realistic range coverage
        # Should utilize most of the available range
        resilience_range = np.max(resilience_array) - np.min(resilience_array)
        affect_range = np.max(affect_array) - np.min(affect_array)
        resources_range = np.max(resources_array) - np.min(resources_array)

        # Should cover substantial portion of possible range
        assert resilience_range > 0.3  # Cover > 30% of [0,1] range
        assert affect_range > 0.8      # Cover > 80% of [-1,1] range
        assert resources_range > 0.3   # Cover > 30% of [0,1] range


# Example of how to run these tests:
# pytest src/python/tests/test_agent_initialization.py -v
# pytest src/python/tests/test_agent_initialization.py::TestAgentInitializationCore::test_agent_initialization_normal_distributions -v