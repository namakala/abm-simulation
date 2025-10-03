"""
Integration tests for the refactored Agent class.

This file demonstrates how to test the Agent class using mocked dependencies
and verifying correct integration with utility functions.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.python.agent import Person
from src.python.stress_utils import StressEvent, AppraisalWeights, ThresholdParams
from src.python.affect_utils import InteractionConfig


class MockModel:
    """Mock Mesa model for testing."""

    def __init__(self, seed=None):
        self.seed = seed
        self.grid = Mock()
        self.grid.get_neighbors.return_value = []
        self.agents = Mock()
        self.register_agent = Mock()  # Required by Mesa Agent base class
        self.rng = np.random.default_rng(seed)  # Required by Mesa Agent base class


class TestAgentInitialization:
    """Test agent initialization with different configurations."""

    def test_agent_initialization_default(self):
        """Test agent initialization with default parameters."""
        model = MockModel(seed=42)
        agent = Person(model)

        assert agent.resilience == 0.5
        assert agent.affect == 0.0
        assert agent.resources == 0.6
        assert len(agent.protective_factors) == 4
        assert hasattr(agent, '_rng')

    def test_agent_initialization_with_config(self):
        """Test agent initialization with custom configuration."""
        model = MockModel()
        config = {
            'initial_resilience': 0.8,
            'initial_affect': 0.2,
            'initial_resources': 0.9,
            'stress_probability': 0.3,
            'coping_success_rate': 0.7,
            'subevents_per_day': 5
        }

        agent = Person(model, config)

        assert agent.resilience == 0.8
        assert agent.affect == 0.2
        assert agent.resources == 0.9

    def test_agent_reproducible_initialization(self):
        """Test that agent initialization is reproducible with same seed."""
        config = {
            'initial_resilience': 0.7,
            'initial_affect': 0.1,
            'initial_resources': 0.6,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        }

        # Two agents with same model seed should behave identically
        model1 = MockModel(seed=123)
        model2 = MockModel(seed=123)

        agent1 = Person(model1, config)
        agent2 = Person(model2, config)

        # Both should have same initial state
        assert agent1.resilience == agent2.resilience
        assert agent1.affect == agent2.affect
        assert agent1.resources == agent2.resources


class TestAgentStepBehavior:
    """Test agent step behavior and event processing."""

    @patch('src.python.agent.sample_poisson')
    def test_agent_step_calls_utility_functions(self, mock_sample_poisson):
        """Test that agent.step() correctly calls utility functions."""
        # Setup mocks
        mock_sample_poisson.return_value = 2

        model = MockModel(seed=42)
        agent = Person(model)

        # Mock the grid to return no neighbors for interaction tests
        model.grid.get_neighbors.return_value = []

        # Execute step
        agent.step()

        # Verify utility functions were called
        mock_sample_poisson.assert_called_once()

    def test_agent_step_preserves_bounds(self):
        """Test that agent step keeps all values within valid bounds."""
        model = MockModel(seed=42)
        agent = Person(model)

        # Set extreme values
        agent.resilience = 1.5  # Above valid range
        agent.affect = -2.0     # Below valid range
        agent.resources = -0.5  # Below valid range

        # Mock no events for this test
        with patch('src.python.agent.sample_poisson', return_value=0):
            agent.step()

        # Values should be clamped to valid ranges
        assert 0.0 <= agent.resilience <= 1.0
        assert -1.0 <= agent.affect <= 1.0
        assert 0.0 <= agent.resources <= 1.0


class TestAgentInteractions:
    """Test agent social interaction behavior."""

    def test_agent_interaction_with_neighbors(self):
        """Test agent interaction when neighbors are present."""
        model = MockModel(seed=42)

        # Create two agents
        agent1 = Person(model, {
            'initial_affect': 0.5,
            'initial_resilience': 0.6,
            'initial_resources': 0.6,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        })
        agent2 = Person(model, {
            'initial_affect': -0.3,
            'initial_resilience': 0.4,
            'initial_resources': 0.6,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        })

        # Setup neighbor relationship
        model.grid.get_neighbors.return_value = [agent2]

        # Record initial states
        initial_affect1 = agent1.affect
        initial_affect2 = agent2.affect
        initial_resilience1 = agent1.resilience
        initial_resilience2 = agent2.resilience

        # Execute interaction
        agent1.interact()

        # Check that interaction occurred and values are still in bounds
        assert -1.0 <= agent1.affect <= 1.0
        assert -1.0 <= agent2.affect <= 1.0
        assert 0.0 <= agent1.resilience <= 1.0
        assert 0.0 <= agent2.resilience <= 1.0

    def test_agent_interaction_no_neighbors(self):
        """Test agent interaction when no neighbors are present."""
        model = MockModel(seed=42)
        agent = Person(model)

        # No neighbors available
        model.grid.get_neighbors.return_value = []

        # Record initial state
        initial_affect = agent.affect
        initial_resilience = agent.resilience

        # Execute interaction (should do nothing)
        agent.interact()

        # State should be unchanged
        assert agent.affect == initial_affect
        assert agent.resilience == initial_resilience


class TestAgentStressEvents:
    """Test agent stress event processing."""

    @patch('src.python.agent.process_stress_event')
    @patch('src.python.agent.generate_stress_event')
    def test_agent_stressful_event_processing(self, mock_generate_stress, mock_process_stress):
        """Test that stressful events are processed correctly."""
        # Setup mocks
        stress_event = StressEvent(0.5, 0.5)
        mock_generate_stress.return_value = stress_event
        mock_process_stress.return_value = (True, 0.7, 0.3)  # is_stressed=True

        model = MockModel(seed=42)
        agent = Person(model)

        # Record initial state
        initial_affect = agent.affect
        initial_resilience = agent.resilience
        initial_resources = agent.resources

        # Execute stressful event
        agent.stressful_event()

        # Verify utility functions were called
        mock_generate_stress.assert_called_once()
        mock_process_stress.assert_called_once()

        # Check bounds are maintained
        assert -1.0 <= agent.affect <= 1.0
        assert 0.0 <= agent.resilience <= 1.0
        assert 0.0 <= agent.resources <= 1.0

    def test_agent_no_stress_event(self):
        """Test behavior when no stress event occurs."""
        model = MockModel(seed=42)
        agent = Person(model)

        # Record initial state
        initial_affect = agent.affect
        initial_resilience = agent.resilience
        initial_resources = agent.resources

        # Execute stressful event (no stress should occur)
        agent.stressful_event()

        # State should be largely unchanged (only resource regeneration)
        # Note: In real implementation, resources might regenerate slightly
        assert -1.0 <= agent.affect <= 1.0
        assert 0.0 <= agent.resilience <= 1.0
        assert 0.0 <= agent.resources <= 1.0


class TestAgentResourceManagement:
    """Test agent resource management and protective factors."""

    def test_agent_resource_usage_during_coping(self):
        """Test that resources are used when coping successfully."""
        model = MockModel(seed=42)
        agent = Person(model, {
            'initial_resources': 0.5,
            'initial_resilience': 0.9,  # High resilience for successful coping
            'initial_affect': 0.0,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        })

        # Mock the random number generator to ensure coping success
        with patch.object(agent, '_rng') as mock_rng:
            mock_rng.random.return_value = 0.5  # Less than resilience (0.9), so coping succeeds
            # Add missing mocks for PSS-10 computation
            mock_rng.multivariate_normal.return_value = np.array([0.5, 0.5])
            mock_rng.normal.side_effect = lambda *args, **kwargs: 2.0 if (args and args[0] != 0) else 0.0
            
            # Mock process_stress_event to ensure stress occurs and coping succeeds
            with patch('src.python.agent.process_stress_event') as mock_process_stress:
                mock_process_stress.return_value = (True, 0.7, 0.3)  # is_stressed=True, challenge=0.7, hindrance=0.3
                
                # Also need to mock the new mechanism function to use agent's RNG
                with patch('src.python.affect_utils.process_stress_event_with_new_mechanism') as mock_new_mechanism:
                    # Mock to return successful coping (coped_successfully=True)
                    mock_new_mechanism.return_value = (0.0, 0.9, 0.2, True)  # affect, resilience, stress, coped_successfully

                    # Mock a high magnitude stress event that will trigger stress
                    with patch('src.python.agent.generate_stress_event') as mock_stress_event:
                        mock_stress_event.return_value = StressEvent(0.0, 0.8)  # High overload stress event
                        agent.stressful_event()

        # Resources should have been consumed for coping
        assert agent.resources < 0.5  # Should be reduced
        assert agent.resources >= 0.0  # But not negative

    def test_agent_resource_preservation_when_not_stressed(self):
        """Test that resources are preserved when not stressed."""
        model = MockModel(seed=42)
        agent = Person(model, {
            'initial_resources': 0.8,
            'initial_resilience': 0.5,
            'initial_affect': 0.0,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        })

        # Mock no stress scenario - create event that won't trigger stress
        with patch('src.python.stress_utils.generate_stress_event') as mock_stress_event, \
             patch('src.python.agent.process_stress_event') as mock_process_stress:
            
            # Create event with low magnitude that won't exceed threshold
            mock_stress_event.return_value = StressEvent(0.5, 0.1)
            # Ensure the event doesn't trigger stress
            mock_process_stress.return_value = (False, 0.0, 0.0)  # is_stressed=False

            # Execute stressful event
            agent.stressful_event()

        # Resources should be largely unchanged (only regeneration)
        assert agent.resources >= 0.7  # Should be close to original value


class TestAgentConfiguration:
    """Test agent behavior with different configurations."""

    def test_agent_with_high_resilience_config(self):
        """Test agent behavior with high resilience configuration."""
        config = {
            'initial_resilience': 0.9,
            'initial_affect': 0.0,
            'initial_resources': 0.8,
            'stress_probability': 0.2,  # Lower stress probability
            'coping_success_rate': 0.9   # High coping success
        }

        model = MockModel(seed=42)
        agent = Person(model, config)

        assert agent.resilience == 0.9

        # With high resilience, agent should cope better with stress
        # This would be tested with more complex scenarios

    def test_agent_with_low_resilience_config(self):
        """Test agent behavior with low resilience configuration."""
        config = {
            'initial_resilience': 0.1,
            'initial_affect': 0.0,
            'initial_resources': 0.3,
            'stress_probability': 0.8,  # High stress probability
            'coping_success_rate': 0.2   # Low coping success
        }

        model = MockModel(seed=42)
        agent = Person(model, config)

        assert agent.resilience == 0.1
        assert agent.resources == 0.3


# Example of how to run integration tests:
# pytest test_agent_integration.py -v
# pytest test_agent_integration.py::TestAgentStepBehavior::test_agent_step_preserves_bounds -v

# Example of running all tests:
# pytest test_*.py -v