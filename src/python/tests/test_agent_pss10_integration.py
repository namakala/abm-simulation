#!/usr/bin/env python3
"""
Tests for agent PSS-10 integration functionality.

Tests the integration of PSS-10 score generation with agent lifecycle:
- PSS-10 initialization during agent creation
- PSS-10 score updates during step execution
- Stress level mapping from PSS-10 responses
- Integration with stress processing mechanisms
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.python.agent import Person
from src.python.model import StressModel
from src.python.stress_utils import generate_pss10_responses, compute_pss10_score


class TestAgentPSS10Initialization:
    """Test PSS-10 initialization during agent creation."""

    def test_agent_pss10_initialization(self):
        """Test that agents initialize with valid PSS-10 scores and stress levels."""
        # Create a mock model
        model = Mock()
        model.seed = 42

        # Create agent
        agent = Person(model)

        # Check that PSS-10 state variables are initialized
        assert hasattr(agent, 'stress_controllability')
        assert hasattr(agent, 'stress_overload')
        assert hasattr(agent, 'pss10')
        assert hasattr(agent, 'pss10_responses')

        # Check that values are in valid ranges
        assert 0 <= agent.stress_controllability <= 1
        assert 0 <= agent.stress_overload <= 1
        assert 0 <= agent.pss10 <= 40
        assert isinstance(agent.pss10_responses, dict)
        assert len(agent.pss10_responses) == 10

        # Check that all PSS-10 responses are valid
        for item_num, response in agent.pss10_responses.items():
            assert 1 <= item_num <= 10
            assert 0 <= response <= 4

    def test_pss10_score_computation(self):
        """Test that PSS-10 score is correctly computed from responses."""
        # Create a mock model
        model = Mock()
        model.seed = 42

        # Create agent
        agent = Person(model)

        # Verify PSS-10 score matches computed score from responses
        expected_score = compute_pss10_score(agent.pss10_responses)
        assert agent.pss10 == expected_score

    def test_pss10_reproducibility(self):
        """Test that PSS-10 initialization is reproducible with same seed."""
        # Create two models with same seed
        model1 = Mock()
        model1.seed = 123

        model2 = Mock()
        model2.seed = 123

        # Create agents with same seed
        agent1 = Person(model1)
        agent2 = Person(model2)

        # Should have identical PSS-10 initialization
        assert agent1.pss10 == agent2.pss10
        assert agent1.pss10_responses == agent2.pss10_responses
        assert agent1.stress_controllability == agent2.stress_controllability
        assert agent1.stress_overload == agent2.stress_overload


class TestAgentPSS10StepIntegration:
    """Test PSS-10 updates during agent step execution."""

    def test_pss10_update_during_step(self):
        """Test that PSS-10 scores are updated during step execution."""
        # Create a mock model with grid
        model = Mock()
        model.seed = 42
        model.current_day = 1
        model.grid = Mock()
        model.grid.get_neighbors.return_value = []  # No neighbors
        model.agents = []

        # Create agent
        agent = Person(model)

        # Store initial PSS-10 state
        initial_pss10 = agent.pss10
        initial_responses = agent.pss10_responses.copy()
        initial_controllability = agent.stress_controllability
        initial_overload = agent.stress_overload

        # Execute one step
        agent.step()

        # PSS-10 should be updated (likely different due to randomness)
        assert agent.pss10 != initial_pss10 or agent.pss10_responses != initial_responses

        # Stress levels should still be in valid range
        assert 0 <= agent.stress_controllability <= 1
        assert 0 <= agent.stress_overload <= 1

    def test_pss10_score_consistency(self):
        """Test that PSS-10 score remains consistent with responses after step."""
        # Create a mock model
        model = Mock()
        model.seed = 42
        model.current_day = 1
        model.grid = Mock()
        model.grid.get_neighbors.return_value = []
        model.agents = []

        # Create agent
        agent = Person(model)

        # Execute one step
        agent.step()

        # PSS-10 score should match computed score from responses
        expected_score = compute_pss10_score(agent.pss10_responses)
        assert agent.pss10 == expected_score


class TestStressLevelPSS10Mapping:
    """Test mapping between stress levels and PSS-10 responses."""

    def test_stress_levels_from_pss10_mapping(self):
        """Test that stress levels are correctly derived from PSS-10 responses."""
        # Create a mock model
        model = Mock()
        model.seed = 42

        # Create agent
        agent = Person(model)

        # Test the mapping function directly
        controllability_items = [4, 5, 7, 8]  # Reverse scored items
        overload_items = [1, 2, 3, 6, 9, 10]  # Non-reverse scored items

        # Calculate expected stress levels from PSS-10 responses
        expected_controllability_scores = []
        for item_num in controllability_items:
            if item_num in agent.pss10_responses:
                response = agent.pss10_responses[item_num]
                # Without reversing the score, higher PSS-10 = higher controllability
                expected_controllability_scores.append(response / 4.0)

        expected_overload_scores = []
        for item_num in overload_items:
            if item_num in agent.pss10_responses:
                response = agent.pss10_responses[item_num]
                # Higher PSS-10 response = higher overload
                expected_overload_scores.append(response / 4.0)

        expected_controllability = np.mean(expected_controllability_scores) if expected_controllability_scores else 0.5
        expected_overload = np.mean(expected_overload_scores) if expected_overload_scores else 0.5

        # Should match agent's stress levels
        assert abs(agent.stress_controllability - expected_controllability) < 1e-10
        assert abs(agent.stress_overload - expected_overload) < 1e-10

    def test_extreme_stress_level_mapping(self):
        """Test mapping with extreme stress levels."""
        # Create a mock model
        model = Mock()
        model.seed = 42

        # Create agent
        agent = Person(model)

        # Test with extreme values
        if agent.stress_controllability < 0.1:  # Very low controllability
            # Should tend to have higher PSS-10 scores on controllability items
            controllability_items = [4, 5, 7, 8]
            for item_num in controllability_items:
                # Reverse scored: low controllability should give higher PSS-10 scores
                response = agent.pss10_responses[item_num]
                expected_response = 4 - int((agent.stress_controllability * 4))  # Reverse mapping
                assert abs(response - expected_response) <= 1  # Allow some tolerance

        if agent.stress_overload > 0.9:  # Very high overload
            # Should tend to have higher PSS-10 scores on overload items
            overload_items = [1, 2, 3, 6, 9, 10]
            for item_num in overload_items:
                response = agent.pss10_responses[item_num]
                expected_response = int(agent.stress_overload * 4)
                assert abs(response - expected_response) <= 1  # Allow some tolerance


class TestPSS10StressMechanismIntegration:
    """Test integration between PSS-10 and stress processing mechanisms."""

    def test_stress_event_without_predictability(self):
        """Test that stress events work correctly without predictability."""
        from src.python.stress_utils import generate_stress_event

        rng = np.random.default_rng(42)

        # Generate stress event
        event = generate_stress_event(rng)

        # Should only have controllability and overload
        assert hasattr(event, 'controllability')
        assert hasattr(event, 'overload')

        # Should not have predictability or magnitude
        assert not hasattr(event, 'predictability')
        assert not hasattr(event, 'magnitude')

        # All values should be in valid range
        assert 0 <= event.controllability <= 1
        assert 0 <= event.overload <= 1

    def test_apply_weights_without_predictability(self):
        """Test that apply_weights function works without predictability."""
        from src.python.stress_utils import apply_weights, StressEvent, AppraisalWeights

        # Create stress event without predictability or magnitude
        event = StressEvent(
            controllability=0.6,
            overload=0.4
        )

        # Create weights (without omega_p)
        weights = AppraisalWeights(omega_c=1.0, omega_o=1.0, bias=0.0, gamma=6.0)

        # Apply weights
        challenge, hindrance = apply_weights(event, weights)

        # Should produce valid challenge/hindrance values
        assert 0 <= challenge <= 1
        assert 0 <= hindrance <= 1
        assert abs((challenge + hindrance) - 1.0) < 1e-10  # Should sum to 1

    def test_agent_stressful_event_integration(self):
        """Test that agent's stressful_event method works with PSS-10 integration."""
        # Create a mock model
        model = Mock()
        model.seed = 42
        model.current_day = 1
        model.grid = Mock()
        model.grid.get_neighbors.return_value = []
        model.agents = []

        # Create agent
        agent = Person(model)

        # Store initial state
        initial_pss10 = agent.pss10
        initial_responses = agent.pss10_responses.copy()
        initial_stress_levels = (agent.stress_controllability, agent.stress_overload)

        # Execute a stressful event
        challenge, hindrance = agent.stressful_event()

        # Should return valid challenge/hindrance values
        assert 0 <= challenge <= 1
        assert 0 <= hindrance <= 1

        # If stress event triggered (challenge > 0 or hindrance > 0), then PSS-10 should be updated
        if challenge > 0 or hindrance > 0:
            # Stress event occurred, so PSS-10 should be updated
            pss10_updated = (agent.pss10 != initial_pss10 or
                           agent.pss10_responses != initial_responses or
                           agent.stress_controllability != initial_stress_levels[0] or
                           agent.stress_overload != initial_stress_levels[1])
            assert pss10_updated, "PSS-10 or stress levels should be updated when stress event occurs"
        else:
            # No stress event occurred (threshold not met), which is also valid
            # In this case, no updates should occur
            assert agent.pss10 == initial_pss10
            assert agent.pss10_responses == initial_responses
            assert agent.stress_controllability == initial_stress_levels[0]
            assert agent.stress_overload == initial_stress_levels[1]


def run_all_tests():
    """Run all agent PSS-10 integration tests."""
    print("Running Agent PSS-10 Integration Test Suite")
    print("=" * 50)

    # Create test instances
    test_classes = [
        TestAgentPSS10Initialization(),
        TestAgentPSS10StepIntegration(),
        TestStressLevelPSS10Mapping(),
        TestPSS10StressMechanismIntegration()
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}:")

        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]

        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_class, test_method)()
                print(f"  ✓ {test_method}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {test_method}: {e}")

    print(f"\n{'=' * 50}")
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)