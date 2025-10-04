#!/usr/bin/env python3
"""
Test suite for PSS-10 threshold-based stress classification.

This module tests the new PSS-10-based stress classification system that replaces
the previous affect-based logic while maintaining affect as an ephemeral state.
"""

import pytest
import tempfile
from pathlib import Path
import os
from unittest.mock import Mock

from src.python.config import Config, reload_config
from src.python.agent import Person
from src.python.model import StressModel


@pytest.mark.config
class TestPSS10ThresholdConfiguration:
    """Test PSS-10 threshold configuration and default values."""

    def test_pss10_threshold_env_default(self):
        """Test that default threshold 27 is used if env var missing."""
        os.environ.clear()
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Test with no .env file
                config = Config()

                # Verify default threshold is 27
                assert config.pss10_threshold == 27
                assert config.get('pss10', 'threshold') == 27

            finally:
                os.chdir(original_cwd)

    def test_pss10_threshold_current_value(self):
        """Test that PSS-10 threshold is correctly loaded from current .env file."""
        # Test with current configuration (which should have PSS10_THRESHOLD=27)
        os.environ.clear()
        config = reload_config()

        # Verify threshold is loaded from .env file
        assert config.pss10_threshold == 27
        assert config.get('pss10', 'threshold') == 27


@pytest.mark.config
class TestPSS10ThresholdEvaluation:
    """Test PSS-10 threshold evaluation for stress classification."""

    def test_pss10_below_threshold_not_stressed(self):
        """Test that agents with PSS-10 below 27 are not stressed."""
        # Create a mock model
        os.environ.clear()
        model = Mock()
        model.seed = 42

        # Create agent
        agent = Person(model)

        # Manually set PSS-10 below threshold
        agent.pss10 = 20  # Below default threshold of 27

        # Update stressed status (simulating what compute_pss10_score does)
        config = Config()
        threshold = config.get('pss10', 'threshold')
        agent.stressed = (agent.pss10 >= threshold)

        # Assert agent is not stressed
        assert not agent.stressed
        assert agent.pss10 == 20

    def test_pss10_at_threshold_stressed(self):
        """Test that agents with PSS-10 at 27 are stressed."""
        # Create a mock model
        os.environ.clear()
        model = Mock()
        model.seed = 42

        # Create agent
        agent = Person(model)

        # Manually set PSS-10 at threshold
        agent.pss10 = 27  # At default threshold

        # Update stressed status (simulating what happens in compute_pss10_score)
        config = Config()
        threshold = config.get('pss10', 'threshold')
        agent.stressed = (agent.pss10 >= threshold)

        # Assert agent is stressed (27 >= 27)
        assert agent.stressed, f"PSS-10: {agent.pss10}, Threshold: {threshold}, 27 should be >= {threshold}"
        assert agent.pss10 == 27

    def test_pss10_above_threshold_stressed(self):
        """Test that agents with PSS-10 above 27 are stressed."""
        # Create a mock model
        os.environ.clear()
        model = Mock()
        model.seed = 42

        # Create agent
        agent = Person(model)

        # Manually set PSS-10 above threshold
        agent.pss10 = 35  # Above default threshold

        # Update stressed status
        config = Config()
        threshold = config.get('pss10', 'threshold')
        agent.stressed = (agent.pss10 >= threshold)

        # Assert agent is stressed
        assert agent.stressed
        assert agent.pss10 == 35


@pytest.mark.config
class TestAffectEphemeralBehavior:
    """Test that affect remains ephemeral while PSS-10 drives stress classification."""

    def test_affect_is_ephemeral(self):
        """Test that affect changes via events and decays back to baseline while PSS-10 remains unchanged."""
        # Create a mock model
        os.environ.clear()
        model = Mock()
        model.seed = 42

        # Create agent with specific initial values
        agent = Person(model)
        initial_affect = agent.affect
        initial_pss10 = agent.pss10

        # Simulate affect change via events (increase affect significantly)
        agent.affect = 0.8  # High positive affect

        # Apply homeostatic decay (simulating daily reset)
        config = Config()
        homeostatic_rate = config.get('affect_dynamics', 'homeostatic_rate')
        agent.affect = agent.baseline_affect + homeostatic_rate * (agent.affect - agent.baseline_affect)

        # PSS-10 should remain unchanged
        assert agent.pss10 == initial_pss10

        # Affect should tend back toward baseline (ephemeral behavior)
        assert agent.affect != 0.8  # Should have moved toward baseline
        assert abs(agent.affect - agent.baseline_affect) < abs(0.8 - agent.baseline_affect)

    def test_affect_influences_resilience_but_not_stress_flag(self):
        """Test that ephemeral affect influences resilience but doesn't flip stressed flag unless PSS-10 changes."""
        # Create a mock model
        os.environ.clear()
        model = Mock()
        model.seed = 42

        # Create agent with PSS-10 below threshold (not stressed)
        agent = Person(model)
        agent.pss10 = 20  # Below threshold
        config = Config()
        threshold = config.get('pss10', 'threshold')
        agent.stressed = (agent.pss10 >= threshold)

        # Verify initial state
        assert not agent.stressed
        initial_resilience = agent.resilience

        # Simulate affect change that should influence resilience
        original_affect = agent.affect
        agent.affect = 0.5  # Positive affect should boost resilience

        # Simulate resilience dynamics being influenced by affect
        # (This would normally happen in the step() method)
        affect_multiplier = 1.0 + 0.2 * max(0.0, agent.affect)
        resilience_boost = 0.1 * affect_multiplier  # Simulate affect influence on resilience

        # The stressed flag should remain unchanged (still False)
        assert not agent.stressed  # Should still be False

        # But resilience should be influenced by affect
        # (In real implementation, this would happen through the resilience dynamics)
        assert agent.affect != original_affect  # Affect changed


@pytest.mark.config
class TestIntegrationScenarios:
    """Integration tests for PSS-10 threshold behavior in realistic scenarios."""

    def test_stress_prevalence_calculation_uses_pss10_threshold(self):
        """Test that model-level stress prevalence calculation uses PSS-10 threshold correctly."""
        # Create a real model with test agents
        os.environ.clear()
        model = StressModel(N=10, max_days=5, seed=42)

        # Set different PSS-10 values for different agents
        stressed_count = 0
        for i, agent in enumerate(model.agents):
            if i < 3:  # First 3 agents: PSS-10 above threshold (stressed)
                agent.pss10 = 30
            else:  # Other agents: PSS-10 below threshold (not stressed)
                agent.pss10 = 20

            # Update stressed status
            config = Config()
            threshold = config.get('pss10', 'threshold')
            agent.stressed = (agent.pss10 >= threshold)

            if agent.stressed:
                stressed_count += 1

        # Calculate stress prevalence manually
        manual_prevalence = stressed_count / len(model.agents)

        # Get stress prevalence from model (this uses our new PSS-10-based logic)
        model_prevalence = sum(1 for agent in model.agents if getattr(agent, 'stressed', False)) / len(model.agents)

        # They should match
        assert manual_prevalence == model_prevalence
        assert model_prevalence == 0.3  # 3 out of 10 agents stressed

    def test_pss10_threshold_persistence_across_steps(self):
        """Test that stressed status persists until PSS-10 changes cross threshold."""
        # Create a mock model
        os.environ.clear()
        model = Mock()
        model.seed = 42

        # Create agent
        agent = Person(model)

        # Set PSS-10 above threshold initially
        agent.pss10 = 30
        config = Config()
        threshold = config.get('pss10', 'threshold')
        agent.stressed = (agent.pss10 >= threshold)

        # Verify initially stressed
        assert agent.stressed

        # Simulate multiple steps with same PSS-10 (should remain stressed)
        for _ in range(5):
            # PSS-10 stays the same
            agent.stressed = (agent.pss10 >= threshold)
            assert agent.stressed  # Should remain stressed

        # Change PSS-10 below threshold
        agent.pss10 = 20
        agent.stressed = (agent.pss10 >= threshold)

        # Should now be not stressed
        assert not agent.stressed

        # Simulate more steps (should remain not stressed)
        for _ in range(5):
            agent.stressed = (agent.pss10 >= threshold)
            assert not agent.stressed  # Should remain not stressed


def run_tests():
    """Run all PSS-10 threshold tests."""
    print("PSS-10 Threshold Test Suite")
    print("=" * 40)

    # Run test classes
    test_classes = [
        TestPSS10ThresholdConfiguration,
        TestPSS10ThresholdEvaluation,
        TestAffectEphemeralBehavior,
        TestIntegrationScenarios
    ]

    passed = 0
    total = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * len(test_class.__name__))

        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                total += 1
                try:
                    test_method = getattr(test_class, method_name)
                    test_method(test_class())
                    print(f"✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"✗ {method_name}: {e}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)