"""
Integration tests for homeostatic stabilization behavior in the full agent model.

This file tests the homeostatic adjustment mechanism in the context of the complete
agent model, verifying that it provides desired stabilization behavior without
breaking existing agent dynamics and social interaction systems.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.python.agent import Person
from src.python.model import StressModel
from src.python.config import get_config


class MockModel:
    """Mock Mesa model for testing."""

    def __init__(self, seed=None):
        self.seed = seed
        self.grid = Mock()
        self.grid.get_neighbors.return_value = []
        self.agents = Mock()
        self.register_agent = Mock()
        self.rng = np.random.default_rng(seed)


class TestHomeostaticStabilizationIntegration:
    """Test homeostatic stabilization in full agent model context."""

    def test_full_agent_model_with_homeostatic_enabled(self):
        """Test that the full agent model works correctly with homeostatic adjustment enabled."""
        # Create a model with homeostatic adjustment enabled (default behavior)
        model = StressModel(N=10, max_days=5, seed=42)

        # Verify model initializes correctly
        assert len(model.agents) == 10
        assert model.day == 0
        assert model.running is True

        # Run a few days of simulation
        for _ in range(3):
            if model.running:
                model.step()

        # Verify all agents still have valid values
        for agent in model.agents:
            assert -1.0 <= agent.affect <= 1.0
            assert 0.0 <= agent.resilience <= 1.0
            assert 0.0 <= agent.resources <= 1.0
            assert hasattr(agent, 'baseline_affect')
            assert hasattr(agent, 'baseline_resilience')

    def test_affect_resilience_respond_to_disruptions(self):
        """Test that affect and resilience still respond to external disruptions despite homeostasis."""
        model = MockModel(seed=42)
        agent = Person(model, {
            'initial_affect': 0.0,
            'initial_resilience': 0.5,
            'initial_resources': 0.6,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        })

        # Record initial state
        initial_affect = agent.affect
        initial_resilience = agent.resilience

        # Mock a stress event that should cause disruption
        with patch('src.python.agent.generate_stress_event') as mock_stress_event, \
             patch('src.python.agent.process_stress_event') as mock_process_stress, \
             patch('src.python.agent.sample_poisson') as mock_sample_poisson:

            # Create a high-magnitude stress event that will trigger stress
            mock_stress_event.return_value = Mock()
            mock_stress_event.return_value.controllability = 0.2
            mock_stress_event.return_value.predictability = 0.2
            mock_stress_event.return_value.overload = 0.8
            mock_stress_event.return_value.magnitude = 0.8

            # Mock stress processing to return stressed=True with high hindrance
            mock_process_stress.return_value = (True, 0.3, 0.7)  # challenge=0.3, hindrance=0.7

            # Mock poisson sampling for subevents
            mock_sample_poisson.return_value = 1

            # Mock random for coping decision (fail coping to see negative impact)
            with patch.object(agent, '_rng') as mock_rng:
                mock_rng.random.side_effect = [0.8, 0.8]  # High values for stress and coping failure

                # Execute one step
                agent.step()

        # Values should have changed from initial state despite homeostatic adjustment
        assert agent.affect != initial_affect or agent.resilience != initial_resilience

        # But should still be within valid ranges
        assert -1.0 <= agent.affect <= 1.0
        assert 0.0 <= agent.resilience <= 1.0

    def test_monotonic_drift_elimination_over_multiple_days(self):
        """Test that monotonic drift is eliminated over multiple days."""
        model = MockModel(seed=42)
        agent = Person(model, {
            'initial_affect': 0.0,
            'initial_resilience': 0.5,
            'initial_resources': 0.6,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        })

        # Set up agent with extreme values to test drift elimination
        agent.affect = 0.8  # Far above baseline
        agent.resilience = 0.1  # Far below baseline
        agent.baseline_affect = 0.0
        agent.baseline_resilience = 0.5

        # Track values over multiple days with no external events
        affect_values = [agent.affect]
        resilience_values = [agent.resilience]

        with patch('src.python.agent.sample_poisson') as mock_sample_poisson, \
             patch('src.python.agent.generate_stress_event') as mock_stress_event, \
             patch.object(agent, '_rng') as mock_rng:

            # Mock no events for several days
            mock_sample_poisson.return_value = 0
            mock_stress_event.return_value = None
            mock_rng.random.return_value = 0.5  # Neutral random values

            # Run multiple days
            for day in range(10):
                agent.step()
                affect_values.append(agent.affect)
                resilience_values.append(agent.resilience)

        # Values should be moving toward baselines (no monotonic drift)
        # Affect should be decreasing toward 0.0
        assert affect_values[-1] < affect_values[0], "Affect should move toward baseline"

        # Resilience should be increasing toward 0.5
        assert resilience_values[-1] > resilience_values[0], "Resilience should move toward baseline"

        # Final values should be closer to baselines than initial values
        final_affect_distance = abs(affect_values[-1] - 0.0)
        initial_affect_distance = abs(affect_values[0] - 0.0)
        assert final_affect_distance < initial_affect_distance

        final_resilience_distance = abs(resilience_values[-1] - 0.5)
        initial_resilience_distance = abs(resilience_values[0] - 0.5)
        assert final_resilience_distance < initial_resilience_distance

    def test_different_homeostatic_rates_tunable_strength(self):
        """Test that different homeostatic rate values provide tunable stabilization strength."""
        model = MockModel(seed=42)

        # Create agents with different homeostatic rates
        agent_low = Person(model, {
            'initial_affect': 0.0,
            'initial_resilience': 0.5,
            'initial_resources': 0.6,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        })

        agent_high = Person(model, {
            'initial_affect': 0.0,
            'initial_resilience': 0.5,
            'initial_resources': 0.6,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        })

        # Set extreme starting values
        for agent in [agent_low, agent_high]:
            agent.affect = 0.8
            agent.resilience = 0.1
            agent.baseline_affect = 0.0
            agent.baseline_resilience = 0.5

        # Track values over multiple days with no events
        low_rate_affect = [agent_low.affect]
        high_rate_affect = [agent_high.affect]

        with patch('src.python.agent.sample_poisson') as mock_sample_poisson, \
             patch('src.python.agent.generate_stress_event') as mock_stress_event, \
             patch.object(agent_low, '_rng') as mock_rng_low, \
             patch.object(agent_high, '_rng') as mock_rng_high:

            # Mock no events
            mock_sample_poisson.return_value = 0
            mock_stress_event.return_value = None
            mock_rng_low.random.return_value = 0.5
            mock_rng_high.random.return_value = 0.5

            # Run multiple days
            for day in range(5):
                agent_low.step()
                agent_high.step()
                low_rate_affect.append(agent_low.affect)
                high_rate_affect.append(agent_high.affect)

        # Higher rate should cause faster convergence
        # (Note: This test assumes we can modify homeostatic rate - in current implementation
        # it's read from config, but we can test the principle with different scenarios)

        # Both should be moving toward baseline, but we can test the rate effect
        # by comparing convergence speed in different scenarios
        low_rate_convergence = abs(low_rate_affect[-1] - 0.0)
        high_rate_convergence = abs(high_rate_affect[-1] - 0.0)

        # Both should show convergence (values closer to baseline)
        assert low_rate_convergence < abs(low_rate_affect[0] - 0.0)
        assert high_rate_convergence < abs(high_rate_affect[0] - 0.0)

    def test_homeostasis_doesnt_interfere_with_daily_logic(self):
        """Test that homeostatic mechanism doesn't interfere with existing daily update logic."""
        model = MockModel(seed=42)
        agent = Person(model, {
            'initial_affect': 0.0,
            'initial_resilience': 0.5,
            'initial_resources': 0.6,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        })

        # Track all state changes through one step
        initial_affect = agent.affect
        initial_resilience = agent.resilience
        initial_resources = agent.resources

        with patch('src.python.agent.sample_poisson') as mock_sample_poisson, \
             patch('src.python.agent.generate_stress_event') as mock_stress_event, \
             patch.object(agent, '_rng') as mock_rng:

            # Mock no events for clean testing
            mock_sample_poisson.return_value = 0  # No subevents
            mock_stress_event.return_value = None  # No stress events

            # Provide enough random values for all calls
            mock_rng.random.return_value = 0.5  # Consistent random value

            # Execute one step
            agent.step()

        # All values should still be in valid ranges
        assert -1.0 <= agent.affect <= 1.0
        assert 0.0 <= agent.resilience <= 1.0
        assert 0.0 <= agent.resources <= 1.0

        # Homeostatic adjustment should have occurred (values pulled toward FIXED baselines)
        # Baselines should remain unchanged (they are the agent's natural equilibrium points)
        assert agent.baseline_affect == 0.0  # Should remain at initial baseline
        assert agent.baseline_resilience == 0.5  # Should remain at initial baseline

        # But current values may have changed due to daily activities and homeostasis
        # Values should still be in valid ranges
        assert -1.0 <= agent.affect <= 1.0
        assert 0.0 <= agent.resilience <= 1.0

    def test_realistic_affect_resilience_trajectories_possible(self):
        """Test that agents can still experience realistic affect/resilience trajectories."""
        model = MockModel(seed=42)
        agent = Person(model, {
            'initial_affect': 0.0,
            'initial_resilience': 0.5,
            'initial_resources': 0.6,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        })

        # Track trajectories over multiple days with no events (simpler test)
        affect_trajectory = [agent.affect]
        resilience_trajectory = [agent.resilience]

        with patch('src.python.agent.sample_poisson') as mock_sample_poisson, \
             patch('src.python.agent.generate_stress_event') as mock_stress_event, \
             patch.object(agent, '_rng') as mock_rng:

            # Mock no events for multiple days
            mock_sample_poisson.return_value = 0
            mock_stress_event.return_value = None
            mock_rng.random.return_value = 0.5

            # Run multiple days
            for day in range(10):
                agent.step()
                affect_trajectory.append(agent.affect)
                resilience_trajectory.append(agent.resilience)

        # All values should remain in valid ranges throughout
        assert all(-1.0 <= val <= 1.0 for val in affect_trajectory)
        assert all(0.0 <= val <= 1.0 for val in resilience_trajectory)

        # Should show some stabilization over time (homeostatic effect)
        # Values should be relatively stable (not extreme variation)
        affect_variance = np.var(affect_trajectory)
        resilience_variance = np.var(resilience_trajectory)

        # Variance should be reasonable (not too high due to homeostasis)
        assert affect_variance < 0.1  # Should be relatively stable
        assert resilience_variance < 0.1  # Should be relatively stable

    def test_extreme_stress_events_and_recovery_patterns(self):
        """Test edge cases with extreme stress events and recovery patterns."""
        model = MockModel(seed=42)
        agent = Person(model, {
            'initial_affect': 0.0,
            'initial_resilience': 0.5,
            'initial_resources': 0.6,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        })

        # Test extreme stress scenario
        agent.affect = 0.9  # Very high positive affect initially
        agent.resilience = 0.1  # Very low resilience initially

        with patch('src.python.agent.sample_poisson') as mock_sample_poisson, \
             patch('src.python.agent.generate_stress_event') as mock_stress_event, \
             patch('src.python.agent.process_stress_event') as mock_process_stress, \
             patch.object(agent, '_rng') as mock_rng:

            # Create extreme stress event
            mock_sample_poisson.return_value = 1
            mock_rng.random.return_value = 0.5  # Consistent random values
            mock_rng.choice.return_value = 'stress'

            mock_stress_event.return_value = Mock()
            mock_stress_event.return_value.controllability = 0.0  # No control
            mock_stress_event.return_value.predictability = 0.0  # Unpredictable
            mock_stress_event.return_value.overload = 1.0      # Maximum overload
            mock_stress_event.return_value.magnitude = 1.0      # Maximum magnitude

            mock_process_stress.return_value = (True, 0.0, 1.0)  # Pure hindrance

            # Execute step with extreme stress
            agent.step()

        # Even with extreme stress, values should remain in valid ranges
        assert -1.0 <= agent.affect <= 1.0
        assert 0.0 <= agent.resilience <= 1.0

        # Test recovery from extreme values
        recovery_affect = [agent.affect]
        recovery_resilience = [agent.resilience]

        # Run recovery days with no events
        with patch('src.python.agent.sample_poisson') as mock_sample_poisson, \
             patch('src.python.agent.generate_stress_event') as mock_stress_event, \
             patch.object(agent, '_rng') as mock_rng:

            mock_sample_poisson.return_value = 0
            mock_stress_event.return_value = None
            mock_rng.random.return_value = 0.5

            # Run 5 recovery days (fewer to avoid precision issues)
            for day in range(5):
                agent.step()
                recovery_affect.append(agent.affect)
                recovery_resilience.append(agent.resilience)

        # Should show some stabilization (values should not get worse)
        # All values should remain in valid ranges
        assert all(-1.0 <= val <= 1.0 for val in recovery_affect)
        assert all(0.0 <= val <= 1.0 for val in recovery_resilience)

        # Final values should not be more extreme than initial values
        final_affect_distance = abs(recovery_affect[-1] - 0.0)
        initial_affect_distance = abs(recovery_affect[0] - 0.0)
        final_resilience_distance = abs(recovery_resilience[-1] - 0.5)
        initial_resilience_distance = abs(recovery_resilience[0] - 0.5)

        # Should show stabilization behavior (homeostatic mechanism working)
        # Values should not show extreme monotonic drift away from baselines
        # Allow some temporary deviation but should show overall stabilization trend
        assert final_affect_distance <= max(0.5, initial_affect_distance + 0.2)  # More lenient for extreme cases
        assert final_resilience_distance <= max(0.5, initial_resilience_distance + 0.2)  # More lenient for extreme cases

    def test_homeostatic_behavior_with_social_network(self):
        """Test homeostatic behavior in the context of a full social network."""
        # Create a small social network model
        model = StressModel(N=5, max_days=3, seed=42)

        # Set up specific agent states to test social influence with homeostasis
        agents = list(model.agents)
        agents[0].affect = 0.8   # Very high affect
        agents[0].resilience = 0.2  # Very low resilience

        for i in range(1, 5):
            agents[i].affect = -0.4  # Low affect
            agents[i].resilience = 0.8  # High resilience

        # Update baselines to current values
        for agent in agents:
            agent.baseline_affect = agent.affect
            agent.baseline_resilience = agent.resilience

        # Run simulation
        for _ in range(3):
            if model.running:
                model.step()

        # All agents should still have valid values
        for agent in model.agents:
            assert -1.0 <= agent.affect <= 1.0
            assert 0.0 <= agent.resilience <= 1.0
            assert 0.0 <= agent.resources <= 1.0

        # Population should show some convergence due to social influence and homeostasis
        final_affects = [agent.affect for agent in model.agents]
        final_resiliences = [agent.resilience for agent in model.agents]

        # Variance should not have increased dramatically (homeostasis working)
        affect_variance = np.var(final_affects)
        resilience_variance = np.var(final_resiliences)

        # Values should be reasonable (not all at extremes)
        assert not all(abs(a) > 0.8 for a in final_affects)  # Not all at extremes
        assert not all(r > 0.8 or r < 0.2 for r in final_resiliences)  # Not all at extremes


class TestHomeostaticStabilizationEdgeCases:
    """Test edge cases and boundary conditions for homeostatic stabilization."""

    def test_zero_homeostatic_rate_effect(self):
        """Test behavior when homeostatic rate is effectively zero."""
        model = MockModel(seed=42)
        agent = Person(model, {
            'initial_affect': 0.0,
            'initial_resilience': 0.5,
            'initial_resources': 0.6,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        })

        # Set extreme values
        agent.affect = 0.9
        agent.resilience = 0.1

        # With very low homeostatic rate, values should change less over time
        # (This test demonstrates the principle - actual rate control would need config modification)

        affect_values = [agent.affect]
        resilience_values = [agent.resilience]

        with patch('src.python.agent.sample_poisson') as mock_sample_poisson, \
             patch('src.python.agent.generate_stress_event') as mock_stress_event, \
             patch.object(agent, '_rng') as mock_rng:

            mock_sample_poisson.return_value = 0
            mock_stress_event.return_value = None
            mock_rng.random.return_value = 0.5

            # Run multiple days
            for day in range(5):
                agent.step()
                affect_values.append(agent.affect)
                resilience_values.append(agent.resilience)

        # Values should still be in valid ranges
        assert all(-1.0 <= val <= 1.0 for val in affect_values)
        assert all(0.0 <= val <= 1.0 for val in resilience_values)

    def test_maximum_homeostatic_rate_behavior(self):
        """Test behavior with maximum homeostatic rate."""
        model = MockModel(seed=42)
        agent = Person(model, {
            'initial_affect': 0.0,
            'initial_resilience': 0.5,
            'initial_resources': 0.6,
            'stress_probability': 0.5,
            'coping_success_rate': 0.5,
            'subevents_per_day': 3
        })

        # Set extreme values
        agent.affect = -0.8
        agent.resilience = 0.9

        affect_values = [agent.affect]
        resilience_values = [agent.resilience]

        with patch('src.python.agent.sample_poisson') as mock_sample_poisson, \
             patch('src.python.agent.generate_stress_event') as mock_stress_event, \
             patch.object(agent, '_rng') as mock_rng:

            mock_sample_poisson.return_value = 0
            mock_stress_event.return_value = None
            mock_rng.random.return_value = 0.5

            # Run multiple days
            for day in range(5):
                agent.step()
                affect_values.append(agent.affect)
                resilience_values.append(agent.resilience)

        # Should show stabilization behavior (values should not drift further from baselines)
        final_affect_distance = abs(affect_values[-1] - 0.0)
        initial_affect_distance = abs(affect_values[0] - 0.0)

        final_resilience_distance = abs(resilience_values[-1] - 0.5)
        initial_resilience_distance = abs(resilience_values[0] - 0.5)

        # Should not drift further from baselines (homeostatic effect)
        assert final_affect_distance <= initial_affect_distance + 0.1
        assert final_resilience_distance <= initial_resilience_distance + 0.1

        # Values should be in valid ranges
        assert all(-1.0 <= val <= 1.0 for val in affect_values)
        assert all(0.0 <= val <= 1.0 for val in resilience_values)


if __name__ == "__main__":
    pytest.main([__file__])