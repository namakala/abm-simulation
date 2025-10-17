"""
Integration tests for configuration system across all modules.

These tests verify that the configuration system works correctly across
all modules and that environment variables propagate properly.
"""

import os
import pytest
import numpy as np
from src.python.affect_utils import (
    InteractionConfig, ProtectiveFactors, ResourceParams,
    compute_social_influence
)
from src.python.stress_utils import (
    StressEvent, AppraisalWeights, ThresholdParams,
    generate_stress_event, apply_weights, evaluate_stress_threshold
)
from src.python.math_utils import RNGConfig, clamp


class TestModuleIntegration:
    """Test integration between configuration and all modules."""

    @pytest.mark.integration
    def test_dataclass_instantiation(self, config):
        """Test that all dataclasses can be instantiated with config values."""
        # Test instantiation
        interaction_config = InteractionConfig()
        protective_factors = ProtectiveFactors()
        resource_params = ResourceParams()
        appraisal_weights = AppraisalWeights()
        threshold_params = ThresholdParams()
        rng_config = RNGConfig()

        # Verify they exist (basic smoke test)
        assert interaction_config is not None
        assert protective_factors is not None
        assert resource_params is not None
        assert appraisal_weights is not None
        assert threshold_params is not None
        assert rng_config is not None

    @pytest.mark.integration
    def test_utility_functions_with_config(self, config, sample_rng, sample_stress_event):
        """Test that utility functions work with configuration values."""
        # Test social influence computation
        influence = compute_social_influence(0.5, 0.3, InteractionConfig())
        assert isinstance(influence, (int, float))

        # Test stress event generation
        stress_event = generate_stress_event(sample_rng)
        assert isinstance(stress_event, StressEvent)
        assert 0 <= stress_event.controllability <= 1

        # Test appraisal weights
        appraisal_weights = AppraisalWeights()
        challenge, hindrance = apply_weights(stress_event, appraisal_weights)
        assert isinstance(challenge, (int, float))
        assert isinstance(hindrance, (int, float))
        assert 0 <= challenge <= 1
        assert 0 <= hindrance <= 1

        # Test threshold evaluation
        appraised_stress = 0.6
        threshold_params = ThresholdParams()
        is_stressed = evaluate_stress_threshold(
            appraised_stress, challenge, hindrance, threshold_params
        )
        # Handle both Python bool and numpy bool
        assert hasattr(is_stressed, '__bool__') or isinstance(is_stressed, bool)

    @pytest.mark.integration
    def test_agent_config_values(self, config):
        """Test that agent configuration values are in valid ranges."""
        # Test that agent config values match global config
        expected_resilience = config.get('agent', 'initial_resilience_mean')
        expected_affect = config.get('agent', 'initial_affect_mean')
        expected_resources = config.get('agent', 'initial_resources_mean')

        assert 0 <= expected_resilience <= 1
        assert -1 <= expected_affect <= 1
        assert 0 <= expected_resources <= 1

    @pytest.mark.integration
    def test_model_config_values(self, config):
        """Test that model configuration values are in valid ranges."""
        expected_num_agents = config.get('simulation', 'num_agents')
        expected_network_k = config.get('network', 'watts_k')
        expected_network_p = config.get('network', 'watts_p')

        assert expected_num_agents > 0
        assert expected_network_k >= 2
        assert 0 <= expected_network_p <= 1

    @pytest.mark.integration
    def test_cross_module_data_flow(self, config, sample_rng):
        """Test data flow between different modules."""
        # Generate a stress event
        stress_event = generate_stress_event(sample_rng)
        appraisal_weights = AppraisalWeights()
        threshold_params = ThresholdParams()

        # Apply appraisal
        challenge, hindrance = apply_weights(stress_event, appraisal_weights)

        # Evaluate threshold
        appraised_stress = 0.5  # Mock appraised stress
        is_stressed = evaluate_stress_threshold(
            appraised_stress, challenge, hindrance, threshold_params
        )

        # Use in social interaction if not stressed
        if not is_stressed:
            interaction_config = InteractionConfig()
            affect_change = compute_social_influence(0.5, 0.3, interaction_config)
            new_affect = clamp(0.5 + affect_change, -1, 1)
            assert -1 <= new_affect <= 1


class TestConfigConsistency:
    """Test configuration consistency across modules."""

    @pytest.mark.config
    def test_parameter_ranges(self, config):
        """Test that all configuration parameters are in valid ranges."""
        assert config.get('simulation', 'num_agents') > 0
        assert config.get('network', 'watts_k') >= 2
        assert 0 <= config.get('network', 'watts_p') <= 1
        assert 0 <= config.get('agent', 'initial_resilience_mean') <= 1
        assert -1 <= config.get('agent', 'initial_affect_mean') <= 1
        assert 0 <= config.get('agent', 'initial_resources_mean') <= 1


class TestEnvironmentVariablePropagation:
    """Test environment variable propagation through modules."""

    @pytest.mark.config
    def test_environment_override(self, config, clean_env, reload_config_fixture):
        """Test that environment variables properly override defaults."""
        # Set test environment variables
        test_env_vars = {
            'SIMULATION_NUM_AGENTS': '30',
            'AGENT_INITIAL_RESILIENCE_MEAN': '0.8',
            'NETWORK_WATTS_K': '6'
        }

        for key, value in test_env_vars.items():
            os.environ[key] = value

        try:
            # Reload configuration
            new_config = reload_config_fixture()

            # Test that new values are reflected
            assert new_config.num_agents == 30
            assert new_config.agent_initial_resilience_mean == 0.8
            assert new_config.network_watts_k == 6

            # Test that dataclasses pick up new values
            interaction_config = InteractionConfig()
            assert interaction_config.influence_rate == new_config.get('interaction', 'influence_rate')

        finally:
            # Clean up environment variables
            for key in test_env_vars.keys():
                if key in os.environ:
                    del os.environ[key]


class TestErrorHandling:
    """Test error handling in configuration system."""

    @pytest.mark.config
    def test_invalid_environment_variable(self, clean_env, reload_config_fixture):
        """Test that invalid environment variables raise appropriate errors."""
        os.environ['SIMULATION_NUM_AGENTS'] = 'invalid_number'

        try:
            with pytest.raises(Exception) as exc_info:
                reload_config_fixture()
            assert "Invalid value" in str(exc_info.value)
        finally:
            # Clean up
            if 'SIMULATION_NUM_AGENTS' in os.environ:
                del os.environ['SIMULATION_NUM_AGENTS']