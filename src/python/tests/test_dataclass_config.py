"""
Tests for dataclass configuration integration.

These tests verify that dataclasses properly use configuration values
and respond correctly to environment variable changes.
"""

import pytest
import os


class TestDataclassConfigIntegration:
    """Test that dataclasses properly integrate with configuration system."""

    @pytest.mark.config
    def test_interaction_config_uses_config_values(self, config):
        """Test that InteractionConfig uses config values correctly."""
        from src.python.affect_utils import InteractionConfig

        interaction_config = InteractionConfig()
        expected_influence_rate = config.get('interaction', 'influence_rate')
        expected_resilience_influence = config.get('interaction', 'resilience_influence')
        expected_max_neighbors = config.get('interaction', 'max_neighbors')

        assert interaction_config.influence_rate == expected_influence_rate
        assert interaction_config.resilience_influence == expected_resilience_influence
        assert interaction_config.max_neighbors == expected_max_neighbors

    @pytest.mark.config
    def test_protective_factors_uses_config_values(self, config):
        """Test that ProtectiveFactors uses config values correctly."""
        from src.python.affect_utils import ProtectiveFactors

        protective_factors = ProtectiveFactors()
        expected_social = config.get('protective', 'social_support')
        expected_family = config.get('protective', 'family_support')
        expected_formal = config.get('protective', 'formal_intervention')
        expected_psychological = config.get('protective', 'psychological_capital')

        assert protective_factors.social_support == expected_social
        assert protective_factors.family_support == expected_family
        assert protective_factors.formal_intervention == expected_formal
        assert protective_factors.psychological_capital == expected_psychological

    @pytest.mark.config
    def test_resource_params_uses_config_values(self, config):
        """Test that ResourceParams uses config values correctly."""
        from src.python.affect_utils import ResourceParams

        resource_params = ResourceParams()
        expected_regeneration = config.get('resource', 'base_regeneration')
        expected_cost = config.get('resource', 'allocation_cost')
        expected_exponent = config.get('resource', 'cost_exponent')

        assert resource_params.base_regeneration == expected_regeneration
        assert resource_params.allocation_cost == expected_cost
        assert resource_params.cost_exponent == expected_exponent

    @pytest.mark.config
    def test_appraisal_weights_uses_config_values(self, config):
        """Test that AppraisalWeights uses config values correctly."""
        from src.python.stress_utils import AppraisalWeights

        appraisal_weights = AppraisalWeights()
        expected_omega_c = config.get('appraisal', 'omega_c')
        expected_omega_o = config.get('appraisal', 'omega_o')
        expected_bias = config.get('appraisal', 'bias')
        expected_gamma = config.get('appraisal', 'gamma')

        assert appraisal_weights.omega_c == expected_omega_c
        assert appraisal_weights.omega_o == expected_omega_o
        assert appraisal_weights.bias == expected_bias
        assert appraisal_weights.gamma == expected_gamma

    @pytest.mark.config
    def test_threshold_params_uses_config_values(self, config):
        """Test that ThresholdParams uses config values correctly."""
        from src.python.stress_utils import ThresholdParams

        threshold_params = ThresholdParams()
        expected_base = config.get('threshold', 'base_threshold')
        expected_challenge = config.get('threshold', 'challenge_scale')
        expected_hindrance = config.get('threshold', 'hindrance_scale')

        assert threshold_params.base_threshold == expected_base
        assert threshold_params.challenge_scale == expected_challenge
        assert threshold_params.hindrance_scale == expected_hindrance

    @pytest.mark.config
    def test_rng_config_no_config_dependency(self):
        """Test that RNGConfig correctly doesn't use config values."""
        from src.python.math_utils import RNGConfig

        rng_config = RNGConfig()
        assert rng_config.seed is None
        assert rng_config.generator is None


class TestConfigEnvironmentOverride:
    """Test environment variable override functionality."""

    @pytest.mark.config
    def test_environment_variable_override(self, clean_env, reload_config_fixture):
        """Test that environment variables properly override defaults."""
        # Set a test environment variable
        os.environ['SIMULATION_NUM_AGENTS'] = '50'

        # Reload config to pick up new environment variable
        new_config = reload_config_fixture()

        # Check that the new value is used
        assert new_config.num_agents == 50


class TestModuleImports:
    """Test module import functionality."""

    @pytest.mark.unit
    def test_no_import_errors(self):
        """Test that all modules can be imported without errors."""
        # Test importing all main modules
        from src.python import config, affect_utils, stress_utils, math_utils, agent, model

        # Test importing test modules
        import src.python.tests.test_affect_utils as test_affect
        import src.python.tests.test_stress_utils as test_stress
        import src.python.tests.test_agent_integration as test_agent

    @pytest.mark.unit
    def test_no_circular_dependencies(self):
        """Test for circular import dependencies."""
        # This should not cause any circular import issues
        from src.python.config import get_config
        from src.python.affect_utils import InteractionConfig, ProtectiveFactors
        from src.python.stress_utils import StressEvent, AppraisalWeights
        from src.python.math_utils import RNGConfig

        # Try to use the dataclasses
        config = get_config()
        interaction = InteractionConfig()
        protective = ProtectiveFactors()
        stress_event = StressEvent(0.5, 0.5)
        weights = AppraisalWeights()
        rng_config = RNGConfig()

        # If we get here without ImportError, no circular dependencies
        assert True
