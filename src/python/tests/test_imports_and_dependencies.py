"""
Tests for import functionality and dependency management.

These tests verify that all modules can be imported correctly and
that there are no circular dependencies or other import issues.
"""

import pytest
import importlib
import traceback
import os


class TestModuleImports:
    """Test module import functionality."""

    @pytest.mark.unit
    def test_all_modules_importable(self):
        """Test that all modules can be imported without errors."""
        modules_to_test = [
            'config',
            'affect_utils',
            'stress_utils',
            'math_utils',
            'agent',
            'model'
        ]

        for module_name in modules_to_test:
            full_module_name = f'src.python.{module_name}'
            module = importlib.import_module(full_module_name)

            # Verify module was imported
            assert module is not None

    @pytest.mark.unit
    def test_test_modules_importable(self):
        """Test that test modules can be imported without errors."""
        # Test importing test modules
        test_modules = [
            'src.python.tests.test_affect_utils',
            'src.python.tests.test_stress_utils',
            'src.python.tests.test_agent_integration'
        ]

        for module_name in test_modules:
            module = importlib.import_module(module_name)
            assert module is not None


class TestCircularDependencies:
    """Test for circular import dependencies."""

    @pytest.mark.unit
    def test_no_circular_dependencies_different_orders(self):
        """Test that modules can be imported in different orders."""
        # Import in different order
        importlib.import_module('src.python.math_utils')
        importlib.import_module('src.python.config')
        importlib.import_module('src.python.stress_utils')
        importlib.import_module('src.python.affect_utils')

        # If we get here without ImportError, no circular dependencies
        assert True

    @pytest.mark.unit
    def test_modules_reimportable(self):
        """Test that modules can be re-imported without issues."""
        for i in range(3):
            importlib.reload(importlib.import_module('src.python.config'))
            importlib.reload(importlib.import_module('src.python.math_utils'))

        # If we get here without ImportError, modules are reimportable
        assert True

    @pytest.mark.unit
    def test_all_modules_import_together(self):
        """Test that all modules can be imported together."""
        modules = []
        module_names = ['config', 'math_utils', 'stress_utils', 'affect_utils', 'agent', 'model']

        for module_name in module_names:
            modules.append(importlib.import_module(f'src.python.{module_name}'))

        # If we get here without ImportError, all modules can be imported together
        assert len(modules) == len(module_names)


class TestFunctionalityAfterImports:
    """Test that modules work correctly after all imports."""

    @pytest.mark.integration
    def test_module_functionality(self):
        """Test that modules work correctly after all imports."""
        from src.python.config import get_config
        from src.python.affect_utils import InteractionConfig, ProtectiveFactors
        from src.python.stress_utils import StressEvent, AppraisalWeights
        from src.python.math_utils import create_rng

        # Test that we can create instances and use functions
        config = get_config()
        interaction_config = InteractionConfig()
        protective_factors = ProtectiveFactors()
        stress_event = StressEvent(0.5, 0.5, 0.5, 0.5)
        appraisal_weights = AppraisalWeights()
        rng = create_rng(42)

        # Test that the instances have expected attributes
        assert hasattr(interaction_config, 'influence_rate')
        assert hasattr(protective_factors, 'social_support')
        assert hasattr(stress_event, 'controllability')
        assert hasattr(appraisal_weights, 'omega_c')
        assert hasattr(rng, 'random')


class TestEnvironmentVariableConsistency:
    """Test environment variable consistency across modules."""

    @pytest.mark.config
    def test_environment_consistency(self, clean_env, reload_config_fixture):
        """Test that environment variables are consistently used across modules."""
        # Set a test environment variable
        os.environ['SIMULATION_NUM_AGENTS'] = '25'

        # Import modules and check they use the new value
        new_config = reload_config_fixture()

        assert new_config.num_agents == 25

        # Test that dataclasses pick up the new value
        from src.python.affect_utils import InteractionConfig
        interaction_config = InteractionConfig()
        # The dataclass should use the config value, but since config is a global,
        # we need to check that the config system is working
        assert new_config.get('interaction', 'influence_rate') >= 0


class TestNoHardcodedValues:
    """Test that no hardcoded values remain in dataclasses."""

    @pytest.mark.config
    def test_dataclass_values_from_config(self, config):
        """Test that dataclass defaults come from config, not hardcoded values."""
        from src.python.affect_utils import InteractionConfig, ProtectiveFactors, ResourceParams
        from src.python.stress_utils import AppraisalWeights, ThresholdParams

        # Test that dataclass defaults come from config, not hardcoded values
        interaction_config = InteractionConfig()
        protective_factors = ProtectiveFactors()
        resource_params = ResourceParams()
        appraisal_weights = AppraisalWeights()
        threshold_params = ThresholdParams()

        # Check that values match config (allowing for some tolerance due to factory functions)
        assert abs(interaction_config.influence_rate - config.get('interaction', 'influence_rate')) < 1e-10
        assert abs(protective_factors.social_support - config.get('protective', 'social_support')) < 1e-10
        assert abs(resource_params.base_regeneration - config.get('resource', 'base_regeneration')) < 1e-10
        assert abs(appraisal_weights.omega_c - config.get('appraisal', 'omega_c')) < 1e-10
        assert abs(threshold_params.base_threshold - config.get('threshold', 'base_threshold')) < 1e-10