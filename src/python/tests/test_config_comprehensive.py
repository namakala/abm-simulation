"""
Comprehensive tests for config.py to improve coverage.

This module tests error handling, validation, edge cases, and missing lines
identified in coverage analysis.
"""

import os
import logging
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.python.config import Config, ConfigurationError, get_config, reload_config


class TestConfigLoadEnvironment:
    """Test _load_environment method edge cases."""

    def test_missing_env_file(self, caplog):
        """Test loading when .env file does not exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                os.environ.clear()

                config = Config()

                # Check that warning was logged
                assert "not found" in caplog.text
                assert "using system environment variables" in caplog.text

            finally:
                os.chdir(original_cwd)

    def test_empty_env_file(self, caplog):
        """Test loading with empty .env file."""
        caplog.set_level(logging.DEBUG)
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                os.environ.clear()

                # Create empty .env file
                env_file = Path(temp_dir) / '.env'
                env_file.touch()

                config = Config()

                # Check that info was logged
                assert "Loaded environment from" in caplog.text

            finally:
                os.chdir(original_cwd)

    @patch('src.python.config.load_dotenv')
    def test_dotenv_import_error(self, mock_load_dotenv, caplog):
        """Test when python-dotenv is not installed."""
        mock_load_dotenv.side_effect = ImportError("No module named 'dotenv'")

        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                os.environ.clear()

                # Create .env file
                env_file = Path(temp_dir) / '.env'
                env_file.write_text("SIMULATION_NUM_AGENTS=30")

                config = Config()

                # Check that warning was logged
                assert "python-dotenv not installed" in caplog.text

            finally:
                os.chdir(original_cwd)


class TestConfigGetEnvValue:
    """Test _get_env_value method error handling."""

    def test_required_missing_variable(self):
        """Test required environment variable missing."""
        os.environ.clear()

        config = Config()
        with pytest.raises(ConfigurationError, match="Required environment variable 'MISSING_VAR' is not set"):
            config._get_env_value('MISSING_VAR', int, 42, required=True)

    def test_invalid_int_conversion(self):
        """Test invalid integer conversion."""
        os.environ['INVALID_INT'] = 'not_a_number'

        config = Config()
        with pytest.raises(ConfigurationError, match="Invalid value for 'INVALID_INT'"):
            config._get_env_value('INVALID_INT', int, 42)

    def test_invalid_float_conversion(self):
        """Test invalid float conversion."""
        os.environ['INVALID_FLOAT'] = 'not_a_number'

        config = Config()
        with pytest.raises(ConfigurationError, match="Invalid value for 'INVALID_FLOAT'"):
            config._get_env_value('INVALID_FLOAT', float, 1.0)

    def test_invalid_boolean_conversion(self):
        """Test invalid boolean conversion."""
        os.environ['INVALID_BOOL'] = 'maybe'

        config = Config()
        with pytest.raises(ConfigurationError, match="Invalid boolean value"):
            config._get_env_value('INVALID_BOOL', bool, False)

    def test_valid_boolean_values(self):
        """Test various valid boolean values."""
        test_cases = [
            ('true', True), ('1', True), ('yes', True), ('on', True),
            ('false', False), ('0', False), ('no', False), ('off', False)
        ]

        for env_value, expected in test_cases:
            os.environ['TEST_BOOL'] = env_value
            config = Config()
            result = config._get_env_value('TEST_BOOL', bool, False)
            assert result == expected

    def test_float_to_int_conversion(self):
        """Test converting float string to int."""
        os.environ['FLOAT_TO_INT'] = '42.0'

        config = Config()
        result = config._get_env_value('FLOAT_TO_INT', int, 0)
        assert result == 42

    def test_debug_log_for_default(self, caplog):
        """Test debug log when using default value."""
        caplog.set_level(logging.DEBUG)
        os.environ.clear()

        config = Config()
        result = config._get_env_value('MISSING_VAR', str, 'default')

        assert result == 'default'
        assert "Using default value for 'MISSING_VAR'" in caplog.text


class TestConfigGetEnvArray:
    """Test _get_env_array method validation and parsing."""

    def test_required_missing_array(self):
        """Test required array missing."""
        os.environ.clear()

        config = Config()
        with pytest.raises(ConfigurationError, match="Required environment variable 'MISSING_ARRAY' is not set"):
            config._get_env_array('MISSING_ARRAY', float, [1.0, 2.0], required=True)

    def test_bracket_notation_parsing(self):
        """Test bracket notation parsing."""
        os.environ['BRACKET_ARRAY'] = '[1.0, 2.0, 3.0]'

        config = Config()
        result = config._get_env_array('BRACKET_ARRAY', float, [])
        assert result == [1.0, 2.0, 3.0]

    def test_space_separated_parsing(self):
        """Test space-separated format parsing."""
        os.environ['SPACE_ARRAY'] = '1.0 2.0 3.0'

        config = Config()
        result = config._get_env_array('SPACE_ARRAY', float, [])
        assert result == [1.0, 2.0, 3.0]

    def test_empty_bracket_notation(self):
        """Test empty bracket notation."""
        os.environ['EMPTY_BRACKET'] = '[]'

        config = Config()
        result = config._get_env_array('EMPTY_BRACKET', float, [1.0])
        assert result == []

    def test_whitespace_handling(self):
        """Test whitespace handling in arrays."""
        os.environ['WHITESPACE_ARRAY'] = ' [ 1.0 , 2.0 , 3.0 ] '

        config = Config()
        result = config._get_env_array('WHITESPACE_ARRAY', float, [])
        assert result == [1.0, 2.0, 3.0]

    def test_invalid_array_length(self):
        """Test invalid array length validation."""
        os.environ['WRONG_LENGTH'] = '1.0 2.0'

        config = Config()
        with pytest.raises(ConfigurationError, match="Array 'WRONG_LENGTH' must have exactly 3 elements"):
            config._get_env_array('WRONG_LENGTH', float, [], expected_length=3)

    def test_invalid_element_conversion(self):
        """Test invalid element conversion in array."""
        os.environ['INVALID_ELEMENT'] = '1.0 not_a_number 3.0'

        config = Config()
        with pytest.raises(ConfigurationError, match="Invalid array value for 'INVALID_ELEMENT'"):
            config._get_env_array('INVALID_ELEMENT', float, [])

    def test_unsupported_array_type(self):
        """Test unsupported array type."""
        os.environ['UNSUPPORTED_TYPE'] = '1 2 3'

        config = Config()
        with pytest.raises(ConfigurationError, match="Unsupported array type"):
            config._get_env_array('UNSUPPORTED_TYPE', str, [])

    def test_debug_log_for_default_array(self, caplog):
        """Test debug log when using default array."""
        caplog.set_level(logging.DEBUG)
        os.environ.clear()

        config = Config()
        result = config._get_env_array('MISSING_ARRAY', float, [1.0, 2.0])

        assert result == [1.0, 2.0]
        assert "Using default array for 'MISSING_ARRAY'" in caplog.text


class TestConfigGetMethod:
    """Test get method error cases."""

    def test_invalid_section(self):
        """Test invalid section access."""
        config = Config()

        with pytest.raises(ConfigurationError, match="Configuration section 'invalid_section' not found"):
            config.get('invalid_section')

    def test_invalid_key_in_section(self):
        """Test invalid key in valid section."""
        config = Config()

        with pytest.raises(ConfigurationError, match="Configuration key 'invalid_key' not found in section 'simulation'"):
            config.get('simulation', 'invalid_key')

    def test_valid_section_access(self):
        """Test valid section access."""
        config = Config()

        sim_section = config.get('simulation')
        assert isinstance(sim_section, dict)
        assert 'num_agents' in sim_section

    def test_valid_key_access(self):
        """Test valid key access."""
        config = Config()

        num_agents = config.get('simulation', 'num_agents')
        assert isinstance(num_agents, int)


class TestConfigValidation:
    """Test validate method error scenarios."""

    def test_network_k_too_small(self):
        """Test network k validation."""
        os.environ['NETWORK_WATTS_K'] = '1'

        config = Config()
        with pytest.raises(ConfigurationError, match="Network k parameter must be >= 2"):
            config.validate()

    def test_network_p_out_of_range(self):
        """Test network p validation."""
        os.environ['NETWORK_WATTS_P'] = '1.5'

        config = Config()
        with pytest.raises(ConfigurationError, match="Network p parameter must be in \\[0, 1\\]"):
            config.validate()

    def test_network_adaptation_threshold_too_small(self):
        """Test adaptation threshold validation."""
        os.environ['NETWORK_ADAPTATION_THRESHOLD'] = '0'

        config = Config()
        with pytest.raises(ConfigurationError, match="Network adaptation threshold must be >= 1"):
            config.validate()

    def test_network_rewire_probability_out_of_range(self):
        """Test rewiring probability validation."""
        os.environ['NETWORK_REWIRE_PROBABILITY'] = '1.5'

        config = Config()
        with pytest.raises(ConfigurationError, match="Network rewiring probability must be in \\[0, 1\\]"):
            config.validate()

    def test_network_homophily_strength_out_of_range(self):
        """Test homophily strength validation."""
        os.environ['NETWORK_HOMOPHILY_STRENGTH'] = '1.5'

        config = Config()
        with pytest.raises(ConfigurationError, match="Network homophily strength must be in \\[0, 1\\]"):
            config.validate()

    def test_agent_initial_resilience_mean_out_of_range(self):
        """Test agent resilience mean validation."""
        os.environ['AGENT_INITIAL_RESILIENCE_MEAN'] = '1.5'

        config = Config()
        with pytest.raises(ConfigurationError, match="Agent initial resilience mean must be in \\[0, 1\\]"):
            config.validate()

    def test_agent_initial_resilience_sd_negative(self):
        """Test agent resilience SD validation."""
        os.environ['AGENT_INITIAL_RESILIENCE_SD'] = '-1.0'

        config = Config()
        with pytest.raises(ConfigurationError, match="Agent initial resilience SD must be positive"):
            config.validate()

    def test_agent_initial_affect_mean_out_of_range(self):
        """Test agent affect mean validation."""
        os.environ['AGENT_INITIAL_AFFECT_MEAN'] = '1.5'

        config = Config()
        with pytest.raises(ConfigurationError, match="Agent initial affect mean must be in \\[-1, 1\\]"):
            config.validate()

    def test_agent_initial_affect_sd_negative(self):
        """Test agent affect SD validation."""
        os.environ['AGENT_INITIAL_AFFECT_SD'] = '-1.0'

        config = Config()
        with pytest.raises(ConfigurationError, match="Agent initial affect SD must be positive"):
            config.validate()

    def test_agent_initial_resources_mean_out_of_range(self):
        """Test agent resources mean validation."""
        os.environ['AGENT_INITIAL_RESOURCES_MEAN'] = '1.5'

        config = Config()
        with pytest.raises(ConfigurationError, match="Agent initial resources mean must be in \\[0, 1\\]"):
            config.validate()

    def test_agent_initial_resources_sd_negative(self):
        """Test agent resources SD validation."""
        os.environ['AGENT_INITIAL_RESOURCES_SD'] = '-1.0'

        config = Config()
        with pytest.raises(ConfigurationError, match="Agent initial resources SD must be positive"):
            config.validate()

    def test_agent_stress_probability_out_of_range(self):
        """Test agent stress probability validation."""
        os.environ['AGENT_STRESS_PROBABILITY'] = '1.5'

        config = Config()
        with pytest.raises(ConfigurationError, match="Agent stress probability must be in \\[0, 1\\]"):
            config.validate()

    def test_stress_controllability_mean_out_of_range(self):
        """Test stress controllability mean validation."""
        os.environ['STRESS_CONTROLLABILITY_MEAN'] = '1.5'

        config = Config()
        with pytest.raises(ConfigurationError, match="Stress event means must be in \\[0, 1\\]"):
            config.validate()

    def test_stress_controllability_sd_negative(self):
        """Test stress controllability SD validation."""
        os.environ['STRESS_CONTROLLABILITY_SD'] = '-1.0'

        config = Config()
        with pytest.raises(ConfigurationError, match="Stress event standard deviations must be positive"):
            config.validate()

    def test_pss10_item_means_wrong_length(self):
        """Test PSS-10 item means length validation."""
        os.environ['PSS10_ITEM_MEAN'] = '2.1 1.8 2.3 1.9 2.2 1.7 2.0 1.6 2.4'

        with pytest.raises(ConfigurationError, match="Array 'PSS10_ITEM_MEAN' must have exactly 10 elements"):
            Config()

    def test_pss10_item_means_invalid_value(self):
        """Test PSS-10 item means value validation."""
        os.environ['PSS10_ITEM_MEAN'] = '2.1 1.8 2.3 1.9 2.2 1.7 2.0 1.6 2.4 5.0'

        config = Config()
        with pytest.raises(ConfigurationError, match="PSS-10 item mean at index 9 must be in \\[0, 4\\]"):
            config.validate()

    def test_pss10_item_sds_invalid_value(self):
        """Test PSS-10 item SDs value validation."""
        os.environ['PSS10_ITEM_SD'] = '1.1 0.9 1.2 1.0 1.1 0.8 1.0 0.9 1.3 0.0'

        config = Config()
        with pytest.raises(ConfigurationError, match="PSS-10 item standard deviation at index 9 must be positive"):
            config.validate()

    def test_pss10_load_controllability_invalid_value(self):
        """Test PSS-10 controllability loading validation."""
        os.environ['PSS10_LOAD_CONTROLLABILITY'] = '0.2 0.8 0.1 0.7 0.6 0.1 0.8 0.6 0.7 1.5'

        config = Config()
        with pytest.raises(ConfigurationError, match="PSS-10 controllability loading at index 9 must be in \\[0, 1\\]"):
            config.validate()

    def test_pss10_bifactor_correlation_out_of_range(self):
        """Test PSS-10 bifactor correlation validation."""
        os.environ['PSS10_BIFACTOR_COR'] = '1.5'

        config = Config()
        with pytest.raises(ConfigurationError, match="PSS-10 bifactor correlation must be in \\[-1, 1\\]"):
            config.validate()

    def test_pss10_controllability_sd_negative(self):
        """Test PSS-10 controllability SD validation."""
        os.environ['PSS10_CONTROLLABILITY_SD'] = '-1.0'

        config = Config()
        with pytest.raises(ConfigurationError, match="PSS-10 controllability SD must be positive"):
            config.validate()

    def test_threshold_base_threshold_out_of_range(self):
        """Test base threshold validation."""
        os.environ['THRESHOLD_BASE_THRESHOLD'] = '1.5'

        config = Config()
        with pytest.raises(ConfigurationError, match="Base threshold must be in \\[0, 1\\]"):
            config.validate()

    def test_coping_base_probability_out_of_range(self):
        """Test coping base probability validation."""
        os.environ['COPING_BASE_PROBABILITY'] = '1.5'

        config = Config()
        with pytest.raises(ConfigurationError, match="Coping base probability must be in \\[0, 1\\]"):
            config.validate()

    def test_stress_decay_rate_out_of_range(self):
        """Test stress decay rate validation."""
        os.environ['STRESS_DECAY_RATE'] = '1.5'

        config = Config()
        with pytest.raises(ConfigurationError, match="Stress decay rate must be in \\[0, 1\\]"):
            config.validate()

    def test_protective_factors_out_of_range(self):
        """Test protective factors validation."""
        os.environ['PROTECTIVE_SOCIAL_SUPPORT'] = '1.5'

        config = Config()
        with pytest.raises(ConfigurationError, match="Protective factors must be in \\[0, 1\\]"):
            config.validate()

    def test_resource_base_regeneration_negative(self):
        """Test resource regeneration validation."""
        os.environ['RESOURCE_BASE_REGENERATION'] = '-1.0'

        config = Config()
        with pytest.raises(ConfigurationError, match="Resource regeneration rate must be >= 0"):
            config.validate()

    def test_resource_allocation_cost_negative(self):
        """Test resource allocation cost validation."""
        os.environ['RESOURCE_ALLOCATION_COST'] = '-1.0'

        config = Config()
        with pytest.raises(ConfigurationError, match="Resource allocation cost must be >= 0"):
            config.validate()

    def test_resource_cost_exponent_less_than_one(self):
        """Test resource cost exponent validation."""
        os.environ['RESOURCE_COST_EXPONENT'] = '0.5'

        config = Config()
        with pytest.raises(ConfigurationError, match="Resource cost exponent must be >= 1"):
            config.validate()

    def test_protective_improvement_rate_out_of_range(self):
        """Test protective improvement rate validation."""
        os.environ['PROTECTIVE_IMPROVEMENT_RATE'] = '1.5'

        config = Config()
        with pytest.raises(ConfigurationError, match="Protective improvement rate must be in \\[0, 1\\]"):
            config.validate()

    def test_utility_softmax_temperature_negative(self):
        """Test softmax temperature validation."""
        os.environ['UTILITY_SOFTMAX_TEMPERATURE'] = '-1.0'

        config = Config()
        with pytest.raises(ConfigurationError, match="Softmax temperature must be > 0"):
            config.validate()


class TestConfigOtherMethods:
    """Test other methods like print_summary and reload_config."""

    @patch('builtins.print')
    def test_print_summary(self, mock_print):
        """Test print_summary method."""
        config = Config()
        config.print_summary()

        # Check that print was called multiple times
        assert mock_print.call_count > 10

    def test_reload_config(self):
        """Test reload_config function."""
        os.environ['SIMULATION_NUM_AGENTS'] = '50'

        try:
            new_config = reload_config()
            assert new_config.num_agents == 50

            # Check that global config is updated
            global_config = get_config()
            assert global_config.num_agents == 50

        finally:
            if 'SIMULATION_NUM_AGENTS' in os.environ:
                del os.environ['SIMULATION_NUM_AGENTS']

    def test_get_config_singleton(self):
        """Test get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


class TestConfigEdgeCases:
    """Test various edge cases."""

    def test_array_with_empty_elements(self):
        """Test array parsing with empty elements."""
        os.environ['ARRAY_WITH_EMPTIES'] = '[1.0,,2.0, ,3.0]'

        config = Config()
        result = config._get_env_array('ARRAY_WITH_EMPTIES', float, [])
        assert result == [1.0, 2.0, 3.0]

    def test_array_with_tabs_and_newlines(self):
        """Test array parsing with tabs and newlines."""
        os.environ['ARRAY_WITH_TABS'] = '1.0\t2.0\n3.0'

        config = Config()
        result = config._get_env_array('ARRAY_WITH_TABS', float, [])
        assert result == [1.0, 2.0, 3.0]

    def test_validation_passes_with_valid_config(self):
        """Test that validation passes with valid configuration."""
        config = Config()
        # Should not raise any exceptions
        config.validate()