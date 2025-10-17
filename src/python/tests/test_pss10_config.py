"""
Comprehensive tests for PSS-10 configuration implementation.

Tests PSS-10 array loading, validation, and access methods with various scenarios:
- Default values (no .env file)
- Valid .env file with PSS-10 arrays
- Invalid array length (should raise ConfigurationError)
- Invalid array values (should raise ConfigurationError)
- Array access methods (direct property, dictionary, individual element access)
- Integration with existing configuration system
"""

import os
import pytest
import tempfile
from pathlib import Path

from src.python.config import Config, ConfigurationError, get_config, reload_config


@pytest.mark.config
class TestPSS10ConfigurationDefaults:
    """Test PSS-10 configuration with default values (no .env file)."""

    def test_pss10_defaults_no_env_file(self):
        """Test PSS-10 configuration loads default values when no .env file exists."""
        # Create a temporary directory without .env file
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Clear all environment variables
                os.environ.clear()

                # Create config without .env file
                config = Config()

                # Check default PSS-10 values
                expected_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
                expected_sds = [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]

                assert config.pss10_item_means == expected_means
                assert config.pss10_item_sds == expected_sds
                assert len(config.pss10_item_means) == 10
                assert len(config.pss10_item_sds) == 10

                # Test dictionary access
                pss10_dict = config.get('pss10')
                assert pss10_dict['item_means'] == expected_means
                assert pss10_dict['item_sds'] == expected_sds

                # Test individual element access
                assert config.pss10_item_means[0] == 2.1
                assert config.pss10_item_means[9] == 1.5
                assert config.pss10_item_sds[0] == 1.1
                assert config.pss10_item_sds[9] == 0.8

            finally:
                os.chdir(original_cwd)

    def test_pss10_defaults_with_empty_env_file(self):
        """Test PSS-10 configuration loads default values when .env file exists but is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Clear all environment variables
                os.environ.clear()

                # Create empty .env file
                env_file = Path(temp_dir) / '.env'
                env_file.touch()

                # Create config with empty .env file
                config = Config()

                # Check default PSS-10 values
                expected_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
                expected_sds = [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]

                assert config.pss10_item_means == expected_means
                assert config.pss10_item_sds == expected_sds

            finally:
                os.chdir(original_cwd)


@pytest.mark.config
class TestPSS10ConfigurationBracketNotation:
    """Test PSS-10 configuration with bracket notation format."""

    def test_pss10_bracket_notation_format(self):
        """Test PSS-10 configuration loads correctly from bracket notation format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Clear all environment variables
                os.environ.clear()

                # Create .env file with bracket notation PSS-10 values
                env_content = """PSS10_ITEM_MEAN=[2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
PSS10_ITEM_SD=[1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]
"""
                env_file = Path(temp_dir) / '.env'
                env_file.write_text(env_content)

                # Create config with .env file
                config = Config(str(env_file))

                # Check loaded PSS-10 values
                expected_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
                expected_sds = [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]

                assert config.pss10_item_means == expected_means
                assert config.pss10_item_sds == expected_sds

                # Test dictionary access
                pss10_dict = config.get('pss10')
                assert pss10_dict['item_means'] == expected_means
                assert pss10_dict['item_sds'] == expected_sds

                # Test individual element access
                assert config.pss10_item_means[0] == 2.1
                assert config.pss10_item_means[4] == 2.2
                assert config.pss10_item_means[9] == 1.5
                assert config.pss10_item_sds[0] == 1.1
                assert config.pss10_item_sds[8] == 1.3

            finally:
                os.chdir(original_cwd)

    def test_pss10_bracket_notation_with_whitespace(self):
        """Test PSS-10 configuration handles bracket notation with extra whitespace."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Clear all environment variables
                os.environ.clear()

                # Create .env file with bracket notation and extra whitespace
                env_content = """PSS10_ITEM_MEAN= [ 2.1 , 1.8 , 2.3 , 1.9 , 2.2 , 1.7 , 2.0 , 1.6 , 2.4 , 1.5 ]
PSS10_ITEM_SD= [1.1,  0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8 ]
"""
                env_file = Path(temp_dir) / '.env'
                env_file.write_text(env_content)

                # Should handle whitespace correctly
                config = Config(str(env_file))

                expected_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
                expected_sds = [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]

                assert config.pss10_item_means == expected_means
                assert config.pss10_item_sds == expected_sds

            finally:
                os.chdir(original_cwd)

    def test_pss10_mixed_bracket_and_space_format(self):
        """Test that both bracket notation and space-separated format work in same config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Clear all environment variables
                os.environ.clear()

                # Create .env file mixing both formats (should work with either)
                env_content = """PSS10_ITEM_MEAN=[2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
PSS10_ITEM_SD=1.1 0.9 1.2 1.0 1.1 0.8 1.0 0.9 1.3 0.8
"""
                env_file = Path(temp_dir) / '.env'
                env_file.write_text(env_content)

                # Should load successfully with mixed formats
                config = Config(str(env_file))

                expected_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
                expected_sds = [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]

                assert config.pss10_item_means == expected_means
                assert config.pss10_item_sds == expected_sds

            finally:
                os.chdir(original_cwd)


@pytest.mark.config
class TestPSS10ConfigurationValidValues:
    """Test PSS-10 configuration with valid .env file values."""

    def test_pss10_valid_env_values(self):
        """Test PSS-10 configuration loads correctly from valid .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Clear all environment variables
                os.environ.clear()

                # Create .env file with valid PSS-10 values
                env_content = """PSS10_ITEM_MEAN=2.1 1.8 2.3 1.9 2.2 1.7 2.0 1.6 2.4 1.5
PSS10_ITEM_SD=1.1 0.9 1.2 1.0 1.1 0.8 1.0 0.9 1.3 0.8
"""
                env_file = Path(temp_dir) / '.env'
                env_file.write_text(env_content)

                # Create config with .env file
                config = Config(str(env_file))

                # Check loaded PSS-10 values
                expected_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
                expected_sds = [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]

                assert config.pss10_item_means == expected_means
                assert config.pss10_item_sds == expected_sds

                # Test dictionary access
                pss10_dict = config.get('pss10')
                assert pss10_dict['item_means'] == expected_means
                assert pss10_dict['item_sds'] == expected_sds

                # Test individual element access
                assert config.pss10_item_means[0] == 2.1
                assert config.pss10_item_means[4] == 2.2
                assert config.pss10_item_means[9] == 1.5
                assert config.pss10_item_sds[0] == 1.1
                assert config.pss10_item_sds[8] == 1.3

            finally:
                os.chdir(original_cwd)

    def test_pss10_boundary_values(self):
        """Test PSS-10 configuration with boundary values (0 and 4 for means, positive SDs)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Clear all environment variables
                os.environ.clear()

                # Create .env file with boundary PSS-10 values
                env_content = """PSS10_ITEM_MEAN=0.0 4.0 0.0 4.0 0.0 4.0 0.0 4.0 0.0 4.0
PSS10_ITEM_SD=0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
"""
                env_file = Path(temp_dir) / '.env'
                env_file.write_text(env_content)

                # Create config with .env file
                config = Config(str(env_file))

                # Check boundary PSS-10 values
                expected_means = [0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0]
                expected_sds = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

                assert config.pss10_item_means == expected_means
                assert config.pss10_item_sds == expected_sds

            finally:
                os.chdir(original_cwd)


@pytest.mark.config
class TestPSS10ConfigurationInvalidValues:
    """Test PSS-10 configuration error handling for invalid values."""

    def test_pss10_invalid_array_length(self):
        """Test PSS-10 configuration raises error for wrong array length."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Clear all environment variables
                os.environ.clear()

                # Create .env file with wrong number of values (9 instead of 10)
                env_content = "PSS10_ITEM_MEAN=2.1 1.8 2.3 1.9 2.2 1.7 2.0 1.6 2.4"
                env_file = Path(temp_dir) / '.env'
                env_file.write_text(env_content)

                # Should raise ConfigurationError for wrong length during validation
                with pytest.raises(ConfigurationError, match="Array 'PSS10_ITEM_MEAN' must have exactly 10 elements"):
                    Config(str(env_file))

            finally:
                os.chdir(original_cwd)

    def test_pss10_invalid_array_length_too_many(self):
        """Test PSS-10 configuration raises error for too many values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Clear all environment variables
                os.environ.clear()

                # Create .env file with too many values (11 instead of 10)
                env_content = "PSS10_ITEM_MEAN=2.1 1.8 2.3 1.9 2.2 1.7 2.0 1.6 2.4 1.5 2.0"
                env_file = Path(temp_dir) / '.env'
                env_file.write_text(env_content)

                # Should raise ConfigurationError for wrong length during validation
                with pytest.raises(ConfigurationError, match="Array 'PSS10_ITEM_MEAN' must have exactly 10 elements"):
                    Config(str(env_file))

            finally:
                os.chdir(original_cwd)

    def test_pss10_invalid_mean_values(self):
        """Test PSS-10 configuration raises error for invalid mean values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Clear all environment variables
                os.environ.clear()

                # Create .env file with invalid mean values (outside [0,4] range)
                env_content = "PSS10_ITEM_MEAN=2.1 1.8 2.3 1.9 2.2 1.7 2.0 1.6 2.4 5.0"
                env_file = Path(temp_dir) / '.env'
                env_file.write_text(env_content)

                # Should raise ConfigurationError for invalid mean value during validation
                config = Config(str(env_file))
                with pytest.raises(ConfigurationError, match="index 9 must be in \\[0, 4\\], got 5.0"):
                    config.validate()

            finally:
                os.chdir(original_cwd)

    def test_pss10_invalid_sd_values(self):
        """Test PSS-10 configuration raises error for invalid standard deviation values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Clear all environment variables
                os.environ.clear()

                # Create .env file with invalid SD values (negative or zero)
                env_content = """PSS10_ITEM_MEAN=2.1 1.8 2.3 1.9 2.2 1.7 2.0 1.6 2.4 3.0
PSS10_ITEM_SD=1.1 0.9 1.2 1.0 1.1 0.8 1.0 0.9 1.3 0.0
"""
                env_file = Path(temp_dir) / '.env'
                env_file.write_text(env_content)

                # Should raise ConfigurationError for invalid SD value during validation
                config = Config(str(env_file))
                with pytest.raises(ConfigurationError, match="PSS-10 item standard deviation at index 9 must be positive"):
                    config.validate()

            finally:
                os.chdir(original_cwd)

    def test_pss10_invalid_float_conversion(self):
        """Test PSS-10 configuration raises error for non-numeric values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Clear all environment variables
                os.environ.clear()

                # Create .env file with non-numeric values
                env_content = "PSS10_ITEM_MEAN=2.1 1.8 2.3 1.9 2.2 1.7 2.0 1.6 2.4 invalid"
                env_file = Path(temp_dir) / '.env'
                env_file.write_text(env_content)

                # Should raise ConfigurationError for invalid float conversion during Config creation
                with pytest.raises(ConfigurationError, match="Invalid array value"):
                    Config(str(env_file))

            finally:
                os.chdir(original_cwd)


@pytest.mark.config
class TestPSS10ConfigurationAccessMethods:
    """Test PSS-10 configuration access methods."""

    def test_pss10_direct_property_access(self):
        """Test direct property access to PSS-10 arrays."""
        os.environ.clear()
        config = Config()

        # Test that properties exist and are accessible
        assert hasattr(config, 'pss10_item_means')
        assert hasattr(config, 'pss10_item_sds')

        # Test that they are lists
        assert isinstance(config.pss10_item_means, list)
        assert isinstance(config.pss10_item_sds, list)

        # Test that they have correct length
        assert len(config.pss10_item_means) == 10
        assert len(config.pss10_item_sds) == 10

        # Test that all elements are floats
        for mean in config.pss10_item_means:
            assert isinstance(mean, float)
        for sd in config.pss10_item_sds:
            assert isinstance(sd, float)

    def test_pss10_dictionary_access(self):
        """Test dictionary-style access to PSS-10 configuration."""
        os.environ.clear()
        config = Config()

        # Test section access
        pss10_section = config.get('pss10')
        assert isinstance(pss10_section, dict)
        assert 'item_means' in pss10_section
        assert 'item_sds' in pss10_section

        # Test individual array access
        item_means = config.get('pss10', 'item_means')
        item_sds = config.get('pss10', 'item_sds')

        assert isinstance(item_means, list)
        assert isinstance(item_sds, list)
        assert len(item_means) == 10
        assert len(item_sds) == 10

    def test_pss10_individual_element_access(self):
        """Test individual element access within PSS-10 arrays."""
        os.environ.clear()
        config = Config()

        # Test first and last elements
        first_mean = config.pss10_item_means[0]
        last_mean = config.pss10_item_means[9]
        first_sd = config.pss10_item_sds[0]
        last_sd = config.pss10_item_sds[9]

        assert isinstance(first_mean, float)
        assert isinstance(last_mean, float)
        assert isinstance(first_sd, float)
        assert isinstance(last_sd, float)

        # Test middle elements
        middle_mean = config.pss10_item_means[5]
        middle_sd = config.pss10_item_sds[5]

        assert isinstance(middle_mean, float)
        assert isinstance(middle_sd, float)

    def test_pss10_invalid_section_access(self):
        """Test error handling for invalid section access."""
        os.environ.clear()
        config = Config()

        # Test invalid section
        with pytest.raises(ConfigurationError, match="Configuration section 'invalid_section' not found"):
            config.get('invalid_section')

    def test_pss10_invalid_key_access(self):
        """Test error handling for invalid key access."""
        os.environ.clear()
        config = Config()

        # Test invalid key within valid section
        with pytest.raises(ConfigurationError, match="Configuration key 'invalid_key' not found in section 'pss10'"):
            config.get('pss10', 'invalid_key')


@pytest.mark.config
class TestPSS10ConfigurationIntegration:
    """Test PSS-10 configuration integration with existing system."""

    def test_pss10_integration_with_global_config(self):
        """Test PSS-10 configuration works with global config system."""
        # Test with default config
        config1 = get_config()
        assert hasattr(config1, 'pss10_item_means')
        assert hasattr(config1, 'pss10_item_sds')
        assert len(config1.pss10_item_means) == 10

        # Test reload functionality
        config2 = reload_config()
        assert config2.pss10_item_means == config1.pss10_item_means
        assert config2.pss10_item_sds == config1.pss10_item_sds

    def test_pss10_validation_integration(self):
        """Test PSS-10 validation integrates properly with config validation."""
        config = Config()

        # Should not raise any exceptions during validation
        config.validate()

        # PSS-10 specific validation should pass
        assert len(config.pss10_item_means) == 10
        assert len(config.pss10_item_sds) == 10

        # All means should be in valid range
        for mean in config.pss10_item_means:
            assert 0 <= mean <= 4

        # All SDs should be positive
        for sd in config.pss10_item_sds:
            assert sd > 0

    def test_pss10_with_other_config_sections(self):
        """Test PSS-10 configuration works alongside other configuration sections."""
        config = Config()

        # Test that PSS-10 config doesn't interfere with other sections
        assert config.get('simulation', 'num_agents') > 0
        assert config.get('network', 'watts_k') >= 2
        assert config.get('agent', 'initial_resilience') >= 0

        # Test that PSS-10 config is accessible
        pss10_means = config.get('pss10', 'item_means')
        pss10_sds = config.get('pss10', 'item_sds')
        assert len(pss10_means) == 10
        assert len(pss10_sds) == 10


@pytest.mark.config
class TestPSS10ConfigurationEdgeCases:
    """Test PSS-10 configuration edge cases and error conditions."""

    def test_pss10_whitespace_handling(self):
        """Test PSS-10 configuration handles whitespace correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Create .env file with extra whitespace
                env_content = """PSS10_ITEM_MEAN=  2.1   1.8  2.3 1.9	2.2	1.7 2.0 1.6 2.4 1.5
PSS10_ITEM_SD=1.1  0.9	1.2  1.0 1.1 0.8 1.0 0.9 1.3 0.8
"""
                env_file = Path(temp_dir) / '.env'
                env_file.write_text(env_content)

                # Should handle whitespace correctly
                config = Config(str(env_file))

                expected_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
                expected_sds = [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]

                assert config.pss10_item_means == expected_means
                assert config.pss10_item_sds == expected_sds

            finally:
                os.chdir(original_cwd)

    def test_pss10_mixed_valid_invalid_env(self):
        """Test PSS-10 configuration with mix of valid and invalid environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                os.environ.clear()

                # Create .env file with mix of valid and invalid values
                env_content = """SIMULATION_NUM_AGENTS=30
PSS10_ITEM_MEAN=2.1 1.8 2.3 1.9 2.2 1.7 2.0 1.6 2.4 1.5
PSS10_ITEM_SD=1.1 0.9 1.2 1.0 1.1 0.8 1.0 0.9 1.3 0.8
AGENT_INITIAL_RESILIENCE_MEAN=0.8
"""
                env_file = Path(temp_dir) / '.env'
                env_file.write_text(env_content)

                # Should load successfully
                config = Config(str(env_file))

                # Test that valid values are loaded
                assert config.num_agents == 30
                assert config.agent_initial_resilience == 0.8
                assert config.pss10_item_means == [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
                assert config.pss10_item_sds == [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]

            finally:
                os.chdir(original_cwd)


def run_pss10_config_tests():
    """Run all PSS-10 configuration tests and report results."""
    print("Running PSS-10 Configuration Tests...")
    print("=" * 50)

    # Test 1: Default values (no .env file)
    print("\n1. Testing default values (no .env file)...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            config = Config()
            assert config.pss10_item_means == [2.0] * 10
            assert config.pss10_item_sds == [1.0] * 10
            print("✓ Default values test passed")
    except Exception as e:
        print(f"✗ Default values test failed: {e}")
    finally:
        if 'original_cwd' in locals():
            os.chdir(original_cwd)

    # Test 2: Valid .env file
    print("\n2. Testing valid .env file...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            env_content = "PSS10_ITEM_MEAN=2.1 1.8 2.3 1.9 2.2 1.7 2.0 1.6 2.4 1.5\nPSS10_ITEM_SD=1.1 0.9 1.2 1.0 1.1 0.8 1.0 0.9 1.3 0.8"
            (Path(temp_dir) / '.env').write_text(env_content)
            config = Config()
            expected_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
            assert config.pss10_item_means == expected_means
            print("✓ Valid .env file test passed")
    except Exception as e:
        print(f"✗ Valid .env file test failed: {e}")
    finally:
        if 'original_cwd' in locals():
            os.chdir(original_cwd)

    # Test 3: Invalid array length
    print("\n3. Testing invalid array length...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            env_content = "PSS10_ITEM_MEAN=2.1 1.8 2.3 1.9 2.2 1.7 2.0 1.6 2.4"
            (Path(temp_dir) / '.env').write_text(env_content)
            Config()
            print("✗ Invalid array length test failed: should have raised ConfigurationError")
    except ConfigurationError as e:
        if "must have exactly 10 elements" in str(e):
            print("✓ Invalid array length test passed")
        else:
            print(f"✗ Invalid array length test failed: wrong error message: {e}")
    except Exception as e:
        print(f"✗ Invalid array length test failed: unexpected error: {e}")
    finally:
        if 'original_cwd' in locals():
            os.chdir(original_cwd)

    # Test 4: Invalid array values
    print("\n4. Testing invalid array values...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            env_content = "PSS10_ITEM_MEAN=2.1 1.8 2.3 1.9 2.2 1.7 2.0 1.6 2.4 5.0"
            (Path(temp_dir) / '.env').write_text(env_content)
            Config()
            print("✗ Invalid array values test failed: should have raised ConfigurationError")
    except ConfigurationError as e:
        if "must be in [0, 4]" in str(e):
            print("✓ Invalid array values test passed")
        else:
            print(f"✗ Invalid array values test failed: wrong error message: {e}")
    except Exception as e:
        print(f"✗ Invalid array values test failed: unexpected error: {e}")
    finally:
        if 'original_cwd' in locals():
            os.chdir(original_cwd)

    # Test 5: Array access methods
    print("\n5. Testing array access methods...")
    try:
        config = Config()

        # Test direct property access
        means = config.pss10_item_means
        sds = config.pss10_item_sds
        assert len(means) == 10 and len(sds) == 10

        # Test dictionary access
        pss10_dict = config.get('pss10')
        assert 'item_means' in pss10_dict and 'item_sds' in pss10_dict

        # Test individual element access
        first_mean = config.pss10_item_means[0]
        last_sd = config.pss10_item_sds[9]
        assert isinstance(first_mean, float) and isinstance(last_sd, float)

        print("✓ Array access methods test passed")
    except Exception as e:
        print(f"✗ Array access methods test failed: {e}")

    # Test 6: Integration with existing system
    print("\n6. Testing integration with existing configuration system...")
    try:
        # Test global config access
        global_config = get_config()
        assert hasattr(global_config, 'pss10_item_means')

        # Test validation
        global_config.validate()

        # Test that PSS-10 doesn't interfere with other config
        num_agents = global_config.get('simulation', 'num_agents')
        assert isinstance(num_agents, int) and num_agents > 0

        print("✓ Integration test passed")
    except Exception as e:
        print(f"✗ Integration test failed: {e}")

    print("\n" + "=" * 50)
    print("PSS-10 Configuration Testing Complete!")


if __name__ == "__main__":
    run_pss10_config_tests()
