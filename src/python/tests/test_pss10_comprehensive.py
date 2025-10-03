#!/usr/bin/env python3
"""
Comprehensive test script for PSS-10 configuration with bracket notation format.

This script demonstrates all the requested functionality:
1. Bracket notation parsing with exact format requested
2. Backward compatibility with space-separated format
3. Mixed format usage
4. Whitespace handling in bracket notation
5. Default values verification
6. Integration with existing configuration system
"""

import os
import tempfile
from pathlib import Path

from src.python.config import Config, ConfigurationError


def test_bracket_notation_parsing():
    """Test bracket notation parsing with the exact format requested by user."""
    print("1. Testing bracket notation parsing with exact format requested...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create .env file with exact bracket notation format requested
        env_content = """PSS10_ITEM_MEAN=[2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
PSS10_ITEM_SD=[1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]
"""
        env_file = Path(temp_dir) / '.env'
        env_file.write_text(env_content)

        # Test configuration loading
        config = Config(str(env_file))

        # Verify exact values requested by user
        expected_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
        expected_sds = [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]

        assert config.pss10_item_means == expected_means, f"Expected {expected_means}, got {config.pss10_item_means}"
        assert config.pss10_item_sds == expected_sds, f"Expected {expected_sds}, got {config.pss10_item_sds}"

        print("✓ Bracket notation parsing works correctly")


def test_backward_compatibility():
    """Test backward compatibility with space-separated format."""
    print("2. Testing backward compatibility with space-separated format...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create .env file with space-separated format (legacy format)
        env_content = """PSS10_ITEM_MEAN=2.1 1.8 2.3 1.9 2.2 1.7 2.0 1.6 2.4 1.5
PSS10_ITEM_SD=1.1 0.9 1.2 1.0 1.1 0.8 1.0 0.9 1.3 0.8
"""
        env_file = Path(temp_dir) / '.env'
        env_file.write_text(env_content)

        # Test configuration loading
        config = Config(str(env_file))

        # Verify values are loaded correctly
        expected_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
        expected_sds = [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]

        assert config.pss10_item_means == expected_means
        assert config.pss10_item_sds == expected_sds

        print("✓ Backward compatibility with space-separated format works correctly")


def test_mixed_format_usage():
    """Test mixed format usage (bracket for one array, space-separated for another)."""
    print("3. Testing mixed format usage...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create .env file mixing both formats
        env_content = """PSS10_ITEM_MEAN=[2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
PSS10_ITEM_SD=1.1 0.9 1.2 1.0 1.1 0.8 1.0 0.9 1.3 0.8
"""
        env_file = Path(temp_dir) / '.env'
        env_file.write_text(env_content)

        # Test configuration loading
        config = Config(str(env_file))

        # Verify both formats work together
        expected_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
        expected_sds = [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]

        assert config.pss10_item_means == expected_means
        assert config.pss10_item_sds == expected_sds

        print("✓ Mixed format usage works correctly")


def test_whitespace_handling():
    """Test whitespace handling in bracket notation."""
    print("4. Testing whitespace handling in bracket notation...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create .env file with extra whitespace in bracket notation
        env_content = """PSS10_ITEM_MEAN= [ 2.1 , 1.8 , 2.3 , 1.9 , 2.2 , 1.7 , 2.0 , 1.6 , 2.4 , 1.5 ]
PSS10_ITEM_SD= [1.1,  0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8 ]
"""
        env_file = Path(temp_dir) / '.env'
        env_file.write_text(env_content)

        # Test configuration loading
        config = Config(str(env_file))

        # Verify whitespace is handled correctly
        expected_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
        expected_sds = [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]

        assert config.pss10_item_means == expected_means
        assert config.pss10_item_sds == expected_sds

        print("✓ Whitespace handling in bracket notation works correctly")


def test_default_values():
    """Verify default values are correctly set to user's specified arrays."""
    print("5. Testing default values...")

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)

            # Test with no .env file
            config = Config()

            # Verify default values match user's specified arrays
            expected_means = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
            expected_sds = [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]

            assert config.pss10_item_means == expected_means
            assert config.pss10_item_sds == expected_sds

            # Test with empty .env file
            env_file = Path(temp_dir) / '.env'
            env_file.touch()
            config2 = Config()

            assert config2.pss10_item_means == expected_means
            assert config2.pss10_item_sds == expected_sds

            print("✓ Default values are correctly set to user's specified arrays")

        finally:
            os.chdir(original_cwd)


def test_integration():
    """Test integration with existing configuration system."""
    print("6. Testing integration with existing configuration system...")

    # Test that PSS-10 configuration works alongside other config sections
    config = Config()

    # Test PSS-10 access
    pss10_means = config.get('pss10', 'item_means')
    pss10_sds = config.get('pss10', 'item_sds')
    assert len(pss10_means) == 10
    assert len(pss10_sds) == 10

    # Test that other config sections still work
    assert config.get('simulation', 'num_agents') > 0
    assert config.get('network', 'watts_k') >= 2
    assert config.get('agent', 'initial_resilience') >= 0

    # Test validation
    config.validate()  # Should not raise any exceptions

    print("✓ Integration with existing configuration system works correctly")


def main():
    """Run all comprehensive PSS-10 configuration tests."""
    print("PSS-10 Configuration Comprehensive Test Suite")
    print("=" * 50)
    print()

    try:
        test_bracket_notation_parsing()
        test_backward_compatibility()
        test_mixed_format_usage()
        test_whitespace_handling()
        test_default_values()
        test_integration()

        print()
        print("=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("PSS-10 configuration with bracket notation format is working correctly.")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
