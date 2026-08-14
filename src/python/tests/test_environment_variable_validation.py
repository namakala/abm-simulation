"""
Tests for environment variable validation and dataclass integration.

These tests verify that dataclasses properly use environment variables
and handle various edge cases correctly.
"""

import os
import pytest
from src.python.affect_utils import InteractionConfig, ProtectiveFactors, ResourceParams
from src.python.stress_utils import AppraisalWeights, ThresholdParams


class TestEnvIsolation:
    """complete_env_isolation must not leak a stale local .env into tests."""

    @pytest.mark.config
    def test_env_isolation_blocks_stale_dotenv(self, tmp_path, monkeypatch):
        """Env isolation must not read a stale .env when DOTENV_FILE is unset.

        Regression: complete_env_isolation clears os.environ (including
        DOTENV_FILE), then reload_config() falls back to reading the real
        `.env`. A stale local `.env` with ASSUMPTION_BUFFERING_A_COEFFICIENT=-0.015
        leaked into get_assumptions(), failing the env-consistency test on
        developer machines (CI has no .env, so it only failed locally).

        This mirrors the fixture's SETUP steps: clear env, set explicit
        defaults, then reload config + assumptions.
        """
        import os

        from src.python.assumption_config import reload_assumptions, get_assumptions
        from src.python.config import reload_config

        # Hostile local .env with a stale value that diverges from code default
        (tmp_path / ".env").write_text("ASSUMPTION_BUFFERING_A_COEFFICIENT=-0.015\n")
        monkeypatch.chdir(tmp_path)

        # Mirror complete_env_isolation SETUP
        current_env = dict(os.environ)
        os.environ.clear()
        os.environ["DOTENV_FILE"] = ".env.empty"  # pin: never read the real .env
        os.environ["APPRAISAL_GAMMA"] = "3.0"
        os.environ["ASSUMPTION_STRESS_DECAY_RATE"] = "0.08"
        os.environ["ASSUMPTION_PF_ALLOCATION_FRACTION"] = "0.05"
        try:
            reload_config()
            reload_assumptions()
            # Stale .env must NOT leak: value must be the code default -0.008
            assert get_assumptions().resource.buffering_a_coefficient == -0.008
        finally:
            os.environ.clear()
            os.environ.update(current_env)


class TestDataclassEnvironmentVariableUsage:
    """Test dataclass integration with environment variables."""

    @pytest.mark.config
    def test_current_environment_variable_usage(self, config):
        """Test that dataclasses use current environment variable values."""
        # Create dataclass instances
        interaction_config = InteractionConfig()
        protective_factors = ProtectiveFactors()
        resource_params = ResourceParams()
        appraisal_weights = AppraisalWeights()
        threshold_params = ThresholdParams()

        # Verify they match current config values
        assert interaction_config.influence_rate == config.get("interaction", "influence_rate")
        assert protective_factors.social_support == config.get("protective", "social_support")
        assert resource_params.base_regeneration == config.get("resource", "base_regeneration")
        assert appraisal_weights.omega_c == config.get("appraisal", "omega_c")
        assert threshold_params.base_threshold == config.get("threshold", "base_threshold")

    @pytest.mark.config
    def test_environment_variable_override(self, clean_env, reload_config_fixture):
        """Test that environment variables properly override defaults."""
        os.environ.clear()

        # Set new environment variables
        test_env_vars = {
            "INTERACTION_INFLUENCE_RATE": "0.123",
            "PROTECTIVE_SOCIAL_SUPPORT": "0.456",
            "RESOURCE_BASE_REGENERATION": "0.789",
            "APPRAISAL_OMEGA_C": "1.5",
            "THRESHOLD_BASE_THRESHOLD": "0.3",
        }

        for key, value in test_env_vars.items():
            os.environ[key] = value

        try:
            # Reload configuration
            reload_config_fixture()

            # Create new dataclass instances
            new_interaction_config = InteractionConfig()
            new_protective_factors = ProtectiveFactors()
            new_resource_params = ResourceParams()
            new_appraisal_weights = AppraisalWeights()
            new_threshold_params = ThresholdParams()

            # Verify they use the new values
            assert new_interaction_config.influence_rate == 0.123
            assert new_protective_factors.social_support == 0.456
            assert new_resource_params.base_regeneration == 0.789
            assert new_appraisal_weights.omega_c == 1.5
            assert new_threshold_params.base_threshold == 0.3

        finally:
            # Clean up environment variables
            for key in test_env_vars.keys():
                if key in os.environ:
                    del os.environ[key]

    @pytest.mark.config
    def test_default_value_fallback(self, clean_env, reload_config_fixture):
        """Test that missing environment variables use defaults."""
        # Temporarily move .env file to test defaults
        env_file = ".env"
        backup_env = ".env.backup"

        if os.path.exists(env_file):
            os.rename(env_file, backup_env)

        try:
            # Clear relevant environment variables
            vars_to_clear = [
                "INTERACTION_INFLUENCE_RATE",
                "PROTECTIVE_SOCIAL_SUPPORT",
                "RESOURCE_BASE_REGENERATION",
                "APPRAISAL_OMEGA_C",
                "THRESHOLD_BASE_THRESHOLD",
            ]

            for var in vars_to_clear:
                if var in os.environ:
                    del os.environ[var]

            # Reload config (should use hardcoded defaults)
            reload_config_fixture()

            # Create dataclass instances (should use defaults)
            default_interaction_config = InteractionConfig()
            default_protective_factors = ProtectiveFactors()

            # Verify they use default values (from the hardcoded defaults in config.py)
            expected_influence_rate = 0.05  # Default value from config.py
            expected_social_support = 0.5  # Default value from config.py

            assert default_interaction_config.influence_rate == expected_influence_rate
            assert default_protective_factors.social_support == expected_social_support

        finally:
            # Restore .env file
            if os.path.exists(backup_env):
                os.rename(backup_env, env_file)

    @pytest.mark.config
    def test_type_conversion(self, clean_env, reload_config_fixture):
        """Test that type conversion works correctly."""
        # Test boolean conversion
        os.environ["OUTPUT_SAVE_TIME_SERIES"] = "false"
        os.environ["OUTPUT_SAVE_NETWORK_SNAPSHOTS"] = "true"

        try:
            bool_config = reload_config_fixture()

            # Note: These config values aren't directly used in dataclasses currently,
            # but we can test that the config system handles them correctly
            assert not bool_config.output_save_time_series
            assert bool_config.output_save_network_snapshots

        finally:
            # Clean up
            for var in ["OUTPUT_SAVE_TIME_SERIES", "OUTPUT_SAVE_NETWORK_SNAPSHOTS"]:
                if var in os.environ:
                    del os.environ[var]


class TestConfigValidation:
    """Test configuration validation functionality."""

    @pytest.mark.config
    def test_invalid_values_handling(self, clean_env, reload_config_fixture):
        """Test that invalid values are properly handled."""
        # Test invalid values
        os.environ["NETWORK_WATTS_K"] = "1"  # Should be >= 2

        try:
            with pytest.raises(Exception) as exc_info:
                reload_config_fixture()
            assert "must be >= 2" in str(exc_info.value)
        finally:
            # Clean up
            if "NETWORK_WATTS_K" in os.environ:
                del os.environ["NETWORK_WATTS_K"]


class TestNoHardcodedPaths:
    """Test that no hardcoded paths remain."""

    @pytest.mark.config
    def test_output_directories_use_env_vars(self, config):
        """Test that output directories use environment variables."""
        # These should come from environment variables, not be hardcoded
        results_dir = config.get("output", "results_dir")
        raw_dir = config.get("output", "raw_dir")
        logs_dir = config.get("output", "logs_dir")

        # Verify these are not hardcoded paths
        assert results_dir == "data/processed"
        assert raw_dir == "data/raw"
        assert logs_dir == "logs"
