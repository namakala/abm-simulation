"""
Integration tests: ASSUMPTION_* env vars propagate through Config.
"""

import os
import pytest


class TestAssumptionConfigInConfig:
    """Assumption values accessible via the main Config class."""

    @pytest.mark.unit
    def test_assumption_section_exists(self, config):
        """Config has an 'assumptions' section."""
        section = config.get("assumptions")
        assert section is not None
        assert isinstance(section, dict)

    @pytest.mark.unit
    def test_assumption_coping_values(self, config):
        """Coping-related assumptions have correct defaults."""
        a = config.get("assumptions")
        assert a["resource_reward"] == 0.75
        assert a["resource_penalty"] == 0.10
        assert a["pf_allocation_fraction"] == 0.30
        assert a["success_affect_change"] == 0.1
        assert a["failure_affect_change"] == -0.2

    @pytest.mark.unit
    def test_assumption_stress_values(self, config):
        """Stress-related assumptions have correct defaults."""
        a = config.get("assumptions")
        assert a["controllability_challenge_weight"] == 0.10
        assert a["controllability_hindrance_weight"] == 0.05
        assert a["baseline_controllability"] == 0.5
        assert a["baseline_overload"] == 0.5
        assert a["event_intensity_challenge_weight"] == 0.7
        assert a["event_intensity_hindrance_weight"] == 1.3

    @pytest.mark.unit
    def test_assumption_resource_values(self, config):
        """Resource-related assumptions have correct defaults."""
        a = config.get("assumptions")
        assert a["resilience_efficiency_factor"] == 0.3
        assert a["min_cost_floor"] == 0.3
        assert a["failed_coping_cost_penalty"] == 1.3
        assert a["max_efficiency_gain"] == 0.5

    @pytest.mark.unit
    def test_assumption_social_values(self, config):
        """Social assumptions have correct defaults."""
        a = config.get("assumptions")
        assert a["support_exchange_threshold"] == 0.05
        assert a["social_support_probability"] == 0.3

    @pytest.mark.unit
    def test_assumption_buffering_values(self, config):
        """Buffering assumptions have correct defaults."""
        a = config.get("assumptions")
        assert a["resilience_low_threshold"] == 0.3
        assert a["resilience_high_threshold"] == 0.7
        assert a["initial_protective_factor_values"] == 0.5

    @pytest.mark.unit
    def test_env_override_coping(self, clean_env, reload_config_fixture):
        """ASSUMPTION_RESOURCE_REWARD env var overrides default."""
        os.environ["ASSUMPTION_RESOURCE_REWARD"] = "0.5"
        try:
            cfg = reload_config_fixture()
            assert cfg.get("assumptions", "resource_reward") == 0.5
        finally:
            os.environ.pop("ASSUMPTION_RESOURCE_REWARD", None)

    @pytest.mark.unit
    def test_env_override_stress(self, clean_env, reload_config_fixture):
        """ASSUMPTION_BASELINE_CONTROLLABILITY env var overrides default."""
        os.environ["ASSUMPTION_BASELINE_CONTROLLABILITY"] = "0.3"
        try:
            cfg = reload_config_fixture()
            assert cfg.get("assumptions", "baseline_controllability") == 0.3
        finally:
            os.environ.pop("ASSUMPTION_BASELINE_CONTROLLABILITY", None)

    @pytest.mark.unit
    def test_env_override_resource(self, clean_env, reload_config_fixture):
        """ASSUMPTION_RESILIENCE_EFFICIENCY_FACTOR env var overrides."""
        os.environ["ASSUMPTION_RESILIENCE_EFFICIENCY_FACTOR"] = "0.5"
        try:
            cfg = reload_config_fixture()
            assert cfg.get("assumptions", "resilience_efficiency_factor") == 0.5
        finally:
            os.environ.pop("ASSUMPTION_RESILIENCE_EFFICIENCY_FACTOR", None)

    @pytest.mark.unit
    def test_env_override_social(self, clean_env, reload_config_fixture):
        """ASSUMPTION_SOCIAL_SUPPORT_PROBABILITY env var overrides."""
        os.environ["ASSUMPTION_SOCIAL_SUPPORT_PROBABILITY"] = "0.5"
        try:
            cfg = reload_config_fixture()
            assert cfg.get("assumptions", "social_support_probability") == 0.5
        finally:
            os.environ.pop("ASSUMPTION_SOCIAL_SUPPORT_PROBABILITY", None)
