"""
Tests for the assumption parameterization config hierarchy.

Verifies that AssumptionConfig with sub-groups exists and has
correct default values matching the specification.
"""

import pytest
from dataclasses import asdict


class TestAssumptionConfigExistence:
    """Test that assumption config classes and defaults exist."""

    @pytest.mark.unit
    def test_assumption_config_importable(self):
        """AssumptionConfig and sub-groups can be imported."""
        from src.python.assumption_config import (
            AssumptionConfig,
            AssumptionCopingConfig,
            AssumptionStressConfig,
            AssumptionResourceConfig,
            AssumptionSocialConfig,
            AssumptionBufferingConfig,
        )

        assert AssumptionConfig is not None
        assert AssumptionCopingConfig is not None
        assert AssumptionStressConfig is not None
        assert AssumptionResourceConfig is not None
        assert AssumptionSocialConfig is not None
        assert AssumptionBufferingConfig is not None

    @pytest.mark.unit
    def test_assumption_config_instantiation(self):
        """AssumptionConfig can be instantiated with default values."""
        from src.python.assumption_config import AssumptionConfig

        config = AssumptionConfig()
        assert config is not None

    @pytest.mark.unit
    def test_assumption_coping_config_defaults(self):
        """AssumptionCopingConfig has correct default values."""
        from src.python.assumption_config import AssumptionCopingConfig

        c = AssumptionCopingConfig()
        assert c.resource_reward == 0.75
        assert c.resource_penalty == 0.10
        assert c.pf_allocation_fraction == 0.30
        assert c.affect_improvement_scale == 0.2
        assert c.affect_deterioration_scale == 0.4
        assert c.resilience_improvement_scale == 0.1
        assert c.resilience_deterioration_scale == 0.2
        assert c.challenge_success_resilience == 0.3
        assert c.challenge_failure_resilience == -0.1
        assert c.hindrance_success_resilience == 0.1
        assert c.hindrance_failure_resilience == -0.4
        assert c.success_stress_reduction == 0.2
        assert c.failure_stress_increase == 0.3
        assert c.success_affect_change == 0.1
        assert c.failure_affect_change == -0.2

    @pytest.mark.unit
    def test_assumption_stress_config_defaults(self):
        """AssumptionStressConfig has correct default values."""
        from src.python.assumption_config import AssumptionStressConfig

        c = AssumptionStressConfig()
        assert c.controllability_challenge_weight == 0.10
        assert c.controllability_hindrance_weight == 0.05
        assert c.overload_challenge_weight == 0.05
        assert c.overload_hindrance_weight == 0.10
        assert c.baseline_controllability == 0.5
        assert c.baseline_overload == 0.5
        assert c.controllability_homeostasis_rate == 0.05
        assert c.overload_homeostasis_rate == 0.05
        assert c.event_intensity_challenge_weight == 0.7
        assert c.event_intensity_hindrance_weight == 1.3
        assert c.failed_coping_intensity_multiplier == 1.5
        assert c.stress_intensity_decay_rate == 0.8
        assert c.new_intensity_weight == 0.2
        assert c.momentum_increase_rate == 0.1
        assert c.momentum_decrease_rate == 0.05
        assert c.momentum_zero_threshold == 0.01
        assert c.momentum_decay_factor == 0.9
        assert c.pss10_estimation_base == 10
        assert c.pss10_controllability_max_effect == 8
        assert c.pss10_overload_max_effect == 12
        assert c.pss10_estimation_variance == 3

    @pytest.mark.unit
    def test_assumption_resource_config_defaults(self):
        """AssumptionResourceConfig has correct default values."""
        from src.python.assumption_config import AssumptionResourceConfig

        c = AssumptionResourceConfig()
        assert c.resilience_efficiency_factor == 0.3
        assert c.min_resource_threshold == 0.05
        assert c.coping_difficulty_scale == 0.5
        assert c.min_cost_floor == 0.3
        assert c.failed_coping_cost_penalty == 1.3
        assert c.max_efficiency_gain == 0.5
        assert c.social_resilience_boost_factor == 0.1
        assert c.support_exchange_benefit_weight == 0.2
        assert c.challenge_resilience_bonus_factor == 0.2
        assert c.hindrance_resilience_bonus_factor == 0.1
        assert c.overload_allocation_penalty_rate == 0.1
        assert c.stress_improvement_effectiveness == 0.1
        assert c.social_resource_boost_factor == 0.1
        assert c.preservable_allocation_fraction == 0.1
        assert c.social_support_allocation_boost == 0.3
        assert c.affect_regeneration_multiplier == 0.5
        assert c.resilience_regeneration_multiplier == 0.3

    @pytest.mark.unit
    def test_assumption_social_config_defaults(self):
        """AssumptionSocialConfig has correct default values."""
        from src.python.assumption_config import AssumptionSocialConfig

        c = AssumptionSocialConfig()
        assert c.support_exchange_threshold == 0.05
        assert c.social_support_probability == 0.3
        assert c.social_support_exchange_boost == 0.1

    @pytest.mark.unit
    def test_assumption_buffering_config_defaults(self):
        """AssumptionBufferingConfig has correct default values."""
        from src.python.assumption_config import AssumptionBufferingConfig

        c = AssumptionBufferingConfig()
        assert c.resilience_low_threshold == 0.3
        assert c.resilience_high_threshold == 0.7
        assert c.volatility_beta_alpha == 1
        assert c.volatility_beta_beta == 1
        assert c.initial_protective_factor_values == 0.5

    @pytest.mark.unit
    def test_assumption_config_sub_groups(self):
        """AssumptionConfig contains all five sub-group dataclasses."""
        from src.python.assumption_config import AssumptionConfig

        c = AssumptionConfig()
        assert hasattr(c, "coping")
        assert hasattr(c, "stress")
        assert hasattr(c, "resource")
        assert hasattr(c, "social")
        assert hasattr(c, "buffering")

    @pytest.mark.unit
    def test_assumption_config_nested_defaults(self):
        """AssumptionConfig nested groups have correct defaults."""
        from src.python.assumption_config import AssumptionConfig

        c = AssumptionConfig()
        assert c.coping.resource_reward == 0.75
        assert c.stress.controllability_challenge_weight == 0.10
        assert c.resource.resilience_efficiency_factor == 0.3
        assert c.social.support_exchange_threshold == 0.05
        assert c.buffering.resilience_low_threshold == 0.3

    @pytest.mark.unit
    def test_assumption_config_total_fields(self):
        """Total config fields count is ~62 (spec-defined constants)."""
        from src.python.assumption_config import (
            AssumptionCopingConfig,
            AssumptionStressConfig,
            AssumptionResourceConfig,
            AssumptionSocialConfig,
            AssumptionBufferingConfig,
        )

        count = (
            len(asdict(AssumptionCopingConfig()))
            + len(asdict(AssumptionStressConfig()))
            + len(asdict(AssumptionResourceConfig()))
            + len(asdict(AssumptionSocialConfig()))
            + len(asdict(AssumptionBufferingConfig()))
        )
        # The spec lists ~84 total, but some are from phase-level constants
        # that will be added later. The sub-group total is ~62.
        assert count >= 60, f"Expected >=60 fields, got {count}"
