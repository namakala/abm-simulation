"""
Comprehensive unit tests for the compute_homeostatic_adjustment function.

This file tests the homeostatic adjustment mechanism that pulls affect and resilience
values back toward their baseline levels over time, simulating natural tendencies
to return to equilibrium states.
"""

import pytest
import numpy as np
from src.python.affect_utils import compute_homeostatic_adjustment
from src.python.config import get_config


class TestHomeostaticAdjustmentBasic:
    """Test basic homeostatic adjustment functionality."""

    def test_values_at_baseline_unchanged(self):
        """Test that values at baseline remain unchanged."""
        # Test affect values
        result = compute_homeostatic_adjustment(
            initial_value=0.0,
            final_value=0.0,
            homeostatic_rate=0.1,
            value_type='affect'
        )
        assert result == 0.0

        # Test resilience values
        result = compute_homeostatic_adjustment(
            initial_value=0.5,
            final_value=0.5,
            homeostatic_rate=0.1,
            value_type='resilience'
        )
        assert result == 0.5

    def test_values_above_baseline_pulled_down_affect(self):
        """Test that affect values above baseline are pulled downward."""
        initial_value = 0.0
        final_value = 0.8
        homeostatic_rate = 0.2

        result = compute_homeostatic_adjustment(
            initial_value, final_value, homeostatic_rate, 'affect'
        )

        # Should be pulled down from 0.8 toward 0.0
        expected_distance = homeostatic_rate * abs(final_value - initial_value)  # 0.2 * 0.8 = 0.16
        expected_result = final_value - expected_distance  # 0.8 - 0.16 = 0.64

        assert result == pytest.approx(expected_result, abs=1e-10)
        assert result < final_value  # Should be lower than final value
        assert result > initial_value  # Should still be above initial value

    def test_values_below_baseline_pulled_up_affect(self):
        """Test that affect values below baseline are pulled upward."""
        initial_value = 0.0
        final_value = -0.6
        homeostatic_rate = 0.3

        result = compute_homeostatic_adjustment(
            initial_value, final_value, homeostatic_rate, 'affect'
        )

        # Should be pulled up from -0.6 toward 0.0
        expected_distance = homeostatic_rate * abs(final_value - initial_value)  # 0.3 * 0.6 = 0.18
        expected_result = final_value + expected_distance  # -0.6 + 0.18 = -0.42

        assert result == pytest.approx(expected_result, abs=1e-10)
        assert result > final_value  # Should be higher than final value
        assert result < initial_value  # Should still be below initial value

    def test_values_above_baseline_pulled_down_resilience(self):
        """Test that resilience values above baseline are pulled downward."""
        initial_value = 0.5
        final_value = 0.9
        homeostatic_rate = 0.15

        result = compute_homeostatic_adjustment(
            initial_value, final_value, homeostatic_rate, 'resilience'
        )

        # Should be pulled down from 0.9 toward 0.5
        expected_distance = homeostatic_rate * abs(final_value - initial_value)  # 0.15 * 0.4 = 0.06
        expected_result = final_value - expected_distance  # 0.9 - 0.06 = 0.84

        assert result == pytest.approx(expected_result, abs=1e-10)
        assert result < final_value  # Should be lower than final value
        assert result > initial_value  # Should still be above initial value

    def test_values_below_baseline_pulled_up_resilience(self):
        """Test that resilience values below baseline are pulled upward."""
        initial_value = 0.5
        final_value = 0.2
        homeostatic_rate = 0.25

        result = compute_homeostatic_adjustment(
            initial_value, final_value, homeostatic_rate, 'resilience'
        )

        # Should be pulled up from 0.2 toward 0.5
        expected_distance = homeostatic_rate * abs(final_value - initial_value)  # 0.25 * 0.3 = 0.075
        expected_result = final_value + expected_distance  # 0.2 + 0.075 = 0.275

        assert result == pytest.approx(expected_result, abs=1e-10)
        assert result > final_value  # Should be higher than final value
        assert result < initial_value  # Should still be below initial value


class TestHomeostaticAdjustmentEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_boundary_values_affect_range(self):
        """Test behavior at the boundaries of affect range [-1, 1]."""
        homeostatic_rate = 0.1

        # Test at upper boundary - should be pulled down toward baseline
        result = compute_homeostatic_adjustment(
            0.0, 1.0, homeostatic_rate, 'affect'
        )
        expected = 1.0 - (0.1 * abs(1.0 - 0.0))  # 1.0 - 0.1 = 0.9
        assert result == pytest.approx(expected, abs=1e-10)

        # Test at lower boundary - should be pulled up toward baseline
        result = compute_homeostatic_adjustment(
            0.0, -1.0, homeostatic_rate, 'affect'
        )
        expected = -1.0 + (0.1 * abs(-1.0 - 0.0))  # -1.0 + 0.1 = -0.9
        assert result == pytest.approx(expected, abs=1e-10)

        # Test near boundaries with adjustment needed
        result = compute_homeostatic_adjustment(
            0.0, 0.95, homeostatic_rate, 'affect'
        )
        expected = 0.95 - (0.1 * 0.95)  # 0.95 - 0.095 = 0.855
        assert result == pytest.approx(expected, abs=1e-10)
        assert -1.0 <= result <= 1.0

    def test_boundary_values_resilience_range(self):
        """Test behavior at the boundaries of resilience range [0, 1]."""
        homeostatic_rate = 0.1

        # Test at upper boundary - should be pulled down toward baseline
        result = compute_homeostatic_adjustment(
            0.5, 1.0, homeostatic_rate, 'resilience'
        )
        expected = 1.0 - (0.1 * abs(1.0 - 0.5))  # 1.0 - 0.05 = 0.95
        assert result == pytest.approx(expected, abs=1e-10)

        # Test at lower boundary - should be pulled up toward baseline
        result = compute_homeostatic_adjustment(
            0.5, 0.0, homeostatic_rate, 'resilience'
        )
        expected = 0.0 + (0.1 * abs(0.0 - 0.5))  # 0.0 + 0.05 = 0.05
        assert result == pytest.approx(expected, abs=1e-10)

        # Test near boundaries with adjustment needed
        result = compute_homeostatic_adjustment(
            0.5, 0.05, homeostatic_rate, 'resilience'
        )
        expected = 0.05 + (0.1 * 0.45)  # 0.05 + 0.045 = 0.095
        assert result == pytest.approx(expected, abs=1e-10)
        assert 0.0 <= result <= 1.0

    def test_extreme_homeostatic_rates(self):
        """Test behavior with extreme homeostatic rate values."""
        initial_value = 0.0
        final_value = 0.5

        # Zero rate should result in no adjustment
        result = compute_homeostatic_adjustment(
            initial_value, final_value, 0.0, 'affect'
        )
        assert result == final_value

        # Rate of 1.0 should pull all the way to baseline
        result = compute_homeostatic_adjustment(
            initial_value, final_value, 1.0, 'affect'
        )
        assert result == initial_value

        # Rate > 1.0 should still pull toward baseline but clamp properly
        result = compute_homeostatic_adjustment(
            initial_value, final_value, 1.5, 'affect'
        )
        # With rate 1.5, distance = 1.5 * 0.5 = 0.75
        # Since final_value (0.5) > initial_value (0.0), we subtract: 0.5 - 0.75 = -0.25
        # Then clamp to [-1, 1], so result should be -0.25
        expected = 0.5 - (1.5 * 0.5)  # -0.25
        assert result == pytest.approx(expected, abs=1e-10)


class TestHomeostaticAdjustmentValueTypes:
    """Test different value types and their valid ranges."""

    def test_affect_value_type_validation(self):
        """Test that affect values are properly constrained to [-1, 1]."""
        test_cases = [
            (0.0, 1.5, 0.1),    # Above range
            (0.0, -1.2, 0.1),   # Below range
            (0.0, 0.8, 0.1),    # Within range
            (0.0, -0.6, 0.1),   # Within range
        ]

        for initial, final, rate in test_cases:
            result = compute_homeostatic_adjustment(
                initial, final, rate, 'affect'
            )
            assert -1.0 <= result <= 1.0

    def test_resilience_value_type_validation(self):
        """Test that resilience values are properly constrained to [0, 1]."""
        test_cases = [
            (0.5, 1.2, 0.1),    # Above range
            (0.5, -0.3, 0.1),   # Below range
            (0.5, 0.8, 0.1),    # Within range
            (0.5, 0.2, 0.1),    # Within range
        ]

        for initial, final, rate in test_cases:
            result = compute_homeostatic_adjustment(
                initial, final, rate, 'resilience'
            )
            assert 0.0 <= result <= 1.0

    def test_invalid_value_type_raises_error(self):
        """Test that invalid value_type parameter raises ValueError."""
        with pytest.raises(ValueError, match="value_type must be 'affect' or 'resilience'"):
            compute_homeostatic_adjustment(
                0.0, 0.5, 0.1, 'invalid_type'
            )


class TestHomeostaticAdjustmentConfiguration:
    """Test integration with configuration system."""

    def test_default_homeostatic_rate_from_config(self):
        """Test using default homeostatic rate from configuration."""
        # Get the default rate from config
        config = get_config()
        default_rate = config.get('affect_dynamics', 'homeostatic_rate')

        initial_value = 0.0
        final_value = 0.5

        # Call without specifying homeostatic_rate (should use config default)
        result = compute_homeostatic_adjustment(
            initial_value, final_value, value_type='affect'
        )

        # Should use the default rate from config
        expected_distance = default_rate * abs(final_value - initial_value)
        expected_result = final_value - expected_distance

        assert result == pytest.approx(expected_result, abs=1e-10)

    def test_explicit_rate_overrides_config(self):
        """Test that explicit rate parameter overrides config default."""
        config = get_config()
        default_rate = config.get('affect_dynamics', 'homeostatic_rate')
        explicit_rate = 0.3  # Different from default

        initial_value = 0.0
        final_value = 0.5

        # Call with explicit rate
        result = compute_homeostatic_adjustment(
            initial_value, final_value, explicit_rate, 'affect'
        )

        # Should use explicit rate, not config default
        expected_distance = explicit_rate * abs(final_value - initial_value)
        expected_result = final_value - expected_distance

        assert result == pytest.approx(expected_result, abs=1e-10)

        # Verify it's different from what config default would produce
        config_result = compute_homeostatic_adjustment(
            initial_value, final_value, value_type='affect'
        )
        assert result != config_result


class TestHomeostaticAdjustmentMultiDay:
    """Test multi-day stabilization behavior."""

    def test_convergence_over_multiple_days_affect(self):
        """Test that affect converges toward baseline over multiple days."""
        initial_value = 0.0
        homeostatic_rate = 0.2

        # Start with extreme value
        current_value = 0.8

        values_over_time = [current_value]

        # Simulate 10 days of homeostatic adjustment
        for day in range(10):
            adjusted_value = compute_homeostatic_adjustment(
                initial_value, current_value, homeostatic_rate, 'affect'
            )
            values_over_time.append(adjusted_value)
            current_value = adjusted_value

        # Should be converging toward baseline
        assert values_over_time[-1] < values_over_time[0]  # Moving toward baseline
        assert all(-1.0 <= val <= 1.0 for val in values_over_time)  # Stay in range

        # Should be getting closer to baseline over time
        final_distance = abs(values_over_time[-1] - initial_value)
        initial_distance = abs(values_over_time[0] - initial_value)
        assert final_distance < initial_distance

    def test_convergence_over_multiple_days_resilience(self):
        """Test that resilience converges toward baseline over multiple days."""
        initial_value = 0.5
        homeostatic_rate = 0.15

        # Start with extreme value
        current_value = 0.1

        values_over_time = [current_value]

        # Simulate 10 days of homeostatic adjustment
        for day in range(10):
            adjusted_value = compute_homeostatic_adjustment(
                initial_value, current_value, homeostatic_rate, 'resilience'
            )
            values_over_time.append(adjusted_value)
            current_value = adjusted_value

        # Should be converging toward baseline
        assert values_over_time[-1] > values_over_time[0]  # Moving toward baseline
        assert all(0.0 <= val <= 1.0 for val in values_over_time)  # Stay in range

        # Should be getting closer to baseline over time
        final_distance = abs(values_over_time[-1] - initial_value)
        initial_distance = abs(values_over_time[0] - initial_value)
        assert final_distance < initial_distance

    def test_oscillation_prevention(self):
        """Test that homeostasis doesn't cause oscillation around baseline."""
        initial_value = 0.0
        homeostatic_rate = 0.3

        # Start above baseline
        current_value = 0.6

        previous_adjusted = current_value

        # Simulate multiple adjustments
        for day in range(20):
            adjusted_value = compute_homeostatic_adjustment(
                initial_value, current_value, homeostatic_rate, 'affect'
            )

            # Each adjustment should bring us closer to baseline (no oscillation)
            current_distance = abs(current_value - initial_value)
            adjusted_distance = abs(adjusted_value - initial_value)

            assert adjusted_distance <= current_distance

            # Update for next iteration
            current_value = adjusted_value

        # Final value should be very close to baseline
        assert abs(current_value - initial_value) < 0.01


class TestHomeostaticAdjustmentMathematicalProperties:
    """Test mathematical properties and consistency."""

    def test_commutativity_of_equal_values(self):
        """Test that adjustment is consistent when initial and final are swapped appropriately."""
        # If we swap initial and final, and negate the result, we should get consistent behavior
        initial1, final1 = 0.0, 0.5
        initial2, final2 = 0.5, 0.0

        result1 = compute_homeostatic_adjustment(initial1, final1, 0.1, 'affect')
        result2 = compute_homeostatic_adjustment(initial2, final2, 0.1, 'affect')

        # The results should be symmetric around the midpoint
        midpoint = (initial1 + final1) / 2
        assert abs(result1 - midpoint) == pytest.approx(abs(result2 - midpoint), abs=1e-10)

    def test_linearity_property(self):
        """Test that adjustment scales linearly with homeostatic rate."""
        initial_value = 0.0
        final_value = 0.5

        rate1 = 0.1
        rate2 = 0.2

        result1 = compute_homeostatic_adjustment(initial_value, final_value, rate1, 'affect')
        result2 = compute_homeostatic_adjustment(initial_value, final_value, rate2, 'affect')

        # Result should scale linearly with rate
        expected_result2 = result1 + (result1 - final_value)  # Double the adjustment

        assert result2 == pytest.approx(expected_result2, abs=1e-10)

    def test_deterministic_behavior(self):
        """Test that function produces consistent results for same inputs."""
        test_cases = [
            (0.0, 0.5, 0.1, 'affect'),
            (0.0, -0.3, 0.2, 'affect'),
            (0.5, 0.8, 0.15, 'resilience'),
            (0.5, 0.1, 0.25, 'resilience'),
        ]

        for initial, final, rate, value_type in test_cases:
            result1 = compute_homeostatic_adjustment(initial, final, rate, value_type)
            result2 = compute_homeostatic_adjustment(initial, final, rate, value_type)

            assert result1 == pytest.approx(result2, abs=1e-15)


class TestHomeostaticAdjustmentIntegration:
    """Test integration scenarios and realistic use cases."""

    def test_realistic_affect_recovery_scenario(self):
        """Test a realistic scenario of affect recovery after stress event."""
        # Simulate an agent starting at neutral affect
        baseline_affect = 0.0

        # Experience a stressful event that drops affect significantly
        stressful_event_impact = -0.7
        current_affect = baseline_affect + stressful_event_impact  # -0.7

        # Apply homeostatic adjustment over several days
        homeostatic_rate = 0.15  # Gradual recovery

        recovery_path = [current_affect]

        for day in range(7):  # One week of recovery
            adjusted_affect = compute_homeostatic_adjustment(
                baseline_affect, current_affect, homeostatic_rate, 'affect'
            )
            recovery_path.append(adjusted_affect)
            current_affect = adjusted_affect

        # Should show gradual recovery toward baseline
        assert recovery_path[0] == -0.7  # Starting point
        assert all(-1.0 <= affect <= 1.0 for affect in recovery_path)  # Valid range

        # Should be moving toward baseline over time
        for i in range(1, len(recovery_path)):
            assert abs(recovery_path[i] - baseline_affect) < abs(recovery_path[i-1] - baseline_affect)

    def test_realistic_resilience_depletion_scenario(self):
        """Test a realistic scenario of resilience depletion and recovery."""
        # Simulate an agent starting with moderate resilience
        baseline_resilience = 0.6

        # Experience prolonged stress that depletes resilience
        depletion_impact = -0.4
        current_resilience = baseline_resilience + depletion_impact  # 0.2

        # Apply homeostatic adjustment over several days
        homeostatic_rate = 0.1  # Slower resilience recovery

        recovery_path = [current_resilience]

        for day in range(10):  # Two weeks of recovery
            adjusted_resilience = compute_homeostatic_adjustment(
                baseline_resilience, current_resilience, homeostatic_rate, 'resilience'
            )
            recovery_path.append(adjusted_resilience)
            current_resilience = adjusted_resilience

        # Should show gradual recovery toward baseline
        assert recovery_path[0] == pytest.approx(0.2, abs=1e-10)  # Starting point
        assert all(0.0 <= resilience <= 1.0 for resilience in recovery_path)  # Valid range

        # Should be moving toward baseline over time
        for i in range(1, len(recovery_path)):
            assert abs(recovery_path[i] - baseline_resilience) < abs(recovery_path[i-1] - baseline_resilience)


if __name__ == "__main__":
    pytest.main([__file__])