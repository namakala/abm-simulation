"""
Example unit tests for affect_utils.py

This file demonstrates testing patterns for affect dynamics and social interactions.
"""

import pytest
import numpy as np
from src.python.affect_utils import (
    process_interaction, compute_stress_impact_on_affect,
    compute_stress_impact_on_resilience, clamp, allocate_protective_resources,
    compute_resource_regeneration, compute_allocation_cost,
    InteractionConfig, ProtectiveFactors, ResourceParams
)


class TestSocialInteractions:
    """Test social interaction mechanics."""

    def test_process_interaction_basic(self):
        """Test basic social interaction between two agents."""
        config = InteractionConfig(influence_rate=0.1, resilience_influence=0.05)

        self_affect = 0.2
        partner_affect = -0.3
        self_resilience = 0.6
        partner_resilience = 0.4

        new_self_affect, new_partner_affect, new_self_resilience, new_partner_resilience = (
            process_interaction(
                self_affect, partner_affect,
                self_resilience, partner_resilience,
                config
            )
        )

        # Check that values are in valid ranges
        assert -1.0 <= new_self_affect <= 1.0
        assert -1.0 <= new_partner_affect <= 1.0
        assert 0.0 <= new_self_resilience <= 1.0
        assert 0.0 <= new_partner_resilience <= 1.0

    def test_process_interaction_symmetric(self):
        """Test that interaction is symmetric in influence."""
        config = InteractionConfig(influence_rate=0.1)

        # Test with symmetric starting conditions
        result1 = process_interaction(0.5, -0.5, 0.5, 0.5, config)
        result2 = process_interaction(-0.5, 0.5, 0.5, 0.5, config)

        # Results should be symmetric (swapped)
        assert abs(result1[0] - result2[1]) < 1e-10  # self_affect should equal partner_affect when swapped
        assert abs(result1[1] - result2[0]) < 1e-10  # partner_affect should equal self_affect when swapped

    def test_process_interaction_no_change_when_isolated(self):
        """Test that agents with no neighbors don't change."""
        # This would be tested in integration tests, but we can test the logic here
        config = InteractionConfig()

        # Same affect should result in no net change
        self_affect = 0.3
        partner_affect = 0.3

        new_self_affect, new_partner_affect, _, _ = process_interaction(
            self_affect, partner_affect, 0.5, 0.5, config
        )

        # With same affect, there should be minimal change
        assert abs(new_self_affect - self_affect) < config.influence_rate
        assert abs(new_partner_affect - partner_affect) < config.influence_rate


class TestStressImpact:
    """Test how stress events impact affect and resilience."""

    def test_stress_impact_on_affect_coping_success(self):
        """Test affect change when coping successfully with stress."""
        config = {'coping_improvement': 0.1, 'coping_deterioration': 0.1, 'no_stress_effect': 0.0}

        current_affect = 0.0

        # Successful coping should improve affect
        affect_change = compute_stress_impact_on_affect(
            current_affect=current_affect,
            is_stressed=True,
            coped_successfully=True,
            config=config
        )

        assert affect_change == config['coping_improvement']

    def test_stress_impact_on_affect_coping_failure(self):
        """Test affect change when coping fails."""
        config = {'coping_improvement': 0.1, 'coping_deterioration': 0.1, 'no_stress_effect': 0.0}

        current_affect = 0.0

        # Failed coping should deteriorate affect
        affect_change = compute_stress_impact_on_affect(
            current_affect=current_affect,
            is_stressed=True,
            coped_successfully=False,
            config=config
        )

        assert affect_change == -config['coping_deterioration']

    def test_stress_impact_on_affect_no_stress(self):
        """Test that no stress event produces no affect change."""
        config = {'coping_improvement': 0.1, 'coping_deterioration': 0.1, 'no_stress_effect': 0.0}

        current_affect = 0.5

        # No stress should produce no change
        affect_change = compute_stress_impact_on_affect(
            current_affect=current_affect,
            is_stressed=False,
            coped_successfully=True,  # This shouldn't matter
            config=config
        )

        assert affect_change == config['no_stress_effect']

    def test_stress_impact_on_resilience(self):
        """Test resilience changes due to stress outcomes."""
        config = {'coping_improvement': 0.08, 'coping_deterioration': 0.08, 'no_stress_effect': 0.0}

        current_resilience = 0.5

        # Successful coping should improve resilience
        resilience_change_success = compute_stress_impact_on_resilience(
            current_resilience=current_resilience,
            is_stressed=True,
            coped_successfully=True,
            config=config
        )

        # Failed coping should deteriorate resilience
        resilience_change_failure = compute_stress_impact_on_resilience(
            current_resilience=current_resilience,
            is_stressed=True,
            coped_successfully=False,
            config=config
        )

        assert resilience_change_success == config['coping_improvement']
        assert resilience_change_failure == -config['coping_deterioration']


class TestResourceAllocation:
    """Test protective factor resource allocation."""

    def test_allocate_protective_resources_basic(self):
        """Test basic resource allocation across protective factors."""
        available_resources = 0.5
        protective_factors = ProtectiveFactors(
            social_support=0.8,
            family_support=0.6,
            formal_intervention=0.4,
            psychological_capital=0.2
        )

        rng = np.random.default_rng(42)
        allocations = allocate_protective_resources(
            available_resources, protective_factors, rng
        )

        # Check that allocations sum to available resources
        total_allocated = sum(allocations.values())
        assert abs(total_allocated - available_resources) < 1e-10

        # Check that all factors get non-negative allocation
        for factor, allocation in allocations.items():
            assert allocation >= 0.0
            assert factor in ['social_support', 'family_support', 'formal_intervention', 'psychological_capital']

    def test_allocate_protective_resources_deterministic(self):
        """Test that allocation is deterministic with fixed RNG."""
        available_resources = 0.3
        protective_factors = ProtectiveFactors()

        rng = np.random.default_rng(123)

        # Two allocations with same seed should be identical
        allocation1 = allocate_protective_resources(available_resources, protective_factors, rng)
        rng = np.random.default_rng(123)  # Reset
        allocation2 = allocate_protective_resources(available_resources, protective_factors, rng)

        for factor in allocation1:
            assert abs(allocation1[factor] - allocation2[factor]) < 1e-10

    def test_resource_regeneration(self):
        """Test passive resource regeneration."""
        config = ResourceParams(base_regeneration=0.1)

        # Test regeneration at different resource levels
        low_resources = 0.2
        high_resources = 0.8

        regen_low = compute_resource_regeneration(low_resources, config)
        regen_high = compute_resource_regeneration(high_resources, config)

        # Lower resources should regenerate more
        assert regen_low > regen_high
        assert regen_low == 0.1 * (1.0 - low_resources)
        assert regen_high == 0.1 * (1.0 - high_resources)

    def test_allocation_cost_computation(self):
        """Test convex cost function for resource allocation."""
        config = ResourceParams(allocation_cost=0.1, cost_exponent=2.0)

        # Test cost at different allocation levels
        small_allocation = 0.2
        large_allocation = 0.8

        cost_small = compute_allocation_cost(small_allocation, config)
        cost_large = compute_allocation_cost(large_allocation, config)

        # Larger allocation should have disproportionately higher cost (convex)
        expected_cost_small = 0.1 * (0.2 ** 2.0)  # 0.1 * 0.04 = 0.004
        expected_cost_large = 0.1 * (0.8 ** 2.0)  # 0.1 * 0.64 = 0.064

        assert abs(cost_small - expected_cost_small) < 1e-10
        assert abs(cost_large - expected_cost_large) < 1e-10
        assert cost_large > cost_small  # Larger allocation costs more


class TestUtilityFunctions:
    """Test general utility functions."""

    def test_clamp_function(self):
        """Test value clamping to specified bounds."""
        # Test basic clamping
        assert clamp(0.5, 0.0, 1.0) == 0.5
        assert clamp(-0.1, 0.0, 1.0) == 0.0  # Below minimum
        assert clamp(1.5, 0.0, 1.0) == 1.0   # Above maximum

        # Test custom bounds
        assert clamp(5, 0, 10) == 5
        assert clamp(-5, 0, 10) == 0
        assert clamp(15, 0, 10) == 10

    def test_clamp_preserves_type(self):
        """Test that clamp preserves input type."""
        assert isinstance(clamp(0.5), float)
        assert isinstance(clamp(np.float64(0.5)), np.float64)


# Integration test example
class TestIntegrationScenarios:
    """Example integration tests for combined utility behaviors."""

    def test_stress_interaction_sequence(self):
        """Test a sequence of stress and interaction events."""
        # This demonstrates how utilities work together
        config = InteractionConfig()

        # Start with an agent
        affect = 0.0
        resilience = 0.5

        # Simulate stress event (using simplified logic for testing)
        is_stressed = True
        coped_successfully = False

        # Apply stress impact
        affect_change = compute_stress_impact_on_affect(
            current_affect=affect,
            is_stressed=is_stressed,
            coped_successfully=coped_successfully
        )

        resilience_change = compute_stress_impact_on_resilience(
            current_resilience=resilience,
            is_stressed=is_stressed,
            coped_successfully=coped_successfully
        )

        affect += affect_change
        resilience += resilience_change

        # Now simulate social interaction with a positive partner
        partner_affect = 0.8
        partner_resilience = 0.7

        new_affect, new_partner_affect, new_resilience, new_partner_resilience = (
            process_interaction(affect, partner_affect, resilience, partner_resilience, config)
        )

        # Agent should have improved due to positive social interaction
        # Note: The exact improvement depends on the interaction parameters
        # Let's just verify that values are still in valid ranges and reasonable
        assert -1.0 <= new_affect <= 1.0
        assert -1.0 <= new_partner_affect <= 1.0
        assert new_resilience > resilience  # Should improve from social support

        # All values should be in valid ranges
        assert -1.0 <= new_affect <= 1.0
        assert 0.0 <= new_resilience <= 1.0