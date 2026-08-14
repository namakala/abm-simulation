#!/usr/bin/env python3
"""
Comprehensive tests for empirically grounded PSS-10 score generation functionality.

Tests the new bifactor model implementation including:
- PSS10Item dataclass structure validation
- Dimension correlation using multivariate normal distribution
- PSS-10 item generation with normal distribution sampling and clamping
- Integration with existing stress processing mechanisms
- Configuration parameter validation
"""

import numpy as np
import pytest
import os
from unittest.mock import patch

from src.python.stress_utils import (
    PSS10Item,
    create_pss10_mapping,
    generate_pss10_dimension_scores,
    generate_pss10_from_stress_dimensions,
    generate_pss10_item_response,
    generate_pss10_responses,
    initialize_pss10_from_items,
    map_agent_stress_to_pss10,
    compute_pss10_score,
)

from src.python.config import Config, ConfigurationError


class TestPSS10Item:
    """Test PSS10Item dataclass structure and validation."""

    def test_pss10_item_creation(self):
        """Test PSS10Item dataclass creation with valid parameters."""
        item = PSS10Item(reverse_scored=True, weight_controllability=0.7, weight_overload=0.3)

        assert item.reverse_scored is True
        assert item.weight_controllability == 0.7
        assert item.weight_overload == 0.3

    def test_pss10_item_default_values(self):
        """Test PSS10Item dataclass default values."""
        item = PSS10Item()

        assert item.reverse_scored is False
        assert item.weight_controllability == 0.0
        assert item.weight_overload == 0.0

    def test_pss10_mapping_structure(self):
        """Test that PSS-10 mapping contains all required items with correct structure."""
        mapping = create_pss10_mapping()

        # Should have exactly 10 items
        assert len(mapping) == 10

        # All items should be PSS10Item instances
        for item_num, item in mapping.items():
            assert isinstance(item, PSS10Item)
            assert 1 <= item_num <= 10

        # Check specific items mentioned in requirements
        # Controllability dimension: items 4, 5, 7, 8 (1-indexed)
        controllability_items = [4, 5, 7, 8]
        for item_num in controllability_items:
            item = mapping[item_num]
            # These should have higher controllability loadings
            assert item.weight_controllability > 0.5

        # Overload dimension: items 1, 2, 3, 5, 6, 9, 10 (1-indexed)
        overload_items = [1, 2, 3, 5, 6, 9, 10]
        for item_num in overload_items:
            item = mapping[item_num]
            # These should have higher overload loadings
            assert item.weight_overload > 0

    def test_reverse_scored_items(self):
        """Test that correct items are marked as reverse scored."""
        mapping = create_pss10_mapping()

        # Items 4, 5, 7, 8 should be reverse scored
        reverse_scored_items = {4, 5, 7, 8}
        for item_num in range(1, 11):
            item = mapping[item_num]
            if item_num in reverse_scored_items:
                assert item.reverse_scored is True, f"Item {item_num} should be reverse scored"
            else:
                assert item.reverse_scored is False, f"Item {item_num} should not be reverse scored"


class TestPSS10DimensionCorrelation:
    """Test dimension correlation functionality using multivariate normal distribution."""

    def test_dimension_correlation_basic(self):
        """Test basic dimension correlation with positive correlation."""
        rng = np.random.default_rng(42)

        controllability = 0.6
        overload = 0.4
        correlation = 0.3

        corr_c, corr_o = generate_pss10_dimension_scores(controllability, overload, correlation, rng)

        # Should return values in [0,1] range
        assert 0 <= corr_c <= 1
        assert 0 <= corr_o <= 1

        # Should be close to input values (with some variation due to correlation)
        assert abs(corr_c - controllability) < 0.5  # Allow some deviation
        assert abs(corr_o - overload) < 0.5

    def test_dimension_correlation_negative(self):
        """Test dimension correlation with negative correlation."""
        rng = np.random.default_rng(42)

        controllability = 0.8
        overload = 0.2
        correlation = -0.4

        corr_c, corr_o = generate_pss10_dimension_scores(controllability, overload, correlation, rng)

        assert 0 <= corr_c <= 1
        assert 0 <= corr_o <= 1

    def test_dimension_correlation_extremes(self):
        """Test dimension correlation with extreme correlation values."""
        rng = np.random.default_rng(42)

        # Test perfect positive correlation
        corr_c, corr_o = generate_pss10_dimension_scores(0.5, 0.5, 1.0, rng)
        assert 0 <= corr_c <= 1
        assert 0 <= corr_o <= 1

        # Test perfect negative correlation
        corr_c, corr_o = generate_pss10_dimension_scores(0.5, 0.5, -1.0, rng)
        assert 0 <= corr_c <= 1
        assert 0 <= corr_o <= 1

    def test_dimension_correlation_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        corr_c1, corr_o1 = generate_pss10_dimension_scores(0.5, 0.5, 0.3, rng1)
        corr_c2, corr_o2 = generate_pss10_dimension_scores(0.5, 0.5, 0.3, rng2)

        assert corr_c1 == corr_c2
        assert corr_o1 == corr_o2


class TestPSS10BifactorGeneration:
    """Test PSS-10 item response generation via bifactor model (phase 2)."""

    def test_basic_generation(self):
        """Test basic item response generation with bifactor model."""
        rng = np.random.default_rng(42)
        response = generate_pss10_item_response(
            item_mean=2.0,
            item_sd=1.0,
            controllability_loading=0.5,
            overload_loading=0.5,
            controllability_score=0.5,
            overload_score=0.5,
            rng=rng,
        )
        assert 0 <= response <= 4
        assert isinstance(response, int)

    def test_stress_direction_controllability(self):
        """Higher controllability (lower stress) → higher raw response.

        Controllability items (4,5,7,8) are positively-worded (e.g.
        "felt confident"), so higher control → more agreement → higher
        raw response. The items are reverse-scored later in
        compute_pss10_score so that higher control→lower PSS-10 total.
        """
        rng = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        # Low controllability = high stress → lower response on positively-worded items
        low_control = generate_pss10_item_response(
            item_mean=2.0,
            item_sd=0.5,
            controllability_loading=0.8,
            overload_loading=0.2,
            controllability_score=0.2,
            overload_score=0.5,
            pss10_scale=3.5,
            pss10_noise_sd=0.1,
            rng=rng,
        )
        # High controllability = low stress → higher response on positively-worded items
        high_control = generate_pss10_item_response(
            item_mean=2.0,
            item_sd=0.5,
            controllability_loading=0.8,
            overload_loading=0.2,
            controllability_score=0.8,
            overload_score=0.5,
            pss10_scale=3.5,
            pss10_noise_sd=0.1,
            rng=rng2,
        )
        assert high_control >= low_control

    def test_stress_direction_overload(self):
        """Higher overload → higher response."""
        rng = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        low_overload = generate_pss10_item_response(
            item_mean=2.0,
            item_sd=0.5,
            controllability_loading=0.2,
            overload_loading=0.8,
            controllability_score=0.5,
            overload_score=0.2,
            pss10_scale=3.5,
            pss10_noise_sd=0.1,
            rng=rng,
        )
        high_overload = generate_pss10_item_response(
            item_mean=2.0,
            item_sd=0.5,
            controllability_loading=0.2,
            overload_loading=0.8,
            controllability_score=0.5,
            overload_score=0.8,
            pss10_scale=3.5,
            pss10_noise_sd=0.1,
            rng=rng2,
        )
        assert high_overload >= low_overload

    def test_item_mean_as_baseline(self):
        """At average stress, response centers near item_mean."""
        rng = np.random.default_rng(42)
        # Average stress: controllability = overload = 0.5
        # With high noise, the mean should still approximate item_mean
        responses = []
        for _ in range(100):
            r = generate_pss10_item_response(
                item_mean=2.5,
                item_sd=0.5,
                controllability_loading=0.5,
                overload_loading=0.5,
                controllability_score=0.5,
                overload_score=0.5,
                pss10_scale=3.5,
                pss10_noise_sd=0.1,
                rng=rng,
            )
            responses.append(r)
        assert abs(np.mean(responses) - 2.5) < 0.5

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        r1 = generate_pss10_item_response(
            item_mean=2.0,
            item_sd=1.0,
            controllability_loading=0.5,
            overload_loading=0.5,
            controllability_score=0.5,
            overload_score=0.5,
            rng=rng1,
        )
        r2 = generate_pss10_item_response(
            item_mean=2.0,
            item_sd=1.0,
            controllability_loading=0.5,
            overload_loading=0.5,
            controllability_score=0.5,
            overload_score=0.5,
            rng=rng2,
        )
        assert r1 == r2


class TestPSS10ItemGeneration:
    """Test PSS-10 item response generation functionality."""

    def test_item_response_generation_basic(self):
        """Test basic item response generation."""
        rng = np.random.default_rng(42)

        response = generate_pss10_item_response(
            item_mean=2.0,
            item_sd=1.0,
            controllability_loading=0.5,
            overload_loading=0.5,
            controllability_score=0.6,
            overload_score=0.4,
            rng=rng,
        )

        # Should return valid PSS-10 response (0-4)
        assert 0 <= response <= 4
        assert isinstance(response, int)

    def test_item_response_clamping(self):
        """Test that responses are properly clamped to [0,4] range."""
        rng = np.random.default_rng(42)

        # Test with extreme values that should be clamped
        response = generate_pss10_item_response(
            item_mean=10.0,  # Very high mean
            item_sd=1.0,
            controllability_loading=0.0,
            overload_loading=0.0,
            controllability_score=0.0,
            overload_score=0.0,
            rng=rng,
        )

        assert 0 <= response <= 4

    def test_item_response_reproducibility(self):
        """Test that item responses are reproducible with same random seed."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        response1 = generate_pss10_item_response(
            item_mean=2.0,
            item_sd=1.0,
            controllability_loading=0.5,
            overload_loading=0.5,
            controllability_score=0.5,
            overload_score=0.5,
            rng=rng1,
        )

        response2 = generate_pss10_item_response(
            item_mean=2.0,
            item_sd=1.0,
            controllability_loading=0.5,
            overload_loading=0.5,
            controllability_score=0.5,
            overload_score=0.5,
            rng=rng2,
        )

        assert response1 == response2


class TestPSS10ResponseGeneration:
    """Test complete PSS-10 response generation functionality."""

    def test_complete_response_generation(self):
        """Test generation of complete PSS-10 responses."""
        rng = np.random.default_rng(42)

        responses = generate_pss10_responses(0.5, 0.5, rng)

        # Should have exactly 10 responses
        assert len(responses) == 10

        # All item numbers should be present
        assert set(responses.keys()) == set(range(1, 11))

        # All responses should be valid integers in [0,4]
        for item_num, response in responses.items():
            assert isinstance(response, int)
            assert 0 <= response <= 4

    def test_response_generation_extreme_inputs(self):
        """Test response generation with extreme input values."""
        rng = np.random.default_rng(42)

        # Test with extreme controllability and overload values
        responses = generate_pss10_responses(1.0, 0.0, rng)  # High controllability, low overload

        assert len(responses) == 10
        for response in responses.values():
            assert 0 <= response <= 4

        # Test with opposite extremes
        responses = generate_pss10_responses(0.0, 1.0, rng)  # Low controllability, high overload

        assert len(responses) == 10
        for response in responses.values():
            assert 0 <= response <= 4

    def test_response_generation_reproducibility(self):
        """Test that complete responses are reproducible with same random seed."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        responses1 = generate_pss10_responses(0.5, 0.5, rng1)
        responses2 = generate_pss10_responses(0.5, 0.5, rng2)

        assert responses1 == responses2

    def test_response_generation_with_config(self):
        """Test response generation with custom configuration."""
        rng = np.random.default_rng(42)

        custom_config = {
            "item_means": [2.0] * 10,
            "item_sds": [1.0] * 10,
            "load_controllability": [0.5] * 10,
            "load_overload": [0.5] * 10,
            "bifactor_correlation": 0.0,
        }

        responses = generate_pss10_responses(0.5, 0.5, rng, custom_config)

        assert len(responses) == 10
        for response in responses.values():
            assert 0 <= response <= 4

    def test_pss10_score_direction_positive_with_stress(self):
        """Final PSS-10 score must increase with stress.

        Higher stress (low controllability + high overload) must produce
        a HIGHER final PSS-10 score than lower stress (high controllability
        + low overload). This tests the full pipeline: bifactor generation
        + reverse-scoring in compute_pss10_score.

        Uses a custom config where all items load purely on controllability
        to isolate the controllability-dimension direction.
        """
        custom_config = {
            "item_means": [2.0] * 10,
            "item_sds": [0.3] * 10,
            "load_controllability": [1.0] * 10,
            "load_overload": [0.0] * 10,
            "bifactor_correlation": 0.0,
            "pss10_scale": 3.5,
            "pss10_noise_sd": 0.1,
        }

        # Low stress: high controllability (0.9), overload neutral (0.5)
        low_rng = np.random.default_rng(42)
        high_rng = np.random.default_rng(42)

        low_stress_responses = generate_pss10_responses(
            controllability=0.9,
            overload=0.5,
            rng=low_rng,
            config=custom_config,
        )
        low_stress_score = compute_pss10_score(low_stress_responses)

        # High stress: low controllability (0.1), overload neutral (0.5)
        high_stress_responses = generate_pss10_responses(
            controllability=0.1,
            overload=0.5,
            rng=high_rng,
            config=custom_config,
        )
        high_stress_score = compute_pss10_score(high_stress_responses)

        assert high_stress_score > low_stress_score, (
            f"PSS-10 should increase with stress (higher when controllability is lower): "
            f"controllability=0.9 -> PSS-10={low_stress_score}, "
            f"controllability=0.1 -> PSS-10={high_stress_score}"
        )

    def test_generate_pss10_from_stress_no_bias_parameter(self):
        """generate_pss10_from_stress_dimensions must not have a pss10_bias param.

        The function was refactored to generate PSS-10 purely from stress
        dimensions. Daily N(0, SD) bias is applied later in
        process_pss10_consolidation. Same seed + same inputs must yield
        the same result.
        """
        import inspect

        sig = inspect.signature(generate_pss10_from_stress_dimensions)
        assert "pss10_bias" not in sig.parameters, "generate_pss10_from_stress_dimensions must not accept pss10_bias"

        # Also verify determinism: same inputs + same seed = same output
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)

        a = generate_pss10_from_stress_dimensions(stress_controllability=0.5, stress_overload=0.5, rng=rng_a)
        b = generate_pss10_from_stress_dimensions(stress_controllability=0.5, stress_overload=0.5, rng=rng_b)
        assert a["pss10_score"] == b["pss10_score"]

    def test_intensity_boost_does_not_invert_controllability(self):
        """intensity_boost must not reduce effective controllability.

        The intensity_boost (= recent_stress_intensity * sensitivity)
        is subtracted from dynamic_controllability in the current code.
        This means high recent intensity REDUCES the effective
        controllability, which via the bifactor model INCREASES PSS-10.

        This creates an inversion: on days with many successfully-handled
        events, stress_controllability goes UP (less stress) but
        dynamic_controllability goes DOWN (due to intensity_boost).
        The result: PSS-10 increases while current_stress decreases.

        Scenario A: high control (0.8), low overload (0.2),
                     high intensity (1.0) — good day, many events
        Scenario B: moderate control (0.5), moderate overload (0.5),
                     low intensity (0.1) — worse stress state, calm

        A must produce LOWER PSS-10 than B because its underlying
        stress dimensions are better, even with high intensity.
        """
        rng_a = np.random.default_rng(0)
        rng_b = np.random.default_rng(0)

        result_a = generate_pss10_from_stress_dimensions(
            stress_controllability=0.8,
            stress_overload=0.2,
            recent_stress_intensity=1.0,
            rng=rng_a,
        )
        result_b = generate_pss10_from_stress_dimensions(
            stress_controllability=0.5,
            stress_overload=0.5,
            recent_stress_intensity=0.1,
            rng=rng_b,
        )

        assert result_a["pss10_score"] < result_b["pss10_score"], (
            f"PSS-10 must be driven by stress dimensions, not intensity_boost: "
            f"A (high control, high intensity)={result_a['pss10_score']} >= "
            f"B (moderate control, low intensity)={result_b['pss10_score']}"
        )


class TestPSS10Integration:
    """Test integration with existing stress processing mechanisms."""

    def test_map_agent_stress_integration(self):
        """Test that map_agent_stress_to_pss10 uses new implementation."""
        rng = np.random.default_rng(42)

        responses = map_agent_stress_to_pss10(0.5, 0.5, rng)

        # Should return properly formatted responses
        assert len(responses) == 10
        assert set(responses.keys()) == set(range(1, 11))

        for response in responses.values():
            assert 0 <= response <= 4

    def test_pss10_score_computation(self):
        """Test that PSS-10 score computation still works with new responses."""
        # Create test responses
        responses = {
            1: 2,
            2: 1,
            3: 3,
            4: 1,
            5: 2,  # Reverse scored items: 4, 5
            6: 3,
            7: 1,
            8: 2,
            9: 2,
            10: 3,  # Reverse scored items: 7, 8
        }

        score = compute_pss10_score(responses)

        # Calculate expected score manually
        # Reverse items: 4, 5, 7, 8 → scores: (4-1) + (4-2) + (4-1) + (4-2) = 3 + 2 + 3 + 2 = 10
        # Normal items: 1, 2, 3, 6, 9, 10 → scores: 2 + 1 + 3 + 3 + 2 + 3 = 14
        # Total: 10 + 14 = 24
        expected_score = 24

        assert score == expected_score

    def test_pss10_score_validation(self):
        """Test PSS-10 score computation validation."""
        # Test missing items
        incomplete_responses = {1: 2, 2: 1, 3: 3}  # Missing items 4-10

        with pytest.raises(ValueError, match="Missing PSS-10 items"):
            compute_pss10_score(incomplete_responses)

        # Test invalid response values
        invalid_responses = {i: 5 for i in range(1, 11)}  # All responses = 5 (invalid)

        with pytest.raises(ValueError, match="Invalid response"):
            compute_pss10_score(invalid_responses)


class TestPSS10Initialization:
    """Test PSS-10 initialization from item parameters (phase 1)."""

    def test_initialize_from_item_params_basic(self):
        """Test that initialize_pss10_from_items generates items from item_mean/sd."""
        rng = np.random.default_rng(42)
        config = {
            "item_means": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "item_sds": [0.5] * 10,
            "threshold": 27,
        }

        result = initialize_pss10_from_items(rng=rng, config=config)

        # Should return all expected keys
        assert "pss10_responses" in result
        assert "stress_controllability" in result
        assert "stress_overload" in result
        assert "pss10_score" in result
        assert "stressed" in result

        # Should have exactly 10 item responses
        responses = result["pss10_responses"]
        assert len(responses) == 10
        assert set(responses.keys()) == set(range(1, 11))

        # All responses should be valid integers in [0,4]
        for item_num, response in responses.items():
            assert isinstance(response, int)
            assert 0 <= response <= 4

        # Stress dimensions should be in [0,1]
        assert 0.0 <= result["stress_controllability"] <= 1.0
        assert 0.0 <= result["stress_overload"] <= 1.0

        # PSS-10 score should be in [0,40]
        assert 0 <= result["pss10_score"] <= 40

        # Stressed should be boolean
        assert isinstance(result["stressed"], bool)

    def test_initialize_reproducibility(self):
        """Test that initialization is reproducible with same seed."""
        config = {
            "item_means": [2.0] * 10,
            "item_sds": [0.5] * 10,
            "threshold": 27,
        }

        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        result1 = initialize_pss10_from_items(rng=rng1, config=config)
        result2 = initialize_pss10_from_items(rng=rng2, config=config)

        assert result1["pss10_responses"] == result2["pss10_responses"]
        assert result1["pss10_score"] == result2["pss10_score"]
        assert result1["stressed"] == result2["stressed"]

    def test_initialize_stress_derivation(self):
        """Test that stress dimensions are correctly derived from items."""
        rng = np.random.default_rng(42)
        config = {
            "item_means": [2.0] * 10,
            "item_sds": [0.1] * 10,  # Low SD for deterministic-like results
            "threshold": 27,
        }

        result = initialize_pss10_from_items(rng=rng, config=config)
        responses = result["pss10_responses"]

        # Stress controllability = mean of items 4,5,7,8 / 4
        expected_controllability = np.mean([responses[i] / 4.0 for i in [4, 5, 7, 8]])
        assert result["stress_controllability"] == pytest.approx(expected_controllability, abs=0.01)

        # Stress overload = mean of items 1,2,3,6,9,10 / 4
        expected_overload = np.mean([responses[i] / 4.0 for i in [1, 2, 3, 6, 9, 10]])
        assert result["stress_overload"] == pytest.approx(expected_overload, abs=0.01)


@pytest.mark.config
class TestPSS10Configuration:
    """Test PSS-10 configuration parameters."""

    def test_pss10_config_loading(self):
        """Test that new PSS-10 configuration parameters load correctly."""
        os.environ.clear()
        config = Config()

        # Test new bifactor model parameters
        controllability_loadings = config.get("pss10", "load_controllability")
        overload_loadings = config.get("pss10", "load_overload")
        correlation = config.get("pss10", "bifactor_correlation")

        assert len(controllability_loadings) == 10
        assert len(overload_loadings) == 10

        # All loadings should be in [0,1] range
        for loading in controllability_loadings + overload_loadings:
            assert 0 <= loading <= 1

        # Correlation should be in [-1,1] range
        assert -1 <= correlation <= 1

    def test_pss10_config_validation(self):
        """Test PSS-10 configuration validation."""
        os.environ.clear()
        config = Config()

        # Should validate without errors
        config.validate()

        # Test that validation catches invalid values
        with patch.object(config, "pss10_load_controllability", [1.5] * 10):  # Invalid loading > 1
            with pytest.raises(ConfigurationError, match="controllability loading"):
                config.validate()

    def test_pss10_config_defaults(self):
        """Test PSS-10 configuration default values."""
        os.environ.clear()
        config = Config()

        # Check that defaults match expected empirical values
        expected_controllability = [0, 0, 0, 1, 1, 0, 1, 1, 0, 0]
        expected_overload = [1, 1, 1, 0, 0, 1, 0, 0, 1, 1]

        assert config.get("pss10", "load_controllability") == expected_controllability
        assert config.get("pss10", "load_overload") == expected_overload
        assert config.get("pss10", "bifactor_correlation") == -0.3


class TestPSS10ResilienceCoupling:
    """PSS-10 resilience coupling at item level is amplified by assumption (Fix 4)."""

    def test_resilience_influence_amplified_in_pss10_generation(self):
        """generate_pss10_from_stress_dimensions uses pss10_resilience_item_amplifier."""
        from src.python.assumption_config import get_assumptions

        a = get_assumptions()
        assert hasattr(a.stress, "pss10_resilience_item_amplifier"), "Missing pss10_resilience_item_amplifier"
        assert a.stress.pss10_resilience_item_amplifier > 1.0, "Amplifier should be > 1.0"

        # Config value is 0.20, amplifier default is 2.0, so effective is 0.40
        config_value_reference = 0.20  # from config.py pss10_resilience_coupling_item
        amplifier = a.stress.pss10_resilience_item_amplifier
        effective_coupling = config_value_reference * amplifier
        assert effective_coupling == pytest.approx(0.40, abs=0.01), (
            f"Effective coupling {effective_coupling:.2f} should be 0.40"
        )

    def test_higher_residence_lowers_pss10_more_with_amplifier(self):
        """Higher resilience produces lower PSS-10 scores when amplifier is active."""
        rng = np.random.default_rng(42)
        # Low resilience
        result_low = generate_pss10_from_stress_dimensions(
            stress_controllability=0.5,
            stress_overload=0.5,
            resilience=0.2,
            affect=0.0,
            resources=0.5,
            rng=rng,
        )
        # High resilience
        rng = np.random.default_rng(42)  # reset seed
        result_high = generate_pss10_from_stress_dimensions(
            stress_controllability=0.5,
            stress_overload=0.5,
            resilience=0.8,
            affect=0.0,
            resources=0.5,
            rng=rng,
        )
        # Higher resilience should produce lower PSS-10 scores
        assert result_high["pss10_score"] < result_low["pss10_score"], (
            f"High resilience PSS-10 ({result_high['pss10_score']}) should be < "
            f"low resilience PSS-10 ({result_low['pss10_score']})"
        )


def run_all_tests():
    """Run all PSS-10 empirical tests."""
    print("Running PSS-10 Empirical Generation Test Suite")
    print("=" * 55)

    # Create test instances
    test_classes = [
        TestPSS10Item(),
        TestPSS10DimensionCorrelation(),
        TestPSS10BifactorGeneration(),
        TestPSS10ItemGeneration(),
        TestPSS10ResponseGeneration(),
        TestPSS10Integration(),
        TestPSS10Initialization(),
        TestPSS10Configuration(),
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}:")

        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith("test_")]

        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_class, test_method)()
                print(f"  ✓ {test_method}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {test_method}: {e}")

    print(f"\n{'=' * 55}")
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
