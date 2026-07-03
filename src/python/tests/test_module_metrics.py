"""Tests for module_metrics.py — shared metric schemas and extraction functions."""

from __future__ import annotations

import pytest
from numpy.random import Generator, PCG64

from src.python.phases.interfaces import AgentState, PhaseOutput


class TestMetricSchemas:
    """Verify metric dataclass/TypedDict schemas exist with correct fields."""

    def test_import_module(self):
        """module_metrics module imports without error."""
        from src.python.demos import module_metrics  # noqa: F401

    def test_perception_metrics_schema(self):
        """PerceptionMetrics has correct fields."""
        from src.python.demos.module_metrics import PerceptionMetrics

        fields = {
            "event_controllability",
            "event_overload",
            "challenge",
            "hindrance",
            "appraised_stress",
            "effective_threshold",
            "is_stressed",
            "post_stress_controllability",
            "post_stress_overload",
            "perception_sensitivity",
        }
        assert set(PerceptionMetrics.__annotations__.keys()) == fields

    def test_activation_metrics_schema(self):
        """ActivationMetrics has correct fields."""
        from src.python.demos.module_metrics import ActivationMetrics

        fields = {
            "coping_probability",
            "coped_successfully",
            "resilience_effect",
            "delta_stress",
            "delta_affect",
            "resource_cost",
            "optimized_cost",
            "consecutive_hindrances",
            "post_coping_pss10",
            "neighbor_affect_modulation",
            "pf_allocation",
        }
        assert set(ActivationMetrics.__annotations__.keys()) == fields

    def test_allocation_metrics_schema(self):
        """AllocationMetrics has correct fields."""
        from src.python.demos.module_metrics import AllocationMetrics

        fields = {
            "regeneration_amount",
            "resources_before",
            "resources_after",
            "allocation_weights",
            "pf_efficacies_before",
            "pf_efficacies_after",
            "total_allocated",
            "net_resource_change",
            "pf_growth_rate",
        }
        assert set(AllocationMetrics.__annotations__.keys()) == fields

    def test_buffering_metrics_schema(self):
        """BufferingMetrics has correct fields."""
        from src.python.demos.module_metrics import BufferingMetrics

        fields = {
            "current_stress_input",
            "resources_before",
            "resources_after",
            "resilience_after",
            "buffering_strength",
            "pf_boost",
            "stress_depletion",
            "mediated_effect",
        }
        assert set(BufferingMetrics.__annotations__.keys()) == fields

    def test_daily_cycle_metrics_schema(self):
        """DailyCycleMetrics has correct fields."""
        from src.python.demos.module_metrics import DailyCycleMetrics

        fields = {
            "pre_reset_affect",
            "post_reset_affect",
            "affect_reset_magnitude",
            "pre_decay_stress",
            "post_decay_stress",
            "stress_decay_amount",
            "daily_stress_event_count",
            "daily_interaction_count",
            "daily_support_count",
            "social_influence_amount",
            "homeostatic_amount",
            "resource_regen_amount",
            "final_daily_pss10",
        }
        assert set(DailyCycleMetrics.__annotations__.keys()) == fields


class TestExtractPerception:
    """Extract perception metrics from PhaseOutput."""

    def test_returns_perception_metrics(self):
        """extract_perception_metrics returns PerceptionMetrics dict."""
        from src.python.demos.module_metrics import extract_perception_metrics, PerceptionMetrics

        rng = Generator(PCG64(42))
        from src.python.phases.stress_perception import run_phase

        state = AgentState(
            stress_controllability=0.5,
            stress_overload=0.5,
            volatility=0.5,
            recent_stress_intensity=0.0,
            stress_momentum=0.0,
            resilience=0.5,
        )
        config = {
            "omega_c": 0.7,
            "omega_o": 0.5,
            "bias": 0.0,
            "gamma": 6.0,
            "delta": 0.2,
            "base_threshold": 0.5,
            "challenge_scale": 0.15,
            "hindrance_scale": 0.25,
        }
        output = run_phase(state, config, rng)
        pre_state = AgentState(
            stress_controllability=0.5,
            stress_overload=0.5,
        )
        metrics = extract_perception_metrics(output, pre_state)

        # TypedDict cannot use isinstance(); verify fields via dict access
        assert set(metrics.keys()) == set(PerceptionMetrics.__annotations__.keys())
        assert isinstance(metrics["is_stressed"], bool)
        assert 0.0 <= metrics["challenge"] <= 1.0
        assert 0.0 <= metrics["hindrance"] <= 1.0
        assert isinstance(metrics["perception_sensitivity"], float)

    def test_no_output_returns_none(self):
        """Returns None when output is None."""
        from src.python.demos.module_metrics import extract_perception_metrics

        assert extract_perception_metrics(None, {}) is None


class TestExtractActivation:
    """Extract activation metrics from PhaseOutput."""

    def test_returns_activation_metrics(self):
        """extract_activation_metrics returns ActivationMetrics dict."""
        from src.python.demos.module_metrics import extract_activation_metrics, ActivationMetrics

        rng = Generator(PCG64(42))
        from src.python.phases.resilience_activation import run_phase

        state = AgentState(
            affect=0.0,
            resilience=0.5,
            current_stress=0.3,
            baseline_resilience=0.5,
            challenge=0.7,
            hindrance=0.3,
            protective_factors={
                "social_support": 0.5,
                "family_support": 0.5,
                "formal_intervention": 0.5,
                "psychological_capital": 0.5,
            },
            resources=0.6,
            stress_controllability=0.5,
            stress_overload=0.5,
            volatility=0.3,
            recent_stress_intensity=0.0,
            stress_momentum=0.0,
        )
        config = {"neighbor_affects": [0.2, -0.1, 0.4], "base_resource_cost": 0.1}
        output = run_phase(state, config, rng)

        pre_state = AgentState(affect=0.0, resilience=0.5, current_stress=0.3)
        metrics = extract_activation_metrics(output, pre_state)

        assert set(metrics.keys()) == set(ActivationMetrics.__annotations__.keys())
        assert isinstance(metrics["coped_successfully"], bool)
        assert isinstance(metrics["pf_allocation"], dict)

    def test_no_output_returns_none(self):
        """Returns None when output is None."""
        from src.python.demos.module_metrics import extract_activation_metrics

        assert extract_activation_metrics(None, {}) is None


class TestExtractAllocation:
    """Extract allocation metrics from PhaseOutput."""

    def test_returns_allocation_metrics(self):
        """extract_allocation_metrics returns AllocationMetrics dict."""
        from src.python.demos.module_metrics import extract_allocation_metrics, AllocationMetrics

        rng = Generator(PCG64(42))
        from src.python.phases.resource_allocation import run_phase

        state = AgentState(
            resources=0.5,
            affect=0.0,
            resilience=0.5,
            protective_factors={
                "social_support": 0.5,
                "family_support": 0.5,
                "formal_intervention": 0.5,
                "psychological_capital": 0.5,
            },
        )
        config = {
            "base_regeneration": 0.1,
            "softmax_temperature": 1.0,
            "protective_improvement_rate": 0.1,
        }
        output = run_phase(state, config, rng)
        pre_state = AgentState(resources=0.5, protective_factors=dict(state["protective_factors"]))

        metrics = extract_allocation_metrics(output, pre_state)

        assert set(metrics.keys()) == set(AllocationMetrics.__annotations__.keys())
        assert 0.0 <= metrics["regeneration_amount"] <= 1.0
        assert isinstance(metrics["allocation_weights"], dict)
        assert len(metrics["allocation_weights"]) == 4
        assert isinstance(metrics["pf_growth_rate"], dict)

    def test_no_output_returns_none(self):
        """Returns None when output is None."""
        from src.python.demos.module_metrics import extract_allocation_metrics

        assert extract_allocation_metrics(None, {}) is None


class TestExtractBuffering:
    """Extract buffering metrics from PhaseOutput."""

    def test_returns_buffering_metrics(self):
        """extract_buffering_metrics returns BufferingMetrics dict."""
        from src.python.demos.module_metrics import extract_buffering_metrics, BufferingMetrics

        rng = Generator(PCG64(42))
        from src.python.phases.stress_buffering import run_phase

        state = AgentState(
            current_stress=0.5,
            resilience=0.5,
            baseline_resilience=0.5,
            protective_factors={
                "social_support": 0.5,
                "family_support": 0.5,
                "formal_intervention": 0.5,
                "psychological_capital": 0.5,
            },
            resources=0.5,
            stress_overload=0.5,
        )
        config = {}
        output = run_phase(state, config, rng)
        pre_state = AgentState(resources=0.5, resilience=0.5)

        metrics = extract_buffering_metrics(output, pre_state)

        assert set(metrics.keys()) == set(BufferingMetrics.__annotations__.keys())
        assert isinstance(metrics["buffering_strength"], float)
        assert isinstance(metrics["pf_boost"], (float, dict))

    def test_no_output_returns_none(self):
        """Returns None when output is None."""
        from src.python.demos.module_metrics import extract_buffering_metrics

        assert extract_buffering_metrics(None, {}) is None


class TestExtractDailyCycle:
    """Extract daily cycle metrics from multiple outputs."""

    def test_returns_daily_cycle_metrics(self):
        """extract_daily_cycle_metrics returns DailyCycleMetrics dict."""
        from src.python.demos.module_metrics import extract_daily_cycle_metrics, DailyCycleMetrics

        pre_state = AgentState(affect=0.5, baseline_affect=0.0, current_stress=0.7)
        post_state = AgentState(affect=0.2, current_stress=0.4)

        reset_output = PhaseOutput(
            state_delta={"affect": 0.3, "current_stress": 0.5},
            observation={"stress_summary": {"num_events": 3}},
        )
        affect_dynamics_output = PhaseOutput(
            state_delta={},
            observation={"social_influence": 0.1, "homeostatic_amount": 0.05},
        )
        pss10_output = PhaseOutput(
            state_delta={"pss10": 18},
            observation={},
        )

        metrics = extract_daily_cycle_metrics(
            pre_state=pre_state,
            post_state=post_state,
            reset_output=reset_output,
            affect_dynamics_output=affect_dynamics_output,
            pss10_output=pss10_output,
        )

        assert set(metrics.keys()) == set(DailyCycleMetrics.__annotations__.keys())
        assert isinstance(metrics["affect_reset_magnitude"], float)
        assert isinstance(metrics["stress_decay_amount"], float)
        assert isinstance(metrics["final_daily_pss10"], int)

    def test_no_output_returns_none(self):
        """Returns None when any required output is None."""
        from src.python.demos.module_metrics import extract_daily_cycle_metrics

        assert (
            extract_daily_cycle_metrics(
                pre_state={},
                post_state={},
                reset_output=None,
                affect_dynamics_output=None,
                pss10_output=None,
            )
            is None
        )


class TestAggregation:
    """Population-level aggregation functions."""

    def test_aggregate_perception_returns_stats_dict(self):
        """aggregate_perception returns dict with mean, std for each metric."""
        from src.python.demos.module_metrics import (
            PerceptionMetrics,
            aggregate_perception,
        )

        metrics = [
            PerceptionMetrics(
                event_controllability=0.5,
                event_overload=0.5,
                challenge=0.7,
                hindrance=0.3,
                appraised_stress=0.6,
                effective_threshold=0.55,
                is_stressed=True,
                post_stress_controllability=0.48,
                post_stress_overload=0.52,
                perception_sensitivity=0.1,
            ),
            PerceptionMetrics(
                event_controllability=0.3,
                event_overload=0.7,
                challenge=0.2,
                hindrance=0.8,
                appraised_stress=0.8,
                effective_threshold=0.35,
                is_stressed=False,
                post_stress_controllability=0.45,
                post_stress_overload=0.55,
                perception_sensitivity=0.2,
            ),
        ]

        result = aggregate_perception(metrics)
        assert "challenge_mean" in result
        assert "challenge_std" in result
        assert "stress_prevalence" in result
        assert result["stress_prevalence"] == 0.5

    def test_aggregate_empty_returns_empty(self):
        """Aggregation on empty list returns empty dict."""
        from src.python.demos.module_metrics import aggregate_perception

        assert aggregate_perception([]) == {}


class TestIsolationInputs:
    """compute_isolation_inputs generates correct grid for each module."""

    @pytest.mark.parametrize(
        "module",
        [
            "perception",
            "activation",
            "allocation",
            "buffering",
            "daily_cycle",
        ],
    )
    def test_returns_sequence_of_tuples(self, module):
        """Returns sequence of (state, config, rng) tuples."""
        from src.python.demos.module_metrics import compute_isolation_inputs

        fixed_state = AgentState()
        variation_grid = {"default": {"resolution": 5}}
        inputs = compute_isolation_inputs(module, variation_grid, fixed_state)

        assert len(inputs) > 0
        for state, config, rng in inputs:
            assert isinstance(state, dict)
            assert isinstance(config, dict)
            assert isinstance(rng, Generator)

    def test_invalid_module(self):
        """Raises ValueError for unknown module."""
        from src.python.demos.module_metrics import compute_isolation_inputs

        with pytest.raises(ValueError, match="Unknown module"):
            compute_isolation_inputs("nonexistent", {}, {})
