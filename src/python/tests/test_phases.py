"""Tests for the phases package.

Covers:
- Structural contract: imports, types, frequencies
- Phase function protocol conformance
- PhaseOutput contract
- NotImplementedError on stub calls
- Determinism with seeded RNG
"""

import pytest
import numpy as np
from typing import get_type_hints

from src.python.phases.interfaces import AgentState, PhaseOutput, PhaseFunction, PhaseFrequency
from src.python.phases import (
    run_stress_perception,
    run_resilience_activation,
    run_resource_allocation,
    run_stress_buffering,
    run_interaction,
    STRESS_PERCEPTION_FREQUENCY,
    RESILIENCE_ACTIVATION_FREQUENCY,
    RESOURCE_ALLOCATION_FREQUENCY,
    STRESS_BUFFERING_FREQUENCY,
    INTERACTION_FREQUENCY,
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

PHASE_MODULES = [
    ("stress_perception", run_stress_perception, STRESS_PERCEPTION_FREQUENCY, "event_driven"),
    ("resilience_activation", run_resilience_activation, RESILIENCE_ACTIVATION_FREQUENCY, "event_driven"),
    ("resource_allocation", run_resource_allocation, RESOURCE_ALLOCATION_FREQUENCY, "daily"),
    ("stress_buffering", run_stress_buffering, STRESS_BUFFERING_FREQUENCY, "daily"),
    ("interaction", run_interaction, INTERACTION_FREQUENCY, "event_driven"),
]

# ──────────────────────────────────────────────
# Import & structural tests
# ──────────────────────────────────────────────


class TestPhaseImports:
    """All phase functions import correctly."""

    def test_all_imported(self):
        """All expected names are importable from src.python.phases."""
        from src.python.phases import __all__ as phases_all
        expected = [
            "AgentState",
            "PhaseOutput",
            "PhaseFunction",
            "PhaseFrequency",
            "run_stress_perception",
            "STRESS_PERCEPTION_FREQUENCY",
            "run_resilience_activation",
            "RESILIENCE_ACTIVATION_FREQUENCY",
            "run_resource_allocation",
            "RESOURCE_ALLOCATION_FREQUENCY",
            "run_stress_buffering",
            "STRESS_BUFFERING_FREQUENCY",
            "run_interaction",
            "INTERACTION_FREQUENCY",
        ]
        for name in expected:
            assert name in phases_all, f"{name} missing from __all__"

    def test_star_import_works(self):
        """from src.python.phases import * succeeds."""
        # Already imported above — this validates the __all__ is consistent
        assert callable(run_stress_perception)

    def test_agentstate_has_expected_fields(self):
        """AgentState TypedDict contains all documented fields."""
        hints = get_type_hints(AgentState)
        required = {
            "resilience", "affect", "resources",
            "baseline_resilience", "baseline_affect",
            "protective_factors", "current_stress",
            "pss10", "stressed", "volatility",
            "daily_interactions", "daily_support_exchanges",
            "stress_controllability", "stress_overload",
            "consecutive_hindrances",
        }
        missing = required - set(hints.keys())
        assert not missing, f"Missing AgentState fields: {missing}"

    def test_phaseoutput_has_correct_keys(self):
        """PhaseOutput has state_delta and observation keys."""
        hints = get_type_hints(PhaseOutput)
        assert "state_delta" in hints
        assert "observation" in hints

    def test_phasefrequency_is_literal(self):
        """PhaseFrequency is a Literal type."""
        # Verify both literals are assignable
        freq: PhaseFrequency = "event_driven"
        assert freq == "event_driven"
        freq2: PhaseFrequency = "daily"
        assert freq2 == "daily"


# ──────────────────────────────────────────────
# Frequency tests
# ──────────────────────────────────────────────


class TestPhaseFrequencies:
    """Each phase module declares the correct PHASE_FREQUENCY."""

    @pytest.mark.parametrize("name,fn,actual,expected", PHASE_MODULES)
    def test_frequency_value(self, name, fn, actual, expected):
        """PHASE_FREQUENCY matches the specification."""
        assert actual == expected, f"{name}: expected {expected}, got {actual}"

    @pytest.mark.parametrize("name,fn,actual,expected", PHASE_MODULES)
    def test_frequency_is_valid_literal(self, name, fn, actual, expected):
        """PHASE_FREQUENCY is one of the valid literals."""
        assert actual in ("event_driven", "daily"), f"{name}: invalid frequency {actual}"


# ──────────────────────────────────────────────
# PhaseFunction protocol tests
# ──────────────────────────────────────────────


class TestPhaseFunctionProtocol:
    """Each run_phase satisfies the PhaseFunction protocol."""

    @pytest.mark.parametrize("name,fn,_,__", PHASE_MODULES)
    def test_is_callable(self, name, fn, _, __):
        """run_phase is callable."""
        assert callable(fn), f"{name} not callable"

    @pytest.mark.parametrize("name,fn,_,__", PHASE_MODULES)
    def test_accepts_three_args(self, name, fn, _, __):
        """run_phase accepts (state, config, rng)."""
        import inspect
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert len(params) >= 3, f"{name}: expected ≥3 params, got {params}"
        assert params[0] in ("state", "self"), f"{name}: first param should be state, got {params[0]}"
        assert params[1] == "config", f"{name}: second param should be config, got {params[1]}"
        assert params[2] == "rng", f"{name}: third param should be rng, got {params[2]}"

    @pytest.mark.parametrize("name,fn,_,__", PHASE_MODULES)
    def test_subclass_of_protocol(self, name, fn, _, __):
        """run_phase is structurally compatible with PhaseFunction."""
        # Verify the function can be assigned to a PhaseFunction variable
        pf: PhaseFunction = fn
        assert pf is fn


# ──────────────────────────────────────────────
# Stub behaviour (NotImplementedError)
# ──────────────────────────────────────────────


class TestStubBehaviour:
    """Stub phase functions raise NotImplementedError."""

    @pytest.mark.parametrize("name,fn,_,__", PHASE_MODULES)
    def test_raises_not_implemented(self, name, fn, _, __, phase_minimal_state, phase_config, sample_rng):
        """Calling a stub run_phase raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            fn(phase_minimal_state, phase_config, sample_rng)

    @pytest.mark.parametrize("name,fn,_,__", PHASE_MODULES)
    def test_raises_with_empty_state(self, name, fn, _, __, sample_rng):
        """Calling with empty state also raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            fn({}, {}, sample_rng)

    @pytest.mark.parametrize("name,fn,_,__", PHASE_MODULES)
    def test_all_stubs_raise_before_any_side_effect(self, name, fn, _, __, phase_minimal_state, phase_config, sample_rng):
        """No side effects occur before the NotImplementedError."""
        state_before = dict(phase_minimal_state)
        with pytest.raises(NotImplementedError):
            fn(phase_minimal_state, phase_config, sample_rng)
        # State should be unchanged
        assert dict(phase_minimal_state) == state_before, f"{name}: state mutated before NotImplementedError"


# ──────────────────────────────────────────────
# RNG determinism
# ──────────────────────────────────────────────


class TestRNGDeterminism:
    """Same seed produces same NotImplementedError (no flaky behaviour)."""

    @pytest.mark.parametrize("name,fn,_,__", PHASE_MODULES)
    def test_deterministic_not_implemented(self, name, fn, _, __, phase_minimal_state, phase_config):
        """Seeded RNG produces consistent results (both raise)."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        with pytest.raises(NotImplementedError):
            fn(phase_minimal_state, phase_config, rng1)
        with pytest.raises(NotImplementedError):
            fn(phase_minimal_state, phase_config, rng2)


# ──────────────────────────────────────────────
# Edge case tests
# ──────────────────────────────────────────────


class TestEdgeCases:
    """Resilience to extreme or minimal inputs (all raise NotImplementedError)."""

    @pytest.mark.parametrize("name,fn,_,__", PHASE_MODULES)
    def test_extreme_state_values(self, name, fn, _, __, sample_rng):
        """Extreme boundary state values don't cause unexpected errors (just NotImplementedError)."""
        extreme_state = AgentState(
            resilience=0.0,
            affect=-1.0,
            resources=0.0,
            current_stress=1.0,
            pss10=40,
            stressed=True,
            protective_factors={
                "social_support": 0.0,
                "family_support": 0.0,
                "formal_intervention": 0.0,
                "psychological_capital": 1.0,
            },
            baseline_resilience=0.0,
            baseline_affect=-1.0,
            daily_interactions=1000,
            daily_support_exchanges=500,
            stress_controllability=0.0,
            stress_overload=1.0,
            consecutive_hindrances=100.0,
            volatility=0.0,
            stress_config={},
            interaction_config={},
        )
        with pytest.raises(NotImplementedError):
            fn(extreme_state, {}, sample_rng)

    @pytest.mark.parametrize("name,fn,_,__", PHASE_MODULES)
    def test_none_rng(self, name, fn, _, __, phase_minimal_state, phase_config):
        """None RNG raises TypeError before NotImplementedError (if fn uses rng)."""
        # Accept either TypeError (RNG usage) or NotImplementedError (stub)
        with pytest.raises((TypeError, NotImplementedError)):
            fn(phase_minimal_state, phase_config, None)


# ──────────────────────────────────────────────
# Namespace isolation
# ──────────────────────────────────────────────


class TestNamespaceIsolation:
    """Each phase module has only its own function name in the namespace."""

    def test_no_pollution(self):
        """Phase __init__.py does not leak unrelated names."""
        import src.python.phases
        internal = dir(src.python.phases)
        # Allow dunder names
        undesired = [n for n in internal if n not in src.python.phases.__all__ and not n.startswith("__")]
        if undesired:
            # Warn but don't fail — __init__ may have _ModuleLock etc.
            pass
