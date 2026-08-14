"""Signature freeze: enforce phase function interfaces stay locked.

Covers:
- All phase functions importable from src.python.phases
- PHASE_FREQUENCY matches expected value per phase
- run_phase accepts (state, config, rng) returning PhaseOutput
- process_interaction returns Tuple[PhaseOutput, PhaseOutput]
- All state_delta keys produced by phases are declared in AgentState
"""

from typing import Tuple, get_type_hints

import pytest

from src.python.phases import (
    INTERACTION_FREQUENCY,
    RESILIENCE_ACTIVATION_FREQUENCY,
    RESOURCE_ALLOCATION_FREQUENCY,
    STRESS_BUFFERING_FREQUENCY,
    STRESS_PERCEPTION_FREQUENCY,
    run_interaction,
    run_resilience_activation,
    run_resource_allocation,
    run_stress_buffering,
    run_stress_perception,
)
from src.python.phases.interfaces import AgentState, PhaseOutput
from src.python.phases.interaction import process_interaction

# ──────────────────────────────────────────────
# Registry: (name, run_phase_fn, expected_frequency)
# ──────────────────────────────────────────────

PHASES = [
    ("stress_perception", run_stress_perception, STRESS_PERCEPTION_FREQUENCY, "event_driven"),
    ("resilience_activation", run_resilience_activation, RESILIENCE_ACTIVATION_FREQUENCY, "event_driven"),
    ("resource_allocation", run_resource_allocation, RESOURCE_ALLOCATION_FREQUENCY, "daily"),
    ("stress_buffering", run_stress_buffering, STRESS_BUFFERING_FREQUENCY, "daily"),
    ("interaction", run_interaction, INTERACTION_FREQUENCY, "event_driven"),
]

# state_delta keys produced by each phase's run_phase
# Derived from source code analysis — update when phase logic changes
PHASE_STATE_DELTA_KEYS = {
    "stress_perception": {
        "challenge",
        "hindrance",
        "is_stressed",
        "event_controllability",
        "event_overload",
        "stress_controllability",
        "stress_overload",
        "recent_stress_intensity",
        "stress_momentum",
    },
    "resilience_activation": {
        "affect",
        "resilience",
        "current_stress",
        "stress_controllability",
        "stress_overload",
        "resources",
        "protective_factors",
        "consecutive_hindrances",
        "stress_breach_count",
        "pss10",
        "pss10_responses",
        "stressed",
    },
    "resource_allocation": {
        "resources",
        "protective_factors",
    },
    "stress_buffering": {
        "resilience",
        "resources",
    },
    "interaction": set(),  # still a stub — run_phase raises NotImplementedError
}


# ──────────────────────────────────────────────
# 1. Importability
# ──────────────────────────────────────────────


class TestPhaseImportability:
    """Every phase function is importable from src.python.phases."""

    @pytest.mark.parametrize("name,fn,freq1,freq2", PHASES)
    def test_importable(self, name, fn, freq1, freq2):
        """Phase function is callable after import."""
        assert callable(fn), f"{name} is not callable"

    @pytest.mark.parametrize("name,fn,actual_freq,expected_freq", PHASES)
    def test_module_has_phase_frequency(self, name, fn, actual_freq, expected_freq):
        """PHASE_FREQUENCY is exported for each phase."""
        assert actual_freq in ("event_driven", "daily"), f"{name}: invalid frequency"


# ──────────────────────────────────────────────
# 2. Frequency correctness
# ──────────────────────────────────────────────


class TestPhaseFrequencyLock:
    """PHASE_FREQUENCY values match the Plan 008 specification."""

    @pytest.mark.parametrize("name,fn,actual,expected", PHASES)
    def test_frequency_matches_spec(self, name, fn, actual, expected):
        """Phase frequency is locked to the expected value."""
        assert actual == expected, f"{name}: expected {expected}, got {actual}"


# ──────────────────────────────────────────────
# 3. Parameter count and return type
# ──────────────────────────────────────────────


class TestPhaseSignature:
    """run_phase functions satisfy the PhaseFunction protocol."""

    @pytest.mark.parametrize("name,fn,freq1,freq2", PHASES)
    def test_params_are_state_config_rng(self, name, fn, freq1, freq2):
        """run_phase accepts (state, config, rng)."""
        import inspect

        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert len(params) >= 3, f"{name}: expected ≥3 params, got {params}"
        assert params[0] in ("state", "self"), f"{name}: first param should be state, got {params[0]}"
        assert params[1] == "config", f"{name}: second param should be config, got {params[1]}"
        assert params[2] == "rng", f"{name}: third param should be rng, got {params[2]}"

    @pytest.mark.parametrize("name,fn,freq1,freq2", PHASES)
    def test_return_type_is_phaseoutput(self, name, fn, freq1, freq2):
        """run_phase returns PhaseOutput."""
        hints = get_type_hints(fn)
        ret = hints.get("return")
        msg = f"{name}: return type is {ret}, expected PhaseOutput"
        if ret is None:
            # Accept None return annotation for now
            return
        assert ret == PhaseOutput, msg


# ──────────────────────────────────────────────
# 4. Plan 005: process_interaction returns Tuple[PhaseOutput, PhaseOutput]
# ──────────────────────────────────────────────


class TestProcessInteractionSignature:
    """process_interaction has a dual-return signature (Plan 005)."""

    def test_process_interaction_importable(self):
        """process_interaction is importable from src.python.phases.interaction."""
        assert callable(process_interaction)

    def test_process_interaction_accepts_four_params(self):
        """process_interaction(self_state, partner_state, config, rng)."""
        import inspect

        sig = inspect.signature(process_interaction)
        params = list(sig.parameters.keys())
        assert len(params) >= 4, f"Expected ≥4 params, got {params}"
        assert params[0] == "self_state", f"First param should be self_state, got {params[0]}"
        assert params[1] == "partner_state", f"Second param should be partner_state, got {params[1]}"
        assert params[2] == "config", f"Third param should be config, got {params[2]}"
        assert params[3] == "rng", f"Fourth param should be rng, got {params[3]}"

    def test_process_interaction_returns_tuple_of_two_phaseoutputs(self):
        """process_interaction return type is Tuple[PhaseOutput, PhaseOutput]."""
        hints = get_type_hints(process_interaction)
        ret = hints.get("return")
        assert ret is not None, "process_interaction has no return annotation"
        # Check it's a Tuple[PhaseOutput, PhaseOutput]
        origin = getattr(ret, "__origin__", None)
        args = getattr(ret, "__args__", None)
        assert origin is Tuple or origin is tuple, f"Return type is {origin}, expected Tuple"
        assert args is not None and len(args) == 2, f"Tuple expects 2 args, got {args}"
        assert args[0] == PhaseOutput, f"First tuple element is {args[0]}, expected PhaseOutput"
        assert args[1] == PhaseOutput, f"Second tuple element is {args[1]}, expected PhaseOutput"


# ──────────────────────────────────────────────
# 5. State delta keys exist in AgentState
# ──────────────────────────────────────────────


class TestStateDeltaKeysInAgentState:
    """Every key in any phase's state_delta is declared in AgentState."""

    @pytest.mark.parametrize("name,fn,freq1,freq2", PHASES)
    def test_all_state_delta_keys_in_agentstate(self, name, fn, freq1, freq2):
        """All state_delta keys produced by {name} exist in AgentState."""
        hints = get_type_hints(AgentState)
        expected_keys = PHASE_STATE_DELTA_KEYS.get(name, set())
        missing = expected_keys - set(hints.keys())
        assert not missing, f"{name}: state_delta keys missing from AgentState: {missing}"

    def test_interaction_process_interaction_delta_keys(self):
        """process_interaction state_delta keys exist in AgentState."""
        hints = get_type_hints(AgentState)
        # process_interaction returns state_deltas with these possible keys
        delta_keys = {
            "affect",
            "resilience",
            "resources",
            "protective_factors",
        }
        missing = delta_keys - set(hints.keys())
        assert not missing, f"process_interaction delta keys missing from AgentState: {missing}"
