"""Resilience activation phase (event-driven).

Determines coping outcome and updates resilience after a stress event.
"""

from typing import Any, Dict

from numpy.random import Generator

from src.python.phases.interfaces import AgentState, PhaseOutput, PhaseFrequency

PHASE_FREQUENCY: PhaseFrequency = "event_driven"


def run_phase(
    state: AgentState,
    config: Dict[str, Any],
    rng: Generator,
) -> PhaseOutput:
    """Activate resilience mechanisms in response to a stress event.

    Args:
        state: Current agent state.
        config: Phase-specific configuration.
        rng: Seeded random number generator.

    Returns:
        PhaseOutput with state_delta (coping outcome, resilience change)
        and observation (coping details).

    Raises:
        NotImplementedError: Stub — implementation pending.
    """
    raise NotImplementedError("resilience_activation.run_phase not yet implemented")
