"""Stress perception phase (event-driven).

Appraises incoming stressors and updates stress dimensions.
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
    """Process a stress perception event.

    Args:
        state: Current agent state.
        config: Phase-specific configuration.
        rng: Seeded random number generator.

    Returns:
        PhaseOutput with state_delta (updated stress dimensions) and
        observation (event details).

    Raises:
        NotImplementedError: Stub — implementation pending.
    """
    raise NotImplementedError("stress_perception.run_phase not yet implemented")
