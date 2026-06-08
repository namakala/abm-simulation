"""Resource allocation phase (daily).

Regenerates resources and allocates to protective factors.
"""

from typing import Any, Dict

from numpy.random import Generator

from src.python.phases.interfaces import AgentState, PhaseOutput, PhaseFrequency

PHASE_FREQUENCY: PhaseFrequency = "daily"


def run_phase(
    state: AgentState,
    config: Dict[str, Any],
    rng: Generator,
) -> PhaseOutput:
    """Allocate resources for the day (regeneration + protective factors).

    Args:
        state: Current agent state.
        config: Phase-specific configuration.
        rng: Seeded random number generator.

    Returns:
        PhaseOutput with state_delta (resource changes) and
        observation (allocation summary).

    Raises:
        NotImplementedError: Stub — implementation pending.
    """
    raise NotImplementedError("resource_allocation.run_phase not yet implemented")
