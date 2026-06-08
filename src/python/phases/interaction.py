"""Social interaction phase (event-driven, dual-state).

Processes an agent-agent interaction and updates both agents' states.
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
    """Execute a single social interaction between two agents.

    This phase may read/write state for both the calling agent and the
    interaction partner.

    Args:
        state: Current state of the calling agent.
        config: Phase-specific configuration.
        rng: Seeded random number generator.

    Returns:
        PhaseOutput with state_delta (affect, resilience, resource changes)
        and observation (partner info, support-exchange flag).

    Raises:
        NotImplementedError: Stub — implementation pending.
    """
    raise NotImplementedError("interaction.run_phase not yet implemented")
