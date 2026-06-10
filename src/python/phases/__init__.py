"""Phase functions for the ABM simulation.

Each phase module exports:
- run_phase(state, config, rng) -> PhaseOutput  (the phase function)
- PHASE_FREQUENCY: Literal["event_driven", "daily"]   (calling context)
"""

from src.python.phases.interfaces import (
    AgentState,
    PhaseOutput,
    PhaseFunction,
    PhaseFrequency,
)

from src.python.phases.stress_perception import (
    run_phase as run_stress_perception,
    PHASE_FREQUENCY as STRESS_PERCEPTION_FREQUENCY,
)
from src.python.phases.resilience_activation import (
    run_phase as run_resilience_activation,
    PHASE_FREQUENCY as RESILIENCE_ACTIVATION_FREQUENCY,
)
from src.python.phases.resource_allocation import (
    run_phase as run_resource_allocation,
    PHASE_FREQUENCY as RESOURCE_ALLOCATION_FREQUENCY,
)
from src.python.phases.stress_buffering import (
    run_phase as run_stress_buffering,
    PHASE_FREQUENCY as STRESS_BUFFERING_FREQUENCY,
)
from src.python.phases.interaction import (
    run_phase as run_interaction,
    PHASE_FREQUENCY as INTERACTION_FREQUENCY,
)

__all__ = [
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
