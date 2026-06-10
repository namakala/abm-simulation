"""Shared type contracts for all phase functions.

Every phase function accepts (state, config, rng) and returns PhaseOutput.
"""

from typing import Any, Dict, List, Literal, Protocol, TypedDict

from numpy.random import Generator

# ──────────────────────────────────────────────
# Agent State (single source of truth)
# ──────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    """All mutable agent variables exposed to phase functions.

    Drawn from Person.__init__ in agent.py.  Optional fields (total=False)
    allow partial state updates / observations.
    """

    # Core state
    baseline_resilience: float  # [0, 1]
    resilience: float  # [0, 1]
    resources: float  # [0, 1]
    baseline_affect: float  # [-1, 1]
    affect: float  # [-1, 1]

    # Protective factors (four-component dict)
    protective_factors: Dict[str, float]

    # Stress tracking
    current_stress: float  # [0, 1]
    recent_stress_intensity: float
    stress_momentum: float
    last_stress_update: int
    daily_stress_events: List[Dict[str, Any]]
    stress_history: List[Dict[str, Any]]
    last_reset_day: int
    consecutive_hindrances: float
    stress_breach_count: int

    # PSS-10 state
    pss10_responses: Dict[str, Any]
    stress_controllability: float  # [0, 1]
    stress_overload: float  # [0, 1]
    pss10: int  # 0-40
    stressed: bool
    daily_pss10_scores: List[int]

    # Daily counters
    daily_interactions: int
    daily_support_exchanges: int

    # Agent-level configuration (read-only in phases)
    stress_config: Dict[str, Any]
    interaction_config: Dict[str, Any]

    # Personality / fixed traits
    volatility: float


# ──────────────────────────────────────────────
# Phase Output Contract
# ──────────────────────────────────────────────


class PhaseOutput(TypedDict):
    """Return type of every phase function.

    Fields:
        state_delta: key-value pairs to apply as updates to AgentState.
        observation: non-state data the caller may log / record.
    """

    state_delta: Dict[str, Any]
    observation: Dict[str, Any]


# ──────────────────────────────────────────────
# Phase Frequency
# ──────────────────────────────────────────────

PhaseFrequency = Literal["event_driven", "daily"]


# ──────────────────────────────────────────────
# Phase Function Protocol
# ──────────────────────────────────────────────


class PhaseFunction(Protocol):
    """Call signature every phase function must satisfy."""

    def __call__(
        self,
        state: AgentState,
        config: Dict[str, Any],
        rng: Generator,
    ) -> PhaseOutput: ...
