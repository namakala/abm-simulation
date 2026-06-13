"""Social interaction phase (event-driven, dual-state).

Provides `process_interaction()` — the core dyadic interaction function — and
`run_phase()` for protocol compatibility.

`process_interaction()`:
1. Converges affect and resilience between two agents
2. Applies negativity bias (negative influence 1.5x stronger)
3. Detects support from convergence magnitude
4. Runs a win-win/lose-lose resource exchange state machine

State machine (self_stressed, partner_stressed, support_occurred):
- (T,T,T) -> both resources +boost
- (T,T,F) -> both resources -cost
- (T,F,T) -> self +boost, partner resources unchanged
- (T,F,F) -> self -cost, partner resources unchanged
- (F,F,_) -> both PF social_support +small_boost, resources unchanged
"""

from typing import Any, Dict, Tuple

from numpy.random import Generator

from src.python.phases.interfaces import AgentState, PhaseOutput, PhaseFrequency
from src.python.math_utils import clamp

PHASE_FREQUENCY: PhaseFrequency = "event_driven"

# ──────────────────────────────────────────────
# Default configuration constants
# ──────────────────────────────────────────────

_DEFAULT_INFLUENCE_RATE = 0.05
_DEFAULT_RESILIENCE_INFLUENCE = 0.05
_DEFAULT_SUPPORT_THRESHOLD = 0.05
_DEFAULT_BOOST = 0.10
_DEFAULT_COST = 0.05
_DEFAULT_SMALL_BOOST = 0.02


# ──────────────────────────────────────────────
# Core interaction logic
# ──────────────────────────────────────────────


def process_interaction(
    self_state: AgentState,
    partner_state: AgentState,
    config: Dict[str, Any],
    rng: Generator,
) -> Tuple[PhaseOutput, PhaseOutput]:
    """Process a dyadic interaction between two agents.

    Converges affect and resilience, detects support, and applies a
    win-win/lose-lose resource exchange based on both agents' stress
    state and whether support occurred during the interaction.

    Args:
        self_state: Current state of the calling agent.
        partner_state: Current state of the partner agent.
        config: Phase configuration with keys:
            influence_rate, resilience_influence, support_threshold,
            boost, cost, small_boost.
        rng: Seeded random number generator (reserved for future stochasticity).

    Returns:
        Tuple of (self_phase_output, partner_phase_output), each with:
            state_delta: affect, resilience, resources, and optionally
                protective_factors changes.
            observation: support_occurred flag.
    """
    # ── 0. Read config ─────────────────────────────────────────────
    influence_rate = config.get("influence_rate", _DEFAULT_INFLUENCE_RATE)
    resilience_influence = config.get("resilience_influence", _DEFAULT_RESILIENCE_INFLUENCE)
    support_threshold = config.get("support_threshold", _DEFAULT_SUPPORT_THRESHOLD)
    boost = config.get("boost", _DEFAULT_BOOST)
    cost = config.get("cost", _DEFAULT_COST)
    small_boost = config.get("small_boost", _DEFAULT_SMALL_BOOST)

    # ── 1. Read inputs ─────────────────────────────────────────────
    self_affect = self_state.get("affect", 0.0)
    partner_affect = partner_state.get("affect", 0.0)
    self_resilience = self_state.get("resilience", 0.5)
    partner_resilience = partner_state.get("resilience", 0.5)
    self_stressed = self_state.get("stressed", False)
    partner_stressed = partner_state.get("stressed", False)

    # ── 2. Affect convergence (with negativity bias) ───────────────
    raw_self_affect_change = influence_rate * partner_affect
    raw_partner_affect_change = influence_rate * self_affect

    # Negativity bias: negative influence is 1.5x stronger
    if raw_self_affect_change < 0:
        raw_self_affect_change *= 1.5
    if raw_partner_affect_change < 0:
        raw_partner_affect_change *= 1.5

    new_self_affect = clamp(self_affect + raw_self_affect_change, -1.0, 1.0)
    new_partner_affect = clamp(partner_affect + raw_partner_affect_change, -1.0, 1.0)

    self_affect_change = new_self_affect - self_affect
    partner_affect_change = new_partner_affect - partner_affect

    # ── 3. Resilience convergence ───────────────────────────────────
    self_resilience_change = resilience_influence * partner_affect
    partner_resilience_change = resilience_influence * self_affect

    new_self_resilience = clamp(self_resilience + self_resilience_change, 0.0, 1.0)
    new_partner_resilience = clamp(partner_resilience + partner_resilience_change, 0.0, 1.0)

    # Recompute clamped changes
    self_resilience_change = new_self_resilience - self_resilience
    partner_resilience_change = new_partner_resilience - partner_resilience

    # ── 4. Support detection from convergence magnitude ─────────────
    total_convergence = (
        abs(self_affect_change)
        + abs(partner_affect_change)
        + abs(self_resilience_change)
        + abs(partner_resilience_change)
    )
    support_occurred = total_convergence > support_threshold

    # ── 5. Resource exchange state machine ─────────────────────────
    self_resource_change: float = 0.0
    partner_resource_change: float = 0.0
    self_pf_delta: Dict[str, float] = {}
    partner_pf_delta: Dict[str, float] = {}

    if self_stressed and partner_stressed:
        # Both stressed
        if support_occurred:
            self_resource_change = boost
            partner_resource_change = boost
        else:
            self_resource_change = -cost
            partner_resource_change = -cost
    elif self_stressed and not partner_stressed:
        # Only self stressed
        if support_occurred:
            self_resource_change = boost
            # partner unchanged
        else:
            self_resource_change = -cost
            # partner unchanged
    elif not self_stressed and not partner_stressed:
        # Neither stressed: PF social_support boost, resources unchanged
        self_pf_delta = {"social_support": small_boost}
        partner_pf_delta = {"social_support": small_boost}
    # else: self not stressed, partner stressed — no resource action
    # (only self-stressed cases get resource changes; partner-stressed
    # alone is handled when that agent is 'self' in their own call)

    # ── 6. Build deltas ────────────────────────────────────────────
    self_delta: Dict[str, Any] = {
        "affect": self_affect_change,
        "resilience": self_resilience_change,
    }
    partner_delta: Dict[str, Any] = {
        "affect": partner_affect_change,
        "resilience": partner_resilience_change,
    }

    # Add resource changes (only if non-zero or key needed)
    if self_resource_change != 0.0:
        self_delta["resources"] = self_resource_change
    if partner_resource_change != 0.0:
        partner_delta["resources"] = partner_resource_change

    # Add PF deltas (only in non-stressed-non-stressed scenario)
    if self_pf_delta:
        self_delta["protective_factors"] = self_pf_delta
    if partner_pf_delta:
        partner_delta["protective_factors"] = partner_pf_delta

    # Shared observation
    observation: Dict[str, Any] = {"support_occurred": support_occurred}

    return (
        PhaseOutput(state_delta=self_delta, observation=observation),
        PhaseOutput(state_delta=partner_delta, observation=observation),
    )


# ──────────────────────────────────────────────
# Phase protocol wrapper (event-driven)
# ──────────────────────────────────────────────


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
