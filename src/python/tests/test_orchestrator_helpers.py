"""Tests for Person orchestrator helpers: _build_agent_state, _apply_delta, _write_back_state.

These helpers manage the state dict that flows through the phase pipeline.
"""

import pytest
from unittest.mock import Mock

from src.python.agent import Person


# ─── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def mock_model():
    """Create a minimal mock model for Person instantiation."""
    model = Mock()
    model.seed = 42
    model.agents = []
    model.register_agent = Mock()
    model.grid = Mock()
    model.grid.get_neighbors.return_value = []
    model.rng = Mock()
    model.rng.beta.return_value = 0.5
    return model


@pytest.fixture
def agent(mock_model):
    """A fully initialized Person agent."""
    return Person(mock_model)


# ─── _build_agent_state tests ────────────────────────────────────


class TestBuildAgentState:
    """_build_agent_state returns a complete AgentState dict from self.*."""

    def test_returns_agentstate_type(self, agent):
        """Result is an AgentState-compatible dict."""
        state = agent._build_agent_state()
        assert isinstance(state, dict)

    def test_contains_core_state(self, agent):
        """Core state variables are present."""
        state = agent._build_agent_state()
        assert "resilience" in state
        assert "affect" in state
        assert "resources" in state
        assert "baseline_resilience" in state
        assert "baseline_affect" in state

    def test_contains_tracking_vars(self, agent):
        """Stress tracking variables are present."""
        state = agent._build_agent_state()
        assert "current_stress" in state
        assert "consecutive_hindrances" in state
        assert "volatility" in state

    def test_contains_pss10_vars(self, agent):
        """PSS-10 state variables are present."""
        state = agent._build_agent_state()
        assert "pss10" in state
        assert "stressed" in state
        assert "pss10_responses" in state
        assert "stress_controllability" in state
        assert "stress_overload" in state

    def test_contains_daily_counters(self, agent):
        """Daily interaction counters are present."""
        state = agent._build_agent_state()
        assert "daily_interactions" in state
        assert "daily_support_exchanges" in state

    def test_values_match_self(self, agent):
        """State values match the agent's attributes."""
        state = agent._build_agent_state()
        assert state["resilience"] == agent.resilience
        assert state["affect"] == agent.affect
        assert state["resources"] == agent.resources
        assert state["current_stress"] == agent.current_stress
        assert state["pss10"] == agent.pss10

    def test_protective_factors_copy(self, agent):
        """state['protective_factors'] is a copy, not a reference."""
        original = agent.protective_factors
        state = agent._build_agent_state()
        state["protective_factors"]["social_support"] = 999.0
        # Original should be unchanged
        assert agent.protective_factors["social_support"] == original["social_support"]

    def test_contains_daily_stress_events(self, agent):
        """daily_stress_events list is present."""
        state = agent._build_agent_state()
        assert "daily_stress_events" in state
        assert isinstance(state["daily_stress_events"], list)

    def test_contains_protective_factors(self, agent):
        """protective_factors dict is present."""
        state = agent._build_agent_state()
        assert "protective_factors" in state
        assert isinstance(state["protective_factors"], dict)
        assert "social_support" in state["protective_factors"]

    def test_contains_configs(self, agent):
        """stress_config and interaction_config are present."""
        state = agent._build_agent_state()
        assert "stress_config" in state
        assert "interaction_config" in state


# ─── _apply_delta tests ──────────────────────────────────────────


class TestApplyDelta:
    """_apply_delta merges a state_delta into an AgentState dict."""

    def test_returns_updated_state(self, agent):
        """Returns a new state dict (no mutation of input)."""
        state = agent._build_agent_state()
        delta = {"resilience": 0.75}
        original_resilience = state["resilience"]
        new_state = agent._apply_delta(state, delta)
        assert new_state["resilience"] == 0.75
        # Original should be unchanged
        assert state["resilience"] == original_resilience

    def test_does_not_mutate_input_state(self, agent):
        """The input state dict is not mutated."""
        state = agent._build_agent_state()
        resilience_before = state["resilience"]
        delta = {"resilience": 0.99}
        agent._apply_delta(state, delta)
        assert state["resilience"] == resilience_before

    def test_does_not_mutate_input_delta(self, agent):
        """The input delta dict is not mutated."""
        state = agent._build_agent_state()
        delta = {"resilience": 0.99, "affect": 0.5}
        original_delta = dict(delta)
        agent._apply_delta(state, delta)
        assert delta == original_delta

    def test_merges_multiple_keys(self, agent):
        """Multiple keys in delta are all applied."""
        state = agent._build_agent_state()
        delta = {"resilience": 0.25, "affect": 0.75, "resources": 0.9}
        new_state = agent._apply_delta(state, delta)
        assert new_state["resilience"] == 0.25
        assert new_state["affect"] == 0.75
        assert new_state["resources"] == 0.9

    def test_preserves_keys_not_in_delta(self, agent):
        """Keys not in delta are preserved from original state."""
        state = agent._build_agent_state()
        original_volatility = state["volatility"]
        delta = {"resilience": 0.5}
        new_state = agent._apply_delta(state, delta)
        assert new_state["volatility"] == original_volatility

    def test_merges_protective_factors_dict(self, agent):
        """protective_factors in delta is merged, not replaced."""
        state = agent._build_agent_state()
        original_pf = dict(state["protective_factors"])
        delta = {"protective_factors": {"social_support": 0.95}}
        new_state = agent._apply_delta(state, delta)
        # social_support should be updated
        assert new_state["protective_factors"]["social_support"] == 0.95
        # Other factors unchanged
        for key in ["family_support", "formal_intervention", "psychological_capital"]:
            assert new_state["protective_factors"][key] == original_pf[key]

    def test_empty_delta_returns_copy(self, agent):
        """Empty delta returns a copy of the original state."""
        state = agent._build_agent_state()
        new_state = agent._apply_delta(state, {})
        assert new_state is not state
        assert new_state == state

    def test_handles_sequential_deltas(self, agent):
        """Sequential apply_delta calls accumulate correctly."""
        state = agent._build_agent_state()
        state = agent._apply_delta(state, {"resilience": 0.3})
        state = agent._apply_delta(state, {"affect": -0.5})
        state = agent._apply_delta(state, {"resilience": 0.8})
        assert state["resilience"] == 0.8
        assert state["affect"] == -0.5

    def test_handles_partial_protective_factors(self, agent):
        """Partial protective_factors delta updates only specified keys."""
        state = agent._build_agent_state()
        delta = {"protective_factors": {"family_support": 0.1}}
        new_state = agent._apply_delta(state, delta)
        assert new_state["protective_factors"]["family_support"] == 0.1
        # social_support should be unchanged from original
        assert new_state["protective_factors"]["social_support"] == state["protective_factors"]["social_support"]

    def test_scalar_float_keys_update_directly(self, agent):
        """Simple float keys like affect, resilience are replaced, not merged."""
        state = agent._build_agent_state()
        delta = {"affect": 0.9}
        new_state = agent._apply_delta(state, delta)
        assert new_state["affect"] == 0.9


# ─── _write_back_state tests ─────────────────────────────────────


class TestWriteBackState:
    """_write_back_state writes AgentState values back to self.*."""

    def test_writes_resilience(self, agent):
        """resilience is written back to self.resilience."""
        state = agent._build_agent_state()
        state["resilience"] = 0.123
        agent._write_back_state(state)
        assert agent.resilience == 0.123

    def test_writes_affect(self, agent):
        """affect is written back to self.affect."""
        state = agent._build_agent_state()
        state["affect"] = 0.456
        agent._write_back_state(state)
        assert agent.affect == 0.456

    def test_writes_current_stress(self, agent):
        """current_stress is written back."""
        state = agent._build_agent_state()
        state["current_stress"] = 0.789
        agent._write_back_state(state)
        assert agent.current_stress == 0.789

    def test_writes_pss10(self, agent):
        """pss10 is written back."""
        state = agent._build_agent_state()
        state["pss10"] = 25
        agent._write_back_state(state)
        assert agent.pss10 == 25

    def test_writes_stressed(self, agent):
        """stressed is written back."""
        state = agent._build_agent_state()
        state["stressed"] = True
        agent._write_back_state(state)
        assert agent.stressed is True

    def test_writes_protective_factors(self, agent):
        """protective_factors dict is written back."""
        state = agent._build_agent_state()
        state["protective_factors"]["social_support"] = 0.88
        state["protective_factors"]["family_support"] = 0.77
        agent._write_back_state(state)
        assert agent.protective_factors["social_support"] == 0.88
        assert agent.protective_factors["family_support"] == 0.77

    def test_writes_daily_counters(self, agent):
        """daily_interactions and daily_support_exchanges are written back."""
        state = agent._build_agent_state()
        state["daily_interactions"] = 5
        state["daily_support_exchanges"] = 3
        agent._write_back_state(state)
        assert agent.daily_interactions == 5
        assert agent.daily_support_exchanges == 3

    def test_writes_stress_controllability(self, agent):
        """stress_controllability is written back."""
        state = agent._build_agent_state()
        state["stress_controllability"] = 0.3
        agent._write_back_state(state)
        assert agent.stress_controllability == 0.3

    def test_writes_stress_overload(self, agent):
        """stress_overload is written back."""
        state = agent._build_agent_state()
        state["stress_overload"] = 0.7
        agent._write_back_state(state)
        assert agent.stress_overload == 0.7

    def test_writes_consecutive_hindrances(self, agent):
        """consecutive_hindrances is written back."""
        state = agent._build_agent_state()
        state["consecutive_hindrances"] = 3.0
        agent._write_back_state(state)
        assert agent.consecutive_hindrances == 3.0

    def test_writes_stress_breach_count(self, agent):
        """stress_breach_count is written back."""
        state = agent._build_agent_state()
        state["stress_breach_count"] = 5
        agent._write_back_state(state)
        assert agent.stress_breach_count == 5

    def test_writes_adapted_network(self, agent):
        """adapted_network is written back to self.adapted_network."""
        state = agent._build_agent_state()
        state["adapted_network"] = True
        agent._write_back_state(state)
        assert agent.adapted_network is True
        state["adapted_network"] = False
        agent._write_back_state(state)
        assert agent.adapted_network is False

    def test_ignores_transient_keys(self, agent):
        """Transient keys like challenge, hindrance, is_stressed are NOT written."""
        state = agent._build_agent_state()
        state["challenge"] = 0.5
        state["hindrance"] = 0.3
        state["is_stressed"] = True
        state["event_controllability"] = 0.6
        state["event_overload"] = 0.4
        # Should not raise AttributeError for non-existent self attributes
        agent._write_back_state(state)
        # Verify core state is still correct
        assert agent.resilience == state.get("resilience", agent.resilience)

    def test_idempotent(self, agent):
        """Build → write → build produces the same state (round-trip)."""
        state1 = agent._build_agent_state()
        agent._write_back_state(state1)
        state2 = agent._build_agent_state()
        # pss10_responses might have RNG differences, compare without it
        assert state1["resilience"] == state2["resilience"]
        assert state1["affect"] == state2["affect"]
        assert state1["resources"] == state2["resources"]
        assert state1["current_stress"] == state2["current_stress"]
        assert state1["pss10"] == state2["pss10"]
        assert state1["adapted_network"] == state2["adapted_network"]
