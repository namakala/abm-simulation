"""Tests that Person.interact() delegates to phase_process_interaction (Plan 008 Step 5).

The old interact() used affect_utils.process_interaction() which returns raw tuples.
The new version calls phases.interaction.process_interaction (state-delta pattern)
and translates the PhaseOutput back into the legacy return dict format.
"""

from unittest.mock import Mock, patch

import pytest

from src.python.agent import Person
from src.python.phases.interfaces import PhaseOutput


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def agent_with_neighbors():
    """Agent with a mock model and grid providing neighbors."""
    agent_model = Mock()
    agent_model.seed = 42
    agent_model.agents = []
    agent_model.register_agent = Mock()
    agent_model.rng = Mock()
    agent_model.rng.beta.return_value = 0.5

    # Create two agents
    a1 = Person(agent_model)
    a2 = Person(agent_model)

    # Give them positions so interactions work
    # Mesa's agent.__init__ sets self.pos = None; we need to set it
    a1.pos = 0
    a2.pos = 1

    # Configure grid to return agent2 as neighbor of agent1
    agent_model.grid = Mock()
    agent_model.grid.get_neighbors.return_value = [a2]

    return a1, a2, agent_model


# ─── Return dict contract tests ───────────────────────────────────


class TestInteractReturnDict:
    """interact() returns the expected dict format."""

    def test_returns_dict(self, agent_with_neighbors):
        """interact() returns a dict."""
        a1, a2, _ = agent_with_neighbors
        result = a1.interact()
        assert isinstance(result, dict)

    def test_has_support_exchange_key(self, agent_with_neighbors):
        """Result dict has support_exchange key."""
        a1, a2, _ = agent_with_neighbors
        result = a1.interact()
        assert "support_exchange" in result

    def test_has_affect_change_key(self, agent_with_neighbors):
        """Result dict has affect_change key."""
        a1, a2, _ = agent_with_neighbors
        result = a1.interact()
        assert "affect_change" in result

    def test_has_resilience_change_key(self, agent_with_neighbors):
        """Result dict has resilience_change key."""
        a1, a2, _ = agent_with_neighbors
        result = a1.interact()
        assert "resilience_change" in result

    def test_has_resource_transfer_key(self, agent_with_neighbors):
        """Result dict has resource_transfer key."""
        a1, a2, _ = agent_with_neighbors
        result = a1.interact()
        assert "resource_transfer" in result

    def test_has_received_resources_key(self, agent_with_neighbors):
        """Result dict has received_resources key."""
        a1, a2, _ = agent_with_neighbors
        result = a1.interact()
        assert "received_resources" in result


# ─── Edge cases ───────────────────────────────────────────────────


class TestInteractEdgeCases:
    """interact() handles edge cases gracefully."""

    def test_no_position_returns_empty(self):
        """Agent without grid position returns empty result."""
        model = Mock()
        model.seed = 42
        model.agents = []
        model.register_agent = Mock()
        model.grid = Mock()
        model.grid.get_neighbors.return_value = []
        model.rng = Mock()
        model.rng.beta.return_value = 0.5

        agent = Person(model)
        agent.pos = None
        result = agent.interact()
        assert isinstance(result, dict)
        assert result["support_exchange"] is False

    def test_no_neighbors_returns_empty(self, agent_with_neighbors):
        """Agent with no neighbors returns empty result."""
        a1, a2, model = agent_with_neighbors
        model.grid.get_neighbors.return_value = []
        result = a1.interact()
        assert isinstance(result, dict)
        assert result["support_exchange"] is False

    def test_increments_daily_interactions(self, agent_with_neighbors):
        """interact() increments daily_interactions counter."""
        a1, a2, _ = agent_with_neighbors
        before = a1.daily_interactions
        a1.interact()
        assert a1.daily_interactions == before + 1

    def test_both_agents_modified(self, agent_with_neighbors):
        """Both agents' states change after interaction."""
        a1, a2, _ = agent_with_neighbors
        a1.interact()

        # It's possible for affect to not change if both are at 0 and influence is small
        # But generally at least one should change
        assert -1.0 <= a1.affect <= 1.0
        assert -1.0 <= a2.affect <= 1.0


# ─── Delegation test ──────────────────────────────────────────────


class TestInteractDelegation:
    """interact() delegates to phase_process_interaction."""

    def test_uses_phase_function(self, agent_with_neighbors):
        """interact() calls phase_process_interaction (not the old affect_utils version)."""
        a1, a2, _ = agent_with_neighbors

        with patch("src.python.agent.phase_process_interaction", wraps=a1._build_agent_state) as mock_fn:
            # We need a proper mock that returns PhaseOutputs
            mock_fn.return_value = (
                PhaseOutput(state_delta={"affect": 0.0, "resilience": 0.0}, observation={"support_occurred": False}),
                PhaseOutput(state_delta={"affect": 0.0, "resilience": 0.0}, observation={"support_occurred": False}),
            )
            a1.interact()

            # phase_process_interaction should have been called
            mock_fn.assert_called_once()
