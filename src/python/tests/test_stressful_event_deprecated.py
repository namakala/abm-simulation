"""Tests that Person.stressful_event() is a thin delegation wrapper (Plan 008 Step 4).

The method now delegates to the phase pipeline (run_stress_perception +
run_resilience_activation) and returns the same (challenge, hindrance) tuple.
"""

from unittest.mock import Mock

import pytest

from src.python.agent import Person


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def agent():
    """A Person agent with minimal mock model."""
    model = Mock()
    model.seed = 42
    model.agents = []
    model.register_agent = Mock()
    model.grid = Mock()
    model.grid.get_neighbors.return_value = []
    model.rng = Mock()
    model.rng.beta.return_value = 0.5
    return Person(model)


# ─── Return contract tests ────────────────────────────────────────


class TestStressfulEventContract:
    """stressful_event() returns the expected tuple format."""

    def test_returns_tuple_of_two_floats(self, agent):
        """stressful_event() returns a (challenge, hindrance) tuple."""
        result = agent.stressful_event()
        assert isinstance(result, tuple)
        assert len(result) == 2
        ch, hi = result
        assert isinstance(ch, float)
        assert isinstance(hi, float)

    def test_challenge_hindrance_in_bounds(self, agent):
        """challenge and hindrance are in [0, 1]."""
        ch, hi = agent.stressful_event()
        assert 0.0 <= ch <= 1.0
        assert 0.0 <= hi <= 1.0


# ─── Delegation tests ─────────────────────────────────────────────


class TestStressfulEventDelegation:
    """stressful_event() delegates to phase functions."""

    def test_updates_pss10(self, agent):
        """stressful_event() updates PSS-10 state (via delegation)."""
        agent.stressful_event()
        # PSS-10 should be in valid range
        assert 0 <= agent.pss10 <= 40

    def test_updates_daily_pss10_scores(self, agent):
        """stressful_event() appends to daily_pss10_scores."""
        before = len(agent.daily_pss10_scores)
        agent.stressful_event()
        if agent.pss10 > 0:
            assert len(agent.daily_pss10_scores) > before

    def test_updates_current_stress(self, agent):
        """current_stress may change after stressful_event."""
        agent.stressful_event()
        assert 0.0 <= agent.current_stress <= 1.0

    def test_increments_stress_breach_count_on_stress(self, agent):
        """stress_breach_count may increase after a stressful event."""
        before = agent.stress_breach_count
        agent.stressful_event()
        assert agent.stress_breach_count >= before

    def test_deterministic_with_seed(self):
        """Same agent same seed -> same result."""

        def make_agent():
            m = Mock()
            m.seed = 42
            m.agents = []
            m.register_agent = Mock()
            m.grid = Mock()
            m.grid.get_neighbors.return_value = []
            m.rng = Mock()
            m.rng.beta.return_value = 0.5
            return Person(m)

        a1 = make_agent()
        a2 = make_agent()
        r1 = a1.stressful_event()
        r2 = a2.stressful_event()
        assert r1 == r2
