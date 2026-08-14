"""Tests for small-N StressModel construction.

Verifies that StressModel can be constructed with N < default watts_k (4),
which previously crashed with NetworkXError "k>n".
"""

from __future__ import annotations

from src.python.model import StressModel


class TestSmallPopulationConstruction:
    """StressModel handles N < default watts_k gracefully."""

    def test_n_1_constructs_without_error(self):
        """N=1 model builds a single-node graph without NetworkXError."""
        model = StressModel(N=1, max_days=1, seed=42)
        # The grid should have exactly 1 node
        assert model.grid is not None
        assert len(list(model.grid.G.nodes())) == 1

    def test_n_1_graph_has_zero_edges(self):
        """N=1 model has 0 edges (no social connections for a single agent)."""
        model = StressModel(N=1, max_days=1, seed=42)
        assert model.grid.G.number_of_edges() == 0

    def test_n_2_constructs_without_error(self):
        """N=2 model builds without NetworkXError."""
        model = StressModel(N=2, max_days=1, seed=42)
        assert len(list(model.grid.G.nodes())) == 2

    def test_n_3_constructs_without_error(self):
        """N=3 model builds without NetworkXError (k clamped to 2)."""
        model = StressModel(N=3, max_days=1, seed=42)
        assert len(list(model.grid.G.nodes())) == 3
        # For n=3, k=2 → ring of 3 nodes, edge count = n*k/2 = 3
        assert model.grid.G.number_of_edges() == 3

    def test_n_5_still_uses_default_k(self):
        """N=5 model still gets full k=4 Watts-Strogatz (no clamping)."""
        model = StressModel(N=5, max_days=1, seed=42)
        # n=5, k=4 → each node connects to 4 neighbors = complete graph
        # Complete graph K5 has n*(n-1)/2 = 10 edges
        assert model.grid.G.number_of_edges() == 10

    def test_n_1_can_step(self):
        """N=1 model runs a full step without errors."""
        model = StressModel(N=1, max_days=3, seed=42)
        model.step()
        agent = list(model.agents)[0]
        # After one step, agent state should be updated
        assert agent.affect is not None
        assert agent.current_stress is not None

    def test_n_2_can_step(self):
        """N=2 model runs a full step without errors (agents have no edges but step is fine)."""
        model = StressModel(N=2, max_days=3, seed=42)
        model.step()
        for agent in model.agents:
            assert agent.affect is not None
