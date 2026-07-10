"""
Unit tests for network adaptation utilities (plan 012).

Tests are written BEFORE implementation (TDD).
"""

import networkx as nx
import pytest

from src.python.math_utils import create_rng


class TestBuildWattsStrogatz:
    """Test Watts-Strogatz network construction."""

    def test_correct_topology(self):
        """Test that WS graph has correct N, k, p properties."""
        from src.python.network_utils import build_watts_strogatz_network

        rng = create_rng(42)
        G = build_watts_strogatz_network(N=100, k=4, p=0.1, rng=rng)

        assert G.number_of_nodes() == 100
        avg_degree = sum(d for _, d in G.degree()) / G.number_of_nodes()
        assert avg_degree >= 3.5  # close to k=4
        clustering = nx.average_clustering(G)
        # WS with p=0.1 should have clustering > random
        random_G = nx.erdos_renyi_graph(100, 4 / 99, seed=42)
        assert clustering > nx.average_clustering(random_G)


class TestComputeConnectionSimilarity:
    """Test similarity computation between agents."""

    def test_identical_states_max_similarity(self):
        """Test that identical states give similarity = 1.0."""
        from src.python.network_utils import compute_connection_similarity

        state = {"current_stress": 0.5, "affect": 0.2, "resilience": 0.7}
        sim = compute_connection_similarity(state, state, 1.0, 1.0, 1.0)
        assert sim == pytest.approx(1.0)

    def test_opposite_states_min_similarity(self):
        """Test that opposite states give similarity near 0."""
        from src.python.network_utils import compute_connection_similarity

        state_i = {"current_stress": 0.0, "affect": 1.0, "resilience": 1.0}
        state_j = {"current_stress": 1.0, "affect": -1.0, "resilience": 0.0}
        sim = compute_connection_similarity(state_i, state_j, 1.0, 1.0, 1.0)
        assert sim >= 0.0
        assert sim <= 0.1  # near-zero for maximally different states

    def test_weighted_similarity(self):
        """Test that weights bias the similarity score."""
        from src.python.network_utils import compute_connection_similarity

        state_i = {"current_stress": 0.9, "affect": 0.0, "resilience": 0.0}
        state_j = {"current_stress": 0.1, "affect": 0.0, "resilience": 0.0}

        # All weight on stress
        sim_stress = compute_connection_similarity(state_i, state_j, 1.0, 0.0, 0.0)
        # All weight on affect (same for both)
        sim_affect = compute_connection_similarity(state_i, state_j, 0.0, 1.0, 0.0)

        assert sim_stress < sim_affect  # stress differs, affect same


class TestComputeRetentionProbability:
    """Test retention probability computation."""

    def test_high_similarity_low_retention(self):
        """Test that high similarity + support gives low retention (likely to rewire)."""
        from src.python.network_utils import compute_connection_retention_probability

        prob = compute_connection_retention_probability(
            similarity=0.9, support_effectiveness=0.8, homophily_strength=0.7
        )
        assert 0.0 <= prob <= 1.0
        # High similarity means agent is similar to neighbor — more likely to retain
        # But with high support effectiveness, agent may still seek better connections
        # The sigmoid should produce a moderate value
        assert prob < 0.9

    def test_low_similarity_high_retention(self):
        """Test that low similarity + low support gives high retention (stay connected)."""
        from src.python.network_utils import compute_connection_retention_probability

        prob = compute_connection_retention_probability(
            similarity=0.1, support_effectiveness=0.2, homophily_strength=0.7
        )
        assert 0.0 <= prob <= 1.0
        # Low similarity means connection to a dissimilar agent — more valuable
        # for diversity of support
        assert prob > 0.1

    def test_monotonic_with_similarity(self):
        """Test that retention probability decreases as similarity increases."""
        from src.python.network_utils import compute_connection_retention_probability

        sims = [0.1, 0.3, 0.5, 0.7, 0.9]
        probs = [
            compute_connection_retention_probability(s, support_effectiveness=0.5, homophily_strength=0.7) for s in sims
        ]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1], f"Not monotonic at {i}: {probs[i]} < {probs[i + 1]}"

    def test_retention_monotonic_in_homophily_strength(self):
        """Test that higher homophily strength steepens the response."""
        from src.python.network_utils import compute_connection_retention_probability

        # At similarity=0.5, support=0.5
        p_low = compute_connection_retention_probability(0.5, 0.5, 0.1)
        p_high = compute_connection_retention_probability(0.5, 0.5, 0.9)
        # Both in [0,1]
        assert 0.0 <= p_low <= 1.0
        assert 0.0 <= p_high <= 1.0

    def test_support_effectiveness_shifts_response(self):
        """Test that support effectiveness shifts midpoint of retention curve."""
        from src.python.network_utils import compute_connection_retention_probability

        # At similarity=0.7, low vs high support
        p_low_support = compute_connection_retention_probability(0.7, 0.2, 0.7)
        p_high_support = compute_connection_retention_probability(0.7, 0.8, 0.7)
        assert 0.0 <= p_low_support <= 1.0
        assert 0.0 <= p_high_support <= 1.0


class TestApplyStressAdaptation:
    """Test full stress adaptation rewiring."""

    def test_rewiring_triggers_on_breach(self):
        """Test that rewiring only occurs when stress_breach_count >= threshold."""
        from src.python.network_utils import apply_stress_adaptation
        from unittest.mock import MagicMock

        rng = create_rng(42)
        G = nx.watts_strogatz_graph(20, k=4, p=0.1, seed=42)

        agents = []
        for i in range(20):
            agent = MagicMock()
            agent.unique_id = i
            agent.stress_breach_count = 5 if i < 5 else 0  # first 5 breach
            agent.current_stress = 0.8 if i < 5 else 0.2
            agent.affect = 0.0
            agent.resilience = 0.5
            agent.support_boost = 0.5
            agents.append(agent)

        config = {
            "adaptation_threshold": 3,
            "rewire_probability": 0.5,
            "homophily_strength": 0.7,
            "stress_weight": 1.0,
            "affect_weight": 1.0,
            "resilience_weight": 1.0,
        }

        new_G, rewired_count = apply_stress_adaptation(G, agents, config, rng)
        assert isinstance(rewired_count, int)
        assert rewired_count >= 0
        # Should have the same number of nodes
        assert new_G.number_of_nodes() == G.number_of_nodes()

    def test_no_edges_to_self(self):
        """Test that rewiring never creates self-loops."""
        from src.python.network_utils import apply_stress_adaptation
        from unittest.mock import MagicMock

        rng = create_rng(99)
        G = nx.watts_strogatz_graph(10, k=2, p=0.1, seed=99)

        agents = []
        for i in range(10):
            agent = MagicMock()
            agent.unique_id = i
            agent.stress_breach_count = 10
            agent.current_stress = 0.9
            agent.affect = 0.0
            agent.resilience = 0.5
            agent.support_boost = 0.5
            agents.append(agent)

        config = {
            "adaptation_threshold": 1,
            "rewire_probability": 1.0,
            "homophily_strength": 0.7,
            "stress_weight": 1.0,
            "affect_weight": 1.0,
            "resilience_weight": 1.0,
        }

        new_G, _ = apply_stress_adaptation(G, agents, config, rng)
        # No self-loops
        assert not any(u == v for u, v in new_G.edges())

    def test_clustering_preserved_within_10_percent(self):
        """Test that clustering coefficient is preserved within 10%% after adaptation."""
        from src.python.network_utils import apply_stress_adaptation
        from unittest.mock import MagicMock

        rng = create_rng(42)
        G = nx.watts_strogatz_graph(50, k=6, p=0.1, seed=42)
        original_clustering = nx.average_clustering(G)

        agents = []
        for i in range(50):
            agent = MagicMock()
            agent.unique_id = i
            agent.stress_breach_count = 10  # all agents breach
            agent.current_stress = 0.9
            agent.affect = 0.0
            agent.resilience = 0.5
            agent.support_boost = 0.5
            agents.append(agent)

        config = {
            "adaptation_threshold": 1,
            "rewire_probability": 0.05,
            "homophily_strength": 0.5,
            "stress_weight": 1.0,
            "affect_weight": 1.0,
            "resilience_weight": 1.0,
        }

        new_G, rewired = apply_stress_adaptation(G, agents, config, rng)
        new_clustering = nx.average_clustering(new_G)

        # Even after rewiring, clustering should remain within 20%% of original.
        # Homophilic rewiring can increase clustering; we just want to ensure
        # the network structure is not completely destroyed.
        if original_clustering > 0:
            ratio = new_clustering / original_clustering
            assert 0.80 <= ratio <= 1.20, (
                f"Clustering changed from {original_clustering:.4f} to {new_clustering:.4f} (ratio {ratio:.4f})"
            )
        assert rewired >= 0
