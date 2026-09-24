"""
Network utilities for the ABM stress simulation.

Provides graph construction, homophily-based similarity scoring,
connection retention probability, and stress-driven network adaptation.
"""

from typing import Any, Dict, List, Tuple

import networkx as nx
import math

import numpy as np


def build_watts_strogatz_network(N: int, k: int, p: float, rng: np.random.Generator) -> nx.Graph:
    """Construct a Watts-Strogatz small-world network.

    Args:
        N: Number of nodes.
        k: Each node is connected to k nearest neighbors (must be < N).
        p: Rewiring probability.
        rng: Seeded random number generator.

    Returns:
        A NetworkX Graph with N nodes.
    """
    k = min(k, max(0, N - 1))
    return nx.watts_strogatz_graph(n=N, k=k, p=p, seed=rng)


def compute_connection_similarity(
    state_i: Dict[str, Any],
    state_j: Dict[str, Any],
    stress_weight: float = 1.0,
    affect_weight: float = 1.0,
    resilience_weight: float = 1.0,
) -> float:
    """Compute weighted similarity between two agent states.

    Uses current_stress, affect, and resilience as homophily dimensions.
    Each dimension contributes a normalised similarity in [0, 1], then
    weighted by the respective weight and averaged.

    Args:
        state_i: First agent's state dict (keys: current_stress, affect, resilience).
        state_j: Second agent's state dict (keys: current_stress, affect, resilience).
        stress_weight: Weight for stress similarity.
        affect_weight: Weight for affect similarity.
        resilience_weight: Weight for resilience similarity.

    Returns:
        Weighted similarity in [0, 1] where 1.0 = identical.
    """
    stress_sim = 1.0 - abs(state_i.get("current_stress", 0.0) - state_j.get("current_stress", 0.0))
    affect_sim = 1.0 - abs(state_i.get("affect", 0.0) - state_j.get("affect", 0.0))
    affect_sim = max(0.0, min(1.0, affect_sim))  # clamp affect diff to [0,1]

    resilience_sim = 1.0 - abs(state_i.get("resilience", 0.5) - state_j.get("resilience", 0.5))

    total_weight = stress_weight + affect_weight + resilience_weight
    if total_weight == 0.0:
        return 0.5  # neutral if no weights

    weighted_sum = stress_weight * stress_sim + affect_weight * affect_sim + resilience_weight * resilience_sim
    return weighted_sum / total_weight


def compute_connection_retention_probability(
    similarity: float,
    support_effectiveness: float,
    homophily_strength: float,
) -> float:
    """Compute probability that an agent retains a connection.

    Uses a sigmoid-based function where higher similarity with current neighbor
    and higher support effectiveness make retention less likely (the agent
    is more willing to rewire toward even more similar agents).

    Args:
        similarity: Current connection similarity in [0, 1].
        support_effectiveness: Recent support quality in [0, 1].
        homophily_strength: Population homophily bias in [0, 1].

    Returns:
        Retention probability in [0, 1].
    """
    # Steepness and midpoint of the sigmoid.
    # Higher similarity -> lower retention (seek more similar peers).
    steepness = 3.0 * homophily_strength
    midpoint = 0.5 + 0.3 * (1.0 - support_effectiveness)

    # Retention decreases with similarity: sigmoid(-steepness * (sim - mid))
    raw = 1.0 / (1.0 + math.exp(steepness * (similarity - midpoint)))
    return float(raw)


def _find_rewire_candidate(
    G: nx.Graph,
    agent_idx: int,
    agents: List,
    rng: np.random.Generator,
    stress_weight: float,
    affect_weight: float,
    resilience_weight: float,
) -> int:
    """Find a non-neighbor node more similar to the agent.

    Searches all nodes not currently connected to agent_idx,
    picks the one with highest similarity score.

    Args:
        G: Current network graph.
        agent_idx: Node index of the rewiring agent.
        agents: List of all agent objects.
        rng: Seeded random number generator.
        stress_weight: Weight for stress similarity.
        affect_weight: Weight for affect similarity.
        resilience_weight: Weight for resilience similarity.

    Returns:
        Node index of best candidate, or -1 if none found.
    """
    state_self = {
        "current_stress": getattr(agents[agent_idx], "current_stress", 0.0),
        "affect": getattr(agents[agent_idx], "affect", 0.0),
        "resilience": getattr(agents[agent_idx], "resilience", 0.5),
    }

    neighbors = set(G.neighbors(agent_idx))
    best_candidate = -1
    best_similarity = -1.0

    for candidate in G.nodes():
        if candidate == agent_idx or candidate in neighbors:
            continue

        state_candidate = {
            "current_stress": getattr(agents[candidate], "current_stress", 0.0),
            "affect": getattr(agents[candidate], "affect", 0.0),
            "resilience": getattr(agents[candidate], "resilience", 0.5),
        }

        sim = compute_connection_similarity(
            state_self, state_candidate, stress_weight, affect_weight, resilience_weight
        )

        if sim > best_similarity:
            best_similarity = sim
            best_candidate = candidate

    return best_candidate


def apply_stress_adaptation(
    G: nx.Graph,
    agents: List,
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[nx.Graph, int]:
    """Apply stress-driven network adaptation (edge rewiring).

    For each agent with ``stress_breach_count >= threshold``, find the edge
    to the most dissimilar neighbor and rewire it to a more similar
    non-neighbor with probability ``(1 - retention_prob)``.

    Args:
        G: Current network graph (will be modified).
        agents: List of agent objects, index-aligned with G.nodes().
        config: Dict with keys:
            - ``adaptation_threshold``: int, minimum breach count to trigger.
            - ``rewire_probability``: float, base rewiring probability.
            - ``homophily_strength``: float, homophily bias.
            - ``stress_weight``: float, similarity weight for stress.
            - ``affect_weight``: float, similarity weight for affect.
            - ``resilience_weight``: float, similarity weight for resilience.
        rng: Seeded random number generator.

    Returns:
        Tuple of (modified graph, number of rewired edges).
    """
    G_prime = G.copy()
    rewired_count = 0

    threshold = config.get("adaptation_threshold", 3)
    homophily_strength = config.get("homophily_strength", 0.7)
    stress_weight = config.get("stress_weight", 1.0)
    affect_weight = config.get("affect_weight", 1.0)
    resilience_weight = config.get("resilience_weight", 1.0)

    for node in list(G_prime.nodes()):
        agent = agents[node]
        breach_count = getattr(agent, "stress_breach_count", 0)

        if breach_count < threshold:
            continue

        neighbors = list(G_prime.neighbors(node))
        if not neighbors:
            continue

        # Find the most dissimilar neighbor (lowest similarity)
        state_self = {
            "current_stress": getattr(agent, "current_stress", 0.0),
            "affect": getattr(agent, "affect", 0.0),
            "resilience": getattr(agent, "resilience", 0.5),
        }

        worst_neighbor = neighbors[0]
        worst_similarity = 1.0

        for nb in neighbors:
            nb_state = {
                "current_stress": getattr(agents[nb], "current_stress", 0.0),
                "affect": getattr(agents[nb], "affect", 0.0),
                "resilience": getattr(agents[nb], "resilience", 0.5),
            }
            sim = compute_connection_similarity(state_self, nb_state, stress_weight, affect_weight, resilience_weight)
            if sim < worst_similarity:
                worst_similarity = sim
                worst_neighbor = nb

        # Decide whether to retain or rewire
        support_effectiveness = getattr(agent, "support_boost", 0.0)
        retention_prob = compute_connection_retention_probability(
            worst_similarity, support_effectiveness, homophily_strength
        )

        if rng.random() > retention_prob:
            # Find a better candidate
            candidate = _find_rewire_candidate(
                G_prime,
                node,
                agents,
                rng,
                stress_weight,
                affect_weight,
                resilience_weight,
            )
            if candidate >= 0:
                G_prime.remove_edge(node, worst_neighbor)
                G_prime.add_edge(node, candidate)
                rewired_count += 1

    return G_prime, rewired_count
