## Network Topology

Social network uses Watts-Strogatz small-world topology with high clustering
and short characteristic path lengths. Network adaptation rewires edges based
on stress breach counts and homophily similarity.

**Algorithm:**

```
FUNCTION build_watts_strogatz_network(N, k, p, rng):
    k ← min(k, N - 1)
    RETURN watts_strogatz_graph(N, k, p, seed=rng)

FUNCTION apply_stress_adaptation(G, agents, config, rng):
    FOR each node with breach_count ≥ threshold:
        worst_neighbor ← most dissimilar neighbor
        retention_prob ← sigmoid(similarity, support_effectiveness, homophily)
        IF rng.random() > retention_prob:
            candidate ← find most similar non-neighbor
            rewire edge: node -- worst_neighbor → node -- candidate
    RETURN (modified_graph, rewired_count)
```

Reference: [src/python/network_utils.py:L16-L245](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/network_utils.py#L16-L245)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `N` | Number of agents | 20 |
| `k` | Mean degree (must be even, < N) | 4 |
| `p` | Rewiring probability | 0.1 |

Returns: `nx.Graph` for `build_watts_strogatz_network`; `Tuple[nx.Graph, int]`
for `apply_stress_adaptation`.

## Data Collection

Mesa's `DataCollector` captures model-level and agent-level metrics using
named reporter functions. Model reporters track population averages; agent
reporters track individual trajectories.

**Reporters:**

```
AGENT_REPORTERS = {
    pss10, resilience, affect, resources, current_stress,
    stress_controllability, stress_overload, consecutive_hindrances,
    coping_success, challenge_appraisal, hindrance_appraisal,
    interaction_frequency, stressed, support_boost
}

MODEL_REPORTERS = {
    avg_pss10, avg_resilience, avg_affect, coping_success_rate,
    avg_resources, avg_stress, social_support_rate, network_density,
    stress_prevalence, low_resilience, high_resilience,
    avg_challenge, avg_hindrance, challenge_hindrance_ratio, ...
}
```

Reference: [src/python/reporters.py:L289-L333](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/reporters.py#L289-L333)

DataCollector initialisation:

Reference: [src/python/model.py:L114-L133](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/model.py#L114-L133)
