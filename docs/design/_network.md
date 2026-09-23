# Network Topology

### Purpose

Social network uses Watts-Strogatz small-world topology with high clustering
and short characteristic path lengths. Network adaptation rewires edges based
on stress breach counts and homophily similarity.

- **Frequency:** model-level (network adaptation runs daily)
- **Inputs:** agent stress_breach_count, affect, resilience
- **Outputs:** modified graph edges

### Algorithm

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

### Parameters

| Name | Description | Default | Source |
|------|-------------|---------|--------|
| `N` | Number of agents | 20 | config |
| `k` | Mean degree (must be even, < N) | 4 | config |
| `p` | Rewiring probability | 0.1 | config |
| `threshold` | Breach count for adaptation | 3 | assumption |
| `homophily` | Similarity weight | 0.7 | assumption |

Reference: [src/python/network_utils.py:L16-L245](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/network_utils.py#L16-L245)
