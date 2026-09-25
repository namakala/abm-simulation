# Network Topology

## Purpose

Social network uses Watts-Strogatz small-world topology with high clustering
and short characteristic path lengths. Network adaptation rewires edges based
on stress breach counts and homophily similarity.

- **Frequency:** model-level (network adaptation runs daily)
- **Inputs:** agent stress_breach_count, affect, resilience
- **Outputs:** modified graph edges

## Algorithm

```
FUNCTION build_network(N, k, p, rng):
    k ← min(k, N - 1)
    RETURN watts_strogatz(N, k, p, rng)

FUNCTION adapt_network(G, agents, config, rng):
    FOR each node WHERE breach_count ≥ threshold:
        worst ← most dissimilar neighbor
        prob ← sigmoid(similarity, support, homophily)
        IF rng > prob:
            best ← most similar non-neighbor
            rewire(node, worst, best)
    RETURN G
```

## Parameters

```{=latex}
\begin{tabularx}{\textwidth}{p{4.5cm}Xp{2cm}}
\toprule
\textbf{Name} & \textbf{Description} & \textbf{Default} \\
\midrule
\trow{N}{Number of agents}{20}
\trow{k}{Mean degree (must be even, < N)}{4}
\trow{p}{Rewiring probability}{0.1}
\trow{threshold}{Breach count for adaptation}{3}
\trow{homophily}{Similarity weight}{0.7}
\bottomrule
\end{tabularx}
```

Reference: [src/python/network_utils.py:L16-L245](https://github.com/namakala/abm-simulation/blob/84ff9b37fb0c569bef9f122e789d9c23a89348ec/src/python/network_utils.py#L16-L245)
