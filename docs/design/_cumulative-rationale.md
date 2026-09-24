# Rationale for Cumulative Block Experiment

## Purpose

Explain why stages 1-7 selectively enable phases and how this demonstrates
the modularity claim made in the main text.

- **Frequency:** N/A (analytical design)
- **Inputs:** N/A
- **Outputs:** N/A

## Design

The cumulative block experiment assembles the model block by block across
seven stages, each running the same seeded population (N = 100 agents,
90 days). Each stage adds one building block on top of the previous ones:

```{=latex}
\begin{tabularx}{\textwidth}{p{1.5cm}p{4cm}X}
\toprule
\textbf{Stage} & \textbf{Blocks Active} & \textbf{Purpose} \\
\midrule
1 & Initialization only & Baseline population state \\
2 & + Stress perception & Isolate appraisal effects \\
3 & + Resilience activation & Isolate coping effects \\
4 & + Interaction & Isolate social contagion \\
5 & + Resource allocation & Isolate resource regeneration \\
6 & + Stress buffering & Isolate protective buffering \\
7 & + Full model & Complete homeostatic cycle \\
\bottomrule
\end{tabularx}
```

## Interpretation

Intermediate stages (2-6) are deliberately partial: they enable stress-adding
mechanisms without the restorative blocks of the daily cycle. This isolation
demonstrates that:

1. **Stress perception alone** (stage 2) updates appraisal dimensions without
   affecting outcome states, so stages 1 and 2 coincide
2. **Resilience activation** (stage 3) introduces coping, causing population
   depletion without regeneration
3. **Interaction** (stage 4) propagates affect/resilience through the network
   with negativity bias, dragging both toward lower bounds
4. **Resource allocation** (stage 5) restores the resource pool but cannot
   move affect or resilience
5. **Stress buffering** (stage 6) begins pushing resilience back toward
   baseline

Stage 7, the complete model, activates the remaining blocks: affect dynamics,
PSS-10 consolidation, daily reset, and network adaptation. This completes the
two-loop daily cycle. The homeostatic pull toward baseline affect and
resilience, the daily stress decay, and adaptive rewiring toward more
similar peers jointly restore equilibrium.

## Analytical vs Runtime Stacking

The cumulative block experiment demonstrates **analytical stacking**:
selectively enabling/disabling phases to isolate contributions. At runtime,
the simulation uses **sequential composition**: all phases execute every day
in a pipeline, with each phase consuming the previous phase's output.

This distinction is important: the modularity claim is validated by the
cumulative experiment, while the runtime behavior uses the full two-loop
orchestration described in S3.

See: article.qmd Results → "Mechanism contribution: cumulative building-block build-up"
