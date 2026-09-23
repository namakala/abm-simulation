# AgentState Schema

## Purpose

Single source of truth for all mutable agent variables. Passed between phases;
phases read from and write to this schema via `state_delta`.

- **Frequency:** N/A (data structure)
- **Inputs:** N/A
- **Outputs:** N/A

## Field Definitions

**Core state:**

```{=latex}
\begin{tabularx}{\textwidth}{p{4cm}p{1.8cm}p{2cm}X}
\toprule
\textbf{Field} & \textbf{Type} & \textbf{Range} & \textbf{Description} \\
\midrule
\trowfour{baseline\_resilience}{float}{[0, 1]}{Initial resilience (fixed)}
\trowfour{resilience}{float}{[0, 1]}{Current resilience}
\trowfour{baseline\_affect}{float}{[-1, 1]}{Initial affect (fixed)}
\trowfour{affect}{float}{[-1, 1]}{Current affect}
\trowfour{resources}{float}{[0, 1]}{Psychological/physical resources}
\bottomrule
\end{tabularx}
```

**Protective factors:**

```{=latex}
\begin{tabularx}{\textwidth}{p{4cm}p{1.8cm}X}
\toprule
\textbf{Field} & \textbf{Type} & \textbf{Description} \\
\midrule
\trow{protective\_factors}{Dict[str, float]}{social\_support, family\_support, formal\_intervention, psychological\_capital}
\bottomrule
\end{tabularx}
```

**Stress tracking:**

```{=latex}
\begin{tabularx}{\textwidth}{p{4cm}p{1.8cm}X}
\toprule
\textbf{Field} & \textbf{Type} & \textbf{Description} \\
\midrule
\trow{current\_stress}{float}{Current stress level [0, 1]}
\trow{recent\_stress\_intensity}{float}{Tracks recent stress for PSS-10 response}
\trow{stress\_momentum}{float}{Rate of stress change for predictive updates}
\trow{consecutive\_hindrances}{float}{Hindrance streak length}
\trow{stress\_breach\_count}{int}{Count of stress threshold breaches}
\bottomrule
\end{tabularx}
```

**PSS-10 state:**

```{=latex}
\begin{tabularx}{\textwidth}{p{4cm}p{1.8cm}X}
\toprule
\textbf{Field} & \textbf{Type} & \textbf{Description} \\
\midrule
\trow{pss10\_responses}{Dict}{Individual PSS-10 item responses}
\trow{stress\_controllability}{float}{Controllability dimension [0, 1]}
\trow{stress\_overload}{float}{Overload dimension [0, 1]}
\trow{pss10}{int}{Total PSS-10 score (0-40)}
\trow{pss10\_smoothed}{float}{Float smoothed value, carried across days}
\trow{stressed}{bool}{Stress classification based on threshold}
\trow{daily\_pss10\_scores}{List[int]}{Scores collected during current day}
\bottomrule
\end{tabularx}
```

**Transient keys** (written by one phase, consumed by next):

```{=latex}
\begin{tabularx}{\textwidth}{p{4cm}p{1.8cm}XX}
\toprule
\textbf{Field} & \textbf{Type} & \textbf{Producer} & \textbf{Consumer} \\
\midrule
\trowfour{challenge}{float}{Stress Perception}{Resilience Activation}
\trowfour{hindrance}{float}{Stress Perception}{Resilience Activation}
\trowfour{is\_stressed}{bool}{Stress Perception}{Orchestrator branching}
\trowfour{event\_controllability}{float}{Stress Perception}{Resilience Activation}
\trowfour{event\_overload}{float}{Stress Perception}{Resilience Activation}
\bottomrule
\end{tabularx}
```

Reference: [src/python/phases/interfaces.py:L18-L75](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/phases/interfaces.py#L18-L75)
