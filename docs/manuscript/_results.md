@fig-initial-population presents a cross-sectional snapshot of agent psychological states at epoch 0, derived from randomly generated baseline data with minimal theoretical associations except for PSS-10 scores and stress levels. This figure underscores the absence of predetermined relationships in the initialization phase, highlighting the random nature of agent attributes that establish a neutral starting point for subsequent dynamics. The initial stress level has an artificially strong correlation to PSS-10 because it was generated from PSS-10. The rationale is to ground the initial stress level to an empirical values reflected in PSS-10. By illustrating uncorrelated variables, it emphasizes the model's reliance on stochastic processes to simulate real-world heterogeneity without biasing initial conditions.

![Cross-sectional visualization of agent psychological states at epoch 0, highlighting randomly generated baseline distributions with minimal associations except between PSS-10 and stress levels](figures/base_initial_population.pdf){#fig-initial-population}

@fig-final-population captures cross-sectional agent states at the simulation's conclusion, revealing theoretically grounded associations among psychological variables. Resilience exhibits a weak positive correlation with affect, a weak-to-moderate negative correlation with stress, and a weak negative correlation with PSS-10. In contrast, stress shows a moderate positive correlation with PSS-10, accompanied by increasing dispersion in stress levels at higher PSS-10 scores, reflecting realistic overdispersion in psychometric instruments. This visualization demonstrates model convergence toward expected relationships, validating the simulation's capacity to generate plausible homeostatic mechanism of psychological dynamics.

@fig-time-series depicts longitudinal trajectories of all agents across simulation epochs, showcasing stable temporal variation indicative of effective homeostatic mechanisms. Drawing from comprehensive agent-state data over time, this figure reveals equilibrium maintenance through consistent fluctuations, with implications for understanding behavioral stability and dynamic resilience in response to stressors. It directly addresses the hypothesis by evidencing stable population metrics within constant ranges, affirming that agent-based simulations can model homeostatic resilience via social network dynamics, where interactions sustain mental health equilibrium without excessive variability.

![The emerging associations of agent psychological states captured at simulation completion](figures/base_final_population.pdf){#fig-final-population}

Under homogeneous initial conditions simulating a general population, the model exhibited stable homeostatic mechanisms with population-level metrics confined to narrow ranges, directly supporting the research question on agent-based modeling of psychological resilience through social network dynamics. Model-level analysis reveals remarkable stability in aggregate measures. Mean perceived stress (PSS-10 = 24.804 ± 0.308, range: 21.93–25.65) demonstrates low variability indicative of effective homeostatic regulation. Resilience (0.523 ± 0.008, range: 0.48–0.54) shows uniform coping capacity across the population. Affect (−0.024 ± 0.003, range: −0.03 to −0.01) exhibits controlled mood variation near neutral. This population-level stability occurs despite substantial individual-level heterogeneity. Agent-level variability in stress shows a coefficient of variation of 31.8%. Resilience demonstrates a coefficient of variation of 55.3%. Affect exhibits a coefficient of variation of 1421%. Elevated resource levels (0.702 ± 0.005) indicate effective resource management through social network dynamics. Moderate stress (0.484 ± 0.007) relative to baselines suggests effective stress mitigation. These outcomes directly affirm the hypothesis that agent interactions determine mental health outcomes within a stable homeostatic state. The findings simultaneously address the research question by demonstrating that complex social interactions can maintain population equilibrium despite individual variability.

```{=latex}
\begin{table}
\centering
\caption{Model-level population statistics demonstrating homeostatic stability}
\label{tbl-model-level}
\begin{tabular}{llp{5.5cm}}
\toprule
\textbf{Variable} & \textbf{Mean $\pm$ SD} & \textbf{Interpretation} \\
\midrule
PSS-10 across agents & 24.804 $\pm$ 0.308 & Stable stress perception with moderate elevation \\
Resilience level & 0.523 $\pm$ 0.008 & Uniform coping capacity across population \\
Affect (valence) & $-0.024 \pm 0.003$ & Controlled mood variation near neutral \\
Coping success rate & 0.448 $\pm$ 0.012 & Reliable coping outcomes maintaining homeostasis \\
Psychological resources & 0.702 $\pm$ 0.005 & Stable resource levels supporting resilience \\
Current stress level & 0.484 $\pm$ 0.007 & Variable patterns within bounded ranges \\
Challenge appraisal & 0.500 $\pm$ 0.007 & Balanced appraisal pattern \\
Hindrance appraisal & 0.500 $\pm$ 0.007 & Balanced appraisal pattern \\
Challenge--hindrance diff. & $-0.000 \pm 0.014$ & Centered near zero, indicating equilibrium \\
Hindrance sequence length & 1.862 $\pm$ 0.066 & Occasional stress streaks but stable overall \\
Coping events per agent & 684.201 $\pm$ 25.642 & Uniform behavioral patterns \\
Social exchanges per agent & 1522.693 $\pm$ 38.794 & Consistent network engagement \\
Support transactions & 395.853 $\pm$ 22.364 & Consistent mutual aid \\
\bottomrule
\end{tabular}
\end{table}
```

@tbl-model-level substantiates a homeostatic system sustained by social network dynamics and agent interactions. The aggregate metrics demonstrate stability across all psychological variables, with coefficients of variation ranging from 1.1% (resource levels) to 5.7% (current stress), indicating tight population-level regulation. These constrained variabilities evidence steady-state dynamics wherein supportive networks bolster resource preservation and stress alleviation, supporting the hypothesis of population metrics within constant ranges. Balanced challenge-hindrance appraisals (0.500 ± 0.007 each, with difference centered at -0.000 ± 0.014) imply neutral cognitive processing and effective appraisal mechanisms. Uniform social exchanges (1522.693 ± 38.794 per agent) denote robust network-based mutual aid that maintains collective stability. Support transactions (395.853 ± 22.364) further validate the mutual aid system. The consistent coping success rate (0.448 ± 0.012) across all agents validates the model's capacity to simulate emergent equilibrium from individual interactions that promote mental health outcomes.

![Stable temporal variation and homeostatic equilibrium of agent psychological states across all simulation epochs](figures/base_time_series.pdf){#fig-time-series}

```{=latex}
\begin{table}
\centering
\caption{Agent-level temporal dynamics demonstrating individual heterogeneity within population stability}
\label{tbl-agent-level}
\begin{tabular}{llp{5.5cm}}
\toprule
\textbf{Variable} & \textbf{Mean $\pm$ SD} & \textbf{Interpretation} \\
\midrule
PSS-10 per agent-step & 24.804 $\pm$ 7.900 & Variable stress levels with realistic range \\
Resilience level & 0.523 $\pm$ 0.289 & Diverse adaptive capacities \\
Emotional state & $-0.024 \pm$ 0.341 & Variable patterns around neutral \\
Psychological resources & 0.702 $\pm$ 0.089 & Fluctuating resource capacity \\
Stress intensity & 0.484 $\pm$ 0.200 & Variable patterns within bounds \\
Controllability perception & 0.477 $\pm$ 0.131 & Diverse agency experiences \\
Overload perception & 0.612 $\pm$ 0.144 & Diverse capacity challenges \\
Hindrance streak length & 0.932 $\pm$ 1.320 & Variable patterns observed \\
\bottomrule
\end{tabular}
\end{table}
```

@tbl-agent-level reveals substantial dynamic variability within the stable population framework, directly supporting the research question on homeostatic resilience modeling through social network dynamics. The contrast between population-level stability (coefficient of variation < 6%) and individual-level heterogeneity (coefficient of variation 31.8-1421%) exemplifies the dual nature of the modeled system. Individual agents experience significant fluctuations in stress (0.00–40.00), resilience (0.00–1.00), and affect (−1.00–1.00). These variations are constrained by social interactions that maintain population equilibrium. The comprehensive range of controllability (0.477 ± 0.131) captures the cognitive appraisal diversity that drives individual adaptation strategies. Overload perceptions (0.612 ± 0.144) further demonstrate diverse capacity challenges. The bounded hindrance streaks (maximum 17.65) demonstrate the system's capacity to prevent pathological cascades through network support. These observations directly validate the hypothesis that agent interactions shape outcomes within a homeostatic state. The model portrays a complex adaptive system where individual diversities reflect realistic psychological heterogeneity. Individual diversities are harmonized through social dynamics to achieve collective stability. The model thus successfully demonstrates that agent-based simulations can effectively capture the homeostatic mechanisms of psychological resilience. Social network dynamics transform individual variability into population-level equilibrium.
