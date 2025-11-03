# Configuration Parameters Reference

_See [`.kilocode/rules/math/notation.md`](../../.kilocode/rules/math/notation.md) for symbol definitions and conventions._

## Overview

This document provides a comprehensive reference mapping all environment variables from the configuration system to their mathematical notation and default values.

## Environment Variable Reference

| Variable | Notation | Value | Description |
|----------|----------|-------|-------------|
| `SIMULATION_NUM_AGENTS` | $N$ | 20 | Number of agents in the simulation network |
| `SIMULATION_MAX_DAYS` | $T_{\max}$ | 100 | Maximum number of days to run the simulation |
| `SIMULATION_SEED` | $S_{\text{seed}}$ | 42 | Random number generator seed for reproducibility |
| `NETWORK_WATTS_K` | $WS_k$ | 4 | Mean degree in Watts-Strogatz network topology |
| `NETWORK_WATTS_P` | $WS_p$ | 0.1 | Rewiring probability in Watts-Strogatz network |
| `NETWORK_ADAPTATION_THRESHOLD` | $\eta_{\text{adapt}}$ | 3 | Threshold for triggering network adaptation |
| `NETWORK_REWIRE_PROBABILITY` | $p_{\text{rewire}}$ | 0.01 | Probability of rewiring network connections |
| `NETWORK_HOMOPHILY_STRENGTH` | $\delta_{\text{homophily}}$ | 0.7 | Strength of homophily in network connections |
| `AGENT_INITIAL_RESILIENCE` | $\mu_{\mathfrak{R}, \text{0}}$ | 0.5 | Initial resilience mean for agents |
| `AGENT_INITIAL_AFFECT` | $\mu_{A, \text{0}}$ | 0.0 | Initial affect mean for agents |
| `AGENT_INITIAL_RESOURCES` | $\mu_{R, \text{0}}$ | 0.6 | Initial resources mean for agents |
| `AGENT_STRESS_PROBABILITY` | - | 0.5 | Probability of stress events for agents |
| `AGENT_COPING_SUCCESS_RATE` | $p_b$ | 0.5 | Base success rate for coping with stress |
| `AGENT_SUBEVENTS_PER_DAY` | $n_{\text{subevents}}$ | 3 | Number of subevents per day for each agent |
| `AGENT_RESOURCE_COST` | $\kappa$ | 0.1 | Resource cost for coping attempts |
| `STRESS_CONTROLLABILITY_MEAN` | $c$ | 0.5 | Mean controllability of stress events |
| `STRESS_OVERLOAD_MEAN` | $o$ | 0.5 | Mean overload of stress events |
| `STRESS_BETA_ALPHA` | $\alpha_s$ | 2.0 | Alpha parameter for Beta distribution of stress events |
| `STRESS_BETA_BETA` | $\beta_s$ | 2.0 | Beta parameter for Beta distribution of stress events |
| `APPRAISAL_OMEGA_C` | $\omega_c$ | 1.0 | Weight for controllability in stress appraisal |
| `APPRAISAL_OMEGA_O` | $\omega_o$ | 1.0 | Weight for overload in stress appraisal |
| `APPRAISAL_BIAS` | $b$ | 0.0 | Bias term in stress appraisal function |
| `APPRAISAL_GAMMA` | $\gamma$ | 6.0 | Steepness parameter for sigmoid in appraisal |
| `THRESHOLD_BASE_THRESHOLD` | $\eta_{\text{0}}$ | 0.5 | Base threshold for stress responses |
| `THRESHOLD_CHALLENGE_SCALE` | $\eta_{\chi}$ | 0.15 | Challenge component threshold scaling |
| `THRESHOLD_HINDRANCE_SCALE` | $\eta_{\zeta}$ | 0.25 | Hindrance component threshold scaling |
| `THRESHOLD_STRESS_THRESHOLD` | $\eta_{\text{stress}}$ | 0.3 | Base stress threshold for agents |
| `THRESHOLD_AFFECT_THRESHOLD` | $\eta_{\text{affect}}$ | 0.3 | Affect threshold for homeostasis |
| `STRESS_ALPHA_CHALLENGE` | $\lambda_C$ | 0.8 | Challenge modifier for stress threshold |
| `STRESS_ALPHA_HINDRANCE` | $\lambda_H$ | 1.2 | Hindrance modifier for stress threshold |
| `STRESS_DELTA` | $\delta$ | 0.2 | Polarity effect strength in stress appraisal |
| `PSS10_ITEM_MEAN` | - | [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5] | Mean values for PSS-10 items |
| `PSS10_ITEM_SD` | - | [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8] | Standard deviations for PSS-10 items |
| `PSS10_LOAD_CONTROLLABILITY` | $\lambda_{c,\Psi,i}$ | [0.2, 0.8, 0.1, 0.7, 0.6, 0.1, 0.8, 0.6, 0.7, 0.1] | Factor loadings for controllability dimension |
| `PSS10_LOAD_OVERLOAD` | $\lambda_{o,\Psi,i}$ | [0.7, 0.3, 0.8, 0.2, 0.4, 0.9, 0.2, 0.3, 0.4, 0.9] | Factor loadings for overload dimension |
| `PSS10_BIFACTOR_COR` | $\rho_\Psi$ | 0.3 | Correlation between PSS-10 dimensions |
| `PSS10_CONTROLLABILITY_SD` | $\sigma_c$ | 1.0 | Standard deviation for controllability dimension |
| `PSS10_OVERLOAD_SD` | $\sigma_o$ | 1.0 | Standard deviation for overload dimension |
| `PSS10_THRESHOLD` | $\eta_\Psi$ | 27 | Threshold for PSS-10 stress classification |
| `COPING_BASE_PROBABILITY` | $p_b$ | 0.5 | Base probability for successful coping |
| `COPING_SOCIAL_INFLUENCE` | $\delta_{\text{soc}}$ | 0.1 | Social influence factor on coping |
| `COPING_CHALLENGE_BONUS` | $\theta_{\chi}$ | 0.2 | Bonus for coping with challenge events |
| `COPING_HINDRANCE_PENALTY` | $\theta_{\zeta}$ | 0.3 | Penalty for coping with hindrance events |
| `INTERACTION_INFLUENCE_RATE` | $\alpha_p$ | 0.05 | Rate of influence in social interactions |
| `INTERACTION_RESILIENCE_INFLUENCE` | $\lambda_{\text{res,interact}}$ | 0.05 | Resilience influence in interactions |
| `INTERACTION_MAX_NEIGHBORS` | $k_{\text{max}}$ | 10 | Maximum neighbors for social influence |
| `AFFECT_PEER_INFLUENCE_RATE` | $\alpha_p$ | 0.1 | Peer influence rate on affect |
| `AFFECT_EVENT_APPRAISAL_RATE` | $\lambda_{\text{appraise}}$ | 0.15 | Event appraisal rate on affect |
| `AFFECT_HOMEOSTATIC_RATE` | $\lambda_{\text{affect}}$ | 0.5 | Homeostatic rate for affect |
| `RESILIENCE_HOMEOSTATIC_RATE` | $\lambda_{\text{resilience}}$ | 0.05 | Homeostatic rate for resilience |
| `RESILIENCE_COPING_SUCCESS_RATE` | $\theta_{\text{boost\|cope}}$ | 0.1 | Boost rate for successful coping |
| `RESILIENCE_SOCIAL_SUPPORT_RATE` | $\alpha_s$ | 0.08 | Social support rate for resilience |
| `RESILIENCE_OVERLOAD_THRESHOLD` | $\eta_{\text{res,overload}}$ | 3 | Threshold for overload effects |
| `RESILIENCE_BOOST_RATE` | $\theta_{\text{boost}}$ | 0.1 | General boost rate for resilience |
| `N_INFLUENCING_NEIGHBORS` | $k_{\text{influence}}$ | 5 | Number of influencing neighbors |
| `N_INFLUENCING_HINDRANCE` | $k_{\text{hindrance}}$ | 3 | Number of hindrance-influencing neighbors |
| `STRESS_DECAY_RATE` | $\delta_{\text{stress}}$ | 0.05 | Rate of stress decay over time |
| `PROTECTIVE_SOCIAL_SUPPORT` | $\epsilon_{\text{soc}}$ | 0.5 | Efficacy of social support |
| `PROTECTIVE_FAMILY_SUPPORT` | $\epsilon_{\text{fam}}$ | 0.5 | Efficacy of family support |
| `PROTECTIVE_FORMAL_INTERVENTION` | $\epsilon_{\text{int}}$ | 0.5 | Efficacy of formal interventions |
| `PROTECTIVE_PSYCHOLOGICAL_CAPITAL` | $\epsilon_{\text{cap}}$ | 0.5 | Efficacy of psychological capital |
| `RESOURCE_BASE_REGENERATION` | $\lambda_R$ | 0.05 | Base rate for resource regeneration |
| `RESOURCE_ALLOCATION_COST` | $\kappa_{\text{alloc}}$ | 0.15 | Cost of resource allocation |
| `RESOURCE_COST_EXPONENT` | $\gamma_c$ | 1.5 | Exponent for resource cost function |
| `PROTECTIVE_IMPROVEMENT_RATE` | $\gamma_p$ | 0.5 | Rate of protective factor improvement |
| `UTILITY_SOFTMAX_TEMPERATURE` | $\beta_{\text{softmax}}$ | 1.0 | Temperature parameter for softmax decisions |
| `LOG_LEVEL` | $L_{\text{level}}$ | 'INFO' | Logging level for the application |
| `OUTPUT_RESULTS_DIR` | $D_{\text{results}}$ | 'data/processed' | Directory for processed output data |
| `OUTPUT_RAW_DIR` | $D_{\text{raw}}$ | 'data/raw' | Directory for raw output data |
| `OUTPUT_LOGS_DIR` | $D_{\text{logs}}$ | 'logs' | Directory for log files |
| `OUTPUT_SAVE_TIME_SERIES` | $F_{\text{ts}}$ | True | Whether to save time series data |
| `OUTPUT_SAVE_NETWORK_SNAPSHOTS` | $F_{\text{net}}$ | True | Whether to save network snapshots |
| `OUTPUT_SAVE_SUMMARY_STATISTICS` | $F_{\text{sum}}$ | True | Whether to save summary statistics |

## Configuration Integration

All parameters are loaded from environment variables with type conversion and validation through the unified configuration system in [`src/python/config.py`](src/python/config.py). The system provides fallback defaults and ensures parameter consistency across all model components.

### Parameter Validation

The configuration system includes comprehensive validation to ensure:

1. **Range Validation**: Parameters fall within acceptable ranges
2. **Type Safety**: Proper type conversion from environment variables
3. **Array Validation**: PSS-10 arrays have correct length and value ranges
4. **Research Compliance**: Parameters align with empirical research constraints

This reference serves as the authoritative mapping between environment variables and their mathematical representations in the agent-based mental health model.