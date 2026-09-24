# Configuration Parameters Reference

_See [`.kilocode/rules/math/notation.md`](../../.kilocode/rules/math/notation.md) for symbol definitions and conventions. The authoritative parameter listing for the manuscript is `docs/manuscript/_parameters.md`; this document maps environment variables to those symbols and values._

## Overview

This document provides a comprehensive reference mapping all environment variables from the configuration system to their mathematical notation and default values. Values reflect the current `.env.example` and source code (`src/python/assumption_config.py`, `src/python/config.py`).

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
| `AGENT_INITIAL_RESILIENCE_MEAN` | $\mu_{\mathfrak{R}, \text{0}}$ | 0.0 | Initial resilience latent mean for agents |
| `AGENT_INITIAL_RESILIENCE_SD` | $\sigma_{\mathfrak{R}, \text{0}}$ | 1.0 | Initial resilience latent SD for agents |
| `AGENT_INITIAL_AFFECT_MEAN` | $\mu_{A, \text{0}}$ | 0.0 | Initial affect latent mean for agents |
| `AGENT_INITIAL_AFFECT_SD` | $\sigma_{A, \text{0}}$ | 1.0 | Initial affect latent SD for agents |
| `AGENT_INITIAL_RESOURCES_MEAN` | $\mu_{R, \text{0}}$ | 0.0 | Initial resources latent mean for agents |
| `AGENT_INITIAL_RESOURCES_SD` | $\sigma_{R, \text{0}}$ | 1.0 | Initial resources latent SD for agents |
| `AGENT_STRESS_PROBABILITY` | $p_{\text{stress}}$ | 0.2 | Probability of stress events for agents |
| `AGENT_COPING_SUCCESS_RATE` | $p_b$ | 0.5 | Base success rate for coping with stress |
| `AGENT_SUBEVENTS_PER_DAY` | $\lambda_s$ | 3 | Subevent (Poisson) rate per day for each agent |
| `AGENT_RESOURCE_COST` | $\kappa$ | 0.03 | Resource cost for coping attempts |
| `STRESS_CONTROLLABILITY_MEAN` | $\mu_c$ | 0.5 | Mean controllability of stress events |
| `STRESS_CONTROLLABILITY_SD` | $\sigma_c$ | 0.2 | SD of controllability of stress events |
| `STRESS_OVERLOAD_MEAN` | $\mu_o$ | 0.5 | Mean overload of stress events |
| `STRESS_OVERLOAD_SD` | $\sigma_o$ | 0.2 | SD of overload of stress events |
| `STRESS_BETA_ALPHA` | $\alpha_\beta$ | 2.0 | Alpha parameter for Beta distribution of stress sampling |
| `STRESS_BETA_BETA` | $\beta_\beta$ | 2.0 | Beta parameter for Beta distribution of stress sampling |
| `APPRAISAL_OMEGA_C` | $\omega_c$ | 1.0 | Weight for controllability in stress appraisal |
| `APPRAISAL_OMEGA_O` | $\omega_o$ | 1.0 | Weight for overload in stress appraisal |
| `APPRAISAL_BIAS` | $b$ | 0.0 | Bias term in stress appraisal function |
| `APPRAISAL_GAMMA` | $\gamma$ | 3.0 | Steepness parameter for sigmoid in appraisal |
| `THRESHOLD_BASE_THRESHOLD` | $\eta_{\text{0}}$ | 0.5 | Base threshold for stress responses |
| `THRESHOLD_CHALLENGE_SCALE` | $\eta_{\chi}$ | 0.15 | Challenge component threshold scaling |
| `THRESHOLD_HINDRANCE_SCALE` | $\eta_{\zeta}$ | 0.25 | Hindrance component threshold scaling |
| `THRESHOLD_STRESS_THRESHOLD` | $\eta_{\text{stress}}$ | 0.7 | Stress threshold for stressed-state detection |
| `THRESHOLD_AFFECT_THRESHOLD` | $\eta_{\text{affect}}$ | 0.3 | Affect threshold for stressed-state detection |
| `STRESS_ALPHA_CHALLENGE` | $\lambda_C$ | 0.8 | Challenge modifier for appraised stress |
| `STRESS_ALPHA_HINDRANCE` | $\lambda_H$ | 1.2 | Hindrance modifier for appraised stress |
| `STRESS_DELTA` | $\delta$ | 0.4 | Polarity effect strength in appraised stress load |
| `STRESS_DECAY_RATE` | $\delta_{\text{stress}}$ | 0.05 | Rate of stress decay over time |
| `PSS10_ITEM_MEAN` | $\mu_i$ | [1.43, 1.38, 1.51, 1.31, 1.50, 1.40, 1.43, 1.60, 1.14, 1.31] | Per-item means for PSS-10 (scaled to US norms) |
| `PSS10_ITEM_SD` | $\sigma_i$ | [0.89, 0.89, 0.93, 0.92, 0.80, 0.78, 0.78, 0.88, 0.91, 0.93] | Per-item SDs for PSS-10 |
| `PSS10_LOAD_OVERLOAD` | $\lambda_{o,\Psi,i}$ | [0.7, 0.3, 0.8, 0.2, 0.4, 0.9, 0.2, 0.3, 0.4, 0.9] | Factor loadings for overload dimension (code) |
| `PSS10_LOAD_CONTROLLABILITY` | $\lambda_{c,\Psi,i}$ | [0.2, 0.8, 0.1, 0.7, 0.6, 0.1, 0.8, 0.6, 0.7, 0.1] | Factor loadings for controllability dimension (code) |
| `PSS10_BIFACTOR_COR` | $\rho_\Psi$ | -0.3 | Correlation between PSS-10 dimensions |
| `PSS10_CONTROLLABILITY_SD` | $\sigma_{c,\Psi}$ | 1.0 (used as /4) | SD for controllability dimension (regularized) |
| `PSS10_OVERLOAD_SD` | $\sigma_{o,\Psi}$ | 1.0 (used as /4) | SD for overload dimension (regularized) |
| `PSS10_THRESHOLD` | $\eta_\Psi$ | 27 | Threshold for PSS-10 stress classification |
| `PSS10_SCALE` | $c_{\Psi}$ | 6.0 | Score scale factor for PSS-10 |
| `PSS10_NOISE_SD` | $\epsilon_{\Psi}$ | 2.0 | Measurement noise SD multiplier for PSS-10 |
| `PSS10_SKEW_A` | $\gamma_{\Psi}$ | 3.0 | Skew parameter for PSS-10 score distribution |
| `PSS10_BIAS_SD` | $\sigma_{\text{bias}}$ | 1.0 | Between-person variance SD for PSS-10 |
| `PSS10_RESILIENCE_COUPLING` | $\beta_{\text{res}}$ | 3.5 | Resilience coupling strength for PSS-10 |
| `PSS10_STRESS_DAMPENING` | $\phi_{\text{damp}}$ | 1.0 | Stress dampening factor (1.0 = none) |
| `PSS10_SENSITIVITY` | $\xi_{\text{sens}}$ | 0.5 | Dynamic PSS-10 sensitivity |
| `PSS10_MOMENTUM_WEIGHT` | $\xi_{\text{mom}}$ | 0.3 | Dynamic PSS-10 momentum weight |
| `COPING_BASE_PROBABILITY` | $p_{\text{cope}}$ | 0.5 | Base probability for successful coping |
| `COPING_SOCIAL_INFLUENCE` | $\delta_{\text{cope,soc}}$ | 0.1 | Social influence factor on coping |
| `COPING_CHALLENGE_BONUS` | $\theta_{\text{cope,}\chi}$ | 0.2 | Bonus for coping with challenge events |
| `COPING_HINDRANCE_PENALTY` | $\theta_{\text{cope,}\zeta}$ | 0.3 | Penalty for coping with hindrance events |
| `INTERACTION_INFLUENCE_RATE` | $\alpha_{\text{int}}$ | 0.05 | Rate of influence in social interactions |
| `INTERACTION_RESILIENCE_INFLUENCE` | $\delta_{\text{res,int}}$ | 0.05 | Resilience influence in interactions |
| `INTERACTION_MAX_NEIGHBORS` | $k_{\text{max}}$ | 10 | Maximum neighbors for social interaction |
| `AFFECT_PEER_INFLUENCE_RATE` | $\alpha_p$ | 0.1 | Peer influence rate on affect |
| `AFFECT_EVENT_APPRAISAL_RATE` | $\alpha_e$ | 0.15 | Event appraisal rate on affect |
| `AFFECT_HOMEOSTATIC_RATE` | $\lambda_{\text{affect}}$ | 0.5 | Homeostatic rate for affect |
| `RESILIENCE_HOMEOSTATIC_RATE` | $\lambda_{\text{resilience}}$ | 0.5 | Homeostatic rate for resilience |
| `RESILIENCE_COPING_SUCCESS_RATE` | $\theta_{\text{boost\|cope}}$ | 0.1 | Boost rate for successful coping |
| `RESILIENCE_SOCIAL_SUPPORT_RATE` | $\alpha_s$ | 0.08 | Social support rate for resilience |
| `RESILIENCE_OVERLOAD_THRESHOLD` | $\eta_{\text{res,overload}}$ | 3 | Threshold for overload effects |
| `RESILIENCE_BOOST_RATE` | $\theta_{\text{boost}}$ | 0.1 | General boost rate for resilience |
| `N_INFLUENCING_NEIGHBORS` | $k_{\text{influence}}$ | 5 | Number of influencing neighbors |
| `N_INFLUENCING_HINDRANCE` | $h_c$ | 3 | Consecutive hindrance count for overload |
| `PROTECTIVE_SOCIAL_SUPPORT` | $e_{\text{soc}}$ | 0.5 | Efficacy of social support |
| `PROTECTIVE_FAMILY_SUPPORT` | $e_{\text{fam}}$ | 0.5 | Efficacy of family support |
| `PROTECTIVE_FORMAL_INTERVENTION` | $e_{\text{int}}$ | 0.5 | Efficacy of formal interventions |
| `PROTECTIVE_PSYCHOLOGICAL_CAPITAL` | $e_{\text{cap}}$ | 0.5 | Efficacy of psychological capital |
| `RESOURCE_BASE_REGENERATION` | $\lambda_R$ | 0.50 | Base rate for resource regeneration (linear $\lambda_R(1-R)$) |
| `RESOURCE_ALLOCATION_COST` | $\kappa_{\text{alloc}}$ | 0.15 | Cost of resource allocation |
| `RESOURCE_COST_EXPONENT` | $\gamma_c$ | 1.5 | Exponent for resource cost function |
| `RESOURCE_SOCIAL_EXCHANGE_RATE` | $\rho_{\text{exch}}$ | 0.5 | Resource exchange rate in social interactions |
| `RESOURCE_EXCHANGE_THRESHOLD` | $t_{\text{exch}}$ | 0.2 | Minimum resource difference for exchange |
| `RESOURCE_MAX_EXCHANGE_RATIO` | $m_{\text{exch}}$ | 0.5 | Maximum fraction of resources exchanged |
| `PROTECTIVE_IMPROVEMENT_RATE` | $\gamma_p$ | 0.5 | Rate of protective factor improvement |
| `UTILITY_SOFTMAX_TEMPERATURE` | $\beta_{\text{softmax}}$ | 1.0 | Temperature parameter for softmax decisions |
| `STRESS_CONTROLLABILITY_UPDATE_RATE` | $\nu_c$ | 0.05 | Per-event learning rate for controllability dimension |
| `STRESS_OVERLOAD_UPDATE_RATE` | $\nu_o$ | 0.05 | Per-event learning rate for overload dimension |
| `LOG_LEVEL` | $L_{\text{level}}$ | 'INFO' | Logging level for the application |
| `OUTPUT_RESULTS_DIR` | $D_{\text{results}}$ | 'data/processed' | Directory for processed output data |
| `OUTPUT_RAW_DIR` | $D_{\text{raw}}$ | 'data/raw' | Directory for raw output data |
| `OUTPUT_LOGS_DIR` | $D_{\text{logs}}$ | 'logs' | Directory for log files |
| `OUTPUT_SAVE_TIME_SERIES` | $F_{\text{ts}}$ | True | Whether to save time series data |
| `OUTPUT_SAVE_NETWORK_SNAPSHOTS` | $F_{\text{net}}$ | True | Whether to save network snapshots |
| `OUTPUT_SAVE_SUMMARY_STATISTICS` | $F_{\text{sum}}$ | True | Whether to save summary statistics |

### Assumption Constants

All `ASSUMPTION_*` variables are model-internal tuning constants loaded from `src/python/assumption_config.py` with defaults documented in `.env.example`. They are grouped into:
- **Coping** (15): resource reward/penalty, PF allocation fraction, affect/resilience change scales, challenge/hindrance resistance outcomes, stress reduction/increase, affect change
- **Stress** (21): controllability/overload challenge-hindrance weights, baselines, homeostasis rates, event intensity weights, failed-coping multiplier, momentum dynamics, PSS-10 estimation
- **Resource** (17): resilience efficiency, thresholds, cost floors, penalties, efficiency gains, social boosts, allocation penalties, regeneration multipliers
- **Social** (3): support exchange threshold, probability, exchange boost
- **Buffering** (24): resilience thresholds, volatility priors, initial PF values, buffering coefficients, homeostasis rates, smoothing alpha, exchange reduction

### Coupling Constants

- `STRESS_RESOURCE_COUPLING` (0.10), `STRESS_AFFECT_COUPLING` (0.15), `STRESS_RESILIENCE_COUPLING` (0.60), `STRESS_EROSION_RATE` (0.15)
- `PSS10_RESILIENCE_COUPLING_ITEM` (0.20), `PSS10_RESOURCE_COUPLING` (0.03), `PSS10_AFFECT_COUPLING` (0.25), `PSS10_AFFECT_ADJUSTMENT` (3.0), `PSS10_RESILIENCE_COUPLING` (3.5)
- Network similarity weights (1.0 each for resilience, stress, affect)

## Configuration Integration

All parameters are loaded from environment variables with type conversion and validation through the unified configuration system in [`src/python/config.py`](src/python/config.py). The system provides fallback defaults and ensures parameter consistency across all model components.

### Parameter Validation

The configuration system includes comprehensive validation to ensure:

1. **Range Validation**: Parameters fall within acceptable ranges
2. **Type Safety**: Proper type conversion from environment variables
3. **Array Validation**: PSS-10 arrays have correct length and value ranges
4. **Research Compliance**: Parameters align with empirical research constraints

This reference serves as the authoritative mapping between environment variables and their mathematical representations in the agent-based mental health model.