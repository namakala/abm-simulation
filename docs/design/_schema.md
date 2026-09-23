# AgentState Schema

### Purpose

Single source of truth for all mutable agent variables. Passed between phases;
phases read from and write to this schema via `state_delta`.

- **Frequency:** N/A (data structure)
- **Inputs:** N/A
- **Outputs:** N/A

### Field Definitions

**Core state:**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `baseline_resilience` | float | [0, 1] | Initial resilience (fixed) |
| `resilience` | float | [0, 1] | Current resilience |
| `baseline_affect` | float | [-1, 1] | Initial affect (fixed) |
| `affect` | float | [-1, 1] | Current affect |
| `resources` | float | [0, 1] | Psychological/physical resources |

**Protective factors:**

| Field | Type | Description |
|-------|------|-------------|
| `protective_factors` | Dict[str, float] | social_support, family_support, formal_intervention, psychological_capital |

**Stress tracking:**

| Field | Type | Description |
|-------|------|-------------|
| `current_stress` | float | Current stress level [0, 1] |
| `recent_stress_intensity` | float | Tracks recent stress for PSS-10 response |
| `stress_momentum` | float | Rate of stress change for predictive updates |
| `consecutive_hindrances` | float | Hindrance streak length |
| `stress_breach_count` | int | Count of stress threshold breaches |

**PSS-10 state:**

| Field | Type | Description |
|-------|------|-------------|
| `pss10_responses` | Dict | Individual PSS-10 item responses |
| `stress_controllability` | float | Controllability dimension [0, 1] |
| `stress_overload` | float | Overload dimension [0, 1] |
| `pss10` | int | Total PSS-10 score (0-40) |
| `pss10_smoothed` | float | Float smoothed value, carried across days |
| `stressed` | bool | Stress classification based on threshold |
| `daily_pss10_scores` | List[int] | Scores collected during current day |

**Transient keys** (written by one phase, consumed by next):

| Field | Type | Producer | Consumer |
|-------|------|----------|----------|
| `challenge` | float | Stress Perception | Resilience Activation |
| `hindrance` | float | Stress Perception | Resilience Activation |
| `is_stressed` | bool | Stress Perception | Orchestrator branching |
| `event_controllability` | float | Stress Perception | Resilience Activation |
| `event_overload` | float | Stress Perception | Resilience Activation |

Reference: [src/python/phases/interfaces.py:L18-L75](https://github.com/namakala/abm-simulation/blob/7e40a44f82da76b18910b774cd882c938bc79992/src/python/phases/interfaces.py#L18-L75)
