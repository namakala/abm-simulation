### Agent Initialization and State Management

Each agent was initialized with baseline values representing natural equilibrium points using mathematical transformations ensuring proper statistical distributions. Resilience baselines utilized sigmoid transformation and affect baselines employed hyperbolic tangent transformation, both using seeded random number generation for reproducibility:

\begin{align*}
\mathfrak{R}_{\text{0}} &= \sigma\left(\frac{X - \mu_{\mathfrak{R},\text{0}}}{\sigma_{\mathfrak{R},\text{0}}}\right) \\
R_{\text{0}} &= \sigma\left(\frac{X - \mu_{R,\text{0}}}{\sigma_{R,\text{0}}}\right) \\
A_{\text{0}} &= \tanh\left(\frac{X - \mu_{A,\text{0}}}{\sigma_{A,\text{0}}}\right)
\end{align*}
\label{eq-baseline-transformations}

where $X \sim \mathcal{N}(\mu, \sigma^2)$ is a normal random variable, $\sigma(x) = \frac{1}{1+e^{-x}}$ is the sigmoid function, and $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ is the hyperbolic tangent function.

Core state variables included resilience ($\mathfrak{R} \in [0,1]$), affect ($A \in [-1,1]$), resources ($R \in [0,1]$), and current stress ($S \in [0,1]$). The model integrated comprehensive Perceived Stress Scale-10 (PSS-10) functionality using a bifactor model with dimension score generation and validated theoretical correlations:

$$
\begin{pmatrix}
c_\Psi \\ o_\Psi
\end{pmatrix}
\sim
\mathcal{N}\left(
\begin{pmatrix}
\mu_c \\ \mu_o
\end{pmatrix},
\begin{pmatrix}
\sigma_c^2 & \rho_\Psi \sigma_c \sigma_o \\
\rho_\Psi \sigma_c \sigma_o & \sigma_o^2
\end{pmatrix}
\right)
$$ {#eq-pss10-dimension-score}

Where $c_\Psi, o_\Psi \in [0,1]$ are PSS-10 dimension scores, $\rho_\Psi \in [-1,1]$ is the bifactor correlation, and $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ denotes the multivariate normal distribution.

### Stress Event Processing and Coping

The stress processing pipeline transformed life events into psychological responses through challenge-hindrance appraisal. Events were generated with controllability ($c$) and overload ($o$) attributes, then appraised using the following mathematical framework:

\begin{align}
z &= \omega_c \cdot c - \omega_o \cdot o + b \\
\chi &= \sigma(\gamma \cdot z) \\
\zeta &= 1 - \chi \\
\eta_{\mathrm{eff}} &= \eta_{\text{0}} + \eta_{\chi} \cdot \chi - \eta_{\zeta} \cdot \zeta \\
p_{\mathrm{coping}} &= p_b + \theta_{\text{cope,}\chi} \cdot \chi - \theta_{\text{cope,}\zeta} \cdot \zeta + \delta_{\text{cope,soc}} \cdot \frac{1}{k} \sum_{j=1}^k A_j
\label{eq-stress-processing}
\end{align}

where $c, o \in [0,1]$ are event attributes, $\omega_c, \omega_o \in \mathbb{R}$ are appraisal weights, $b \in \mathbb{R}$ is bias, $\gamma > 0$ controls sigmoid steepness, $\eta_{\text{0}}, \eta_{\chi}, \eta_{\zeta} \in [0,1]$ are threshold parameters, $p_b \in [0,1]$ is base coping probability, $\theta_{\text{cope,}\chi}, \theta_{\text{cope,}\zeta} > 0$ are coping modifiers, $\delta_{\text{cope,soc}} \in [0,1]$ is social influence, $k$ is number of neighbors, and $A_j \in [-1,1]$ are neighbor affect values.

Dynamic threshold evaluation adjusted stress response based on event characteristics, and coping success determination integrated challenge/hindrance effects with social influence through probability calculation.

### Social Interaction Mechanism

Social interactions occurred between neighboring agents in the network topology, enabling emotional contagion and mutual support. The system processed interactions through utility functions computing mutual emotional and resilience effects, with social influence on affect determined by neighbor states and relationship dynamics.

#### Mutual Influence Mechanism

Social interactions created bidirectional changes where both individuals affect each other's emotional state. The model recognized that negative emotional states tended to have stronger influence than positive ones:

$$
\Delta A_i = \alpha_p \cdot (A_j - A_i) \cdot \begin{cases}
1.5 & \text{if } A_j - A_i < 0 \\
1.0 & \text{if } A_j - A_i \geq 0
\end{cases}
$$ {#eq-mutual-influence}

Where $\Delta A_i$ is affect change for agent $i$, $\alpha_p \in [0,1]$ is peer influence rate, and $A_i, A_j \in [-1,1]$ are affect values.

#### Network Adaptation Mechanism

When individuals experienced repeated stress, they adapted their social connections to better suit their needs. Network adaptation triggers based on stress breach counts:

$$
\mathrm{trigger\ adaptation} = \begin{cases}
1 & \text{if } c_{\text{breach}} \geq \eta_{\text{adapt}} \\
0 & \text{otherwise}
\end{cases}
$$ {#eq-adaptation-trigger}

Where $c_{\text{breach}} \in \mathbb{N}$ is stress breach count and $\eta_{\text{adapt}} \in \mathbb{N}$ is adaptation threshold. $c_{\text{breach}}$ is a mechanism to record chronic stress pattern, reflecting the count of perceived stress exceeding $\eta_{\mathrm{eff}}$.

Connection preferences were determined by similarity and support effectiveness:

$$
s_{ij} = 1 - \frac{|A_i - A_j| + |\mathfrak{R}_i - \mathfrak{R}_j|}{2}
$$ {#eq-connection-similarity}

Where $s_{ij} \in [0,1]$ is similarity between agents $i,j$, $A_i, A_j \in [-1,1]$ are affect values, and $\mathfrak{R}_i, \mathfrak{R}_j \in [0,1]$ are resilience values.

Connection retention probability balanced homophily with support effectiveness:

$$
p_{\text{keep}} = s_{ij} \cdot \delta_{\text{homophily}} + e_s \cdot (1 - \delta_{\text{homophily}})
$$ {#eq-retention-probability}

where $p_{\text{keep}} \in [0,1]$ is probability of keeping connection, $\delta_{\text{homophily}} \in [0,1]$ is homophily strength, and $e_s \in [0,1]$ is support effectiveness.

#### Social Support Dynamics

Support effectiveness depended on the neighbor's resilience and affect:

$$
e_s = \frac{\mathfrak{R}_j + (1 + A_j)/2}{2} + 0.2
$$ {#eq-support-effectiveness}

where $e_s \in [0,1]$ is support effectiveness, $\mathfrak{R}_j \in [0,1]$ is neighbor's resilience, and $A_j \in [-1,1]$ is neighbor's affect.

Social support exchange was detected when meaningful improvements occurred:

$$
\mathrm{support\ exchange} = \begin{cases}
1 & \text{if } \max\{|\Delta A_i|, |\Delta \mathfrak{R}_i|, |\Delta A_j|, |\Delta \mathfrak{R}_j|, |\Delta \text{resources}|\} > \eta_{\text{exchange}} \\
0 & \text{otherwise}
\end{cases}
$$ {#eq-support-exchange}

where $\Delta A_i, \Delta A_j$ are affect changes, $\Delta \mathfrak{R}_i, \Delta \mathfrak{R}_j$ are resilience changes, $\Delta \text{resources}$ is resource transfer, and $\eta_{\text{exchange}} \in [0,1]$ is exchange threshold.

### Daily Simulation Step

Each simulation day followed a structured sequence ensuring proper mechanism integration. The process initiated with daily initialization, capturing initial affect and resilience values while obtaining neighbor emotional states. Subevent generation determined daily activity through Poisson sampling, creating random sequences of interactions and stress events:

\begin{align}
n_s &\sim \max(\mathcal{P}(\lambda_s), 1) \\
\bar{\chi}_d &= \frac{1}{n_e}\sum_{i=1}^{n_e} \chi_i \\
\bar{\zeta}_d &= \frac{1}{n_e}\sum_{i=1}^{n_e} \zeta_i
\label{eq-daily-integration}
\end{align}

where $n_s \in \mathbb{N}$ is number of subevents; $\mathcal{P}(\lambda_s)$ is Poisson distribution with rate $\lambda_s$; $\bar{\chi}_d, \bar{\zeta}_d \in [0,1]$ are daily averages; and $n_e$ is number of stress events.

Daily integration normalized challenge/hindrance values by event count, providing inputs for dynamics application. Dynamics updates applied integrated affect and resilience dynamics:

\begin{align}
A_{t+1} &= A_t + \Delta A_p + \Delta A_e + \Delta A_h \\
\mathfrak{R}_{t+1} &= \mathfrak{R}_t + \Delta \mathfrak{R}_{\chi\zeta} + \Delta \mathfrak{R}_p + \Delta \mathfrak{R}_o + \Delta \mathfrak{R}_s + \lambda_{\text{resilience}} \cdot (\mathfrak{R}_{\text{0}} - \mathfrak{R}_t) \\
S_{t+1} &= S_t \cdot (1 - \delta_{\text{stress}})
\label{eq-dynamics-updates}
\end{align}

where $A_t, \mathfrak{R}_t, S_t \in [0,1]$ are current values; $\Delta A_p, \Delta A_e, \Delta A_h$ are affect changes; $\Delta \mathfrak{R}_{\chi\zeta}, \Delta \mathfrak{R}_p, \Delta \mathfrak{R}_o, \Delta \mathfrak{R}_s$ are resilience changes; $\lambda_{\text{resilience}} \in [0,1]$ is homeostatic rate; $\mathfrak{R}_{\text{0}} \in [0,1]$ is baseline resilience; and $\delta_{\text{stress}} \in [0,1]$ is stress decay rate.

Homeostatic adjustment applied natural pull toward baseline equilibrium for affect and resilience, while stress decay followed exponential reduction. Daily reset procedures cleared tracking variables and stored summaries for analysis.

### Affect and Resilience Dynamics

Integrated affect dynamics combined peer influence, event appraisal effects, and homeostasis through the following mathematical framework:

\begin{align}
\Delta A_p &= \frac{1}{k} \sum_{j=1}^{k} \alpha_p \cdot (A_j - A_t) \cdot \mathbb{1}_{j \leq k_{\text{influence}}} \\
\Delta A_e &= \alpha_e \cdot \bar{\chi}_d \cdot (1 - A_t) - \alpha_e \cdot \bar{\zeta}_d \cdot \max(0.1, A_t + 1) \\
\Delta A_h &= \lambda_{\text{affect}} \cdot (A_{\text{0}} - A_t)
\label{eq-affect-dynamics}
\end{align}

where $\Delta A_p, \Delta A_e, \Delta A_h$ are affect change components; $k$ is number of neighbors; $k_{\text{influence}}$ is number of influencing neighbors; $\alpha_p, \alpha_e \in [0,1]$ are influence rates; $\bar{\chi}_d, \bar{\zeta}_d \in [0,1]$ are daily averages; $A_t, A_j \in [-1,1]$ are affect values; $\lambda_{\text{affect}} \in [0,1]$ is homeostatic rate; $A_{\text{0}} \in [-1,1]$ is baseline affect; and $\mathbb{1}$ is indicator function.

Resilience dynamics integrated challenge-hindrance effects, protective factor boosts, overload effects, and social support contributions:

$$
\Delta \mathfrak{R}_p = \sum_{f \in F} e_f \cdot (\mathfrak{R}_{\text{0}} - \mathfrak{R}_t) \cdot \theta_{\text{boost}}
$$ {#eq-resilience-boosts}

where $\Delta \mathfrak{R}_p$ is resilience boost from protective factors; $F = \{\mathrm{soc}, \mathrm{fam}, \mathrm{int}, \mathrm{cap}\}$ is set of protective factors; $e_f \in [0,1]$ is efficacy of factor $f$; $\mathfrak{R}_{\text{0}}, \mathfrak{R}_t \in [0,1]$ are baseline and current resilience; and $\theta_{\text{boost}} > 0$ is boost rate parameter.

Challenge-hindrance effects varied based on coping outcomes: successful coping yields positive resilience changes while failed coping produces negative impacts.

### Resource Management System

Resources represented finite psychological and physical capacity for coping and protective factor maintenance. Resource regeneration followed affect-modulated recovery, while consumption occurred during coping attempts. Protective factor allocation utilized softmax decision framework for bounded rational resource distribution across social support, family support, formal intervention, and psychological capital:

\begin{align}
R' &= \lambda_R \cdot (R_{\max} - R) \cdot (1 + \beta_a \cdot \max(0, A)) \\
w_f &= \frac{\exp(e_f / \beta_{\text{softmax}})}{\sum_{k \in F} \exp(e_k / \beta_{\text{softmax}})}
\label{eq-resource-dynamics}
\end{align}

where $R' > 0$ is resource regeneration; $\lambda_R \in [0,1]$ is regeneration rate; $R_{\max} = 1$ is maximum resources; $R \in [0,1]$ is current resources; $\beta_a > 0$ is affect influence parameter; $A \in [-1,1]$ is current affect; $w_f \in [0,1]$ is allocation weight for factor $f$; $e_f \in [0,1]$ is efficacy of factor $f$; $\beta_{\text{softmax}} > 0$ is softmax temperature; and $F$ is set of protective factors.

### Model-Level Simulation Orchestration

The simulation employed Mesa's agent-based modeling framework with dual-class architecture separating agent behaviors from model orchestration. Network structure utilized Watts-Strogatz small-world topology providing realistic social connection patterns with local clustering and short path lengths. The network was initialized with a mean degree of $WS_k$ and rewiring probability of $WS_p$.

Model-level coordination managed population statistics, network adaptation tracking, and cumulative social support monitoring. The orchestration ensured proper temporal sequencing and data collection while maintaining computational efficiency for large-scale simulations.

### Data Collection and Analysis Framework

The simulation implemented Mesa's `DataCollector` for standardized, efficient data collection replacing manual tracking systems. Model-level reporters captured population metrics including average PSS-10 scores ($\bar{\Psi}$), resilience levels ($\bar{\mathfrak{R}}$), affect states ($\bar{A}$), stress prevalence ($P_{\text{stressed}}$), and social network characteristics. Agent-level reporters tracked individual trajectories for PSS-10 scores, resilience, affect, resources, current stress, stress controllability, stress overload, and consecutive hindrances. Recovery potential, vulnerability index, challenge-hindrance balance, and coping success rates were also computed for comprehensive analysis. The data collection framework supported comprehensive research applications including baseline versus intervention comparison, individual trajectory analysis, network analysis, and parameter sensitivity studies.
