```{=latex}
\setlength\LTleft{0pt}
\setlength\LTright{0pt}
\begin{longtable}{lp{2.4cm}p{1.6cm}p{1.7cm}p{5.2cm}}
\caption{Summary of model parameters and variables used in the agent-based mental health simulation. Status: Validated = value grounded in peer-reviewed empirical literature; Plausible = theoretically grounded within a literature-supported range; Assumed = model-internal constant without direct empirical evidence; variable = computed state derived from the update equations.} \\
\label{tbl-parameters} \\
\toprule
\textbf{Symbol} & \textbf{Description} & \textbf{Value} & \textbf{Status} & \textbf{Note} \\
\midrule
\multicolumn{5}{c}{\textbf{Simulation and Network}} \\
\midrule
$N$ & Number of agents (population size) & 20 & Plausible & \cite{watts1998}: prototype scale; scalable to 1000+ \\
$T_{\max}$ & Maximum simulation days & 100 & Assumed & Assumes 100 days suffice for stress dynamics to stabilise; the model steps through daily loops over this horizon. \\
$S_{\mathrm{seed}}$ & Random seed & 42 & Assumed & Assumes a fixed seed for full reproducibility; controls the RNG only and carries no behavioural content. \\
$WS_k$ & Mean degree in Watts-Strogatz network & 4 & Plausible & \cite{watts1998}: k must be even; degree 4 typical for small-world topology. \\
$WS_p$ & Rewiring probability in Watts-Strogatz network & 0.1 & Plausible & \cite{watts1998}: p = 0.1 is the standard small-world regime. \\
\midrule
\multicolumn{5}{c}{\textbf{Agent Initialization}} \\
\midrule
$\mu_{\mathfrak{R},0}$ & Initial resilience latent mean & 0.0 & Plausible & \cite{shively2020}: sigmoid(N(0,1)) centres near 0.5, consistent with CD-RISC norms $\approx$ 0.74--0.80. \\
$\sigma_{\mathfrak{R},0}$ & Initial resilience latent SD & 1.0 & Plausible & \cite{connor2003}: empirical SD $\approx$ 0.13--0.19 on [0,1]; latent SD 1.0 spans the full interval. \\
$\mu_{A,0}$ & Initial affect latent mean & 0.0 & Plausible & \cite{watson1988}: tanh maps latent 0 to neutral valence; empirical population mean $\approx$ 0.1--0.2 on [-1,1]. \\
$\sigma_{A,0}$ & Initial affect latent SD & 1.0 & Plausible & \cite{watson1988}: empirical SD $\approx$ 0.3--0.4 on [-1,1]; latent SD 1.0 gives full-range spread. \\
$\mu_{R,0}$ & Initial resources latent mean & 0.0 & Plausible & \cite{hobfoll2001}: sigmoid centres resources near 0.5; right-skewed population norms. \\
$\sigma_{R,0}$ & Initial resources latent SD & 1.0 & Plausible & \cite{hobfoll2001}: latent default spanning [0,1]. \\
\midrule
\multicolumn{5}{c}{\textbf{Agent Behaviour}} \\
\midrule
$p_{\mathrm{stress}}$ & Probability of a daily stressor & 0.2 & Validated & \cite{almeida2002}: 0.1--0.3 for clinically significant events; 40\% of days contain at least one stressor. \\
$p_b$ & Base coping success rate & 0.5 & Validated & \cite{skinner2003}: approach coping 0.60--0.75, avoidance 0.30--0.40; 0.5 is the community midpoint. \\
$\lambda_s$ & Subevent (Poisson) rate per day & 3 & Plausible & \cite{almeida2002}: Poisson rate 2--5 events/day. \\
$\kappa$ & Resource cost of a coping attempt & 0.03 & Assumed & Assumes each coping attempt consumes 3\% of the resource pool; applied linearly per coping event. \\
\midrule
\multicolumn{5}{c}{\textbf{Stress Events}} \\
\midrule
$\mu_c$ & Controllability mean of stress events & 0.5 & Validated & \cite{bolger1991}: 0.4--0.6 on [0,1]. \\
$\sigma_c$ & Controllability SD of stress events & 0.2 & Validated & \cite{bolger1991}: SD 0.2--0.3. \\
$\mu_o$ & Overload mean of stress events & 0.5 & Validated & \cite{bolger1991}: 0.4--0.6 on [0,1]. \\
$\sigma_o$ & Overload SD of stress events & 0.2 & Validated & \cite{bolger1991}: SD 0.2--0.3. \\
\midrule
\multicolumn{5}{c}{\textbf{Appraisal and Threshold}} \\
\midrule
$\omega_c$ & Controllability weight in appraisal & 1.0 & Plausible & \cite{cavanaugh2000}: ratio $\omega_c:\omega_o \approx 1:1$ (0.8--1.2). \\
$\omega_o$ & Overload weight in appraisal & 1.0 & Plausible & \cite{cavanaugh2000}: ratio $\approx 1:1$. \\
$b$ & Appraisal bias term & 0.0 & Assumed & Assumes no systematic appraisal bias; at neutral event attributes the population splits evenly between challenge and hindrance. \\
$\gamma$ & Sigmoid steepness in appraisal & 3.0 & Plausible & \cite{cavanaugh2000}: 4--6 recommended; 3.0 yields graded rather than binary appraisal. \\
$\alpha_\beta$ & Beta shape parameter for stress sampling & 2.0 & Plausible & \cite{lazarus1984}: Beta(2,2) is symmetric and unimodal on [0,1]. \\
$\beta_\beta$ & Beta shape parameter for stress sampling & 2.0 & Plausible & \cite{lazarus1984}: Beta(2,2) is symmetric and unimodal. \\
$\eta_0$ & Base stress threshold & 0.5 & Plausible & \cite{cavanaugh2000}: midpoint of the appraisal space. \\
$\eta_{\chi}$ & Challenge threshold scale & 0.15 & Plausible & \cite{cavanaugh2000}: challenge raises the effective threshold (protective). \\
$\eta_{\zeta}$ & Hindrance threshold scale & 0.25 & Plausible & \cite{podsakoff2007}: hindrance lowers the threshold; negativity bias (0.25 > 0.15). \\
$\lambda_C$ & Challenge modifier for appraised stress & 0.8 & Plausible & \cite{cavanaugh2000}: challenge dampens appraised stress ($<1$). \\
$\lambda_H$ & Hindrance modifier for appraised stress & 1.2 & Plausible & \cite{cavanaugh2000}: hindrance amplifies appraised stress ($>1$). \\
$\delta$ & Polarity effect strength & 0.4 & Assumed & Assumes balanced events anchor appraised stress at 0.5; applied in $L = 0.5 + \delta(\zeta - \chi)$. \\
\midrule
\multicolumn{5}{c}{\textbf{Social Interaction}} \\
\midrule
$\alpha_{\mathrm{int}}$ & Influence rate in social interactions & 0.05 & Plausible & \cite{centola2018}: affect convergence 0.05--0.15 per interaction. \\
$\delta_{\mathrm{res,int}}$ & Resilience influence in interactions & 0.05 & Plausible & \cite{centola2018}: contagion rate for resilience transfer. \\
$k_{\max}$ & Maximum neighbours per interaction & 10 & Assumed & Assumes up to 10 neighbours can influence one interaction; caps social reach and computational load. \\
\midrule
\multicolumn{5}{c}{\textbf{Affect Dynamics}} \\
\midrule
$\alpha_p$ & Peer influence rate on affect & 0.1 & Validated & \cite{centola2018}: 0.05--0.15 per interaction. \\
$\alpha_e$ & Event appraisal rate on affect & 0.15 & Plausible & \cite{lazarus1984}: daily challenge/hindrance shift affect. \\
$\lambda_{\mathrm{affect}}$ & Affect homeostatic rate & 0.5 & Plausible & \cite{friston2016}: set-point return to baseline affect. \\
$k_{\mathrm{influence}}$ & Number of influencing neighbours & 5 & Assumed & Assumes 5 neighbours drive peer influence; affect differences are averaged over this subset. \\
\midrule
\multicolumn{5}{c}{\textbf{Resilience Dynamics}} \\
\midrule
$\lambda_{\mathrm{resilience}}$ & Resilience homeostatic rate & 0.5 & Plausible & \cite{shively2020}: return to baseline resilience; scaled by resources and stress in code. \\
$\theta_{\mathrm{boost|cope}}$ & Resilience gain from successful coping & 0.1 & Plausible & \cite{cavanaugh2000}: successful challenge coping builds resilience. \\
$\alpha_s$ & Social support rate for resilience & 0.08 & Plausible & \cite{Hobfoll1989}: supportive exchanges transfer resilience. \\
$\eta_{\mathrm{res,overload}}$ & Overload threshold & 3 & Assumed & Assumes 3 consecutive hindrance events trigger resilience depletion; coarse allostatic-load mechanism. \\
$h_c$ & Consecutive hindrance count & 3 & Assumed & Assumes 3 consecutive hindrance events count toward overload detection; decays by $\delta_{\mathrm{stress}}$ per day. \\
$\theta_{\mathrm{boost}}$ & Protective factor boost rate & 0.1 & Plausible & \cite{Hobfoll1989}: protective factors boost resilience toward baseline. \\
\midrule
\multicolumn{5}{c}{\textbf{Resource Dynamics}} \\
\midrule
$e_{f}$ & Protective factor efficacy (social, family, intervention, PsyCap) & 0.5 each & Plausible & \cite{Hobfoll1989}: baseline efficacy 0.5 on [0,1]. \\
$\gamma_p$ & Protective improvement rate & 0.5 & Assumed & Assumes allocated resources convert to protective effect at 50\% per day; applied as a linear conversion in the allocation update. \\
$\lambda_R$ & Resource base regeneration rate & 0.50 & Assumed & Assumes 50\% of the deficit regenerates daily ($R' = \lambda_R(1 - R_t)$); exceeds the empirical 0.2--0.3 range and models fast recovery. \\
$\kappa_{\mathrm{alloc}}$ & Resource allocation cost & 0.15 & Plausible & \cite{Hobfoll1989}: convex cost of allocation, $c_a = \kappa_{\mathrm{alloc}} \cdot a^{\gamma_c}$. \\
$\gamma_c$ & Resource cost exponent & 1.5 & Plausible & \cite{Hobfoll1989}: diminishing returns on allocation. \\
$\rho_{\mathrm{exch}}$ & Social resource exchange rate & 0.5 & Assumed & Assumes 50\% of the resource difference transfers between agents during a support interaction. \\
$t_{\mathrm{exch}}$ & Resource exchange threshold & 0.2 & Assumed & Assumes exchange occurs only when the resource gap exceeds 0.2, preventing trivial transfers. \\
$m_{\mathrm{exch}}$ & Maximum exchange ratio & 0.5 & Assumed & Assumes at most 50\% of an agent's resources are exchanged in a single interaction. \\
\midrule
\multicolumn{5}{c}{\textbf{Utility}} \\
\midrule
$\beta_{\mathrm{softmax}}$ & Softmax temperature & 1.0 & Plausible & \cite{simon1955a}: bounded-rationality default balancing exploration and exploitation. \\
\midrule
\multicolumn{5}{c}{\textbf{Coping Mechanism}} \\
\midrule
$p_{\mathrm{cope}}$ & Base coping probability & 0.5 & Validated & \cite{skinner2003}: community midpoint. \\
$\theta_{\mathrm{cope},\chi}$ & Challenge bonus for coping & 0.2 & Plausible & \cite{cavanaugh2000}: challenge appraisal raises coping probability. \\
$\theta_{\mathrm{cope},\zeta}$ & Hindrance penalty for coping & 0.3 & Plausible & \cite{bakker2017}: negativity bias (penalty $>$ bonus). \\
$\delta_{\mathrm{cope,soc}}$ & Social influence on coping & 0.1 & Plausible & \cite{centola2018}: contagion range. \\
\midrule
\multicolumn{5}{c}{\textbf{Daily Dynamics}} \\
\midrule
$\delta_{\mathrm{stress}}$ & Stress decay rate & 0.05 & Plausible & \cite{lazarus1984}: half-life of about 14 days; applied as $S_{t+1} = \max(0.03, S_t(1 - \delta_{\mathrm{stress}}))$. \\
$\eta_{\mathrm{stress}}$ & Stress threshold for stressed state & 0.7 & Assumed & Assumes a stress value above 0.7 marks a stressed state; combined with the affect threshold for detection. \\
$\eta_{\mathrm{affect}}$ & Affect threshold for stressed state & 0.3 & Assumed & Assumes affect below the 0.3 band contributes to stressed-state detection alongside stress. \\
\midrule
\multicolumn{5}{c}{\textbf{Network Adaptation}} \\
\midrule
$\eta_{\mathrm{adapt}}$ & Network adaptation threshold & 3 & Assumed & Assumes rewiring is triggered after 3 stress breaches; counts adaptation events. \\
$\delta_{\mathrm{homophily}}$ & Homophily strength & 0.7 & Plausible & \cite{centola2018}: homophily is well documented; 0.7 is an assumed magnitude. \\
$p_{\mathrm{rewire}}$ & Network rewiring probability & 0.01 & Plausible & \cite{watts1998}: low rewiring preserves small-world structure. \\
\midrule
\multicolumn{5}{c}{\textbf{PSS-10}} \\
\midrule
$\mu_i$ & Per-item means & [1.43, 1.38, 1.51, 1.31, 1.50, 1.40, 1.43, 1.60, 1.14, 1.31] & Validated & \cite{liu2020}: raw 1.59--2.22 scaled by 0.72 to US norms; total $\approx$ 14/40. \\
$\sigma_i$ & Per-item SDs & [0.89, 0.89, 0.93, 0.92, 0.80, 0.78, 0.78, 0.88, 0.91, 0.93] & Validated & \cite{liu2020}: 0.78--0.93. \\
$\lambda_{o,\Psi,i}$ & Overload factor loadings & [1, 1, 1, 0, 0, 1, 0, 0, 1, 1] (empirical); 0.2--0.9 (code) & Validated & \cite{liu2020}: items 1, 2, 3, 6, 9, 10; code uses graded loadings. \\
$\lambda_{c,\Psi,i}$ & Controllability factor loadings & [0, 0, 0, 1, 1, 0, 1, 1, 0, 0] (empirical); 0.1--0.8 (code) & Validated & \cite{liu2020}: items 4, 5, 7, 8; code uses graded loadings. \\
$\rho_{\Psi}$ & Bifactor correlation & $-0.3$ & Validated & \cite{reis2017}: r = $-0.3$. \\
$\sigma_{o,\Psi}$ & Overload dimension SD & 1.0 (used as /4) & Assumed & Assumes dimension SD of 1.0, regularized to 0.25 in the multivariate normal generating PSS dimensions. \\
$\sigma_{c,\Psi}$ & Controllability dimension SD & 1.0 (used as /4) & Assumed & Assumes dimension SD of 1.0, regularized to 0.25 in the multivariate normal. \\
$\eta_{\Psi}$ & PSS-10 clinical cut-off & 27 & Validated & \cite{cohen1983}: 27--28 clinically elevated. \\
$c_{\Psi}$ & PSS-10 score scale & 6.0 & Assumed & Assumes a scaling factor of 6.0 mapping latent stress to the 0--40 PSS range. \\
$\epsilon_{\Psi}$ & PSS-10 measurement noise SD & 2.0 & Assumed & Assumes measurement error SD of 2.0 scaled by item SD; applied in item response generation. \\
$\gamma_{\Psi}$ & PSS-10 skew parameter & 3.0 & Assumed & Assumes a skew parameter of 3.0 controlling the skew of the score distribution. \\
$\sigma_{\mathrm{bias}}$ & PSS-10 between-person SD & 1.0 & Assumed & Assumes between-person variance SD of 1.0 in the resilience coupling plus noise term; added as static bias. \\
$\beta_{\mathrm{res}}$ & PSS-10 resilience coupling & 3.5 & Assumed & Assumes a coupling strength of 3.5; applied as daily penalty $-0.5\beta_{\mathrm{res}}(\mathfrak{R} - 0.5)$ and init bias. \\
$\phi_{\mathrm{damp}}$ & PSS-10 stress dampening & 1.0 & Assumed & Assumes a dampening factor of 1.0 (no dampening); values below 1 weaken the resource--stress cascade. \\
$\xi_{\mathrm{sens}}$ & PSS-10 sensitivity & 0.5 & Assumed & Assumes a response rate of 0.5 for how quickly perceived stress reacts to events. \\
$\xi_{\mathrm{mom}}$ & PSS-10 momentum weight & 0.3 & Assumed & Assumes 0.3 autocorrelation smoothing of stress perception, reflecting rumination. \\
\midrule
\multicolumn{5}{c}{\textbf{Stress Dynamics}} \\
\midrule
$\nu_c$ & Controllability update rate & 0.05 & Assumed & Assumes a per-event learning rate of 0.05 updating the controllability dimension. \\
$\nu_o$ & Overload update rate & 0.05 & Assumed & Assumes a per-event learning rate of 0.05 updating the overload dimension. \\
\midrule
\multicolumn{5}{c}{\textbf{Output Configuration}} \\
\midrule
$L_{\mathrm{level}}$ & Logging level & INFO & Assumed & Technical configuration; controls verbosity only. \\
$D_{\mathrm{results}}$ & Results output directory & data/processed & Assumed & Technical configuration for output writing. \\
$D_{\mathrm{raw}}$ & Raw output directory & data/raw & Assumed & Technical configuration for output writing. \\
$D_{\mathrm{logs}}$ & Logs output directory & logs & Assumed & Technical configuration for output writing. \\
$F_{\mathrm{ts}}$ & Save time series flag & True & Assumed & Technical configuration; data collection switch. \\
$F_{\mathrm{net}}$ & Save network snapshots flag & True & Assumed & Technical configuration; data collection switch. \\
$F_{\mathrm{sum}}$ & Save summary statistics flag & True & Assumed & Technical configuration; data collection switch. \\
\midrule
\multicolumn{5}{c}{\textbf{Assumption Constants: Coping}} \\
\midrule
$A_{\mathrm{cope}}$ & Coping outcome constants (15) & 0.10; 0.20; 0.05; 0.2; 0.4; 0.15; 0.2; 0.3; $-0.1$; 0.1; $-0.4$; 0.2; 0.3; 0.2; $-0.4$ & Assumed & Assumes model-internal tuning of resource rewards/penalties, protective-factor allocation fraction, affect/resilience change scales, and challenge/hindrance coping outcomes; applied as fixed increments to state variables after each coping result. \\
\midrule
\multicolumn{5}{c}{\textbf{Assumption Constants: Stress}} \\
\midrule
$A_{\mathrm{stress}}$ & Stress constants (21) & 0.10; 0.05; 0.05; 0.10; 0.5; 0.5; 0.05; 0.05; 0.7; 1.3; 1.5; 0.8; 0.2; 0.1; 0.05; 0.01; 0.9; 10.0; 8.0; 12.0; 3.0 & Assumed & Assumes tuning of appraisal weights, baselines, homeostasis rates, event intensity weights, failed-coping multiplier, momentum dynamics, and PSS-10 estimation; applied in the stress update equations. \\
\midrule
\multicolumn{5}{c}{\textbf{Assumption Constants: Resource}} \\
\midrule
$A_{\mathrm{res}}$ & Resource constants (17) & 0.15; 0.05; 0.5; 0.3; 1.3; 0.3; 0.1; 0.2; 0.1; 0.1; 0.1; 0.1; 0.1; 0.5; 0.3; 0.02; 0.02 & Assumed & Assumes tuning of resilience efficiency, thresholds, cost floors, penalties, efficiency gains, social boosts, allocation penalties, and regeneration multipliers; applied in the resource allocation and regeneration equations. \\
\midrule
\multicolumn{5}{c}{\textbf{Assumption Constants: Social}} \\
\midrule
$A_{\mathrm{soc}}$ & Social constants (3) & 0.05; 0.3; 0.1 & Assumed & Assumes tuning of support exchange threshold, social support probability, and exchange boost; applied in interaction mechanics. \\
\midrule
\multicolumn{5}{c}{\textbf{Assumption Constants: Buffering}} \\
\midrule
$A_{\mathrm{buf}}$ & Buffering constants (24) & 0.3; 0.7; 1.0; 1.0; 0.5; 0.1; 0.05; 0.15; 0.2; 0.2; 0.35; 0.30; 0.20; 0.85; 0.05; 0.10; $-0.008$; 0.15; 0.15; 0.50; $-0.2$; 0.5; 0.5; $-0.0$ & Assumed & Assumes tuning of resilience thresholds, volatility priors, initial protective-factor values, buffering coefficients, homeostasis rates, smoothing alpha, and exchange reduction; applied in the buffering and resilience-moderation equations. \\
\midrule
\multicolumn{5}{c}{\textbf{Coupling Constants}} \\
\midrule
$C_{\mathrm{coup}}$ & State-variable couplings (15) & 0.10; 0.20; 0.08; 1.0; 1.0; 0.03; 1.0; 0.15; 0.60; 2.0; 0.15; 3.0; 0.08; 0.08; 0.25 & Assumed & Assumes linear coupling strengths between stress, resilience, affect, resources, and PSS-10 score, plus stress erosion and network-similarity weights; applied as additive coupling terms between state variables each day. \\
\midrule
\multicolumn{5}{c}{\textbf{Core Variables}} \\
\midrule
$\mathfrak{R}_0$ & Baseline resilience & [0,1] & variable & $\mathfrak{R}_0 = \sigma_6\left(\frac{X - \mu_{\mathfrak{R},0}}{\sigma_{\mathfrak{R},0}}\right)$ with $X \sim \mathcal{N}(\mu_{\mathfrak{R},0},\sigma_{\mathfrak{R},0}^2)$ and sigmoid steepness fixed at 6. \\
$A_0$ & Baseline affect & [-1,1] & variable & $A_0 = \tanh\left(\frac{X - \mu_{A,0}}{\sigma_{A,0}}\right)$ with $X \sim \mathcal{N}(\mu_{A,0},\sigma_{A,0}^2)$. \\
$R_0$ & Baseline resources & [0,1] & variable & $R_0 = \sigma_6\left(\frac{X - \mu_{R,0}}{\sigma_{R,0}}\right)$, sigmoid-transformed normal. \\
$\mathfrak{R}_t$ & Current resilience & [0,1] & variable & Update below in Daily Integration Variables; clamped to [0,1]. \\
$A_t$ & Current affect & [-1,1] & variable & Update below in Daily Integration Variables; clamped to [-1,1]. \\
$S_t$ & Current stress & [0,1] & variable & $S_{t+1} = \max(0.03,\; S_t (1 - \delta_{\mathrm{stress}}))$ and smoothed daily from dimensions. \\
$R_t$ & Current resources & [0,1] & variable & $R_{t+1} = R_t + \lambda_R (1 - R_t) - \text{costs}$; regeneration is linear in code. \\
$z$ & Weighted appraisal score & $\mathbb{R}$ & variable & $z = \omega_c \cdot c - \omega_o \cdot o + b$. \\
$\chi$ & Challenge component & [0,1] & variable & $\chi = \sigma_\gamma(z) = 1/(1 + e^{-\gamma z})$. \\
$\zeta$ & Hindrance component & [0,1] & variable & $\zeta = 1 - \chi$. \\
$L$ & Appraised stress load & [0,1] & variable & $L = \mathrm{clamp}\big(0.5 + \delta(\zeta - \chi),\, 0,\, 1\big)$. \\
$\eta_{\mathrm{eff}}$ & Effective stress threshold & [0,1] & variable & $\eta_{\mathrm{eff}} = \mathrm{clamp}(\eta_0 + \eta_\chi \chi - \eta_\zeta \zeta,\, 0,\, 1)$. \\
$p_{\mathrm{coping}}$ & Coping success probability & [0,1] & variable & $p_{\mathrm{coping}} = \mathrm{clamp}\big(p_b + \theta_{\mathrm{cope},\chi}\chi - \theta_{\mathrm{cope},\zeta}\zeta + \delta_{\mathrm{cope,soc}}\bar{A}_{\mathrm{nb}} + \gamma_{rc}\mathfrak{R} + \gamma_{ss}e_{\mathrm{soc}} + \gamma_{sb}\beta_0,\, 0,\, 1\big)$. \\
$c_\Psi$ & PSS-10 controllability dimension & [0,1] & variable & $(c_\Psi, o_\Psi) \sim \mathcal{N}\big((\mu_c,\mu_o), \Sigma\big)$, $\Sigma_{11} = (\sigma_c/4)^2$, $\Sigma_{22} = (\sigma_o/4)^2$, $\Sigma_{12} = \rho_\Psi (\sigma_c\sigma_o)/16$; clamped. \\
$o_\Psi$ & PSS-10 overload dimension & [0,1] & variable & Same bivariate draw; $o_\Psi$ updated per event and via PSS-10 feedback. \\
$c$ & Event controllability attribute & [0,1] & variable & $c \sim \mathrm{clamp}\big(\mathcal{N}(\mu_c, \sigma_c^2), 0, 1\big)$ per stress event. \\
$o$ & Event overload attribute & [0,1] & variable & $o \sim \mathrm{clamp}\big(\mathcal{N}(\mu_o, \sigma_o^2), 0, 1\big)$ per stress event. \\
$s$ & Event intensity (magnitude) & $[0,\infty)$ & variable & $s = \lambda_{\chi}^{\mathrm{int}} \cdot \chi + \lambda_{\zeta}^{\mathrm{int}} \cdot \zeta$ (challenge/hindrance weights 0.7/1.3). \\
$\varepsilon$ & PSS-10 item residual & $\mathbb{R}$ & variable & $\varepsilon \sim \mathcal{N}(0, (\sigma_i \cdot \epsilon_{\Psi})^2)$. \\
$\Psi$ & PSS-10 total score & [0,40] & variable & $\Psi = \sum_{i=1}^{10} y_i$, with $y_i \sim \mathrm{clamp}\big(\mathcal{N}\big(\mu_i + \lambda_{c,i}c'_\Psi + \lambda_{o,i}o'_\Psi,\, (\sigma_i \epsilon_{\Psi})^2\big), 0, 4\big)$ rounded, items 4,5,7,8 reverse-scored; smoothed across days. \\
\midrule
\multicolumn{5}{c}{\textbf{Daily Integration Variables}} \\
\midrule
$n_s$ & Number of subevents per day & $\mathbb{N}$ & variable & $n_s \sim \max(\mathcal{P}(\lambda_s), 1)$. \\
$n_e$ & Number of stress events in day & $\mathbb{N}$ & variable & Count of stress-processing subevents in the day. \\
$\bar{\chi}_d$ & Daily average challenge & [0,1] & variable & $\bar{\chi}_d = \frac{1}{n_e}\sum_{i=1}^{n_e}\chi_i$. \\
$\bar{\zeta}_d$ & Daily average hindrance & [0,1] & variable & $\bar{\zeta}_d = \frac{1}{n_e}\sum_{i=1}^{n_e}\zeta_i$. \\
$\Delta A_p$ & Peer influence on affect & $\mathbb{R}$ & variable & $\Delta A_p = \frac{1}{k}\sum_{j=1}^{k} \alpha_p (A_j - A_t)$, $k = \min(n_{\mathrm{nb}}, k_{\mathrm{influence}})$. \\
$\Delta A_e$ & Event appraisal effect on affect & $\mathbb{R}$ & variable & $\Delta A_e = \alpha_e \bar{\chi}_d (1 - A_t) - \alpha_e \bar{\zeta}_d \max(0.1, A_t + 1)$. \\
$\Delta A_h$ & Homeostatic effect on affect & $\mathbb{R}$ & variable & $\Delta A_h = \lambda_{\mathrm{affect}}^s \cdot (A_0 - A_t)$ with stress- and resource-scaled rate. \\
$\Delta A_{\mathrm{ero}}$ & Stress erosion effect on affect & $\mathbb{R}$ & variable & $\Delta A_{\mathrm{ero}} = -\lambda_{\mathrm{ero}} \cdot S_t \cdot m_{\mathrm{ero}}$ (STRESS\_EROSION\_RATE $\times$ stress $\times$ assumption multiplier). \\
$\Delta A_{\mathrm{res}}$ & Resource effect on affect & $\mathbb{R}$ & variable & $\Delta A_{\mathrm{res}} = \beta_{ra}(R_t - 0.5)\big(1 + (\mathfrak{R}_t - 0.5)\cdot 0.5\big)$; $\beta_{ra} = $ ASSUMPTION\_RESOURCE\_AFFECT\_COUPLING. \\
$A_{t+1}$ & Affect at time t+1 & [-1,1] & variable & $A_{t+1} = \mathrm{clamp}\big(A_t + \Delta A_p + \Delta A_e + \Delta A_h + \Delta A_{\mathrm{ero}} + \Delta A_{\mathrm{res}},\, -1, 1\big)$. \\
$\Delta \mathfrak{R}_{\chi\zeta}$ & Resilience change from coping & $\mathbb{R}$ & variable & Success: $(0.3\chi + 0.1\zeta)(1 - \mathfrak{R}_t)$; failure: $(-0.1\chi - 0.4\zeta)(1 - \mathfrak{R}_t)$; ceiling-damped. \\
$\Delta \mathfrak{R}_p$ & Protective factor boost & $\mathbb{R}$ & variable & $\Delta \mathfrak{R}_p = \sum_{f \in F} e_f (\mathfrak{R}_0 - \mathfrak{R}_t) \theta_{\mathrm{boost}}$. \\
$\Delta \mathfrak{R}_o$ & Overload effect & $\mathbb{R}$ & variable & $\Delta \mathfrak{R}_o = -0.2 \cdot \min(h_c / \eta_{\mathrm{res,overload}},\, 2)$ if $h_c \geq \eta_{\mathrm{res,overload}}$, else 0. \\
$\Delta \mathfrak{R}_s$ & Social support effect & $\mathbb{R}$ & variable & $\Delta \mathfrak{R}_s = \alpha_s$ if support exchange occurred, else 0. \\
$\Delta \mathfrak{R}_{\mathrm{int}}$ & Interaction-frequency boost & $\mathbb{R}$ & variable & $\Delta \mathfrak{R}_{\mathrm{int}} = n_{\mathrm{int}} \cdot 0.005$ per daily interaction. \\
$\mathfrak{R}_{t+1}$ & Resilience at time t+1 & [0,1] & variable & $\mathfrak{R}_{t+1} = \mathrm{clamp}\big(\mathfrak{R}_t + \Delta\mathfrak{R}_{\chi\zeta} + \Delta\mathfrak{R}_p + \Delta\mathfrak{R}_o + \Delta\mathfrak{R}_s + \Delta\mathfrak{R}_{\mathrm{int}} + \lambda_{\mathrm{res}}^s(\mathfrak{R}_0 - \mathfrak{R}_t),\, 0, 1\big)$. \\
$S^{\mathrm{new}}$ & Dimension-derived stress & [0,1] & variable & $S^{\mathrm{new}} = \mathrm{clamp}\big(\tfrac{o_{\Psi}^{m} + (1 - c_{\Psi}^{m})}{2} \cdot \phi_{\mathrm{damp}},\, 0, 1\big)$ with couplings on dimensions. \\
$S_{t+1}$ & Stress at time t+1 & [0,1] & variable & $S_{t+1} = \mathrm{clamp}(0.5 S^{\mathrm{new}} + 0.5 S_t,\, 0, 1)$ then decay $\max(0.03, S_{t+1}(1 - \delta_{\mathrm{stress}}))$. \\
$\mathrm{clamp}(x,a,b)$ & Clamping function & $[a,b]$ & variable & $\min(b, \max(a, x))$, applied to all state variables after every update. \\
\bottomrule
\end{longtable}
```