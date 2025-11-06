```{=latex}
\setlength\LTleft{0pt}
\setlength\LTright{0pt}
\begin{longtable}{ll}
\caption{Summary of model parameters and variables used in the agent-based mental health simulation} \\
\label{tbl-parameters} \\
\toprule
\multicolumn{2}{c}{\textbf{Model Structure}} \\
\midrule
$N$ & Network size (number of agents) \\
$WS_k$ & Mean degree in Watts-Strogatz network topology \\
$WS_p$ & Rewiring probability in Watts-Strogatz network \\
\midrule
\multicolumn{2}{c}{\textbf{Agent Initialization}} \\
\midrule
$\mu_{\mathfrak{R}, \text{0}}$ & Initial resilience mean \\
$\sigma_{\mathfrak{R},\text{0}}$ & Initial resilience standard deviation \\
$\mu_{A, \text{0}}$ & Initial affect mean \\
$\sigma_{A,\text{0}}$ & Initial affect standard deviation \\
$\mu_{R, \text{0}}$ & Initial resources mean \\
$\sigma_{R,\text{0}}$ & Initial resources standard deviation \\
$X$ & Random variable for baseline generation \\
\midrule
\multicolumn{2}{c}{\textbf{Stress Processing}} \\
\midrule
$\omega_c$ & Controllability weight in appraisal \\
$\omega_o$ & Overload weight in appraisal \\
$b$ & Bias term in appraisal function \\
$\gamma$ & Sigmoid steepness parameter \\
$\eta_{\chi}$ & Challenge threshold modifier \\
$\eta_{\zeta}$ & Hindrance threshold modifier \\
$\eta_{\text{0}}$ & Base stress threshold \\
\midrule
\multicolumn{2}{c}{\textbf{Coping}} \\
\midrule
$p_b$ & Base probability for successful coping \\
$\theta_{\text{cope,}\chi}$ & Challenge bonus for coping \\
$\theta_{\text{cope,}\zeta}$ & Hindrance penalty for coping \\
$\delta_{\text{cope,soc}}$ & Social influence on coping \\
\midrule
\multicolumn{2}{c}{\textbf{Affect Dynamics}} \\
\midrule
$\alpha_p$ & Peer influence rate \\
$\alpha_e$ & Event appraisal rate \\
$\lambda_{\text{affect}}$ & Affect homeostasis rate \\
$k$ & Number of neighbors \\
$k_{\text{influence}}$ & Number of influencing neighbors \\
\midrule
\multicolumn{2}{c}{\textbf{Resilience Dynamics}} \\
\midrule
$\lambda_{\text{resilience}}$ & Resilience homeostasis rate \\
$\theta_{\text{boost}}$ & Resilience boost rate \\
$\theta_{\text{boost|cope}}$ & Boost rate for successful coping \\
$\alpha_s$ & Social support rate for resilience \\
$\eta_{\text{res,overload}}$ & Threshold for overload effects \\
$h_c$ & Consecutive hindrances count \\
$F$ & Set of protective factors \\
$e_f$ & Efficacy of factor $f$ \\
\midrule
\multicolumn{2}{c}{\textbf{Resource}} \\
\midrule
$\lambda_R$ & Resource regeneration rate \\
$\beta_{\text{softmax}}$ & Softmax temperature parameter \\
$\beta_a$ & Affect influence parameter \\
$R_{\max}$ & Maximum resources \\
$R_a$ & Available resources \\
$w_f$ & Allocation weight for factor $f$ \\
$r_f$ & Resources allocated to factor $f$ \\
$\gamma_p$ & Protective improvement rate \\
\midrule
\multicolumn{2}{c}{\textbf{Social Network}} \\
\midrule
$\delta_{\text{homophily}}$ & Homophily strength \\
$\eta_{\text{exchange}}$ & Support exchange threshold \\
$s_{ij}$ & Similarity between agents \\
$e_s$ & Support effectiveness \\
$p_{\text{keep}}$ & Connection retention probability \\
$c_{\text{breach}}$ & Stress breach count \\
$\eta_{\text{adapt}}$ & Adaptation threshold \\
$p_{\text{rewire}}$ & Rewiring probability \\
$A_i, A_j$ & Affect values for agents $i,j$ \\
$\mathfrak{R}_i, \mathfrak{R}_j$ & Resilience values for agents $i,j$ \\
\midrule
\multicolumn{2}{c}{\textbf{PSS-10}} \\
\midrule
$\mu_c$ & Controllability dimension mean \\
$\sigma_c$ & Controllability dimension standard deviation \\
$\mu_o$ & Overload dimension mean \\
$\sigma_o$ & Overload dimension standard deviation \\
$\rho_\Psi$ & PSS-10 dimension correlation \\
$\eta_\Psi$ & PSS-10 stress threshold \\
$\Psi_i$ & PSS-10 item response for item $i$ \\
$\lambda_{c,\Psi,i}$ & Factor loading for item $j$ on controllability dimension \\
$\lambda_{o,\Psi,i}$ & Factor loading for item $j$ on overload dimension \\
$\epsilon$ & Measurement error \\
$c_\Psi$ & PSS-10 controllability dimension \\
$o_\Psi$ & PSS-10 overload dimension \\
$\Psi$ & Total PSS-10 score \\
\midrule
\multicolumn{2}{c}{\textbf{Core Variables}} \\
\midrule
$\mathfrak{R}$ & Agent resilience level \\
$A$ & Agent affect level \\
$R$ & Agent resource level \\
$S$ & Agent stress level \\
$z$ & Weighted appraisal score \\
$\chi$ & Challenge component \\
$\zeta$ & Hindrance component \\
$\eta_{\mathrm{eff}}$ & Effective stress threshold \\
$p_{\mathrm{coping}}$ & Probability of successful coping \\
$c$ & Controllability \\
$o$ & Overload \\
$s$ & Event magnitude \\
$\delta$ & Polarity effect strength \\
$L$ & Appraised stress load \\
$A_t$ & Current affect at time t \\
$\mathfrak{R}_t$ & Current resilience at time t \\
$S_t$ & Current stress at time t \\
$\mathfrak{R}_0$ & Baseline resilience \\
\midrule
\multicolumn{2}{c}{\textbf{Daily Integration Variables}} \\
\midrule
$\bar{\chi}_d$ & Daily average challenge \\
$\bar{\zeta}_d$ & Daily average hindrance \\
$\Delta A_p$ & Peer influence effect on affect \\
$\Delta A_e$ & Event appraisal effect on affect \\
$\Delta A_h$ & Homeostatic effect on affect \\
$\Delta \mathfrak{R}_p$ & Protective factor contribution \\
$\delta_{\text{stress}}$ & Stress decay rate \\
$\lambda_s$ & Subevent rate parameter \\
$\mathcal{P}(\lambda)$ & Poisson distribution \\
$\Delta \mathfrak{R}_{\chi\zeta}$ & Resilience change from challenge/hindrance \\
$\Delta \mathfrak{R}_p$ & Resilience boost from protective factors \\
$\Delta \mathfrak{R}_o$ & Resilience change from overload \\
$\Delta \mathfrak{R}_s$ & Resilience change from social support \\
$\Delta A_{t+1}$ & Affect change at time t+1 \\
$\mathfrak{R}_{t+1}$ & Resilience at time t+1 \\
$S_{t+1}$ & Stress level at time t+1 \\
$\mathrm{clamp}(x, a, b)$ & Clamping function to bind the value of $x$ into the range of $[a, b]$ \\
$n_s$ & Number of subevents per day \\
$n_e$ & Number of stress events in day \\
\bottomrule
\end{longtable}
```
