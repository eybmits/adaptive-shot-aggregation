# Budget-Aware Quantum Reservoir Computing: Grouped-Trajectory Training Under Fixed Measurement Budgets

## Abstract

Quantum reservoir computing (QRC) on near-term hardware faces a fundamental measurement budget constraint: the total number of circuit executions $S$ is fixed, forcing a trade-off between shots per timestep $N_\text{shots}$ (measurement precision) and the number of timesteps $T = S / N_\text{shots}$ (training data volume). We introduce **Grouped-Trajectory Training (G-Traj)**, which partitions single-shot measurement trajectories into $G = N_\text{shots}/K$ groups of $K$ averaged shots, multiplying the effective training set by a factor $G$ while retaining controlled noise levels. Across two chaotic benchmarks (Mackey--Glass and Lorenz attractor), G-Traj reduces the normalized NRMSE by 18--19\% relative to standard expectation-value (EV) training when the sample efficiency ratio $\rho_\text{EV} < 3$. We identify $\rho_\text{EV} = (\alpha/d)(T - t_0)$ as a universal order parameter that governs the transition between the EV-sufficient and G-Traj-advantageous regimes, with the performance gap following a power law $\Delta(\rho) \approx 0.12\,\rho^{-0.65}$. These results provide a practical, parameter-free guideline for budget allocation in shot-limited quantum machine learning.

---

## 1. Introduction

Quantum reservoir computing harnesses the exponentially large Hilbert space of a quantum system as a nonlinear feature map for temporal data processing. Unlike classical reservoir computing, however, QRC on real hardware faces a unique constraint: each measurement collapses the quantum state, yielding a **single binary outcome** ($\pm 1$) rather than a continuous expectation value. Obtaining reliable statistics therefore requires repeated circuit executions (*shots*), each consuming part of a finite measurement budget.

We formalize this as the **budget constraint**:

$$S = N_\text{shots} \times T$$

where $S$ is the total number of circuit executions available, $N_\text{shots}$ is the number of shots per input timestep, and $T$ is the number of distinct input timesteps that can be processed. This creates an inherent trade-off:

- **High $N_\text{shots}$, low $T$**: Precise feature estimates but few training samples.
- **Low $N_\text{shots}$, high $T$**: Many training samples but noisy features.

The standard approach (**EV training**) averages all shots per timestep to approximate expectation values, then trains a Ridge regression. This discards the shot-level structure entirely. We ask: *can the individual measurement trajectories themselves be leveraged to improve learning, especially when the training set is small?*

We answer affirmatively by introducing **Grouped-Trajectory Training (G-Traj)**, which adaptively partitions shots into groups, averages within groups, and trains on the resulting multiplied dataset. We show that the **sample efficiency ratio** $\rho_\text{EV}$ acts as a universal order parameter determining when G-Traj outperforms EV training, independent of the target time series.

---

## 2. Methods

### 2.1 Quantum Reservoir

We simulate a 6-qubit quantum reservoir ($\dim \mathcal{H} = 64$) with the following architecture:

| Component | Specification |
|-----------|---------------|
| Qubits | $n = 6$ |
| Input encoding | $R_y(W_i \cdot u_t)$ per qubit, $W_i \in [-\pi, \pi]$ fixed random |
| Entangling layer | Ring of CNOT gates + random single-qubit rotations ($R_x, R_y, R_z$) |
| Circuit depth | 5 layers |
| Observables | Pauli $Z_i$ and $X_i$ on all qubits, $M = 12$ |

At each input timestep $t$, the reservoir maps the scalar input $u_t$ to a quantum state $|\psi_t\rangle = U_\text{res} \cdot U_\text{in}(u_t)|0\rangle^{\otimes 6}$, from which $M = 12$ observables are measured.

**Measurement model.** Each shot $n$ produces binary outcomes $m_{t,i}^{(n)} \in \{+1, -1\}$. We apply a controlled noise model:

$$\tilde{m}_{t,i}^{(n)} = \xi_i \cdot m_{t,i}^{(n)} + \sigma \cdot \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, 1)$$

where $\xi_i$ flips the sign with probability $\min(\sigma, 0.5)$, and $\sigma$ controls the overall noise level.

**Temporal processing.** A leaky integrator with decay rate $\eta = 0.5$ processes the shot-level measurements:

$$r_{t,i}^{(n)} = (1 - \eta)\, r_{t-1,i}^{(n)} + \eta\, \tilde{m}_{t,i}^{(n)}$$

The feature vector at timestep $t$ for shot $n$ is constructed by concatenating $L = 20$ past integrator states plus a bias term:

$$\phi_t^{(n)} = \bigl[r_t^{(n)},\; r_{t-1}^{(n)},\; \ldots,\; r_{t-L+1}^{(n)},\; 1\bigr] \in \mathbb{R}^d, \quad d = L \cdot M + 1 = 241$$

### 2.2 Training Algorithms

All algorithms use Ridge regression with fixed regularization $\lambda = 10.0$, a burn-in of $t_0 = 30$ timesteps, and a 70/30 train/test split ($\alpha = 0.7$).

#### Algorithm 1: EV Training (Expectation Value)

Average all $N_\text{shots}$ feature vectors per timestep to approximate the expectation value:

$$\bar{\phi}_t = \frac{1}{N_\text{shots}} \sum_{n=1}^{N_\text{shots}} \phi_t^{(n)}$$

Train Ridge regression on $\{(\bar{\phi}_t, y_t)\}_{t=t_0+1}^{T_\text{train}}$, yielding $T_\text{train} = \alpha(T - t_0)$ training samples.

**Effective training samples:** $T_\text{train}$ (one per timestep).

#### Algorithm 2: Trajectory Training (Raw, $K = 1$)

Treat each shot as an independent training sample. Stack all $N_\text{shots}$ trajectories:

$$\mathcal{D}_\text{traj} = \bigl\{(\phi_t^{(n)}, y_t) : n = 1, \ldots, N_\text{shots},\; t \in \text{train}\bigr\}$$

**Effective training samples:** $N_\text{shots} \times T_\text{train}$.

At inference, average predictions across shots: $\hat{y}_t = \frac{1}{N_\text{shots}} \sum_n w^\top \phi_t^{(n)}$.

#### Algorithm 3: Grouped-Trajectory Training (G-Traj)

Partition the $N_\text{shots}$ shots into $G = N_\text{shots} / K$ groups of size $K$, and average within each group:

$$\phi_t^{(g)} = \frac{1}{K} \sum_{n \in \text{group}_g} \phi_t^{(n)}, \quad g = 1, \ldots, G$$

Stack grouped trajectories as training samples ($G \times T_\text{train}$ total). The group size $K$ is selected automatically via validation over all divisors of $N_\text{shots}$.

**Effective training samples:** $G \times T_\text{train} = (N_\text{shots}/K) \times T_\text{train}$.

**Key property:** G-Traj interpolates continuously between raw trajectory training ($K = 1$, maximum samples, maximum noise) and EV training ($K = N_\text{shots}$, minimum samples, minimum noise). The automatic $K$-selection finds the optimal trade-off without manual tuning.

### 2.3 Order Parameter: Sample Efficiency Ratio

We define the **sample efficiency ratio** for EV training:

$$\rho_\text{EV} = \frac{\alpha}{d}\,(T - t_0) = \frac{\alpha}{d}\left(\frac{S}{N_\text{shots}} - t_0\right)$$

This ratio measures whether the EV regression problem is over- or underdetermined:

| Regime | Condition | Interpretation |
|--------|-----------|----------------|
| Overdetermined | $\rho_\text{EV} \gg 1$ | Many more samples than features; EV sufficient |
| Critical | $\rho_\text{EV} \approx 1$ | Samples $\approx$ features; G-Traj most beneficial |
| Underdetermined | $\rho_\text{EV} < 1$ | Fewer samples than features; severe overfitting risk |

With our parameters ($d = 241$, $t_0 = 30$, $\alpha = 0.7$), the critical shot count where $\rho_\text{EV} = 1$ is:

$$N^* = \frac{S}{t_0 + d/\alpha} \approx \frac{S}{374.3}$$

### 2.4 Evaluation Metrics

**NRMSE** (Normalized Root Mean Squared Error):

$$\text{NRMSE} = \frac{\sqrt{\frac{1}{T_\text{test}} \sum_t (\hat{y}_t - y_t)^2}}{\sigma_y}$$

where $\sigma_y$ is the standard deviation of the true test signal. NRMSE $< 1$ indicates the model outperforms naive prediction of the mean.

**Signal quality ratio** (Std ratio): $\text{SR} = \sigma_{\hat{y}} / \sigma_y$. Values near 1.0 indicate the prediction preserves the amplitude of the true signal. Values $\ll 1$ indicate pathological amplitude damping.

### 2.5 Benchmark Time Series

**Mackey--Glass equation** (delay differential equation, $\tau = 17$):

$$\frac{dx}{dt} = \beta \frac{x(t - \tau)}{1 + x(t-\tau)^{10}} - \gamma x(t)$$

with $\beta = 0.2$, $\gamma = 0.1$. Produces mildly chaotic dynamics with characteristic oscillatory structure.

**Lorenz attractor** ($\sigma_L = 10$, $\rho_L = 28$, $\beta_L = 8/3$, $dt = 0.02$):

$$\dot{x} = \sigma_L(y - x), \quad \dot{y} = x(\rho_L - z) - y, \quad \dot{z} = xy - \beta_L z$$

The $x$-component is used, producing fully chaotic dynamics. Both series are min-max normalized to $[0, 1]$.

---

## 3. Results

### 3.1 Benchmark Comparison

We evaluate all three algorithms at fixed operating points chosen to be in the underdetermined regime ($\rho_\text{EV} < 3$). Results are averaged over 8 random seeds.

**Table 1.** Benchmark results (8 seeds). Gap = NRMSE$_\text{EV}$ $-$ NRMSE$_\text{G-Traj}$.

| Dataset | $S$ | $N_\text{shots}$ | $\sigma$ | $\rho_\text{EV}$ | EV NRMSE | Traj NRMSE | G-Traj NRMSE | Gap | Improvement |
|---------|-----|---------|------|------|----------|----------|----------|------|------|
| Mackey--Glass | 20,000 | 40 | 0.03 | 1.37 | $0.592 \pm 0.085$ | $0.810 \pm 0.048$ | $0.478 \pm 0.087$ | +0.114 | 19% |
| Lorenz | 30,000 | 40 | 0.0 | 2.09 | $0.581 \pm 0.067$ | $0.737 \pm 0.045$ | $0.478 \pm 0.066$ | +0.103 | 18% |

Key observations:
- **G-Traj consistently outperforms EV** by 18--19% in NRMSE.
- **Raw trajectory training is pathological**: NRMSE $> 0.7$ in both cases, with severe amplitude damping (Std ratio $\approx 0.2$--$0.35$).
- **G-Traj preserves signal quality**: Std ratio $\approx 0.82$, compared to EV's $\approx 1.02$. G-Traj predictions retain $\sim$80% of the true signal amplitude while achieving lower error.

*(See Figure 1: fig\_benchmarks.pdf)*

### 3.2 Phase Diagrams

We sweep a $7 \times 7$ grid over $N_\text{shots} \in \{10, 15, 20, 25, 30, 40, 50\}$ and $\sigma \in \{0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10\}$, with 5 seeds per configuration, for both datasets.

**Table 2.** Selected phase sweep results (5 seeds).

| Dataset | $N$ | $\sigma$ | $\rho_\text{EV}$ | EV NRMSE | G-Traj NRMSE | Gap |
|---------|-----|------|------|----------|----------|------|
| Mackey--Glass | 20 | 0.0 | 2.82 | $0.548 \pm 0.151$ | $0.491 \pm 0.128$ | +0.057 |
| Mackey--Glass | 30 | 0.03 | 1.85 | $0.658 \pm 0.101$ | $0.557 \pm 0.136$ | +0.101 |
| Mackey--Glass | 40 | 0.03 | 1.37 | $0.574 \pm 0.084$ | $0.446 \pm 0.082$ | +0.128 |
| Mackey--Glass | 50 | 0.05 | 1.07 | $0.664 \pm 0.095$ | $0.581 \pm 0.097$ | +0.083 |
| Lorenz | 20 | 0.0 | 4.27 | $0.554 \pm 0.046$ | $0.554 \pm 0.046$ | +0.000 |
| Lorenz | 30 | 0.0 | 2.82 | $0.491 \pm 0.057$ | $0.479 \pm 0.064$ | +0.012 |
| Lorenz | 40 | 0.0 | 2.09 | $0.562 \pm 0.074$ | $0.444 \pm 0.019$ | +0.118 |
| Lorenz | 50 | 0.0 | 1.66 | $0.572 \pm 0.068$ | $0.494 \pm 0.045$ | +0.078 |

The phase diagrams (Figure 3) reveal a clear structure:
- **High $N_\text{shots}$ (low $\rho_\text{EV}$):** Strong G-Traj advantage (dark red, gap up to +0.19).
- **Low $N_\text{shots}$ (high $\rho_\text{EV}$):** Algorithms converge (gap $\approx 0$, white/blue).
- The **phase boundary** (gap $= 0$) aligns with constant-$\rho_\text{EV}$ contours, confirming $\rho_\text{EV}$ as the governing parameter.
- The boundary position is consistent between datasets: $\rho_\text{EV} \approx 3$ for Mackey--Glass and Lorenz alike.

*(See Figure 3: fig\_phase.pdf)*

### 3.3 Universal Collapse and Scaling Law

When all $(N_\text{shots}, \sigma)$ configurations from both datasets are plotted as Gap vs. $\rho_\text{EV}$, the data from both benchmarks collapse onto a **single universal curve** (Figure 2):

$$\boxed{\Delta(\rho_\text{EV}) \approx 0.124 \cdot \rho_\text{EV}^{-0.65}}$$

This power-law decay has the following properties:

| Property | Value |
|----------|-------|
| Prefactor $C$ | 0.124 |
| Exponent $\beta$ | 0.65 |
| Crossover $\rho^*$ (Gap $\to 0$) | $\approx 3$ |
| Maximum gap observed | +0.325 (MG, $\rho_\text{EV} = 1.37$, $\sigma = 0.10$) |

The **dataset-independence** of this collapse is the central result: the performance gap between EV and G-Traj is governed entirely by $\rho_\text{EV}$, regardless of whether the target is a delay-differential equation (Mackey--Glass) or a chaotic ODE (Lorenz).

*(See Figure 2: fig\_collapse.pdf)*

### 3.4 Signal Quality Analysis

Raw trajectory training exhibits pathological amplitude damping:

| Algorithm | Std Ratio (MG) | Std Ratio (Lorenz) | Interpretation |
|-----------|----------------|-------------------|----------------|
| EV | $\approx 1.02$ | $\approx 1.03$ | Correct amplitude |
| G-Traj | $\approx 0.82$ | $\approx 0.82$ | Slight undershoot, good quality |
| Traj (raw) | $\approx 0.23$ | $\approx 0.35$ | Severe damping, flat predictions |

Raw trajectory training with $K = 1$ produces predictions with only 20--35% of the true signal amplitude. This occurs because Ridge regression, faced with high feature variance from single-shot noise, learns overly conservative (mean-regressing) weights. G-Traj avoids this by averaging within groups before training, reducing per-sample variance while retaining the sample multiplication advantage.

---

## 4. Discussion

### 4.1 Why G-Traj Works

The advantage of G-Traj arises from a fundamental trade-off between **sample count** and **per-sample noise**:

| Configuration | Noise per sample | Training samples | Regime |
|--------------|-----------------|-----------------|--------|
| $K = N_\text{shots}$ (EV) | Minimal ($\propto 1/\sqrt{N}$) | $T_\text{train}$ | Data-limited |
| $K = 1$ (raw Traj) | Maximal (single-shot) | $N \cdot T_\text{train}$ | Noise-limited |
| $K = K^*$ (G-Traj) | Moderate ($\propto 1/\sqrt{K}$) | $(N/K) \cdot T_\text{train}$ | Optimal |

When $\rho_\text{EV} < 3$, EV training has barely enough samples to fit the $d = 241$-dimensional regression. G-Traj alleviates this by:

1. **Sample multiplication:** With $G = N/K$ groups, the effective training set grows by a factor $G$.
2. **Noise reduction:** Intra-group averaging reduces per-sample variance by a factor $\sqrt{K}$.
3. **Automatic adaptation:** Validation-based $K$-selection finds the optimal noise-data trade-off without manual tuning.

### 4.2 Why Raw Trajectory Training Fails

Raw trajectory training ($K = 1$) always fails because:
- Each single-shot feature vector has variance $\mathcal{O}(1)$ (binary $\pm 1$ measurements after leaky integration).
- Ridge regression minimizes $\|Xw - y\|^2 + \lambda\|w\|^2$. With high feature variance, the regularization dominates, pushing weights toward zero.
- The result is amplitude-damped predictions that approximate the training mean, regardless of the input.

This effect is robust across datasets and hyperparameter choices, making raw trajectory training unsuitable as a practical algorithm.

### 4.3 $\rho_\text{EV}$ as Universal Order Parameter

The ratio $\rho_\text{EV} = (\alpha/d)(T - t_0)$ captures the essential physics of the budget allocation problem:

- It is **computable a priori** from the experimental parameters $(S, N_\text{shots}, d, t_0, \alpha)$ without running any experiment.
- It **determines the algorithmic choice**: use G-Traj when $\rho_\text{EV} < 3$, EV otherwise.
- It is **dataset-independent**: the same crossover point $\rho^* \approx 3$ holds for both Mackey--Glass and Lorenz.

The interpretation is physical: $\rho_\text{EV}$ counts how many training samples EV has per feature dimension. When this ratio drops below $\sim$3, the regression problem becomes sufficiently ill-conditioned that sample multiplication via grouped trajectories yields a net benefit despite the increased per-sample noise.

### 4.4 Practical Guidelines

Based on our results, we recommend the following decision rule for budget allocation in shot-limited QRC:

1. **Compute** $\rho_\text{EV} = (\alpha/d)(S/N_\text{shots} - t_0)$ for the planned experimental configuration.
2. **If $\rho_\text{EV} > 3$:** Use standard EV training. G-Traj will automatically select $K = N_\text{shots}$ and reduce to EV.
3. **If $\rho_\text{EV} < 3$:** Use G-Traj. Expected improvement: $\Delta \approx 0.12 \cdot \rho^{-0.65}$ in NRMSE.
4. **Never use raw trajectory training** ($K = 1$): it produces amplitude-damped predictions regardless of the regime.
5. **G-Traj is always safe to use**: in the overdetermined regime, it degrades gracefully to EV (no performance penalty).

### 4.5 Limitations

- The scaling law $\Delta \propto \rho^{-0.65}$ is empirical, fitted to two datasets. Validation on additional time series (e.g., NARMA, chaotic maps) would strengthen the universality claim.
- The exponent $\beta = 0.65$ and prefactor $C = 0.12$ may depend on the reservoir architecture (qubit count, depth, observable set).
- The noise parameter $\sigma$ does not appear explicitly in $\rho_\text{EV}$, though the phase diagrams show that moderate noise ($\sigma \approx 0.03$) can amplify the G-Traj advantage.

---

## 5. Figure Descriptions

**Figure 1** (`fig_benchmarks.pdf`): Benchmark comparison on Mackey--Glass ($S = 20\text{k}$, $N = 40$, $\sigma = 0.03$) and Lorenz ($S = 30\text{k}$, $N = 40$, $\sigma = 0.0$). Left: test time traces comparing true signal (black), EV prediction (gray dashed), Traj prediction (light gray), and G-Traj prediction (red). Center: NRMSE bar charts with standard deviations over 8 seeds. Right: signal quality (Std ratio) bars; the dashed line at 1.0 marks perfect amplitude preservation. G-Traj (red) achieves the lowest NRMSE while maintaining $\sim$80% of the true signal amplitude.

**Figure 2** (`fig_collapse.pdf`): Universal collapse of the performance gap $\Delta = \text{NRMSE}_\text{EV} - \text{NRMSE}_\text{G-Traj}$ as a function of the sample efficiency ratio $\rho_\text{EV}$. Black circles: Mackey--Glass; gray open squares: Lorenz. Red curve: power-law fit $\Delta = 0.124 \cdot \rho^{-0.65}$. Data from both datasets collapse onto a single curve, with the crossover at $\rho_\text{EV} \approx 3$ where the gap vanishes.

**Figure 3** (`fig_phase.pdf`): Phase diagrams showing the performance gap as a function of $\sigma$ (x-axis) and $N_\text{shots}$ (y-axis) for Mackey--Glass (left) and Lorenz (right). Smooth filled contours with diverging colormap (red: G-Traj wins, blue: EV wins). Black line: empirical phase boundary. Dashed gray lines: constant-$\rho_\text{EV}$ contours at $\rho = 2, 3, 5$. Black dots: measured grid points. The phase boundary aligns with $\rho_\text{EV} \approx 3$ in both datasets.

---

## 6. Summary of Key Numbers

| Parameter | Value |
|-----------|-------|
| Qubits | 6 |
| Observables ($M$) | 12 (Pauli $Z$ + $X$) |
| Feature dimension ($d$) | 241 |
| Window length ($L$) | 20 |
| Burn-in ($t_0$) | 30 |
| Train/test split ($\alpha$) | 0.7 |
| Ridge regularization ($\lambda$) | 10.0 |
| Leaky integrator decay ($\eta$) | 0.5 |
| | |
| **MG benchmark NRMSE** | |
| EV | $0.592 \pm 0.085$ |
| G-Traj | $0.478 \pm 0.087$ |
| Improvement | 19% ($\Delta = +0.114$) |
| | |
| **Lorenz benchmark NRMSE** | |
| EV | $0.581 \pm 0.067$ |
| G-Traj | $0.478 \pm 0.066$ |
| Improvement | 18% ($\Delta = +0.103$) |
| | |
| **Scaling law** | $\Delta = 0.124 \cdot \rho^{-0.65}$ |
| Crossover | $\rho_\text{EV} \approx 3$ |
| Max gap observed | +0.325 |
| | |
| **Phase sweep** | $7 \times 7$ grid, 5 seeds, 2 datasets |
| $N_\text{shots}$ range | 10--50 |
| $\sigma$ range | 0.0--0.10 |
