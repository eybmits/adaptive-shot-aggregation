# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Budget-aware Quantum Reservoir Computing (QRC) research project. Investigates three training algorithms (EV, Trajectory, Grouped-Trajectory) under a fixed measurement budget constraint $S = N_\text{shots} \times T$. The central finding is that **G-Traj reduces NRMSE by 18–19%** vs EV when the sample efficiency ratio $\rho_\text{EV} < 3$, following a universal power law $\Delta \approx 0.124 \cdot \rho^{-0.65}$.

## File Structure

```
qrc_budget.py          # Core implementation (~1400 lines, single-file)
plot_paper.py           # Figure 1: Benchmark comparison (2×3 grid)
plot_collapse.py        # Figure 2: Universal collapse (Gap vs ρ_EV)
plot_phase.py           # Figure 3: Phase diagrams (contourf, Nature-style)
make_table.py           # Generate results tables (Markdown + LaTeX)
theorie.md              # Full theoretical framework (14 sections)
results/
  paper_report.md       # Paper-ready research report
  paper_results.md      # Numerical results tables
  fig_benchmarks.pdf    # Figure 1
  fig_collapse.pdf      # Figure 2
  fig_phase.pdf         # Figure 3
  data_benchmarks.pkl   # Cached training data (8 seeds)
  data_collapse.pkl     # Cached collapse data (9×7×5 seeds)
  data_phase.pkl        # Cached phase data (7×7×5 seeds)
```

## Running

```bash
# Generate individual figures (loads from cache if available)
python3 plot_paper.py       # → results/fig_benchmarks.pdf
python3 plot_collapse.py    # → results/fig_collapse.pdf
python3 plot_phase.py       # → results/fig_phase.pdf
python3 make_table.py       # → results/paper_results.md

# Full pipeline (old, runs everything from scratch)
python3 qrc_budget.py
```

All plot scripts use **pickle-based caching**: first run computes data and saves `.pkl`, subsequent runs load instantly for fast plot iteration. Delete `results/data_*.pkl` to force recomputation.

## Dependencies

Standard scientific Python stack (no requirements.txt): `numpy`, `scipy`, `sklearn`, `matplotlib`. All assumed pre-installed.

## Architecture: qrc_budget.py

The core file is organized into 8 sections:

1. **Quantum Gates & Statevector Simulator** (lines ~31–91) — Custom NumPy-based gate implementations (Rx, Ry, Rz, CNOT) operating on $2^6$ statevectors. No external quantum framework.

2. **QuantumReservoir class** (lines ~93–222) — 6-qubit reservoir with Ry input encoding, fixed random unitary (ring of CNOTs + random rotations), Pauli Z/X measurement ($M = 12$ observables). Core methods: `prepare_state()`, `get_expectation_values()`, `sample_single_shot()`.

3. **Benchmark Generators** (lines ~224–279) — `mackey_glass()` (delay-differential, $\tau=17$) and `narma10()` time series.

4. **Noise Model** (lines ~281–311) — `apply_noise()`: sign flips with $p = \min(\sigma, 0.5)$ + additive Gaussian $\varepsilon \sim \mathcal{N}(0, 1)$.

5. **Reservoir Pipeline** (lines ~314–371) — `run_reservoir_pipeline()`: state prep → noisy sampling → leaky integrator ($\eta = 0.5$) → windowed features ($L = 20$, $d = L \times M + 1 = 241$).

6. **Training Protocols** (lines ~373–572):
   - **`ev_training()`** — Algorithm 1: Average shots → Ridge regression. $T_\text{train}$ samples.
   - **`trajectory_training()`** — Algorithm 2: Stack all shots ($K = 1$) → Ridge → average predictions. $N \times T_\text{train}$ samples. Always pathological (Std ratio ≈ 0.2).
   - **`grouped_trajectory_training()`** — Algorithm 3: Group $K$ shots (auto-selected via validation) → Ridge. $(N/K) \times T_\text{train}$ samples. Best algorithm when $\rho_\text{EV} < 3$.

7. **`compute_rho_ev()`** (line 578) — Order parameter: $\rho_\text{EV} = (\alpha/d)(S/N_\text{shots} - t_0)$.

8. **Phase Diagram & Visualization** (lines ~584–1106) — Sweep and plotting functions.

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $n$ (qubits) | 6 | Hilbert space $2^6 = 64$ |
| $M$ (observables) | 12 | Pauli Z + X per qubit |
| $d$ (features) | 241 | $L \times M + 1$ |
| $L$ (window) | 20 | Temporal memory |
| $t_0$ (burn-in) | 30 | Discarded timesteps |
| $\alpha$ (train split) | 0.7 | 70% training |
| $\lambda$ (Ridge) | 10.0 | Regularization |
| $\eta$ (leaky integrator) | 0.5 | Decay rate |

## Key Results

| Metric | Value |
|--------|-------|
| MG: EV NRMSE | $0.592 \pm 0.085$ (8 seeds) |
| MG: G-Traj NRMSE | $0.478 \pm 0.087$ (19% better) |
| Lorenz: EV NRMSE | $0.581 \pm 0.067$ (8 seeds) |
| Lorenz: G-Traj NRMSE | $0.478 \pm 0.066$ (18% better) |
| Scaling law | $\Delta = 0.124 \cdot \rho^{-0.65}$ |
| Crossover | $\rho_\text{EV} \approx 3$ |

## Plot Style Convention

All figures use a consistent publication style:
- **Colors:** True = `#000000`, EV = `#555555` (///), Traj = `#aaaaaa` (...), G-Traj = `#c0392b` (solid red)
- **Font:** Helvetica Neue / sans-serif, 8pt base
- **Output:** PDF only, 300 DPI, `pdf.fonttype = 42`
- **No** panel labels (a, b, c), **no** titles
- Config annotations with white background boxes

## Data Flow

```
Time series → Ry encoding → Statevector → Shot sampling
→ Noise (σ) → Leaky integrator (η) → Windowed features (L=20, d=241)
→ Training (EV / Traj / G-Traj) → Ridge regression → NRMSE evaluation
```

## Budget Constraint

$$S = N_\text{shots} \times T$$

- High $N$, low $T$: precise features, few training samples → EV sufficient
- Low $N$, high $T$: noisy features, many training samples → EV sufficient
- Moderate $N$ where $\rho_\text{EV} < 3$: G-Traj wins via sample multiplication

## Benchmarks

| Dataset | $S$ | $N$ | $\sigma$ | $\rho_\text{EV}$ |
|---------|-----|-----|----------|---------|
| Mackey-Glass | 20,000 | 40 | 0.03 | 1.37 |
| Lorenz | 30,000 | 40 | 0.0 | 2.09 |

Phase sweep: $N \in \{10,15,20,25,30,40,50\}$, $\sigma \in \{0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10\}$, 5 seeds.
Collapse sweep: $N \in \{5,8,10,15,20,25,30,40,50\}$, same $\sigma$ range, 5 seeds.
