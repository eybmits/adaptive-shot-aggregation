# Technical Report (Paper Draft)
## Budget-Aware Quantum Reservoir Computing with NARMA-10 Stabilization

### Status and Scope
This report documents the current paper-ready state of the project using the existing figure artifacts:
- `/Users/superposition/Coding/qrc_budget/results/fig_benchmarks.pdf`
- `/Users/superposition/Coding/qrc_budget/results/fig_collapse.pdf`
- `/Users/superposition/Coding/qrc_budget/results/fig_phase.pdf`

The goal is to provide a fair and technically defensible narrative after introducing NARMA-10 and the corresponding stabilization in grouped-trajectory training.

---

## 1. Problem Setting and Core Claim

Under a fixed measurement budget

\[
S = N_{\text{shots}} \cdot T,
\]

quantum reservoir training must trade off:
- larger `N_shots` (cleaner features, fewer time points), versus
- larger `T` (more samples, noisier features).

The central question is whether grouped shot trajectories improve generalization relative to standard EV training when sample efficiency is limited.

We evaluate three targets:
- Mackey-Glass,
- Lorenz attractor,
- NARMA-10.

The two key outputs are:
1. predictive error (NRMSE),
2. signal fidelity (std-ratio = `std(y_pred)/std(y_true)`).

This second metric is critical for NARMA-10, where a method can appear better in NRMSE by collapsing to a smoothed mean-like prediction.

---

## 2. Methods and Fairness Protocol

### 2.1 Reservoir and training backbone
- 6-qubit random-circuit reservoir, fixed random parameters per seed.
- Readout: Ridge regression (same model class and regularization protocol across methods).
- Same train/test split convention and burn-in conventions across all comparisons.

### 2.2 Compared training modes
- `EV`: average all shots first, then train.
- `Traj`: stack raw trajectories (`K=1`).
- `G-Traj` (grouped): average shots within groups and stack groups.

### 2.3 NARMA-10-specific stabilization (why it was necessary)
For NARMA-10, naive grouped training can improve NRMSE while severely shrinking output variance (mean-line effect). To avoid a misleading win, we introduced a guarded NARMA fix for grouped training in benchmark evaluation:
- group-size selection on **aggregated validation prediction** (same inference mode as test),
- std-aware objective in selection,
- bounded affine calibration on validation only,
- explicit guardrail: calibration is rejected if validation NRMSE degradation exceeds the configured tolerance.

This is a methodological correction, not a cosmetic one: it enforces that NRMSE gains are not achieved by destroying dynamical amplitude.

---

## 3. Figure-by-Figure Analysis

## 3.1 Figure 1: Benchmark comparison
**File:** `/Users/superposition/Coding/qrc_budget/results/fig_benchmarks.pdf`

### What this figure shows
Three rows (Mackey-Glass, Lorenz, NARMA-10) with:
- time trace,
- NRMSE bars,
- std-ratio bars.

For NARMA-10, the shown time window was shortened to improve visual interpretability (fixed interval in script), avoiding unreadable full-length overlays.

### Quantitative summary (8 seeds)
Using the current benchmark cache (`data_benchmarks_narma_fix.pkl`):

- Mackey-Glass:
  - EV: `0.592 ± 0.085`
  - Traj: `0.810 ± 0.048`
  - G-Traj: `0.434 ± 0.063`
  - Gap (EV - G): `+0.158` (26.7% relative improvement)
  - Std-ratio EV / Traj / G: `1.019 / 0.224 / 0.730`

- Lorenz attractor:
  - EV: `0.581 ± 0.067`
  - Traj: `0.737 ± 0.045`
  - G-Traj: `0.444 ± 0.067`
  - Gap: `+0.137` (23.6% relative improvement)
  - Std-ratio EV / Traj / G: `1.032 / 0.358 / 0.772`

- NARMA-10:
  - EV: `1.080 ± 0.072`
  - Traj: `1.001 ± 0.030`
  - G-Traj (stabilized): `0.978 ± 0.050`
  - Gap: `+0.102` (9.4% relative improvement)
  - Std-ratio EV / Traj / G: `0.854 / 0.050 / 0.695`

### Interpretation
- Raw trajectory training is consistently variance-collapsed.
- Grouped training is best in NRMSE across all three tasks.
- The NARMA row now reflects a fairer regime: G-Traj still wins, but with materially better amplitude behavior than the pathological trajectory baseline.

---

## 3.2 Figure 2: Empirical collapse (gap vs \(\rho_{EV}\))
**File:** `/Users/superposition/Coding/qrc_budget/results/fig_collapse.pdf`

### What this figure shows
Each point is a configuration-level mean gap:
\[
\Delta = \text{NRMSE}_{EV} - \text{NRMSE}_{G\text{-}Traj}
\]
plotted against
\[
\rho_{EV}=\frac{\alpha}{d}(T-t_0).
\]

Positive \(\Delta\) means G-Traj outperforms EV.

### Quantitative summary (current collapse cache)
- total points: `189`
- \(\rho\)-range: `1.07` to `28.96`
- gap range: `-0.008` to `0.195`
- fitted law:
\[
\Delta(\rho) \approx 0.128\,\rho^{-0.77}
\]

Dataset-level mean gaps in this collapse set:
- Mackey-Glass: `0.044`
- Lorenz: `0.030`
- NARMA-10: `0.022`

### Interpretation
- The dominant trend is monotone decay of benefit with increasing \(\rho_{EV}\): when EV has enough effective samples, grouping adds less.
- The fitted exponent (`~0.77`) supports a smooth scaling regime rather than a hard threshold.
- Small negative outliers exist near high-\(\rho\) regimes, but the global behavior remains positive and decaying.

### Fairness point
This collapse figure is used as the **within-setup scaling statement** across datasets/configurations. Its value is in consistent cross-task structure under one protocol and one order parameter, not in claiming architecture-agnostic universality.

---

## 3.3 Figure 3: Phase diagrams
**File:** `/Users/superposition/Coding/qrc_budget/results/fig_phase.pdf`

### What this figure shows
Gap heatmaps over (`N_shots`, `sigma`) for each dataset, with overlaid guides and a phase boundary around zero-gap regions.

### Quantitative summary (current phase cache)
Grid size: `7 x 7` per dataset.

- Mackey-Glass:
  - mean gap: `0.057`
  - max gap: `0.195`
  - min gap: `-0.001`
  - positive-cell fraction: `71.4%`

- Lorenz:
  - mean gap: `0.038`
  - max gap: `0.173`
  - min gap: `0.000`
  - positive-cell fraction: `49.0%`

- NARMA-10:
  - mean gap: `0.027`
  - max gap: `0.069`
  - min gap: `-0.008`
  - positive-cell fraction: `91.8%`

### Interpretation
- Benefit regions are broad for Mackey-Glass and especially NARMA-10, with larger gains in more sample-limited corners.
- Lorenz shows a narrower but still meaningful advantage region.
- The heatmap and collapse are consistent: gains are regime-dependent and linked to effective sample pressure.

---

## 4. Why this overall story is technically fair

### 4.1 No metric gaming
We do not accept NRMSE alone as success for NARMA-10; std-ratio is reported explicitly. This blocks the trivial “predict the mean” failure mode.

### 4.2 Method change is justified and scoped
The NARMA stabilization is introduced exactly where pathology appears. It is a targeted correction to evaluation validity, not an across-the-board hyperparameter inflation.

### 4.3 Separation of claims
- Figure 1 supports the practical benchmark claim including stabilized NARMA behavior.
- Figures 2 and 3 support the structural regime claim (budget/sample-efficiency geometry and scaling).

This separation is methodologically clean: practical stabilization for task validity, plus global regime analysis for theory-facing claims.

### 4.4 External-validity boundary is explicit
Current evidence is intentionally narrow in architecture/backend breadth:
- one circuit family (depth-5 random reservoir with ring-CNOT entanglers),
- one hardware backend (`ibm_marrakesh`) in the hardware section.

Therefore, any quoted crossover scale in \(\rho_{EV}\) (including values around \(\rho_{EV}\approx 3\) when referenced) is treated as an empirical marker of this setup, not as a universal constant. The power-law fit is a within-family regularity; cross-family and cross-backend universality remains open.

---

## 5. Additional Evidence for NARMA Stabilization (supplementary)
From `data_narma_fix_compare_s8.pkl` (N=60, sigma=0.03, S=50000, 8 seeds):

- EV: NRMSE `0.913 ± 0.100`, SR `0.893`
- G-old: NRMSE `0.867 ± 0.044`, SR `0.279`
- G-fix: NRMSE `0.860 ± 0.081`, SR `0.770`

Key point: the fix improves or preserves NRMSE while substantially restoring variance fidelity relative to old grouped behavior.

### 5.1 Small-\(\rho_{EV}\) architecture ablation (single operating point)
To directly test the reviewer-facing scope question without a full matrix sweep per architecture, we run a single matched low-\(\rho_{EV}\) operating point:
- dataset: Mackey-Glass,
- budget and noise: \(S=20000\), \(N_{\text{shots}}=40\), \(\sigma=0.02\),
- protocol constants: \(t_0=30\), \(\alpha=0.7\), \(L=20\),
- resulting \(\rho_{EV}=1.365<3\),
- 8 seeds per architecture, same split-ensemble training pipeline with automatic \(K\)-selection.

Operating-point ablation table (`results/table_small_rho_architecture_claim.csv`):

| Circuit | Depth | Entangler | \(\rho_{EV}\) | NRMSE_EV | NRMSE_Split | \(\Delta\) (= EV - Split) | \(\Delta\) std | \(\Delta>0\) seed fraction | \(K_{\text{mode}}\) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Ring CNOT | 5 | Ring | 1.365 | 0.5842 | 0.4280 | +0.1562 | 0.1081 | 1.00 | 10 |
| Linear CNOT | 5 | Chain | 1.365 | 0.4728 | 0.3439 | +0.1289 | 0.0524 | 1.00 | 10 |
| All-to-all CZ | 5 | All-to-all | 1.365 | 0.5465 | 0.3668 | +0.1797 | 0.0671 | 1.00 | 10 |
| Ring CNOT | 2 | Ring | 1.365 | 0.5991 | 0.3774 | +0.2217 | 0.2231 | 1.00 | 10 |
| Ring CNOT | 10 | Ring | 1.365 | 0.5919 | 0.4578 | +0.1340 | 0.0901 | 0.75 | 10 |

Strong tested-scope claim:
- At this low-\(\rho_{EV}\) operating point, split-ensemble is better than EV in mean NRMSE for all tested architecture variants (\(\Delta>0\) in all rows).
- This is architecture-robust evidence in the tested set (entangler family and depth), while still not a proof of universal behavior over all possible circuit constructions.

### 5.2 Crossover sensitivity of \(\rho_{EV}\) across architectures
To quantify how sensitive the reported crossover scale is, we ran a multi-\(\rho\) sweep for five architecture variants (ring/chain/all-to-all at depth 5, plus ring at depth 2 and 10), then estimated smoothed \(\Delta(\rho_{EV})\) curves and extracted two markers:
- \(\rho^\star_{\Delta=0}\): where the smoothed gap reaches zero,
- \(\rho^\star_{\Delta=0.02}\): practical crossover where gain shrinks to \(\Delta=0.02\).

Artifacts:
- table: `results/table_crossover_sensitivity.csv`
- plot: `results/fig_crossover_sensitivity_ieee.pdf`

Summary (5 seeds, shots=`8,10,15,20,25,30,40,50`, sigma=`0,0.02,0.05`):
- \(\rho^\star_{\Delta=0}\) range: `4.48` to `4.75`
- \(\rho^\star_{\Delta=0.02}\) range: `3.91` to `4.12`
- low-\(\rho\) robustness (\(\rho<3\)): mean gap remains positive for every variant (`0.076` to `0.116`), with positive-point fraction `1.000` for all variants.

Interpretation:
- The crossover scale is not a fixed constant and shifts with architecture/depth.
- The low-\(\rho\) advantage of split-ensemble remains intact across tested variants.

---

## 6. Paper-ready claim set

A defensible final claim set is:

1. Under fixed-shot budgets, grouped trajectory learning consistently improves NRMSE over EV in sample-limited regimes.
2. Within the tested circuit family/protocol, the improvement follows an empirical power-law decay with \(\rho_{EV}\), supporting \(\rho_{EV}\) as an operational regime variable.
3. For NARMA-10, naive grouped/trajectory variants can be variance-pathological; a guarded grouped protocol is required for fair evaluation.
4. With this correction, NARMA-10 remains consistent with the main hypothesis: grouped training yields non-trivial error reductions without collapsing fully to a mean-line solution.
5. We do not claim universality of a specific crossover value (for example near \(\rho_{EV}\approx 3\)); that scale is currently setup-dependent.
6. In a matched low-\(\rho_{EV}\) operating-point ablation (\(\rho_{EV}=1.365<3\), 8 seeds), split-ensemble achieves \(\Delta>0\) across all tested architecture variants (ring/chain/all-to-all entanglers, depth 2/5/10).

---

## 7. Reproducibility (current scripts)

- Benchmark figure:
```bash
python /Users/superposition/Coding/qrc_budget/plot_paper.py
```

- Collapse figure:
```bash
python /Users/superposition/Coding/qrc_budget/plot_collapse.py
```

- Phase figure:
```bash
python /Users/superposition/Coding/qrc_budget/plot_phase.py
```

- NARMA old-vs-fix supplementary:
```bash
python /Users/superposition/Coding/qrc_budget/plot_narma_fix.py --seeds 8
```

- Circuit-family ablation (scope extension):
```bash
python /Users/superposition/Coding/qrc_budget/ablate_circuit_families.py --narma-fix --seeds 3
```

- Small-\(\rho_{EV}\) operating-point architecture table:
```bash
python /Users/superposition/Coding/qrc_budget/study_small_rho_split_ensemble.py --seeds 8 --S 20000 --n-shots 40 --sigma 0.02
```

- Crossover sensitivity plot/table across architectures:
```bash
python /Users/superposition/Coding/qrc_budget/plot_crossover_sensitivity.py --narma-fix --seeds 5 --shots-list "8,10,15,20,25,30,40,50" --sigma-list "0.0,0.02,0.05" --out-csv /Users/superposition/Coding/qrc_budget/results/table_crossover_sensitivity.csv --out-plot /Users/superposition/Coding/qrc_budget/results/fig_crossover_sensitivity_ieee.pdf
```

---

## 8. Limitations and next concrete paper step

Primary external-validity limitation:
- full-grid collapse/phase analyses are still centered on the baseline circuit family (depth-5 random, ring-CNOT),
- architecture breadth is currently demonstrated at a single low-\(\rho_{EV}\) operating point (ring/chain/all-to-all and depth variants),
- one hardware backend (`ibm_marrakesh`),
- one observed crossover scale in \(\rho_{EV}\), treated as non-universal.

The remaining practical step for maximal consistency is to publish a matched “all-figures narma-fix” sweep set (same seed budget and same protocol flags across all three figures), then freeze those artifacts as final camera-ready data products. The next breadth extension should explicitly test whether the \(\rho_{EV}\) crossover location shifts under alternative circuit families and backends.

The current report is already technically coherent and publication-usable, with explicit caveats and transparent metric controls.
