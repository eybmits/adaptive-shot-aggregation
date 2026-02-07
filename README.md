# Budget-Aware Quantum Reservoir Computing

Minimal, paper-focused repository for generating the three core figures:
- `results/fig_benchmarks.pdf`
- `results/fig_collapse.pdf`
- `results/fig_phase.pdf`

The project studies EV vs grouped trajectory training under fixed measurement budgets in quantum reservoir computing, including the NARMA-10 handling used in the current paper draft.

## Install

```bash
pip install numpy scipy scikit-learn matplotlib
```

## Generate Core Figures

From repo root:

```bash
python plot_paper.py
python plot_collapse.py
python plot_phase.py
```

Notes:
- Each script writes directly to `results/`.
- Cache files (`results/data_*.pkl`) are optional and ignored by git.
- If no cache is present, scripts recompute data automatically.

## Repository Structure

```text
qrc_budget.py            # Core simulator + training methods
plot_paper.py            # Figure 1: benchmark comparison
plot_collapse.py         # Figure 2: rho_EV collapse
plot_phase.py            # Figure 3: phase diagrams
results/
  fig_benchmarks.pdf
  fig_collapse.pdf
  fig_phase.pdf
  paper_report.md        # Technical paper-style report
```

## Reproducibility Scope

This cleaned repository intentionally keeps only what is needed for:
1. generating the three core paper figures,
2. reproducing the technical narrative in `results/paper_report.md`.
