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

## Hardware Validation (ibm_marrakesh)

This repository now also contains a finalized hardware validation package for
Mackey-Glass on IBM Quantum backend `ibm_marrakesh`.

Fixed configuration:
- `Q=6`, `depth=5`, `L=20`
- `S=20000`, `N_shots=40`, `sigma=0.03`
- `t0=30`, `alpha=0.7`, `lambda=10`
- seeds `0..7`

Final artifacts (camera-ready):
- `results/camera_ready/fig_hardware_mg_main.pdf`
- `results/camera_ready/fig_hardware_mg_appendix_collage.pdf`
- `results/camera_ready/table_hardware_mg_nrmse.csv`
- `results/camera_ready/supplementary/fig_hardware_mg_seed_00..07.(pdf|png)`
- `results/camera_ready_hardware_mg.zip`

Hardware summary (8 seeds):
- EV: `1.023 ± 0.235`
- Traj: `0.932 ± 0.021`
- G-Traj: `0.670 ± 0.080`
- Gap (EV-GTraj): `+0.353`

## Hardware Scripts

From repo root:

```bash
# 1) Run on IBM hardware and store raw hardware outputs
python run_best_mg_ibm_marrakesh.py --backend ibm_marrakesh --seeds 8 --chunk-size 200

# 2) Replot in benchmark style (time series spans two left columns)
python plot_best_mg_ibm_marrakesh_wide.py \
  --outfile fig_best_mg_ibm_marrakesh_benchmark_style.pdf

# 3) Build camera-ready package
python export_camera_ready_hardware.py --make-zip

# 4) Build appendix overview collage
python make_hardware_appendix_collage.py
```

Notes:
- No new hardware submission is needed for re-plotting if the required IBM job IDs
  are already present in `results/table_best_mg_ibm_marrakesh.csv`.
- `results/camera_ready/MANIFEST.md` and `results/camera_ready/SHA256SUMS.txt`
  define the final package contents.
