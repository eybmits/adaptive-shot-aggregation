# Budget-Aware Quantum Reservoir Computing

Paper-focused repository for reproducible figure generation under fixed measurement budgets in QRC.

## Final Method Names
- `EV`
- `Single-Path Readout (SPR)` (short tick label: `SPR`)
- `Split-Ensemble`

## Environment

```bash
pip install numpy scipy scikit-learn matplotlib pypdf
```

## Final Publication Figures

### Main IEEE figures
- `results/fig_benchmarks_ieee.pdf`
- `results/fig_collapse_ieee.pdf`
- `results/fig_phase_ieee.pdf`
- `results/fig_gtraj_robustness_oneplot_notitle.pdf`
- `results/fig_crossover_sensitivity_ieee.pdf` (architecture crossover sensitivity)

### Hardware camera-ready figures
- `results/camera_ready/fig_hardware_mg_main.pdf`
- `results/camera_ready/fig_hardware_mg_appendix_collage.pdf`
- `results/camera_ready/collage.pdf`
- `results/camera_ready/table_hardware_mg_nrmse.csv`

## Reproduce Main IEEE Figures

From repo root:

```bash
# 1) Benchmarks (left + middle columns)
python plot_paper.py \
  --output results/fig_benchmarks_ieee.pdf \
  --left-middle-only

# 2) Collapse (cached)
python plot_collapse.py \
  --cache-file results/data_collapse_ieee.pkl \
  --output results/fig_collapse_ieee.pdf \
  --ours-label "Split-Ensemble"

# 3) Phase (cached)
python plot_phase.py \
  --cache-file results/data_phase_ieee.pkl \
  --output results/fig_phase_ieee.pdf

# 4) Robustness one-plot (no title, compact)
python plot_gtraj_robustness_styled.py \
  --table results/table_gtraj_robustness_oneplot.csv \
  --out-pdf results/fig_gtraj_robustness_oneplot_notitle.pdf \
  --out-png results/fig_gtraj_robustness_oneplot_notitle.png \
  --fig-width 7.16 \
  --fig-height 2.2
```

## Reproduce Hardware Figures (No New Hardware Runs)

This uses existing camera-ready table + supplementary seed panels only.

```bash
# Re-render camera-ready hardware main + appendix
python replot_hardware_camera_ready_ieee.py

# Re-render alternative collage output path
python replot_hardware_camera_ready_ieee.py \
  --appendix-pdf results/camera_ready/collage.pdf \
  --appendix-png results/camera_ready/collage.png
```

## Optional Cache Preparation

```bash
python precompute_plot_data.py
```

## Circuit-Family Ablation (Reviewer Scope)

```bash
# Multi-family simulator ablation (ring/line/star CNOT + all-to-all CZ)
python ablate_circuit_families.py \
  --narma-fix \
  --seeds 3 \
  --shots-list "8,15,25,40" \
  --sigma-list "0.0,0.02,0.05,0.08" \
  --out-csv results/table_circuit_family_ablation.csv

# Family-specific collapse plot (optional)
python plot_collapse.py \
  --narma-fix \
  --circuit-family line_cnot \
  --seeds 3 \
  --output results/fig_collapse_narma_fix_ieee_line_cnot.pdf

# Single operating-point claim table (rho_EV < 3)
python study_small_rho_split_ensemble.py \
  --seeds 8 \
  --S 20000 \
  --n-shots 40 \
  --sigma 0.02 \
  --out-csv results/table_small_rho_architecture_claim.csv

# Crossover sensitivity quantification + validation plot
python plot_crossover_sensitivity.py \
  --narma-fix \
  --seeds 5 \
  --shots-list "8,10,15,20,25,30,40,50" \
  --sigma-list "0.0,0.02,0.05" \
  --out-csv results/table_crossover_sensitivity.csv \
  --out-plot results/fig_crossover_sensitivity_ieee.pdf
```

## Repository Structure

```text
qrc_budget.py                            # Core simulator + training methods
ieee_plot_style.py                       # Shared publication style + labels
plot_paper.py                            # Main benchmark figure
plot_collapse.py                         # rho_EV collapse
ablate_circuit_families.py               # multi-family ablation table
study_small_rho_split_ensemble.py        # single-point small-rho architecture claim
plot_crossover_sensitivity.py            # crossover sensitivity quantification + plot
plot_phase.py                            # phase figure
plot_gtraj_robustness_styled.py          # robustness 3-panel figure
replot_hardware_camera_ready_ieee.py     # hardware replot from existing artifacts
precompute_plot_data.py                  # cache generation helper
results/
  fig_benchmarks_ieee.pdf
  fig_collapse_ieee.pdf
  fig_phase_ieee.pdf
  fig_gtraj_robustness_oneplot_notitle.pdf
  camera_ready/
```

## Notes
- Cache files (`results/data_*.pkl`) are used for fast rerender iteration.
- Hardware rerender commands do not submit jobs to IBM hardware.
- `results/camera_ready/MANIFEST.md` and `results/camera_ready/SHA256SUMS.txt` track camera-ready package contents.

## License
This repository is licensed under the MIT License. See `LICENSE`.
