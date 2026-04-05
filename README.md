# Split-Ensemble QRC ground-truth paper repo

This repository is the canonical local paper package for the current Split-Ensemble QRC manuscript. The tracked `results/`, `figures/`, `tex/`, and `docs/` directories are the source of truth for the current paper state.

## Canonical sources

- Manuscript source: `tex/main.tex`
- Results overview: `docs/results_summary.md`
- Active plotting / experiment code:
  - `scripts/replot_manuscript_exp2.py`
  - `scripts/run_20seed_experiments.py`
  - `scripts/compute_paired_stats.py`
  - `scripts/run_ibm_hardware_validation_best.py`

Generated PDFs such as `docs/full_paper.pdf` and `figures/experiment_*.pdf` are local build products and are intentionally ignored by Git.

## Experiment map

1. `experiment_1.pdf`: common operating-point comparison with representative traces and 20-seed outcome summaries
2. `experiment_2.pdf`: empirical EV-versus-Split gap as a function of `rho_EV`
3. `experiment_3.pdf`: larger-reservoir / more-qubits robustness sweep
4. `experiment_4.pdf`: baseline controls at the common operating point
5. `experiment_5.pdf`: duplication-control sweep (`EV`, `EV-dup`, `Split`)
7. `experiment_7.pdf`: reservoir-architecture ablation at the common operating point

Supplementary reproducibility artifacts:
- `lambda_*.csv`: ridge-regularization sweep at the common operating point
- `experiment_*_paired_stats.csv`: paired delta / CI / win-rate / Wilcoxon tables backing the revised claims
- `reservoir_size_*.csv`: full-format version of the Experiment 3 size sweep
- `hardware_validation_experiment2_best_*`: full audit bundle for the best-regime IBM validation rerun

Appendix-only robustness figures extend the story with the `experiment_7` reservoir-architecture ablation at fixed `Q=4` and a targeted `Q=12` follow-up on the strongest variants.

The manuscript additionally imports a narrow hardware validation as a table-backed result via `results/hardware_validation_summary.csv`.

## Canonical result files

- `results/representative_summary.csv`
- `results/representative_seed_level.csv`
- `results/representative_traces.npz`
- `results/empirical_gap_summary.csv`
- `results/empirical_gap_seed_level.csv`
- `results/more_qubits_summary.csv`
- `results/more_qubits_seed_level.csv`
- `results/baseline_controls_summary.csv`
- `results/baseline_controls_seed_level.csv`
- `results/duplication_control_summary.csv`
- `results/duplication_control_seed_level.csv`
- `results/experiment_1_paired_stats.csv`
- `results/experiment_2_cell_paired_stats.csv`
- `results/experiment_2_peak_paired_stats.csv`
- `results/experiment_3_paired_stats.csv`
- `results/experiment_4_paired_stats.csv`
- `results/experiment_5_paired_stats.csv`
- `results/architecture_ablation_summary.csv`
- `results/architecture_ablation_seed_level.csv`
- `results/architecture_ablation_q12_summary.csv`
- `results/architecture_ablation_q12_seed_level.csv`
- `results/architecture_ablation_paired_stats.csv`
- `results/hardware_validation_summary.csv`

## Supplementary reproducibility files

- `results/lambda_seed_level.csv`
- `results/lambda_summary.csv`
- `results/reservoir_size_seed_level.csv`
- `results/reservoir_size_summary.csv`
- `results/hardware_validation_experiment2_best_seed_level.csv`
- `results/hardware_validation_experiment2_best_summary.csv`
- `results/hardware_validation_experiment2_best_trace_long.csv`
- `results/hardware_validation_experiment2_best_trace_wide.csv`
- `results/hardware_validation_experiment2_best_job_rows.csv`
- `results/hardware_validation_experiment2_best_manifest.json`
- `results/hardware_validation_experiment2_best_traces.npz`
- `results/hardware_validation_experiment2_best_audit_bundle.npz`

## Active scripts

- `scripts/replot_manuscript_exp2.py`: regenerates `figures/experiment_1.pdf` through `figures/experiment_5.pdf` plus `figures/experiment_7.pdf` from tracked result files
- `scripts/run_20seed_experiments.py`: synthetic-suite runner retained for the current codebase
- `scripts/compute_paired_stats.py`: writes the paired-delta, CI, win-rate, and Wilcoxon tables used in the revised results text
- `scripts/update_common_operating_point_results.py`: refreshes the representative operating-point artifacts
- `scripts/run_ibm_hardware_validation_best.py`: IBM Runtime validation runner for the three best `experiment_2` settings (`mackey_glass`, `lorenz`, `narma10`) with 20 seeds by default; writes seed-level and summary CSVs, exploded trace tables, an exploded job table, compressed trace/audit bundles, a manifest JSON, and an optional summary figure using the saved local Qiskit account

## Minimal rebuild workflow

1. Regenerate the figure PDFs:
   - `python scripts/replot_manuscript_exp2.py`
2. Rebuild the paper:
   - `cd tex`
   - `pdflatex -interaction=nonstopmode -halt-on-error main.tex`
   - `pdflatex -interaction=nonstopmode -halt-on-error main.tex`
3. Refresh the canonical paper artifact:
   - copy `tex/main.pdf` to `docs/full_paper.pdf`

`docs/full_paper.pdf`, `figures/*.pdf`, and `figures/*.png` are treated as disposable local build outputs and are ignored by Git.
