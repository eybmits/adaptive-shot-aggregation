# Split-Ensemble QRC ground-truth paper repo

This repository is the canonical local paper package for the current Split-Ensemble QRC manuscript. The tracked `results/`, `figures/`, `tex/`, and `docs/` directories are the source of truth for the current paper state.

## Canonical outputs

- Paper PDF: `docs/full_paper.pdf`
- Manuscript source: `tex/main.tex`
- Main figure set:
  - `figures/experiment_1.pdf`
  - `figures/experiment_2.pdf`
  - `figures/experiment_3.pdf`
  - `figures/experiment_4.pdf`
  - `figures/experiment_5.pdf`
- Results overview: `docs/results_summary.md`

## Experiment map

1. `experiment_1.pdf`: common operating-point comparison with representative traces and 20-seed outcome summaries
2. `experiment_2.pdf`: empirical EV-versus-Split gap as a function of `rho_EV`
3. `experiment_3.pdf`: larger-reservoir / more-qubits robustness sweep
4. `experiment_4.pdf`: baseline controls at the common operating point
5. `experiment_5.pdf`: duplication-control sweep (`EV`, `EV-dup`, `Split`)

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
- `results/hardware_validation_summary.csv`

## Active scripts

- `scripts/replot_manuscript_exp2.py`: regenerates `figures/experiment_1.pdf` through `figures/experiment_5.pdf` from tracked result files
- `scripts/run_20seed_experiments.py`: synthetic-suite runner retained for the current codebase
- `scripts/update_common_operating_point_results.py`: refreshes the representative operating-point artifacts

## Minimal rebuild workflow

1. Regenerate the figure PDFs:
   - `python scripts/replot_manuscript_exp2.py`
2. Rebuild the paper:
   - `cd tex`
   - `pdflatex -interaction=nonstopmode -halt-on-error main.tex`
   - `pdflatex -interaction=nonstopmode -halt-on-error main.tex`
3. Refresh the canonical paper artifact:
   - copy `tex/main.pdf` to `docs/full_paper.pdf`

`docs/full_paper.pdf` is the tracked compiled-paper artifact. `tex/main.pdf` and the other TeX byproducts are treated as disposable build outputs and are ignored.
