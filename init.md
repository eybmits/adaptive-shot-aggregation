# init.md - Repo bootstrap notes for plotting workflow

## Scope
This repository was refactored to support fast, repeatable IEEE-style figure iteration with:
- one shared style source
- one consistent method naming (`Single-Path Readout (SPR)` and `Split-Ensemble`)
- cached data for quick rerenders
- cleaned `results/` root while keeping important experiments/hardware outputs

## Canonical output files (keep)
- `results/fig_benchmarks_ieee.pdf`
- `results/fig_collapse_ieee.pdf`
- `results/fig_phase_ieee.pdf`
- `results/fig_gtraj_robustness_oneplot_notitle.pdf`
- `results/table_gtraj_robustness_oneplot.csv` (source table for robustness plot)

## Important experiment artifacts (keep)
- `results/camera_ready/` (all hardware camera-ready figures and manifest)
- `results/camera_ready_hardware_mg.zip`
- tracked legacy baseline PDFs:
  - `results/fig_benchmarks.pdf`
  - `results/fig_collapse.pdf`
  - `results/fig_phase.pdf`

## Core style source of truth
- `ieee_plot_style.py`
  - central method and architecture color palettes
  - shared axis styling helpers
  - `OURS_LABEL = "Split-Ensemble"`
  - `TRAJ_LABEL = "Single-Path Readout (SPR)"`
  - `TRAJ_TICK_LABEL = "SPR"`

## Plot scripts changed
- `plot_paper.py`
  - IEEE style integration
  - consistent method colors
  - white legend background
  - improved left-panel readability (red curve without markers)
  - white-backed bottom-left text box
  - added `--left-middle-only` mode (`width_ratios=[2.8, 1]`)
- `plot_collapse.py`
  - IEEE style integration
  - annotation moved right and renamed to `Split-Ensemble wins`
  - added `--ours-label` (default `Split-Ensemble`)
  - supports IEEE cache naming
- `plot_phase.py`
  - IEEE style integration
  - diverging map with `RdBu_r` and centered norm at zero
  - removed internal overlay lines by default
  - optional `--show-overlay-lines` switch
- `plot_best_mg_ibm_marrakesh_wide.py`
  - visible labels updated to `Split-Ensemble`
- `plot_gtraj_robustness_styled.py` (new)
  - compact, no-title robustness figure
  - shared IEEE styling and color conventions

## Helper scripts added
- `precompute_plot_data.py`
  - writes reusable cached data for faster iteration
- `replot_ieee_all.py`
  - one-shot rerender for main IEEE plots

## Data caches for fast iteration
- `results/data_benchmarks_narma_fix.pkl`
- `results/data_collapse_ieee.pkl`
- `results/data_phase_ieee.pkl`

Note:
- current cache workflow prioritizes iteration speed (`seeds=1`)
- increase seeds when statistical stability is needed

## Cleanup done
- removed temporary/intermediate render artifacts
- removed duplicate robustness PDFs (kept only styled canonical file)
- removed non-camera-ready preview PNGs from `results/` root
- removed `.DS_Store` and `__pycache__`
- explicitly preserved hardware camera-ready outputs
- restored tracked legacy `results/fig_collapse.pdf` to avoid accidental binary drift

## Rebuild commands

### Benchmarks (left + middle panels only)
```bash
python plot_paper.py \
  --output results/fig_benchmarks_ieee.pdf \
  --left-middle-only
```

### Collapse
```bash
python plot_collapse.py \
  --cache-file results/data_collapse_ieee.pkl \
  --output results/fig_collapse_ieee.pdf \
  --ours-label "Split-Ensemble"
```

### Phase (default: no overlay lines)
```bash
python plot_phase.py \
  --cache-file results/data_phase_ieee.pkl \
  --output results/fig_phase_ieee.pdf
```

### Phase (if overlays are required)
```bash
python plot_phase.py \
  --cache-file results/data_phase_ieee.pkl \
  --output results/fig_phase_ieee.pdf \
  --show-overlay-lines
```

### Robustness (compact, no title)
```bash
python plot_gtraj_robustness_styled.py \
  --table results/table_gtraj_robustness_oneplot.csv \
  --out-pdf results/fig_gtraj_robustness_oneplot_notitle.pdf \
  --fig-width 7.16 \
  --fig-height 2.2
```

### Full IEEE rerender
```bash
python replot_ieee_all.py
```

## Practical guardrails for next agent pass
- do not delete anything under `results/camera_ready/`
- keep `results/camera_ready_hardware_mg.zip`
- keep the four canonical IEEE PDFs listed above
- keep `results/table_gtraj_robustness_oneplot.csv` for robustness rerenders
- when changing style, edit `ieee_plot_style.py` first and rerender via `replot_ieee_all.py`
