# From Quantum Shots to Training Data: Reorganizing Measurement Records for Improved Finite-Shot Quantum Learning

This repository contains the code, evidence, and paper for the QCE26 submission.

The submission is a **QCE26 New Ideas and Emergent Results (NIER) short
paper**. Its focused claim is that adaptive shot aggregation can tune how many
shots form each training view without additional quantum executions. It does not present the
idea as a universal law or a completed quantum advantage result.

The build follows the
[QCE26 technical-paper instructions](https://qce.quantum.ieee.org/2026/call-for-technical-papers/):
NIER is declared in the submission filenames, the source uses
`\documentclass[10pt,conference]{IEEEtran}`, and the main document is six pages
including its reference page.

Quantum reservoir computing normally averages all repeated measurements at a
time step into one expectation value feature vector. This work treats the
number of measurements per average as a readout hyperparameter:

| Representation | Aggregation scale | Training samples per time step |
|---|---:|---:|
| Raw | `k = 1` | `N` individual measurements |
| Grouped | `1 < k < N` | `N / k` partial averages |
| EV | `k = N` | one full average |

All candidates use the same measurement record and execution budget. The
aggregation scale and ridge penalty are tuned jointly on chronological
development folds. The tuned pair is frozen before the test block is evaluated.

## Main result

The simulation benchmark contains Mackey Glass, Santa Fe laser, Rössler, and
sunspots. Every task uses the same six-qubit reservoir, feature set,
preprocessing, execution budgets, development folds, and parameter grids.

At the primary setting, `T = 500` and `N = 20`, positive values favor Grouped:

| Task | EV minus Grouped NRMSE | 95% interval |
|---|---:|---:|
| Mackey Glass | 0.0138 | [-0.0023, 0.0300] |
| Santa Fe laser | **0.0848** | **[0.0174, 0.1522]** |
| Rössler | **0.0349** | **[0.0139, 0.0559]** |
| Sunspots | **0.0360** | **[0.0103, 0.0616]** |

The Cov EV control performs similarly to Grouped. This agrees with the exact
identity between grouped squared loss and EV loss with an empirical covariance
penalty. The paper does not claim quantum advantage or a universal aggregation scale.

## Repository map

```text
configs/benchmark.json       current four task simulation protocol
configs/qai_robust_*.json    immutable configs hashed by source manifests
config.json                  compact preprocessing grid for hardware analysis
data/                        pinned Santa Fe and sunspot source series
scripts/                     simulation, analysis, plotting, and release tools
results/                     frozen evidence used by the paper
paper/                       named and anonymous manuscript sources
output/pdf/                  submission PDFs and checksums
docs/                        concise results and reproducibility notes
tests/                       tests for the retained implementation
release/qce2026_nier_anonymous/ generated anonymous source package
```

The result directories retain their original artifact identifiers because
their manifests bind paths, hashes, and row counts. Their roles are:

| Directory | Contents |
|---|---|
| `results/expressive_reservoir/` | ideal feature audit |
| `results/qai_robust_protocol_full/` | frozen source rows for three tasks |
| `results/qai_santa_fe_laser_screen/` | frozen Santa Fe source rows |
| `results/qai_robust_protocol_santa_fe_manuscript/` | merged four task paper evidence |
| `results/qai_hardware/` | Fez and Marrakesh records and analysis tables |
| `results/qai_hardware_santa_fe_extension/` | Santa Fe IBM job and budget manifest |

## Quick start

Python 3.11 or newer is recommended. Install the dependencies in an isolated
environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The supported commands are listed by:

```bash
make help
```

The complete local verification is:

```bash
make all
```

This runs the test suite, verifies all admitted numerical results, regenerates
the paper figures, builds the named and anonymous papers and supplements, and
checks the anonymous release.

## Reproduce the simulations

Inspect the current four task plan without running simulations:

```bash
python scripts/run_robust_evaluation.py \
  --config configs/benchmark.json \
  --results-dir results/benchmark_dry_run \
  --dry-run
```

Run or resume the full benchmark:

```bash
python scripts/run_robust_evaluation.py \
  --config configs/benchmark.json \
  --results-dir results/benchmark \
  --workers 4 \
  --resume
```

The checked paper package is already included, so reproducing the expensive
simulation is not required to verify the published numbers.

## Reproduce the paper from frozen evidence

```bash
make figures
make paper
make verify
```

Final PDFs:

- [`output/pdf/QCE26_NIER_split_ensemble_qrc_named.pdf`](output/pdf/QCE26_NIER_split_ensemble_qrc_named.pdf)
- [`output/pdf/QCE26_NIER_split_ensemble_qrc_anonymous.pdf`](output/pdf/QCE26_NIER_split_ensemble_qrc_anonymous.pdf)
- [`output/pdf/QCE26_NIER_split_ensemble_qrc_supplement.pdf`](output/pdf/QCE26_NIER_split_ensemble_qrc_supplement.pdf)
- [`output/pdf/QCE26_NIER_split_ensemble_qrc_supplement_anonymous.pdf`](output/pdf/QCE26_NIER_split_ensemble_qrc_supplement_anonymous.pdf)

## Hardware evidence

The hardware panel uses archived ordered measurements from `ibm_fez` and
`ibm_marrakesh`. This is deliberately a transfer test on a different reservoir:
simulation uses a six-qubit, two-layer data reuploading circuit with local Z
and neighboring ZZ features, whereas hardware uses a ten-qubit, depth-one
transverse field Ising ring with local Z features. The hardware evidence does
not compare these architectures or claim to reproduce the six-qubit simulation.
Within each study, all aggregation methods receive the same fixed circuit
record.

Each hardware task and processor has one acquisition with 500 time steps and
400 shots per step. The analysis partitions each acquisition into twenty
disjoint records with `N = 20`. These records measure variation within one
acquisition, not independent calibration periods.

The Santa Fe jobs used 55 and 56 QPU seconds, for 111 seconds in total. No IBM
connection is needed to reproduce the analysis:

```bash
make hardware
```

See [`docs/HARDWARE.md`](docs/HARDWARE.md) for job identifiers and scope.

## Scientific scope

- Aggregation scale is tuned on development data only.
- All finite measurement methods reuse the same raw record.
- Uncertainty is computed across trajectories or chronological windows after
  nested reservoir, shot, and grouping repetitions are averaged.
- The ideal reservoir is nonlinear and full rank, but it does not consistently
  outperform the matched classical input.
- The hardware intervals describe measurement blocks within an acquisition.
- No result establishes quantum advantage, a universal aggregation scale, or
  robustness across hardware calibration periods.

Detailed protocol and artifact information is in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
