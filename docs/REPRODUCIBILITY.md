# Reproduce the paper

Start with the [README setup](../README.md#start-here). Use the pinned Python 3.11
environment and run all commands from the repository root. Checking saved
evidence, rebuilding figures, and running new simulations are separate steps.
None of the commands below submit a quantum hardware job.

## 1. Check the saved evidence

```bash
make check
```

This runs the tests and verifies the eight merged simulation CSV tables. The
checks compare file hashes, row counts, task names, and the hashes of the three
source manifests. They read the frozen evidence without rewriting it.

To also check the numerical claims, manuscript source, figures, and shipped PDFs:

```bash
make verify
```

This requires Poppler (`pdfinfo` and `pdftotext`), but no LaTeX build. It reads the
tracked PDFs in `output/pdf/`, checks their `SHA256SUMS`, and checks the main
papers' six-page length, title, task labels, and anonymous text. Success ends with
`Santa Fe manuscript package verified.` The supplement bytes are checked too;
full supplement build checks are part of the release commands below.

## 2. Install tools for the paper build

On Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y make latexmk poppler-utils \
  texlive-fonts-recommended texlive-latex-extra texlive-science
```

On macOS, install MacTeX for LaTeX and `latexmk`, and install Poppler through
Homebrew (`brew install poppler`). Ensure `latexmk`, `pdflatex`, `bibtex`,
`pdfinfo`, and `pdftotext` are on your `PATH`. Windows users can follow the Linux
instructions inside WSL.

## 3. Rebuild from the saved records

Use a disposable clone for this step: the commands write generated files into
`results/`, `paper/`, and `output/pdf/`.

```bash
make hardware      # reanalyze the saved shots from both IBM processors
make figures       # rebuild Figures 1–5
make paper         # build named/anonymous paper and supplement PDFs
make verify        # check the rebuilt published PDFs and evidence
make release       # build and test the anonymous review archive
make arxiv         # build and test the named arXiv source archive
```

Or run all steps, including tests, in order:

```bash
make all
```

Use plain `make all`, without `-j`: the publication steps share generated files.
The CI workflow first verifies the checked-in evidence and replays one complete
simulation job, then runs `make all` on Ubuntu with Python 3.11.

| Output | Location |
|---|---|
| Named paper and supplement | `output/pdf/adaptive_shot_aggregation.pdf`, `output/pdf/adaptive_shot_aggregation_supplement.pdf` |
| Anonymous PDFs | The corresponding `_anonymous.pdf` files in `output/pdf/` |
| PDF checksums | `output/pdf/SHA256SUMS` |
| Anonymous review ZIP | `release/adaptive_shot_aggregation_anonymous_review.zip` |
| Named arXiv source ZIP | `release/adaptive_shot_aggregation_arxiv.zip` |

`make release` and `make arxiv` also test builds using only the packaged files.
Generated release folders are ignored by Git. Rebuilding can change PDF bytes
because fonts and TeX versions differ. The numerical checks use explicit
tolerances; passing them does not imply byte-identical PDFs across platforms.

## Find the evidence behind each result

| Paper content | Saved evidence | Rebuild or inspection command |
|---|---|---|
| Main comparison and intervals | `results/qai_robust_protocol_santa_fe_manuscript/cluster_summary.csv` and `cluster_paired_stats.csv` | `make verify-data` |
| Per-repeat scores and selected group sizes | `replicate_results.csv`, `selected_k_frequency.csv` in the same directory | Inspect the CSV files; method `split` means Grouped |
| Validation curves | `split_validation_spectrum.csv` in the same directory | `python scripts/plot_manuscript_figures.py --fig 2` |
| Figures 2–4 | Merged tables and `results/expressive_reservoir/` diagnostics | `python scripts/plot_manuscript_figures.py --fig 2` (or `3`, `4`) |
| Figure 1 | `paper/gfx/fig1_shot_organization.tex` | `make figures` |
| Hardware results and Figure 5 | `results/qai_hardware/` and its `marrakesh/` subdirectory | `make hardware` |
| Input time series | `data/` | See [DATA.md](DATA.md) for sources and checksums |

The merged simulation tables combine these original runs:

| Paper component | Configuration | Frozen result directory |
|---|---|---|
| Mackey–Glass: 20 independent trajectories, indices 5–24 | `configs/mackey_glass_primary.json` | `results/mackey_glass_primary/` |
| Rössler and sunspots: five trajectories/windows each | `configs/qai_robust_protocol.json` | `results/qai_robust_protocol_full/` (only those two tasks enter the merged paper tables) |
| Santa Fe: five chronological windows | `configs/qai_robust_protocol_santa_fe.json` | `results/qai_santa_fe_laser_screen/` |

The older full run also contains Lorenz and five Mackey–Glass trajectories.
These are retained for provenance, but they are not the current paper's main
comparison. The merge manifest records exactly which sources were used.
Santa Fe was screened after the original task outcomes were known, so its
intervals are exploratory. The merge script copies and combines saved tables;
it does not rerun the simulations or recalculate their statistics.

## Run new simulations

A bounded comparison with the paper's saved results is available:

```bash
make replay
```

This executes the full balanced Mackey–Glass trajectory 5 / reservoir seed 1729
job, including all its shot/grouping seeds and all six methods. It compares all
38 result rows, chosen parameters, validation fold scores, and test errors with
the saved evidence using an absolute tolerance of `1e-9`. It prints a JSON
summary and does not overwrite evidence. This is one full job, not the entire
experiment. See [VALIDATION.md](VALIDATION.md) for the observed release checks.

A small wiring check runs one reduced Mackey–Glass job:

```bash
python scripts/run_robust_evaluation.py \
  --config configs/benchmark.json \
  --results-dir results/smoke \
  --smoke --workers 1 --resume
```

Its manifest is marked `smoke-test`. Its scores are not paper evidence.

Inspect the current benchmark plan before starting a larger run:

```bash
python scripts/run_robust_evaluation.py \
  --config configs/benchmark.json \
  --results-dir results/benchmark_dry_run \
  --dry-run
```

This writes a planned manifest, without simulation. The current benchmark uses
five forecasting units per task. **It does not reproduce the paper's separate
20-trajectory Mackey–Glass evaluation.** To rerun the current benchmark:

```bash
python scripts/run_robust_evaluation.py \
  --config configs/benchmark.json \
  --results-dir results/benchmark \
  --workers 4 --resume
```

To rerun the original paper components, use the three configurations in the
evidence table above, with a new output directory for each. For example:

```bash
python scripts/run_robust_evaluation.py \
  --config configs/mackey_glass_primary.json \
  --results-dir tmp/rerun_mackey_glass \
  --workers 4 --resume
python scripts/run_robust_evaluation.py \
  --config configs/qai_robust_protocol.json \
  --results-dir tmp/rerun_original \
  --workers 4 --resume
python scripts/run_robust_evaluation.py \
  --config configs/qai_robust_protocol_santa_fe.json \
  --results-dir tmp/rerun_santa_fe \
  --workers 4 --resume
```

Full simulation runs are much more expensive than `make check`; no fixed runtime
is promised. Reduce `--workers` if memory is limited. On a multi-core machine,
setting `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` can avoid nested parallelism.

Always use a new output directory, as shown. The runner writes its manifest and
result tables there. Do not point it at a frozen result directory. Resume keys
include the configuration and implementation hashes: only matching job shards
are reused. If either changes, start a new directory to keep runs separate.
New outputs include `replicate_results.csv`, `cluster_summary.csv`,
`cluster_paired_stats.csv`, `split_validation_spectrum.csv`, and `manifest.json`.

The frozen merge uses fixed source paths. Do not replace it with new results
just to make checks pass. Compare a rerun against the appropriate original
source directory first. Small numerical differences can arise from numerical
libraries and hardware; changed scientific conclusions require review.

## Version history and replay limits

Frozen manifests record the source hash used when a run was produced. The current
runner has later documentation and metadata changes, so its combined source hash
differs. A new result must keep its own provenance. The older report generator
deliberately rejects a source-hash mismatch; do not bypass that check.

The four hashed files are `run_robust_evaluation.py`, `robust_protocol.py`,
`robust_reservoir_adapter.py`, and `expressive_reservoir.py`, all in `scripts/`.
The hash covers each filename and its bytes, in that order. It does not cover
every dependency or capture the original operating system.

| Frozen run | Historical source recovery |
|---|---|
| `mackey_glass_primary` | The four-file hash `08e59534adc96ca3a4ddd70b9587187af8b93a3a3dd2ace664c32b37ac3ec01b` matches commit `7f8481ed787205875ca7e581da41ec8487c5522a`. The current primary config matches the manifest, but was committed later. Recovering both requires combining these verified components; no single committed tree contains both. |
| `qai_robust_protocol_full` | Hash `608649ed7ddf50c1d4ccd403f10cafbbab7f06aed108c283bf7b69627e8abe43` was not found in the retained Git history. Exact original source bytes cannot be recovered from this repository. |
| `qai_santa_fe_laser_screen` | Hash `3b0c41de5d1962051cdcde489df7ed4a46df089149af23a51c2d517e3897e94a` was not found in the retained Git history. Exact original source bytes cannot be recovered from this repository. |

To recover the verified Mackey–Glass components, start from the current checkout
with its Python environment active. Use a new disposable directory (and a full
clone, not a shallow clone):

```bash
git clone https://github.com/eybmits/adaptive-shot-aggregation.git tmp/historical-mg
git -C tmp/historical-mg restore \
  --source=7f8481ed787205875ca7e581da41ec8487c5522a -- \
  scripts/run_robust_evaluation.py scripts/robust_protocol.py \
  scripts/robust_reservoir_adapter.py scripts/expressive_reservoir.py
cd tmp/historical-mg
make replay
```

The replay summary must show the Mackey–Glass source hash above. The retained
`configs/mackey_glass_primary.json` has SHA-256
`dd7349575b2ce6950fc7ac369b2fd04828d52fa74ffb5a7165bc21f072944496`, matching
the frozen manifest. The full Mackey–Glass command in the preceding section can
then be run from this disposable checkout. This recovers the hashed code/config
pair, not a historical Python environment or a single original Git snapshot.

Commit `7f8481e` also retains the earlier balanced-only experiment under
`configs/mackey_glass_confirmatory.json` and `results/mackey_glass_confirmatory/`.
That is a historical 60-job campaign. It is different from the present
three-budget primary configuration, which has 180 jobs.

For the two older runs, the repository supports verification of saved tables,
paper regeneration, and new runs of the retained implementation. It does not
support a claim that every byte of the original simulation source has survived.
The full new-simulation commands above have not been rerun as part of the public
release validation; the tested full-job replay is explicitly bounded.

## Understand the protocol

Each simulation allocation has 10,000 circuit executions:
`(time steps, shots per step) = (250, 40), (500, 20), (1000, 10)`.
The middle allocation is the primary comparison. Every method receives the
same measurement record. Scaling is fitted within training data. Three expanding
chronological validation folds choose group size and ridge regularization;
grouped predictions are averaged to one prediction per time step before scoring.
The final test block is evaluated after tuning is complete.

Reservoir, shot, and grouping seeds are nested within each trajectory or window.
They are averaged before confidence intervals are computed; they are not extra
independent forecasting units. The hardware study uses a separate ten-qubit
circuit and one acquisition per task/device. See [HARDWARE.md](HARDWARE.md) for
its raw records, acquisition metadata, and limits.

## If a command fails

- Missing Python module: activate `.venv` and install `requirements-lock.txt`.
- Missing `pdfinfo` or `pdftotext`: install Poppler. `make check` works without it.
- Missing `latexmk` or a TeX package: install the build tools above.
- Hash mismatch: check `git status --short`. Use a clean clone to distinguish a
  local modification from a repository problem; do not rewrite the manifest.
- Build works but PDF bytes differ: compare page counts, text, figure content,
  and numerical checks before assuming the scientific result changed.

For a bug report, include the command, commit (`git rev-parse HEAD`), Python
version, operating system, and error message. See [CONTRIBUTING.md](../CONTRIBUTING.md).
