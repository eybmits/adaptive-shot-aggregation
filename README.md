# From Quantum Shots to Training Data

**Reorganizing Measurement Records in Quantum Machine Learning**

Accepted at [IEEE QAI 2026](https://attend.ieee.org/qai-2026/).

[Paper on arXiv](https://arxiv.org/abs/2604.28160) · [Paper PDF](output/pdf/adaptive_shot_aggregation.pdf) · [Supplement](output/pdf/adaptive_shot_aggregation_supplement.pdf) · [Reproduction guide](docs/REPRODUCIBILITY.md)

[![CI](https://github.com/eybmits/adaptive-shot-aggregation/actions/workflows/verify.yml/badge.svg)](https://github.com/eybmits/adaptive-shot-aggregation/actions/workflows/verify.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the code, saved results, measurement records, and paper
sources for the study. You can check the saved evidence and rebuild the paper
without an IBM account or a quantum computer.

## What is the idea?

A quantum experiment measures the same circuit many times. Each measurement is
called a **shot**. Usually, all shots are averaged into one training example.
We instead try averaging smaller groups of shots into several training examples.
This changes the training data while using the same measurement record.

| Representation | Shots averaged per training example | Examples per time step |
|---|---:|---:|
| Raw | `1` | `N` |
| Grouped | `k` | `N / k` |
| Expectation value (EV) | `N` | `1` |

The groups do not overlap. Validation data choose the group size and ridge
regularization together. The final test data are used only after that choice.
Grouped also allows the Raw and EV endpoints as fallback choices.

<p align="center">
  <a href="paper/gfx/fig1_shot_organization.pdf">
    <img src="docs/assets/shot_grouping_overview.svg"
         alt="The same measurement record becomes one EV example, several grouped examples, or individual raw examples"
         width="760">
  </a>
</p>

## Start here

Use **Python 3.11**, Git, and a terminal. These commands work on macOS and Linux;
Windows users can use WSL. Run commands from the repository root.

```bash
git clone https://github.com/eybmits/adaptive-shot-aggregation.git
cd adaptive-shot-aggregation
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
make check
```

`make check` runs the implementation tests and checks the saved simulation tables
against their recorded hashes and row counts. It needs only Python dependencies;
it does not run the full simulation or rebuild the paper. Success ends with a
passed test summary and `validated ... and 8 frozen tables`.

If `make` is unavailable, use the same two commands directly:

```bash
python -m pytest -q
python scripts/build_santa_fe_manuscript_results.py --validate-only
```

The lock file gives exact package versions. `requirements.txt` is available for
experiments with other versions; that environment is not the pinned one.

## Choose what to reproduce

| Goal | Command | Extra requirements |
|---|---|---|
| Check tests and saved simulation tables | `make check` | None beyond the Python setup |
| Also check published PDFs and manuscript values | `make verify` | Poppler (`pdfinfo`, `pdftotext`) |
| Reanalyze the archived hardware shots | `make hardware` | Python setup; rewrites hardware tables and Figure 5 |
| Rebuild all figures and paper PDFs | `make figures paper verify` | LaTeX, `latexmk`, and Poppler |
| Rebuild and verify all publication packages | `make all` | Same tools as the paper build |
| Run new simulations | See [the full guide](docs/REPRODUCIBILITY.md#run-new-simulations) | More computation time; a new output directory |

The [reproduction guide](docs/REPRODUCIBILITY.md) gives system installation
commands, expected outputs, the exact paper-to-file map, and separate commands
for the original simulation configurations. Run rebuilds in a disposable clone:
`make hardware`, `make figures`, `make paper`, and `make all` rewrite generated
files. PDF bytes and small floating-point differences can depend on the platform.

## What did the paper find?

At 500 time steps and 20 shots per step, Grouped lowers mean test NRMSE on all
four tasks. NRMSE measures prediction error; lower is better. The relative
reduction is `(EV - Grouped) / EV`.

| Task | EV | Grouped | Relative error reduction | 95% interval for EV minus Grouped |
|---|---:|---:|---:|---:|
| Mackey–Glass | 0.4838 | 0.4392 | 9.21% | [0.0372, 0.0519] |
| Santa Fe laser | 0.7644 | 0.6797 | 11.09% | [0.0174, 0.1522] |
| Rössler | 0.4137 | 0.3788 | 8.44% | [0.0139, 0.0559] |
| Sunspots | 0.5531 | 0.5171 | 6.50% | [0.0103, 0.0616] |

The mean NRMSE across tasks falls from 0.5537 to 0.5037: a 9.04% reduction at the
same quantum execution cost. [The result guide](docs/RESULTS.md) identifies the
source tables. Santa Fe was added after the original task outcomes were known;
its intervals are exploratory, not a preregistered confirmation.

Regularized EV closely matches Grouped. This supports the paper's explanation:
grouping adds a regularization effect from the record's own shot fluctuations.
These results do not establish quantum advantage or one best group size for
every task. Keeping quantum execution cost fixed also does not mean classical
training costs are identical.

The hardware results use archived shots from `ibm_fez` and `ibm_marrakesh`.
They test the grouping procedure on a separate ten-qubit circuit; the simulation
uses six qubits. Each device/task result comes from one acquisition split into
twenty shot blocks. Its intervals describe variation within that acquisition.
See [hardware records and limits](docs/HARDWARE.md).

## Where are the files?

| Location | Contents |
|---|---|
| [`configs/`](configs/) | Frozen experiment settings and the current four-task benchmark |
| [`data/`](data/) | Pinned Santa Fe laser and sunspot time series |
| [`results/`](results/) | Saved simulation results, hardware shots, and provenance records |
| [`scripts/`](scripts/) | Simulation, analysis, plotting, verification, and packaging code |
| [`tests/`](tests/) | Tests for the scientific implementation and reporting rules |
| [`paper/`](paper/) | LaTeX sources and final figures |
| [`output/pdf/`](output/pdf/) | Published paper and supplement PDFs, with checksums |
| [`docs/`](docs/) | [Method](docs/METHOD.md), [results](docs/RESULTS.md), [data](docs/DATA.md), and reproduction guides |

The paper uses the merged tables in
[`results/qai_robust_protocol_santa_fe_manuscript/`](results/qai_robust_protocol_santa_fe_manuscript/).
Its manifest links each table to its original result directory. Older directory
names are retained so those links remain valid. The full reproduction guide
explains which parts of the older runs enter the paper.

## Cite this work

```bibtex
@article{baumann2026quantumshots,
  title = {From Quantum Shots to Training Data: Reorganizing Measurement Records in Quantum Machine Learning},
  author = {Baumann, Markus and Zorn, Maximilian and Gabor, Thomas and Linnhoff-Popien, Claudia and Stein, Jonas},
  year = {2026},
  eprint = {2604.28160},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2604.28160},
  url = {https://arxiv.org/abs/2604.28160}
}
```

GitHub's **Cite this repository** button uses [`CITATION.cff`](CITATION.cff).
See [CONTRIBUTING.md](CONTRIBUTING.md) for corrections and bug reports.

## License

The software and original documentation use the [MIT License](LICENSE).
Included third-party data keep their providers' terms; see
[data provenance](docs/DATA.md). The repository does not relicense those data.
