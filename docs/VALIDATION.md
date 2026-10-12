# Public release validation

Local checks performed on 2026-10-13, using macOS 15.6.1 on Apple Silicon,
Python 3.11.15, `requirements-lock.txt`, Latexmk 4.86a, and Poppler 26.06.0.
The Python installation passed `pip check`. Scientific implementation files,
frozen result tables, input data, and shipped paper PDFs were kept unchanged
in the public cleanup.

| Check | Observed result |
|---|---|
| Implementation tests | 50 passed |
| Frozen merged simulation evidence | All eight tables passed hashes, row counts, task-set checks, and source-manifest checks |
| Fresh-clone `make verify` | Passed using shipped `output/pdf/` files, without first building ignored PDFs |
| Current benchmark dry run | Planned 180 jobs; all four retained configurations passed their feature-validity gate |
| Reduced simulation smoke run | Completed; 38 rows, manifest marked `smoke-test` |
| `make replay`, current retained source | All 38 rows of one full paper job matched within `1e-9` absolute tolerance |
| Same replay, restored historical Mackey–Glass source | Passed with the exact frozen four-file source hash |
| Full `make all` in a disposable fresh clone | Passed: tests, both-device hardware reanalysis, all five figures, four PDFs, verification, anonymous package, and arXiv source package |
| Anonymous package isolated build | Passed; main paper six pages, supplement four pages |
| Rebuilt main-paper visual check | All six pages inspected; text, figures, and layout intact |

The complete paper-job replay used the balanced Mackey–Glass trajectory 5 and
reservoir seed 1729, with the full configuration's shot/grouping seeds and all six
methods. Chosen group sizes, group counts, ridge strengths, and covariance
strengths matched exactly. The maximum absolute difference in test NRMSE was
`6.51e-15`; validation fold scores also passed the `1e-9` tolerance. The current
and restored historical source gave the same comparison outcome. This checks
one job, not the whole simulation campaign.

Both hardware reanalyses passed the manuscript's numerical checks. Changed CSV
entries differed by at most `1.00e-13` in the compared numeric columns, with
matching table shapes. Some generated PDFs had different bytes after rebuilding;
original released PDFs and frozen tables were preserved. The disposable rebuild
was used for validation only.

## Known limits

- The full simulation campaigns were not rerun for this cleanup.
- Two older frozen simulation source hashes are not recoverable from the retained
  Git history. The guide documents the exact hashes and the separately tested
  Mackey–Glass recovery route. See
  [version history and replay limits](REPRODUCIBILITY.md#version-history-and-replay-limits).
- The lock file supplies a repeatable current package environment. It does not
  prove that the original simulations used identical numerical-library builds.
- Archived hardware records support reanalysis. A new acquisition on a physical
  processor is a separate experiment.
- These are local checks. The [GitHub Actions history](https://github.com/eybmits/adaptive-shot-aggregation/actions/workflows/verify.yml)
  separately reports CI results for each pushed commit.
