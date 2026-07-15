# Reproducibility

## Protocol

In simulation, the four tasks share one six-qubit reservoir, feature map,
preprocessing pipeline,
execution budget, validation-fold design, and parameter grid. Mackey Glass uses
twenty independent trajectories and Rössler uses five. Santa Fe and sunspots use five
chronological windows with disjoint test blocks. Three reservoir seeds, three
shot seeds, and three grouping seeds are nested within each forecasting unit.
Nested repetitions are averaged before confidence intervals are computed.

The three execution allocations are `(T, N) = (250, 40), (500, 20),
(1000, 10)`. Each contains 10,000 circuit executions. The middle allocation is
the primary comparison.

## Joint tuning

All methods receive the same measurement record. Scaling is fitted inside each
training fold. Three expanding chronological validation folds tune aggregation
scale and ridge regularization jointly. Grouped predictions are averaged to one
prediction per time step before validation NRMSE is scored. The tuned pair is
frozen before the final test block is evaluated once.

## Evidence files

The paper reads the merged tables in
`results/qai_robust_protocol_santa_fe_manuscript/`. Its `manifest.json` records
the source artifacts, row counts, and SHA256 hashes. The source result
directories are retained under their frozen identifiers because changing them
would break that provenance chain.

Hardware tables and ordered raw records are in `results/qai_hardware/`. The
Santa Fe job and budget record is
`results/qai_hardware_santa_fe_extension/manifest.json`.

Hardware uses a separate fixed ten-qubit reservoir with local Z readout. It is
a transfer check for the aggregation procedure, not a device realization of
the six-qubit simulation reservoir. The circuit distinction and both exact
contracts are documented in the paper supplement and `docs/HARDWARE.md`.

## Verification

```bash
make test
make verify
make figures
make paper
make release
make arxiv
```

`make verify` checks table hashes and row counts, admitted numerical results,
hardware jobs and results, six page limits, PDF titles, figure task labels, and
anonymous PDF privacy. `make release` creates an isolated anonymous source
package and rebuilds it in a temporary directory. `make arxiv` creates a
named-only source ZIP, removes local build products and anonymous-review
branches, and verifies a clean six-page build using only the archive contents.

The consolidated four-task merge can also be checked directly without writing
any evidence file:

```bash
python scripts/build_santa_fe_manuscript_results.py --validate-only
```

The full frozen-evidence check is:

```bash
make all
```

Generated packages are written to `release/` and are not tracked. The release
archives are `adaptive_shot_aggregation_anonymous_review.zip` and
`adaptive_shot_aggregation_arxiv.zip`. Final named and anonymous PDFs and their
SHA256 checksums are stored under `output/pdf/`.

## Expensive rerun

The current experiment definition is `configs/benchmark.json`.

```bash
python scripts/run_robust_evaluation.py \
  --config configs/benchmark.json \
  --results-dir results/benchmark \
  --workers 4 \
  --resume
```

The runner stores independent shards and resumes only when configuration and
implementation hashes match. The checked frozen evidence is included, so a
full simulation rerun is not required for paper verification.

Every manifest fails closed if its configuration or implementation hash does
not match. New runs must use a new results directory and receive their own
manifest and hashes, leaving the released evidence unchanged.
