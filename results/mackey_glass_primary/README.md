# Mackey Glass primary evidence

This directory contains the frozen Mackey Glass evidence used by the paper.
The evaluation covers twenty independent trajectories under the same three
fixed execution allocations as the other simulation tasks. Reservoir, shot,
and grouping seeds are nested within each trajectory and are not treated as
independent forecasting units.

The balanced setting contains 500 time steps and 20 shots per step. Its primary
endpoint is the paired difference between EV and Grouped test NRMSE. The group
size and ridge strength are chosen on three chronological validation folds, and
the final test block is evaluated once.

The protocol is defined by `configs/mackey_glass_primary.json`. The manifest
records the configuration and implementation hashes, row counts, output paths,
and test-access contract. `SHA256SUMS` covers every consolidated evidence file.

The resumable worker shards are excluded from version control because the
consolidated tables and manifest contain the complete paper-facing evidence.
