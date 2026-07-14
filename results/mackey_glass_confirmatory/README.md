# Mackey-Glass confirmatory expansion

This run evaluates the balanced setting on twenty previously unused,
independently initialized Mackey-Glass trajectories with indices 5 through 24.
It keeps the main protocol's reservoir, shot, and grouping seeds nested within
each trajectory. The original headline trajectories 0 through 4 are excluded.

The endpoint was fixed before execution as the trajectory-level paired
difference in test NRMSE, `EV - Grouped`. A positive value favors Grouped, and
confirmation requires the lower bound of the two-sided 95% t interval to exceed
zero.

The result is a mean difference of `0.044543`, with 95% CI
`[0.037176, 0.051911]`. Grouped has lower mean test NRMSE on all twenty
trajectories. Mean EV and Grouped errors are `0.483764` and `0.439221`,
respectively, corresponding to a relative reduction of `9.21%`.

Run from the repository root with:

```sh
python scripts/run_robust_evaluation.py \
  --config configs/mackey_glass_confirmatory.json \
  --results-dir results/mackey_glass_confirmatory \
  --workers 8
```

The registered output is `cluster_paired_stats.csv`. The full nested results,
validation spectra, selected group-size frequencies, configuration hash, code
hash, and execution manifest are stored alongside it. Worker checkpoints are
ignored because they are derivable from the released configuration and code.
