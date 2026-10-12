#!/usr/bin/env python3
"""Replay one complete paper simulation job and compare it with saved evidence.

This is the balanced Mackey–Glass trajectory 5, reservoir seed 1729 job. It uses
the full paper settings, all shot/grouping seeds, and all six methods. It is a
bounded numerical check, not a rerun of the entire experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_robust_evaluation as runner

ROOT = Path(__file__).resolve().parent.parent
ATOL = 1e-9
KEYS = [
    "dataset", "cluster_id", "budget_setting", "reservoir_variant",
    "reservoir_seed", "shot_seed", "partition_seed", "partition_mode",
    "partition_role", "scope_role", "feature_set", "method",
]
METRICS = [
    "k", "groups", "lambda_selected", "alpha_selected", "mean_val_nrmse",
    "second_best_val_nrmse", "selection_margin", "test_nrmse",
]


def main() -> None:
    config_path = ROOT / "configs" / "mackey_glass_primary.json"
    config = runner.load_config(config_path)
    runner.feature_gate(config, smoke=False, allow_override=False)
    jobs = [job for job in runner.planned_jobs(config, config_path, smoke=False)
            if job["setting"]["name"] == "balanced"
            and job["cluster_index"] == 5 and job["reservoir_seed"] == 1729]
    if len(jobs) != 1:
        raise RuntimeError(f"expected exactly one replay job, got {len(jobs)}")
    rows, _, _ = runner.run_job(jobs[0])
    replay = pd.DataFrame(rows)
    saved = pd.read_csv(ROOT / "results" / "mackey_glass_primary" / "replicate_results.csv")
    saved = saved.loc[
        saved["budget_setting"].eq("balanced") & saved["cluster_index"].eq(5)
        & (saved["reservoir_seed"].eq(1729) | saved["method"].eq("classical_input"))
    ]
    # pandas matches missing seed keys, which occur for the classical/ideal controls.
    matched = saved.merge(replay, on=KEYS, how="outer", suffixes=("_saved", "_replay"),
                          validate="one_to_one", indicator=True)
    if not matched["_merge"].eq("both").all() or len(matched) != 38:
        raise RuntimeError("replay does not match the expected 38 paper rows")
    errors = {}
    for metric in METRICS:
        expected = matched[f"{metric}_saved"].to_numpy(dtype=float)
        observed = matched[f"{metric}_replay"].to_numpy(dtype=float)
        if not np.allclose(observed, expected, atol=ATOL, rtol=0.0, equal_nan=True):
            raise RuntimeError(f"{metric} differs from the saved evidence by more than {ATOL}")
        finite = np.isfinite(expected) & np.isfinite(observed)
        errors[metric] = float(np.max(np.abs(observed[finite] - expected[finite]))) if finite.any() else 0.0
    for _, row in matched.iterrows():
        expected = json.loads(row["fold_val_nrmse_saved"])
        observed = json.loads(row["fold_val_nrmse_replay"])
        if not np.allclose(observed, expected, atol=ATOL, rtol=0.0):
            raise RuntimeError("validation fold scores differ from the saved evidence")
    print(json.dumps({
        "status": "passed", "dataset": "mackey_glass", "cluster_index": 5,
        "budget_setting": "balanced", "reservoir_seed": 1729, "rows": len(matched),
        "absolute_tolerance": ATOL, "maximum_absolute_error_by_metric": errors,
        "implementation_sha256": runner.implementation_sha256(),
        "scope": "one full simulation job; not the complete paper experiment",
    }, indent=2))


if __name__ == "__main__":
    main()
