#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"

spec = importlib.util.spec_from_file_location("run20_common_point", HERE / "run_20seed_experiments.py")
run20 = importlib.util.module_from_spec(spec)
sys.modules["run20_common_point"] = run20
spec.loader.exec_module(run20)

base = run20.base
Config = run20.Config
build_representative_summary = run20.build_representative_summary


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    cfg = Config()
    print("rerunning common operating point", flush=True)
    print(f"config: n_shots={cfg.representative_n_shots}, sigma={cfg.representative_sigma:.2f}, lambda={cfg.lambda_default}", flush=True)

    rep_rows, rep_seed, _ = base.run_representative_suite(cfg)
    rep_seed = rep_seed.drop(columns=["sigma"], errors="ignore")
    rep_summary = build_representative_summary(rep_rows, rep_seed, cfg)
    rep_seed.to_csv(RESULTS / "representative_seed_level.csv", index=False)
    rep_summary.to_csv(RESULTS / "representative_summary.csv", index=False)
    np.savez_compressed(
        RESULTS / "representative_traces.npz",
        **{
            f"{dataset}_y_test": np.stack([row["trace_y_test"] for row in rep_rows if row["dataset"] == dataset], axis=0)
            for dataset in run20.TASK_ORDER
        },
        **{
            f"{dataset}_{protocol}": np.stack([row[f"trace_{protocol}"] for row in rep_rows if row["dataset"] == dataset], axis=0)
            for dataset in run20.TASK_ORDER
            for protocol in ["ev", "raw", "split"]
        },
    )

    baseline_seed, baseline_summary = base.run_baseline_suite(cfg)
    baseline_seed.to_csv(RESULTS / "baseline_controls_seed_level.csv", index=False)
    baseline_summary.to_csv(RESULTS / "baseline_controls_summary.csv", index=False)
    print("updated representative and baseline result files", flush=True)


if __name__ == "__main__":
    main()
