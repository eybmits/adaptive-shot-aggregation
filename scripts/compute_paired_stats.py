#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
TASK_ORDER = ["mackey_glass", "lorenz", "narma10"]


def paired_stats_frame(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    a_col: str,
    b_col: str,
    a_label: str,
    b_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(list(group_cols), dropna=False) if group_cols else [((), df)]
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        a = sub[a_col].to_numpy(dtype=float)
        b = sub[b_col].to_numpy(dtype=float)
        d = a - b
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "compare_a": a_label,
                "compare_b": b_label,
                "metric_a_col": a_col,
                "metric_b_col": b_col,
                "n": int(len(sub)),
                "mean_a": float(np.mean(a)),
                "mean_b": float(np.mean(b)),
                "delta_mean": float(np.mean(d)),
                "delta_median": float(np.median(d)),
                "delta_std": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                "delta_ci95": float(1.96 * np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0,
                "win_rate_a_beats_b": float(np.mean(d > 0.0)),
                "loss_rate_a_loses_to_b": float(np.mean(d < 0.0)),
                "ties": int(np.sum(d == 0.0)),
            }
        )
        try:
            row["wilcoxon_p"] = float(wilcoxon(d).pvalue)
        except Exception:
            row["wilcoxon_p"] = np.nan
        try:
            row["ttest_p"] = float(ttest_rel(a, b).pvalue)
        except Exception:
            row["ttest_p"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if "dataset" in out.columns:
        out["dataset"] = pd.Categorical(out["dataset"], TASK_ORDER, ordered=True)
        out = out.sort_values(["dataset", *[c for c in group_cols if c != "dataset"]]).reset_index(drop=True)
    return out


def write_csv(df: pd.DataFrame, name: str) -> None:
    path = RESULTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote: {path}")


def main() -> None:
    rep = pd.read_csv(RESULTS_DIR / "representative_seed_level.csv")
    exp1 = pd.concat(
        [
            paired_stats_frame(
                rep,
                group_cols=["dataset"],
                a_col="ev_nrmse",
                b_col="split_nrmse",
                a_label="EV",
                b_label="Split",
            ),
            paired_stats_frame(
                rep,
                group_cols=["dataset"],
                a_col="raw_nrmse",
                b_col="split_nrmse",
                a_label="Raw",
                b_label="Split",
            ),
        ],
        ignore_index=True,
    )
    write_csv(exp1, "experiment_1_paired_stats.csv")

    emp = pd.read_csv(RESULTS_DIR / "empirical_gap_seed_level.csv")
    exp2_cells = paired_stats_frame(
        emp,
        group_cols=["dataset", "n_shots", "rho_ev"],
        a_col="ev_nrmse",
        b_col="split_nrmse",
        a_label="EV",
        b_label="Split",
    )
    write_csv(exp2_cells, "experiment_2_cell_paired_stats.csv")
    peak_rows = []
    for dataset in TASK_ORDER:
        sub = exp2_cells[exp2_cells["dataset"] == dataset].copy()
        peak_rows.append(sub.sort_values(["delta_mean", "n_shots"], ascending=[False, True]).iloc[0].to_dict())
    exp2_peak = pd.DataFrame(peak_rows)
    exp2_peak["dataset"] = pd.Categorical(exp2_peak["dataset"], TASK_ORDER, ordered=True)
    exp2_peak = exp2_peak.sort_values("dataset").reset_index(drop=True)
    write_csv(exp2_peak, "experiment_2_peak_paired_stats.csv")

    size = pd.read_csv(RESULTS_DIR / "more_qubits_seed_level.csv")
    exp3 = paired_stats_frame(
        size,
        group_cols=["dataset", "q", "rho_ev"],
        a_col="ev_nrmse",
        b_col="split_nrmse",
        a_label="EV",
        b_label="Split",
    )
    write_csv(exp3, "experiment_3_paired_stats.csv")

    base = pd.read_csv(RESULTS_DIR / "baseline_controls_seed_level.csv")
    exp4_frames = []
    for metric, label in [
        ("ev_nrmse", "EV"),
        ("raw_nrmse", "Raw"),
        ("evdup_nrmse", "EV-dup"),
        ("ev_na_nrmse", "EV-NA"),
        ("split_na_nrmse", "Split-NA"),
    ]:
        exp4_frames.append(
            paired_stats_frame(
                base,
                group_cols=["dataset"],
                a_col=metric,
                b_col="split_nrmse",
                a_label=label,
                b_label="Split",
            )
        )
    exp4 = pd.concat(exp4_frames, ignore_index=True)
    write_csv(exp4, "experiment_4_paired_stats.csv")

    dup = pd.read_csv(RESULTS_DIR / "duplication_control_seed_level.csv")
    exp5 = paired_stats_frame(
        dup,
        group_cols=["dataset", "n_shots", "rho_ev"],
        a_col="evdup_nrmse",
        b_col="split_nrmse",
        a_label="EV-dup",
        b_label="Split",
    )
    write_csv(exp5, "experiment_5_paired_stats.csv")

    arch = pd.read_csv(RESULTS_DIR / "architecture_ablation_seed_level.csv")
    arch_frames = [
        paired_stats_frame(
            arch,
            group_cols=["dataset", "q", "entangler", "depth", "n_shots", "rho_ev"],
            a_col="ev_nrmse",
            b_col="split_nrmse",
            a_label="EV",
            b_label="Split",
        )
    ]
    q12_path = RESULTS_DIR / "architecture_ablation_q12_seed_level.csv"
    if q12_path.exists():
        arch_q12 = pd.read_csv(q12_path)
        arch_frames.append(
            paired_stats_frame(
                arch_q12,
                group_cols=["dataset", "q", "entangler", "depth", "n_shots", "rho_ev"],
                a_col="ev_nrmse",
                b_col="split_nrmse",
                a_label="EV",
                b_label="Split",
            )
        )
    arch_stats = pd.concat(arch_frames, ignore_index=True)
    if "dataset" in arch_stats.columns:
        arch_stats["dataset"] = pd.Categorical(arch_stats["dataset"], TASK_ORDER, ordered=True)
        arch_stats = arch_stats.sort_values(
            ["dataset", "q", "entangler", "depth", "n_shots", "rho_ev"]
        ).reset_index(drop=True)
    write_csv(arch_stats, "architecture_ablation_paired_stats.csv")


if __name__ == "__main__":
    main()
