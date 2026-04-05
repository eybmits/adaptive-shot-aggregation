#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qrc_budget_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from exp2_plot_style import (
    AXIS_COLOR,
    EVDUP_COLOR,
    EVNA_COLOR,
    FD_COLOR,
    FD_LS,
    FD_LW,
    FD_MARKER,
    FIG_W,
    FILL_ALPHA_FD,
    FILL_ALPHA_ID,
    FILL_ALPHA_RAW,
    ID_COLOR,
    ID_LS,
    ID_LW,
    ID_MARKER,
    MARKER_EDGE_COLOR,
    MARKER_EDGE_WIDTH,
    RAW_COLOR,
    RAW_LS,
    RAW_LW,
    RAW_MARKER,
    REFERENCE_ALPHA,
    REFERENCE_COLOR,
    REFERENCE_LS,
    REFERENCE_LW,
    SPLITNA_COLOR,
    TITLE_MAP,
    TRUE_COLOR,
    apply_style,
    save_pdf,
    scaled_height,
    style_axis,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TASK_ORDER = ["mackey_glass", "lorenz", "narma10"]
ARCH_LABELS = {
    ("ring_cnot", 1): "Ring d=1",
    ("ring_cnot", 2): "Ring d=2",
    ("ring_cnot", 4): "Ring d=4",
    ("line_cnot", 1): "Line d=1",
    ("all_to_all_cz", 1): "All-to-all d=1",
}


def load_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path)

def load_representative_traces() -> Dict[str, np.ndarray]:
    path = RESULTS / "representative_traces.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    traces = np.load(path)
    return {key: traces[key] for key in traces.files}


def representative_limits(values: Iterable[float]) -> tuple[float, float]:
    vals = np.asarray(list(values), dtype=float)
    ymin = max(0.0, np.floor((vals.min() - 0.05) / 0.05) * 0.05)
    ymax = min(1.6, np.ceil((vals.max() + 0.05) / 0.05) * 0.05)
    if ymax - ymin < 0.20:
        ymax = ymin + 0.20
    return float(ymin), float(ymax)


def protocol_summary_panel(ax, data: Dict[str, np.ndarray], *, ylabel: str) -> None:
    order = [
        ("EV", FD_COLOR, FD_MARKER),
        ("Raw", RAW_COLOR, RAW_MARKER),
        ("Split", ID_COLOR, ID_MARKER),
    ]
    x = np.arange(len(order))
    for i, (label, color, marker) in enumerate(order):
        vals = np.asarray(data[label], dtype=float)
        jitter = np.linspace(-0.06, 0.06, len(vals))
        ax.scatter(
            np.full_like(vals, i, dtype=float) + jitter,
            vals,
            s=16,
            color=color,
            alpha=0.25,
            edgecolors="none",
            zorder=2,
        )
        ax.errorbar(
            i,
            vals.mean(),
            yerr=vals.std(ddof=1),
            fmt=marker,
            color=color,
            markersize=5.5 if label != "Raw" else 5.2,
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            capsize=3.0,
            linewidth=1.4,
            zorder=3,
        )
    ax.set_xticks(x, [label for label, _, _ in order])
    ax.set_ylabel(ylabel)
    style_axis(ax, grid=True)

def representative_legend_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], color=TRUE_COLOR, lw=1.8, label="True"),
        Line2D([0], [0], color=RAW_COLOR, lw=RAW_LW, ls=RAW_LS, label="Raw"),
        Line2D([0], [0], color=FD_COLOR, lw=FD_LW, ls=FD_LS, label="EV"),
        Line2D([0], [0], color=ID_COLOR, lw=ID_LW, ls=ID_LS, label="Split"),
    ]


TRACE_XMAX = {
    "mackey_glass": 100,
    "lorenz": 100,
    "narma10": 20,
}


def plot_representative_trace(
    ax,
    row: pd.Series,
    traces: Dict[str, np.ndarray],
    trace_index_map: Dict[str, Dict[int, int]],
    *,
    ylabel: str | None = None,
) -> None:
    dataset = str(row["dataset"])
    seed = int(row["representative_seed"])
    idx = trace_index_map[dataset][seed]
    y_true = traces[f"{dataset}_y_test"][idx]
    y_raw = traces[f"{dataset}_raw"][idx]
    y_ev = traces[f"{dataset}_ev"][idx]
    y_split = traces[f"{dataset}_split"][idx]
    x = np.arange(len(y_true))
    ax.plot(x, y_true, color=TRUE_COLOR, lw=1.8, label="True")
    ax.plot(x, y_raw, color=RAW_COLOR, lw=RAW_LW, ls=RAW_LS, label="Raw")
    ax.plot(x, y_ev, color=FD_COLOR, lw=FD_LW, ls=FD_LS, label="EV")
    ax.plot(x, y_split, color=ID_COLOR, lw=ID_LW, ls=ID_LS, label="Split")
    dataset = str(row["dataset"])
    xmax = TRACE_XMAX.get(dataset, int(x[-1]))
    ax.set_xlim(0, xmax)
    if xmax <= 20:
        ax.set_xticks([0, 10, 20])
    else:
        ax.set_xticks([0, 50, 100])
    ax.set_title(TITLE_MAP[dataset], pad=0, y=1.03)
    ax.set_xlabel("Test time step")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    style_axis(ax, grid=True)


def make_experiment_1(rep_summary: pd.DataFrame, rep_seed: pd.DataFrame, traces: Dict[str, np.ndarray]) -> None:
    apply_style()
    handles = representative_legend_handles()
    trace_index_map = {
        dataset: {
            int(seed): idx
            for idx, seed in enumerate(
                sorted(rep_seed[rep_seed["dataset"] == dataset]["seed"].astype(int).unique().tolist())
            )
        }
        for dataset in TASK_ORDER
    }

    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, scaled_height(7.0, 5.55)))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.91, bottom=0.20, hspace=0.36, wspace=0.25)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        frameon=False,
        ncol=4,
        columnspacing=1.4,
        handlelength=2.0,
    )
    for cidx, dataset in enumerate(TASK_ORDER):
        row = rep_summary[rep_summary["dataset"] == dataset].iloc[0]
        plot_representative_trace(
            axes[0, cidx],
            row,
            traces,
            trace_index_map,
            ylabel="Target" if cidx == 0 else None,
        )

        sub = rep_seed[rep_seed["dataset"] == dataset]
        score_map = {
            "EV": sub["ev_nrmse"].to_numpy(dtype=float),
            "Raw": sub["raw_nrmse"].to_numpy(dtype=float),
            "Split": sub["split_nrmse"].to_numpy(dtype=float),
        }
        protocol_summary_panel(axes[1, cidx], score_map, ylabel="Test NRMSE" if cidx == 0 else "")
        ymin, ymax = representative_limits(np.concatenate(list(score_map.values())))
        axes[1, cidx].set_ylim(ymin, ymax)
    save_pdf(fig, FIGURES / "experiment_1.pdf")

def make_experiment_2(emp_seed: pd.DataFrame, emp_summary: pd.DataFrame) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_W, scaled_height(7.0, 3.15)))
    fig.subplots_adjust(left=0.15, right=0.985, top=0.965, bottom=0.17)
    x_min, x_max = 0.45, 5.30
    cond_end, trans_end = 1.00, 2.70
    specs = [
        ("mackey_glass", ID_COLOR, ID_MARKER),
        ("lorenz", FD_COLOR, FD_MARKER),
        ("narma10", REFERENCE_COLOR, "D"),
    ]
    for dataset, color, marker in specs:
        seed_sub = emp_seed[emp_seed["dataset"] == dataset]
        xvals = seed_sub["rho_ev"].to_numpy(dtype=float)
        yvals = seed_sub["gap"].to_numpy(dtype=float)
        jitter = np.linspace(-0.03, 0.03, len(yvals))
        ax.scatter(xvals + jitter, yvals, s=12, color=color, alpha=0.16, edgecolors="none", zorder=1)
        sub = emp_summary[emp_summary["dataset"] == dataset].sort_values("rho_ev")
        ax.fill_between(
            sub["rho_ev"],
            sub["gap_mean"] - sub["gap_std"].fillna(0.0),
            sub["gap_mean"] + sub["gap_std"].fillna(0.0),
            color=color,
            alpha=0.10 if dataset == "mackey_glass" else (0.09 if dataset == "lorenz" else 0.08),
            linewidth=0.0,
            zorder=2,
        )
        ax.plot(
            sub["rho_ev"],
            sub["gap_mean"],
            color=color,
            lw=ID_LW if dataset == "mackey_glass" else (FD_LW if dataset == "lorenz" else REFERENCE_LW + 0.4),
            ls=ID_LS if dataset == "mackey_glass" else (FD_LS if dataset == "lorenz" else "-"),
            marker=marker,
            markersize=5.5 if dataset != "lorenz" else 5.2,
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            markerfacecolor=color,
            label=TITLE_MAP[dataset],
            zorder=3,
        )
    ax.axhline(0.0, color=REFERENCE_COLOR, lw=REFERENCE_LW, ls=REFERENCE_LS, alpha=REFERENCE_ALPHA)
    ax.axvline(cond_end, color="#000000", lw=1.0, ls="--", alpha=0.8)
    ax.axvline(trans_end, color="#000000", lw=1.0, ls="--", alpha=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.1, 0.7)
    ax.set_yticks(np.arange(-0.1, 0.71, 0.2))
    ax.set_xlabel(r"$\rho_{\mathrm{EV}}$")
    ax.set_ylabel("NRMSE gap (EV - Split)")
    ax.legend(loc="upper right", frameon=False)
    style_axis(ax, grid=True)
    save_pdf(fig, FIGURES / "experiment_2.pdf")


def make_experiment_4(base_seed: pd.DataFrame) -> None:
    apply_style()
    protocols = [
        ("ev_nrmse", "EV", FD_COLOR),
        ("raw_nrmse", "Raw", RAW_COLOR),
        ("evdup_nrmse", "EV-dup", EVDUP_COLOR),
        ("ev_na_nrmse", "EV-NA", EVNA_COLOR),
        ("split_nrmse", "Split", ID_COLOR),
        ("split_na_nrmse", "Split-NA", SPLITNA_COLOR),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, scaled_height(7.0, 3.05)))
    fig.subplots_adjust(left=0.092, right=0.99, top=0.80, bottom=0.30, wspace=0.29)
    for ax, dataset in zip(axes, TASK_ORDER):
        sub = base_seed[base_seed["dataset"] == dataset]
        vals = [sub[col].to_numpy(dtype=float) for col, _, _ in protocols]
        bplot = ax.boxplot(
            vals,
            patch_artist=True,
            widths=0.58,
            showfliers=False,
            medianprops={"color": AXIS_COLOR, "linewidth": 1.2},
            whiskerprops={"color": AXIS_COLOR, "linewidth": 0.9},
            capprops={"color": AXIS_COLOR, "linewidth": 0.9},
            boxprops={"edgecolor": AXIS_COLOR, "linewidth": 0.9},
        )
        for patch, (_, _, color) in zip(bplot["boxes"], protocols):
            patch.set_facecolor(color)
            patch.set_alpha(0.64)
        for xpos, (series, (_, _, color)) in enumerate(zip(vals, protocols), start=1):
            if len(series) == 0:
                continue
            sorted_vals = np.sort(np.asarray(series, dtype=float))
            jitter = np.linspace(-0.08, 0.08, len(sorted_vals))
            ax.scatter(
                np.full(len(sorted_vals), xpos, dtype=float) + jitter,
                sorted_vals,
                s=9,
                color=color,
                alpha=0.22,
                edgecolors="none",
                zorder=2.2,
            )
        ax.set_title(TITLE_MAP[dataset], pad=0, y=1.03)
        ax.set_xticks(np.arange(1, len(protocols) + 1), [label for _, label, _ in protocols], rotation=28, ha="right")
        if dataset == TASK_ORDER[0]:
            ax.set_ylabel("Test NRMSE")
        style_axis(ax, grid=True)
    save_pdf(fig, FIGURES / "experiment_4.pdf")

def make_experiment_5(dup_seed: pd.DataFrame, dup_summary: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, scaled_height(7.0, 3.00)))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.92, bottom=0.23, wspace=0.22)
    marker_map = {
        "EV": (FD_MARKER, 5.2),
        "EV-dup": ("D", 5.0),
        "Split": (ID_MARKER, 5.5),
    }
    for ax, dataset in zip(axes, TASK_ORDER):
        seed_sub = dup_seed[dup_seed["dataset"] == dataset]
        sub = dup_summary[dup_summary["dataset"] == dataset].sort_values("n_shots")
        for col, sd_col, label, color, ls, lw, alpha in [
            ("ev_nrmse_mean", "ev_nrmse_std", "EV", FD_COLOR, FD_LS, FD_LW, FILL_ALPHA_FD),
            ("evdup_nrmse_mean", "evdup_nrmse_std", "EV-dup", EVDUP_COLOR, "-.", 1.6, 0.12),
            ("split_nrmse_mean", "split_nrmse_std", "Split", ID_COLOR, ID_LS, ID_LW, FILL_ALPHA_ID),
        ]:
            x = sub["n_shots"].to_numpy(dtype=float)
            y = sub[col].to_numpy(dtype=float)
            e = sub[sd_col].to_numpy(dtype=float)
            seed_col = {
                "EV": "ev_nrmse",
                "EV-dup": "evdup_nrmse",
                "Split": "split_nrmse",
            }[label]
            ax.scatter(
                seed_sub["n_shots"],
                seed_sub[seed_col],
                s=11,
                color=color,
                alpha=0.12,
                edgecolors="none",
                zorder=1,
            )
            ax.fill_between(x, y - e, y + e, color=color, alpha=alpha, linewidth=0.0)
            marker, markersize = marker_map[label]
            ax.plot(
                x,
                y,
                color=color,
                ls=ls,
                lw=lw,
                marker=marker,
                markersize=markersize,
                markeredgecolor=MARKER_EDGE_COLOR,
                markeredgewidth=MARKER_EDGE_WIDTH,
                label=label,
            )
        ax.set_title(TITLE_MAP[dataset], pad=6)
        ax.set_xlabel(r"$N_{\mathrm{shots}}$")
        if dataset == TASK_ORDER[0]:
            ax.set_ylabel("Test NRMSE")
            ax.legend(frameon=False, loc="upper left")
        style_axis(ax, grid=True)
    save_pdf(fig, FIGURES / "experiment_5.pdf")


def make_experiment_3(more_qubits_seed: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, scaled_height(7.0, 3.72)))
    fig.subplots_adjust(left=0.11, right=0.95, top=0.91, bottom=0.19, wspace=0.38)
    gap_min = float(np.floor((more_qubits_seed["gap"].min() - 0.03) / 0.05) * 0.05)
    gap_max = float(np.ceil((more_qubits_seed["gap"].max() + 0.03) / 0.05) * 0.05)
    legend_handles = [
        Line2D([0], [0], color=FD_COLOR, lw=FD_LW, ls=FD_LS, marker=FD_MARKER, markersize=5.2, markeredgecolor=MARKER_EDGE_COLOR, markeredgewidth=MARKER_EDGE_WIDTH, label="EV"),
        Line2D([0], [0], color=ID_COLOR, lw=ID_LW, ls=ID_LS, marker=ID_MARKER, markersize=5.5, markeredgecolor=MARKER_EDGE_COLOR, markeredgewidth=MARKER_EDGE_WIDTH, label="Split"),
        Line2D([0], [0], color=REFERENCE_COLOR, lw=1.3, ls="-", marker="o", markersize=4.4, markerfacecolor="white", markeredgecolor=REFERENCE_COLOR, label="Gap"),
    ]
    for cidx, dataset in enumerate(TASK_ORDER):
        ax = axes[cidx]
        ax_gap = ax.twinx()
        sub = more_qubits_seed[more_qubits_seed["dataset"] == dataset]
        gap_grp = sub.groupby("q", as_index=False).agg(gap_mean=("gap", "mean"), gap_std=("gap", "std"))
        abs_grp = sub.groupby("q", as_index=False).agg(
            ev_mean=("ev_nrmse", "mean"),
            ev_std=("ev_nrmse", "std"),
            split_mean=("split_nrmse", "mean"),
            split_std=("split_nrmse", "std"),
        )
        xs = abs_grp["q"].to_numpy(dtype=float)

        ax.scatter(sub["q"], sub["ev_nrmse"], s=11, color=FD_COLOR, alpha=0.12, edgecolors="none", zorder=1)
        ax.scatter(sub["q"], sub["split_nrmse"], s=11, color=ID_COLOR, alpha=0.12, edgecolors="none", zorder=1)
        ax.fill_between(
            xs,
            abs_grp["ev_mean"] - abs_grp["ev_std"].fillna(0.0),
            abs_grp["ev_mean"] + abs_grp["ev_std"].fillna(0.0),
            color=FD_COLOR,
            alpha=0.10,
            linewidth=0.0,
            zorder=1,
        )
        ax.fill_between(
            xs,
            abs_grp["split_mean"] - abs_grp["split_std"].fillna(0.0),
            abs_grp["split_mean"] + abs_grp["split_std"].fillna(0.0),
            color=ID_COLOR,
            alpha=0.12,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(xs, abs_grp["ev_mean"], color=FD_COLOR, lw=FD_LW, ls=FD_LS, marker=FD_MARKER, markersize=5.2, markeredgecolor=MARKER_EDGE_COLOR, markeredgewidth=MARKER_EDGE_WIDTH, zorder=3)
        ax.plot(xs, abs_grp["split_mean"], color=ID_COLOR, lw=ID_LW, ls=ID_LS, marker=ID_MARKER, markersize=5.5, markeredgecolor=MARKER_EDGE_COLOR, markeredgewidth=MARKER_EDGE_WIDTH, zorder=3)

        ax_gap.scatter(sub["q"], sub["gap"], s=10, color=REFERENCE_COLOR, alpha=0.10, edgecolors="none", zorder=1)
        ax_gap.fill_between(
            gap_grp["q"],
            gap_grp["gap_mean"] - gap_grp["gap_std"].fillna(0.0),
            gap_grp["gap_mean"] + gap_grp["gap_std"].fillna(0.0),
            color=REFERENCE_COLOR,
            alpha=0.08,
            linewidth=0.0,
            zorder=1,
        )
        ax_gap.plot(
            gap_grp["q"],
            gap_grp["gap_mean"],
            color=REFERENCE_COLOR,
            lw=1.3,
            ls="-",
            marker="o",
            markersize=4.4,
            markerfacecolor="white",
            markeredgecolor=REFERENCE_COLOR,
            zorder=2,
        )
        ax_gap.axhline(0.0, color=REFERENCE_COLOR, lw=REFERENCE_LW, ls=REFERENCE_LS, alpha=REFERENCE_ALPHA, zorder=1)

        ax.set_title(TITLE_MAP[dataset], pad=4)
        ax.set_xlabel(r"Reservoir size $Q$")
        if cidx == 0:
            ax.set_ylabel("Test NRMSE")
        if cidx == len(TASK_ORDER) - 1:
            ax_gap.set_ylabel("Gap (EV - Split)", color=REFERENCE_COLOR)
        else:
            ax_gap.set_yticklabels([])
            ax_gap.tick_params(length=0)
        ax.set_xticks(sorted(sub["q"].unique()))
        ax.set_box_aspect(1.0)
        style_axis(ax, grid=True)
        ax_gap.set_ylim(gap_min, gap_max)
        ax_gap.grid(False)
        ax_gap.spines["right"].set_linewidth(0.8)
        ax_gap.spines["right"].set_color(REFERENCE_COLOR)
        ax_gap.spines["top"].set_visible(False)
        ax_gap.spines["left"].set_visible(False)
        ax_gap.spines["bottom"].set_visible(False)
        ax_gap.tick_params(direction="out", width=0.8, length=3.0, pad=4, colors=REFERENCE_COLOR)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        frameon=False,
        ncol=3,
        columnspacing=1.6,
        handlelength=2.0,
    )

    out_more = FIGURES / "experiment_3.pdf"
    out_more.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_more, format="pdf")
    plt.close(fig)


def make_experiment_7(arch_seed: pd.DataFrame, arch_summary: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, scaled_height(7.0, 3.55)))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.87, bottom=0.36, wspace=0.30)
    order = list(ARCH_LABELS.keys())
    xtick_labels = [ARCH_LABELS[key] for key in order]
    for ax, dataset in zip(axes, TASK_ORDER):
        sub_seed = arch_seed[arch_seed["dataset"] == dataset].copy()
        sub_summary = arch_summary[arch_summary["dataset"] == dataset].copy()
        gap_values = sub_seed["ev_nrmse"].to_numpy(dtype=float) - sub_seed["split_nrmse"].to_numpy(dtype=float)
        y_min = min(-0.02, float(np.floor((gap_values.min() - 0.03) / 0.05) * 0.05))
        y_max = max(0.05, float(np.ceil((gap_values.max() + 0.03) / 0.05) * 0.05))
        for xpos, key in enumerate(order):
            entangler, depth = key
            seed_cell = sub_seed[
                (sub_seed["entangler"] == entangler)
                & (sub_seed["depth"].astype(int) == depth)
            ].copy()
            if seed_cell.empty:
                continue
            seed_gap = seed_cell["ev_nrmse"].to_numpy(dtype=float) - seed_cell["split_nrmse"].to_numpy(dtype=float)
            jitter = np.linspace(-0.10, 0.10, len(seed_gap))
            ax.scatter(
                np.full(len(seed_gap), xpos, dtype=float) + jitter,
                np.sort(seed_gap),
                s=10,
                color=ID_COLOR,
                alpha=0.16,
                edgecolors="none",
                zorder=1,
            )
            row = sub_summary[
                (sub_summary["entangler"] == entangler)
                & (sub_summary["depth"].astype(int) == depth)
            ].iloc[0]
            ax.errorbar(
                xpos,
                float(row["gap_ev_minus_split"]),
                yerr=float(np.sqrt(row["ev_nrmse_std"] ** 2 + row["split_nrmse_std"] ** 2)),
                fmt="o",
                color=ID_COLOR,
                markersize=5.6,
                markeredgecolor=MARKER_EDGE_COLOR,
                markeredgewidth=MARKER_EDGE_WIDTH,
                capsize=3.0,
                linewidth=1.3,
                zorder=3,
            )
        ax.axhline(0.0, color=REFERENCE_COLOR, lw=REFERENCE_LW, ls=REFERENCE_LS, alpha=REFERENCE_ALPHA, zorder=0)
        ax.set_title(TITLE_MAP[dataset], pad=3)
        ax.set_xticks(np.arange(len(order)), xtick_labels, rotation=28, ha="right")
        if dataset == TASK_ORDER[0]:
            ax.set_ylabel("Gap (EV - Split)")
        ax.set_ylim(y_min, y_max)
        style_axis(ax, grid=True)
    save_pdf(fig, FIGURES / "experiment_7.pdf")


def main() -> None:
    global FIGURES
    parser = argparse.ArgumentParser(description="Replot manuscript figures in the exp2 PDF style.")
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    args = parser.parse_args()

    FIGURES = args.out_dir.resolve()

    rep_summary = load_csv("representative_summary.csv")
    rep_seed = load_csv("representative_seed_level.csv")
    base_seed = load_csv("baseline_controls_seed_level.csv")
    emp_seed = load_csv("empirical_gap_seed_level.csv")
    emp_summary = load_csv("empirical_gap_summary.csv")
    dup_seed = load_csv("duplication_control_seed_level.csv")
    dup_summary = load_csv("duplication_control_summary.csv")
    more_qubits_seed = load_csv("more_qubits_seed_level.csv")
    arch_seed = load_csv("architecture_ablation_seed_level.csv")
    arch_summary = load_csv("architecture_ablation_summary.csv")
    traces = load_representative_traces()

    print("writing PDFs", flush=True)
    make_experiment_1(rep_summary, rep_seed, traces)
    make_experiment_2(emp_seed, emp_summary)
    make_experiment_4(base_seed)
    make_experiment_5(dup_seed, dup_summary)
    make_experiment_3(more_qubits_seed)
    make_experiment_7(arch_seed, arch_summary)
    print(f"done: {FIGURES}", flush=True)


if __name__ == "__main__":
    main()
