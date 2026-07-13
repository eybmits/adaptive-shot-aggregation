#!/usr/bin/env python3
"""Build simulation figures for the current manuscript.

The script deliberately performs inference at the independent
trajectory/window level.  Reservoir, shot, and partition seeds are averaged
inside each cluster before means and 95% t intervals are drawn.

Outputs
-------
paper/figures/fig2_composite.{pdf,png}
paper/figures/fig3_regime_map.{pdf,png}
paper/figures/fig4_controls_compact.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "qai_robust_protocol_santa_fe_manuscript"
FIGURE_DIR = ROOT / "paper" / "figures"

# Original manuscript palette.
FULL_AVERAGE = "#4a6fa5"
SPLIT = "#2a9d8f"
RAW = "#e3b341"  # deliberately yellow: individual shots, not a gray reference
COVARIANCE_EV = "#9c6b9e"
NEUTRAL = "#777777"
BLACK = "#171717"

TASKS = ("mackey_glass", "santa_fe_laser", "rossler", "sunspots")
TASK_TITLES = {
    "mackey_glass": "Mackey–Glass",
    "santa_fe_laser": "Santa Fe laser",
    "rossler": "Rössler",
    "sunspots": "Sunspots (real)",
}
TASK_SHORT = {
    "mackey_glass": "M–Glass",
    "santa_fe_laser": "Santa Fe",
    "rossler": "Rössler",
    "sunspots": "Sunspots",
}
TASK_COLORS = {
    "mackey_glass": FULL_AVERAGE,
    "santa_fe_laser": SPLIT,
    "rossler": "#c8793a",
    "sunspots": COVARIANCE_EV,
}
TASK_MARKERS = {
    "mackey_glass": "o",
    "santa_fe_laser": "D",
    "rossler": "s",
    "sunspots": "^",
}
METHOD_LABELS = {
    "ev": "EV",
    "split": "Grouped",
    "raw": "Raw",
    "covariance_ev": "Cov-EV",
}
METHOD_COLORS = {
    "raw": RAW,
    "ev": FULL_AVERAGE,
    "covariance_ev": COVARIANCE_EV,
    "split": SPLIT,
}
METHOD_MARKERS = {"raw": "o", "ev": "o", "covariance_ev": "s", "split": "D"}

PAGE_COLUMN_WIDTH = 3.50
K_BALANCED = (1, 2, 4, 5, 10, 20)
TIME_STEPS = (250, 500, 1000)
TIME_TO_SETTING = {250: "shot_rich", 500: "balanced", 1000: "data_rich"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 5.7,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    }
)


def _open_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(width=0.7, length=2.5, pad=1.8)
    ax.grid(False)


def _panel_letter(ax: plt.Axes, letter: str, *, x: float = -0.13) -> None:
    """Place a bare panel identifier outside the upper-left plot corner."""

    ax.text(
        x,
        1.055,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        clip_on=False,
    )


def _t_interval(values: Iterable[float]) -> tuple[float, float, float]:
    data = np.asarray(tuple(values), dtype=float)
    if data.size == 0:
        raise ValueError("cannot summarize an empty cluster sample")
    mean = float(np.mean(data))
    if data.size == 1:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, data.size - 1) * np.std(data, ddof=1) / np.sqrt(data.size))
    return mean, mean - half, mean + half


def _validate_cluster_count(frame: pd.DataFrame, grouping: list[str]) -> None:
    counts = frame.groupby(grouping, dropna=False)["cluster_id"].nunique()
    if counts.empty or not counts.eq(5).all():
        raise ValueError(
            "registered figures require exactly five independent trajectory/window clusters "
            f"per comparison; observed {counts.to_dict()}"
        )


def _load_registered_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    replicate_path = RESULTS_DIR / "replicate_results.csv"
    spectrum_path = RESULTS_DIR / "split_validation_spectrum.csv"
    if not replicate_path.exists() or not spectrum_path.exists():
        raise FileNotFoundError(
            "run the registered robust protocol before plotting: "
            f"missing {replicate_path if not replicate_path.exists() else spectrum_path}"
        )
    rows = pd.read_csv(replicate_path)
    spectrum = pd.read_csv(spectrum_path)

    headline = rows.loc[
        rows["reservoir_variant"].eq("expressive_reuploading")
        & rows["feature_set"].eq("z_plus_zz")
        & rows["scope_role"].eq("headline")
    ].copy()
    validation = spectrum.loc[
        spectrum["reservoir_variant"].eq("expressive_reuploading")
        & spectrum["feature_set"].eq("z_plus_zz")
        & spectrum["scope_role"].eq("headline")
        & spectrum["metric_role"].eq("validation_only_no_test_access")
    ].copy()

    if set(headline["dataset"].unique()) != set(TASKS):
        raise ValueError("headline results do not contain the four manuscript tasks")
    if set(validation["dataset"].unique()) != set(TASKS):
        raise ValueError("validation spectrum does not contain the four manuscript tasks")
    if set(headline["budget_setting"].unique()) != set(TIME_TO_SETTING.values()):
        raise ValueError("headline results do not contain the three fixed-budget settings")

    _validate_cluster_count(headline, ["dataset", "budget_setting", "method"])
    _validate_cluster_count(validation, ["dataset", "budget_setting", "k"])
    return headline, validation


def _cluster_method_means(rows: pd.DataFrame, *, budget: str) -> pd.DataFrame:
    methods = ("raw", "ev", "covariance_ev", "split")
    selected = rows.loc[
        rows["budget_setting"].eq(budget) & rows["method"].isin(methods)
    ]
    means = (
        selected.groupby(["dataset", "cluster_id", "method"], as_index=False)["test_nrmse"]
        .mean()
        .sort_values(["dataset", "cluster_id", "method"])
    )
    _validate_cluster_count(means, ["dataset", "method"])
    return means


def _cluster_validation_means(spectrum: pd.DataFrame, *, budget: str) -> pd.DataFrame:
    selected = spectrum.loc[spectrum["budget_setting"].eq(budget)]
    means = (
        selected.groupby(["dataset", "cluster_id", "k"], as_index=False)["mean_val_nrmse"]
        .mean()
        .sort_values(["dataset", "cluster_id", "k"])
    )
    _validate_cluster_count(means, ["dataset", "k"])
    return means


def _paired_cluster_gains(rows: pd.DataFrame) -> pd.DataFrame:
    """Average paired EV-minus-Split differences within each cluster."""

    selected = rows.loc[rows["method"].isin(("ev", "split"))].copy()
    pairing = [
        "dataset",
        "budget_setting",
        "cluster_id",
        "reservoir_seed",
        "shot_seed",
        "partition_seed",
    ]
    paired = selected.pivot_table(
        index=pairing,
        columns="method",
        values="test_nrmse",
        aggfunc="mean",
    ).dropna(subset=["ev", "split"])
    paired["gain"] = paired["ev"] - paired["split"]
    cluster = paired["gain"].groupby(level=[0, 1, 2]).mean().reset_index()
    _validate_cluster_count(cluster, ["dataset", "budget_setting"])
    return cluster


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_DIR / f"{stem}.pdf"
    png = FIGURE_DIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


def _fig2_panel_a(ax: plt.Axes, balanced: pd.DataFrame) -> None:
    """Matched outcomes across the five independent Mackey--Glass series."""

    sub = balanced.loc[balanced["dataset"].eq("mackey_glass")]
    wide = sub.pivot(index="cluster_id", columns="method", values="test_nrmse").sort_index()
    if not {"raw", "ev", "split"}.issubset(wide.columns):
        raise ValueError("Mackey--Glass paired-outcome panel is missing a headline method")
    y = np.arange(len(wide))[::-1]
    for yi, (_, row) in zip(y, wide.iterrows()):
        ax.plot(
            [row["ev"], row["split"]],
            [yi, yi],
            color=SPLIT,
            lw=1.25,
            alpha=0.82,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(row["raw"], yi, "o", color=RAW, ms=3.7, mec="white", mew=0.5, zorder=3)
        ax.plot(row["ev"], yi, "o", mfc="white", mec=FULL_AVERAGE, mew=0.9, ms=4.1, zorder=4)
        ax.plot(row["split"], yi, "D", color=SPLIT, mec="white", mew=0.55, ms=4.0, zorder=5)
    ax.set_yticks(y, [str(i + 1) for i in range(len(y))])
    ax.set_ylim(-0.55, len(y) - 0.45)
    ax.set_xlabel("Test error (NRMSE)", labelpad=2)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(axis="y", length=0)
    _panel_letter(ax, "a", x=-0.16)
    _open_axes(ax)


def _fig2_panel_b(ax: plt.Axes, validation: pd.DataFrame) -> None:
    sub = validation.loc[validation["dataset"].eq("mackey_glass")]
    x = np.arange(len(K_BALANCED), dtype=float)
    means, lows, highs = [], [], []
    for k in K_BALANCED:
        values = sub.loc[sub["k"].eq(k), "mean_val_nrmse"].to_numpy(dtype=float)
        mean, low, high = _t_interval(values)
        means.append(mean)
        lows.append(low)
        highs.append(high)
        jitter = np.linspace(-0.075, 0.075, len(values))
        ax.scatter(
            x[K_BALANCED.index(k)] + jitter,
            values,
            s=8,
            color=SPLIT,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
    mean_arr = np.asarray(means)
    low_arr = np.asarray(lows)
    high_arr = np.asarray(highs)
    ax.fill_between(x, low_arr, high_arr, color=SPLIT, alpha=0.12, linewidth=0, zorder=2)
    ax.plot(x, mean_arr, color=SPLIT, lw=1.35, marker="o", ms=3.2, mec="white", mew=0.55, zorder=3)
    ax.plot(x[0], mean_arr[0], "s", color=RAW, ms=4.3, mec="white", mew=0.55, zorder=4)
    ax.plot(x[-1], mean_arr[-1], "s", color=FULL_AVERAGE, ms=4.3, mec="white", mew=0.55, zorder=4)
    best = int(np.argmin(mean_arr))
    ax.plot(x[best], mean_arr[best], "*", color="#14705f", ms=8.5, mec="white", mew=0.5, zorder=5)
    ax.set_xticks(x, [str(k) for k in K_BALANCED])
    ax.set_xlabel("Shots per group $k$", labelpad=2)
    ax.set_ylabel("Validation error (NRMSE)", labelpad=2)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    _panel_letter(ax, "b", x=-0.16)
    _open_axes(ax)


def _fig2_panel_c(ax: plt.Axes, validation: pd.DataFrame) -> None:
    x = np.arange(len(K_BALANCED), dtype=float)
    for task in TASKS:
        task_rows = validation.loc[validation["dataset"].eq(task)]
        wide = task_rows.pivot(index="cluster_id", columns="k", values="mean_val_nrmse").reindex(
            columns=K_BALANCED
        )
        relative = wide.divide(wide[20], axis=0) * 100.0
        means, lows, highs = [], [], []
        for k in K_BALANCED:
            mean, low, high = _t_interval(relative[k].to_numpy(dtype=float))
            means.append(mean)
            lows.append(low)
            highs.append(high)
        color = TASK_COLORS[task]
        ax.fill_between(x, lows, highs, color=color, alpha=0.075, linewidth=0)
        ax.plot(
            x,
            means,
            color=color,
            lw=1.12,
            marker=TASK_MARKERS[task],
            ms=2.8,
            mec="white",
            mew=0.45,
            label=TASK_SHORT[task],
        )
    ax.axhline(100.0, color=NEUTRAL, lw=0.65, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(x, [str(k) for k in K_BALANCED])
    ax.set_xlabel("Shots per group $k$", labelpad=2)
    ax.set_ylabel("Error relative to EV (%)", labelpad=2)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.legend(
        loc="upper center",
        ncol=2,
        fontsize=5.3,
        handlelength=1.1,
        handletextpad=0.3,
        columnspacing=0.65,
        labelspacing=0.25,
        borderaxespad=0.15,
    )
    _panel_letter(ax, "c", x=-0.23)
    _open_axes(ax)


def _fig2_panel_d(ax: plt.Axes, balanced: pd.DataFrame) -> None:
    summary = balanced.groupby(["dataset", "method"], as_index=False)["test_nrmse"].mean()
    wide = summary.pivot(index="dataset", columns="method", values="test_nrmse").reindex(TASKS)
    y = np.arange(len(TASKS))[::-1]
    all_values: list[float] = []
    for yi, task in zip(y, TASKS):
        row = wide.loc[task]
        all_values.extend((float(row["raw"]), float(row["ev"]), float(row["split"])))
        ax.plot(
            [row["ev"], row["split"]],
            [yi, yi],
            color=SPLIT,
            lw=1.5,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(row["raw"], yi, "o", color=RAW, ms=3.8, mec="white", mew=0.5, zorder=3)
        ax.plot(row["ev"], yi, "o", mfc="white", mec=FULL_AVERAGE, mew=0.95, ms=4.4, zorder=4)
        ax.plot(row["split"], yi, "D", color=SPLIT, mec="white", mew=0.55, ms=4.2, zorder=5)

    xmin, xmax = min(all_values), max(all_values)
    span = xmax - xmin
    ax.set_xlim(xmin - 0.10 * span, xmax + 0.55 * span)
    label_x = xmax + 0.12 * span
    for yi, task in zip(y, TASKS):
        row = wide.loc[task]
        delta = float(row["ev"] - row["split"])
        ax.text(label_x, yi, f"{delta:.3f} lower", ha="left", va="center", fontsize=5.2, color="#14705f")
    ax.set_yticks(y, [TASK_SHORT[task] for task in TASKS])
    ax.set_ylim(-0.55, len(TASKS) - 0.45)
    ax.set_xlabel("Test error (NRMSE)", labelpad=2)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(axis="y", length=0, pad=1.0)
    _panel_letter(ax, "d", x=-0.22)
    _open_axes(ax)


def figure_2(rows: pd.DataFrame, spectrum: pd.DataFrame) -> None:
    balanced = _cluster_method_means(rows, budget="balanced")
    validation = _cluster_validation_means(spectrum, budget="balanced")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(PAGE_COLUMN_WIDTH, 3.02),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (0.92, 1.08)},
    )
    fig.get_layout_engine().set(h_pad=0.025, w_pad=0.025, hspace=0.015, wspace=0.015)
    _fig2_panel_a(axes[0, 0], balanced)
    _fig2_panel_b(axes[0, 1], validation)
    _fig2_panel_c(axes[1, 0], validation)
    _fig2_panel_d(axes[1, 1], balanced)
    _save(fig, "fig2_composite")


def figure_3(rows: pd.DataFrame) -> None:
    gains = _paired_cluster_gains(rows)
    rng = np.random.default_rng(240719)
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_COLUMN_WIDTH, 2.72), constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.035, w_pad=0.025, hspace=0.02, wspace=0.02)
    letters = "abcd"
    x = np.arange(len(TIME_STEPS), dtype=float)
    for index, (ax, task) in enumerate(zip(axes.ravel(), TASKS)):
        sub = gains.loc[gains["dataset"].eq(task)]
        means, lows, highs = [], [], []
        for xpos, time_steps in zip(x, TIME_STEPS):
            setting = TIME_TO_SETTING[time_steps]
            values = sub.loc[sub["budget_setting"].eq(setting), "gain"].to_numpy(dtype=float)
            mean, low, high = _t_interval(values)
            means.append(mean)
            lows.append(low)
            highs.append(high)
            jitter = rng.uniform(-0.065, 0.065, size=len(values))
            ax.scatter(
                np.full(len(values), xpos) + jitter,
                values,
                s=10,
                color=TASK_COLORS[task],
                alpha=0.22,
                linewidth=0,
                zorder=1,
            )
        color = TASK_COLORS[task]
        ax.axhline(0.0, color=NEUTRAL, lw=0.65, ls=(0, (3, 2)), zorder=0)
        ax.fill_between(x, lows, highs, color=color, alpha=0.10, linewidth=0, zorder=2)
        ax.plot(x, means, color=color, lw=1.25, solid_capstyle="round", zorder=3)
        ax.errorbar(
            x,
            means,
            yerr=[np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)],
            fmt=TASK_MARKERS[task],
            ms=4.0,
            mfc=color,
            mec="white",
            mew=0.55,
            ecolor=color,
            elinewidth=0.9,
            capsize=2.0,
            capthick=0.9,
            zorder=4,
        )
        low_bound = min(0.0, min(lows), float(sub["gain"].min()))
        high_bound = max(0.0, max(highs), float(sub["gain"].max()))
        span = max(high_bound - low_bound, 0.025)
        ax.set_ylim(low_bound - 0.10 * span, high_bound + 0.13 * span)
        ax.set_xticks(x, [str(value) for value in TIME_STEPS])
        ax.yaxis.set_major_locator(MaxNLocator(4))
        if index >= 2:
            ax.set_xlabel("Observed time steps $T$", labelpad=2)
        if index % 2 == 0:
            ax.set_ylabel("EV minus Grouped", labelpad=2)
        _panel_letter(ax, letters[index], x=-0.17 if index % 2 == 0 else -0.13)
        _open_axes(ax)
    _save(fig, "fig3_regime_map")


def figure_4(rows: pd.DataFrame) -> None:
    balanced = _cluster_method_means(rows, budget="balanced")
    rng = np.random.default_rng(481516)
    # Top-to-bottom comparison order: EV, Grouped, Raw, Cov-EV.
    methods = ("ev", "split", "raw", "covariance_ev")
    y = np.arange(len(methods))[::-1]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(PAGE_COLUMN_WIDTH, 2.72),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.025, hspace=0.02, wspace=0.015)
    letters = "abcd"
    for index, (ax, task) in enumerate(zip(axes.ravel(), TASKS)):
        sub = balanced.loc[balanced["dataset"].eq(task)]
        for yi, method in zip(y, methods):
            values = sub.loc[sub["method"].eq(method), "test_nrmse"].to_numpy(dtype=float)
            mean, low, high = _t_interval(values)
            color = METHOD_COLORS[method]
            jitter = rng.uniform(-0.095, 0.095, size=len(values))
            ax.scatter(
                values,
                np.full(len(values), yi) + jitter,
                s=9,
                color=color,
                alpha=0.22,
                linewidth=0,
                zorder=1,
            )
            marker = METHOD_MARKERS[method]
            marker_face = "white" if method == "ev" else color
            ax.errorbar(
                mean,
                yi,
                xerr=[[mean - low], [high - mean]],
                fmt=marker,
                ms=4.3 if method != "split" else 4.7,
                mfc=marker_face,
                mec=color if method == "ev" else "white",
                mew=0.85 if method == "ev" else 0.55,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.2,
                capthick=1.0,
                zorder=4,
            )
        ax.set_yticks(y, [METHOD_LABELS[m] for m in methods])
        ax.set_ylim(-0.55, len(methods) - 0.45)
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        if index >= 2:
            ax.set_xlabel("Test NRMSE", labelpad=2)
        _panel_letter(ax, letters[index], x=-0.38 if index % 2 == 0 else -0.13)
        _open_axes(ax)
    all_values = balanced["test_nrmse"].to_numpy(dtype=float)
    xmin, xmax = float(all_values.min()), float(all_values.max())
    padding = 0.055 * (xmax - xmin)
    axes[0, 0].set_xlim(xmin - padding, xmax + padding)
    _save(fig, "fig4_controls_compact")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig", choices=("2", "3", "4", "all"), default="all")
    args = parser.parse_args()
    rows, spectrum = _load_registered_results()
    if args.fig in ("2", "all"):
        figure_2(rows, spectrum)
    if args.fig in ("3", "all"):
        figure_3(rows)
    if args.fig in ("4", "all"):
        figure_4(rows)


if __name__ == "__main__":
    main()
