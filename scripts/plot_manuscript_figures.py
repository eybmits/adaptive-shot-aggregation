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
PDF_METADATA = {
    "Creator": "Adaptive shot aggregation publication pipeline",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}

# Original manuscript palette.
FULL_AVERAGE = "#4a6fa5"
SPLIT = "#2a9d8f"
RAW = "#e3b341"  # deliberately yellow: individual shots, not a gray reference
COVARIANCE_EV = "#9c6b9e"
NEUTRAL = "#777777"
BLACK = "#171717"

TASKS = ("mackey_glass", "santa_fe_laser", "rossler", "sunspots")
EXPECTED_CLUSTERS = {
    "mackey_glass": 20,
    "santa_fe_laser": 5,
    "rossler": 5,
    "sunspots": 5,
}
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
TASK_LEGEND = {
    "mackey_glass": "Mackey–Glass",
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
    "covariance_ev": "Regularized EV",
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

# Level 2 publication enlargement.  The physical figure dimensions and data
# geometry stay fixed; only strokes, markers, and typography are enlarged.
LINE_SCALE = 1.70
MARKER_SCALE = 1.50
TEXT_SCALE = 1.15


def _lw(value: float) -> float:
    return value * LINE_SCALE


def _ms(value: float) -> float:
    return value * MARKER_SCALE


def _mew(value: float) -> float:
    return value * MARKER_SCALE


def _ma(value: float) -> float:
    """Scale scatter-marker area while preserving marker shape."""

    return value * MARKER_SCALE**2


def _fs(value: float) -> float:
    return value * TEXT_SCALE

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": _fs(7.0),
        "axes.titlesize": _fs(7.5),
        "axes.labelsize": _fs(7.0),
        "xtick.labelsize": _fs(6.3),
        "ytick.labelsize": _fs(6.3),
        "legend.fontsize": _fs(5.7),
        "axes.linewidth": _lw(0.7),
        "xtick.major.width": _lw(0.7),
        "ytick.major.width": _lw(0.7),
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
    ax.spines["left"].set_linewidth(_lw(0.7))
    ax.spines["bottom"].set_linewidth(_lw(0.7))
    ax.tick_params(width=_lw(0.7), length=2.5, pad=1.8)
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
        fontsize=_fs(9.0),
        fontweight="bold",
        clip_on=False,
    )


def _task_label(ax: plt.Axes, task: str) -> None:
    """Identify the task directly inside an individual panel."""

    ax.text(
        0.97,
        0.96,
        TASK_LEGEND[task],
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=_fs(6.2),
        fontweight="bold",
        color=BLACK,
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.90},
        zorder=8,
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
    if counts.empty or "dataset" not in grouping:
        raise ValueError("cluster-count validation requires dataset in the grouping")
    observed = counts.reset_index(name="clusters")
    observed["expected"] = observed["dataset"].map(EXPECTED_CLUSTERS)
    if observed["expected"].isna().any() or not observed["clusters"].eq(observed["expected"]).all():
        raise ValueError(
            "manuscript figures require the registered independent trajectory/window counts; "
            f"observed {observed.to_dict(orient='records')}"
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
    fig.savefig(pdf, metadata=PDF_METADATA)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


def _fig2_panel_a(ax: plt.Axes, balanced: pd.DataFrame) -> None:
    """Distributions across the independent Mackey--Glass trajectories."""

    sub = balanced.loc[balanced["dataset"].eq("mackey_glass")]
    wide = sub.pivot(index="cluster_id", columns="method", values="test_nrmse").sort_index()
    if not {"raw", "ev", "split"}.issubset(wide.columns):
        raise ValueError("Mackey--Glass paired-outcome panel is missing a headline method")
    methods = ("ev", "split", "raw")
    positions = np.arange(1, len(methods) + 1, dtype=float)
    colors = (FULL_AVERAGE, SPLIT, RAW)
    markers = ("o", "D", "o")
    values = [wide[method].to_numpy(dtype=float) for method in methods]

    violins = ax.violinplot(
        values,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.14)
        body.set_linewidth(_lw(0.55))

    rng = np.random.default_rng(26)
    for position, method_values, color, marker in zip(positions, values, colors, markers):
        jitter = rng.uniform(-0.13, 0.13, len(method_values))
        facecolor = "white" if color == FULL_AVERAGE else color
        ax.scatter(
            np.full(len(method_values), position) + jitter,
            method_values,
            s=_ma(8.0),
            marker=marker,
            facecolor=facecolor,
            edgecolor=color,
            linewidth=_mew(0.42),
            alpha=0.72,
            zorder=3,
        )
        mean, low, high = _t_interval(method_values)
        ax.errorbar(
            position,
            mean,
            yerr=[[mean - low], [high - mean]],
            color=color,
            lw=_lw(0.85),
            capsize=2.7,
            capthick=_lw(0.75),
            zorder=4,
        )
        ax.scatter(
            position,
            mean,
            s=_ma(24.0),
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=_mew(0.5),
            zorder=5,
        )

    ax.set_xticks(positions, ("EV", "Grouped", "Raw"))
    ax.set_ylabel("Test NRMSE", labelpad=2)
    ax.set_ylim(0.28, 0.78)
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
            s=_ma(8),
            color=SPLIT,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
    mean_arr = np.asarray(means)
    low_arr = np.asarray(lows)
    high_arr = np.asarray(highs)
    ax.fill_between(x, low_arr, high_arr, color=SPLIT, alpha=0.12, linewidth=0, zorder=2)
    ax.plot(x, mean_arr, color=SPLIT, lw=_lw(1.35), marker="o", ms=_ms(3.2), mec="white", mew=_mew(0.55), zorder=3)
    ax.plot(x[0], mean_arr[0], "s", color=RAW, ms=_ms(4.3), mec="white", mew=_mew(0.55), zorder=4)
    ax.plot(x[-1], mean_arr[-1], "s", color=FULL_AVERAGE, ms=_ms(4.3), mec="white", mew=_mew(0.55), zorder=4)
    best = int(np.argmin(mean_arr))
    ax.plot(x[best], mean_arr[best], "*", color="#14705f", ms=_ms(8.5), mec="white", mew=_mew(0.5), zorder=5)
    ax.set_xticks(x, [str(k) for k in K_BALANCED])
    ax.set_xlabel("Shots per group $k$", labelpad=2)
    ax.set_ylabel("NRMSE", labelpad=2)
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
            lw=_lw(1.12),
            marker=TASK_MARKERS[task],
            ms=_ms(2.8),
            mec="white",
            mew=_mew(0.45),
            label="M-Glass" if task == "mackey_glass" else TASK_SHORT[task],
        )
    ax.axhline(100.0, color=NEUTRAL, lw=_lw(0.65), ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(x, [str(k) for k in K_BALANCED])
    ax.set_xlabel("Shots per group $k$", labelpad=2)
    ax.set_ylabel("Error relative to EV (%)", labelpad=2)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    # Keep the enlarged two-row legend in a clear header above every band.
    ax.set_ylim(80.0, 152.0)
    legend = ax.legend(
        loc="upper center",
        ncol=2,
        fontsize=_fs(4.9),
        frameon=True,
        facecolor="white",
        edgecolor="#d6dbe3",
        framealpha=1.0,
        fancybox=True,
        borderpad=0.30,
        handlelength=1.1,
        handletextpad=0.3,
        columnspacing=0.65,
        labelspacing=0.25,
        borderaxespad=0.15,
    )
    legend.get_frame().set_linewidth(_lw(0.8))
    _panel_letter(ax, "c", x=-0.23)
    _open_axes(ax)


def _fig2_panel_d(ax: plt.Axes, balanced: pd.DataFrame) -> None:
    paired = balanced.loc[balanced["method"].isin(("ev", "split"))].pivot(
        index=["dataset", "cluster_id"],
        columns="method",
        values="test_nrmse",
    )
    if paired[["ev", "split"]].isna().any().any():
        raise ValueError("balanced outcome panel requires paired EV and Grouped unit means")
    y = np.arange(len(TASKS))[::-1]
    ax.axvline(0.0, color=NEUTRAL, lw=_lw(0.65), ls=(0, (3, 2)), zorder=0)
    for yi, task in zip(y, TASKS):
        task_pairs = paired.xs(task, level="dataset")
        gains = (task_pairs["ev"] - task_pairs["split"]).to_numpy(dtype=float)
        mean, low, high = _t_interval(gains)
        color = TASK_COLORS[task]
        ax.errorbar(
            mean,
            yi,
            xerr=[[mean - low], [high - mean]],
            color=color,
            lw=_lw(1.1),
            capsize=2.7,
            capthick=_lw(0.85),
            zorder=2,
        )
        ax.plot(
            mean,
            yi,
            marker=TASK_MARKERS[task],
            color=color,
            ms=_ms(4.0),
            mec="white",
            mew=_mew(0.5),
            zorder=3,
        )
        ax.text(
            0.98,
            yi,
            f"{mean:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=_fs(5.2),
            color=BLACK,
        )
    ax.set_yticks(y, [TASK_SHORT[task] for task in TASKS])
    ax.set_ylim(-0.55, len(TASKS) - 0.45)
    ax.set_xlim(-0.015, 0.21)
    ax.set_xlabel("EV minus Grouped\n(NRMSE)", labelpad=2)
    ax.xaxis.set_major_locator(MaxNLocator(4))
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
                s=_ma(10),
                color=TASK_COLORS[task],
                alpha=0.22,
                linewidth=0,
                zorder=1,
            )
        color = TASK_COLORS[task]
        ax.axhline(0.0, color=NEUTRAL, lw=_lw(0.65), ls=(0, (3, 2)), zorder=0)
        ax.fill_between(x, lows, highs, color=color, alpha=0.10, linewidth=0, zorder=2)
        ax.plot(x, means, color=color, lw=_lw(1.25), solid_capstyle="round", zorder=3)
        ax.errorbar(
            x,
            means,
            yerr=[np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)],
            fmt=TASK_MARKERS[task],
            ms=_ms(4.0),
            mfc=color,
            mec="white",
            mew=_mew(0.55),
            ecolor=color,
            elinewidth=_lw(0.9),
            capsize=2.0,
            capthick=_lw(0.9),
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
        _task_label(ax, task)
        _panel_letter(ax, letters[index], x=-0.17 if index % 2 == 0 else -0.13)
        _open_axes(ax)
    _save(fig, "fig3_regime_map")


def figure_4(rows: pd.DataFrame) -> None:
    balanced = _cluster_method_means(rows, budget="balanced")
    rng = np.random.default_rng(481516)
    # Top-to-bottom comparison order: EV, Grouped, Raw, Regularized EV.
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
                s=_ma(9),
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
                ms=_ms(4.3 if method != "split" else 4.7),
                mfc=marker_face,
                mec=color if method == "ev" else "white",
                mew=_mew(0.85 if method == "ev" else 0.55),
                ecolor=color,
                elinewidth=_lw(1.0),
                capsize=2.2,
                capthick=_lw(1.0),
                zorder=4,
            )
        ax.set_yticks(y, [METHOD_LABELS[m] for m in methods])
        # Keep a clear header strip for the enlarged in-panel task label.
        ax.set_ylim(-0.55, len(methods) - 0.15)
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        if index >= 2:
            ax.set_xlabel("Test NRMSE", labelpad=2)
        _task_label(ax, task)
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
