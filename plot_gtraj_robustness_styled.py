#!/usr/bin/env python3
"""Render the Mackey-Glass robustness 3-panel figure with publication styling."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieee_plot_style import (
    ARCH_COLORS,
    METHOD_COLORS,
    NEUTRAL_ZERO,
    OURS_LABEL,
    apply_axis_chrome,
    open_marker_kwargs,
    remove_axis_title,
    setup_ieee_style,
)

COLORS = {
    "ev": METHOD_COLORS["ev"],
    "evdup": METHOD_COLORS["evdup"],
    "gtraj": METHOD_COLORS["gtraj"],
    "q6": ARCH_COLORS["q6"],
    "q8": ARCH_COLORS["q8"],
    "q10": ARCH_COLORS["q10"],
}

MARKERS = {
    "ev": "o",
    "evdup": "s",
    "gtraj": "D",
    "q6": "o",
    "q8": "s",
    "q10": "^",
}

LINESTYLES = {
    "ev": "-",
    "evdup": "--",
    "gtraj": "-",
    "q6": "-",
    "q8": "--",
    "q10": "-.",
}


def setup_style() -> None:
    setup_ieee_style()


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def _style_axis(ax: plt.Axes) -> None:
    apply_axis_chrome(ax, grid=True)


def _load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["q"] = int(r["q"])
        r["n_shots"] = int(r["n_shots"])
        r["lambda"] = float(r["lambda"])
        r["seed"] = int(r["seed"])
        r["nrmse"] = float(r["nrmse"])
        r["best_k"] = int(r["best_k"])
        r["dup_factor"] = int(r["dup_factor"])
        r["gap_ev_minus_gtraj"] = float(r["gap_ev_minus_gtraj"])
    return rows


def _aggregate_panel(
    rows: list[dict], *, panel: str, method: str | None, x_key: str, y_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if r["panel"] != panel:
            continue
        if method is not None and r["method"] != method:
            continue
        grouped[r[x_key]].append(r[y_key])
    xs = np.array(sorted(grouped.keys()), dtype=float)
    means = np.array([_mean_std(grouped[x])[0] for x in xs], dtype=float)
    stds = np.array([_mean_std(grouped[x])[1] for x in xs], dtype=float)
    return xs, means, stds


def _plot_series(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    color: str,
    marker: str,
    linestyle: str,
    label: str,
) -> None:
    ax.plot(
        x,
        mean,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=4,
        **open_marker_kwargs(color),
        linewidth=1.3,
        label=label,
    )
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)


def render(
    table_path: Path,
    out_pdf: Path,
    out_png: Path,
    fig_width: float = 7.16,
    fig_height: float = 2.4,
) -> None:
    setup_style()
    rows = _load_rows(table_path)

    fig, axes = plt.subplots(
        1, 3, figsize=(float(fig_width), float(fig_height)), constrained_layout=True
    )

    # Panel (A): lambda robustness
    ax = axes[0]
    x_l, ev_m, ev_s = _aggregate_panel(
        rows, panel="A_lambda", method="EV", x_key="lambda", y_key="nrmse"
    )
    _, gt_m, gt_s = _aggregate_panel(
        rows, panel="A_lambda", method="G-Traj", x_key="lambda", y_key="nrmse"
    )
    _plot_series(
        ax,
        x_l,
        ev_m,
        ev_s,
        color=COLORS["ev"],
        marker=MARKERS["ev"],
        linestyle=LINESTYLES["ev"],
        label="EV",
    )
    _plot_series(
        ax,
        x_l,
        gt_m,
        gt_s,
        color=COLORS["gtraj"],
        marker=MARKERS["gtraj"],
        linestyle=LINESTYLES["gtraj"],
        label=OURS_LABEL,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Ridge lambda")
    ax.set_ylabel("NRMSE")
    remove_axis_title(ax)
    _style_axis(ax)
    ax.legend(frameon=False, loc="upper right")

    # Panel (B): duplication control
    ax = axes[1]
    x_s, ev_m, ev_s = _aggregate_panel(
        rows, panel="B_dup", method="EV", x_key="n_shots", y_key="nrmse"
    )
    _, dup_m, dup_s = _aggregate_panel(
        rows, panel="B_dup", method="EV-dup", x_key="n_shots", y_key="nrmse"
    )
    _, gt_m, gt_s = _aggregate_panel(
        rows, panel="B_dup", method="G-Traj", x_key="n_shots", y_key="nrmse"
    )
    _plot_series(
        ax,
        x_s,
        ev_m,
        ev_s,
        color=COLORS["ev"],
        marker=MARKERS["ev"],
        linestyle=LINESTYLES["ev"],
        label="EV",
    )
    _plot_series(
        ax,
        x_s,
        dup_m,
        dup_s,
        color=COLORS["evdup"],
        marker=MARKERS["evdup"],
        linestyle=LINESTYLES["evdup"],
        label="EV-dup",
    )
    _plot_series(
        ax,
        x_s,
        gt_m,
        gt_s,
        color=COLORS["gtraj"],
        marker=MARKERS["gtraj"],
        linestyle=LINESTYLES["gtraj"],
        label=OURS_LABEL,
    )
    ax.set_xlabel(r"$N_{\mathrm{shots}}$")
    ax.set_ylabel("NRMSE")
    remove_axis_title(ax)
    _style_axis(ax)
    ax.legend(frameon=False, loc="upper right")

    # Panel (C): architecture check
    ax = axes[2]
    for q, color_key, label in [(6, "q6", "Q=6"), (8, "q8", "Q=8"), (10, "q10", "Q=10")]:
        sub = [r for r in rows if r["panel"] == "C_arch" and r["q"] == q and r["method"] == "EV"]
        grouped: dict[int, list[float]] = defaultdict(list)
        for r in sub:
            grouped[r["n_shots"]].append(r["gap_ev_minus_gtraj"])
        x = np.array(sorted(grouped.keys()), dtype=float)
        y = np.array([_mean_std(grouped[n])[0] for n in x], dtype=float)
        s = np.array([_mean_std(grouped[n])[1] for n in x], dtype=float)
        _plot_series(
            ax,
            x,
            y,
            s,
            color=COLORS[color_key],
            marker=MARKERS[color_key],
            linestyle=LINESTYLES[color_key],
            label=label,
        )
    ax.axhline(0.0, color=NEUTRAL_ZERO, linestyle=":", linewidth=0.5)
    ax.set_xlabel(r"$N_{\mathrm{shots}}$")
    ax.set_ylabel(f"Gap = EV - {OURS_LABEL}")
    remove_axis_title(ax)
    _style_axis(ax)
    ax.legend(frameon=False, loc="upper right")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render styled robustness figure from table_gtraj_robustness_oneplot.csv."
    )
    parser.add_argument(
        "--table",
        type=str,
        default="results/table_gtraj_robustness_oneplot.csv",
    )
    parser.add_argument(
        "--out-pdf",
        type=str,
        default="results/fig_gtraj_robustness_oneplot_notitle_ieee.pdf",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default="results/fig_gtraj_robustness_oneplot_notitle_ieee.png",
    )
    parser.add_argument("--fig-width", type=float, default=7.16)
    parser.add_argument("--fig-height", type=float, default=2.4)
    args = parser.parse_args()

    render(
        Path(args.table),
        Path(args.out_pdf),
        Path(args.out_png),
        fig_width=float(args.fig_width),
        fig_height=float(args.fig_height),
    )
    print(f"Saved: {args.out_pdf}")
    print(f"Saved: {args.out_png}")


if __name__ == "__main__":
    main()
