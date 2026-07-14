#!/usr/bin/env python3
"""Build Fig. 5 from the checked Fez and Marrakesh hardware tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
HARDWARE = ROOT / "results" / "qai_hardware"
FIGURES = ROOT / "paper" / "figures"
PDF_METADATA = {
    "Creator": "Adaptive shot aggregation publication pipeline",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}
TASKS = ("mackey_glass", "santa_fe_laser", "rossler", "sunspots")
COLORS = {
    "mackey_glass": "#4a6fa5",
    "santa_fe_laser": "#2a9d8f",
    "rossler": "#c8793a",
    "sunspots": "#9c6b9e",
}
MARKERS = {
    "mackey_glass": "o",
    "santa_fe_laser": "D",
    "rossler": "s",
    "sunspots": "^",
}
LABELS = {
    "mackey_glass": "Mackey–Glass",
    "santa_fe_laser": "Santa Fe",
    "rossler": "Rössler",
    "sunspots": "Sunspots",
}
LEGEND_LABELS = {**LABELS, "mackey_glass": "M-Glass"}

# Preserve the pre-enlargement standalone PDF frame exactly.
HARDWARE_FIGURE_SIZE = (246.815 / 72.0, 244.168 / 72.0)

# Level 2 publication enlargement.  Keep the canvas and data geometry fixed.
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
    return value * MARKER_SCALE**2


def _fs(value: float) -> float:
    return value * TEXT_SCALE


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": _fs(7.5),
        "axes.titlesize": _fs(8.0),
        "axes.labelsize": _fs(7.5),
        "xtick.labelsize": _fs(7.0),
        "ytick.labelsize": _fs(7.0),
        "axes.linewidth": _lw(0.8),
        "xtick.major.width": _lw(0.8),
        "ytick.major.width": _lw(0.8),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def ci95(values: np.ndarray) -> float:
    """Return the two sided 95 percent t interval half width."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def load_tables(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    spectrum = pd.read_csv(directory / "hardware_spectrum.csv")
    spectrum = spectrum[spectrum["grouping"].eq("contiguous")]
    records = pd.read_csv(directory / "hardware_seed_level.csv")
    for name, table in (("spectrum", spectrum), ("record", records)):
        missing = set(TASKS) - set(table["dataset"].unique())
        if missing:
            raise ValueError(f"{directory.name} {name} table is missing {sorted(missing)}")
    return spectrum, records


def main() -> None:
    devices = (("ibm_fez", HARDWARE), ("ibm_marrakesh", HARDWARE / "marrakesh"))
    tables = {name: load_tables(directory) for name, directory in devices}
    fig, axes = plt.subplots(2, 2, figsize=HARDWARE_FIGURE_SIZE, sharey="row")
    fig.subplots_adjust(left=0.245, right=0.985, bottom=0.19, top=0.925, hspace=0.48, wspace=0.11)
    rng = np.random.default_rng(23)

    k_values = sorted(tables["ibm_fez"][0]["k"].unique())
    x = np.arange(len(k_values), dtype=float)
    offsets = dict(zip(TASKS, (-0.15, -0.05, 0.05, 0.15)))
    handles = []
    for column, (name, _) in enumerate(devices):
        axis = axes[0, column]
        spectrum = tables[name][0]
        for task in TASKS:
            rows = spectrum[spectrum["dataset"].eq(task)]
            means = np.array([rows[rows["k"].eq(k)]["val_nrmse"].mean() for k in k_values])
            intervals = np.array([ci95(rows[rows["k"].eq(k)]["val_nrmse"].to_numpy()) for k in k_values])
            xx = x + offsets[task]
            for index, k in enumerate(k_values):
                values = rows[rows["k"].eq(k)]["val_nrmse"].to_numpy()
                axis.scatter(
                    np.full(len(values), xx[index]) + rng.uniform(-0.028, 0.028, len(values)),
                    values,
                    s=_ma(5.5),
                    color=COLORS[task],
                    alpha=0.16,
                    linewidth=0,
                )
            axis.fill_between(xx, means - intervals, means + intervals, color=COLORS[task], alpha=0.08, linewidth=0)
            line, = axis.plot(
                xx,
                means,
                color=COLORS[task],
                linewidth=_lw(1.05),
                marker=MARKERS[task],
                markersize=_ms(2.9),
                markeredgecolor="white",
                markeredgewidth=_mew(0.5),
                label=LEGEND_LABELS[task],
            )
            if column == 0:
                handles.append(line)
        axis.set_xticks(x, [str(k) for k in k_values])
        axis.set_xlabel("Aggregation scale $k$")
        axis.text(-0.30 if column == 0 else -0.10, 1.05, "ab"[column], transform=axis.transAxes, fontsize=_fs(9), fontweight="bold")
    axes[0, 0].set_ylabel("NRMSE")
    legend = axes[0, 0].legend(
        handles=handles,
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
    # Reserve the upper part of both top panels for the enlarged legend.
    axes[0, 0].set_ylim(0.25, 1.30)

    y_positions = np.arange(len(TASKS))[::-1]
    maximum = 0.0
    for column, (name, _) in enumerate(devices):
        axis = axes[1, column]
        records = tables[name][1]
        ev = records[records["protocol"].eq("ev")].set_index(["dataset", "record"])["test_nrmse"]
        grouped = records[records["protocol"].eq("split")].set_index(["dataset", "record"])["test_nrmse"]
        gains = (ev - grouped).rename("gain").reset_index()
        maximum = max(maximum, float(gains["gain"].max()))
        for y, task in zip(y_positions, TASKS):
            values = gains[gains["dataset"].eq(task)]["gain"].to_numpy()
            axis.scatter(values, np.full(len(values), y) + rng.uniform(-0.17, 0.17, len(values)), s=_ma(9), color=COLORS[task], alpha=0.30, linewidth=0)
            axis.errorbar(
                values.mean(),
                y,
                xerr=ci95(values),
                fmt="D",
                markersize=_ms(4.2),
                color=COLORS[task],
                capsize=2.3,
                elinewidth=_lw(1.5),
                capthick=_mew(0.6),
                markeredgecolor="white",
                markeredgewidth=_mew(0.6),
            )
        axis.axvline(0, color="#8a8a8a", linewidth=_lw(0.7), linestyle=(0, (2, 2)))
        axis.set_xlabel("EV minus Grouped\n(NRMSE)", linespacing=0.95)
        axis.text(-0.30 if column == 0 else -0.10, 1.05, "cd"[column], transform=axis.transAxes, fontsize=_fs(9), fontweight="bold")
    for axis in axes[1]:
        axis.set_xlim(-0.045, maximum + 0.035)
        axis.set_xticks([0.0, 0.1, 0.2])
    axes[1, 0].set_yticks(y_positions, [LABELS[task] for task in TASKS], fontsize=_fs(6.2))

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURES / "fig5_hardware.pdf"
    fig.savefig(figure_path, metadata=PDF_METADATA)
    plt.close(fig)
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
