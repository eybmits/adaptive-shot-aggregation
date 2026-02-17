#!/usr/bin/env python3
"""Replot camera-ready hardware figures in IEEE style without rerunning hardware jobs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from ieee_plot_style import (
    BAR_EDGE,
    METHOD_COLORS,
    OURS_LABEL,
    TRAJ_LABEL,
    TRAJ_TICK_LABEL,
    apply_axis_chrome,
    setup_ieee_style,
)

BAR_HATCHES = ["///", "...", ""]


def parse_seed(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def load_table(table_csv: Path) -> List[Dict]:
    if not table_csv.exists():
        raise FileNotFoundError(f"Missing table: {table_csv}")
    rows = list(csv.DictReader(table_csv.open()))
    if not rows:
        raise ValueError(f"Empty table: {table_csv}")
    return rows


def select_best_seed(rows: List[Dict]) -> int:
    best = min(rows, key=lambda r: float(r["nrmse_gtraj"]))
    return int(best["seed"])


def load_seed_png(supplementary_dir: Path, seed: int) -> np.ndarray:
    p = supplementary_dir / f"fig_hardware_mg_seed_{seed:02d}.png"
    if not p.exists():
        raise FileNotFoundError(f"Missing seed panel: {p}")
    return mpimg.imread(p)


def _crop_trace_panel_main(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[0], img.shape[1]
    x0 = int(0.00 * w)
    # Keep only the trace panel and trim the original right-side legend area.
    x1 = int(0.695 * w)
    y0 = int(0.03 * h)
    y1 = int(0.97 * h)
    return img[y0:y1, x0:x1]


def _crop_trace_panel_collage(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[0], img.shape[1]
    x0 = int(0.00 * w)
    # Keep more right axis content and clean legend remnants in post.
    x1 = int(0.58 * w)
    y0 = int(0.03 * h)
    y1 = int(0.97 * h)
    crop = img[y0:y1, x0:x1].copy()
    # Remove thin right-edge artifacts introduced by source legend border.
    if crop.shape[1] >= 8:
        crop[:, -6:, ...] = crop[:, -7:-6, ...]
    return crop


def _method_stats(rows: List[Dict]) -> tuple[list[float], list[float]]:
    ev = np.array([float(r["nrmse_ev"]) for r in rows], dtype=float)
    traj = np.array([float(r["nrmse_traj"]) for r in rows], dtype=float)
    ours = np.array([float(r["nrmse_gtraj"]) for r in rows], dtype=float)
    means = [float(np.mean(ev)), float(np.mean(traj)), float(np.mean(ours))]
    stds = [float(np.std(ev)), float(np.std(traj)), float(np.std(ours))]
    return means, stds


def _render_boxplot_panel_image(rows: List[Dict], target_h: int, target_w: int) -> np.ndarray:
    dpi = 320
    fig = plt.figure(figsize=(target_w / dpi, target_h / dpi), dpi=dpi, constrained_layout=False)
    ax = fig.add_subplot(111)

    labels = ["EV", TRAJ_TICK_LABEL, OURS_LABEL]
    data = [
        [float(r["nrmse_ev"]) for r in rows],
        [float(r["nrmse_traj"]) for r in rows],
        [float(r["nrmse_gtraj"]) for r in rows],
    ]
    bplot = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        medianprops={"color": BAR_EDGE, "linewidth": 1.0},
        whiskerprops={"color": BAR_EDGE, "linewidth": 0.8},
        capprops={"color": BAR_EDGE, "linewidth": 0.8},
        boxprops={"edgecolor": BAR_EDGE, "linewidth": 0.9},
        flierprops={
            "marker": "o",
            "markersize": 3.0,
            "markerfacecolor": BAR_EDGE,
            "markeredgecolor": "none",
            "alpha": 0.55,
        },
    )
    for patch, c, h in zip(
        bplot["boxes"],
        [METHOD_COLORS["ev"], METHOD_COLORS["traj"], METHOD_COLORS["gtraj"]],
        BAR_HATCHES,
    ):
        patch.set_facecolor(c)
        patch.set_hatch(h)
        patch.set_alpha(0.95)

    ax.set_ylabel("NRMSE", fontsize=11.0)
    ax.tick_params(axis="x", labelsize=9.5, pad=1.5)
    ax.tick_params(axis="y", labelsize=9.5)
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(9.5)
    apply_axis_chrome(ax, grid=True)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.13, right=0.995, top=0.99, bottom=0.20)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def plot_main(
    *,
    rows: List[Dict],
    supplementary_dir: Path,
    out_pdf: Path,
    out_png: Path,
) -> None:
    setup_ieee_style()
    best_seed = select_best_seed(rows)
    trace_img = _crop_trace_panel_main(load_seed_png(supplementary_dir, best_seed))
    means, stds = _method_stats(rows)

    fig = plt.figure(figsize=(7.16, 2.25), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.8, 1.0], wspace=0.08)
    ax_trace = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    ax_trace.imshow(trace_img, aspect="auto")
    ax_trace.axis("off")
    bg = tuple(float(v) for v in trace_img[10, -10, :3])
    ax_trace.add_patch(
        Rectangle(
            (0.745, 0.675),
            0.255,
            0.325,
            transform=ax_trace.transAxes,
            facecolor=bg,
            edgecolor="none",
            zorder=6,
        )
    )
    legend_handles = [
        Line2D([0], [0], color=METHOD_COLORS["true"], lw=1.8, label="True"),
        Line2D([0], [0], color=METHOD_COLORS["traj"], lw=1.2, label=TRAJ_TICK_LABEL),
        Line2D([0], [0], color=METHOD_COLORS["ev"], lw=1.2, ls="--", label="EV"),
        Line2D([0], [0], color=METHOD_COLORS["gtraj"], lw=1.5, label=OURS_LABEL),
    ]
    legend = ax_trace.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fancybox=False,
        facecolor=bg,
        edgecolor=BAR_EDGE,
        framealpha=0.92,
        fontsize=6.8,
        borderpad=0.25,
        labelspacing=0.22,
        handlelength=1.8,
    )
    legend.set_zorder(8)

    x = np.arange(3)
    bar_colors = [METHOD_COLORS["ev"], METHOD_COLORS["traj"], METHOD_COLORS["gtraj"]]
    for xi, m, s, c, h in zip(x, means, stds, bar_colors, BAR_HATCHES):
        ax_bar.bar(
            xi,
            m,
            yerr=s,
            width=0.55,
            color=c,
            hatch=h,
            edgecolor=BAR_EDGE,
            linewidth=0.6,
            capsize=2.4,
            error_kw={"linewidth": 0.8, "capthick": 0.8},
            zorder=3,
        )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["EV", TRAJ_TICK_LABEL, OURS_LABEL])
    ax_bar.set_ylabel("NRMSE")
    apply_axis_chrome(ax_bar, grid=True)
    ax_bar.set_axisbelow(True)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_appendix(
    *,
    rows: List[Dict],
    supplementary_dir: Path,
    out_pdf: Path,
    out_png: Path,
) -> None:
    setup_ieee_style()

    # Keep a strict 3x3 grid with enough spacing so nothing overlaps.
    fig, axes = plt.subplots(3, 3, figsize=(11.4, 7.1), constrained_layout=False)
    fig.patch.set_facecolor("white")

    # Collect seed panels first so we can enforce exact same panel geometry.
    seed_pngs = sorted(supplementary_dir.glob("fig_hardware_mg_seed_*.png"), key=parse_seed)
    if len(seed_pngs) < 8:
        raise FileNotFoundError(
            f"Expected at least 8 seed PNGs in {supplementary_dir}, found {len(seed_pngs)}."
        )
    sample_crop = _crop_trace_panel_collage(mpimg.imread(seed_pngs[0]))
    target_h, target_w = sample_crop.shape[0], sample_crop.shape[1]

    # Panel 1: aggregated NRMSE boxplot rendered as an image with same pixel size.
    ax0 = axes.ravel()[0]
    ax0.imshow(_render_boxplot_panel_image(rows, target_h=target_h, target_w=target_w), aspect="auto")
    ax0.axis("off")

    # Panels 2-9: per-seed trace crops from existing camera-ready seed PNGs.

    for ax, path in zip(axes.ravel()[1:], seed_pngs[:8]):
        crop = _crop_trace_panel_collage(mpimg.imread(path))
        ax.imshow(crop, aspect="auto")
        bg = tuple(float(v) for v in crop[8, 8, :3])
        # Hide residual embedded legend snippet (top-right) without touching axis labels.
        ax.add_patch(
            Rectangle(
                (0.92, 0.68),
                0.08,
                0.32,
                transform=ax.transAxes,
                facecolor=bg,
                edgecolor="none",
                zorder=4,
            )
        )
        seed = parse_seed(path)
        ax.text(
            0.02,
            1.005,
            f"seed {seed}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.6,
            color="#2d3436",
            clip_on=False,
            zorder=6,
        )
        ax.axis("off")

    plt.subplots_adjust(
        left=0.045,
        right=0.995,
        top=0.982,
        bottom=0.045,
        wspace=0.12,
        hspace=0.20,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replot camera-ready hardware main + appendix figures in IEEE style without rerunning hardware."
    )
    parser.add_argument(
        "--table",
        type=str,
        default="results/camera_ready/table_hardware_mg_nrmse.csv",
    )
    parser.add_argument(
        "--supplementary",
        type=str,
        default="results/camera_ready/supplementary",
    )
    parser.add_argument(
        "--main-pdf",
        type=str,
        default="results/camera_ready/fig_hardware_mg_main.pdf",
    )
    parser.add_argument(
        "--main-png",
        type=str,
        default="results/camera_ready/fig_hardware_mg_main.png",
    )
    parser.add_argument(
        "--appendix-pdf",
        type=str,
        default="results/camera_ready/fig_hardware_mg_appendix_collage.pdf",
    )
    parser.add_argument(
        "--appendix-png",
        type=str,
        default="results/camera_ready/fig_hardware_mg_appendix_collage.png",
    )
    args = parser.parse_args()

    rows = load_table(Path(args.table))
    supp = Path(args.supplementary)

    plot_main(
        rows=rows,
        supplementary_dir=supp,
        out_pdf=Path(args.main_pdf),
        out_png=Path(args.main_png),
    )
    plot_appendix(
        rows=rows,
        supplementary_dir=supp,
        out_pdf=Path(args.appendix_pdf),
        out_png=Path(args.appendix_png),
    )

    print(f"Wrote: {args.main_pdf}")
    print(f"Wrote: {args.main_png}")
    print(f"Wrote: {args.appendix_pdf}")
    print(f"Wrote: {args.appendix_png}")


if __name__ == "__main__":
    main()
