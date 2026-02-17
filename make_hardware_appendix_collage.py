#!/usr/bin/env python3
"""Create appendix collage: aggregated NRMSE boxplot + all seed traces."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from ieee_plot_style import OURS_LABEL, TRAJ_LABEL


def parse_seed(path: Path) -> int:
    # fig_hardware_mg_seed_07.png -> 7
    return int(path.stem.split("_")[-1])


def load_seed_panels(supplementary_dir: Path) -> List[Path]:
    if not supplementary_dir.exists():
        raise FileNotFoundError(f"Missing supplementary directory: {supplementary_dir}")

    seed_pngs = sorted(
        supplementary_dir.glob("fig_hardware_mg_seed_*.png"),
        key=parse_seed,
    )
    if len(seed_pngs) < 8:
        raise FileNotFoundError(
            f"Expected at least 8 seed PNGs in {supplementary_dir}, found {len(seed_pngs)}."
        )
    return seed_pngs[:8]


def load_nrmse_table(table_csv: Path) -> dict[str, list[float]]:
    if not table_csv.exists():
        raise FileNotFoundError(f"Missing table: {table_csv}")
    rows = list(csv.DictReader(table_csv.open()))
    if not rows:
        raise ValueError(f"Empty table: {table_csv}")
    return {
        "EV": [float(r["nrmse_ev"]) for r in rows],
        TRAJ_LABEL: [float(r["nrmse_traj"]) for r in rows],
        OURS_LABEL: [float(r["nrmse_gtraj"]) for r in rows],
    }


def make_collage(
    *,
    seed_panels: List[Path],
    nrmse_data: dict[str, list[float]],
    out_pdf: Path,
    out_png: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(3, 3, figsize=(11.4, 7.0))
    fig.patch.set_facecolor("white")

    # Panel 1: aggregated NRMSE boxplot over all seeds.
    ax0 = axes.ravel()[0]
    labels = ["EV", TRAJ_LABEL, OURS_LABEL]
    data = [nrmse_data[k] for k in labels]
    bplot = ax0.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        medianprops={"color": "#222222", "linewidth": 1.1},
        whiskerprops={"color": "#444444", "linewidth": 0.9},
        capprops={"color": "#444444", "linewidth": 0.9},
        boxprops={"edgecolor": "#333333", "linewidth": 0.9},
        flierprops={"marker": "o", "markersize": 3.0, "markerfacecolor": "#666666", "markeredgecolor": "none", "alpha": 0.7},
    )
    for patch, c, h in zip(
        bplot["boxes"],
        ["#555555", "#aaaaaa", "#c0392b"],
        ["///", "...", ""],
    ):
        patch.set_facecolor(c)
        patch.set_hatch(h)
        patch.set_alpha(0.95)
    ax0.set_ylabel("NRMSE")
    ax0.tick_params(axis="x", labelsize=8)
    ax0.tick_params(axis="y", labelsize=8)
    ax0.grid(axis="y", linestyle=":", alpha=0.25)

    # Panels 2-9: per-seed trace only (crop left time-series part).
    for ax, path in zip(axes.ravel()[1:], seed_panels):
        img = mpimg.imread(path)
        h, w = img.shape[0], img.shape[1]
        # Keep only left trace area, remove right NRMSE bar area.
        x0 = int(0.00 * w)
        x1 = int(0.72 * w)
        y0 = int(0.03 * h)
        y1 = int(0.97 * h)
        crop = img[y0:y1, x0:x1]
        ax.imshow(crop)
        seed = parse_seed(path)
        ax.text(
            0.02,
            1.005,
            f"seed {seed}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#333333",
            clip_on=False,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.2},
        )
        ax.axis("off")

    # Compact layout with enough left margin so panel-1 y-scale stays readable.
    plt.subplots_adjust(left=0.04, right=0.996, top=0.985, bottom=0.01, wspace=0.02, hspace=0.05)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create appendix collage for hardware results.")
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
        "--out-pdf",
        type=str,
        default="results/camera_ready/fig_hardware_mg_appendix_collage.pdf",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default="results/camera_ready/fig_hardware_mg_appendix_collage.png",
    )
    args = parser.parse_args()

    seed_panels = load_seed_panels(Path(args.supplementary))
    nrmse_data = load_nrmse_table(Path(args.table))
    make_collage(
        seed_panels=seed_panels,
        nrmse_data=nrmse_data,
        out_pdf=Path(args.out_pdf),
        out_png=Path(args.out_png),
    )

    print(f"Wrote: {args.out_pdf}")
    print(f"Wrote: {args.out_png}")


if __name__ == "__main__":
    main()
