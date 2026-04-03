#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

FIG_W = 5.7295

ID_COLOR = "#EE6677"
FD_COLOR = "#4477AA"
REFERENCE_COLOR = "#888888"
GRID_COLOR = "#D7D7D7"
AXIS_COLOR = "#1F1F1F"
TICK_COLOR = "#222222"

RAW_COLOR = REFERENCE_COLOR
EVDUP_COLOR = "#6F6F6F"
EVNA_COLOR = "#77AADD"
SPLITNA_COLOR = "#E89AA8"
TRUE_COLOR = "#111111"

FILL_ALPHA_FD = 0.14
FILL_ALPHA_ID = 0.18
FILL_ALPHA_RAW = 0.10
REFERENCE_LW = 1.0
REFERENCE_LS = ":"
REFERENCE_ALPHA = 0.85
FD_LW = 2.0
FD_LS = (0, (4, 2))
ID_LW = 2.2
ID_LS = "-"
RAW_LW = 1.4
RAW_LS = ":"
MARKER_EDGE_COLOR = "white"
MARKER_EDGE_WIDTH = 0.9
ID_MARKER = "o"
FD_MARKER = "s"
RAW_MARKER = "^"

TITLE_MAP = {
    "mackey_glass": "Mackey--Glass",
    "lorenz": "Lorenz",
    "narma10": "NARMA10",
}


def scaled_height(width: float, height: float) -> float:
    return FIG_W * (height / width)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 1.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.7,
            "grid.linestyle": "-",
            "grid.linewidth": 0.75,
            "grid.color": GRID_COLOR,
        }
    )


def style_axis(ax, *, grid: bool = True) -> None:
    ax.tick_params(direction="out", width=0.8, length=3.0, pad=4, colors=TICK_COLOR)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color(AXIS_COLOR)
    if grid:
        ax.grid(True)
    else:
        ax.grid(False)


def save_pdf(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)


def save_pdf_png(fig, pdf_path: Path, png_path: Path, *, png_dpi: int = 300) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=png_dpi)
    plt.close(fig)
