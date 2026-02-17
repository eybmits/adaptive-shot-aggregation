"""Central IEEE-style plotting config and helpers for this repository."""

from __future__ import annotations

import matplotlib as mpl


# Method colors (global, consistent across plots)
GROUND_TRUTH = "#2d3436"
OURS = "#c0392b"
EV = "#636e72"
TRAJ = "#b2bec3"
EV_DUP = "#95a5a6"

# Architecture colors
Q6 = "#2d3436"
Q8 = "#c0392b"
Q10 = "#2471a3"

# Diverging phase color map
PHASE_CMAP = "RdBu_r"

# Display names
OURS_LABEL = "Split-Ensemble"
TRAJ_LABEL = "Single-Path Readout (SPR)"
TRAJ_TICK_LABEL = "SPR"

# Neutral UI/support tones used consistently across figures
NEUTRAL_ANNOT = "#4f575b"
NEUTRAL_ZERO = "#7f8c8d"
LEGEND_EDGE = "#bfc5c8"
BAR_EDGE = "#4f575b"
GRID_COLOR = "#b0b0b0"


METHOD_COLORS = {
    "true": GROUND_TRUTH,
    "gtraj": OURS,
    "ev": EV,
    "traj": TRAJ,
    "evdup": EV_DUP,
    "fit": OURS,
}

ARCH_COLORS = {
    "q6": Q6,
    "q8": Q8,
    "q10": Q10,
}


def setup_ieee_style() -> None:
    """Apply a shared IEEE-like rcParams profile."""
    mpl.rcParams.update(
        {
            # Fonts
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            # Lines and markers
            "lines.linewidth": 1.3,
            "lines.markersize": 4,
            # Axes and ticks
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            # Grid (enabled per axis where needed)
            "axes.grid": False,
            "grid.linewidth": 0.3,
            "grid.alpha": 0.4,
            # Export
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.constrained_layout.use": True,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Legend
            "legend.frameon": False,
        }
    )


def apply_axis_chrome(ax, *, grid: bool = True) -> None:
    """Set consistent axis styling (spines/ticks/grid) for line/scatter plots."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(direction="in", width=0.5)
    if grid:
        ax.grid(True, linewidth=0.3, alpha=0.4, color=GRID_COLOR)


def remove_axis_title(ax) -> None:
    """Remove axis title text to satisfy no-title figure variants."""
    ax.set_title("")


def open_marker_kwargs(color: str) -> dict:
    """Open marker style for clearer overlap in compact IEEE figures."""
    return {
        "markerfacecolor": "white",
        "markeredgecolor": color,
        "markeredgewidth": 0.8,
    }
