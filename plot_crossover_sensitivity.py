"""Crossover sensitivity analysis across circuit families and depths."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieee_plot_style import (
    ARCH_COLORS,
    METHOD_COLORS,
    NEUTRAL_ZERO,
    apply_axis_chrome,
    remove_axis_title,
    setup_ieee_style,
)
from plot_collapse import load_or_compute
from qrc_budget import OUT_DIR


@dataclass(frozen=True)
class ArchConfig:
    label: str
    family: str
    depth: int


ARCH_CONFIGS = [
    ArchConfig(label="Ring CNOT (d=5)", family="ring_cnot", depth=5),
    ArchConfig(label="Linear CNOT (d=5)", family="line_cnot", depth=5),
    ArchConfig(label="All-to-all CZ (d=5)", family="all_to_all_cz", depth=5),
    ArchConfig(label="Ring CNOT (d=2)", family="ring_cnot", depth=2),
    ArchConfig(label="Ring CNOT (d=10)", family="ring_cnot", depth=10),
]

FAMILY_COLOR = {
    "ring_cnot": METHOD_COLORS["fit"],
    "line_cnot": ARCH_COLORS["q10"],
    "all_to_all_cz": METHOD_COLORS["true"],
}

def _slug(values: list[int | float]) -> str:
    return "-".join(str(v).replace(".", "p") for v in values)


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _smooth_curve(rho: np.ndarray, gap: np.ndarray, x_grid: np.ndarray, bandwidth: float) -> np.ndarray:
    y = np.full_like(x_grid, np.nan, dtype=float)
    if rho.size == 0:
        return y
    for i, x in enumerate(x_grid):
        w = np.exp(-0.5 * ((rho - x) / bandwidth) ** 2)
        w_sum = float(np.sum(w))
        if w_sum > 1e-12:
            y[i] = float(np.sum(w * gap) / w_sum)
    return y


def _first_crossing(x: np.ndarray, y: np.ndarray, target: float) -> float:
    d = y - target
    valid = np.isfinite(d)
    x = x[valid]
    d = d[valid]
    if x.size < 2:
        return float("nan")
    if d[0] < 0:
        return float(x[0])
    for i in range(x.size - 1):
        d0, d1 = d[i], d[i + 1]
        if d0 == 0:
            return float(x[i])
        if d0 > 0 and d1 <= 0:
            x0, x1 = x[i], x[i + 1]
            if d1 == d0:
                return float(x0)
            t = d0 / (d0 - d1)
            return float(x0 + t * (x1 - x0))
    return float("nan")


def _as_float(v: float) -> str:
    if np.isfinite(v):
        return f"{v:.4f}"
    return "nan"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot crossover sensitivity across architecture variants.")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds per grid point.")
    parser.add_argument(
        "--shots-list",
        type=str,
        default="8,10,15,20,25,30,40,50",
        help='Comma-separated shot counts, e.g. "8,10,15,20,25,30,40,50".',
    )
    parser.add_argument(
        "--sigma-list",
        type=str,
        default="0.0,0.02,0.05",
        help='Comma-separated sigma values, e.g. "0.0,0.02,0.05".',
    )
    parser.add_argument("--bandwidth", type=float, default=0.9, help="Kernel bandwidth in rho units.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Practical crossover threshold for Delta (in NRMSE units).",
    )
    parser.add_argument("--narma-fix", action="store_true", help="Use NARMA-10 grouped fix in collapse data.")
    parser.add_argument(
        "--out-csv",
        type=str,
        default=os.path.join(OUT_DIR, "table_crossover_sensitivity.csv"),
        help="Output table CSV path.",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default=os.path.join(OUT_DIR, "fig_crossover_sensitivity_ieee.pdf"),
        help="Output plot path.",
    )
    parser.add_argument("--fig-width", type=float, default=3.5, help="Figure width in inches.")
    parser.add_argument("--fig-height", type=float, default=2.5, help="Figure height in inches.")
    args = parser.parse_args()

    setup_ieee_style()
    shots_list = _parse_int_list(args.shots_list)
    sigma_list = _parse_float_list(args.sigma_list)
    threshold = float(args.threshold)

    x_grid = np.linspace(0.8, 18.0, 320)
    rows: list[dict[str, float | int | str]] = []

    fig, ax = plt.subplots(figsize=(float(args.fig_width), float(args.fig_height)), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for cfg in ARCH_CONFIGS:
        shots_slug = _slug(shots_list)
        sigma_slug = _slug(sigma_list)
        cache_name = (
            f"data_crossover_sens_{'narma_fix' if args.narma_fix else 'legacy'}_"
            f"{cfg.family}_d{cfg.depth}_seeds{int(args.seeds)}_"
            f"shots{shots_slug}_sigma{sigma_slug}.pkl"
        )
        cache_file = os.path.join(OUT_DIR, cache_name)

        data = load_or_compute(
            cache_file=cache_file,
            shots_list=shots_list,
            sigma_list=sigma_list,
            n_seeds=int(args.seeds),
            use_narma_fix=bool(args.narma_fix),
            circuit_family=cfg.family,
            depth=int(cfg.depth),
        )

        rho = np.asarray(data["all_rho"], dtype=float)
        gap = np.asarray(data["all_gap"], dtype=float)
        y_smooth = _smooth_curve(rho, gap, x_grid, bandwidth=float(args.bandwidth))

        rho_cross_zero = _first_crossing(x_grid, y_smooth, target=0.0)
        rho_cross_thr = _first_crossing(x_grid, y_smooth, target=threshold)

        low_mask = rho < 3.0
        low_mean = float(np.mean(gap[low_mask])) if np.any(low_mask) else float("nan")
        low_pos = float(np.mean(gap[low_mask] > 0.0)) if np.any(low_mask) else float("nan")

        rows.append(
            {
                "architecture": cfg.label,
                "family": cfg.family,
                "depth": int(cfg.depth),
                "n_points": int(gap.size),
                "rho_min": float(np.min(rho)) if rho.size else float("nan"),
                "rho_max": float(np.max(rho)) if rho.size else float("nan"),
                "low_rho_mean_gap": low_mean,
                "low_rho_pos_frac": low_pos,
                "rho_cross_delta0": rho_cross_zero,
                "rho_cross_delta_threshold": rho_cross_thr,
                "threshold": threshold,
                "seeds": int(args.seeds),
            }
        )

        color = FAMILY_COLOR.get(cfg.family, METHOD_COLORS["ev"])
        ax.plot(x_grid, y_smooth, lw=1.3, ls="-", color=color, label=cfg.label, alpha=0.95, zorder=3)
        if np.isfinite(rho_cross_zero):
            y_zero = 0.0
            ax.scatter(
                [rho_cross_zero],
                [y_zero],
                s=16,
                zorder=4,
                facecolors="white",
                edgecolors=color,
                linewidths=0.7,
            )

    ax.axhline(0.0, color=NEUTRAL_ZERO, lw=0.5, ls=":", zorder=1)
    ax.set_xlabel(r"$\rho_{\mathrm{EV}}$")
    ax.set_ylabel(r"Gap (NRMSE$_{EV}$ - NRMSE$_{Split}$)")
    ax.set_xlim(0.8, 18.0)
    ax.set_ylim(-0.04, 0.28)
    remove_axis_title(ax)
    apply_axis_chrome(ax, grid=True)
    ax.legend(loc="upper right", borderpad=0.2, labelspacing=0.18, handletextpad=0.35, fontsize=6.2)
    fig.savefig(args.out_plot)
    plt.close(fig)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = [
        "architecture",
        "family",
        "depth",
        "n_points",
        "rho_min",
        "rho_max",
        "low_rho_mean_gap",
        "low_rho_pos_frac",
        "rho_cross_delta0",
        "rho_cross_delta_threshold",
        "threshold",
        "seeds",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {args.out_csv}")
    print(f"Saved {args.out_plot}")
    print("\nCrossover sensitivity summary:")
    for r in rows:
        print(
            f"  {r['architecture']}: "
            f"low-rho mean={_as_float(float(r['low_rho_mean_gap']))}, "
            f"low-rho pos-frac={_as_float(float(r['low_rho_pos_frac']))}, "
            f"rho*(Delta=0)={_as_float(float(r['rho_cross_delta0']))}, "
            f"rho*(Delta={threshold:.3f})={_as_float(float(r['rho_cross_delta_threshold']))}"
        )


if __name__ == "__main__":
    main()
