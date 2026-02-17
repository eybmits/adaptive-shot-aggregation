"""IEEE-style benchmark comparison figure generation."""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ieee_plot_style import (  # noqa: E402
    BAR_EDGE,
    LEGEND_EDGE,
    METHOD_COLORS,
    NEUTRAL_ANNOT,
    NEUTRAL_ZERO,
    OURS_LABEL,
    TRAJ_LABEL,
    TRAJ_TICK_LABEL,
    apply_axis_chrome,
    remove_axis_title,
    setup_ieee_style,
)
from qrc_budget import (  # noqa: E402
    OUT_DIR,
    QuantumReservoir,
    ev_training,
    grouped_trajectory_training_legacy,
    grouped_trajectory_training_narma_fix,
    mackey_glass,
    narma10,
    run_reservoir_pipeline,
    trajectory_training,
)


def lorenz_x(n_steps, rng, dt=0.02, warmup=2000):
    s, r, b = 10.0, 28.0, 8.0 / 3.0
    total = n_steps + warmup
    x, y, z = 1.0, 1.0, 1.0
    xs = np.zeros(total)
    for i in range(total):
        xs[i] = x
        dx = s * (y - x) * dt
        dy = (x * (r - z) - y) * dt
        dz = (x * y - b * z) * dt
        x += dx
        y += dy
        z += dz
    out = xs[warmup:]
    return (out - out.min()) / (out.max() - out.min() + 1e-10)


def narma_series(n_steps, rng):
    _, y = narma10(n_steps, rng=rng)
    return y


def select_trace_window(y_true, window):
    n = len(y_true)
    if window is None or window <= 0 or window >= n:
        return 0, n
    best_s = 0
    best_var = -np.inf
    for s in range(0, n - window + 1):
        v = np.var(y_true[s : s + window])
        if v > best_var:
            best_var = v
            best_s = s
    return best_s, best_s + window


def resolve_trace_interval(y_true, cfg):
    n = len(y_true)
    if "trace_start" in cfg and "trace_end" in cfg:
        s = int(max(0, cfg["trace_start"]))
        e = int(min(n, cfg["trace_end"]))
        if e > s:
            return s, e
    return select_trace_window(y_true, cfg.get("trace_window", None))


def run_dataset(dataset_name, series_func, n_shots, sigma, S, n_seeds=8):
    all_results = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        T = S // n_shots
        series = series_func(T + 50, rng)
        inputs, targets = series[:-1][:T], series[1:][:T]
        res = QuantumReservoir(n_qubits=6, depth=5, rng=rng)
        sf = run_reservoir_pipeline(
            inputs,
            res,
            n_shots,
            sigma,
            rng=np.random.default_rng(seed + 1000),
        )
        ev, yp_ev, yt = ev_training(sf, targets, 30, 0.7)
        tr, yp_tr, _ = trajectory_training(sf, targets, 30, 0.7)
        if dataset_name == "NARMA-10":
            gt, yp_gt, _, k = grouped_trajectory_training_narma_fix(sf, targets, 30, 0.7)
        else:
            gt, yp_gt, _, k = grouped_trajectory_training_legacy(sf, targets, 30, 0.7)
        all_results.append(
            {
                "ev": ev,
                "traj": tr,
                "gtraj": gt,
                "yp_ev": yp_ev,
                "yp_tr": yp_tr,
                "yp_gt": yp_gt,
                "yt": yt,
                "k": k,
            }
        )
        print(f"  seed {seed + 1}/{n_seeds} done")
    return all_results


def compute_data(n_seeds: int):
    configs = [
        {
            "name": "Mackey-Glass",
            "func": lambda n, rng: mackey_glass(n, tau=17, rng=rng),
            "S": 20000,
            "N": 40,
            "sigma": 0.03,
        },
        {"name": "Lorenz attractor", "func": lorenz_x, "S": 30000, "N": 40, "sigma": 0.0},
        {"name": "NARMA-10", "func": narma_series, "S": 50000, "N": 60, "sigma": 0.03},
    ]
    all_data = {}
    for cfg in configs:
        print(f"\n  Running {cfg['name']}...")
        results = run_dataset(
            cfg["name"],
            cfg["func"],
            cfg["N"],
            cfg["sigma"],
            cfg["S"],
            n_seeds=n_seeds,
        )
        all_data[cfg["name"]] = {
            "results": results,
            "cfg": {k: v for k, v in cfg.items() if k != "func"},
        }
    return all_data


def load_or_compute(cache_file: str, n_seeds: int):
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    print("No cache found, computing...")
    data = compute_data(n_seeds=n_seeds)
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved cache to {cache_file}")
    return data


def _plot_bars(ax, means, stds, colors):
    x_pos = np.arange(3)
    hatches = ["///", "...", ""]
    for xi, m, s, c, h in zip(x_pos, means, stds, colors, hatches):
        ax.bar(
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
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["EV", TRAJ_TICK_LABEL, OURS_LABEL])


def make_figure1(
    cache_file: str,
    output: str,
    save_png: bool,
    left_middle_only: bool = False,
) -> None:
    print("=== Figure 1: Benchmark comparison (IEEE) ===")
    setup_ieee_style()

    all_data = load_or_compute(cache_file=cache_file, n_seeds=8)

    configs = [
        {"name": "Mackey-Glass", "S": 20000, "N": 40, "sigma": 0.03},
        {"name": "Lorenz attractor", "S": 30000, "N": 40, "sigma": 0.0},
        {
            "name": "NARMA-10",
            "S": 50000,
            "N": 60,
            "sigma": 0.03,
            "trace_start": 60,
            "trace_end": 120,
        },
    ]

    n_cols = 2 if left_middle_only else 3
    width_ratios = [2.8, 1.0] if left_middle_only else [2.8, 1.0, 1.0]
    if left_middle_only:
        # Compact IEEE double-column layout with reduced whitespace.
        fig = plt.figure(figsize=(7.16, 5.05), constrained_layout=True)
        gs = fig.add_gridspec(3, n_cols, width_ratios=width_ratios, wspace=0.08, hspace=0.18)
    else:
        fig = plt.figure(figsize=(7.16, 5.55), constrained_layout=True)
        gs = fig.add_gridspec(3, n_cols, width_ratios=width_ratios, wspace=0.20, hspace=0.24)

    for row, cfg in enumerate(configs):
        results = all_data[cfg["name"]]["results"]

        ev_s = [r["ev"] for r in results]
        tr_s = [r["traj"] for r in results]
        gt_s = [r["gtraj"] for r in results]

        # Time trace
        ax = fig.add_subplot(gs[row, 0])
        r0 = results[0]
        full_x = np.arange(len(r0["yt"]))
        s, e = resolve_trace_interval(r0["yt"], cfg)
        x = full_x[s:e]
        y_true = r0["yt"][s:e]
        yp_tr = r0["yp_tr"][s:e]
        yp_ev = r0["yp_ev"][s:e]
        yp_gt = r0["yp_gt"][s:e]

        ax.plot(x, y_true, color=METHOD_COLORS["true"], lw=2.0, label="True", zorder=4)
        if len(yp_tr) == len(x):
            ax.plot(x, yp_tr, color=METHOD_COLORS["traj"], lw=1.2, alpha=0.92, label=TRAJ_LABEL, zorder=1)
        if len(yp_ev) == len(x):
            ax.plot(
                x,
                yp_ev,
                color=METHOD_COLORS["ev"],
                lw=1.7,
                ls="--",
                label="EV",
                zorder=3,
            )
        if len(yp_gt) == len(x):
            ax.plot(
                x,
                yp_gt,
                color=METHOD_COLORS["gtraj"],
                lw=1.8,
                label=OURS_LABEL,
                zorder=2,
            )
        ax.set_xlabel("Test time step")
        ax.set_ylabel(cfg["name"])
        remove_axis_title(ax)
        apply_axis_chrome(ax, grid=False)
        ax.legend(
            loc="upper right",
            handlelength=1.6,
            borderpad=0.25,
            labelspacing=0.22,
            frameon=True,
            fancybox=False,
            facecolor="white",
            edgecolor=LEGEND_EDGE,
            framealpha=0.95,
        )

        # NRMSE bars
        ax = fig.add_subplot(gs[row, 1])
        means = [np.mean(ev_s), np.mean(tr_s), np.mean(gt_s)]
        stds = [np.std(ev_s), np.std(tr_s), np.std(gt_s)]
        _plot_bars(ax, means, stds, [METHOD_COLORS["ev"], METHOD_COLORS["traj"], METHOD_COLORS["gtraj"]])
        ax.set_ylabel("NRMSE")
        remove_axis_title(ax)
        apply_axis_chrome(ax, grid=True)

        if not left_middle_only:
            # Std ratio bars (right column)
            ax = fig.add_subplot(gs[row, 2])
            metrics = {"EV": [], TRAJ_LABEL: [], OURS_LABEL: []}
            for r in results:
                yt = r["yt"]
                std_yt = max(np.std(yt), 1e-10)
                for name, key in [("EV", "yp_ev"), (TRAJ_LABEL, "yp_tr"), (OURS_LABEL, "yp_gt")]:
                    p = r[key]
                    if len(p) == len(yt):
                        metrics[name].append(np.std(p) / std_yt)
            m = [np.mean(metrics[k]) for k in ["EV", TRAJ_LABEL, OURS_LABEL]]
            sdev = [np.std(metrics[k]) for k in ["EV", TRAJ_LABEL, OURS_LABEL]]
            _plot_bars(ax, m, sdev, [METHOD_COLORS["ev"], METHOD_COLORS["traj"], METHOD_COLORS["gtraj"]])
            ax.axhline(y=1.0, color=NEUTRAL_ZERO, ls=":", lw=0.5)
            ax.set_ylabel("Std ratio")
            remove_axis_title(ax)
            apply_axis_chrome(ax, grid=True)

    # Config annotations in each trace panel
    for row, cfg in enumerate(configs):
        ax = fig.axes[row * n_cols]
        ax.text(
            0.02,
            0.03,
            f"S={cfg['S'] // 1000}k, N={cfg['N']}, sigma={cfg['sigma']}",
            transform=ax.transAxes,
            fontsize=6.2,
            color=NEUTRAL_ANNOT,
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="none",
                alpha=0.9,
            ),
        )

    out_path = output
    fig.savefig(out_path)
    if save_png:
        fig.savefig(os.path.splitext(out_path)[0] + ".png", dpi=300)
    print(f"Saved {out_path}")
    if save_png:
        print(f"Saved {os.path.splitext(out_path)[0] + '.png'}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render IEEE-style benchmark comparison figure.")
    parser.add_argument(
        "--cache-file",
        type=str,
        default=os.path.join(OUT_DIR, "data_benchmarks_narma_fix.pkl"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(OUT_DIR, "fig_benchmarks_ieee.pdf"),
    )
    parser.add_argument("--png", action="store_true", help="Also export PNG alongside PDF.")
    parser.add_argument(
        "--left-middle-only",
        action="store_true",
        help="Render only left + middle columns (drop right column).",
    )
    args = parser.parse_args()

    make_figure1(
        cache_file=args.cache_file,
        output=args.output,
        save_png=bool(args.png),
        left_middle_only=bool(args.left_middle_only),
    )


if __name__ == "__main__":
    main()
