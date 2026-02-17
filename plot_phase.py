"""IEEE-style phase diagram (smooth contourf with zero-centered diverging map)."""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(__file__))

from ieee_plot_style import (  # noqa: E402
    METHOD_COLORS,
    NEUTRAL_ANNOT,
    OURS_LABEL,
    PHASE_CMAP,
    apply_axis_chrome,
    remove_axis_title,
    setup_ieee_style,
)
from qrc_budget import (  # noqa: E402
    OUT_DIR,
    QuantumReservoir,
    compute_rho_ev,
    ev_training,
    grouped_trajectory_training_legacy,
    grouped_trajectory_training_narma_fix,
    mackey_glass,
    narma10,
    run_reservoir_pipeline,
)


def _parse_int_list(raw, default):
    if not raw:
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw, default):
    if not raw:
        return default
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def lorenz_x(n_steps, rng, dt=0.02, warmup=2000):
    sigma_l, rho_l, beta = 10.0, 28.0, 8.0 / 3.0
    total = n_steps + warmup
    x, y, z = 1.0, 1.0, 1.0
    xs = np.zeros(total)
    for i in range(total):
        xs[i] = x
        dx = sigma_l * (y - x) * dt
        dy = (x * (rho_l - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x += dx
        y += dy
        z += dz
    series = xs[warmup:]
    return (series - series.min()) / (series.max() - series.min() + 1e-10)


def mg_series(n_steps, rng):
    return mackey_glass(n_steps, tau=17, rng=rng)


def narma_series(n_steps, rng):
    _, y = narma10(n_steps, rng=rng)
    return y


def run_one(dataset_name, series_func, n_shots, sigma, S, seed=0, use_narma_fix=False):
    rng = np.random.default_rng(seed)
    T = S // n_shots
    if T < 60:
        return None

    series = series_func(T + 50, rng)
    inputs, targets = series[:-1][:T], series[1:][:T]
    res = QuantumReservoir(n_qubits=6, depth=5, rng=rng)
    sf = run_reservoir_pipeline(inputs, res, n_shots, sigma, rng=np.random.default_rng(seed + 1000))
    ev, _, _ = ev_training(sf, targets, 30, 0.7)
    if use_narma_fix and dataset_name == "NARMA-10":
        gt, _, _, _ = grouped_trajectory_training_narma_fix(sf, targets, 30, 0.7)
    else:
        gt, _, _, _ = grouped_trajectory_training_legacy(sf, targets, 30, 0.7)
    return {"ev": ev, "gt": gt, "gap": ev - gt}


def compute_data(shots_list, sigma_list, n_seeds, use_narma_fix):
    ds_configs = {
        "Mackey-Glass": {"func": mg_series, "S": 20000},
        "Lorenz": {"func": lorenz_x, "S": 30000},
        "NARMA-10": {"func": narma_series, "S": 50000},
    }

    all_data = {}
    for ds_name, ds in ds_configs.items():
        S = ds["S"]
        func = ds["func"]
        print(f"\n=== {ds_name} (S={S}) ===")

        gap_grid = np.full((len(shots_list), len(sigma_list)), np.nan)
        ev_grid = np.full_like(gap_grid, np.nan)
        gt_grid = np.full_like(gap_grid, np.nan)
        rho_grid = np.full_like(gap_grid, np.nan)

        total = len(shots_list) * len(sigma_list)
        idx = 0
        for i, n_shots in enumerate(shots_list):
            for j, sigma in enumerate(sigma_list):
                idx += 1
                rho = compute_rho_ev(S, n_shots)
                rho_grid[i, j] = rho
                evs, gts = [], []
                for seed in range(n_seeds):
                    r = run_one(
                        ds_name,
                        func,
                        n_shots,
                        sigma,
                        S,
                        seed,
                        use_narma_fix=use_narma_fix,
                    )
                    if r is None:
                        continue
                    evs.append(r["ev"])
                    gts.append(r["gt"])
                if not evs:
                    continue
                ev_grid[i, j] = np.mean(evs)
                gt_grid[i, j] = np.mean(gts)
                gap_grid[i, j] = np.mean(evs) - np.mean(gts)
                g = gap_grid[i, j]
                print(
                    f"  [{idx:>2d}/{total}] N={n_shots:>2d} sigma={sigma:.2f} rho={rho:.2f} "
                    f"EV={ev_grid[i, j]:.3f} GT={gt_grid[i, j]:.3f} gap={g:+.3f}"
                )

        all_data[ds_name] = {
            "gap": gap_grid,
            "ev": ev_grid,
            "gt": gt_grid,
            "rho": rho_grid,
            "S": S,
        }
    return all_data


def load_or_compute(cache_file, shots_list, sigma_list, n_seeds, use_narma_fix):
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("No cache found, computing...")
    data = compute_data(
        shots_list=shots_list,
        sigma_list=sigma_list,
        n_seeds=n_seeds,
        use_narma_fix=use_narma_fix,
    )
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved cache to {cache_file}")
    return data


def make_plot(
    out_path,
    all_data,
    shots_list,
    sigma_list,
    save_png=False,
    show_overlay_lines=False,
    fig_width=7.16,
    fig_height=2.2,
):
    setup_ieee_style()

    fig = plt.figure(figsize=(float(fig_width), float(fig_height)), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.05], wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cbar_ax = fig.add_subplot(gs[0, 3])

    all_gaps = []
    for data in all_data.values():
        g = np.nan_to_num(data["gap"], nan=0)
        all_gaps.append(g)

    all_flat = np.concatenate([g.ravel() for g in all_gaps])
    absmax = max(abs(np.nanmin(all_flat)), abs(np.nanmax(all_flat)), 0.01)
    norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)

    sigma_arr = np.array(sigma_list)
    shots_arr = np.array(shots_list)
    sigma_fine = np.linspace(sigma_arr[0], sigma_arr[-1], 200)
    shots_fine = np.linspace(shots_arr[0], shots_arr[-1], 200)
    Sf, Nf = np.meshgrid(sigma_fine, shots_fine)

    im_last = None
    for col, (ds_name, data) in enumerate(all_data.items()):
        ax = axes[col]
        S = data["S"]
        gap = data["gap"]
        gap_clean = np.nan_to_num(gap, nan=0)

        pad_shots = np.concatenate(
            [[shots_arr[0] - (shots_arr[1] - shots_arr[0])], shots_arr, [shots_arr[-1] + (shots_arr[-1] - shots_arr[-2])]]
        )
        pad_sigma = np.concatenate(
            [[sigma_arr[0] - (sigma_arr[1] - sigma_arr[0])], sigma_arr, [sigma_arr[-1] + (sigma_arr[-1] - sigma_arr[-2])]]
        )
        gap_padded = np.pad(gap_clean, 1, mode="edge")

        spline = RectBivariateSpline(pad_shots, pad_sigma, gap_padded, kx=3, ky=3)
        gap_smooth = spline(shots_fine, sigma_fine)
        gap_smooth = gaussian_filter(gap_smooth, sigma=6)

        levels = np.linspace(-absmax, absmax, 25)
        im = ax.contourf(Sf, Nf, gap_smooth, levels=levels, cmap=PHASE_CMAP, norm=norm, extend="both")
        im_last = im

        if show_overlay_lines:
            interp_lin = RegularGridInterpolator(
                (shots_arr, sigma_arr),
                gap_clean,
                method="linear",
                bounds_error=False,
                fill_value=0,
            )
            gap_lin = interp_lin((Nf, Sf))
            gap_lin = gaussian_filter(gap_lin, sigma=5)
            cs = ax.contour(Sf, Nf, gap_lin, levels=[0.005], colors="none")
            segs = cs.allsegs[0] if cs.allsegs else []
            wide_segs = [
                s
                for s in segs
                if len(s) > 1 and (s[:, 0].max() - s[:, 0].min()) > 0.4 * (sigma_arr[-1] - sigma_arr[0])
            ]
            if wide_segs:
                longest = max(wide_segs, key=lambda s: len(s))
                ax.plot(longest[:, 0], longest[:, 1], color=METHOD_COLORS["true"], lw=0.8, ls="--", zorder=4)

            # rho_EV guide lines
            d, t0, alpha = 241, 30, 0.7
            for rho_t in [1, 2, 3, 5]:
                N_for_rho = S / (rho_t * d / alpha + t0)
                if shots_arr[0] <= N_for_rho <= shots_arr[-1]:
                    ax.axhline(N_for_rho, color=METHOD_COLORS["ev"], ls=":", lw=0.45, alpha=0.8)

        Sg, Ng = np.meshgrid(sigma_arr, shots_arr)
        ax.scatter(
            Sg.ravel(),
            Ng.ravel(),
            s=8,
            c=METHOD_COLORS["true"],
            edgecolors="white",
            linewidths=0.25,
            zorder=5,
            alpha=0.7,
        )

        ax.set_xlabel(r"$\sigma$")
        if col == 0:
            ax.set_ylabel(r"$N_{\mathrm{shots}}$")
        else:
            ax.set_yticklabels([])

        ax.set_xlim(sigma_arr[0], sigma_arr[-1])
        ax.set_ylim(shots_arr[0], shots_arr[-1])
        remove_axis_title(ax)
        apply_axis_chrome(ax, grid=False)

        # Dataset id as lightweight annotation (not title)
        ax.text(
            0.03,
            0.04,
            ds_name,
            transform=ax.transAxes,
            fontsize=6.3,
            color=NEUTRAL_ANNOT,
            ha="left",
            va="bottom",
        )

    cb = fig.colorbar(im_last, cax=cbar_ax)
    cb.set_label(f"Gap = EV - {OURS_LABEL}", fontsize=7.0)
    cb.ax.tick_params(labelsize=6.5, width=0.4)
    cb.outline.set_linewidth(0.4)

    fig.savefig(out_path)
    if save_png:
        fig.savefig(os.path.splitext(out_path)[0] + ".png", dpi=300)
    print(f"Saved {out_path}")
    if save_png:
        print(f"Saved {os.path.splitext(out_path)[0] + '.png'}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IEEE-style phase diagram.")
    parser.add_argument(
        "--narma-fix",
        action="store_true",
        help="Use NARMA-10 G-Traj fix (std-aware selection + guarded calibration).",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds per grid point.")
    parser.add_argument("--cache-file", type=str, default=None, help="Optional cache file path.")
    parser.add_argument("--output", type=str, default=None, help="Optional output pdf path.")
    parser.add_argument("--png", action="store_true", help="Also export PNG alongside PDF.")
    parser.add_argument(
        "--show-overlay-lines",
        action="store_true",
        help="Show phase boundary and rho guide lines on top of contourf.",
    )
    parser.add_argument(
        "--shots-list",
        type=str,
        default=None,
        help='Comma-separated shot counts, e.g. "10,20,30,40".',
    )
    parser.add_argument(
        "--sigma-list",
        type=str,
        default=None,
        help='Comma-separated sigmas, e.g. "0.0,0.01,0.03,0.05".',
    )
    parser.add_argument("--fig-width", type=float, default=7.16, help="Figure width in inches.")
    parser.add_argument("--fig-height", type=float, default=2.2, help="Figure height in inches.")
    args = parser.parse_args()

    use_narma_fix = bool(args.narma_fix)
    shots_list = _parse_int_list(args.shots_list, [10, 15, 20, 25, 30, 40, 50])
    sigma_list = _parse_float_list(args.sigma_list, [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10])

    if args.cache_file:
        cache_file = args.cache_file
    else:
        if args.shots_list or args.sigma_list:
            cache_name = "data_phase_narma_fix_ieee_custom.pkl" if use_narma_fix else "data_phase_ieee_custom.pkl"
        else:
            cache_name = "data_phase_narma_fix_ieee.pkl" if use_narma_fix else "data_phase_ieee.pkl"
        cache_file = os.path.join(OUT_DIR, cache_name)
    out_path = (
        args.output
        if args.output
        else os.path.join(OUT_DIR, "fig_phase_narma_fix_ieee.pdf" if use_narma_fix else "fig_phase_ieee.pdf")
    )

    mode_label = "narma_fix" if use_narma_fix else "legacy"
    print(f"Mode: {mode_label} | seeds={args.seeds}")

    all_data = load_or_compute(
        cache_file=cache_file,
        shots_list=shots_list,
        sigma_list=sigma_list,
        n_seeds=int(args.seeds),
        use_narma_fix=use_narma_fix,
    )
    make_plot(
        out_path=out_path,
        all_data=all_data,
        shots_list=shots_list,
        sigma_list=sigma_list,
        save_png=bool(args.png),
        show_overlay_lines=bool(args.show_overlay_lines),
        fig_width=float(args.fig_width),
        fig_height=float(args.fig_height),
    )


if __name__ == "__main__":
    main()
