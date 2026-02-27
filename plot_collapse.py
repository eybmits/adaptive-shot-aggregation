"""IEEE-style rho_EV collapse plot."""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(__file__))

from ieee_plot_style import (  # noqa: E402
    METHOD_COLORS,
    NEUTRAL_ANNOT,
    NEUTRAL_ZERO,
    OURS_LABEL,
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


def _parse_int_list(raw, default):
    if not raw:
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw, default):
    if not raw:
        return default
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def run_one(dataset_name, func, n_shots, sigma, S, seed, use_narma_fix, circuit_family, depth=5):
    rng = np.random.default_rng(seed)
    T = S // n_shots
    if T < 60:
        return None
    series = func(T + 50, rng)
    inputs, targets = series[:-1][:T], series[1:][:T]
    res = QuantumReservoir(n_qubits=6, depth=int(depth), rng=rng, circuit_family=circuit_family)
    sf = run_reservoir_pipeline(inputs, res, n_shots, sigma, rng=np.random.default_rng(seed + 1000))
    ev, _, _ = ev_training(sf, targets, 30, 0.7)
    if use_narma_fix and dataset_name == "NARMA-10":
        gt, _, _, _ = grouped_trajectory_training_narma_fix(sf, targets, 30, 0.7)
    else:
        gt, _, _, _ = grouped_trajectory_training_legacy(sf, targets, 30, 0.7)
    return ev - gt


def compute_data(shots_list, sigma_list, n_seeds, use_narma_fix, circuit_family, depth=5):
    datasets = {
        "Mackey-Glass": (lambda n, rng: mackey_glass(n, tau=17, rng=rng), 20000),
        "Lorenz": (lorenz_x, 30000),
        "NARMA-10": (narma_series, 50000),
    }

    all_rho, all_gap, all_label = [], [], []
    for ds_name, (func, S) in datasets.items():
        print(f"Sweeping {ds_name} (family={circuit_family}, depth={depth})...")
        for n_shots in shots_list:
            for sigma in sigma_list:
                rho = compute_rho_ev(S, n_shots)
                gaps = []
                for seed in range(n_seeds):
                    g = run_one(
                        ds_name,
                        func,
                        n_shots,
                        sigma,
                        S,
                        seed,
                        use_narma_fix,
                        circuit_family,
                        depth=depth,
                    )
                    if g is not None:
                        gaps.append(g)
                if gaps:
                    all_rho.append(rho)
                    all_gap.append(np.mean(gaps))
                    all_label.append(ds_name)
    return {
        "all_rho": np.array(all_rho),
        "all_gap": np.array(all_gap),
        "all_label": np.array(all_label),
    }


def load_or_compute(cache_file, shots_list, sigma_list, n_seeds, use_narma_fix, circuit_family, depth=5):
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
        circuit_family=circuit_family,
        depth=depth,
    )
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved cache to {cache_file}")
    return data


def make_plot(out_path, data, save_png, ours_label: str):
    setup_ieee_style()

    all_rho = data["all_rho"]
    all_gap = data["all_gap"]
    all_label = data["all_label"]

    mask_pos = all_gap > 0.005
    rho_fit = all_rho[mask_pos]
    gap_fit = all_gap[mask_pos]

    def power_law(x, C, beta):
        return C * x ** (-beta)

    has_fit = False
    C_fit = 0.0
    beta_fit = 0.0
    try:
        popt, _ = curve_fit(power_law, rho_fit, gap_fit, p0=[0.3, 1.5], maxfev=5000)
        C_fit, beta_fit = popt
        has_fit = True
        print(f"Fit: Gap = {C_fit:.3f} * rho^(-{beta_fit:.2f})")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for ds, marker, color in [
        ("Mackey-Glass", "o", METHOD_COLORS["true"]),
        ("Lorenz", "s", METHOD_COLORS["evdup"]),
        ("NARMA-10", "^", METHOD_COLORS["ev"]),
    ]:
        mask = all_label == ds
        ax.scatter(
            all_rho[mask],
            all_gap[mask],
            s=20,
            marker=marker,
            color=color,
            alpha=0.85,
            linewidths=0.8,
            label=ds,
            zorder=3,
            facecolors="white",
            edgecolors=color,
        )

    if has_fit:
        rho_curve = np.linspace(0.9, 18, 200)
        gap_curve = np.clip(power_law(rho_curve, C_fit, beta_fit), 0, None)
        ax.plot(
            rho_curve,
            gap_curve,
            color=METHOD_COLORS["fit"],
            lw=1.3,
            ls="-",
            zorder=2,
            label=rf"$C \cdot \rho^{{-\beta}}$, $C$={C_fit:.2f}, $\beta$={beta_fit:.1f}",
        )

    ax.axhline(y=0, color=NEUTRAL_ZERO, lw=0.5, ls=":", zorder=1)

    ax.annotate(f"{ours_label} wins", xy=(2.25, 0.28), fontsize=6.5, color=NEUTRAL_ANNOT)
    ax.annotate("EV wins / equal", xy=(10.0, -0.03), fontsize=6.5, color=NEUTRAL_ANNOT, ha="center")

    ax.set_xlabel(r"$\rho_{\mathrm{EV}}$")
    ax.set_ylabel(f"Gap (NRMSE_EV - NRMSE_{ours_label})")
    ax.set_xlim(0.8, 18)
    ax.set_ylim(-0.04, 0.38)
    remove_axis_title(ax)
    apply_axis_chrome(ax, grid=True)
    ax.legend(loc="upper right", borderpad=0.25, labelspacing=0.2, handletextpad=0.35)

    fig.savefig(out_path)
    if save_png:
        fig.savefig(os.path.splitext(out_path)[0] + ".png", dpi=300)
    print(f"Saved {out_path}")
    if save_png:
        print(f"Saved {os.path.splitext(out_path)[0] + '.png'}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IEEE-style rho_EV collapse.")
    parser.add_argument(
        "--narma-fix",
        action="store_true",
        help="Use NARMA-10 G-Traj fix (std-aware selection + guarded calibration).",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds per grid point.")
    parser.add_argument("--cache-file", type=str, default=None, help="Optional cache file path.")
    parser.add_argument("--output", type=str, default=None, help="Optional output pdf path.")
    parser.add_argument("--png", action="store_true", help="Also export PNG alongside PDF.")
    parser.add_argument("--ours-label", type=str, default=OURS_LABEL, help="Label for the proposed method annotation.")
    parser.add_argument(
        "--circuit-family",
        type=str,
        default="ring_cnot",
        choices=list(QuantumReservoir.ALLOWED_CIRCUIT_FAMILIES),
        help="Reservoir entangler family.",
    )
    parser.add_argument("--depth", type=int, default=5, help="Reservoir depth.")
    parser.add_argument(
        "--shots-list",
        type=str,
        default=None,
        help='Comma-separated shot counts, e.g. "5,10,20,30,40".',
    )
    parser.add_argument(
        "--sigma-list",
        type=str,
        default=None,
        help='Comma-separated sigmas, e.g. "0.0,0.01,0.03,0.05".',
    )
    args = parser.parse_args()

    use_narma_fix = bool(args.narma_fix)
    shots_list = _parse_int_list(args.shots_list, [5, 8, 10, 15, 20, 25, 30, 40, 50])
    sigma_list = _parse_float_list(args.sigma_list, [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10])

    base_cache = "data_collapse_narma_fix_ieee" if use_narma_fix else "data_collapse_ieee"
    suffix_parts = []
    if args.circuit_family != "ring_cnot":
        suffix_parts.append(args.circuit_family)
    if int(args.depth) != 5:
        suffix_parts.append(f"d{int(args.depth)}")
    if args.shots_list or args.sigma_list:
        suffix_parts.append("custom")
    cache_name = base_cache + (f"_{'_'.join(suffix_parts)}" if suffix_parts else "") + ".pkl"

    if args.cache_file:
        cache_file = args.cache_file
    else:
        cache_file = os.path.join(OUT_DIR, cache_name)

    out_base = "fig_collapse_narma_fix_ieee" if use_narma_fix else "fig_collapse_ieee"
    if args.circuit_family != "ring_cnot":
        out_base = f"{out_base}_{args.circuit_family}"
    if int(args.depth) != 5:
        out_base = f"{out_base}_d{int(args.depth)}"
    out_path = (
        args.output
        if args.output
        else os.path.join(OUT_DIR, f"{out_base}.pdf")
    )

    mode_label = "narma_fix" if use_narma_fix else "legacy"
    print(f"Mode: {mode_label} | family={args.circuit_family} | depth={args.depth} | seeds={args.seeds}")
    data = load_or_compute(
        cache_file=cache_file,
        shots_list=shots_list,
        sigma_list=sigma_list,
        n_seeds=int(args.seeds),
        use_narma_fix=use_narma_fix,
        circuit_family=str(args.circuit_family),
        depth=int(args.depth),
    )
    make_plot(out_path=out_path, data=data, save_png=bool(args.png), ours_label=str(args.ours_label))


if __name__ == "__main__":
    main()
