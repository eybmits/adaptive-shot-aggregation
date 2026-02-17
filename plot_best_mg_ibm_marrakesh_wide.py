#!/usr/bin/env python3
"""Replot existing ibm_marrakesh Mackey-Glass run in benchmark style (no new hardware jobs)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

from qrc_budget import (
    OUT_DIR,
    apply_noise,
    ev_training,
    grouped_trajectory_training_legacy,
    mackey_glass,
    trajectory_training,
)
from ieee_plot_style import OURS_LABEL, TRAJ_LABEL, TRAJ_TICK_LABEL


COLORS = {
    "true": "#000000",
    "ev": "#555555",
    "traj": "#aaaaaa",
    "gtraj": "#c0392b",
}
HATCHES = {
    "ev": "///",
    "traj": "...",
    "gtraj": "",
}
BAR_EDGE = "#333333"


def setup_style() -> None:
    # Match the benchmark panel style from plot_paper.py.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "medium",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.0,
        "axes.grid": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _bitstrings_to_pm_one(bitstrings: Sequence[str], q: int) -> np.ndarray:
    arr = np.empty((len(bitstrings), q), dtype=float)
    for s, bits in enumerate(bitstrings):
        rev = bits[::-1]  # from c_{q-1}...c_0 to q0...q_{q-1}
        arr[s, :] = [1.0 if b == "0" else -1.0 for b in rev]
    return arr


def _window_features(r: np.ndarray, L: int) -> np.ndarray:
    n_shots, T, M = r.shape
    shot_features = np.zeros((n_shots, T, L * M), dtype=float)
    for t in range(T):
        for l in range(L):
            idx = t - L + 1 + l
            if idx >= 0:
                shot_features[:, t, l * M:(l + 1) * M] = r[:, idx, :]
    return shot_features


def _select_trace_window(y_true: np.ndarray, window: int | None) -> tuple[int, int]:
    n = len(y_true)
    if window is None or window <= 0 or window >= n:
        return 0, n
    best_s = 0
    best_var = -np.inf
    for s in range(0, n - window + 1):
        v = np.var(y_true[s:s + window])
        if v > best_var:
            best_var = v
            best_s = s
    return best_s, best_s + window


def _read_job_table(path: Path) -> List[Dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["seed"]))
    return rows


def _reconstruct_seed_from_jobs(
    *,
    service: QiskitRuntimeService,
    row: Dict,
    S: int,
    n_shots: int,
    sigma: float,
    q: int,
    L: int,
    eta: float,
    t0: int,
    alpha: float,
    ridge_lambda: float,
    chunk_size: int,
) -> Dict:
    seed = int(row["seed"])
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 2000)
    T = S // n_shots
    series = mackey_glass(T + 50, tau=17, rng=rng)
    targets = series[1:][:T]

    job_ids = [j for j in row["job_ids"].split(";") if j.strip()]
    expected_pairs = int(np.ceil(T / chunk_size))
    if len(job_ids) != 2 * expected_pairs:
        raise ValueError(
            f"Seed {seed}: expected {2 * expected_pairs} job ids, got {len(job_ids)}."
        )

    M = 2 * q
    r = np.zeros((n_shots, T, M), dtype=float)
    t = 0

    for pair_idx in range(expected_pairs):
        job_z_id = job_ids[2 * pair_idx]
        job_x_id = job_ids[2 * pair_idx + 1]
        res_z = service.job(job_z_id).result()
        res_x = service.job(job_x_id).result()

        n_batch = len(res_z)
        if n_batch != len(res_x):
            raise ValueError(
                f"Seed {seed}: batch size mismatch for jobs {job_z_id} vs {job_x_id}."
            )

        for j in range(n_batch):
            if t >= T:
                break
            z_bits = res_z[j].data.meas.get_bitstrings()
            x_bits = res_x[j].data.meas.get_bitstrings()
            z_arr = _bitstrings_to_pm_one(z_bits, q)
            x_arr = _bitstrings_to_pm_one(x_bits, q)
            m = np.concatenate([z_arr, x_arr], axis=1)
            m_noisy = apply_noise(m, sigma, noise_rng)

            if t == 0:
                r[:, t, :] = eta * m_noisy
            else:
                r[:, t, :] = (1.0 - eta) * r[:, t - 1, :] + eta * m_noisy
            t += 1

        print(f"[reconstruct] seed {seed}: processed {t}/{T} time steps")

    if t != T:
        raise RuntimeError(f"Seed {seed}: reconstructed {t}/{T} time steps.")

    shot_features = _window_features(r, L)

    ev, yp_ev, yt = ev_training(
        shot_features, targets, t0, alpha, ridge_lambda=ridge_lambda
    )
    traj, yp_traj, _ = trajectory_training(
        shot_features, targets, t0, alpha, ridge_lambda=ridge_lambda
    )
    gtraj, yp_gtraj, _, best_k = grouped_trajectory_training_legacy(
        shot_features, targets, t0, alpha, ridge_lambda=ridge_lambda
    )

    return {
        "seed": seed,
        "ev": float(ev),
        "traj": float(traj),
        "gtraj": float(gtraj),
        "best_k": int(best_k),
        "yt": yt,
        "yp_ev": yp_ev,
        "yp_traj": yp_traj,
        "yp_gtraj": yp_gtraj,
        "gap": float(ev - gtraj),
    }


def _summarize(results: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for method in ("ev", "traj", "gtraj"):
        vals = np.array([r[method] for r in results], dtype=float)
        out[method] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return out


def _save_reconstructed_table(results: Sequence[Dict], outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "nrmse_ev",
                "nrmse_traj",
                "nrmse_gtraj",
                "best_k",
                "gap_ev_minus_gtraj",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "seed": r["seed"],
                "nrmse_ev": r["ev"],
                "nrmse_traj": r["traj"],
                "nrmse_gtraj": r["gtraj"],
                "best_k": r["best_k"],
                "gap_ev_minus_gtraj": r["gap"],
            })


def _make_single_row_plot(
    *,
    trace_result: Dict,
    bar_results: Sequence[Dict],
    summary: Dict[str, Dict[str, float]],
    outpath: Path,
    S: int,
    n_shots: int,
    sigma: float,
    trace_window: int | None,
) -> None:
    setup_style()

    s, e = _select_trace_window(trace_result["yt"], trace_window)
    x = np.arange(len(trace_result["yt"]))[s:e]

    # One-row variant of benchmark layout: same style, time-series spans first 2 columns.
    fig = plt.figure(figsize=(7.2, 2.25))
    gs = gridspec.GridSpec(
        1, 3,
        hspace=0.55, wspace=0.45,
        left=0.08, right=0.97, top=0.95, bottom=0.18
    )
    ax_ts = fig.add_subplot(gs[0, 0:2])
    ax_bar = fig.add_subplot(gs[0, 2])

    # Time-series panel with benchmark style ordering/colors.
    y_true = trace_result["yt"][s:e]
    yp_tr = trace_result["yp_traj"][s:e]
    yp_ev = trace_result["yp_ev"][s:e]
    yp_gt = trace_result["yp_gtraj"][s:e]

    ax_ts.plot(x, y_true, color=COLORS["true"], lw=1.8, label="True", zorder=4)
    if len(yp_tr) == len(x):
        ax_ts.plot(x, yp_tr, color=COLORS["traj"], lw=0.9, alpha=0.5, label=TRAJ_LABEL, zorder=1)
    if len(yp_ev) == len(x):
        ax_ts.plot(x, yp_ev, color=COLORS["ev"], lw=1.2, ls="--", alpha=0.85, label="EV", zorder=3)
    if len(yp_gt) == len(x):
        ax_ts.plot(x, yp_gt, color=COLORS["gtraj"], lw=1.2, alpha=0.9, label=OURS_LABEL, zorder=2)
    ax_ts.set_xlabel("Test time step")
    ax_ts.set_ylabel("Mackey-Glass")
    ax_ts.legend(
        frameon=True,
        fancybox=False,
        edgecolor="#999999",
        facecolor="white",
        framealpha=0.95,
        loc="upper right",
        handlelength=1.5,
        borderpad=0.4,
        labelspacing=0.25,
        fontsize=6.5,
    )
    ax_ts.text(
        0.02,
        0.05,
        f"S={S//1000}k, N={n_shots}, σ={sigma:.2f}",
        transform=ax_ts.transAxes,
        fontsize=6,
        color="#666666",
        va="bottom",
    )

    # NRMSE bars panel.
    methods = ["EV", TRAJ_TICK_LABEL, OURS_LABEL]
    means = [summary["ev"]["mean"], summary["traj"]["mean"], summary["gtraj"]["mean"]]
    stds = [summary["ev"]["std"], summary["traj"]["std"], summary["gtraj"]["std"]]
    colors = [COLORS["ev"], COLORS["traj"], COLORS["gtraj"]]
    hatches = [HATCHES["ev"], HATCHES["traj"], HATCHES["gtraj"]]

    for i, (m, sdev, c, h) in enumerate(zip(means, stds, colors, hatches)):
        ax_bar.bar(
            i,
            m,
            yerr=sdev,
            width=0.55,
            color=c,
            hatch=h,
            edgecolor=BAR_EDGE,
            linewidth=0.6,
            capsize=2.5,
            error_kw={"linewidth": 0.7, "capthick": 0.7},
        )
    ax_bar.set_xticks(np.arange(3))
    ax_bar.set_xticklabels(methods)
    ax_bar.set_ylabel("NRMSE")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replot existing ibm_marrakesh jobs in benchmark style (time-series spans two columns)."
    )
    parser.add_argument("--backend", type=str, default="ibm_marrakesh")
    parser.add_argument("--table", type=str, default="results/table_best_mg_ibm_marrakesh.csv")
    parser.add_argument("--S", type=int, default=20000)
    parser.add_argument("--n-shots", type=int, default=40)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--q", type=int, default=6)
    parser.add_argument("--L", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--t0", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--lambda", dest="ridge_lambda", type=float, default=10.0)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--trace-window", type=int, default=80)
    parser.add_argument("--outdir", type=str, default=OUT_DIR)
    parser.add_argument("--outfile", type=str, default="fig_best_mg_ibm_marrakesh_benchmark_style.pdf")
    parser.add_argument(
        "--reconstructed-table",
        type=str,
        default="table_best_mg_ibm_marrakesh_reconstructed.csv",
    )
    parser.add_argument("--individual-dir", type=str, default="hardware_individual")
    parser.add_argument("--no-individual", action="store_true")
    args = parser.parse_args()

    rows = _read_job_table(Path(args.table))
    if not rows:
        raise ValueError(f"No rows found in {args.table}")

    service = QiskitRuntimeService()
    results: List[Dict] = []
    for row in rows:
        results.append(
            _reconstruct_seed_from_jobs(
                service=service,
                row=row,
                S=int(args.S),
                n_shots=int(args.n_shots),
                sigma=float(args.sigma),
                q=int(args.q),
                L=int(args.L),
                eta=float(args.eta),
                t0=int(args.t0),
                alpha=float(args.alpha),
                ridge_lambda=float(args.ridge_lambda),
                chunk_size=int(args.chunk_size),
            )
        )

    summary = _summarize(results)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _save_reconstructed_table(results, outdir / args.reconstructed_table)
    # Main combined figure: time series from best hardware seed, bars from all seeds.
    trace_result = min(results, key=lambda r: r["gtraj"])
    _make_single_row_plot(
        trace_result=trace_result,
        bar_results=results,
        summary=summary,
        outpath=outdir / args.outfile,
        S=int(args.S),
        n_shots=int(args.n_shots),
        sigma=float(args.sigma),
        trace_window=int(args.trace_window) if args.trace_window > 0 else None,
    )

    # Optional individual seed figures with same 2-panel layout.
    individual_written: List[Path] = []
    if not args.no_individual:
        ind_dir = outdir / args.individual_dir
        ind_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            seed_summary = {
                "ev": {"mean": r["ev"], "std": 0.0},
                "traj": {"mean": r["traj"], "std": 0.0},
                "gtraj": {"mean": r["gtraj"], "std": 0.0},
            }
            p = ind_dir / f"fig_best_mg_ibm_marrakesh_seed_{r['seed']}.pdf"
            _make_single_row_plot(
                trace_result=r,
                bar_results=[r],
                summary=seed_summary,
                outpath=p,
                S=int(args.S),
                n_shots=int(args.n_shots),
                sigma=float(args.sigma),
                trace_window=int(args.trace_window) if args.trace_window > 0 else None,
            )
            individual_written.append(p)

    print("=== Reconstructed Hardware Summary (NRMSE) ===")
    print(f"EV     : {summary['ev']['mean']:.3f} +/- {summary['ev']['std']:.3f}")
    print(f"{TRAJ_LABEL}: {summary['traj']['mean']:.3f} +/- {summary['traj']['std']:.3f}")
    print(f"{OURS_LABEL}: {summary['gtraj']['mean']:.3f} +/- {summary['gtraj']['std']:.3f}")
    print(f"Gap EV-{OURS_LABEL}: {summary['ev']['mean'] - summary['gtraj']['mean']:+.3f}")
    print(f"Wrote: {outdir / args.outfile}")
    print(f"Wrote: {(outdir / args.outfile).with_suffix('.png')}")
    print(f"Wrote: {outdir / args.reconstructed_table}")
    if individual_written:
        print(f"Wrote individual seed plots to: {outdir / args.individual_dir}")


if __name__ == "__main__":
    main()
