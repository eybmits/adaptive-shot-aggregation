#!/usr/bin/env python3
"""Run the best Mackey-Glass configuration on IBM hardware (ibm_marrakesh)."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from qrc_budget import (
    OUT_DIR,
    mackey_glass,
    apply_noise,
    ev_training,
    trajectory_training,
    grouped_trajectory_training_legacy,
)


def setup_style() -> None:
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
        "lines.linewidth": 1.1,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.35,
        "figure.dpi": 240,
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _build_parameterized_circuits(
    *,
    q: int,
    depth: int,
    W: np.ndarray,
    res_params: Sequence[np.ndarray],
) -> Tuple[QuantumCircuit, QuantumCircuit]:
    u = Parameter("u")

    qc_z = QuantumCircuit(q)
    for i in range(q):
        qc_z.ry(float(W[i]) * u, i)

    for layer in res_params:
        for i in range(q):
            j = (i + 1) % q
            qc_z.cx(i, j)
        for i in range(q):
            qc_z.rx(float(layer[i, 0]), i)
            qc_z.ry(float(layer[i, 1]), i)
            qc_z.rz(float(layer[i, 2]), i)

    qc_x = qc_z.copy()
    for i in range(q):
        qc_x.h(i)

    qc_z.measure_all()
    qc_x.measure_all()
    return qc_z, qc_x


def _bind_parameter(circuit: QuantumCircuit, value: float) -> QuantumCircuit:
    params = list(circuit.parameters)
    if len(params) != 1:
        raise ValueError(f"Expected exactly one circuit parameter, got {len(params)}.")
    return circuit.assign_parameters({params[0]: float(value)}, inplace=False)


def _bitstrings_to_pm_one(bitstrings: Sequence[str], q: int) -> np.ndarray:
    arr = np.empty((len(bitstrings), q), dtype=float)
    for s, bits in enumerate(bitstrings):
        rev = bits[::-1]  # Qiskit returns c_{q-1}...c_0; we want q0..q_{q-1}
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


def _run_single_seed_hardware(
    *,
    service: QiskitRuntimeService,
    backend_name: str,
    seed: int,
    S: int,
    n_shots: int,
    sigma: float,
    q: int,
    depth: int,
    L: int,
    eta: float,
    t0: int,
    alpha: float,
    ridge_lambda: float,
    chunk_size: int,
) -> Dict:
    if n_shots <= 0:
        raise ValueError("n_shots must be positive.")

    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 2000)
    T = S // n_shots
    series = mackey_glass(T + 50, tau=17, rng=rng)
    inputs = series[:-1][:T]
    targets = series[1:][:T]

    # Keep RNG protocol aligned with the simulator baseline for reservoir params.
    W = rng.uniform(-np.pi, np.pi, size=q)
    res_params = [rng.uniform(0, 2 * np.pi, size=(q, 3)) for _ in range(depth)]

    backend = service.backend(backend_name)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)

    qc_z, qc_x = _build_parameterized_circuits(q=q, depth=depth, W=W, res_params=res_params)
    isa_z = pm.run(qc_z)
    isa_x = pm.run(qc_x)

    sampler = SamplerV2(mode=backend)
    M = 2 * q
    r = np.zeros((n_shots, T, M), dtype=float)
    job_ids: List[str] = []

    indices = list(range(T))
    for start in range(0, T, chunk_size):
        batch_idx = indices[start:start + chunk_size]
        pubs_z = [_bind_parameter(isa_z, float(inputs[t])) for t in batch_idx]
        pubs_x = [_bind_parameter(isa_x, float(inputs[t])) for t in batch_idx]

        job_z = sampler.run(pubs_z, shots=n_shots)
        job_x = sampler.run(pubs_x, shots=n_shots)
        job_ids.extend([job_z.job_id(), job_x.job_id()])

        res_z = job_z.result()
        res_x = job_x.result()

        for j, t in enumerate(batch_idx):
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

        print(
            f"[seed {seed}] processed {min(T, start + chunk_size)}/{T} time steps "
            f"(last job ids: {job_ids[-2]}, {job_ids[-1]})"
        )

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
        "seed": int(seed),
        "ev": float(ev),
        "traj": float(traj),
        "gtraj": float(gtraj),
        "best_k": int(best_k),
        "yt": yt,
        "yp_ev": yp_ev,
        "yp_traj": yp_traj,
        "yp_gtraj": yp_gtraj,
        "job_ids": job_ids,
    }


def _summarize(results: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for method in ("ev", "traj", "gtraj"):
        vals = np.array([r[method] for r in results], dtype=float)
        summary[method] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return summary


def _save_table(results: Sequence[Dict], outpath: Path) -> None:
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
                "job_ids",
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
                "gap_ev_minus_gtraj": float(r["ev"] - r["gtraj"]),
                "job_ids": ";".join(r["job_ids"]),
            })


def _make_plot(
    *,
    results: Sequence[Dict],
    summary: Dict[str, Dict[str, float]],
    outpath: Path,
    title: str,
) -> None:
    setup_style()

    gvals = np.array([r["gtraj"] for r in results], dtype=float)
    rep = results[int(np.argmin(gvals))]

    methods = ["EV", "Traj", "G-Traj"]
    means = [summary["ev"]["mean"], summary["traj"]["mean"], summary["gtraj"]["mean"]]
    stds = [summary["ev"]["std"], summary["traj"]["std"], summary["gtraj"]["std"]]
    colors = ["#555555", "#aaaaaa", "#c0392b"]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), gridspec_kw={"wspace": 0.34})

    x = np.arange(len(rep["yt"]))
    axes[0].plot(x, rep["yt"], color="#000000", lw=1.8, label="True", zorder=4)
    axes[0].plot(x, rep["yp_ev"], color=colors[0], lw=1.2, ls="--", alpha=0.9, label="EV", zorder=3)
    axes[0].plot(x, rep["yp_traj"], color=colors[1], lw=0.9, alpha=0.55, label="Traj", zorder=1)
    axes[0].plot(x, rep["yp_gtraj"], color=colors[2], lw=1.2, alpha=0.95, label="G-Traj", zorder=2)
    axes[0].set_xlabel("Test time step")
    axes[0].set_ylabel("Mackey-Glass")
    axes[0].set_title(f"Best Hardware Seed (seed={rep['seed']})")
    axes[0].legend(
        frameon=True,
        fancybox=False,
        edgecolor="#999999",
        facecolor="white",
        framealpha=0.95,
        loc="upper right",
        handlelength=1.5,
        borderpad=0.4,
        labelspacing=0.25,
    )

    xpos = np.arange(len(methods))
    for i, (m, s, c) in enumerate(zip(means, stds, colors)):
        axes[1].bar(
            i,
            m,
            yerr=s,
            width=0.58,
            color=c,
            edgecolor="#333333",
            linewidth=0.6,
            capsize=2.5,
            error_kw={"linewidth": 0.7, "capthick": 0.7},
        )
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel("NRMSE (mean ± std)")
    axes[1].set_title("Hardware Results")

    k_mode = Counter(int(r["best_k"]) for r in results).most_common(1)[0][0]
    gap = summary["ev"]["mean"] - summary["gtraj"]["mean"]
    axes[1].text(
        0.03,
        0.97,
        f"Mean gap EV-GTraj: {gap:+.3f}\nMode best_k: {k_mode}",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#aaaaaa", "alpha": 0.95, "pad": 2.5},
    )

    fig.suptitle(title, y=1.02, fontsize=10, fontweight="medium")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run best Mackey-Glass config on IBM hardware with SamplerV2 shot memory."
    )
    parser.add_argument("--backend", type=str, default="ibm_marrakesh")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--S", type=int, default=20000)
    parser.add_argument("--n-shots", type=int, default=40)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--q", type=int, default=6)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--L", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--t0", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--lambda", dest="ridge_lambda", type=float, default=10.0)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--outdir", type=str, default=OUT_DIR)
    parser.add_argument("--outfile", type=str, default="fig_best_mg_ibm_marrakesh.pdf")
    parser.add_argument("--tablefile", type=str, default="table_best_mg_ibm_marrakesh.csv")
    args = parser.parse_args()

    if args.chunk_size > 300:
        raise ValueError("chunk-size must be <= 300 for ibm_marrakesh max_experiments limit.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    service = QiskitRuntimeService()
    results: List[Dict] = []

    for seed in range(int(args.seeds)):
        print(f"\n=== Hardware seed {seed + 1}/{args.seeds} on {args.backend} ===")
        r = _run_single_seed_hardware(
            service=service,
            backend_name=str(args.backend),
            seed=seed,
            S=int(args.S),
            n_shots=int(args.n_shots),
            sigma=float(args.sigma),
            q=int(args.q),
            depth=int(args.depth),
            L=int(args.L),
            eta=float(args.eta),
            t0=int(args.t0),
            alpha=float(args.alpha),
            ridge_lambda=float(args.ridge_lambda),
            chunk_size=int(args.chunk_size),
        )
        results.append(r)
        print(
            f"[seed {seed}] EV={r['ev']:.3f}, Traj={r['traj']:.3f}, "
            f"G-Traj={r['gtraj']:.3f}, best_k={r['best_k']}"
        )

    summary = _summarize(results)
    _save_table(results, outdir / args.tablefile)

    title = (
        "IBM Hardware Best MG Config "
        f"({args.backend}, Q={args.q}, depth={args.depth}, L={args.L}, "
        f"S={args.S}, N={args.n_shots}, sigma={args.sigma}, lambda={args.ridge_lambda})"
    )
    _make_plot(
        results=results,
        summary=summary,
        outpath=outdir / args.outfile,
        title=title,
    )

    print("\n=== Hardware Summary (NRMSE) ===")
    print(f"EV     : {summary['ev']['mean']:.3f} +/- {summary['ev']['std']:.3f}")
    print(f"Traj   : {summary['traj']['mean']:.3f} +/- {summary['traj']['std']:.3f}")
    print(f"G-Traj : {summary['gtraj']['mean']:.3f} +/- {summary['gtraj']['std']:.3f}")
    print(f"Gap EV-GTraj: {summary['ev']['mean'] - summary['gtraj']['mean']:+.3f}")
    print(f"Wrote: {outdir / args.outfile}")
    print(f"Wrote: {(outdir / args.outfile).with_suffix('.png')}")
    print(f"Wrote: {outdir / args.tablefile}")


if __name__ == "__main__":
    main()
