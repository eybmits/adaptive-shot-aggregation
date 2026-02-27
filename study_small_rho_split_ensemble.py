"""Operating-point architecture study for the small-rho split-ensemble claim."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter

import numpy as np

from qrc_budget import OUT_DIR, compute_rho_ev, run_single_experiment


OPERATING_POINT_CONFIGS = [
    {
        "circuit": "Ring CNOT",
        "depth": 5,
        "entangler": "Ring",
        "family": "ring_cnot",
    },
    {
        "circuit": "Linear CNOT",
        "depth": 5,
        "entangler": "Chain",
        "family": "line_cnot",
    },
    {
        "circuit": "All-to-all CZ",
        "depth": 5,
        "entangler": "All-to-all",
        "family": "all_to_all_cz",
    },
    {
        "circuit": "Ring CNOT",
        "depth": 2,
        "entangler": "Ring",
        "family": "ring_cnot",
    },
    {
        "circuit": "Ring CNOT",
        "depth": 10,
        "entangler": "Ring",
        "family": "ring_cnot",
    },
]


def _mode_int(values: list[int]) -> int:
    counts = Counter(values)
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return int(best)


def _evaluate_config(
    config: dict[str, str | int],
    seeds: int,
    n_shots: int,
    sigma: float,
    S: int,
    t0: int,
    alpha: float,
    L: int,
    eta: float,
) -> dict[str, float | int | str]:
    ev_scores: list[float] = []
    split_scores: list[float] = []
    deltas: list[float] = []
    best_ks: list[int] = []

    for seed in range(seeds):
        result = run_single_experiment(
            n_shots=n_shots,
            sigma=sigma,
            S=S,
            benchmark="mackey_glass",
            seed=seed,
            eta=eta,
            L=L,
            t0=t0,
            alpha=alpha,
            depth=int(config["depth"]),
            circuit_family=str(config["family"]),
        )
        ev_scores.append(float(result["nrmse_ev"]))
        split_scores.append(float(result["nrmse_gtraj"]))
        deltas.append(float(result["gap_g"]))
        best_ks.append(int(result["best_k"]))

    rho_ev = compute_rho_ev(S=S, n_shots=n_shots, d=L * 12 + 1, t0=t0, alpha=alpha)
    return {
        "Circuit": str(config["circuit"]),
        "Depth": int(config["depth"]),
        "Entangler": str(config["entangler"]),
        "rho_EV": float(rho_ev),
        "NRMSE_EV": float(np.mean(ev_scores)),
        "NRMSE_Split": float(np.mean(split_scores)),
        "Delta": float(np.mean(deltas)),
        "Delta_std": float(np.std(deltas)),
        "Delta_pos_frac": float(np.mean(np.array(deltas) > 0.0)),
        "K_mode": int(_mode_int(best_ks)),
        "K_mean": float(np.mean(best_ks)),
        "seeds": int(seeds),
        "S": int(S),
        "n_shots": int(n_shots),
        "sigma": float(sigma),
    }


def _write_csv(rows: list[dict[str, float | int | str]], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fields = [
        "Circuit",
        "Depth",
        "Entangler",
        "rho_EV",
        "NRMSE_EV",
        "NRMSE_Split",
        "Delta",
        "Delta_std",
        "Delta_pos_frac",
        "K_mode",
        "K_mean",
        "seeds",
        "S",
        "n_shots",
        "sigma",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _print_markdown_table(rows: list[dict[str, float | int | str]]) -> None:
    print("| Circuit | Depth | Entangler | rho_EV | NRMSE_EV | NRMSE_Split | Delta | K_mode |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['Circuit']} | {r['Depth']} | {r['Entangler']} | "
            f"{r['rho_EV']:.3f} | {r['NRMSE_EV']:.4f} | {r['NRMSE_Split']:.4f} | "
            f"{r['Delta']:+.4f} | {r['K_mode']} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Small-rho split-ensemble operating-point study.")
    parser.add_argument("--seeds", type=int, default=8, help="Number of random seeds.")
    parser.add_argument("--S", type=int, default=20000, help="Total budget.")
    parser.add_argument("--n-shots", type=int, default=40, help="Shots per time step.")
    parser.add_argument("--sigma", type=float, default=0.02, help="Noise level.")
    parser.add_argument("--t0", type=int, default=30, help="Washout time.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Train/test split.")
    parser.add_argument("--L", type=int, default=20, help="Number of delay taps.")
    parser.add_argument("--eta", type=float, default=0.5, help="Input scaling.")
    parser.add_argument(
        "--out-csv",
        type=str,
        default=os.path.join(OUT_DIR, "table_small_rho_architecture_claim.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    for cfg in OPERATING_POINT_CONFIGS:
        print(f"Running {cfg['circuit']} depth={cfg['depth']} family={cfg['family']}")
        rows.append(
            _evaluate_config(
                config=cfg,
                seeds=int(args.seeds),
                n_shots=int(args.n_shots),
                sigma=float(args.sigma),
                S=int(args.S),
                t0=int(args.t0),
                alpha=float(args.alpha),
                L=int(args.L),
                eta=float(args.eta),
            )
        )

    _write_csv(rows, args.out_csv)
    print(f"Saved {args.out_csv}")
    print("\nOperating-point table:")
    _print_markdown_table(rows)

    all_positive = all(float(r["Delta"]) > 0.0 for r in rows)
    print("\nClaim check (rho_EV < 3 => Delta > 0 across listed architectures):")
    print(f"  rho_EV values: {[round(float(r['rho_EV']), 3) for r in rows]}")
    print(f"  all Delta > 0: {all_positive}")


if __name__ == "__main__":
    main()
