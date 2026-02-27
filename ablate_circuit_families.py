"""Ablation study: collapse robustness across reservoir circuit families."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Iterable

import numpy as np
from scipy.optimize import curve_fit

from plot_collapse import compute_data
from qrc_budget import OUT_DIR, QuantumReservoir


def _parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if not raw:
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str | None, default: list[float]) -> list[float]:
    if not raw:
        return default
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_families(raw: str | None) -> list[str]:
    if not raw:
        return list(QuantumReservoir.ALLOWED_CIRCUIT_FAMILIES)
    families = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [f for f in families if f not in QuantumReservoir.ALLOWED_CIRCUIT_FAMILIES]
    if bad:
        raise ValueError(
            f"Unknown family entries: {bad}. "
            f"Allowed: {QuantumReservoir.ALLOWED_CIRCUIT_FAMILIES}."
        )
    return families


def _fit_power_law(rho: np.ndarray, gap: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(rho) & np.isfinite(gap) & (gap > 0.005) & (rho > 0)
    if int(np.sum(mask)) < 4:
        return float("nan"), float("nan")

    def power_law(x, c, beta):
        return c * x ** (-beta)

    try:
        popt, _ = curve_fit(power_law, rho[mask], gap[mask], p0=[0.2, 0.8], maxfev=5000)
    except Exception:
        return float("nan"), float("nan")
    return float(popt[0]), float(popt[1])


def _summarize_slice(family: str, dataset: str, rho: np.ndarray, gap: np.ndarray) -> dict[str, float | str | int]:
    c_fit, beta_fit = _fit_power_law(rho, gap)
    return {
        "family": family,
        "dataset": dataset,
        "n_points": int(gap.size),
        "mean_gap": float(np.mean(gap)) if gap.size else float("nan"),
        "median_gap": float(np.median(gap)) if gap.size else float("nan"),
        "positive_fraction": float(np.mean(gap > 0.0)) if gap.size else float("nan"),
        "fit_C": c_fit,
        "fit_beta": beta_fit,
    }


def run_family_ablation(
    family: str,
    shots_list: Iterable[int],
    sigma_list: Iterable[float],
    n_seeds: int,
    use_narma_fix: bool,
) -> list[dict[str, float | str | int]]:
    data = compute_data(
        shots_list=list(shots_list),
        sigma_list=list(sigma_list),
        n_seeds=int(n_seeds),
        use_narma_fix=bool(use_narma_fix),
        circuit_family=family,
    )
    all_rho = np.asarray(data["all_rho"])
    all_gap = np.asarray(data["all_gap"])
    all_label = np.asarray(data["all_label"])

    rows = [_summarize_slice(family, "ALL", all_rho, all_gap)]
    for dataset in ("Mackey-Glass", "Lorenz", "NARMA-10"):
        mask = all_label == dataset
        rows.append(_summarize_slice(family, dataset, all_rho[mask], all_gap[mask]))
    return rows


def write_csv(rows: list[dict[str, float | str | int]], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = [
        "family",
        "dataset",
        "n_points",
        "mean_gap",
        "median_gap",
        "positive_fraction",
        "fit_C",
        "fit_beta",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run circuit-family ablation for rho_EV collapse.")
    parser.add_argument("--families", type=str, default=None, help="Comma-separated family list.")
    parser.add_argument("--narma-fix", action="store_true", help="Use NARMA-10 grouped fix.")
    parser.add_argument("--seeds", type=int, default=3, help="Random seeds per grid point.")
    parser.add_argument(
        "--shots-list",
        type=str,
        default="8,15,25,40",
        help='Comma-separated shot counts, e.g. "8,15,25,40".',
    )
    parser.add_argument(
        "--sigma-list",
        type=str,
        default="0.0,0.02,0.05,0.08",
        help='Comma-separated sigmas, e.g. "0.0,0.02,0.05,0.08".',
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=os.path.join(OUT_DIR, "table_circuit_family_ablation.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    families = _parse_families(args.families)
    shots_list = _parse_int_list(args.shots_list, [8, 15, 25, 40])
    sigma_list = _parse_float_list(args.sigma_list, [0.0, 0.02, 0.05, 0.08])

    all_rows: list[dict[str, float | str | int]] = []
    for family in families:
        print(f"Running family={family} | seeds={args.seeds} | narma_fix={bool(args.narma_fix)}")
        all_rows.extend(
            run_family_ablation(
                family=family,
                shots_list=shots_list,
                sigma_list=sigma_list,
                n_seeds=int(args.seeds),
                use_narma_fix=bool(args.narma_fix),
            )
        )

    write_csv(all_rows, args.out_csv)
    print(f"Saved {args.out_csv}")

    print("\nTop-level summary (dataset=ALL):")
    for row in all_rows:
        if row["dataset"] == "ALL":
            print(
                f"  {row['family']}: mean_gap={row['mean_gap']:.4f}, "
                f"positive_fraction={row['positive_fraction']:.3f}, "
                f"fit_beta={row['fit_beta']:.3f}"
            )


if __name__ == "__main__":
    main()
