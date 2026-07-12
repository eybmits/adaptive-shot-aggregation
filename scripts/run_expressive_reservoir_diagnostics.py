#!/usr/bin/env python3
"""Run the frozen expressive reservoir's ideal-state validity gate.

This is intentionally a cheap pre-forecast check: no shot sampling, model
selection, or test targets are used.  The JSON ``passed`` field is consumed by
the robust forecasting runner before it starts a full experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from expressive_reservoir import (
    EXPRESSIVE_RESERVOIR_Q,
    EXPRESSIVE_RESERVOIR_SEED,
    diagnose_expressive_reservoir,
    ideal_feature_matrix,
    make_expressive_reservoir,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "expressive_reservoir" / "ideal_diagnostics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--q", type=int, default=EXPRESSIVE_RESERVOIR_Q)
    parser.add_argument("--seed", type=int, default=EXPRESSIVE_RESERVOIR_SEED)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument(
        "--feature-set",
        choices=("z0", "all_z", "z_plus_zz"),
        default="z_plus_zz",
    )
    parser.add_argument(
        "--feature-grid-csv",
        type=Path,
        default=None,
        help="Optional exact ideal feature curves for a paper diagnostic panel.",
    )
    return parser.parse_args()


def write_feature_grid(
    path: Path,
    *,
    q: int,
    seed: int,
    eta: float,
    window: int,
    feature_set: str,
) -> None:
    reservoir = make_expressive_reservoir(
        q=q,
        seed=seed,
        eta=eta,
        window=window,
        feature_set=feature_set,
    )
    input_grid = np.linspace(0.0, 1.0, 257)
    features = ideal_feature_matrix(reservoir, input_grid)
    if feature_set == "z0":
        names = ["Z0"]
    elif feature_set == "all_z":
        names = [f"Z{qubit}" for qubit in range(q)]
    else:
        names = [f"Z{qubit}" for qubit in range(q)] + [
            f"Z{qubit}Z{qubit + 1}" for qubit in range(q - 1)
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input", *names])
        writer.writerows(np.column_stack((input_grid, features)))


def main() -> int:
    args = parse_args()
    diagnostics = diagnose_expressive_reservoir(
        q=args.q,
        parameter_seed=args.seed,
        eta=args.eta,
        window=args.window,
        feature_set=args.feature_set,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    if args.feature_grid_csv is not None:
        write_feature_grid(
            args.feature_grid_csv,
            q=args.q,
            seed=args.seed,
            eta=args.eta,
            window=args.window,
            feature_set=args.feature_set,
        )

    summary = {
        "passed": diagnostics["passed"],
        "seed": diagnostics["architecture"]["parameter_seed"],
        "q": diagnostics["architecture"]["q"],
        "features": diagnostics["feature_dimension"],
        "dependent": diagnostics["input_dependent_features"],
        "nonlinear": diagnostics["nonlinear_features"],
        "numerical_rank": diagnostics["numerical_rank"],
        "effective_rank": round(float(diagnostics["effective_rank"]), 4),
        "second_layer_coupling_sensitive_features": diagnostics[
            "second_layer_coupling_sensitive_features"
        ],
        "architecture_sha256": diagnostics["architecture"]["architecture_sha256"],
        "classical_projection_error": round(
            float(diagnostics["classical_to_quantum_relative_projection_error"]), 4
        ),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))
    return 0 if diagnostics["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
