#!/usr/bin/env python3
"""Replot core + robustness figures in IEEE style."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def run(cmd: list[str], *, env: dict | None = None) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def ensure_robustness_table(table_path: Path, force_recompute: bool) -> None:
    if table_path.exists() and not force_recompute:
        print(f"[ok] using existing table: {table_path}")
        return

    pyc_path = ROOT / "__pycache__" / "plot_gtraj_robust_oneplot.cpython-313.pyc"
    if not pyc_path.exists():
        raise FileNotFoundError(
            f"Missing robustness source table and pyc fallback: {pyc_path}"
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    run(
        [
            sys.executable,
            str(pyc_path),
            "--outdir",
            "results",
            "--outfile",
            "fig_gtraj_robustness_oneplot_recomputed.pdf",
            "--tablefile",
            table_path.name,
        ],
        env=env,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replot core + robustness IEEE figures."
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also export PNG files alongside PDFs.",
    )
    parser.add_argument(
        "--force-recompute-robustness-table",
        action="store_true",
        help="Force recomputation of robustness table before plotting.",
    )
    parser.add_argument(
        "--collapse-seeds",
        type=int,
        default=1,
        help="Seed count for collapse replot (set 5 for full default sweep).",
    )
    parser.add_argument(
        "--phase-seeds",
        type=int,
        default=1,
        help="Seed count for phase replot (set 5 for full default sweep).",
    )
    parser.add_argument("--collapse-shots-list", type=str, default=None)
    parser.add_argument("--collapse-sigma-list", type=str, default=None)
    parser.add_argument("--phase-shots-list", type=str, default=None)
    parser.add_argument("--phase-sigma-list", type=str, default=None)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "plot_paper.py",
            "--output",
            "results/fig_benchmarks_ieee.pdf",
            *(["--png"] if args.png else []),
        ]
    )

    run(
        [
            sys.executable,
            "plot_collapse.py",
            "--output",
            "results/fig_collapse_ieee.pdf",
            "--seeds",
            str(args.collapse_seeds),
            *(
                ["--shots-list", args.collapse_shots_list]
                if args.collapse_shots_list
                else []
            ),
            *(
                ["--sigma-list", args.collapse_sigma_list]
                if args.collapse_sigma_list
                else []
            ),
            *(["--png"] if args.png else []),
        ]
    )

    run(
        [
            sys.executable,
            "plot_phase.py",
            "--output",
            "results/fig_phase_ieee.pdf",
            "--seeds",
            str(args.phase_seeds),
            *(
                ["--shots-list", args.phase_shots_list]
                if args.phase_shots_list
                else []
            ),
            *(
                ["--sigma-list", args.phase_sigma_list]
                if args.phase_sigma_list
                else []
            ),
            *(["--png"] if args.png else []),
        ]
    )

    table_path = RESULTS / "table_gtraj_robustness_oneplot.csv"
    ensure_robustness_table(
        table_path=table_path,
        force_recompute=bool(args.force_recompute_robustness_table),
    )

    run(
        [
            sys.executable,
            "plot_gtraj_robustness_styled.py",
            "--table",
            "results/table_gtraj_robustness_oneplot.csv",
            "--out-pdf",
            "results/fig_gtraj_robustness_oneplot_notitle_ieee.pdf",
            "--out-png",
            "results/fig_gtraj_robustness_oneplot_notitle_ieee.png",
        ]
    )

    print("\nWrote IEEE outputs:")
    print("- results/fig_benchmarks_ieee.pdf")
    print("- results/fig_collapse_ieee.pdf")
    print("- results/fig_phase_ieee.pdf")
    print("- results/fig_gtraj_robustness_oneplot_notitle_ieee.pdf")


if __name__ == "__main__":
    main()
