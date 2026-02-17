#!/usr/bin/env python3
"""Precompute and persist plot training-data caches for fast style iteration."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _run(cmd: list[str], *, env: dict | None = None) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def _ensure_file(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected {desc} missing: {path}")


def ensure_benchmark_cache(force: bool) -> Path:
    cache = RESULTS / "data_benchmarks_narma_fix.pkl"
    if cache.exists() and not force:
        print(f"[ok] benchmark cache exists: {cache}")
        return cache

    _run(
        [
            sys.executable,
            "plot_paper.py",
            "--cache-file",
            str(cache),
            "--output",
            "/tmp/_cache_warm_benchmarks_ieee.pdf",
        ]
    )
    _ensure_file(cache, "benchmark cache")
    return cache


def ensure_collapse_cache(
    *,
    seeds: int,
    force: bool,
    narma_fix: bool,
    shots_list: str | None,
    sigma_list: str | None,
) -> Path:
    cache_name = (
        "data_collapse_narma_fix_ieee.pkl"
        if narma_fix
        else "data_collapse_ieee.pkl"
    )
    cache = RESULTS / cache_name
    if cache.exists() and not force:
        print(f"[ok] collapse cache exists: {cache}")
        return cache

    out_pdf = (
        "/tmp/_cache_warm_collapse_narma_fix_ieee.pdf"
        if narma_fix
        else "/tmp/_cache_warm_collapse_ieee.pdf"
    )
    cmd = [
        sys.executable,
        "plot_collapse.py",
        "--seeds",
        str(seeds),
        "--cache-file",
        str(cache),
        "--output",
        out_pdf,
    ]
    if narma_fix:
        cmd.append("--narma-fix")
    if shots_list:
        cmd.extend(["--shots-list", shots_list])
    if sigma_list:
        cmd.extend(["--sigma-list", sigma_list])

    _run(cmd)
    _ensure_file(cache, "collapse cache")
    return cache


def ensure_phase_cache(
    *,
    seeds: int,
    force: bool,
    narma_fix: bool,
    shots_list: str | None,
    sigma_list: str | None,
) -> Path:
    cache_name = "data_phase_narma_fix_ieee.pkl" if narma_fix else "data_phase_ieee.pkl"
    cache = RESULTS / cache_name
    if cache.exists() and not force:
        print(f"[ok] phase cache exists: {cache}")
        return cache

    out_pdf = (
        "/tmp/_cache_warm_phase_narma_fix_ieee.pdf"
        if narma_fix
        else "/tmp/_cache_warm_phase_ieee.pdf"
    )
    cmd = [
        sys.executable,
        "plot_phase.py",
        "--seeds",
        str(seeds),
        "--cache-file",
        str(cache),
        "--output",
        out_pdf,
    ]
    if narma_fix:
        cmd.append("--narma-fix")
    if shots_list:
        cmd.extend(["--shots-list", shots_list])
    if sigma_list:
        cmd.extend(["--sigma-list", sigma_list])

    _run(cmd)
    _ensure_file(cache, "phase cache")
    return cache


def ensure_robustness_table(force: bool) -> Path:
    table = RESULTS / "table_gtraj_robustness_oneplot.csv"
    if table.exists() and not force:
        print(f"[ok] robustness table exists: {table}")
        return table

    pyc_path = ROOT / "__pycache__" / "plot_gtraj_robust_oneplot.cpython-313.pyc"
    _ensure_file(pyc_path, "robustness pyc fallback")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    _run(
        [
            sys.executable,
            str(pyc_path),
            "--outdir",
            "results",
            "--outfile",
            "fig_gtraj_robustness_oneplot_recomputed.pdf",
            "--tablefile",
            table.name,
        ],
        env=env,
    )
    _ensure_file(table, "robustness table")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute and save all plotting data caches for fast iterative styling."
    )
    parser.add_argument("--force", action="store_true", help="Recompute caches even if present.")
    parser.add_argument(
        "--collapse-seeds",
        type=int,
        default=1,
        help="Seed count for collapse cache generation (use 5 for full default fidelity).",
    )
    parser.add_argument(
        "--phase-seeds",
        type=int,
        default=1,
        help="Seed count for phase cache generation (use 5 for full default fidelity).",
    )
    parser.add_argument(
        "--include-narma-fix-caches",
        action="store_true",
        help="Also precompute narma-fix collapse/phase caches.",
    )
    parser.add_argument("--collapse-shots-list", type=str, default=None)
    parser.add_argument("--collapse-sigma-list", type=str, default=None)
    parser.add_argument("--phase-shots-list", type=str, default=None)
    parser.add_argument("--phase-sigma-list", type=str, default=None)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    paths.append(ensure_benchmark_cache(force=bool(args.force)))
    paths.append(
        ensure_collapse_cache(
            seeds=int(args.collapse_seeds),
            force=bool(args.force),
            narma_fix=False,
            shots_list=args.collapse_shots_list,
            sigma_list=args.collapse_sigma_list,
        )
    )
    paths.append(
        ensure_phase_cache(
            seeds=int(args.phase_seeds),
            force=bool(args.force),
            narma_fix=False,
            shots_list=args.phase_shots_list,
            sigma_list=args.phase_sigma_list,
        )
    )
    if args.include_narma_fix_caches:
        paths.append(
            ensure_collapse_cache(
                seeds=int(args.collapse_seeds),
                force=bool(args.force),
                narma_fix=True,
                shots_list=args.collapse_shots_list,
                sigma_list=args.collapse_sigma_list,
            )
        )
        paths.append(
            ensure_phase_cache(
                seeds=int(args.phase_seeds),
                force=bool(args.force),
                narma_fix=True,
                shots_list=args.phase_shots_list,
                sigma_list=args.phase_sigma_list,
            )
        )

    paths.append(ensure_robustness_table(force=bool(args.force)))

    print("\nSaved/available plot-data caches:")
    for p in paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()
