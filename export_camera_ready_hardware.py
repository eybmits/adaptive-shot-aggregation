#!/usr/bin/env python3
"""Create a camera-ready export package for hardware Mackey-Glass figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path
from statistics import mean, pstdev


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_seed(path: Path) -> int:
    # expected: fig_best_mg_ibm_marrakesh_seed_7.pdf
    stem = path.stem
    return int(stem.split("_")[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export camera-ready hardware artifacts with fixed filenames."
    )
    parser.add_argument(
        "--src-main-pdf",
        type=str,
        default="results/fig_best_mg_ibm_marrakesh_benchmark_style.pdf",
    )
    parser.add_argument(
        "--src-main-png",
        type=str,
        default="results/fig_best_mg_ibm_marrakesh_benchmark_style.png",
    )
    parser.add_argument(
        "--src-table",
        type=str,
        default="results/table_best_mg_ibm_marrakesh_reconstructed.csv",
    )
    parser.add_argument(
        "--src-individual-dir",
        type=str,
        default="results/hardware_individual",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/camera_ready",
    )
    parser.add_argument("--make-zip", action="store_true")
    args = parser.parse_args()

    src_main_pdf = Path(args.src_main_pdf)
    src_main_png = Path(args.src_main_png)
    src_table = Path(args.src_table)
    src_individual_dir = Path(args.src_individual_dir)
    outdir = Path(args.outdir)
    supp_dir = outdir / "supplementary"

    for p in (src_main_pdf, src_main_png, src_table):
        if not p.exists():
            raise FileNotFoundError(f"Missing required source file: {p}")
    if not src_individual_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {src_individual_dir}")

    outdir.mkdir(parents=True, exist_ok=True)
    supp_dir.mkdir(parents=True, exist_ok=True)

    # Fixed camera-ready names
    dst_main_pdf = outdir / "fig_hardware_mg_main.pdf"
    dst_main_png = outdir / "fig_hardware_mg_main.png"
    dst_table = outdir / "table_hardware_mg_nrmse.csv"

    shutil.copy2(src_main_pdf, dst_main_pdf)
    shutil.copy2(src_main_png, dst_main_png)
    shutil.copy2(src_table, dst_table)

    # Copy individual seed figures with fixed names
    seed_pdf_src = sorted(src_individual_dir.glob("fig_best_mg_ibm_marrakesh_seed_*.pdf"), key=parse_seed)
    seed_png_src = sorted(src_individual_dir.glob("fig_best_mg_ibm_marrakesh_seed_*.png"), key=parse_seed)
    if not seed_pdf_src or not seed_png_src:
        raise FileNotFoundError("No individual seed figures found.")

    copied_files: list[Path] = [dst_main_pdf, dst_main_png, dst_table]

    for src in seed_pdf_src:
        seed = parse_seed(src)
        dst = supp_dir / f"fig_hardware_mg_seed_{seed:02d}.pdf"
        shutil.copy2(src, dst)
        copied_files.append(dst)

    for src in seed_png_src:
        seed = parse_seed(src)
        dst = supp_dir / f"fig_hardware_mg_seed_{seed:02d}.png"
        shutil.copy2(src, dst)
        copied_files.append(dst)

    # Compute summary from table
    rows = list(csv.DictReader(dst_table.open()))
    ev = [float(r["nrmse_ev"]) for r in rows]
    traj = [float(r["nrmse_traj"]) for r in rows]
    gtraj = [float(r["nrmse_gtraj"]) for r in rows]
    best_k = [int(float(r["best_k"])) for r in rows]
    gap = [float(r["gap_ev_minus_gtraj"]) for r in rows]

    manifest = outdir / "MANIFEST.md"
    manifest.write_text(
        "\n".join(
            [
                "# Camera-Ready Hardware Export",
                "",
                "## Main Files",
                "- `fig_hardware_mg_main.pdf`",
                "- `fig_hardware_mg_main.png`",
                "- `table_hardware_mg_nrmse.csv`",
                "",
                "## Supplementary Files",
                "- `supplementary/fig_hardware_mg_seed_00..07.(pdf|png)`",
                "",
                "## Fixed Run Configuration",
                "- Backend: `ibm_marrakesh`",
                "- Dataset: `Mackey-Glass`",
                "- `Q=6`, `depth=5`, `L=20`",
                "- `S=20000`, `N_shots=40`, `sigma=0.03`",
                "- `t0=30`, `alpha=0.7`, `lambda=10`",
                "- Seeds: `0..7`",
                "",
                "## Hardware Results (from table)",
                f"- EV: `{mean(ev):.3f} ± {pstdev(ev):.3f}`",
                f"- Traj: `{mean(traj):.3f} ± {pstdev(traj):.3f}`",
                f"- G-Traj: `{mean(gtraj):.3f} ± {pstdev(gtraj):.3f}`",
                f"- Gap EV-GTraj: `{mean(gap):+.3f}`",
                f"- Mode best_k: `{max(set(best_k), key=best_k.count)}`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    copied_files.append(manifest)

    checksums = outdir / "SHA256SUMS.txt"
    with checksums.open("w", encoding="utf-8") as f:
        for p in sorted(copied_files):
            rel = p.relative_to(outdir)
            f.write(f"{sha256(p)}  {rel}\n")
    copied_files.append(checksums)

    zip_path = None
    if args.make_zip:
        zip_base = outdir.parent / "camera_ready_hardware_mg"
        zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=outdir)

    print(f"Wrote export dir: {outdir}")
    print(f"Wrote manifest: {manifest}")
    print(f"Wrote checksums: {checksums}")
    if zip_path:
        print(f"Wrote zip: {zip_path}")


if __name__ == "__main__":
    main()
