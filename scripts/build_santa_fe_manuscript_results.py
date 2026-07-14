#!/usr/bin/env python3
"""Build the manuscript-facing four-task tables with Santa Fe laser.

This is an evidence-preserving merge, not a statistical recomputation.  It
keeps the completed Mackey--Glass, Roessler, and sunspot rows from the frozen
robust campaign and substitutes the completed Santa Fe laser screen for the
former fourth task.  The output manifest records the post-hoc substitution so
that the merged directory cannot be mistaken for a preregistered campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT / "results" / "qai_robust_protocol_full"
SANTA_FE = ROOT / "results" / "qai_santa_fe_laser_screen"
OUTPUT = ROOT / "results" / "qai_robust_protocol_santa_fe_manuscript"
TASKS = ("mackey_glass", "santa_fe_laser", "rossler", "sunspots")
CSV_FILES = (
    "replicate_results.csv",
    "cluster_summary.csv",
    "cluster_paired_stats.csv",
    "selected_k_frequency.csv",
    "partition_robustness_summary.csv",
    "fixed_lane_sensitivity.csv",
    "ideal_vs_classical.csv",
    "split_validation_spectrum.csv",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_csv(name: str) -> dict[str, object]:
    original = pd.read_csv(ORIGINAL / name)
    santa = pd.read_csv(SANTA_FE / name)
    original = original.loc[original["dataset"].isin(("mackey_glass", "rossler", "sunspots"))]

    # Santa Fe uses rolling windows, so trajectory- and month-specific columns
    # present in the original multi-task campaign are intentionally not set.
    for column in original.columns:
        if column not in santa.columns:
            santa[column] = pd.NA
    for column in santa.columns:
        if column not in original.columns:
            original[column] = pd.NA
    santa = santa.loc[:, original.columns]

    merged = pd.concat([original, santa], ignore_index=True)
    if set(merged["dataset"].unique()) != set(TASKS):
        raise RuntimeError(f"{name}: unexpected task set")
    if merged.duplicated().any():
        raise RuntimeError(f"{name}: duplicate rows after merge")
    output = OUTPUT / name
    merged.to_csv(output, index=False)
    return {"rows": int(len(merged)), "sha256": sha256(output)}


def build() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = {name: merge_csv(name) for name in CSV_FILES}

    original_manifest = json.loads((ORIGINAL / "manifest.json").read_text(encoding="utf-8"))
    santa_manifest = json.loads((SANTA_FE / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "qai-manuscript-merge-v1",
        "status": "complete-exploratory-posthoc-santa-fe-substitution",
        "tasks": list(TASKS),
        "clusters_per_task": 5,
        "planned_worker_jobs_equivalent": 200,
        "selection_disclosure": (
            "Santa Fe laser was screened after the original task outcomes were known; "
            "its intervals are exploratory and are not a preregistered confirmation."
        ),
        "source_results": {
            "original_three_tasks": {
                "path": str(ORIGINAL.relative_to(ROOT)),
                "manifest_sha256": sha256(ORIGINAL / "manifest.json"),
                "tasks": ["mackey_glass", "rossler", "sunspots"],
            },
            "santa_fe_laser": {
                "path": str(SANTA_FE.relative_to(ROOT)),
                "manifest_sha256": sha256(SANTA_FE / "manifest.json"),
                "task": "santa_fe_laser",
            },
        },
        "test_access": original_manifest.get("test_access"),
        "nested_design": original_manifest.get("nested_design"),
        "santa_fe_data_provenance": santa_manifest.get("data_provenance", {}).get(
            "santa_fe_laser"
        ),
        "outputs": outputs,
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    for name, metadata in outputs.items():
        print(f"wrote {OUTPUT / name} ({metadata['rows']} rows)")


def validate() -> None:
    """Validate the frozen merge without modifying any evidence file."""

    manifest_path = OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete-exploratory-posthoc-santa-fe-substitution":
        raise RuntimeError("unexpected merged-evidence status")
    if tuple(manifest.get("tasks", ())) != TASKS:
        raise RuntimeError("unexpected merged-evidence task order")
    sources = manifest.get("source_results", {})
    expected_sources = {
        "original_three_tasks": ORIGINAL / "manifest.json",
        "santa_fe_laser": SANTA_FE / "manifest.json",
    }
    for key, path in expected_sources.items():
        expected = sources.get(key, {}).get("manifest_sha256")
        if expected != sha256(path):
            raise RuntimeError(f"{key}: source manifest hash mismatch")
    outputs = manifest.get("outputs", {})
    for name in CSV_FILES:
        path = OUTPUT / name
        metadata = outputs.get(name, {})
        if metadata.get("sha256") != sha256(path):
            raise RuntimeError(f"{name}: SHA-256 mismatch")
        if int(metadata.get("rows", -1)) != len(pd.read_csv(path)):
            raise RuntimeError(f"{name}: row-count mismatch")
    print(f"validated {manifest_path} and {len(CSV_FILES)} frozen tables")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check the frozen merged evidence without rewriting it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        validate()
    else:
        build()


if __name__ == "__main__":
    main()
