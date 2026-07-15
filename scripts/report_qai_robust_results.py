#!/usr/bin/env python3
"""Build paper-ready reports from the *completed* robust QAI campaign.

This module is intentionally strict.  It will not render smoke tests, running
campaigns, incomplete factorial designs, legacy reservoirs, or ablation rows as
headline evidence.  Every interval is recomputed after averaging shot,
reservoir, and partition seeds within the independent trajectory/window unit.

The group-size spectrum is a validation result.  It is read from
``split_validation_spectrum.csv`` and is never reconstructed from the held-out
test scores.  This keeps the final test block out of model-selection graphics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
REPORTER_PATH = Path(__file__).resolve()
DEFAULT_CONFIG = ROOT / "configs" / "qai_robust_protocol.json"
DEFAULT_RESULTS = ROOT / "results" / "qai_robust_protocol_full"
DEFAULT_DIAGNOSTICS = ROOT / "results" / "expressive_reservoir" / "ideal_diagnostics.json"
DEFAULT_FEATURE_GRID = ROOT / "results" / "expressive_reservoir" / "ideal_feature_grid.csv"
DEFAULT_PAPER_FIGURES = ROOT / "paper" / "figures"

PRIMARY_SCOPE = "headline"
PRIMARY_RESERVOIR = "expressive_reuploading"
PRIMARY_FEATURE_SET = "z_plus_zz"
REQUIRED_METHODS = (
    "classical_input",
    "ideal_quantum",
    "ev",
    "raw",
    "split",
    "covariance_ev",
)

DATASET_ORDER = ("mackey_glass", "lorenz", "rossler", "sunspots")
DATASET_LABEL = {
    "mackey_glass": "Mackey–Glass",
    "lorenz": "Lorenz",
    "rossler": "Rössler",
    "sunspots": "Sunspots",
}
BUDGET_ORDER = ("shot_rich", "balanced", "data_rich")
BUDGET_LABEL = {
    "shot_rich": "250 × 40",
    "balanced": "500 × 20",
    "data_rich": "1000 × 10",
}
METHOD_LABEL = {
    "classical_input": "Classical",
    "ideal_quantum": "Ideal QRC",
    "ev": "EV",
    "raw": "Raw",
    "split": "Split",
    "covariance_ev": "Regularized EV",
}

COLORS = {
    "mackey_glass": "#3D6FA8",
    "lorenz": "#1D9A86",
    "rossler": "#D9822B",
    "sunspots": "#94689B",
}
MARKERS = {
    "mackey_glass": "o",
    "lorenz": "D",
    "rossler": "s",
    "sunspots": "^",
}
INK = "#1A1A1A"
MID_GRAY = "#777777"
LIGHT_GRAY = "#D7D7D7"
ACCENT = "#D81B60"
PAGE_WIDTH = 7.16
COLUMN_WIDTH = 3.45


class ReportingContractError(RuntimeError):
    """Raised when results are unsafe to report as headline evidence."""


@dataclass(frozen=True)
class ReportInputs:
    results_dir: Path
    config_path: Path
    diagnostics_path: Path
    feature_grid_path: Path
    config: Mapping[str, object]
    manifest: Mapping[str, object]
    replicate: pd.DataFrame
    spectrum: pd.DataFrame
    diagnostics: Mapping[str, object]
    feature_grid: pd.DataFrame
    headline: pd.DataFrame


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 450,
        }
    )


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.13,
        1.04,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=INK,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_label(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ReportingContractError(f"{label} is missing columns: {sorted(missing)}")


def _t_interval(values: Sequence[float]) -> Tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(np.mean(data))
    if data.size == 1:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, data.size - 1) * stats.sem(data))
    return mean, mean - half, mean + half


def _manifest_datasets(manifest: Mapping[str, object]) -> Tuple[str, ...]:
    protocol = manifest.get("protocol", {})
    return tuple(str(value) for value in protocol.get("datasets", ()))


def _manifest_budgets(manifest: Mapping[str, object]) -> Tuple[str, ...]:
    return tuple(str(item["name"]) for item in manifest.get("budget_settings", ()))


def _resolve_repo_path(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _validate_live_manifest_contract(
    manifest: Mapping[str, object], config_path: Path
) -> Mapping[str, object]:
    """Bind a completed manifest to the live frozen code and input data.

    A row-complete CSV is not sufficient evidence if its configuration, runner
    implementation, or pinned real-data snapshot has changed since the
    campaign was executed.  The campaign manifest already records all three;
    the reporter checks them again immediately before reading any outcomes.
    """

    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise ReportingContractError(f"frozen live config is missing: {config_path}")
    manifest_config_path = _resolve_repo_path(manifest.get("config_path", ""))
    if manifest_config_path != config_path:
        raise ReportingContractError(
            f"manifest config_path resolves to {manifest_config_path}, expected {config_path}"
        )
    observed_config_hash = _sha256(config_path)
    if str(manifest.get("config_sha256", "")) != observed_config_hash:
        raise ReportingContractError("live config SHA-256 differs from campaign manifest")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    # Import lazily so plotting-only imports of this module remain lightweight.
    from run_robust_evaluation import implementation_sha256, verify_sunspot_dataset_pin

    observed_implementation_hash = implementation_sha256()
    if str(manifest.get("implementation_sha256", "")) != observed_implementation_hash:
        raise ReportingContractError("live implementation SHA-256 differs from campaign manifest")

    if manifest.get("decision_gate") != config.get("decision_gate"):
        raise ReportingContractError("manifest decision gate differs from the frozen live config")

    protocol = manifest.get("protocol", {})
    expected_protocol = {
        "datasets": list(config.get("datasets", ())),
        "primary_reservoir_variant": config.get("primary_reservoir_variant"),
        "primary_reservoir_seed": int(config.get("primary_reservoir_seed", -1)),
        "primary_feature_set": config.get("primary_feature_set"),
        "methods": list(config.get("methods", ())),
        "normalization": config.get("normalization"),
        "evaluation_scopes": config.get("evaluation_scopes"),
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ReportingContractError(f"manifest protocol field {key!r} differs from live config")

    expected_budgets = [
        {
            "name": str(item["name"]),
            "time_steps": int(item["time_steps"]),
            "shots_per_step": int(item["shots_per_step"]),
            "executions": int(item["time_steps"]) * int(item["shots_per_step"]),
        }
        for item in config.get("budget_settings", ())
    ]
    if manifest.get("budget_settings") != expected_budgets:
        raise ReportingContractError("manifest budget settings differ from live config")

    nested = manifest.get("nested_design", {})
    expected_nested = {
        "trajectory_indices": list(config.get("trajectory_indices", ())),
        "reservoir_seeds": list(config.get("reservoir_seeds", ())),
        "shot_seeds": list(config.get("shot_seeds", ())),
        "partition_seeds": list(config.get("partition_seeds", ())),
        "sunspot_origin_spacing_months": int(config.get("sunspot_origin_spacing_months", -1)),
        "partition_robustness_subset": config.get("partition_robustness_subset"),
        "fixed_lane_ablation": config.get("fixed_lane_ablation"),
    }
    for key, expected in expected_nested.items():
        if nested.get(key) != expected:
            raise ReportingContractError(f"manifest nested-design field {key!r} differs from live config")

    search = manifest.get("hyperparameter_search", {})
    expected_search = {
        "lambda_grid": list(config.get("lambda_grid", ())),
        "covariance_alpha_grid": list(config.get("covariance_alpha_grid", ())),
        "k_grid_by_setting": config.get("organization_k_grid_by_setting"),
    }
    for key, expected in expected_search.items():
        if search.get(key) != expected:
            raise ReportingContractError(f"manifest hyperparameter field {key!r} differs from live config")

    try:
        live_sunspots = verify_sunspot_dataset_pin(config)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ReportingContractError(f"live pinned data verification failed: {exc}") from exc
    manifest_sunspots = manifest.get("data_provenance", {}).get("sunspots")
    if manifest_sunspots != live_sunspots:
        raise ReportingContractError("live sunspot data SHA-256/provenance differs from manifest")
    return config


def _seed_value(value: object) -> int | None:
    return None if pd.isna(value) else int(value)


def _seed_tuples(frame: pd.DataFrame) -> set[Tuple[int | None, int | None, int | None]]:
    return {
        (_seed_value(row.reservoir_seed), _seed_value(row.shot_seed), _seed_value(row.partition_seed))
        for row in frame.itertuples(index=False)
    }


def _validate_headline_seed_and_partition_contract(
    headline: pd.DataFrame,
    *,
    datasets: Sequence[str],
    budgets: Sequence[str],
    clusters_per_task: int,
    reservoir_seeds: Sequence[int],
    shot_seeds: Sequence[int],
    partition_seeds: Sequence[int],
) -> None:
    """Require exact method-specific seed products and partition semantics."""

    finite_product = set(product(map(int, reservoir_seeds), map(int, shot_seeds), map(int, partition_seeds)))
    ideal_product = {(int(seed), None, None) for seed in reservoir_seeds}
    classical_product = {(None, None, None)}
    for dataset in datasets:
        for budget in budgets:
            budget_frame = headline[
                (headline["dataset"] == dataset) & (headline["budget_setting"] == budget)
            ]
            clusters = tuple(sorted(str(value) for value in budget_frame["cluster_id"].unique()))
            if len(clusters) != int(clusters_per_task):
                raise ReportingContractError(
                    f"unexpected cluster count for {dataset}/{budget}: {len(clusters)}"
                )
            for cluster in clusters:
                cell = budget_frame[budget_frame["cluster_id"].astype(str) == cluster]
                if set(cell["method"]) != set(REQUIRED_METHODS):
                    raise ReportingContractError(
                        f"headline method set differs for {dataset}/{budget}/{cluster}"
                    )
                for method in REQUIRED_METHODS:
                    rows = cell[cell["method"] == method]
                    if method == "classical_input":
                        expected_seeds = classical_product
                        expected_role, expected_mode = "not_applicable", "not_applicable"
                    elif method == "ideal_quantum":
                        expected_seeds = ideal_product
                        expected_role, expected_mode = "not_applicable", "not_applicable"
                    else:
                        expected_seeds = finite_product
                        expected_role, expected_mode = "primary", "per_time_permuted"
                    observed_seeds = _seed_tuples(rows)
                    if observed_seeds != expected_seeds or len(rows) != len(expected_seeds):
                        raise ReportingContractError(
                            "exact method-specific seed set differs for "
                            f"{dataset}/{budget}/{cluster}/{method}"
                        )
                    if set(rows["partition_role"].astype(str)) != {expected_role}:
                        raise ReportingContractError(
                            f"headline partition_role differs for {dataset}/{budget}/{cluster}/{method}"
                        )
                    if set(rows["partition_mode"].astype(str)) != {expected_mode}:
                        raise ReportingContractError(
                            f"headline partition_mode differs for {dataset}/{budget}/{cluster}/{method}"
                        )


def headline_filter(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the only rows permitted in headline reporting.

    Classical input has no reservoir/quantum feature-set identifiers.  Every
    other method must be the expressive reservoir with Z+ZZ observables.
    """

    _require_columns(
        frame,
        ("scope_role", "method", "reservoir_variant", "feature_set"),
        "replicate results",
    )
    scoped = frame[frame["scope_role"] == PRIMARY_SCOPE].copy()
    classical = scoped[
        (scoped["method"] == "classical_input")
        & (scoped["reservoir_variant"] == "not_applicable")
        & (scoped["feature_set"] == "classical_input")
    ]
    quantum = scoped[
        (scoped["method"] != "classical_input")
        & (scoped["reservoir_variant"] == PRIMARY_RESERVOIR)
        & (scoped["feature_set"] == PRIMARY_FEATURE_SET)
    ]
    selected = pd.concat([classical, quantum], ignore_index=True)
    if selected.empty:
        raise ReportingContractError("headline filter produced no rows")
    bad = selected[~selected["method"].isin(REQUIRED_METHODS)]
    if len(bad):
        raise ReportingContractError("headline frame contains an unregistered method")
    return selected


def spectrum_filter(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        frame,
        (
            "dataset",
            "budget_setting",
            "cluster_id",
            "reservoir_variant",
            "reservoir_seed",
            "shot_seed",
            "partition_seed",
            "partition_mode",
            "partition_role",
            "scope_role",
            "feature_set",
            "method",
            "k",
            "lambda_selected",
            "mean_val_nrmse",
            "metric_role",
        ),
        "split validation spectrum",
    )
    selected = frame[
        (frame["scope_role"] == PRIMARY_SCOPE)
        & (frame["reservoir_variant"] == PRIMARY_RESERVOIR)
        & (frame["feature_set"] == PRIMARY_FEATURE_SET)
        & (frame["partition_mode"] == "per_time_permuted")
        & (frame["partition_role"] == "primary")
        & (frame["method"] == "split")
        & (frame["metric_role"] == "validation_only_no_test_access")
    ].copy()
    if selected.empty:
        raise ReportingContractError("spectrum headline filter produced no rows")
    if "test_nrmse" in frame.columns:
        raise ReportingContractError("validation spectrum must not contain held-out test metrics")
    return selected


def validate_complete_inputs(
    results_dir: Path,
    diagnostics_path: Path = DEFAULT_DIAGNOSTICS,
    feature_grid_path: Path = DEFAULT_FEATURE_GRID,
    config_path: Path = DEFAULT_CONFIG,
) -> ReportInputs:
    """Read and validate a completed, full-factorial campaign."""

    results_dir = Path(results_dir)
    required_paths = {
        "manifest": results_dir / "manifest.json",
        "replicate": results_dir / "replicate_results.csv",
        "spectrum": results_dir / "split_validation_spectrum.csv",
        "diagnostics": Path(diagnostics_path),
        "feature_grid": Path(feature_grid_path),
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise ReportingContractError(f"completed reporting inputs are missing: {missing}")

    manifest = json.loads(required_paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ReportingContractError(
            f"refusing non-complete campaign with status={manifest.get('status')!r}"
        )
    if "smoke" in str(manifest.get("status", "")).lower():
        raise ReportingContractError("refusing smoke-test manifest")
    gate = manifest.get("feature_validity_gate", {})
    if gate.get("passed") is not True or gate.get("hash_match") is not True:
        raise ReportingContractError("feature-validity gate or frozen architecture hash failed")
    config = _validate_live_manifest_contract(manifest, Path(config_path))

    replicate = pd.read_csv(required_paths["replicate"])
    _require_columns(
        replicate,
        (
            "dataset",
            "budget_setting",
            "cluster_id",
            "reservoir_variant",
            "reservoir_seed",
            "shot_seed",
            "partition_seed",
            "partition_mode",
            "partition_role",
            "scope_role",
            "feature_set",
            "method",
            "k",
            "test_nrmse",
        ),
        "replicate results",
    )
    if int(manifest.get("rows", -1)) != len(replicate):
        raise ReportingContractError(
            f"manifest rows={manifest.get('rows')} but replicate_results has {len(replicate)}"
        )
    if any("smoke" in str(value).lower() for value in replicate["budget_setting"].unique()):
        raise ReportingContractError("smoke-only rows found in a purported full campaign")

    headline = headline_filter(replicate)
    raw_headline = replicate[replicate["scope_role"] == PRIMARY_SCOPE]
    if len(headline) != len(raw_headline):
        raise ReportingContractError(
            "headline scope contains rows outside the registered reservoir/feature contract"
        )
    datasets = tuple(str(value) for value in config["datasets"])
    budgets = tuple(str(item["name"]) for item in config["budget_settings"])
    if set(datasets) != set(DATASET_ORDER):
        raise ReportingContractError(f"unexpected dataset set: {datasets}")
    if set(budgets) != set(BUDGET_ORDER):
        raise ReportingContractError(f"unexpected budget set: {budgets}")

    expected_clusters = len(config["trajectory_indices"])
    if int(manifest.get("clusters_per_task", 0)) != expected_clusters:
        raise ReportingContractError("manifest cluster count differs from frozen config")
    if expected_clusters < 5:
        raise ReportingContractError("fewer than five independent clusters per task")
    coverage = (
        headline.groupby(["dataset", "budget_setting", "method"], dropna=False)["cluster_id"]
        .nunique()
        .reindex(
            pd.MultiIndex.from_product(
                [datasets, budgets, REQUIRED_METHODS],
                names=["dataset", "budget_setting", "method"],
            ),
            fill_value=0,
        )
    )
    bad_coverage = coverage[coverage != expected_clusters]
    if len(bad_coverage):
        raise ReportingContractError(
            "incomplete headline factorial design: " + str(bad_coverage.to_dict())
        )

    nested = manifest.get("nested_design", {})
    reservoir_seeds = tuple(int(value) for value in config["reservoir_seeds"])
    shot_seeds = tuple(int(value) for value in config["shot_seeds"])
    partition_seeds = tuple(int(value) for value in config["partition_seeds"])
    if not reservoir_seeds or not shot_seeds or not partition_seeds:
        raise ReportingContractError("manifest has an empty nested-factor seed set")
    _validate_headline_seed_and_partition_contract(
        headline,
        datasets=datasets,
        budgets=budgets,
        clusters_per_task=expected_clusters,
        reservoir_seeds=reservoir_seeds,
        shot_seeds=shot_seeds,
        partition_seeds=partition_seeds,
    )
    expected_rows = {
        "classical_input": 1,
        "ideal_quantum": len(reservoir_seeds),
        "ev": len(reservoir_seeds) * len(shot_seeds) * len(partition_seeds),
        "raw": len(reservoir_seeds) * len(shot_seeds) * len(partition_seeds),
        "split": len(reservoir_seeds) * len(shot_seeds) * len(partition_seeds),
        "covariance_ev": len(reservoir_seeds) * len(shot_seeds) * len(partition_seeds),
    }
    row_counts = headline.groupby(
        ["dataset", "budget_setting", "cluster_id", "method"], dropna=False
    ).size()
    for key, count in row_counts.items():
        method = str(key[-1])
        if int(count) != expected_rows[method]:
            raise ReportingContractError(
                f"nested design is incomplete for {key}: got {count}, expected {expected_rows[method]}"
            )

    spectrum_raw = pd.read_csv(required_paths["spectrum"])
    spectrum = spectrum_filter(spectrum_raw)
    raw_headline_spectrum = spectrum_raw[spectrum_raw["scope_role"] == PRIMARY_SCOPE]
    if len(spectrum) != len(raw_headline_spectrum):
        raise ReportingContractError(
            "headline validation spectrum contains non-primary or non-validation rows"
        )
    spectrum_coverage = (
        spectrum.groupby(["dataset", "budget_setting", "k"])["cluster_id"].nunique()
    )
    if len(spectrum_coverage) == 0 or (spectrum_coverage != expected_clusters).any():
        raise ReportingContractError("split validation spectrum has incomplete cluster coverage")
    if set(spectrum["dataset"]) != set(datasets) or set(spectrum["budget_setting"]) != set(budgets):
        raise ReportingContractError("spectrum does not cover every registered task and budget")
    registered_k = manifest.get("hyperparameter_search", {}).get("k_grid_by_setting", {})
    expected_spectrum_rows = len(reservoir_seeds) * len(shot_seeds) * len(partition_seeds)
    for dataset in datasets:
        for budget in budgets:
            cell = spectrum[
                (spectrum["dataset"] == dataset) & (spectrum["budget_setting"] == budget)
            ]
            observed_k = set(int(value) for value in cell["k"])
            expected_k = set(int(value) for value in registered_k.get(budget, ()))
            if observed_k != expected_k:
                raise ReportingContractError(
                    f"spectrum k grid differs for {dataset}/{budget}: {observed_k} != {expected_k}"
                )
            counts = cell.groupby(["cluster_id", "k"]).size()
            if (counts != expected_spectrum_rows).any():
                raise ReportingContractError(
                    f"spectrum nested-factor coverage is incomplete for {dataset}/{budget}"
                )
            expected_seed_product = set(product(reservoir_seeds, shot_seeds, partition_seeds))
            for (cluster_id, k), rows in cell.groupby(["cluster_id", "k"], dropna=False):
                if _seed_tuples(rows) != expected_seed_product:
                    raise ReportingContractError(
                        "spectrum exact seed set differs for "
                        f"{dataset}/{budget}/{cluster_id}/k={int(k)}"
                    )

    robustness = replicate[
        (replicate["scope_role"] == "partition_robustness")
        & (replicate["partition_role"] == "twenty_partition_robustness")
        & (replicate["reservoir_variant"] == PRIMARY_RESERVOIR)
        & (replicate["feature_set"] == PRIMARY_FEATURE_SET)
    ]
    expected_robust_partitions = tuple(
        int(value)
        for value in nested.get("partition_robustness_subset", {}).get("partition_seeds", ())
    )
    if len(expected_robust_partitions) < 20:
        raise ReportingContractError("manifest does not register twenty partition-robustness seeds")
    robust_counts = robustness.groupby(
        ["dataset", "budget_setting", "cluster_id", "method"], dropna=False
    )["partition_seed"].nunique()
    if robust_counts.empty or (robust_counts != len(expected_robust_partitions)).any():
        raise ReportingContractError("twenty-partition robustness factorial is incomplete")

    diagnostics = json.loads(required_paths["diagnostics"].read_text(encoding="utf-8"))
    if diagnostics.get("passed") is not True:
        raise ReportingContractError("ideal feature diagnostics did not pass")
    if diagnostics.get("architecture", {}).get("feature_set") != PRIMARY_FEATURE_SET:
        raise ReportingContractError("feature diagnostics are not for z_plus_zz")
    expected_arch = gate.get("observed_architecture_sha256")
    observed_arch = diagnostics.get("architecture", {}).get("architecture_sha256")
    if expected_arch != observed_arch:
        raise ReportingContractError("diagnostic architecture hash differs from campaign manifest")

    feature_grid = pd.read_csv(required_paths["feature_grid"])
    if "input" not in feature_grid or feature_grid.shape[1] - 1 != int(
        diagnostics.get("feature_dimension", -1)
    ):
        raise ReportingContractError("feature grid dimension disagrees with diagnostics")

    return ReportInputs(
        results_dir=results_dir,
        config_path=Path(config_path),
        diagnostics_path=required_paths["diagnostics"],
        feature_grid_path=required_paths["feature_grid"],
        config=config,
        manifest=manifest,
        replicate=replicate,
        spectrum=spectrum,
        diagnostics=diagnostics,
        feature_grid=feature_grid,
        headline=headline,
    )


def cluster_method_summary(headline: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Average nested factors first, then form t intervals over clusters."""

    cluster = (
        headline.groupby(
            ["dataset", "budget_setting", "cluster_id", "method"], dropna=False
        )["test_nrmse"]
        .mean()
        .reset_index(name="cluster_nrmse")
    )
    rows = []
    for keys, frame in cluster.groupby(["dataset", "budget_setting", "method"]):
        mean, low, high = _t_interval(frame["cluster_nrmse"])
        rows.append(
            {
                "dataset": keys[0],
                "budget_setting": keys[1],
                "method": keys[2],
                "clusters": int(frame["cluster_id"].nunique()),
                "mean_nrmse": mean,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return cluster, pd.DataFrame(rows)


def cluster_paired_gain(
    headline: pd.DataFrame, baseline: str, method: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cluster, _ = cluster_method_summary(headline)
    wide = cluster.pivot(
        index=["dataset", "budget_setting", "cluster_id"],
        columns="method",
        values="cluster_nrmse",
    )
    if baseline not in wide or method not in wide:
        raise ReportingContractError(f"missing paired methods {baseline}/{method}")
    paired = wide[[baseline, method]].dropna().reset_index()
    paired["gain"] = paired[baseline] - paired[method]
    rows = []
    for keys, frame in paired.groupby(["dataset", "budget_setting"]):
        mean, low, high = _t_interval(frame["gain"])
        rows.append(
            {
                "dataset": keys[0],
                "budget_setting": keys[1],
                "baseline": baseline,
                "method": method,
                "clusters": int(frame["cluster_id"].nunique()),
                "mean_gain": mean,
                "ci95_low": low,
                "ci95_high": high,
                "cluster_win_rate": float(np.mean(frame["gain"] > 0)),
            }
        )
    return paired, pd.DataFrame(rows)


def nested_paired_gain(
    headline: pd.DataFrame, baseline: str = "ev", method: str = "split"
) -> pd.DataFrame:
    """Descriptive nested points; never used for inferential intervals."""

    selected = headline[headline["method"].isin([baseline, method])].copy()
    keys = [
        "dataset",
        "budget_setting",
        "cluster_id",
        "reservoir_seed",
        "shot_seed",
        "partition_seed",
        "method",
    ]
    means = selected.groupby(keys, dropna=False)["test_nrmse"].mean().reset_index()
    index = keys[:-1]
    wide = means.pivot(index=index, columns="method", values="test_nrmse")
    if baseline not in wide or method not in wide:
        return pd.DataFrame(columns=index + ["gain"])
    output = wide[[baseline, method]].dropna().reset_index()
    output["gain"] = output[baseline] - output[method]
    return output


def validation_spectrum_summary(spectrum: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cluster = (
        spectrum.groupby(["dataset", "budget_setting", "cluster_id", "k"], dropna=False)[
            "mean_val_nrmse"
        ]
        .mean()
        .reset_index(name="cluster_val_nrmse")
    )
    rows = []
    for keys, frame in cluster.groupby(["dataset", "budget_setting", "k"]):
        mean, low, high = _t_interval(frame["cluster_val_nrmse"])
        rows.append(
            {
                "dataset": keys[0],
                "budget_setting": keys[1],
                "k": int(keys[2]),
                "clusters": int(frame["cluster_id"].nunique()),
                "mean_val_nrmse": mean,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return cluster, pd.DataFrame(rows)


def selected_k_frequency(
    headline: pd.DataFrame,
    k_grid_by_setting: Mapping[str, Sequence[int]] | None = None,
) -> pd.DataFrame:
    """Count validation-selected group sizes, retaining zero-count candidates.

    The selected rows alone do not contain a candidate that was evaluated but
    never chosen.  When the frozen registered grid is supplied, explicitly
    complete every dataset/budget cell with those zero-frequency candidates so
    the selection-frequency panel cannot make them look untested.
    """

    split = headline[
        (headline["method"] == "split")
        & (headline["partition_role"] == "primary")
        & (headline["partition_mode"] == "per_time_permuted")
    ].copy()
    keys = ["dataset", "budget_setting", "k"]
    counts = split.groupby(keys, dropna=False).size().reset_index(name="selections")
    if k_grid_by_setting is not None:
        registered_rows = []
        for dataset, budget_setting in (
            split[["dataset", "budget_setting"]].drop_duplicates().itertuples(index=False)
        ):
            if str(budget_setting) not in k_grid_by_setting:
                raise ReportingContractError(
                    f"registered k grid is missing budget setting {budget_setting!r}"
                )
            registered = tuple(
                int(value) for value in k_grid_by_setting[str(budget_setting)]
            )
            observed = set(
                int(value)
                for value in counts[
                    (counts["dataset"] == dataset)
                    & (counts["budget_setting"] == budget_setting)
                ]["k"]
            )
            if not observed.issubset(set(registered)):
                raise ReportingContractError(
                    f"selected k lies outside the registered grid for {dataset}/{budget_setting}"
                )
            registered_rows.extend(
                {
                    "dataset": dataset,
                    "budget_setting": budget_setting,
                    "k": value,
                }
                for value in registered
            )
        registered_frame = pd.DataFrame(registered_rows)
        counts = registered_frame.merge(counts, on=keys, how="left", validate="one_to_one")
        counts["selections"] = counts["selections"].fillna(0).astype(int)
    totals = counts.groupby(["dataset", "budget_setting"])["selections"].transform("sum")
    if (totals <= 0).any():
        raise ReportingContractError("selection-frequency cell has no Split selections")
    counts["selection_frequency"] = counts["selections"] / totals
    return counts


def partition_gain_summary(replicate: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected = replicate[
        (replicate["scope_role"] == "partition_robustness")
        & (replicate["reservoir_variant"] == PRIMARY_RESERVOIR)
        & (replicate["feature_set"] == PRIMARY_FEATURE_SET)
        & (replicate["partition_role"] == "twenty_partition_robustness")
        & (replicate["partition_mode"] == "per_time_permuted")
        & (replicate["method"].isin(["ev", "split"]))
    ].copy()
    if selected.empty:
        raise ReportingContractError("twenty-partition robustness rows are missing")
    keys = ["dataset", "budget_setting", "cluster_id", "partition_seed", "method"]
    means = selected.groupby(keys, dropna=False)["test_nrmse"].mean().reset_index()
    wide = means.pivot(index=keys[:-1], columns="method", values="test_nrmse")
    if "ev" not in wide or "split" not in wide:
        raise ReportingContractError("partition robustness lacks paired EV/Split rows")
    partition = wide[["ev", "split"]].dropna().reset_index()
    partition["gain"] = partition["ev"] - partition["split"]
    cluster = (
        partition.groupby(["dataset", "budget_setting", "cluster_id"])["gain"]
        .mean()
        .reset_index(name="cluster_gain")
    )
    rows = []
    for keys_value, frame in cluster.groupby(["dataset", "budget_setting"]):
        mean, low, high = _t_interval(frame["cluster_gain"])
        rows.append(
            {
                "dataset": keys_value[0],
                "budget_setting": keys_value[1],
                "clusters": int(frame["cluster_id"].nunique()),
                "partitions_per_cluster_min": int(
                    partition[
                        (partition["dataset"] == keys_value[0])
                        & (partition["budget_setting"] == keys_value[1])
                    ]
                    .groupby("cluster_id")["partition_seed"]
                    .nunique()
                    .min()
                ),
                "mean_gain": mean,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return partition, pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, generated: Path, paper_figures: Path, stem: str) -> None:
    generated.mkdir(parents=True, exist_ok=True)
    paper_figures.mkdir(parents=True, exist_ok=True)
    pdf = generated / f"{stem}.pdf"
    png = generated / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, dpi=450, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    shutil.copy2(pdf, paper_figures / pdf.name)
    shutil.copy2(png, paper_figures / png.name)


def plot_feature_validity_and_ideal(
    inputs: ReportInputs, generated: Path, paper_figures: Path
) -> None:
    grid = inputs.feature_grid
    feature_names = list(grid.columns[1:])
    values = grid[feature_names].to_numpy(dtype=float).T
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    standardized = values / scale

    paired, summary = cluster_paired_gain(inputs.headline, "classical_input", "ideal_quantum")
    paired = paired[paired["budget_setting"] == "balanced"]
    summary = summary[summary["budget_setting"] == "balanced"]

    fig = plt.figure(figsize=(PAGE_WIDTH, 2.15))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.27, 1.0), wspace=0.38)
    ax0 = fig.add_subplot(gs[0, 0])
    image = ax0.imshow(
        standardized,
        aspect="auto",
        origin="lower",
        extent=(float(grid["input"].min()), float(grid["input"].max()), -0.5, len(feature_names) - 0.5),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0),
        interpolation="nearest",
    )
    ax0.set_yticks(range(len(feature_names)), feature_names)
    ax0.set_xlabel("Normalized input")
    ax0.set_ylabel("Observable")
    ax0.set_title("Ideal features respond nonlinearly")
    ax0.spines["top"].set_visible(True)
    ax0.spines["right"].set_visible(True)
    cbar = fig.colorbar(image, ax=ax0, fraction=0.035, pad=0.025)
    cbar.set_label("Feature / max |feature|", labelpad=2)
    diag = inputs.diagnostics
    ax0.text(
        0.01,
        -0.28,
        (
            f"{diag['input_dependent_features']}/{diag['feature_dimension']} input-dependent; "
            f"{diag['nonlinear_features']} nonlinear; rank {diag['numerical_rank']}"
        ),
        transform=ax0.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color=MID_GRAY,
    )
    _panel_letter(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(DATASET_ORDER))[::-1]
    ax1.axvline(0, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=0)
    rng = np.random.default_rng(1729)
    for index, dataset in enumerate(DATASET_ORDER):
        yy = y[index]
        points = paired[paired["dataset"] == dataset]["gain"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.09, 0.09, size=len(points))
        ax1.scatter(
            points,
            yy + jitter,
            s=13,
            color=COLORS[dataset],
            alpha=0.23,
            edgecolors="none",
            zorder=1,
        )
        row = summary[summary["dataset"] == dataset].iloc[0]
        ax1.errorbar(
            row["mean_gain"],
            yy,
            xerr=[[row["mean_gain"] - row["ci95_low"]], [row["ci95_high"] - row["mean_gain"]]],
            fmt=MARKERS[dataset],
            ms=5.2,
            mfc=COLORS[dataset],
            mec="white",
            mew=0.7,
            color=COLORS[dataset],
            capsize=2.2,
            lw=1.4,
            zorder=3,
        )
    ax1.set_yticks(y, [DATASET_LABEL[d] for d in DATASET_ORDER])
    ax1.set_xlabel("Classical − ideal QRC NRMSE\n(positive favors ideal QRC)")
    ax1.set_title("Ideal reservoir vs matched classical")
    _panel_letter(ax1, "b")
    _save_figure(fig, generated, paper_figures, "qai_feature_validity_ideal_classical")


def plot_balanced_spectrum(inputs: ReportInputs, generated: Path, paper_figures: Path) -> None:
    spectrum_cluster, spectrum_summary = validation_spectrum_summary(inputs.spectrum)
    spectrum_cluster = spectrum_cluster[spectrum_cluster["budget_setting"] == "balanced"]
    spectrum_summary = spectrum_summary[spectrum_summary["budget_setting"] == "balanced"]
    frequency = selected_k_frequency(
        inputs.headline,
        inputs.config["organization_k_grid_by_setting"],
    )
    frequency = frequency[frequency["budget_setting"] == "balanced"]
    cluster_gain, gain_summary = cluster_paired_gain(inputs.headline, "ev", "split")
    cluster_gain = cluster_gain[cluster_gain["budget_setting"] == "balanced"]
    gain_summary = gain_summary[gain_summary["budget_setting"] == "balanced"]
    nested = nested_paired_gain(inputs.headline)
    nested = nested[nested["budget_setting"] == "balanced"]

    fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH, 5.35), constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.07)

    ax = axes[0]
    for dataset in DATASET_ORDER:
        raw = spectrum_cluster[spectrum_cluster["dataset"] == dataset]
        for _, frame in raw.groupby("cluster_id"):
            ax.plot(
                frame["k"],
                frame["cluster_val_nrmse"],
                color=COLORS[dataset],
                alpha=0.09,
                lw=0.6,
                zorder=1,
            )
        frame = spectrum_summary[spectrum_summary["dataset"] == dataset].sort_values("k")
        ax.plot(
            frame["k"],
            frame["mean_val_nrmse"],
            marker=MARKERS[dataset],
            ms=3.6,
            color=COLORS[dataset],
            lw=1.45,
            mfc=COLORS[dataset],
            mec="white",
            mew=0.5,
            label=DATASET_LABEL[dataset],
            zorder=3,
        )
        ax.fill_between(
            frame["k"].to_numpy(dtype=float),
            frame["ci95_low"].to_numpy(dtype=float),
            frame["ci95_high"].to_numpy(dtype=float),
            color=COLORS[dataset],
            alpha=0.09,
            linewidth=0,
            zorder=0,
        )
    ax.set_xscale("log", base=2)
    ticks = sorted(set(int(value) for value in spectrum_summary["k"]))
    ax.set_xticks(ticks, [str(value) for value in ticks])
    ax.set_ylabel("Validation NRMSE")
    ax.set_title("Grouping spectrum (500 × 20 budget)")
    _panel_letter(ax, "a")

    ax = axes[1]
    y = np.arange(len(DATASET_ORDER))[::-1]
    k_values = sorted(set(int(value) for value in frequency["k"]))
    for index, dataset in enumerate(DATASET_ORDER):
        yy = y[index]
        frame = frequency[frequency["dataset"] == dataset]
        ax.hlines(yy, min(k_values), max(k_values), color=LIGHT_GRAY, lw=0.7, zorder=0)
        positive = frame[frame["selection_frequency"] > 0]
        zero = frame[frame["selection_frequency"] == 0]
        ax.scatter(
            positive["k"],
            np.full(len(positive), yy),
            s=12 + 165 * positive["selection_frequency"],
            marker=MARKERS[dataset],
            color=COLORS[dataset],
            alpha=0.88,
            edgecolors="white",
            linewidths=0.55,
            zorder=2,
        )
        ax.scatter(
            zero["k"],
            np.full(len(zero), yy),
            s=10,
            marker=MARKERS[dataset],
            facecolors="none",
            edgecolors=COLORS[dataset],
            linewidths=0.65,
            zorder=2,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(k_values, [str(value) for value in k_values])
    ax.set_yticks(y, [DATASET_LABEL[d] for d in DATASET_ORDER])
    ax.set_xlabel("Validation-selected group size $k$")
    ax.set_title("Selection frequency (135 nested choices; open = 0)")
    _panel_letter(ax, "b")

    ax = axes[2]
    ax.axvline(0, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=0)
    rng = np.random.default_rng(31415)
    for index, dataset in enumerate(DATASET_ORDER):
        yy = y[index]
        nested_values = nested[nested["dataset"] == dataset]["gain"].to_numpy(dtype=float)
        ax.scatter(
            nested_values,
            yy + rng.uniform(-0.14, 0.14, size=len(nested_values)),
            s=7,
            color=COLORS[dataset],
            alpha=0.09,
            edgecolors="none",
            zorder=1,
        )
        points = cluster_gain[cluster_gain["dataset"] == dataset]["gain"].to_numpy(dtype=float)
        ax.scatter(
            points,
            yy + rng.uniform(-0.075, 0.075, size=len(points)),
            s=18,
            color=COLORS[dataset],
            alpha=0.45,
            edgecolors="white",
            linewidths=0.4,
            zorder=2,
        )
        row = gain_summary[gain_summary["dataset"] == dataset].iloc[0]
        ax.errorbar(
            row["mean_gain"],
            yy,
            xerr=[[row["mean_gain"] - row["ci95_low"]], [row["ci95_high"] - row["mean_gain"]]],
            fmt=MARKERS[dataset],
            ms=5.4,
            mfc=COLORS[dataset],
            mec="white",
            mew=0.7,
            color=COLORS[dataset],
            lw=1.4,
            capsize=2.2,
            zorder=4,
        )
    ax.set_yticks(y, [DATASET_LABEL[d] for d in DATASET_ORDER])
    ax.set_xlabel("EV − Split test NRMSE\n(positive favors Split)")
    ax.set_title("Trajectory/window inference (faint = nested)")
    _panel_letter(ax, "c")
    _save_figure(fig, generated, paper_figures, "qai_balanced_split_spectrum")


def plot_budget_partition(inputs: ReportInputs, generated: Path, paper_figures: Path) -> None:
    paired, summary = cluster_paired_gain(inputs.headline, "ev", "split")
    partition, partition_summary = partition_gain_summary(inputs.replicate)

    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.25), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.08)
    ax = axes[0]
    x = np.arange(len(BUDGET_ORDER))
    offsets = np.linspace(-0.24, 0.24, len(DATASET_ORDER))
    ax.axhline(0, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=0)
    for offset, dataset in zip(offsets, DATASET_ORDER):
        frame = summary[summary["dataset"] == dataset].set_index("budget_setting").loc[list(BUDGET_ORDER)]
        means = frame["mean_gain"].to_numpy(float)
        low = frame["ci95_low"].to_numpy(float)
        high = frame["ci95_high"].to_numpy(float)
        ax.errorbar(
            x + offset,
            means,
            yerr=np.vstack([means - low, high - means]),
            fmt=MARKERS[dataset],
            ms=4.3,
            mfc=COLORS[dataset],
            mec="white",
            mew=0.55,
            color=COLORS[dataset],
            lw=1.1,
            capsize=1.9,
            label=DATASET_LABEL[dataset],
        )
    ax.set_xticks(x, [BUDGET_LABEL[b] for b in BUDGET_ORDER])
    ax.set_xlabel("Time steps × shots per step (10,000 executions)")
    ax.set_ylabel("EV − Split test NRMSE")
    ax.set_title("Three fixed-budget settings")
    _panel_letter(ax, "a")

    ax = axes[1]
    y = np.arange(len(DATASET_ORDER))[::-1]
    ax.axvline(0, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=0)
    rng = np.random.default_rng(2718)
    for index, dataset in enumerate(DATASET_ORDER):
        yy = y[index]
        values = partition[partition["dataset"] == dataset]["gain"].to_numpy(float)
        ax.scatter(
            values,
            yy + rng.uniform(-0.13, 0.13, size=len(values)),
            s=8,
            color=COLORS[dataset],
            alpha=0.13,
            edgecolors="none",
        )
        row = partition_summary[partition_summary["dataset"] == dataset].iloc[0]
        ax.errorbar(
            row["mean_gain"],
            yy,
            xerr=[[row["mean_gain"] - row["ci95_low"]], [row["ci95_high"] - row["mean_gain"]]],
            fmt=MARKERS[dataset],
            ms=5.2,
            mfc=COLORS[dataset],
            mec="white",
            mew=0.65,
            color=COLORS[dataset],
            lw=1.35,
            capsize=2.1,
            zorder=3,
        )
    ax.set_yticks(y, [DATASET_LABEL[d] for d in DATASET_ORDER])
    ax.set_xlabel("EV − Split test NRMSE\n(positive favors Split)")
    ax.set_title("Twenty per-time shot permutations")
    _panel_letter(ax, "b")
    _save_figure(fig, generated, paper_figures, "qai_budget_partition_robustness")


def _tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _macro_name(*parts: object) -> str:
    tokens = []
    for part in parts:
        tokens.extend(re.findall(r"[A-Za-z0-9]+", str(part)))
    name = "QAI" + "".join(token[:1].upper() + token[1:] for token in tokens)
    if not re.fullmatch(r"[A-Za-z]+", name):
        name = re.sub(r"[^A-Za-z]", "", name)
    return name


def write_main_table(summary: pd.DataFrame, path: Path) -> None:
    balanced = summary[summary["budget_setting"] == "balanced"]
    lines = [
        "% Generated by scripts/report_qai_robust_results.py; do not edit by hand.",
        "% Cluster means after averaging reservoir, shot, and partition seeds.",
    ]
    finite = ("ev", "raw", "split", "covariance_ev")
    for dataset in DATASET_ORDER:
        task = balanced[balanced["dataset"] == dataset].set_index("method")
        finite_values = {method: float(task.loc[method, "mean_nrmse"]) for method in finite}
        best_finite = min(finite_values, key=finite_values.get)
        cells = []
        for method in REQUIRED_METHODS:
            value = f"{float(task.loc[method, 'mean_nrmse']):.3f}"
            if method == best_finite:
                value = rf"\textbf{{{value}}}"
            cells.append(value)
        lines.append(f"{_tex_escape(DATASET_LABEL[dataset])} & " + " & ".join(cells) + r" \\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{float(value):.{digits}f}"


def build_summary(inputs: ReportInputs) -> Dict[str, object]:
    cluster, method_summary = cluster_method_summary(inputs.headline)
    _, ev_split = cluster_paired_gain(inputs.headline, "ev", "split")
    _, split_cov = cluster_paired_gain(inputs.headline, "split", "covariance_ev")
    _, ideal = cluster_paired_gain(inputs.headline, "classical_input", "ideal_quantum")
    _, partition = partition_gain_summary(inputs.replicate)
    k_freq = selected_k_frequency(
        inputs.headline,
        inputs.config["organization_k_grid_by_setting"],
    )

    decision = inputs.config.get("decision_gate", {})
    try:
        primary_setting = str(decision["primary_confirmatory_setting"])
        practical_margin = float(decision["practical_noninferiority_margin_absolute_nrmse"])
        ev_every_lower = float(
            decision["split_vs_ev"]["require_every_task_ci_lower_bound_above"]
        )
        ev_one_lower = float(
            decision["split_vs_ev"]["require_at_least_one_task_ci_lower_bound_above"]
        )
        covariance_every_upper = float(
            decision["split_vs_covariance_ev"]["require_every_task_ci_upper_bound_below"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReportingContractError(f"malformed frozen decision gate: {exc}") from exc
    if (
        primary_setting != "balanced"
        or not math.isclose(practical_margin, 0.05)
        or not math.isclose(ev_every_lower, -practical_margin)
        or not math.isclose(ev_one_lower, 0.0)
        or not math.isclose(covariance_every_upper, practical_margin)
        or decision.get("other_budget_settings") != "descriptive robustness only"
    ):
        raise ReportingContractError("live decision gate is not the preregistered balanced 0.05 contract")

    confirmatory_ev_split = ev_split[ev_split["budget_setting"] == primary_setting].copy()
    confirmatory_split_cov = split_cov[split_cov["budget_setting"] == primary_setting].copy()
    if set(confirmatory_ev_split["dataset"]) != set(DATASET_ORDER):
        raise ReportingContractError("balanced Split-vs-EV gate does not cover every task")
    if set(confirmatory_split_cov["dataset"]) != set(DATASET_ORDER):
        raise ReportingContractError("balanced Split-vs-covariance-EV gate does not cover every task")

    noninferior = confirmatory_ev_split["ci95_low"] > ev_every_lower
    improved = confirmatory_ev_split["ci95_low"] > ev_one_lower
    covariance_control = confirmatory_split_cov["ci95_high"] < covariance_every_upper

    ideal_supported = ideal[ideal["ci95_low"] > 0]
    ideal_chaotic = ideal_supported[ideal_supported["dataset"] != "sunspots"]
    ideal_real = ideal_supported[ideal_supported["dataset"] == "sunspots"]
    partition_harm = partition[partition["ci95_high"] < 0]

    gates = {
        "feature_validity": {
            "passed": bool(inputs.diagnostics.get("passed") is True),
            "rule": "all preregistered ideal feature-validity criteria pass",
        },
        "split_vs_ev_balanced": {
            "passed": bool(noninferior.all() and improved.any()),
            "rule": (
                "for every task, lower 95% CI of EV-Split is strictly above -0.05 NRMSE; "
                "for at least one task it is strictly above zero"
            ),
            "primary_confirmatory_setting": primary_setting,
            "noninferiority_margin_absolute_nrmse": practical_margin,
            "tasks_noninferior": int(noninferior.sum()),
            "tasks_improved": int(improved.sum()),
            "tasks_required": len(DATASET_ORDER),
        },
        "split_vs_covariance_ev_balanced": {
            "passed": bool(covariance_control.all()),
            "rule": (
                "for every task, upper 95% CI of Split-Covariance-EV is strictly below "
                "0.05 NRMSE"
            ),
            "primary_confirmatory_setting": primary_setting,
            "practical_margin_absolute_nrmse": practical_margin,
            "tasks_within_margin": int(covariance_control.sum()),
            "tasks_required": len(DATASET_ORDER),
        },
    }
    overall = all(bool(item["passed"]) for item in gates.values())
    return {
        "schema_version": "qai-result-report-v1",
        "status": "GO" if overall else "NO-GO",
        "reporting_filter": {
            "scope_role": PRIMARY_SCOPE,
            "reservoir_variant": PRIMARY_RESERVOIR,
            "feature_set": PRIMARY_FEATURE_SET,
            "classical_exception": "classical_input/not_applicable",
            "excluded": ["smoke", "legacy", "feature_ablation", "degenerate_circuit_ablation"],
            "primary_confirmatory_setting": primary_setting,
            "descriptive_budget_settings": [
                setting for setting in BUDGET_ORDER if setting != primary_setting
            ],
        },
        "inference": {
            "independent_unit": "trajectory_or_sunspot_window",
            "nested_factors_averaged_before_ci": [
                "reservoir_seed",
                "shot_seed",
                "partition_seed",
            ],
            "interval": "two-sided 95% Student-t interval over cluster means",
        },
        "feature_validity": {
            key: inputs.diagnostics[key]
            for key in (
                "passed",
                "input_dependent_features",
                "input_dependent_fraction",
                "nonlinear_features",
                "nonlinear_fraction",
                "numerical_rank",
                "effective_rank",
                "classical_to_quantum_relative_projection_error",
            )
        },
        "method_summary": method_summary.to_dict(orient="records"),
        "ev_minus_split": ev_split.to_dict(orient="records"),
        "split_minus_covariance_ev": split_cov.to_dict(orient="records"),
        "confirmatory_ev_minus_split": confirmatory_ev_split.to_dict(orient="records"),
        "confirmatory_split_minus_covariance_ev": confirmatory_split_cov.to_dict(orient="records"),
        "classical_minus_ideal": ideal.to_dict(orient="records"),
        "partition_robustness": partition.to_dict(orient="records"),
        "selected_k_frequency": k_freq.to_dict(orient="records"),
        "descriptive_checks_not_in_go_no_go": {
            "ideal_vs_classical_supported_cells": int(len(ideal_supported)),
            "ideal_vs_classical_has_chaotic_support": bool(len(ideal_chaotic) > 0),
            "ideal_vs_classical_has_sunspot_support": bool(len(ideal_real) > 0),
            "partition_supported_harm_tasks": int(len(partition_harm)),
            "note": "These diagnostics do not alter the frozen balanced-setting decision gate.",
        },
        "gates": gates,
        "overall_passed": overall,
    }


def write_macros(inputs: ReportInputs, summary: Mapping[str, object], path: Path) -> None:
    _, method_summary = cluster_method_summary(inputs.headline)
    _, ev_split = cluster_paired_gain(inputs.headline, "ev", "split")
    lines = [
        "% Candidate result macros generated from a complete campaign.",
        "% Review scientifically before including this file in the manuscript.",
        rf"\newcommand{{\QAIFeatureDependentCount}}{{{int(inputs.diagnostics['input_dependent_features'])}}}",
        rf"\newcommand{{\QAIFeatureDimension}}{{{int(inputs.diagnostics['feature_dimension'])}}}",
        rf"\newcommand{{\QAINonlinearFeatureCount}}{{{int(inputs.diagnostics['nonlinear_features'])}}}",
        rf"\newcommand{{\QAIFeatureNumericalRank}}{{{int(inputs.diagnostics['numerical_rank'])}}}",
        rf"\newcommand{{\QAIFeatureEffectiveRank}}{{{float(inputs.diagnostics['effective_rank']):.2f}}}",
        rf"\newcommand{{\QAIOverallGateStatus}}{{{summary['status']}}}",
    ]
    for _, row in method_summary.iterrows():
        name = _macro_name(row["budget_setting"], row["dataset"], row["method"], "NRMSE")
        lines.append(rf"\newcommand{{\{name}}}{{{_fmt(row['mean_nrmse'])}}}")
    for _, row in ev_split.iterrows():
        name = _macro_name(row["budget_setting"], row["dataset"], "EVMinusSplit")
        lines.append(rf"\newcommand{{\{name}}}{{{_fmt(row['mean_gain'])}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_report_manifest(
    inputs: ReportInputs,
    paper_figures: Path,
    artifacts: Sequence[Path],
) -> Dict[str, object]:
    source_paths = [
        REPORTER_PATH,
        inputs.config_path,
        inputs.results_dir / "manifest.json",
        inputs.results_dir / "replicate_results.csv",
        inputs.results_dir / "split_validation_spectrum.csv",
        inputs.diagnostics_path,
        inputs.feature_grid_path,
    ]
    source_sha256 = {_path_label(path): _sha256(path) for path in source_paths}
    reporter_label = _path_label(REPORTER_PATH)
    return {
        "schema_version": "qai-result-report-manifest-v1",
        "source_sha256": source_sha256,
        "artifact_sha256": {path.name: _sha256(path) for path in artifacts},
        "reporter": {
            "path": reporter_label,
            "sha256": source_sha256[reporter_label],
        },
        "paper_figure_directory": _path_label(paper_figures),
        "headline_filter": {
            "scope_role": PRIMARY_SCOPE,
            "reservoir_variant": PRIMARY_RESERVOIR,
            "feature_set": PRIMARY_FEATURE_SET,
        },
        "verified_campaign_hashes": {
            "config_sha256": inputs.manifest["config_sha256"],
            "implementation_sha256": inputs.manifest["implementation_sha256"],
            "sunspot_data_sha256": inputs.manifest["data_provenance"]["sunspots"][
                "observed_sha256"
            ],
        },
    }


def _report_artifact_paths(generated: Path) -> list[Path]:
    """Return hashable report artifacts, excluding the manifest itself."""

    return sorted(
        path
        for path in generated.iterdir()
        if path.is_file() and path.name != "report_manifest.json"
    )


def generate_report(inputs: ReportInputs, paper_figures: Path) -> Path:
    generated = inputs.results_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _apply_style()
    plot_feature_validity_and_ideal(inputs, generated, paper_figures)
    plot_balanced_spectrum(inputs, generated, paper_figures)
    plot_budget_partition(inputs, generated, paper_figures)

    _, method_summary = cluster_method_summary(inputs.headline)
    write_main_table(method_summary, generated / "main_table_body.tex")
    summary = build_summary(inputs)
    (generated / "summary_go_no_go.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_macros(inputs, summary, generated / "result_macros_candidate.tex")

    artifacts = _report_artifact_paths(generated)
    report_manifest = _build_report_manifest(inputs, paper_figures, artifacts)
    (generated / "report_manifest.json").write_text(
        json.dumps(report_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--feature-grid", type=Path, default=DEFAULT_FEATURE_GRID)
    parser.add_argument("--paper-figures", type=Path, default=DEFAULT_PAPER_FIGURES)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="enforce the reporting contract without writing any artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = validate_complete_inputs(
        args.results_dir,
        args.diagnostics,
        args.feature_grid,
        args.config,
    )
    if args.validate_only:
        print(f"complete reporting contract passed for {args.results_dir}")
        return
    generated = generate_report(inputs, args.paper_figures)
    print(f"wrote validated result report to {generated}")


if __name__ == "__main__":
    main()
