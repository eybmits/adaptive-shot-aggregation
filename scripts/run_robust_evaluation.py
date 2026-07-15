#!/usr/bin/env python3
"""Run the preregistered robustness campaign for adaptive shot aggregation.

This is intentionally separate from the historical QAI scripts.  A full run
uses five independent trajectories/windows per task, three reservoir seeds,
three shot seeds, and three per-time partition seeds at each of three exact
10,000-execution settings.  A 20-partition subset and a fixed-lane ablation
probe sensitivity to the arbitrary shot-lane construction.

The primary expressive reservoir is gated by ideal-state diagnostics.  Use
``--dry-run`` to inspect the complete plan without simulation.  ``--smoke`` is
only a wiring test and is labeled as such in its manifest; it is not paper
evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import base_impl as base
import robust_protocol as robust
from robust_reservoir_adapter import make_reservoir_provider


DEFAULT_CONFIG = ROOT / "configs" / "benchmark.json"
DEFAULT_RESULTS = ROOT / "results" / "qai_robust_protocol"
IMPLEMENTATION_FILES = (
    HERE / "run_robust_evaluation.py",
    HERE / "robust_protocol.py",
    HERE / "robust_reservoir_adapter.py",
    HERE / "expressive_reservoir.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-job JSON checkpoints from results-dir/shards.",
    )
    parser.add_argument(
        "--allow-unvalidated-reservoir",
        action="store_true",
        help="Allowed only with --smoke; never use this for headline results.",
    )
    return parser.parse_args()


def verify_sunspot_dataset_pin(config: Mapping[str, object]) -> Dict[str, object]:
    """Verify the exact cached real-data snapshot before any job is planned."""

    pin = config.get("sunspot_dataset_pin", {})
    required = {"path", "sha256", "rows", "start_month", "end_month", "series", "source"}
    if set(pin) != required:
        raise ValueError(f"sunspot_dataset_pin must contain exactly {sorted(required)}")
    if str(pin["path"]) != "data/sunspots_monthly_sidc.csv":
        raise ValueError("sunspot dataset pin must reference the runner's canonical cache path")
    path = ROOT / str(pin["path"])
    if not path.exists():
        raise ValueError(f"pinned sunspot dataset is missing: {path}")
    observed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_sha != str(pin["sha256"]):
        raise ValueError("pinned sunspot dataset SHA-256 mismatch")
    table = np.loadtxt(path, delimiter=";", usecols=(0, 1, 3), dtype=float)
    rows = int(table.shape[0])
    start_month = f"{int(table[0, 0]):04d}-{int(table[0, 1]):02d}"
    end_month = f"{int(table[-1, 0]):04d}-{int(table[-1, 1]):02d}"
    if rows != int(pin["rows"]) or start_month != pin["start_month"] or end_month != pin["end_month"]:
        raise ValueError("pinned sunspot row count or date range mismatch")
    return {
        **dict(pin),
        "verified": True,
        "observed_sha256": observed_sha,
        "observed_rows": rows,
        "observed_start_month": start_month,
        "observed_end_month": end_month,
    }


def load_santa_fe_laser(config: Mapping[str, object]) -> np.ndarray:
    """Load the original Series A record and its continuation."""

    pin = config.get("santa_fe_dataset_pin", {})
    train = np.loadtxt(ROOT / str(pin["train_path"]), dtype=float)
    continuation = np.loadtxt(ROOT / str(pin["continuation_path"]), dtype=float)
    return np.concatenate([np.ravel(train), np.ravel(continuation)]).astype(float, copy=False)


def verify_santa_fe_dataset_pin(config: Mapping[str, object]) -> Dict[str, object]:
    """Verify the exact McNames archive files used for the laser screen."""

    pin = config.get("santa_fe_dataset_pin", {})
    required = {
        "train_path",
        "train_sha256",
        "train_rows",
        "train_source",
        "continuation_path",
        "continuation_sha256",
        "continuation_rows",
        "continuation_source",
        "total_rows",
        "series",
        "source_page",
    }
    if set(pin) != required:
        raise ValueError(f"santa_fe_dataset_pin must contain exactly {sorted(required)}")
    observed: Dict[str, object] = {}
    for prefix in ("train", "continuation"):
        path = ROOT / str(pin[f"{prefix}_path"])
        if not path.exists():
            raise ValueError(f"pinned Santa Fe {prefix} file is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != str(pin[f"{prefix}_sha256"]):
            raise ValueError(f"pinned Santa Fe {prefix} SHA-256 mismatch")
        values = np.ravel(np.loadtxt(path, dtype=float))
        if len(values) != int(pin[f"{prefix}_rows"]):
            raise ValueError(f"pinned Santa Fe {prefix} row count mismatch")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"pinned Santa Fe {prefix} contains non-finite values")
        observed[f"observed_{prefix}_sha256"] = digest
        observed[f"observed_{prefix}_rows"] = int(len(values))
    series = load_santa_fe_laser(config)
    if len(series) != int(pin["total_rows"]):
        raise ValueError("pinned Santa Fe combined row count mismatch")
    if float(series.min()) < 0.0 or float(series.max()) > 255.0:
        raise ValueError("pinned Santa Fe values fall outside the documented 8-bit intensity range")
    return {
        **dict(pin),
        **observed,
        "observed_total_rows": int(len(series)),
        "observed_min": float(series.min()),
        "observed_max": float(series.max()),
        "verified": True,
    }


def santa_fe_test_index_ranges(config: Mapping[str, object]) -> Dict[str, List[Dict[str, int]]]:
    """Return disjoint held-out target ranges for the laser rolling windows."""

    pin = verify_santa_fe_dataset_pin(config)
    data_length = int(pin["observed_total_rows"])
    continuation_start = int(pin["train_rows"])
    spacing = int(config["santa_fe_origin_spacing_samples"])
    ranges_by_setting: Dict[str, List[Dict[str, int]]] = {}
    maximum_window = 0
    for item in config["budget_settings"]:
        setting = robust.BudgetSetting(**item)
        maximum_window = max(maximum_window, int(setting.time_steps + 1))
        design = robust.make_temporal_design(
            setting.time_steps,
            t0=int(config["t0"]),
            test_fraction=float(config["test_fraction"]),
            validation_folds=int(config["validation_folds"]),
            initial_train_fraction=float(config["initial_train_fraction"]),
        )
        registered: List[Dict[str, int]] = []
        for cluster_index in config["trajectory_indices"]:
            end = int(data_length - int(cluster_index) * spacing)
            window_start = int(end - (setting.time_steps + 1))
            if window_start < continuation_start:
                raise ValueError(
                    f"Santa Fe window {cluster_index} crosses the original competition boundary"
                )
            registered.append(
                {
                    "cluster_index": int(cluster_index),
                    "window_start": window_start,
                    "window_stop": end,
                    "start": int(window_start + design.test_start + 1),
                    "stop": end,
                    "length": int(end - (window_start + design.test_start + 1)),
                }
            )
        ranges_by_setting[setting.name] = registered
    if spacing < maximum_window:
        raise ValueError(
            f"Santa Fe origin spacing {spacing} is shorter than the largest full window {maximum_window}"
        )
    for setting_name, registered in ranges_by_setting.items():
        ordered = sorted(registered, key=lambda row: row["window_start"])
        if any(left["window_stop"] > right["window_start"] for left, right in zip(ordered, ordered[1:])):
            raise ValueError(f"Santa Fe full windows overlap for setting {setting_name}")
    return ranges_by_setting


def sunspot_test_index_ranges(config: Mapping[str, object]) -> Dict[str, List[Dict[str, int]]]:
    """Return absolute half-open held-out target ranges for every sunspot origin."""

    pin = verify_sunspot_dataset_pin(config)
    path = ROOT / str(pin["path"])
    table = np.loadtxt(path, delimiter=";", usecols=(0, 1, 3), dtype=float)
    data_length = int(np.sum(table[:, 2] >= 0.0))
    spacing = int(config["sunspot_origin_spacing_months"])
    ranges_by_setting: Dict[str, List[Dict[str, int]]] = {}
    max_test_length = 0
    for item in config["budget_settings"]:
        setting = robust.BudgetSetting(**item)
        design = robust.make_temporal_design(
            setting.time_steps,
            t0=int(config["t0"]),
            test_fraction=float(config["test_fraction"]),
            validation_folds=int(config["validation_folds"]),
            initial_train_fraction=float(config["initial_train_fraction"]),
        )
        test_length = int(setting.time_steps - design.test_start)
        max_test_length = max(max_test_length, test_length)
        registered: List[Dict[str, int]] = []
        for cluster_index in config["trajectory_indices"]:
            end = int(data_length - int(cluster_index) * spacing)
            window_start = int(end - (setting.time_steps + 1))
            if window_start < 0:
                raise ValueError(f"sunspot origin {cluster_index} is too early for {setting.name}")
            # Target at forecast time t is raw[t+1]. The half-open absolute
            # test range is therefore [window_start+test_start+1, end).
            test_start_index = int(window_start + design.test_start + 1)
            registered.append(
                {
                    "cluster_index": int(cluster_index),
                    "start": test_start_index,
                    "stop": end,
                    "length": int(end - test_start_index),
                }
            )
        ranges_by_setting[setting.name] = registered
    if spacing < max_test_length:
        raise ValueError(
            f"sunspot origin spacing {spacing} is shorter than the maximum test block {max_test_length}"
        )
    for setting_name, registered in ranges_by_setting.items():
        ordered = sorted(registered, key=lambda row: row["start"])
        if any(left["stop"] > right["start"] for left, right in zip(ordered, ordered[1:])):
            raise ValueError(f"sunspot held-out target blocks overlap for setting {setting_name}")
    return ranges_by_setting


def load_config(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "datasets",
        "budget_settings",
        "trajectory_indices",
        "reservoir_seeds",
        "shot_seeds",
        "partition_seeds",
        "feature_sets",
        "lambda_grid",
        "covariance_alpha_grid",
        "evaluation_scopes",
        "sunspot_origin_spacing_months",
        "sunspot_dataset_pin",
        "decision_gate",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"robust protocol config is missing {sorted(missing)}")
    if len(data["trajectory_indices"]) < 5:
        raise ValueError("at least five independent trajectory/window clusters are required")
    for factor in ("reservoir_seeds", "shot_seeds", "partition_seeds"):
        if len(data[factor]) < 3:
            raise ValueError(f"{factor} must contain at least three preregistered values")
    settings = robust.validate_budget_settings(
        robust.BudgetSetting(
            name=str(item["name"]),
            time_steps=int(item["time_steps"]),
            shots_per_step=int(item["shots_per_step"]),
        )
        for item in data["budget_settings"]
    )
    registered_budget = int(data["execution_budget_per_record"])
    if any(setting.executions != registered_budget for setting in settings):
        raise ValueError("budget setting does not match execution_budget_per_record")
    feature_sets = set(data["feature_sets"])
    if not feature_sets or not feature_sets.issubset(set(robust.FEATURE_SETS)):
        raise ValueError(f"feature_sets must be a nonempty subset of {robust.FEATURE_SETS}")
    scopes = data["evaluation_scopes"]
    required_scopes = {"headline", "partition_robustness"}
    allowed_scopes = required_scopes | {"feature_ablation", "degenerate_circuit_ablation"}
    if not required_scopes.issubset(scopes) or not set(scopes).issubset(allowed_scopes):
        raise ValueError(
            f"evaluation_scopes must include {sorted(required_scopes)} and use only "
            f"{sorted(allowed_scopes)}"
        )
    if scopes["headline"]["reservoir_variant"] != data["primary_reservoir_variant"]:
        raise ValueError("headline scope must use the primary expressive reservoir")
    if scopes["headline"]["feature_sets"] != [data["primary_feature_set"]]:
        raise ValueError("headline scope must contain only the primary z_plus_zz feature set")
    if (
        "degenerate_circuit_ablation" in scopes
        and scopes["degenerate_circuit_ablation"]["budget_settings"] != ["balanced"]
    ):
        raise ValueError("degenerate circuit is restricted to the balanced-setting ablation")
    setting_shots = {str(item["name"]): int(item["shots_per_step"]) for item in data["budget_settings"]}
    if set(data["organization_k_grid_by_setting"]) != set(setting_shots):
        raise ValueError("organization_k_grid_by_setting must cover exactly the three settings")
    for name, n_shots in setting_shots.items():
        k_grid = tuple(int(value) for value in data["organization_k_grid_by_setting"][name])
        expected = (1, *robust.proper_group_sizes(n_shots), n_shots)
        if k_grid != expected:
            raise ValueError(f"{name} must register every divisor of N={n_shots}: {expected}")
    if len(data["covariance_alpha_grid"]) < max(
        len(values) for values in data["organization_k_grid_by_setting"].values()
    ):
        raise ValueError("covariance-EV must have at least as many alpha candidates as any Split k grid")
    normalization = data.get("normalization", {})
    if normalization.get("clip_circuit_inputs_only") is not True:
        raise ValueError("registered preprocessing must clip circuit inputs only")
    if "never clipped" not in str(normalization.get("targets", "")):
        raise ValueError("registered forecast targets must remain train-scaled and unclipped")
    decision_gate = data["decision_gate"]
    if decision_gate.get("primary_confirmatory_setting") != "balanced":
        raise ValueError("the frozen primary confirmatory setting must be balanced")
    if float(decision_gate.get("practical_noninferiority_margin_absolute_nrmse", np.nan)) != 0.05:
        raise ValueError("the frozen practical non-inferiority margin must be 0.05 NRMSE")
    if "sunspots" in data["datasets"]:
        verify_sunspot_dataset_pin(data)
        sunspot_test_index_ranges(data)
    if "santa_fe_laser" in data["datasets"]:
        if "santa_fe_dataset_pin" not in data or "santa_fe_origin_spacing_samples" not in data:
            raise ValueError("Santa Fe runs require a pinned dataset and origin spacing")
        verify_santa_fe_dataset_pin(data)
        santa_fe_test_index_ranges(data)
    return data


def registered_k_grid(config: Mapping[str, object], setting: robust.BudgetSetting) -> Tuple[int, ...]:
    configured = config["organization_k_grid_by_setting"].get(setting.name)
    if configured is None and setting.name == "smoke_only":
        configured = (1, 2, 5, setting.shots_per_step)
    if configured is None:
        raise ValueError(f"no registered k grid for setting {setting.name}")
    grid = tuple(int(value) for value in configured)
    if grid != (1, *robust.proper_group_sizes(setting.shots_per_step), setting.shots_per_step):
        raise ValueError(f"invalid registered k grid {grid} for N={setting.shots_per_step}")
    return grid


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in IMPLEMENTATION_FILES:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def job_id(job: Mapping[str, object]) -> str:
    canonical = json.dumps(job, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def path_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def feature_gate(config: Mapping[str, object], *, smoke: bool, allow_override: bool) -> Dict[str, object]:
    report_path = ROOT / str(config["feature_validity_report"])
    required = bool(config.get("require_feature_validity_gate", True))
    report: Dict[str, object] = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    architecture = report.get("architecture", {}) if isinstance(report.get("architecture", {}), dict) else {}
    expected_architecture = str(config.get("primary_architecture_sha256", ""))
    expected_parameters = str(config.get("primary_parameter_sha256", ""))
    hash_match = bool(
        architecture.get("architecture_sha256") == expected_architecture
        and architecture.get("parameter_sha256") == expected_parameters
    )
    passed = bool(report.get("passed", False)) and hash_match
    if required and not passed and not (smoke and allow_override):
        raise RuntimeError(
            f"expressive feature-validity gate has not passed: {report_path}. "
            "Run `python scripts/run_expressive_reservoir_diagnostics.py "
            "--feature-grid-csv results/expressive_reservoir/ideal_feature_grid.csv` "
            "before forecasting."
        )
    return {
        "required": required,
        "report": str(report_path.relative_to(ROOT)),
        "passed": passed,
        "expected_architecture_sha256": expected_architecture,
        "observed_architecture_sha256": architecture.get("architecture_sha256"),
        "expected_parameter_sha256": expected_parameters,
        "observed_parameter_sha256": architecture.get("parameter_sha256"),
        "hash_match": hash_match,
        "override_for_smoke_only": bool(smoke and allow_override and not passed),
        "criteria": report.get("criteria", {}),
    }


def _rossler_independent(n_steps: int, trajectory_index: int) -> np.ndarray:
    rng = np.random.default_rng(7043 + int(trajectory_index))
    x, y, z = np.array([1.0, 1.0, 1.0]) + rng.uniform(-0.35, 0.35, size=3)
    warmup = 2500
    subsample = 5
    dt = 0.02
    output = np.empty(n_steps + warmup, dtype=float)
    stored = 0
    for step in range((n_steps + warmup) * subsample):
        if step % subsample == 0:
            output[stored] = x
            stored += 1
        dx = -y - z
        dy = x + 0.2 * y
        dz = 0.2 + z * (x - 5.7)
        x += dt * dx
        y += dt * dy
        z += dt * dz
    return output[warmup:]


def independent_series(
    dataset: str,
    *,
    time_steps: int,
    trajectory_index: int,
    sunspot_spacing: int,
) -> Tuple[np.ndarray, str, Dict[str, object]]:
    """Return a raw T+1 sequence and its independent cluster identifier."""

    count = int(time_steps + 1)
    index = int(trajectory_index)
    if dataset == "mackey_glass":
        seed = 5101 + index
        return (
            base.mackey_glass(count, seed=seed),
            f"mackey_glass_trajectory_{index:02d}",
            {"cluster_kind": "independent_trajectory", "trajectory_seed": seed},
        )
    if dataset == "lorenz":
        rng = np.random.default_rng(6101 + index)
        initial = tuple((np.array([1.0, 1.0, 1.0]) + rng.uniform(-1.0, 1.0, size=3)).tolist())
        return (
            base.lorenz_x(count, init_state=initial),
            f"lorenz_trajectory_{index:02d}",
            {"cluster_kind": "independent_trajectory", "initial_state": list(initial)},
        )
    if dataset == "rossler":
        return (
            _rossler_independent(count, index),
            f"rossler_trajectory_{index:02d}",
            {"cluster_kind": "independent_trajectory", "trajectory_seed": 7043 + index},
        )
    if dataset == "sunspots":
        path = ROOT / "data" / "sunspots_monthly_sidc.csv"
        table = np.loadtxt(path, delimiter=";", usecols=(0, 1, 3), dtype=float)
        table = table[table[:, 2] >= 0.0]
        end = int(len(table) - index * int(sunspot_spacing))
        start = int(end - count)
        if start < 0:
            raise ValueError(f"sunspot rolling window {index} is too early for T={time_steps}")
        segment = table[start:end]
        year, month = int(segment[-1, 0]), int(segment[-1, 1])
        return (
            segment[:, 2].copy(),
            f"sunspots_origin_{year:04d}_{month:02d}",
            {
                "cluster_kind": "rolling_origin_window",
                "window_start_index": start,
                "window_end_index": end,
                "origin_year": year,
                "origin_month": month,
            },
        )
    if dataset == "santa_fe_laser":
        # The exploratory screen uses five fully disjoint windows from the
        # 9,093-point continuation.  The original 1,000 competition points are
        # retained for provenance but never enter these windows.
        pin = {
            "train_path": "data/santa_fe_laser_A.dat",
            "continuation_path": "data/santa_fe_laser_A_continuation.dat",
        }
        train = np.ravel(np.loadtxt(ROOT / pin["train_path"], dtype=float))
        continuation = np.ravel(np.loadtxt(ROOT / pin["continuation_path"], dtype=float))
        values = np.concatenate([train, continuation])
        spacing = int(sunspot_spacing)
        end = int(len(values) - index * spacing)
        start = int(end - count)
        if start < len(train):
            raise ValueError(f"Santa Fe rolling window {index} crosses the competition boundary")
        return (
            values[start:end].copy(),
            f"santa_fe_laser_window_{index:02d}",
            {
                "cluster_kind": "disjoint_continuation_window",
                "window_start_index": start,
                "window_end_index": end,
            },
        )
    raise ValueError(f"unsupported dataset={dataset!r}")


def result_row(common: Mapping[str, object], method: str, result: Mapping[str, object]) -> Dict[str, object]:
    return {
        **common,
        "method": method,
        "k": int(result.get("k", 1)),
        "groups": int(result.get("groups", 1)),
        "interior_k_selected": bool(result.get("interior_k_selected", False)),
        "lambda_selected": float(result.get("lambda_selected", np.nan)),
        "alpha_selected": float(result.get("alpha_selected", np.nan)),
        "mean_val_nrmse": float(result.get("mean_val_nrmse", np.nan)),
        "second_best_val_nrmse": float(result.get("second_best_val_nrmse", np.nan)),
        "selection_margin": float(result.get("selection_margin", np.nan)),
        "fold_val_nrmse": json.dumps([float(value) for value in result.get("fold_val_nrmse", ())]),
        "test_nrmse": float(result["test_nrmse"]),
    }


def validation_spectrum_rows(
    common: Mapping[str, object], result: Mapping[str, object]
) -> List[Dict[str, object]]:
    """Flatten the validation-only fixed-k spectrum; never includes test data."""

    validation_metadata = {
        key: value
        for key, value in common.items()
        if not key.startswith(("test", "target")) and "postfit" not in key
    }
    return [
        {
            **validation_metadata,
            "method": "split",
            "k": int(point["k"]),
            "lambda_selected": float(point["lambda_selected"]),
            "mean_val_nrmse": float(point["mean_val_nrmse"]),
            "fold_val_nrmse": json.dumps([float(value) for value in point["fold_val_nrmse"]]),
            "lambda_selection_margin": float(point["lambda_selection_margin"]),
            "metric_role": "validation_only_no_test_access",
        }
        for point in result.get("k_validation_spectrum", [])
    ]


def partition_specs_for_job(
    config: Mapping[str, object],
    *,
    setting_name: str,
    reservoir_variant: str,
    reservoir_seed: int,
    feature_methods: Mapping[str, Sequence[str]],
) -> List[Tuple[int, str, str, Dict[str, Tuple[str, ...]], str | None]]:
    main_mapping = {
        str(feature): tuple(str(method) for method in methods if method in {"ev", "raw", "split", "covariance_ev"})
        for feature, methods in feature_methods.items()
    }
    specs = [
        (int(seed), "per_time_permuted", "primary", main_mapping, None)
        for seed in config["partition_seeds"]
    ]
    extra = config.get("partition_robustness_subset", {})
    if (
        setting_name == extra.get("budget_setting")
        and reservoir_variant == extra.get("reservoir_variant")
        and reservoir_seed == int(extra.get("reservoir_seed", -1))
    ):
        for seed in extra.get("partition_seeds", []):
            specs.append(
                (
                    int(seed),
                    "per_time_permuted",
                    "twenty_partition_robustness",
                    {str(extra["feature_set"]): ("ev", "split")},
                    "partition_robustness",
                )
            )
    fixed = config.get("fixed_lane_ablation", {})
    if (
        setting_name == fixed.get("budget_setting")
        and reservoir_variant == fixed.get("reservoir_variant")
        and reservoir_seed == int(fixed.get("reservoir_seed", -1))
    ):
        for seed in fixed.get("partition_seeds", []):
            specs.append(
                (
                    int(seed),
                    "fixed_lane",
                    "fixed_lane_ablation",
                    {str(fixed["feature_set"]): ("ev", "split")},
                    "partition_robustness",
                )
            )
    return specs


def run_job(
    job: Mapping[str, object],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    config_path = Path(str(job["config_path"]))
    observed_config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if observed_config_hash != str(job["config_sha256"]):
        raise RuntimeError("protocol config changed after jobs were planned; refusing mixed-config results")
    if implementation_sha256() != str(job["implementation_sha256"]):
        raise RuntimeError("protocol implementation changed after jobs were planned; refusing mixed-code results")
    config = load_config(config_path)
    setting = robust.BudgetSetting(**job["setting"])
    dataset = str(job["dataset"])
    cluster_index = int(job["cluster_index"])
    reservoir_variant = str(job["reservoir_variant"])
    reservoir_seed = int(job["reservoir_seed"])
    feature_scopes = {str(key): str(value) for key, value in job["feature_scopes"].items()}
    feature_methods = {
        str(key): tuple(str(method) for method in value)
        for key, value in job["feature_methods"].items()
    }
    design = robust.make_temporal_design(
        setting.time_steps,
        t0=int(config["t0"]),
        test_fraction=float(config["test_fraction"]),
        validation_folds=int(config["validation_folds"]),
        initial_train_fraction=float(config["initial_train_fraction"]),
    )
    raw, cluster_id, cluster_meta = independent_series(
        dataset,
        time_steps=setting.time_steps,
        trajectory_index=cluster_index,
        sunspot_spacing=int(
            config["santa_fe_origin_spacing_samples"]
            if dataset == "santa_fe_laser"
            else config["sunspot_origin_spacing_months"]
        ),
    )
    inputs, targets, normalization = robust.normalize_train_prefix(
        raw,
        time_steps=setting.time_steps,
        fit_stop=design.normalization_fit_stop,
        clip_inputs=True,
    )
    provider = make_reservoir_provider(
        reservoir_variant,
        q=int(config["q"]),
        parameter_seed=reservoir_seed,
        eta=float(config["eta"]),
        window=int(config["window"]),
    )
    common = {
        "dataset": dataset,
        "cluster_id": cluster_id,
        "cluster_index": cluster_index,
        **cluster_meta,
        "budget_setting": setting.name,
        "time_steps": setting.time_steps,
        "shots_per_step": setting.shots_per_step,
        "executions": setting.executions,
        "test_start": design.test_start,
        "reservoir_variant": reservoir_variant,
        "reservoir_seed": reservoir_seed,
        **normalization,
    }
    lambda_grid = tuple(float(value) for value in config["lambda_grid"])
    alpha_grid = tuple(float(value) for value in config["covariance_alpha_grid"])
    rows: List[Dict[str, object]] = []
    spectrum_rows: List[Dict[str, object]] = []

    # The classical baseline is emitted once per dataset/cluster/setting.
    if (
        bool(job.get("emit_classical", False))
        and reservoir_variant == str(config["primary_reservoir_variant"])
        and reservoir_seed == int(config["primary_reservoir_seed"])
    ):
        classical = robust.classical_leak_lag(inputs, eta=float(config["eta"]), window=int(config["window"]))
        result = robust.evaluate_grouped(classical, targets, design, k_grid=(1,), lambda_grid=lambda_grid)
        rows.append(
            result_row(
                {
                    **common,
                    "reservoir_variant": "not_applicable",
                    "reservoir_seed": np.nan,
                    "shot_seed": np.nan,
                    "effective_shot_seed": np.nan,
                    "partition_seed": np.nan,
                    "effective_partition_seed": np.nan,
                    "partition_mode": "not_applicable",
                    "partition_role": "not_applicable",
                    "scope_role": "headline",
                    "feature_set": "classical_input",
                },
                "classical_input",
                result,
            )
        )

    # Exact/infinite-shot baseline: no shot or partition randomness.
    for feature_set, methods_for_feature in feature_methods.items():
        if "ideal_quantum" not in methods_for_feature:
            continue
        ideal_steps = provider.ideal_features(inputs, feature_set=feature_set)
        ideal = robust.leak_and_lag(ideal_steps, eta=float(config["eta"]), window=int(config["window"]))
        result = robust.evaluate_grouped(ideal[None, :, :], targets, design, k_grid=(1,), lambda_grid=lambda_grid)
        rows.append(
            result_row(
                {
                    **common,
                    "shot_seed": np.nan,
                    "effective_shot_seed": np.nan,
                    "partition_seed": np.nan,
                    "effective_partition_seed": np.nan,
                    "partition_mode": "not_applicable",
                    "partition_role": "not_applicable",
                    "scope_role": feature_scopes[feature_set],
                    "feature_set": feature_set,
                },
                "ideal_quantum",
                result,
            )
        )

    all_k = registered_k_grid(config, setting)
    for registered_shot_seed in tuple(int(value) for value in config["shot_seeds"]):
        effective_shot_seed = stable_seed(
            "shot",
            registered_shot_seed,
            dataset,
            cluster_id,
            setting.name,
            reservoir_variant,
            reservoir_seed,
        )
        local_z = provider.shot_local_z(
            inputs,
            n_shots=setting.shots_per_step,
            shot_seed=effective_shot_seed,
        )
        for registered_partition_seed, mode, role, scoped_methods, scope_override in partition_specs_for_job(
            config,
            setting_name=setting.name,
            reservoir_variant=reservoir_variant,
            reservoir_seed=reservoir_seed,
            feature_methods=feature_methods,
        ):
            effective_partition_seed = stable_seed(
                "partition",
                registered_partition_seed,
                dataset,
                cluster_id,
                setting.name,
                reservoir_variant,
                reservoir_seed,
                registered_shot_seed,
                mode,
            )
            for feature_set, methods in scoped_methods.items():
                step_features = robust.derive_feature_set(local_z, feature_set)
                base_common = {
                    **common,
                    "shot_seed": registered_shot_seed,
                    "effective_shot_seed": effective_shot_seed,
                    "partition_seed": registered_partition_seed,
                    "effective_partition_seed": effective_partition_seed,
                    "partition_mode": mode,
                    "partition_role": role,
                    "scope_role": scope_override or feature_scopes[feature_set],
                    "feature_set": feature_set,
                }
                if "raw" in methods:
                    result = robust.evaluate_shot_organization(
                        step_features,
                        targets,
                        design,
                        k_grid=(1,),
                        lambda_grid=lambda_grid,
                        partition_seed=effective_partition_seed,
                        partition_mode=mode,
                        eta=float(config["eta"]),
                        window=int(config["window"]),
                    )
                    rows.append(result_row(base_common, "raw", result))
                if "ev" in methods:
                    result = robust.evaluate_shot_organization(
                        step_features,
                        targets,
                        design,
                        k_grid=(setting.shots_per_step,),
                        lambda_grid=lambda_grid,
                        partition_seed=effective_partition_seed,
                        partition_mode=mode,
                        eta=float(config["eta"]),
                        window=int(config["window"]),
                    )
                    rows.append(result_row(base_common, "ev", result))
                if "split" in methods:
                    result = robust.evaluate_shot_organization(
                        step_features,
                        targets,
                        design,
                        k_grid=all_k,
                        lambda_grid=lambda_grid,
                        partition_seed=effective_partition_seed,
                        partition_mode=mode,
                        eta=float(config["eta"]),
                        window=int(config["window"]),
                    )
                    rows.append(result_row(base_common, "split", result))
                    if (scope_override or feature_scopes[feature_set]) == "headline" and role == "primary":
                        spectrum_rows.extend(validation_spectrum_rows(base_common, result))
                if "covariance_ev" in methods:
                    shot_phi = robust.organized_views_from_shots(
                        step_features,
                        k=1,
                        partition_seed=effective_partition_seed,
                        partition_mode=mode,
                        eta=float(config["eta"]),
                        window=int(config["window"]),
                    )
                    result = robust.evaluate_covariance_ev(
                        shot_phi,
                        targets,
                        design,
                        lambda_grid=lambda_grid,
                        alpha_grid=alpha_grid,
                    )
                    rows.append(result_row(base_common, "covariance_ev", result))
    return rows, spectrum_rows, provider.architecture_manifest()


def planned_jobs(config: Mapping[str, object], config_path: Path, *, smoke: bool) -> List[Dict[str, object]]:
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    implementation_hash = implementation_sha256()
    settings_by_name = {
        str(item["name"]): robust.BudgetSetting(**item)
        for item in config["budget_settings"]
    }
    if smoke:
        return [{
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "implementation_sha256": implementation_hash,
            "dataset": "mackey_glass",
            "setting": {"name": "smoke_only", "time_steps": 90, "shots_per_step": 10},
            "cluster_index": 0,
            "reservoir_variant": str(config["primary_reservoir_variant"]),
            "reservoir_seed": int(config["primary_reservoir_seed"]),
            "feature_scopes": {"z_plus_zz": "headline"},
            "feature_methods": {
                "z_plus_zz": ["ideal_quantum", "ev", "raw", "split", "covariance_ev"]
            },
            "emit_classical": True,
        }]

    # Merge logical scopes sharing the same expensive raw shot record.  The
    # partition-robustness scope is evaluated inside its matching headline job.
    merged: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for scope_name, scope in config["evaluation_scopes"].items():
        if scope_name == "partition_robustness":
            continue
        role = str(scope["role"])
        for dataset in config["datasets"]:
            for setting_name in scope["budget_settings"]:
                setting = settings_by_name[str(setting_name)]
                for cluster in config["trajectory_indices"]:
                    for reservoir_seed in scope["reservoir_seeds"]:
                        key = (
                            str(dataset),
                            setting.name,
                            int(cluster),
                            str(scope["reservoir_variant"]),
                            int(reservoir_seed),
                        )
                        if key not in merged:
                            merged[key] = {
                                "config_path": str(config_path),
                                "config_sha256": config_sha256,
                                "implementation_sha256": implementation_hash,
                                "dataset": str(dataset),
                                "setting": {
                                    "name": setting.name,
                                    "time_steps": setting.time_steps,
                                    "shots_per_step": setting.shots_per_step,
                                },
                                "cluster_index": int(cluster),
                                "reservoir_variant": str(scope["reservoir_variant"]),
                                "reservoir_seed": int(reservoir_seed),
                                "feature_scopes": {},
                                "feature_methods": {},
                                "emit_classical": False,
                            }
                        job = merged[key]
                        for feature_set in scope["feature_sets"]:
                            feature = str(feature_set)
                            existing_role = job["feature_scopes"].get(feature)
                            if existing_role is not None and existing_role != role:
                                raise ValueError(f"feature {feature} receives conflicting scope roles")
                            job["feature_scopes"][feature] = role
                            methods = set(job["feature_methods"].get(feature, []))
                            methods.update(
                                str(method) for method in scope["methods"] if method != "classical_input"
                            )
                            job["feature_methods"][feature] = sorted(methods)
                        if (
                            "classical_input" in scope["methods"]
                            and int(reservoir_seed) == int(config["primary_reservoir_seed"])
                        ):
                            job["emit_classical"] = True
    return [merged[key] for key in sorted(merged)]


def make_manifest(
    config: Mapping[str, object],
    *,
    config_path: Path,
    results_dir: Path,
    status: str,
    gate: Mapping[str, object],
    jobs: Sequence[Mapping[str, object]],
    architectures: Sequence[Mapping[str, object]] = (),
) -> Dict[str, object]:
    settings = [robust.BudgetSetting(**item) for item in config["budget_settings"]]
    designs = {
        setting.name: robust.make_temporal_design(
            setting.time_steps,
            t0=int(config["t0"]),
            test_fraction=float(config["test_fraction"]),
            validation_folds=int(config["validation_folds"]),
            initial_train_fraction=float(config["initial_train_fraction"]),
        ).manifest_dict()
        for setting in settings
    }
    nested_design = {
        "trajectory_indices": list(config["trajectory_indices"]),
        "reservoir_seeds": list(config["reservoir_seeds"]),
        "shot_seeds": list(config["shot_seeds"]),
        "partition_seeds": list(config["partition_seeds"]),
        "partition_robustness_subset": config["partition_robustness_subset"],
        "fixed_lane_ablation": config["fixed_lane_ablation"],
    }
    data_provenance = {}
    if "sunspots" in config["datasets"]:
        nested_design.update(
            {
                "sunspot_origin_spacing_months": int(config["sunspot_origin_spacing_months"]),
                "sunspot_test_index_ranges_by_setting": sunspot_test_index_ranges(config),
                "sunspot_test_blocks_pairwise_disjoint": True,
            }
        )
        data_provenance["sunspots"] = verify_sunspot_dataset_pin(config)
    if "santa_fe_laser" in config["datasets"]:
        nested_design.update(
            {
                "santa_fe_origin_spacing_samples": int(config["santa_fe_origin_spacing_samples"]),
                "santa_fe_test_index_ranges_by_setting": santa_fe_test_index_ranges(config),
                "santa_fe_full_windows_pairwise_disjoint": True,
            }
        )
        data_provenance["santa_fe_laser"] = verify_santa_fe_dataset_pin(config)
    return {
        "schema_version": "qai-robust-protocol-v1",
        "status": status,
        "script": "scripts/run_robust_evaluation.py",
        "config_path": path_label(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "implementation_sha256": implementation_sha256(),
        "git_commit": git_commit(),
        "protocol": {
            "datasets": list(config["datasets"]),
            "reservoir_variants": list(config["reservoir_variants"]),
            "reservoir_roles": dict(config["reservoir_roles"]),
            "primary_reservoir_variant": config["primary_reservoir_variant"],
            "primary_reservoir_seed": int(config["primary_reservoir_seed"]),
            "feature_sets": list(config["feature_sets"]),
            "primary_feature_set": config["primary_feature_set"],
            "methods": list(config["methods"]),
            "temporal_design_by_setting": designs,
            "normalization": config["normalization"],
            "partitioning": config["partitioning"],
            "registered_objectives": config["registered_objectives"],
            "evaluation_scopes": config["evaluation_scopes"],
        },
        "nested_design": nested_design,
        "budget_settings": [
            {
                "name": setting.name,
                "time_steps": setting.time_steps,
                "shots_per_step": setting.shots_per_step,
                "executions": setting.executions,
            }
            for setting in settings
        ],
        "hyperparameter_search": {
            "validation_folds": 3,
            "fold_kind": "expanding_train_blocked_validation",
            "lambda_grid": list(config["lambda_grid"]),
            "covariance_alpha_grid": list(config["covariance_alpha_grid"]),
            "k_grid_by_setting": {
                setting.name: list(registered_k_grid(config, setting))
                for setting in settings
            },
            "candidate_budget": dict(config["tuning_budget"]),
            "split_endpoint_policy": "k=1 and k=N are allowed safe fallbacks; interior selections are reported",
        },
        "test_access": "once_after_all_applicable_hyperparameters_are_frozen_on_validation",
        "inference": {
            "cluster": "trajectory_or_sunspot_window",
            "nested_factors": ["reservoir_seed", "shot_seed", "partition_seed"],
            "ci": "95_percent_t_interval_over_cluster_means",
        },
        "feature_validity_gate": dict(gate),
        "decision_gate": dict(config["decision_gate"]),
        "data_provenance": data_provenance,
        "planned_worker_jobs": int(len(jobs)),
        "planned_scope_feature_jobs": {
            role: int(
                sum(
                    1
                    for job in jobs
                    for job_role in job.get("feature_scopes", {}).values()
                    if job_role == role
                )
            )
            for role in ("headline", "feature_ablation", "degenerate_circuit_ablation")
        },
        "checkpointing": {
            "unit": "dataset_x_budget_x_cluster_x_reservoir_variant_x_reservoir_seed",
            "directory": path_label(results_dir / "shards"),
            "resume_flag": "--resume",
            "config_sha256_in_job_id": True,
            "implementation_sha256_in_job_id": True,
        },
        "architectures": list(architectures),
        "outputs": {
            "replicate_results": path_label(results_dir / "replicate_results.csv"),
            "cluster_summary": path_label(results_dir / "cluster_summary.csv"),
            "cluster_paired_stats": path_label(results_dir / "cluster_paired_stats.csv"),
            "partition_robustness_summary": path_label(results_dir / "partition_robustness_summary.csv"),
            "fixed_lane_sensitivity": path_label(results_dir / "fixed_lane_sensitivity.csv"),
            "ideal_vs_classical": path_label(results_dir / "ideal_vs_classical.csv"),
            "selected_k_frequency": path_label(results_dir / "selected_k_frequency.csv"),
            "split_validation_spectrum": path_label(results_dir / "split_validation_spectrum.csv"),
        },
    }


def main() -> None:
    args = parse_args()
    if args.allow_unvalidated_reservoir and not args.smoke:
        raise ValueError("--allow-unvalidated-reservoir is restricted to --smoke")
    config = load_config(args.config)
    gate = feature_gate(
        config,
        smoke=bool(args.smoke or args.dry_run),
        allow_override=bool(args.allow_unvalidated_reservoir or args.dry_run),
    )
    jobs = planned_jobs(config, args.config, smoke=bool(args.smoke))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    initial_status = "smoke-test" if args.smoke else "planned" if args.dry_run else "running"
    manifest = make_manifest(
        config,
        config_path=args.config,
        results_dir=args.results_dir,
        status=initial_status,
        gate=gate,
        jobs=jobs,
    )
    (args.results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    rows: List[Dict[str, object]] = []
    spectrum_rows: List[Dict[str, object]] = []
    architectures: Dict[str, Dict[str, object]] = {}
    shard_dir = args.results_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    pending_jobs: List[Dict[str, object]] = []
    resumed_jobs = 0
    for job in jobs:
        shard_path = shard_dir / f"{job_id(job)}.json"
        if args.resume and shard_path.exists():
            payload = json.loads(shard_path.read_text(encoding="utf-8"))
            rows.extend(payload["rows"])
            spectrum_rows.extend(payload.get("spectrum_rows", []))
            architecture = payload["architecture"]
            architectures[json.dumps(architecture, sort_keys=True)] = architecture
            resumed_jobs += 1
        else:
            pending_jobs.append(dict(job))
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        for index, (job, result) in enumerate(zip(pending_jobs, pool.map(run_job, pending_jobs)), start=1):
            job_rows, job_spectrum_rows, architecture = result
            rows.extend(job_rows)
            spectrum_rows.extend(job_spectrum_rows)
            key = json.dumps(architecture, sort_keys=True)
            architectures[key] = architecture
            shard_payload = {
                "job_id": job_id(job),
                "job": job,
                "rows": job_rows,
                "spectrum_rows": job_spectrum_rows,
                "architecture": architecture,
            }
            (shard_dir / f"{job_id(job)}.json").write_text(
                json.dumps(shard_payload, sort_keys=True), encoding="utf-8"
            )
            print(
                f"robust protocol progress {resumed_jobs + index}/{len(jobs)} "
                f"({resumed_jobs} resumed)",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    sort_columns = [
        "dataset",
        "budget_setting",
        "cluster_id",
        "reservoir_variant",
        "reservoir_seed",
        "shot_seed",
        "partition_seed",
        "feature_set",
        "method",
    ]
    frame = frame.sort_values(sort_columns, na_position="first").reset_index(drop=True)
    frame.to_csv(args.results_dir / "replicate_results.csv", index=False)
    spectrum_frame = pd.DataFrame(spectrum_rows)
    if len(spectrum_frame):
        spectrum_frame = spectrum_frame.sort_values(
            [
                "dataset",
                "budget_setting",
                "cluster_id",
                "reservoir_seed",
                "shot_seed",
                "partition_seed",
                "k",
            ]
        ).reset_index(drop=True)
    spectrum_frame.to_csv(args.results_dir / "split_validation_spectrum.csv", index=False)
    # Headline inference uses only the three preregistered per-time partitions.
    # The 20-partition and fixed-lane rows are reported separately and cannot
    # silently reweight the primary estimator.
    main_frame = frame[frame["partition_role"].isin(["primary", "not_applicable"])].copy()
    summary_grouping = (
        "dataset",
        "budget_setting",
        "reservoir_variant",
        "feature_set",
        "scope_role",
        "method",
    )
    paired_grouping = summary_grouping[:-1]
    summary = robust.clustered_method_summary(main_frame, grouping=summary_grouping)
    summary.to_csv(args.results_dir / "cluster_summary.csv", index=False)
    paired = pd.concat(
        [
            robust.clustered_paired_comparison(main_frame, grouping=paired_grouping),
            robust.clustered_paired_comparison(
                main_frame,
                baseline="split",
                methods=("covariance_ev",),
                grouping=paired_grouping,
            ),
        ],
        ignore_index=True,
    )
    paired.to_csv(args.results_dir / "cluster_paired_stats.csv", index=False)
    ideal_classical = robust.clustered_ideal_classical_comparison(
        main_frame,
        reservoir_variant=str(config["primary_reservoir_variant"]),
        feature_set=str(config["primary_feature_set"]),
    )
    ideal_classical.to_csv(args.results_dir / "ideal_vs_classical.csv", index=False)
    headline_split = main_frame[
        (main_frame["scope_role"] == "headline")
        & (main_frame["method"] == "split")
        & (main_frame["partition_role"] == "primary")
    ].copy()
    frequency_keys = [
        "dataset",
        "budget_setting",
        "reservoir_variant",
        "feature_set",
        "scope_role",
    ]
    k_frequency = (
        headline_split.groupby(frequency_keys + ["k"], dropna=False)
        .agg(
            selections=("k", "size"),
            clusters=("cluster_id", "nunique"),
            selection_margin_mean=("selection_margin", "mean"),
            selection_margin_median=("selection_margin", "median"),
        )
        .reset_index()
    )
    totals = k_frequency.groupby(frequency_keys, dropna=False)["selections"].transform("sum")
    k_frequency["selection_frequency"] = k_frequency["selections"] / totals
    k_frequency.to_csv(args.results_dir / "selected_k_frequency.csv", index=False)
    robustness_frame = frame[frame["partition_role"] == "twenty_partition_robustness"].copy()
    if len(robustness_frame):
        robustness_summary = robust.clustered_method_summary(
            robustness_frame, grouping=summary_grouping
        )
    else:
        robustness_summary = pd.DataFrame()
    robustness_summary.to_csv(args.results_dir / "partition_robustness_summary.csv", index=False)
    fixed_sensitivity = robust.clustered_partition_sensitivity(frame)
    fixed_sensitivity.to_csv(args.results_dir / "fixed_lane_sensitivity.csv", index=False)

    final_status = "smoke-test" if args.smoke else "complete"
    manifest = make_manifest(
        config,
        config_path=args.config,
        results_dir=args.results_dir,
        status=final_status,
        gate=gate,
        jobs=jobs,
        architectures=list(architectures.values()),
    )
    manifest["rows"] = int(len(frame))
    manifest["split_validation_spectrum_rows"] = int(len(spectrum_frame))
    manifest["resumed_worker_jobs"] = int(resumed_jobs)
    manifest["new_worker_jobs"] = int(len(pending_jobs))
    manifest["clusters_per_task"] = int(len(config["trajectory_indices"]))
    manifest["main_analysis_filter"] = (
        "partition_role in {primary, not_applicable}; scope_role retained in every grouped summary"
    )
    manifest["interior_split_selection_rate"] = float(
        main_frame.loc[main_frame["method"] == "split", "interior_k_selected"].mean()
    )
    (args.results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(frame)} replicate rows to {args.results_dir}", flush=True)


if __name__ == "__main__":
    main()
