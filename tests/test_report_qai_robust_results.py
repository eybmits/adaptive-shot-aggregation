from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_qai_robust_results as report
import run_robust_evaluation as runner


def _row(
    *,
    dataset: str = "mackey_glass",
    budget: str = "balanced",
    cluster: str = "cluster_0",
    method: str = "split",
    score: float = 0.4,
    reservoir: str = report.PRIMARY_RESERVOIR,
    feature: str = report.PRIMARY_FEATURE_SET,
    scope: str = report.PRIMARY_SCOPE,
    reservoir_seed: float = 1729,
    shot_seed: float = 1009,
    partition_seed: float = 17,
    role: str = "primary",
    mode: str = "per_time_permuted",
    k: int = 5,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "budget_setting": budget,
        "cluster_id": cluster,
        "reservoir_variant": reservoir,
        "reservoir_seed": reservoir_seed,
        "shot_seed": shot_seed,
        "partition_seed": partition_seed,
        "partition_mode": mode,
        "partition_role": role,
        "scope_role": scope,
        "feature_set": feature,
        "method": method,
        "k": k,
        "test_nrmse": score,
    }


def test_headline_filter_never_mixes_legacy_or_ablation_rows() -> None:
    rows = [
        _row(method="split"),
        _row(method="ev"),
        _row(method="split", reservoir="degenerate_gate", scope="degenerate_circuit_ablation"),
        _row(method="split", feature="z0", scope="feature_ablation"),
        _row(method="split", reservoir="legacy_random", scope="headline"),
        _row(
            method="classical_input",
            reservoir="not_applicable",
            feature="classical_input",
            reservoir_seed=np.nan,
            shot_seed=np.nan,
            partition_seed=np.nan,
            role="not_applicable",
            mode="not_applicable",
            k=1,
        ),
    ]
    selected = report.headline_filter(pd.DataFrame(rows))
    assert set(selected["method"]) == {"split", "ev", "classical_input"}
    quantum = selected[selected["method"] != "classical_input"]
    assert set(quantum["reservoir_variant"]) == {report.PRIMARY_RESERVOIR}
    assert set(quantum["feature_set"]) == {report.PRIMARY_FEATURE_SET}


def test_cluster_summary_averages_nested_seeds_before_ci() -> None:
    rows = []
    rows.extend(_row(cluster="a", method="ev", score=1.0, partition_seed=i) for i in range(10))
    rows.append(_row(cluster="b", method="ev", score=3.0, partition_seed=99))
    cluster, summary = report.cluster_method_summary(pd.DataFrame(rows))
    assert cluster.set_index("cluster_id").loc["a", "cluster_nrmse"] == pytest.approx(1.0)
    assert cluster.set_index("cluster_id").loc["b", "cluster_nrmse"] == pytest.approx(3.0)
    assert summary.iloc[0]["mean_nrmse"] == pytest.approx(2.0)
    assert pd.DataFrame(rows)["test_nrmse"].mean() != pytest.approx(2.0)


def test_validation_spectrum_averages_nested_seeds_before_ci() -> None:
    rows = []
    for cluster, values in {"a": [1.0, 1.0, 1.0], "b": [3.0]}.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "dataset": "mackey_glass",
                    "budget_setting": "balanced",
                    "cluster_id": cluster,
                    "k": 5,
                    "mean_val_nrmse": value,
                    "reservoir_variant": report.PRIMARY_RESERVOIR,
                    "reservoir_seed": 1729,
                    "shot_seed": 1009 + index,
                    "partition_seed": 17,
                    "partition_mode": "per_time_permuted",
                    "scope_role": "headline",
                    "feature_set": report.PRIMARY_FEATURE_SET,
                    "lambda_selected": 0.1,
                    "metric_role": "validation_only_no_test_access",
                }
            )
    cluster, summary = report.validation_spectrum_summary(pd.DataFrame(rows))
    assert summary.iloc[0]["mean_val_nrmse"] == pytest.approx(2.0)
    assert len(cluster) == 2


def test_selection_frequency_retains_registered_zero_count_candidates() -> None:
    rows = [
        _row(method="split", k=4, partition_seed=17),
        _row(method="split", k=5, partition_seed=43),
        _row(method="split", k=5, partition_seed=97),
    ]
    registered = {"balanced": [1, 2, 4, 5, 10, 20]}
    frequency = report.selected_k_frequency(pd.DataFrame(rows), registered)

    assert list(frequency["k"]) == registered["balanced"]
    assert int(frequency.loc[frequency["k"] == 1, "selections"].iloc[0]) == 0
    assert float(
        frequency.loc[frequency["k"] == 1, "selection_frequency"].iloc[0]
    ) == 0.0
    assert frequency["selection_frequency"].sum() == pytest.approx(1.0)


def test_reporter_refuses_running_or_smoke_campaign(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    diagnostics = ROOT / "results" / "expressive_reservoir" / "ideal_diagnostics.json"
    grid = ROOT / "results" / "expressive_reservoir" / "ideal_feature_grid.csv"
    (results / "manifest.json").write_text(
        json.dumps({"status": "running", "feature_validity_gate": {"passed": True, "hash_match": True}})
    )
    pd.DataFrame().to_csv(results / "replicate_results.csv", index=False)
    pd.DataFrame().to_csv(results / "split_validation_spectrum.csv", index=False)
    with pytest.raises(report.ReportingContractError, match="non-complete"):
        report.validate_complete_inputs(results, diagnostics, grid)


def _write_complete_synthetic_campaign(results: Path) -> tuple[Path, Path]:
    diagnostics_path = ROOT / "results" / "expressive_reservoir" / "ideal_diagnostics.json"
    feature_grid_path = ROOT / "results" / "expressive_reservoir" / "ideal_feature_grid.csv"
    diagnostics = json.loads(diagnostics_path.read_text())
    config = json.loads(report.DEFAULT_CONFIG.read_text())
    reservoir_seeds = tuple(int(value) for value in config["reservoir_seeds"])
    shot_seeds = tuple(int(value) for value in config["shot_seeds"])
    partition_seeds = tuple(int(value) for value in config["partition_seeds"])
    robust_seeds = tuple(
        int(value) for value in config["partition_robustness_subset"]["partition_seeds"]
    )
    k_grid = {
        str(key): tuple(int(value) for value in values)
        for key, values in config["organization_k_grid_by_setting"].items()
    }

    rows: list[dict[str, object]] = []
    spectrum: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(report.DATASET_ORDER):
        for budget_index, budget in enumerate(report.BUDGET_ORDER):
            for cluster_index in range(5):
                cluster = f"{dataset}_cluster_{cluster_index}"
                base = 0.4 + 0.02 * dataset_index + 0.01 * budget_index + 0.005 * cluster_index
                rows.append(
                    _row(
                        dataset=dataset,
                        budget=budget,
                        cluster=cluster,
                        method="classical_input",
                        score=base + 0.08,
                        reservoir="not_applicable",
                        feature="classical_input",
                        reservoir_seed=np.nan,
                        shot_seed=np.nan,
                        partition_seed=np.nan,
                        role="not_applicable",
                        mode="not_applicable",
                        k=1,
                    )
                )
                for reservoir_seed in reservoir_seeds:
                    rows.append(
                        _row(
                            dataset=dataset,
                            budget=budget,
                            cluster=cluster,
                            method="ideal_quantum",
                            score=base,
                            reservoir_seed=reservoir_seed,
                            shot_seed=np.nan,
                            partition_seed=np.nan,
                            role="not_applicable",
                            mode="not_applicable",
                            k=1,
                        )
                    )
                    for shot_seed in shot_seeds:
                        for partition_seed in partition_seeds:
                            score_by_method = {
                                "ev": base + 0.10,
                                "raw": base + 0.14,
                                "split": base + 0.05,
                                "covariance_ev": base + 0.055,
                            }
                            for method, score in score_by_method.items():
                                rows.append(
                                    _row(
                                        dataset=dataset,
                                        budget=budget,
                                        cluster=cluster,
                                        method=method,
                                        score=score,
                                        reservoir_seed=reservoir_seed,
                                        shot_seed=shot_seed,
                                        partition_seed=partition_seed,
                                        k=5 if method == "split" else (1 if method == "raw" else 20),
                                    )
                                )
                            for k in k_grid[budget]:
                                spectrum.append(
                                    {
                                        "dataset": dataset,
                                        "budget_setting": budget,
                                        "cluster_id": cluster,
                                        "reservoir_variant": report.PRIMARY_RESERVOIR,
                                        "reservoir_seed": reservoir_seed,
                                        "shot_seed": shot_seed,
                                        "partition_seed": partition_seed,
                                        "partition_mode": "per_time_permuted",
                                        "partition_role": "primary",
                                        "scope_role": "headline",
                                        "feature_set": report.PRIMARY_FEATURE_SET,
                                        "method": "split",
                                        "k": k,
                                        "lambda_selected": 0.1,
                                        "mean_val_nrmse": base + abs(np.log2(k) - 2.0) * 0.01,
                                        "metric_role": "validation_only_no_test_access",
                                    }
                                )
                if budget == "balanced":
                    for shot_seed in shot_seeds:
                        for partition_seed in robust_seeds:
                            for method, score in (("ev", base + 0.10), ("split", base + 0.05)):
                                rows.append(
                                    _row(
                                        dataset=dataset,
                                        budget=budget,
                                        cluster=cluster,
                                        method=method,
                                        score=score,
                                        reservoir_seed=1729,
                                        shot_seed=shot_seed,
                                        partition_seed=partition_seed,
                                        scope="partition_robustness",
                                        role="twenty_partition_robustness",
                                        k=5 if method == "split" else 20,
                                    )
                                )
    replicate = pd.DataFrame(rows)
    spectrum_frame = pd.DataFrame(spectrum)
    results.mkdir(parents=True)
    replicate.to_csv(results / "replicate_results.csv", index=False)
    spectrum_frame.to_csv(results / "split_validation_spectrum.csv", index=False)
    manifest = {
        "schema_version": "qai-robust-protocol-v1",
        "status": "complete",
        "config_path": str(report.DEFAULT_CONFIG.relative_to(ROOT)),
        "config_sha256": report._sha256(report.DEFAULT_CONFIG),
        "implementation_sha256": runner.implementation_sha256(),
        "rows": len(replicate),
        "clusters_per_task": 5,
        "protocol": {
            "datasets": list(config["datasets"]),
            "primary_reservoir_variant": config["primary_reservoir_variant"],
            "primary_reservoir_seed": config["primary_reservoir_seed"],
            "primary_feature_set": config["primary_feature_set"],
            "methods": list(config["methods"]),
            "normalization": dict(config["normalization"]),
            "evaluation_scopes": dict(config["evaluation_scopes"]),
        },
        "budget_settings": [
            {
                **item,
                "executions": int(item["time_steps"]) * int(item["shots_per_step"]),
            }
            for item in config["budget_settings"]
        ],
        "nested_design": {
            "trajectory_indices": list(config["trajectory_indices"]),
            "reservoir_seeds": list(reservoir_seeds),
            "shot_seeds": list(shot_seeds),
            "partition_seeds": list(partition_seeds),
            "sunspot_origin_spacing_months": config["sunspot_origin_spacing_months"],
            "partition_robustness_subset": dict(config["partition_robustness_subset"]),
            "fixed_lane_ablation": dict(config["fixed_lane_ablation"]),
        },
        "hyperparameter_search": {
            "lambda_grid": list(config["lambda_grid"]),
            "covariance_alpha_grid": list(config["covariance_alpha_grid"]),
            "k_grid_by_setting": {
                key: list(values) for key, values in k_grid.items()
            },
        },
        "decision_gate": dict(config["decision_gate"]),
        "data_provenance": {
            "sunspots": runner.verify_sunspot_dataset_pin(config),
        },
        "feature_validity_gate": {
            "passed": True,
            "hash_match": True,
            "observed_architecture_sha256": diagnostics["architecture"]["architecture_sha256"],
        },
    }
    (results / "manifest.json").write_text(json.dumps(manifest))
    return diagnostics_path, feature_grid_path


def test_complete_synthetic_campaign_passes_strict_contract(tmp_path: Path) -> None:
    results = tmp_path / "full"
    diagnostics, grid = _write_complete_synthetic_campaign(results)
    inputs = report.validate_complete_inputs(results, diagnostics, grid)
    assert set(inputs.headline["scope_role"]) == {"headline"}
    assert set(inputs.headline["reservoir_variant"]) == {
        "not_applicable",
        report.PRIMARY_RESERVOIR,
    }
    summary = report.build_summary(inputs)
    assert summary["status"] == "GO"
    assert summary["inference"]["independent_unit"] == "trajectory_or_sunspot_window"


def test_report_manifest_pins_live_reporter_source(tmp_path: Path) -> None:
    results = tmp_path / "full"
    diagnostics, grid = _write_complete_synthetic_campaign(results)
    inputs = report.validate_complete_inputs(results, diagnostics, grid)
    manifest = report._build_report_manifest(inputs, tmp_path / "paper", ())

    reporter_label = report._path_label(report.REPORTER_PATH)
    live_hash = report._sha256(report.REPORTER_PATH)
    assert reporter_label == "scripts/report_qai_robust_results.py"
    assert manifest["source_sha256"][reporter_label] == live_hash
    assert manifest["reporter"] == {"path": reporter_label, "sha256": live_hash}


def test_report_artifact_inventory_excludes_manifest_self_hash(tmp_path: Path) -> None:
    (tmp_path / "summary_go_no_go.json").write_text("{}")
    (tmp_path / "report_manifest.json").write_text("{}")
    artifacts = report._report_artifact_paths(tmp_path)

    assert [path.name for path in artifacts] == ["summary_go_no_go.json"]


def test_contract_rejects_one_missing_nested_seed_row(tmp_path: Path) -> None:
    results = tmp_path / "full"
    diagnostics, grid = _write_complete_synthetic_campaign(results)
    frame = pd.read_csv(results / "replicate_results.csv")
    drop_index = frame[
        (frame["scope_role"] == "headline") & (frame["method"] == "split")
    ].index[0]
    frame = frame.drop(drop_index)
    frame.to_csv(results / "replicate_results.csv", index=False)
    manifest = json.loads((results / "manifest.json").read_text())
    manifest["rows"] = len(frame)
    (results / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(report.ReportingContractError, match="exact method-specific seed set"):
        report.validate_complete_inputs(results, diagnostics, grid)


def test_contract_rejects_live_config_implementation_and_data_hash_mismatch(
    tmp_path: Path,
) -> None:
    results = tmp_path / "full"
    diagnostics, grid = _write_complete_synthetic_campaign(results)
    path = results / "manifest.json"
    original = json.loads(path.read_text())

    tampered = dict(original)
    tampered["config_sha256"] = "0" * 64
    path.write_text(json.dumps(tampered))
    with pytest.raises(report.ReportingContractError, match="live config SHA-256"):
        report.validate_complete_inputs(results, diagnostics, grid)

    tampered = dict(original)
    tampered["implementation_sha256"] = "0" * 64
    path.write_text(json.dumps(tampered))
    with pytest.raises(report.ReportingContractError, match="live implementation SHA-256"):
        report.validate_complete_inputs(results, diagnostics, grid)

    tampered = json.loads(json.dumps(original))
    tampered["data_provenance"]["sunspots"]["observed_sha256"] = "0" * 64
    path.write_text(json.dumps(tampered))
    with pytest.raises(report.ReportingContractError, match="sunspot data SHA-256"):
        report.validate_complete_inputs(results, diagnostics, grid)


def test_contract_rejects_same_count_wrong_seed_product_and_partition_semantics(
    tmp_path: Path,
) -> None:
    results = tmp_path / "full"
    diagnostics, grid = _write_complete_synthetic_campaign(results)
    path = results / "replicate_results.csv"
    original = pd.read_csv(path)
    target = original[
        (original["scope_role"] == "headline")
        & (original["method"] == "split")
        & (original["partition_seed"] == 17)
    ].index[0]

    wrong_seed = original.copy()
    wrong_seed.loc[target, "partition_seed"] = 43
    wrong_seed.to_csv(path, index=False)
    with pytest.raises(report.ReportingContractError, match="exact method-specific seed set"):
        report.validate_complete_inputs(results, diagnostics, grid)

    wrong_mode = original.copy()
    wrong_mode.loc[target, "partition_mode"] = "fixed_lane"
    wrong_mode.to_csv(path, index=False)
    with pytest.raises(report.ReportingContractError, match="headline partition_mode"):
        report.validate_complete_inputs(results, diagnostics, grid)

    wrong_role = original.copy()
    wrong_role.loc[target, "partition_role"] = "fixed_lane_ablation"
    wrong_role.to_csv(path, index=False)
    with pytest.raises(report.ReportingContractError, match="headline partition_role"):
        report.validate_complete_inputs(results, diagnostics, grid)


def test_balanced_confirmatory_margins_and_descriptive_budget_separation(
    tmp_path: Path,
) -> None:
    results = tmp_path / "full"
    diagnostics, grid = _write_complete_synthetic_campaign(results)
    inputs = report.validate_complete_inputs(results, diagnostics, grid)

    descriptive_failure = inputs.headline.copy()
    descriptive_failure.loc[
        (descriptive_failure["budget_setting"] != "balanced")
        & (descriptive_failure["method"] == "split"),
        "test_nrmse",
    ] += 1.0
    descriptive_summary = report.build_summary(replace(inputs, headline=descriptive_failure))
    assert descriptive_summary["status"] == "GO"
    assert descriptive_summary["reporting_filter"]["primary_confirmatory_setting"] == "balanced"

    noninferiority_failure = inputs.headline.copy()
    noninferiority_failure.loc[
        (noninferiority_failure["budget_setting"] == "balanced")
        & (noninferiority_failure["dataset"] == "mackey_glass")
        & (noninferiority_failure["method"] == "split"),
        "test_nrmse",
    ] += 0.11
    noninferiority_summary = report.build_summary(
        replace(inputs, headline=noninferiority_failure)
    )
    assert noninferiority_summary["status"] == "NO-GO"
    assert not noninferiority_summary["gates"]["split_vs_ev_balanced"]["passed"]

    no_supported_win = inputs.headline.copy()
    no_supported_win.loc[
        (no_supported_win["budget_setting"] == "balanced")
        & (no_supported_win["method"] == "split"),
        "test_nrmse",
    ] += 0.06
    no_win_summary = report.build_summary(replace(inputs, headline=no_supported_win))
    ev_gate = no_win_summary["gates"]["split_vs_ev_balanced"]
    assert ev_gate["tasks_noninferior"] == len(report.DATASET_ORDER)
    assert ev_gate["tasks_improved"] == 0
    assert not ev_gate["passed"]

    covariance_failure = inputs.headline.copy()
    covariance_failure.loc[
        (covariance_failure["budget_setting"] == "balanced")
        & (covariance_failure["dataset"] == "lorenz")
        & (covariance_failure["method"] == "covariance_ev"),
        "test_nrmse",
    ] -= 0.065
    covariance_summary = report.build_summary(replace(inputs, headline=covariance_failure))
    assert covariance_summary["status"] == "NO-GO"
    assert not covariance_summary["gates"]["split_vs_covariance_ev_balanced"]["passed"]
