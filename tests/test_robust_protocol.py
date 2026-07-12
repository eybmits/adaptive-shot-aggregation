from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import robust_protocol as robust
import run_robust_evaluation as runner
from robust_reservoir_adapter import make_reservoir_provider


def test_registered_budget_and_nested_design() -> None:
    config = runner.load_config(ROOT / "configs" / "qai_robust_protocol.json")
    settings = [robust.BudgetSetting(**item) for item in config["budget_settings"]]
    assert [(item.time_steps, item.shots_per_step) for item in settings] == [
        (250, 40),
        (500, 20),
        (1000, 10),
    ]
    assert {item.executions for item in settings} == {10000}
    assert len(config["trajectory_indices"]) == 5
    assert len(config["reservoir_seeds"]) == 3
    assert len(config["shot_seeds"]) == 3
    assert len(config["partition_seeds"]) == 3
    assert len(config["partition_robustness_subset"]["partition_seeds"]) == 20
    assert config["sunspot_origin_spacing_months"] == 240
    assert config["sunspot_dataset_pin"]["sha256"] == (
        "afb272b758d8db30446e43930091a98dbcf09210086da7fac8d5337f5532f68c"
    )
    assert config["decision_gate"]["primary_confirmatory_setting"] == "balanced"
    assert config["decision_gate"]["practical_noninferiority_margin_absolute_nrmse"] == 0.05
    assert config["organization_k_grid_by_setting"] == {
        "shot_rich": [1, 2, 4, 5, 8, 10, 20, 40],
        "balanced": [1, 2, 4, 5, 10, 20],
        "data_rich": [1, 2, 5, 10],
    }
    assert config["covariance_alpha_grid"] == [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    gate = runner.feature_gate(config, smoke=False, allow_override=False)
    assert gate["passed"] is True
    assert gate["hash_match"] is True


def test_current_benchmark_contains_exactly_the_paper_tasks() -> None:
    config = runner.load_config(ROOT / "configs" / "benchmark.json")
    assert config["status"] == "current-four-task-benchmark"
    assert config["datasets"] == [
        "mackey_glass",
        "santa_fe_laser",
        "rossler",
        "sunspots",
    ]


def test_temporal_design_has_three_expanding_blocked_folds() -> None:
    design = robust.make_temporal_design(250, t0=30)
    assert len(design.folds) == 3
    assert design.normalization_fit_stop == design.folds[0].train_stop
    assert design.folds[-1].validation_stop == design.test_start
    previous_stop = None
    for fold in design.folds:
        assert fold.train_start == 30
        assert fold.train_stop == fold.validation_start
        assert fold.validation_start < fold.validation_stop <= design.test_start
        if previous_stop is not None:
            assert fold.train_stop == previous_stop
        previous_stop = fold.validation_stop


def test_normalization_never_reads_validation_or_test_extrema() -> None:
    time_steps = 40
    fit_stop = 15
    raw = np.linspace(2.0, 4.0, time_steps + 1)
    inputs_a, targets_a, meta_a = robust.normalize_train_prefix(
        raw, time_steps=time_steps, fit_stop=fit_stop
    )
    changed = raw.copy()
    changed[fit_stop + 1 :] += 1000.0
    inputs_b, targets_b, meta_b = robust.normalize_train_prefix(
        changed, time_steps=time_steps, fit_stop=fit_stop
    )
    assert meta_a["normalization_min"] == meta_b["normalization_min"]
    assert meta_a["normalization_max"] == meta_b["normalization_max"]
    assert meta_a["circuit_inputs_clipped"] is True
    assert meta_a["targets_clipped"] is False
    assert "postfit_clipped_fraction" not in meta_a
    np.testing.assert_allclose(inputs_a[:fit_stop], inputs_b[:fit_stop])
    np.testing.assert_allclose(targets_a[: fit_stop - 1], targets_b[: fit_stop - 1])


def test_postfit_extrema_clip_inputs_but_not_train_scaled_targets() -> None:
    # The scale is fitted to [0,2]. Later observations exceed both ends.
    raw = np.array([0.0, 1.0, 2.0, 1.5, 5.0, -3.0, 1.0])
    inputs, targets, metadata = robust.normalize_train_prefix(
        raw,
        time_steps=6,
        fit_stop=2,
        clip_inputs=True,
    )
    assert np.all((0.0 <= inputs) & (inputs <= 1.0))
    assert float(np.max(targets)) == 2.5
    assert float(np.min(targets)) == -1.5
    assert metadata["targets_clipped"] is False
    assert metadata["input_postfit_clipped_fraction"] > 0.0
    assert metadata["target_postfit_outside_fraction"] > 0.0


def test_feature_ablation_uses_same_local_z_shots() -> None:
    z = np.array([[[1.0, -1.0, -1.0], [-1.0, -1.0, 1.0]]])
    assert robust.derive_feature_set(z, "all_z").shape[-1] == 3
    assert robust.derive_feature_set(z, "z0").shape[-1] == 1
    with_parity = robust.derive_feature_set(z, "z_plus_zz")
    assert with_parity.shape[-1] == 5
    np.testing.assert_allclose(with_parity[..., 3:], z[..., :-1] * z[..., 1:])


def test_group_mean_equals_ev_after_per_time_partition_and_leak_lags() -> None:
    rng = np.random.default_rng(91)
    shots = rng.choice((-1.0, 1.0), size=(12, 45, 4))
    ev = robust.organized_views_from_shots(
        shots,
        k=12,
        partition_seed=17,
        partition_mode="per_time_permuted",
        eta=0.2,
        window=5,
    )[0]
    for k in (1, 2, 3, 4, 6):
        grouped = robust.organized_views_from_shots(
            shots,
            k=k,
            partition_seed=17,
            partition_mode="per_time_permuted",
            eta=0.2,
            window=5,
        )
        np.testing.assert_allclose(grouped.mean(axis=0), ev, atol=1e-12)


def test_partition_modes_expose_arbitrary_lane_pairing() -> None:
    rng = np.random.default_rng(812)
    shots = rng.choice((-1.0, 1.0), size=(10, 40, 3))
    per_time = robust.organized_views_from_shots(
        shots, k=2, partition_seed=7, partition_mode="per_time_permuted", eta=0.2, window=4
    )
    fixed = robust.organized_views_from_shots(
        shots, k=2, partition_seed=7, partition_mode="fixed_lane", eta=0.2, window=4
    )
    assert not np.allclose(per_time, fixed)
    np.testing.assert_allclose(per_time.mean(axis=0), fixed.mean(axis=0), atol=1e-12)


def test_validation_selects_endpoints_or_interior_then_tests_once(monkeypatch) -> None:
    rng = np.random.default_rng(22)
    steps = rng.choice((-1.0, 1.0), size=(10, 90, 2))
    targets = rng.normal(size=90)
    design = robust.make_temporal_design(90, t0=10)
    calls = 0
    original = robust._grouped_test_once

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(robust, "_grouped_test_once", counted)
    result = robust.evaluate_shot_organization(
        steps,
        targets,
        design,
        k_grid=(1, 2, 5, 10),
        lambda_grid=(0.1, 1.0),
        partition_seed=97,
        partition_mode="per_time_permuted",
        eta=0.2,
        window=3,
    )
    assert result["k"] in {1, 2, 5, 10}
    assert result["lambda_selected"] in {0.1, 1.0}
    assert len(result["fold_val_nrmse"]) == 3
    assert result["selection_margin"] >= 0.0
    assert np.isfinite(result["second_best_val_nrmse"])
    assert [point["k"] for point in result["k_validation_spectrum"]] == [1, 2, 5, 10]
    assert all(len(point["fold_val_nrmse"]) == 3 for point in result["k_validation_spectrum"])
    assert calls == 1


def test_direct_covariance_ev_tunes_both_weights() -> None:
    rng = np.random.default_rng(404)
    phi = rng.normal(size=(8, 90, 5))
    targets = rng.normal(size=90)
    design = robust.make_temporal_design(90, t0=10)
    result = robust.evaluate_covariance_ev(
        phi,
        targets,
        design,
        lambda_grid=(0.1, 1.0),
        alpha_grid=(0.0, 1.0, 10.0),
    )
    assert result["lambda_selected"] in {0.1, 1.0}
    assert result["alpha_selected"] in {0.0, 1.0, 10.0}
    assert len(result["fold_val_nrmse"]) == 3
    assert np.isfinite(result["test_nrmse"])
    assert result["selection_margin"] >= 0.0


def test_fixed_k_split_equals_ev_plus_within_group_covariance_at_mean_loss_scale() -> None:
    rng = np.random.default_rng(505)
    innovations = rng.normal(size=(4, 80, 7))
    grouped = np.empty_like(innovations)
    grouped[:, 0, :] = innovations[:, 0, :]
    for time_index in range(1, grouped.shape[1]):
        grouped[:, time_index, :] = (
            0.75 * grouped[:, time_index - 1, :] + 0.25 * innovations[:, time_index, :]
        )
    targets = rng.normal(size=80)
    for ridge_alpha in (0.001, 0.1, 10.0):
        direct = robust._grouped_test_once(
            grouped,
            targets,
            train_start=8,
            test_start=60,
            ridge_alpha=ridge_alpha,
        )
        decomposed = robust.grouped_covariance_equivalent_prediction(
            grouped,
            targets,
            train_start=8,
            train_stop=60,
            eval_start=60,
            eval_stop=80,
            ridge_alpha=ridge_alpha,
        )
        np.testing.assert_allclose(decomposed["y_pred"], direct["y_pred"], rtol=1e-10, atol=1e-11)


def test_cluster_ci_counts_trajectories_not_nested_replicates() -> None:
    rows = []
    for cluster in range(5):
        for nested in range(12):
            rows.extend(
                [
                    {
                        "dataset": "mackey_glass",
                        "budget_setting": "balanced",
                        "reservoir_variant": "expressive_reuploading",
                        "feature_set": "z_plus_zz",
                        "cluster_id": f"trajectory_{cluster}",
                        "method": "ev",
                        "test_nrmse": 0.5 + 0.01 * cluster + 0.001 * nested,
                    },
                    {
                        "dataset": "mackey_glass",
                        "budget_setting": "balanced",
                        "reservoir_variant": "expressive_reuploading",
                        "feature_set": "z_plus_zz",
                        "cluster_id": f"trajectory_{cluster}",
                        "method": "split",
                        "test_nrmse": 0.4 + 0.01 * cluster + 0.001 * nested,
                    },
                ]
            )
    frame = pd.DataFrame(rows)
    summary = robust.clustered_method_summary(frame)
    assert set(summary["clusters"]) == {5}
    paired = robust.clustered_paired_comparison(frame, methods=("split",))
    assert int(paired.iloc[0]["clusters"]) == 5
    assert abs(float(paired.iloc[0]["delta_baseline_minus_method"]) - 0.1) < 1e-12


def test_fixed_lane_sensitivity_is_clustered() -> None:
    rows = []
    for cluster in range(5):
        for shot_seed in (1009, 2003, 3001):
            for partition_seed in (17, 43):
                for mode, role, value in (
                    ("per_time_permuted", "primary", 0.40 + 0.01 * cluster),
                    ("fixed_lane", "fixed_lane_ablation", 0.45 + 0.01 * cluster),
                ):
                    rows.append(
                        {
                            "dataset": "lorenz",
                            "budget_setting": "balanced",
                            "reservoir_variant": "expressive_reuploading",
                            "feature_set": "z_plus_zz",
                            "reservoir_seed": 1729,
                            "cluster_id": f"trajectory_{cluster}",
                            "shot_seed": shot_seed,
                            "partition_seed": partition_seed,
                            "method": "split",
                            "partition_mode": mode,
                            "partition_role": role,
                            "test_nrmse": value + shot_seed * 1e-7 + partition_seed * 1e-6,
                        }
                    )
        # Unmatched primary rows from another reservoir seed must not enter the
        # seed-1729 fixed-lane comparison.
        rows.append(
            {
                "dataset": "lorenz",
                "budget_setting": "balanced",
                "reservoir_variant": "expressive_reuploading",
                "feature_set": "z_plus_zz",
                "reservoir_seed": 2718,
                "cluster_id": f"trajectory_{cluster}",
                "shot_seed": 1009,
                "partition_seed": 17,
                "method": "split",
                "partition_mode": "per_time_permuted",
                "partition_role": "primary",
                "test_nrmse": 9.0,
            }
        )
    result = robust.clustered_partition_sensitivity(pd.DataFrame(rows))
    assert len(result) == 1
    assert int(result.iloc[0]["reservoir_seed"]) == 1729
    assert int(result.iloc[0]["clusters"]) == 5
    assert int(result.iloc[0]["matched_nested_pairs"]) == 30
    assert int(result.iloc[0]["matched_pairs_per_cluster_min"]) == 6
    assert abs(float(result.iloc[0]["delta_fixed_minus_per_time"]) - 0.05) < 1e-12


def test_ideal_vs_classical_uses_cluster_pairs_after_reservoir_averaging() -> None:
    rows = []
    for cluster in range(5):
        rows.append(
            {
                "dataset": "rossler",
                "budget_setting": "balanced",
                "cluster_id": f"trajectory_{cluster}",
                "reservoir_variant": "not_applicable",
                "feature_set": "classical_input",
                "method": "classical_input",
                "test_nrmse": 0.6 + 0.01 * cluster,
            }
        )
        for reservoir_seed in range(3):
            rows.append(
                {
                    "dataset": "rossler",
                    "budget_setting": "balanced",
                    "cluster_id": f"trajectory_{cluster}",
                    "reservoir_variant": "expressive_reuploading",
                    "feature_set": "z_plus_zz",
                    "method": "ideal_quantum",
                    "test_nrmse": 0.5 + 0.01 * cluster + 0.001 * reservoir_seed,
                }
            )
    result = robust.clustered_ideal_classical_comparison(pd.DataFrame(rows))
    assert int(result.iloc[0]["clusters"]) == 5
    assert abs(float(result.iloc[0]["delta_classical_minus_ideal"]) - 0.099) < 1e-12


def test_independent_clusters_and_sunspot_rolling_origins() -> None:
    mg0, id0, _ = runner.independent_series(
        "mackey_glass", time_steps=100, trajectory_index=0, sunspot_spacing=120
    )
    mg1, id1, _ = runner.independent_series(
        "mackey_glass", time_steps=100, trajectory_index=1, sunspot_spacing=120
    )
    assert id0 != id1
    assert not np.allclose(mg0, mg1)
    sun0, sid0, meta0 = runner.independent_series(
        "sunspots", time_steps=100, trajectory_index=0, sunspot_spacing=240
    )
    sun1, sid1, meta1 = runner.independent_series(
        "sunspots", time_steps=100, trajectory_index=1, sunspot_spacing=240
    )
    assert sid0 != sid1
    assert len(sun0) == len(sun1) == 101
    assert int(meta0["window_end_index"]) - int(meta1["window_end_index"]) == 240


def test_sunspot_test_blocks_are_pairwise_disjoint_at_every_budget() -> None:
    config = runner.load_config(ROOT / "configs" / "qai_robust_protocol.json")
    ranges_by_setting = runner.sunspot_test_index_ranges(config)
    assert max(row["length"] for rows in ranges_by_setting.values() for row in rows) == 194
    assert int(config["sunspot_origin_spacing_months"]) >= 194
    for rows in ranges_by_setting.values():
        ordered = sorted(rows, key=lambda row: row["start"])
        assert len(ordered) == 5
        assert all(left["stop"] <= right["start"] for left, right in zip(ordered, ordered[1:]))


def test_short_sunspot_origin_spacing_is_rejected(tmp_path: Path) -> None:
    source = json.loads((ROOT / "configs" / "qai_robust_protocol.json").read_text())
    source["sunspot_origin_spacing_months"] = 120
    path = tmp_path / "bad_spacing.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="shorter than the maximum test block"):
        runner.load_config(path)


def test_sunspot_dataset_pin_is_verified_and_tampering_is_rejected(tmp_path: Path) -> None:
    config_path = ROOT / "configs" / "qai_robust_protocol.json"
    config = runner.load_config(config_path)
    observed = runner.verify_sunspot_dataset_pin(config)
    assert observed["verified"] is True
    assert observed["observed_rows"] == 3330
    assert observed["observed_start_month"] == "1749-01"
    assert observed["observed_end_month"] == "2026-06"

    tampered = json.loads(config_path.read_text())
    tampered["sunspot_dataset_pin"]["sha256"] = "0" * 64
    path = tmp_path / "bad_pin.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runner.load_config(path)


def test_santa_fe_laser_pin_and_disjoint_continuation_windows() -> None:
    config_path = ROOT / "configs" / "qai_robust_protocol_santa_fe.json"
    config = runner.load_config(config_path)
    observed = runner.verify_santa_fe_dataset_pin(config)
    assert observed["verified"] is True
    assert observed["observed_train_rows"] == 1000
    assert observed["observed_continuation_rows"] == 9093
    assert observed["observed_total_rows"] == 10093
    assert observed["observed_min"] == 0.0
    assert observed["observed_max"] == 255.0

    ranges_by_setting = runner.santa_fe_test_index_ranges(config)
    for rows in ranges_by_setting.values():
        ordered = sorted(rows, key=lambda row: row["window_start"])
        assert len(ordered) == 5
        assert all(row["window_start"] >= 1000 for row in ordered)
        assert all(
            left["window_stop"] <= right["window_start"]
            for left, right in zip(ordered, ordered[1:])
        )

    window0, id0, meta0 = runner.independent_series(
        "santa_fe_laser", time_steps=1000, trajectory_index=0, sunspot_spacing=1001
    )
    window1, id1, meta1 = runner.independent_series(
        "santa_fe_laser", time_steps=1000, trajectory_index=1, sunspot_spacing=1001
    )
    assert id0 != id1
    assert len(window0) == len(window1) == 1001
    assert int(meta0["window_start_index"]) == int(meta1["window_end_index"])
    assert not np.array_equal(window0, window1)


def test_santa_fe_laser_tampered_pin_is_rejected(tmp_path: Path) -> None:
    source = json.loads(
        (ROOT / "configs" / "qai_robust_protocol_santa_fe.json").read_text()
    )
    source["santa_fe_dataset_pin"]["continuation_sha256"] = "0" * 64
    path = tmp_path / "bad_laser_pin.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="continuation SHA-256 mismatch"):
        runner.load_config(path)


def test_degenerate_adapter_supports_same_bitstring_ablation_interface() -> None:
    provider = make_reservoir_provider(
        "degenerate_gate", q=2, parameter_seed=1729, eta=0.2, window=2
    )
    inputs = np.linspace(0.1, 0.9, 5)
    shots = provider.shot_local_z(inputs, n_shots=4, shot_seed=9)
    ideal = provider.ideal_features(inputs, feature_set="z_plus_zz")
    assert shots.shape == (4, 5, 2)
    assert ideal.shape == (5, 3)
    assert set(np.unique(shots)).issubset({-1.0, 1.0})


def test_manifest_schema_declares_clustered_inference_and_test_lock() -> None:
    schema = json.loads((ROOT / "scripts" / "robust_protocol_manifest.schema.json").read_text())
    required = set(schema["required"])
    assert {"nested_design", "hyperparameter_search", "test_access", "inference"}.issubset(required)
    assert schema["properties"]["test_access"]["const"].startswith("once_after")
    assert schema["properties"]["inference"]["properties"]["cluster"]["const"] == (
        "trajectory_or_sunspot_window"
    )


def test_checkpoint_job_id_is_stable_and_content_addressed() -> None:
    job = {
        "dataset": "lorenz",
        "setting": {"name": "balanced", "time_steps": 500, "shots_per_step": 20},
        "cluster_index": 2,
        "reservoir_variant": "expressive_reuploading",
        "reservoir_seed": 1729,
        "config_path": "config.json",
    }
    same_reordered = dict(reversed(list(job.items())))
    assert runner.job_id(job) == runner.job_id(same_reordered)
    changed = {**job, "cluster_index": 3}
    assert runner.job_id(job) != runner.job_id(changed)


def test_planned_jobs_follow_explicit_headline_and_ablation_scope() -> None:
    config_path = ROOT / "configs" / "qai_robust_protocol.json"
    config = runner.load_config(config_path)
    jobs = runner.planned_jobs(config, config_path, smoke=False)
    assert len(jobs) == 200
    assert all(len(job["implementation_sha256"]) == 64 for job in jobs)
    expressive = [job for job in jobs if job["reservoir_variant"] == "expressive_reuploading"]
    degenerate = [job for job in jobs if job["reservoir_variant"] == "degenerate_gate"]
    assert len(expressive) == 180
    assert len(degenerate) == 20
    assert all(job["setting"]["name"] == "balanced" for job in degenerate)
    assert all(job["reservoir_seed"] == 1729 for job in degenerate)
    assert all(set(job["feature_scopes"]) == {"z0", "all_z", "z_plus_zz"} for job in degenerate)
    assert all(set(job["feature_scopes"].values()) == {"degenerate_circuit_ablation"} for job in degenerate)

    feature_ablation_jobs = [
        job for job in expressive if "feature_ablation" in set(job["feature_scopes"].values())
    ]
    assert len(feature_ablation_jobs) == 20
    assert all(job["setting"]["name"] == "balanced" for job in feature_ablation_jobs)
    assert all(job["reservoir_seed"] == 1729 for job in feature_ablation_jobs)
    assert all(job["feature_scopes"]["z_plus_zz"] == "headline" for job in expressive)
    assert sum(bool(job["emit_classical"]) for job in jobs) == 60

    roles = [role for job in jobs for role in job["feature_scopes"].values()]
    assert roles.count("headline") == 180
    assert roles.count("feature_ablation") == 40
    assert roles.count("degenerate_circuit_ablation") == 60


def test_partition_robustness_is_folded_only_into_matching_headline_job() -> None:
    config = runner.load_config(ROOT / "configs" / "qai_robust_protocol.json")
    feature_methods = {
        "z_plus_zz": ("ideal_quantum", "ev", "raw", "split", "covariance_ev"),
        "z0": ("ideal_quantum", "ev", "raw", "split", "covariance_ev"),
        "all_z": ("ideal_quantum", "ev", "raw", "split", "covariance_ev"),
    }
    primary = runner.partition_specs_for_job(
        config,
        setting_name="balanced",
        reservoir_variant="expressive_reuploading",
        reservoir_seed=1729,
        feature_methods=feature_methods,
    )
    assert sum(role == "primary" for _, _, role, _, _ in primary) == 3
    assert sum(role == "twenty_partition_robustness" for _, _, role, _, _ in primary) == 20
    assert sum(role == "fixed_lane_ablation" for _, _, role, _, _ in primary) == 3
    for _, _, role, scoped_methods, _ in primary:
        if role in {"twenty_partition_robustness", "fixed_lane_ablation"}:
            assert scoped_methods == {"z_plus_zz": ("ev", "split")}
    nonmatching = runner.partition_specs_for_job(
        config,
        setting_name="shot_rich",
        reservoir_variant="expressive_reuploading",
        reservoir_seed=1729,
        feature_methods={"z_plus_zz": feature_methods["z_plus_zz"]},
    )
    assert len(nonmatching) == 3


def test_validation_spectrum_rows_are_explicitly_test_free() -> None:
    common = {
        "dataset": "lorenz",
        "budget_setting": "balanced",
        "cluster_id": "lorenz_trajectory_00",
        "scope_role": "headline",
        "test_start": 406,
        "test_nrmse": 0.25,
        "target_postfit_outside_fraction": 0.4,
        "targets_clipped": False,
        "target_scaling": "train_prefix_minmax_affine_unclipped",
        "input_postfit_clipped_fraction": 0.1,
    }
    result = {
        "k_validation_spectrum": [
            {
                "k": 1,
                "lambda_selected": 0.1,
                "mean_val_nrmse": 0.5,
                "fold_val_nrmse": (0.4, 0.5, 0.6),
                "lambda_selection_margin": 0.02,
            }
        ]
    }
    rows = runner.validation_spectrum_rows(common, result)
    assert rows[0]["metric_role"] == "validation_only_no_test_access"
    assert "test_nrmse" not in rows[0]
    assert "test_start" not in rows[0]
    assert not any(column.startswith("target") for column in rows[0])
    assert not any("postfit" in column for column in rows[0])
