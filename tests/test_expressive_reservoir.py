from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import expressive_reservoir as expressive
import base_impl as base


def test_frozen_architecture_is_serializable_and_nonclifford() -> None:
    reservoir = expressive.make_expressive_reservoir()
    manifest = reservoir.architecture_manifest()

    assert manifest["parameter_seed"] == 1729
    assert manifest["q"] == 6
    assert manifest["layers"] == 2
    assert manifest["feature_set"] == "z_plus_zz"
    assert manifest["feature_dimension"] == 11
    assert len(manifest["parameter_sha256"]) == 64
    assert len(manifest["architecture_sha256"]) == 64
    assert manifest["layer_order"] == [
        "multi-qubit input RY/RZ",
        "nearest-neighbour line RZZ",
        "heterogeneous local RX/RY/RZ readout mixing",
    ]
    assert np.asarray(manifest["input_y_weights"]).shape == (2, 6)
    assert np.asarray(manifest["input_z_weights"]).shape == (2, 6)
    local = np.asarray(manifest["local_angles_rad"])
    coupling = np.asarray(manifest["coupling_angles_rad"])
    assert local.shape == (2, 6, 3)
    assert coupling.shape == (2, 5)
    assert np.all(expressive._distance_to_clifford_grid(local) >= 0.07 * np.pi - 1e-12)
    assert np.all(expressive._distance_to_clifford_grid(coupling) >= 0.07 * np.pi - 1e-12)


def test_state_and_parameters_are_deterministic_but_seed_sensitive() -> None:
    first = expressive.make_expressive_reservoir(seed=1729)
    second = expressive.make_expressive_reservoir(seed=1729)
    changed = expressive.make_expressive_reservoir(seed=1730)

    np.testing.assert_allclose(first.prepare_state(0.37), second.prepare_state(0.37))
    assert first.architecture_manifest()["parameter_sha256"] == second.architecture_manifest()["parameter_sha256"]
    assert not np.allclose(first.prepare_state(0.37), changed.prepare_state(0.37))
    assert first.architecture_manifest()["parameter_sha256"] != changed.architecture_manifest()["parameter_sha256"]


def test_all_features_come_from_one_z_basis_shot_record() -> None:
    reservoir = expressive.make_expressive_reservoir(q=4, feature_set="z_plus_zz")
    state = reservoir.prepare_state(0.61)
    rng_bits = np.random.default_rng(44)
    rng_features = np.random.default_rng(44)
    bits = reservoir.sample_bitstrings(state, n_shots=128, rng=rng_bits)
    features = reservoir.sample_measurements(
        state, n_shots=128, sigma=0.0, rng=rng_features
    )
    z = 1.0 - 2.0 * bits.astype(float)

    assert bits.shape == (128, 4)
    assert features.shape == (128, 7)
    np.testing.assert_array_equal(features[:, :4], z)
    np.testing.assert_array_equal(features[:, 4:], z[:, :-1] * z[:, 1:])


def test_feature_set_ablation_shapes_and_exact_expectations() -> None:
    state_reservoir = expressive.make_expressive_reservoir(q=4)
    state = state_reservoir.prepare_state(0.23)
    probabilities = np.abs(state) ** 2
    z_basis = state_reservoir._basis_z

    z0 = state_reservoir.ideal_expectations(state, feature_set="z0")
    all_z = state_reservoir.ideal_expectations(state, feature_set="all_z")
    z_plus_zz = state_reservoir.ideal_expectations(state, feature_set="z_plus_zz")
    assert z0.shape == (1,)
    assert all_z.shape == (4,)
    assert z_plus_zz.shape == (7,)
    np.testing.assert_allclose(z0, probabilities @ z_basis[:, :1])
    np.testing.assert_allclose(all_z, probabilities @ z_basis)
    np.testing.assert_allclose(
        z_plus_zz[4:], probabilities @ (z_basis[:, :-1] * z_basis[:, 1:])
    )


def test_existing_leak_lag_runner_accepts_expressive_interface() -> None:
    reservoir = expressive.make_expressive_reservoir(
        q=3, seed=1729, eta=0.2, window=2, feature_set="z_plus_zz"
    )
    inputs = np.linspace(0.1, 0.9, 8)
    phi = base.run_reservoir(
        inputs,
        reservoir,
        n_shots=5,
        sigma=0.0,
        shot_seed=9001,
    )

    assert phi.shape == (5, 8, 2 * (2 * 3 - 1))
    assert np.isfinite(phi).all()
    # Same frozen circuit and shot seed must reproduce the identical record.
    repeated = base.run_reservoir(
        inputs,
        expressive.make_expressive_reservoir(
            q=3, seed=1729, eta=0.2, window=2, feature_set="z_plus_zz"
        ),
        n_shots=5,
        sigma=0.0,
        shot_seed=9001,
    )
    np.testing.assert_array_equal(phi, repeated)


def test_second_layer_rzz_is_observable_under_z_readout() -> None:
    primary = expressive.make_expressive_reservoir(q=4, seed=1729)
    ablated = expressive.make_expressive_reservoir(q=4, seed=1729)
    ablated._rzz_phases = (
        ablated._rzz_phases[0],
        tuple(np.ones(ablated.dim, dtype=complex) for _ in ablated.edges),
    )
    inputs = np.linspace(0.0, 1.0, 33)
    primary_features = expressive.ideal_feature_matrix(primary, inputs)
    ablated_features = expressive.ideal_feature_matrix(ablated, inputs)
    effect = np.max(np.abs(primary_features - ablated_features), axis=0)

    assert np.max(effect) > expressive.COUPLING_SENSITIVITY_THRESHOLD
    assert np.count_nonzero(effect > expressive.COUPLING_SENSITIVITY_THRESHOLD) >= 3


def test_ideal_feature_validity_gate_passes_before_forecasting() -> None:
    diagnostics = expressive.diagnose_expressive_reservoir()

    assert diagnostics["passed"] is True
    assert diagnostics["diagnostic_status"] == "PASS"
    assert all(diagnostics["criteria"].values())
    assert diagnostics["input_dependent_fraction"] >= 0.75
    assert diagnostics["nonlinear_features"] >= 3
    assert diagnostics["numerical_rank"] >= 4
    assert diagnostics["effective_rank"] >= 2.0
    assert diagnostics["criteria"]["second_layer_couplings_change_z_features"]
    assert diagnostics["second_layer_coupling_sensitive_features"] > 0
    assert diagnostics["second_layer_coupling_max_effect"] > expressive.COUPLING_SENSITIVITY_THRESHOLD
    assert diagnostics["joint_classical_quantum_rank"] > diagnostics["classical_filtered_rank"]
    assert (
        diagnostics["classical_to_quantum_relative_projection_error"]
        > expressive.NON_EQUIVALENCE_RELATIVE_ERROR_THRESHOLD
    )
