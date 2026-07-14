#!/usr/bin/env python3
"""Frozen expressive gate-model reservoir and ideal feature diagnostics.

This module lives beside ``ExactQRCReservoir`` in :mod:`base_impl`, which is
used as the degenerate circuit control. The primary architecture here is small
enough for present IBM hardware and avoids single-input-qubit and
homogeneous-angle degeneracies:

* the normalized scalar input is encoded on every qubit and re-uploaded in
  each of exactly two layers;
* every layer has heterogeneous fixed non-Clifford single-qubit rotations;
* heterogeneous nearest-neighbour RZZ gates form a hardware-friendly line;
* one Z-basis shot yields all local Z and adjacent ZZ features.

The architecture seed is part of the scientific protocol and must not be
selected using forecasting results.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np

from base_impl import (
    apply_single_qubit_gate,
    rx,
    ry,
    rz,
    rzz_phase_vector,
)


EXPRESSIVE_RESERVOIR_SEED = 1729
EXPRESSIVE_RESERVOIR_Q = 6
EXPRESSIVE_RESERVOIR_LAYERS = 2
FEATURE_SETS = ("z0", "all_z", "z_plus_zz")

# These are declared diagnostic criteria, not fitted forecasting thresholds.
DEPENDENCE_RANGE_THRESHOLD = 1.0e-3
NONLINEAR_UNEXPLAINED_VARIANCE_THRESHOLD = 0.05
NON_EQUIVALENCE_RELATIVE_ERROR_THRESHOLD = 0.10
COUPLING_SENSITIVITY_THRESHOLD = 1.0e-3


def _distance_to_clifford_grid(angle: np.ndarray) -> np.ndarray:
    """Distance to the nearest multiple of pi/2."""

    scaled = np.asarray(angle, dtype=float) / (0.5 * np.pi)
    return np.abs(scaled - np.round(scaled)) * (0.5 * np.pi)


def _draw_nonclifford_angles(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    *,
    exclusion: float = 0.07 * np.pi,
) -> np.ndarray:
    """Draw reproducible angles bounded away from Clifford rotations.

    Rejection is deterministic for a fixed generator seed.  Keeping the
    angles away from multiples of pi/2 also prevents a rare continuous draw
    from becoming effectively Clifford after finite-precision serialization.
    """

    result = np.empty(int(np.prod(shape)), dtype=float)
    filled = 0
    while filled < result.size:
        candidates = rng.uniform(-0.95 * np.pi, 0.95 * np.pi, size=result.size - filled)
        valid = candidates[_distance_to_clifford_grid(candidates) >= exclusion]
        take = min(len(valid), result.size - filled)
        if take:
            result[filled : filled + take] = valid[:take]
            filled += take
    return result.reshape(shape)


@dataclass(frozen=True)
class ExpressiveArchitecture:
    """Serializable fixed parameters of the two-layer circuit."""

    parameter_seed: int
    q: int
    input_y_weights: np.ndarray
    input_z_weights: np.ndarray
    local_angles: np.ndarray
    coupling_angles: np.ndarray

    @classmethod
    def from_seed(cls, q: int, parameter_seed: int) -> "ExpressiveArchitecture":
        if q < 2:
            raise ValueError("the expressive reservoir requires at least two qubits")
        rng = np.random.default_rng(int(parameter_seed))
        # Both signs occur and every qubit receives a nonzero input rotation.
        input_y_weights = rng.uniform(0.55, 1.45, size=(EXPRESSIVE_RESERVOIR_LAYERS, q))
        input_z_weights = rng.uniform(0.20, 0.70, size=(EXPRESSIVE_RESERVOIR_LAYERS, q))
        input_y_weights *= rng.choice((-1.0, 1.0), size=input_y_weights.shape)
        input_z_weights *= rng.choice((-1.0, 1.0), size=input_z_weights.shape)
        local_angles = _draw_nonclifford_angles(
            rng, (EXPRESSIVE_RESERVOIR_LAYERS, q, 3)
        )
        coupling_angles = _draw_nonclifford_angles(
            rng, (EXPRESSIVE_RESERVOIR_LAYERS, q - 1)
        )
        return cls(
            parameter_seed=int(parameter_seed),
            q=int(q),
            input_y_weights=input_y_weights,
            input_z_weights=input_z_weights,
            local_angles=local_angles,
            coupling_angles=coupling_angles,
        )

    def parameter_bytes(self) -> bytes:
        arrays: Iterable[np.ndarray] = (
            self.input_y_weights,
            self.input_z_weights,
            self.local_angles,
            self.coupling_angles,
        )
        return b"".join(np.asarray(array, dtype="<f8").tobytes() for array in arrays)

    def manifest(self) -> Dict[str, Any]:
        """Return a JSON-serializable, audit-ready parameter manifest."""

        manifest = {
            "name": "expressive_reuploading_qrc",
            "circuit_version": "expressive_reuploading_v2",
            "parameter_seed": self.parameter_seed,
            "seed_status": "fixed as part of the forecasting protocol",
            "q": self.q,
            "layers": EXPRESSIVE_RESERVOIR_LAYERS,
            "input_encoding": "RY and RZ on every qubit, re-uploaded in both layers",
            "entanglement": "heterogeneous nearest-neighbour RZZ line",
            "layer_order": [
                "multi-qubit input RY/RZ",
                "nearest-neighbour line RZZ",
                "heterogeneous local RX/RY/RZ readout mixing",
            ],
            "measurement": "one Z-basis circuit; local Z and adjacent ZZ from the same bitstrings",
            "input_y_weights": self.input_y_weights.tolist(),
            "input_z_weights": self.input_z_weights.tolist(),
            "local_angles_rad": self.local_angles.tolist(),
            "coupling_angles_rad": self.coupling_angles.tolist(),
            "parameter_sha256": hashlib.sha256(self.parameter_bytes()).hexdigest(),
        }
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        manifest["architecture_sha256"] = hashlib.sha256(canonical).hexdigest()
        return manifest


class ExpressiveReuploadingReservoir:
    """Two-layer, multi-qubit data-reuploading reservoir.

    The object follows the interface consumed by ``base_impl.run_reservoir``.
    ``sigma`` in :meth:`sample_measurements` is interpreted as an independent
    per-bit readout-flip probability; the paper's ideal/noiseless simulations
    use ``sigma=0``.
    """

    def __init__(
        self,
        q: int = EXPRESSIVE_RESERVOIR_Q,
        eta: float = 0.2,
        window: int = 10,
        parameter_seed: int = EXPRESSIVE_RESERVOIR_SEED,
        feature_set: str = "z_plus_zz",
    ) -> None:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")
        if not 0.0 < float(eta) <= 1.0:
            raise ValueError("eta must lie in (0, 1]")
        if int(window) < 1:
            raise ValueError("window must be positive")

        self.q = int(q)
        self.dim = 2 ** self.q
        self.depth = EXPRESSIVE_RESERVOIR_LAYERS
        self.entangler = "line_rzz"
        self.eta = float(eta)
        self.window = int(window)
        self.parameter_seed = int(parameter_seed)
        self.seed = self.parameter_seed
        self.feature_set = feature_set
        self.architecture = ExpressiveArchitecture.from_seed(self.q, self.parameter_seed)
        self.edges = tuple((qubit, qubit + 1) for qubit in range(self.q - 1))
        self._rzz_phases = tuple(
            tuple(
                rzz_phase_vector(float(self.architecture.coupling_angles[layer, edge]), a, b, self.q)
                for edge, (a, b) in enumerate(self.edges)
            )
            for layer in range(self.depth)
        )
        self._basis_z = self._basis_z_signs()
        self.m = self.feature_dimension(feature_set)

    def feature_dimension(self, feature_set: str | None = None) -> int:
        selected = self.feature_set if feature_set is None else feature_set
        if selected == "z0":
            return 1
        if selected == "all_z":
            return self.q
        if selected == "z_plus_zz":
            return 2 * self.q - 1
        raise ValueError(f"unknown feature_set={selected!r}")

    def _basis_z_signs(self) -> np.ndarray:
        indices = np.arange(self.dim, dtype=np.uint64)
        shifts = np.arange(self.q - 1, -1, -1, dtype=np.uint64)
        bits = ((indices[:, None] >> shifts[None, :]) & 1).astype(np.int8)
        return 1.0 - 2.0 * bits.astype(float)

    def _features_from_z(self, z_signs: np.ndarray, feature_set: str | None = None) -> np.ndarray:
        selected = self.feature_set if feature_set is None else feature_set
        if selected == "z0":
            return z_signs[..., :1]
        if selected == "all_z":
            return z_signs
        if selected == "z_plus_zz":
            adjacent = z_signs[..., :-1] * z_signs[..., 1:]
            return np.concatenate((z_signs, adjacent), axis=-1)
        raise ValueError(f"unknown feature_set={selected!r}")

    def _initial_state(self) -> np.ndarray:
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0
        return state

    def prepare_state(self, u: float) -> np.ndarray:
        """Return the ideal statevector for one normalized scalar input."""

        centered = 2.0 * float(np.clip(u, 0.0, 1.0)) - 1.0
        state = self._initial_state()
        for layer in range(self.depth):
            # Multi-qubit data encoding/re-uploading.
            for qubit in range(self.q):
                state = apply_single_qubit_gate(
                    state,
                    ry(np.pi * self.architecture.input_y_weights[layer, qubit] * centered),
                    qubit,
                    self.q,
                )
                state = apply_single_qubit_gate(
                    state,
                    rz(np.pi * self.architecture.input_z_weights[layer, qubit] * centered),
                    qubit,
                    self.q,
                )

            # Entanglement precedes the non-diagonal local mix.  In particular,
            # the second layer's diagonal RZZ phases therefore affect the final
            # Z-basis probabilities instead of disappearing at measurement.
            for phase in self._rzz_phases[layer]:
                state = state * phase

            # Frozen heterogeneous, explicitly non-Clifford local/readout mix.
            for qubit in range(self.q):
                alpha, beta, gamma = self.architecture.local_angles[layer, qubit]
                state = apply_single_qubit_gate(state, rx(float(alpha)), qubit, self.q)
                state = apply_single_qubit_gate(state, ry(float(beta)), qubit, self.q)
                state = apply_single_qubit_gate(state, rz(float(gamma)), qubit, self.q)

        norm = float(np.linalg.norm(state))
        if not np.isfinite(norm) or norm <= 1.0e-12:
            raise FloatingPointError("expressive reservoir produced an invalid state")
        return state / norm

    def ideal_expectations(
        self,
        state: np.ndarray,
        feature_set: str | None = None,
    ) -> np.ndarray:
        """Exact statevector expectations for the selected Z observables."""

        probabilities = np.abs(np.asarray(state, dtype=complex)) ** 2
        probabilities = np.clip(probabilities.real, 0.0, None)
        probabilities /= probabilities.sum()
        basis_features = self._features_from_z(self._basis_z, feature_set=feature_set)
        return probabilities @ basis_features

    def sample_bitstrings(
        self,
        state: np.ndarray,
        n_shots: int,
        rng: np.random.Generator,
        readout_flip_probability: float = 0.0,
    ) -> np.ndarray:
        """Sample one Z-basis circuit and return bits ordered by qubit."""

        if int(n_shots) < 1:
            raise ValueError("n_shots must be positive")
        probabilities = np.abs(np.asarray(state, dtype=complex)) ** 2
        probabilities = np.clip(probabilities.real, 0.0, None)
        probabilities /= probabilities.sum()
        indices = rng.choice(self.dim, size=int(n_shots), p=probabilities)
        shifts = np.arange(self.q - 1, -1, -1)
        bits = ((indices[:, None] >> shifts[None, :]) & 1).astype(np.int8)
        flip_probability = float(np.clip(readout_flip_probability, 0.0, 0.5))
        if flip_probability:
            bits = np.bitwise_xor(bits, rng.random(bits.shape) < flip_probability).astype(np.int8)
        return bits

    def sample_measurements(
        self,
        state: np.ndarray,
        n_shots: int,
        sigma: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        bits = self.sample_bitstrings(
            state,
            n_shots=n_shots,
            rng=rng,
            readout_flip_probability=sigma,
        )
        z_signs = 1.0 - 2.0 * bits.astype(float)
        return self._features_from_z(z_signs)

    def architecture_manifest(self) -> Dict[str, Any]:
        manifest = self.architecture.manifest()
        manifest.update(
            {
                "feature_set": self.feature_set,
                "feature_dimension": self.m,
                "eta": self.eta,
                "window": self.window,
                "legacy_ablation": "base_impl.ExactQRCReservoir",
            }
        )
        return manifest


def make_expressive_reservoir(
    q: int = EXPRESSIVE_RESERVOIR_Q,
    seed: int = EXPRESSIVE_RESERVOIR_SEED,
    eta: float = 0.2,
    window: int = 10,
    feature_set: str = "z_plus_zz",
) -> ExpressiveReuploadingReservoir:
    """Factory used by experiment runners and robustness sweeps."""

    return ExpressiveReuploadingReservoir(
        q=q,
        eta=eta,
        window=window,
        parameter_seed=seed,
        feature_set=feature_set,
    )


def ideal_feature_matrix(
    reservoir: ExpressiveReuploadingReservoir,
    inputs: np.ndarray,
    feature_set: str | None = None,
) -> np.ndarray:
    return np.vstack(
        [
            reservoir.ideal_expectations(reservoir.prepare_state(float(u)), feature_set=feature_set)
            for u in np.asarray(inputs, dtype=float)
        ]
    )


def _leak_and_lag(values: np.ndarray, eta: float, window: int) -> np.ndarray:
    """Apply the same causal leak and tapped-delay construction as the paper."""

    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    filtered = np.zeros_like(values)
    for t in range(len(values)):
        filtered[t] = eta * values[t] if t == 0 else (1.0 - eta) * filtered[t - 1] + eta * values[t]
    lagged = np.zeros((len(values), int(window) * values.shape[1]), dtype=float)
    for t in range(len(values)):
        for lag in range(int(window)):
            source = t - int(window) + 1 + lag
            if source >= 0:
                lagged[t, lag * values.shape[1] : (lag + 1) * values.shape[1]] = filtered[source]
    return lagged


def _effective_rank(singular_values: np.ndarray) -> float:
    energy = np.square(np.asarray(singular_values, dtype=float))
    if energy.sum() <= 0.0:
        return 0.0
    probabilities = energy / energy.sum()
    probabilities = probabilities[probabilities > 0.0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def diagnostic_input_sequence(n_steps: int = 420) -> np.ndarray:
    """Fixed bounded probe containing several incommensurate time scales."""

    t = np.arange(int(n_steps), dtype=float)
    values = (
        0.50
        + 0.22 * np.sin(2.0 * np.pi * t / 37.0)
        + 0.15 * np.sin(2.0 * np.pi * t / 71.0 + 0.4)
        + 0.08 * np.sin(2.0 * np.pi * t / 19.0 + 0.0015 * t * t)
    )
    return np.clip(values, 0.02, 0.98)


def diagnose_expressive_reservoir(
    q: int = EXPRESSIVE_RESERVOIR_Q,
    parameter_seed: int = EXPRESSIVE_RESERVOIR_SEED,
    eta: float = 0.2,
    window: int = 10,
    feature_set: str = "z_plus_zz",
) -> Dict[str, Any]:
    """Run pre-forecast ideal-state diagnostics with no shot sampling.

    The non-equivalence diagnostic projects the identical leaked/lagged ideal
    quantum features onto the full linear span of the leaked/lagged scalar
    input.  A nonzero residual is a direct linear-span test, not a forecasting
    score and not a claim of quantum advantage.
    """

    reservoir = make_expressive_reservoir(
        q=q,
        seed=parameter_seed,
        eta=eta,
        window=window,
        feature_set=feature_set,
    )

    input_grid = np.linspace(0.0, 1.0, 257)
    grid_features = ideal_feature_matrix(reservoir, input_grid, feature_set=feature_set)
    feature_ranges = np.ptp(grid_features, axis=0)
    dependent = feature_ranges > DEPENDENCE_RANGE_THRESHOLD

    # Directly audit the once-subtle final-layer issue: replacing only the
    # second layer's RZZ phases by identity must change measured Z features.
    coupling_ablated = make_expressive_reservoir(
        q=q,
        seed=parameter_seed,
        eta=eta,
        window=window,
        feature_set=feature_set,
    )
    second_layer_identity = tuple(
        np.ones(coupling_ablated.dim, dtype=complex) for _ in coupling_ablated.edges
    )
    coupling_ablated._rzz_phases = (
        coupling_ablated._rzz_phases[0],
        second_layer_identity,
    )
    ablated_grid_features = ideal_feature_matrix(
        coupling_ablated, input_grid, feature_set=feature_set
    )
    second_layer_coupling_effect = np.max(
        np.abs(grid_features - ablated_grid_features), axis=0
    )
    coupling_sensitive = second_layer_coupling_effect > COUPLING_SENSITIVITY_THRESHOLD

    affine_design = np.column_stack((np.ones_like(input_grid), input_grid))
    affine_coefficients = np.linalg.lstsq(affine_design, grid_features, rcond=None)[0]
    affine_residuals = grid_features - affine_design @ affine_coefficients
    centered_grid = grid_features - grid_features.mean(axis=0, keepdims=True)
    total_energy = np.sum(np.square(centered_grid), axis=0)
    residual_energy = np.sum(np.square(affine_residuals), axis=0)
    unexplained_variance = np.divide(
        residual_energy,
        total_energy,
        out=np.zeros_like(residual_energy),
        where=total_energy > 1.0e-14,
    )
    nonlinear = unexplained_variance > NONLINEAR_UNEXPLAINED_VARIANCE_THRESHOLD

    scaled_grid = centered_grid[:, dependent]
    scales = np.std(scaled_grid, axis=0, ddof=1)
    scaled_grid = scaled_grid[:, scales > 1.0e-12] / scales[scales > 1.0e-12]
    singular_values = np.linalg.svd(scaled_grid, compute_uv=False)
    numerical_rank = int(np.linalg.matrix_rank(scaled_grid, tol=singular_values[0] * 1.0e-9))
    effective_rank = _effective_rank(singular_values)

    probe_inputs = diagnostic_input_sequence()
    ideal_features = ideal_feature_matrix(reservoir, probe_inputs, feature_set=feature_set)
    classical_lagged = _leak_and_lag(probe_inputs, eta=eta, window=window)
    quantum_lagged = _leak_and_lag(ideal_features, eta=eta, window=window)
    start = int(window) - 1
    classical_lagged = classical_lagged[start:]
    quantum_lagged = quantum_lagged[start:]
    c_mean = classical_lagged.mean(axis=0)
    c_scale = classical_lagged.std(axis=0)
    c_scale[c_scale <= 1.0e-12] = 1.0
    q_mean = quantum_lagged.mean(axis=0)
    q_scale = quantum_lagged.std(axis=0)
    q_scale[q_scale <= 1.0e-12] = 1.0
    classical_scaled = (classical_lagged - c_mean) / c_scale
    quantum_scaled = (quantum_lagged - q_mean) / q_scale
    classical_design = np.column_stack((np.ones(len(classical_scaled)), classical_scaled))
    mapping = np.linalg.lstsq(classical_design, quantum_scaled, rcond=None)[0]
    projection = classical_design @ mapping
    relative_projection_error = float(
        np.linalg.norm(quantum_scaled - projection) / max(np.linalg.norm(quantum_scaled), 1.0e-12)
    )

    joint = np.column_stack((classical_scaled, quantum_scaled))
    classical_rank = int(np.linalg.matrix_rank(classical_scaled))
    joint_rank = int(np.linalg.matrix_rank(joint))
    minimum_meaningful_rank = min(4, reservoir.m)
    minimum_nonlinear_features = min(3, reservoir.m)

    criteria = {
        "most_features_depend_on_input": bool(np.mean(dependent) >= 0.75),
        "several_features_are_nonlinear": bool(np.sum(nonlinear) >= minimum_nonlinear_features),
        "feature_matrix_has_meaningful_rank": bool(
            numerical_rank >= minimum_meaningful_rank and effective_rank >= 2.0
        ),
        "second_layer_couplings_change_z_features": bool(np.any(coupling_sensitive)),
        "not_linear_equivalent_to_filtered_input": bool(
            relative_projection_error > NON_EQUIVALENCE_RELATIVE_ERROR_THRESHOLD
            and joint_rank > classical_rank
        ),
    }
    passed = bool(all(criteria.values()))
    return {
        "architecture": reservoir.architecture_manifest(),
        "passed": passed,
        "diagnostic_status": "PASS" if passed else "FAIL",
        "criteria": criteria,
        "input_grid_points": int(len(input_grid)),
        "input_dependent_features": int(np.sum(dependent)),
        "input_dependent_fraction": float(np.mean(dependent)),
        "feature_ranges": feature_ranges.tolist(),
        "second_layer_coupling_sensitive_features": int(np.sum(coupling_sensitive)),
        "second_layer_coupling_max_effect": float(np.max(second_layer_coupling_effect)),
        "second_layer_coupling_effect_by_feature": second_layer_coupling_effect.tolist(),
        "nonlinear_features": int(np.sum(nonlinear)),
        "nonlinear_fraction": float(np.mean(nonlinear)),
        "affine_unexplained_variance": unexplained_variance.tolist(),
        "numerical_rank": numerical_rank,
        "effective_rank": effective_rank,
        "feature_dimension": int(reservoir.m),
        "classical_filtered_rank": classical_rank,
        "joint_classical_quantum_rank": joint_rank,
        "classical_to_quantum_relative_projection_error": relative_projection_error,
        "thresholds": {
            "input_dependence_range": DEPENDENCE_RANGE_THRESHOLD,
            "nonlinear_unexplained_variance": NONLINEAR_UNEXPLAINED_VARIANCE_THRESHOLD,
            "non_equivalence_relative_error": NON_EQUIVALENCE_RELATIVE_ERROR_THRESHOLD,
            "minimum_dependent_fraction": 0.75,
            "minimum_nonlinear_features": minimum_nonlinear_features,
            "minimum_numerical_rank": minimum_meaningful_rank,
            "minimum_effective_rank": 2.0,
            "second_layer_coupling_effect": COUPLING_SENSITIVITY_THRESHOLD,
        },
        "interpretation": (
            "Ideal statevector feature validity only; this is neither a forecasting result "
            "nor evidence of quantum advantage."
        ),
    }


__all__ = [
    "DEPENDENCE_RANGE_THRESHOLD",
    "COUPLING_SENSITIVITY_THRESHOLD",
    "EXPRESSIVE_RESERVOIR_LAYERS",
    "EXPRESSIVE_RESERVOIR_Q",
    "EXPRESSIVE_RESERVOIR_SEED",
    "FEATURE_SETS",
    "ExpressiveArchitecture",
    "ExpressiveReuploadingReservoir",
    "diagnose_expressive_reservoir",
    "diagnostic_input_sequence",
    "ideal_feature_matrix",
    "make_expressive_reservoir",
]
