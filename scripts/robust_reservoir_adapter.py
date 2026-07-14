#!/usr/bin/env python3
"""Common statevector/shot interface for the robust evaluation campaign.

The primary implementation is ``expressive_reservoir.ExpressiveReuploadingReservoir``.
The previous one-input-qubit circuit is available only as the registered
``degenerate_gate`` ablation.  Both are sampled first as local-Z bitstrings;
all-Z, Z0-only, and local-Z-plus-neighbor-parity feature sets therefore use
the same quantum executions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

import base_impl as base
from robust_protocol import FEATURE_SETS, derive_feature_set


ReservoirFactory = Callable[[], object]


@dataclass
class StatevectorFeatureProvider:
    name: str
    q: int
    parameter_seed: int
    factory: ReservoirFactory

    def shot_local_z(self, inputs: np.ndarray, *, n_shots: int, shot_seed: int) -> np.ndarray:
        """Return unfiltered local-Z shot outcomes, shape (shots,time,q)."""

        reservoir = self.factory()
        rng = np.random.default_rng(int(shot_seed))
        out = np.empty((int(n_shots), len(inputs), self.q), dtype=float)
        for time_index, value in enumerate(np.asarray(inputs, dtype=float)):
            state = reservoir.prepare_state(float(value))
            measured = np.asarray(
                reservoir.sample_measurements(state, n_shots=int(n_shots), sigma=0.0, rng=rng),
                dtype=float,
            )
            if measured.ndim != 2 or measured.shape[0] != n_shots or measured.shape[1] < self.q:
                raise ValueError(
                    f"{self.name} returned measurement shape {measured.shape}; "
                    f"expected ({n_shots}, >= {self.q})"
                )
            out[:, time_index, :] = measured[:, : self.q]
        return out

    def ideal_features(self, inputs: np.ndarray, *, feature_set: str) -> np.ndarray:
        """Return exact per-step expectations before leak/lag preprocessing."""

        if feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown feature_set={feature_set!r}")
        reservoir = self.factory()
        rows = []
        signs = _basis_local_z(self.q)
        derived_basis = derive_feature_set(signs, feature_set)
        for value in np.asarray(inputs, dtype=float):
            state = np.asarray(reservoir.prepare_state(float(value)), dtype=complex)
            if hasattr(reservoir, "ideal_expectations"):
                row = np.asarray(reservoir.ideal_expectations(state, feature_set=feature_set), dtype=float)
            else:
                probabilities = np.abs(state) ** 2
                probabilities = probabilities / probabilities.sum()
                row = probabilities @ derived_basis
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def architecture_manifest(self) -> Dict[str, object]:
        reservoir = self.factory()
        if hasattr(reservoir, "architecture_manifest"):
            return dict(reservoir.architecture_manifest())
        return {
            "name": self.name,
            "q": int(self.q),
            "parameter_seed": int(self.parameter_seed),
            "role": "degenerate_ablation",
        }


def _basis_local_z(q: int) -> np.ndarray:
    indices = np.arange(2 ** int(q))
    shifts = np.arange(q - 1, -1, -1)
    bits = ((indices[:, None] >> shifts[None, :]) & 1).astype(np.int8)
    return 1.0 - 2.0 * bits.astype(float)


def make_reservoir_provider(
    variant: str,
    *,
    q: int,
    parameter_seed: int,
    eta: float,
    window: int,
) -> StatevectorFeatureProvider:
    """Build the expressive primary provider or the original-circuit ablation."""

    if variant == "expressive_reuploading":
        try:
            from expressive_reservoir import make_expressive_reservoir
        except ModuleNotFoundError as exc:  # explicit failure for partial checkouts
            raise RuntimeError(
                "scripts/expressive_reservoir.py is required for the primary campaign. "
                "Run only --reservoir-variants degenerate_gate for a legacy smoke test."
            ) from exc

        def factory():
            return make_expressive_reservoir(
                q=int(q),
                seed=int(parameter_seed),
                eta=float(eta),
                window=int(window),
                feature_set="all_z",
            )

        return StatevectorFeatureProvider(
            name="expressive_reuploading",
            q=int(q),
            parameter_seed=int(parameter_seed),
            factory=factory,
        )

    if variant == "degenerate_gate":

        def factory():
            return base.ExactQRCReservoir(
                q=int(q),
                depth=1,
                entangler="ring_rzz",
                eta=float(eta),
                window=int(window),
                seed=int(parameter_seed),
            )

        return StatevectorFeatureProvider(
            name="degenerate_gate",
            q=int(q),
            parameter_seed=int(parameter_seed),
            factory=factory,
        )

    raise ValueError(
        f"unknown reservoir variant {variant!r}; expected expressive_reuploading or degenerate_gate"
    )
