#!/usr/bin/env python3
"""Leakage-safe evaluation primitives for the redesigned QAI experiment.

This module deliberately contains no paper-specific constants and performs no
I/O.  The campaign driver in :mod:`run_robust_evaluation` supplies independent
time-series clusters and raw local-Z shot records.  The functions here enforce
the statistical contract:

* train-prefix-only normalization;
* a held-out chronological test block;
* three expanding-train, blocked-validation folds;
* validation-only selection of group size, covariance weight, and ridge weight;
* per-time shot-lane randomization before leak/lag construction; and
* confidence intervals whose independent units are trajectories/windows.

Array convention
----------------
Raw local-Z shot records have shape ``(shots, time, qubits)``.  Processed
feature records have shape ``(views, time, features)``.  All slice end points
are exclusive.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


FEATURE_SETS: Tuple[str, ...] = ("all_z", "z0", "z_plus_zz")


@dataclass(frozen=True)
class BudgetSetting:
    """A fixed-execution setting with an exact ``T * N`` budget."""

    name: str
    time_steps: int
    shots_per_step: int

    @property
    def executions(self) -> int:
        return int(self.time_steps * self.shots_per_step)


@dataclass(frozen=True)
class ChronologicalFold:
    train_start: int
    train_stop: int
    validation_start: int
    validation_stop: int

    def manifest_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalDesign:
    t0: int
    test_start: int
    normalization_fit_stop: int
    folds: Tuple[ChronologicalFold, ...]

    def manifest_dict(self) -> Dict[str, object]:
        return {
            "t0": int(self.t0),
            "test_start": int(self.test_start),
            "normalization_fit_stop": int(self.normalization_fit_stop),
            "folds": [fold.manifest_dict() for fold in self.folds],
        }


def make_temporal_design(
    time_steps: int,
    *,
    t0: int = 30,
    test_fraction: float = 0.20,
    validation_folds: int = 3,
    initial_train_fraction: float = 0.50,
) -> TemporalDesign:
    """Construct expanding-train/blocked-validation folds and a final test.

    The last ``test_fraction`` of the post-washout series is never touched by
    model selection.  The first half of the pre-test block initializes the
    training prefix; the remaining half is split into consecutive validation
    blocks.  Normalization is frozen on the first fold's training prefix, so
    neither validation nor test observations affect it.
    """

    if validation_folds != 3:
        raise ValueError("The registered protocol requires exactly three validation folds")
    if not (0.0 < test_fraction < 0.5):
        raise ValueError("test_fraction must lie in (0, 0.5)")
    if not (0.25 <= initial_train_fraction <= 0.75):
        raise ValueError("initial_train_fraction must lie in [0.25, 0.75]")
    if time_steps <= t0 + 12:
        raise ValueError("time_steps is too small for washout, three folds, and test")

    usable = int(time_steps - t0)
    test_start = int(t0 + np.floor((1.0 - test_fraction) * usable))
    search_span = int(test_start - t0)
    initial_stop = int(t0 + np.floor(initial_train_fraction * search_span))
    edges = np.linspace(initial_stop, test_start, validation_folds + 1)
    edges = np.rint(edges).astype(int)
    edges[0] = initial_stop
    edges[-1] = test_start
    if np.any(np.diff(edges) <= 0):
        raise ValueError("validation blocks are empty; increase time_steps")

    folds = tuple(
        ChronologicalFold(
            train_start=int(t0),
            train_stop=int(edges[index]),
            validation_start=int(edges[index]),
            validation_stop=int(edges[index + 1]),
        )
        for index in range(validation_folds)
    )
    return TemporalDesign(
        t0=int(t0),
        test_start=int(test_start),
        normalization_fit_stop=int(initial_stop),
        folds=folds,
    )


def normalize_train_prefix(
    raw: np.ndarray,
    *,
    time_steps: int,
    fit_stop: int,
    clip_inputs: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Train-scale inputs and targets, clipping only physical circuit inputs.

    ``raw`` must contain ``time_steps + 1`` values.  The minimum and maximum
    are computed only from values available through ``fit_stop``.  Values
    outside that range later in time are clipped only on the input branch,
    because the registered angle encoding accepts inputs in [0, 1].  Targets
    retain the same train-fitted affine scale without clipping; they may
    legitimately be below zero or above one.
    """

    values = np.asarray(raw, dtype=float).reshape(-1)
    if values.size < time_steps + 1:
        raise ValueError(f"expected at least {time_steps + 1} values, got {values.size}")
    if not (1 <= fit_stop <= time_steps):
        raise ValueError("fit_stop must be inside the observed input interval")
    values = values[: time_steps + 1]
    fit_values = values[: fit_stop + 1]
    lo = float(np.min(fit_values))
    hi = float(np.max(fit_values))
    scale = float(hi - lo) if hi > lo else 1.0
    normalized_unclipped = (values - lo) / scale
    input_unclipped = normalized_unclipped[:-1]
    targets_unclipped = normalized_unclipped[1:]
    input_outside = np.logical_or(input_unclipped < 0.0, input_unclipped > 1.0)
    target_outside = np.logical_or(targets_unclipped < 0.0, targets_unclipped > 1.0)
    circuit_inputs = np.clip(input_unclipped, 0.0, 1.0) if clip_inputs else input_unclipped.copy()
    postfit_input_slice = slice(min(fit_stop + 1, time_steps), time_steps)
    postfit_target_slice = slice(min(fit_stop, time_steps), time_steps)
    return (
        circuit_inputs,
        targets_unclipped,
        {
            "normalization_min": lo,
            "normalization_max": hi,
            "normalization_fit_stop": int(fit_stop),
            "circuit_inputs_clipped": bool(clip_inputs),
            "targets_clipped": False,
            "input_postfit_clipped_fraction": float(np.mean(input_outside[postfit_input_slice])),
            "target_postfit_outside_fraction": float(np.mean(target_outside[postfit_target_slice])),
            "target_scaling": "train_prefix_minmax_affine_unclipped",
        },
    )


def derive_feature_set(local_z: np.ndarray, feature_set: str) -> np.ndarray:
    """Derive registered observables from the same local-Z bitstrings.

    Neighbor parities use the hardware-friendly line, ``Z_i Z_{i+1}``;
    they do not require additional circuit executions.
    """

    values = np.asarray(local_z, dtype=float)
    if values.ndim not in (2, 3):
        raise ValueError("local_z must have shape (time,q) or (shots,time,q)")
    q = int(values.shape[-1])
    if q < 1:
        raise ValueError("at least one local-Z feature is required")
    if feature_set == "all_z":
        return values.copy()
    if feature_set == "z0":
        return values[..., :1].copy()
    if feature_set == "z_plus_zz":
        parity = values[..., :-1] * values[..., 1:] if q > 1 else np.empty((*values.shape[:-1], 0))
        return np.concatenate([values, parity], axis=-1)
    raise ValueError(f"unknown feature_set={feature_set!r}; expected one of {FEATURE_SETS}")


def permute_lanes_per_time(local_z: np.ndarray, partition_seed: int) -> np.ndarray:
    """Randomize the arbitrary shot-lane assignment independently per step."""

    values = np.asarray(local_z, dtype=float)
    if values.ndim != 3:
        raise ValueError("local_z must have shape (shots,time,qubits)")
    rng = np.random.default_rng(int(partition_seed))
    out = np.empty_like(values)
    for time_index in range(values.shape[1]):
        out[:, time_index, :] = values[rng.permutation(values.shape[0]), time_index, :]
    return out


def permute_fixed_lanes(local_z: np.ndarray, partition_seed: int) -> np.ndarray:
    """Apply one lane permutation at every time (fixed-lane ablation)."""

    values = np.asarray(local_z, dtype=float)
    if values.ndim != 3:
        raise ValueError("local_z must have shape (shots,time,qubits)")
    rng = np.random.default_rng(int(partition_seed))
    return values[rng.permutation(values.shape[0]), :, :].copy()


def leak_and_lag(step_features: np.ndarray, *, eta: float, window: int) -> np.ndarray:
    """Apply the same linear leak and lag stack to one or several views."""

    values = np.asarray(step_features, dtype=float)
    squeeze = values.ndim == 2
    if squeeze:
        values = values[None, :, :]
    if values.ndim != 3:
        raise ValueError("step_features must have shape (time,d) or (views,time,d)")
    if not (0.0 < eta <= 1.0):
        raise ValueError("eta must lie in (0,1]")
    if window < 1:
        raise ValueError("window must be positive")

    views, time_steps, base_dim = values.shape
    leaked = np.empty_like(values)
    leaked[:, 0, :] = eta * values[:, 0, :]
    for time_index in range(1, time_steps):
        leaked[:, time_index, :] = (1.0 - eta) * leaked[:, time_index - 1, :] + eta * values[:, time_index, :]
    lagged = np.zeros((views, time_steps, window * base_dim), dtype=float)
    for lag in range(window):
        destination = slice(lag * base_dim, (lag + 1) * base_dim)
        offset = window - 1 - lag
        if offset == 0:
            lagged[:, :, destination] = leaked
        else:
            lagged[:, offset:, destination] = leaked[:, : time_steps - offset, :]
    return lagged[0] if squeeze else lagged


def classical_leak_lag(inputs: np.ndarray, *, eta: float, window: int) -> np.ndarray:
    """Classical baseline with exactly the quantum leak and lag preprocessing."""

    scalar = np.asarray(inputs, dtype=float).reshape(-1, 1)
    return leak_and_lag(scalar, eta=eta, window=window)[None, :, :]


def proper_group_sizes(n_shots: int) -> Tuple[int, ...]:
    return tuple(k for k in range(2, int(n_shots)) if n_shots % k == 0)


def grouped_views(phi: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(phi, dtype=float)
    if values.ndim != 3:
        raise ValueError("phi must have shape (shots,time,features)")
    n_shots = int(values.shape[0])
    if k < 1 or n_shots % int(k) != 0:
        raise ValueError(f"k={k} must be a positive divisor of n_shots={n_shots}")
    return values.reshape(n_shots // int(k), int(k), values.shape[1], values.shape[2]).mean(axis=1)


def organized_views_from_shots(
    step_features: np.ndarray,
    *,
    k: int,
    partition_seed: int,
    partition_mode: str,
    eta: float,
    window: int,
) -> np.ndarray:
    """Partition raw step observations, average, then construct leak/lags.

    The registered per-time mode assigns a fresh permutation at every time
    step.  The k value is mixed into the seed, so candidate group sizes do not
    inherit nested partitions from one common permutation.  ``fixed_lane`` is
    retained only as a sensitivity analysis.
    """

    seed_sequence = np.random.SeedSequence([int(partition_seed), int(k), 73471])
    k_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
    if partition_mode == "per_time_permuted":
        organized = permute_lanes_per_time(step_features, k_seed)
    elif partition_mode == "fixed_lane":
        organized = permute_fixed_lanes(step_features, k_seed)
    else:
        raise ValueError("partition_mode must be per_time_permuted or fixed_lane")
    grouped_steps = grouped_views(organized, int(k))
    return leak_and_lag(grouped_steps, eta=eta, window=window)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    scale = float(np.std(y))
    if scale <= 1e-12:
        scale = 1.0
    return float(np.sqrt(np.mean((y - pred) ** 2)) / scale)


def _grouped_fold_score(
    grouped: np.ndarray,
    targets: np.ndarray,
    fold: ChronologicalFold,
    ridge_alpha: float,
) -> float:
    groups, _, dimension = grouped.shape
    X_train = grouped[:, fold.train_start : fold.train_stop, :].reshape(-1, dimension)
    y_train = np.tile(targets[fold.train_start : fold.train_stop], groups)
    scaler = StandardScaler().fit(X_train)
    # Registered objective: mean squared loss + lambda * ||w||^2.  sklearn's
    # Ridge uses a summed loss, so alpha must scale with the number of rows.
    sklearn_alpha = float(ridge_alpha) * int(X_train.shape[0])
    model = Ridge(alpha=sklearn_alpha, fit_intercept=True).fit(scaler.transform(X_train), y_train)
    X_val = grouped[:, fold.validation_start : fold.validation_stop, :].reshape(-1, dimension)
    pred = model.predict(scaler.transform(X_val)).reshape(groups, -1).mean(axis=0)
    return nrmse(targets[fold.validation_start : fold.validation_stop], pred)


def _grouped_test_once(
    grouped: np.ndarray,
    targets: np.ndarray,
    *,
    train_start: int,
    test_start: int,
    ridge_alpha: float,
) -> Dict[str, object]:
    groups, _, dimension = grouped.shape
    X_train = grouped[:, train_start:test_start, :].reshape(-1, dimension)
    y_train = np.tile(targets[train_start:test_start], groups)
    scaler = StandardScaler().fit(X_train)
    sklearn_alpha = float(ridge_alpha) * int(X_train.shape[0])
    model = Ridge(alpha=sklearn_alpha, fit_intercept=True).fit(scaler.transform(X_train), y_train)
    X_test = grouped[:, test_start:, :].reshape(-1, dimension)
    prediction = model.predict(scaler.transform(X_test)).reshape(groups, -1).mean(axis=0)
    return {
        "test_nrmse": nrmse(targets[test_start:], prediction),
        "y_test": np.asarray(targets[test_start:], dtype=float),
        "y_pred": np.asarray(prediction, dtype=float),
    }


def evaluate_grouped(
    phi: np.ndarray,
    targets: np.ndarray,
    design: TemporalDesign,
    *,
    k_grid: Sequence[int],
    lambda_grid: Sequence[float],
) -> Dict[str, object]:
    """Select ``k`` and ridge ``lambda`` using folds, then touch test once."""

    candidates: List[Dict[str, object]] = []
    for k in tuple(int(value) for value in k_grid):
        grouped = grouped_views(phi, k)
        for ridge_alpha in tuple(float(value) for value in lambda_grid):
            scores = tuple(
                _grouped_fold_score(grouped, targets, fold, ridge_alpha)
                for fold in design.folds
            )
            candidates.append(
                {
                    "k": int(k),
                    "lambda_selected": float(ridge_alpha),
                    "fold_val_nrmse": scores,
                    "mean_val_nrmse": float(np.mean(scores)),
                }
            )
    if not candidates:
        raise ValueError("empty k/lambda candidate grid")
    ordered = sorted(
        candidates,
        key=lambda item: (item["mean_val_nrmse"], item["lambda_selected"], item["k"]),
    )
    best = ordered[0]
    second_best = float(ordered[1]["mean_val_nrmse"]) if len(ordered) > 1 else np.nan
    k_validation_spectrum: List[Dict[str, object]] = []
    for k in tuple(int(value) for value in k_grid):
        at_k = sorted(
            (item for item in candidates if int(item["k"]) == k),
            key=lambda item: (item["mean_val_nrmse"], item["lambda_selected"]),
        )
        winner = at_k[0]
        runner_up = float(at_k[1]["mean_val_nrmse"]) if len(at_k) > 1 else np.nan
        k_validation_spectrum.append(
            {
                "k": k,
                "lambda_selected": float(winner["lambda_selected"]),
                "mean_val_nrmse": float(winner["mean_val_nrmse"]),
                "fold_val_nrmse": tuple(float(value) for value in winner["fold_val_nrmse"]),
                "lambda_selection_margin": runner_up - float(winner["mean_val_nrmse"]),
            }
        )
    grouped = grouped_views(phi, int(best["k"]))
    test = _grouped_test_once(
        grouped,
        targets,
        train_start=design.t0,
        test_start=design.test_start,
        ridge_alpha=float(best["lambda_selected"]),
    )
    return {
        **best,
        **test,
        "groups": int(grouped.shape[0]),
        "second_best_val_nrmse": second_best,
        "selection_margin": second_best - float(best["mean_val_nrmse"]),
        "k_validation_spectrum": k_validation_spectrum,
    }


def evaluate_shot_organization(
    step_features: np.ndarray,
    targets: np.ndarray,
    design: TemporalDesign,
    *,
    k_grid: Sequence[int],
    lambda_grid: Sequence[float],
    partition_seed: int,
    partition_mode: str,
    eta: float,
    window: int,
) -> Dict[str, object]:
    """Tune organization directly from the fixed raw shot record.

    Each candidate k receives its own per-time partition, group averages are
    formed first, and the identical linear leak/lag map is then applied to each
    group sequence.  This ordering is both operationally transparent and
    preserves the exact equality between the mean grouped view and EV.
    """

    candidates: List[Dict[str, object]] = []
    grouped_by_k: Dict[int, np.ndarray] = {}
    for k in tuple(int(value) for value in k_grid):
        grouped = organized_views_from_shots(
            step_features,
            k=k,
            partition_seed=partition_seed,
            partition_mode=partition_mode,
            eta=eta,
            window=window,
        )
        grouped_by_k[k] = grouped
        for ridge_alpha in tuple(float(value) for value in lambda_grid):
            scores = tuple(
                _grouped_fold_score(grouped, targets, fold, ridge_alpha)
                for fold in design.folds
            )
            candidates.append(
                {
                    "k": k,
                    "lambda_selected": ridge_alpha,
                    "fold_val_nrmse": scores,
                    "mean_val_nrmse": float(np.mean(scores)),
                }
            )
    if not candidates:
        raise ValueError("empty organization/lambda candidate grid")
    ordered = sorted(
        candidates,
        key=lambda item: (item["mean_val_nrmse"], item["lambda_selected"], item["k"]),
    )
    best = ordered[0]
    second_best = float(ordered[1]["mean_val_nrmse"]) if len(ordered) > 1 else np.nan
    k_validation_spectrum: List[Dict[str, object]] = []
    for k in tuple(int(value) for value in k_grid):
        at_k = sorted(
            (item for item in candidates if int(item["k"]) == k),
            key=lambda item: (item["mean_val_nrmse"], item["lambda_selected"]),
        )
        winner = at_k[0]
        runner_up = float(at_k[1]["mean_val_nrmse"]) if len(at_k) > 1 else np.nan
        k_validation_spectrum.append(
            {
                "k": k,
                "lambda_selected": float(winner["lambda_selected"]),
                "mean_val_nrmse": float(winner["mean_val_nrmse"]),
                "fold_val_nrmse": tuple(float(value) for value in winner["fold_val_nrmse"]),
                "lambda_selection_margin": runner_up - float(winner["mean_val_nrmse"]),
            }
        )
    grouped = grouped_by_k[int(best["k"])]
    test = _grouped_test_once(
        grouped,
        targets,
        train_start=design.t0,
        test_start=design.test_start,
        ridge_alpha=float(best["lambda_selected"]),
    )
    n_shots = int(np.asarray(step_features).shape[0])
    selected_k = int(best["k"])
    return {
        **best,
        **test,
        "groups": int(grouped.shape[0]),
        "interior_k_selected": bool(1 < selected_k < n_shots),
        "partition_mode": partition_mode,
        "second_best_val_nrmse": second_best,
        "selection_margin": second_best - float(best["mean_val_nrmse"]),
        "k_validation_spectrum": k_validation_spectrum,
    }


def grouped_covariance_equivalent_prediction(
    grouped: np.ndarray,
    targets: np.ndarray,
    *,
    train_start: int,
    train_stop: int,
    eval_start: int,
    eval_stop: int,
    ridge_alpha: float,
) -> Dict[str, object]:
    """Solve fixed-k grouped ridge as EV plus its within-group covariance.

    Both forms use the registered mean-loss objective and the scaler fitted to
    the grouped training rows.  The returned prediction must match direct
    grouped Ridge to numerical precision; this is an identity check, not an
    additional baseline.
    """

    values = np.asarray(grouped, dtype=float)
    groups, _, dimension = values.shape
    train_steps = int(train_stop - train_start)
    X_rows = values[:, train_start:train_stop, :].reshape(-1, dimension)
    scaler = StandardScaler().fit(X_rows)
    Xs = scaler.transform(X_rows).reshape(groups, train_steps, dimension)
    xbar = Xs.mean(axis=0)
    deviations = Xs - xbar[None, :, :]
    within_covariance = (
        np.einsum("gtd,gte->de", deviations, deviations, optimize=True)
        / float(groups * train_steps)
    )
    y = np.asarray(targets[train_start:train_stop], dtype=float)
    y_mean = float(y.mean())
    yc = y - y_mean
    # The grouped-row scaler centers the grand mean, hence mean_t(xbar_t)=0.
    matrix = xbar.T @ xbar / float(train_steps)
    matrix += within_covariance
    matrix += float(ridge_alpha) * np.eye(dimension)
    rhs = xbar.T @ yc / float(train_steps)
    try:
        coefficient = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.pinv(matrix) @ rhs
    eval_rows = values[:, eval_start:eval_stop, :].reshape(-1, dimension)
    eval_scaled = scaler.transform(eval_rows).reshape(groups, eval_stop - eval_start, dimension)
    prediction = eval_scaled.mean(axis=0) @ coefficient + y_mean
    return {
        "y_pred": prediction,
        "coefficient": coefficient,
        "within_covariance": within_covariance,
    }


def _covariance_ev_stage(
    phi: np.ndarray,
    targets: np.ndarray,
    *,
    train_start: int,
    train_stop: int,
    eval_start: int,
    eval_stop: int,
    ridge_alpha: float,
    covariance_alpha: float,
) -> Tuple[float, np.ndarray]:
    """Fit the directly penalized EV objective for one chronological stage."""

    n_shots, _, dimension = phi.shape
    ev = phi.mean(axis=0)
    X_train = ev[train_start:train_stop]
    y_train = np.asarray(targets[train_start:train_stop], dtype=float)
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    shot_train = scaler.transform(phi[:, train_start:train_stop, :].reshape(-1, dimension))
    shot_train = shot_train.reshape(n_shots, train_stop - train_start, dimension)
    deviations = shot_train - Xs[None, :, :]
    denominator = max(n_shots - 1, 1) * max(train_stop - train_start, 1)
    sigma_shot = np.einsum("ntd,nte->de", deviations, deviations, optimize=True) / denominator

    y_mean = float(y_train.mean())
    yc = y_train - y_mean
    train_steps = float(train_stop - train_start)
    # Registered objective:
    # mean_t loss + alpha * w^T Sigma_shot w + lambda * ||w||^2.
    matrix = Xs.T @ Xs / train_steps
    matrix += float(covariance_alpha) * sigma_shot
    matrix += float(ridge_alpha) * np.eye(dimension)
    rhs = Xs.T @ yc / train_steps
    try:
        coefficient = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        coefficient = np.linalg.pinv(matrix) @ rhs
    prediction = scaler.transform(ev[eval_start:eval_stop]) @ coefficient + y_mean
    return nrmse(targets[eval_start:eval_stop], prediction), prediction


def evaluate_covariance_ev(
    phi: np.ndarray,
    targets: np.ndarray,
    design: TemporalDesign,
    *,
    lambda_grid: Sequence[float],
    alpha_grid: Sequence[float],
) -> Dict[str, object]:
    """Tune direct shot-covariance and ridge penalties using validation only."""

    candidates: List[Dict[str, object]] = []
    for covariance_alpha in tuple(float(value) for value in alpha_grid):
        for ridge_alpha in tuple(float(value) for value in lambda_grid):
            scores = tuple(
                _covariance_ev_stage(
                    phi,
                    targets,
                    train_start=fold.train_start,
                    train_stop=fold.train_stop,
                    eval_start=fold.validation_start,
                    eval_stop=fold.validation_stop,
                    ridge_alpha=ridge_alpha,
                    covariance_alpha=covariance_alpha,
                )[0]
                for fold in design.folds
            )
            candidates.append(
                {
                    "alpha_selected": covariance_alpha,
                    "lambda_selected": ridge_alpha,
                    "fold_val_nrmse": scores,
                    "mean_val_nrmse": float(np.mean(scores)),
                }
            )
    if not candidates:
        raise ValueError("empty alpha/lambda candidate grid")
    ordered = sorted(
        candidates,
        key=lambda item: (item["mean_val_nrmse"], item["lambda_selected"], item["alpha_selected"]),
    )
    best = ordered[0]
    second_best = float(ordered[1]["mean_val_nrmse"]) if len(ordered) > 1 else np.nan
    score, prediction = _covariance_ev_stage(
        phi,
        targets,
        train_start=design.t0,
        train_stop=design.test_start,
        eval_start=design.test_start,
        eval_stop=phi.shape[1],
        ridge_alpha=float(best["lambda_selected"]),
        covariance_alpha=float(best["alpha_selected"]),
    )
    return {
        **best,
        "k": int(phi.shape[0]),
        "groups": 1,
        "test_nrmse": float(score),
        "y_test": np.asarray(targets[design.test_start:], dtype=float),
        "y_pred": np.asarray(prediction, dtype=float),
        "second_best_val_nrmse": second_best,
        "selection_margin": second_best - float(best["mean_val_nrmse"]),
    }


def _t_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    mean = float(np.mean(data))
    if data.size < 2:
        return mean, mean, mean
    sem = float(np.std(data, ddof=1) / np.sqrt(data.size))
    half = float(stats.t.ppf(0.5 + confidence / 2.0, data.size - 1) * sem)
    return mean, mean - half, mean + half


def clustered_method_summary(
    rows: pd.DataFrame,
    *,
    grouping: Sequence[str] = ("dataset", "budget_setting", "reservoir_variant", "feature_set", "method"),
    cluster_column: str = "cluster_id",
) -> pd.DataFrame:
    """Summarize method NRMSE with trajectory/window as the independent unit."""

    required = set(grouping) | {cluster_column, "test_nrmse"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing result columns: {sorted(missing)}")
    cluster_keys = list(grouping) + [cluster_column]
    cluster_means = rows.groupby(cluster_keys, dropna=False)["test_nrmse"].mean().reset_index()
    output: List[Dict[str, object]] = []
    for key, frame in cluster_means.groupby(list(grouping), dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        mean, lo, hi = _t_interval(frame["test_nrmse"].to_numpy(dtype=float))
        output.append(
            {
                **dict(zip(grouping, key_tuple)),
                "clusters": int(frame[cluster_column].nunique()),
                "cluster_mean_nrmse": mean,
                "cluster_ci95_low": lo,
                "cluster_ci95_high": hi,
                "inference_unit": "trajectory_or_sunspot_window",
            }
        )
    return pd.DataFrame(output)


def clustered_paired_comparison(
    rows: pd.DataFrame,
    *,
    baseline: str = "ev",
    methods: Sequence[str] = ("raw", "split", "covariance_ev"),
    grouping: Sequence[str] = ("dataset", "budget_setting", "reservoir_variant", "feature_set"),
    cluster_column: str = "cluster_id",
) -> pd.DataFrame:
    """Compute paired deltas after averaging every nested factor per cluster."""

    cluster_keys = list(grouping) + [cluster_column, "method"]
    means = rows.groupby(cluster_keys, dropna=False)["test_nrmse"].mean().reset_index()
    output: List[Dict[str, object]] = []
    for key, frame in means.groupby(list(grouping), dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        wide = frame.pivot(index=cluster_column, columns="method", values="test_nrmse")
        for method in methods:
            if baseline not in wide or method not in wide:
                continue
            paired = wide[[baseline, method]].dropna()
            delta = (paired[baseline] - paired[method]).to_numpy(dtype=float)
            mean, lo, hi = _t_interval(delta)
            output.append(
                {
                    **dict(zip(grouping, key_tuple)),
                    "baseline": baseline,
                    "method": method,
                    "clusters": int(delta.size),
                    "delta_baseline_minus_method": mean,
                    "cluster_ci95_low": lo,
                    "cluster_ci95_high": hi,
                    "cluster_win_rate": float(np.mean(delta > 0.0)) if delta.size else np.nan,
                    "inference_unit": "trajectory_or_sunspot_window",
                }
            )
    return pd.DataFrame(output)


def clustered_partition_sensitivity(rows: pd.DataFrame) -> pd.DataFrame:
    """Compare per-time random partitions with the fixed-lane ablation.

    Rows are paired on reservoir seed, trajectory/window, shot seed, and
    registered partition seed before nested deltas are averaged within each
    cluster. Positive delta means the primary per-time construction has lower
    NRMSE than fixed lanes.
    """

    required = {
        "dataset",
        "budget_setting",
        "reservoir_variant",
        "feature_set",
        "reservoir_seed",
        "cluster_id",
        "shot_seed",
        "partition_seed",
        "method",
        "partition_mode",
        "partition_role",
        "test_nrmse",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"missing result columns: {sorted(missing)}")
    selected = rows[
        (rows["method"] == "split")
        & (rows["partition_role"].isin(["primary", "fixed_lane_ablation"]))
    ].copy()
    config_keys = ["dataset", "budget_setting", "reservoir_variant", "feature_set", "reservoir_seed"]
    nested_keys = config_keys + ["cluster_id", "shot_seed", "partition_seed"]
    mode_means = (
        selected.groupby(nested_keys + ["partition_mode"], dropna=False)["test_nrmse"]
        .mean()
        .reset_index()
    )
    output: List[Dict[str, object]] = []
    for key, frame in mode_means.groupby(config_keys, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        wide = frame.pivot(
            index=["cluster_id", "shot_seed", "partition_seed"],
            columns="partition_mode",
            values="test_nrmse",
        )
        if "per_time_permuted" not in wide or "fixed_lane" not in wide:
            continue
        paired = wide[["per_time_permuted", "fixed_lane"]].dropna()
        paired = paired.assign(
            delta_fixed_minus_per_time=paired["fixed_lane"] - paired["per_time_permuted"]
        ).reset_index()
        cluster_delta = paired.groupby("cluster_id", dropna=False)["delta_fixed_minus_per_time"].mean()
        delta = cluster_delta.to_numpy(dtype=float)
        mean, lo, hi = _t_interval(delta)
        pairs_per_cluster = paired.groupby("cluster_id", dropna=False).size()
        output.append(
            {
                **dict(zip(config_keys, key_tuple)),
                "clusters": int(delta.size),
                "matched_nested_pairs": int(len(paired)),
                "matched_pairs_per_cluster_min": int(pairs_per_cluster.min()),
                "delta_fixed_minus_per_time": mean,
                "cluster_ci95_low": lo,
                "cluster_ci95_high": hi,
                "inference_unit": "trajectory_or_sunspot_window",
            }
        )
    return pd.DataFrame(output)


def clustered_ideal_classical_comparison(
    rows: pd.DataFrame,
    *,
    reservoir_variant: str = "expressive_reuploading",
    feature_set: str = "z_plus_zz",
) -> pd.DataFrame:
    """Pair ideal quantum and matched classical preprocessing by cluster."""

    classical = rows[rows["method"] == "classical_input"].copy()
    ideal = rows[
        (rows["method"] == "ideal_quantum")
        & (rows["reservoir_variant"] == reservoir_variant)
        & (rows["feature_set"] == feature_set)
    ].copy()
    selected = pd.concat([classical, ideal], ignore_index=True)
    keys = ["dataset", "budget_setting", "cluster_id", "method"]
    means = selected.groupby(keys, dropna=False)["test_nrmse"].mean().reset_index()
    output: List[Dict[str, object]] = []
    for (dataset, setting), frame in means.groupby(["dataset", "budget_setting"], dropna=False):
        wide = frame.pivot(index="cluster_id", columns="method", values="test_nrmse")
        if "classical_input" not in wide or "ideal_quantum" not in wide:
            continue
        paired = wide[["classical_input", "ideal_quantum"]].dropna()
        delta = (paired["classical_input"] - paired["ideal_quantum"]).to_numpy(dtype=float)
        mean, lo, hi = _t_interval(delta)
        output.append(
            {
                "dataset": dataset,
                "budget_setting": setting,
                "reservoir_variant": reservoir_variant,
                "feature_set": feature_set,
                "baseline": "classical_input",
                "method": "ideal_quantum",
                "clusters": int(delta.size),
                "delta_classical_minus_ideal": mean,
                "cluster_ci95_low": lo,
                "cluster_ci95_high": hi,
                "cluster_win_rate": float(np.mean(delta > 0.0)) if delta.size else np.nan,
                "inference_unit": "trajectory_or_sunspot_window",
            }
        )
    return pd.DataFrame(output)


def validate_budget_settings(settings: Iterable[BudgetSetting]) -> Tuple[BudgetSetting, ...]:
    settings = tuple(settings)
    if len(settings) != 3:
        raise ValueError("The registered experiment requires exactly three representative settings")
    budgets = {setting.executions for setting in settings}
    if len(budgets) != 1:
        raise ValueError(f"all representative settings must use the same budget, got {sorted(budgets)}")
    for setting in settings:
        if not proper_group_sizes(setting.shots_per_step):
            raise ValueError(f"{setting.name} has no interior divisor for split training")
    return settings
