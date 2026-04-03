
#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ============================
# Plot style (Thesis / Exp2-like)
# ============================

ID_COLOR = "#EE6677"       # Split / ID
FD_COLOR = "#4477AA"       # EV / FD
REFERENCE_COLOR = "#888888"
MUTED_COLOR = "#666666"
AXIS_COLOR = "#1F1F1F"
TICK_COLOR = "#222222"
GRID_COLOR = "#D7D7D7"
LIGHT_GRID = "#ECECEC"

FIG_W = 5.7295
FIG_H = 2.45

def scaled_height(width: float, height: float) -> float:
    return FIG_W * (height / width)

def apply_style() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "mathtext.fontset": "cm",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "axes.grid": True,
        "grid.alpha": 0.7,
        "grid.linestyle": "-",
        "grid.linewidth": 0.75,
        "grid.color": GRID_COLOR,
        "axes.axisbelow": True,
    })

def style_axis(ax):
    ax.tick_params(direction="out", width=0.8, length=3.0, pad=4, colors=TICK_COLOR)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color(AXIS_COLOR)

def save_figure(fig, stem: Path, dpi: int = 260) -> None:
    fig.savefig(str(stem.with_suffix(".pdf")), dpi=dpi)
    fig.savefig(str(stem.with_suffix(".png")), dpi=dpi)
    plt.close(fig)


# ============================
# Config
# ============================

@dataclass(frozen=True)
class PaperConfig:
    budget_total: int = 12000
    q_default: int = 5
    depth_default: int = 2
    entangler_default: str = "ring_cnot"
    eta: float = 0.3
    window: int = 10
    t0: int = 30
    train_frac: float = 0.70
    val_frac_within_train: float = 0.25
    lambda_default: float = 10.0
    lambda_grid: Tuple[float, ...] = (
        1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1,
        3e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0
    )
    gamma_grid: Tuple[float, ...] = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
    phase_nshots: Tuple[int, ...] = (10, 20, 30, 40)
    phase_sigmas: Tuple[float, ...] = (0.00, 0.02, 0.08)
    confirmation_seeds: Tuple[int, ...] = tuple(range(100, 112))   # 12 seeds
    architecture_seeds: Tuple[int, ...] = tuple(range(200, 212))   # 12 seeds
    appendix_seeds: Tuple[int, ...] = tuple(range(300, 312))       # 12 seeds
    architecture_variants: Tuple[Tuple[str, int], ...] = (
        ("ring_cnot", 1),
        ("ring_cnot", 2),
        ("ring_cnot", 4),
        ("line_cnot", 2),
        ("all_to_all_cz", 2),
    )
    reservoir_sizes: Tuple[int, ...] = (4, 5, 6)

    @property
    def m(self) -> int:
        return 2 * self.q_default

    @property
    def d_default(self) -> int:
        return self.window * (2 * self.q_default)

    def d(self, q: Optional[int] = None) -> int:
        qq = self.q_default if q is None else q
        return self.window * (2 * qq)

    def rho_ev(self, t_steps: int, q: Optional[int] = None) -> float:
        return self.train_frac * max(t_steps - self.t0, 0) / self.d(q=q)

    @property
    def representative_configs(self) -> Dict[str, Dict[str, float]]:
        return {
            "mackey_glass": {"budget_total": 12000, "n_shots": 40, "sigma": 0.02},
            "lorenz": {"budget_total": 12000, "n_shots": 30, "sigma": 0.00},
            "narma10": {"budget_total": 12000, "n_shots": 20, "sigma": 0.00},
        }


# ============================
# Dynamical systems
# ============================

def mackey_glass(
    n_steps: int,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n_power: int = 10,
    dt: float = 1.0,
    warmup: int = 500,
    seed: int = 1234,
) -> np.ndarray:
    total = n_steps + warmup + tau
    rng = np.random.default_rng(seed)
    x = np.zeros(total, dtype=float)
    x[:tau] = 0.9 + 0.2 * rng.random(tau)
    for t in range(tau, total - 1):
        x_tau = x[t - tau]
        dx = beta * x_tau / (1.0 + x_tau ** n_power) - gamma * x[t]
        x[t + 1] = x[t] + dt * dx
    return x[warmup + tau : warmup + tau + n_steps]

def lorenz_x(
    n_steps: int,
    dt: float = 0.02,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    warmup: int = 2500,
    init_state: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    total = n_steps + warmup
    xs = np.zeros(total, dtype=float)
    x, y, z = init_state
    for t in range(total):
        xs[t] = x
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dt * dx
        y += dt * dy
        z += dt * dz
    return xs[warmup:]

def narma10_y(n_steps: int, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 10
    warmup = 200
    total = n_steps + warmup + n
    u = rng.uniform(0.0, 0.5, size=total)
    y = np.zeros(total, dtype=float)
    for t in range(total - 1):
        if t < n:
            y_sum = np.sum(y[max(0, t - n + 1): t + 1])
        else:
            y_sum = np.sum(y[t - n + 1: t + 1])
        lag_idx = max(t - n + 1, 0)
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * y_sum + 1.5 * u[lag_idx] * u[t] + 0.1
        y[t + 1] = np.clip(y[t + 1], -10.0, 10.0)
    return y[warmup + n : warmup + n + n_steps]

def make_series(dataset: str, total_steps: int) -> np.ndarray:
    if dataset == "mackey_glass":
        return mackey_glass(total_steps + 1, seed=1234)
    if dataset == "lorenz":
        return lorenz_x(total_steps + 1)
    if dataset == "narma10":
        return narma10_y(total_steps + 1, seed=1234)
    raise ValueError(dataset)

def normalize_and_shift(raw: np.ndarray, t_steps: int, cfg: PaperConfig) -> Tuple[np.ndarray, np.ndarray]:
    train_end = cfg.t0 + int(cfg.train_frac * (t_steps - cfg.t0))
    usable = raw[: t_steps + 1]
    train_raw = usable[: train_end + 1]
    lo = float(np.min(train_raw))
    hi = float(np.max(train_raw))
    scale = hi - lo if hi > lo else 1.0
    scaled = (usable - lo) / scale
    return scaled[:-1], scaled[1:]


# ============================
# Quantum reservoir simulator
# ============================

H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)

def rx(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

def ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)

def rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2.0), 0.0], [0.0, np.exp(1j * theta / 2.0)]], dtype=complex)

def apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    state_view = state.reshape([2] * n_qubits)
    state_view = np.moveaxis(state_view, qubit, 0)
    shp = state_view.shape
    state_view = state_view.reshape(2, -1)
    state_view = gate @ state_view
    state_view = state_view.reshape(shp)
    state_view = np.moveaxis(state_view, 0, qubit)
    return state_view.reshape(-1)

def apply_cnot(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    state_view = state.reshape([2] * n_qubits)
    idx = [slice(None)] * n_qubits
    idx[control] = 1
    sub = state_view[tuple(idx)].copy()
    target_adj = target if target < control else target - 1
    sub = np.moveaxis(sub, target_adj, 0)
    shp = sub.shape
    sub = sub.reshape(2, -1)
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sub = x_gate @ sub
    sub = sub.reshape(shp)
    sub = np.moveaxis(sub, 0, target_adj)
    state_view[tuple(idx)] = sub
    return state_view.reshape(-1)

def apply_cz(state: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    state_view = state.reshape([2] * n_qubits)
    idx = [slice(None)] * n_qubits
    idx[control] = 1
    idx[target] = 1
    state_view[tuple(idx)] *= -1.0
    return state_view.reshape(-1)

class ExactQRCReservoir:
    def __init__(self, q: int, depth: int, entangler: str, eta: float, window: int, seed: int) -> None:
        self.q = q
        self.depth = depth
        self.entangler = entangler
        self.eta = eta
        self.window = window
        self.dim = 2 ** q
        self.m = 2 * q
        self.rng = np.random.default_rng(seed)
        self.input_weights = self.rng.uniform(-np.pi, np.pi, size=q)
        self.layer_params = self.rng.uniform(0.0, 2.0 * np.pi, size=(depth, q, 3))

    def _initial_state(self) -> np.ndarray:
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0
        return state

    def _apply_entangler_layer(self, state: np.ndarray) -> np.ndarray:
        if self.entangler == "ring_cnot":
            edges = [(i, (i + 1) % self.q) for i in range(self.q)]
            for control, target in edges:
                state = apply_cnot(state, control, target, self.q)
        elif self.entangler == "line_cnot":
            edges = [(i, i + 1) for i in range(self.q - 1)]
            for control, target in edges:
                state = apply_cnot(state, control, target, self.q)
        elif self.entangler == "all_to_all_cz":
            for control in range(self.q):
                for target in range(control + 1, self.q):
                    state = apply_cz(state, control, target, self.q)
        else:
            raise ValueError(self.entangler)
        return state

    def prepare_state(self, u: float) -> np.ndarray:
        state = self._initial_state()
        for qubit in range(self.q):
            state = apply_single_qubit_gate(state, ry(self.input_weights[qubit] * u), qubit, self.q)
        for layer in range(self.depth):
            state = self._apply_entangler_layer(state)
            for qubit in range(self.q):
                state = apply_single_qubit_gate(state, rx(self.layer_params[layer, qubit, 0]), qubit, self.q)
                state = apply_single_qubit_gate(state, ry(self.layer_params[layer, qubit, 1]), qubit, self.q)
                state = apply_single_qubit_gate(state, rz(self.layer_params[layer, qubit, 2]), qubit, self.q)
        return state

    def _sample_basis(self, state: np.ndarray, n_shots: int, basis: str, rng: np.random.Generator) -> np.ndarray:
        if basis == "X":
            for qubit in range(self.q):
                state = apply_single_qubit_gate(state, H, qubit, self.q)
        probs = np.abs(state) ** 2
        probs = np.clip(probs.real, 0.0, None)
        probs /= probs.sum()
        idx = rng.choice(self.dim, size=n_shots, p=probs)
        shifts = np.arange(self.q - 1, -1, -1)
        bits = ((idx[:, None] >> shifts[None, :]) & 1).astype(np.int8)
        return bits

    def sample_measurements(self, state: np.ndarray, n_shots: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
        z_bits = self._sample_basis(state.copy(), n_shots, basis="Z", rng=rng)
        x_bits = self._sample_basis(state.copy(), n_shots, basis="X", rng=rng)
        z_signs = 1.0 - 2.0 * z_bits.astype(float)
        x_signs = 1.0 - 2.0 * x_bits.astype(float)
        measurements = np.concatenate([z_signs, x_signs], axis=1)
        if sigma > 0.0:
            flip_prob = min(float(sigma), 0.5)
            flip_mask = rng.random(measurements.shape) < flip_prob
            measurements = np.where(flip_mask, -measurements, measurements)
            measurements = measurements + float(sigma) * rng.normal(size=measurements.shape)
        return measurements

def run_reservoir(
    inputs: np.ndarray,
    reservoir: ExactQRCReservoir,
    n_shots: int,
    sigma: float,
    shot_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(shot_seed)
    t_steps = len(inputs)
    m = reservoir.m
    eta = reservoir.eta
    lags = reservoir.window
    r = np.zeros((n_shots, t_steps, m), dtype=float)
    for t in range(t_steps):
        state = reservoir.prepare_state(float(inputs[t]))
        meas = reservoir.sample_measurements(state, n_shots=n_shots, sigma=sigma, rng=rng)
        if t == 0:
            r[:, t, :] = eta * meas
        else:
            r[:, t, :] = (1.0 - eta) * r[:, t - 1, :] + eta * meas
    phi = np.zeros((n_shots, t_steps, lags * m), dtype=float)
    for t in range(t_steps):
        for lag in range(lags):
            idx = t - lags + 1 + lag
            if idx >= 0:
                phi[:, t, lag * m : (lag + 1) * m] = r[:, idx, :]
    return phi


# ============================
# Readout helpers and protocols
# ============================

def split_points(t_steps: int, cfg: PaperConfig) -> Tuple[int, int, int]:
    t0 = cfg.t0
    train_end = t0 + int(cfg.train_frac * (t_steps - t0))
    fit_end = t0 + int((1.0 - cfg.val_frac_within_train) * (train_end - t0))
    return t0, fit_end, train_end

def proper_divisors(n: int) -> List[int]:
    return [k for k in range(2, n) if n % k == 0]

def grouped_features(phi: np.ndarray, k: int) -> np.ndarray:
    n_shots, t_steps, d = phi.shape
    if n_shots % k != 0:
        raise ValueError(f"k={k} must divide n_shots={n_shots}")
    if k == 1:
        return phi.copy()
    if k == n_shots:
        return np.mean(phi, axis=0, keepdims=True)
    groups = n_shots // k
    return phi.reshape(groups, k, t_steps, d).mean(axis=1)

def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.std(y_true)
    denom = denom if denom > 1e-12 else 1.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / denom)

def sr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.std(y_true)
    denom = denom if denom > 1e-12 else 1.0
    return float(np.std(y_pred) / denom)

def feature_noise_ratio(phi: np.ndarray, cfg: PaperConfig) -> float:
    t0, _, train_end = split_points(phi.shape[1], cfg)
    phi_train = phi[:, t0:train_end, :]
    ev = phi_train.mean(axis=0)
    signal = np.var(ev, axis=0, ddof=1).mean()
    within_single = np.var(phi_train - phi_train.mean(axis=0, keepdims=True), axis=0, ddof=1)
    within_single = np.nanmean(within_single)
    # Use the EV-scale feature-noise pressure: single-shot within variance reduced by N_shots.
    ev_noise = within_single / max(phi.shape[0], 1)
    if signal <= 1e-12:
        return np.nan
    return float(ev_noise / signal)

def ev_condition_diagnostics(phi: np.ndarray, cfg: PaperConfig) -> Tuple[float, float]:
    t0, _, train_end = split_points(phi.shape[1], cfg)
    ev = grouped_features(phi, phi.shape[0])[0]
    X = ev[t0:train_end, :]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n = Xs.shape[0]
    sv = np.linalg.svd(Xs / np.sqrt(max(n, 1)), compute_uv=False)
    smin = float(np.min(sv)) if len(sv) else np.nan
    smax = float(np.max(sv)) if len(sv) else np.nan
    cond = float(smax / smin) if smin > 1e-12 else np.nan
    return smin, cond

def fit_predict_grouped(grouped: np.ndarray, targets: np.ndarray, cfg: PaperConfig, ridge_alpha: float) -> Dict[str, object]:
    _, fit_end, train_end = split_points(grouped.shape[1], cfg)
    t0 = cfg.t0

    X_fit = grouped[:, t0:fit_end, :].reshape(-1, grouped.shape[2])
    y_fit = np.tile(targets[t0:fit_end], grouped.shape[0])

    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit)
    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(X_fit_s, y_fit)

    X_val = grouped[:, fit_end:train_end, :].reshape(-1, grouped.shape[2])
    pred_val = model.predict(scaler.transform(X_val)).reshape(grouped.shape[0], -1).mean(axis=0)

    X_train = grouped[:, t0:train_end, :].reshape(-1, grouped.shape[2])
    y_train = np.tile(targets[t0:train_end], grouped.shape[0])

    scaler_final = StandardScaler()
    X_train_s = scaler_final.fit_transform(X_train)
    model_final = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model_final.fit(X_train_s, y_train)

    X_test = grouped[:, train_end:, :].reshape(-1, grouped.shape[2])
    pred_test = model_final.predict(scaler_final.transform(X_test)).reshape(grouped.shape[0], -1).mean(axis=0)
    y_test = targets[train_end:]

    return {
        "val_nrmse": nrmse(targets[fit_end:train_end], pred_val),
        "test_nrmse": nrmse(y_test, pred_test),
        "test_sr": sr(y_test, pred_test),
        "y_test": y_test,
        "y_pred": pred_test,
    }

def estimate_group_noise(phi: np.ndarray, k: int, cfg: PaperConfig) -> np.ndarray:
    t0, _, train_end = split_points(phi.shape[1], cfg)
    phi_train = phi[:, t0:train_end, :]
    base = np.var(phi_train, axis=0, ddof=1).mean(axis=0)
    return np.asarray(base / max(k, 1), dtype=float)

def fit_diagonal_ridge(X: np.ndarray, y: np.ndarray, alpha: float, noise_diag: np.ndarray, gamma: float) -> Dict[str, object]:
    # X already standardized (centered and scaled), y raw.
    y_mean = float(np.mean(y))
    yc = y - y_mean
    d = X.shape[1]
    A = X.T @ X + alpha * np.eye(d)
    if gamma > 0.0:
        A = A + gamma * np.diag(noise_diag)
    b = X.T @ yc
    beta = np.linalg.solve(A, b)
    return {"coef": beta, "intercept": y_mean}

def predict_diagonal_ridge(X: np.ndarray, fit: Dict[str, object]) -> np.ndarray:
    return X @ fit["coef"] + fit["intercept"]

def fit_predict_grouped_na(
    phi: np.ndarray,
    grouped: np.ndarray,
    k: int,
    targets: np.ndarray,
    cfg: PaperConfig,
    ridge_alpha: float,
    gamma: float,
) -> Dict[str, object]:
    _, fit_end, train_end = split_points(grouped.shape[1], cfg)
    t0 = cfg.t0
    noise_var = estimate_group_noise(phi, k=k, cfg=cfg)

    X_fit = grouped[:, t0:fit_end, :].reshape(-1, grouped.shape[2])
    y_fit = np.tile(targets[t0:fit_end], grouped.shape[0])

    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit)
    scales = np.where(scaler.scale_ > 1e-12, scaler.scale_, 1.0)
    noise_diag_fit = noise_var / (scales ** 2)
    model = fit_diagonal_ridge(X_fit_s, y_fit, alpha=ridge_alpha, noise_diag=noise_diag_fit, gamma=gamma)

    X_val = grouped[:, fit_end:train_end, :].reshape(-1, grouped.shape[2])
    pred_val = predict_diagonal_ridge(scaler.transform(X_val), model).reshape(grouped.shape[0], -1).mean(axis=0)

    X_train = grouped[:, t0:train_end, :].reshape(-1, grouped.shape[2])
    y_train = np.tile(targets[t0:train_end], grouped.shape[0])

    scaler_final = StandardScaler()
    X_train_s = scaler_final.fit_transform(X_train)
    scales_final = np.where(scaler_final.scale_ > 1e-12, scaler_final.scale_, 1.0)
    noise_diag_train = noise_var / (scales_final ** 2)
    model_final = fit_diagonal_ridge(X_train_s, y_train, alpha=ridge_alpha, noise_diag=noise_diag_train, gamma=gamma)

    X_test = grouped[:, train_end:, :].reshape(-1, grouped.shape[2])
    pred_test = predict_diagonal_ridge(scaler_final.transform(X_test), model_final).reshape(grouped.shape[0], -1).mean(axis=0)
    y_test = targets[train_end:]

    return {
        "val_nrmse": nrmse(targets[fit_end:train_end], pred_val),
        "test_nrmse": nrmse(y_test, pred_test),
        "test_sr": sr(y_test, pred_test),
        "gamma": gamma,
        "y_test": y_test,
        "y_pred": pred_test,
    }

def evaluate_protocol(
    phi: np.ndarray,
    targets: np.ndarray,
    protocol: str,
    cfg: PaperConfig,
    ridge_alpha: float,
) -> Dict[str, object]:
    n_shots = phi.shape[0]
    if protocol == "ev":
        candidates = [n_shots]
    elif protocol == "raw":
        candidates = [1]
    elif protocol in {"split", "evdup"}:
        candidates = proper_divisors(n_shots)
        if not candidates:
            raise ValueError(f"n_shots={n_shots} has no proper divisors for protocol={protocol}")
    else:
        raise ValueError(protocol)

    best = None
    for k in candidates:
        if protocol == "ev":
            grouped = grouped_features(phi, n_shots)
        elif protocol == "raw":
            grouped = grouped_features(phi, 1)
        elif protocol == "split":
            grouped = grouped_features(phi, k)
        elif protocol == "evdup":
            ev = grouped_features(phi, n_shots)
            duplicates = n_shots // k
            grouped = np.repeat(ev, duplicates, axis=0)
        result = fit_predict_grouped(grouped, targets, cfg, ridge_alpha=ridge_alpha)
        current = {"k": k, "group_count": grouped.shape[0], **result}
        if best is None or current["val_nrmse"] < best["val_nrmse"] - 1e-12 or (
            abs(current["val_nrmse"] - best["val_nrmse"]) <= 1e-12 and current["k"] < best["k"]
        ):
            best = current
    best["protocol"] = protocol
    best["ridge_alpha"] = ridge_alpha
    return best

def evaluate_protocol_na(
    phi: np.ndarray,
    targets: np.ndarray,
    protocol: str,
    cfg: PaperConfig,
    ridge_alpha: float,
) -> Dict[str, object]:
    n_shots = phi.shape[0]
    if protocol == "ev_na":
        candidates = [n_shots]
        protocol_kind = "ev"
    elif protocol == "split_na":
        candidates = proper_divisors(n_shots)
        protocol_kind = "split"
        if not candidates:
            raise ValueError(f"n_shots={n_shots} has no proper divisors for split_na")
    else:
        raise ValueError(protocol)

    best = None
    for k in candidates:
        grouped = grouped_features(phi, n_shots if protocol_kind == "ev" else k)
        for gamma in cfg.gamma_grid:
            result = fit_predict_grouped_na(phi, grouped, k=(n_shots if protocol_kind == "ev" else k),
                                            targets=targets, cfg=cfg, ridge_alpha=ridge_alpha, gamma=gamma)
            current = {"k": (n_shots if protocol_kind == "ev" else k), "group_count": grouped.shape[0], **result}
            if best is None or current["val_nrmse"] < best["val_nrmse"] - 1e-12 or (
                abs(current["val_nrmse"] - best["val_nrmse"]) <= 1e-12 and current["k"] < best["k"]
            ):
                best = current
    best["protocol"] = protocol
    best["ridge_alpha"] = ridge_alpha
    return best

def build_phi(
    dataset: str,
    budget_total: int,
    n_shots: int,
    sigma: float,
    reservoir_seed: int,
    cfg: PaperConfig,
    q: Optional[int] = None,
    depth: Optional[int] = None,
    entangler: Optional[str] = None,
) -> Dict[str, object]:
    qq = cfg.q_default if q is None else q
    dd = cfg.depth_default if depth is None else depth
    ee = cfg.entangler_default if entangler is None else entangler
    t_steps = budget_total // (2 * n_shots)
    raw = make_series(dataset, total_steps=t_steps)
    inputs, targets = normalize_and_shift(raw, t_steps, cfg)
    reservoir = ExactQRCReservoir(q=qq, depth=dd, entangler=ee, eta=cfg.eta, window=cfg.window, seed=reservoir_seed)
    phi = run_reservoir(inputs=inputs, reservoir=reservoir, n_shots=n_shots, sigma=sigma, shot_seed=reservoir_seed + 100000)
    nu_hat = feature_noise_ratio(phi, cfg)
    # Conditioning diagnostics are not needed for the figure suite itself.
    ev_smin, ev_cond = np.nan, np.nan
    return {
        "phi": phi,
        "targets": targets,
        "t_steps": t_steps,
        "rho_ev": cfg.rho_ev(t_steps, q=qq),
        "nu_hat": nu_hat,
        "ev_smin_scaled": ev_smin,
        "ev_cond": ev_cond,
        "q": qq,
        "depth": dd,
        "entangler": ee,
    }

def run_main_protocols(
    dataset: str,
    budget_total: int,
    n_shots: int,
    sigma: float,
    reservoir_seed: int,
    cfg: PaperConfig,
    q: Optional[int] = None,
    depth: Optional[int] = None,
    entangler: Optional[str] = None,
    include_evdup: bool = False,
    traces: bool = False,
) -> Dict[str, object]:
    built = build_phi(dataset, budget_total, n_shots, sigma, reservoir_seed, cfg, q=q, depth=depth, entangler=entangler)
    phi = built["phi"]
    targets = built["targets"]

    out = {
        "dataset": dataset,
        "budget_total": budget_total,
        "n_shots": n_shots,
        "sigma": sigma,
        "seed": reservoir_seed,
        "rho_ev": built["rho_ev"],
        "nu_hat": built["nu_hat"],
        "ev_smin_scaled": built["ev_smin_scaled"],
        "ev_cond": built["ev_cond"],
        "q": built["q"],
        "depth": built["depth"],
        "entangler": built["entangler"],
        "t_steps": built["t_steps"],
    }

    protocols = ["ev", "raw", "split"] + (["evdup"] if include_evdup else [])
    for protocol in protocols:
        result = evaluate_protocol(phi, targets, protocol=protocol, cfg=cfg, ridge_alpha=cfg.lambda_default)
        out[f"{protocol}_nrmse"] = result["test_nrmse"]
        out[f"{protocol}_sr"] = result["test_sr"]
        out[f"{protocol}_k"] = result["k"]
        out[f"{protocol}_val_nrmse"] = result["val_nrmse"]
        if traces and protocol in {"ev", "raw", "split"}:
            out[f"trace_{protocol}"] = result["y_pred"]
            out["trace_y_test"] = result["y_test"]
    return out

def run_baseline_protocols(
    dataset: str,
    budget_total: int,
    n_shots: int,
    sigma: float,
    reservoir_seed: int,
    cfg: PaperConfig,
) -> Dict[str, object]:
    built = build_phi(dataset, budget_total, n_shots, sigma, reservoir_seed, cfg)
    phi = built["phi"]
    targets = built["targets"]
    out = {
        "dataset": dataset,
        "budget_total": budget_total,
        "n_shots": n_shots,
        "sigma": sigma,
        "seed": reservoir_seed,
        "rho_ev": built["rho_ev"],
        "nu_hat": built["nu_hat"],
        "ev_smin_scaled": built["ev_smin_scaled"],
        "ev_cond": built["ev_cond"],
    }
    for protocol in ["ev", "raw", "split", "evdup"]:
        result = evaluate_protocol(phi, targets, protocol=protocol, cfg=cfg, ridge_alpha=cfg.lambda_default)
        out[f"{protocol}_nrmse"] = result["test_nrmse"]
        out[f"{protocol}_sr"] = result["test_sr"]
        out[f"{protocol}_k"] = result["k"]
    for protocol in ["ev_na", "split_na"]:
        result = evaluate_protocol_na(phi, targets, protocol=protocol, cfg=cfg, ridge_alpha=cfg.lambda_default)
        out[f"{protocol}_nrmse"] = result["test_nrmse"]
        out[f"{protocol}_sr"] = result["test_sr"]
        out[f"{protocol}_k"] = result["k"]
        out[f"{protocol}_gamma"] = result["gamma"]
    return out


# ============================
# Summaries
# ============================

def flatten_rows(rows: List[Dict[str, object]]) -> pd.DataFrame:
    flat_rows = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if isinstance(value, np.ndarray):
                continue
            flat[key] = value
        flat_rows.append(flat)
    return pd.DataFrame(flat_rows)

def summarize(rows_df: pd.DataFrame, group_cols: Sequence[str], protocols: Sequence[str]) -> pd.DataFrame:
    out_rows = []
    for keys, group in rows_df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["seeds"] = int(len(group))
        for protocol in protocols:
            row[f"{protocol}_nrmse_mean"] = float(group[f"{protocol}_nrmse"].mean())
            row[f"{protocol}_nrmse_std"] = float(group[f"{protocol}_nrmse"].std(ddof=1)) if len(group) > 1 else 0.0
            if f"{protocol}_sr" in group.columns:
                row[f"{protocol}_sr_mean"] = float(group[f"{protocol}_sr"].mean())
                row[f"{protocol}_sr_std"] = float(group[f"{protocol}_sr"].std(ddof=1)) if len(group) > 1 else 0.0
            if f"{protocol}_k" in group.columns:
                modes = group[f"{protocol}_k"].mode()
                row[f"{protocol}_k_mode"] = int(modes.iloc[0]) if not modes.empty else np.nan
            if f"{protocol}_gamma" in group.columns:
                modes = group[f"{protocol}_gamma"].mode()
                row[f"{protocol}_gamma_mode"] = float(modes.iloc[0]) if not modes.empty else np.nan
        if {"ev_nrmse", "split_nrmse"}.issubset(group.columns):
            row["gap_ev_minus_split"] = float(group["ev_nrmse"].mean() - group["split_nrmse"].mean())
            row["split_win_rate"] = float(np.mean(group["split_nrmse"] < group["ev_nrmse"]))
        for col in ["rho_ev", "nu_hat", "ev_smin_scaled", "ev_cond"]:
            if col in group.columns:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
                if col not in row:
                    row[col] = float(group[col].mean())
        out_rows.append(row)
    df = pd.DataFrame(out_rows)
    sort_cols = [c for c in ["dataset", "entangler", "depth", "q", "budget_total", "n_shots", "sigma"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


# ============================
# Theory helpers
# ============================

def mp_edge(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=float)
    out = np.zeros_like(rho)
    mask = rho > 0.0
    out[mask] = np.maximum(1.0 - rho[mask] ** (-0.5), 0.0) ** 2
    return out

def risk_proxy_u(rho: np.ndarray, nu: np.ndarray, u: float) -> np.ndarray:
    gamma = mp_edge(np.asarray(rho) / u)
    return np.asarray(nu) / u + 1.0 / np.maximum(gamma, 1e-8)

def split_favored_region(rho_grid: np.ndarray, nu_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u_vals = np.linspace(0.05, 1.0, 240)
    region = np.zeros((len(nu_grid), len(rho_grid)), dtype=float)
    best_u = np.ones_like(region)
    for i, nu in enumerate(nu_grid):
        for j, rho in enumerate(rho_grid):
            risks = risk_proxy_u(rho, nu, u_vals)
            idx = int(np.argmin(risks))
            region[i, j] = 1.0 if u_vals[idx] < 0.999 else 0.0
            best_u[i, j] = u_vals[idx]
    return region, best_u

def summarize_theory_empirics(phase_summary: pd.DataFrame) -> pd.DataFrame:
    df = phase_summary.copy()
    df["theory_split_region"] = np.nan
    rho_grid = np.linspace(0.20, 5.50, 420)
    nu_grid = np.geomspace(2e-3, 2.0, 360)
    region, _ = split_favored_region(rho_grid, nu_grid)
    def interp_region(rho: float, nu: float) -> float:
        if not np.isfinite(rho) or not np.isfinite(nu):
            return np.nan
        i = int(np.clip(np.searchsorted(nu_grid, nu), 1, len(nu_grid)-1))
        j = int(np.clip(np.searchsorted(rho_grid, rho), 1, len(rho_grid)-1))
        return float(region[i, j])
    df["theory_split_region"] = [interp_region(r, n) for r, n in zip(df["rho_ev_mean"], df["nu_hat_mean"])]
    df["empirical_split_better"] = (df["gap_ev_minus_split"] > 0.0).astype(float)
    df["theory_correct"] = (df["theory_split_region"] == df["empirical_split_better"]).astype(float)
    return df


# ============================
# Figure builders
# ============================

TITLE_MAP = {"mackey_glass": "Mackey--Glass", "lorenz": "Lorenz", "narma10": "NARMA10"}

def representative_errorbar_limits(values: np.ndarray, errors: np.ndarray) -> Tuple[float, float]:
    ymin = float(np.min(values - errors))
    ymax = float(np.max(values + errors))
    ymin = max(0.0, min(0.45, math.floor(ymin / 0.05) * 0.05))
    ymax = max(1.01, math.ceil(ymax / 0.05) * 0.05)
    return ymin, ymax

def make_figure_1(rep_rows: List[Dict[str, object]], out_stem: Path) -> None:
    datasets = ["mackey_glass", "lorenz", "narma10"]
    fig, axes = plt.subplots(
        3, 2, figsize=(13.2, 10.6),
        gridspec_kw={"width_ratios": [2.60, 1.05]},
        constrained_layout=True,
    )
    for i, dataset in enumerate(datasets):
        sub = [r for r in rep_rows if r["dataset"] == dataset]
        y_true = np.stack([r["trace_y_test"] for r in sub], axis=0)
        y_ref = y_true.mean(axis=0)
        trace_ax = axes[i, 0]
        for key, color, ls, lw, alpha in [
            ("ev", FD_COLOR, (0, (4, 2)), 2.0, 0.14),
            ("raw", REFERENCE_COLOR, ":", 1.4, 0.10),
            ("split", ID_COLOR, "-", 2.2, 0.18),
        ]:
            arr = np.stack([r[f"trace_{key}"] for r in sub], axis=0)
            mean = arr.mean(axis=0)
            sd = arr.std(axis=0, ddof=1)
            trace_ax.fill_between(np.arange(len(mean)), mean - sd, mean + sd, color=color, alpha=alpha, linewidth=0.0)
            trace_ax.plot(mean, color=color, ls=ls, lw=lw, label={"ev": "EV", "raw": "Raw", "split": "Split"}[key])
        trace_ax.plot(y_ref, color="#222222", lw=2.0, label="True")
        cfg0 = sub[0]
        trace_ax.set_title(
            rf"{TITLE_MAP[dataset]} --- $S={int(cfg0['budget_total'])},\,N={int(cfg0['n_shots'])},\,\sigma={cfg0['sigma']:.2f},\,\rho_{{EV}}={cfg0['rho_ev']:.2f}$"
        )
        trace_ax.set_ylabel("Normalized target")
        trace_ax.set_xlabel("Test time step")
        style_axis(trace_ax)
        if i == 0:
            handles, labels = trace_ax.get_legend_handles_labels()
            order = [3, 0, 1, 2]
            trace_ax.legend([handles[j] for j in order], [labels[j] for j in order], ncol=2, loc="upper left", frameon=False)

        bar_ax = axes[i, 1]
        protos = ["ev", "raw", "split"]
        xs = np.arange(3)
        means = np.array([np.mean([r[f"{p}_nrmse"] for r in sub]) for p in protos], dtype=float)
        sds = np.array([np.std([r[f"{p}_nrmse"] for r in sub], ddof=1) for p in protos], dtype=float)
        colors = [FD_COLOR, REFERENCE_COLOR, ID_COLOR]
        markers = ["s", "^", "o"]
        labels = ["EV", "Raw", "Split"]
        rng = np.random.default_rng(42 + i)
        for j, p in enumerate(protos):
            seed_vals = np.array([r[f"{p}_nrmse"] for r in sub], dtype=float)
            jitter = rng.normal(scale=0.03, size=len(seed_vals))
            bar_ax.scatter(np.full_like(seed_vals, xs[j], dtype=float) + jitter, seed_vals,
                           s=18, color=colors[j], alpha=0.22, edgecolors="none")
            bar_ax.errorbar(xs[j], means[j], yerr=sds[j], fmt=markers[j], color=colors[j],
                            markersize=(5.2 if p == "raw" else 5.5), markeredgecolor="white",
                            markeredgewidth=0.9, capsize=4.0, lw=1.7)
        ymin, ymax = representative_errorbar_limits(means, sds)
        bar_ax.set_ylim(ymin, ymax)
        bar_ax.set_xticks(xs, labels)
        bar_ax.set_ylabel("Test NRMSE")
        bar_ax.set_title(r"mean $\pm$ 1 SD over 12 seeds")
        style_axis(bar_ax)
    save_figure(fig, out_stem)

def make_figure_2_phase(phase_summary: pd.DataFrame, out_stem: Path) -> None:
    datasets = ["mackey_glass", "lorenz", "narma10"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 3.7), constrained_layout=True)
    vmax = max(0.05, float(np.nanmax(np.abs(phase_summary["gap_ev_minus_split"].to_numpy()))))
    for ax, dataset in zip(axes, datasets):
        sub = phase_summary[phase_summary["dataset"] == dataset].copy()
        pivot = sub.pivot(index="n_shots", columns="sigma", values="gap_ev_minus_split").sort_index()
        yvals = np.array(pivot.index.to_list(), dtype=float)
        xvals = np.array(pivot.columns.to_list(), dtype=float)
        Z = pivot.loc[yvals, xvals].to_numpy()
        im = ax.imshow(
            Z, origin="lower", aspect="auto",
            extent=[xvals.min(), xvals.max(), yvals.min(), yvals.max()],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax
        )
        # overlay stars when split wins >= 75% seeds
        win_pivot = sub.pivot(index="n_shots", columns="sigma", values="split_win_rate").sort_index()
        W = win_pivot.loc[yvals, xvals].to_numpy()
        for iy, y in enumerate(yvals):
            for ix, x in enumerate(xvals):
                if W[iy, ix] >= 0.75:
                    ax.text(x, y, r"$\star$", ha="center", va="center", color="white", fontsize=9)
        ax.set_title(TITLE_MAP[dataset])
        ax.set_xlabel(r"$\sigma$")
        ax.set_ylabel(r"$N_{\mathrm{shots}}$")
        ax.set_xticks(xvals)
        ax.set_yticks(yvals)
        ax.grid(False)
        style_axis(ax)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label(r"Gap = NRMSE(EV) $-$ NRMSE(Split)")
    save_figure(fig, out_stem)

def make_figure_3_theory(theory_df: pd.DataFrame, out_stem: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.1), constrained_layout=True)

    # Panel A: theory region
    rho_grid = np.linspace(0.20, 5.50, 300)
    nu_grid = np.geomspace(2e-3, 2.0, 220)
    region, best_u = split_favored_region(rho_grid, nu_grid)
    ax = axes[0]
    cmap = ListedColormap([FD_COLOR, ID_COLOR])
    ax.pcolormesh(rho_grid, nu_grid, region, shading="auto", cmap=cmap, alpha=0.22)
    cs = ax.contour(rho_grid, nu_grid, best_u, levels=[0.125, 0.25, 0.5, 1.0], colors=[AXIS_COLOR]*4, linewidths=0.7)
    ax.clabel(cs, fmt={0.125: r"$u^\star=0.125$", 0.25: r"$u^\star=0.25$", 0.5: r"$u^\star=0.5$", 1.0: r"$u^\star=1$"}, fontsize=8)
    for xv, label in [(1.0, r"$\rho_{EV}=1$"), (4.0, r"$\rho_{EV}=4$")]:
        ax.axvline(xv, color=REFERENCE_COLOR, lw=1.0, ls=":", alpha=0.85)
        ax.text(xv + 0.05, 0.0024, label, rotation=90, va="bottom", ha="left", color=MUTED_COLOR, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlim(0.20, 5.50)
    ax.set_xlabel(r"$\rho_{EV}$")
    ax.set_ylabel(r"feature-noise pressure $\hat{\nu}$")
    ax.set_title("Theory surrogate")
    ax.text(0.58, 0.84, "Split region", color=ID_COLOR, transform=ax.transAxes)
    ax.text(0.58, 0.11, "EV region", color=FD_COLOR, transform=ax.transAxes)
    style_axis(ax)

    # Panel B: empirical scatter in regime plane
    ax = axes[1]
    for dataset, marker, color in [("mackey_glass", "o", ID_COLOR), ("lorenz", "s", FD_COLOR), ("narma10", "D", MUTED_COLOR)]:
        sub = theory_df[theory_df["dataset"] == dataset]
        split_sub = sub[sub["empirical_split_better"] > 0.5]
        ev_sub = sub[sub["empirical_split_better"] <= 0.5]
        ax.scatter(split_sub["rho_ev_mean"], split_sub["nu_hat_mean"], marker=marker, color=color,
                   s=36, alpha=0.88, edgecolors="white", linewidths=0.5, label=f"{TITLE_MAP[dataset]} (split)")
        if len(ev_sub) > 0:
            ax.scatter(ev_sub["rho_ev_mean"], ev_sub["nu_hat_mean"], marker=marker, facecolors="none",
                       edgecolors=color, s=36, alpha=0.95, linewidths=1.0, label=f"{TITLE_MAP[dataset]} (EV)")
    ax.axvline(1.0, color=REFERENCE_COLOR, lw=1.0, ls=":", alpha=0.85)
    ax.axvline(4.0, color=REFERENCE_COLOR, lw=1.0, ls=":", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlim(0.20, 5.50)
    ax.set_xlabel(r"$\rho_{EV}$")
    ax.set_ylabel(r"estimated $\hat{\nu}$")
    acc = float(theory_df["theory_correct"].mean())
    ax.set_title(rf"Empirical regime test (agreement {acc:.2f})")
    # deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    dedup_h, dedup_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            dedup_h.append(h); dedup_l.append(l); seen.add(l)
    ax.legend(dedup_h, dedup_l, frameon=False, fontsize=8, loc="upper right")
    style_axis(ax)

    # Panel C: binned gap vs rho
    ax = axes[2]
    bins = np.array([0.20, 0.60, 1.00, 1.50, 2.25, 3.25, 5.50], dtype=float)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for dataset, marker, color in [("mackey_glass", "o", ID_COLOR), ("lorenz", "s", FD_COLOR), ("narma10", "D", MUTED_COLOR)]:
        sub = theory_df[theory_df["dataset"] == dataset].copy()
        sub["rho_bin"] = pd.cut(sub["rho_ev_mean"], bins=bins, include_lowest=True, labels=False)
        grp = sub.dropna(subset=["rho_bin"]).groupby("rho_bin")["gap_ev_minus_split"]
        xs, ys, es = [], [], []
        for b, vals in grp:
            vals = np.asarray(vals, dtype=float)
            xs.append(bin_centers[int(b)])
            ys.append(vals.mean())
            es.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker=marker, color=color, ms=5.0,
                        markeredgecolor="white", markeredgewidth=0.7, lw=1.4, capsize=2.5, label=TITLE_MAP[dataset])
    ax.axhline(0.0, color=AXIS_COLOR, lw=1.0, ls=":", alpha=0.85)
    ax.axvline(1.0, color=REFERENCE_COLOR, lw=1.0, ls=":", alpha=0.85)
    ax.axvline(4.0, color=REFERENCE_COLOR, lw=1.0, ls=":", alpha=0.85)
    ax.set_xlim(0.20, 5.50)
    ax.set_xlabel(r"$\rho_{EV}$")
    ax.set_ylabel(r"Gap = NRMSE(EV) $-$ NRMSE(Split)")
    ax.set_title("Binned split gain")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axis(ax)

    save_figure(fig, out_stem)

def make_figure_4_baselines(summary_df: pd.DataFrame, out_stem: Path) -> None:
    datasets = ["mackey_glass", "lorenz", "narma10"]
    methods = [
        ("ev", "EV", FD_COLOR, "s"),
        ("raw", "Raw", REFERENCE_COLOR, "^"),
        ("evdup", "EV-dup", MUTED_COLOR, "v"),
        ("ev_na", "EV-NA", FD_COLOR, "P"),
        ("split", "Split", ID_COLOR, "o"),
        ("split_na", "Split-NA", ID_COLOR, "X"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.1, 3.8), constrained_layout=True)
    for ax, dataset in zip(axes, datasets):
        row = summary_df[summary_df["dataset"] == dataset].iloc[0]
        xs = np.arange(len(methods))
        means = np.array([row[f"{m[0]}_nrmse_mean"] for m in methods], dtype=float)
        sds = np.array([row[f"{m[0]}_nrmse_std"] for m in methods], dtype=float)
        for x, (key, label, color, marker), mean, sd in zip(xs, methods, means, sds):
            ax.errorbar(x, mean, yerr=sd, fmt=marker, color=color, ms=5.8, markeredgecolor="white",
                        markeredgewidth=0.8, lw=1.6, capsize=3.2)
        ax.set_xticks(xs, [m[1] for m in methods], rotation=30, ha="right")
        ax.set_ylabel("Test NRMSE")
        ax.set_title(TITLE_MAP[dataset])
        ymin, ymax = representative_errorbar_limits(means, sds)
        ax.set_ylim(ymin, ymax)
        style_axis(ax)
    save_figure(fig, out_stem)

def make_figure_5_lambda(lambda_summary: pd.DataFrame, out_stem: Path) -> None:
    datasets = ["mackey_glass", "lorenz", "narma10"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 3.8), constrained_layout=True)
    for ax, dataset in zip(axes, datasets):
        sub = lambda_summary[lambda_summary["dataset"] == dataset]
        for protocol, color, ls, lw, alpha in [
            ("ev", FD_COLOR, (0, (4, 2)), 2.0, 0.14),
            ("raw", REFERENCE_COLOR, ":", 1.2, 0.10),
            ("split", ID_COLOR, "-", 2.2, 0.18),
        ]:
            ss = sub[sub["protocol"] == protocol].sort_values("lambda")
            x = ss["lambda"].to_numpy()
            y = ss["nrmse_mean"].to_numpy()
            e = ss["nrmse_std"].to_numpy()
            ax.semilogx(x, y, color=color, ls=ls, lw=lw, label={"ev": "EV", "raw": "Raw", "split": "Split"}[protocol])
            ax.fill_between(x, y - e, y + e, color=color, alpha=alpha, linewidth=0.0)
        ax.axvline(10.0, color=REFERENCE_COLOR, lw=1.0, ls=":", alpha=0.85)
        ax.set_title(TITLE_MAP[dataset])
        ax.set_xlabel(r"Ridge $\lambda$")
        ax.set_ylabel("Test NRMSE")
        if dataset == "mackey_glass":
            ax.legend(frameon=False, loc="upper right")
        style_axis(ax)
    save_figure(fig, out_stem)

def make_figure_6_architecture(arch_summary: pd.DataFrame, out_stem: Path) -> None:
    datasets = ["mackey_glass", "lorenz", "narma10"]
    name_map = {
        ("ring_cnot", 1): "Ring d=1",
        ("ring_cnot", 2): "Ring d=2",
        ("ring_cnot", 4): "Ring d=4",
        ("line_cnot", 2): "Line d=2",
        ("all_to_all_cz", 2): "All-to-all d=2",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.1, 3.85), constrained_layout=True)
    for ax, dataset in zip(axes, datasets):
        sub = arch_summary[arch_summary["dataset"] == dataset].copy()
        sub["variant"] = sub.apply(lambda r: name_map[(r["entangler"], int(r["depth"]))], axis=1)
        sub["gap_std_proxy"] = np.sqrt(sub["ev_nrmse_std"] ** 2 + sub["split_nrmse_std"] ** 2)
        sub = sub.sort_values("gap_ev_minus_split", ascending=False)
        x = np.arange(len(sub))
        ax.bar(x, sub["gap_ev_minus_split"], color=ID_COLOR, alpha=0.65, width=0.72)
        ax.errorbar(x, sub["gap_ev_minus_split"], yerr=sub["gap_std_proxy"], fmt="none", ecolor=AXIS_COLOR, lw=0.9, capsize=2.5)
        ax.axhline(0.0, color=AXIS_COLOR, lw=1.0, ls=":", alpha=0.85)
        ax.set_xticks(x, sub["variant"], rotation=28, ha="right")
        ax.set_title(TITLE_MAP[dataset])
        ax.set_ylabel(r"Gap = NRMSE(EV) $-$ NRMSE(Split)")
        style_axis(ax)
    save_figure(fig, out_stem)

def make_appendix_duplication(dup_summary: pd.DataFrame, out_stem: Path) -> None:
    datasets = ["mackey_glass", "lorenz", "narma10"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 3.75), constrained_layout=True)
    for ax, dataset in zip(axes, datasets):
        sub = dup_summary[dup_summary["dataset"] == dataset].sort_values("n_shots")
        for protocol, color, label, ls, lw, alpha in [
            ("ev", FD_COLOR, "EV", (0, (4, 2)), 2.0, 0.12),
            ("evdup", MUTED_COLOR, "EV-dup", "-.", 1.5, 0.10),
            ("split", ID_COLOR, "Split", "-", 2.2, 0.16),
        ]:
            x = sub["n_shots"].to_numpy()
            y = sub[f"{protocol}_nrmse_mean"].to_numpy()
            e = sub[f"{protocol}_nrmse_std"].to_numpy()
            ax.plot(x, y, color=color, ls=ls, lw=lw, label=label)
            ax.fill_between(x, y - e, y + e, color=color, alpha=alpha, linewidth=0.0)
        ax.set_title(TITLE_MAP[dataset])
        ax.set_xlabel(r"$N_{\mathrm{shots}}$")
        ax.set_ylabel("Test NRMSE")
        if dataset == "mackey_glass":
            ax.legend(frameon=False, loc="upper right")
        style_axis(ax)
    save_figure(fig, out_stem)

def make_appendix_size(size_summary: pd.DataFrame, out_stem: Path) -> None:
    datasets = ["mackey_glass", "lorenz", "narma10"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 3.75), constrained_layout=True)
    for ax, dataset in zip(axes, datasets):
        sub = size_summary[size_summary["dataset"] == dataset].sort_values("q")
        x = sub["q"].to_numpy()
        y = sub["gap_ev_minus_split"].to_numpy()
        e = np.sqrt(sub["ev_nrmse_std"].to_numpy() ** 2 + sub["split_nrmse_std"].to_numpy() ** 2)
        ax.plot(x, y, color=ID_COLOR, marker="o", ms=5.5, markeredgecolor="white", markeredgewidth=0.9, lw=2.0)
        ax.fill_between(x, y - e, y + e, color=ID_COLOR, alpha=0.18, linewidth=0.0)
        ax.axhline(0.0, color=AXIS_COLOR, lw=1.0, ls=":", alpha=0.85)
        ax.set_xticks(x)
        ax.set_title(TITLE_MAP[dataset])
        ax.set_xlabel(r"Reservoir size $Q$")
        ax.set_ylabel(r"Gap = NRMSE(EV) $-$ NRMSE(Split)")
        style_axis(ax)
    save_figure(fig, out_stem)


# ============================
# Docs
# ============================

def build_readme() -> str:
    return """# Split-Ensemble QRC paper repo (corrected synthetic suite)

This repository is a clean, corrected synthetic study for the split-ensemble paper idea.

## What is fixed relative to the earlier draft
- Exact budget accounting: `S_total = 2 * N_shots * T`.
- Joint bitstring sampling for the two non-commuting measurement settings (`Z` and `X`).
- One consistent protocol across Mackey-Glass, Lorenz, and NARMA10.
- Representative plots are seed-averaged, not cherry-picked single trajectories.
- The `rho_EV` section is no longer framed as a universal empirical crossover; instead it is tied to a grouped-view conditioning theory and tested against data.

## Repository layout
- `scripts/run_paper_suite.py`: end-to-end experiment runner.
- `results/`: seed-level and summary CSV files.
- `figures/`: all paper and appendix figures in both PDF and PNG.
- `docs/`: technical report, theory note, captions, and review-response notes.

## Main figures
- `experiment_1.pdf`
- `figure_2_phase_maps.pdf`
- `figure_3_theory_regime_test.pdf`
- `experiment_4.pdf`
- `figure_5_lambda_sweeps.pdf`
- `figure_6_architecture_ablation.pdf`

## Appendix figures
- `experiment_5.pdf`
- `experiment_3.pdf`

## Reproduction
```bash
python scripts/run_paper_suite.py
```

The default run uses 12 seeds for the confirmation experiments and writes all outputs into the repository itself.
"""

def build_technical_report(
    cfg: PaperConfig,
    rep_summary: pd.DataFrame,
    phase_summary: pd.DataFrame,
    theory_df: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    lambda_summary: pd.DataFrame,
    arch_summary: pd.DataFrame,
    size_summary: pd.DataFrame,
) -> str:
    rep = rep_summary.set_index("dataset")
    lines = []
    lines.append("# Split-Ensemble QRC: corrected synthetic paper suite\n\n")
    lines.append("## Executive summary\n\n")
    lines.append("This bundle rebuilds the synthetic part of the split-ensemble study under a fully corrected protocol. ")
    lines.append("The design is budget-fair (`S_total = 2 * N_shots * T`), uses joint bitstring sampling in both measurement settings, ")
    lines.append("and uses the same estimator and inference rule across protocols. Representative plots are seed-averaged over 12 confirmation seeds.\n\n")
    lines.append("## Fixed experimental protocol\n\n")
    lines.append(f"- Reservoir: `Q={cfg.q_default}`, depth `{cfg.depth_default}`, entangler `{cfg.entangler_default}`, `eta={cfg.eta}`, window `L={cfg.window}`.\n")
    lines.append(f"- Feature dimension: `d = 2 * Q * L = {cfg.d_default}`.\n")
    lines.append(f"- Washout `t0 = {cfg.t0}`, training fraction `{cfg.train_frac}`, validation-within-train fraction `{cfg.val_frac_within_train}`.\n")
    lines.append(f"- Default ridge parameter: `lambda = {cfg.lambda_default}`.\n")
    lines.append(f"- Confirmation seeds: `{cfg.confirmation_seeds[0]}..{cfg.confirmation_seeds[-1]}` (`n={len(cfg.confirmation_seeds)}`).\n\n")
    lines.append("## Representative operating points (12-seed confirmation averages)\n\n")
    lines.append("| Dataset | Nshots | sigma | rho_EV | EV | Raw | Split | Gap EV-Split |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for ds in ["mackey_glass", "lorenz", "narma10"]:
        row = rep.loc[ds]
        lines.append(
            f"| {TITLE_MAP[ds]} | {int(row['n_shots'])} | {row['sigma']:.2f} | {row['rho_ev']:.2f} | "
            f"{row['ev_nrmse_mean']:.3f} +/- {row['ev_nrmse_std']:.3f} | "
            f"{row['raw_nrmse_mean']:.3f} +/- {row['raw_nrmse_std']:.3f} | "
            f"{row['split_nrmse_mean']:.3f} +/- {row['split_nrmse_std']:.3f} | "
            f"{row['gap_ev_minus_split']:.3f} |\n"
        )
    lines.append("\n")
    lines.append("## Stronger scientific insight behind `rho_EV`\n\n")
    lines.append("The old draft framed the split/EV transition as an empirical crossover near `rho_EV ~ 2-4`. ")
    lines.append("This refresh replaces that claim with a grouped-view regime theory.\n\n")
    lines.append("Let `x_{t,g}^{(k)} = z_t + eps_{t,g}^{(k)}` denote the grouped feature view obtained by averaging `k` shots, ")
    lines.append("with `G = Nshots / k` groups. Then the grouped design has effective aspect ratio\n\n")
    lines.append(r"$$\rho_k = \frac{G T_{\mathrm{train}}}{d} = \frac{N_{\mathrm{shots}}}{k}\rho_{EV}.$$" + "\n\n")
    lines.append("The grouped-view surrogate then balances two effects:\n\n")
    lines.append(r"$$\mathcal{R}(u;\rho,\nu) \approx \frac{\nu}{u} + \frac{1}{\gamma_{MP}(\rho/u)}, \qquad \gamma_{MP}(\rho)=(1-\rho^{-1/2})_+^2,$$" + "\n\n")
    lines.append("with `u = k / Nshots`. The first term penalizes noisier grouped rows; the second penalizes poor conditioning. ")
    lines.append("The theory therefore predicts that split-ensemble helps exactly in the regime where the conditioning gain dominates the denoising loss. ")
    lines.append("The landmarks `rho=1` and `rho=4` come from the Marchenko-Pastur lower edge and should be described as conditioning landmarks, not universal empirical thresholds.\n\n")
    lines.append("## Theory-vs-experiment summary\n\n")
    agree = float(theory_df["theory_correct"].mean())
    split_frac = float(theory_df["empirical_split_better"].mean())
    lines.append(f"- Fraction of aggregated phase-map configurations where Split beats EV: `{split_frac:.2f}`.\n")
    lines.append(f"- Agreement between the theory-region classifier and empirical winner on the aggregated phase-map points: `{agree:.2f}`.\n")
    lines.append("- This should be written as a theory-guided regime picture, not as a theorem for all QRC tasks.\n\n")
    lines.append("## Stronger baseline controls\n\n")
    lines.append("We include both protocol-matched controls (`Raw`, `EV-dup`) and estimator-level controls (`EV-NA`, `Split-NA`). ")
    lines.append("The latter are implemented as variance-aware diagonal-penalty ridge fits using an estimated grouped measurement-noise covariance. ")
    lines.append("They operationalize the complementary 'noise-aware regularization' axis discussed in the paper.\n\n")
    lines.append("| Dataset | EV | Raw | EV-dup | EV-NA | Split | Split-NA |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    base = baseline_summary.set_index("dataset")
    for ds in ["mackey_glass", "lorenz", "narma10"]:
        row = base.loc[ds]
        lines.append(
            f"| {TITLE_MAP[ds]} | {row['ev_nrmse_mean']:.3f} | {row['raw_nrmse_mean']:.3f} | {row['evdup_nrmse_mean']:.3f} | "
            f"{row['ev_na_nrmse_mean']:.3f} | {row['split_nrmse_mean']:.3f} | {row['split_na_nrmse_mean']:.3f} |\n"
        )
    lines.append("\n")
    lines.append("## Robustness checks\n\n")
    lines.append(f"- Lambda sweep: {lambda_summary['lambda'].nunique()} log-spaced values between `{lambda_summary['lambda'].min():.1e}` and `{lambda_summary['lambda'].max():.1e}`.\n")
    lines.append(f"- Architecture ablation: {arch_summary.shape[0]} aggregated points covering {len(cfg.architecture_variants)} variants.\n")
    lines.append(f"- Reservoir-size appendix: `Q in {list(cfg.reservoir_sizes)}`.\n\n")
    lines.append("## What should be claimed in the paper\n\n")
    lines.append("1. Split-ensemble is a clean finite-shot data-organization mechanism under a fixed quantum budget.\n")
    lines.append("2. `rho_EV` is a theory-motivated regime axis, but the exact threshold is task- and architecture-dependent.\n")
    lines.append("3. The novelty claim should remain moderate: this is a principled protocol contribution plus a theory-tested regime insight, not a new learning paradigm.\n")
    lines.append("4. The hardware claim should remain narrow unless the hardware section is expanded substantially.\n\n")
    lines.append("## Embedded figure previews\n\n")
    for fig in [
        "experiment_1.png",
        "figure_2_phase_maps.png",
        "figure_3_theory_regime_test.png",
        "experiment_4.png",
        "figure_5_lambda_sweeps.png",
        "figure_6_architecture_ablation.png",
    ]:
        lines.append(f"![{fig}](../figures/{fig})\n\n")
    return "".join(lines)

def build_theory_note() -> str:
    return r"""# Grouped-view theory note

## Core model

Let the latent clean feature at time step `t` be `z_t in R^d`, and suppose the target satisfies

$$
y_t = \beta_\star^\top z_t + \xi_t.
$$

Under finite shots, the grouped feature obtained by averaging `k` shots is modeled as

$$
x^{(k)}_{t,g} = z_t + \varepsilon^{(k)}_{t,g}, \qquad
\mathbb{E}\,\varepsilon^{(k)}_{t,g}=0, \qquad
\mathrm{Cov}(\varepsilon^{(k)}_{t,g})=\frac{\Sigma_\varepsilon}{k},
$$

with `g = 1,...,G` and `G = Nshots / k`.

This yields three protocol endpoints:
- EV: `k = Nshots`, `G = 1`
- Raw: `k = 1`, `G = Nshots`
- Split: `1 < k < Nshots`

## Proposition 1: effective aspect ratio

If EV has aspect ratio

$$
\rho_{EV} = \frac{T_{\mathrm{train}}}{d},
$$

then grouped training with group size `k` has aspect ratio

$$
\rho_k = \frac{G T_{\mathrm{train}}}{d}
       = \frac{N_{\mathrm{shots}}}{k}\rho_{EV}.
$$

### Consequence
Grouping does not create new information, but it does reallocate the fixed shot budget between
row-wise denoising and design conditioning.

## Proposition 2: conditioning landmarks

Under the isotropic-design idealization, the asymptotic lower spectral edge of the grouped design is

$$
\gamma_{MP}(\rho_k) = (1-\rho_k^{-1/2})_+^2.
$$

Hence:
- if `rho_k <= 1`, the grouped design is rank-deficient in the MP limit;
- if `1 < rho_k < 4`, the lower edge is below `0.25`, so conditioning is still fragile;
- if `rho_k >= 4`, the lower edge is at least `0.25`, so EV is no longer strongly sample-starved.

These values `1` and `4` are therefore theory-based conditioning landmarks.

## Proposition 3: grouped-view risk proxy

Introduce the normalized feature-noise pressure

$$
\nu = \frac{\operatorname{tr}(\Sigma_\varepsilon)/d}{\operatorname{Var}(z)}.
$$

In the grouped-view surrogate, a simple excess-risk proxy is

$$
\mathcal{R}(u;\rho,\nu)
\approx
\frac{\nu}{u} + \frac{1}{\gamma_{MP}(\rho/u)},
\qquad
u = k/N_{\mathrm{shots}} \in (0,1].
$$

Interpretation:
- `nu/u` penalizes feature noise; it prefers larger groups (larger `u`);
- `1/gamma_MP(rho/u)` penalizes poor conditioning; it prefers smaller groups (smaller `u`).

Therefore the optimal `u^*` is split-like (`u^* < 1`) exactly when the conditioning gain from more grouped rows
dominates the denoising gain from full EV averaging.

## What is theoretical vs. what is empirical

### Theoretical
- `rho_EV` is the correct EV conditioning axis;
- grouping rescales aspect ratio as `rho_k = (Nshots/k) rho_EV`;
- the landmarks `rho=1` and `rho=4` arise from the MP lower edge;
- split should be favored in a low- or moderate-`rho_EV` region whose boundary shifts with `nu`.

### Empirical
- the exact location of the split/EV boundary for a given task;
- the best discrete group size `k`;
- the precise mapping from the simulator noise model to the surrogate's `nu`.

## Paper-safe claim

A defensible statement is:

> Split-ensemble should help when EV is still weakly conditioned (`rho_EV` small or moderate) and the grouped feature-noise pressure is not so large that full EV averaging dominates.

That is a theory-guided regime claim, not a universal threshold theorem.
"""

def build_captions() -> str:
    return """# Figure captions

## Experiment 1
Seed-averaged representative operating points for Mackey--Glass, Lorenz, and NARMA10 under the corrected finite-shot protocol `S_total = 2 N_shots T`.
Left panels show the mean predicted trajectories over 12 confirmation seeds with one-standard-deviation bands.
Right panels show the corresponding seed-level test NRMSE points together with the mean and one-standard-deviation error bars.

## Figure 2
Phase maps of the split gain at fixed total budget.
Each cell shows `NRMSE(EV) - NRMSE(Split)`, so positive values favor Split-Ensemble.
Stars mark cells where Split wins on at least 75\% of the confirmation seeds.

## Figure 3
Theory-backed regime figure.
Panel A shows the split-favored region of the grouped-view risk surrogate in the `(rho_EV, nu)` plane.
Panel B overlays the aggregated phase-map configurations in the same plane and marks whether Split or EV wins empirically.
Panel C shows the binned split gain versus `rho_EV`; the vertical lines at `rho_EV=1` and `rho_EV=4` are conditioning landmarks from the Marchenko--Pastur lower edge.

## Figure 4
Representative-point comparison against stronger finite-shot baselines.
`EV-dup` controls for row-count inflation without creating new grouped views, while `EV-NA` and `Split-NA` add variance-aware estimator-level corrections.

## Figure 5
Lambda sweep at the representative operating points.
Curves show mean test NRMSE over 12 seeds with one-standard-deviation bands.
The vertical dotted line marks the default value `lambda=10` used in the main comparisons.

## Figure 6
Architecture ablation across entangler families and depths.
Bars show `NRMSE(EV) - NRMSE(Split)`, so positive bars favor Split-Ensemble.

## Appendix Figure A
Duplication-control sweep versus `N_shots` at the representative sigma level for each dataset.

## Appendix Figure B
Reservoir-size ablation (`Q in {4,5,6}`) at the representative operating points.
"""

def build_review_response_notes() -> str:
    return """# Review-response notes

## Criticism: novelty is moderate rather than deep
Response:
- Keep the novelty framing modest.
- Present the contribution as a clean finite-shot protocol mechanism plus a theory-guided regime analysis, not as a fundamentally new learning paradigm.

## Criticism: theory is mostly heuristic
Response:
- Replace the empirical-crossover language with the grouped-view surrogate theory.
- State clearly what is theoretical (effective aspect ratio, MP landmarks, surrogate trade-off) and what remains empirical (exact threshold location, best discrete group size).

## Criticism: no direct comparison to stronger finite-shot correction methods
Response:
- Add estimator-level variance-aware baselines (`EV-NA`, `Split-NA`) in the representative-point comparison.
- Describe them as complementary estimator-level corrections that act after sample construction.

## Criticism: hardware validation is narrow
Response:
- Keep hardware out of the central paper claim unless the hardware section is expanded.
- Position the synthetic suite as the main evidence package.

## Criticism: crossover near rho_EV ~ 2-4 is empirical and setup-dependent
Response:
- Do not claim a universal crossover number.
- Replace it with theory-based conditioning landmarks (`rho_EV=1` and `rho_EV=4`) and a task-dependent threshold picture governed by `(rho_EV, nu)`.
"""

def build_manifest(repo: Path) -> Dict[str, object]:
    return {
        "repo_root": str(repo),
        "main_script": "scripts/run_paper_suite.py",
        "figures_pdf": sorted([p.name for p in (repo / "figures").glob("*.pdf")]),
        "figures_png": sorted([p.name for p in (repo / "figures").glob("*.png")]),
        "docs_md": sorted([p.name for p in (repo / "docs").glob("*.md")]),
        "docs_pdf": sorted([p.name for p in (repo / "docs").glob("*.pdf")]),
        "results_csv": sorted([p.name for p in (repo / "results").glob("*.csv")]),
    }


# ============================
# Run suites
# ============================

def run_phase_suite(cfg: PaperConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    total = 3 * len(cfg.phase_nshots) * len(cfg.phase_sigmas) * len(cfg.confirmation_seeds)
    count = 0
    for dataset in ["mackey_glass", "lorenz", "narma10"]:
        for n_shots in cfg.phase_nshots:
            for sigma in cfg.phase_sigmas:
                for seed in cfg.confirmation_seeds:
                    row = run_main_protocols(dataset, cfg.budget_total, n_shots, sigma, seed, cfg, traces=False)
                    rows.append(row)
                    count += 1
                    if count % 24 == 0:
                        print(f"phase progress {count}/{total}", flush=True)
    seed_df = flatten_rows(rows)
    summary = summarize(seed_df, group_cols=["dataset", "budget_total", "n_shots", "sigma"], protocols=["ev", "raw", "split"])
    return seed_df, summary

def run_representative_suite(cfg: PaperConfig) -> Tuple[List[Dict[str, object]], pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    total = len(cfg.representative_configs) * len(cfg.confirmation_seeds)
    count = 0
    for dataset, rcfg in cfg.representative_configs.items():
        for seed in cfg.confirmation_seeds:
            row = run_main_protocols(dataset, int(rcfg["budget_total"]), int(rcfg["n_shots"]), float(rcfg["sigma"]), seed, cfg, traces=True)
            rows.append(row)
            count += 1
            if count % 12 == 0:
                print(f"representative progress {count}/{total}", flush=True)
    seed_df = flatten_rows(rows)
    summary = summarize(seed_df, group_cols=["dataset", "budget_total", "n_shots", "sigma"], protocols=["ev", "raw", "split"])
    return rows, seed_df, summary

def run_baseline_suite(cfg: PaperConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    total = len(cfg.representative_configs) * len(cfg.confirmation_seeds)
    count = 0
    for dataset, rcfg in cfg.representative_configs.items():
        for seed in cfg.confirmation_seeds:
            rows.append(run_baseline_protocols(dataset, int(rcfg["budget_total"]), int(rcfg["n_shots"]), float(rcfg["sigma"]), seed, cfg))
            count += 1
            if count % 12 == 0:
                print(f"baseline progress {count}/{total}", flush=True)
    seed_df = flatten_rows(rows)
    summary = summarize(seed_df, group_cols=["dataset", "budget_total", "n_shots", "sigma"], protocols=["ev", "raw", "evdup", "ev_na", "split", "split_na"])
    return seed_df, summary

def run_lambda_suite(cfg: PaperConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    total = len(cfg.representative_configs) * len(cfg.confirmation_seeds)
    count = 0
    for dataset, rcfg in cfg.representative_configs.items():
        for seed in cfg.confirmation_seeds:
            built = build_phi(dataset, int(rcfg["budget_total"]), int(rcfg["n_shots"]), float(rcfg["sigma"]), seed, cfg)
            phi = built["phi"]
            targets = built["targets"]
            for lam in cfg.lambda_grid:
                for protocol in ["ev", "raw", "split"]:
                    result = evaluate_protocol(phi, targets, protocol=protocol, cfg=cfg, ridge_alpha=float(lam))
                    rows.append({
                        "dataset": dataset,
                        "budget_total": int(rcfg["budget_total"]),
                        "n_shots": int(rcfg["n_shots"]),
                        "sigma": float(rcfg["sigma"]),
                        "seed": seed,
                        "lambda": float(lam),
                        "protocol": protocol,
                        "nrmse": result["test_nrmse"],
                        "sr": result["test_sr"],
                    })
            count += 1
            if count % 12 == 0:
                print(f"lambda progress {count}/{total}", flush=True)
    seed_df = pd.DataFrame(rows)
    summary = seed_df.groupby(["dataset", "lambda", "protocol"], as_index=False).agg(
        nrmse_mean=("nrmse", "mean"),
        nrmse_std=("nrmse", lambda x: float(np.std(x, ddof=1))),
        sr_mean=("sr", "mean"),
        sr_std=("sr", lambda x: float(np.std(x, ddof=1))),
    )
    return seed_df, summary

def run_architecture_suite(cfg: PaperConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    total = len(cfg.representative_configs) * len(cfg.architecture_variants) * len(cfg.architecture_seeds)
    count = 0
    for dataset, rcfg in cfg.representative_configs.items():
        for entangler, depth in cfg.architecture_variants:
            for seed in cfg.architecture_seeds:
                rows.append(run_main_protocols(dataset, int(rcfg["budget_total"]), int(rcfg["n_shots"]),
                                               float(rcfg["sigma"]), seed, cfg, depth=depth, entangler=entangler, traces=False))
                count += 1
                if count % 20 == 0:
                    print(f"architecture progress {count}/{total}", flush=True)
    seed_df = flatten_rows(rows)
    summary = summarize(seed_df, group_cols=["dataset", "entangler", "depth", "n_shots", "sigma"], protocols=["ev", "raw", "split"])
    return seed_df, summary

def run_duplication_suite(cfg: PaperConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    total = len(cfg.representative_configs) * len(cfg.phase_nshots) * len(cfg.appendix_seeds)
    count = 0
    for dataset, rcfg in cfg.representative_configs.items():
        sigma = float(rcfg["sigma"])
        for n_shots in cfg.phase_nshots:
            for seed in cfg.appendix_seeds:
                row = run_main_protocols(dataset, cfg.budget_total, n_shots, sigma, seed, cfg, include_evdup=True, traces=False)
                rows.append(row)
                count += 1
                if count % 24 == 0:
                    print(f"duplication progress {count}/{total}", flush=True)
    seed_df = flatten_rows(rows)
    summary = summarize(seed_df, group_cols=["dataset", "budget_total", "n_shots", "sigma"], protocols=["ev", "evdup", "split"])
    return seed_df, summary

def run_size_suite(cfg: PaperConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    total = len(cfg.representative_configs) * len(cfg.reservoir_sizes) * len(cfg.appendix_seeds)
    count = 0
    for dataset, rcfg in cfg.representative_configs.items():
        for q in cfg.reservoir_sizes:
            for seed in cfg.appendix_seeds:
                rows.append(run_main_protocols(dataset, int(rcfg["budget_total"]), int(rcfg["n_shots"]),
                                               float(rcfg["sigma"]), seed, cfg, q=q, depth=2, entangler="ring_cnot", traces=False))
                count += 1
                if count % 12 == 0:
                    print(f"size progress {count}/{total}", flush=True)
    seed_df = flatten_rows(rows)
    summary = summarize(seed_df, group_cols=["dataset", "q", "n_shots", "sigma"], protocols=["ev", "raw", "split"])
    return seed_df, summary


# ============================
# Main entry
# ============================

def convert_markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    cmd = [
        "python",
        "/home/oai/skills/pdfs/scripts/md_to_pdf.py",
        str(md_path),
        "-o",
        str(pdf_path),
    ]
    subprocess.run(cmd, check=True)

def main() -> None:
    print("starting suite", flush=True)
    apply_style()
    repo = Path(__file__).resolve().parents[1]
    results_dir = repo / "results"
    figures_dir = repo / "figures"
    docs_dir = repo / "docs"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)

    cfg = PaperConfig()

    # run suites
    print("running phase suite", flush=True)
    phase_seed, phase_summary = run_phase_suite(cfg)
    print("phase suite done", flush=True)
    print("running representative suite", flush=True)
    rep_rows, rep_seed, rep_summary = run_representative_suite(cfg)
    print("representative suite done", flush=True)
    print("running baseline suite", flush=True)
    baseline_seed, baseline_summary = run_baseline_suite(cfg)
    print("baseline suite done", flush=True)
    print("running lambda suite", flush=True)
    lambda_seed, lambda_summary = run_lambda_suite(cfg)
    print("lambda suite done", flush=True)
    print("running architecture suite", flush=True)
    arch_seed, arch_summary = run_architecture_suite(cfg)
    print("architecture suite done", flush=True)
    print("running duplication suite", flush=True)
    dup_seed, dup_summary = run_duplication_suite(cfg)
    print("duplication suite done", flush=True)
    print("running size suite", flush=True)
    size_seed, size_summary = run_size_suite(cfg)
    print("size suite done", flush=True)

    theory_df = summarize_theory_empirics(phase_summary)

    # save results
    phase_seed.to_csv(results_dir / "phase_seed_level.csv", index=False)
    phase_summary.to_csv(results_dir / "phase_summary.csv", index=False)

    rep_seed.to_csv(results_dir / "representative_seed_level.csv", index=False)
    rep_summary.to_csv(results_dir / "representative_summary.csv", index=False)
    np.savez_compressed(
        results_dir / "representative_traces.npz",
        **{
            f"{dataset}_y_test": np.stack([r["trace_y_test"] for r in rep_rows if r["dataset"] == dataset], axis=0)
            for dataset in cfg.representative_configs.keys()
        },
        **{
            f"{dataset}_{protocol}": np.stack([r[f"trace_{protocol}"] for r in rep_rows if r["dataset"] == dataset], axis=0)
            for dataset in cfg.representative_configs.keys() for protocol in ["ev", "raw", "split"]
        },
    )

    baseline_seed.to_csv(results_dir / "baseline_controls_seed_level.csv", index=False)
    baseline_summary.to_csv(results_dir / "baseline_controls_summary.csv", index=False)

    lambda_seed.to_csv(results_dir / "lambda_seed_level.csv", index=False)
    lambda_summary.to_csv(results_dir / "lambda_summary.csv", index=False)

    arch_seed.to_csv(results_dir / "architecture_seed_level.csv", index=False)
    arch_summary.to_csv(results_dir / "architecture_summary.csv", index=False)

    dup_seed.to_csv(results_dir / "duplication_control_seed_level.csv", index=False)
    dup_summary.to_csv(results_dir / "duplication_control_summary.csv", index=False)

    size_seed.to_csv(results_dir / "reservoir_size_seed_level.csv", index=False)
    size_summary.to_csv(results_dir / "reservoir_size_summary.csv", index=False)

    theory_df.to_csv(results_dir / "theory_empirical_summary.csv", index=False)

    print("saving results", flush=True)
    # figures
    make_figure_1(rep_rows, figures_dir / "experiment_1")
    make_figure_2_phase(phase_summary, figures_dir / "figure_2_phase_maps")
    make_figure_3_theory(theory_df, figures_dir / "figure_3_theory_regime_test")
    make_figure_4_baselines(baseline_summary, figures_dir / "experiment_4")
    make_figure_5_lambda(lambda_summary, figures_dir / "figure_5_lambda_sweeps")
    make_figure_6_architecture(arch_summary, figures_dir / "figure_6_architecture_ablation")
    make_appendix_duplication(dup_summary, figures_dir / "experiment_5")
    make_appendix_size(size_summary, figures_dir / "experiment_3")

    print("making figures", flush=True)
    # docs
    (repo / "README.md").write_text(build_readme(), encoding="utf-8")
    (docs_dir / "technical_report.md").write_text(
        build_technical_report(cfg, rep_summary, phase_summary, theory_df, baseline_summary, lambda_summary, arch_summary, size_summary),
        encoding="utf-8",
    )
    (docs_dir / "theory_note.md").write_text(build_theory_note(), encoding="utf-8")
    (docs_dir / "figure_captions.md").write_text(build_captions(), encoding="utf-8")
    (docs_dir / "review_response_notes.md").write_text(build_review_response_notes(), encoding="utf-8")

    print("building docs", flush=True)
    convert_markdown_to_pdf(docs_dir / "technical_report.md", docs_dir / "technical_report.pdf")
    convert_markdown_to_pdf(docs_dir / "theory_note.md", docs_dir / "theory_note.pdf")
    convert_markdown_to_pdf(docs_dir / "review_response_notes.md", docs_dir / "review_response_notes.pdf")

    (repo / "requirements.txt").write_text(
        "numpy\npandas\nmatplotlib\nscikit-learn\n",
        encoding="utf-8",
    )
    (repo / "manifest.json").write_text(json.dumps(build_manifest(repo), indent=2), encoding="utf-8")

if __name__ == "__main__":
    import traceback
    try:
        main()
        print("suite complete", flush=True)
    except Exception as e:
        traceback.print_exc()
        raise
