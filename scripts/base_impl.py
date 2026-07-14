
#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from protocol_config import RESERVOIR_ALIASES, ProtocolConfig, load_protocol_config
except ModuleNotFoundError:
    import importlib.util as _importlib_util

    _CONFIG_SPEC = _importlib_util.spec_from_file_location(
        "protocol_config",
        Path(__file__).resolve().parent / "protocol_config.py",
    )
    _CONFIG_MODULE = _importlib_util.module_from_spec(_CONFIG_SPEC)
    assert _CONFIG_SPEC.loader is not None
    sys.modules["protocol_config"] = _CONFIG_MODULE
    _CONFIG_SPEC.loader.exec_module(_CONFIG_MODULE)
    ProtocolConfig = _CONFIG_MODULE.ProtocolConfig
    load_protocol_config = _CONFIG_MODULE.load_protocol_config
    RESERVOIR_ALIASES = _CONFIG_MODULE.RESERVOIR_ALIASES


# ============================
# Config
# ============================

PaperConfig = ProtocolConfig


def load_config(path: str | Path | None = None) -> ProtocolConfig:
    return load_protocol_config(path)


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

def rossler_x(
    n_steps: int,
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
    dt: float = 0.02,
    subsample: int = 5,
    warmup: int = 2500,
) -> np.ndarray:
    total = (n_steps + warmup) * subsample
    xs = np.zeros(n_steps + warmup, dtype=float)
    x, y, z = 1.0, 1.0, 1.0
    for t in range(total):
        if t % subsample == 0:
            xs[t // subsample] = x
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dt * dx
        y += dt * dy
        z += dt * dz
    return xs[warmup:]

def henon_x(n_steps: int, a: float = 1.4, b: float = 0.3, warmup: int = 200) -> np.ndarray:
    total = n_steps + warmup
    xs = np.zeros(total, dtype=float)
    x, y = 0.1, 0.1
    for t in range(total):
        xs[t] = x
        x, y = 1.0 - a * x * x + y, b * x
    return xs[warmup:]

def sunspots_monthly(n_steps: int) -> np.ndarray:
    """Real-world series: SIDC monthly mean total sunspot number (V2.0).

    Cached at data/sunspots_monthly_sidc.csv (1749-01 onward). Returns the
    most recent n_steps values so longer records extend the same window
    backward in time.
    """
    path = Path(__file__).resolve().parent.parent / "data" / "sunspots_monthly_sidc.csv"
    values = np.loadtxt(path, delimiter=";", usecols=3, dtype=float)
    values = values[values >= 0.0]
    if n_steps > len(values):
        raise ValueError(f"sunspots series has only {len(values)} months, requested {n_steps}")
    return values[-n_steps:].copy()

def make_series(dataset: str, total_steps: int) -> np.ndarray:
    if dataset == "mackey_glass":
        return mackey_glass(total_steps + 1, seed=1234)
    if dataset == "lorenz":
        return lorenz_x(total_steps + 1)
    if dataset == "narma10":
        return narma10_y(total_steps + 1, seed=1234)
    if dataset == "henon":
        return henon_x(total_steps + 1)
    if dataset == "rossler":
        return rossler_x(total_steps + 1)
    if dataset == "sunspots":
        return sunspots_monthly(total_steps + 1)
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

def input_amplitude_angle(u: float) -> float:
    clipped = min(max(float(u), 0.0), 1.0)
    return float(2.0 * np.arcsin(np.sqrt(clipped)))

def entangler_edges(entangler: str, q: int) -> List[Tuple[int, int]]:
    if entangler in {"ring_cnot", "ring_rzz"}:
        return [(i, (i + 1) % q) for i in range(q)]
    if entangler in {"line_cnot", "line_rzz"}:
        return [(i, i + 1) for i in range(q - 1)]
    if entangler in {"all_to_all_cz", "all_to_all_rzz"}:
        return [(i, j) for i in range(q) for j in range(i + 1, q)]
    raise ValueError(entangler)

def rzz_phase_vector(theta: float, qubit_a: int, qubit_b: int, n_qubits: int) -> np.ndarray:
    idx = np.arange(2 ** n_qubits)
    shift_a = n_qubits - 1 - qubit_a
    shift_b = n_qubits - 1 - qubit_b
    za = 1.0 - 2.0 * ((idx >> shift_a) & 1).astype(float)
    zb = 1.0 - 2.0 * ((idx >> shift_b) & 1).astype(float)
    return np.exp(-0.5j * float(theta) * za * zb)

class ExactQRCReservoir:
    """Canonical IBM-compatible gate-model QRC reservoir.

    The circuit is a fixed Trotterized transverse-field-Ising block: one
    amplitude-encoded input qubit, initialized memory qubits, RZZ couplings,
    transverse RX rotations, and local-Z bitstring readout.
    """

    def __init__(self, q: int, depth: int, entangler: str, eta: float, window: int, seed: int) -> None:
        self.q = q
        self.depth = depth
        self.entangler = entangler
        self.eta = eta
        self.window = window
        self.dim = 2 ** q
        self.m = q
        self.seed = int(seed)
        self.input_qubit = 0
        self.field_angle = np.pi / 4.0
        self.coupling_angle = np.pi / 2.0
        self.edges = entangler_edges(entangler, q)
        self.rzz_phases = [rzz_phase_vector(self.coupling_angle, a, b, q) for a, b in self.edges]

    def _initial_state(self) -> np.ndarray:
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0
        return state

    def prepare_state(self, u: float) -> np.ndarray:
        state = self._initial_state()
        state = apply_single_qubit_gate(state, ry(input_amplitude_angle(u)), self.input_qubit, self.q)
        for qubit in range(1, self.q):
            state = apply_single_qubit_gate(state, H, qubit, self.q)
        for _ in range(self.depth):
            for phase in self.rzz_phases:
                state = state * phase
            for qubit in range(self.q):
                state = apply_single_qubit_gate(state, rx(self.field_angle), qubit, self.q)
        return state

    def _sample_basis(self, state: np.ndarray, n_shots: int, rng: np.random.Generator) -> np.ndarray:
        probs = np.abs(state) ** 2
        probs = np.clip(probs.real, 0.0, None)
        probs /= probs.sum()
        idx = rng.choice(self.dim, size=n_shots, p=probs)
        shifts = np.arange(self.q - 1, -1, -1)
        bits = ((idx[:, None] >> shifts[None, :]) & 1).astype(np.int8)
        return bits

    def sample_measurements(self, state: np.ndarray, n_shots: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
        z_bits = self._sample_basis(state.copy(), n_shots, rng=rng)
        measurements = 1.0 - 2.0 * z_bits.astype(float)
        if sigma > 0.0:
            flip_prob = min(float(sigma), 0.5)
            flip_mask = rng.random(measurements.shape) < flip_prob
            measurements = np.where(flip_mask, -measurements, measurements)
            measurements = measurements + float(sigma) * rng.normal(size=measurements.shape)
        return measurements

class LegacyRandomCircuitReservoir:
    """Fixed random-circuit baseline for architecture comparisons."""

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
        if self.entangler in {"ring_cnot", "ring_rzz"}:
            edges = [(i, (i + 1) % self.q) for i in range(self.q)]
            for control, target in edges:
                state = apply_cnot(state, control, target, self.q)
        elif self.entangler in {"line_cnot", "line_rzz"}:
            edges = [(i, i + 1) for i in range(self.q - 1)]
            for control, target in edges:
                state = apply_cnot(state, control, target, self.q)
        elif self.entangler in {"all_to_all_cz", "all_to_all_rzz"}:
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

class CanonicalSpinQRCReservoir:
    """Stateful critical TFIM ring reservoir with sampled Z/X observables."""

    def __init__(self, q: int, eta: float, window: int, seed: int) -> None:
        self.q = q
        self.depth = 1
        self.entangler = "canonical_spin"
        self.eta = eta
        self.window = window
        self.dim = 2 ** q
        self.m = 2 * q
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.h_fields = rng.uniform(0.75, 1.25, size=q)
        self.j_couplings = rng.uniform(0.75, 1.25, size=q)
        self.base_hamiltonian = self._build_base_hamiltonian()
        norm_bound = float(np.sum(np.abs(self.h_fields)) + np.sum(np.abs(self.j_couplings)))
        self.tau = np.pi / (2.0 * norm_bound) if norm_bound > 1e-12 else 1.0
        self._basis_z_signs = self._basis_signs()
        self._basis_x_signs = self._basis_signs()
        self.state = self._initial_state()

    def _basis_signs(self) -> np.ndarray:
        idx = np.arange(self.dim)
        shifts = np.arange(self.q - 1, -1, -1)
        bits = ((idx[:, None] >> shifts[None, :]) & 1).astype(np.int8)
        return 1.0 - 2.0 * bits.astype(float)

    def _build_base_hamiltonian(self):
        rows: List[int] = []
        cols: List[int] = []
        vals: List[complex] = []
        for basis in range(self.dim):
            z_values = np.array(
                [1.0 if ((basis >> (self.q - 1 - qubit)) & 1) == 0 else -1.0 for qubit in range(self.q)],
                dtype=float,
            )
            diag = 0.0
            for qubit in range(self.q):
                nxt = (qubit + 1) % self.q
                diag += self.j_couplings[qubit] * z_values[qubit] * z_values[nxt]
                flipped = basis ^ (1 << (self.q - 1 - qubit))
                rows.append(flipped)
                cols.append(basis)
                vals.append(complex(self.h_fields[qubit]))
            rows.append(basis)
            cols.append(basis)
            vals.append(complex(diag))
        return coo_matrix((vals, (rows, cols)), shape=(self.dim, self.dim), dtype=complex).tocsr()

    def _initial_state(self) -> np.ndarray:
        state = np.ones(self.dim, dtype=complex)
        state /= np.sqrt(self.dim)
        return state

    def prepare_state(self, u: float) -> np.ndarray:
        centered = 2.0 * float(u) - 1.0
        for qubit in range(self.q):
            scale = (qubit + 1) / self.q
            self.state = apply_single_qubit_gate(self.state, ry(np.pi * scale * centered), qubit, self.q)
        self.state = np.asarray(expm_multiply((-1j * self.tau) * self.base_hamiltonian, self.state), dtype=complex)
        norm = float(np.linalg.norm(self.state))
        if norm > 1e-12:
            self.state = self.state / norm
        return self.state

    def _sample_probs(self, probs: np.ndarray, n_shots: int, rng: np.random.Generator) -> np.ndarray:
        probs = np.clip(probs.real, 0.0, None)
        probs /= probs.sum()
        return rng.choice(self.dim, size=n_shots, p=probs)

    def sample_measurements(self, state: np.ndarray, n_shots: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
        probs_z = np.abs(state) ** 2
        idx_z = self._sample_probs(probs_z, n_shots=n_shots, rng=rng)
        state_x = state.copy()
        for qubit in range(self.q):
            state_x = apply_single_qubit_gate(state_x, H, qubit, self.q)
        probs_x = np.abs(state_x) ** 2
        idx_x = self._sample_probs(probs_x, n_shots=n_shots, rng=rng)
        measurements = np.concatenate([self._basis_z_signs[idx_z], self._basis_x_signs[idx_x]], axis=1)
        if sigma > 0.0:
            flip_prob = min(float(sigma), 0.5)
            flip_mask = rng.random(measurements.shape) < flip_prob
            measurements = np.where(flip_mask, -measurements, measurements)
            measurements = measurements + float(sigma) * rng.normal(size=measurements.shape)
        return measurements

def run_reservoir(
    inputs: np.ndarray,
    reservoir: ExactQRCReservoir | LegacyRandomCircuitReservoir | CanonicalSpinQRCReservoir,
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

def fit_validate_grouped(
    grouped: np.ndarray,
    targets: np.ndarray,
    cfg: PaperConfig,
    ridge_alpha: float,
) -> Dict[str, object]:
    """Fit on the fit block and score only the validation block.

    Keeping this operation separate from :func:`fit_test_grouped` makes it
    impossible for a hyperparameter search to inspect the held-out test block.
    """
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

    return {
        "val_nrmse": nrmse(targets[fit_end:train_end], pred_val),
        "y_val": targets[fit_end:train_end],
        "y_pred_val": pred_val,
    }


def fit_test_grouped(
    grouped: np.ndarray,
    targets: np.ndarray,
    cfg: PaperConfig,
    ridge_alpha: float,
) -> Dict[str, object]:
    """Refit on fit+validation and evaluate the test block exactly once."""
    _, _, train_end = split_points(grouped.shape[1], cfg)
    t0 = cfg.t0

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
        "test_nrmse": nrmse(y_test, pred_test),
        "test_sr": sr(y_test, pred_test),
        "y_test": y_test,
        "y_pred": pred_test,
    }


def fit_predict_grouped(grouped: np.ndarray, targets: np.ndarray, cfg: PaperConfig, ridge_alpha: float) -> Dict[str, object]:
    """Compatibility wrapper for a preselected grouped model.

    New hyperparameter searches call ``fit_validate_grouped`` for every
    candidate and ``fit_test_grouped`` only for the validation winner.
    """
    validation = fit_validate_grouped(grouped, targets, cfg, ridge_alpha)
    test = fit_test_grouped(grouped, targets, cfg, ridge_alpha)
    return {**validation, **test}


def _fit_full_covariance_stage(
    grouped: np.ndarray,
    targets: np.ndarray,
    cfg: PaperConfig,
    ridge_alpha: float,
    *,
    final: bool,
) -> Dict[str, object]:
    """Exact full-average plus empirical full-covariance ridge fit.

    In the standardized grouped coordinates, the grouped squared loss is
    identically the loss on the group mean plus the within-group covariance
    quadratic.  This routine solves that decomposed normal equation directly;
    it is an implementation check, not a distinct learning algorithm.
    """
    _, fit_end, train_end = split_points(grouped.shape[1], cfg)
    fit_stop = train_end if final else fit_end
    eval_start = train_end if final else fit_end
    eval_stop = grouped.shape[1] if final else train_end
    t0 = cfg.t0
    groups = grouped.shape[0]
    d = grouped.shape[2]

    X_rows = grouped[:, t0:fit_stop, :].reshape(-1, d)
    scaler = StandardScaler().fit(X_rows)
    Xs = scaler.transform(X_rows).reshape(groups, fit_stop - t0, d)
    xbar = Xs.mean(axis=0)
    xbar_mean = xbar.mean(axis=0)
    xbar_centered = xbar - xbar_mean
    deviations = Xs - xbar[None, :, :]

    y = np.asarray(targets[t0:fit_stop], dtype=float)
    y_mean = float(y.mean())
    yc = y - y_mean
    covariance_penalty = np.einsum("gtd,gte->de", deviations, deviations, optimize=True)
    A = groups * (xbar_centered.T @ xbar_centered) + covariance_penalty
    A = A + float(ridge_alpha) * np.eye(d)
    rhs = groups * (xbar_centered.T @ yc)
    coef = np.linalg.solve(A, rhs)
    intercept = y_mean - float(xbar_mean @ coef)

    X_eval = grouped[:, eval_start:eval_stop, :].reshape(-1, d)
    X_eval_s = scaler.transform(X_eval).reshape(groups, eval_stop - eval_start, d)
    pred = X_eval_s.mean(axis=0) @ coef + intercept
    y_eval = np.asarray(targets[eval_start:eval_stop], dtype=float)
    out: Dict[str, object] = {
        "coef": coef,
        "intercept": intercept,
        "covariance_penalty": covariance_penalty,
    }
    if final:
        out.update(
            {
                "test_nrmse": nrmse(y_eval, pred),
                "test_sr": sr(y_eval, pred),
                "y_test": y_eval,
                "y_pred": pred,
            }
        )
    else:
        out.update(
            {
                "val_nrmse": nrmse(y_eval, pred),
                "y_val": y_eval,
                "y_pred_val": pred,
            }
        )
    return out


def fit_validate_full_covariance_equivalent(
    grouped: np.ndarray,
    targets: np.ndarray,
    cfg: PaperConfig,
    ridge_alpha: float,
) -> Dict[str, object]:
    return _fit_full_covariance_stage(grouped, targets, cfg, ridge_alpha, final=False)


def fit_test_full_covariance_equivalent(
    grouped: np.ndarray,
    targets: np.ndarray,
    cfg: PaperConfig,
    ridge_alpha: float,
) -> Dict[str, object]:
    return _fit_full_covariance_stage(grouped, targets, cfg, ridge_alpha, final=True)


def fit_predict_full_covariance_equivalent(
    grouped: np.ndarray,
    targets: np.ndarray,
    cfg: PaperConfig,
    ridge_alpha: float,
) -> Dict[str, object]:
    validation = fit_validate_full_covariance_equivalent(grouped, targets, cfg, ridge_alpha)
    test = fit_test_full_covariance_equivalent(grouped, targets, cfg, ridge_alpha)
    return {**validation, **test}

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
        result = fit_validate_grouped(grouped, targets, cfg, ridge_alpha=ridge_alpha)
        current = {"k": k, "group_count": grouped.shape[0], **result}
        if best is None or current["val_nrmse"] < best["val_nrmse"] - 1e-12 or (
            abs(current["val_nrmse"] - best["val_nrmse"]) <= 1e-12 and current["k"] < best["k"]
        ):
            best = current
    selected_k = int(best["k"])
    if protocol == "ev":
        grouped = grouped_features(phi, n_shots)
    elif protocol == "raw":
        grouped = grouped_features(phi, 1)
    elif protocol == "split":
        grouped = grouped_features(phi, selected_k)
    else:
        ev = grouped_features(phi, n_shots)
        grouped = np.repeat(ev, n_shots // selected_k, axis=0)
    test = fit_test_grouped(grouped, targets, cfg, ridge_alpha=ridge_alpha)
    return {**best, **test, "protocol": protocol, "ridge_alpha": ridge_alpha}

def evaluate_protocol_lambda_grid(
    phi: np.ndarray,
    targets: np.ndarray,
    protocol: str,
    cfg: PaperConfig,
    ridge_grid: Sequence[float],
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
    for ridge_alpha in ridge_grid:
        for k in candidates:
            if protocol == "ev":
                grouped = grouped_features(phi, n_shots)
            elif protocol == "raw":
                grouped = grouped_features(phi, 1)
            elif protocol == "split":
                grouped = grouped_features(phi, k)
            else:
                ev = grouped_features(phi, n_shots)
                grouped = np.repeat(ev, n_shots // k, axis=0)
            validation = fit_validate_grouped(grouped, targets, cfg, ridge_alpha=float(ridge_alpha))
            current = {
                "k": int(k),
                "group_count": int(grouped.shape[0]),
                "ridge_alpha": float(ridge_alpha),
                **validation,
            }
            if best is None or current["val_nrmse"] < best["val_nrmse"] - 1e-12 or (
                abs(current["val_nrmse"] - best["val_nrmse"]) <= 1e-12
                and (current["ridge_alpha"], current["k"]) < (best["ridge_alpha"], best["k"])
            ):
                best = current

    selected_k = int(best["k"])
    if protocol == "ev":
        grouped = grouped_features(phi, n_shots)
    elif protocol == "raw":
        grouped = grouped_features(phi, 1)
    elif protocol == "split":
        grouped = grouped_features(phi, selected_k)
    else:
        ev = grouped_features(phi, n_shots)
        grouped = np.repeat(ev, n_shots // selected_k, axis=0)
    test = fit_test_grouped(grouped, targets, cfg, ridge_alpha=float(best["ridge_alpha"]))
    return {**best, **test, "protocol": protocol}

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
    reservoir_family: str = "canonical_gate_qrc",
) -> Dict[str, object]:
    qq = cfg.q_default if q is None else q
    dd = cfg.depth_default if depth is None else depth
    ee = cfg.entangler_default if entangler is None else entangler
    family = RESERVOIR_ALIASES.get(reservoir_family, reservoir_family)
    budget_factor = cfg.budget_factor(family) if hasattr(cfg, "budget_factor") else (2 if family in {"legacy_random_circuit", "canonical_spin"} else 1)
    t_steps = budget_total // (budget_factor * n_shots)
    raw = make_series(dataset, total_steps=t_steps)
    inputs, targets = normalize_and_shift(raw, t_steps, cfg)
    if family == "canonical_gate_qrc":
        reservoir = ExactQRCReservoir(q=qq, depth=dd, entangler=ee, eta=cfg.eta, window=cfg.window, seed=reservoir_seed)
        reservoir_family = "canonical_gate_qrc"
        reservoir_variant = f"tfim_gate_{ee}_d{dd}_local_z"
    elif family == "legacy_random_circuit":
        reservoir = LegacyRandomCircuitReservoir(q=qq, depth=dd, entangler=ee, eta=cfg.eta, window=cfg.window, seed=reservoir_seed)
        reservoir_family = "legacy_random_circuit"
        reservoir_variant = f"legacy_random_{ee}_d{dd}_zx"
    elif family == "canonical_spin":
        reservoir = CanonicalSpinQRCReservoir(q=qq, eta=cfg.eta, window=cfg.window, seed=reservoir_seed)
        reservoir_family = "canonical_spin"
        reservoir_variant = "tfim_ring_stateful"
        ee = "canonical_spin"
        dd = 1
    else:
        raise ValueError(f"Unsupported reservoir_family={reservoir_family!r}")
    phi = run_reservoir(inputs=inputs, reservoir=reservoir, n_shots=n_shots, sigma=sigma, shot_seed=reservoir_seed + 100000)
    nu_hat = feature_noise_ratio(phi, cfg)
    feature_dim = int(phi.shape[2])
    rho_ev = cfg.train_frac * max(int(t_steps) - cfg.t0, 0) / feature_dim if feature_dim > 0 else np.nan
    # Conditioning diagnostics are not needed for the figure suite itself.
    ev_smin, ev_cond = np.nan, np.nan
    return {
        "phi": phi,
        "targets": targets,
        "t_steps": t_steps,
        "rho_ev": rho_ev,
        "nu_hat": nu_hat,
        "ev_smin_scaled": ev_smin,
        "ev_cond": ev_cond,
        "q": qq,
        "depth": dd,
        "entangler": ee,
        "reservoir_family": reservoir_family,
        "reservoir_variant": reservoir_variant,
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
    reservoir_family: str = "canonical_gate_qrc",
    include_evdup: bool = False,
    traces: bool = False,
    tune_lambda: bool = False,
) -> Dict[str, object]:
    built = build_phi(
        dataset,
        budget_total,
        n_shots,
        sigma,
        reservoir_seed,
        cfg,
        q=q,
        depth=depth,
        entangler=entangler,
        reservoir_family=reservoir_family,
    )
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
        "reservoir_family": built["reservoir_family"],
        "reservoir_variant": built["reservoir_variant"],
        "t_steps": built["t_steps"],
    }

    protocols = ["ev", "raw", "split"] + (["evdup"] if include_evdup else [])
    for protocol in protocols:
        if tune_lambda:
            result = evaluate_protocol_lambda_grid(phi, targets, protocol=protocol, cfg=cfg, ridge_grid=cfg.lambda_grid)
        else:
            result = evaluate_protocol(phi, targets, protocol=protocol, cfg=cfg, ridge_alpha=cfg.lambda_default)
        out[f"{protocol}_nrmse"] = result["test_nrmse"]
        out[f"{protocol}_sr"] = result["test_sr"]
        out[f"{protocol}_k"] = result["k"]
        out[f"{protocol}_val_nrmse"] = result["val_nrmse"]
        out[f"{protocol}_ridge_alpha"] = result["ridge_alpha"]
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
    reservoir_family: str = "canonical_gate_qrc",
) -> Dict[str, object]:
    built = build_phi(dataset, budget_total, n_shots, sigma, reservoir_seed, cfg, reservoir_family=reservoir_family)
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
        "reservoir_family": built["reservoir_family"],
        "reservoir_variant": built["reservoir_variant"],
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

# This module intentionally exposes reusable simulation and fitting primitives
# only. Publication workflows are implemented by the dedicated scripts in this
# directory and by the root Makefile.
