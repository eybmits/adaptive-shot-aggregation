"""
Budget-aware Quantum Reservoir Computing
=========================================
Full reproduction of the paper:
  "Budget-aware quantum reservoir learning from single-shot measurement trajectories"

Implements:
  - 6-qubit random-circuit reservoir (numpy statevector simulator)
  - Mackey-Glass and NARMA-10 benchmarks
  - EV training (Algorithm 1) and trajectory-level training (Algorithm 2)
  - Full (Nshots, sigma) phase-diagram sweep at S=10000
  - rho_EV collapse analysis
  - Grouped-measurement diagnostics
  - All figures from the paper
"""

import numpy as np
from scipy.linalg import expm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os, sys, json, warnings
warnings.filterwarnings('ignore')

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# 1. Quantum gates and statevector simulator
# =============================================================================

def rx(theta):
    """Single-qubit X rotation."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

def ry(theta):
    """Single-qubit Y rotation."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def rz(theta):
    """Single-qubit Z rotation."""
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)

def cnot():
    """Two-qubit CNOT gate."""
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,0,1],
                     [0,0,1,0]], dtype=complex)

def cz():
    """Two-qubit CZ gate."""
    return np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0],
                     [0,0,0,-1]], dtype=complex)

def apply_single_qubit_gate(state, gate, qubit, n_qubits):
    """Apply a single-qubit gate to a statevector."""
    N = 2 ** n_qubits
    state = state.reshape([2] * n_qubits)
    state = np.moveaxis(state, qubit, 0)
    shape = state.shape
    state = state.reshape(2, -1)
    state = gate @ state
    state = state.reshape(shape)
    state = np.moveaxis(state, 0, qubit)
    return state.reshape(N)

def apply_cnot(state, control, target, n_qubits):
    """Apply CNOT between control and target qubits."""
    N = 2 ** n_qubits
    state = state.reshape([2] * n_qubits)
    # CNOT: if control=1, flip target
    # Extract slices
    idx_c1 = [slice(None)] * n_qubits
    idx_c1[control] = 1
    sub = state[tuple(idx_c1)].copy()
    # Flip target in the control=1 subspace
    idx_t0 = [slice(None)] * (n_qubits - 1)  # after removing control dim
    # We need to swap the target qubit in `sub`
    target_adj = target if target < control else target - 1
    sub = np.moveaxis(sub, target_adj, 0)
    sub_shape = sub.shape
    sub = sub.reshape(2, -1)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    sub = X @ sub
    sub = sub.reshape(sub_shape)
    sub = np.moveaxis(sub, 0, target_adj)
    state[tuple(idx_c1)] = sub
    return state.reshape(N)


def apply_cz(state, control, target, n_qubits):
    """Apply CZ between control and target qubits."""
    if control == target:
        raise ValueError("control and target must be different for CZ.")
    N = 2 ** n_qubits
    state = state.reshape([2] * n_qubits)
    idx = [slice(None)] * n_qubits
    idx[control] = 1
    idx[target] = 1
    state[tuple(idx)] *= -1.0
    return state.reshape(N)


class QuantumReservoir:
    """
    6-qubit random-circuit quantum reservoir.

    Architecture:
      - Input encoding: single-qubit Ry rotations scaled by random input weights W
      - Fixed reservoir unitary: configurable entangler family + random single-qubit rotations
      - Measurement: Pauli Z and X on all qubits (M = 2Q = 12 observables)
    """

    ALLOWED_CIRCUIT_FAMILIES = ("ring_cnot", "line_cnot", "star_cnot", "all_to_all_cz")

    def __init__(self, n_qubits=6, depth=1, rng=None, circuit_family="ring_cnot"):
        self.Q = n_qubits
        self.N = 2 ** n_qubits
        self.M = 2 * n_qubits  # Z and X measurements
        self.depth = depth
        self.circuit_family = circuit_family

        if self.circuit_family not in self.ALLOWED_CIRCUIT_FAMILIES:
            raise ValueError(
                f"Unknown circuit_family='{self.circuit_family}'. "
                f"Expected one of {self.ALLOWED_CIRCUIT_FAMILIES}."
            )

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        # Random input weights W (one per qubit)
        self.W = rng.uniform(-np.pi, np.pi, size=n_qubits)

        # Precompute the fixed reservoir unitary parameters
        self.res_params = []
        for _ in range(depth):
            layer_params = rng.uniform(0, 2 * np.pi, size=(n_qubits, 3))
            self.res_params.append(layer_params)
        self.entangler_ops = self._build_entangler_ops()

    def _build_entangler_ops(self):
        """Return a fixed list of (op_name, control, target) entanglers for one layer."""
        if self.circuit_family == "ring_cnot":
            return [("cnot", i, (i + 1) % self.Q) for i in range(self.Q)]
        if self.circuit_family == "line_cnot":
            return [("cnot", i, i + 1) for i in range(self.Q - 1)]
        if self.circuit_family == "star_cnot":
            return [("cnot", 0, j) for j in range(1, self.Q)]
        if self.circuit_family == "all_to_all_cz":
            ops = []
            for i in range(self.Q):
                for j in range(i + 1, self.Q):
                    ops.append(("cz", i, j))
            return ops
        raise ValueError(f"Unsupported circuit_family: {self.circuit_family}")

    def _initial_state(self):
        """Return |0...0> state."""
        state = np.zeros(self.N, dtype=complex)
        state[0] = 1.0
        return state

    def _apply_input_encoding(self, state, u):
        """Apply Ry(W_i * u) to each qubit."""
        for i in range(self.Q):
            angle = self.W[i] * u
            state = apply_single_qubit_gate(state, ry(angle), i, self.Q)
        return state

    def _apply_reservoir_unitary(self, state):
        """Apply fixed random circuit family + random single-qubit rotations."""
        for layer_params in self.res_params:
            for op_name, control, target in self.entangler_ops:
                if op_name == "cnot":
                    state = apply_cnot(state, control, target, self.Q)
                elif op_name == "cz":
                    state = apply_cz(state, control, target, self.Q)
                else:
                    raise ValueError(f"Unsupported entangler op: {op_name}")
            # Random single-qubit rotations
            for i in range(self.Q):
                state = apply_single_qubit_gate(state, rx(layer_params[i, 0]), i, self.Q)
                state = apply_single_qubit_gate(state, ry(layer_params[i, 1]), i, self.Q)
                state = apply_single_qubit_gate(state, rz(layer_params[i, 2]), i, self.Q)
        return state

    def prepare_state(self, u):
        """Prepare the quantum state for input u: |psi> = U_res U_in |0>."""
        state = self._initial_state()
        state = self._apply_input_encoding(state, u)
        state = self._apply_reservoir_unitary(state)
        return state

    def get_expectation_values(self, state):
        """Compute exact expectation values of Z_i and X_i."""
        evs = np.zeros(self.M)
        for i in range(self.Q):
            # <Z_i>
            probs = self._get_qubit_probs(state, i)
            evs[i] = probs[0] - probs[1]  # P(0) - P(1)
            # <X_i>
            # Apply Hadamard, then measure Z
            H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
            state_h = apply_single_qubit_gate(state.copy(), H, i, self.Q)
            probs_x = self._get_qubit_probs(state_h, i)
            evs[self.Q + i] = probs_x[0] - probs_x[1]
        return evs

    def get_measurement_probs(self, state):
        """
        Precompute all measurement probabilities for Z and X bases.
        Returns array of shape (M, 2) with P(0) and P(1) for each observable.
        """
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        probs = np.zeros((self.M, 2))
        for i in range(self.Q):
            probs[i] = self._get_qubit_probs(state, i)
            state_x = apply_single_qubit_gate(state.copy(), H, i, self.Q)
            probs[self.Q + i] = self._get_qubit_probs(state_x, i)
        return probs

    def sample_multiple_shots(self, state, n_shots, rng=None):
        """
        Sample n_shots measurement outcomes for all M observables at once.
        Returns array of shape (n_shots, M) with values in {-1, +1}.
        """
        if rng is None:
            rng = self.rng

        probs = self.get_measurement_probs(state)  # (M, 2)
        m = np.zeros((n_shots, self.M))

        for i in range(self.M):
            # Sample n_shots outcomes for observable i
            outcomes = (rng.random(n_shots) > probs[i, 0]).astype(float)
            m[:, i] = 1.0 - 2.0 * outcomes  # 0 -> +1, 1 -> -1

        return m

    def sample_single_shot(self, state, rng=None):
        """Sample a single-shot measurement outcome for all M observables."""
        return self.sample_multiple_shots(state, 1, rng)[0]

    def _get_qubit_probs(self, state, qubit):
        """Get probability of measuring |0> and |1> on a given qubit."""
        state_r = state.reshape([2] * self.Q)
        # Sum over all other qubits
        axes_to_sum = list(range(self.Q))
        axes_to_sum.remove(qubit)
        # Move qubit axis to front
        state_r = np.moveaxis(state_r, qubit, 0)
        # Compute probabilities
        probs = np.array([
            np.sum(np.abs(state_r[0])**2),
            np.sum(np.abs(state_r[1])**2)
        ])
        probs = np.clip(probs, 0, None)
        probs /= probs.sum()
        return probs


# =============================================================================
# 2. Benchmark time-series generators
# =============================================================================

def mackey_glass(n_steps, tau=17, beta=0.2, gamma=0.1, n_power=10, dt=1.0,
                 warmup=500, rng=None):
    """
    Generate Mackey-Glass time series.

    dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
    """
    total = n_steps + warmup + tau
    x = np.zeros(total)
    if rng is not None:
        x[:tau] = 0.9 + 0.2 * rng.random(tau)
    else:
        x[:tau] = 0.9 + 0.2 * np.random.rand(tau)

    for t in range(tau, total - 1):
        x_tau = x[t - tau]
        dx = beta * x_tau / (1 + x_tau ** n_power) - gamma * x[t]
        x[t + 1] = x[t] + dt * dx

    return x[warmup + tau: warmup + tau + n_steps]


def narma10(n_steps, rng=None):
    """
    Generate NARMA-10 time series.

    y[t+1] = 0.3*y[t] + 0.05*y[t]*sum(y[t-i], i=0..9) + 1.5*u[t-9]*u[t] + 0.1
    """
    if rng is None:
        rng = np.random.default_rng()

    n = 10
    warmup = 200
    total = n_steps + warmup + n

    u = rng.uniform(0, 0.5, size=total)
    y = np.zeros(total)
    y[0] = 0.0

    for t in range(total - 1):
        if t < n:
            y_sum = np.sum(y[max(0, t-n+1):t+1])
        else:
            y_sum = np.sum(y[t-n+1:t+1])

        y[t+1] = 0.3 * y[t] + 0.05 * y[t] * y_sum + 1.5 * u[t-n+1] * u[t] + 0.1

        # Clip to avoid divergence
        y[t+1] = np.clip(y[t+1], -10, 10)

    return u[warmup + n: warmup + n + n_steps], y[warmup + n: warmup + n + n_steps]


# =============================================================================
# 3. Noise model (Eq. 10)
# =============================================================================

def apply_noise(m, sigma, rng):
    """
    Apply controlled readout noise: sign flips + additive Gaussian.
    m_tilde = xi * m + sigma * epsilon
    where xi in {-1,+1}^M (random sign flips) and epsilon ~ N(0,I).

    NOTE: The sign-flip vector xi should flip each bit with some probability.
    From the paper: xi is a random sign-flip vector. At sigma=0, if xi flips
    randomly, this would destroy the signal. Looking at the paper's results
    where sigma=0 gives good performance, the sign-flip probability must be
    controlled by sigma as well. We interpret this as: the sign-flip
    probability is proportional to sigma, making the noise model purely
    controlled by sigma.

    Actually, re-reading Eq (10): xi * m + sigma * epsilon with xi random in {-1,+1}^M.
    If xi is fully random (50/50), the mean of xi*m is 0, destroying all signal
    even at sigma=0. This contradicts results.

    The most consistent interpretation: sigma controls BOTH effects. At sigma=0,
    xi = +1 (no flips). We use flip probability = sigma (capped at 0.5).
    """
    M = m.shape[-1]
    # Sign flips with probability proportional to sigma
    flip_prob = min(sigma, 0.5)
    xi = np.where(rng.random(m.shape) < flip_prob, -1.0, 1.0)
    epsilon = rng.standard_normal(m.shape)
    return xi * m + sigma * epsilon


# =============================================================================
# 4. Full reservoir pipeline
# =============================================================================

def run_reservoir_pipeline(input_series, reservoir, n_shots, sigma,
                           eta=0.5, L=20, rng=None):
    """
    Run the quantum reservoir on a time series and collect shot-level features.

    Args:
        input_series: 1D array of input values (length T)
        reservoir: QuantumReservoir instance
        n_shots: number of shots per time step
        sigma: noise level
        eta: leaky integrator decay
        L: window length
        rng: random number generator

    Returns:
        shot_features: array of shape (n_shots, T, d) where d = L*M + 1
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(input_series)
    M = reservoir.M

    # Leaky integrator states: (n_shots, T, M)
    r = np.zeros((n_shots, T, M))

    for t in range(T):
        u = input_series[t]
        state = reservoir.prepare_state(u)

        # Sample all shots at once (vectorized)
        m_all = reservoir.sample_multiple_shots(state, n_shots, rng=rng)  # (n_shots, M)

        # Apply noise model (Eq. 10) - vectorized for all shots
        m_noisy = apply_noise(m_all, sigma, rng)  # (n_shots, M)

        # Leaky integration (Eq. 11)
        if t == 0:
            r[:, t, :] = eta * m_noisy
        else:
            r[:, t, :] = (1 - eta) * r[:, t-1, :] + eta * m_noisy

    # Build windowed features (Eq. 12) - vectorized
    # phi_t^(n) = [r_{t-L+1}^(n); ...; r_t^(n)]
    d_feat = L * M
    shot_features = np.zeros((n_shots, T, d_feat))
    for t in range(T):
        for l in range(L):
            idx = t - L + 1 + l
            if idx >= 0:
                shot_features[:, t, l*M:(l+1)*M] = r[:, idx, :]

    return shot_features


# =============================================================================
# 5. EV and Trajectory training protocols
# =============================================================================

def nrmse(y_true, y_pred):
    """Normalized Root Mean Square Error (Eq. 6)."""
    return np.sqrt(np.mean((y_true - y_pred)**2)) / np.std(y_true)


RIDGE_LAMBDA = 10.0  # Fixed regularization for all protocols (fair: same model, same lambda)


def ev_training(shot_features, targets, t0, alpha, ridge_lambda=RIDGE_LAMBDA):
    """
    Algorithm 1: EV training.
    Average shots before learning, then train ridge regression.
    """
    n_shots, T, d = shot_features.shape

    # Average across shots (Eq. 15)
    ev_features = np.mean(shot_features, axis=0)  # (T, d)

    # Split into train/test
    usable_start = t0
    usable_T = T - usable_start
    train_end = usable_start + int(alpha * usable_T)

    X_train = ev_features[usable_start:train_end]
    y_train = targets[usable_start:train_end]
    X_test = ev_features[train_end:]
    y_test = targets[train_end:]

    if len(X_train) < 2 or len(X_test) < 2:
        return np.inf, np.array([]), np.array([])

    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    ridge = Ridge(alpha=ridge_lambda, fit_intercept=True)
    ridge.fit(X_train_s, y_train)

    y_pred_test = ridge.predict(X_test_s)
    score = nrmse(y_test, y_pred_test)

    return score, y_pred_test, y_test


def trajectory_training(shot_features, targets, t0, alpha, ridge_lambda=RIDGE_LAMBDA):
    """
    Algorithm 2: Trajectory-level training.
    Train on stacked shot-level dataset, average predictions at inference.
    """
    n_shots, T, d = shot_features.shape

    # Split into train/test
    usable_start = t0
    usable_T = T - usable_start
    train_end = usable_start + int(alpha * usable_T)

    # Stack all shots for training (Eq. in Algorithm 2, step 9)
    X_train_list = []
    y_train_list = []
    for n in range(n_shots):
        X_train_list.append(shot_features[n, usable_start:train_end, :])
        y_train_list.append(targets[usable_start:train_end])

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    if len(X_train) < 2 or train_end >= T:
        return np.inf, np.array([]), np.array([])

    # Standardize features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    ridge = Ridge(alpha=ridge_lambda, fit_intercept=True)
    ridge.fit(X_train_s, y_train)

    # Predict: average across shots (Eq. 17)
    y_test = targets[train_end:]
    test_len = T - train_end

    preds = np.zeros((n_shots, test_len))
    for n in range(n_shots):
        X_test_n = shot_features[n, train_end:, :]
        X_test_s = scaler.transform(X_test_n)
        preds[n] = ridge.predict(X_test_s)

    y_pred_test = np.mean(preds, axis=0)
    score = nrmse(y_test, y_pred_test)

    return score, y_pred_test, y_test


def grouped_trajectory_training(
    shot_features, targets, t0, alpha, group_size=None,
    ridge_lambda=RIDGE_LAMBDA, std_objective_weight=0.0,
    calibrate_amplitude=False, calibration_std_weight=None,
    calibration_blends=(0.0, 0.3, 0.5, 0.7, 1.0), calibration_max_scale=2.0,
    candidate_group_sizes=None, calibration_max_val_nrmse_rel_increase=0.08
):
    """
    Grouped trajectory training with optional std-aware model selection.

    If `group_size` is None, K is selected on a validation split using the
    aggregated (group-averaged) validation prediction, i.e. the same inference
    mode used at test time.

    Optional amplitude calibration uses validation data only:
      y_cal = a * y_pred + b
    where `a` is bounded and blended to avoid over-correction.
    """
    n_shots, T, d = shot_features.shape

    usable_start = t0
    usable_T = T - usable_start
    train_end = usable_start + int(alpha * usable_T)

    y_test = targets[train_end:]
    if train_end >= T or len(y_test) < 2:
        return np.inf, np.array([]), np.array([]), 1

    if calibration_std_weight is None:
        calibration_std_weight = std_objective_weight

    def _safe_std(x):
        return max(np.std(x), 1e-10)

    def _group_features(k):
        return [
            np.mean(shot_features[g * k:(g + 1) * k], axis=0)
            for g in range(n_shots // k)
        ]

    def _fit_model(group_feats, start_idx, end_idx):
        X_train_list = [gf[start_idx:end_idx] for gf in group_feats]
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate([targets[start_idx:end_idx] for _ in group_feats])
        if len(X_train) < 2:
            return None, None
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        ridge = Ridge(alpha=ridge_lambda, fit_intercept=True)
        ridge.fit(X_train_s, y_train)
        return scaler, ridge

    def _predict_aggregated(group_feats, scaler, ridge, start_idx, end_idx):
        if end_idx <= start_idx:
            return np.array([])
        preds = []
        for gf in group_feats:
            X = gf[start_idx:end_idx]
            preds.append(ridge.predict(scaler.transform(X)))
        return np.mean(preds, axis=0)

    def _objective(y_true, y_pred, std_weight):
        if len(y_true) < 2 or len(y_pred) != len(y_true):
            return np.inf
        obj = nrmse(y_true, y_pred)
        if std_weight > 0:
            sr = _safe_std(y_pred) / _safe_std(y_true)
            obj += std_weight * (np.log(max(sr, 1e-6)) ** 2)
        return obj

    # Candidate group sizes
    if candidate_group_sizes is None:
        default_candidates = [1, 2, 4, 5, 10, n_shots]
        candidates = sorted({
            int(k) for k in default_candidates
            if int(k) > 0 and int(k) <= n_shots and (n_shots % int(k) == 0)
        })
    else:
        candidates = sorted({
            int(k) for k in candidate_group_sizes
            if int(k) > 0 and int(k) <= n_shots and (n_shots % int(k) == 0)
        })
    if not candidates:
        candidates = [1, n_shots]

    # Auto-select K if not specified
    if group_size is None:
        val_split = usable_start + int(0.7 * (train_end - usable_start))
        y_val = targets[val_split:train_end]

        best_k = candidates[0]
        best_obj = np.inf

        for k in candidates:
            group_feats = _group_features(k)
            scaler, ridge = _fit_model(group_feats, usable_start, val_split)
            if scaler is None:
                continue
            y_val_pred = _predict_aggregated(
                group_feats, scaler, ridge, val_split, train_end
            )
            obj = _objective(y_val, y_val_pred, std_objective_weight)
            if obj < best_obj:
                best_obj = obj
                best_k = k

        group_size = best_k

    # Train final grouped model on full train window
    group_feats = _group_features(group_size)
    scaler, ridge = _fit_model(group_feats, usable_start, train_end)
    if scaler is None:
        return np.inf, np.array([]), np.array([]), group_size

    y_pred_test = _predict_aggregated(group_feats, scaler, ridge, train_end, T)
    if len(y_pred_test) != len(y_test):
        return np.inf, np.array([]), np.array([]), group_size

    # Optional validation-based affine amplitude calibration
    if calibrate_amplitude:
        val_split = usable_start + int(0.7 * (train_end - usable_start))
        y_val = targets[val_split:train_end]

        cal_scaler, cal_ridge = _fit_model(group_feats, usable_start, val_split)
        if cal_scaler is not None and len(y_val) >= 2:
            y_val_pred = _predict_aggregated(
                group_feats, cal_scaler, cal_ridge, val_split, train_end
            )
            if len(y_val_pred) == len(y_val):
                raw_scale = _safe_std(y_val) / _safe_std(y_val_pred)
                raw_scale = np.clip(
                    raw_scale, 1.0 / max(calibration_max_scale, 1.0), calibration_max_scale
                )

                blend_candidates = sorted({
                    float(b) for b in calibration_blends
                    if 0.0 <= float(b) <= 1.0
                })
                if not blend_candidates:
                    blend_candidates = [0.0]
                if blend_candidates[0] > 0.0:
                    blend_candidates = [0.0] + blend_candidates

                best_a, best_b = 1.0, 0.0
                base_val_nrmse = nrmse(y_val, y_val_pred)
                best_cal_obj = _objective(y_val, y_val_pred, calibration_std_weight)

                for blend in blend_candidates:
                    a = 1.0 + blend * (raw_scale - 1.0)
                    b = np.mean(y_val) - a * np.mean(y_val_pred)
                    y_val_cal = a * y_val_pred + b
                    cal_val_nrmse = nrmse(y_val, y_val_cal)
                    if (
                        calibration_max_val_nrmse_rel_increase is not None
                        and cal_val_nrmse
                        > base_val_nrmse * (1.0 + calibration_max_val_nrmse_rel_increase)
                    ):
                        continue
                    obj = _objective(y_val, y_val_cal, calibration_std_weight)
                    if obj < best_cal_obj:
                        best_cal_obj = obj
                        best_a, best_b = a, b

                y_pred_test = best_a * y_pred_test + best_b

    score = nrmse(y_test, y_pred_test)
    return score, y_pred_test, y_test, group_size


def grouped_trajectory_training_legacy(
    shot_features, targets, t0, alpha, group_size=None, ridge_lambda=RIDGE_LAMBDA
):
    """Legacy grouped trajectory training (paper baseline)."""
    n_shots = shot_features.shape[0]
    legacy_candidates = [
        k for k in [1, 2, 4, 5, 10, n_shots]
        if n_shots % k == 0 and k <= n_shots
    ]
    return grouped_trajectory_training(
        shot_features, targets, t0, alpha,
        group_size=group_size,
        ridge_lambda=ridge_lambda,
        std_objective_weight=0.0,
        calibrate_amplitude=False,
        candidate_group_sizes=legacy_candidates,
    )


def grouped_trajectory_training_narma_fix(
    shot_features, targets, t0, alpha, group_size=None, ridge_lambda=RIDGE_LAMBDA
):
    """NARMA-10 fix: std-aware K selection + guarded amplitude calibration."""
    n_shots = shot_features.shape[0]
    return grouped_trajectory_training(
        shot_features, targets, t0, alpha,
        group_size=group_size,
        ridge_lambda=ridge_lambda,
        std_objective_weight=0.2,
        calibrate_amplitude=True,
        calibration_std_weight=1.2,
        calibration_blends=(0.0, 0.3, 0.5, 0.7, 1.0),
        calibration_max_scale=2.0,
        candidate_group_sizes=[k for k in range(1, n_shots + 1) if n_shots % k == 0],
        calibration_max_val_nrmse_rel_increase=0.08,
    )


# =============================================================================
# 6. Phase diagram sweep
# =============================================================================

def compute_rho_ev(S, n_shots, d=241, t0=30, alpha=0.7):
    """Compute EV sample ratio (Eq. 21). d = L*M + 1 = 241 as in paper."""
    T = S // n_shots
    return (alpha / d) * (T - t0)


def run_single_experiment(n_shots, sigma, S, benchmark='mackey_glass',
                          seed=0, eta=0.5, L=20, t0=30, alpha=0.7,
                          grouped=False, G=2, depth=5, circuit_family="ring_cnot"):
    """
    Run a single EV vs trajectory experiment.

    Returns:
        dict with nrmse_ev, nrmse_traj, gap
    """
    rng = np.random.default_rng(seed)

    # Compute T from budget
    if grouped:
        T = S // (G * n_shots)
    else:
        T = S // n_shots

    if T < t0 + 20:
        return {'nrmse_ev': np.inf, 'nrmse_traj': np.inf, 'gap': 0.0,
                'y_pred_ev': np.array([]), 'y_pred_traj': np.array([]),
                'y_test': np.array([])}

    # Generate benchmark data
    total_len = T + 50  # extra margin
    if benchmark == 'mackey_glass':
        series = mackey_glass(total_len, rng=rng)
        # Use lookback=1: predict x_t from reservoir driven by x_{t-1}
        inputs = series[:-1][:T]
        targets = series[1:][:T]
    elif benchmark == 'narma10':
        u_series, y_series = narma10(total_len, rng=rng)
        inputs = u_series[:T]
        targets = y_series[:T]
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    # Build reservoir
    reservoir = QuantumReservoir(
        n_qubits=6,
        depth=depth,
        rng=rng,
        circuit_family=circuit_family,
    )

    # Run reservoir pipeline
    shot_features = run_reservoir_pipeline(
        inputs, reservoir, n_shots, sigma, eta=eta, L=L,
        rng=np.random.default_rng(seed + 1000)
    )

    # EV training
    nrmse_ev, y_pred_ev, y_test_ev = ev_training(
        shot_features, targets, t0, alpha
    )

    # Trajectory training (pure, K=1)
    nrmse_traj, y_pred_traj, y_test_traj = trajectory_training(
        shot_features, targets, t0, alpha
    )

    # Grouped trajectory training (auto-K)
    if benchmark == 'narma10':
        nrmse_gtraj, y_pred_gtraj, y_test_gtraj, best_k = grouped_trajectory_training_narma_fix(
            shot_features, targets, t0, alpha
        )
    else:
        nrmse_gtraj, y_pred_gtraj, y_test_gtraj, best_k = grouped_trajectory_training_legacy(
            shot_features, targets, t0, alpha
        )

    gap = nrmse_ev - nrmse_traj
    gap_g = nrmse_ev - nrmse_gtraj

    return {
        'nrmse_ev': nrmse_ev,
        'nrmse_traj': nrmse_traj,
        'nrmse_gtraj': nrmse_gtraj,
        'gap': gap,
        'gap_g': gap_g,
        'best_k': best_k,
        'y_pred_ev': y_pred_ev,
        'y_pred_traj': y_pred_traj,
        'y_pred_gtraj': y_pred_gtraj,
        'y_test': y_test_ev
    }


def run_phase_diagram(S=10000, n_seeds=10, eta=0.5, L=20, t0=30, alpha=0.7):
    """
    Run the full (Nshots, sigma) phase diagram at fixed S.
    """
    shots_list = [5, 10, 15, 20, 25, 30, 40, 50]
    sigma_list = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]

    results = {}
    total_configs = len(shots_list) * len(sigma_list)
    config_idx = 0

    for n_shots in shots_list:
        for sigma in sigma_list:
            config_idx += 1
            T = S // n_shots

            ev_scores = []
            traj_scores = []
            gtraj_scores = []
            best_ks = []

            for seed in range(n_seeds):
                print(f"  Config {config_idx}/{total_configs}: "
                      f"Nshots={n_shots}, sigma={sigma:.2f}, seed={seed}/{n_seeds}")

                result = run_single_experiment(
                    n_shots, sigma, S, benchmark='mackey_glass',
                    seed=seed, eta=eta, L=L, t0=t0, alpha=alpha, depth=5
                )
                ev_scores.append(result['nrmse_ev'])
                traj_scores.append(result['nrmse_traj'])
                gtraj_scores.append(result['nrmse_gtraj'])
                best_ks.append(result['best_k'])
                sys.stdout.flush()

            ev_mean = np.mean(ev_scores)
            ev_std = np.std(ev_scores)
            traj_mean = np.mean(traj_scores)
            traj_std = np.std(traj_scores)
            gtraj_mean = np.mean(gtraj_scores)
            gtraj_std = np.std(gtraj_scores)
            gap_mean = ev_mean - traj_mean
            gap_g_mean = ev_mean - gtraj_mean

            rho_ev = compute_rho_ev(S, n_shots, d=241, t0=t0, alpha=alpha)

            results[(n_shots, sigma)] = {
                'T': T,
                'n_shots': n_shots,
                'sigma': sigma,
                'rho_ev': rho_ev,
                'ev_mean': ev_mean,
                'ev_std': ev_std,
                'traj_mean': traj_mean,
                'traj_std': traj_std,
                'gtraj_mean': gtraj_mean,
                'gtraj_std': gtraj_std,
                'gap_mean': gap_mean,
                'gap_g_mean': gap_g_mean,
                'best_ks': best_ks,
                'ev_scores': ev_scores,
                'traj_scores': traj_scores,
                'gtraj_scores': gtraj_scores,
            }

    return results, shots_list, sigma_list


# =============================================================================
# 7. Plotting functions
# =============================================================================

# Nature-style color palette
COLOR_EV = '#4477AA'
COLOR_TRAJ = '#CC6677'
COLOR_GTRAJ = '#228833'
COLOR_TRUE = 'black'


def setup_nature_style():
    """Configure matplotlib rcParams for Nature-style two-column figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 7,
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'lines.linewidth': 1.0,
        'axes.grid': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,       # TrueType fonts in PDF
        'ps.fonttype': 42,
    })


def _add_panel_label(ax, label, x=-0.12, y=1.08):
    """Add bold panel label (a), (b), etc. in top-left of axes."""
    ax.text(x, y, f'({label})', transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left')


def plot_phase_diagram(results, shots_list, sigma_list, S=10000):
    """Nature-style phase diagram: 3 panels (EV NRMSE, G-Traj NRMSE, Gap)."""
    setup_nature_style()

    n_s = len(shots_list)
    n_sig = len(sigma_list)

    ev_grid = np.zeros((n_sig, n_s))
    gtraj_grid = np.zeros((n_sig, n_s))
    gap_grid = np.zeros((n_sig, n_s))

    for i, sigma in enumerate(sigma_list):
        for j, n_shots in enumerate(shots_list):
            r = results[(n_shots, sigma)]
            ev_grid[i, j] = r['ev_mean']
            gtraj_grid[i, j] = r.get('gtraj_mean', r['traj_mean'])
            gap_grid[i, j] = r['ev_mean'] - r.get('gtraj_mean', r['traj_mean'])

    shot_labels = [str(s) for s in shots_list]
    sigma_labels = [f'{s:.2f}' for s in sigma_list]

    # Shared NRMSE colorbar range for panels (a) and (b)
    vmin_nrmse = min(ev_grid.min(), gtraj_grid.min()) * 0.95
    vmax_nrmse = max(ev_grid.max(), gtraj_grid.max()) * 1.05
    vmin_nrmse = max(vmin_nrmse, 0.3)
    vmax_nrmse = min(vmax_nrmse, 2.0)

    fig, axes = plt.subplots(1, 3, figsize=(7, 2))

    # --- Panel (a): EV NRMSE ---
    im_a = axes[0].imshow(ev_grid, aspect='auto', cmap='YlOrRd',
                           vmin=vmin_nrmse, vmax=vmax_nrmse, interpolation='nearest')
    axes[0].set_xticks(range(n_s))
    axes[0].set_xticklabels(shot_labels)
    axes[0].set_yticks(range(n_sig))
    axes[0].set_yticklabels(sigma_labels)
    axes[0].set_xlabel(r'$N_{\mathrm{shots}}$')
    axes[0].set_ylabel(r'Noise $\sigma$')
    axes[0].set_title('EV')
    _add_panel_label(axes[0], 'a')

    # --- Panel (b): G-Traj NRMSE ---
    im_b = axes[1].imshow(gtraj_grid, aspect='auto', cmap='YlOrRd',
                           vmin=vmin_nrmse, vmax=vmax_nrmse, interpolation='nearest')
    axes[1].set_xticks(range(n_s))
    axes[1].set_xticklabels(shot_labels)
    axes[1].set_yticks(range(n_sig))
    axes[1].set_yticklabels([])
    axes[1].set_xlabel(r'$N_{\mathrm{shots}}$')
    axes[1].set_title('Grouped-Traj')
    _add_panel_label(axes[1], 'b')

    # Shared colorbar for (a) and (b)
    cbar_ab = fig.colorbar(im_b, ax=axes[:2].tolist(), fraction=0.04, pad=0.03)
    cbar_ab.set_label('NRMSE', fontsize=6)
    cbar_ab.ax.tick_params(labelsize=5)

    # --- Panel (c): Gap = EV - G-Traj (diverging) ---
    vmax_gap = max(abs(gap_grid.min()), abs(gap_grid.max()), 0.05)
    try:
        norm = TwoSlopeNorm(vmin=-vmax_gap, vcenter=0, vmax=vmax_gap)
    except Exception:
        norm = None
    im_c = axes[2].imshow(gap_grid, aspect='auto', cmap='RdBu_r', norm=norm,
                           interpolation='nearest')
    axes[2].set_xticks(range(n_s))
    axes[2].set_xticklabels(shot_labels)
    axes[2].set_yticks(range(n_sig))
    axes[2].set_yticklabels([])
    axes[2].set_xlabel(r'$N_{\mathrm{shots}}$')
    axes[2].set_title(r'$\Delta$ = EV $-$ G-Traj')
    _add_panel_label(axes[2], 'c')

    cbar_c = fig.colorbar(im_c, ax=axes[2], fraction=0.06, pad=0.03)
    cbar_c.set_label(r'$\Delta$ NRMSE', fontsize=6)
    cbar_c.ax.tick_params(labelsize=5)

    fig.subplots_adjust(wspace=0.15)
    for fmt in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'fig1_phase_diagram.{fmt}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close(fig)


def plot_rho_collapse(results, shots_list, sigma_list):
    """Reproduce Figure 2: Collapse under rho_EV."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Fig. 2: Mechanism via ρ_EV collapse', fontsize=14)

    # Panel (a): Gap vs rho_EV colored by sigma
    colors = plt.cm.viridis(np.linspace(0, 1, len(sigma_list)))

    for i, sigma in enumerate(sigma_list):
        rhos = []
        gaps = []
        for n_shots in shots_list:
            r = results[(n_shots, sigma)]
            rhos.append(r['rho_ev'])
            gaps.append(r['gap_mean'])
        axes[0].plot(rhos, gaps, 'o-', color=colors[i], label=f'σ={sigma:.2f}',
                     markersize=5)

    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0].axvline(x=1, color='r', linestyle='--', alpha=0.5, label='ρ_EV = 1')
    axes[0].set_xlabel('EV sample ratio ρ_EV')
    axes[0].set_ylabel('Δ = NRMSE_EV − NRMSE_traj')
    axes[0].set_title('(a) Collapse: Δ(ρ_EV; σ)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel (b): Interpolated gap field in (rho_EV, sigma) space
    rho_vals = sorted(set(r['rho_ev'] for r in results.values()))
    gap_field = np.zeros((len(sigma_list), len(shots_list)))
    rho_field = np.zeros(len(shots_list))

    for j, n_shots in enumerate(shots_list):
        rho_field[j] = results[(n_shots, sigma_list[0])]['rho_ev']
        for i, sigma in enumerate(sigma_list):
            gap_field[i, j] = results[(n_shots, sigma)]['gap_mean']

    vmax = max(abs(gap_field.min()), abs(gap_field.max()), 0.1)
    try:
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    except Exception:
        norm = None
    im = axes[1].imshow(gap_field, aspect='auto', cmap='RdBu', norm=norm,
                         extent=[rho_field.min(), rho_field.max(),
                                 sigma_list[-1], sigma_list[0]])
    axes[1].set_xlabel('EV sample ratio ρ_EV')
    axes[1].set_ylabel('Noise level σ')
    axes[1].set_title('(b) Trajectory-win band (collapsed field)')
    axes[1].axvline(x=1, color='k', linestyle='--', alpha=0.7)
    plt.colorbar(im, ax=axes[1], label='Δ')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_rho_collapse.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def print_table1(results, shots_list, sigma_list, S=10000, d=241, t0=30, alpha=0.7):
    """Print comparison table: EV vs Traj vs Grouped-Traj."""
    print("\n" + "="*110)
    print("TABLE I: Shot-wise summary at fixed S = 10^4  (EV vs Trajectory vs Grouped-Trajectory)")
    print("="*110)
    print(f"{'Nshots':>6} {'T':>5} {'rho':>6} {'NRMSE_EV':>9} {'NRMSE_traj':>11} "
          f"{'NRMSE_gtraj':>12} {'K':>3} {'Δ_traj':>7} {'Δ_gtraj':>8} {'best':>8}")
    print("-"*110)

    for n_shots in shots_list:
        T = S // n_shots
        rho = compute_rho_ev(S, n_shots, d=d, t0=t0, alpha=alpha)

        ev_m, traj_m, gtraj_m = [], [], []
        ks = []
        for sigma in sigma_list:
            r = results[(n_shots, sigma)]
            ev_m.append(r['ev_mean'])
            traj_m.append(r['traj_mean'])
            gtraj_m.append(r.get('gtraj_mean', r['traj_mean']))
            ks.extend(r.get('best_ks', [1]))

        ev_avg = np.mean(ev_m)
        traj_avg = np.mean(traj_m)
        gtraj_avg = np.mean(gtraj_m)
        k_mode = int(np.median(ks)) if ks else 1
        gap_t = ev_avg - traj_avg
        gap_g = ev_avg - gtraj_avg

        best = 'EV' if ev_avg <= min(traj_avg, gtraj_avg) else \
               ('G-Traj' if gtraj_avg <= traj_avg else 'Traj')

        print(f"{n_shots:>6} {T:>5} {rho:>6.2f} {ev_avg:>9.3f} {traj_avg:>11.3f} "
              f"{gtraj_avg:>12.3f} {k_mode:>3} {gap_t:>+7.3f} {gap_g:>+8.3f} {best:>8}")

    print("="*110)


def print_table2(results, shots_list, sigma_list):
    """Reproduce Table II: Interpolated phase boundary."""
    print("\n" + "="*60)
    print("TABLE II: Interpolated phase boundary in collapsed coordinate")
    print("="*60)
    print(f"{'sigma':>8} {'rho_low':>10} {'rho_high':>10}")
    print("-"*60)

    for sigma in sigma_list:
        rhos = []
        gaps = []
        for n_shots in shots_list:
            r = results[(n_shots, sigma)]
            rhos.append(r['rho_ev'])
            gaps.append(r['gap_mean'])

        # Sort by rho
        order = np.argsort(rhos)
        rhos = np.array(rhos)[order]
        gaps = np.array(gaps)[order]

        # Find zero crossings by linear interpolation
        crossings = []
        for i in range(len(gaps) - 1):
            if gaps[i] * gaps[i+1] < 0:
                # Linear interpolation
                rho_cross = rhos[i] + (rhos[i+1] - rhos[i]) * (-gaps[i]) / (gaps[i+1] - gaps[i])
                crossings.append(rho_cross)

        if len(crossings) >= 2:
            print(f"{sigma:>8.2f} {crossings[0]:>10.3f} {crossings[-1]:>10.3f}")
        elif len(crossings) == 1:
            print(f"{sigma:>8.2f} {crossings[0]:>10.3f} {'N/A':>10}")
        else:
            print(f"{sigma:>8.2f} {'N/A':>10} {'N/A':>10}")

    print("="*60)


def run_grouped_diagnostic(benchmark='mackey_glass', n_shots=20, sigma=0.01,
                           S_total=10000, G=2, n_seeds=4, L=20, t0=30, alpha=0.7,
                           eta=0.5):
    """
    Run grouped-measurement diagnostic (Figs 3-5).
    Stotal = G * Nshots * T => T = Stotal / (G * Nshots)
    """
    T = S_total // (G * n_shots)
    print(f"\nGrouped diagnostic: {benchmark}, Nshots={n_shots}, sigma={sigma}, "
          f"G={G}, Stotal={S_total}, T={T}")

    all_results = []
    for seed in range(n_seeds):
        print(f"  Seed {seed}/{n_seeds}")
        result = run_single_experiment(
            n_shots, sigma, S_total, benchmark=benchmark,
            seed=seed + 100, eta=eta, L=L, t0=t0, alpha=alpha,
            grouped=True, G=G, depth=5
        )
        all_results.append(result)

    ev_scores = [r['nrmse_ev'] for r in all_results]
    traj_scores = [r['nrmse_traj'] for r in all_results]
    mean_gap = np.mean(ev_scores) - np.mean(traj_scores)
    win_frac = sum(1 for e, t in zip(ev_scores, traj_scores) if e > t) / n_seeds

    return all_results, ev_scores, traj_scores, mean_gap, win_frac, T


def plot_grouped_diagnostic(all_results, ev_scores, traj_scores, mean_gap,
                            win_frac, T, benchmark, n_shots, sigma, G,
                            S_total, fig_name):
    """Nature-style diagnostic figure with 3 protocols."""
    setup_nature_style()

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2))

    bm_label = 'Mackey-Glass' if benchmark == 'mackey_glass' else 'NARMA-10'
    fig.suptitle(rf'{bm_label} | $N_{{\mathrm{{shots}}}}$={n_shots}, '
                 rf'$\sigma$={sigma}, G={G}', fontsize=8, y=1.02)

    gtraj_scores = [r.get('nrmse_gtraj', np.inf) for r in all_results]
    gap_g = np.mean(ev_scores) - np.mean(gtraj_scores)
    bar_colors = [COLOR_EV, COLOR_TRAJ, COLOR_GTRAJ]

    # Panel (a): Time traces (seed 0)
    ax = axes[0]
    for i, result in enumerate(all_results):
        if len(result['y_test']) > 0 and i == 0:
            x_axis = np.arange(len(result['y_test']))
            ax.plot(x_axis, result['y_test'], color=COLOR_TRUE,
                    linewidth=1.5, label='True', zorder=3)
            if len(result['y_pred_ev']) == len(x_axis):
                ax.plot(x_axis, result['y_pred_ev'], color=COLOR_EV,
                        linewidth=1.0, label='EV', alpha=0.85)
            if len(result['y_pred_traj']) == len(x_axis):
                ax.plot(x_axis, result['y_pred_traj'], color=COLOR_TRAJ,
                        linewidth=0.8, label='Traj', alpha=0.7)
            if 'y_pred_gtraj' in result and len(result['y_pred_gtraj']) == len(x_axis):
                ax.plot(x_axis, result['y_pred_gtraj'], color=COLOR_GTRAJ,
                        linewidth=1.0, label='G-Traj', alpha=0.85)
    ax.set_xlabel('Test time step')
    ax.set_ylabel(bm_label)
    ax.legend(frameon=False, loc='upper right', handlelength=1.5)
    _add_panel_label(ax, 'a')

    # Panel (b): NRMSE bar chart
    ax = axes[1]
    x_pos = np.arange(3)
    means = [np.mean(ev_scores), np.mean(traj_scores), np.mean(gtraj_scores)]
    stds = [np.std(ev_scores), np.std(traj_scores), np.std(gtraj_scores)]
    ax.bar(x_pos, means, yerr=stds, width=0.6, color=bar_colors,
           edgecolor='white', linewidth=0.3, capsize=2,
           error_kw={'linewidth': 0.5})
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['EV', 'Traj', 'G-Traj'])
    ax.set_ylabel('NRMSE')
    ax.set_title(rf'$\Delta_g$ = {gap_g:+.3f}', fontsize=7, pad=3)
    _add_panel_label(ax, 'b')

    # Panel (c): Signal quality
    ax = axes[2]
    metrics = {'EV': [], 'Traj': [], 'G-Traj': []}
    for result in all_results:
        yt = result['y_test']
        if len(yt) < 2:
            continue
        for name, pred_key in [('EV', 'y_pred_ev'), ('Traj', 'y_pred_traj'),
                                ('G-Traj', 'y_pred_gtraj')]:
            pred = result.get(pred_key, np.array([]))
            if len(pred) == len(yt):
                sr = np.std(pred) / max(np.std(yt), 1e-10)
                metrics[name].append(sr)
    means_sq = [np.mean(metrics[k]) if metrics[k] else 0
                for k in ['EV', 'Traj', 'G-Traj']]
    stds_sq = [np.std(metrics[k]) if metrics[k] else 0
               for k in ['EV', 'Traj', 'G-Traj']]
    ax.bar(x_pos, means_sq, yerr=stds_sq, width=0.6, color=bar_colors,
           edgecolor='white', linewidth=0.3, capsize=2,
           error_kw={'linewidth': 0.5})
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['EV', 'Traj', 'G-Traj'])
    ax.set_ylabel('Std ratio')
    _add_panel_label(ax, 'c')

    fig.subplots_adjust(wspace=0.40)
    for fmt in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, fig_name.replace('.png', f'.{fmt}'))
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close(fig)


def plot_paper_figure2(n_seeds=10, S=10000, L=20, t0=30, alpha=0.7, eta=0.5):
    """
    Nature-style Figure 2: 2x3 grid.
    Row 1 = Mackey-Glass (Nshots=20, sigma=0.01)
    Row 2 = NARMA-10 (Nshots=10, sigma=0.0)
    Columns = time trace | NRMSE bars | signal quality
    """
    setup_nature_style()

    configs = [
        {'benchmark': 'mackey_glass', 'n_shots': 20, 'sigma': 0.01,
         'label': 'Mackey-Glass'},
        {'benchmark': 'narma10', 'n_shots': 10, 'sigma': 0.0,
         'label': 'NARMA-10'},
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7, 3.5))
    panel_labels = [['a', 'b', 'c'], ['d', 'e', 'f']]

    for row, cfg in enumerate(configs):
        # Run benchmark
        all_results = []
        for seed in range(n_seeds):
            print(f"  Fig2 {cfg['label']}: seed {seed+1}/{n_seeds}")
            sys.stdout.flush()
            result = run_single_experiment(
                cfg['n_shots'], cfg['sigma'], S,
                benchmark=cfg['benchmark'],
                seed=seed, eta=eta, L=L, t0=t0, alpha=alpha, depth=5
            )
            all_results.append(result)

        ev_scores = [r['nrmse_ev'] for r in all_results]
        traj_scores = [r['nrmse_traj'] for r in all_results]
        gtraj_scores = [r['nrmse_gtraj'] for r in all_results]

        # --- Column 0: Time trace (seed 0) ---
        ax = axes[row, 0]
        r0 = all_results[0]
        if len(r0['y_test']) > 0:
            x_axis = np.arange(len(r0['y_test']))
            ax.plot(x_axis, r0['y_test'], color=COLOR_TRUE, linewidth=1.5,
                    label='True', zorder=3)
            if len(r0['y_pred_ev']) == len(x_axis):
                ax.plot(x_axis, r0['y_pred_ev'], color=COLOR_EV, linewidth=1.0,
                        label='EV', alpha=0.85)
            if len(r0.get('y_pred_gtraj', [])) == len(x_axis):
                ax.plot(x_axis, r0['y_pred_gtraj'], color=COLOR_GTRAJ,
                        linewidth=1.0, label='G-Traj', alpha=0.85)
        ax.set_xlabel('Test time step')
        ax.set_ylabel(cfg['label'])
        if row == 0:
            ax.legend(frameon=False, loc='upper right', handlelength=1.5)
        _add_panel_label(ax, panel_labels[row][0])

        # --- Column 1: NRMSE bar chart ---
        ax = axes[row, 1]
        means = [np.mean(ev_scores), np.mean(traj_scores), np.mean(gtraj_scores)]
        stds = [np.std(ev_scores), np.std(traj_scores), np.std(gtraj_scores)]
        bar_colors = [COLOR_EV, COLOR_TRAJ, COLOR_GTRAJ]
        x_pos = np.arange(3)
        bars = ax.bar(x_pos, means, yerr=stds, width=0.6, color=bar_colors,
                       edgecolor='white', linewidth=0.3, capsize=2,
                       error_kw={'linewidth': 0.5})
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['EV', 'Traj', 'G-Traj'])
        ax.set_ylabel('NRMSE')
        # Annotate gap
        gap_val = means[0] - means[2]
        ax.set_title(rf'$\Delta$ = {gap_val:+.3f}', fontsize=7, pad=3)
        _add_panel_label(ax, panel_labels[row][1])

        # --- Column 2: Signal quality (std ratio) ---
        ax = axes[row, 2]
        metrics = {'EV': [], 'Traj': [], 'G-Traj': []}
        for result in all_results:
            yt = result['y_test']
            if len(yt) < 2:
                continue
            for name, pred_key in [('EV', 'y_pred_ev'), ('Traj', 'y_pred_traj'),
                                    ('G-Traj', 'y_pred_gtraj')]:
                pred = result.get(pred_key, np.array([]))
                if len(pred) == len(yt):
                    sr = np.std(pred) / max(np.std(yt), 1e-10)
                    metrics[name].append(sr)
        means_sq = [np.mean(metrics[k]) if metrics[k] else 0
                    for k in ['EV', 'Traj', 'G-Traj']]
        stds_sq = [np.std(metrics[k]) if metrics[k] else 0
                   for k in ['EV', 'Traj', 'G-Traj']]
        bars = ax.bar(x_pos, means_sq, yerr=stds_sq, width=0.6,
                       color=bar_colors, edgecolor='white', linewidth=0.3,
                       capsize=2, error_kw={'linewidth': 0.5})
        ax.axhline(y=1.0, color='k', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['EV', 'Traj', 'G-Traj'])
        ax.set_ylabel('Std ratio')
        _add_panel_label(ax, panel_labels[row][2])

    fig.subplots_adjust(hspace=0.45, wspace=0.40)
    for fmt in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'fig2_benchmarks.{fmt}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved {path}")
    plt.close(fig)


# =============================================================================
# 8. Main execution
# =============================================================================

def main():
    print("="*80)
    print("Budget-aware Quantum Reservoir Computing - Full Reproduction")
    print("="*80)

    S = 10000
    L = 20
    M = 12
    d = L * M + 1  # = 241
    t0 = 30
    alpha = 0.7
    eta = 0.5
    n_seeds = 10

    # ----- Step 1: Run the full phase diagram sweep -----
    print("\n[Step 1] Running phase diagram sweep (S=10000, 10 seeds per config)...")
    print("This is the main computational workload.\n")

    results, shots_list, sigma_list = run_phase_diagram(
        S=S, n_seeds=n_seeds, eta=eta, L=L, t0=t0, alpha=alpha
    )

    # ----- Step 2: Print Table I -----
    print_table1(results, shots_list, sigma_list, S=S, d=d, t0=t0, alpha=alpha)

    # ----- Step 3: Print headline optimum -----
    r_opt = results[(20, 0.01)]
    print(f"\nHeadline optimum (Nshots=20, sigma=0.01, S=10^4):")
    print(f"  NRMSE_EV     = {r_opt['ev_mean']:.3f} +/- {r_opt['ev_std']:.3f}")
    print(f"  NRMSE_traj   = {r_opt['traj_mean']:.3f} +/- {r_opt['traj_std']:.3f}")
    print(f"  NRMSE_gtraj  = {r_opt.get('gtraj_mean', 0):.3f} +/- {r_opt.get('gtraj_std', 0):.3f}")
    print(f"  Gap (EV-traj)  = {r_opt['gap_mean']:.3f}")
    print(f"  Gap (EV-gtraj) = {r_opt.get('gap_g_mean', 0):.3f}")

    # ----- Step 4: Print Table II -----
    print_table2(results, shots_list, sigma_list)

    # ----- Step 5: Verify N*_shots formula (Eq. 23) -----
    N_star = S / (t0 + d / alpha)
    print(f"\nN*_shots (Eq. 23) = S / (t0 + d/alpha) = {S} / ({t0} + {d}/{alpha}) = {N_star:.1f}")
    print(f"Paper claims N*_shots ≈ 26.7")

    # ----- Step 6: Plot Figure 1 (Phase Diagram) -----
    plot_phase_diagram(results, shots_list, sigma_list, S=S)

    # ----- Step 7: Plot Figure 2 (rho_EV collapse) -----
    plot_rho_collapse(results, shots_list, sigma_list)

    # ----- Step 8: Grouped measurement diagnostics -----

    # Fig 3: Mackey-Glass at Nshots=20, sigma=0.01, G=2
    print("\n[Step 8a] Grouped diagnostic: Mackey-Glass (Nshots=20, sigma=0.01, G=2)")
    res3, ev3, traj3, gap3, win3, T3 = run_grouped_diagnostic(
        benchmark='mackey_glass', n_shots=20, sigma=0.01,
        S_total=10000, G=2, n_seeds=4, L=L, t0=t0, alpha=alpha, eta=eta
    )
    plot_grouped_diagnostic(res3, ev3, traj3, gap3, win3, T3,
                            'mackey_glass', 20, 0.01, 2, 10000,
                            'fig3_mg_grouped.png')

    # Fig 4: Mackey-Glass at Nshots=10, sigma=0.05, G=2
    print("\n[Step 8b] Grouped diagnostic: Mackey-Glass (Nshots=10, sigma=0.05, G=2)")
    res4, ev4, traj4, gap4, win4, T4 = run_grouped_diagnostic(
        benchmark='mackey_glass', n_shots=10, sigma=0.05,
        S_total=10000, G=2, n_seeds=4, L=L, t0=t0, alpha=alpha, eta=eta
    )
    plot_grouped_diagnostic(res4, ev4, traj4, gap4, win4, T4,
                            'mackey_glass', 10, 0.05, 2, 10000,
                            'fig4_mg_grouped2.png')

    # Fig 5: NARMA-10 at Nshots=10, sigma=0, G=2
    print("\n[Step 8c] Grouped diagnostic: NARMA-10 (Nshots=10, sigma=0, G=2)")
    res5, ev5, traj5, gap5, win5, T5 = run_grouped_diagnostic(
        benchmark='narma10', n_shots=10, sigma=0.0,
        S_total=10000, G=2, n_seeds=4, L=L, t0=t0, alpha=alpha, eta=eta
    )
    plot_grouped_diagnostic(res5, ev5, traj5, gap5, win5, T5,
                            'narma10', 10, 0.0, 2, 10000,
                            'fig5_narma10_grouped.png')

    # ----- Step 9: Summary of claim validation -----
    print("\n" + "="*80)
    print("CLAIM VALIDATION SUMMARY")
    print("="*80)

    # Claim 1: Trajectory wins in intermediate-shot regime
    intermediate_wins = True
    for n_shots in [15, 20, 25, 30]:
        wins = sum(1 for sigma in sigma_list if results[(n_shots, sigma)]['gap_mean'] > 0)
        total = len(sigma_list)
        status = "PASS" if wins >= total // 2 else "FAIL"
        print(f"  Nshots={n_shots}: trajectory wins {wins}/{total} noise levels [{status}]")
        if wins < total // 2:
            intermediate_wins = False

    print(f"\n  Claim 1 (intermediate-shot regime): "
          f"{'CONFIRMED' if intermediate_wins else 'NOT FULLY CONFIRMED'}")

    # Claim 2: EV wins at extreme shots
    low_shot_ev_wins = all(
        results[(n_shots, sigma)]['gap_mean'] < 0
        for n_shots in [5]
        for sigma in sigma_list
    )
    high_shot_ev_wins = all(
        results[(n_shots, sigma)]['gap_mean'] < 0
        for n_shots in [50]
        for sigma in sigma_list
    )
    print(f"\n  Claim 2a (EV wins at low shots, Nshots=5): "
          f"{'CONFIRMED' if low_shot_ev_wins else 'NOT FULLY CONFIRMED'}")
    print(f"  Claim 2b (EV wins at high shots, Nshots=50): "
          f"{'CONFIRMED' if high_shot_ev_wins else 'NOT FULLY CONFIRMED'}")

    # Claim 3: Headline gap
    print(f"\n  Claim 3 (headline gap at Nshots=20, sigma=0.01):")
    print(f"    Paper:  NRMSE_EV=1.067, NRMSE_traj=0.686, gap=0.381")
    print(f"    Ours:   NRMSE_EV={r_opt['ev_mean']:.3f}, "
          f"NRMSE_traj={r_opt['traj_mean']:.3f}, gap={r_opt['gap_mean']:.3f}")
    gap_match = r_opt['gap_mean'] > 0
    print(f"    Direction match: {'YES' if gap_match else 'NO'}")

    # Claim 4: rho_EV crossover near 1
    # Check that trajectory wins cluster around rho_EV ~ 1
    traj_win_rhos = []
    for n_shots in shots_list:
        for sigma in sigma_list:
            r = results[(n_shots, sigma)]
            if r['gap_mean'] > 0:
                traj_win_rhos.append(r['rho_ev'])
    if traj_win_rhos:
        mean_win_rho = np.mean(traj_win_rhos)
        print(f"\n  Claim 4 (crossover near rho_EV ~ 1):")
        print(f"    Mean rho_EV where trajectory wins: {mean_win_rho:.3f}")
        print(f"    Crossover region consistent with rho_EV ~ 1: "
              f"{'YES' if 0.5 < mean_win_rho < 2.5 else 'UNCERTAIN'}")

    # Claim 5: N*_shots formula
    print(f"\n  Claim 5 (N*_shots formula):")
    print(f"    Computed N*_shots = {N_star:.1f}")
    print(f"    Paper claims ≈ 26.7")
    print(f"    Match: {'YES' if abs(N_star - 26.7) < 1.0 else 'NO'}")

    # Claim 6: NARMA-10 replication
    if gap5 > 0:
        print(f"\n  Claim 6 (NARMA-10 replication): CONFIRMED (gap = {gap5:.3f})")
    else:
        print(f"\n  Claim 6 (NARMA-10 replication): NOT CONFIRMED (gap = {gap5:.3f})")

    print("\n" + "="*80)
    print(f"All figures saved to: {OUT_DIR}/")
    print("="*80)

    # Save numerical results
    summary = {}
    for key, val in results.items():
        k_str = f"shots{key[0]}_sigma{key[1]}"
        summary[k_str] = {
            'n_shots': val['n_shots'],
            'sigma': val['sigma'],
            'T': val['T'],
            'rho_ev': val['rho_ev'],
            'ev_mean': val['ev_mean'],
            'ev_std': val['ev_std'],
            'traj_mean': val['traj_mean'],
            'traj_std': val['traj_std'],
            'gtraj_mean': val.get('gtraj_mean', val['traj_mean']),
            'gtraj_std': val.get('gtraj_std', val['traj_std']),
            'gap_mean': val['gap_mean'],
            'gap_g_mean': val.get('gap_g_mean', val['gap_mean']),
        }

    with open(os.path.join(OUT_DIR, 'sweep_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Numerical results saved to {OUT_DIR}/sweep_results.json")


if __name__ == '__main__':
    main()
