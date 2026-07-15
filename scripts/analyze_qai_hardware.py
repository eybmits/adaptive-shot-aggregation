#!/usr/bin/env python3
"""Analyze the hardware shot records at the shared operating point.

Input: results/qai_hardware/raw_{task}.npz (bits: bindings x shots x qubits,
shot order preserved) plus inputs_{task}.npz (inputs, targets).  The 400
shots per step are split into 20 consecutive, disjoint shot blocks of N=20
lanes.  These are shot-level replicates from one campaign per task, not
independent devices or calibration periods; every analysis is classical
post-processing:

  * protocols: full average / single shots / adaptive shot aggregation
    (lambda and k selected chronologically per record, exactly as in simulation)
  * the full group-size spectrum at every admissible k
  * an optional noise-to-signal initial candidate k0 = 2a (never a substitute
    for validation)
  * shot-order ablation: contiguous vs interleaved lane grouping
  * noise diagnostics: shot-index autocorrelation, excess inter-qubit
    correlation vs the exact simulator, drift across execution position

Outputs CSVs into results/qai_hardware/.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import base_impl as base  # noqa: E402
from protocol_config import load_protocol_config  # noqa: E402

HW_DIR = ROOT / "results" / "qai_hardware"
TASKS = ("mackey_glass", "santa_fe_laser", "rossler", "sunspots")
RECORD_SHOTS = 20
CFG = load_protocol_config(ROOT / "config.json")
C_WARM = 2.0  # fixed by the leave-one-task-out simulation calibration


def phi_from_measurements(meas: np.ndarray) -> np.ndarray:
    """meas: (lanes, T, m) in {-1,+1}. Mirrors base.run_reservoir exactly."""
    n_lanes, t_steps, m = meas.shape
    eta, lags = CFG.eta, CFG.window
    r = np.zeros((n_lanes, t_steps, m), dtype=float)
    r[:, 0, :] = eta * meas[:, 0, :]
    for t in range(1, t_steps):
        r[:, t, :] = (1.0 - eta) * r[:, t - 1, :] + eta * meas[:, t, :]
    phi = np.zeros((n_lanes, t_steps, lags * m), dtype=float)
    for t in range(t_steps):
        for lag in range(lags):
            idx = t - lags + 1 + lag
            if idx >= 0:
                phi[:, t, lag * m : (lag + 1) * m] = r[:, idx, :]
    return phi


def grouped_interleaved(phi: np.ndarray, k: int) -> np.ndarray:
    """Group lanes {g, g+G, g+2G, ...} instead of contiguous blocks."""
    n_lanes, t_steps, d = phi.shape
    if k == 1 or k == n_lanes:
        return base.grouped_features(phi, k)
    groups = n_lanes // k
    return phi.reshape(k, groups, t_steps, d).mean(axis=0)


def select_lambda_for_k(phi: np.ndarray, targets: np.ndarray, grouped: np.ndarray) -> Dict[str, float]:
    """Select lambda from validation only; do not access the test block."""
    best = None
    for lam in CFG.lambda_grid:
        res = base.fit_validate_grouped(grouped, targets, CFG, ridge_alpha=float(lam))
        if best is None or res["val_nrmse"] < best["val_nrmse"] - 1e-12:
            best = {**res, "lambda": float(lam)}
    return best


def noise_to_signal(phi: np.ndarray) -> float:
    t0, _, train_end = base.split_points(phi.shape[1], CFG)
    phi_train = phi[:, t0:train_end, :]
    ev = phi_train.mean(axis=0)
    signal = float(np.var(ev, axis=0, ddof=1).mean())
    noise = float(np.nanmean(np.var(phi_train, axis=0, ddof=1)))
    return noise / signal if signal > 1e-12 else float("nan")


def warm_start_k(a: float, ks: List[int]) -> int:
    interior = [k for k in ks if k not in (ks[0], ks[-1])]
    target = max(C_WARM * a, 1e-9)
    return min(interior, key=lambda k: abs(math.log(k) - math.log(target)))


def paired(delta: np.ndarray) -> Dict[str, float]:
    n = len(delta)
    sd = float(np.std(delta, ddof=1))
    ci = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else 0.0
    try:
        p = float(stats.wilcoxon(delta, alternative="two-sided").pvalue) if np.any(delta != 0) else 1.0
    except ValueError:
        p = float("nan")
    return {
        "delta_mean": float(np.mean(delta)), "delta_sd": sd, "delta_ci95": ci,
        "win_rate": float(np.mean(delta > 0)), "wilcoxon_p": p,
        "cohens_d": float(np.mean(delta) / sd) if sd > 0 else float("nan"),
        "n": n,
    }


def exact_sim_probs(dataset: str, t_steps: int, q: int) -> np.ndarray:
    inp = HW_DIR / f"inputs_{dataset}.npz"
    if not inp.exists():
        inp = ROOT / "results" / "qai_hardware" / f"inputs_{dataset}.npz"
    inputs = np.load(inp)["inputs"]
    res = base.ExactQRCReservoir(q=q, depth=1, entangler="ring_rzz",
                                 eta=CFG.eta, window=CFG.window, seed=0)
    probs = np.zeros((t_steps, 2 ** q))
    for t in range(t_steps):
        state = res.prepare_state(float(inputs[t]))
        probs[t] = np.abs(state) ** 2
    return probs


def sim_corr_matrix(probs: np.ndarray, q: int) -> np.ndarray:
    """Exact per-step correlation of Z outcomes, averaged over steps."""
    idx = np.arange(probs.shape[1])
    zvals = np.stack([1.0 - 2.0 * ((idx >> (q - 1 - j)) & 1) for j in range(q)], axis=1)
    corrs = []
    for t in range(probs.shape[0]):
        p = probs[t]
        mz = zvals.T @ p                       # <Z_j>
        second = (zvals.T * p) @ zvals         # <Z_i Z_j>
        cov = second - np.outer(mz, mz)
        sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        corrs.append(cov / np.outer(sd, sd))
    return np.mean(corrs, axis=0)


def offdiag_mean_abs(mat: np.ndarray) -> float:
    q = mat.shape[0]
    mask = ~np.eye(q, dtype=bool)
    return float(np.mean(np.abs(mat[mask])))


def load_task_archives(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one task and enforce the archived-hardware data contract.

    The hardware protocol uses one 500-binding, 400-shot, ten-qubit acquisition
    per task and backend. Santa Fe must have exactly the same dimensions; rejecting
    malformed or partial archives here prevents a silent change of protocol.
    """
    raw_path = HW_DIR / f"raw_{dataset}.npz"
    input_path = HW_DIR / f"inputs_{dataset}.npz"
    if not raw_path.exists() or not input_path.exists():
        missing = [str(path) for path in (raw_path, input_path) if not path.exists()]
        raise FileNotFoundError(f"missing hardware archive(s) for {dataset}: {missing}")

    with np.load(raw_path) as raw:
        if set(raw.files) != {"bits"}:
            raise ValueError(f"{raw_path.name} must contain only the 'bits' array")
        bits = np.asarray(raw["bits"])
    if bits.shape != (500, 400, 10):
        raise ValueError(
            f"{raw_path.name} has shape {bits.shape}; expected (500, 400, 10)"
        )
    if not np.issubdtype(bits.dtype, np.integer) or not np.all((bits == 0) | (bits == 1)):
        raise ValueError(f"{raw_path.name} must contain binary integer outcomes")

    with np.load(input_path) as archive:
        if set(archive.files) != {"inputs", "targets", "angles"}:
            raise ValueError(
                f"{input_path.name} must contain inputs, targets, and angles"
            )
        io = {name: np.asarray(archive[name], dtype=float) for name in archive.files}
    for name, values in io.items():
        if values.shape != (500,):
            raise ValueError(f"{input_path.name}:{name} has shape {values.shape}; expected (500,)")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{input_path.name}:{name} contains non-finite values")
    expected_angles = np.array([base.input_amplitude_angle(u) for u in io["inputs"]])
    if not np.allclose(io["angles"], expected_angles, rtol=0.0, atol=1e-12):
        raise ValueError(f"{input_path.name}:angles do not match the archived inputs")
    return bits.astype(np.uint8, copy=False), io["targets"]


def analyze_task(dataset: str) -> Dict[str, object]:
    bits, targets = load_task_archives(dataset)  # bits: (T, S, q)
    t_steps, n_shots_total, q = bits.shape
    meas_all = (1.0 - 2.0 * bits.astype(float)).transpose(1, 0, 2)  # (S, T, q)
    n_records = n_shots_total // RECORD_SHOTS
    ks = [k for k in range(1, RECORD_SHOTS + 1) if RECORD_SHOTS % k == 0]

    seed_rows, spectrum_rows, ablation_rows = [], [], []
    a_values, k0_values = [], []
    for rec in range(n_records):
        lanes = meas_all[rec * RECORD_SHOTS:(rec + 1) * RECORD_SHOTS]
        phi = phi_from_measurements(lanes)
        a = noise_to_signal(phi)
        a_values.append(a)

        per_k = {}
        for k in ks:
            res = select_lambda_for_k(phi, targets, base.grouped_features(phi, k))
            per_k[k] = res
            spectrum_rows.append({"dataset": dataset, "record": rec, "k": k,
                                  "grouping": "contiguous",
                                  "val_nrmse": res["val_nrmse"],
                                  "lambda": res["lambda"]})
            if k not in (1, RECORD_SHOTS):
                res_i = select_lambda_for_k(phi, targets, grouped_interleaved(phi, k))
                spectrum_rows.append({"dataset": dataset, "record": rec, "k": k,
                                      "grouping": "interleaved",
                                      "val_nrmse": res_i["val_nrmse"],
                                      "lambda": res_i["lambda"]})
                ablation_rows.append({"dataset": dataset, "record": rec, "k": k,
                                      "metric": "validation_nrmse",
                                      "contiguous": res["val_nrmse"],
                                      "interleaved": res_i["val_nrmse"]})

        interior = {k: v for k, v in per_k.items() if k not in (1, RECORD_SHOTS)}
        k_val = min(interior, key=lambda k: (interior[k]["val_nrmse"], k))
        k0 = warm_start_k(a, ks)
        k0_values.append(k0)
        for proto, k_used in (("ev", RECORD_SHOTS), ("raw", 1), ("split", k_val)):
            final = base.fit_test_grouped(
                base.grouped_features(phi, k_used), targets, CFG,
                ridge_alpha=float(per_k[k_used]["lambda"]),
            )
            seed_rows.append({"dataset": dataset, "record": rec, "protocol": proto,
                              "k": k_used, "test_nrmse": final["test_nrmse"],
                              "val_nrmse": per_k[k_used]["val_nrmse"]})
        warm_final = base.fit_test_grouped(
            base.grouped_features(phi, k0), targets, CFG,
            ridge_alpha=float(per_k[k0]["lambda"]),
        )
        seed_rows.append({"dataset": dataset, "record": rec, "protocol": "warm_start",
                          "k": k0, "test_nrmse": warm_final["test_nrmse"],
                          "val_nrmse": per_k[k0]["val_nrmse"]})

    # --- noise diagnostics on the full 400-shot blocks -------------------
    # shot-index autocorrelation within steps (i.i.d. null: ~N(0, 1/S))
    x = meas_all.transpose(1, 0, 2)  # (T, S, q)
    xc = x - x.mean(axis=1, keepdims=True)
    denom = (xc ** 2).sum(axis=1)
    ac = {}
    for lag in (1, 2, 5, 20):
        num = (xc[:, :-lag, :] * xc[:, lag:, :]).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ac[lag] = float(np.nanmean(num / np.where(denom > 1e-12, denom, np.nan)))
    # inter-qubit correlation: hardware vs exact simulator
    probs = exact_sim_probs(dataset, t_steps, q)
    sim_corr = sim_corr_matrix(probs, q)
    hw_corrs = []
    for t in range(t_steps):
        c = np.corrcoef(x[t].T)
        hw_corrs.append(c)
    hw_corr = np.nanmean(hw_corrs, axis=0)
    excess = hw_corr - sim_corr
    # drift: per-record readout mean vs execution position within the block
    rec_means = meas_all.reshape(n_records, RECORD_SHOTS, t_steps, q).mean(axis=(1, 2, 3))
    drift_slope, _, drift_r, drift_p, _ = stats.linregress(np.arange(n_records), rec_means)

    diagnostics = {
        "dataset": dataset,
        "autocorr_by_lag": {str(l): v for l, v in ac.items()},
        "autocorr_iid_null_sd": 1.0 / math.sqrt(n_shots_total),
        "mean_abs_offdiag_corr_hw": offdiag_mean_abs(hw_corr),
        "mean_abs_offdiag_corr_sim": offdiag_mean_abs(sim_corr),
        "mean_abs_offdiag_excess": offdiag_mean_abs(excess),
        "max_abs_offdiag_excess": float(np.max(np.abs(excess[~np.eye(q, dtype=bool)]))),
        "drift_slope_per_record": float(drift_slope),
        "drift_corr_r": float(drift_r),
        "drift_p": float(drift_p),
        "a_mean": float(np.mean(a_values)),
        "k0_mode": int(pd.Series(k0_values).mode().iloc[0]),
    }
    return {"seed_rows": seed_rows, "spectrum_rows": spectrum_rows,
            "ablation_rows": ablation_rows, "diagnostics": diagnostics}


def validate_analysis_tables(seed_df: pd.DataFrame, spectrum_df: pd.DataFrame,
                             ablation_df: pd.DataFrame) -> None:
    """Fail closed unless all four tasks have a complete analysis."""
    expected_tasks = set(TASKS)
    for name, frame in (("seed-level", seed_df), ("spectrum", spectrum_df),
                        ("ablation", ablation_df)):
        observed = set(frame["dataset"].unique()) if not frame.empty else set()
        if observed != expected_tasks:
            raise ValueError(
                f"{name} results contain tasks {sorted(observed)}; "
                f"expected {sorted(expected_tasks)}"
            )

    expected_protocols = {"ev", "raw", "split", "warm_start"}
    grouped = seed_df.groupby(["dataset", "record"])["protocol"].agg(set)
    if len(grouped) != len(TASKS) * 20 or any(value != expected_protocols for value in grouped):
        raise ValueError("seed-level results do not contain four protocols for every shot block")

    ks = {1, 2, 4, 5, 10, 20}
    contiguous = spectrum_df[spectrum_df["grouping"] == "contiguous"]
    grouped_ks = contiguous.groupby(["dataset", "record"])["k"].agg(set)
    if len(grouped_ks) != len(TASKS) * 20 or any(value != ks for value in grouped_ks):
        raise ValueError("group-size spectrum is incomplete")

    expected_interior = {2, 4, 5, 10}
    ablation_ks = ablation_df.groupby(["dataset", "record"])["k"].agg(set)
    if len(ablation_ks) != len(TASKS) * 20 or any(
        value != expected_interior for value in ablation_ks
    ):
        raise ValueError("shot-order ablation is incomplete")


def main() -> None:
    import argparse
    global HW_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", type=str, default="")
    args = ap.parse_args()
    if args.subdir:
        HW_DIR = ROOT / "results" / "qai_hardware" / args.subdir
    all_seed, all_spec, all_abl, diags = [], [], [], []
    missing = [
        str(path)
        for task in TASKS
        for path in (HW_DIR / f"raw_{task}.npz", HW_DIR / f"inputs_{task}.npz")
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "refusing to analyze or overwrite hardware results with an incomplete "
            f"task set; missing {missing}"
        )
    for task in TASKS:
        print(f"analyzing {task} ...")
        out = analyze_task(task)
        all_seed.extend(out["seed_rows"])
        all_spec.extend(out["spectrum_rows"])
        all_abl.extend(out["ablation_rows"])
        diags.append(out["diagnostics"])

    seed_df = pd.DataFrame(all_seed)
    spectrum_df = pd.DataFrame(all_spec)
    abl_df = pd.DataFrame(all_abl)
    validate_analysis_tables(seed_df, spectrum_df, abl_df)
    # A fixed significant-digit format prevents harmless last-bit differences
    # between NumPy and SciPy builds from changing the released evidence files.
    float_format = "%.14g"
    seed_df.to_csv(HW_DIR / "hardware_seed_level.csv", index=False, float_format=float_format)
    spectrum_df.to_csv(HW_DIR / "hardware_spectrum.csv", index=False, float_format=float_format)
    abl_df.to_csv(HW_DIR / "hardware_ablation.csv", index=False, float_format=float_format)

    def stable_json(value):
        if isinstance(value, float):
            return float(format(value, ".14g"))
        if isinstance(value, list):
            return [stable_json(item) for item in value]
        if isinstance(value, dict):
            return {key: stable_json(item) for key, item in value.items()}
        return value

    (HW_DIR / "hardware_noise_diagnostics.json").write_text(
        json.dumps(stable_json(diags), indent=1) + "\n"
    )

    # paired statistics: split vs ev, warm start vs ev, interleaved vs contiguous
    stats_rows = []
    for dataset, grp in seed_df.groupby("dataset"):
        wide = grp.pivot_table(index="record", columns="protocol", values="test_nrmse")
        for method in ("split", "warm_start", "raw"):
            d = paired((wide["ev"] - wide[method]).to_numpy())
            stats_rows.append({"dataset": dataset, "baseline": "ev", "method": method, **d})
    if not abl_df.empty:
        for dataset, grp in abl_df.groupby("dataset"):
            at = grp.groupby("record")[["contiguous", "interleaved"]].mean()
            stats_rows.append({"dataset": dataset, "baseline": "contiguous",
                               "method": "interleaved",
                               **paired((at["contiguous"] - at["interleaved"]).to_numpy())})
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(HW_DIR / "hardware_paired_stats.csv", index=False, float_format=float_format)

    summary = seed_df.groupby(["dataset", "protocol"])["test_nrmse"].agg(["mean", "std"])
    summary.to_csv(HW_DIR / "hardware_summary.csv", float_format=float_format)
    print(summary.round(3).to_string())
    print()
    print(stats_df.round(4).to_string(index=False))
    for d in diags:
        print(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
