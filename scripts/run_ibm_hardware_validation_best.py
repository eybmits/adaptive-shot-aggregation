#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "qrc_budget_mplcache"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "qrc_budget_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

import base_impl as base
from exp2_plot_style import (
    FD_COLOR,
    FD_LS,
    FD_LW,
    FIG_W,
    FILL_ALPHA_FD,
    FILL_ALPHA_ID,
    FILL_ALPHA_RAW,
    ID_COLOR,
    ID_LS,
    ID_LW,
    MARKER_EDGE_COLOR,
    MARKER_EDGE_WIDTH,
    RAW_COLOR,
    RAW_LS,
    RAW_LW,
    TITLE_MAP,
    TRUE_COLOR,
    apply_style,
    save_pdf_png,
    scaled_height,
    style_axis,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures"
DEFAULT_SEEDS = tuple(range(100, 120))
TASK_ORDER = ("mackey_glass", "lorenz", "narma10")


@dataclass(frozen=True)
class HardwareCandidate:
    dataset: str
    n_shots: int
    budget_total: int = 12000
    q: int = 4
    depth: int = 1
    entangler: str = "ring_cnot"
    eta: float = 0.2
    window: int = 10
    t0: int = 30
    train_frac: float = 0.70
    val_frac_within_train: float = 0.25
    lambda_default: float = 10.0
    sigma: float = 0.0
    source_rho_ev: float = np.nan
    source_gap_mean: float = np.nan

    @property
    def t_steps(self) -> int:
        return self.budget_total // (2 * self.n_shots)

    @property
    def feature_dim(self) -> int:
        return self.window * (2 * self.q)

    @property
    def rho_ev(self) -> float:
        return self.train_frac * max(self.t_steps - self.t0, 0) / self.feature_dim


BEST_CANDIDATES: Dict[str, HardwareCandidate] = {
    "mackey_glass": HardwareCandidate(
        dataset="mackey_glass",
        n_shots=50,
        source_rho_ev=0.7875,
        source_gap_mean=0.335334843656075,
    ),
    "lorenz": HardwareCandidate(
        dataset="lorenz",
        n_shots=25,
        source_rho_ev=1.8375,
        source_gap_mean=0.45075966938436507,
    ),
    "narma10": HardwareCandidate(
        dataset="narma10",
        n_shots=40,
        source_rho_ev=1.05,
        source_gap_mean=0.6479098410015063,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run IBM hardware validation for the best experiment-2 settings "
            "using a unified architecture and 20 confirmation seeds by default."
        )
    )
    parser.add_argument("--backend", type=str, default="ibm_marrakesh")
    parser.add_argument("--datasets", type=str, default=",".join(TASK_ORDER))
    parser.add_argument("--seed0", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--num-seeds", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--account-name", type=str, default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--seed-level-csv",
        type=str,
        default="hardware_validation_experiment2_best_seed_level.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="hardware_validation_experiment2_best_summary.csv",
    )
    parser.add_argument(
        "--jobs-json",
        type=str,
        default="hardware_validation_experiment2_best_manifest.json",
    )
    parser.add_argument(
        "--traces-npz",
        type=str,
        default="hardware_validation_experiment2_best_traces.npz",
    )
    parser.add_argument(
        "--audit-npz",
        type=str,
        default="hardware_validation_experiment2_best_audit_bundle.npz",
    )
    parser.add_argument(
        "--trace-wide-csv",
        type=str,
        default="hardware_validation_experiment2_best_trace_wide.csv",
    )
    parser.add_argument(
        "--trace-long-csv",
        type=str,
        default="hardware_validation_experiment2_best_trace_long.csv",
    )
    parser.add_argument(
        "--job-rows-csv",
        type=str,
        default="hardware_validation_experiment2_best_job_rows.csv",
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default="experiment_6",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_candidates(dataset_arg: str) -> List[HardwareCandidate]:
    chosen = [item.strip() for item in dataset_arg.split(",") if item.strip()]
    if not chosen:
        raise ValueError("No datasets selected.")
    unknown = [name for name in chosen if name not in BEST_CANDIDATES]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return [BEST_CANDIDATES[name] for name in TASK_ORDER if name in chosen]


def default_seed_list(seed0: int, num_seeds: int) -> List[int]:
    return list(range(int(seed0), int(seed0) + int(num_seeds)))


def resolve_service(account_name: str | None) -> Tuple[QiskitRuntimeService, List[str]]:
    saved_accounts = sorted(QiskitRuntimeService.saved_accounts().keys())
    if account_name:
        return QiskitRuntimeService(name=account_name), saved_accounts
    return QiskitRuntimeService(), saved_accounts


def _apply_entangler_layer(qc: QuantumCircuit, entangler: str, q: int) -> None:
    if entangler == "ring_cnot":
        for control in range(q):
            qc.cx(control, (control + 1) % q)
        return
    if entangler == "line_cnot":
        for control in range(q - 1):
            qc.cx(control, control + 1)
        return
    if entangler == "all_to_all_cz":
        for control in range(q):
            for target in range(control + 1, q):
                qc.cz(control, target)
        return
    raise ValueError(f"Unsupported entangler: {entangler}")


def build_parameterized_circuits(
    candidate: HardwareCandidate,
    reservoir: base.ExactQRCReservoir,
) -> Tuple[QuantumCircuit, QuantumCircuit]:
    u = Parameter("u")
    qc_z = QuantumCircuit(candidate.q)

    for qubit in range(candidate.q):
        qc_z.ry(float(reservoir.input_weights[qubit]) * u, qubit)

    for layer_idx in range(candidate.depth):
        _apply_entangler_layer(qc_z, candidate.entangler, candidate.q)
        layer = reservoir.layer_params[layer_idx]
        for qubit in range(candidate.q):
            qc_z.rx(float(layer[qubit, 0]), qubit)
            qc_z.ry(float(layer[qubit, 1]), qubit)
            qc_z.rz(float(layer[qubit, 2]), qubit)

    qc_x = qc_z.copy()
    for qubit in range(candidate.q):
        qc_x.h(qubit)

    qc_z.measure_all()
    qc_x.measure_all()
    return qc_z, qc_x


def bind_parameter(circuit: QuantumCircuit, value: float) -> QuantumCircuit:
    params = list(circuit.parameters)
    if len(params) != 1:
        raise ValueError(f"Expected one circuit parameter, got {len(params)}.")
    return circuit.assign_parameters({params[0]: float(value)}, inplace=False)


def extract_bitstrings(pub_result: object) -> Sequence[str]:
    data = getattr(pub_result, "data", None)
    meas = getattr(data, "meas", None)
    if meas is None or not hasattr(meas, "get_bitstrings"):
        raise RuntimeError(
            "Sampler result does not expose shot bitstrings via result.data.meas.get_bitstrings()."
        )
    return meas.get_bitstrings()


def bitstrings_to_pm_one(bitstrings: Sequence[str], q: int) -> np.ndarray:
    arr = np.empty((len(bitstrings), q), dtype=float)
    for shot_idx, bits in enumerate(bitstrings):
        rev = bits[::-1]
        arr[shot_idx, :] = [1.0 if bit == "0" else -1.0 for bit in rev]
    return arr


def window_features(r: np.ndarray, window: int) -> np.ndarray:
    n_shots, t_steps, m = r.shape
    phi = np.zeros((n_shots, t_steps, window * m), dtype=float)
    for t in range(t_steps):
        for lag in range(window):
            idx = t - window + 1 + lag
            if idx >= 0:
                phi[:, t, lag * m : (lag + 1) * m] = r[:, idx, :]
    return phi


def measure_hardware_phi(
    *,
    sampler: SamplerV2,
    pass_manager: object,
    candidate: HardwareCandidate,
    backend_name: str,
    seed: int,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], List[str], List[Dict[str, object]]]:
    raw = base.make_series(candidate.dataset, total_steps=candidate.t_steps)
    inputs, targets = base.normalize_and_shift(raw, candidate.t_steps, candidate)
    reservoir = base.ExactQRCReservoir(
        q=candidate.q,
        depth=candidate.depth,
        entangler=candidate.entangler,
        eta=candidate.eta,
        window=candidate.window,
        seed=seed,
    )

    qc_z, qc_x = build_parameterized_circuits(candidate, reservoir)
    isa_z = pass_manager.run(qc_z)
    isa_x = pass_manager.run(qc_x)

    r = np.zeros((candidate.n_shots, candidate.t_steps, 2 * candidate.q), dtype=float)
    job_ids: List[str] = []
    job_rows: List[Dict[str, object]] = []
    time_indices = list(range(candidate.t_steps))

    for chunk_idx, start in enumerate(range(0, candidate.t_steps, chunk_size)):
        batch = time_indices[start : start + chunk_size]
        pubs_z = [bind_parameter(isa_z, float(inputs[t])) for t in batch]
        pubs_x = [bind_parameter(isa_x, float(inputs[t])) for t in batch]

        job_z = sampler.run(pubs_z, shots=candidate.n_shots)
        job_x = sampler.run(pubs_x, shots=candidate.n_shots)
        job_ids.extend([job_z.job_id(), job_x.job_id()])
        job_rows.extend(
            [
                {
                    "dataset": candidate.dataset,
                    "seed": int(seed),
                    "backend": backend_name,
                    "chunk_idx": int(chunk_idx),
                    "basis": "Z",
                    "job_id": job_z.job_id(),
                    "time_start": int(batch[0]),
                    "time_stop_exclusive": int(batch[-1] + 1),
                    "n_time_steps": int(len(batch)),
                    "n_shots": int(candidate.n_shots),
                    "q": int(candidate.q),
                    "depth": int(candidate.depth),
                },
                {
                    "dataset": candidate.dataset,
                    "seed": int(seed),
                    "backend": backend_name,
                    "chunk_idx": int(chunk_idx),
                    "basis": "X",
                    "job_id": job_x.job_id(),
                    "time_start": int(batch[0]),
                    "time_stop_exclusive": int(batch[-1] + 1),
                    "n_time_steps": int(len(batch)),
                    "n_shots": int(candidate.n_shots),
                    "q": int(candidate.q),
                    "depth": int(candidate.depth),
                },
            ]
        )

        result_z = job_z.result()
        result_x = job_x.result()

        for batch_idx, t in enumerate(batch):
            z_arr = bitstrings_to_pm_one(extract_bitstrings(result_z[batch_idx]), candidate.q)
            x_arr = bitstrings_to_pm_one(extract_bitstrings(result_x[batch_idx]), candidate.q)
            measurements = np.concatenate([z_arr, x_arr], axis=1)
            if t == 0:
                r[:, t, :] = candidate.eta * measurements
            else:
                r[:, t, :] = (1.0 - candidate.eta) * r[:, t - 1, :] + candidate.eta * measurements

        print(
            f"[{candidate.dataset} seed={seed}] "
            f"{min(candidate.t_steps, start + chunk_size)}/{candidate.t_steps} time steps "
            f"submitted on hardware",
            flush=True,
        )

    _, fit_end, train_end = base.split_points(candidate.t_steps, candidate)
    phi = window_features(r, candidate.window)
    audit_arrays = {
        "raw_series": np.asarray(raw, dtype=float),
        "inputs": np.asarray(inputs, dtype=float),
        "targets": np.asarray(targets, dtype=float),
        "test_index_abs": np.arange(train_end, len(targets), dtype=int),
        "test_index_rel": np.arange(len(targets) - train_end, dtype=int),
        "split_points": np.asarray([candidate.t0, fit_end, train_end], dtype=int),
        "phi": np.asarray(phi, dtype=float),
    }
    return phi, targets, audit_arrays, job_ids, job_rows


def evaluate_single_seed(
    *,
    sampler: SamplerV2,
    pass_manager: object,
    backend_name: str,
    candidate: HardwareCandidate,
    seed: int,
    chunk_size: int,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object], List[Dict[str, object]]]:
    phi, targets, audit_arrays, job_ids, job_rows = measure_hardware_phi(
        sampler=sampler,
        pass_manager=pass_manager,
        candidate=candidate,
        backend_name=backend_name,
        seed=seed,
        chunk_size=chunk_size,
    )

    protocol_results = {
        protocol: base.evaluate_protocol(
            phi,
            targets,
            protocol=protocol,
            cfg=candidate,
            ridge_alpha=candidate.lambda_default,
        )
        for protocol in ("ev", "raw", "split")
    }

    row: Dict[str, object] = {
        "dataset": candidate.dataset,
        "backend": backend_name,
        "seed": int(seed),
        "budget_total": int(candidate.budget_total),
        "n_shots": int(candidate.n_shots),
        "t_steps": int(candidate.t_steps),
        "q": int(candidate.q),
        "depth": int(candidate.depth),
        "entangler": candidate.entangler,
        "eta": float(candidate.eta),
        "window": int(candidate.window),
        "t0": int(candidate.t0),
        "train_frac": float(candidate.train_frac),
        "val_frac_within_train": float(candidate.val_frac_within_train),
        "lambda_default": float(candidate.lambda_default),
        "sigma": float(candidate.sigma),
        "rho_ev": float(candidate.rho_ev),
        "source_rho_ev": float(candidate.source_rho_ev),
        "source_gap_mean": float(candidate.source_gap_mean),
        "fit_end": int(audit_arrays["split_points"][1]),
        "train_end": int(audit_arrays["split_points"][2]),
        "test_length": int(len(protocol_results["split"]["y_test"])),
    }
    for protocol, result in protocol_results.items():
        row[f"{protocol}_nrmse"] = float(result["test_nrmse"])
        row[f"{protocol}_sr"] = float(result["test_sr"])
        row[f"{protocol}_k"] = int(result["k"])
        row[f"{protocol}_val_nrmse"] = float(result["val_nrmse"])

    traces = {
        "test_index_abs": np.asarray(audit_arrays["test_index_abs"], dtype=int),
        "test_index_rel": np.asarray(audit_arrays["test_index_rel"], dtype=int),
        "y_test": np.asarray(protocol_results["split"]["y_test"], dtype=float),
        "ev": np.asarray(protocol_results["ev"]["y_pred"], dtype=float),
        "raw": np.asarray(protocol_results["raw"]["y_pred"], dtype=float),
        "split": np.asarray(protocol_results["split"]["y_pred"], dtype=float),
    }
    jobs = {
        "dataset": candidate.dataset,
        "seed": int(seed),
        "backend": backend_name,
        "job_ids": job_ids,
        "settings": asdict(candidate),
    }
    audit_arrays = {
        **audit_arrays,
        "y_test": np.asarray(protocol_results["split"]["y_test"], dtype=float),
        "y_pred_ev": np.asarray(protocol_results["ev"]["y_pred"], dtype=float),
        "y_pred_raw": np.asarray(protocol_results["raw"]["y_pred"], dtype=float),
        "y_pred_split": np.asarray(protocol_results["split"]["y_pred"], dtype=float),
    }
    return row, traces, audit_arrays, jobs, job_rows


def choose_representative_seed(group: pd.DataFrame) -> int:
    median_split = float(group["split_nrmse"].median())
    ranked = group.assign(split_dist=(group["split_nrmse"] - median_split).abs())
    ranked = ranked.sort_values(["split_dist", "seed"])
    return int(ranked.iloc[0]["seed"])


def summarize_seed_rows(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset in TASK_ORDER:
        group = seed_df[seed_df["dataset"] == dataset].copy()
        if group.empty:
            continue
        gap = group["ev_nrmse"] - group["split_nrmse"]
        rows.append(
            {
                "dataset": dataset,
                "backend": group["backend"].iloc[0],
                "budget_total": int(group["budget_total"].iloc[0]),
                "n_shots": int(group["n_shots"].iloc[0]),
                "t_steps": int(group["t_steps"].iloc[0]),
                "q": int(group["q"].iloc[0]),
                "depth": int(group["depth"].iloc[0]),
                "entangler": group["entangler"].iloc[0],
                "eta": float(group["eta"].iloc[0]),
                "window": int(group["window"].iloc[0]),
                "lambda_default": float(group["lambda_default"].iloc[0]),
                "sigma": float(group["sigma"].iloc[0]),
                "rho_ev": float(group["rho_ev"].iloc[0]),
                "source_rho_ev": float(group["source_rho_ev"].iloc[0]),
                "source_gap_mean": float(group["source_gap_mean"].iloc[0]),
                "seeds": int(len(group)),
                "ev_nrmse_mean": float(group["ev_nrmse"].mean()),
                "ev_nrmse_std": float(group["ev_nrmse"].std(ddof=1)),
                "raw_nrmse_mean": float(group["raw_nrmse"].mean()),
                "raw_nrmse_std": float(group["raw_nrmse"].std(ddof=1)),
                "split_nrmse_mean": float(group["split_nrmse"].mean()),
                "split_nrmse_std": float(group["split_nrmse"].std(ddof=1)),
                "gap_mean": float(gap.mean()),
                "gap_std": float(gap.std(ddof=1)),
                "gap_ci95": float(1.96 * gap.std(ddof=1) / np.sqrt(len(group))),
                "split_win_rate": float((gap > 0.0).mean()),
                "split_k_mode": int(Counter(group["split_k"].astype(int)).most_common(1)[0][0]),
                "representative_seed": choose_representative_seed(group),
            }
        )
    return pd.DataFrame(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_trace_bundle(path: Path, traces: Dict[Tuple[str, int], Dict[str, np.ndarray]]) -> None:
    bundle: Dict[str, np.ndarray] = {}
    for (dataset, seed), trace_map in traces.items():
        prefix = f"{dataset}__{seed}"
        for name, array in trace_map.items():
            bundle[f"{prefix}__{name}"] = np.asarray(array, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bundle)


def save_audit_bundle(path: Path, audits: Dict[Tuple[str, int], Dict[str, np.ndarray]]) -> None:
    bundle: Dict[str, np.ndarray] = {}
    for (dataset, seed), audit_map in audits.items():
        prefix = f"{dataset}__{seed}"
        for name, array in audit_map.items():
            bundle[f"{prefix}__{name}"] = np.asarray(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bundle)


def build_trace_tables(
    seed_df: pd.DataFrame,
    traces: Dict[Tuple[str, int], Dict[str, np.ndarray]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metadata_cols = [
        "dataset",
        "backend",
        "seed",
        "budget_total",
        "n_shots",
        "t_steps",
        "q",
        "depth",
        "entangler",
        "eta",
        "window",
        "lambda_default",
        "sigma",
        "rho_ev",
        "fit_end",
        "train_end",
    ]
    wide_rows: List[Dict[str, object]] = []
    long_rows: List[Dict[str, object]] = []

    for row in seed_df.to_dict(orient="records"):
        key = (str(row["dataset"]), int(row["seed"]))
        trace = traces[key]
        y_true = np.asarray(trace["y_test"], dtype=float)
        y_ev = np.asarray(trace["ev"], dtype=float)
        y_raw = np.asarray(trace["raw"], dtype=float)
        y_split = np.asarray(trace["split"], dtype=float)
        idx_abs = np.asarray(trace["test_index_abs"], dtype=int)
        idx_rel = np.asarray(trace["test_index_rel"], dtype=int)

        for i in range(len(y_true)):
            shared = {col: row[col] for col in metadata_cols}
            shared.update(
                {
                    "test_index_rel": int(idx_rel[i]),
                    "test_index_abs": int(idx_abs[i]),
                    "y_true": float(y_true[i]),
                    "y_pred_ev": float(y_ev[i]),
                    "y_pred_raw": float(y_raw[i]),
                    "y_pred_split": float(y_split[i]),
                    "abs_error_ev": float(abs(y_true[i] - y_ev[i])),
                    "abs_error_raw": float(abs(y_true[i] - y_raw[i])),
                    "abs_error_split": float(abs(y_true[i] - y_split[i])),
                }
            )
            wide_rows.append(shared)

            for protocol, pred in (
                ("ev", y_ev[i]),
                ("raw", y_raw[i]),
                ("split", y_split[i]),
            ):
                long_rows.append(
                    {
                        **{col: row[col] for col in metadata_cols},
                        "protocol": protocol,
                        "test_index_rel": int(idx_rel[i]),
                        "test_index_abs": int(idx_abs[i]),
                        "y_true": float(y_true[i]),
                        "y_pred": float(pred),
                        "abs_error": float(abs(y_true[i] - pred)),
                    }
                )

    return pd.DataFrame(wide_rows), pd.DataFrame(long_rows)


def make_summary_plot(
    seed_df: pd.DataFrame,
    pdf_path: Path,
    png_path: Path,
) -> None:
    apply_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIG_W, scaled_height(7.0, 2.55)),
        gridspec_kw={"wspace": 0.34},
    )

    protocol_specs = [
        ("EV", "ev_nrmse", ID_COLOR),
        ("Raw", "raw_nrmse", RAW_COLOR),
        ("Split", "split_nrmse", FD_COLOR),
    ]

    for col, dataset in enumerate(TASK_ORDER):
        group = seed_df[seed_df["dataset"] == dataset].copy().sort_values("seed")
        if group.empty:
            continue
        ax_box = axes[col]
        values = [group["ev_nrmse"].to_numpy(), group["raw_nrmse"].to_numpy(), group["split_nrmse"].to_numpy()]
        box = ax_box.boxplot(
            values,
            positions=[0, 1, 2],
            widths=0.56,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.0},
            whiskerprops={"color": "#555555", "linewidth": 0.8},
            capprops={"color": "#555555", "linewidth": 0.8},
            boxprops={"linewidth": 0.8, "edgecolor": "#444444"},
        )
        for patch, color, alpha in zip(
            box["boxes"],
            [ID_COLOR, RAW_COLOR, FD_COLOR],
            [FILL_ALPHA_ID, FILL_ALPHA_RAW, FILL_ALPHA_FD],
        ):
            patch.set_facecolor(color)
            patch.set_alpha(max(alpha * 1.8, 0.18))

        rng = np.random.default_rng(1000 + col)
        for xpos, (_, column, color) in enumerate(protocol_specs):
            vals = group[column].to_numpy(dtype=float)
            jitter = rng.normal(loc=0.0, scale=0.045, size=len(vals))
            ax_box.scatter(
                np.full(len(vals), xpos, dtype=float) + jitter,
                vals,
                s=16,
                alpha=0.22,
                color=color,
                edgecolors="none",
                zorder=2,
            )

        gap_mean = float((group["ev_nrmse"] - group["split_nrmse"]).mean())
        ax_box.text(
            0.04,
            0.96,
            f"Mean gap: {gap_mean:+.3f}",
            transform=ax_box.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92, "pad": 2.0},
        )
        ax_box.set_title(TITLE_MAP.get(dataset, dataset), y=1.03)
        ax_box.set_xticks([0, 1, 2], ["EV", "Raw", "Split"])
        if col == 0:
            ax_box.set_ylabel("Test NRMSE")
        style_axis(ax_box)

    legend_handles = [
        Line2D([0], [0], color=ID_COLOR, lw=ID_LW, ls=ID_LS, marker="o", markerfacecolor=ID_COLOR,
               markeredgecolor=MARKER_EDGE_COLOR, markeredgewidth=MARKER_EDGE_WIDTH, label="EV"),
        Line2D([0], [0], color=RAW_COLOR, lw=RAW_LW, ls=RAW_LS, marker="^", markerfacecolor=RAW_COLOR,
               markeredgecolor=MARKER_EDGE_COLOR, markeredgewidth=MARKER_EDGE_WIDTH, label="Raw"),
        Line2D([0], [0], color=FD_COLOR, lw=FD_LW, ls=FD_LS, marker="s", markerfacecolor=FD_COLOR,
               markeredgecolor=MARKER_EDGE_COLOR, markeredgewidth=MARKER_EDGE_WIDTH, label="Split"),
    ]
    fig.legend(
        legend_handles,
        ["EV", "Raw", "Split"],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        handlelength=1.9,
        columnspacing=2.0,
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.24)
    save_pdf_png(fig, pdf_path, png_path)


def ensure_chunk_size(chunk_size: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk-size must be positive.")
    if chunk_size > 300:
        raise ValueError("chunk-size must be <= 300 to stay below common backend max-circuit limits.")


def main() -> None:
    args = parse_args()
    ensure_chunk_size(int(args.chunk_size))
    candidates = selected_candidates(args.datasets)
    seed_list = default_seed_list(args.seed0, args.num_seeds)

    seed_csv_path = args.results_dir / args.seed_level_csv
    summary_csv_path = args.results_dir / args.summary_csv
    jobs_json_path = args.results_dir / args.jobs_json
    traces_npz_path = args.results_dir / args.traces_npz
    audit_npz_path = args.results_dir / args.audit_npz
    trace_wide_csv_path = args.results_dir / args.trace_wide_csv
    trace_long_csv_path = args.results_dir / args.trace_long_csv
    job_rows_csv_path = args.results_dir / args.job_rows_csv
    figure_pdf_path = args.figures_dir / f"{args.figure_name}.pdf"
    figure_png_path = args.figures_dir / f"{args.figure_name}.png"

    account_names = sorted(QiskitRuntimeService.saved_accounts().keys())
    if args.dry_run:
        print("IBM hardware validation dry run", flush=True)
        print(f"Saved runtime accounts: {account_names}", flush=True)
        print(f"Backend: {args.backend}", flush=True)
        print(f"Seeds: {seed_list}", flush=True)
        print("Selected settings:", flush=True)
        for candidate in candidates:
            print(
                f"  - {candidate.dataset}: "
                f"budget_total={candidate.budget_total}, n_shots={candidate.n_shots}, "
                f"q={candidate.q}, depth={candidate.depth}, entangler={candidate.entangler}, "
                f"eta={candidate.eta}, window={candidate.window}, lambda={candidate.lambda_default}, "
                f"rho_ev={candidate.rho_ev:.4f}",
                flush=True,
            )
        return

    if args.plot_only:
        if not seed_csv_path.exists():
            raise FileNotFoundError(f"Missing seed-level CSV for plot-only mode: {seed_csv_path}")
        seed_df = pd.read_csv(seed_csv_path)
        make_summary_plot(seed_df, figure_pdf_path, figure_png_path)
        print(f"Wrote: {figure_pdf_path}", flush=True)
        print(f"Wrote: {figure_png_path}", flush=True)
        return

    service, saved_accounts = resolve_service(args.account_name)
    print(f"Using saved Qiskit Runtime accounts: {saved_accounts}", flush=True)
    backend = service.backend(args.backend)
    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=int(args.optimization_level),
    )
    sampler = SamplerV2(mode=backend)

    seed_rows: List[Dict[str, object]] = []
    trace_store: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    audit_store: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    job_records: List[Dict[str, object]] = []
    job_rows_all: List[Dict[str, object]] = []

    for candidate in candidates:
        print(
            f"\n=== {candidate.dataset} on {args.backend} "
            f"(n_shots={candidate.n_shots}, q={candidate.q}, depth={candidate.depth}) ===",
            flush=True,
        )
        for seed in seed_list:
            print(
                f"Running dataset={candidate.dataset} seed={seed} "
                f"[{seed_list.index(seed) + 1}/{len(seed_list)}]",
                flush=True,
            )
            row, traces, audit_arrays, jobs, job_rows = evaluate_single_seed(
                sampler=sampler,
                pass_manager=pass_manager,
                backend_name=args.backend,
                candidate=candidate,
                seed=seed,
                chunk_size=int(args.chunk_size),
            )
            seed_rows.append(row)
            trace_store[(candidate.dataset, seed)] = traces
            audit_store[(candidate.dataset, seed)] = audit_arrays
            job_records.append(jobs)
            job_rows_all.extend(job_rows)

            pd.DataFrame(seed_rows).to_csv(seed_csv_path, index=False)
            pd.DataFrame(job_rows_all).to_csv(job_rows_csv_path, index=False)
            write_json(jobs_json_path, {"jobs": job_records})

            print(
                f"Completed {candidate.dataset} seed={seed}: "
                f"EV={row['ev_nrmse']:.3f}, Raw={row['raw_nrmse']:.3f}, Split={row['split_nrmse']:.3f}, "
                f"best_split_k={row['split_k']}",
                flush=True,
            )

    seed_df = pd.DataFrame(seed_rows)
    summary_df = summarize_seed_rows(seed_df)
    trace_wide_df, trace_long_df = build_trace_tables(seed_df, trace_store)
    seed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv_path, index=False)
    trace_wide_df.to_csv(trace_wide_csv_path, index=False)
    trace_long_df.to_csv(trace_long_csv_path, index=False)
    save_trace_bundle(traces_npz_path, trace_store)
    save_audit_bundle(audit_npz_path, audit_store)
    pd.DataFrame(job_rows_all).to_csv(job_rows_csv_path, index=False)
    write_json(
        jobs_json_path,
        {
            "backend": args.backend,
            "saved_accounts": saved_accounts,
            "seed_list": seed_list,
            "candidates": [asdict(candidate) for candidate in candidates],
            "jobs": job_records,
        },
    )

    if not args.no_plot:
        try:
            make_summary_plot(seed_df, figure_pdf_path, figure_png_path)
        except Exception as exc:  # pragma: no cover - plotting must not erase collected hardware results
            print(f"Plot generation failed after data collection: {exc}", flush=True)

    print("\n=== Hardware validation summary ===", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f"\nWrote: {seed_csv_path}", flush=True)
    print(f"Wrote: {summary_csv_path}", flush=True)
    print(f"Wrote: {trace_wide_csv_path}", flush=True)
    print(f"Wrote: {trace_long_csv_path}", flush=True)
    print(f"Wrote: {traces_npz_path}", flush=True)
    print(f"Wrote: {audit_npz_path}", flush=True)
    print(f"Wrote: {jobs_json_path}", flush=True)
    print(f"Wrote: {job_rows_csv_path}", flush=True)
    if not args.no_plot:
        print(f"Wrote: {figure_pdf_path}", flush=True)
        print(f"Wrote: {figure_png_path}", flush=True)


if __name__ == "__main__":
    main()
