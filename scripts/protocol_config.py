#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


TASK_ORDER: Tuple[str, ...] = ("mackey_glass", "lorenz", "rossler", "sunspots")
RESERVOIR_FAMILIES: Tuple[str, ...] = ("canonical_gate_qrc", "legacy_random_circuit", "canonical_spin")
RESERVOIR_ALIASES: Dict[str, str] = {
    "canonical_circuit": "canonical_gate_qrc",
    "gate_circuit": "canonical_gate_qrc",
    "standard_qrc": "canonical_gate_qrc",
    "spin_ring": "canonical_spin",
}


def _tuple(values: Iterable[Any], cast: type = float) -> Tuple[Any, ...]:
    return tuple(cast(value) for value in values)


def _tuple_pairs(values: Iterable[Iterable[Any]]) -> Tuple[Tuple[str, int], ...]:
    return tuple((str(name), int(depth)) for name, depth in values)


@dataclass(frozen=True)
class ProtocolConfig:
    budget_total: int = 12000
    q_default: int = 4
    depth_default: int = 1
    entangler_default: str = "ring_rzz"
    eta: float = 0.2
    window: int = 10
    t0: int = 30
    train_frac: float = 0.70
    val_frac_within_train: float = 0.25
    lambda_default: float = 10.0
    lambda_grid: Tuple[float, ...] = (
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
        300.0,
        1000.0,
    )
    gamma_grid: Tuple[float, ...] = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
    phase_nshots: Tuple[int, ...] = (10, 12, 15, 18, 20, 25, 30, 40, 50)
    phase_sigmas: Tuple[float, ...] = (0.0,)
    confirmation_seeds: Tuple[int, ...] = tuple(range(100, 120))
    architecture_seeds: Tuple[int, ...] = tuple(range(200, 220))
    architecture_q12_topk: int = 2
    architecture_q12_q: int = 12
    architecture_q12_seeds: Tuple[int, ...] = tuple(range(300, 308))
    architecture_q12_datasets: Tuple[str, ...] = ("mackey_glass", "narma10")
    appendix_seeds: Tuple[int, ...] = tuple(range(300, 320))
    architecture_variants: Tuple[Tuple[str, int], ...] = (
        ("ring_rzz", 1),
        ("ring_rzz", 2),
        ("ring_rzz", 4),
        ("line_rzz", 1),
        ("all_to_all_rzz", 1),
    )
    reservoir_sizes: Tuple[int, ...] = (4, 6, 8, 10, 12)
    representative_n_shots: int = 18
    representative_sigma: float = 0.0
    comparison_datasets: Tuple[str, ...] = TASK_ORDER
    comparison_reservoir_families: Tuple[str, ...] = ("canonical_gate_qrc",)
    comparison_q_values: Tuple[int, ...] = (2, 4, 6, 8)
    comparison_time_steps: Tuple[int, ...] = (60, 120, 200, 300, 400, 600)
    comparison_seeds: Tuple[int, ...] = tuple(range(100, 120))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def m(self) -> int:
        return self.q_default

    @property
    def d_default(self) -> int:
        return self.window * self.q_default

    def d(self, q: int | None = None) -> int:
        qq = self.q_default if q is None else int(q)
        return self.window * qq

    def budget_factor(self, reservoir_family: str | None = None) -> int:
        family = RESERVOIR_ALIASES.get(reservoir_family or "canonical_gate_qrc", reservoir_family or "canonical_gate_qrc")
        return 2 if family in {"legacy_random_circuit", "canonical_spin"} else 1

    def t_steps(self, n_shots: int, budget_total: int | None = None) -> int:
        budget = self.budget_total if budget_total is None else int(budget_total)
        return budget // int(n_shots)

    def shots_per_time_step(
        self,
        time_steps: int,
        budget_total: int | None = None,
        reservoir_family: str | None = None,
    ) -> int:
        budget = self.budget_total if budget_total is None else int(budget_total)
        shots = budget // (self.budget_factor(reservoir_family) * int(time_steps))
        if shots < 2:
            raise ValueError(f"time_steps={time_steps} leaves fewer than two shots per measurement setting")
        return int(shots)

    def budget_used(
        self,
        time_steps: int,
        shots_per_time_step: int,
        reservoir_family: str | None = None,
    ) -> int:
        return self.budget_factor(reservoir_family) * int(time_steps) * int(shots_per_time_step)

    def rho_ev(self, t_steps: int, q: int | None = None) -> float:
        return self.train_frac * max(int(t_steps) - self.t0, 0) / self.d(q=q)

    @property
    def representative_configs(self) -> Dict[str, Dict[str, float]]:
        common = {
            "budget_total": self.budget_total,
            "n_shots": self.representative_n_shots,
            "sigma": self.representative_sigma,
        }
        return {dataset: dict(common) for dataset in TASK_ORDER}

    def manifest_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, tuple):
                data[key] = list(value)
        data["protocol_question"] = (
            "For each dataset and reservoir family, does Split improve test "
            "NRMSE under a matched time-step grid and the same shot-budget cap?"
        )
        data["main_reservoir"] = {
            "name": "canonical_gate_qrc",
            "description": (
                "IBM-compatible gate-model QRC: amplitude input encoding on one qubit, "
                "fixed Trotterized transverse-field-Ising circuit block, local-Z readout, "
                "and the same EV/Raw/Split readout protocol."
            ),
        }
        data["benefit_rule"] = {
            "delta": "mean(ev_nrmse - split_nrmse)",
            "beneficial": "delta > 0",
            "majority_win": "split_nrmse < ev_nrmse on more than half of matched seeds",
            "strong_cell": "paired 95% CI for delta is above zero",
        }
        return data


def _coerce_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(raw)
    tuple_float_keys = ("lambda_grid", "gamma_grid", "phase_sigmas")
    tuple_int_keys = (
        "phase_nshots",
        "confirmation_seeds",
        "architecture_seeds",
        "architecture_q12_seeds",
        "appendix_seeds",
        "reservoir_sizes",
        "comparison_q_values",
        "comparison_time_steps",
        "comparison_seeds",
    )
    tuple_str_keys = (
        "architecture_q12_datasets",
        "comparison_datasets",
        "comparison_reservoir_families",
    )
    if "region_datasets" in data and "comparison_datasets" not in data:
        data["comparison_datasets"] = data.pop("region_datasets")
    if "region_reservoir_families" in data and "comparison_reservoir_families" not in data:
        data["comparison_reservoir_families"] = data.pop("region_reservoir_families")
    if "comparison_reservoir_families" in data:
        data["comparison_reservoir_families"] = [
            RESERVOIR_ALIASES.get(str(value), str(value))
            for value in data["comparison_reservoir_families"]
        ]
    if "region_q_values" in data and "comparison_q_values" not in data:
        data["comparison_q_values"] = data.pop("region_q_values")
    if "region_seeds" in data and "comparison_seeds" not in data:
        data["comparison_seeds"] = data.pop("region_seeds")
    if "region_n_shots" in data and "comparison_time_steps" not in data:
        budget = int(data.get("budget_total", 12000))
        data["comparison_time_steps"] = [budget // int(shots) for shots in data.pop("region_n_shots")]
    for key in tuple_float_keys:
        if key in data:
            data[key] = _tuple(data[key], float)
    for key in tuple_int_keys:
        if key in data:
            data[key] = _tuple(data[key], int)
    for key in tuple_str_keys:
        if key in data:
            data[key] = _tuple(data[key], str)
    if "architecture_variants" in data:
        data["architecture_variants"] = _tuple_pairs(data["architecture_variants"])
    if "representative_configs" in data:
        data.pop("representative_configs")
    known = set(ProtocolConfig.__dataclass_fields__)
    metadata = {key: data.pop(key) for key in list(data.keys()) if key not in known}
    if metadata:
        data["metadata"] = metadata
    return data


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config.json"


def load_protocol_config(path: str | Path | None = None) -> ProtocolConfig:
    cfg_path = default_config_path() if path is None else Path(path)
    if not cfg_path.exists():
        return ProtocolConfig()
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    return ProtocolConfig(**_coerce_config(raw))


def write_manifest(path: str | Path, cfg: ProtocolConfig, extra: Dict[str, Any] | None = None) -> None:
    manifest = cfg.manifest_dict()
    if extra:
        manifest.update(extra)
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
