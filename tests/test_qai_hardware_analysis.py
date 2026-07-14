from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_qai_hardware as hardware


def _write_valid_task(directory: Path, task: str = "santa_fe_laser") -> None:
    inputs = np.linspace(0.0, 1.0, 500)
    targets = np.roll(inputs, -1)
    angles = np.array([hardware.base.input_amplitude_angle(value) for value in inputs])
    np.savez_compressed(directory / f"raw_{task}.npz",
                        bits=np.zeros((500, 400, 10), dtype=np.uint8))
    np.savez_compressed(directory / f"inputs_{task}.npz",
                        inputs=inputs, targets=targets, angles=angles)


def test_hardware_task_order_replaces_lorenz_with_santa_fe() -> None:
    assert hardware.TASKS == (
        "mackey_glass", "santa_fe_laser", "rossler", "sunspots"
    )


def test_santa_fe_archive_uses_exact_legacy_hardware_grain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_valid_task(tmp_path)
    monkeypatch.setattr(hardware, "HW_DIR", tmp_path)

    bits, targets = hardware.load_task_archives("santa_fe_laser")

    assert bits.shape == (500, 400, 10)
    assert bits.dtype == np.uint8
    assert targets.shape == (500,)


def test_partial_santa_fe_archive_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_valid_task(tmp_path)
    np.savez_compressed(
        tmp_path / "raw_santa_fe_laser.npz",
        bits=np.zeros((499, 400, 10), dtype=np.uint8),
    )
    monkeypatch.setattr(hardware, "HW_DIR", tmp_path)

    with pytest.raises(ValueError, match="expected \\(500, 400, 10\\)"):
        hardware.load_task_archives("santa_fe_laser")


def _complete_analysis_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_rows, spectrum_rows, ablation_rows = [], [], []
    for task in hardware.TASKS:
        for record in range(20):
            for protocol in ("ev", "raw", "split", "warm_start"):
                seed_rows.append({"dataset": task, "record": record, "protocol": protocol})
            for k in (1, 2, 4, 5, 10, 20):
                spectrum_rows.append({
                    "dataset": task, "record": record, "grouping": "contiguous", "k": k
                })
            for k in (2, 4, 5, 10):
                spectrum_rows.append({
                    "dataset": task, "record": record, "grouping": "interleaved", "k": k
                })
                ablation_rows.append({"dataset": task, "record": record, "k": k})
    return (pd.DataFrame(seed_rows), pd.DataFrame(spectrum_rows),
            pd.DataFrame(ablation_rows))


def test_analysis_tables_require_all_four_tasks_and_all_shot_blocks() -> None:
    seed, spectrum, ablation = _complete_analysis_tables()
    hardware.validate_analysis_tables(seed, spectrum, ablation)

    incomplete = seed[seed["dataset"] != "santa_fe_laser"]
    with pytest.raises(ValueError, match="expected"):
        hardware.validate_analysis_tables(incomplete, spectrum, ablation)
