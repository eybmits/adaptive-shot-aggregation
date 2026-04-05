#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('split_base_impl', HERE / 'base_impl.py')
base = importlib.util.module_from_spec(spec)
sys.modules['split_base_impl'] = base
spec.loader.exec_module(base)

TITLE_MAP = {
    'mackey_glass': 'Mackey-Glass',
    'lorenz': 'Lorenz',
    'narma10': 'NARMA10',
}
TASK_ORDER = ['mackey_glass', 'lorenz', 'narma10']


@dataclass(frozen=True)
class Config:
    budget_total: int = 12000
    q_default: int = 4
    depth_default: int = 1
    entangler_default: str = 'ring_cnot'
    eta: float = 0.2
    window: int = 10
    t0: int = 30
    train_frac: float = 0.70
    val_frac_within_train: float = 0.25
    lambda_default: float = 10.0
    lambda_grid: Tuple[float, ...] = (
        1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0,
        3.0, 10.0, 30.0, 100.0, 300.0, 1000.0
    )
    gamma_grid: Tuple[float, ...] = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
    phase_nshots: Tuple[int, ...] = (10, 12, 15, 18, 20, 25, 30, 40, 50)
    confirmation_seeds: Tuple[int, ...] = tuple(range(100, 120))
    architecture_seeds: Tuple[int, ...] = tuple(range(200, 220))
    appendix_seeds: Tuple[int, ...] = tuple(range(300, 320))
    architecture_variants: Tuple[Tuple[str, int], ...] = (
        ('ring_cnot', 1),
        ('ring_cnot', 2),
        ('ring_cnot', 4),
        ('line_cnot', 1),
        ('all_to_all_cz', 1),
    )
    architecture_q12_topk: int = 2
    architecture_q12_q: int = 12
    architecture_q12_seeds: Tuple[int, ...] = tuple(range(300, 308))
    architecture_q12_datasets: Tuple[str, ...] = ('mackey_glass', 'narma10')
    reservoir_sizes: Tuple[int, ...] = (4, 6, 8, 10, 12)
    representative_n_shots: int = 18
    representative_sigma: float = 0.0

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
        common = {
            'budget_total': self.budget_total,
            'n_shots': self.representative_n_shots,
            'sigma': self.representative_sigma,
        }
        return {dataset: dict(common) for dataset in TASK_ORDER}


def aggregate_by_rho(seed_df: pd.DataFrame) -> pd.DataFrame:
    df = seed_df.copy()
    df['gap'] = df['ev_nrmse'] - df['split_nrmse']
    rows = []
    for (dataset, rho_ev), group in df.groupby(['dataset', 'rho_ev'], as_index=False):
        gap = group['gap'].to_numpy(dtype=float)
        win = (gap > 0.0).astype(float)
        rows.append({
            'dataset': dataset,
            'rho_ev': float(rho_ev),
            'n': int(len(group)),
            'gap_mean': float(gap.mean()),
            'gap_sd': float(gap.std(ddof=1)) if len(gap) > 1 else 0.0,
            'gap_ci95': float(1.96 * gap.std(ddof=1) / np.sqrt(len(gap))) if len(gap) > 1 else 0.0,
            'win_rate': float(win.mean()),
            'win_se': float(np.sqrt(win.mean() * (1.0 - win.mean()) / len(win))),
            'win_ci95': float(1.96 * np.sqrt(win.mean() * (1.0 - win.mean()) / len(win))),
        })
    return pd.DataFrame(rows).sort_values(['dataset', 'rho_ev']).reset_index(drop=True)


def _std(series: pd.Series) -> float:
    return float(series.std(ddof=1)) if len(series) > 1 else 0.0


def choose_representative_seed(seed_df: pd.DataFrame) -> int:
    median_split = float(seed_df['split_nrmse'].median())
    ranked = seed_df.assign(split_dist=(seed_df['split_nrmse'] - median_split).abs())
    ranked = ranked.sort_values(['split_dist', 'seed'])
    return int(ranked.iloc[0]['seed'])


def build_representative_summary(rep_rows: list[dict[str, object]], rep_seed_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    trace_lengths = {
        (str(row['dataset']), int(row['seed'])): int(len(row['trace_y_test']))
        for row in rep_rows
    }
    rows: list[dict[str, object]] = []
    for dataset in TASK_ORDER:
        sub = rep_seed_df[rep_seed_df['dataset'] == dataset].copy().sort_values('seed')
        representative_seed = choose_representative_seed(sub)
        rows.append(
            {
                'dataset': dataset,
                'q': int(sub['q'].mode().iloc[0]),
                'budget_total': int(sub['budget_total'].iloc[0]),
                'n_shots': int(sub['n_shots'].iloc[0]),
                'lambda_default': float(cfg.lambda_default),
                'rho_ev': float(sub['rho_ev'].iloc[0]),
                'test_length': trace_lengths[(dataset, representative_seed)],
                'representative_seed': representative_seed,
                'ev_nrmse_mean': float(sub['ev_nrmse'].mean()),
                'ev_nrmse_std': _std(sub['ev_nrmse']),
                'ev_sr_mean': float(sub['ev_sr'].mean()),
                'ev_sr_std': _std(sub['ev_sr']),
                'raw_nrmse_mean': float(sub['raw_nrmse'].mean()),
                'raw_nrmse_std': _std(sub['raw_nrmse']),
                'raw_sr_mean': float(sub['raw_sr'].mean()),
                'raw_sr_std': _std(sub['raw_sr']),
                'split_nrmse_mean': float(sub['split_nrmse'].mean()),
                'split_nrmse_std': _std(sub['split_nrmse']),
                'split_sr_mean': float(sub['split_sr'].mean()),
                'split_sr_std': _std(sub['split_sr']),
            }
        )
    return pd.DataFrame(rows)


def run_empirical_gap_suite(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    total = len(TASK_ORDER) * len(cfg.phase_nshots) * len(cfg.confirmation_seeds)
    count = 0
    for dataset in TASK_ORDER:
        for n_shots in cfg.phase_nshots:
            for seed in cfg.confirmation_seeds:
                out = base.run_main_protocols(dataset, cfg.budget_total, int(n_shots), 0.0, int(seed), cfg, traces=False)
                rows.append(
                    {
                        'dataset': dataset,
                        'seed': int(seed),
                        'n_shots': int(n_shots),
                        'rho_ev': float(out['rho_ev']),
                        'ev_nrmse': float(out['ev_nrmse']),
                        'split_nrmse': float(out['split_nrmse']),
                        'gap': float(out['ev_nrmse'] - out['split_nrmse']),
                    }
                )
                count += 1
                if count % 30 == 0:
                    print(f"empirical gap progress {count}/{total}", flush=True)
    seed_df = pd.DataFrame(rows).sort_values(['dataset', 'rho_ev', 'seed']).reset_index(drop=True)
    summary = seed_df.groupby(['dataset', 'rho_ev', 'n_shots'], as_index=False).agg(
        gap_mean=('gap', 'mean'),
        gap_std=('gap', lambda x: float(np.std(x, ddof=1))),
        ev_nrmse_mean=('ev_nrmse', 'mean'),
        split_nrmse_mean=('split_nrmse', 'mean'),
    )
    summary['gap_ci95'] = 1.96 * summary['gap_std'] / np.sqrt(len(cfg.confirmation_seeds))
    return seed_df, summary.sort_values(['dataset', 'rho_ev']).reset_index(drop=True)


ARCHITECTURE_ORDER = {
    ('ring_cnot', 1): 0,
    ('ring_cnot', 2): 1,
    ('ring_cnot', 4): 2,
    ('line_cnot', 1): 3,
    ('all_to_all_cz', 1): 4,
}


def run_architecture_q12_targeted_suite(cfg: Config, arch_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    selected_rows = []
    for dataset in cfg.architecture_q12_datasets:
        sub = arch_summary[arch_summary['dataset'] == dataset].copy()
        sub['variant_order'] = [
            ARCHITECTURE_ORDER[(str(entangler), int(depth))]
            for entangler, depth in zip(sub['entangler'], sub['depth'])
        ]
        ranked = sub.sort_values(
            ['gap_ev_minus_split', 'split_nrmse_mean', 'depth', 'variant_order'],
            ascending=[False, True, True, True],
        ).head(cfg.architecture_q12_topk)
        selected_rows.extend(ranked.to_dict(orient='records'))
        for _, row in ranked.iterrows():
            entangler = str(row['entangler'])
            depth = int(row['depth'])
            for seed in cfg.architecture_q12_seeds:
                out = base.run_main_protocols(
                    dataset,
                    int(cfg.budget_total),
                    int(cfg.representative_n_shots),
                    float(cfg.representative_sigma),
                    int(seed),
                    cfg,
                    q=int(cfg.architecture_q12_q),
                    depth=depth,
                    entangler=entangler,
                    traces=False,
                )
                rows.append(out)
    seed_df = pd.DataFrame(rows).sort_values(['dataset', 'entangler', 'depth', 'seed']).reset_index(drop=True)
    summary = seed_df.groupby(['dataset', 'q', 'entangler', 'depth', 'n_shots', 'sigma'], as_index=False).agg(
        seeds=('seed', 'nunique'),
        ev_nrmse_mean=('ev_nrmse', 'mean'),
        ev_nrmse_std=('ev_nrmse', lambda x: float(np.std(x, ddof=1))),
        raw_nrmse_mean=('raw_nrmse', 'mean'),
        raw_nrmse_std=('raw_nrmse', lambda x: float(np.std(x, ddof=1))),
        split_nrmse_mean=('split_nrmse', 'mean'),
        split_nrmse_std=('split_nrmse', lambda x: float(np.std(x, ddof=1))),
        rho_ev=('rho_ev', 'mean'),
    )
    summary['gap_ev_minus_split'] = summary['ev_nrmse_mean'] - summary['split_nrmse_mean']
    win_rate = (
        seed_df.assign(split_beats_ev=(seed_df['split_nrmse'] < seed_df['ev_nrmse']).astype(float))
        .groupby(['dataset', 'q', 'entangler', 'depth', 'n_shots', 'sigma'], as_index=False)
        .agg(split_win_rate=('split_beats_ev', 'mean'))
    )
    summary = summary.merge(win_rate, on=['dataset', 'q', 'entangler', 'depth', 'n_shots', 'sigma'], how='left')
    selected_df = pd.DataFrame(selected_rows)[['dataset', 'entangler', 'depth', 'gap_ev_minus_split', 'split_nrmse_mean']]
    summary = summary.merge(
        selected_df.rename(
            columns={
                'gap_ev_minus_split': 'source_q4_gap_ev_minus_split',
                'split_nrmse_mean': 'source_q4_split_nrmse_mean',
            }
        ),
        on=['dataset', 'entangler', 'depth'],
        how='left',
    )
    return seed_df, summary.sort_values(['dataset', 'entangler', 'depth']).reset_index(drop=True)


def build_more_qubits_outputs(size_seed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_df = (
        size_seed.loc[:, ['dataset', 'q', 'seed', 'rho_ev', 'ev_nrmse', 'split_nrmse', 'split_k']]
        .copy()
        .sort_values(['dataset', 'q', 'seed'])
        .reset_index(drop=True)
    )
    seed_df['gap'] = seed_df['ev_nrmse'] - seed_df['split_nrmse']
    summary = (
        seed_df.groupby(['dataset', 'q'], as_index=False)
        .agg(
            ev_mean=('ev_nrmse', 'mean'),
            split_mean=('split_nrmse', 'mean'),
            gap_mean=('gap', 'mean'),
            gap_std=('gap', lambda x: float(np.std(x, ddof=1))),
            split_k_mode=('split_k', lambda x: int(pd.Series(x).mode().iloc[0])),
            rho_ev=('rho_ev', 'mean'),
        )
        .sort_values(['dataset', 'q'])
        .reset_index(drop=True)
    )
    return seed_df, summary


def main() -> None:
    repo = HERE.parent
    results_dir = repo / 'results'
    results_dir.mkdir(exist_ok=True)
    cfg = Config()
    print('Running 20-seed corrected synthetic suite...', flush=True)
    print('Representative configs:', cfg.representative_configs, flush=True)

    print('empirical gap suite', flush=True)
    emp_seed, emp_summary = run_empirical_gap_suite(cfg)
    print('representative suite', flush=True)
    rep_rows, rep_seed, rep_summary = base.run_representative_suite(cfg)
    rep_seed = rep_seed.drop(columns=['sigma'], errors='ignore')
    rep_summary = build_representative_summary(rep_rows, rep_seed, cfg)
    print('baseline suite', flush=True)
    baseline_seed, baseline_summary = base.run_baseline_suite(cfg)
    print('lambda suite', flush=True)
    lambda_seed, lambda_summary = base.run_lambda_suite(cfg)
    print('architecture suite', flush=True)
    arch_seed, arch_summary = base.run_architecture_suite(cfg)
    arch_seed = arch_seed.copy()
    arch_summary = arch_summary.copy()
    arch_seed['q'] = int(cfg.q_default)
    arch_summary['q'] = int(cfg.q_default)
    print('targeted q12 architecture suite', flush=True)
    arch_q12_seed, arch_q12_summary = run_architecture_q12_targeted_suite(cfg, arch_summary)
    print('duplication suite', flush=True)
    dup_seed, dup_summary = base.run_duplication_suite(cfg)
    print('reservoir-size suite', flush=True)
    size_seed, size_summary = base.run_size_suite(cfg)

    # Save all results
    emp_seed.to_csv(results_dir / 'empirical_gap_seed_level.csv', index=False)
    emp_summary.to_csv(results_dir / 'empirical_gap_summary.csv', index=False)

    rep_seed.to_csv(results_dir / 'representative_seed_level.csv', index=False)
    rep_summary.drop(columns=['sigma'], errors='ignore').to_csv(results_dir / 'representative_summary.csv', index=False)
    baseline_seed.drop(columns=['sigma'], errors='ignore').to_csv(results_dir / 'baseline_controls_seed_level.csv', index=False)
    baseline_summary.drop(columns=['sigma'], errors='ignore').to_csv(results_dir / 'baseline_controls_summary.csv', index=False)
    lambda_seed.to_csv(results_dir / 'lambda_seed_level.csv', index=False)
    lambda_summary.to_csv(results_dir / 'lambda_summary.csv', index=False)
    arch_seed.to_csv(results_dir / 'architecture_ablation_seed_level.csv', index=False)
    arch_summary.to_csv(results_dir / 'architecture_ablation_summary.csv', index=False)
    arch_q12_seed.to_csv(results_dir / 'architecture_ablation_q12_seed_level.csv', index=False)
    arch_q12_summary.to_csv(results_dir / 'architecture_ablation_q12_summary.csv', index=False)
    dup_seed.to_csv(results_dir / 'duplication_control_seed_level.csv', index=False)
    dup_summary.to_csv(results_dir / 'duplication_control_summary.csv', index=False)
    more_qubits_seed, more_qubits_summary = build_more_qubits_outputs(size_seed)
    more_qubits_seed.to_csv(results_dir / 'more_qubits_seed_level.csv', index=False)
    more_qubits_summary.to_csv(results_dir / 'more_qubits_summary.csv', index=False)
    size_seed.to_csv(results_dir / 'reservoir_size_seed_level.csv', index=False)
    size_summary.to_csv(results_dir / 'reservoir_size_summary.csv', index=False)

    np.savez_compressed(
        results_dir / 'representative_traces.npz',
        **{f'{dataset}_y_test': np.stack([r['trace_y_test'] for r in rep_rows if r['dataset'] == dataset], axis=0) for dataset in cfg.representative_configs.keys()},
        **{f'{dataset}_{protocol}': np.stack([r[f'trace_{protocol}'] for r in rep_rows if r['dataset'] == dataset], axis=0) for dataset in cfg.representative_configs.keys() for protocol in ['ev', 'raw', 'split']},
    )

    config_json = {
        'budget_total': cfg.budget_total,
        'q_default': cfg.q_default,
        'depth_default': cfg.depth_default,
        'entangler_default': cfg.entangler_default,
        'eta': cfg.eta,
        'window': cfg.window,
        't0': cfg.t0,
        'train_frac': cfg.train_frac,
        'val_frac_within_train': cfg.val_frac_within_train,
        'lambda_default': cfg.lambda_default,
        'lambda_grid': list(cfg.lambda_grid),
        'gamma_grid': list(cfg.gamma_grid),
        'phase_nshots': list(cfg.phase_nshots),
        'confirmation_seeds': list(cfg.confirmation_seeds),
        'architecture_seeds': list(cfg.architecture_seeds),
        'architecture_q12_topk': cfg.architecture_q12_topk,
        'architecture_q12_q': cfg.architecture_q12_q,
        'architecture_q12_seeds': list(cfg.architecture_q12_seeds),
        'architecture_q12_datasets': list(cfg.architecture_q12_datasets),
        'appendix_seeds': list(cfg.appendix_seeds),
        'architecture_variants': [list(v) for v in cfg.architecture_variants],
        'reservoir_sizes': list(cfg.reservoir_sizes),
        'representative_configs': cfg.representative_configs,
    }
    (repo / 'config.json').write_text(json.dumps(config_json, indent=2), encoding='utf-8')
    print('Done. Results saved to', results_dir, flush=True)


if __name__ == '__main__':
    main()
