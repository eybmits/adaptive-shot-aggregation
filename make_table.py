"""Generate paper results tables from phase sweep data."""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from qrc_budget import (
    QuantumReservoir, run_reservoir_pipeline, ev_training,
    trajectory_training, grouped_trajectory_training, mackey_glass,
    compute_rho_ev, OUT_DIR
)

def lorenz_x(n_steps, rng, dt=0.02, warmup=2000):
    sigma_l, rho_l, beta = 10.0, 28.0, 8.0/3.0
    total = n_steps + warmup
    x, y, z = 1.0, 1.0, 1.0
    xs = np.zeros(total)
    for i in range(total):
        xs[i] = x
        dx = sigma_l * (y - x) * dt
        dy = (x * (rho_l - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x += dx; y += dy; z += dz
    series = xs[warmup:]
    return (series - series.min()) / (series.max() - series.min() + 1e-10)

def mg_series(n_steps, rng):
    return mackey_glass(n_steps, tau=17, rng=rng)

def run_full(series_func, n_shots, sigma, S, n_seeds=8):
    rng_seeds = range(n_seeds)
    evs, trajs, gtrajs, srs_ev, srs_gt = [], [], [], [], []
    for seed in rng_seeds:
        rng = np.random.default_rng(seed)
        T = S // n_shots
        if T < 60: continue
        series = series_func(T + 50, rng)
        inputs, targets = series[:-1][:T], series[1:][:T]
        res = QuantumReservoir(n_qubits=6, depth=5, rng=rng)
        sf = run_reservoir_pipeline(inputs, res, n_shots, sigma,
                                     rng=np.random.default_rng(seed + 1000))
        ev, yp_ev, yt = ev_training(sf, targets, 30, 0.7)
        tr, yp_tr, _ = trajectory_training(sf, targets, 30, 0.7)
        gt, yp_gt, _, k = grouped_trajectory_training(sf, targets, 30, 0.7)
        evs.append(ev); trajs.append(tr); gtrajs.append(gt)
        std_yt = max(np.std(yt), 1e-10)
        if len(yp_ev) == len(yt): srs_ev.append(np.std(yp_ev) / std_yt)
        if len(yp_gt) == len(yt): srs_gt.append(np.std(yp_gt) / std_yt)
    if not evs: return None
    return {
        'ev': np.mean(evs), 'ev_std': np.std(evs),
        'traj': np.mean(trajs), 'traj_std': np.std(trajs),
        'gtraj': np.mean(gtrajs), 'gtraj_std': np.std(gtrajs),
        'sr_ev': np.mean(srs_ev) if srs_ev else 0,
        'sr_gt': np.mean(srs_gt) if srs_gt else 0,
        'gap': np.mean(evs) - np.mean(gtrajs),
        'rho': compute_rho_ev(S, n_shots),
        'n_seeds': len(evs),
    }

# --- Configs ---
configs = [
    {'name': 'Mackey-Glass', 'func': mg_series, 'S': 20000, 'N': 40, 'sigma': 0.03},
    {'name': 'Lorenz', 'func': lorenz_x, 'S': 30000, 'N': 40, 'sigma': 0.0},
]

n_seeds = 8
results_all = []

for cfg in configs:
    print(f"\n=== {cfg['name']} (S={cfg['S']}, N={cfg['N']}, σ={cfg['sigma']}) ===")
    r = run_full(cfg['func'], cfg['N'], cfg['sigma'], cfg['S'], n_seeds)
    if r:
        r['name'] = cfg['name']
        r['S'] = cfg['S']
        r['N'] = cfg['N']
        r['sigma'] = cfg['sigma']
        results_all.append(r)
        print(f"  EV     = {r['ev']:.3f} ± {r['ev_std']:.3f}  (SR={r['sr_ev']:.2f})")
        print(f"  Traj   = {r['traj']:.3f} ± {r['traj_std']:.3f}")
        print(f"  G-Traj = {r['gtraj']:.3f} ± {r['gtraj_std']:.3f}  (SR={r['sr_gt']:.2f})")
        print(f"  Gap    = {r['gap']:+.3f}")
        print(f"  ρ_EV   = {r['rho']:.2f}")

# --- Markdown table ---
md = []
md.append("# Paper Results\n")
md.append("## Table 1: Benchmark comparison (8 seeds)\n")
md.append("| Dataset | S | N | σ | ρ_EV | EV NRMSE | Traj NRMSE | G-Traj NRMSE | Gap | SR(EV) | SR(G-Traj) |")
md.append("|---------|---|---|---|------|----------|------------|--------------|-----|--------|------------|")
for r in results_all:
    md.append(f"| {r['name']} | {r['S']} | {r['N']} | {r['sigma']} | {r['rho']:.2f} | "
              f"{r['ev']:.3f}±{r['ev_std']:.3f} | {r['traj']:.3f}±{r['traj_std']:.3f} | "
              f"{r['gtraj']:.3f}±{r['gtraj_std']:.3f} | {r['gap']:+.3f} | "
              f"{r['sr_ev']:.2f} | {r['sr_gt']:.2f} |")

# --- Extended sweep table ---
md.append("\n## Table 2: Phase sweep summary (selected configs)\n")
md.append("| Dataset | N | σ | ρ_EV | EV | G-Traj | Gap | SR(GT) |")
md.append("|---------|---|---|------|----|--------|-----|--------|")

sweep_configs = [
    ('Mackey-Glass', mg_series, 20000, [(20, 0.0), (30, 0.03), (40, 0.03), (50, 0.05)]),
    ('Lorenz', lorenz_x, 30000, [(20, 0.0), (30, 0.0), (40, 0.0), (50, 0.0)]),
]

for ds_name, func, S, nshots_sigmas in sweep_configs:
    for n_shots, sigma in nshots_sigmas:
        print(f"\nSweep: {ds_name} N={n_shots} σ={sigma}")
        r = run_full(func, n_shots, sigma, S, n_seeds=5)
        if r:
            md.append(f"| {ds_name} | {n_shots} | {sigma} | {r['rho']:.2f} | "
                      f"{r['ev']:.3f}±{r['ev_std']:.3f} | "
                      f"{r['gtraj']:.3f}±{r['gtraj_std']:.3f} | "
                      f"{r['gap']:+.3f} | {r['sr_gt']:.2f} |")

# --- LaTeX table ---
md.append("\n## LaTeX Table 1\n")
md.append("```latex")
md.append("\\begin{table}[t]")
md.append("\\centering")
md.append("\\caption{Benchmark results (8 seeds). Gap = NRMSE\\textsubscript{EV} $-$ NRMSE\\textsubscript{G-Traj}.}")
md.append("\\label{tab:benchmark}")
md.append("\\begin{tabular}{lcccccc}")
md.append("\\toprule")
md.append("Dataset & $S$ & $N$ & $\\sigma$ & $\\rho_{\\text{EV}}$ & EV & G-Traj & Gap \\\\")
md.append("\\midrule")
for r in results_all:
    md.append(f"{r['name']} & {r['S']//1000}k & {r['N']} & {r['sigma']} & {r['rho']:.2f} & "
              f"${r['ev']:.3f} \\pm {r['ev_std']:.3f}$ & "
              f"${r['gtraj']:.3f} \\pm {r['gtraj_std']:.3f}$ & "
              f"${r['gap']:+.3f}$ \\\\")
md.append("\\bottomrule")
md.append("\\end{tabular}")
md.append("\\end{table}")
md.append("```\n")

out_text = "\n".join(md)
out_path = os.path.join(OUT_DIR, 'paper_results.md')
with open(out_path, 'w') as f:
    f.write(out_text)
print(f"\n\nResults table saved to {out_path}")
print(out_text)
