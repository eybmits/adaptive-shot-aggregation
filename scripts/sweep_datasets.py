"""Find the S/N sweet spot per dataset where G-Traj wins AND SR > 0.75."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from qrc_budget import (
    QuantumReservoir, run_reservoir_pipeline, ev_training,
    grouped_trajectory_training, compute_rho_ev
)

def mackey_glass_tau30(n_steps, rng):
    from qrc_budget import mackey_glass
    return mackey_glass(n_steps, tau=30, rng=rng)

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
    series = (series - series.min()) / (series.max() - series.min() + 1e-10)
    return series

def sine_nonlinear(n_steps, rng):
    t = np.arange(n_steps)
    series = np.sin(t / 10.0)**2 + 0.3 * np.sin(t / 25.0)
    series += 0.02 * rng.standard_normal(n_steps)
    return series

def run_one(series_func, n_shots, sigma, S, seed=0):
    rng = np.random.default_rng(seed)
    T = S // n_shots
    if T < 60: return None
    series = series_func(T + 50, rng)
    inputs, targets = series[:-1][:T], series[1:][:T]
    res = QuantumReservoir(n_qubits=6, depth=5, rng=rng)
    sf = run_reservoir_pipeline(inputs, res, n_shots, sigma, rng=np.random.default_rng(seed+1000))
    ev, yp_ev, yt = ev_training(sf, targets, 30, 0.7)
    gt, yp_gt, _, k = grouped_trajectory_training(sf, targets, 30, 0.7)
    sre = np.std(yp_ev)/max(np.std(yt),1e-10) if len(yp_ev)==len(yt) else 0
    srg = np.std(yp_gt)/max(np.std(yt),1e-10) if len(yp_gt)==len(yt) else 0
    return {'ev':ev,'gt':gt,'sre':sre,'srg':srg,'k':k}

datasets = {'Lorenz-x': lorenz_x, 'Sine-NL': sine_nonlinear}

# Sweep: increase S to push SR up, keep N high enough for gap
params = [
    (30000, 40, 0.0),
    (40000, 40, 0.0),
    (50000, 40, 0.0),
    (30000, 30, 0.0),
    (40000, 30, 0.0),
    (50000, 30, 0.0),
    (40000, 50, 0.0),
    (50000, 50, 0.0),
]

for name, func in datasets.items():
    print(f"\n{name}:")
    print(f"{'S':>6s} {'N':>3s} {'ρ':>5s}  {'EV':>7s} {'G-Tr':>7s} {'Gap':>7s} {'SR_EV':>6s} {'SR_GT':>6s}")
    print("-"*55)
    for S, N, sigma in params:
        rho = compute_rho_ev(S, N)
        ev_a, gt_a, sre_a, srg_a = [], [], [], []
        for seed in range(3):
            r = run_one(func, N, sigma, S, seed)
            if r: ev_a.append(r['ev']); gt_a.append(r['gt']); sre_a.append(r['sre']); srg_a.append(r['srg'])
        if not ev_a: continue
        em, gm = np.mean(ev_a), np.mean(gt_a)
        gap = em - gm
        sre, srg = np.mean(sre_a), np.mean(srg_a)
        tag = " ***" if gap > 0.02 and srg > 0.75 else ""
        print(f"{S:>6d} {N:>3d} {rho:>5.2f}  {em:>7.3f} {gm:>7.3f} {gap:>+7.3f} {sre:>6.2f} {srg:>6.2f}{tag}")
