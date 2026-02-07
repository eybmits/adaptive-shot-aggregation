"""Sweep S and Nshots to find configs where Traj/G-Traj excel."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from qrc_budget import (
    run_single_experiment, compute_rho_ev, setup_nature_style,
    COLOR_EV, COLOR_TRAJ, COLOR_GTRAJ, COLOR_TRUE, OUT_DIR
)

# Theory: rho_EV = (alpha/d) * (S/Nshots - t0)
# Sweet spot for trajectory: rho_EV ~ 1
# d=241, t0=30, alpha=0.7 => S/Nshots - 30 ~ 241/0.7 ~ 344 => S/Nshots ~ 374
# So for Nshots=20 => S ~ 7480;  Nshots=30 => S ~ 11220
# Larger S = more time steps = lower NRMSE overall

S_list = [20000, 50000, 100000]
nshots_list = [10, 15, 20, 25, 30, 40]
sigma = 0.01
n_seeds = 3  # quick scan

results = []

total = len(S_list) * len(nshots_list)
idx = 0
for S in S_list:
    for n_shots in nshots_list:
        idx += 1
        T = S // n_shots
        rho = compute_rho_ev(S, n_shots)
        if T < 60:
            continue
        ev_scores, traj_scores, gtraj_scores = [], [], []
        for seed in range(n_seeds):
            r = run_single_experiment(
                n_shots, sigma, S, benchmark='mackey_glass',
                seed=seed, eta=0.5, L=20, t0=30, alpha=0.7, depth=5
            )
            ev_scores.append(r['nrmse_ev'])
            traj_scores.append(r['nrmse_traj'])
            gtraj_scores.append(r['nrmse_gtraj'])

        ev_m = np.mean(ev_scores)
        traj_m = np.mean(traj_scores)
        gtraj_m = np.mean(gtraj_scores)
        gap = ev_m - gtraj_m

        results.append({
            'S': S, 'n_shots': n_shots, 'T': T, 'rho_ev': rho,
            'ev': ev_m, 'traj': traj_m, 'gtraj': gtraj_m, 'gap': gap
        })
        print(f"[{idx}/{total}] S={S:>6d} Nshots={n_shots:>2d} T={T:>5d} "
              f"rho={rho:.2f}  EV={ev_m:.3f}  Traj={traj_m:.3f}  "
              f"G-Traj={gtraj_m:.3f}  gap={gap:+.3f}")

# Sort by best G-Traj NRMSE
results.sort(key=lambda r: r['gtraj'])

print("\n" + "="*80)
print("TOP 10 configs by G-Traj NRMSE:")
print("="*80)
print(f"{'S':>7s} {'Nshots':>6s} {'T':>5s} {'rho_EV':>7s} "
      f"{'EV':>7s} {'Traj':>7s} {'G-Traj':>7s} {'Gap':>7s}")
print("-"*60)
for r in results[:10]:
    print(f"{r['S']:>7d} {r['n_shots']:>6d} {r['T']:>5d} {r['rho_ev']:>7.2f} "
          f"{r['ev']:>7.3f} {r['traj']:>7.3f} {r['gtraj']:>7.3f} {r['gap']:>+7.3f}")

print("\nTOP 10 configs by largest gap (EV - G-Traj):")
print("-"*60)
results_gap = sorted(results, key=lambda r: -r['gap'])
for r in results_gap[:10]:
    print(f"{r['S']:>7d} {r['n_shots']:>6d} {r['T']:>5d} {r['rho_ev']:>7.2f} "
          f"{r['ev']:>7.3f} {r['traj']:>7.3f} {r['gtraj']:>7.3f} {r['gap']:>+7.3f}")
