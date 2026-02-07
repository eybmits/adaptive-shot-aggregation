"""NARMA-10: target rho_EV 2-5, the gap between 'G-Traj wins' and 'both good'."""
import numpy as np
from qrc_budget import run_single_experiment, compute_rho_ev

configs = [
    # rho ~ 2-5 range for NARMA-10
    (50000, 15, 0.0),
    (50000, 15, 0.01),
    (50000, 15, 0.03),
    (50000, 20, 0.0),
    (50000, 20, 0.01),
    (50000, 20, 0.03),
    (80000, 20, 0.0),
    (80000, 20, 0.01),
    (80000, 25, 0.0),
    (80000, 25, 0.01),
    (80000, 30, 0.0),
    (80000, 30, 0.01),
    (80000, 40, 0.0),
    (80000, 40, 0.01),
]
n_seeds = 3

print(f"{'S':>6s} {'N':>3s} {'σ':>5s} {'T':>5s} {'ρ':>5s}  "
      f"{'EV':>7s} {'G-Tr':>7s} {'Gap':>7s} "
      f"{'SR_EV':>6s} {'SR_GT':>6s}")
print("-"*70)

for S, n_shots, sigma in configs:
    T = S // n_shots
    rho = compute_rho_ev(S, n_shots)
    ev_all, gtraj_all, sr_ev, sr_gt = [], [], [], []
    for seed in range(n_seeds):
        r = run_single_experiment(n_shots, sigma, S, benchmark='narma10', seed=seed)
        ev_all.append(r['nrmse_ev'])
        gtraj_all.append(r['nrmse_gtraj'])
        yt = r['y_test']
        std_yt = max(np.std(yt), 1e-10)
        if len(r['y_pred_ev']) == len(yt):
            sr_ev.append(np.std(r['y_pred_ev']) / std_yt)
        yg = r.get('y_pred_gtraj', np.array([]))
        if len(yg) == len(yt):
            sr_gt.append(np.std(yg) / std_yt)

    ev_m, gt_m = np.mean(ev_all), np.mean(gtraj_all)
    gap = ev_m - gt_m
    sre = np.mean(sr_ev) if sr_ev else 0
    srg = np.mean(sr_gt) if sr_gt else 0
    tag = " ***" if gap > 0.02 and srg > 0.6 else ""
    print(f"{S:>6d} {n_shots:>3d} {sigma:>5.2f} {T:>5d} {rho:>5.2f}  "
          f"{ev_m:>7.3f} {gt_m:>7.3f} {gap:>+7.3f} "
          f"{sre:>6.2f} {srg:>6.2f}{tag}")
