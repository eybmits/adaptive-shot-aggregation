"""Find best G-Traj configs that clearly beat EV with good signal quality."""
import numpy as np
from qrc_budget import run_single_experiment, compute_rho_ev

configs = [
    # S, Nshots, sigma — rho_EV ~1-2 sweet spot, varying noise
    (20000, 40, 0.00),
    (20000, 40, 0.01),
    (20000, 40, 0.03),
    (20000, 50, 0.00),
    (20000, 50, 0.01),
    (20000, 60, 0.00),
    (20000, 60, 0.01),
    (30000, 50, 0.00),
    (30000, 50, 0.01),
    (30000, 60, 0.00),
    (30000, 60, 0.01),
]
n_seeds = 3

print(f"{'S':>6s} {'N':>3s} {'σ':>5s} {'T':>5s} {'ρ':>5s}  "
      f"{'EV':>7s} {'G-Tr':>7s} {'Gap':>7s} "
      f"{'SR_EV':>6s} {'SR_GT':>6s}")
print("-"*70)

for S, n_shots, sigma in configs:
    T = S // n_shots
    rho = compute_rho_ev(S, n_shots)
    if T < 60:
        continue
    ev_all, gtraj_all, sr_ev, sr_gtraj = [], [], [], []
    for seed in range(n_seeds):
        r = run_single_experiment(n_shots, sigma, S, seed=seed)
        ev_all.append(r['nrmse_ev'])
        gtraj_all.append(r['nrmse_gtraj'])
        yt = r['y_test']
        std_yt = max(np.std(yt), 1e-10)
        if len(r['y_pred_ev']) == len(yt):
            sr_ev.append(np.std(r['y_pred_ev']) / std_yt)
        yg = r.get('y_pred_gtraj', np.array([]))
        if len(yg) == len(yt):
            sr_gtraj.append(np.std(yg) / std_yt)

    ev_m, gt_m = np.mean(ev_all), np.mean(gtraj_all)
    gap = ev_m - gt_m
    tag = " ***" if gap > 0.05 and gt_m < 0.5 else ""
    print(f"{S:>6d} {n_shots:>3d} {sigma:>5.2f} {T:>5d} {rho:>5.2f}  "
          f"{ev_m:>7.3f} {gt_m:>7.3f} {gap:>+7.3f} "
          f"{np.mean(sr_ev):>6.2f} {np.mean(sr_gtraj):>6.2f}{tag}")
