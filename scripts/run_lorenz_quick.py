"""Lorenz attractor validation: G-Traj vs EV with good signal quality."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from qrc_budget import (
    QuantumReservoir, run_reservoir_pipeline, ev_training,
    trajectory_training, grouped_trajectory_training,
    setup_nature_style, COLOR_EV, COLOR_TRAJ, COLOR_GTRAJ, COLOR_TRUE, OUT_DIR
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
    series = (series - series.min()) / (series.max() - series.min() + 1e-10)
    return series

S = 30000
n_shots = 40
sigma = 0.0
n_seeds = 8

setup_nature_style()

all_results = []
for seed in range(n_seeds):
    print(f"Lorenz: seed {seed+1}/{n_seeds}")
    rng = np.random.default_rng(seed)
    T = S // n_shots
    series = lorenz_x(T + 50, rng)
    inputs, targets = series[:-1][:T], series[1:][:T]
    reservoir = QuantumReservoir(n_qubits=6, depth=5, rng=rng)
    sf = run_reservoir_pipeline(inputs, reservoir, n_shots, sigma,
                                 rng=np.random.default_rng(seed + 1000))
    ev, yp_ev, yt = ev_training(sf, targets, 30, 0.7)
    traj, yp_traj, _ = trajectory_training(sf, targets, 30, 0.7)
    gtraj, yp_gtraj, _, k = grouped_trajectory_training(sf, targets, 30, 0.7)
    all_results.append({
        'nrmse_ev': ev, 'nrmse_traj': traj, 'nrmse_gtraj': gtraj,
        'y_pred_ev': yp_ev, 'y_pred_traj': yp_traj, 'y_pred_gtraj': yp_gtraj,
        'y_test': yt, 'best_k': k
    })

ev_scores = [r['nrmse_ev'] for r in all_results]
traj_scores = [r['nrmse_traj'] for r in all_results]
gtraj_scores = [r['nrmse_gtraj'] for r in all_results]

print(f"\nLorenz-x (Nshots={n_shots}, sigma={sigma}, S={S}, {n_seeds} seeds):")
print(f"  EV     NRMSE = {np.mean(ev_scores):.3f} +/- {np.std(ev_scores):.3f}")
print(f"  Traj   NRMSE = {np.mean(traj_scores):.3f} +/- {np.std(traj_scores):.3f}")
print(f"  G-Traj NRMSE = {np.mean(gtraj_scores):.3f} +/- {np.std(gtraj_scores):.3f}")
print(f"  Gap (EV - G-Traj) = {np.mean(ev_scores) - np.mean(gtraj_scores):.3f}")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

ax = axes[0]
r0 = all_results[0]
x_axis = np.arange(len(r0['y_test']))
ax.plot(x_axis, r0['y_test'], color=COLOR_TRUE, linewidth=1.8, label='True', zorder=4)
if len(r0.get('y_pred_traj', [])) == len(x_axis):
    ax.plot(x_axis, r0['y_pred_traj'], color=COLOR_TRAJ, linewidth=1.0, label='Traj', alpha=0.7, zorder=1)
if len(r0['y_pred_ev']) == len(x_axis):
    ax.plot(x_axis, r0['y_pred_ev'], color=COLOR_EV, linewidth=1.3, linestyle='--', label='EV', alpha=0.9, zorder=3)
if len(r0.get('y_pred_gtraj', [])) == len(x_axis):
    ax.plot(x_axis, r0['y_pred_gtraj'], color=COLOR_GTRAJ, linewidth=1.3, label='G-Traj', alpha=0.9, zorder=2)
ax.set_xlabel('Test time step')
ax.set_ylabel('Lorenz x')
ax.set_title('Time trace (seed 0)')
ax.legend(frameon=False, loc='upper right')

ax = axes[1]
means = [np.mean(ev_scores), np.mean(traj_scores), np.mean(gtraj_scores)]
stds = [np.std(ev_scores), np.std(traj_scores), np.std(gtraj_scores)]
bar_colors = [COLOR_EV, COLOR_TRAJ, COLOR_GTRAJ]
x_pos = np.arange(3)
ax.bar(x_pos, means, yerr=stds, width=0.6, color=bar_colors,
       edgecolor='white', linewidth=0.3, capsize=3, error_kw={'linewidth': 0.8})
ax.set_xticks(x_pos)
ax.set_xticklabels(['EV', 'Traj', 'G-Traj'])
ax.set_ylabel('NRMSE')
ax.set_title(f'NRMSE comparison ({n_seeds} seeds)')

ax = axes[2]
metrics = {'EV': [], 'Traj': [], 'G-Traj': []}
for result in all_results:
    yt = result['y_test']
    if len(yt) < 2: continue
    for name, pk in [('EV','y_pred_ev'),('Traj','y_pred_traj'),('G-Traj','y_pred_gtraj')]:
        pred = result.get(pk, np.array([]))
        if len(pred) == len(yt):
            metrics[name].append(np.std(pred) / max(np.std(yt), 1e-10))
means_sq = [np.mean(metrics[k]) if metrics[k] else 0 for k in ['EV', 'Traj', 'G-Traj']]
stds_sq = [np.std(metrics[k]) if metrics[k] else 0 for k in ['EV', 'Traj', 'G-Traj']]
ax.bar(x_pos, means_sq, yerr=stds_sq, width=0.6, color=bar_colors,
       edgecolor='white', linewidth=0.3, capsize=3, error_kw={'linewidth': 0.8})
ax.axhline(y=1.0, color='k', linestyle='--', linewidth=0.5, alpha=0.6)
ax.set_xticks(x_pos)
ax.set_xticklabels(['EV', 'Traj', 'G-Traj'])
ax.set_ylabel('Std ratio (pred / true)')
ax.set_title('Signal quality')

fig.suptitle(f'Lorenz attractor  |  Nshots={n_shots}, σ={sigma}, S={S}', fontsize=9, y=1.02)
fig.tight_layout()

out_path = os.path.join(OUT_DIR, 'lorenz_quick.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {out_path}")
plt.close(fig)
