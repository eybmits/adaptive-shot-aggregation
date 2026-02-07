"""Quick Mackey-Glass demo: run a single config and plot results."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from qrc_budget import (
    run_single_experiment, setup_nature_style,
    COLOR_EV, COLOR_TRAJ, COLOR_GTRAJ, COLOR_TRUE, OUT_DIR
)

# Sweet spot: rho_EV ~ 1.4, G-Traj biggest gap at sigma=0.03
n_shots = 40
sigma = 0.03
S = 20000
n_seeds = 8

setup_nature_style()

all_results = []
for seed in range(n_seeds):
    print(f"Mackey-Glass: seed {seed+1}/{n_seeds}")
    result = run_single_experiment(
        n_shots, sigma, S, benchmark='mackey_glass',
        seed=seed, eta=0.5, L=20, t0=30, alpha=0.7, depth=5
    )
    all_results.append(result)

ev_scores = [r['nrmse_ev'] for r in all_results]
traj_scores = [r['nrmse_traj'] for r in all_results]
gtraj_scores = [r['nrmse_gtraj'] for r in all_results]

print(f"\nResults (Nshots={n_shots}, sigma={sigma}, S={S}, {n_seeds} seeds):")
print(f"  EV     NRMSE = {np.mean(ev_scores):.3f} +/- {np.std(ev_scores):.3f}")
print(f"  Traj   NRMSE = {np.mean(traj_scores):.3f} +/- {np.std(traj_scores):.3f}")
print(f"  G-Traj NRMSE = {np.mean(gtraj_scores):.3f} +/- {np.std(gtraj_scores):.3f}")
print(f"  Gap (EV - G-Traj) = {np.mean(ev_scores) - np.mean(gtraj_scores):.3f}")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

# Panel 1: Time trace (seed 0)
ax = axes[0]
r0 = all_results[0]
x_axis = np.arange(len(r0['y_test']))
ax.plot(x_axis, r0['y_test'], color=COLOR_TRUE, linewidth=1.5, label='True', zorder=3)
if len(r0['y_pred_ev']) == len(x_axis):
    ax.plot(x_axis, r0['y_pred_ev'], color=COLOR_EV, linewidth=1.0, label='EV', alpha=0.85)
if len(r0.get('y_pred_traj', [])) == len(x_axis):
    ax.plot(x_axis, r0['y_pred_traj'], color=COLOR_TRAJ, linewidth=1.0, label='Traj', alpha=0.85)
if len(r0.get('y_pred_gtraj', [])) == len(x_axis):
    ax.plot(x_axis, r0['y_pred_gtraj'], color=COLOR_GTRAJ, linewidth=1.0, label='G-Traj', alpha=0.85)
ax.set_xlabel('Test time step')
ax.set_ylabel('Mackey-Glass')
ax.set_title('Time trace (seed 0)')
ax.legend(frameon=False, loc='upper right')

# Panel 2: NRMSE bar chart
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

# Panel 3: Std ratio (signal quality / amplitude match)
ax = axes[2]
metrics = {'EV': [], 'Traj': [], 'G-Traj': []}
for result in all_results:
    yt = result['y_test']
    if len(yt) < 2:
        continue
    for name, pred_key in [('EV', 'y_pred_ev'), ('Traj', 'y_pred_traj'),
                            ('G-Traj', 'y_pred_gtraj')]:
        pred = result.get(pred_key, np.array([]))
        if len(pred) == len(yt):
            sr = np.std(pred) / max(np.std(yt), 1e-10)
            metrics[name].append(sr)
means_sq = [np.mean(metrics[k]) if metrics[k] else 0 for k in ['EV', 'Traj', 'G-Traj']]
stds_sq = [np.std(metrics[k]) if metrics[k] else 0 for k in ['EV', 'Traj', 'G-Traj']]
ax.bar(x_pos, means_sq, yerr=stds_sq, width=0.6, color=bar_colors,
       edgecolor='white', linewidth=0.3, capsize=3, error_kw={'linewidth': 0.8})
ax.axhline(y=1.0, color='k', linestyle='--', linewidth=0.5, alpha=0.6)
ax.set_xticks(x_pos)
ax.set_xticklabels(['EV', 'Traj', 'G-Traj'])
ax.set_ylabel('Std ratio (pred / true)')
ax.set_title('Signal quality')

fig.suptitle(f'Mackey-Glass  |  Nshots={n_shots}, σ={sigma}, S={S}', fontsize=9, y=1.02)
fig.tight_layout()

out_path = os.path.join(OUT_DIR, 'mg_quick.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {out_path}")
plt.close(fig)
