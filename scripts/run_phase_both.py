"""Phase diagrams for MG and Lorenz: where does G-Traj beat EV?"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from qrc_budget import (
    QuantumReservoir, run_reservoir_pipeline, ev_training,
    grouped_trajectory_training, mackey_glass, compute_rho_ev,
    setup_nature_style, COLOR_EV, COLOR_GTRAJ, OUT_DIR
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

def run_one(series_func, n_shots, sigma, S, seed=0):
    rng = np.random.default_rng(seed)
    T = S // n_shots
    if T < 60: return None
    series = series_func(T + 50, rng)
    inputs, targets = series[:-1][:T], series[1:][:T]
    res = QuantumReservoir(n_qubits=6, depth=5, rng=rng)
    sf = run_reservoir_pipeline(inputs, res, n_shots, sigma,
                                 rng=np.random.default_rng(seed + 1000))
    ev, yp_ev, yt = ev_training(sf, targets, 30, 0.7)
    gt, yp_gt, _, k = grouped_trajectory_training(sf, targets, 30, 0.7)
    sr_gt = np.std(yp_gt) / max(np.std(yt), 1e-10) if len(yp_gt) == len(yt) else 0
    return {'ev': ev, 'gt': gt, 'gap': ev - gt, 'sr_gt': sr_gt, 'k': k}

def mg_series(n_steps, rng):
    return mackey_glass(n_steps, tau=17, rng=rng)

# --- Grid ---
shots_list = [10, 15, 20, 25, 30, 40, 50]
sigma_list = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
n_seeds = 3

datasets = {
    'Mackey-Glass': {'func': mg_series, 'S': 20000},
    'Lorenz': {'func': lorenz_x, 'S': 30000},
}

all_data = {}

for ds_name, ds in datasets.items():
    S = ds['S']
    func = ds['func']
    print(f"\n=== {ds_name} (S={S}) ===")

    gap_grid = np.full((len(shots_list), len(sigma_list)), np.nan)
    ev_grid = np.full_like(gap_grid, np.nan)
    gt_grid = np.full_like(gap_grid, np.nan)
    sr_grid = np.full_like(gap_grid, np.nan)
    rho_grid = np.full_like(gap_grid, np.nan)

    total = len(shots_list) * len(sigma_list)
    idx = 0
    for i, n_shots in enumerate(shots_list):
        for j, sigma in enumerate(sigma_list):
            idx += 1
            rho = compute_rho_ev(S, n_shots)
            rho_grid[i, j] = rho
            evs, gts, srs = [], [], []
            for seed in range(n_seeds):
                r = run_one(func, n_shots, sigma, S, seed)
                if r is None: continue
                evs.append(r['ev']); gts.append(r['gt']); srs.append(r['sr_gt'])
            if not evs: continue
            ev_grid[i, j] = np.mean(evs)
            gt_grid[i, j] = np.mean(gts)
            gap_grid[i, j] = np.mean(evs) - np.mean(gts)
            sr_grid[i, j] = np.mean(srs)
            g = gap_grid[i, j]
            tag = " ***" if g > 0.05 else ""
            print(f"  [{idx:>2d}/{total}] N={n_shots:>2d} σ={sigma:.2f} ρ={rho:.2f} "
                  f"EV={ev_grid[i,j]:.3f} GT={gt_grid[i,j]:.3f} gap={g:+.3f} SR={sr_grid[i,j]:.2f}{tag}")

    all_data[ds_name] = {
        'gap': gap_grid, 'ev': ev_grid, 'gt': gt_grid,
        'sr': sr_grid, 'rho': rho_grid, 'S': S
    }

# --- Phase diagram plots ---
setup_nature_style()
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8})

fig, axes = plt.subplots(2, 3, figsize=(11, 7))

for row, (ds_name, data) in enumerate(all_data.items()):
    S = data['S']
    gap = data['gap']
    ev = data['ev']
    gt = data['gt']

    # Panel 1: EV NRMSE
    ax = axes[row, 0]
    vmin, vmax = np.nanmin(ev), np.nanmax(ev)
    im = ax.imshow(ev, aspect='auto', cmap='YlOrRd',
                   vmin=max(0.3, vmin), vmax=min(1.5, vmax))
    ax.set_xticks(range(len(sigma_list)))
    ax.set_xticklabels([f'{s:.2f}' for s in sigma_list], rotation=45)
    ax.set_yticks(range(len(shots_list)))
    ax.set_yticklabels(shots_list)
    ax.set_xlabel('σ (noise)')
    ax.set_ylabel('N_shots')
    ax.set_title(f'{ds_name}: EV NRMSE (S={S})')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: G-Traj NRMSE
    ax = axes[row, 1]
    im = ax.imshow(gt, aspect='auto', cmap='YlOrRd',
                   vmin=max(0.3, np.nanmin(gt)), vmax=min(1.5, np.nanmax(gt)))
    ax.set_xticks(range(len(sigma_list)))
    ax.set_xticklabels([f'{s:.2f}' for s in sigma_list], rotation=45)
    ax.set_yticks(range(len(shots_list)))
    ax.set_yticklabels(shots_list)
    ax.set_xlabel('σ (noise)')
    ax.set_ylabel('N_shots')
    ax.set_title(f'{ds_name}: G-Traj NRMSE')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 3: Gap (EV - G-Traj) with diverging colormap
    ax = axes[row, 2]
    gap_clean = np.nan_to_num(gap, nan=0)
    absmax = max(abs(np.nanmin(gap_clean)), abs(np.nanmax(gap_clean)), 0.01)
    norm = TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
    im = ax.imshow(gap_clean, aspect='auto', cmap='RdBu', norm=norm)
    ax.set_xticks(range(len(sigma_list)))
    ax.set_xticklabels([f'{s:.2f}' for s in sigma_list], rotation=45)
    ax.set_yticks(range(len(shots_list)))
    ax.set_yticklabels(shots_list)
    ax.set_xlabel('σ (noise)')
    ax.set_ylabel('N_shots')
    ax.set_title(f'Gap (EV − G-Traj) [blue=G-Traj wins]')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Annotate rho_EV values
    for i in range(len(shots_list)):
        for j in range(len(sigma_list)):
            rho = data['rho'][i, j]
            g = gap_clean[i, j]
            if not np.isnan(rho):
                ax.text(j, i, f'{rho:.1f}', ha='center', va='center',
                        fontsize=5, color='black' if abs(g) < absmax * 0.5 else 'white')

fig.suptitle('Phase Diagrams: G-Traj vs EV across Datasets\n'
             '(numbers in gap panel = ρ_EV)', fontsize=10, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_path = os.path.join(OUT_DIR, 'phase_diagram_both.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nPhase diagram saved to {out_path}")
plt.close(fig)

# --- Collapse plot: Gap vs rho_EV ---
fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

for ds_name, data in all_data.items():
    rhos = data['rho'].flatten()
    gaps = data['gap'].flatten()
    mask = ~np.isnan(rhos) & ~np.isnan(gaps)
    rhos, gaps = rhos[mask], gaps[mask]
    color = COLOR_EV if 'Mackey' in ds_name else COLOR_GTRAJ
    ax.scatter(rhos, gaps, s=20, alpha=0.7, label=ds_name, color=color, edgecolors='white', linewidth=0.3)

ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
ax.axvline(x=1.0, color='grey', linewidth=0.5, linestyle=':', alpha=0.7)
ax.set_xlabel('ρ_EV (sample efficiency ratio)')
ax.set_ylabel('Gap (EV − G-Traj)')
ax.set_title('Universal collapse: G-Traj advantage vs ρ_EV')
ax.legend(frameon=False)
ax.annotate('G-Traj wins', xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=7, color='blue', va='top')
ax.annotate('EV wins', xy=(0.05, 0.05), xycoords='axes fraction',
            fontsize=7, color='red', va='bottom')

fig.tight_layout()
out_path2 = os.path.join(OUT_DIR, 'rho_collapse_both.png')
fig.savefig(out_path2, dpi=300, bbox_inches='tight')
print(f"Collapse plot saved to {out_path2}")
plt.close(fig)
