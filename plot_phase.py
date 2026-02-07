"""Nature-quality phase diagram — smooth contourf with phase boundary."""
import numpy as np
import pickle
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from qrc_budget import (
    QuantumReservoir, run_reservoir_pipeline, ev_training,
    grouped_trajectory_training_legacy, grouped_trajectory_training_narma_fix,
    mackey_glass, narma10,
    compute_rho_ev, OUT_DIR
)

parser = argparse.ArgumentParser(description="Plot phase diagram.")
parser.add_argument(
    '--narma-fix', action='store_true',
    help='Use NARMA-10 G-Traj fix (std-aware selection + guarded calibration).'
)
parser.add_argument('--seeds', type=int, default=5, help='Number of random seeds per grid point.')
parser.add_argument('--cache-file', type=str, default=None, help='Optional cache file path.')
parser.add_argument('--output', type=str, default=None, help='Optional output pdf path.')
parser.add_argument(
    '--shots-list', type=str, default=None,
    help='Comma-separated shot counts, e.g. "10,20,30,40".'
)
parser.add_argument(
    '--sigma-list', type=str, default=None,
    help='Comma-separated sigmas, e.g. "0.0,0.01,0.03,0.05".'
)
args = parser.parse_args()

USE_NARMA_FIX = bool(args.narma_fix)
n_seeds = int(args.seeds)


def _parse_int_list(raw, default):
    if not raw:
        return default
    return [int(x.strip()) for x in raw.split(',') if x.strip()]


def _parse_float_list(raw, default):
    if not raw:
        return default
    return [float(x.strip()) for x in raw.split(',') if x.strip()]

CACHE_FILE = (
    args.cache_file
    if args.cache_file
    else os.path.join(OUT_DIR, 'data_phase_narma_fix.pkl' if USE_NARMA_FIX else 'data_phase.pkl')
)
OUT_PATH = (
    args.output
    if args.output
    else os.path.join(OUT_DIR, 'fig_phase_narma_fix.pdf' if USE_NARMA_FIX else 'fig_phase.pdf')
)

# --- Style (same as other paper figures) ---
def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.titleweight': 'medium',
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'lines.linewidth': 1.0,
        'axes.grid': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

# --- Data generators ---
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

def narma_series(n_steps, rng):
    _, y = narma10(n_steps, rng=rng)
    return y

def run_one(dataset_name, series_func, n_shots, sigma, S, seed=0):
    rng = np.random.default_rng(seed)
    T = S // n_shots
    if T < 60: return None
    series = series_func(T + 50, rng)
    inputs, targets = series[:-1][:T], series[1:][:T]
    res = QuantumReservoir(n_qubits=6, depth=5, rng=rng)
    sf = run_reservoir_pipeline(inputs, res, n_shots, sigma,
                                 rng=np.random.default_rng(seed + 1000))
    ev, yp_ev, yt = ev_training(sf, targets, 30, 0.7)
    if USE_NARMA_FIX and dataset_name == 'NARMA-10':
        gt, yp_gt, _, k = grouped_trajectory_training_narma_fix(sf, targets, 30, 0.7)
    else:
        gt, yp_gt, _, k = grouped_trajectory_training_legacy(sf, targets, 30, 0.7)
    return {'ev': ev, 'gt': gt, 'gap': ev - gt}

# Full paper grid
shots_list = _parse_int_list(args.shots_list, [10, 15, 20, 25, 30, 40, 50])
sigma_list = _parse_float_list(args.sigma_list, [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10])

ds_configs = {
    'Mackey-Glass': {'func': mg_series, 'S': 20000},
    'Lorenz': {'func': lorenz_x, 'S': 30000},
    'NARMA-10': {'func': narma_series, 'S': 50000},
}


def compute_data():
    all_data = {}
    for ds_name, ds in ds_configs.items():
        S = ds['S']
        func = ds['func']
        print(f"\n=== {ds_name} (S={S}) ===")

        gap_grid = np.full((len(shots_list), len(sigma_list)), np.nan)
        ev_grid = np.full_like(gap_grid, np.nan)
        gt_grid = np.full_like(gap_grid, np.nan)
        rho_grid = np.full_like(gap_grid, np.nan)

        total = len(shots_list) * len(sigma_list)
        idx = 0
        for i, n_shots in enumerate(shots_list):
            for j, sigma in enumerate(sigma_list):
                idx += 1
                rho = compute_rho_ev(S, n_shots)
                rho_grid[i, j] = rho
                evs, gts = [], []
                for seed in range(n_seeds):
                    r = run_one(ds_name, func, n_shots, sigma, S, seed)
                    if r is None: continue
                    evs.append(r['ev']); gts.append(r['gt'])
                if not evs: continue
                ev_grid[i, j] = np.mean(evs)
                gt_grid[i, j] = np.mean(gts)
                gap_grid[i, j] = np.mean(evs) - np.mean(gts)
                g = gap_grid[i, j]
                print(f"  [{idx:>2d}/{total}] N={n_shots:>2d} σ={sigma:.2f} ρ={rho:.2f} "
                      f"EV={ev_grid[i,j]:.3f} GT={gt_grid[i,j]:.3f} gap={g:+.3f}")

        all_data[ds_name] = {
            'gap': gap_grid, 'ev': ev_grid, 'gt': gt_grid, 'rho': rho_grid, 'S': S
        }
    return all_data


def load_or_compute():
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    print("No cache found, computing...")
    data = compute_data()
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved cache to {CACHE_FILE}")
    return data


mode_label = 'narma_fix' if USE_NARMA_FIX else 'legacy'
print(f"Mode: {mode_label} | seeds={n_seeds}")
all_data = load_or_compute()

# --- Plot: Nature-quality contourf phase diagram ---
setup_style()
# Enable all spines for contour plots
plt.rcParams.update({
    'axes.spines.top': True,
    'axes.spines.right': True,
})

fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0),
                         gridspec_kw={'wspace': 0.08, 'left': 0.06,
                                      'right': 0.90, 'top': 0.95, 'bottom': 0.16})

# Shared color scale
all_gaps = []
for data in all_data.values():
    g = np.nan_to_num(data['gap'], nan=0)
    all_gaps.append(g)
absmax = max(abs(np.nanmin(np.concatenate([g.ravel() for g in all_gaps]))),
             abs(np.nanmax(np.concatenate([g.ravel() for g in all_gaps]))),
             0.01)
norm = TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)

sigma_arr = np.array(sigma_list)
shots_arr = np.array(shots_list)

# Fine grid for interpolation
sigma_fine = np.linspace(sigma_arr[0], sigma_arr[-1], 200)
shots_fine = np.linspace(shots_arr[0], shots_arr[-1], 200)
Sf, Nf = np.meshgrid(sigma_fine, shots_fine)

im_last = None
for col, (ds_name, data) in enumerate(all_data.items()):
    ax = axes[col]
    S = data['S']
    gap = data['gap']
    gap_clean = np.nan_to_num(gap, nan=0)

    # Smooth interpolation (shots on rows, sigma on cols)
    # Pad data by one cell on all sides to stabilise edge behaviour
    pad_shots = np.concatenate([[shots_arr[0] - (shots_arr[1]-shots_arr[0])],
                                 shots_arr,
                                 [shots_arr[-1] + (shots_arr[-1]-shots_arr[-2])]])
    pad_sigma = np.concatenate([[sigma_arr[0] - (sigma_arr[1]-sigma_arr[0])],
                                 sigma_arr,
                                 [sigma_arr[-1] + (sigma_arr[-1]-sigma_arr[-2])]])
    gap_padded = np.pad(gap_clean, 1, mode='edge')

    spline = RectBivariateSpline(pad_shots, pad_sigma, gap_padded, kx=3, ky=3)
    gap_smooth = spline(shots_fine, sigma_fine)
    # Light smoothing for contourf — enough to remove spline ringing
    gap_smooth = gaussian_filter(gap_smooth, sigma=6)

    # Filled contours
    levels = np.linspace(-absmax, absmax, 25)
    im = ax.contourf(Sf, Nf, gap_smooth, levels=levels, cmap='RdBu_r', norm=norm,
                     extend='both')
    im_last = im

    # Phase boundary: use LINEAR interpolation on raw data (no spline ringing)
    from scipy.interpolate import RegularGridInterpolator
    interp_lin = RegularGridInterpolator((shots_arr, sigma_arr), gap_clean,
                                          method='linear', bounds_error=False,
                                          fill_value=0)
    gap_lin = interp_lin((Nf, Sf))
    gap_lin = gaussian_filter(gap_lin, sigma=5)
    cs = ax.contour(Sf, Nf, gap_lin, levels=[0.005], colors='none')
    segs = cs.allsegs[0] if cs.allsegs else []
    # Keep only segments that span most of the σ range (skip small islands)
    wide_segs = [
        s for s in segs
        if len(s) > 1 and (s[:, 0].max() - s[:, 0].min()) > 0.4 * (sigma_arr[-1] - sigma_arr[0])
    ]
    if wide_segs:
        longest = max(wide_segs, key=lambda s: len(s))
        ax.plot(longest[:, 0], longest[:, 1], color='black', lw=1.0, zorder=4)
    elif any(len(s) > 1 for s in segs):
        longest = max([s for s in segs if len(s) > 1], key=lambda s: len(s))
        ax.plot(longest[:, 0], longest[:, 1], color='black', lw=1.0, zorder=4)

    # rho_EV iso-contours (ρ = α/d × (S/N - t0))
    # ρ depends only on N_shots (not σ), so iso-lines are horizontal
    # But we show them as labeled reference lines
    d, t0, alpha = 241, 30, 0.7
    rho_targets = [1, 2, 3, 5]
    for rho_t in rho_targets:
        # ρ = α/d × (S/N - t0) → N = S / (ρ × d/α + t0)
        N_for_rho = S / (rho_t * d / alpha + t0)
        if shots_arr[0] <= N_for_rho <= shots_arr[-1]:
            ax.axhline(N_for_rho, color='#555555', ls='--', lw=0.5, alpha=0.7)
            ax.text(sigma_arr[0] + 0.002, N_for_rho + 0.8,
                    rf'$\rho_{{\mathrm{{EV}}}}$={rho_t}', fontsize=5.5,
                    color='#444444', ha='left', va='bottom')

    # Data points overlay
    Sg, Ng = np.meshgrid(sigma_arr, shots_arr)
    ax.scatter(Sg.ravel(), Ng.ravel(), s=8, c='black', zorder=5, alpha=0.6)

    ax.set_xlabel(r'$\sigma$')
    if col == 0:
        ax.set_ylabel(r'$N_{\mathrm{shots}}$')
    else:
        ax.set_yticklabels([])

    ax.set_xlim(sigma_arr[0], sigma_arr[-1])
    ax.set_ylim(shots_arr[0], shots_arr[-1])

    # Dataset label
    ax.text(0.03, 0.96, ds_name, transform=ax.transAxes, fontsize=7,
            fontweight='medium', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.8))


# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.16, 0.012, 0.79])
cb = fig.colorbar(im_last, cax=cbar_ax)
cb.set_label(r'Gap (NRMSE$_{\mathrm{EV}}$ $-$ NRMSE$_{\mathrm{G\text{-}Traj}}$)',
             fontsize=7)
cb.ax.tick_params(labelsize=6)

fig.savefig(OUT_PATH, dpi=300)
print(f"Saved {OUT_PATH}")
plt.close(fig)
