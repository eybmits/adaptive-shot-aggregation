"""Publication-quality rho_EV collapse plot — compact style."""
import numpy as np
import pickle
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from qrc_budget import (
    QuantumReservoir, run_reservoir_pipeline, ev_training,
    grouped_trajectory_training_legacy, grouped_trajectory_training_narma_fix,
    mackey_glass, narma10,
    compute_rho_ev, OUT_DIR
)

parser = argparse.ArgumentParser(description="Plot rho_EV collapse.")
parser.add_argument(
    '--narma-fix', action='store_true',
    help='Use NARMA-10 G-Traj fix (std-aware selection + guarded calibration).'
)
parser.add_argument('--seeds', type=int, default=5, help='Number of random seeds per grid point.')
parser.add_argument('--cache-file', type=str, default=None, help='Optional cache file path.')
parser.add_argument('--output', type=str, default=None, help='Optional output pdf path.')
parser.add_argument(
    '--shots-list', type=str, default=None,
    help='Comma-separated shot counts, e.g. "5,10,20,30,40".'
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
    else os.path.join(OUT_DIR, 'data_collapse_narma_fix.pkl' if USE_NARMA_FIX else 'data_collapse.pkl')
)
OUT_PATH = (
    args.output
    if args.output
    else os.path.join(OUT_DIR, 'fig_collapse_narma_fix.pdf' if USE_NARMA_FIX else 'fig_collapse.pdf')
)

# --- Style (same as fig_benchmarks) ---
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

COLORS = {
    'mg': '#000000',
    'lorenz': '#888888',
    'narma': '#555555',
    'fit': '#B00020',
}

# --- Data ---
def lorenz_x(n_steps, rng, dt=0.02, warmup=2000):
    s, r, b = 10.0, 28.0, 8.0/3.0
    total = n_steps + warmup
    x, y, z = 1.0, 1.0, 1.0
    xs = np.zeros(total)
    for i in range(total):
        xs[i] = x
        dx = s*(y-x)*dt; dy = (x*(r-z)-y)*dt; dz = (x*y-b*z)*dt
        x += dx; y += dy; z += dz
    out = xs[warmup:]
    return (out - out.min()) / (out.max() - out.min() + 1e-10)

def narma_series(n_steps, rng):
    _, y = narma10(n_steps, rng=rng)
    return y

def run_one(dataset_name, func, n_shots, sigma, S, seed):
    rng = np.random.default_rng(seed)
    T = S // n_shots
    if T < 60: return None
    series = func(T + 50, rng)
    inputs, targets = series[:-1][:T], series[1:][:T]
    res = QuantumReservoir(n_qubits=6, depth=5, rng=rng)
    sf = run_reservoir_pipeline(inputs, res, n_shots, sigma, rng=np.random.default_rng(seed+1000))
    ev, _, _ = ev_training(sf, targets, 30, 0.7)
    if USE_NARMA_FIX and dataset_name == 'NARMA-10':
        gt, _, _, _ = grouped_trajectory_training_narma_fix(sf, targets, 30, 0.7)
    else:
        gt, _, _, _ = grouped_trajectory_training_legacy(sf, targets, 30, 0.7)
    return ev - gt

# Full paper grid — extended to low N_shots for high rho_EV
shots_list = _parse_int_list(args.shots_list, [5, 8, 10, 15, 20, 25, 30, 40, 50])
sigma_list = _parse_float_list(args.sigma_list, [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10])

datasets = {
    'Mackey-Glass': (lambda n, rng: mackey_glass(n, tau=17, rng=rng), 20000),
    'Lorenz': (lorenz_x, 30000),
    'NARMA-10': (narma_series, 50000),
}


def compute_data():
    all_rho, all_gap, all_label = [], [], []
    for ds_name, (func, S) in datasets.items():
        print(f"Sweeping {ds_name}...")
        for n_shots in shots_list:
            for sigma in sigma_list:
                rho = compute_rho_ev(S, n_shots)
                gaps = []
                for seed in range(n_seeds):
                    g = run_one(ds_name, func, n_shots, sigma, S, seed)
                    if g is not None: gaps.append(g)
                if gaps:
                    all_rho.append(rho)
                    all_gap.append(np.mean(gaps))
                    all_label.append(ds_name)
    return {
        'all_rho': np.array(all_rho),
        'all_gap': np.array(all_gap),
        'all_label': np.array(all_label),
    }


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
data = load_or_compute()
all_rho = data['all_rho']
all_gap = data['all_gap']
all_label = data['all_label']

# --- Fit ---
mask_pos = all_gap > 0.005
rho_fit = all_rho[mask_pos]
gap_fit = all_gap[mask_pos]

def power_law(x, C, beta):
    return C * x ** (-beta)

try:
    popt, _ = curve_fit(power_law, rho_fit, gap_fit, p0=[0.3, 1.5], maxfev=5000)
    C_fit, beta_fit = popt
    print(f"Fit: Gap = {C_fit:.3f} * rho^(-{beta_fit:.2f})")
    has_fit = True
except:
    has_fit = False

# --- Plot ---
setup_style()

fig, ax = plt.subplots(figsize=(3.5, 2.6))

# Subtle shaded regions
ax.axhspan(0.0, -0.05, color='#e8e8e8', alpha=0.5, zorder=0)
ax.axhspan(0.0, 0.40, color='#f7f7f7', alpha=0.6, zorder=0)

# Data: black filled circles (MG), gray open squares (Lorenz), dark-gray open triangles (NARMA)
for ds, marker, fc, ec, lw, zorder in [
    ('Mackey-Glass', 'o', COLORS['mg'], COLORS['mg'], 0.3, 3),
    ('Lorenz', 's', 'none', COLORS['lorenz'], 0.8, 2),
    ('NARMA-10', '^', 'none', COLORS['narma'], 1.0, 3),
]:
    mask = all_label == ds
    ax.scatter(all_rho[mask], all_gap[mask], s=22, facecolors=fc, marker=marker,
               alpha=0.8, edgecolors=ec, linewidth=lw, label=ds, zorder=zorder)

# Fit curve
if has_fit:
    rho_curve = np.linspace(0.9, 18, 200)
    gap_curve = np.clip(power_law(rho_curve, C_fit, beta_fit), 0, None)
    ax.plot(rho_curve, gap_curve, color=COLORS['fit'], lw=1.5, ls='-', alpha=1.0, zorder=1,
            label=f'$C \\cdot \\rho^{{-\\beta}}$, $C$={C_fit:.2f}, $\\beta$={beta_fit:.1f}')

# Zero line
ax.axhline(y=0, color='#555555', lw=0.5, ls='-', zorder=1)

# Annotations
ax.annotate('G-Traj wins', xy=(1.15, 0.28), fontsize=6.5, color='#333333',
            fontweight='medium')
ax.annotate('EV wins / equal', xy=(10.0, -0.03), fontsize=6.5, color='#555555',
            fontweight='medium', ha='center')

ax.set_xlabel(r'$\rho_{\mathrm{EV}}$')
ax.set_ylabel(r'Gap (NRMSE$_{\mathrm{EV}}$ $-$ NRMSE$_{\mathrm{G\text{-}Traj}}$)')
ax.set_xlim(0.8, 18)
ax.set_ylim(-0.04, 0.38)

ax.legend(frameon=True, fancybox=False, edgecolor='#999999',
          facecolor='white', framealpha=0.95,
          loc='upper right', borderpad=0.3, labelspacing=0.3,
          handletextpad=0.4, fontsize=6.5)

fig.savefig(OUT_PATH, dpi=300)
print(f"Saved {OUT_PATH}")
plt.close(fig)
