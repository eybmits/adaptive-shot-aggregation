"""Publication-quality figures for the paper."""
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from qrc_budget import (
    QuantumReservoir, run_reservoir_pipeline, ev_training,
    trajectory_training, grouped_trajectory_training_legacy,
    grouped_trajectory_training_narma_fix, mackey_glass,
    narma10, compute_rho_ev, OUT_DIR
)

# Cache with NARMA-10 fix enabled in the benchmark panel.
CACHE_FILE = os.path.join(OUT_DIR, 'data_benchmarks_narma_fix.pkl')

# --- Style ---
COLORS = {
    'true': '#000000',
    'ev': '#555555',
    'traj': '#aaaaaa',
    'gtraj': '#c0392b',
}
HATCHES = {
    'ev': '///',
    'traj': '...',
    'gtraj': '',
}
BAR_EDGE = '#333333'

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


def select_trace_window(y_true, window):
    """Select a high-variance window for clearer short-time visualization."""
    n = len(y_true)
    if window is None or window <= 0 or window >= n:
        return 0, n
    best_s = 0
    best_var = -np.inf
    for s in range(0, n - window + 1):
        v = np.var(y_true[s:s + window])
        if v > best_var:
            best_var = v
            best_s = s
    return best_s, best_s + window


def resolve_trace_interval(y_true, cfg):
    """
    Resolve trace interval from config.
    If explicit start/end is provided, use it; otherwise use high-variance window.
    """
    n = len(y_true)
    if 'trace_start' in cfg and 'trace_end' in cfg:
        s = int(max(0, cfg['trace_start']))
        e = int(min(n, cfg['trace_end']))
        if e > s:
            return s, e
    return select_trace_window(y_true, cfg.get('trace_window', None))


def run_dataset(dataset_name, series_func, n_shots, sigma, S, n_seeds=8):
    all_results = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        T = S // n_shots
        series = series_func(T + 50, rng)
        inputs, targets = series[:-1][:T], series[1:][:T]
        res = QuantumReservoir(n_qubits=6, depth=5, rng=rng)
        sf = run_reservoir_pipeline(inputs, res, n_shots, sigma,
                                     rng=np.random.default_rng(seed + 1000))
        ev, yp_ev, yt = ev_training(sf, targets, 30, 0.7)
        tr, yp_tr, _ = trajectory_training(sf, targets, 30, 0.7)
        if dataset_name == 'NARMA-10':
            gt, yp_gt, _, k = grouped_trajectory_training_narma_fix(
                sf, targets, 30, 0.7
            )
        else:
            gt, yp_gt, _, k = grouped_trajectory_training_legacy(
                sf, targets, 30, 0.7
            )
        all_results.append({
            'ev': ev, 'traj': tr, 'gtraj': gt,
            'yp_ev': yp_ev, 'yp_tr': yp_tr, 'yp_gt': yp_gt,
            'yt': yt, 'k': k
        })
        print(f"  seed {seed+1}/{n_seeds} done")
    return all_results


def compute_data():
    """Run simulations and return data dict."""
    configs = [
        {'name': 'Mackey-Glass', 'func': lambda n, rng: mackey_glass(n, tau=17, rng=rng),
         'S': 20000, 'N': 40, 'sigma': 0.03},
        {'name': 'Lorenz attractor', 'func': lorenz_x,
         'S': 30000, 'N': 40, 'sigma': 0.0},
        {'name': 'NARMA-10', 'func': narma_series,
         'S': 50000, 'N': 60, 'sigma': 0.03},
    ]
    all_data = {}
    for cfg in configs:
        print(f"\n  Running {cfg['name']}...")
        results = run_dataset(cfg['name'], cfg['func'], cfg['N'], cfg['sigma'], cfg['S'], n_seeds=8)
        all_data[cfg['name']] = {'results': results, 'cfg': {k: v for k, v in cfg.items() if k != 'func'}}
    return all_data


def load_or_compute():
    """Load cached data or compute and save."""
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


def make_figure1():
    print("=== Figure 1: Benchmark comparison ===")
    setup_style()

    all_data = load_or_compute()

    configs = [
        {'name': 'Mackey-Glass', 'S': 20000, 'N': 40, 'sigma': 0.03},
        {'name': 'Lorenz attractor', 'S': 30000, 'N': 40, 'sigma': 0.0},
        {'name': 'NARMA-10', 'S': 50000, 'N': 60, 'sigma': 0.03, 'trace_start': 60, 'trace_end': 120},
    ]

    fig = plt.figure(figsize=(7.2, 6.4))
    gs = gridspec.GridSpec(3, 3, hspace=0.55, wspace=0.45,
                           left=0.08, right=0.97, top=0.97, bottom=0.07)

    for row, cfg in enumerate(configs):
        results = all_data[cfg['name']]['results']

        ev_s = [r['ev'] for r in results]
        tr_s = [r['traj'] for r in results]
        gt_s = [r['gtraj'] for r in results]

        # --- Time trace ---
        ax = fig.add_subplot(gs[row, 0])
        r0 = results[0]
        full_x = np.arange(len(r0['yt']))
        s, e = resolve_trace_interval(r0['yt'], cfg)
        x = full_x[s:e]
        y_true = r0['yt'][s:e]
        yp_tr = r0['yp_tr'][s:e]
        yp_ev = r0['yp_ev'][s:e]
        yp_gt = r0['yp_gt'][s:e]

        ax.plot(x, y_true, color=COLORS['true'], lw=1.8, label='True', zorder=4)
        if len(yp_tr) == len(x):
            ax.plot(x, yp_tr, color=COLORS['traj'], lw=0.9, alpha=0.5, label='Traj', zorder=1)
        if len(yp_ev) == len(x):
            ax.plot(x, yp_ev, color=COLORS['ev'], lw=1.2, ls='--', alpha=0.85, label='EV', zorder=3)
        if len(yp_gt) == len(x):
            ax.plot(x, yp_gt, color=COLORS['gtraj'], lw=1.2, alpha=0.9, label='G-Traj', zorder=2)
        ax.set_xlabel('Test time step')
        ax.set_ylabel(cfg['name'])
        ax.legend(frameon=True, fancybox=False, edgecolor='#999999',
                  facecolor='white', framealpha=0.95,
                  loc='upper right', handlelength=1.5,
                  borderpad=0.4, labelspacing=0.25, fontsize=6.5)

        # --- NRMSE bars ---
        ax = fig.add_subplot(gs[row, 1])
        means = [np.mean(ev_s), np.mean(tr_s), np.mean(gt_s)]
        stds = [np.std(ev_s), np.std(tr_s), np.std(gt_s)]
        x_pos = np.arange(3)
        colors = [COLORS['ev'], COLORS['traj'], COLORS['gtraj']]
        hatches = [HATCHES['ev'], HATCHES['traj'], HATCHES['gtraj']]
        for xi, m, s, c, h in zip(x_pos, means, stds, colors, hatches):
            ax.bar(xi, m, yerr=s, width=0.55, color=c, hatch=h,
                   edgecolor=BAR_EDGE, linewidth=0.6, capsize=2.5,
                   error_kw={'linewidth': 0.7, 'capthick': 0.7})
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['EV', 'Traj', 'G-Traj'])
        ax.set_ylabel('NRMSE')

        # --- Std ratio ---
        ax = fig.add_subplot(gs[row, 2])
        metrics = {'EV': [], 'Traj': [], 'G-Traj': []}
        for r in results:
            yt = r['yt']
            std_yt = max(np.std(yt), 1e-10)
            for name, key in [('EV', 'yp_ev'), ('Traj', 'yp_tr'), ('G-Traj', 'yp_gt')]:
                p = r[key]
                if len(p) == len(yt):
                    metrics[name].append(np.std(p) / std_yt)
        m = [np.mean(metrics[k]) for k in ['EV', 'Traj', 'G-Traj']]
        s = [np.std(metrics[k]) for k in ['EV', 'Traj', 'G-Traj']]
        for xi, mi, si, c, h in zip(x_pos, m, s, colors, hatches):
            ax.bar(xi, mi, yerr=si, width=0.55, color=c, hatch=h,
                   edgecolor=BAR_EDGE, linewidth=0.6, capsize=2.5,
                   error_kw={'linewidth': 0.7, 'capthick': 0.7})
        ax.axhline(y=1.0, color='#888888', ls='--', lw=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['EV', 'Traj', 'G-Traj'])
        ax.set_ylabel('Std ratio')

    # Subtle config annotations
    for row, cfg in enumerate(configs):
        ax = fig.axes[row * 3]
        ax.text(0.03, 0.05, f'$S$={cfg["S"]//1000}k, $N$={cfg["N"]}, $\\sigma$={cfg["sigma"]}',
                transform=ax.transAxes, fontsize=6, color='#666666', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.85))

    path = os.path.join(OUT_DIR, 'fig_benchmarks.pdf')
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")
    plt.close(fig)


if __name__ == '__main__':
    make_figure1()
