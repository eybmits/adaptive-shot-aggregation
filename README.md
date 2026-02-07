# Budget-Aware Quantum Reservoir Computing

Grouped-trajectory training under fixed measurement budgets for quantum reservoir computing on near-term hardware.

## Key Finding

**G-Traj reduces prediction error (NRMSE) by 18–19%** compared to standard expectation-value training when the sample efficiency ratio ρ_EV < 3, following a universal power law:

$$\Delta(\rho_\text{EV}) \approx 0.124 \cdot \rho_\text{EV}^{-0.65}$$

## The Problem

On real quantum computers, each measurement yields a single binary outcome (±1). With a fixed measurement budget *S*, there is a fundamental trade-off:

```
S = N_shots × T
```

- **More shots** → better statistics, fewer training timesteps
- **Fewer shots** → noisier features, more training data

## Three Algorithms

| Algorithm | Training Samples | Noise Level | Result |
|-----------|-----------------|-------------|--------|
| **EV** (average shots) | T_train | Low | Baseline |
| **Trajectory** (stack shots) | N × T_train | High | Pathological (amplitude damping) |
| **G-Traj** (group K shots) | (N/K) × T_train | Medium | **Best when ρ_EV < 3** |

G-Traj automatically selects the group size *K* via validation — no manual tuning required.

## Results

| Dataset | EV NRMSE | G-Traj NRMSE | Improvement |
|---------|----------|--------------|-------------|
| Mackey-Glass | 0.592 ± 0.085 | 0.478 ± 0.087 | **19%** |
| Lorenz | 0.581 ± 0.067 | 0.478 ± 0.066 | **18%** |

## Figures

| Figure | Script | Description |
|--------|--------|-------------|
| `fig_benchmarks.pdf` | `plot_paper.py` | Time traces, NRMSE bars, signal quality |
| `fig_collapse.pdf` | `plot_collapse.py` | Universal collapse: Gap vs ρ_EV |
| `fig_phase.pdf` | `plot_phase.py` | Phase diagrams with phase boundary |

## Quick Start

```bash
# Install dependencies
pip install numpy scipy scikit-learn matplotlib

# Generate all figures (first run computes data, subsequent runs use cache)
python plot_paper.py
python plot_collapse.py
python plot_phase.py

# Generate results tables
python make_table.py
```

## Project Structure

```
qrc_budget.py        # Core implementation (reservoir, training, metrics)
plot_paper.py         # Figure 1: Benchmark comparison
plot_collapse.py      # Figure 2: Universal collapse plot
plot_phase.py         # Figure 3: Phase diagrams
make_table.py         # Results tables (Markdown + LaTeX)
theorie.md            # Theoretical framework
results/
  fig_benchmarks.pdf  # Generated figures
  fig_collapse.pdf
  fig_phase.pdf
  paper_report.md     # Paper-ready research report
  paper_results.md    # Numerical results tables
scripts/              # Exploration and development scripts
```

## Quantum Reservoir

- 6 qubits, Ry input encoding, Pauli Z/X measurements (M=12)
- Fixed random unitary: ring of CNOTs + random single-qubit rotations
- Leaky integrator temporal processing (η=0.5)
- Windowed features: d = L×M + 1 = 241

## License

Research code — see paper for citation details.
