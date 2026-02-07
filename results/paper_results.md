# Paper Results

## Table 1: Benchmark comparison (8 seeds)

| Dataset | S | N | σ | ρ_EV | EV NRMSE | Traj NRMSE | G-Traj NRMSE | Gap | SR(EV) | SR(G-Traj) |
|---------|---|---|---|------|----------|------------|--------------|-----|--------|------------|
| Mackey-Glass | 20000 | 40 | 0.03 | 1.37 | 0.592±0.085 | 0.810±0.048 | 0.478±0.087 | +0.114 | 1.02 | 0.82 |
| Lorenz | 30000 | 40 | 0.0 | 2.09 | 0.581±0.067 | 0.737±0.045 | 0.478±0.066 | +0.103 | 1.03 | 0.82 |

## Table 2: Phase sweep summary (selected configs)

| Dataset | N | σ | ρ_EV | EV | G-Traj | Gap | SR(GT) |
|---------|---|---|------|----|--------|-----|--------|
| Mackey-Glass | 20 | 0.0 | 2.82 | 0.548±0.151 | 0.491±0.128 | +0.057 | 0.82 |
| Mackey-Glass | 30 | 0.03 | 1.85 | 0.658±0.101 | 0.557±0.136 | +0.101 | 0.87 |
| Mackey-Glass | 40 | 0.03 | 1.37 | 0.574±0.084 | 0.446±0.082 | +0.128 | 0.77 |
| Mackey-Glass | 50 | 0.05 | 1.07 | 0.664±0.095 | 0.581±0.097 | +0.083 | 0.84 |
| Lorenz | 20 | 0.0 | 4.27 | 0.554±0.046 | 0.554±0.046 | +0.000 | 0.90 |
| Lorenz | 30 | 0.0 | 2.82 | 0.491±0.057 | 0.479±0.064 | +0.012 | 0.95 |
| Lorenz | 40 | 0.0 | 2.09 | 0.562±0.074 | 0.444±0.019 | +0.118 | 0.82 |
| Lorenz | 50 | 0.0 | 1.66 | 0.572±0.068 | 0.494±0.045 | +0.078 | 0.81 |

## LaTeX Table 1

```latex
\begin{table}[t]
\centering
\caption{Benchmark results (8 seeds). Gap = NRMSE\textsubscript{EV} $-$ NRMSE\textsubscript{G-Traj}.}
\label{tab:benchmark}
\begin{tabular}{lcccccc}
\toprule
Dataset & $S$ & $N$ & $\sigma$ & $\rho_{\text{EV}}$ & EV & G-Traj & Gap \\
\midrule
Mackey-Glass & 20k & 40 & 0.03 & 1.37 & $0.592 \pm 0.085$ & $0.478 \pm 0.087$ & $+0.114$ \\
Lorenz & 30k & 40 & 0.0 & 2.09 & $0.581 \pm 0.067$ & $0.478 \pm 0.066$ & $+0.103$ \\
\bottomrule
\end{tabular}
\end{table}
```
