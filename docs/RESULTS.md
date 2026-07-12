# Results used in the paper

The primary comparison uses 500 time steps and 20 shots per step. Aggregation
scale and ridge regularization are tuned jointly on chronological development
data. The
table reports the paired difference between EV and Grouped test NRMSE.
Positive values favor Grouped.

| Task | Mean difference | 95% interval |
|---|---:|---:|
| Mackey Glass | 0.013846 | [-0.002290, 0.029982] |
| Santa Fe laser | 0.084787 | [0.017418, 0.152156] |
| Rössler | 0.034926 | [0.013947, 0.055904] |
| Sunspots | 0.035954 | [0.010328, 0.061580] |

Source:
`results/qai_robust_protocol_santa_fe_manuscript/cluster_paired_stats.csv`.
The corresponding manifest records row counts and SHA256 hashes for every
table in the paper evidence package.

The effect is largest when the execution budget contains many shots at fewer
time steps. Cov EV is close to Grouped at the primary setting, supporting the
covariance regularization interpretation of adaptive shot aggregation.
