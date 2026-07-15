# Results used in the paper

The primary comparison uses 500 time steps and 20 shots per step. Aggregation
scale and ridge regularization are tuned jointly on chronological validation
data. The
table reports the paired difference between EV and Grouped test NRMSE.
Positive values favor Grouped.

| Task | EV | Grouped | Relative reduction | Mean difference | 95% interval |
|---|---:|---:|---:|---:|---:|
| Mackey-Glass | 0.483764 | 0.439221 | 9.21% | 0.044543 | [0.037176, 0.051911] |
| Santa Fe laser | 0.764448 | 0.679661 | 11.09% | 0.084787 | [0.017418, 0.152156] |
| Rössler | 0.413728 | 0.378802 | 8.44% | 0.034926 | [0.013947, 0.055904] |
| Sunspots | 0.553056 | 0.517102 | 6.50% | 0.035954 | [0.010328, 0.061580] |

The mean NRMSE across tasks falls from 0.553749 to 0.503696, which is a
9.04% reduction. All four paired intervals are above zero.

Source:
`results/qai_robust_protocol_santa_fe_manuscript/cluster_paired_stats.csv`.
The corresponding manifest records row counts and SHA256 hashes for every
table in the paper evidence package.

The effect is largest when the execution budget contains many shots at fewer
time steps. Regularized EV is close to Grouped at the primary setting, supporting the
covariance regularization interpretation of adaptive shot aggregation.
