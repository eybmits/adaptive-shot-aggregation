# Results used in the paper

The primary comparison uses 500 time steps and 20 shots per step. Aggregation
scale and ridge regularization are tuned jointly on chronological development
data. The
table reports the paired difference between EV and Grouped test NRMSE.
Positive values favor Grouped.

| Task | EV | Grouped | Relative reduction | Mean difference | 95% interval |
|---|---:|---:|---:|---:|---:|
| Mackey-Glass | 0.467590 | 0.453745 | 2.96% | 0.013846 | [-0.002290, 0.029982] |
| Santa Fe laser | 0.764448 | 0.679661 | 11.09% | 0.084787 | [0.017418, 0.152156] |
| Rössler | 0.413728 | 0.378802 | 8.44% | 0.034926 | [0.013947, 0.055904] |
| Sunspots | 0.553056 | 0.517102 | 6.50% | 0.035954 | [0.010328, 0.061580] |

The mean NRMSE across tasks falls from 0.549706 to 0.507327, which is a
7.71% reduction. Mackey-Glass has a positive mean difference but its interval
contains zero. The remaining three intervals are above zero.

The Santa Fe row is an exploratory extension evaluated after the original
three-task campaign had been analyzed. It follows the same frozen protocol,
but it is not presented as a preregistered confirmation. The provenance and
selection timing are recorded in the merged manifest below.

Source:
`results/qai_robust_protocol_santa_fe_manuscript/cluster_paired_stats.csv`.
The corresponding manifest records row counts and SHA256 hashes for every
table in the paper evidence package.

The effect is largest when the execution budget contains many shots at fewer
time steps. Cov EV is close to Grouped at the primary setting, supporting the
covariance regularization interpretation of adaptive shot aggregation.
