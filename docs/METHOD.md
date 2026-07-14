# Adaptive shot aggregation

## The design choice

Assume a circuit is evaluated at `T` time steps with `N` shots at every step.
Each shot produces one feature vector. The same finite record supports three
training representations:

- **Raw** uses one shot per training view, so `k = 1`.
- **Grouped** averages disjoint groups of `k` shots, producing `N / k` views.
- **Expectation value (EV)** averages all shots, so `k = N` and there is one
  view per time step.

The quantum execution is identical in every case. Only the classical
organization of the retained outcomes changes.

```python
import numpy as np


def grouped_views(shots: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Return grouped means from shots shaped (time, shot, feature)."""
    time_steps, n_shots, features = shots.shape
    if n_shots % k:
        raise ValueError("k must divide the number of shots")
    shuffled = np.empty_like(shots)
    for t in range(time_steps):
        shuffled[t] = shots[t, rng.permutation(n_shots)]
    return shuffled.reshape(time_steps, n_shots // k, k, features).mean(axis=2)
```

Every view at a time step receives the same forecasting target. Candidate
values of `k` and the ridge penalty are compared on chronological development
folds. The chosen settings are frozen before the held-out test block is scored.
Raw and EV remain available as safe endpoints of the same search.

## Why it can help

Full averaging gives the readout one precise view of each target. Smaller
groups give it several noisier views. For a linear readout with squared loss,
training on all grouped views is exactly the EV loss plus a quadratic penalty
formed from the variation between the views. Feature combinations that vary
strongly across groups receive a larger penalty. The aggregation scale controls
the strength and realization of that regularization.

This interpretation explains the empirical regime in the paper. Grouping is
most useful when many shots are available at relatively few time steps. When
shots per step are already scarce, there is little room between Raw and EV and
the benefit becomes smaller.

## Practical checklist

1. Keep individual outcomes rather than storing expectation values alone.
2. Compare every admissible divisor of `N`, including `1` and `N`.
3. Form groups independently at each time step because shot indices have no
   physical identity across steps.
4. Fit scaling inside each training fold.
5. Tune with earlier data and reserve the final test block for one evaluation.
6. Average grouped predictions to one forecast per time step before scoring.
7. Quantify uncertainty over independent trajectories or windows, not over
   correlated views of the same record.

The complete implementation is in `scripts/robust_protocol.py` and
`scripts/run_robust_evaluation.py`. Tests in `tests/` cover the grouping,
chronological selection, covariance identity, and fail-closed provenance
contracts.
