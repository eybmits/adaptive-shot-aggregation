# Ground-truth results summary

## Experiment 1: common operating point (20 seeds)
- Mackey-Glass: EV `0.446 +- 0.103`, Raw `0.563 +- 0.076`, Split `0.414 +- 0.072`
- Lorenz: EV `0.925 +- 0.168`, Raw `0.990 +- 0.126`, Split `0.845 +- 0.142`
- NARMA10: EV `0.947 +- 0.098`, Raw `0.911 +- 0.027`, Split `0.863 +- 0.035`
- Paired EV-vs-Split deltas:
  - Mackey-Glass: `+0.032 +- 0.018` CI95 half-width, win-rate `0.75`, Wilcoxon `p=2.7e-3`
  - Lorenz: `+0.080 +- 0.055`, win-rate `0.90`, Wilcoxon `p=4.2e-3`
  - NARMA10: `+0.084 +- 0.034`, win-rate `0.85`, Wilcoxon `p=3.6e-5`
- Important nuance: at the shared operating point, Raw is already below EV on NARMA10; the defensible claim is therefore `best overall protocol under matched budget`, not uniform dominance over every baseline ordering.

## Experiment 2: empirical gap versus `rho_EV`
- Shared sweep range: `rho_EV = 0.7875` to `4.9875`
- Mackey-Glass: maximum mean gap `0.335` at `rho_EV = 0.7875`
- Lorenz: maximum mean gap `0.451` at `rho_EV = 1.8375`
- NARMA10: maximum mean gap `0.648` at `rho_EV = 1.0500`
- Peak paired EV-vs-Split stats:
  - Mackey-Glass at `N_shots=50`: `+0.335 +- 0.102`, win-rate `0.95`, Wilcoxon `p=3.8e-6`
  - Lorenz at `N_shots=25`: `+0.451 +- 0.221`, win-rate `0.95`, Wilcoxon `p=3.6e-5`
  - NARMA10 at `N_shots=40`: `+0.648 +- 0.228`, win-rate `1.00`, Wilcoxon `p=1.9e-6`

## Experiment 3: more qubits / larger reservoir
Selected mean gaps `Delta = NRMSE(EV) - NRMSE(Split)`:
- Mackey-Glass: `0.042` at `Q=4`, `0.299` at `Q=8`, `0.194` at `Q=12`
- Lorenz: `0.254` at `Q=4`, `0.732` at `Q=10`, `0.404` at `Q=12`
- NARMA10: `0.113` at `Q=4`, `0.313` at `Q=8`, `0.600` at `Q=12`
- Readout:
  - `20` seeds per point for every reported `Q`
  - paired EV-vs-Split delta is positive in `15/15` size cells and Wilcoxon-significant in `14/15`
  - the only weak corner is Mackey-Glass at `Q=4`: `+0.042 +- 0.034`, win-rate `0.65`, Wilcoxon `p=8.3e-2`
  - from `Q=6` onward, Mackey-Glass is positive and significant in every cell; Lorenz and NARMA10 are positive and significant throughout the full sweep

## Experiment 4: baseline controls at the common operating point
- Mackey-Glass: Split `0.414` remains below EV `0.446`, EV-dup `0.446`, EV-NA `0.444`, and Raw `0.563`
- Lorenz: Split `0.845` remains below EV `0.925`, EV-dup `0.916`, EV-NA `0.920`, and Raw `0.990`
- NARMA10: Split `0.863` remains below EV `0.947`, EV-dup `0.966`, EV-NA `0.937`, while Raw is `0.911`
- Paired EV-dup-vs-Split deltas stay positive on all three datasets:
  - Mackey-Glass: `+0.032`, win-rate `0.80`, Wilcoxon `p=2.7e-3`
  - Lorenz: `+0.071`, win-rate `0.65`, Wilcoxon `p=5.8e-2`
  - NARMA10: `+0.103`, win-rate `0.90`, Wilcoxon `p=1.3e-5`
- Important nuance: Split-NA is effectively tied with Split; the non-oversold claim is that Split is the strongest plain protocol and remains competitive with the diagonal noise-aware proxy.

## Experiment 5: duplication control
- Mackey-Glass: Split beats EV-dup in `7/9` shot cells
- Lorenz: Split beats EV-dup in `9/9` shot cells
- NARMA10: Split beats EV-dup in `9/9` shot cells
- Important nuance: the weakest Mackey-Glass cells are the highest-`rho_EV` end (`N_shots=10,12,15,18`); from `N_shots=20` onward, the paired delta is positive and statistically clear in every reported cell.

## Appendix: reservoir-architecture ablation
- Fixed `Q=4` architecture sweep at the common operating point (`N_shots=18`, `sigma=0`, `lambda=10`) across five variants: `ring_cnot` depth `1/2/4`, `line_cnot` depth `1`, `all_to_all_cz` depth `1`
- Global readout:
  - paired EV-vs-Split delta is positive in `15/15` architecture cells
  - Wilcoxon significance holds in `12/15` cells
- Strongest `Q=4` variant in all three datasets: `ring_cnot`, depth `2`
  - Mackey-Glass: `+0.084 +- 0.041`, win-rate `0.85`, Wilcoxon `p=2.6e-4`
  - Lorenz: `+0.258 +- 0.100`, win-rate `0.85`, Wilcoxon `p=6.3e-5`
  - NARMA10: `+0.088 +- 0.028`, win-rate `0.80`, Wilcoxon `p=8.2e-5`
- Weakest `Q=4` cells are still positive:
  - Mackey-Glass `line_cnot d=1`: `+0.021 +- 0.018`, win-rate `0.75`
  - Lorenz `line_cnot d=1`: `+0.079 +- 0.072`, win-rate `0.70`
- Targeted `Q=12` follow-up on the top-2 `Q=4` variants for Mackey-Glass and NARMA10:
  - Mackey-Glass:
    - `ring_cnot d=2`: `+0.275 +- 0.131`, win-rate `0.875`, Wilcoxon `p=1.56e-2`
    - `all_to_all_cz d=1`: `+0.150 +- 0.078`, win-rate `0.875`, Wilcoxon `p=1.56e-2`
  - NARMA10:
    - `ring_cnot d=2`: `+0.648 +- 0.127`, win-rate `1.00`, Wilcoxon `p=7.8e-3`
    - `ring_cnot d=4`: `+0.496 +- 0.090`, win-rate `1.00`, Wilcoxon `p=7.8e-3`
- Main interpretation: the split gain is not confined to the default four-qubit depth-1 ring reservoir; it remains positive across all tested architecture cells and strengthens further in the targeted `Q=12` extension.

## Imported hardware validation
- Backend: `ibm_marrakesh`
- Task: Mackey-Glass
- EV: `1.023 +- 0.235`
- Raw: `0.932 +- 0.021`
- Split: `0.670 +- 0.080`
