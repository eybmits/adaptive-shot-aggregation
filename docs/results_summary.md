# Ground-truth results summary

## Experiment 1: common operating point (20 seeds)
- Mackey-Glass: EV `0.446 +- 0.103`, Raw `0.563 +- 0.076`, Split `0.414 +- 0.072`
- Lorenz: EV `0.925 +- 0.168`, Raw `0.990 +- 0.126`, Split `0.845 +- 0.142`
- NARMA10: EV `0.947 +- 0.098`, Raw `0.911 +- 0.027`, Split `0.863 +- 0.035`

## Experiment 2: empirical gap versus `rho_EV`
- Shared sweep range: `rho_EV = 0.7875` to `4.9875`
- Mackey-Glass: maximum mean gap `0.335` at `rho_EV = 0.7875`
- Lorenz: maximum mean gap `0.451` at `rho_EV = 1.8375`
- NARMA10: maximum mean gap `0.648` at `rho_EV = 1.0500`

## Experiment 3: more qubits / larger reservoir
Selected mean gaps `Delta = NRMSE(EV) - NRMSE(Split)`:
- Mackey-Glass: `0.117` at `Q=4`, `0.320` at `Q=8`, `0.340` at `Q=12`
- Lorenz: `0.020` at `Q=4`, `0.042` at `Q=10`, `0.050` at `Q=12`
- NARMA10: `0.043` at `Q=4`, `0.208` at `Q=8`, `0.443` at `Q=12`

## Experiment 4: baseline controls at the common operating point
- Mackey-Glass: Split `0.414` remains below EV `0.446`, EV-dup `0.446`, EV-NA `0.444`, and Raw `0.563`
- Lorenz: Split `0.845` remains below EV `0.925`, EV-dup `0.916`, EV-NA `0.920`, and Raw `0.990`
- NARMA10: Split `0.863` remains below EV `0.947`, EV-dup `0.966`, EV-NA `0.937`, while Raw is `0.911`

## Experiment 5: duplication control
- Mackey-Glass: Split beats EV-dup in `7/9` shot cells
- Lorenz: Split beats EV-dup in `9/9` shot cells
- NARMA10: Split beats EV-dup in `9/9` shot cells

## Imported hardware validation
- Backend: `ibm_marrakesh`
- Task: Mackey-Glass
- EV: `1.023 +- 0.235`
- Raw: `0.932 +- 0.021`
- Split: `0.670 +- 0.080`
