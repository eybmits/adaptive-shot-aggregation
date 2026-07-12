# Hardware records

The paper uses unmitigated ordered measurements from two IBM processors. Each
task has 500 time steps and 400 shots per step. The analysis divides the shots
into twenty consecutive records with 20 shots per time step and applies the
same chronological joint-tuning procedure used in simulation.

The device experiment uses a different reservoir realization from the main
simulation. Simulation uses a six-qubit, two-layer data reuploading circuit and
eleven local Z and neighboring ZZ features. Hardware uses a ten-qubit,
depth-one transverse field Ising ring with one amplitude-encoded input qubit, nine
memory qubits, and ten local Z features. The hardware experiment asks whether
the aggregation procedure transfers to another fixed circuit. It is not a
hardware reproduction of the six-qubit simulation and does not compare qubit
counts. EV, Grouped, and Raw are always compared within the same circuit record.

| Processor | Santa Fe job | QPU time |
|---|---|---:|
| `ibm_fez` | `d99mf4t2su3c739kl9k0` | 55 s |
| `ibm_marrakesh` | `d99mf52f47jc73a99fj0` | 56 s |

The raw Santa Fe records are stored once in each processor directory under
`results/qai_hardware/`. The acquisition manifest with budget, job state,
hashes, and archived layouts is
`results/qai_hardware_santa_fe_extension/manifest.json`.

The twenty records are parts of one acquisition. They quantify variation
across shot blocks, not across devices, jobs, calibration periods, or hardware
campaigns. Task comparisons are not used as evidence for calibration
robustness.

Reanalyze the archived records without IBM access:

```bash
make hardware
make verify
```
