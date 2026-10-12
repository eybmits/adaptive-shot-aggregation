# Data provenance

The repository contains two measured time series and generated or archived
quantum-measurement records. The MIT License covers the repository's software
and original documentation. Third-party datasets retain the terms of their
providers.

## Santa Fe laser

Files:

- `data/santa_fe_laser_A.dat`, 1,000 rows, SHA-256
  `14faec41629ead4f45d9c055a4494269dd9122dd559594502c0c522dc98a9eb2`
- `data/santa_fe_laser_A_continuation.dat`, 9,093 rows, SHA-256
  `feebf2460cb8c945b87dde368cccde95a05ee06955827c4bec93f2043db6ec5e`

The series is the laser-intensity benchmark from the Santa Fe Institute time
series competition. The paper cites N. A. Gershenfeld and A. S. Weigend,
"The Future of Time Series," in *Time Series Prediction: Forecasting the
Future and Understanding the Past*, 1993.

Sources:

- <https://web.cecs.pdx.edu/mcnames/DataSets/SantaFeA.dat>
- <https://web.cecs.pdx.edu/mcnames/DataSets/SantaFeA2.dat>
- <https://web.cecs.pdx.edu/mcnames/DataSets/index.html>

The frozen metadata does not assert a separate redistribution license for
these benchmark files. They are included to reproduce the published analysis,
remain attributed to their provider, and are not relicensed by this repository.

## Sunspots

File: `data/sunspots_monthly_sidc.csv`, 3,330 rows from 1749-01 through
2026-06, SHA-256
`afb272b758d8db30446e43930091a98dbcf09210086da7fac8d5337f5532f68c`.

The monthly mean total sunspot number is provided by the SILSO World Data
Center of the Royal Observatory of Belgium. The paper cites the SILSO data
record and version used by the artifact.

Source: <https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv>. The file is SILSO
monthly mean total sunspot number, version 2.0. The
[SILSO monthly-data page](https://www.sidc.be/SILSO/infosnmtot) identifies the
[Creative Commons Attribution-NonCommercial 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).
Credit: SILSO World Data Center, Royal Observatory of Belgium, Brussels.
Retain that attribution and license link, and indicate changes when adapting
the data. The MIT License does not apply to this third-party record.

This repository pins the 3,330-row snapshot and date range above rather than
refreshing from the live source. Analysis filters missing values and applies
training-fitted normalization in memory; the included source file is not
rewritten by those steps. See the experiment configurations for the exact
window and preprocessing settings.

## Simulated trajectories

Mackey-Glass and Rossler trajectories are generated from frozen protocols,
including `configs/mackey_glass_primary.json`. Generator settings and seeds are recorded in the
result manifests.

## Quantum measurement records

Archived IBM measurement records and their layouts are stored under
`results/qai_hardware/`. Job identifiers, processors, shot budgets, hashes, and
the interpretation boundary are documented in `docs/HARDWARE.md` and
`results/qai_hardware_santa_fe_extension/manifest.json`.

The consolidated simulation evidence is stored under
`results/qai_robust_protocol_santa_fe_manuscript/`. Its manifest records the
source artifacts, row counts, and SHA256 hashes used by the paper.

The manifest binds every consolidated table to its source artifact, row count,
and SHA256 hash.
