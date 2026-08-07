# GN100 paper data

These six CSVs are compact figure inputs, not a second analysis framework.
Each row names the evidence release/archive and SHA-256 or, for Sprints 1–5
before per-sprint releases existed, the retained source artifact and its hash.
Values with different workload bases must not be treated as a controlled
time-series; `arc-timeline.csv` carries the basis explicitly.

| File | Intended figure |
| --- | --- |
| `arc-timeline.csv` | Development arc from 0.065 tok/s to qualification |
| `phase-profiles.csv` | How the measured decode bottleneck moved |
| `mechanism-blocks.csv` | Sprint 3 A/B/C scheduling decomposition |
| `power.csv` | Task 0 calibration and Task 2 definitive wall-power rows |
| `saturation.csv` | Task 3 fio/engine storage saturation |
| `speculation-bounds.csv` | Sprints 16–18 speculative-decoding bounds |

The consolidated release attaches this directory as a checksum-anchored
archive alongside the untouched Task 0–4 evidence archives. The CSVs contain
derived summaries only; raw evidence remains authoritative.
