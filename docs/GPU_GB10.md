# GB10 CUDA experiment

This document is the pre-registered decision record for the CUDA work on the
NVIDIA GB10 / DGX Spark / Acer Veriton GN100. The work stays on
`exp/cuda-gb10` until both its numerical and integrated performance gates pass.
It is not an upstream proposal and it is not part of the qualified
`spark/integration` profile.

## What must be established first

Moving attention arithmetic to CUDA can perturb the residual stream before
the next router. Routing is discrete: a small change at the top-16 boundary
can select a different expert, change I/O and cache state, and then cascade.
Kernel speed is therefore not enough to establish equivalence.

The actual K3 selection score is

```text
sigmoid(raw router logit) + correction bias
```

`WASTE_DUMP_ROUTE` records selected, renormalized weights and has no rank-17
entry, so it cannot recover that boundary. Commit `b2944b5` added the separate
`WASTE_DUMP_ROUTE_MARGIN` CSV and `tools/route_margins.py`; enabling it is a
correctness capture, never a timed performance row.

The earlier SDOT maximum difference (`0.0589654`) was in final vocabulary
logits. It is not a router-score error and is not dimensionally comparable to
the margins below. CPU/CUDA router deltas and route sets must be measured
directly once a CUDA path exists.

## Route-margin result

The fixed capture used K3 manifest SHA-256
`da3232925d8eeaf0df07ee45743f6467c7eecbc4c0e2513899028e788a1a07f2`,
usage hotlist SHA-256
`e2949e8d14da3c46d766daa99e6b93f946c8cfc0425eba4815450079996899dd`,
prompt IDs `1008,10484,318,15383,387`, and 64 greedy decode tokens. It ran
under the qualified eight-thread, CPU-set, Q0, Q8, two-reader/depth-two,
lookahead-zero profile. The run produced the established 0.34 tok/s class
(`0.342234`), but tracing disqualifies that row from performance comparison.

There were 6,348 boundary decisions: 460 from the five-token chunked prefill
and 5,888 from decode, across 68 KDA and 24 MLA routed layers. There were no
ties and no non-finite scores.

| margin statistic | score gap |
| --- | ---: |
| minimum | `4.17e-7` |
| p0.1 | `1.16e-6` |
| p1 | `9.77e-6` |
| p5 | `5.04e-5` |
| median | `8.21e-4` |
| p95 | `6.33e-3` |
| p99 | `1.46e-2` |
| maximum | `4.63e-2` |

If every expert score can move independently by at most `epsilon`, margin
alone certifies a route only when the gap is greater than `2*epsilon`:

| per-score error bound | decisions not certified invariant |
| ---: | ---: |
| `1e-6` | 14 / 6,348 (0.221%) |
| `1e-5` | 126 / 6,348 (1.985%) |
| `1e-4` | 1,122 / 6,348 (17.675%) |
| `1e-3` | 4,812 / 6,348 (75.803%) |

These are upper-bound vulnerability counts, not predicted flip counts; score
errors are correlated and may cancel. They do establish that route invariance
cannot be assumed. The compact machine-readable result is
[gn100/sprint11-route-margin-summary.json](gn100/sprint11-route-margin-summary.json).
The 1.1 MiB raw CSV is retained in the Spark evidence directory with SHA-256
`b4c67249af0f3d565a19510ae51e4548c782f729b7d10633c0d3e92ba181524d`.

## Amdahl result

Commit `867c2c2` separated the recurrent update from the existing full-KDA
profile bucket. On the fixed 16-token CPU profile:

| phase | cumulative time | accounted decode share |
| --- | ---: | ---: |
| full KDA layers | 16.10 s | 26.7% |
| KDA recurrence only | 0.24 s | 0.4% |

The recurrent update is roughly 1.5% of KDA time. Even making it free cannot
clear the 5% integrated engine gate. A recurrence kernel remains useful for
validating CUDA arithmetic and handoff, but the performance pilot must capture
the whole KDA layer: projections, convolutions, decay/beta path, recurrence,
gated normalization, and output projection.

## GB10 memory result

The standalone CUDA 13 probe at `dc165d5` found compute capability 12.1 and:

- coherent pageable-memory access through host page tables;
- concurrent managed access and host-native atomics;
- the same virtual pointer for registered host memory;
- 64-bit stream memory operations;
- no direct host access to device-prefetched managed memory.

For a 6 MiB one-layer state repeatedly resident in cache, aligned pageable and
registered host pointers took about 2.0 ms for 100 passes; managed-prefetched
and device-resident pointers took about 1.05 ms. A full explicit 6 MiB
host-device-kernel-host round trip took 0.233 ms. These are pointer-path
screening numbers, not LPDDR bandwidth claims: repeated access to 6 MiB is
cache-resident. They say which paths deserve the real KDA benchmark, not how
fast the model will run.

## Numerical contracts

The strict contract remains the promotion gate:

- zero changed top-16 route memberships over the fixed corpus;
- identical greedy tokens and per-step argmax;
- existing maximum absolute logit error at or below `0.01`;
- no NaNs, changed expert bytes, or silent CPU fallback.

Because the margin study found real mass near zero, a second contract is
pre-registered before any CUDA model result is seen. It allows an experiment
to continue for performance diagnosis but does **not** promote it into
`spark/integration` or make it suitable for upstream:

- changed route sets in at most 1% of decode layer-token rows (58 of 5,888);
- at most 0.1% membership replacements over all selected decode slots
  (94 of 94,208), and no more than two replacements in one row;
- identical 64-token greedy sequence and per-step argmax;
- identical top-10 logit set at every step;
- per-step maximum absolute logit error at most `0.1` and mean absolute error
  at most `0.005`;
- no NaNs; expert bytes, hit/miss counts, and route-linked byte changes must be
  reported, with bytes read within 1% of control for a compute-speed claim.

Those limits are an operational experimental boundary, not a general quality
theorem. Failing them ends decode integration until a stricter arithmetic path
or a separate quality study is designed.

## Pilot gates and fallback ladder

1. Implement the precise recurrence kernel without `--use_fast_math`, using a
   coarse all-head dispatch so eight CPU pool workers cannot launch it eight
   times. Measure direct launch, CUDA Graph launch, event wait, and mapped-flag
   polling. This validates state access and numerical instrumentation; it is
   not the end-to-end speed target.
2. Capture a fixed-shape whole-KDA-layer graph. Kernel-only and host-observed
   median time must each be at least 2x faster than the qualified CPU layer.
3. Add a runtime one-load A/B arm, CUDA warmup, per-step logit deltas, ordered
   route hash, route-set replacements, and explicit fallback reporting.
4. The interleaved integrated campaign must improve median decode throughput
   by at least 5%, with zero faults/swap/thermal invalidation and one of the
   numerical contracts above.
5. Measure storage contention separately: pair the active GPU loop with the
   existing 11 MiB O_DIRECT/QD2 fio workload and compare bandwidth plus p95/p99
   latency. Nsight runs are diagnostic and never promotion rows.

If the whole-layer graph loses primarily to launch/handoff overhead, increase
granularity to a multi-layer or whole-block graph. If decode shapes remain too
small, restrict CUDA to chunked prefill. Only after dense attention clears its
gate does VQ expert gather become the stretch target. If decode offload is
still uneconomic, speculative decoding is the next architectural experiment;
its batch dimension may make a later GPU attempt materially different.
