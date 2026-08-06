# GB10 CUDA experiment

This document is the pre-registered decision and result record for CUDA work on the
NVIDIA GB10 / DGX Spark / Acer Veriton GN100. The work stays on
`exp/cuda-gb10` until it is deliberately promoted, even when its numerical and
integrated performance gates pass. It is not an upstream proposal and it is
not part of the qualified `spark/integration` profile.

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
profile bucket. Commit `32a3b6b` then split the layer into the phases a CUDA
graph would have to replace. On the fixed 16-token CPU profile:

| phase | cumulative time | accounted decode share |
| --- | ---: | ---: |
| full KDA layers | 16.10 s | 26.7% |
| KDA recurrence only | 0.24 s | 0.4% |

The recurrent update is roughly 1.5% of KDA time. Even making it free cannot
clear the 5% integrated engine gate. A recurrence kernel remains useful for
validating CUDA arithmetic and handoff, but phase profiling must identify a
coarse enough portion of the KDA layer to clear the integrated gate.

The detailed repeat measured 1,449 KDA layer calls across five chunked-prefill
and 16 decode tokens. Its sub-totals normalize to about 14.6 ms per layer:

| KDA sub-phase | cumulative time | time per KDA layer |
| --- | ---: | ---: |
| q/k/v Q4 projections | 11.75 s | 8.11 ms |
| output-gate Q4 projection | 3.92 s | 2.71 ms |
| output Q4 projection | 4.18 s | 2.89 ms |
| auxiliary projections + decay/beta | 0.52 s | 0.36 ms |
| three short convolutions | 0.47 s | 0.32 ms |
| recurrence | 0.22 s | 0.15 ms |
| gated output norm | 0.05 s | 0.03 ms |

The five large Q4 projections account for 19.85 of 21.11 seconds of detailed
KDA time (94%). The next pilot therefore targets the real Q4G/group-128 trunk
format, not recurrence in isolation. Raw profile SHA-256:
`85973216030dd136f68c32301c6603c1793611ccd11c777c0322d1c9dcab2829`.

## Recurrence and handoff pilot

The standalone GB10 recurrence kernel is numerically close to the precise CPU
reference: maximum absolute error was `5.59e-9` for output and `2.98e-8` for
the recurrent state, with no non-finite values. For 96 heads of 128 x 128
state, its matched kernel-only control and CUDA paths were:

| path | time per call | speedup over persistent 8-thread CPU control |
| --- | ---: | ---: |
| CPU control | 0.0529 ms | 1.00x |
| CUDA direct, event timed | 0.0274 ms | 1.93x |
| CUDA Graph, event timed | 0.0287 ms | 1.84x |
| CUDA Graph, host synchronize | 0.0304 ms | 1.74x |
| CUDA Graph, mapped-flag poll | 0.0308 ms | 1.72x |

This narrowly misses a 2x isolated screen and graph launch does not help a
single kernel. More importantly, recurrence is only about 1.5% of KDA time,
so it is rejected as a standalone speed feature while retained as the
correctness and handoff building block for a coarse graph. Raw result SHA-256:
`d6d3ab154ea7ceaaa96980010be7a2a1ad43edae33f5906eca100641708762b9`.

## GB10 memory result

The corrected standalone CUDA 13 probe at `f220fab` found compute capability
12.1 and:

- coherent pageable-memory access through host page tables;
- concurrent managed access and host-native atomics;
- the same virtual pointer for registered host memory;
- 64-bit stream memory operations;
- no direct host access to device-prefetched managed memory.

For a 6 MiB one-layer state repeatedly resident in cache, aligned pageable and
registered host pointers took 1.97 and 1.92 ms for 100 passes;
managed-prefetched and device-resident pointers took 1.09 and 1.05 ms. A full explicit 6 MiB
host-device-kernel-host round trip took 0.233 ms. These are pointer-path
screening numbers, not LPDDR bandwidth claims: repeated access to 6 MiB is
cache-resident. They say which paths deserve the real KDA benchmark, not how
fast the model will run. Raw result SHA-256:
`0751ffb86cded8ce4a37efee64b25bbe1561fcef5197010ae84222d1ec772c02`.

## Real Q4 projection pilot

The next pilot read K3's actual Q4G/group-128 matrices directly from
`trunk.bin`. The candidate path leaves the 29 GiB trunk in ordinary pageable
host memory and lets GB10's host-page-table support read it directly. Only the
small activation and result vectors use mapped pinned staging. It therefore
does not create a second trunk allocation behind the expert cache's back.

Two representative 43.4 MiB projection scans at commit `4662c46` measured:

| projection | CPU control | CUDA fast | fast speedup | CUDA CPU-order | CPU-order speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12,288 x 7,168 | 3.080 ms | 0.352 ms | 8.74x | 1.239 ms | 2.49x |
| 7,168 x 12,288 | 3.081 ms | 0.368 ms | 8.38x | 1.644 ms | 1.87x |

The fast kernel's maximum element error against the four-accumulator CPU
control was `2.53e-7`; the diagnostic CPU-order kernel was byte-exact. Direct
launch, host synchronization, graph-plus-poll, and registered-host results
were effectively tied for the fast kernel. This rejected two assumptions at
once: GB10 pageable access is fast enough for the trunk, and a graph is not
needed merely to amortize one projection this large. The engine pilot could
therefore stay small: offload the eight decode-time KDA Q4 projections and
leave convolution, recurrence, normalization, MLA, MoE, prefill, and the
language-model head on the CPU.

## Integrated CUDA path

The experimental implementation is opt-in at both build and runtime:

```sh
make clean
make -j8 WASTE_NATIVE=1 WASTE_ENABLE_CUDA=1
WASTE_CUDA_KDA=1 ./waste ...
```

`WASTE_CUDA_KDA=1` selects the fast reduction. Mode `2` retains the
CPU-order kernel for diagnosis, and `WASTE_BACKEND=cpu` forces the control
path even if the CUDA variable is present. A clean build is required when
toggling `WASTE_ENABLE_CUDA`.

The path performs 69 KDA layers times eight projections, or 552 CUDA calls per
decode token. It synchronizes after each projection; no graph or CPU/GPU
overlap is claimed. Preflight validates every selected projection's format
before decode and warm-tests the real kernel outside the timer. A runtime
failure aborts the token rather than mixing GPU and CPU arithmetic, poisons
the partially changed recurrent
state, blocks prefill/decode/snapshot export, and requires a reset for CPU
recovery after selecting CPU mode, or a model reload before CUDA can be
attempted again. Captures record requested and effective mode, fallbacks,
actual calls, and expected calls. The comparator refuses missing metadata,
missing decode routes, or a call shortfall.

The balanced two-repeat, 16-token campaign at `cb53b2c` produced:

| arm | repeat 1 | repeat 2 | median |
| --- | ---: | ---: | ---: |
| qualified CPU | 0.353799 tok/s | 0.341846 tok/s | 0.347823 tok/s |
| CUDA fast | 0.477685 tok/s | 0.500151 tok/s | 0.488918 tok/s |

That is a 40.6% median improvement. Every CUDA arm executed exactly 8,832
projections with zero fallbacks. Both arms had the same token and route hashes,
12,526 hits, 11,026 misses, and 136,797,200,384 expert bytes read. The whole
campaign had zero major faults and zero swaps.

The fully audited final commit is `378f881`. Its short capture was deliberately
rerun after the acceptance hardening, not reused from an earlier binary:

| arm | throughput | calls | fallbacks |
| --- | ---: | ---: | ---: |
| qualified CPU | 0.348210 tok/s | 0 | 0 |
| CUDA fast | 0.473351 tok/s | 2,208 / 2,208 | 0 |

The final short proof improved throughput by 35.9%, preserved all 368 route
sets (zero replacements in 5,888 selected slots), preserved every greedy
argmax and top-10 set, and had maximum absolute logit error `1.19e-5` and
maximum per-step mean error `1.68e-6`. It also recorded zero major faults and
zero swaps. The longer final strict-contract result is recorded below.

## Shared-memory contention

The storage screen paired 12 MiB random O_DIRECT reads at queue depth two with
an 85-second fast-Q4 loop. The GPU loop continuously scanned a 43.4 MiB real
K3 projection at 123.3 GiB/s of logical model traffic through the same
pageable/HMM path as the engine. This is a controlled near-100%-duty stressor,
not a claim about physical DRAM counters or the engine's bursty duty cycle.

| arm | bandwidth | p95 completion | p99 completion |
| --- | ---: | ---: | ---: |
| no GPU, before | 13.394 GB/s | 1.548 ms | 1.647 ms |
| sustained GPU | 12.857 GB/s | 1.958 ms | 2.179 ms |
| no GPU, after | 13.393 GB/s | 1.630 ms | 1.729 ms |

The two bracketing baselines agree within 0.004% on bandwidth. Against their
mean, sustained GPU traffic reduced SSD bandwidth by 4.0%, raised p95 by
23.2%, and raised p99 by 29.1%. Shared LPDDR contention is therefore real but
bounded in this deliberately harsher-than-engine screen. It is not hiding in
the integrated speed result: the matched engine arms read identical expert
bytes, and CUDA remained 35.9-40.6% faster after that traffic was included.

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

## Strict-contract result

The final 64-token capture at `378f881` passed the strict contract, so the
relaxed contract was not needed:

| measurement | CPU | CUDA fast |
| --- | ---: | ---: |
| decode time | 189.752 s | 134.501 s |
| throughput | 0.337282 tok/s | 0.475833 tok/s |
| CUDA calls | 0 / 0 | 35,328 / 35,328 |
| fallbacks | 0 | 0 |
| expert hits / misses | 43,435 / 50,773 | 43,435 / 50,773 |
| expert bytes read | 629,929,644,032 | 629,929,644,032 |

CUDA improved throughput by 41.1%. All 65 causal logit rows were comparable.
Across 5,888 routed layer-token rows and 94,208 selected expert slots there
were zero membership changes, zero order-only changes, and zero replacements.
Every greedy argmax and top-10 set was unchanged. Maximum absolute logit error
was `3.62e-5`, maximum per-step mean error was `2.59e-6`, and there were no
non-finite values. The combined process recorded zero major faults, zero
swaps, and a 79.9 GiB peak RSS. The PM-QoS holder and child both exited
cleanly.

This is stronger than the 5% engine gate and stays well inside the `0.01`
strict logit bound. It also reproduces the earlier 64-token hashes exactly,
which confirms that the later failure and acceptance hardening did not change
the arithmetic path. Exact settings, counters, artifact hashes, and fio values
are retained in
[gn100/sprint11-gpu-summary.json](gn100/sprint11-gpu-summary.json).

## Pilot-gate disposition and fallback ladder

1. **Recurrence screen: rejected alone.** It reached 1.93x kernel-only, missed
   the 2x screen, and represented only about 1.5% of KDA time.
2. **Whole-layer graph: not built; re-scoped by measurement.** Profiling
   showed that five large Q4 projections were 94% of KDA time. Representative
   real matrices exceeded 8x on the fast kernel, while graph launch did not
   improve a single projection. We explicitly deferred the original
   whole-layer-graph pilot and tested the smaller projection-only integration;
   the engine result below shows that it was already coarse enough.
3. **Runtime and instrumentation: passed.** One-load balanced arms, untimed
   preflight/warmup, full logits, ordered and set route comparisons, exact
   CUDA call counts, and fail-closed behavior are all present.
4. **Integrated engine: passed.** The repeated campaign improved the median by
   40.6%; the final 64-token proof improved by 41.1%, passed the strict
   numerical contract, and recorded zero major faults and swaps. The repeated
   campaign's GPU endpoint moved from 47 to 48 C, so there was no thermal
   invalidation in the qualified comparison.
5. **Storage contention: measured.** A sustained worst-case loop costs 4.0%
   bandwidth and 23-29% p95/p99 latency against stable bracketing baselines.

KDA dense decode cleared its gate. Sprint 12 then expanded the same kernel
class through the conventional dense Q4 projections. Cumulative scope 2
passed the stronger ordered-route contract at 0.673474 tok/s; scope 3 remains
diagnostic. Results and exact evidence hashes are in
[GPU_DENSE_GB10.md](GPU_DENSE_GB10.md) and
[gn100/sprint12-dense-gpu-summary.json](gn100/sprint12-dense-gpu-summary.json).

Sprint 13 moved expert VQ LUT construction and gather to CUDA while retaining
SiTU and router-ordered weighted accumulation on the CPU. Mode 2/group 1
passed a byte-exact 64-token comparison at 0.901801 tok/s versus 0.707878
tok/s with CPU VQ. A capture-trained, explicitly in-sample studio profile
reached a repeated 1.003615 tok/s median over 16 tokens; its 64-token endpoint
was 0.944126 tok/s. Grouped synchronization was rejected because the reduced
wait count exposed more expert I/O and made every group larger than one
slower. See [GPU_VQ_GB10.md](GPU_VQ_GB10.md) for the contract, profile split,
and storage-contention result. Whole-layer graphs, CPU/GPU overlap, and
prefill CUDA remain possible follow-ons, but each must be justified by a
measured profile rather than added by default.
