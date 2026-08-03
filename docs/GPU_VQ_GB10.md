# GB10 CUDA VQ gather: sprint 13 preregistration

Sprint 12 selected `WASTE_CUDA_KDA=1 WASTE_CUDA_DENSE=2` at 0.673474
tok/s over the fixed 64-token K3 corpus. Its post-expansion profile assigns
51.0% of accounted decode time to expert VQ LUT application. Sprint 13 tests
that kernel class without changing routing, expert-cache policy, model format,
or the qualified `spark/integration` branch.

The work stays on `exp/cuda-vq-gb10`. One token evaluates 92 MoE layers times
16 experts: 1,472 expert triplets, 4,416 VQ matrix applications and 1,656
semantic LUT builds. K3's blocked VQ3R payload performs 18,232,639,488 LUT
gathers and reads the same number of index bytes per token. The logical core
traffic is about 84.9 GiB/token, but the 3.75 MiB live LUT working set should
remain GPU-cache resident. All static K3 expert codebooks occupy only about
6.47 MiB.

At the measured 1.4848 seconds/token baseline, LUT application accounts for
about 0.757 seconds. Reaching 1 tok/s with the rest of the engine unchanged
requires reducing that phase to about 0.272 seconds: a 2.8x component
speedup. One tok/s is the stretch target, not permission to weaken the
correctness contract or discard a smaller repeatable gain.

## Runtime arms

`WASTE_CUDA_VQ` is a separate decode-time experimental control. It requires
`WASTE_CUDA_KDA=1` and `WASTE_CUDA_DENSE=2`; it does not widen either older
variable's meaning.

| mode | GPU work | CPU work retained | purpose |
| ---: | --- | --- | --- |
| 0 | none | existing LUT build, apply, SiTU and reduction | matched control |
| 1 | VQ LUT application | LUT build, SiTU and router-ordered reduction | isolate the gather |
| 2 | LUT build and application, with device-resident codebooks | SiTU and router-ordered reduction | intended strict candidate |
| 3 | mode 2 plus SiTU inside the per-expert stream | router-ordered reduction | registered fallback for measured handoff/activation cost |

Mode 3 is built only if mode 2 remains below 1 tok/s and a post-mode-2 trace
attributes at least 5% of token time to the retained activation or the second
expert synchronization. It is not introduced merely because it exists in
this table. GPU router arithmetic and atomics/tree reductions across experts
remain out of scope in every mode.

Modes 1 and 2 return each expert's down vector to the CPU. The existing
`j = 0..15` weighted accumulation remains byte-for-byte the same source loop.
Mode 1 also retains CPU LUT arithmetic, providing a clean fallback if device
LUT construction fails numerical acceptance.

## Data ownership and launch shape

- Expert records stay in the existing bounded host expert cache and are read
  directly through GB10 pageable host-page-table access. The experiment must
  not duplicate 17.01 GiB/token into a second device cache or register the
  complete expert cache outside its budget.
- Modes 2 and 3 copy the static codebooks to one approximately 6.47 MiB device
  allocation when the CUDA context is created. Three reusable LUT buffers
  consume approximately 3.75 MiB. Small activation/output staging remains
  bounded by the existing CUDA context.
- One apply thread owns one output row. A 64-row CTA follows the container's
  index block, walks vector positions in ascending order and reads the three
  adjacent stage bytes for each row. No output-row reduction is needed.
- Gate and up share their layer LUTs across all 16 experts. Down's LUT is
  rebuilt once per expert because its activation differs.
- The first implementation synchronizes before the next expert-cache pointer
  can become evictable. Launch count, host synchronization time and kernel
  time are measured separately. Cache pinning or multi-expert batching is a
  later optimization, not an assumption in the pilot.

## Arithmetic contract

The strict apply kernel preserves the CPU scalar dependency chain for each
row:

1. vector positions are consumed from zero upward;
2. the stage term is `(stage0 + stage1) + stage2`;
3. the term is added to the row accumulator in that order; and
4. the final accumulator is multiplied once by the original f16 row scale.

Device LUT construction consumes dimensions zero through seven in order and
uses explicit `fmaf` for the same per-code chain as the NEON CPU builder.
Compilation disables unintended contraction around the remaining additions.
Synthetic tests include normal, negative, minimum-normal and subnormal f16
scales, K3 gate/up and down shapes, partial row blocks, nonzero ranges and a
VQ2 rejection/fallback case.

Mode 3 retains the same VQ term order but adds CUDA `tanhf`/`expf` arithmetic.
It therefore has to clear the complete model contract independently; a mode-2
pass does not grandfather it.

## Failure contract and accounting

The existing sticky CUDA failure domain extends to VQ. Any failed prepare,
launch, synchronization or copy:

- aborts the current token;
- marks recurrent state dirty;
- sets every CUDA effective mode to zero;
- increments the common failure counter; and
- forbids CPU fallback until model reload/reset.

Preflight validates VQ3R, `stages=3`, `entries=256`, `vec_dim=8`,
`index_block=64`, K3 matrix geometry, codebook bounds and the selected dense
baseline. It creates the CUDA context, uploads codebooks and executes an
untimed warmup before capture. Record headers and offsets retain their normal
per-read validation.

Counters are semantic so later fusion cannot rewrite the acceptance target:

| counter | per token | over 64 decode tokens |
| --- | ---: | ---: |
| expert triplets completed | 1,472 | 94,208 |
| VQ matrix applications | 4,416 | 282,624 |
| LUT builds, mode 2/3 | 1,656 | 105,984 |

Actual kernel launches and synchronizations are reported separately. Every
accepted capture requires exact semantic counts and zero fallback.

## Gates and sequence

1. **Standalone real-layout pilot.** Use K3 gate/up `[3072,3584]` and down
   `[3584,3072]` records in pageable, pre-touched host memory. Compare against
   the current eight-thread CPU path in alternating order. Report kernel-only
   and launch-plus-sync timing. Both apply shapes must reach at least 3.0x in
   launch-plus-sync time, remain finite and pass the synthetic arithmetic
   checks. A result below 2.0x ends integration; 2.0-3.0x permits one measured
   launch/layout revision before a decision.
2. **Two-token smoke.** Run modes 0, 1 and 2 after one model load. Require
   exact semantic counters, zero fallback, identical token and route hashes,
   zero faults/swaps and clean Q0 teardown.
3. **Balanced 16-token campaign.** Alternate modes forward then reverse for
   two repeats in one load. Mode 1 must improve median engine throughput by at
   least 5% over mode 0. Mode 2 must improve at least 5% over mode 1 or explain
   a measured component saving that is hidden by campaign variance; it may
   not regress the cumulative candidate.
4. **Strict 64-token capture.** The selected mode must retain identical greedy
   tokens, per-step argmax and top-10 sets; zero changed top-16 memberships;
   zero ordered-route changes; maximum absolute logit error at most `0.01`;
   no non-finite values; identical expert hits, misses and bytes; zero major
   faults/swaps; and clean Q0/child exit. A membership-exact but order-changing
   mode remains diagnostic and is not the selected sprint-13 profile.
5. **Target and profile.** Report both the matched improvement and whether the
   repeated median and 64-token endpoint reach 1.0 tok/s. Profile the best
   strict mode before considering mode 3, graphs, cache pinning or overlap.
6. **Contention.** If a VQ mode passes the engine gate, repeat the engine-level
   same-SSD fio screen because the GPU now reads the same expert-cache pages
   that the direct-I/O readers fill.

No result from this sprint is promoted into `spark/integration` or proposed
upstream automatically. Publication means an `exp/*` branch, exact source and
model hashes, a machine-readable summary and immutable raw evidence.

## Post-pilot amendment: fixed-budget hotlist and grouped synchronization

The standalone pilot passed bit-exactly on 16 distinct records at 9.85-11.93x
for the individual apply shapes. Integrated mode 2 then exposed the next
bottleneck rather than reproducing that factor at engine level: the original
3,008-entry hotlist sustains about 0.81 tok/s over eight decode tokens, while
expert I/O grows as the CPU VQ work that used to hide it disappears. Mode 3's
measured non-LUT residual is only about 3.5% of token time, below its registered
build trigger, so mode 3 remains deferred.

A diagnostic hotlist built from the existing fixed 64-token route capture
fills all 4,495 slots at the unchanged 53,196 MiB cache budget. Its top-4,495
entries cover 58.82% of captured selections, versus 49.28% for the first 3,008.
Matched two-repeat results are 0.903/0.923 tok/s against 0.803/0.807 tok/s for
the original hotlist, with 56.92 GB versus 72.96 GB read over eight tokens.
This is an in-sample ceiling result, not a general cache claim. The converter
and hotlist are retained because representative recurring prompt families are
a real target workload. Sprint 14 subsequently used a separately captured
calibration set and frozen held-out prompt families; its decode-only aging arm
missed the registered selection gates and was not promoted. See
[GPU_VQ_HELDOUT_GB10.md](GPU_VQ_HELDOUT_GB10.md).

The remaining mode-2 micro-arm changes scheduling only. Experimental
`WASTE_CUDA_VQ_GROUP` values 1, 2, 4, 8 and 16 queue already-validated kernels
and synchronize twice per expert group rather than twice per expert. Kernel
count, LUT construction, SiTU, expert weights, and the CPU `j = 0..15`
accumulation order do not change. Expert-cache records receive explicit holds
until the group's stream work is drained; no expert bytes or LUTs are copied
into a second cache. Mode 1 remains group 1.

For K3, the registered synchronization counts are 2,944, 1,472, 736, 368 and
184 per token for groups 1, 2, 4, 8 and 16 respectively; launch count remains
4,508. Each group arm must retain byte-identical logits, routes and tokens,
the same expert hit/miss/byte counters, exact semantic and launch/sync counts,
zero fallback, clean held-slot release on errors, and clean Q0 teardown. A
group is selected only if two matched repeats improve median throughput by at
least 5% over group 1. Group sizes are tested in ascending order and the work
stops once larger groups reduce I/O/compute overlap or no longer improve the
median. One tok/s remains the stretch target rather than an acceptance waiver.

## Measured result

The strict CUDA kernel passed. On 16 distinct real K3 expert records, the
standalone pilot preserved the CPU scalar dependency chain bit-for-bit and
improved the individual apply shapes by 11.02-16.85x in the archived
confirmation. The integrated mode-2
path completed all three expert matrices on CUDA, retained SiTU and the final
weighted sum on the CPU, and never copied an expert record into a second
cache. Mode 3 was not built: the retained activation/handoff share measured
about 3.5%, below its registered trigger.

The definitive lookahead-off 64-token comparison used ten compute threads, a
59,340 MiB expert cache, two direct-I/O readers at depth two, the
capture-derived hotlist, and group one:

| measurement | CPU VQ | CUDA VQ mode 2 |
| --- | ---: | ---: |
| decode time | 90.411068 s | 70.969114 s |
| throughput | 0.707878 tok/s | **0.901801 tok/s** |
| expert hits / misses | 56,136 / 38,072 | 56,136 / 38,072 |
| expert bytes | 472,351,080,448 | 472,351,080,448 |

That is a 27.4% engine improvement. Across 65 causal logit rows, 10,649,600
finite logit pairs, 5,888 routed rows and 94,208 selected expert slots, the
captures were byte-identical: zero changed logits, tokens, top-ten sets,
route memberships, route orders or expert replacements. The CUDA arm
reported exactly 94,208 experts, 282,624 applications, 105,984 LUT builds,
288,512 launches and 188,416 synchronizations, with zero fallback. The
process and Q0 holder exited cleanly.

The faster single-user profile adds three measured policy changes rather than
changing CUDA arithmetic: ten compute threads, a 59,340 MiB cache, and
lookahead width six. Its hotlist was trained from the existing fixed 64-token
route capture, so this is explicitly an in-sample recurring-prompt/studio
result, not a general K3 claim. The saved final confirmation was:

| repeat | time | throughput | hit rate | expert bytes |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 15.932697 s | **1.004224 tok/s** | 71.15% | 111,561,801,728 |
| 2 | 15.952056 s | **1.003005 tok/s** | 71.15% | 111,561,801,728 |

The median is **1.003615 tok/s**, which clears the sprint's stretch target.
Both repeats have identical token, logit and route hashes and exact CUDA
counters. A separate confirmation produced a 1.009791 tok/s median, and the
final queue-depth screen reproduced 1.007297 tok/s at depth two. Depths four
and eight were slightly slower at 1.002326 and 1.001715 tok/s, so depth two
remains selected.

The result does not sustain 1 tok/s across this entire 64-token continuation.
With width six, the longer CUDA arm reached 0.944126 tok/s versus 0.722252
tok/s for CPU VQ, a 30.7% improvement. It remained fully byte-identical, but
asynchronous prefetch timing changed 12 of 94,208 cache outcomes and moved
0.05% more bytes. The lookahead-off proof above supplies the exact cache-I/O
accounting gate; the lookahead-on run supplies the realistic performance
endpoint.

### Grouped synchronization decision

Grouping reduced the registered two synchronizations per expert exactly as
designed, but it also delayed router-order consumption and destroyed useful
expert-I/O/compute cadence:

| experts per group | median tok/s | change from group 1 |
| ---: | ---: | ---: |
| 1 | **0.936831** | — |
| 2 | 0.934069 | -0.29% |
| 4 | 0.919360 | -1.86% |
| 8 | 0.905899 | -3.30% |
| 16 | 0.863330 | -7.85% |

A group-16 trace cut VQ waiting by about 1.51 seconds over eight tokens, but
expert I/O grew by about 2.28 seconds. All group arms preserved byte-identical
outputs, exact semantic/launch/sync counts, explicit cache-hold lifetimes and
zero fallback. Group one is selected; larger groups remain an opt-in negative
experiment on this branch rather than part of the performance profile.

### Same-SSD contention

The engine-level contention screen repeated Sprint 12's 12 MiB random
`O_DIRECT` fio job at queue depth two, bracketed by uncontended 60-second
baselines:

| arm | fio bandwidth | p95 | p99 |
| --- | ---: | ---: | ---: |
| before | 13.388 GB/s | 1.630 ms | 1.729 ms |
| concurrent CUDA VQ engine | 8.384 GB/s | 3.949 ms | 4.145 ms |
| after | 13.186 GB/s | 1.663 ms | 1.745 ms |

The bracketing bandwidth drift was 1.50%. Against their mean, contention
reduced fio bandwidth by 36.9% and raised p95/p99 by 139.8%/138.7%. The model
completed at 0.808481 tok/s while fio was active, versus 0.944126 tok/s in its
uncontended 64-token capture. This is an expected shared-device saturation
bound: the 1 TPS studio result assumes exclusive use of the model SSD.

## Disposition

CUDA VQ mode 2/group 1 passes the strict arithmetic and engine gates. The
capture-derived hotlist, larger safe cache, width-six lookahead and ten-thread
selection together cross 1 tok/s for the measured recurring-prompt workload.
The claim remains experimental and in-sample; nothing is promoted to
`spark/integration` or proposed upstream from this sprint. That next validation
was the frozen held-out prompt-family experiment. Its decode-only aging arm
improved unseen-family throughput by 4.5573% and misses by 7.8974%,
narrowly missing its respective 5% and 10% gates; it stopped without selecting
the policy. Exact Sprint 13 counters, commands, captures and hashes are in
[the Sprint 13 summary](gn100/sprint13-vq-gpu-summary.json); the held-out
disposition is in [GPU_VQ_HELDOUT_GB10.md](GPU_VQ_HELDOUT_GB10.md).
