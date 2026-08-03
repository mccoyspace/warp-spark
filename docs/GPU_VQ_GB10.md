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
