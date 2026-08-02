# Add typed whole-expert scheduling

Candidate commit: `5e76aea6b75db4a846337e322300cceef41464f8`

Candidate status: implementation-complete, but performance-gated.  It remains
an experimental branch and is not part of the recommended GN100 integration.

## Problem

The established routed-expert path divides each expert's matrix rows across
the compute pool.  That gives every expert the full pool, but it also submits
and joins several pool jobs per routed expert.  A K3 decode token crosses
thousands of those synchronization points.  Off-CPU measurements on the GN100
made whole-expert ownership worth testing: stage the complete top-k set, give
each worker complete experts, and reduce the results in the original route
order.

## Change

- Add `WASTE_EXPERT_SCHEDULE_ROW` and `WASTE_EXPERT_SCHEDULE_WHOLE` as a typed
  per-context setting, with matching CLI and server options.
- Add a schedule-aware memory planner and reserve the complete per-expert
  scratch allocation inside the hard budget.
- Acquire and reference the routed cache records as one bounded set before
  arithmetic; release every reference on success and failure.
- Run each complete expert as one outer pool job, using serial inner kernels,
  then accumulate outputs in the established route order.
- Report the effective schedule and fall back to row scheduling when reader
  threads or cache geometry cannot keep the full set valid.
- Normalize a whole-schedule request to row on a dense/no-MoE model without
  reserving unusable scratch.

The default remains row.  This candidate integrates with upstream's reader
pipeline; it does not restore the archived `io_uring` backend.

## Correctness

- Synthetic row and whole paths produce byte-identical logits, routes, and
  serialized conversation state across multiple tokens.
- With router lookahead disabled, hit, miss, eviction, byte, and expert-read
  counters are identical.
- Batch acquisition tests cover partial read failure and leaked-reference
  cleanup.
- Planner tests prove the exact scratch delta, overflow refusal, floor-minus-
  one failure, small-cache fallback, disabled-reader fallback, and dense-model
  normalization/evaluation.
- Default lookahead remains speculative; its completed traffic is explicitly
  treated as a measurement rather than an invariant.

## Performance and safety evidence

- Portable suite: 29 passed, 0 failed, 12 skipped; server: 168 passed.
- ASan/UBSan suite: 28 passed, 0 failed, 13 skipped.
- GN100 Linux ARM64 native suite: 27 passed, 0 failed, 13 skipped; server:
  168 passed.
- On K3, whole scheduling adds 19,496,960 bytes of scratch to the plan.
- Exact final-commit, same-binary, same-budget, raw/Q0 qualification:

| Schedule | Decode | Process wall | Demand cache | Filesystem inputs |
| --- | ---: | ---: | ---: | ---: |
| row | 2.97 s, 0.34 tok/s | 67.87 s | 695 hit / 777 miss | 355,508,408 |
| whole | 3.06 s, 0.33 tok/s | 68.23 s | 696 hit / 776 miss | 355,484,176 |

Both runs produced stdout SHA-256
`bb9af6089119e3cd934e7dba08807efff93bbdb591e1b58706eece73c17f6d5b`,
used effective direct I/O and a scoped zero-microsecond PM-QoS request, exited
zero, and recorded zero major faults and zero process swaps.  The one-record
cache difference is expected under the same total budget because whole's
extra scratch displaces a small amount of expert cache.

This pair does not establish a statistically significant regression.  It does
establish that the candidate failed its adoption gate: it showed no
incremental win over the current reader.  The branch remains useful for other
machines and further scheduler experiments, but it is not selected on GN100.

## Compatibility

No container, arithmetic, route-order, or state-format change is intended.
The new enum, query function, planner entry point, and appended public
configuration field require pre-1.0 binary clients to rebuild; the in-tree
CLI and Python `ctypes` layout change atomically.  A future stable ABI should
use a size-versioned options entry point rather than continuing to grow the
unversioned struct.

## Rollback

Row remains the default and `--expert-schedule row` selects it explicitly.
Reverting the single candidate commit removes the alternate scheduler and its
public API.
