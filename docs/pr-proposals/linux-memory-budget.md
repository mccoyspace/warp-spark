# Size Linux automatic budgets from usable memory

Candidate commit: `5e4147f1ebc79e04ed31df70beccb24fafb5b48e`

## Problem

Automatic sizing currently derives its ceiling only from physical RAM.  On
Linux, that can be much larger than the memory the process can safely allocate:
another workload may already hold RAM, or a cgroup-v2 parent may impose a
smaller limit.  The engine can therefore remain below seven eighths of
`MemTotal` and still reclaim or swap its own expert cache.

The GN100 investigation also exposed a flaw in our first proposed correction:
subtracting one eighth of *host* RAM from a small cgroup's headroom can make an
otherwise usable group appear exhausted.  An 8 GiB group on a 128 GiB host
must retain a reserve based on the 8 GiB effective capacity, not a 16 GiB host
reserve.

## Change

- Add a private, injectable Linux memory reader for `/proc/meminfo`,
  `/proc/self/cgroup`, and the cgroup-v2 tree.
- Bound automatic sizing by the minimum of seven eighths physical RAM,
  `MemAvailable` minus the host reserve, and finite cgroup-v2 headroom minus
  one eighth of that cgroup's effective capacity.
- Walk finite cgroup ancestors; an unlimited leaf does not cancel a limited
  parent.
- Return `WASTE_E_MEMORY` before model-sized allocation when an automatic
  budget cannot hold the floor.
- Preserve explicit budgets as caller policy, with a warning above the current
  safe ceiling, and preserve non-Linux behavior.
- Expose `waste_memory_ceiling()` and include the value in JSON plan output.

This is open-time protection.  It does not predict a workload that begins
after the model opens, and it intentionally does not add cgroup-v1 parsing.

## Correctness

- Pure policy tests cover missing/malformed inputs, unknown values, exhausted
  constraints, whole-working-set selection, and floor refusal.
- Synthetic proc/cgroup tests cover an 8 GiB child, an unlimited child under a
  finite parent, current usage above a limit, and parent traversal rejection.
- Existing explicit-budget and non-Linux behavior remains tested.

## Performance and safety evidence

- macOS portable suite: 29 passed, 0 failed, 12 skipped; server: 168 passed.
- ASan/UBSan suite: 28 passed, 0 failed, 13 skipped.
- GN100 Linux ARM64 native suite: 27 passed, 0 failed, 13 skipped, including
  the synthetic Linux memory test.
- A real K3 automatic open on the otherwise idle GN100 selected the full
  recommendation, including a 55,780,900,864-byte expert cache, and completed
  in 24.1 seconds.  The change does not claim a throughput improvement.

## Compatibility

No container, arithmetic, routing, or state format changes.  Explicit nonzero
budgets retain their existing authority.  One public status and one public
query function are added; the in-tree Python binding changes atomically.

## Rollback

An explicit `--budget` bypasses automatic selection.  Reverting the single
candidate commit restores the physical-RAM-only policy.
