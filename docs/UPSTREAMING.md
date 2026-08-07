# Upstreaming the GN100 work

The GN100 development tree is a research history, not a pull request.  Upstream
changes must be reconstructed on the latest upstream commit as narrow,
independently testable patches.  This avoids asking maintainers to review
hardware campaigns, obsolete experiments, and several interacting engine
changes at once.

## Branch and refresh policy

The public fork uses these rules:

- `main` tracks `sqliteai/waste:main` without local commits.
- `pr/<topic>` starts from the current upstream commit, or an explicitly named
  prerequisite branch, and contains one reviewable concern.
- `exp/<topic>` contains work that is still being measured or that produced a
  useful negative result; it is not presented as ready for upstream review.
- `spark/integration` combines accepted generic branches with documented
  GN100-only features.
- `archive/*` and `gn100-*-results-*` tags identify measured source and never
  move.

Before a formalization pass:

```sh
git fetch upstream --prune
git switch main
git merge --ff-only upstream/main
git push origin main
git switch -c pr/TOPIC upstream/main
```

If upstream changes while a PR is under development, rebase the PR branch and
repeat its full acceptance set.  Do not rebase a measured archive or copy its
old benchmark number onto the rebased code.

## Upstream refresh: 2026-08-07

The fork's `main` now fast-forwards to upstream
`d9b919a791148b571e643d0af666bf19b4d733ab` (v0.6.6 plus two README magnet
corrections).  Since the previous shared base, upstream added optional VQ4P,
converter reclaim, explicit `--cpus` placement, a corrected direct-I/O disk
bench, and server/converter fixes.  VQ3R remains the default format, so the
qualified GN100 container does not need conversion or another download.

A no-commit merge into `spark/integration` found 14 textual conflicts.  Most
are documentation or build-list combinations; the production work is
concentrated in four interfaces:

1. combine upstream's batch expert holds with the CUDA path's epoch-scoped
   record pins;
2. port the qualified VQ3R CUDA dispatch around upstream's generalized
   VQ3R/VQ4P CPU API, failing closed for VQ4P until it has its own CUDA gate;
3. compose upstream's compute-thread `--cpus` with the qualified launcher's
   whole-process `taskset` without silently changing reader placement; and
4. reconcile the fork's API-2 additions and version string with upstream's
   new public CPU-list field.

This is a focused integration and requalification project, not a container
format migration.  Keep the consolidated-results tag immutable; perform the
merge on a new candidate branch and promote it only after model-free suites,
the retained K3 exactness check, and a short matched throughput run pass.

The three portable candidates were replayed on this base:

- `pr/posix-model-lock` (`293d06e`) needed three mechanical conflict
  resolutions and passed `31 passed, 0 failed, 13 skipped` plus 168 server
  checks on macOS;
- `pr/in-memory-state-snapshots` (`6fba880`) applied without conflicts and
  passed `30 passed, 0 failed, 13 skipped` plus 173 server checks; and
- `pr/server-prefix-cache` (`0c53bc7`), stacked on snapshots, applied without
  conflicts and passed `30 passed, 0 failed, 13 skipped` plus 202 server
  checks.

All three also passed their native Linux ARM64 model-free suites on the GN100:
30/0/13 for the lock and 29/0/13 for each stacked snapshot branch, including
the real CPU-binding check, with 168/173/202 server checks respectively.  The
rebased snapshot candidate additionally passed `test_state` against the real
K3 container: next logits, the complete recurrent state, and the legacy file
continuation were bit-exact after restore.  These are current-base candidates,
not submitted PRs.  Re-run the longer real-K3 server miss-hit-hit sequence only
if the snapshot prerequisite lands and the prefix-cache PR is next to submit.

## Proposed pull-request stack

The order below follows dependencies, not the historical sprint order.

| Order | Proposed PR | Upstream boundary | Status on current base |
| ---: | --- | --- | --- |
| 0 | Linux 4 KiB `O_DIRECT` eligibility and transfer probing | Portable correctness | Already implemented upstream; no duplicate PR |
| 1 | Auto-budget from Linux `MemAvailable` and cgroup-v2 headroom | Generic safety; preserve other platforms | Issue #14 is closed; upstream has stable cgroup capacity and the GN100 pressure row rejects the remaining hard ceiling |
| 2 | POSIX model-container ownership lock with explicit opt-out | Generic safety/policy | Rebased on `d9b919a`; macOS and GN100 suites pass at `pr/posix-model-lock` |
| 3 | Explicit CPU-list affinity | Generic configuration | Implemented upstream in v0.6.6 as `--cpus`; no duplicate PR |
| 4 | Final phase/layer trace and request-boundary flushing | Generic observability | Reconcile with upstream cache traces |
| 5 | Transactional in-memory state export/import and caller-owned budget reservation | Generic engine API | Rebased cleanly on `d9b919a`; macOS/GN100 suites and real-K3 bit-exact round-trip pass |
| 6 | Whole-expert scheduling through typed per-context configuration | Generic optimization | Complete experiment on `exp/whole-expert-scheduler`; no GN100 gain, excluded from integration |
| 7 | Exact, renderer-delimited family-root server cache | Generic server feature | Rebased cleanly on `d9b919a`; macOS/GN100 suites pass; depends on state snapshots |
| 8 | Mutable conversation-head reuse | Generic follow-on | Implemented on `spark/sprint7`; exact next-turn state/output, divergent-history, replacement, and shared-budget gates pass |

Sprint 9 produced one additional discuss-first candidate: measurement
correctness for the one-load sweep. Keep it as one narrow patch that drains
speculative reads before timing ends or cache state is cleared, resets all
mutable cache state, reports effective direct-I/O and reader settings, and
adds portable reset and race tests. The GN100 lookahead result and Spark launch
policy are evidence for that patch, not behavior the portable PR should impose.
Because upstream introduced the sweep recently, first send the maintainer the
short failure mode and offer the patch; open the PR if they want that shape.

Upstream 0.6.3 independently fixed the stable half of the Linux memory issue:
`waste_usable_ram()` now respects finite cgroup-v2 max/high over the hierarchy.
Do not submit the archived pre-0.6.3 Linux memory branch as written. The remaining
candidate was only the dynamic half (`MemAvailable`, `memory.current`, early
under-floor refusal, and host reservations). Sprint 10's pressure row below
does not support sending its current hard-ceiling shape upstream. The Spark
fork's API-2 ABI and host-reservation fields remain integration dependencies,
not changes to smuggle into another memory-policy discussion.

The old public `pread_batch` backend remains a diagnostic control, not a
production API.  A raw `io_uring` PR is conditional: first show that it adds
value over the current two-thread/depth-two upstream reader on the GN100.  If
it does, the PR must make fallback visible and test `ENOSYS`, `EPERM`, short
reads, out-of-order completion, and batches larger than the ring.  If it does
not, retain it only in the archive.

PM QoS, fixed CPU sets, thermal/CPPC/NVML/PMU collection, `bpftrace` sidecars,
cooldown gates, and fixed K3 prompts/budgets are Spark integration and evidence
tools.  They do not belong in the portable engine PRs.

The CUDA work is not a PR candidate on this pass.  Upstream issue
[#11](https://github.com/sqliteai/waste/issues/11) is already the active CUDA
design thread, and upstream VQ4P arrived after the qualified GB10 VQ3R path.
The least noisy contribution is one concise results comment there: exactness,
engine-level CPU/CUDA and held-out results, power and storage bounds, the
consolidated release link, and an offer to discuss the source.  A CUDA PR
should wait for maintainer interest and a decision about VQ4P coverage.

`pr/gcc-aarch64-native-note` is a separate documentation-only candidate.  It
records the observed GCC 13 failure mode where `-mcpu=native` accepts the GB10
target but omits the dot-product and i8mm feature macros, compiling both
kernels away.  It adds a macro probe and does not recommend enabling the
numerically non-default paths.

### Current-pressure decision

Sprint 10 measured the remaining memory-policy question before proposing code.
With a freshly faulted-in 31.3 GB host workload, every K3 open-time snapshot
put the Spark ceiling near 77.18 GB: enough for `floor + 2x`, not the 86.58 GB
`floor + 3x` recommendation. Three matched pressure arms at each budget found
47.12 s median decode at 3x versus 48.31 s at 2x. The step-down reduced
reclaim and retained roughly 18.5 GB more headroom, but cut the expert hit rate
from 49% to 36% and raised median filesystem input from 342.62 to 396.73 GB.

That is not evidence for an upstream default. Keep the Spark hard ceiling as
fork safety policy while its tradeoff is intentional; do not send it as a
performance optimization. A future generic proposal needs either an explicit
safety contract or a deeper pressure sweep that measures the crossover where
3x actually loses to 2x. Raw rows are in
[gn100/sprint10-pressure.csv](gn100/sprint10-pressure.csv).

## Shape of one PR

Each PR should be understandable without reading the sprint reports.  Use this
structure in the description:

```markdown
## Problem

Describe the observed failure or cost and the affected platforms. Include the
smallest measured fact that establishes the problem.

## Change

Describe the API and behavior. State defaults, limits, fallback behavior, and
what deliberately remains unchanged.

## Correctness

- unit/integration tests added
- deterministic output or state-round-trip result
- failure and cleanup paths exercised

## Performance / safety evidence

Give exact commits, machine, command/configuration, sample count and ordering,
raw result location, median or paired estimate, and rejection criteria. Say
"not performance-sensitive" when that is true.

## Compatibility

Call out C ABI, Python `ctypes`, container-format, state-format, platform, and
environment-variable consequences. State explicitly when there is no format
or numerical change.

## Rollback

Name the option that disables the behavior, or explain why reverting the one
commit is sufficient.
```

## Acceptance checklist

Every proposed branch must satisfy the applicable items before it is pushed as
a PR candidate:

- clean diff against the named current upstream commit;
- no unrelated formatting or campaign artifacts;
- portable build and upstream unit/server suites;
- focused tests for the new success, error, fallback, and cleanup paths;
- Linux ARM64 native build and focused GN100 test;
- deterministic tokens, routes, logits, state bytes, and I/O traffic wherever
  the change claims semantic transparency;
- no swap, major-fault, reclaim/PSI, or thermal invalidation in performance
  evidence;
- exact binary, manifest, source, command, and result hashes retained;
- public structs and Python `ctypes` layouts changed atomically;
- documentation states whether the feature is default, opt-in, or diagnostic.

For memory-budget work, synthetic tests must cover missing/malformed procfs,
unlimited and bounded cgroup-v2 values, current cgroup usage, and non-Linux
fallback.  The reserve should be proportional to the effective limit rather
than subtracting a fraction of host RAM from a much smaller cgroup.

For snapshots, failed imports must leave the live context untouched; successful
serialize/restore must produce identical next-token logits.  Persistent or
cross-context snapshots also require a model/manifest identity fingerprint.

For scheduling, the memory planner and execution path must read the same typed
context setting.  Tests must prove exact planned scratch use, fallback to the
row scheduler, identical numerical behavior, and no changed expert traffic.

## Commit discipline

Prefer one logical commit per PR.  During review, fixup commits are acceptable,
but squash only after retaining the review history in GitHub.  Commit messages
should state the behavior, not the sprint name; for example:

```text
Size Linux auto-budgets from available memory

Use MemAvailable and cgroup-v2 headroom when choosing an automatic RAM
ceiling. Preserve the existing physical-memory fallback on other platforms
and when Linux telemetry is missing or malformed.
```

Benchmark data should be committed only when it is compact, machine-readable,
and needed to support the PR.  Full raw GN100 campaigns belong with the Spark
archive/report bundle, referenced by hash from the PR.
