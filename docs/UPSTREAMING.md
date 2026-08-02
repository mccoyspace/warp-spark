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

## Proposed pull-request stack

The order below follows dependencies, not the historical sprint order.

| Order | Proposed PR | Upstream boundary | Status on current base |
| ---: | --- | --- | --- |
| 0 | Linux 4 KiB `O_DIRECT` eligibility and transfer probing | Portable correctness | Already implemented upstream; no duplicate PR |
| 1 | Auto-budget from Linux `MemAvailable` and cgroup-v2 headroom | Generic safety; preserve other platforms | Upstream 0.6.3 has stable cgroup capacity; current-pressure layer is discuss-first |
| 2 | POSIX model-container ownership lock with explicit opt-out | Generic safety/policy | PR-ready on `pr/posix-model-lock` |
| 3 | Explicit Linux CPU-list affinity | Generic configuration | Keep the GN100 CPU choice outside core |
| 4 | Final phase/layer trace and request-boundary flushing | Generic observability | Reconcile with upstream cache traces |
| 5 | Transactional in-memory state export/import and caller-owned budget reservation | Generic engine API | PR-ready on `pr/in-memory-state-snapshots`; required before server prefix reuse |
| 6 | Whole-expert scheduling through typed per-context configuration | Generic optimization | Complete experiment on `pr/whole-expert-scheduler`; no GN100 gain, excluded from integration |
| 7 | Exact, renderer-delimited family-root server cache | Generic server feature | PR-ready on `pr/server-prefix-cache`; real-K3 miss-hit-hit acceptance complete; depends on state snapshots |
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
Do not submit the old `pr/linux-memory-budget` branch as written. The remaining
proposal is only the dynamic half (`MemAvailable`, `memory.current`, early
under-floor refusal, and host reservations), reconstructed on current upstream
and presented as a policy question before a PR. The Spark fork's API-2 ABI and
host-reservation fields are integration dependencies, not changes to smuggle
into that first discussion.

The old public `pread_batch` backend remains a diagnostic control, not a
production API.  A raw `io_uring` PR is conditional: first show that it adds
value over the current two-thread/depth-two upstream reader on the GN100.  If
it does, the PR must make fallback visible and test `ENOSYS`, `EPERM`, short
reads, out-of-order completion, and batches larger than the ring.  If it does
not, retain it only in the archive.

PM QoS, fixed CPU sets, thermal/CPPC/NVML/PMU collection, `bpftrace` sidecars,
cooldown gates, and fixed K3 prompts/budgets are Spark integration and evidence
tools.  They do not belong in the portable engine PRs.

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
