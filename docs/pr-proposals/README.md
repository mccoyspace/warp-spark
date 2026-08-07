# Pull-request candidates

These are review drafts for changes rebuilt on current upstream rather than
cherry-picked from the measured GN100 branch.  The implementation branches are
independently reviewable; a proposal that is stacked on a named prerequisite
says so explicitly. Rebase every branch that is not first to land, rerun its
acceptance set, and resolve public status-code or struct-layout additions
against the new base before opening the next PR.

| Candidate | Fork branch | Draft | State |
| --- | --- | --- | --- |
| POSIX model-container ownership | [`pr/posix-model-lock`](https://github.com/mccoyspace/waste-spark/tree/pr/posix-model-lock) | [Description](posix-model-lock.md) | Open upstream as [PR #29](https://github.com/sqliteai/waste/pull/29); mergeable and awaiting review |
| In-memory state snapshots | [`pr/in-memory-state-snapshots`](https://github.com/mccoyspace/waste-spark/tree/pr/in-memory-state-snapshots) | [Description](in-memory-state-snapshots.md) | Rebased on upstream `d9b919a`; macOS/GN100 suites and real-K3 exact round-trip pass |
| Exact server prefix cache | [`pr/server-prefix-cache`](https://github.com/mccoyspace/waste-spark/tree/pr/server-prefix-cache) | [Description](server-prefix-cache.md) | Rebased on upstream `d9b919a`; macOS and GN100 suites pass; real-K3 evidence retained |
| GCC AArch64 native-feature note | [`pr/gcc-aarch64-native-note`](https://github.com/mccoyspace/waste-spark/tree/pr/gcc-aarch64-native-note) | Branch diff | Documentation-only current-base candidate; not submitted upstream |

The superseded Linux budgeting implementation is retained at
[`archive/pre-063-linux-memory-budget`](https://github.com/mccoyspace/waste-spark/tree/archive/pre-063-linux-memory-budget);
upstream 0.6.3 independently fixed its stable cgroup-capacity half. The
no-gain whole-expert scheduler remains available at
[`exp/whole-expert-scheduler`](https://github.com/mccoyspace/waste-spark/tree/exp/whole-expert-scheduler).
Neither branch is an active PR candidate.

The exact measured Sprint 5 source remains available at
[`archive/sprint-5-measured`](https://github.com/mccoyspace/waste-spark/tree/archive/sprint-5-measured)
and tag `gn100-sprint5-results-2026-08-01`.  It is evidence and design history,
not a substitute for the focused branches above.
