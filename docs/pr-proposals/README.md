# Pull-request candidates

These are review drafts for changes rebuilt on current upstream rather than
cherry-picked from the measured GN100 branch.  The implementation branches are
independently reviewable; a proposal that is stacked on a named prerequisite
says so explicitly. Rebase every branch that is not first to land, rerun its
acceptance set, and resolve public status-code or struct-layout additions
against the new base before opening the next PR.

| Candidate | Fork branch | Draft | State |
| --- | --- | --- | --- |
| Linux available-memory budgeting | [`pr/linux-memory-budget`](https://github.com/mccoyspace/waste-spark/tree/pr/linux-memory-budget) | [Description](linux-memory-budget.md) | PR-ready; not yet submitted upstream |
| POSIX model-container ownership | [`pr/posix-model-lock`](https://github.com/mccoyspace/waste-spark/tree/pr/posix-model-lock) | [Description](posix-model-lock.md) | PR-ready; not yet submitted upstream |
| In-memory state snapshots | [`pr/in-memory-state-snapshots`](https://github.com/mccoyspace/waste-spark/tree/pr/in-memory-state-snapshots) | [Description](in-memory-state-snapshots.md) | PR-ready; not yet submitted upstream |
| Exact server prefix cache | [`pr/server-prefix-cache`](https://github.com/mccoyspace/waste-spark/tree/pr/server-prefix-cache) | [Description](server-prefix-cache.md) | PR-ready; real-K3 miss-hit-hit acceptance complete |
| Whole-expert scheduler | [`pr/whole-expert-scheduler`](https://github.com/mccoyspace/waste-spark/tree/pr/whole-expert-scheduler) | [Description](whole-expert-scheduler.md) | Complete experimental candidate; no GN100 gain, excluded from integration |

The exact measured Sprint 5 source remains available at
[`archive/sprint-5-measured`](https://github.com/mccoyspace/waste-spark/tree/archive/sprint-5-measured)
and tag `gn100-sprint5-results-2026-08-01`.  It is evidence and design history,
not a substitute for the focused branches above.
