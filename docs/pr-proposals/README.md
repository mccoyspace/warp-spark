# Pull-request proposal

This branch carries one focused, copy-ready upstream proposal. Its implementation
is stacked on the in-memory state-snapshot prerequisite and should be rebased and
retested if that prerequisite changes before submission.

| Candidate | Fork branch | Draft | State |
| --- | --- | --- | --- |
| Exact server prefix cache | [`pr/server-prefix-cache`](https://github.com/mccoyspace/waste-spark/tree/pr/server-prefix-cache) | [Description](server-prefix-cache.md) | Implementation complete, independently reviewed, and exercised with real K3 on this project's GN100; practical warm-server results and their confound are documented |
