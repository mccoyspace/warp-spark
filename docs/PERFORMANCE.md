# Performance acceptance contract

Performance changes are compared against a frozen command, not an informal
interactive run. On the Acer GN100 the current baseline is:

```text
commit                 ce96e38e573cb1befd45623d0213027d09dce8a5
Linux direct-I/O patch enabled
budget                 86583021568 bytes
threads                8
CPU set                performance (CPUs 5-9,15-19 on this host)
learned list           retained usage.waste
16-token bench median  0.1000 tok/s
64-token decode        366.06 s, 0.1748 tok/s
64-token wall          544.60 s
```

`tools/perf_acceptance.py` appends `waste.acceptance.v1` JSON Lines and keeps
stdout, stderr and optional `waste.layer_trace.v1` data beside it. Each record
identifies the binary, model manifest, learned list, exact command, wall time,
peak RSS, minimum Linux `MemAvailable`, process swap, VM/PSI deltas and the
engine's own result.

`tools/server_acceptance.py` applies the same contract to a persistent
OpenAI-compatible server. It records startup separately, issues repeated
identical requests against one resident context, probes the container lock,
and captures per-request usage/stats plus the same Linux memory evidence.

Example:

```bash
python3 tools/perf_acceptance.py /models/k3.waste \
  --binary ./waste --out results/acceptance.jsonl \
  --artifacts results/raw --label learned-3x --workload bench \
  --tokens 16 --repetitions 2 --budget 86583021568 \
  --threads 8 --cpu-set performance --trace
```

```bash
python3 tools/server_acceptance.py /models/k3.waste \
  --repo . --binary ./waste --out results/server.jsonl \
  --artifacts results/server-raw --label k3-server \
  --budget 86583021568 --threads 8 --cpu-set performance \
  --tokens 16 --requests 2
```

An optimization is acceptable only when deterministic output is unchanged,
all tests pass, direct I/O remains active, `pswpout` and process swap remain
zero, pressure does not become sustained, and repeated end-to-end performance
improves beyond ordinary run-to-run variation. Invalid or overlapping runs
remain in the result file with an explanation; they are not deleted.

## Acer GN100 first-sprint acceptance

The first safety/instrumentation sprint preserved the performance baseline:

| Check | Result |
|---|---:|
| K3 16-token bench | 0.0966, 0.0960 tok/s |
| Previous median | 0.1000 tok/s |
| K3 64-token decode | 371.17 s, 0.17243 tok/s |
| Previous 64-token decode | 366.06 s, 0.1748 tok/s |
| 64-token wall | 549.83 s (previous 544.60 s) |
| Persistent server startup | 21.03 s |
| Persistent 16-token requests | 246.38 s, then 237.58 s |
| Competing K3 open | refused in 0.86 ms |

Every K3 acceptance run used direct I/O and CPUs `5-9,15-19`; process swap,
swap-out, direct reclaim and kswapd reclaim remained zero. A separate
four-token trace confirmed the current backend is synchronous `pread`, queue
depth 1, with no I/O/compute overlap. Its decoded MoE layers recorded 3.60 s
expert-read time, 8.32 s routed-expert compute, 1.84 s shared-expert compute,
5.45 s attention and 0.20 s routing. This supports bounded async/prefetch as
the next experiment, but puts its first-order opportunity near 15–20% of
decoded-layer time rather than enough to explain the full Mac/Acer gap.
