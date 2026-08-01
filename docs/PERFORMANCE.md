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
stdout, stderr and optional `waste.layer_trace.v2` data beside it. Each record
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
  --threads 8 --cpu-set performance \
  --io-backend io_uring --io-queue-depth 4 --trace
```

```bash
python3 tools/server_acceptance.py /models/k3.waste \
  --repo . --binary ./waste --out results/server.jsonl \
  --artifacts results/server-raw --label k3-server \
  --budget 86583021568 --prefix-cache 1073741824 \
  --threads 8 --cpu-set performance \
  --io-backend io_uring --io-queue-depth 4 --tokens 16 --requests 2
```

With a prefix reservation, server acceptance records the exact-output digest,
per-request cache status, reused/evaluated prompt tokens, snapshot size and
first-to-second request speedup. The first request must be a stored miss and
the second an exact hit; both answers must have the same digest.

Scheduler hypotheses are measured separately from the primary benchmark:

```bash
python3 tools/offcpu_diagnostic.py record \
  --cpus 5-9,15-19 --trace results/layers.jsonl \
  --events results/offcpu.tsv --summary results/offcpu.json \
  --markdown results/offcpu.md -- \
  ./waste run /models/k3.waste "Explain NVMe clearly." -n 1 \
    --budget 86583021568 --threads 8 --cpu-set performance \
    --io-backend io_uring --io-queue-depth 4 \
    --trace-layers results/layers.jsonl
```

The layer trace stores exact monotonic start/end pairs for each routed-expert
arithmetic interval without writing from inside the hot loop. A small
bpftrace sidecar gates capture with the decode-only `moe_layer` symbol, then
records the target thread group's sched-switch, wakeup and futex events, plus
CPU-idle transitions on recently targeted CPUs. The analyzer narrows them to
the exact intervals and separates futex/barrier wakes with a tightly matched
idle exit from wakes where no such exit was observed. It also reports the
matched idle state and residency. Sidecar numbers are diagnostic evidence,
never primary throughput measurements. Compare a matched QD1/QD4 pair with:

```bash
python3 tools/analyze_offcpu_pair.py \
  --pread-summary results/pread/offcpu.json \
  --qd4-summary results/qd4/offcpu.json \
  --pread-trace results/pread/layers.jsonl \
  --qd4-trace results/qd4/layers.jsonl \
  --output results/offcpu-comparison.json \
  --markdown results/offcpu-comparison.md
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

## Acer GN100 second-sprint acceptance

The second sprint added a raw Linux `io_uring` transport with no liburing
dependency. It submits only the router's exact expert records, in bounded
batches, and computes them in the original order. Cache slots returned to a
batch are pinned until all reads finish, so an in-flight miss cannot evict and
overwrite another live record. Ring setup failure reports a fallback and uses
synchronous `pread`.

The 16-token K3 queue-depth sweep used the same model, learned list, budget,
threads and CPU set as the first sprint:

| Backend | QD | Runs (tok/s) | Median | vs QD1 |
|---|---:|---:|---:|---:|
| synchronous `pread` | 1 | 0.0969, 0.0955 | 0.0962 | — |
| `io_uring` | 2 | 0.1029, 0.1038 | 0.10335 | +7.4% |
| `io_uring` | 4 | 0.1106, 0.1095 | **0.11005** | **+14.4%** |

QD4 passed the longer gates as well:

| Check | Pread/control | QD4 | Change |
|---|---:|---:|---:|
| Same-request persistent server | 208.75 s | 182.67 s | -12.49% |
| 64-token generation | 371.17 s | 315.15 s | -15.09% |
| 64-token wall | 549.83 s | 478.04 s | -13.06% |
| 64-token throughput | 0.17243 tok/s | 0.20308 tok/s | +17.78% |

The server responses and the 64-token CLI output were byte-identical between
the compared transports. All 368 routed-expert arrays in the four-token QD4
decode trace exactly matched the first-sprint trace. That trace reduced expert
I/O from 3.60 s to 2.81 s (-21.8%) with identical decode cache counts and
bytes. QD4 remained direct-I/O throughout, never fell back, and every
acceptance run recorded zero process swap, swap-out, direct/kswapd reclaim and
sustained memory pressure. Use `--io-queue-depth 4` on Linux to select this
measured configuration; synchronous `pread` remains the portable default.
