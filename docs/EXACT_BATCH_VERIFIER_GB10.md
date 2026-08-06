# GB10 exact width-four verifier: Sprint 18 preregistration

Sprint 18 tests the remaining software-side prerequisite exposed by the K3
self-draft result: an exact width-four target verifier using the full 59,340
MiB expert cache. It is a diagnostic engine path, not a production speculative
decoder. NVMe-oF hardware remains deferred and H1/H2 remain untouched.

## Frozen design

| Item | Value |
|---|---|
| Parent commit | `08c3854` |
| Branch | `exp/exact-batch-verifier-gb10` |
| Width | 4 only |
| Model/cache | K3, 59,340 MiB, A-only hotlist |
| Corpus | frozen Sprint 17 A+B trajectories |
| Arithmetic | qualified CUDA T=1 kernels per row; CPU router and LM head |
| Schedule | causal layer-major execution across four positions |
| Expert reduction | original router order for every row |

The layer-major schedule may reuse a resident expert across positions at the
same layer, but it may not batch a projection by changing its reduction order,
move router or head arithmetic, renormalize routes, or reduce top-k. CUDA KDA
1, dense 2, VQ 2/group 1, Q8 on, SDOT/I8MM off, ten performance CPUs, Q0,
two direct readers at depth two, lookahead zero, no aging/prior compression,
and direct I/O are frozen.

## Sequence and gates

One retained rejection block and one retained full-accept block are the pilot.
Each exact block is compared with four ordinary serial target steps from the
same root. It must have byte-identical logits at all four positions,
byte-identical ordered routes at all four positions, and a byte-identical final
KDA/MLA state. CUDA fallback, process swap, and timed major faults must be zero.
The diagnostic records wall time, physical expert reads, demand misses, bytes,
and the distinct `(layer, expert)` union.

If either block misses exactness, Sprint 18 stops. If both pass, the same fixed
path is measured at every selected Sprint 17 A+B root after its reduced-expert
draft branch. No alternate width, CUDA mode, expert count, or arithmetic is
tried. This is development evidence only and does not spend H1 or H2.

The retained speculative admission arithmetic is:

```text
full-cache baseline                             302.954579 s
maximum time for +15% throughput                263.438764 s
measured Sprint 17 reduced-draft work            115.110682 s
remaining exact-verifier and transaction budget 148.328083 s
selected verifier blocks                                  67
average remaining budget per block                 2.213852 s
```

The A+B path is worth an integrated speculative decoder only if every exactness
and safety gate passes and its measured verifier plus actual snapshot/restore
cost keeps total modeled time at or below 263.438764 seconds. Rejected-branch
replay and other not-yet-integrated costs remain reported separately; they are
not silently assigned zero in the final disposition. A time miss still yields
the requested full-cache unique-miss measurement and ends integration work.

## Result: exact scheduling works, but not within the block budget

The diagnostic implementation processes the four causal positions layer by
layer while calling the same qualified CUDA T=1 attention, dense, routed-VQ,
and head routines for each row as ordinary decode. It retains CPU routing and
the original router-order expert accumulation. This changes scheduling and
cache locality without changing row arithmetic; it is not a fused multi-row
CUDA kernel.

The frozen `color_value_a` pilot compared that path directly with four serial
target steps from identical roots. Both the retained immediate-rejection block
and the retained full-accept block matched all four full-logit hashes, all four
ordered-route hashes, all four argmax values, and the final recurrent-state
hash. The exact schedule saved 104 reads (3.661%) on the rejection block and 93
reads (3.716%) on the full-accept block. Its wall-time changes were only +1.10%
and -0.44%, respectively. Serial and exact arms issued the same 18,032 VQ
launches and 11,776 VQ synchronizations per four-position pilot, locating the
remaining cost above the cross-position cache-locality improvement.

With that exactness gate passed, the fixed path ran at all 67 Sprint 17 A+B
roots with the full 59,340-MiB cache. The measured unique-miss result is:

| Metric | A+B result |
|---|---:|
| Exact verifier blocks | 67 |
| Verifier time | 299.480692 s |
| Time per block | 4.469861 s |
| Physical misses | 153,424 |
| Physical expert bytes | 1,903,498,428,416 B (1,772.771 GiB) |
| Physical bytes per block | 26.459 GiB |
| Distinct `(layer, expert)` union | 259,712 |
| Physical misses / union | 59.075% |
| Effective physical read rate | 5.919 GiB/s |

Per-case verifier cost ranged from 4.087 to 5.127 seconds per block, and
physical reads ranged from 23.645 to 31.626 GiB per block. Thus the larger
self-draft cache does reduce the prior representative 33.846-GB miss set on
average, but unfamiliar or difficult rows can still approach it.

The campaign's draft branches took 117.753958 seconds. Adding the exact
verifier gives a deliberately favorable lower bound of 417.234650 seconds;
adding only the measured snapshot and two restores per root raises it to
421.946527 seconds. That is 60.17% over the 263.438764-second admission limit
and produces 28.20% less throughput than the 302.954579-second ordinary-decode
baseline. Rejection replay, synchronization bookkeeping, and decoder
integration remain unimplemented and uncharged, so they cannot rescue the
result.

All eight canonical replays retained their expected final state hashes. Every
timed verifier block recorded zero major faults, process swap was zero, and all
CUDA fallback counters were zero. The exact path is therefore retained as a
useful diagnostic foundation, but the performance gate rejects an integrated
speculative decoder. H1 and H2 remain unspent, no runtime profile is promoted,
and no upstream submission is proposed.

The next credible software reopening condition is a verifier that reduces
CUDA launch/synchronization granularity across positions while preserving the
pilot's row arithmetic and router-order contract. Cache scheduling alone has
now been measured and is insufficient. See the
[machine-readable result](gn100/sprint18-exact-verifier-summary.json) and the
[raw-evidence checksum](gn100/gn100-sprint18-exact-verifier-evidence-2026-08-06.tar.gz.sha256).
