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
