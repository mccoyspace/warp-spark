# GB10 reduced-expert K3 self-draft screen

This is a deliberately small software-only follow-up to Sprint 16. It asks
whether K3 can draft for itself cheaply enough to justify future speculative
decoder work. It does not implement speculation and cannot promote a runtime
profile.

The expensive NVMe-oF storage proposal is deferred. The untouched Sprint 15
H2 tier also remains unspent.

## Frozen experiment

| Item | Value |
|---|---|
| Parent record commit | `206c14e` |
| Branch | `exp/k3-self-draft-gb10` |
| Development cases | Sprint 14 A+B only (8 cases, 32 tokens each) |
| Prompt and target IDs | frozen Sprint 16 K3 trajectories |
| Model | K3, no second model resident |
| Target cache | 59,340 MiB, A-only hotlist |
| Draft rule | select and normalize the ordinary top 16, execute the first 4 in original router order |
| Block width | 4 only |
| First proposal | exact argmax already available at the canonical block root |
| Later proposals | three reduced-expert K3 forward steps at most |

The remaining profile is the qualified Sprint 13 control: CUDA KDA 1, dense
2, VQ 2/group 1, ten performance CPUs, Q0, Q8 on, SDOT/I8MM off, two direct
I/O readers at depth two, lookahead zero, and no cache aging or prior
compression. Reduced-expert mode never changes router selection, full-top-16
weight normalization, or accumulation order among the experts it retains.

No top-2 sweep, alternate width, layer truncation, prompt retuning, or H-tier
model run is allowed in this screen. Those are separate experiments only if
this simplest candidate earns them.

## Measurements and gate

For each case, the offline probe restores the canonical target root after
every reduced branch and then teacher-forces the frozen full-top-16 target
trajectory. It records proposal survival and first rejection, committed tokens
per block, reduced-branch time, cache hits/misses/bytes, CUDA work and fallback,
process swap and major faults, and the final exact target-state hash.

The practical target is still Sprint 16's registered 15% modeled throughput
gain over the full-cache A+B control:

```text
baseline F                         = 302.954579 s / 256 tokens
maximum speculative time          = 302.954579 / 1.15
                                  = 263.438764 s
optimistic candidate time         = measured reduced-branch seconds
                                  + 1.5 s * verifier blocks
```

The 1.5-second verifier is an explicitly optimistic future assumption, below
what the current SSD can support for the representative Sprint 16 miss set. It
charges zero for snapshots, restores, replay, synchronization, bookkeeping,
and every other integration cost. Passing therefore means only that this draft
is worth revisiting after an exact batched verifier and adequate residency or
storage bandwidth exist. It does not claim a realizable speedup today.

The candidate stops here unless all eight A+B rows have exact canonical final
state hashes, zero CUDA fallback, zero process swap and timed major faults, and
the pooled optimistic time is at most 263.438764 seconds. A miss ends this
software experiment without an integrated decoder or further candidate
shopping. A pass permits a separately recorded integration decision; it does
not spend H2 automatically.
