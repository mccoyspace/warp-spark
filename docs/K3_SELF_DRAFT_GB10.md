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

## Result: the draft works; the present verifier path still does not

The frozen A+B campaign ran on the Acer Veriton GN100 on 2026-08-06. The
top-4 self-draft produced 261 proposals at 67 selected block roots. It matched
200 proposals (76.63% marginal agreement) and committed 256 canonical tokens,
or 3.8209 tokens per verifier block. Prefix survival was 100%, 80.60%, 63.08%,
and 51.61% through positions one to four. The first value is 100% by design:
proposal one reuses the exact logit row already available at the block root.

Reduced branches took 115.111 seconds and read 241.008 GiB. Under the
preregistered hypothetical 1.5-second exact verifier, total modeled time is
215.611 seconds, below the 263.439-second limit and equivalent to a 40.51%
throughput gain over the retained full-cache baseline. This is a useful
draft-side result, not a measured engine speedup. Applying Sprint 16's
representative 2.526-second same-SSD storage floor instead would raise the
already overhead-free projection to 284.339 seconds, only 6.55% faster than
baseline and below the gate. The exact verifier's miss set under the larger
self-draft cache has not yet been measured, so that projection is a bound from
the prior representative block rather than a new claim about all 67 blocks.

All eight full-top-16 replay trajectories ended with the exact retained
471-MiB state hash. There was no process swap, CUDA fallback, profile drift, or
leftover Q0 holder. Seven rows recorded zero major faults; the first row
recorded one during the complete probe window. Consequently the literal
all-zero preregistered safety gate returned `viability_miss`, even though its
timing, trajectory, swap, and fallback sub-gates passed. The result is retained
as-is rather than rerun or relabeled.

The engineering disposition is therefore narrower than the runner status:
reduced-work K3 satisfies Sprint 16's draft-side reopening condition, but an
integrated decoder is still premature. The next speculative software target is
an exact batched verifier that measures its unique misses with the 59,340-MiB
cache. It must establish a realizable block cost before any H-tier campaign or
integrated decoder. H1 and H2 each consumed zero model steps, no runtime profile
was promoted, and no upstream submission is proposed.

See the [machine-readable result](gn100/sprint17-k3-selfdraft-summary.json) and
the [raw-evidence checksum](gn100/gn100-sprint17-selfdraft-evidence-2026-08-06.tar.gz.sha256).
