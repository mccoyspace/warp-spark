# GB10 dense CUDA expansion: sprint 12

Sprint 11 established that direct HMM access to K3's real Q4G/group-128
weights can accelerate the 69 KDA layers while preserving the strict CUDA
contract. Sprint 12 tests the remaining conventional decode-time Q4
projections before any new kernel class is attempted. This work remains on an
`exp/*` branch and is not part of the qualified `spark/integration` profile.

The control for every arm is the accepted sprint-11 fast CUDA path documented
in [GPU_GB10.md](GPU_GB10.md): 552 KDA projection calls per decode token and
zero fallback. The primary performance gate uses a balanced 16-token
one-load campaign; strict correctness uses the complete fixed 64-token corpus.
Both retain the qualified CPU set, Q0, Q8, two-reader/depth-two,
lookahead-zero profile and the same model, hotlist, prompt and greedy decoding
settings.

## Result

Sprint 12 accepts cumulative scope 2 as the experimental GB10 profile:

```sh
WASTE_CUDA_KDA=1 WASTE_CUDA_DENSE=2 ./waste ...
```

This offloads KDA, shared-expert FFNs, latent-MoE bridges and the five
conventional MLA projections. It executes 1,132 CUDA Q4 projections per
decode token. Scope 3 remains available as a diagnostic arm, but is not the
selected sprint-12 experimental profile. The implementation is commit
`1bca4d4` on `exp/cuda-dense-gb10`; no part of this experiment is promoted
into `spark/integration` by this result.

The balanced 16-token one-load campaign produced:

| scope | repeat 1 | repeat 2 | median | incremental change | vs KDA-only |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0: KDA-only | 0.513522 | 0.510998 | 0.512260 | — | — |
| 1: shared + latent | 0.617254 | 0.618831 | 0.618043 | +20.7% | +20.7% |
| 2: plus MLA | 0.682626 | 0.681030 | 0.681828 | +10.3% | +33.1% |
| 3: plus dense layer 0 | 0.700494 | 0.693742 | 0.697118 | +2.2% | +36.1% |

Every row executed exactly 552 KDA calls/token plus the expected dense calls,
reported zero fallback, recorded 12,526 hits, 11,026 misses and
136,797,200,384 expert bytes, and produced identical greedy-token and
route-set hashes. Arms A and B clear their 5% gates. Arm C's 2.2% is evaluated
under the machinery-reuse clause.

The GPU endpoint rose from 44 C before the campaign to 62 C afterward. The
reverse-order second half kept every scope within 1% of its first result, so
the performance decision is not based on a simple cold-first ordering effect.
At scope 2's 1.485 seconds/token, the preregistered 33.96 ms synchronization
estimate is 2.29% of token time. It remains below the 5% trigger for revisiting
CUDA Graphs.

The 64-token captures exposed a useful distinction hidden by the shorter
campaign:

| scope | throughput | dense calls | membership changes | order-only changes | max logit abs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.489927 | 0 | — | — | — |
| 1 | 0.623578 | 29,440 | 0 | 1 | `4.10e-5` |
| 2 | 0.673474 | 37,120 | 0 | 0 | `3.05e-5` |
| 3 | 0.687369 | 37,312 | 0 | 1 | `5.67e-5` |

Scope 2 executed 35,328 KDA plus 37,120 dense calls, with zero fallback. All
5,888 routed rows and 94,208 selected expert slots were identical in both
membership and order; all 65 causal logit rows retained their argmax and
top-10 set, maximum per-step mean absolute error was `3.48e-6`, and no value
was non-finite. Its 0.673474 tok/s is 37.5% above the 0.489927 KDA-only
64-token endpoint. The repeated 16-token campaign remains the primary
performance gate because all scopes shared one load and alternated order.

Scopes 1 and 3 each swapped ranks 15 and 16 in one row at layer 58 near the
end of the corpus, without replacing an expert or changing I/O. That still
passes the preregistered membership contract literally. Nevertheless, Arm C
is not selected as the sprint-12 experimental profile: its small
machinery-reuse gain is not worth giving up the stronger ordered-route result
that scope 2 already achieves. This is a conservative profile choice, not a
claim that Arm C failed the registered membership contract. Scope 3 stays
available to reproduce and study that decision.

The balanced campaign recorded an 83,738,376 KiB peak RSS (79.9 GiB), zero
major faults and zero swaps. Its phase-scoped Q0 holder opened and closed the
latency descriptor exactly once, reported no close error, and left no active
constraint. The clean Spark suite passed 33 checks with zero failures and 13
environmental skips; the CUDA build and all 13 capture-comparator checks also
passed.

### Engine-level storage contention

The final scope-2 screen paired the actual 64-token decode with a 12 MiB
O_DIRECT random-read fio job at queue depth two, pinned outside the engine CPU
set. Bracketing fio-only controls were stable to 0.15% in bandwidth:

| arm | fio bandwidth | p95 completion | p99 completion |
| --- | ---: | ---: | ---: |
| before | 13.223 GB/s | 1.253 ms | 1.352 ms |
| scope-2 engine overlap | 8.569 GB/s | 3.588 ms | 3.817 ms |
| after | 13.204 GB/s | 1.270 ms | 1.434 ms |

Against the bracket mean, overlap reduced fio bandwidth 35.2% and raised p95
and p99 by 184.4% and 174.1%. This is an end-to-end storage saturation bound:
the engine itself reads experts from the same SSD, so the delta must not be
attributed solely to coherent GPU trunk traffic. The engine delivered
0.586172 tok/s during the deliberately saturating overlap, 13.0% below its
uncontended 64-token scope-2 endpoint, while retaining exact call counts,
route hash and expert bytes. The screen sets no independent promotion gate,
but it supersedes the earlier synthetic-stressor result for end-to-end
capacity planning. The synthetic screen remains the cleaner isolation of
coherent GPU trunk traffic.

### Post-expansion residual profile

The accepted scope-2 profile makes the next target unambiguous:

| phase | accounted share |
| --- | ---: |
| KDA layers | 7.7% |
| MLA layers | 2.4% |
| MoE total | 89.3% |
| expert I/O sub-total | 22.2% |
| expert matmul sub-total | 56.6% |
| VQ LUT application sub-total | 51.0% |
| language-model head | 0.7% |

Sub-totals overlap their parent buckets and are not added together. The Q8
head and absorbed MLA no longer justify priority. The next GPU experiment is
the VQ gather: keep static codebooks device-resident, build and consume the
activation-dependent LUTs on device, keep expert records under the existing
host cache budget, and preserve scalar term plus router-ordered expert
accumulation as registered below.

## Scope and arms

The arms are cumulative and are enabled independently so a failed arm can be
removed without changing the accepted KDA path.

| arm | newly offloaded work | new calls/token | approximate new Q4 traffic/token | cumulative calls/token | expected calls over 64 decode tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| sprint-11 control | KDA projections only | 0 | 0 | 552 | 35,328 |
| A | shared-expert FFN plus latent MoE down/up bridges | 460 | 8.11 GiB | 1,012 | 64,768 |
| B | five conventional projections in each of 24 MLA layers | 120 | 2.53 GiB | 1,132 | 72,448 |
| C | the three projections in the layer-0 dense FFN | 3 | 0.35 GiB | 1,135 | 72,640 |

Arm A comprises 276 shared-expert calls and 184 latent-bridge calls. Arm B
comprises `q_a`, `q_b`, `kv_a`, gate and output projections; it requires the
CUDA staging capacity to cover the 18,432-row `q_b` output. Arm C requires
staging for the dense FFN's 33,792-row intermediate. The traffic figures
include quantization metadata and are rounded; call counts are exact.

The existing KDA path scans approximately 14.70 GiB per token. The cumulative
logical Q4 traffic is therefore approximately 22.81 GiB after A, 25.34 GiB
after B and 25.69 GiB after C: the final arm is about 1.75 times the accepted
KDA-only traffic, not three times it.

## Explicit exclusions

- The MoE router remains on the CPU. Its 92 Q4 calls scan only about
  0.28 GiB/token, while changing its arithmetic would conflate residual-stream
  perturbation with router-kernel perturbation. Keeping it bit-identical makes
  the route-invariance result interpretable.
- MLA's absorbed `kv_b_proj` remains on the CPU. Its transposed-K and forward-V
  operations are not the conventional matvec already validated. A residual
  profile after this sprint will price a specialized or fused kernel.
- The language-model head is Q8G/group-128, not Q4. A Q8 kernel and its own
  component gate are separate future work.
- VQ expert lookup, CUDA Graphs, CPU/GPU overlap and prefill offload are out of
  scope. No result from this sprint is allowed to depend on them.

`WASTE_CUDA_KDA` continues to mean the accepted KDA-only behavior. The new
scope must have a separate experimental control rather than silently widening
that variable's meaning. Preflight must validate every enabled tensor before
timing, and one failure anywhere in a token must poison CUDA sequence state;
mid-token CPU fallback is forbidden.

## Synchronization decision

The measured projection handoff is about 0.03 ms. At full scope,
`1,135 * 0.03 ms = 34.05 ms`, reported as 34.1 ms/token. The 583 new calls add
`17.49 ms`, reported as 17.5 ms/token. Against a projected 1.5-1.9 second
token, total synchronization is about 1.8-2.3% and the incremental cost is
about 0.9-1.2%.

Sprint 12 therefore uses the existing launch-and-synchronize path. Graphs are
not reconsidered during the sprint unless measured integrated timing, rather
than this estimate, attributes at least 5% of token time to handoff. Any such
finding is recorded and becomes a separate experiment; it does not silently
change these arms.

## Acceptance contracts

Every arm must satisfy the inherited sprint-11 strict numerical and runtime
contract:

- exactly the expected CUDA calls for its enabled scope and zero fallback;
- zero changed top-16 route memberships over the fixed decode corpus;
- identical greedy tokens and per-step argmax;
- maximum absolute logit error at or below `0.01`, with no non-finite values;
- identical expert bytes read and consistent hit/miss accounting against its
  matched control;
- zero major faults, zero swap activity, and clean Q0-holder and child exit.

Performance is evaluated on matched, thermally valid arms with the usual
one-load sweep and repeated endpoints:

1. **Arm A uses the normal engine gate.** Its median decode throughput must
   improve by at least 5% over the sprint-11 KDA-only CUDA control.
2. **Arm B uses the normal engine gate.** Report both its incremental result
   against A and the cumulative A+B result against the KDA-only control. It is
   accepted as a throughput arm only if the incremental improvement is at
   least 5% and the cumulative result does not regress.
3. **Arm C uses the machinery-reuse clause.** Its 0.35 GiB/token is too small
   to promise a 5% engine result. It may be retained when it reuses the
   already accepted Q4 machinery, improves its isolated component time,
   passes every numerical and runtime gate above, and produces no measurable
   regression in the cumulative candidate. It is removed if any of those
   conditions fails.

The machinery-reuse clause does not excuse a new arithmetic format, silent
fallback or a numerical difference. It only replaces an impossible
standalone macro-throughput threshold for small, mechanically identical work.

## Final contention and residual profile

After C, rerun the SSD contention screen with the complete dense engine arm,
not only the standalone projection stressor. Pair 12 MiB random O_DIRECT
reads at queue depth two with an actual fixed-corpus decode and bracket the
run with no-GPU fio controls. Report bandwidth, p95 and p99 completion latency,
expert bytes, faults and swaps. This is a dated characterization rather than
an independent percentage gate, but the final candidate must still clear its
macro gate while the real expert I/O is present.

Only after all accepted dense arms are active, rerun the decode phase profile.
That residual profile decides whether absorbed MLA, the Q8 head or VQ gather
is worth a specialized kernel; sprint 12 does not infer their value from the
old CPU profile.

## Contract reserved for a future VQ arm

A future fused VQ kernel must preserve the CPU contract's arithmetic ordering:
each expert partial is accumulated in the original scalar term order, and
weighted expert partials are reduced in original router order. It must not use
an atomic or tree reduction across experts, and unintended FMA contraction
must be controlled. One implementation may write a partial output per expert
and perform the final 16-expert reduction in router order.

Preserving order does not promise CPU/GPU bit identity, but it keeps the VQ
experiment in the existing strict contract class: zero route changes,
identical token decisions and bounded logits. If an implementation cannot
preserve that order, a different numerical contract must be registered before
its model outputs or performance are inspected.
