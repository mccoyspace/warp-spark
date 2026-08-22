# Preregistered GN100 full-layer prefill experiment

Experiment id: `gn100-prefill-full-layer-v1`

Status: stopped at Gate 0 on 2026-08-21 before a full-layer loader or treatment
arm was implemented. Complete-layer traffic failed the preregistered 1.75x
physical-byte ceiling even under an unrealistically favorable cache-credit
bound. The qualified product path remains unchanged.

## Question and prior evidence

Can exact K3 prompt prefill become faster by streaming every expert in layer
`L+1` while computing layer `L`, using two alternating layer banks carved from
the existing unified-memory budget?

This is a high-risk hypothesis. WARP already built and removed next-layer
router lookahead from chunked prefill. A 64-token chunk touched roughly 550 of
896 experts per layer, physical reads rose 6.9% (193.9 to 207.2 GB), and paired
wall times did not improve (132.6/137.0 seconds control versus 130.6/138.3
seconds treatment). See [TECHNICAL.md](TECHNICAL.md) and LEARNED section 36.
A complete-layer stream reads still more data and needs approximately 21 GB
for two K3 layer banks, so it must pass an instrumentation/simulation gate
before implementation.

## Frozen hypotheses

- Primary: treatment improves median prefill wall time per prompt token by at
  least 5% against the product control. Ten percent is the desired result.
- Mechanism: at least 80% of routed expert uses are staged before first demand,
  reducing demand-wait time rather than moving it elsewhere.
- End to end: prompt plus the following 16 decode tokens improves by at least
  5%; prefill may not borrow performance from the warmed decode tail.
- Energy: joules per prompt token regress by no more than 3%.
- Null: extra bytes, cache displacement, and GB10 unified-memory contention
  erase the overlap benefit.

## Gate 0: trace and simulate before building

Capture unmodified-product traces with the new opt-in telemetry:

```bash
WASTE_DUMP_PREFILL=prefill.jsonl waste bench MODEL
python3 tools/prefill_trace.py --json prefill.jsonl > prefill-summary.json
```

Each multi-token chunk/layer row records exact unique expert density, logical
and full-bank bytes, cache deltas, and layer/attention/feed-forward wall time.
Do not enable `WASTE_PROFILE`, `WASTE_DUMP_ROUTE`, or route-margin dumping in
the primary timing pass; those diagnostics add unrelated work. Trace runs are
for mechanism and threshold selection. Performance arms later run untraced.

Before treatment code, add a two-bank shadow simulation using the traced layer
sequence, manifest-derived record sizes, cache state, and measured SSD
bandwidth. Its optimistic steady-state lower bound is:

```text
prime(layer 0)
+ sum(max(non-I/O compute[L], missing full-bank bytes[L+1] / SSD bandwidth))
+ final drain
```

Stop before implementation if any condition holds:

- The optimistic bound cannot beat control by 5%.
- Projected full-layer physical traffic exceeds product control by 75%.
- Reserving two banks predicts paging or a cache-displacement reread regime.
- The following 16-token decode is projected to read over 10% more data.

The density threshold, if any, is frozen from development traces before the
validation set is opened. Changing it afterward creates experiment v2.

### Gate 0 result: stop before implementation

The first real-K3 trace ran on the project GN100 / NVIDIA GB10 with the
qualified CUDA environment, ten CPUs `5-9,15-19`, child-scoped Q0, direct I/O,
two readers at depth two, lookahead zero, and the fixed CLI budget
`93,024,636,928`. The source was base revision `4bba3d37` plus the uncommitted
instrumentation tree whose tested content digest was
`e48b35b044459f83cd3f1c555ff7b77a0a66ebe324ce492c4fce83327527b52a`.

The natural coding prompt was 64 raw tokens. K3's chat template made the
runtime prompt one complete 64-token chunk plus a 21-token tail. The complete
chunk produced 93 layer rows (92 MoE) with:

- mean expert density `0.3511` and p95 density `0.4587`;
- `339,027,779,584` physical bytes read by the product path;
- `359,089,549,312` logical distinct-expert bytes;
- `1,022,716,018,688` bytes for one complete stream of every expert layer;
- `179.840` seconds summed layer time, of which `97.676` seconds was attention,
  `81.428` seconds feed-forward, and `3.608` seconds blocked in expert acquire.

A literal complete-layer stream is `3.0166x` the observed product traffic. Even
subtracting the entire qualified `62,075,813,472`-byte expert cache as if every
resident byte perfectly avoided a stream read leaves a lower bound of
`960,640,205,216` bytes, or `2.8335x` product traffic. That is already above the
frozen `1.75x` stop threshold before charging two-bank displacement, priming,
unused records, or the following decode. The 21-token tail was worse at
`6.5977x` full-layer bytes versus physical demand, supporting the preregistered
choice to exclude partial chunks from any treatment.

Gate 0 therefore rejects arm C for this design and model. No full-layer loader
should be built under experiment id `gn100-prefill-full-layer-v1`. A redesigned
sparse or compressed staging hypothesis requires a new preregistration rather
than relaxing this result after seeing it. Compact evidence and hashes are in
[gn100/freetoken-gate0-20260821-summary.json](gn100/freetoken-gate0-20260821-summary.json).

## Treatment arms

Use one binary and model load with balanced interleaving, a complete logical
state/cache reset, and the same frozen hotlist before every row.

- **A, product control:** current `moe_chunk`, full qualified LFRU cache.
- **B, capacity control:** reserve exactly the two-bank treatment space while
  retaining current exact-demand loading. This measures cache displacement.
- **C, full stream:** prime layer 0, then compute from one aligned bank while
  filling the other with the next complete expert layer. Priming, cancellation,
  final drain, and post-prefill cache effects stay inside the timer.

C must beat A to be useful. C versus B identifies overlap separately from the
capacity tax. The first implementation enables C only for complete 64-token
chunks; smaller tail chunks retain the product path.

The existing `waste_ecache_hint` is not the treatment: it is capped at 64
experts and uses ordinary cache slots, while K3 has 896 experts per layer.
Treatment needs two aligned full-bank buffers with explicit ownership,
cancellation, and drain semantics.

## Fixed memory and runtime invariants

Use the qualified settings from [GB10_QUALIFIED_PROFILE.md](GB10_QUALIFIED_PROFILE.md):

- CLI engine budget `93,024,636,928` bytes.
- Server budget `95,172,120,576` bytes when the 2 GiB prefix reservation is
  included.
- Ten compute CPUs `5-9,15-19`.
- Direct I/O; two readers at depth two; LFRU; lookahead zero.
- Q8 trunk; SDOT and I8MM off.
- CUDA KDA 1, dense scope 2, VQ mode 2/group 1.
- Child-scoped Q0.

Bank size is derived from the manifest and checked for overflow. The two banks
are carved from the existing expert-cache allocation; allocating them on top
invalidates the row. Prefer slot-ownership transfer so a record is not retained
simultaneously in a staging bank and LFRU.

## Workloads and ordering

Development set:

- Existing A+B+H1 fixed families.
- Prompt lengths 64 and 256 for inexpensive mechanism sweeps.
- The existing 1,102-prompt-token plus 16-generated-token prefill-heavy case.
- At least four balanced repetitions for short arms and two repetitions for
  the 1,102-token arm.

Validation set:

- Mint and hash a new H4 set before treatment tuning finishes: six long
  coding/agent families including tool transcripts.
- Two interleaved repetitions per H4 family.
- Existing H3 may be a regression corpus but is no longer held out because it
  was opened during Consolidation Task 4.
- No thresholds change after H4 is opened.

Store the exact prompt bytes/token ids and their SHA-256, arm order, source
revision, build configuration, and effective runtime profile with every row.

## Recorded metrics

Primary:

- Prefill wall seconds and prompt tokens/second, excluding model load.
- Time to first generated token.
- Prompt plus 16-token full-request wall time.

Mechanism:

- Unique experts and density by chunk/layer.
- Demand hits, misses, physical bytes, and demand-wait seconds.
- Stream records issued/completed, ready before use, late, and never used.
- Reader occupancy, achieved SSD bandwidth, priming, and final-drain time.
- Cache evictions plus the following 16-token decode bytes/time.

Correctness and safety:

- Identical final-prefill logit, generated-token, route, and exported-state
  hashes across arms.
- Effective direct I/O/readers/CUDA modes and zero fallbacks.
- Planned/actual memory, peak RSS, minimum `MemAvailable`, major faults, swap,
  PSI/allocation stalls, GPU/NVMe temperatures, clocks, and Q0 status.
- Shelly total and marginal energy when available.

Staging traffic is reported separately from demand misses. A higher apparent
hit rate is not evidence if physical bytes or full-request time regress.

## Promotion and abort rules

Promote only when all are true:

- C is at least 5% faster than A and B on development and H4 medians.
- No H4 family regresses by more than 5%.
- Prompt plus 16-token full-request time improves by at least 5%.
- Physical traffic is no more than 1.75 times A.
- Energy per prompt token is no worse than 3%.
- Every exactness and safety check passes.

Abort or invalidate immediately on output/state divergence, allocation above
the fixed budget, swap, process major faults, OOM, reader deadlock/starvation,
undrained work crossing arm boundaries, loss of direct I/O, or thermal
throttling. Stop the expensive campaign after two consecutive 64-token C runs
more than 10% slower than their paired A controls, or after any first pilot
exceeds twice the control's physical bytes.

Passing this experiment would select an opt-in experimental branch. It does
not modify the qualified `spark-cuda` profile without a separate qualification
and evidence archive.
