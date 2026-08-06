# GB10 lossless speculative decoding: Sprint 16 result

Sprint 16 asked whether Kimi-Linear, a model-native MTP head, or prompt lookup
could make greedy K3 decoding materially faster without changing a single
output token. The development answer is **not with the current draft,
cache budget, verifier path, and internal SSD**. The experiment stopped before
an integrated speculative decoder and left the fresh H2 corpus unspent.

This is a viability miss for the measured design, not a rejection of
speculative decoding as a class. The measurements identify what would have to
change before another implementation attempt is worthwhile.

## What was tested

The experiment was frozen on `exp/speculative-decoding-gb10` before new model
inference. It used the existing 12-prompt development corpus: A and B for
selection and legacy H1 as veto-only data. H2 was never supplied to a model
runner and remained at zero model steps.

Two draft classes were priced, and one possible third class was audited before
the trace campaign:

- Kimi-Linear with either canonical K3 token IDs or its native chat format;
- exact prompt lookup using the already-known K3 context; and
- a K3 MTP head, if one existed in the released checkpoint, was checked for
  availability rather than tested as a candidate.

The authenticated checkpoint audit found 497,220 tensor names, zero MTP-name
matches, and `num_nextn_predict_layers=0`. No MTP candidate was therefore
available. Prompt lookup was nearly free but weak: at `k=4`, A+B committed only
85 tokens through 77 verifier blocks, or 1.104 tokens per block. It was rejected
before any engine work.

Kimi-Linear with canonical K3 IDs was the best measured draft class and prompt
format. Across A+B its marginal argmax agreement was 53.906%. Exact trajectory
simulation gave:

| width | verifier blocks | accepted proposals | committed/block | measured draft branch time |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 143 | 110 | 1.7692 | 9.494 s |
| 4 | 121 | 132 | **2.0909** | 23.436 s |
| 8 | 118 | 135 | 2.1441 | 50.016 s |

`k=8` bought almost no additional amortization for more than twice the draft
work. Kimi-Linear's native prompt format was also worse at `k=4`, committing
2.016 tokens per block. The provisional arm was therefore canonical-ID
Kimi-Linear at `k=4`, subject to the registered cache-rent and verifier-cost
screen.

## The state round-trip is exact

The actual K3 state after an 81-token prompt and 31 decoded tokens was
471,085,216 bytes (449.3 MiB), including 6,193,152 bytes of live MLA latent
rows. In-memory export and import took 20.018 and 20.107 ms. A pre-touched
ten-thread copy took 6.437 ms best and 7.855 ms median.

Export, restore, replay, and re-export produced byte-identical logits, ordered
routes, and post-state. This confirms the earlier durable-checkpoint timing
was not the transactional-state floor. Repeated snapshots, restores, and
replay were not integrated or priced; they did not need to be, because the
verifier bound fails while assigning all of them zero cost.

## Draft residency has a real rent

The paired rent screen compared ordinary 32-token K3 decode in two states:

- **F:** 59,340 MiB K3 expert cache, no resident draft;
- **R:** 40,502 MiB K3 cache, fully resident 16,926 MiB Kimi-Linear cache, and
  both rollback allocations live, with speculation disabled.

All eight A+B F/R trajectories were byte-identical. Aggregate F completed 256
tokens in 302.955 s (0.8450 tok/s); R took 347.717 s (0.7362 tok/s). The median
per-case throughput loss was 9.93%, while pooled throughput fell 12.87%. R added
39,423 misses and 455.52 GiB of expert traffic across those tokens.

This is the measured rent before speculative work earns anything. It strongly
favors a future in-model head, prompt lookup on a repetition-heavy workload,
or a materially smaller draft.

## The available verifier misses the cost and exactness gates

One immediate-rejection block and one full-accept block were measured at
`k=4`. The serial reference uses four ordinary, CUDA-enabled T=1 K3 steps and
is exact. This diagnostic deliberately executed all four positions even for
the immediate-rejection block; it measured a fixed-width exact reference, not
an abort-aware serial implementation. The existing chunked prefill path
deduplicates experts, but runs on the CPU, changes arithmetic, and is not a
lossless verifier.

| arm | full-accept block | immediate-reject block | expert bytes at full accept | exact lossless contract |
| --- | ---: | ---: | ---: | --- |
| four serial CUDA steps | 5.119 s | **4.956 s** | 39.181 GB | yes |
| chunk, i8mm off | 9.918 s | 9.989 s | 33.858 GB | no |
| chunk, i8mm on | **9.459 s** | not repeated | 33.846 GB | no |

The chunk rows retained the same argmax IDs in these probes, but every logit
row hash and the final state hash differed from the serial reference. They
therefore cannot qualify under greedy lossless speculation. The i8mm row was
compiled once with `-march=armv8.6-a+i8mm`: GCC 13's `-mcpu=native` on the
Spark did not define the compile-time i8mm feature even though the CPU reports
it. That explicit build improved the already-slow chunk by 4.63%; no compiler
flag was added to the normal build.

## The generous viability bound still fails

The full-cache A+B baseline used 302.955 s for 256 tokens. The registered 15%
modeled-gain threshold permits 263.439 s. The measured `k=4` draft branches
consume 23.436 s over 121 verifier blocks, leaving at most **1.9835 s per
block**.

That bound deliberately charges zero for snapshots, restores, accepted and
correction replay, bonus and direct-tail target steps, draft resynchronization,
CPU/GPU synchronization, and bookkeeping. It also ignores the measured
cache-rent penalty. Even under those impossible favors, the faster fixed-width
exact-reference block is 4.956 s, or 2.50 times the available budget. That
comparison rejects fixed-width serial execution; it is not an estimate of an
abort-aware serial verifier.

The fixed four-step serial calibration is not a model of early abort. A real
serial verifier stops at the first mismatch, but that does not create useful
amortization: apart from root/bonus bookkeeping, it performs approximately one
ordinary K3 target step per committed token, reproducing baseline decoding and
then adding draft and transactional overhead. Any genuine verification gain
must therefore come from a batched target pass.

The existing chunk path cannot be repaired by launch tuning alone. Its fastest
row read 33.846 GB after cross-position expert deduplication. For the same
starting cache state and routes, exact batching can make compute cheaper and
overlap it with I/O, but it cannot reduce that measured unique-miss set without
changing effective residency or the record representation. The same Spark
SSD's prior uncontended direct-I/O controls measured about 13.2-13.4 GB/s,
implying a roughly 2.526 s storage floor before compute or state work. That
floor already exceeds the 1.9835 s block budget. Even 16 GB/s would require
about 2.116 s; 20 GB/s leaves only about 0.291 s inside the budget for all
verifier compute, while 23 GB/s leaves about 0.512 s. These are inferences from
the same-host fio controls and this representative block, not a new fio run or
a universal byte floor for every cache state.

## Decision

The pre-integration viability screen failed before the formal per-case S/F and
S/R candidate gate. No integrated speculative decoder was built, H2 consumed
zero model steps, the qualified profile is unchanged, and nothing from Sprint
16 is proposed upstream.

A future sprint should reopen integration only after both its draft side and
verifier side are viable. Draft-side candidates include:

- a native MTP head or substantially higher-agreement draft appears;
- a smaller draft preserves most of the 59,340 MiB K3 cache; or
- a reduced-work K3 self-draft proves both cheap and accurate without a second
  model allocation.

The verifier side requires an exact batched CUDA path **and** enough storage
bandwidth or effective residency to service its unique expert set within the
block budget. Under the measured representative cache state, approximately
20-25 GB/s is the credible storage range rather than an independent bonus.
Faster storage remains independently useful without speculation, especially
on held-out prompt families with weaker calibration-hotlist coverage; any
storage claim should therefore be paired on held-out families as well as the
warm studio profile.

The detailed values are in the
[machine-readable summary](gn100/sprint16-speculative-summary.json). Raw
evidence is attached to the `gn100-sprint16-results-2026-08-03` release and
anchored by its
[SHA-256 checksum](gn100/gn100-sprint16-evidence-2026-08-03.tar.gz.sha256).
