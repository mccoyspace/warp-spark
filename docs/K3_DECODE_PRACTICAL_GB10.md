# Kimi K3 practical decode on NVIDIA GB10

Date: 2026-08-23

Status: experimental campaign complete; practical studio profile promoted

This note records a sanitized Kimi K3 decode experiment on a 128 GiB NVIDIA
GB10 system. It tests lower routed-expert counts as an explicitly approximate
operating mode. The generic registered prompt corpus is already public in this
repository and is identified below. Raw completions, routes, captures, learned
usage files, and machine paths remain private.

This is a branch result, not an upstream-supported profile or a request for
upstream adoption.

## Contracts, not one “fast mode”

The profile names separate two independent choices: prompt-prefill arithmetic
and routed-expert count.

| Profile | Routed experts | Prompt prefill | Intended use |
| --- | ---: | --- | --- |
| `k3-exact` | trained top-16 | exact-prefill fallback | strongest available prefill-preservation contract |
| `k3-top16` | trained top-16 | practical dense mode 1 + exact VQ | full-quality routing; retained fallback |
| `k3-top12` | approximate top-12 | practical dense mode 1 + exact VQ | measured intermediate fallback, not promoted |
| `k3-top8` | approximate top-8 | practical dense mode 1 + exact VQ | held-out-qualified practical studio default |

“Exact” above describes the prompt-prefill fallback. It is not a blanket claim
that every CUDA dense operation in the full request is bit-identical. Practical
prefill is separately qualified in `K3_PREFILL_GB10.md` and may cause bounded
trajectory divergence.

Top-12 and top-8 change the model computation: fewer router-ordered experts
are accumulated and the retained weights are normalized for that effective
top-k. They are not numerical approximations of an otherwise identical top-16
sum and must not be described as exact K3.

## Implementation and fail-closed boundary

The experimental engine commit is `dfaa108edfcec71e8187924ce9c67fbde747cdb0`.
`WASTE_K3_APPROX_TOP_K=0|12|8` is read at startup. Zero preserves the trained
top-16 contract; 12 and 8 are accepted only for the complete released K3
geometry. Other values, non-K3 shapes, post-load selector mutation, and test
sweep mutation are rejected. Each routing contract requires a fresh process.

Focused geometry tests, the local serve/QoS suite, and the Spark CUDA geometry
test passed. Same-effective-k CPU-VQ versus CUDA-VQ gates were exact:

| Effective top-k | CPU VQ | CUDA VQ | Result |
| ---: | ---: | ---: | --- |
| 12 | 0.781315 tok/s | 0.954553 tok/s | all logits and 8,832 routed slots byte/order identical |
| 8 | 0.977940 tok/s | 1.217075 tok/s | all logits and 5,888 routed slots byte/order identical |

These gates isolate CUDA dispatch correctness at a fixed effective top-k. They
do not establish quality equivalence between top-16 and an approximate arm.

## Reproducibility contract

- converted K3 container, 93 layers, 92 MoE layers, 896 routed experts, trained
  top-16;
- explicit server budget `95172120576` bytes;
- ten externally pinned CPUs `5-9,15-19`, child-scoped PM-QoS Q0;
- direct I/O, LFRU, two readers at depth two, router lookahead six;
- `WASTE_CUDA_KDA=1`, `WASTE_CUDA_DENSE=2`, `WASTE_CUDA_VQ=2`,
  `WASTE_CUDA_VQ_GROUP=1`, `WASTE_XPAR=0`;
- practical arms use `WASTE_CUDA_PREFILL_VQ=1` and
  `WASTE_CUDA_PREFILL_DENSE=1`;
- one fresh server and cold prefix per arm, greedy sampling, 64 output tokens;
- fixed opening/closing top-16 control bracket, frozen common top-16 hotlist,
  and usage learning disabled;
- no competing GPU services; accepted arms reported zero server `VmSwap` and
  no resource distress; exact reported effective top-k, complete 92-layer
  route shape, finite normalized weights, and zero fallback required.

The approximate candidate was selected from the product screen before the
unseen-family holdout corpus was opened. The frozen corpus is
`9dcfa0d4fc220805c262d21746258709e97043a2:docs/gn100/sprint14-heldout-corpus.json`,
SHA-256 `4fb4ca60d82ed521c0d59732c748a5f99b22f720d35e19d5590b41bc01c423fe`.
The four registered prompt lengths are 76, 77, 80, and 79 tokens; rendered IDs
must match the preregistration byte-for-byte before an arm is accepted.

## Product screen

The fresh-process order was top-16 control 1, top-12, top-8, top-16 control 2.
The two controls differed by at most 0.52%, passing the registered 5% drift
gate.

| Arm | TTFT | Post-first-token decode | Full-request throughput |
| --- | ---: | ---: | ---: |
| Top-16 control mean | 94.678 s | 0.805641 tok/s | 0.370207 tok/s |
| Top-12 | 82.082 s | 0.993685 tok/s | 0.439917 tok/s |
| Top-8 | **68.889 s** | **1.238816 tok/s** | **0.534475 tok/s** |

Against the control mean, top-12 improved decode by 23.3% and full-request
throughput by 18.8%, while reducing TTFT by 13.3%. Top-8 improved decode by
53.8% and full-request throughput by 44.4%, while reducing TTFT by 27.2%.

All structural and simple anti-collapse screens passed. The retained outputs
were coherent and on task in private review. All arms reached the common
64-token cap before completing the multi-part brief, so shared truncation is a
length-cap limitation rather than evidence of top-8 collapse. Top-8 was fixed
as the only candidate advanced to the unseen-family holdout; top-12 remains an
unpromoted fallback and is not eligible for candidate-shopping on that holdout.

## Unseen-family holdout

Each frozen case ran top-16 control 1, top-8, top-16 control 2 in fresh
processes. All prompt-ID, system, behavior, drift, and private comparative
review gates completed before calibration.

| Measure | Top-16 control mean | Top-8 | Relative result |
| --- | ---: | ---: | ---: |
| TTFT | 99.822 s | **72.337 s** | 27.5% less time |
| Decode throughput | 0.779545 tok/s | **1.212526 tok/s** | **+55.5%** |
| Full-request throughput | 0.354364 tok/s | **0.515019 tok/s** | **+45.3%** |

- exact preregistered prompt gates: all 12 arms passed;
- system, route, and simple behavior gates: all passed;
- maximum control drift: 0.82%, below the registered 5% limit;
- private comparative review: no top-8-specific regression or collapse found
  in the four-family sample; one family favored top-8's factual restraint;
- shared caveat: both profiles invented absent-image details in one family,
  exposing a media-grounding/harness limitation rather than a top-8-specific
  failure;
- holdout decision: advance top-8 while preserving both trained-top-16
  profiles. This small holdout does not establish general quality parity.

## Profile-specific hotlist calibration

Learned expert rankings are namespaced by contract and are never shared as a
live candidate:

```text
.k3/learning/profiles/k3-top16/studio-usage.waste
.k3/learning/profiles/k3-top12/studio-usage.waste
.k3/learning/profiles/k3-top8/studio-usage.waste
.k3/learning/profiles/k3-exact/studio-usage.waste
```

Top-8 calibration uses only calibration/train families already available
before the unseen-family holdout. The opened holdout is not used for tuning.

| Top-8 arm | TTFT | Decode | Full request |
| --- | ---: | ---: | ---: |
| Frozen common hotlist | 73.119 s | 1.232488 tok/s | 0.187656 tok/s |
| Calibrated top-8 hotlist | **71.746 s** | **1.367036 tok/s** | **0.193509 tok/s** |

The sealed calibration candidate was exact in outputs and routes, stable
between controls, and reduced expert traffic by 15.24%. It improved decode by
10.67% at the family median, but its median full-request gain was only 2.97%,
below the preregistered 5% adoption gate. No extension was warranted. The
candidate remains private evidence and is not promoted; the frozen common
hotlist, SHA-256
`a0e70d0612e4e42958ea313d503f96536a78840d1da1cfaa3e8a93653e69a482`,
is retained as a copy in top-8's own namespace.

## Fresh phase trace and scheduler decision

The selected top-8 routing profile received a fresh, decode-isolated phase
trace before any further pipeline engineering. It used the retained common
hotlist and recorded whole MoE time, exposed expert-acquire wait, expert body,
and CUDA apply/handoff/synchronization separately. A repeat was made only to
check whether the profile reproduced after the first run tripped the strict
host-global no-swap-I/O gate.

| Phase | Time or share |
| --- | ---: |
| Whole MoE | 7.333 s mean; 59.53% of measured decode time |
| Exposed expert acquire wait | 2.237 s mean; 18.16% |
| Expert body | 2.875 s mean; 23.34% |
| CUDA apply/handoff/sync | 2.448 s mean; 19.88% |
| Decode throughput | 1.297821 and 1.300065 tok/s; 1.298943 tok/s mean |

The phase timers overlap; their subphase shares are diagnostic and are not
additive.

The phase mix reproduced closely, but the original strict clean-host gate
technically failed: the host-global `pswpout` counter advanced by one page and
two pages respectively. The traced process itself reported zero swap in both
runs, more than 121 GiB remained available, and there was no sign of memory
pressure. The repeat is therefore retained as practical directional mechanism
evidence, not discarded and not promoted to strict quantitative proof. There
will be no further rerun or threshold shopping.

Source audit then established that the existing `WASTE_CUDA_VQ_GROUP=2` path
is already the smallest bounded two-ticket overlap pilot: it acquires expert
zero, enqueues the pair asynchronously, acquires expert one while CUDA runs,
then finishes in router order with both records held live. A direct fresh-
process `group1 control -> group2 -> group1 control` bracket was consequently
the only remaining screen. It had to preserve byte-exact logits, tokens, and
routes, halve synchronization count without changing semantic work, keep
controls within 5%, avoid more than 2% full-request regression, and improve
decode by at least 5% before any follow-up.

| Arm | TTFT | Decode | Full request |
| --- | ---: | ---: | ---: |
| Group-one control mean | 71.969 s | **1.228321 tok/s** | **0.519242 tok/s** |
| Group two | 71.743 s | 1.219740 tok/s | 0.518667 tok/s |

Group two was safe but not useful. All 17 captured full-vocabulary logit rows,
generated tokens, and routes were byte-identical; non-synchronization work was
unchanged; synchronization count was exactly halved; and CUDA fallback stayed
at zero. Controls differed by less than 1%. Despite that, group two reduced
decode throughput by 0.70% and full-request throughput by 0.11%, while TTFT
improved by only 0.31%. It therefore missed the 5% practical trigger and is
not advanced. Group one remains selected, and no new grouping or pipeline code
is justified by this campaign.

For this final screen, global swap counters had to be quiet for five seconds
before each arm and were then retained as diagnostics. The actual server's
`VmSwap` was recorded before and after. Small unrelated host-global activity
is not treated as a reason to discard an otherwise stable, useful studio
result; this is a practical operating qualification, not a claim that the host
was mathematically motionless.

## Promotion and scope

The switchable harness names are `k3-exact`, `k3-top16`, `k3-top12`, and
`k3-top8`. The backward-compatible `k3` alias remains whitelist-only. A clean
public install defaults conservatively to `k3-top16`; the studio's ignored,
machine-private default selects `k3-top8`. Exact-prefill and practical-prefill
trained-top-16 fallbacks remain available. The retained common hotlist is
installed as a copy in top-8's own private namespace rather than shared live
across routing contracts.

Final selected held-out throughput: **1.212526 decode tok/s** and **0.515019
full-request tok/s** across four frozen studio-task families.

Final profile decision: promote **`k3-top8` with CUDA VQ group one** as the
private practical studio default; retain `k3-top16` and `k3-exact` as full-
routing fallbacks; retain `k3-top12` as unpromoted experimental evidence.

Nothing in this note claims portable quality, support for another K3 revision
or geometry, or upstream readiness. Any upstream discussion would require a
separate maintainer-facing design and maintenance assessment after the private
campaign is complete.
