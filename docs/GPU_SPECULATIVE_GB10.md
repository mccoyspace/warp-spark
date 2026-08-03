# GB10 lossless speculative decoding: Sprint 16 preregistration

Sprint 16 tests greedy speculative decoding with Kimi-Linear as the draft and
K3 as the verifier. Cache-policy development is frozen. The qualified K3 CUDA
profile, A-only hotlist, linear LFRU policy, router arithmetic, expert layout,
and storage settings are not tuning dimensions.

This document is frozen before any new target or draft forward pass. A+B are
development data, legacy H1 is veto-only, and the untouched Sprint 15 H2 tier
remains under inference embargo until one candidate has passed every earlier
gate. Tokenization and load-only memory planning do not spend H2; the first
forward step by either model on an H2 prompt spends the whole tier.

## Frozen starting point

| Item | Value |
|---|---|
| Parent commit | `c8945e44331c9ffedbd8a2849634a014f4c3a916` |
| Branch | `exp/speculative-decoding-gb10` |
| A+B/H1 corpus | `docs/gn100/sprint14-heldout-corpus.json` |
| A+B/H1 SHA-256 | `4fb4ca60d82ed521c0d59732c748a5f99b22f720d35e19d5590b41bc01c423fe` |
| Unspent H2 corpus | `docs/gn100/sprint15-h2-corpus.json` |
| H2 SHA-256 | `5c5b806358a26bb8d9ce782aab190b7e10bd83f00e445d1bab7367333cee5bee` |
| K3 manifest SHA-256 | `da3232925d8eeaf0df07ee45743f6467c7eecbc4c0e2513899028e788a1a07f2` |
| Kimi-Linear manifest SHA-256 | `d287b2cc45b7f6b3800c1a8114aafcc3eedf163947cf164e5aa3d3df18494369` |
| Shared tokenizer SHA-256 | `b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103` |
| K3 specials SHA-256 | `d3350d087a42ad97d800f7187264f09e8188b2ab9e02485b89d494831c99d0c3` |
| Kimi specials SHA-256 | `64fdacd414acbc34c28745b96a0197d4f8246b78b178c44a1dbfb416a658b18a` |

The registered K3 control is the Sprint 13 strict profile: CUDA KDA 1, CUDA
dense 2, CUDA VQ 2/group 1, 10 performance CPUs, Q0, Q8 enabled, SDOT/I8MM
disabled, 59,340 MiB expert cache, two readers at depth two, lookahead zero,
direct I/O, and the A-only calibration hotlist. Prior compression and decode
aging are disabled. The sustained studio reference is 0.944126 tok/s; the
short in-sample reference is 1.0036145 tok/s. These are historical anchors,
not substitutes for the paired Sprint 16 controls.

The K3 and draft token-ID-to-piece maps must match. Sprint 16 permits no token
translation, approximate vocabulary mapping, sampling, temperature, alternate
draft, or adaptive block width.

## Candidate cells

Two draft prompt encodings are registered:

1. `target_ids`: give Kimi-Linear the canonical K3 prompt token IDs.
2. `kimi_native`: render the same system and user text with Kimi-Linear's
   native chat markers. Only the draft's initial context differs; every
   committed continuation token is shared.

The only widths are `k=2`, `k=4`, and `k=8`. Both encodings are traced once at
maximum width eight. All narrower statistics and trajectories are derived
from that fixed trace; they are not separate experiments.

## Memory and comparator contract

Three arms separate speculation from the cost of making speculation possible:

- `F`, full control: K3 alone with the qualified 59,340 MiB expert cache.
- `R`, resident control: K3 decodes normally while Kimi-Linear, both prompt
  states, and all rollback buffers are allocated, touched, and resident.
- `S`, speculation: byte-for-byte the same allocation and reduced K3 cache as
  `R`, with the selected speculative loop enabled.

`S/R` measures mechanism gain. `S/F` is the practical result. Recovering only
the cost of loading the draft is not a win.

Before prompt inference, one load-only step will record the exact model floors,
full Kimi expert-cache charge, live snapshot sizes, and a single joint budget.
The deterministic rule is to keep total engine-plus-rollback memory no larger
than the qualified `F` allocation: Kimi's complete engine charge and one
touched root snapshot per model displace K3 expert cache. No alternate joint
budget may be tried in this sprint. Loaded high-water `MemAvailable` must
remain at least 24 GiB with zero process swap; otherwise Sprint 16 stops before
agreement tracing.

`R` and `S` use that same budget, warm the same K3 entries in the same order,
and allocate the same buffers. `F` remains the best usable no-draft profile.

## One maximum-width viability campaign

The four A and four B cases select. The four H1 cases are measured in the same
fixed campaign but can only veto. Development continuations are 32 greedy K3
tokens per case. Before B or H1 is used, the first 16 tokens of each new A
capture must reproduce the retained Sprint 14 A token sequence and token,
full-logit, and ordered-route hashes; a mismatch stops the campaign.

At every canonical K3 block root, the draft proposes up to eight tokens in
order. K3 serial verification records the exact first mismatch and the bonus
token. For each format report, by case and tier:

- marginal next-token match;
- conditional match after `j-1` accepted proposals;
- prefix survival `Q_j = P(first j proposals all match)`, `j=1..8`;
- accepted-run and first-mismatch histograms;
- full-block acceptance;
- proposal, verifier, replay, snapshot, restore, synchronization, expert-union,
  miss, and byte costs.

For descriptive checking,

```text
E[N_k] = 1 + sum(j=1..k, Q_j)
```

under standard correction-or-bonus semantics. Selection uses the deterministic
trajectory simulation because block roots and failures are correlated. It may
not substitute a marginal-agreement or independent-`p` approximation.

The timing calibration includes actual draft branches, exact serial target
verification, the existing chunk path as a non-exact performance bound, state
export/import, committed replay, direct tails, and co-resident cache effects.
Predicted throughput is:

```text
committed tokens * ordinary-K3 seconds/token
---------------------------------------------------------------
draft + verify + snapshot/restore + replay + sync + direct-tail
```

calculated separately against `R` and `F`.

For each of the six `(format,k)` cells, the score is the smaller of its A+B
median predicted gains against `R` and `F`. Highest score wins. Within one
percentage point choose smaller `k`; if still tied choose `target_ids`.

The selected cell proceeds only when all are true:

- A+B predicted median gain is at least 15% against both `R` and `F`;
- no A+B case is worse than either comparator by more than 5%;
- H1 median predicted gain is nonnegative against both comparators; and
- no H1 case is worse by more than 5%.

H1 cannot select or rescue. If the A+B winner fails H1, no runner-up is tried.
A miss stops implementation and leaves H2 unspent.

## Exact verifier and semantic contract

Only the selected cell may be integrated. A serial per-position verifier is
the bit-exact reference. Any optimized verifier must preserve the qualified
decode arithmetic, including router-order expert accumulation, and return the
logit row and ordered route row at every proposal position.

Greedy verification uses standard semantics:

- first mismatch commits K3's exact argmax correction;
- a full match commits K3's exact next-token bonus;
- every committed correction or bonus is fed back into the draft;
- rejected tokens are never exposed;
- stop, EOS, cancellation, and the token limit leave both states at exactly
  the externally committed boundary.

For every row retain verifier blocks `Q`, proposals `P`, accepted proposals
`A`, rejected proposals `P-A`, corrections `M`, bonuses `U`, direct-tail
tokens `D`, replay/resynchronization tokens, state-copy counts/bytes, target
verifier tokens, and first-mismatch positions. The identities are:

```text
committed = A + M + U + D
Q = M + U
proposals = accepted + rejected
```

Every target/draft forward call must reconcile with those semantic counts.
Rejected-branch routes, cache traffic, and logits are retained separately.
Cache metadata is not rolled back, so rejected and replay traffic remains
charged.

The numerical gate is strict:

- byte-identical emitted K3 token IDs;
- byte-identical K3 logits and ordered routes on the committed trajectory;
- byte-identical exported final KDA/MLA state after every block and row;
- finite logits, deterministic counters, correct CUDA call counts;
- zero CUDA or I/O fallback.

No tolerance, route-divergence allowance, or third contract class is permitted.
If exact batched verification is impossible, that is the Sprint 16 result.

## Integrated development gate

Only a selected cell receives a 64-token A+B/H1 campaign. Each arm has two
valid repeats, with fixed bracketing:

1. `F`, forward case order;
2. joint-load `R,S`, reverse case order;
3. joint-load `S,R`, forward case order;
4. `F`, reverse case order.

No valid row is discarded and no selective third repeat is allowed. An
infrastructure retry is allowed only when no valid timed row was produced and
its cause is archived. The case statistic is its two-repeat median.

All must pass:

- A+B median `S/F` and `S/R` gain at least 10%;
- at least six of eight A+B cases improve against `F`;
- no A+B case regresses more than 5% against either comparator;
- H1 median gain is nonnegative against both comparators;
- no H1 case regresses more than 5%;
- median end-to-end request time, including draft preparation, is no worse
  than `F`;
- exactness, accounting, memory, direct-I/O, Q0, and process-safety gates pass.

H1 remains veto-only. Any miss ends the sprint without changing the format,
width, joint budget, verifier, or thresholds. A passing candidate receives one
engine-level SSD-contention screen before H2.

## H2 single-use confirmation

Before an H2 model step, seal the implementation and binary hashes, selected
format and width, model/tokenizer/template hashes, full and joint budgets,
hotlist and warm counts, runner, analyzer, phase order, statistics, thresholds,
retry rule, A+B/H1 raw evidence, and expected accounting identities.

H2 then tests only that frozen candidate on its six existing families, 64
generated tokens, two repeats, and the same `F/R/S` bracketing. The first model
step consumes H2. A substantive change or retry after a valid row requires a
fresh H3 corpus.

The H2 gate requires:

- six-family median `S/F` and `S/R` gain at least 10%;
- at least five of six families improve against `F`;
- no family regresses more than 2% against either comparator;
- overall median `S` throughput at least 1.000 tok/s;
- every family's two-repeat `S` median at least 0.950 tok/s;
- median end-to-end `S` request time no worse than `F`;
- every exactness, accounting, memory, storage, Q0, and zero-fallback check.

A miss is retained as the Sprint 16 result. Passing does not automatically
promote to `spark/integration` or trigger an upstream submission.

## Publication and stop rules

Stop at the registered boundary on tokenizer incompatibility, insufficient
memory, swap, timed major faults, profile drift, non-finite logits, numerical
or state mismatch, accounting mismatch, fallback, or dirty Q0 teardown. A
valid performance miss does not authorize another candidate.

Every outcome is published on the experimental branch with the preregistration,
machine-readable summary, source/model/corpus hashes, manifest-verified raw
evidence, and explicit H2 spent/unspent status.
