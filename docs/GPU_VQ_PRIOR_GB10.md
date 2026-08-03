# GB10 inherited-hotlist prior compression: Sprint 15 preregistration

Sprint 14 tested one decode-aging interval on a held-out corpus and missed its
frozen gate. Testing more intervals against that same corpus would turn the
campaign into an adaptive search. Sprint 15 therefore retires Sprint 14's H
tier to development use, tests one different policy shape, and reserves a new
six-family H2 tier for a single confirmatory decision.

The experiment stays on `exp/cuda-vq-prior-compression-gb10`. It changes only
the inherited LFRU score after warm selection and before prompt prefill. It
does not change model arithmetic, routing, expert layout, cache capacity,
warm selection, CUDA kernels, router order, or the qualified execution
profile.

## Frozen inputs

These values were fixed before candidate implementation or any H2 model step.

| input | frozen value |
| --- | --- |
| branch parent | `3d61aab9034ef806a4e915d4f047db019e012507` |
| model manifest SHA-256 | `da3232925d8eeaf0df07ee45743f6467c7eecbc4c0e2513899028e788a1a07f2` |
| calibration hotlist SHA-256 | `6ef8b0e752b4c5e369eae1e088010bb47243f6ee4d1d9470053067c4e85f693e` |
| calibration hotlist | Sprint 14 A-only, 64 routed tokens, 23,828 entries |
| effective warm count | 5,015 entries in a 59,340 MiB cache |
| legacy development corpus | `docs/gn100/sprint14-heldout-corpus.json` |
| fresh confirmatory corpus | `docs/gn100/sprint15-h2-corpus.json` |
| H2 corpus SHA-256 | `5c5b806358a26bb8d9ce782aab190b7e10bd83f00e445d1bab7367333cee5bee` |
| renderer | `serve.xtml.build_chat_segments` at the branch parent |
| rendering controls | generation prompt on, thinking off, no tools, no BOS |
| tokenizer boundary | every XTML segment encoded separately in markup or plain mode |

The common system message remains:

> You are an incisive studio critic helping a practicing visual artist make
> concrete decisions. Be concise, specific, and attentive to formal
> relationships.

H2 was frozen with `tools/freeze_prompt_corpus.py` and the converted K3
tokenizer on the Spark. Before writing H2, that process exactly reproduced the
token IDs and segment hashes of all twelve frozen Sprint 14 cases. Tokenizing
H2 did not execute a model forward step and did not expose routes, cache
behavior, logits, or timing.

The new families and their fixed forward order are:

| case | family | tokens | rendered-segments SHA-256 |
| --- | --- | ---: | --- |
| `documentation_h2` | documentation/reproduction | 78 | `11fc1cbe25eab9db6c0293c7de3d2c3ba54185443b2d4df56e409359a85246ed` |
| `conservation_h2` | conservation/stewardship | 76 | `3b3278ccb5b15de8d4d3697073d179020ec40d99a1554fd7e02f5899b1e9f17f` |
| `accessibility_h2` | accessibility/description | 79 | `da4d8cf7e5ef69f8e90f2ec3adf0581dda38619d60cff33b455cb5451e419e68` |
| `rights_h2` | rights/permissions | 81 | `10129f1bb95c5aa7041343c5f238d64640a24ca608604c92a6463cc7d89161df` |
| `portfolio_h2` | portfolio/submission | 79 | `44c129c8da937d60e765b851ab3f32d5135838fa023c126d5e5b921f747df99e` |
| `proposal_h2` | proposal/commission | 82 | `cb1c4233080c81617e12dc57adeb024f215b6e105792eba5b280c7b928114807` |

These professional and operational studio tasks are absent from calibration
A, within-family B, and legacy H1. H2 never trains or rebuilds a hotlist.

## The single candidate

The control and treatment both use the complete, identical calibration
hotlist. Raw hit counts first sort and select the same 5,015 entries in the
same order. As each selected entry is loaded, treatment assigns its initial
LFRU score from the raw imported score `p`:

```text
C(0) = 0
C(p) = 1 + floor(log2(p)) for p > 0
initial hits = C(p)
```

Control assigns the original `p`. The transform occurs only after raw sorting
and top-5,015 selection, so initial membership and load order are identical.
All later current-prompt and decode hits increment the assigned score normally
and are never transformed. Prefill-only entries, later misses, and rebound
slots therefore contain only live evidence. Zero is defined defensively even
though the sealed hotlist contains no zero counts.

The mode is default-off, LFRU-only, and exposed as
`WASTE_LFRU_PRIOR_LOG2=0|1`; the registered sweep is `lfru_prior=0,1`.
Treatment must report `lfru_prior_events=1`,
`lfru_prior_entries=5015`, and `warmed=5015`. Control must report zero prior
events and entries while still reporting `warmed=5015`. A failed warm or any
mismatch among those counts invalidates the row. Decode aging remains disabled
and must report zero aging events in both arms.

The transform is monotone but not strictly monotone. For the warmed set it
maps raw scores 5 through 63 to four score buckets: 2,102 entries at 3, 1,943
at 4, 740 at 5, and 230 at 6. It preserves bucket ordering, not the complete
raw ordering inside a bucket. The hotter raw entries load first and therefore
start with older synthetic recency; after bucketing they can be evicted before
cooler entries in the same bucket. Raw counts determine initial membership;
log-bucket score plus current-request recency governs residency thereafter.
This experiment does not add a secondary raw-score field.

The candidate operates during CPU prefill as well as decode. Each row must
therefore report prefill elapsed time, prefill hits, misses, and bytes, plus
the cache state entering decode. These are secondary diagnostics, not H2
selection statistics, but make thermal/DVFS carryover and residency changes
visible. The frozen hotlist is read-only: no arm may save usage over it or
feed compressed-plus-live scores into a later warm.

No transform alternative, interval, exponent, threshold, warm fraction,
cache size, or tie-break rule may be selected after development begins.

## Fixed profile

- `WASTE_CUDA_KDA=1`, `WASTE_CUDA_DENSE=2`, `WASTE_CUDA_VQ=2`, and
  `WASTE_CUDA_VQ_GROUP=1`;
- ten compute threads on CPUs `5-9,15-19`, under child-scoped Q0;
- Q8 on and SDOT/i8mm off;
- 59,340 MiB LFRU expert cache, fully warmed from the sealed hotlist;
- direct I/O, two effective reader threads, queue depth two;
- mlock, purgeable slots, and LFRU decode aging off;
- greedy decoding from the canonical rendered token IDs;
- CPU prefill, with CUDA restored only for decode;
- lookahead zero until the conditional qualification phases.

No qualified-profile parameter is reopened.

## Fixed phases and orders

Every case uses one model load, exactly one clear-and-warm per arm, two
interleaved repeats, and arm order
`0,1` followed by `1,0`. Phase names determine the permitted corpus, token
count, lookahead, capture requirement, and prompt order; callers cannot
override them.

1. `dev-look0-16`: the Sprint 14 corpus in its original forward order:
   `composition_a, revision_a, color_value_a, material_a, composition_b,
   revision_b, color_value_b, material_b, ambiguity_h, history_h, series_h,
   display_h`; 16 generated tokens, lookahead zero.
2. `h2-look0-16`: the six H2 cases in the table's forward order; 16 generated
   tokens, lookahead zero.
3. `look6-h2-16`: only after an H2 pass, the exact reverse H2 order
   `proposal_h2, portfolio_h2, rights_h2, accessibility_h2,
   conservation_h2, documentation_h2`; 16 tokens, lookahead six.
4. `look6-h2-64`: only after a valid interaction phase, H2 in forward order;
   64 tokens, lookahead six.

The runner must enforce predecessor completion and cross-prompt order. The
analyzer uses stages `development`, `h2`, and `final`: exit 0 is valid/pass,
exit 2 is valid/performance miss, and exit 1 is invalid or incomplete. The
final stage returns 2 when the 64-token practical target misses even if H2
passed earlier.

## Development admission

A, B, and legacy H1 are development data. They can reject this candidate but
cannot support a confirmatory claim.

For A+B alone, all of these must hold:

- the median of the eight paired prompt throughput gains is greater than 0;
- aggregate treatment misses are lower than control;
- aggregate treatment expert bytes are lower than control; and
- no A/B prompt throughput gain is below -5%.

Legacy H1 is veto-only: its aggregate misses and bytes may not worsen and no
H1 prompt may regress below -5%. An H1 improvement cannot rescue a failed A+B
condition. Every row must also pass the numerical, accounting, and safety
contracts below.

A development miss ends the sprint. It does not authorize another transform,
a relaxed gate, or an H2 run. A pass freezes the already registered candidate
and permits H2; development results do not alter it.

## Single-use H2 contract and gate

H2 is a one-candidate, single-use confirmatory tier. Before its first model
step, the implementation commit, transform, model and hotlist hashes, warm
order/count, corpus and token hashes, profile, phase orders, runner, analyzer,
statistics, thresholds, and retry policy must be sealed in the evidence
manifest. The first model step consuming any H2 case spends the entire tier.

After that point, H2 may evaluate only the sealed candidate. No H2 output,
route, cache, or timing result may change the policy, capacity, hotlist,
profile, prompts, order, statistics, or thresholds. A substantive change
requires a fresh H3 tier. An infrastructure-only rerun is allowed only when
no valid timed row was produced; its cause and disposition must be recorded
before resuming.

For each H2 family and arm, throughput is the median of its two repeats.
Family gain is `median(treatment) / median(control) - 1`. The lookahead-zero
H2 gate requires every condition:

- median of the six family gains at least 5%;
- no family gain below -2%;
- at least four of six family gains greater than 0;
- aggregate treatment misses across all twelve treatment rows at least 10%
  below control;
- aggregate treatment expert bytes no greater than control; and
- every numerical, accounting, and safety check below passes.

A miss is the Sprint 15 result. H2 then becomes development data and another
candidate requires H3. A pass permits only the fixed lookahead-six interaction
and final-target phases; those characterize the accepted candidate and are
not additional selection opportunities.

The final practical target is evaluated on the twelve treatment rows from
`look6-h2-64`:

- overall median throughput at least 1.000 tok/s; and
- each H2 family's two-repeat median at least 0.950 tok/s.

## Numerical and accounting contract

For each case, both arms and both repeats must have identical greedy-token,
full-logit, and ordered-route hashes. The retained 64-token comparator must
also report byte-identical finite logits, zero argmax or top-ten changes, and
zero route membership or order changes.

CUDA decode accounting remains exact:

| counter | per token | 16 tokens | 64 tokens |
| --- | ---: | ---: | ---: |
| KDA calls | 552 | 8,832 | 35,328 |
| dense calls | 580 | 9,280 | 37,120 |
| routed VQ experts | 1,472 | 23,552 | 94,208 |
| VQ applications | 4,416 | 70,656 | 282,624 |
| VQ LUT builds | 1,656 | 26,496 | 105,984 |
| CUDA launches | 4,508 | 72,128 | 288,512 |
| CUDA synchronizations | 2,944 | 47,104 | 188,416 |

Every row must report VQ mode 2/group 1, the exact semantic counts, zero CUDA
fallback, effective prior mode, prior event/entry counts, and zero aging
events. The two repeats of a given arm must also have identical hits, misses,
and expert bytes; a mismatch invalidates evidence rather than becoming a
performance observation.

## Safety, stopping, and publication

Every phase records the load header and requires direct I/O 1, two readers,
depth two, the expected usage path, prompt/generated lengths, warmed entries,
process major faults, host swap before and after, loaded high-water
`MemAvailable`, Q0 holder state, and child exit. Acceptance requires no
process swap, no increase in host swap use, at least 24 GiB `MemAvailable` at
loaded high-water, clean Q0 teardown, and no competing model, server, or fio
job.

A correctness, accounting, profile, or safety failure stops immediately. A
valid performance miss stops at its registered boundary. All outcomes are
published on the experimental branch with machine-readable stage summaries,
the pre-H2 seal if reached, source references, manifest-verified raw evidence,
a tag, and an experimental release. This sprint does not promote a policy to
`spark/integration` or propose it upstream automatically.
