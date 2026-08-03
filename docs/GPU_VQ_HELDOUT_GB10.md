# GB10 CUDA VQ held-out hotlist adaptation: Sprint 14 preregistration

Sprint 13 crossed 1 tok/s over 16 generated tokens with a hotlist built from
the route capture it was measured on. That was a useful ceiling, but it was
in-sample, and the same profile reached 0.944126 tok/s over the full 64-token
continuation. Sprint 14 asks whether a calibration-only hotlist can generalize
to new studio prompts and whether a small, decode-only LFRU aging rule lets the
fully warmed cache adapt without giving up the useful calibration records.

The work stays on `exp/cuda-vq-heldout-gb10`. It does not change model
arithmetic, routing, expert layout, cache capacity, CUDA kernels, or router
order. The only experimental policy is an interval that ages LFRU hit counts
after completed decode steps. The no-aging control and aging treatment use the
same sealed hotlist and fully warm the same 59,340 MiB cache.

## Frozen inputs

The corpus, split, rendering, and order were frozen before any calibration
route capture or throughput result.

| input | frozen value |
| --- | --- |
| branch parent | `dbaf26f9f0144f6193215987461d9c934e2f2ea5` |
| model manifest SHA-256 | `da3232925d8eeaf0df07ee45743f6467c7eecbc4c0e2513899028e788a1a07f2` |
| Sprint 13 baseline usage SHA-256 | `e2949e8d14da3c46d766daa99e6b93f946c8cfc0425eba4815450079996899dd` |
| canonical corpus | `docs/gn100/sprint14-heldout-corpus.json` |
| corpus SHA-256 | `4fb4ca60d82ed521c0d59732c748a5f99b22f720d35e19d5590b41bc01c423fe` |
| renderer | `serve.xtml.build_chat_segments` at the branch parent |
| rendering controls | generation prompt on, thinking off, no tools, no BOS |
| tokenizer boundary | every XTML segment encoded separately in markup or plain mode |

The common system message is:

> You are an incisive studio critic helping a practicing visual artist make
> concrete decisions. Be concise, specific, and attentive to formal
> relationships.

The JSON file is authoritative for the complete rendered token IDs. The IDs
include the common system turn, user turn, and opened assistant turn; they are
not the tokenization of the user sentence alone. Concatenating XTML and
tokenizing it once is forbidden because that changes markup/plain security
boundaries and BPE boundaries.

The twelve cases form three deliberately different evidence tiers:

| tier | case | family | tokens | rendered-segments SHA-256 |
| --- | --- | --- | ---: | --- |
| calibration A | `composition_a` | composition | 74 | `19f32429fba4ee3c901cb89317780d37057fa61ab540e083f3373d246bf8a20e` |
| calibration A | `revision_a` | revision | 73 | `3753744603e7e534449053b08705c5dd6fbdf016d1e69fe6911d5766c33d5dd0` |
| calibration A | `color_value_a` | color/value/edge | 74 | `b2edf368da10208f92c14166b79a3c041c309477867c898e830386a0ffaeb734` |
| calibration A | `material_a` | material/process | 76 | `f9c60b5c2e3f33ea87b72406db61ecb50ee1e57d7d88a166e8598ae673010810` |
| within-family B | `composition_b` | composition | 78 | `ab4fdafd63a6d4cb8a8c9ec0140dfb208c999f75a6d3de75283fef54ee1a6a7f` |
| within-family B | `revision_b` | revision | 77 | `2386ed0352dd2018020d760a7ab8c8165f2283c5a0acb332e896b1403bed521f` |
| within-family B | `color_value_b` | color/value/edge | 78 | `a55fdd7f5e32430835f395acd140e33f2df7d40490c7b5e3b0053830ffc3e209` |
| within-family B | `material_b` | material/process | 81 | `f53aa671d0aac636d77b1ba234d15d759fe8d82f1c31cb65795a38f25587a244` |
| unseen H | `ambiguity_h` | meaning/ambiguity | 76 | `24123790e5a478a558b3defae03993a33a66dc39cc7e2ae5648f95588aa77381` |
| unseen H | `history_h` | art history | 77 | `9415b827d50879e560ed78ab2f3c47ea9fec0e33e073ca31b350fc0c53faf1b7` |
| unseen H | `series_h` | series development | 80 | `a4336625d50f3c963e8191074c48e5b9e7ff145ddedb400664500cb44dfac0cb` |
| unseen H | `display_h` | display/installation | 79 | `41c14a962385d1d54aa62cb416d2baeac7b48d9a7baff7092189ad70a303115a` |

A cases alone train the hotlist. B cases test new wording inside the four known
task families. H cases test four task families absent from calibration and
supply the only generalization gate. All cases intentionally share the studio
system root, so “unseen family” means an unseen studio task family, not an
unseen system prefix.

Before the first B or H inference, the evidence manifest must record the
implementation commit, all four A-capture hashes, the resulting full-hotlist
SHA-256, its entry/selection/token counts, and its effective warm count. Those
values are filled once from artifacts; they may not be changed after a B or H
route exists.

## Fixed execution orders

Prompt order is part of the preregistration.

1. Calibration capture:
   `composition_a, revision_a, color_value_a, material_a`.
2. Lookahead-zero 16-token campaign:
   `composition_a, revision_a, color_value_a, material_a,
   composition_b, revision_b, color_value_b, material_b,
   ambiguity_h, history_h, series_h, display_h`.
3. Lookahead-six H qualification, run only after the lookahead-zero H gate:
   `display_h, series_h, history_h, ambiguity_h`.
4. Final 64-token H campaign:
   `ambiguity_h, history_h, series_h, display_h`.

Every comparative prompt run uses one model load and `lfru_age=0,4` for two
interleaved repeats. The existing sweep reversal therefore fixes the arm order
as age 0 then age 4 in repeat one, and age 4 then age 0 in repeat two. The four
one-arm A captures that train the hotlist precede these comparative runs.

## Profile held fixed

- `WASTE_CUDA_KDA=1`, `WASTE_CUDA_DENSE=2`,
  `WASTE_CUDA_VQ=2`, and `WASTE_CUDA_VQ_GROUP=1`;
- ten compute threads on CPUs `5-9,15-19`, under child-scoped Q0;
- Q8 on and SDOT/i8mm off;
- 59,340 MiB expert cache with LFRU policy;
- direct I/O, two reader threads, queue depth two;
- mlock and purgeable slots off;
- greedy decoding from the exact rendered IDs in the canonical corpus;
- lookahead zero for the first campaign, then width six only after the H gate.

No cache-size, hotlist-fraction, thread-count, queue-depth, grouped-VQ,
CUDA-graph, router, or kernel arm is reopened. The Sprint 13 in-sample hotlist
may be reported separately but cannot train, select, or satisfy Sprint 14.

## The decode-only LFRU aging arm

The sweep key is `lfru_age=0,4`.

- Age 0 disables aging.
- Age 4 increments a per-arm decode phase counter only after a generated
  `waste_model_step` completes successfully.
- After each fourth successful generated step, while holding the cache
  metadata lock, every resident `EC_READY` entry is updated as
  `hits = hits > 1 ? hits / 2 : hits`.
- One remains one. The `last` recency field is not changed.
- Warming, prompt prefill, failed steps, and speculative-tail draining do not
  advance or apply aging.
- The decode phase counter is reset for every sweep arm.

Both arms clear the cache, fully warm from the identical sealed calibration
hotlist, and perform the identical CPU prefill before CUDA decode is enabled.
There is no partial warm, `--max-entries`, timestamp offset, second hotlist,
or held-out retraining. The experiment asks whether inherited calibration
frequency should decay during a new continuation; it does not test admission
fraction.

The implementation must make `lfru_age` a first-class CUDA-VQ sweep arm:
CUDA is disabled during prefill and restored for decode, semantic counters are
reset and asserted, captures identify the aging arm correctly, and the
effective interval and completed aging passes are reported. Expected aging
passes are zero for age 0, four for an age-4 16-token arm, and sixteen for an
age-4 64-token arm.

## Calibration and leakage boundary

Each A case produces one 16-token route capture under the fixed profile,
lookahead zero, no aging, and the immutable Sprint 13 baseline usage file.
Together they contain 64 routed decode tokens and exactly 94,208 selected
expert slots. Cache policy cannot alter routes, but fixing the capture
conditions removes an avoidable provenance difference.

One candidate is then built from the four explicit A manifests:

```bash
python3 tools/capture_to_usage.py \
  captures/composition_a/cuda_vq-2-rep1.json \
  captures/revision_a/cuda_vq-2-rep1.json \
  captures/color_value_a/cuda_vq-2-rep1.json \
  captures/material_a/cuda_vq-2-rep1.json \
  -o usage-sprint14-calibration.waste
```

The converter writes every aggregated entry. The output is not truncated.
The candidate SHA-256 and counts are sealed before any B or H model step. The
same pathname and hash must appear in every age-0 and age-4 row. B and H
routes are evidence only: they are never added to the hotlist, and no
parameter may be changed in response to them.

## Campaign sequence

1. Verify the branch source, model-manifest, baseline-usage, and corpus hashes;
   verify exclusive model-SSD use, recovered memory, and no active
   WASTE/fio/server process.
2. Capture the four A prompts in the fixed order and build the single full
   calibration hotlist.
3. Seal the implementation, A captures, hotlist, and effective warm count.
4. Run all twelve prompts at lookahead zero for 16 tokens, with two
   interleaved repeats of `lfru_age=0,4`. A is an in-sample diagnostic, B is
   a within-family diagnostic, and H alone decides the policy.
5. If and only if the H gate passes, run the four H prompts at lookahead six
   for 16 tokens and two interleaved repeats, in the fixed reverse order.
   This is an interaction qualification, not a new selection opportunity.
6. Run the final four H prompts at lookahead six for 64 tokens and two
   interleaved repeats, in the fixed forward order. Age 0 remains the matched
   control; age 4 is the fixed treatment evaluated against the 1 tok/s target.
7. Retain complete age-0/age-4 logit and route captures for
   `ambiguity_h` in the final campaign and compare them with
   `tools/compare_gpu_runs.py`.

A correctness or safety failure stops immediately. A performance miss ends
the sprint with the fixed result; it does not authorize another interval,
hotlist, cache budget, or prompt.

## Exact accounting and correctness

CUDA is decode-only, so prompt work is excluded from these registered counts.

| counter | per token | 16 tokens | 64 tokens |
| --- | ---: | ---: | ---: |
| KDA calls | 552 | 8,832 | 35,328 |
| dense calls | 580 | 9,280 | 37,120 |
| routed VQ experts | 1,472 | 23,552 | 94,208 |
| VQ applications | 4,416 | 70,656 | 282,624 |
| VQ LUT builds | 1,656 | 26,496 | 105,984 |
| CUDA launches | 4,508 | 72,128 | 288,512 |
| CUDA synchronizations | 2,944 | 47,104 | 188,416 |

Every row must report the exact applicable counts, VQ mode 2/group 1, and zero
fallback. For a given prompt, both aging arms and both repeats must have
identical greedy-token, full-logit, and ordered-route hashes. The retained
64-token capture additionally requires byte-identical finite logits, zero
argmax/top-ten changes, and zero route membership or order changes.

Hits, misses, expert bytes, and eviction timing are expected to differ between
aging arms; requiring those to match would invalidate the policy experiment.
The full-hotlist SHA and number of initially warmed entries must match.

## Gates

For each H family and arm, throughput is the median of its two 16-token
repeats. Family gain is `median(age4) / median(age0) - 1`. The primary H gain
is the median of the four family gains. Across-H misses and bytes are sums of
the eight rows for each arm.

The lookahead-zero aging gate requires all of the following:

- primary H throughput gain at least 5%;
- no H-family throughput gain below -2%;
- age-4 aggregate H expert bytes no greater than age 0;
- age-4 aggregate H misses at least 10% lower than age 0;
- identical token/logit/route hashes, exact counters, expected aging-pass
  counts, and zero fallback;
- all memory, storage, and Q0 safety checks below.

A and B results are reported by tier but cannot compensate for an H failure.
If this gate fails, lookahead six and the 64-token target are not run.

Passing the gate fixes interval four. The lookahead-six 16-token campaign must
retain the numerical and safety contract, but it does not tune or replace the
interval. The final practical target is evaluated on the eight age-4 H rows
from the 64-token campaign:

- their overall median must be at least 1.000 tok/s; and
- each H family's two-repeat median must be at least 0.950 tok/s.

The paired age-0 rows quantify the gain but do not satisfy the age-4 target.
A result below target is published as a held-out miss, without training on H.

## Memory, storage, and publication

Every campaign records process major faults, host swap before and after,
loaded high-water `MemAvailable`, effective reader state, warmed entries,
Q0 holder status, and child exit. Acceptance requires no process swap, no
increase in host swap use, at least 24 GiB `MemAvailable` at loaded
high-water, clean Q0 teardown, and no competing model, server, or fio job.

Publication means the experimental branch, a machine-readable result summary,
exact source/model/corpus/capture/hotlist hashes, and immutable raw evidence.
It does not promote the aging policy to `spark/integration` or propose it
upstream automatically.

## Measured result and disposition

The frozen campaign completed all calibration-A, within-family-B, and unseen-H
lookahead-zero rows. The frozen analyzer returned exit 2: the evidence was
valid, but the H selection gate did not pass. Median family gains were 0.2365%
for calibration A and 0.9740% for within-family B; neither diagnostic tier was
used for selection.

The unseen-family result was:

| H case | family | age-4 throughput gain |
| --- | --- | ---: |
| `ambiguity_h` | meaning/ambiguity | 4.6457975% |
| `history_h` | art history | 7.3350770% |
| `series_h` | series development | 4.4689000% |
| `display_h` | display/installation | 1.5946348% |
| **median** | **selection statistic** | **4.5573488%** |

The median gain missed the required 5%. Across the eight H rows per arm,
age-four misses fell from 96,614 to 88,984, a 7.8974% reduction against the
required 10%. Expert bytes moved in the required direction, falling by
94,663,761,920 bytes from 1,198,669,029,376 to 1,104,005,267,456. No H family
regressed, so the -2% family floor passed.

All numerical and safety contracts passed: paired arms retained identical
token, full-logit, and ordered-route hashes; CUDA semantic, launch,
synchronization, fallback, and aging-event counts were exact; the byte gate,
memory/storage checks, and clean Q0 teardown passed. The failed throughput and
miss-reduction thresholds were nevertheless decisive because the gate required
every condition.

The preregistered stop was honored. The lookahead-six H qualification and the
64-token H campaign were not run, interval four was not selected, and no part
of this policy moves to `spark/integration` or an upstream proposal. This is a
scoped negative result for this frozen corpus, hotlist, cache budget, and aging
interval; it does not reject cache adaptation as a class. See the
[machine-readable Sprint 14 summary](gn100/sprint14-heldout-aging-summary.json)
and [frozen corpus](gn100/sprint14-heldout-corpus.json).
