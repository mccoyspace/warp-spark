# Route margins and cache-conditional selection: what sprint11 does and does not settle

Source data: `docs/gn100/sprint11-route-margin-summary.json` (mccoyspace/warp-spark,
`spark/integration` @ `cb05b14`). 6,348 routed rows, 92 layers, K3.

Cross-read against: sqliteai/warp PR #46, "Prefer experts the cache already has,
for ranking only" (mfethe1), commit `33b4d2e`.

This note contributes no mechanism. It is an independent read of the sprint11
distribution applied to a proposal on another fork, plus a correction of our own
first reading of it.

## Correcting our first reading

Our initial pass treated `exact_boundary_ties = 0` as decisive against PR #46, on
the assumption that cache preference acted as a tie-break. **That was wrong.**
PR #46 is not a tie-break. It adds an explicit bias to the selection value:

```c
v = score[e] + bias[e] + lambda * (hi - lo) * resident[e]
```

where `hi - lo` is the layer's own selection-value range. The tie count is
therefore irrelevant to it, and we withdraw that argument.

The second correction matters more. `margin < eps` is **necessary but not
sufficient** for a route to flip. A flip also requires the specific rejected
expert to be resident *and* the boundary winner not to be. Margin thresholds
bound **susceptibility**, not observed divergence. Any flip percentage derived
from margins alone — including ours — is an upper bound on an upper bound, and
we are not publishing those interpolated numbers as measurements.

## What the published quantiles actually say

Margin = selection value of the last selected expert minus the first rejected.

| quantile | margin |
| --- | ---: |
| min | 4.172e-07 |
| p1 | 9.774e-06 |
| p5 | 5.039e-05 |
| p50 | 8.212e-04 |
| p95 | 6.327e-03 |
| max | 4.628e-02 |

Per token position, the weakest decision anywhere in the 92-layer stack:
p50 = 7.957e-06, p95 = 2.630e-05, max = 4.682e-05. 43 of 92 layers contain a
decision below 1e-05; 91 of 92 below 1e-04.

These are the reported quantiles, unmodified. We do not interpolate between them
here.

## Where the two datasets genuinely meet

PR #46's bump is `lambda * (hi - lo)`, scaled by the layer's selection-value
range. The sprint11 margins are in the same units. That makes one comparison
available without any modelling:

**For any plausible layer range, `lambda = 0.1` produces a bump that is orders of
magnitude larger than the median margin (8.2e-04).** The selected set is
therefore expected to move on most tokens — not as a subtle perturbation but as
a routine one.

PR #46 already reports exactly that, and prices it honestly rather than hiding
it: KL 0.0232 at `lambda = 0.1`, against README's own accepted bar of KL 0.037
for top-8, with the caveat stated up front that it does **not** reproduce the
greedy continuation the way top-8 does, and that it is off unless
`WASTE_CCR_LAMBDA` is set.

**So sprint11 does not refute PR #46. It supplies the mechanism behind PR #46's
measured KL cost.** The margin distribution is the structural reason a
residency prior buys 22.7% of expert bytes at that KL: the top-k boundary in K3
is genuinely close, so a modest bias reaches a large number of decisions.

The two results are corroborating, and they converge on the same conclusion your
`LEARNED.md` §54 already states — what routing buys is the per-token exclusion.
A residency prior edits precisely that, which is why PR #46 ships off by default
as an instrument rather than as a scheduling optimization.

## The distinction we think is worth naming

Sprint11's margins argue for a boundary that neither fork has drawn explicitly:

- **Selection-affecting** work (a residency prior, top-k reduction) changes the
  per-token exclusion. It requires a KL or quality curve. It cannot be a default.
- **Post-selection** work (fetch order, prefetch priority, eviction preference
  among already-routed experts) never touches the boundary. Sprint11's margins
  are simply irrelevant to it, which is a useful licence: that space is free to
  optimize without a quality argument.

PR #46 is honestly in the first category and says so. We think the second
category is under-explored on both forks precisely because the first is more
interesting to measure.

## Two requests

1. **Raw margin rows, if you can publish them.** Threshold *incidence* needs the
   joint distribution of margin and residency, which the summary quantiles cannot
   provide. With raw rows plus a residency trace, the flip rate becomes a
   measurement instead of a bound.

2. **Is sprint11 aimed at decode?** `by_phase` reports decode 5,888 rows against
   prefill 460 (~7%), and prefill margins run tighter at the low end
   (p1 = 2.107e-06 vs decode 1.082e-05). Several prefill-side conclusions would
   read differently if that tail is undersampled rather than genuinely tighter.

## What we can contribute

To be precise about whose constraint is whose: the "no NVIDIA hardware, no way to
regress-check a numerical contract" line in `LEARNED.md` is **upstream's**
(Marco Bambini, `bc08fe7`), inherited here by merge. It is not a statement about
this fork, which plainly has GB10/GN100 hardware and has published extensive
measurements on it. We misread that on our first pass and corrected it.

The real gap it describes is a *maintenance* boundary: upstream cannot
regression-test a backend it cannot run, which is exactly the out-of-tree
position agreed on issue #11.

What we offer is therefore complementary platform coverage, not hardware you
lack:

- **Windows and x86-64 CPU-only** runs of the portable and server suites, with
  full provenance (OS, compiler, CPU, source SHA, exact commands,
  pass/fail/skip, raw logs, synthetic vs real-model labelled);
- **a prefill-balanced margin capture** on that path, with raw rows and route
  hashes rather than summary quantiles.

That coverage is useful precisely because it is the platform neither you nor
upstream is measuring, and because a portability regression is the kind of thing
an ARM/CUDA vehicle cannot see.

We are not asking you to merge anything. Offered against whichever branch is
most useful — `pr/server-prefix-cache` and `pr/in-memory-state-snapshots` look
like the ones where an independent platform would say the most.
