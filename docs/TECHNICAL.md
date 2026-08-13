# Technical measurements and experiments

This document keeps the detailed numbers that are useful when changing the
engine but too specific for the main README. Unless stated otherwise, Kimi K3
measurements use a 64 GB MacBook Pro with an M5 Pro and the 982 GB container on
the internal SSD. They are snapshots, not portable promises: hardware state,
memory pressure, container revision, and commit all matter.

The experiment log in [LEARNED.md](LEARNED.md) remains the source of truth for
chronology and negative results. [EFFICIENCY.md](EFFICIENCY.md) explains the
performance model, while [GATES.md](GATES.md) records the criteria used to
decide whether an idea is worth implementing.

## Performance snapshot

### Kimi K3

| Quantity | Measurement |
|---|---:|
| Model parameters | 2.78T |
| Container | 982 GB |
| Minimum RAM, 4K context | 29.06 GB |
| Minimum RAM, 32K context | 30.54 GB |
| Minimum RAM, 128K context | 35.64 GB |
| Minimum RAM, 1M context | 83.22 GB |
| Resident trunk | 27.28 GB |
| Expert data read per cold token | 17.0 GB |
| Expert data read at a 38% hit rate | 10.5 GB/token |
| Model load | about 20 s |
| Prefill | 0.47 tok/s chunked; 0.29 tok/s sequential before read-ahead |
| Decode | 0.45–0.62 tok/s at the default budget |
| Vision tower | 15.7 s for 1024 patches |
| 896×896 image | 256 prompt positions, about 2.8 s each |

The image tower is not the expensive part. Its output positions still pass
through the language model, so an image is priced approximately like the same
number of text positions. At the default patch budget, the 256 positions from
an 896×896 image account for roughly 731 seconds of prefill after the tower has
finished. See [K3.md](K3.md) for the architecture and validation.

### Kimi-Linear

| Quantity | Measurement |
|---|---:|
| Container | 19 GB |
| Minimum RAM | 1.28 GB |
| Decode | 10.65 tok/s at an 8 GB budget |
| Expert-cache hit rate | 89% |

Measured on a default `tools/convert.py` container, which gives a 4-bit
trunk. Figures of 1.87 GB and 78% appear in older material; those are a
`--trunk8` container, which raises the floor by 46% and is not what the
converter produces unless asked.

## Where K3 decode time goes

This cold-cache profile used a 17.32 GB expert cache and measured a 6.7% hit
rate over ten steps. Lookahead was disabled to expose the underlying demand
path.

| Component | Share |
|---|---:|
| MoE, total | **82.5%** |
| Expert I/O | 53.5% |
| Expert matrix multiplication | 20.0% |
| KDA layers | 14.5% |
| MLA layers | 2.8% |
| `lm_head` | 0.2% |

Reproduce that profile with:

```bash
WASTE_PROFILE=1 WASTE_LOOKAHEAD=0 WASTE_CACHE_MB=17735 \
./test_forward MODEL 1008,10484,318,15383,387 out.bin 5
```

The I/O share falls as the cache warms or lookahead turns demand reads into
hits. The ranking does not change. Cold expert reads run at about 9.9 GB/s
against 12.78 GB/s measured from the internal SSD.

## Read-ahead and chunked prefill

All sixteen expert IDs for a layer are known before its first expert is used.
Issuing reads concurrently and consuming experts as they arrive overlaps I/O
with arithmetic. On K3 this changed 16-token runs from 48.41–58.61 seconds to
30.03–32.85 seconds: approximately **1.6× faster**, with identical logits and
cache counts. Kimi-Linear improved from 8.96 to 10.81 tok/s.

Chunked prefill deduplicates experts across prompt tokens. It measures
0.47 tok/s against 0.29 tok/s for sequential prefill, a 1.62× gain. Grouping
removes 70–76% of expert I/O but none of the per-token expert computation, so
the observed result closely matches the 1.63× ceiling predicted by the profile.

See [EFFICIENCY.md](EFFICIENCY.md) for the full decomposition and
[ENGINE.md](ENGINE.md) for the implementation.

## Memory budget and the paging cliff

The current one-process sweep, with router lookahead enabled, is:

| Budget | Expert cache | Slots | Hit rate | Decode |
|---:|---:|---:|---:|---:|
| 32 GB | 3.32 GB | 287 | 29.1% | 0.56–0.58 tok/s |
| 46 GB | 17.32 GB | 1498 | 36.2% | **0.63 tok/s** |
| 52 GB | 23.32 GB | 2018 | 38.4% | 0.07–0.09 tok/s |
| 58 GB | 29.32 GB | 2537 | 41.3% | 0.07–0.08 tok/s |

The last two rows are the important failure mode: hit rate rises and bytes
read fall while throughput collapses. The engine remains inside its own
budget, but the machine starts paging its memory. A cache hit then becomes a
page fault. This is why the automatic budget deliberately leaves RAM unused.

The old rule that a useful cache must hold one complete token working set
(17.0 GB) remains true for a demand-only cache. Lookahead changes the relevant
lifetime: a prefetched expert only has to survive from one layer to the next,
so even the 3.32 GB cache records a 29.1% demand hit rate.

## Router lookahead

At the end of layer L, WARP runs layer L+1's resident router on layer L's
hidden state and starts fetching the six highest-ranked experts. The real
router still selects the experts after the next hidden state exists; the
prediction changes only when bytes move, so logits remain bit-identical.

Prediction precision on the measured K3 routing trace is:

| Prediction width | Precision |
|---:|---:|
| Rank 1 | 92.2% |
| Top 6 cumulative | 81.4% |
| Top 16 cumulative | 59.0% |

Six is the current operating point because those reads fit in the idle window
and remain correct about four times in five. Ten speculative experts increased
traffic, so the width is a property of this model and storage device rather
than a universal constant. `WASTE_LOOKAHEAD=0` disables the mechanism.

Initial paired runs moved the demand hit rate from 14–19% to 38–40% and had a
median throughput ratio of 1.17×. A later one-load controlled sweep measured:

| Lookahead | Median decode | Hit rate | Data read |
|---:|---:|---:|---:|
| Off | 0.506 tok/s | 7.2–7.7% | 204–205 GB |
| Top 6 | 0.541 tok/s | 38.0–38.2% | 191 GB |

The controlled result corrected an earlier conclusion: at the 17 GB cache,
lookahead reads 6.6% fewer bytes, not the same number of bytes. Cache size
changes the economics. With only 287 slots it reads about 8% more (165 GB
instead of 153 GB) because speculative records are sometimes evicted and read
again before use.

The same idea was implemented and removed from chunked prefill. A 64-token
chunk touches roughly 550 distinct experts per layer, evicts speculative
records before using them, and increased reads by 6.9% without improving wall
time. The detailed record is in [LEARNED.md](LEARNED.md), sections 34–39.

## The failed 3-bit trunk

K3 was trained with quantization-aware training for its expert weights, not
for the shared trunk. Reducing the trunk from four bits to three freed memory
for the expert cache and raised its hit rate exactly as predicted, but lost on
both speed and output quality:

| Trunk | Resident | Cache at 46 GB | Hit rate | Decode | Output |
|---|---:|---:|---:|---:|---|
| Q4G | 27.50 GB | 17.10 GB | 12% | 0.23 tok/s | coherent |
| Q3G | 21.13 GB | 23.48 GB | 29% | 0.16 tok/s | collapsed |

The scalar 3-bit unpack made trunk matrix operations slower, but vectorizing
it would not rescue the experiment: Q3G logits were 36% away from Q4G and the
generated output collapsed into punctuation and spaces. The quality wall is
in front of the speed wall, so four bits remains the default. The full history
and the superseded earlier interpretation are preserved in
[LEARNED.md](LEARNED.md), section 13.

## The failed non-uniform expert allocation

The format originally reserved room for assigning more bits to important
experts and fewer bits to unimportant ones. The experiment found almost no
variation in the value of the third residual stage:

| Possible source of variation | Maximum/minimum error delta | Greedy vs random allocation |
|---|---:|---:|
| Experts within a layer | 1.06–1.15× | 0.2–1.4% |
| Layers 1, 5, 23, 46, 69, and 92 | **1.01×** | — |
| Gate, up, and down matrices | 1.09–1.30× | 0.3–0.6% |

After per-output-channel scaling, experts differ in what they compute but not
in how difficult they are to quantize. An optimal allocator and a random one
therefore produce nearly the same quality at a given average bit width.

Routing frequency is not flat, but it produces the wrong trade-off. Demoting
cold experts saves disk space and almost no I/O; demoting hot experts saves I/O
but causes a large quality loss. For example, demoting the hottest 25% saved
25.6% of routed I/O while raising measured error from 19.57% to 30.66%.

Variable-width records, cache classes, and an allocator were therefore not
built. The revive criterion is evidence of a model where quantization error
varies materially by expert or layer. See [FORMAT.md](FORMAT.md),
[GATES.md](GATES.md), and [LEARNED.md](LEARNED.md), section 20.

## Measurement discipline

- Measurements identify the model, container, budget, hardware, and relevant
  environment switches.
- Deterministic counters such as hits and bytes are separated from noisy wall
  time.
- Unstable results are ranges rather than averages that hide paging behavior.
- Controlled sweeps clear state and cache between arms; budget comparisons
  still require separate processes because the cache is sized at open.
- Superseded results remain in [LEARNED.md](LEARNED.md) with the correction
  beside them.

For validation rather than performance methodology, see
[GATES.md](GATES.md). For backend-specific measurements and open accelerator
work, see [BACKENDS.md](BACKENDS.md) and [RESEARCH.md](RESEARCH.md).
