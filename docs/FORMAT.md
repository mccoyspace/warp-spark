# WASTE Container Format v0

Status: **v0, frozen against the released weights.** This started as a
pre-release design; the TBDs were settled on 2026-07-27 when K3 dropped
and are recorded as answers below. Two things in it are *specified and
not implemented* — the shared low-rank block and the SUB1 substitute
bank — and both say so where they appear. A third, the per-expert bit
allocator, was measured instead of built and then dropped: design goal 5.

`format_version` is enforced: a container from another version, or one
without the field, is refused rather than read against the wrong rules.

## Design goals

1. **One coalesced read per expert.** An expert's gate/up/down matrices are
   adjacent on disk and loaded with a single `pread` — measured as the
   difference between usable and unusable NVMe throughput.
2. **Placement decides speed, never precision.** The invariant: output is
   bit-identical whether an expert came from RAM cache or disk.
   The *substitute* path (below) is the one deliberate, bounded exception,
   and it is off by default.
3. **O_DIRECT-friendly.** Every independently-readable record is aligned to
   4 KiB and sized in 4 KiB multiples, so the page cache can be bypassed
   (measured wins from `F_NOCACHE`/O_DIRECT on some drives).
4. **Sub-4-bit without collapse.** Per-expert weights are vector-quantized:
   multi-stage (residual) VQ over 8-dim vectors with per-channel scales.
   Gate 3 measured this on real Kimi experts — VQ decisively beats RTN
   below 4 bits, and **3 bits (19.4% error) is the operating point**; 2-bit
   VQ (33%) is more than double the known-good production int4 baseline.

   The KBVQ-MoE shared-low-rank component (arXiv 2602.11184) is
   **specified but NOT implemented in v0** — see "Shared low-rank: on
   probation" below.
5. ~~**Non-uniform bits.**~~ **Measured and dropped.** The design goal was
   global per-expert bit allocation à la GEMQ (arXiv 2605.23078) —
   important experts at 3 bits, unimportant at 2. It assumes experts
   differ in how much the third bit buys them, and on this family they do
   not: the spread in that quantity is 1.06–1.15x between experts in a
   layer, **1.01x between layers**, and 1.09–1.30x between gate, up and
   down, on both K3 and Kimi-Linear. An optimal allocator and a coin flip
   therefore write the same container, and the only importance signal
   that is not flat — routing frequency — buys disk footprint and no I/O.
   Every expert in a container stays at one width. The measurement, and
   the criterion that would revive this, are in
   [LEARNED.md](LEARNED.md) §20; the instrument is
   `tools/bitalloc_lab.py`.

   Attention / router / norms / shared experts / MTP head stay at 4–8 bit
   (asymmetric recipe; the MTP head must be int8+). That part stands.

## File layout

A WASTE model is a directory (not a single file — shard-friendly, resumable
conversion, multi-drive splitting):

```
model.waste/
  manifest.json        # config, trunk tensor index, expert bank index
  trunk.bin            # resident dense part
  experts-L{layer}.bin # one expert bank per layer
  codebooks.bin        # VQ codebooks, resident
  tokenizer.model      # the model's tiktoken rank file, copied in
  specials.json        # special-token ids and their exact strings
  vision.json          # the tower's shape, if the container has one
  chat.json            # conversation format — hand-written, see examples/
  chat_template.jinja  # the release's own template, when it ships one
  usage.waste          # runtime-appended routing stats / learned hotlist
  subs-L{layer}.bin    # 1-bit substitute bank — specified, NOT written in v0
  lowrank.bin          # shared low-rank factors — specified, NOT written in v0
```

Everything up to `usage.waste` is produced by `tools/convert.py`, except
`chat.json`: neither Kimi release distributes a chat template, so that one
is written by hand. See [examples/](../examples/).

### manifest.json

JSON (hardened parser, treat as untrusted — `cfg_sane()` bounds every
dimension, tensor sizes are bounded by the file that holds them, and the
fuzzer in `tools/fuzz_container.py` exists for this file). Keys:

| key | what it holds |
|---|---|
| `format_version` | 0. Absent or different ⇒ refused |
| `arch` | descriptive; the engine derives its own from `config` |
| `tensor_prefix` | `language_model.` on K3, `""` on Kimi-Linear |
| `config` | the release's own config verbatim, with the multimodal wrapper under `_outer` |
| `expert_quant` | `stages`, `vec_dim`, `entries`, `index_block`, `bits_per_weight` |
| `layers` | per MoE layer: `file`, `experts`, `bytes`, `codebook_base` |
| `trunk` | per tensor: `name`, `fmt`, `off`, `shape`, `group`, `scale_off`, `bytes` |

Two fields from the original design are **not** here. There is no
`bits[]`: the GEMQ-style per-expert bit allocator was specified, measured
(design goal 5 above, [LEARNED.md](LEARNED.md) §20) and dropped, so every
expert in a container is at the same width — deliberately, now, rather
than for want of the code. And there is
no `blake3` — the only checksum in the format is the per-expert `crc32`
in the record header.

### What the engine checks on the read path

Two different things happen to every record that comes off the disk, and
only one of them is optional.

**Always: the header.** A record must carry the right magic, be the
expert the bank index asked for at that offset, be the bank's stride
long, be a VQ format with `lowrank_id == 0`, name a codebook that
exists, and have ordered offsets that fit. All of it is derivable from
the manifest, so it is derived rather than believed. This is O(1) per
record and it is not a integrity feature — it is what keeps an offset out
of a damaged header from reaching the arithmetic downstream, which is
also why the checksum could not be bolted on without it: the offsets that
say where the payload *ends* live in the header, so deciding how much to
checksum from an unvalidated one would be a read past the buffer rather
than a check. A short read is caught here too.

**On request: the checksum.** `waste_cfg.verify_records`, `--verify` on
the CLI, or `WASTE_VERIFY=1` in the environment. It is **off by
default**, and that is a throughput decision: it is a pass over every
record on every cache miss, about **5% on Kimi-Linear** (4.7% and 5.5%
in two sittings — [LEARNED.md](LEARNED.md) §21) and about **1% on K3**,
where 11.83 MB records make the read dominate — 7,287 misses at 0.376 ms
of CRC each is 2.7 s of a 268 s run. Turn it on for a container that was
copied, downloaded, or has sat on a disk you do not trust; leave it off
for the one you converted locally and have been reading all week. A cache
hit is never re-checked either way: the unit being verified is bytes
entering RAM.

The `crc32` covers the payload from the end of the header to the end of
the per-channel scales, excluding the 4 KiB padding — bit for bit what
`zlib.crc32` returns, since `tools/convert.py` is what writes it, and
`tests/test_container.c` pins the two together against known values. The
implementation is [`src/crc32.c`](../src/crc32.c): 33 GB/s where the
ARMv8 CRC extension is available, 2.7 GB/s from the slice-by-8 table
everywhere else.

Either check failing ends the generation with `WASTE_E_IO`, and
`waste_error_detail` names the record — "expert 412 of layer 37:
checksum mismatch" — rather than answering with whatever the damaged
bytes decode to. `tools/verify_container.py` checks every record's
checksum unconditionally and is the right tool for auditing a container
once, as against paying for it on every token.

Not checked at all: the trunk, which has no checksum in the format, and
the codebooks. Both are read once at load rather than per token, so the
argument for a per-record check does not carry over — it is simply not
built.

### Quantization formats (`fmt`)

| fmt | name | bits/weight | use |
|---|---|---|---|
| 0 | F32 | 32 | norms, router, `e_score_correction_bias` |
| 1 | F16/BF16 | 16 | codebooks; low-rank factors if they ever land |
| 2 | Q8G | 8 (+f16 scale /g128) | the embedding table and the LM head on every container, the vision tower on K3, and anything under `--trunk8` / `--trunk-bits 8` |
| 3 | Q4G | 4 (+f16 scale /g128) | the trunk default, on every model |
| 4 | **VQ3R** | 3.00 (3 stages x 256 entries, dim 8) | default for experts (Gate 3) |
| 5 | **VQ2R** | 2.00 (2 stages x 256 entries, dim 8) | only where Gate 3 quality allows |
| 6 | **SUB1** | ~1.0 direct VQ | cache-miss substitutes — specified, not written |
| 7 | **Q3G** | 3 (+f16 scale /g128) | implemented, default for nothing — see below |

The bits/weight column for VQxR is exact — one byte of index per 8-dim
vector per stage — plus one f16 scale per output row, i.e. 16/n_in
amortized. A reader must take `stages`/`vec_dim`/`entries` from the
manifest rather than from the format id.

**Q3G exists and is not recommended.** The trunk is the RAM floor and the
floor is what the expert cache does not get, so a 3-bit trunk looks like
free cache. It was built and measured, twice. It does get the better hit
rate — 29% against 12% at the same budget — and it is 1.4x slower anyway,
because the scalar 3-bit unpack costs more in the trunk matvecs than the
cache saves in I/O. Worse, generation collapses: K3's QAT covered the
*expert* weights only, so the trunk has no trained tolerance for being
squeezed, and the logits land 36% off. Vectorizing the unpack would not
save it — the quality wall sits in front of the speed wall.
[LEARNED.md](LEARNED.md) §13 has both measurements.

**VQxR record** (per expert matrix): N stages of 8-dim VQ indices into
per-layer codebooks of 256 entries each (N=3 for VQ3R, N=2 for VQ2R), plus
one FP16 scale per output channel:

```
W_expert ≈ scale_per_channel * sum_{s=1..N} codebook_s[index_s]
```

Residual (multi-stage) VQ: each successive codebook quantizes what the
previous stages left over. Bits/weight = N, plus 16/n_in for the channel
scale. Measured on real Kimi experts in Gate 3.

**Index layout is blocked by 64 rows** (`index_block` in the manifest):
`[row_block][vector_position][row_in_block][stage]`. The engine walks a
tile of rows for one vector position at a time; in plain row-major order
those rows sit `n_in/8 * stages` bytes apart, so each is a separate cache
line. Blocked, a tile's indices for one position are contiguous. The block
size matches `VQ_TILE` in the engine; **a reader must honour
`index_block`** (0 = plain row-major), because it changes where the bytes
are, not merely how fast they are read.

*The speed argument for it was refuted.* Blocking measured 1.44x on the
gather loop in isolation and changed nothing in the real engine — the
microbenchmark did not model 12 threads sharing L2 — and finding that out
cost a full reconversion. The layout stays because containers are written
in it and it is not worse; do not repeat the 1.44x as a result.
[LEARNED.md](LEARNED.md) §7.

### Shared low-rank: on probation — specified, NOT implemented in v0

`lowrank.bin` and the `lowrank_id` field exist in the spec but the
converter does not emit them and the engine does not read them;
`lowrank_id` must be 0 in v0.

**Why parked** (Gate 3 plus a follow-up subspace measurement, 2026-07-27,
on real Kimi-Linear-48B experts):

- at rank N/128 the shared basis costs 0.12 bits/weight and reduces error
  by 0.3 pp — noise;
- at equal budget it loses badly: kbvq2 at 4.01 bits = 28.87% error, plain
  per-row INT4 at 4.01 bits = 15.20%;
- structurally, Kimi's experts are nearly mutually orthogonal — pairwise
  overlap of their rank-72 dominant subspaces is **0.046 against a random
  baseline of 0.031** (identical = 1.0); a shared basis captures 7.1% of
  energy vs 3.1% for random directions and 20.0% for each expert's own.

**Why not deleted.** Those measurements are in the *unweighted* weight
metric. "KLT-guided" in the paper most likely means a basis chosen after
whitening by the activation covariance, and there is a credible mechanism
by which that flips the result: every expert sees the same hidden-state
distribution, and LLM hidden states concentrate in a few dominant
directions, so in the activation-weighted metric the useful directions may
be shared *by construction* even though the weights are orthogonal here.

**What settles it:** rerun the Gate 3 comparison with an importance matrix
from real activations, in the same rented GPU session as Gate 2 and the
Gate 4 oracle. Revive if a whitened shared basis buys >1.5 pp of error at
≤0.15 bits/weight; otherwise delete the section and the field. No data has
been written in this format yet, so either way it is a cheap change.

### Expert bank record (`experts-L{n}.bin`)

```
[4 KiB-aligned]
ExpertRec {
  u32 magic 'WEXP', u16 layer, u16 expert_id
  u8  fmt (VQ3R|VQ2R), u8 flags, u16 codebook_id
  u32 gate_off, up_off, down_off, correction_off   // within record
  u32 record_4k_blocks
  -- gate indices | up indices | down indices | per-channel corrections --
}
```

One `pread` of `record_4k_blocks * 4096` bytes yields the whole expert.
On K3 that record is **12 406 784 bytes, exactly 3029 pages** — which is
what makes O_DIRECT possible, and why `bank_open` checks the alignment
rather than assuming it: a record that is not a page multiple makes every
read fail `EINVAL` instead of merely running slow.

Records for the same layer are contiguous and sorted by expert id.
`subs-L{n}.bin` would mirror the layout at SUB1 precision (~5× smaller
reads, used only when the engine's miss-latency budget is exceeded,
HOBBIT-style, arXiv 2411.01433 — it would need a flag, because it breaks
bit-exactness). **Specified, not implemented:** the converter writes no
substitute bank, so there is nothing to substitute. `waste_cfg` carried an
`allow_substitutes` flag for this until 0.6.0; it was removed, because a
switch the engine never read described a capability the engine did not
have.

### trunk.bin

Everything needed for a forward pass with zero expert reads: KDA/MLA
attention weights, routers, shared experts, the latent MoE projections,
norms and the LM head — plus the embedding table and, on a multimodal
container, the vision tower.

The last two are in the file but not in the resident set. `embed_tokens`
is 1.11 GB of which one 7 KB row is read per token, so it stays on disk
and the row is `pread` on use. The tower is loaded only when a caller
asks for images: 434 MB of weights, and 1.12 GB reserved once the bounded
source decode, the tower's activations and the queued image embeddings are
counted, all of it otherwise straight out of the
expert cache. Everything else is touched in full on every token, so
streaming it would cost more I/O than the freed cache could save.

Measured on K3: 27.28 GB resident out of a 29.06 GB floor at 4K context.
The pre-release target was ≤ 25 GB; the real trunk missed it, and that
overshoot is most of why decode sits at 0.5 tok/s rather than 1.5. (0.3
before read-ahead — [EFFICIENCY.md](EFFICIENCY.md).)

### usage.waste

Append-only runtime log: per-(layer, expert) hit counts + decayed recency
for the LFRU policy, written by `--learn` and preloaded on the next open
so a run starts warm instead of empty. Measured on Kimi-Linear at a 5 GB
budget: 1602 misses cold against 1175 warm, 61% → 72%. The
cross-layer routing pairs for a pilot/COUPLE prefetcher have a field in
the entry struct and no code behind them, and the converter cannot yet
bake an initial hotlist from a calibration corpus.

## Converter pipeline (`tools/convert.py`)

1. Stream release shards one at a time — never needs the full 1.42 TB
   locally beyond the shard in flight plus the output.
2. Dequant MXFP4 → f32 blocks (`tools/mxfp4.py`, verified bit-identical
   to `compressed_tensors`' own unpacker). Only the routed experts are
   packed; the whole trunk, latent projections and shared experts
   included, ships as plain bf16. No shared-basis pass — see "Shared
   low-rank: on probation".
3. Fit VQ codebooks by k-means in 8-dim space on a sample of experts
   (`--cb-sample`, default 12), then load, quantize and write each expert
   on its own, so peak memory is a few hundred MB regardless of model
   size. Layers convert in separate processes (`--jobs`), each with its
   own codebook file, merged by concatenation.
4. Write the trunk, the tokenizer, the special tokens, the vision config
   and the manifest. Verify with `tools/verify_container.py`, which
   dequantizes records back and diffs them against the source weights.

**Step 3 of the original design is not missing, it is refused:** there is
no GEMQ-style per-expert bit allocation, because the importance it would
allocate against does not vary — 1.01x between layers, 1.15x between
experts. Every expert in a container is at `--stages` bits.
[LEARNED.md](LEARNED.md) §20 has the numbers and the revive criterion.

## Questions the weights drop answered

- **Per-layer expert count and shape.** 896 flat, top-16, 92 MoE layers
  of 93 — but operating on a 3584-wide *latent*, not the 7168 hidden,
  which halves both the expert size and the per-token I/O.
- **MXFP4 → VQ requantization error compounding.** It does not compound
  badly: requantizing an already-4-bit source down to 3 costs about 2 pp
  over rtn4-row (20.3% against 18.4%), close to the 19.4% measured
  going to 3 bits from bf16 on Kimi-Linear. QAT-from-MXFP4 does appear
  to leave the weights tolerant.
- **Codebook granularity.** Per layer, 256 entries per stage, fitted on
  12 sampled experts. Per-expert-group was never needed.

## Still open

- Whether SUB1 substitutes measurably hurt K3 quality — untestable until
  the substitute bank is written at all.
- Whether the shared low-rank basis survives an activation-weighted
  metric (the revive-or-delete criterion above).
- Whether per-expert importance spreads under that same metric. Both
  questions want the same rented GPU session and the same importance
  matrix, so they are one experiment, not two.
- A read-path integrity check that does not cost a pass per miss.
