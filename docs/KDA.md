# Kimi Delta Attention (KDA) — analysis & C kernel plan

**Status: grounded in released reference code.** Moonshot ships
`Kimi-Linear-48B-A3B` publicly, and it is the same architecture family K3
builds on: KDA linear-attention layers interleaved with MLA, fine-grained
MoE, 1M context. Everything below is read off its `config.json` and
`modeling_kimi.py` (fetched 2026-07-27), so the kernel can be written and
validated *before* K3 lands. K3-specific numbers (layer count, head
counts, KDA:MLA ratio) still get confirmed at Gate 1.

> **Gate 1 answered it on 2026-07-27, and the kernel needed no change
> (noted 2026-08-02).** The analysis below is Kimi-Linear's and is still
> exactly what `src/kda.c` implements; K3 differs in two places, both of
> which turned out to be config-detected rather than structural — a
> full-rank output gate (`g_proj` instead of the `g_a`/`g_b` bottleneck in
> §5 below) and a *different* decay-gate formula, `g_min·σ(e^{A_log}z)`
> rather than `-e^{A_log}·softplus(z)`. Shapes, ratios and the four open
> questions at the end are all settled in [K3.md](K3.md); the numbers in
> this file that were predictions are annotated where they appear.

## Why this matters for WARP

KDA state is O(1) in sequence length: one matrix state per head plus a
tiny conv window. At 1M context the attention side costs megabytes, not
gigabytes — RAM stays available for the expert cache, which is where
tokens/sec actually comes from.

## Confirmed architecture (Kimi-Linear-48B-A3B)

| property | value |
|---|---|
| layers | 27 (layer 0 dense FFN, rest MoE) |
| **full-attention (MLA) layers** | 4, 8, 12, 16, 20, 24, 27 — **7 of 27** |
| **KDA layers** | the other 20 → **ratio 3:1**, every 4th layer is MLA |
| KDA heads × head_dim | 32 × 128 |
| KDA short conv | kernel size **4**, SiLU, on q/k/v separately |
| MLA | `q_lora_rank=null` (direct q_proj), kv_lora 512, qk_nope 128, qk_rope 64, v_head 128 |
| **`mla_use_nope: true`** | MLA layers carry **no positional encoding** — KDA supplies position implicitly (asserted in code) |
| MoE | 256 experts, top-8, 1 shared, sigmoid router, grouped top-k, `routed_scaling_factor` 2.446 |
| vocab / max len | 163840 / 1048576 |

The KDA:MLA 3:1 interleave is exactly what `tools/memplan.py` assumed —
that estimate stands.

## The KDA layer, precisely

Per layer, from `KimiDeltaAttention`:

```
q = SiLU(ShortConv_4(W_q · x))        # per-channel causal conv, then SiLU
k = SiLU(ShortConv_4(W_k · x))
v = SiLU(ShortConv_4(W_v · x))
g = fused_kda_gate(W_fb · (W_fa · x), A_log, bias=dt_bias)   # per-channel decay
beta = sigmoid(W_b · x)                                       # per-head, scalar
o, S = KDA_recurrence(q, k, v, g, beta, S)    # L2-norms q,k inside the kernel
o = RMSNormGated(o, W_gb · (W_ga · x))        # sigmoid-gated output norm
y = W_o · o
```

Structural facts that shape the C kernel:

1. **There is a short causal conv (k=4) in front of q/k/v**, with SiLU.
   This was an open question; it is now settled. Decode needs a 3-token
   ring buffer per projection per layer — trivial memory, but it must be
   part of the persisted session state.
2. **The decay gate `g` is low-rank**: `hidden → head_dim(128) → n_heads*head_dim`,
   then combined with a per-head learned `A_log` and `dt_bias`. So it is
   **per-channel** (as assumed), produced through a rank-128 bottleneck —
   cheap, and the two small matmuls fuse into the trunk GEMM pass.
3. **`beta` is per-head scalar**, not per-channel.
4. **q and k are L2-normalized inside the kernel** — a normalization step
   the C kernel must reproduce exactly (it changes numerics).
5. **The output gate is a gated RMSNorm with sigmoid activation**, gate
   also produced via a rank-128 bottleneck (`g_a_proj`/`g_b_proj`). This
   is the "gated" in Gated MLA/KDA — it is on the *output*, elementwise.
6. Reference switches to a fused-recurrent path when `q_len <= 64` and a
   chunked path otherwise — the same split WARP should use
   (decode = recurrent, prefill = chunked).

## Recurrence

```
S'   = Diag(exp(g_t)) · S_{t-1}          # decay first, along the K axis
S_t  = S' + β_t · k_t (v_t − S'ᵀ k_t)ᵀ   # delta uses the DECAYED state
o_t  = S_tᵀ q_t                          # q L2-normalized, then × K^-0.5
```

**Correction over the first draft:** the delta term is computed against the
*decayed* state `S'`, not `S_{t-1}`, and `g` is log-space (the kernel
exponentiates). Confirmed against `fla/ops/kda/naive.py`
(`naive_recurrent_kda`), which is the reference shipped with Kimi-Linear.

Per token per head: two GEMVs + one rank-1 update + a diagonal scale ≈
`3·d_k·d_v` MACs. For 32 heads × 128×128: ~1.6 MFLOP/token/layer — the
engine stays NVMe-bound, as intended.

## State budget (per session, K3-shaped)

- recurrent state: `n_kda_layers × heads × d_k × d_v × 4 B`.
  For 45 KDA layers × 32 heads × 128 × 128 f32 ≈ **377 MB** — flat in
  context length.
- conv windows: `3 projections × (kernel−1) × proj_size` per KDA layer —
  a few MB total.
- MLA latent KV: `(kv_lora + qk_rope) = 576` values/token/layer, only on
  the ~1/4 of layers that are MLA.

> **Measured on the real shape (2026-08-02).** K3 is 69 KDA layers ×
> 96 heads, not 45 × 32, so the recurrent state is **414 MiB**. All three
> lines together are what `waste plan` reports as "KDA state + KV cache":
> **659 MiB at 4K context** — 414 of recurrent state, 216 of MLA latent
> (24 layers × 4096 × 576 × 4 B) and ~30 of conv rings. Only the middle
> term grows with context, and the 1M floor of 83.22 GB in [K3.md](K3.md)
> is almost entirely it.
>
> The 377 MB above is not what the shape beside it gives: 45 × 32 × 128 ×
> 128 × 4 B is **90 MiB**, so that line was off by 4x in its own terms and
> then landed near the right answer for a model with 4.6x more state than
> it assumed. Two errors, opposite directions, and no way to tell from the
> number alone — which is the argument for `waste plan` reading the figure
> off the container rather than a document carrying one.
>
> The structural claim held, and it is the one that mattered: the
> recurrence is O(1) in sequence length, which is what leaves the RAM to
> the expert cache.

## C kernel plan

### Decode (batch 1): fused recurrent step

```c
/* per head: S[dk][dv] f32, row-major by dk */
/* 0. q,k <- l2norm(q), l2norm(k)                                  */
/* 1. u   = Sᵀk            (GEMV over rows, accumulate into dv)    */
/* 2. d   = beta * (v - u)                                          */
/* 3. S   = Diag(g)·S + k·dᵀ    (row-scale + rank-1, fused)         */
/* 4. o   = Sᵀq            (fused into pass 3's row loop)           */
```

NEON: 4×f32 lanes over `dv` (`vfmaq_f32`), `g` row-scale one `vmulq` per
lane; AVX-512: 16 lanes, same shape. Row-major-by-`d_k` keeps the rank-1
update writing contiguous rows and both GEMVs reading them sequentially,
so `S` streams through cache once per token.

Short conv: 3 taps × SiLU per projection — a handful of FMAs, fused into
the projection epilogue.

### Prefill: chunked

Chunk 64/128; intra-chunk as small GEMMs parallel over chunks (OpenMP),
inter-chunk carried by decayed cumulative products. Mirrors the
reference's `chunk_kda`. Batches naturally with the expert batch-union
reads used for MTP verification.

### Gating projections

`f_a/f_b`, `b_proj`, `g_a/g_b` are ordinary dense matmuls living in
`trunk.bin` at Q8G/Q4G — reuse the int8-activation IDOT kernels
(NEON SDOT / AVX-VNNI). No new matmul work.

## Validation (Gate 4)

**Step 1 and 2: DONE (2026-07-27).** [src/kda.c](../src/kda.c) implements
the decode step (NEON/AVX2/scalar), short conv and gated RMSNorm;
[tools/kda_ref.py](../tools/kda_ref.py) diffs it against fla's own
`naive_recurrent_kda`. Results, f32:

| dims | output max\|diff\| | state max\|diff\| |
|---|---|---|
| T=24, H=4, K=V=32 | 3.7e-08 | 2.4e-07 |
| T=64, H=32, K=V=128 (Kimi-Linear's real shape) | 4.1e-08 | 1.8e-07 |

Note `fla/ops/__init__.py` imports Triton-backed kernels, which do not
exist on macOS; `kda_ref.py` loads `naive.py` directly by path, so the
official reference runs on Apple Silicon.

**Step 3 (token-exact on real weights) needs a Linux+CUDA box**: the HF
modeling code hard-requires `fla-core`, which requires Triton. Plan: one
rented GPU session dumps per-layer KDA inputs/outputs *and* the batch-1
routing trace (Gate 2) in the same run, then both are checked offline
here. One rental, two gates.

> **No rental was needed (2026-07-27, noted here 2026-08-02).** Step 3
> passed the same day against `tools/kimi_ref.py` — a pure-PyTorch oracle
> reading the WARP container directly, which sidesteps `fla-core` and
> Triton entirely because it never loads the HF modeling code. Kimi-Linear
> end to end at rel 1.58e-06, and K3's 93 layers at ≤1.14e-05 with final
> logits at 3.56e-06 once the weights landed. The routing trace came from
> the same oracle, and later from `WASTE_DUMP_ROUTE` in the engine itself.
> [GATES.md](GATES.md) Gate 4, [K3.md](K3.md).

## Remaining unknowns (Gate 1, from K3's own config)

*(All four answered 2026-07-27 when the weights dropped. Kept as written
with the answers beside them; the full read is in [K3.md](K3.md).)*

- layer count and KDA:MLA ratio at K3 scale (expect 3:1).
  → **93 layers, 69 KDA : 24 MLA**, about 2.9:1. The ratio held.
- head counts / d_state (expect 128; memplan uses that).
  → **96 heads × 128**. d_state right, head count 3x the estimate.
- whether K3 keeps `mla_use_nope` and the null `q_lora_rank`.
  → NoPE kept; `q_lora_rank` is **1536**, not null, so MLA's query is
  factorized on K3 and the oracle had to learn about it.
- MTP head presence (`num_nextn_predict_layers` is 0 in Kimi-Linear;
  K2 shipped one, and speculative decoding is a throughput lever for us).
  → **none.** `num_nextn_predict_layers: 0` on K3 too, so there is no
  cheap draft model. Speculative decoding was separately refused on
  arithmetic anyway — [EFFICIENCY.md](EFFICIENCY.md) §4D.
