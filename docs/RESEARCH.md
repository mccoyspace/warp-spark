# Deep research: trillion-scale MoE on ≤64 GB RAM (July 23, 2026)

Condensed from a 24-source arXiv-focused deep research pass with adversarial
claim verification (3-vote refutation panels). Full model specs from the
[Kimi K3 announcement](https://www.kimi.com/blog/kimi-k3): 2.8T total params,
896 experts / 16 active per token, MXFP4 weights + QAT from SFT, Kimi Delta
Attention + Gated MLA, 1M context, open weights July 27, 2026.

> **This is the literature review that started the project, kept as
> written.** Four days later the weights dropped and most of it was tested
> against real measurements. Two entries below did not survive that
> contact — the KBVQ shared basis and the throughput estimate — and each
> is annotated where it appears. What the engine actually does is in
> [LEARNED.md](LEARNED.md); what the format actually is, in
> [FORMAT.md](FORMAT.md).

## Verified findings (survived 3-0 adversarial votes)

### Quantization

- **MoE quantization must be non-uniform** — sensitivity varies per block,
  expert activation frequencies are heterogeneous, rare experts get biased
  calibration. MxMoE (ICML 2025) beats GPTQ by 2.4 PPL at 2.25 bits avg.
  [arXiv:2505.05799](https://arxiv.org/abs/2505.05799),
  [arXiv:2505.03804](https://arxiv.org/abs/2505.03804)
  **The premise does not hold on this family.** Activation frequencies are
  indeed heterogeneous; *sensitivity* is not. Measured per expert, per
  layer and per matrix on K3 and Kimi-Linear, the value of the third bit
  varies by 1.01–1.15x — see [LEARNED.md](LEARNED.md) §20.
- **KBVQ-MoE (ICLR 2026)** — strongest sub-4-bit result: shared low-rank
  components across experts kept FP16 (~0.1 bit/param overhead), per-expert
  residuals vector-quantized. Near-lossless at 3 bit; usable at 2 bit where
  GPTQ collapses (PPL 11.87 vs 438 on Qwen3-30B-A3B).
  [arXiv:2602.11184](https://arxiv.org/pdf/2602.11184)
  **Half of it survived.** The residual VQ is the WARP expert format and
  earns its place — VQ beats round-to-nearest decisively below 4 bits. The
  *shared low-rank* half was the original centrepiece and does not pay for
  itself on Kimi's experts: 0.12 bits for 0.3 pp, and a loss at equal
  budget. It is specified and not implemented, with a revive-or-delete
  criterion in [FORMAT.md](FORMAT.md); the measurements are in
  [LEARNED.md](LEARNED.md) §3.
- **GEMQ (May 2026)** — global per-expert bit allocation (LP over expert
  importance, mix 1/2/3-bit) + cheap router fine-tune: Mixtral 87→16 GB with
  −7% MMLU. Quality cliff below 2 bits ⇒ **practical floor ~2–2.5
  bits/expert**, with attention/dense at 4+ bits (consistent with Unsloth's
  1.58-bit DeepSeek-R1 recipe).
  [arXiv:2605.23078](https://arxiv.org/pdf/2605.23078)
  **The floor held; the allocation did not.** 3 bits is where this engine
  runs and 2 is unusable, which is the cliff. But the LP has nothing to
  optimize over here — the importance is flat to within 1.15x, so the
  allocator was measured and dropped rather than built
  ([LEARNED.md](LEARNED.md) §20). The paper's Mixtral is not QAT'd from
  MXFP4 and its experts are 8, not 896; neither difference explains it,
  since Kimi-Linear is not QAT'd either and measures the same.
- **DynaExq** — expert utilization is heavy-tailed and the hot set shifts
  per workload; runtime hot/cold precision promotion beats static
  quantization at equal memory (Qwen3-80B: 73.09→77.57 avg).
  [arXiv:2511.15015](https://arxiv.org/abs/2511.15015)

### Offloading / streaming

- **KTransformers (SOSP 2025, peer-reviewed)** — best-documented 671B-class
  recipe: DeepSeek-V3 decode 4.68→5.87 tok/s (CUDA graphs) and **Expert
  Deferral** (defer ~6/8 routed experts, overlap with next layer's
  attention) for +33–45% decode, <0.5% accuracy drop. ⚠️ Testbed is **1 TB
  DDR5 @ 220 GB/s** — numbers do NOT transfer to NVMe streaming.
  [SOSP25 paper](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- **HOBBIT** — mixed-precision cache-miss handling (substitute low-bit
  expert versions instead of full fetch), built on llama.cpp, up to 9.93×
  decode speedup. Closest C-engine precedent.
  [arXiv:2411.01433](https://arxiv.org/abs/2411.01433)
- **MoE-Infinity** — trace-driven expert cache/prefetch, 3.1–16.7× per-token
  latency vs generic baselines.
  [arXiv:2401.14361](https://arxiv.org/abs/2401.14361)
- **FlashMoE (Jan 2026)** — learned cache replacement (recency+frequency):
  +21% hit rate over LRU, measured on consumer PCIe5 NVMe hardware.
  [arXiv:2601.17063](https://arxiv.org/abs/2601.17063)

## Refuted claims — treat as design risks

1. **"Batch-1 expert temporal locality is strong"** — REFUTED (1-2).
   Aggregate heavy-tailedness is real; strong per-token decode reuse is not
   established. The LFRU cache may hit less than hoped.
2. **"SSD expert offloading is viable for very large MoE on RAM-constrained
   machines"** — REFUTED (0-3) *as a literature claim*: no paper
   demonstrates trillion-scale NVMe streaming. Counter-evidence: our own
   prior measurements (GLM-5.2 744B streamed from NVMe, 0.3–3.3 tok/s, 71.6%
   next-layer routing predictability) — empirical, ours, at 744B not 2.8T.

## Uncovered areas (no claims survived verification)

MLA/linear-attention CPU kernels, speculative decoding on Apple Silicon,
community llama.cpp/ktransformers NVMe throughput reports for
DeepSeek-R1/Kimi K2. These need first-party measurement, not literature.

## Open questions → measurement plan (post-weights-drop)

1. **Expert reuse rate at batch 1 for K3's 896-expert routing** — THE
   number that decides tokens/sec. Measure from released router weights +
   calibration corpus before building anything heavy.
2. MXFP4 → sub-4-bit requantization error compounding (does QAT-MXFP4 make
   VQ residuals better- or worse-behaved than from BF16?).
3. KDA+Gated MLA compute/memory profile at long context on CPU.
4. Does MTP/speculative decoding help when NVMe-bound? (Draft tokens grow
   the expert working set — but batch-union reads amortize unique experts
   across speculated tokens, so it may still win. Measure.)

## Throughput expectation (honest)

~12.5 GB expert reads per cold token at 2 bit ⇒ on 10–14 GB/s NVMe:
**~1 tok/s cold, 2–3 tok/s with warm learned cache** — extrapolation, not
measurement. Same order as our own GLM-5.2 numbers at equivalent
bytes/token. Sub-1 tok/s on a single ~3 GB/s SSD. Plan B if unacceptable:
workload-driven expert pruning (drop the cold tail of 896 experts).

> **Measured: ~0.3 tok/s.** Three things the estimate got wrong, all in
> the same direction. 2 bits was unsafe, so experts are 3 — 17.0 GB per
> cold token, not 12.5. The trunk is 27.28 GB resident on a 64 GB
> machine, so the warm cache the second half assumed never fits. And the
> I/O is under half a decode step, so even free reads would not give
> 2–3 tok/s. The one part that held is the shape: this is an I/O-bound
> engine whose only real lever is bytes read per token. Plan B — expert
> pruning — is still unexplored. The non-uniform bit allocation that was
> to make it cheap has since been measured and dropped (§20), and the
> same measurement is why pruning would have to be justified on its own
> terms: dropping the cold tail costs disk, which is not scarce, and
> saves almost none of the reads, which are.

> **And the "only real lever" was wrong too (2026-07-31).** Bytes per token
> is what this section names, and every lever on it has since been measured
> and refused — per-activation bit allocation because the router has no
> tail, a larger cache because the machine will not leave it resident. What
> actually paid was not reading fewer bytes but not *waiting* for them:
> overlapping the reads with the arithmetic is ~1.6x on K3, and it took no
> format change at all. [EFFICIENCY.md](EFFICIENCY.md) is the ledger,
> [LEARNED.md](LEARNED.md) §22–25 the account.
>
> *(And again on 2026-08-01: the router lookahead is the same lever one
> layer further out — start the next layer's reads from a prediction made
> by its own router, during the boundary the disk would otherwise spend
> idle. Also no format change, also not about bytes. §34–36.)*
