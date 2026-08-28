# GLM-5.3-Flash intake and bounded support plan

The released `zai-org/GLM-5.3-Flash` is not a GLM-5.2 checkpoint update.
It is a new hybrid `glm5_next` architecture. This document records the
smallest useful WARP target and the gates that precede any performance claim.

## Pinned source contract

The first qualification is pinned to commit
`04c4e9e95c5da8862dced7e5056455116f83a7e0` (released 2026-08-26,
MIT license):

- 321.323 billion stored parameters; 320B nominal and 18B active;
- 62 safetensor shards, 328,326,771,576 tensor bytes (305.78 GiB);
- E4M3 block FP8 with dynamic activations and 128x128 inverse-scale grids;
- 45 base text layers plus one source-only MTP layer;
- hidden width 4096, 288 routed experts, top 8, one shared expert;
- 34 Kimi Delta Attention layers and 11 NoPE sparse-MLA layers;
- four mHC residual streams, with learned mixing around attention and FFN;
- a separate 24-layer vision tower for image and video input.

`tools/check_glm53_release.py --headers` first validates the immutable repo,
the complete config contract, all index-to-shard assignments, and every
safetensor name, dtype, shape-derived byte count, and data range using HTTP
range requests. On 2026-08-27 it bound 76,108 tensors across all 62 shards;
the summed header payload exactly matched the index.

## First useful WARP milestone

The initial target is deliberately bounded:

- text generation only;
- exact dense NoPE MLA through 2,048 context tokens, where DSA top-2048
  selects the complete causal history;
- exact four-stream mHC and final mean collapse;
- exact GLM-5.3 KDA normalization epsilon and zero-based layer schedule;
- GLM-5.3's clamped SwiGLU in dense, shared, and routed FFNs;
- MTP, DSA indexer tensors, and the vision tower omitted explicitly;
- token-serial prefill until an mHC-aware chunk path has its own oracle;
- a hard refusal above the proven context bound.

This is not native million-token, image, video, or speculative-decoding
support. Those features have separate execution contracts and should not be
smuggled into the first model bring-up.

## Why the target is credible

WARP already has the expensive reusable pieces: 128x128 block-FP8 source
dequantization, VQ3R expert conversion, sigmoid/no-aux routing, streamed MoE,
Kimi Delta Attention recurrence, NoPE latent attention, and coherent-memory
CUDA dense/VQ primitives. The new engine-core work is concentrated in mHC,
the SwiGLU bound, and exact architecture dispatch.

The model has about half as many layers and substantially fewer active
parameters than full GLM-4.7. That makes a practical speed result plausible,
but it is only a hypothesis until the source-backed correctness gates pass.

## Qualification sequence

1. Run the metadata and complete remote-header gate at the pinned revision.
2. Download the 62 shards resumably without stopping normal inference
   services.
3. Prove the converter and runtime on synthetic mHC, KDA, MLA, MoE, and
   clamp fixtures, including reset/replay.
4. Convert into a new container without overwriting any K2, K3, or GLM
   asset. Preserve source shards through source-backed sampled verification.
5. Run minimal token/logit, routing, state-reset, continuation, and semantic
   checks on CPU before enabling a CUDA profile.
6. Compare CPU and exact allowlisted CUDA profiles from the same prompt and
   loaded container. Record TTFT, decode tok/s, route equality, and backend
   counters.
7. Restore normal services regardless of outcome. Publish results only after
   the measured boundary and unsupported features are stated alongside them.

The chat template includes reasoning controls, XML tools, and image/video
markers. Raw token inference comes first; a distilled text/tool harness
profile is a separate frontend qualification after model logits are sound.
