# VQ4P CUDA crossover on NVIDIA GB10

Date: 2026-08-07  
Status: completed bounded experiment on `exp/cuda-vq4p-gb10`  
Vehicle: Kimi-Linear-48B, 26 layers, 17.9 GiB VQ4P WASTE container

This experiment closes the VQ4P refusal that accompanied the first GB10 CUDA
path. It does **not** replace the qualified K3 result: Kimi-Linear is a much
smaller model, and the 9.138 tok/s number below must not be compared directly
with K3's 0.637 tok/s held-out profile.

## Result

The selected CUDA path builds and quantizes the VQ4P LUT on the GPU, then
applies the packed expert records there. It is byte-exact against the scalar
CPU reference and 3.98x faster than NEON in the matched 16-token engine run.

| Engine path | Repeats (tok/s) | Median | Relative to NEON |
| --- | --- | ---: | ---: |
| Scalar CPU | 1.478499, 1.470860, 1.467248 | **1.470860** | 0.64x |
| NEON CPU | 2.304586, 2.264450, 2.293125 | **2.293125** | 1.00x |
| CUDA, CPU-built LUT | 3.733551, 3.771248, 3.809792 | **3.771248** | 1.64x |
| CUDA, GPU-built LUT | 9.404292, 9.130235, 9.137839 | **9.137839** | **3.98x** |

All four paths used the same prompt, 16 generated tokens, 3,221 expert-cache
slots, two readers and depth two. They produced token hash
`0xb553a16b6da4d4cc`, logit hash `0xd64812f1651c4037`, and route hash
`0x2693b3c6354e9838`, with identical hits, misses and expert bytes.

The apply profile explains why the second CUDA arm matters. Times below are
milliseconds per routed expert; the selected CUDA pair/down rows include the
asynchronous GPU LUT-build work charged at their synchronization points.

| Work per routed expert | Scalar | NEON | CUDA, CPU LUT | CUDA, GPU LUT |
| --- | ---: | ---: | ---: | ---: |
| CPU LUT build/quantize | 1.046278 | 0.797244 | 0.724450 | 0 |
| Gate/up pair | 1.196925 | 0.216336 | 0.065248 | 0.153224 |
| Down | 0.738855 | 0.605821 | 0.051401 | 0.049600 |
| Accounted total | **2.982057** | **1.619400** | **0.841100** | **0.202824** |

In the CPU-LUT CUDA arm, LUT construction was 619.76% of apply time, far over
the preregistered 10% trigger. GPU-side construction was therefore built and
selected rather than added speculatively.

## Exactness contract

VQ4P is easier to parallelize exactly than VQ3R. The 4-stage int8 sum inside
each 32-vector block is bounded integer arithmetic, so its additions may be
reordered without changing a bit. The CUDA kernel preserves the remaining
ordered operations: eight-FMA fp32 LUT dots, block folds in increasing order,
and the final channel scale.

The accepted path passed:

- all 16,777,216 possible 4x6-bit packed-index patterns;
- real-record gate, up and down preflight comparisons;
- byte comparisons of the GPU-built fp32 LUT, int8 LUT and block scales;
- 17 causal states with byte-identical logits (`max_abs = 0`), routes and
  tokens; and
- exact launch/apply counters with zero CPU fallback.

Mode 2 is deliberately ungrouped. `WASTE_CUDA_VQ_GROUP` values above one fail
closed instead of silently selecting a different contract. The generalized
host interface recognizes only complete VQ3R and VQ4P manifest geometries;
unknown or partial tuples also fail closed.

## What this establishes

The GB10 CUDA source now implements both supported pluggable VQ schemes:
ordered-fp32 VQ3R and integer-accumulating VQ4P. CPU-built tables are a valid,
exact coherent-memory reference, but they are not the fast configuration on
this workload. The experiment remains quarantined until there is maintainer
interest in a CUDA architecture and a larger qualification is warranted.

The preregistration is
[gn100/vq4p-cuda-preregistration.md](gn100/vq4p-cuda-preregistration.md), and
the compact result is
[gn100/vq4p-cuda-gb10-summary.json](gn100/vq4p-cuda-gb10-summary.json).
