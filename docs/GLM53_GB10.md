# GLM-5.3-Flash on NVIDIA GB10

This note records a sanitized experimental qualification of
`zai-org/GLM-5.3-Flash` on a 128 GiB NVIDIA GB10 system. It is not an
upstream-supported model or a general performance claim. Raw prompts,
captures, machine paths, and private usage data are intentionally excluded.

The source checkpoint was pinned to Hugging Face revision
`04c4e9e95c5da8862dced7e5056455116f83a7e0`, released 2026-08-26. Its 62
safetensor shards held 76,108 tensors and 328,326,771,576 tensor bytes
(305.78 GiB). Every indexed file, tensor assignment, dtype, shape-derived
byte count, and payload interval passed before conversion.

## Bounded architecture contract

GLM-5.3-Flash is a new hybrid `glm5_next` architecture rather than a
GLM-5.2 checkpoint update. The released model has about 320 billion total
and 18 billion active parameters, 45 base text layers, hidden width 4,096,
288 routed experts with top-8 selection, one shared expert, and three initial
dense layers. Its attention schedule mixes 34 Kimi Delta Attention layers
with 11 NoPE sparse-MLA layers, and learned mHC mixing carries four residual
streams around attention and FFN blocks.

The first WARP profile is deliberately bounded to text generation and an
exact 2,048-token context. At that length the DSA top-2,048 selection contains
the complete causal history, so dense NoPE MLA is equivalent to the sparse
path. Four-stream mHC, GLM-5.3's KDA schedule and normalization, and its
clamped SwiGLU are implemented. DSA indexer tensors, the appended MTP layer,
and the separate image/video tower are omitted explicitly. The runtime
refuses contexts above the proven bound.

## Conversion and source verification

Conversion used two CUDA worker processes and VQ3R expert storage. It
finished in 45 minutes 29 seconds and produced a 111.5 GiB directory: 42
expert banks of about 2.54 GiB each plus a 4.96 GiB, 1,169-tensor trunk.

The source-backed verifier sampled two experts from every MoE layer and all
three expert matrices, for 252 matrix comparisons. Relative VQ error ranged
from 19.46% to 19.79%, within the registered 30% ceiling, and the container
round-trip gate passed. The 305.78 GiB source checkpoint remains retained for
follow-on validation.

## Runtime qualification

The short matched profile used:

```text
context limit: 2048
expert cache: 40000 MiB (4,434 slots)
compute: 9 threads on CPUs 5-9,15-18
readers / queue depth: 2 / 2, direct I/O
router lookahead: 6
Q8 trunk: enabled
CUDA tuple: KDA selector 1, dense scope 3, VQ mode 2/group 1
workload: 7 prompt tokens + 8 measured greedy decode tokens
```

| Bracket arm | Elapsed for 8 tokens | Decode rate |
| --- | ---: | ---: |
| CPU control, before | 4.9453 s | 1.6177 tok/s |
| Full registered CUDA profile | 2.2818 s | **3.5059 tok/s** |
| CPU control, after | 4.9522 s | 1.6155 tok/s |

The complete CUDA profile was 2.1688x the mean CPU control by elapsed time;
the two CPU endpoints differed by 0.14%. On a separate matched six-token
screen, each CUDA layer paid independently: KDA selector 1 was 1.49x its CPU
base, dense scope 3 added 1.34x versus KDA alone, and VQ mode 2 added 1.67x
versus CUDA KDA+dense with CPU VQ. These are within-stage ratios, not
multiplicative or sustained-production claims.

A small preregistered screen selected the 40,000 MiB cache over 20,000 MiB
by 1.067x and lookahead 6 over lookahead 0 by 1.201x using mean elapsed time.
The selected full profile then passed the CPU/CUDA/CPU bracket above. Every
arm generated the same tokens and ordered expert routes. Top-10 sets and
argmax were unchanged, maximum absolute logit error was `3.1605e-5`, all
registered KDA/dense/VQ dispatch counts matched, and CUDA fallbacks were zero.

Two resident-model, fresh-engine-cache TTFT captures for the seven-token
prompt were 3.847 and 3.538 seconds (3.693-second mean). Model opening was a
separate roughly 1.7 seconds and is not included in those TTFT figures.

## Interpretation and limits

- This establishes a source-backed, runnable 320B-total/18B-active hybrid
  model on one 128 GiB coherent-memory system, bounded to 2,048 tokens.
- The 3.51 tok/s result is a short, warm-cache decode row. It is not a
  sustained studio-workload, long-run stability, semantic-quality, or
  concurrency result.
- The installed chat profile is fixed-Max, one-shot, plain text. Generated
  reasoning and answer text currently share one content stream and should not
  be replayed as history.
- Sparse DSA beyond 2,048 tokens, MTP, image/video input, tools, dynamic
  reasoning control, response-channel splitting, and stateful chat formatting
  remain unsupported or fail closed.
