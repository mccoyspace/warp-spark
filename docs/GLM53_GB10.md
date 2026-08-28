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
clamped SwiGLU are implemented. DSA indexer tensors and the separate
image/video tower are omitted explicitly. The base container omits the
appended MTP layer; a separate opt-in research container includes it under
the exact recurrent contract described below. The runtime refuses contexts
above the proven bound.

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

## MTP and task-major VQ follow-up

The released checkpoint has one recurrent MTP layer. An opt-in conversion
now preserves that layer, its distinct expert bank, normalization, embedding,
and output head. The runtime can recursively propose one to three tokens using
the released recurrence, restore exact target state after rejected proposals,
and verify depth-one proposals with either a serial oracle or an exact
layer-major scheduler. Ordinary conversion and generation remain unchanged
when MTP is not requested.

A 64-target-token shadow run first measured whether a wider verifier was worth
building. Every depth replayed to the same ordinary token hash:

| Proposal depth | Position-1 match | Later conditional matches | Observed output / complete cycle | Verifier time allowed for a 5% win |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 77.8% | — | 1.778 | 0.452 s |
| 2 | 75.9% | 59.1% | 2.250 | 0.564 s |
| 3 | 77.8% | 52.4%, then 45.5% | 2.423 | 0.585 s |

The depth-three chain accepted all three proposals in only 18.5% of complete
cycles. A four-row target verifier would therefore need roughly a 47% target-
work discount to clear the registered 5% end-to-end promotion bar. The
measured kernel opportunity was much smaller, so depth-three target fusion was
stopped before it became a larger engineering project.

Profiling did expose a useful general optimization: ordinary decode launched
one VQ kernel per selected expert. A new opt-in task-major CUDA path instead
batches four, eight, or sixteen independent expert tasks while preserving the
router-ordered fp32 accumulation contract. A clean 16-token bracket selected
group 4:

| Arm | Decode rate | Change vs control mean |
| --- | ---: | ---: |
| Control, before | 3.5841 tok/s | — |
| Control, after | 3.5994 tok/s | — |
| Task-major group 4 | **3.8512 tok/s** | **+7.22%** |
| Task-major group 8 | 3.7528 tok/s | +4.48% |

All four arms produced identical token, logit, and route hashes, and all
registered dispatch, launch, synchronization, and fallback counters passed.
Group 8 reduced launches further, but its profiled expert-acquisition bucket
grew; that is consistent with its larger barrier giving some of the gain back.
The selected model-specific setting is
`WASTE_CUDA_VQ_FUSED=4`; it is load-static, requires CUDA VQ mode 2 with the
legacy group-1 path, and remains off for other models until qualified there.

Enabling the paired CUDA VQ2 verifier under fused group 8 improved the MTP arm
from 3.3593 to 3.6106 tok/s (+7.48%) on a favorable short trace with 87.5%
proposal acceptance. That was still 6.25% slower than ordinary group-4 decode.
Depth-one target verification is consequently retained as exact experimental
support and a measurement tool, but it is off by default. The practical
promotion from this work is task-major VQ batching, not speculative decoding.

## Interpretation and limits

- This establishes a source-backed, runnable 320B-total/18B-active hybrid
  model on one 128 GiB coherent-memory system, bounded to 2,048 tokens.
- The original 3.51 tok/s result and the later 3.85 tok/s fused result are
  short, warm-cache decode rows. They are not
  sustained studio-workload, long-run stability, semantic-quality, or
  concurrency result.
- The installed chat profile is fixed-Max, one-shot, plain text. Generated
  reasoning and answer text currently share one content stream and should not
  be replayed as history.
- Sparse DSA beyond 2,048 tokens, image/video input, tools, dynamic reasoning
  control, response-channel splitting, and stateful chat formatting remain
  unsupported or fail closed. MTP is supported only by the separate opt-in
  research container and is not a promoted speed profile.
