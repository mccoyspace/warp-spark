# GLM-5.2 on NVIDIA GB10

This note records a sanitized experimental qualification of
`zai-org/GLM-5.2-FP8` on a 128 GiB NVIDIA GB10 system. It is not an
upstream-supported model or a general performance claim. Raw prompts,
captures, machine paths, and private usage data are intentionally excluded.

The source checkpoint was pinned to Hugging Face revision
`ba978f7d347eaf65d22f1a86833408afdb953541`. Its 141 safetensor files held
118,629 indexed tensors and occupied 755,632,050,320 bytes including file
headers. Every indexed file, tensor membership, payload interval, and exact
file length passed before conversion.

## Bounded architecture contract

The first WARP implementation targets the exact released
`GlmMoeDsaForCausalLM` / `glm_moe_dsa` geometry: 78 base layers, hidden width
6,144, 256 routed experts with sigmoid top-8 selection, one shared expert,
three initial dense layers, and MLA dimensions 2,048/512/192+64/256.

The released DSA indexer selects up to 2,048 prior positions. At causal
context lengths no greater than 2,048, that set contains the entire available
history, so ordinary dense MLA is exact and the indexer cannot change logits.
The container therefore records a hard 2,048-token context limit and refuses
larger allocations. DSA indexer tensors and the appended MTP layer are
omitted explicitly; neither sparse DSA nor MTP is claimed.

## Conversion and source verification

Conversion used two worker processes and CUDA for converter work. It finished
in 1 hour 49 minutes 55 seconds and produced a roughly 264 GiB directory:
75 VQ3R expert banks plus a 9,962 MB, 1,089-tensor trunk.

The source-backed verifier sampled two experts from every MoE layer and all
three expert matrices, for 450 matrix comparisons. Relative VQ error ranged
from 19.40% to 20.57%, within the established 30% acceptance ceiling, and the
container round-trip gate passed. The source safetensors were deleted only
after this gate and the runtime qualification below passed.

## Runtime qualification

The short matched profile used:

```text
context limit: 2048
expert cache: 40000 MiB
compute: 9 threads on CPUs 5-9,15-18
readers / queue depth: 2 / 2
router lookahead: 0
Q8 trunk: enabled
CUDA tuple: KDA selector 1, dense scope 3, VQ mode 2/group 1
workload: 7 prompt tokens + 6 measured greedy decode tokens
```

The KDA selector executes no KDA calls on this all-MLA model; it selects the
already-qualified coherent-memory Q4 kernel for dense projections. CUDA is
admitted only by the exact release geometry and profile above. Alternate
release shapes and selector tuples fail closed.

| Matched arm | Repeat 1 | Repeat 2 | Rate from mean time |
| --- | ---: | ---: | ---: |
| CPU dense + CPU VQ | 0.5483 tok/s | 0.5672 tok/s | 0.5576 tok/s |
| CUDA dense + CPU VQ | 0.9238 tok/s | 0.9120 tok/s | 0.9178 tok/s |
| CUDA dense + CPU VQ, second bracket | 0.8996 tok/s | 0.9362 tok/s | 0.9175 tok/s |
| CUDA dense + CUDA VQ3R | 1.6882 tok/s | 1.7246 tok/s | 1.7062 tok/s |

CUDA dense was 1.646x the first CPU control by mean time. CUDA VQ3R was
1.860x its CUDA-dense/CPU-VQ control. Comparing the first CPU control with
the complete CUDA profile gives a 3.060x short-row throughput ratio. The best
individual measured row was 1.7246 tok/s.

Every arm selected the same generated tokens and ordered expert routes. Dense
CUDA preserved every top-10 set with maximum absolute logit error
`1.5259e-5`; the VQ comparison was byte-exact. Kernel call counts matched the
registered targets and every arm reported zero CUDA fallbacks. A separate
same-process reset/replay check was exact; its CPU/CUDA top-10 sets and argmax
matched with maximum absolute logit error `3.2067e-5`.

## Interpretation and limits

- This establishes a source-backed, runnable roughly 750B-class model on one 128 GiB
  coherent-memory system, bounded to 2,048 context tokens.
- The 1.71 tok/s figure is generated-token decode throughput from a short,
  warm-cache qualification row. It excludes prompt time and is not a held-out
  studio-workload or sustained-throughput result.
- Sparse DSA beyond 2,048 tokens, MTP, long-run stability, semantic quality,
  tool use, serving profiles, and calibrated cache/lookahead settings remain
  unqualified.
- The converted container is retained locally. The 704 GiB source staging
  checkpoint was removed after all source-backed and runtime gates passed.
