# GLM-4.7-Flash on NVIDIA GB10

This note records a sanitized first qualification of `zai-org/GLM-4.7-Flash`
on a 128 GiB NVIDIA GB10 system. It is an experimental branch result, not an
upstream-supported model or performance claim. Raw prompts, completions,
captures, and machine paths are intentionally excluded.

The source model was pinned to Hugging Face revision
[`7dd20894`](https://huggingface.co/zai-org/GLM-4.7-Flash/tree/7dd20894a642a0aa287e9827cb1a1f7f91386b67).
Architecture behavior was checked against the
[official Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py).

## Result

The official BF16 checkpoint was converted to an 11.07 GiB VQ3R WARP
container. The text model has 47 MLA layers: one dense layer followed by 46
MoE layers with 64 routed experts and top-4 routing. Its complete 9,970.5 MiB
expert bank fits in the selected 10,240 MiB cache.

The matched full-stack correctness pair generated 16 tokens from the same
14-token prompt and empty engine cache:

| Measure | CPU fallback | Full CUDA |
| --- | ---: | ---: |
| Decode throughput | 6.730 tok/s | **11.890 tok/s** |
| Relative gain | — | **+76.68%** |
| Expert routes changed | 0 | 0 |

The longer CPU–GPU–CPU check generated 64 tokens per fresh process:

| Measure | CPU 1 | Full CUDA | CPU 2 |
| --- | ---: | ---: | ---: |
| Decode throughput | 6.795 tok/s | **11.789 tok/s** | 6.849 tok/s |
| Expert hits / misses | 11,625 / 151 | 11,625 / 151 | 11,625 / 151 |
| Expert bytes read | 4,048,404,480 | 4,048,404,480 | 4,048,404,480 |

CUDA was **72.81% faster** than the 6.822 tok/s CPU mean. The two CPU
brackets differed by 0.79%; generated-token and ordered-route hashes were
identical across all three runs. GPU temperature moved from 45 C to 48 C,
with no swap delta or CUDA error.

The selected experimental profile was:

```text
expert cache: 10240 MiB (3023 slots; all 2944 records fit)
router lookahead: 6
readers / queue depth: 2 / 2
compute: 10 threads on CPUs 5-9,15-19
idle policy: child-scoped PM QoS Q0
I/O and cache: direct I/O, LFRU
arithmetic: Q8 on; SDOT and I8MM off
WASTE_CUDA_KDA=1
WASTE_CUDA_DENSE=3
WASTE_CUDA_VQ=2
WASTE_CUDA_VQ_GROUP=1
WASTE_XPAR=0
```

This all-MLA model executes zero KDA calls; `WASTE_CUDA_KDA=1` is the
existing Q4-kernel selector needed by the dense and VQ CUDA arms. The exact
GLM-4.7-Flash geometry is allowlisted and other shapes fail closed.

Lookahead 6 was about 20.3% faster than lookahead 0 in the small tuning
sweep. VQ group 2 did not beat group 1, and 8, 10, and 20 compute threads
were effectively tied, so the profile retains the simpler established
10-thread GB10 placement.

## Correctness

- Source-backed verification sampled one expert in every MoE layer. All 46
  banks passed shape, CRC, alignment, and codebook-ID checks; relative VQ
  error was approximately 19.4–19.7%, inside the established 30% gate.
- The converted byte-level BPE matched the official tokenizer on all 29
  differential cases.
- Against the official model implementation, one complete 154,880-logit row
  had maximum absolute error `3.2806396484375e-4`, mean absolute error
  `7.218230854381214e-5`, and relative L2 error
  `1.0388010494939642e-6`. Argmax was token 785 in both paths and the ordered
  top 10 was identical.
- Across the 16-token CPU/CUDA capture, generated tokens and ordered routes
  were identical, with zero changed routes in 2,944 selected slots, no top-10
  change, no fallback, and maximum absolute logit difference
  `7.62939453125e-5`.
- CUDA counters matched the precomputed work exactly: 5,264 dense calls;
  2,944 VQ experts; 8,832 applies; 4,416 LUT builds; 9,568 launches; and
  5,888 synchronizations.

The dense fast path is not bit-identical at every logit. Its contract is
unchanged greedy output and routing, bounded numerical error, exact semantic
work counters, and no fallback.

## Functional scope

A simple factual continuation produced the expected answer and stopped on a
secondary model turn marker after 10 tokens at 11.05 decode tok/s. A separate
studio-planning smoke test completed 109 tokens at 12.25 decode tok/s and
about 7.56 completion tok/s end to end.

The converted container retains the vendor Jinja template, but WARP does not
interpret arbitrary Jinja and no qualified `chat.json` exists yet. The tests
therefore used explicitly framed raw prompts. `/v1/chat/completions`, native
tool calling, and a promoted service profile remain future work; this branch
does not change the existing qualified K2 or K3 profiles.
