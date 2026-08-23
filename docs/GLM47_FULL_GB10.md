# Full GLM-4.7 on NVIDIA GB10

This note records a sanitized experimental qualification of the full
`zai-org/GLM-4.7` model on a 128 GiB NVIDIA GB10 system, from the initial CPU
bring-up through narrowly gated coherent-memory CUDA pilots. It is not an
upstream-supported model or a general performance claim. Raw prompts,
completions, captures, and machine paths are intentionally excluded.

The source checkpoint was pinned to Hugging Face revision
[`602d01ef`](https://huggingface.co/zai-org/GLM-4.7/tree/602d01efcdd332c5238ca4bcede555defbe83eb7).
Its 92 BF16 base shards total 705,601,209,080 bytes. The separate MTP weights
were deliberately omitted because this experimental path does not implement
MTP.

## Conversion

The source-boundary gate validated 44,691 tensors from the pinned checkpoint,
including the official separate routed-expert layout. Conversion completed in
55 minutes 48.91 seconds and produced a roughly 127 GB directory containing
the trunk and 89 MoE expert banks.

The source-backed verifier sampled two experts in every MoE layer and all
three expert matrices (`gate`, `up`, and `down`): 534 matrices in total. The
worst relative VQ error was 20.03%, inside the established 30% acceptance
gate. The original source shards were retained after qualification.

## CPU qualification

The real-checkpoint checks used this deliberately conservative profile:

```text
expert cache: 59340 MiB
compute: 10 threads on CPUs 5-9,15-19
readers / queue depth: 2 / 2
router lookahead: 0
CUDA: disabled
```

An exact same-process reset/replay check ran a three-token prompt plus one
decode step twice. Logits, ordered expert routes, router weights, and hashes
were identical across the two passes, covering 712 route rows in total.

One clean three-prompt-token plus two-decode-token timing row produced:

| Wall time | Effective generated rate | Expert bytes read | Cache hit rate |
| ---: | ---: | ---: | ---: |
| 3.867034 s | 0.517192 tok/s | 4,762,030,080 | 62.29% |

The rate above is two generated tokens divided by the measured decode time;
prompt evaluation is excluded. It is a qualification datapoint, not a
workload benchmark or a claim about sustained throughput.

A separate raw-format arithmetic smoke returned exactly `323` and stopped on
the secondary EOS marker `<|user|>`. It generated three tokens in 5.52 seconds
(0.54 tok/s), with 1,819 expert hits and 317 misses (85% hit rate). This tests
basic semantic execution and stopping only; it is not a quality evaluation.

## Initial qualification scope

- The initial result used WARP's CPU implementation of standard grouped query
  attention. CUDA was not used in that first qualification row.
- Standard-GQA state snapshots are not implemented, so snapshot export/import
  is unavailable for this model.
- The checkpoint's optional MTP layer was omitted and is unsupported.
- The conversion and runtime gates target the exact pinned GLM-4.7 source
  contract and fail closed outside that experimental scope.
- Source weights remain retained. Nothing here is an upstream support or
  performance claim.

## First optimization pass

A subsequent experimental pass kept the exact full-model architecture while
testing the same software controls used on the Kimi profiles. On one fixed
eight-token development row, the original CPU setup produced 0.587 tok/s.
The useful CPU changes were:

- a child-scoped Linux PM-QoS Q0 request;
- reducing the expert cache from 59,340 to 18,809 MiB; and
- using nine compute threads inside the same ten-CPU mask.

Together those changes reached 1.26 tok/s with identical tokens, logits, and
routes. Lookahead, additional reader concurrency, deeper I/O queues, and
expert-parallel execution did not improve the row and remain disabled.

The existing coherent-memory CUDA Q4 and VQ3R kernels were then admitted only
for the exact full-GLM geometry. Standard-GQA attention and prompt prefill
remain on the CPU. CUDA dense FFN work improved the matched row from 1.27 to
1.37 tok/s; adding VQ3R reached 1.70 tok/s. The complete CUDA arm preserved
all routed expert selections and top-10 logits, with a maximum logit delta of
1.43e-5 against CPU in the held-out check.

A separate 16-token route pattern measured 1.105 tok/s for the bracketed CPU
control and 1.311 tok/s for CUDA, an 18.6% improvement. It read substantially
more expert data and is the more conservative generalization result. The raw
arithmetic smoke continued to return `323` and stop correctly, at 1.47 tok/s.

Reducing routing from top-8 to top-6 or top-4 also produced 1.38-1.40 tok/s on
CPU, but changed the generated path. Those settings are retained only as
unqualified approximate-profile candidates; neither is the default.

After CUDA FFN/VQ promotion, standard GQA accounted for roughly 64% of the
remaining decode time. This motivated one further bounded pilot: move only
its Q/K/V/O projections through the already qualified CUDA Q4 kernel.

## CUDA GQA projection qualification

The new arm, qualified at code commit `83add64`, is default-off and restricted
to the exact full-GLM geometry and the qualified KDA-1, dense-3,
VQ-2/group-1 base. Bias, per-head Q/K RMSNorm, half-split partial RoPE, fp32
K/V state, score/value attention, routing, expert accumulation, and prompt
prefill remain on the CPU. All 368 projection tensors are checked before
decode, both matrix orientations receive a real preflight launch, and any
CUDA failure stops the request without a CPU fallback.

An interleaved control/test/test/control development run produced:

| Arm | Repeat 1 | Repeat 2 | Mean effective rate |
| --- | ---: | ---: | ---: |
| CUDA FFN + VQ3R control | 4.694486 s, 1.704127 tok/s | 4.698418 s, 1.702701 tok/s | 1.703414 tok/s |
| + CUDA GQA projections | 2.223362 s, 3.598155 tok/s | 2.217875 s, 3.607056 tok/s | 3.602606 tok/s |

The projection arm was 2.115x the control's mean throughput. In a separately
profiled pair, effective rate moved from 1.699467 to 3.627511 tok/s while the
standard-GQA phase fell from 2.967868618 to 0.483058310 seconds, an 83.72%
reduction. The candidate repeats were byte-identical. Against the controls,
all greedy tokens, argmaxes, top-10 sets, and ordered routes were unchanged;
maximum and mean absolute logit error were `9.5367e-6` and `6.5221e-7`.
This passed the strict numerical contract, but the candidate logits were not
bit-exact to the control.

A different 16-token route pattern provided the held-out bracket:

| Arm | Repeat 1 | Repeat 2 |
| --- | ---: | ---: |
| Control | 12.062276 s, 1.326449 tok/s | 12.224253 s, 1.308873 tok/s |
| GQA projection candidate | 7.078026 s, 2.260517 tok/s | 7.077308 s, 2.260747 tok/s |

The candidate was approximately 1.716x the bracketed control mean. Every arm
read 52,160,634,880 expert bytes; all 11,392 routed selections were unchanged,
and maximum absolute logit error was `1.04904e-5`. No swap I/O occurred.

The arithmetic smoke still returned exactly `323` and stopped on the expected
secondary EOS, now at 2.54 tok/s. A short studio-oriented check generated 70
coherent tokens in 27.82 seconds (2.52 tok/s). These are short qualification
rows on one host, not sustained-throughput or broad quality benchmarks.

## Layer-major GQA chunk-prefill qualification

Commit `ae4cb6d` adds a bounded, default-off prompt path for the full model.
`WASTE_GQA_CHUNK_PREFILL=1` walks each standard-GQA layer through the prompt
in strict causal token order, then reuses the existing chunk-wide dense-FFN
and MoE machinery. On the qualified CUDA profile it must be paired with
`WASTE_CUDA_PREFILL_VQ=1`; the exact full-GLM geometry, KDA-1/dense-3/GQA-1/
VQ-2/group-1 selectors, and completed CUDA preflights are all required. Other
CUDA combinations fail closed.

The router, routing correction, fp32 K/V state, Q/K normalization, partial
RoPE, score/value attention, and router-ordered decode accumulation retain
their existing contracts. Prompt GQA projections use the qualified CUDA Q4
path, routed prompt experts use the qualified CUDA VQ3R path, and the LM head
is evaluated once for the completed prompt. A one-token tail uses the normal
qualified step path rather than creating a degenerate chunk.

The matched 34-token prompt bracket was:

| Arm | Repeat 1 | Repeat 2 | Mean prompt time |
| --- | ---: | ---: | ---: |
| Token-major control | 33.31 s | 33.42 s | 33.365 s |
| Layer-major CUDA candidate | 13.88 s | 13.80 s | 13.840 s |

The candidate reduced prompt latency by 58.519% and was 2.41077x faster. Its
prompt rate was 2.45-2.46 tok/s. Reported expert traffic fell from 110.69 GB
and 13,403 misses to 66.22 GB and 8,018 misses. The candidate trace contained
92 completed layer rows, including all 89 MoE layers and no failed rows; its
prompt-only physical read count was 68,096,143,360 bytes.

Profiling explains both halves of the gain. The standard-GQA phase fell from
12.74-12.78 seconds to 2.15 seconds, while MoE fell from 20.09-20.15 seconds
to 11.86-11.93 seconds. The candidate spent about 0.85 seconds in 276 batched
matrix calls. Its prompt-plus-one-step counters were 13,156 aggregate dense
calls, 12,880 GQA projection calls, 24,920 experts, 74,760 VQ applies, 36,338
LUT builds, 80,469 launches, 49,840 synchronizations, and zero fallbacks.

Each control repeated byte-for-byte, as did each candidate. Across arms,
argmax and ordered top-10 logits were unchanged; maximum and mean absolute
logit differences were `1.144409e-5` and `1.538187e-6`. All 3,115 routes had
identical expert IDs after keying by position and layer, and the maximum
router-weight difference was `1e-5`. The next eight generated tokens were
also identical.

Including those eight tokens, total time fell from 36.512 to 17.332 seconds,
a 2.10662x full-request improvement. Subsequent decode steps averaged 0.40914
seconds for the control and 0.43557 seconds for the candidate, a 6.46%
regression, so the promotion is specifically a prompt/full-request result and
not a decode-speed claim. The arithmetic smoke still returned exactly `323`
and the expected EOS, and every arm recorded zero swap-I/O delta.

The result passed the registered performance and numerical gates and is
retained as the local experimental full-GLM prompt profile. The selector
remains opt-in rather than default and does not imply upstream support or a
general result for other models or hardware.

## Fused VQ pipeline experiment

An earlier default-off experiment kept each expert's VQ gate/up result on the
GPU, applied SiLU and the down projection there, and synchronized only the
final down vector. Experimental commit `85e011f` was tested and then reverted
by `0aa550b` after it missed the performance gate.

The interleaved mode-2/mode-3/mode-3/mode-2 development run produced:

| Arm | Repeat 1 | Repeat 2 | Mean effective rate |
| --- | ---: | ---: | ---: |
| Existing VQ mode 2 | 2.218469 s, 3.606090 tok/s | 2.210495 s, 3.619099 tok/s | 3.6125945 tok/s |
| Fused VQ mode 3 | 2.191513 s, 3.650445 tok/s | 2.224014 s, 3.597099 tok/s | 3.623772 tok/s |

The measured gain was 0.309%, below the registered gate. Expert traffic was
identical at 7,147,479,040 bytes. Greedy tokens, top-10 logits, and ordered
routes were unchanged; maximum absolute logit difference was `6.6757e-6`.
Semantic counters were exact. Synchronizations fell from 11,392 to 5,696,
while launches rose from 17,800 to 23,496. The profiled VQ phase P7 was
effectively unchanged (`1.077652455` versus `1.079494174` seconds), showing
that removing this host handoff did not remove the dominant work. No held-out
run was performed for the rejected candidate.
