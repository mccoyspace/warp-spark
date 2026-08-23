# Full GLM-4.7 on NVIDIA GB10

This note records a sanitized first experimental qualification of the full
`zai-org/GLM-4.7` model on a 128 GiB NVIDIA GB10 system. It is a CPU-only
bring-up result, not an upstream-supported model or performance claim. Raw
prompts, completions, captures, and machine paths are intentionally excluded.

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

After CUDA FFN/VQ promotion, standard GQA accounts for roughly 64% of the
remaining decode time. Offloading only its Q/K/V/O projections is therefore
the next bounded experiment. These short rows remain qualification evidence,
not sustained or quality-benchmark claims.
