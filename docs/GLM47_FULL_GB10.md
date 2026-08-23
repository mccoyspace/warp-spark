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

## Current scope

- The full model currently uses WARP's CPU implementation of standard grouped
  query attention. CUDA was not used or qualified for this result.
- Standard-GQA state snapshots are not implemented, so snapshot export/import
  is unavailable for this model.
- The checkpoint's optional MTP layer was omitted and is unsupported.
- The conversion and runtime gates target the exact pinned GLM-4.7 source
  contract and fail closed outside that experimental scope.
- Source weights remain retained. Nothing here is an upstream support or
  performance claim.
