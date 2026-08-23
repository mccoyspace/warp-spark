# Kimi K2 CUDA prefill on NVIDIA GB10

Date: 2026-08-23  
Status: experimental, practical profile selected

This note records the sanitized Kimi K2 chunk-prefill qualification on a
128 GiB NVIDIA GB10 system. It extends the existing K2 CUDA decode profile;
it does not change the model container or claim a higher steady-state decode
rate. Raw prompts, completions, captures, and machine paths remain private.

## Result

Two existing CUDA primitives were admitted behind the complete K2 geometry
and runtime preflight:

- VQ3R expert application during chunk prefill (`PREFILL_VQ=1`), which keeps
  the established byte-exact contract;
- dense Q4 projection mode 1 during chunk prefill (`PREFILL_DENSE=1`), which
  is a faster bounded-numerical path.

Every arm was a fresh process under the same selected K2 decode profile. The
opening and closing controls bracketed each prompt shape.

| Prompt | Bracket control | Exact VQ | Practical dense 1 | Dense 2 |
| --- | ---: | ---: | ---: | ---: |
| 53 tokens | 41.99 s / 1.26 tok/s | 37.68 s / 1.41 tok/s | **28.78 s / 1.84 tok/s** | 32.61 s / 1.63 tok/s |
| 288 tokens | 210.64 s / 1.37 tok/s | 162.38 s / 1.77 tok/s | **116.84 s / 2.46 tok/s** | 138.68 s / 2.08 tok/s |

On the 288-token studio-shaped prompt, the selected practical arm reduced
prefill wall time by **44.5%** and increased prompt throughput by **1.80x**.
The exact VQ-only fallback still reduced wall time by 22.9% and increased
prompt throughput by 1.30x.

This is primarily a time-to-first-token improvement. The previously qualified
K2 decode result remains 2.748 tok/s mean and 3.001 tok/s best observed.

## Behavioral gates

Across both matched prompt shapes:

- all arms produced the same first eight continuation token IDs;
- exact VQ was byte-identical in final logits and routes;
- practical dense mode 1 kept the same argmax, ordered top 10, and expert
  membership, with zero CUDA fallbacks;
- its worst final-logit absolute difference was `1.5259e-5`;
- one of 17,760 long-prompt route rows reordered the same expert set; no
  expert was replaced.

The practical profile is intentionally not described as bit-exact. The exact
VQ-only profile remains available when that stronger contract is needed.

A separate 64-token continuation confirmation used the 288-token prompt.
Exact VQ measured 163.02 seconds of prefill (1.77 tok/s); practical dense mode
1 measured 116.29 seconds (2.48 tok/s). All 64 generated token IDs matched,
and the numerical, routing, and fallback observations above repeated.

## Why it helps

On the 288-token prompt, dense mode 1 reduced measured attention time from
72.7 to 26.2 seconds. VQ/cache reuse reduced feed-forward time from 138.4 to
90.3 seconds and physical expert reads from about 700 to 539 GB. The two
savings compose, which is why the combined gain is materially larger than
either isolated change.

## Selected settings

The practical profile adds the following to the existing qualified K2 CUDA
decode configuration:

```text
WASTE_CUDA_PREFILL_VQ=1
WASTE_CUDA_PREFILL_DENSE=1
```

The exact fallback uses:

```text
WASTE_CUDA_PREFILL_VQ=1
WASTE_CUDA_PREFILL_DENSE=0
```

The implementation fails closed unless the exact K2 geometry, dense scope 3,
CUDA VQ mode 2, and their runtime preflights all succeed.
