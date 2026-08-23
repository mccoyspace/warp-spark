# Kimi K3 CUDA prefill on NVIDIA GB10

Date: 2026-08-23
Status: experimental, practical profile selected by the matched campaign

This note records the sanitized Kimi K3 chunk-prefill qualification on a
128 GiB NVIDIA GB10 system. It extends the existing K3 CUDA decode profile;
it does not change the model container or claim a higher steady-state decode
rate. Raw prompts, completions, captures, and machine paths remain private.

## What changed

Two existing CUDA primitives are admitted behind a complete K3 geometry and
runtime preflight:

- VQ3R expert application during chunk prefill (`PREFILL_VQ=1`), preserving
  the established byte-exact expert-application contract;
- dense Q4 projection mode 1 during chunk prefill (`PREFILL_DENSE=1`), a
  faster bounded-numerical path covering K3's mixed KDA/MLA scope 2.

The practical arm composes both paths. The exact fallback enables VQ prefill
but keeps dense attention projections on their CPU path. K3 deliberately
rejects dense prefill mode 2: only the measured mode-1 arithmetic has been
admitted for this geometry.

## Implementation and smoke gates

The allowlist checks the complete text-side K3 fingerprint rather than only
its headline dimensions: 93 layers, the exact 69-KDA/24-MLA layer pattern,
latent-MoE dimensions and normalization, full-rank KDA gate, SiTU, Attention
Residuals, `language_model.` prefix, expert shapes, and VQ3R layout. Static
geometry, runtime CUDA preflight, dense scope, and selected mode must all
agree; stale or directly mutated state fails closed before prefill changes
recurrent state.

Model-free negative tests reject mismatches across every checked geometry,
KDA-pattern, VQ-layout, scope, mode, and runtime-preflight field. The local
CUDA-geometry test and 21-test comparison-helper suite pass at source
`0e8a84a`. The final full test suite reported 54 passed, zero failed, and 14
skipped tests.

A real-container smoke used a 20-token prompt, two generated tokens, and the
current K3 fast-decode profile:

| Arm | Prefill | Prompt throughput | Relative to control |
| --- | ---: | ---: | ---: |
| Control | 63.78 s | 0.31 tok/s | — |
| Exact VQ | 61.60 s | 0.32 tok/s | 1.035x |
| Practical dense 1 + VQ | **36.90 s** | **0.54 tok/s** | **1.728x; 42.1% less wall** |

Exact VQ was byte-identical in final logits and ordered routes. Practical
dense mode 1 produced the same two-token continuation, argmax, ordered top 10,
and routes, with no route-order or membership changes. Its maximum final-logit
absolute difference was `1.0490e-5` and its mean absolute difference was
`1.5519e-6`. The real-model fingerprint and runtime preflights passed, and
there were zero KDA fallbacks.

Work counters also reflected the intended expansion from decode-only CUDA to
prefill plus decode: control/exact-VQ recorded 1,104 KDA and 1,160 dense calls;
the practical arm recorded 12,144 and 3,560. Control recorded 2,944 CUDA VQ
experts and 8,832 applies, while each VQ-prefill arm recorded 32,384 experts
and 97,152 applies.

## Matched performance campaign

Every arm uses a fresh process under the same selected K3 decode profile. An
opening and closing control bracket each prompt shape. Prompt descriptions are
intentionally generic; the retained raw fixtures remain private.

| Prompt | Bracket control | Exact VQ | Practical dense 1 |
| --- | ---: | ---: | ---: |
| 53 tokens | 150.47 s / 0.35 tok/s | 139.33 s / 0.38 tok/s | **71.59 s / 0.74 tok/s** |
| 288 tokens | 794.44 s / 0.36 tok/s | 718.49 s / 0.40 tok/s | **347.11 s / 0.83 tok/s** |

On the longer prompt, the selected practical arm reduced prefill wall time by
**56.3%** and increased prompt throughput by **2.29x** versus the bracket
control. The exact VQ-only arm reduced wall time by 9.6% and increased prompt
throughput by 1.11x. On the short prompt, practical dense mode 1 was 2.10x
faster and reduced wall time by 52.4%; the opening and closing controls differed
by less than 0.04%.

This is primarily a time-to-first-token result. The existing qualified K3
decode measurements remain separate and unchanged.

## Behavioral gates

- exact VQ final logits and ordered routes were byte-identical on both prompt
  shapes;
- practical mode 1 produced the same first eight continuation token IDs on
  both shapes;
- practical mode 1 retained argmax, ordered top 10, and expert membership;
  four of 32,844 route rows reordered the same expert set, with zero expert
  replacements;
- its worst final-logit absolute difference was `3.0398e-5`;
- all qualified arms had zero CUDA fallbacks, complete prefill traces, and the
  exact registered K3 semantic work counts.

The automatic campaign analyzer passed all 91 registered gates.

The practical profile must not be described as bit-exact unless the retained
evidence establishes that stronger result. The exact VQ-only profile remains
available when the ordered CPU dense arithmetic is required.

## Product-style confirmation

A separate fresh-server confirmation used a 74-token chat prompt and generated
64 tokens greedily under the same qualified decode settings. Raw prompt and
completion text remain private.

| Arm | Time to first token | Full request | Full-request throughput | Post-first-token decode |
| --- | ---: | ---: | ---: | ---: |
| Exact VQ | 189.401 s | 266.145 s | 0.24047 tok/s | 0.82091 tok/s |
| Practical dense 1 + VQ | **94.646 s** | **171.648 s** | **0.37286 tok/s** | 0.81815 tok/s |

The practical profile halved time to first token (**2.001x**), increased
full-request throughput by **1.55052x**, and reduced full-request wall time by
**35.5056%**. Post-first-token decode was effectively unchanged at 0.996637x
the exact arm, as expected for an optimization confined to prompt prefill.

The first 49 of 64 generated token IDs matched before a late trajectory
divergence. Both retained completions were reviewed privately and remained
coherent and on task, with no collapse or repetition. This is useful practical
behavior, not an exact-generation guarantee: the exact VQ-only profile remains
available when trajectory preservation is required.

## Selected settings

The practical profile adds the following to the existing qualified K3 CUDA
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

The implementation fails closed unless the exact K3 geometry, dense scope 2,
CUDA VQ mode 2, and their runtime preflights all succeed.
