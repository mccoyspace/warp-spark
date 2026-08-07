# Draft comment for upstream issue #11

CUDA is viable for K3 on the NVIDIA GB10, and we now have an engine-level
result rather than a kernel projection. The qualified VQ3R path is
byte-identical to the CPU path. In a matched fresh-server request it raised
full-request throughput from **0.1112 to 0.2125 tok/s (+91.05%)** while reducing
wall energy from **1,018.39 to 609.61 J/generated token (−40.14%)**. It was
promoted only after a **4.52-hour soak with 50/50 exact 64-token trajectories**.

Scope first: this does **not** close gate 4 for a discrete card. GB10 is a third
regime between the Apple result and the PCIe measurements already in this
thread: its CPU and GPU share coherent LPDDR5x, so routed expert records do not
pay a per-token H2D copy, while its NVIDIA GPU still requires explicit CUDA
kernels and synchronization. What it establishes is narrower and useful: the
original conclusion that CUDA must begin as a whole-forward, residual-resident
engine is not universal. On GB10, incremental kernel-class offload paid while
attention state, routing and final expert accumulation remained on the CPU.

The acceptance arc was:

| Stage | K3 decode result | Contract |
| --- | ---: | --- |
| CPU baseline before GPU work | 0.34–0.35 tok/s | deterministic CPU reference |
| KDA CUDA pilot | 0.476 tok/s | bounded logit tolerance; routes and tokens unchanged |
| all accepted dense Q4 projections | 0.673 tok/s | same route/token contract |
| VQ3R gather, strict 64-token capture | **0.902 tok/s** | byte-identical logits, routes and tokens |
| qualified held-out H3 default | **0.637 tok/s at 121.35 W** | 12 unseen 64-token studio prompts |
| optional held-out lookahead-six arm | 0.728 tok/s at 132.13 W | exact output; remains opt-in |

The design element that made the strict VQ result possible was preserving the
original router accumulation order. CUDA produces a separate partial for each
selected expert; the CPU applies the router weights and reduces those partials
in original router order. Across the strict capture that produced zero changes
in 10,649,600 logits and 5,888 routed rows, with exact traffic/call counters and
zero fallback. Keeping the router itself on the CPU also made route invariance
an interpretable gate.

This does not contradict the earlier Metal result. Metal's backend-shaped path
paid synchronous launch/round-trip economics for small calls and measured 22%
slower. The GB10 path uses coherent unified memory, first established a ≥2×
isolated KDA-layer gate, and then offloaded successively larger dense and routed
kernel classes without a PCIe expert transfer. CUDA graphs were not needed at
the measured per-call synchronization cost.

The current v0.6.6 realignment consumes upstream's generalized VQ metadata, so
the kernel is now the CUDA implementation of a pluggable VQ interface rather
than a separate hardcoded record parser. Its contract is deliberately narrow:
**VQ3R-complete; VQ4P rejected explicitly pending its own numerical and
performance contract.** A focused real-K3 check after realignment again found
byte-identical logits/routes/tokens and no measurable regression against the
previous qualified source.

The compact public evidence and raw-archive links are in the
[consolidated GN100 release](https://github.com/mccoyspace/waste-spark/releases/tag/gn100-consolidated-results-2026-08-07),
and the current-upstream compatibility check is in the
[v0.6.6 realignment release](https://github.com/mccoyspace/waste-spark/releases/tag/gn100-upstream-v066-realignment-2026-08-07),
with the operating profile in
[`docs/GB10_QUALIFIED_PROFILE.md`](https://github.com/mccoyspace/waste-spark/blob/spark/integration/docs/GB10_QUALIFIED_PROFILE.md).
I am happy to walk through or separate the CUDA source if that would be useful;
I have not opened a CUDA PR because VQ4P coverage and the preferred upstream
architecture should be agreed first.
