# CUDA backend provider (review layer 2)

This directory is a source-separated provider for the public
`waste_backend_v1` seam. It is intentionally not wired into WARP's root build.
An embedding host builds and links `libwaste_cuda_backend.a`, then passes
`&waste_cuda_backend_provider` (plus an optional provider config) to
`waste_open_with_backend`.

The review scope is the promoted, synchronous group-one implementation:

- Q4G/group-128 projections, with the fast kernel selected by default;
- strict VQ3R gate/up and down gathers with device-built LUTs;
- router selection, SiLU/SiTU, expert weighting, and router-ordered final
  accumulation remain in WARP;
- absorbed MLA `kv_b`, routers, embeddings, vision, and the language-model
  head are never claimed;
- VQ4P, grouped expert execution, non-128 Q4 groups, and model geometries
  other than the qualified K2 and K3 fingerprints fail closed.

The kernels are extracted from the promoted group-one state at commit
`96613e5` (whose CUDA lineage is `842a8f7`, `cb53b2c`, `1bca4d4`, then
`96613e5`). K2's allowlist and promoted composition were established later in
`95a0e27`, `32c8aa8`, `2cfde58`, and `d8634d5`. This provider deliberately
excludes the grouped-execution and VQ4P experiment branches. Public tensor
views replace every former `model.h` dependency; the per-record scheme,
matrix lengths, and channel-scale counts are checked before a borrowed expert
record is dispatched.

`make test` is a host-only policy test and does not require CUDA. `make`
requires an NVIDIA CUDA toolkit. The selected numerical flags are part of the
contract: device code is built for `CUDA_ARCH=native` with `-fmad=false` and
host code with `-ffp-contract=off`. These flags are appended in the recipes so
caller-supplied `CFLAGS` or `NVCCFLAGS` cannot silently remove them. Set
`CUDA_ARCH` explicitly only when cross-compiling. Link the archive with CUDA
runtime, C++ runtime, and pthread libraries appropriate to the host toolchain.

Version 1 targets device ordinal zero and requires coherent pageable memory
with host page tables, as exposed by the CUDA runtime. The planner charges its
staging/LUT/codebook allocations plus a visible 64 MiB allowance for opaque
CUDA process/context overhead. That allowance is a qualified GB10 policy, not
a portable measurement of every CUDA driver.

The v1 open view contains the trunk and codebooks but no expert record, so the
old in-engine preflight that ran one real VQ record before the first token
cannot live in this provider. The first callback failure is still sticky and
never falls back, but a host that wants the old warm-before-mutation gate must
perform a sacrificial/resettable inference before serving requests (or a
future seam can add an explicit warm-record callback).
