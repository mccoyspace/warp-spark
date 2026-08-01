# Portability and acceleration backends

Scope requirement (Marco, 2026-07-27): the engine runs on **macOS, Windows
and Linux**; acceleration backends (BLAS, CUDA, Metal, NEON, AVX-512, …)
are selected without burdening a build that does not want them, over a
**universal CPU version always available**. The dispatch discipline follows [sqlite-vector](https://github.com/sqliteai/sqlite-vector).

**No dynamic loading.** An earlier draft resolved accelerators with
`dlopen`/`LoadLibrary`; that was complexity without a matching problem, and
sqlite-vector, the model here, uses none. Backends are chosen by
**conditional compilation plus runtime feature detection**: one binary
still adapts to the CPU it runs on, and a build without CUDA simply has no
CUDA code in it.

## The sqlite-vector pattern, and what we take from it

sqlite-vector keeps one global table of function pointers
(`dispatch_distance_table[VECTOR_DISTANCE_MAX][VECTOR_TYPE_MAX]`,
`src/distance-cpu.c`). At init it `memcpy`s a fully-populated CPU table in,
then the best backend detected at runtime **overwrites the entries it
implements** — `init_distance_functions()` tries AVX-512 → AVX2 → SSE2 on
x86, NEON on ARM, RVV on RISC-V, with a `force_cpu` escape hatch and the
selected backend's name exposed for introspection (`vector_backend()`).
Detection is careful: `cpu_supports_avx512()` checks CPUID *and* XGETBV, so
a CPU whose OS has not enabled ZMM state is correctly rejected.

WASTE adopts all of it:

- one dispatch table, filled with a baseline that is **always compiled in**;
- backends **partially override** — an unimplemented kernel keeps the CPU
  version, so a new backend can start with one hot kernel and grow;
- **runtime detection with OS-support checks**, best-first with fallback;
- **force-CPU escape hatch** (`WASTE_BACKEND=cpu`, or
  `waste_backend_init(WASTE_BE_FORCE_CPU)`) — indispensable for bisecting
  numeric differences;
- **name introspection** (`waste_backend_name()`), surfaced by the CLI.

One difference: sqlite-vector's distance functions all share a signature,
so a 2-D array works. WASTE's kernels do not, so the table is a **struct of
function pointers** (`waste_kernels` in
[src/waste_backend.h](../src/waste_backend.h)). Same idea, C-idiomatic for
heterogeneous ops.

## How a backend gets in

*Design, not inventory.* Today's inventory, kept in one place so the rest
of this section can stay about the mechanism:

| backend | file | state |
|---|---|---|
| portable C baseline | `kda.c`, `model.c`, `vq.c` | always compiled in |
| NEON | `kda_neon.c` + inline in `model.c`/`vq.c` | default on ARM |
| AVX2 | `simd_avx2.c` | verified on Linux/x86_64 |
| AVX-512 | `simd_avx512.c` | compiled and dispatched, never executed — CI runner is Zen 3, confirmed 2026-07-29 |
| Metal | `metal.m` | correct, off by default, 22% slower |
| CUDA, BLAS, ROCm | — | not implemented; the flag refuses to build |
| SVE, RVV | — | not implemented |

Details for each are in the dated sections below. The rest of this one
describes where a new backend plugs in.

**SIMD** (NEON, dotprod/i8mm, AVX2, AVX-512, SVE, RVV): one translation
unit per ISA (`kda.c`, `kda_neon.c`, `kda_avx2.c`, …), each guarded by
`#if defined(__ARM_NEON)`-style fences so a build for another architecture
compiles them away, and each built with its own flags. All the ones valid
for the target architecture are compiled in, and `waste_cpu_features()`
picks at runtime — that is what lets a single x86 binary use AVX-512 on a
machine that has it and AVX2 on one that does not.

**Accelerators** (CUDA, Metal, BLAS, ROCm): build-time options, each
needing a source file that registers it. `src/metal.m` exists; `src/cuda.cu`
and `src/blas.c` do not, and setting their flag stops the build with a
message saying so rather than failing at link time on an undefined
`waste_register_*`. Deleting that check is the last step of adding the
backend it guards.

```
make                     # CPU + SIMD only, zero extra dependencies
make WASTE_ENABLE_METAL=1
make WASTE_ENABLE_CUDA=1
```

A build without `WASTE_ENABLE_CUDA` contains no CUDA code and no link
dependency — which is the whole reason dlopen looked tempting, solved more
simply by not linking it. A build *with* it still calls
`waste_register_cuda()` at init, which probes for a usable device and
**returns NULL to decline** if there is none; the engine then keeps the
backend it already had. Declining is normal, not an error.

Metal deserves a note: it is present on every Mac that can run this engine,
so on macOS it is a plain `#ifdef __APPLE__` decision with no runtime
uncertainty beyond device selection.

## Platform abstraction

Beyond kernels, a handful of calls differ per OS. They live in
[src/platform.h](../src/platform.h) rather than sprinkled through the
engine — one Windows implementation and one POSIX line each, so no call
site carries a branch:

| concern | macOS | Linux | Windows |
|---|---|---|---|
| cache-bypass read | `fcntl(F_NOCACHE)` | `O_DIRECT` | `FILE_FLAG_NO_BUFFERING` |
| positional read | `pread` | `pread`, optional raw `io_uring` | `ReadFile` + `OVERLAPPED` |
| aligned allocation | `posix_memalign` | `posix_memalign` | `_aligned_malloc` |
| CPU count | `sysconf` | `sysconf` | `GetActiveProcessorCount` |
| physical RAM | `sysctlbyname("hw.memsize")` | `sysconf(_SC_PHYS_PAGES)` | `GlobalMemoryStatusEx` |
| CPU features | `sysctlbyname("hw.optional.arm.*")` | `getauxval(AT_HWCAP/2)` | `IsProcessorFeaturePresent` / CPUID |
| threads | pthreads | pthreads | pthreads (winpthreads) |

Model data is never mapped: the engine reads by offset. The optional Linux
ring maps only the kernel's submission/completion queues and SQE array, as the
`io_uring` ABI requires; expert banks remain direct positional reads.

The expert-streaming path is the one that really cares: Gate H showed
throughput is set by random 12 MB reads with the page cache bypassed, and
that call differs on all three platforms.

The Linux ring is implemented directly against `linux/io_uring.h`, with no
runtime liburing dependency. A depth is bounded to 64, completion results are
matched by request index, and setup failure falls back to `pread` while
surfacing that fact in `waste_stats`. QD4 on the Acer GN100 improved the K3
16-token median by 14.4% and the 64-token generation time by 15.1%; it is an
opt-in measured backend rather than a portable assumption.

`pread_batch` is the causal control for that result. It uses the same batched
cache reservation and unchanged expert order, but executes one synchronous
`pread` at a time and finishes the layer's reads before compute. Its physical
queue depth is always one. A speedup over interleaved `pread` therefore comes
from I/O/compute phase separation; the remaining QD4 advantage comes from
concurrent storage requests.

## Status

Implemented and verified today:

- `waste_cpu_features()` — x86 CPUID + XGETBV (checks OS-enabled AVX/AVX-512
  state, as sqlite-vector does), aarch64 with per-OS dot-product/i8mm
  detection (macOS `sysctlbyname`, Linux `getauxval`, Windows
  `IsProcessorFeaturePresent`);
- `waste_backend_init()` — CPU baseline, then the best SIMD backend for the
  machine, then any accelerator compiled into this build;
- first kernel family (KDA) wired through the table:
  [src/kda.c](../src/kda.c) is the universal baseline,
  [src/kda_neon.c](../src/kda_neon.c) the NEON specialization.

Verified on this machine (MacBook Pro M5 Pro): `tools/kda_ref.py` passes
against the official reference on *both* paths —

| path | output max\|diff\| | state max\|diff\| |
|---|---|---|
| `WASTE_BACKEND=cpu` (baseline) | 4.10e-08 | 1.79e-07 |
| auto (NEON) | 4.47e-08 | 1.19e-07 |

`waste_cpu_features()` also detects dotprod and i8mm on this CPU, and the
build used to name itself `NEON+i8mm` on the strength of that. Nothing in
the engine emits SDOT or SMMLA, so the name has been cut back to `NEON`:
the backend string reports what the binary *uses*, not what the silicon
offers. Put the suffix back in the commit that adds the kernel.

That equivalence is the contract every future backend must meet: **same
results, only faster.** Windows is built and run in CI as of 2026-07-29;
what it cost, and what is still not claimed, is at the end.

## Machine-specific optimization: measured, not guessed (2026-07-27)

Optimizing started from a profile of the C forward pass on Kimi-Linear
(`WASTE_PROFILE=1`), not from intuition. The order the profile dictated:

| step | s/token | what the profile then said |
|---|---|---|
| first correct version | 2.15 | MoE 71%, of which dequant 37% + matmul 30% |
| NEON f32 dot + thread pool | 0.93 | expert **dequantization 87.5%** |
| fused VQ matvec (no dequant) | 0.22 | expert matmul 67%, KDA 12%, I/O 12% |
| hoist gate/up tables out of the expert loop | **0.18** | expert matmul 56%, I/O 17%, KDA 15% |

**11.9x, and the logits still match the oracle** (max abs diff 4.6e-05,
relative 1.5e-06, argmax and top-10 identical) — the same check is rerun
after every step, because an optimization that changes results is not an
optimization.

The two that mattered:

1. **Never dequantize an expert.** The first version expanded VQ indices
   into f32 weights and then multiplied — 87% of the time. Instead, note
   that `sum_s C_s[i] . x_v` depends only on (stage, code, vector
   position), never on the output row: tabulate it once per matrix and
   every row becomes 3 table lookups per 8 weights. This is
   sqlite-vector's turbo-LUT idea applied to a weight matrix rather than a
   distance. Dequantization dropped from 87.5% to nothing, and the
   remaining 16% under "expert deq" is now purely file I/O.
2. **Hoist what does not vary.** Every routed expert in a layer sees the
   same input and the same per-layer codebooks for its gate and up
   matrices, so those two tables are built once per token instead of once
   per expert — 8x less table-building on two of the three matrices.

**Thread scaling** on this M5 Pro (18 logical cores, 6 performance):

| threads | 1 | 4 | 8 | 12 | 18 |
|---|---|---|---|---|---|
| s/token | 0.45 | 0.20 | 0.20 | 0.18 | 0.18 |

2.5x, flattening after ~4 — consistent with the performance-core count and
with a workload that is becoming memory-bound. The pool splits by row, so
results are bit-identical at any thread count; `WASTE_THREADS` overrides.

### int8 and SDOT: where the instruction actually fits

The expert matmul's inner loop is a **gather** (`acc += lut[block + s*256 +
code]`), with no multiply — SDOT cannot vectorize it, and ARM has no gather
instruction. Cache-blocking it (`VQ_TILE`, swept: 64 and 128 tie, larger is
worse) bought 0.18 -> 0.15 s/token; the loop is latency-bound on dependent
loads, not bandwidth-bound.

Where SDOT *does* fit is the trunk: those are dense dots, and the trunk is
already stored Q8G (int8 + one fp16 scale per 128 inputs). Keeping it int8
instead of expanding to f32 at load gives three modes:

| mode | s/token | RSS | logits vs oracle | top-10 |
|---|---|---|---|---|
| f32 weights (expand at load) | 0.15 | 9.5 GB | rel 1.6e-06 | identical |
| **int8 stored, f32 math** | **0.13** | **3.9 GB** | **rel 1.5e-06** | **identical** |
| int8 stored, int8 acts + SDOT | 0.13 | 3.9 GB | rel 1.2e-02 | **reordered** |

SDOT does what it promises on its own slice — the trunk phases drop from
0.30 s to 0.16 s, ~1.9x — but the trunk is only ~16% of a token, so Amdahl
caps the end-to-end gain at ~13%, which the f32-math path matches without
quantizing activations. Quantizing them costs four orders of magnitude of
accuracy and reorders the top-10; argmax survived here, but that is luck,
not a guarantee.

**Default: int8 storage with f32 arithmetic.** It keeps the container's
precision exactly, and the memory saving is the part that matters — 5.6 GB
freed is 5.6 GB more expert cache, and Gate 2 says cache is what buys
tokens/sec. `WASTE_SDOT=1` enables the activation-quantized path for anyone
who wants to measure the trade on their own workload.

Still on the table: i8mm/SMMLA for batched prefill (where activations are a
matrix and the accuracy trade is amortized over more work), a NEON pass
over the LUT accumulation, and Metal for the prefill GEMMs.

## Not yet done

*(This list is what was outstanding on 2026-07-27. AVX2, AVX-512, Metal,
the platform I/O wrapper and the CI matrix all landed the following day
and have their own sections below; what remains of it is CUDA, BLAS and
Windows.)*

The CUDA and BLAS backends. Windows was on this list until 2026-07-29;
"the branches are written" had turned out to mean one branch, in
dot-product detection. The measurement, and the port that followed it,
are at the end of this document.

## i8mm/SMMLA for the batched matmul: 2x on its own work, 1.2% overall

SMMLA multiplies a 2x8 int8 tile by an 8x2 tile into a 2x2 int32
accumulator — 32 MACs per instruction against 4 for an fp32 FMA. The
natural target is `mmq_rows`, the batched trunk matmul the chunked
prefill uses for the latent projections, the shared experts and the
dense FFN. `src/model.c` now has `mmq_rows_i8mm`, tiling two weight rows
by two tokens and accumulating per quantization group so each group's
pair of scales applies once.

It works, and it is off by default. Measured on a 19-token chunked
prefill of K3:

| | batched mm | total prefill |
|---|---|---|
| f32 path | 2.38 s (6.0%) | 46.20 s |
| SMMLA | 1.15 s (3.0%) | 45.66 s |

**2.07x on the kernel, 1.2% end to end** — because the batched matmul is
only 6% of prefill to begin with. The time is in the VQ path: LUT apply
37.0% plus LUT build 17.7%, then expert I/O 37.6%. That is where the
next SIMD work belongs, and it is gather-shaped rather than GEMM-shaped,
so SMMLA does not reach it.

It also is not free numerically. SMMLA needs int8 on both sides, so the
activations get quantized per group, which the f32 path deliberately
avoids. Logits move by 6.8e-02 relative — argmax and top-5 held on the
prompt tested, but that is a real change, not fp noise.

Hence two switches, both off: build with `make WASTE_NATIVE=1` (the
default build targets baseline ARM, where `__ARM_FEATURE_MATMUL_INT8` is
not defined and the kernel compiles away) and run with `WASTE_I8MM=1`.
Turning it on by default would trade measurable accuracy for 1.2%. If
the VQ path ever gets fast enough that the batched matmul's share grows,
revisit — and at that point move the kernel into its own translation
unit so runtime dispatch can pick it, instead of requiring a native
build.

## First Linux runs (2026-07-28)

The engine had never been built on Linux. Docker, both architectures,
`tests/run.sh` against the Kimi-Linear container:

| | build | suite | backend | generation |
|---|---|---|---|---|
| Linux arm64 | ok | 12 pass, 0 fail, 4 skip | `NEON` | correct |
| Linux x86_64 | ok | 12 pass, 0 fail, 4 skip | `CPU` | correct |
| macOS arm64 | ok | 17 pass | `NEON` | correct |

*(2026-07-29: re-run on ubuntu:24.04 / gcc 13.3, model-free, after the CI
defects in [LEARNED.md](LEARNED.md) §17 — both Linux targets 17 pass,
0 fail, 8 skip, plus the sanitizer suite and 400 fuzz cases. x86_64 now
names itself `AVX2` rather than `CPU`.)*

Both Linux targets produce "The capital of France is Paris, and the
capital of Italy is Rome" — the same continuation as macOS — and both
pass *engine matches the PyTorch oracle*, so the numerics carry across
platforms and architectures. The four skips need `uv` or the source
weights, neither of which is in the image.

x86_64 names itself `CPU`, not AVX2, even though `waste_cpu_features()
`detects AVX2 there. That is the version string doing its job: detection
is not a kernel, and there is no x86 SIMD in this engine.

Three defects turned up, none in the engine and all invisible from macOS:

- The Makefile added `kda_neon.c` when `uname -m` contained "arm". Linux
  on aarch64 reports **aarch64**, which does not, so the translation unit
  was dropped while `backend.c` — which tests `__aarch64__` — still
  emitted the call. Undefined `waste_kda_register_neon` at link.
- `check_budget.sh` measured peak RSS with `/usr/bin/time -l`: a BSD-only
  flag, and the tool is not in a plain debian image at all. It reported
  the budget as exceeded when it had simply measured nothing. Now
  `getrusage(RUSAGE_CHILDREN)` through python3.
- The first attempt bind-mounted the working tree, so the Linux build
  overwrote the macOS objects and `./waste` became "Exec format error"
  on the host. `Dockerfile.test` copies instead.

O_DIRECT works on Docker's volumes: `waste_stats.direct_io` stays 1 and
no warning prints, while `WASTE_DIRECT=0` produces the fallback and the
"hit rate is partly the kernel's" note. Both directions of that path are
now exercised, which is more than the macOS-only build could do.

CUDA remains untested and untestable here on two counts: there is no
NVIDIA GPU on this machine, and there is still no CUDA source to compile
— `WASTE_ENABLE_CUDA=1` stops the build with a message saying so.

## AVX2 and AVX-512 (2026-07-28)

x86 ran pure scalar C until now. Two translation units, `src/simd_avx2.c`
and `src/simd_avx512.c`, each built with its own `-mavx*` flags and
selected by `waste_backend_init` from CPUID — so one binary adapts, which
is why they are separate files rather than `#ifdef`s inside model.c.

They implement the two range kernels that carry the arithmetic:
`mvq_rows_f32` (every trunk projection) and `lutb_range` (the VQ table).
Those moved behind the dispatch table for this, with their argument
structs and the two shared inlines in a new `src/simd.h`. The third hot
path, the VQ gather, gets nothing — no x86 SIMD helps it either, for the
same reason NEON does not.

**AVX2 is verified.** On Linux/x86_64 the engine reports `backend AVX2`,
the suite is 12 passed / 0 failed, and *SIMD backend matches the CPU
baseline* passes — that check runs `WASTE_BACKEND=cpu` against the
dispatched path and compares logits, so it is exactly the claim that
matters. It is now "within fp noise" rather than bit-identical, because
AVX2 accumulates in a different order.

**AVX-512 is compiled and dispatched, never executed.** No machine here
has it: the container's CPU reports `AVX512F=0` in CPUID leaf 7, and so
does `qemu-x86_64 -cpu max` under Rosetta, whose XCR0 leaves the opmask
and ZMM bits clear. The detection is doing the right thing by declining —
that much is confirmed — but the kernels themselves have never run an
instruction. Treat the first AVX-512 machine as the test, and expect the
same "matches the CPU baseline" check to be the thing that decides it.

No performance numbers from any of this: x86 here is emulated, so timings
would measure Rosetta.

### Emulation cannot close this gap (2026-07-29)

Asked directly whether Docker could execute the AVX-512 path, and the
answer is no, for a reason worth writing down rather than re-deriving.
A CPUID probe inside `--platform linux/amd64` reports:

```
cpu        : VirtualApple @ 2.50GHz
AVX2       : 1
AVX512F    : 0
XCR0       : 0x7  (SSE 1 YMM 1 opmask 0 ZMM_hi 0 hi16 0)
```

That brand string is **Rosetta 2**, not QEMU — Docker Desktop on Apple
silicon translates x86 with Rosetta by default, which is also why
`QEMU_CPU=max`, `Skylake-Server-v4` and `Sapphire-Rapids` all change
nothing: the variable means nothing to a translator that is not QEMU.
Rosetta implements AVX2 and not AVX-512, and `XCR0 = 0x7` says the
opmask and ZMM state is not enabled, so an AVX-512 instruction could not
retire even if it decoded. Switching Docker to QEMU would not obviously
help either: TCG has implemented AVX/AVX2 for some releases and AVX-512
is not among them.

So the gap needs hardware. The cheapest that might already exist is the
CI runner, and the workflow now prints the CPU model and its avx512
flags before it builds — if a hosted runner has them, `waste version`
on the next push says `AVX-512` instead of `AVX2` and the *SIMD backend
matches the CPU baseline* check becomes the confirmation, at no cost.
If it does not, this stays open until someone runs the suite on an Ice
Lake, a Sapphire Rapids or a Zen 4. It did not; see below.

### The runner answered, and the answer is no (2026-07-29)

The workflow asked, and the hosted x86 runner closed the cheap option:

```
model name	: AMD EPYC 7763 64-Core Processor
  avx2: yes
  avx512f: no   avx512bw: no   avx512dq: no   avx512vl: no
WASTE 0.6.0 (container v0, backend AVX2, crc32 slice8, x86_64)
```

An EPYC 7763 is Milan, which is Zen 3 — AMD added AVX-512 in Zen 4, so
this is the generation before. The Windows x86_64 job reports `backend
AVX2` as well. The AVX-512 kernels are therefore still compiled, linked,
dispatched past and never entered, on every target this project builds
for.

One consequence is worth stating, because the check invites exactly the
wrong reading: *SIMD backend matches the CPU baseline* **passes** on
linux-x86_64 and says nothing whatsoever about AVX-512. It compares the
dispatched path against `WASTE_BACKEND=cpu`, and the dispatched path
there is AVX2. A green CI is not evidence about the AVX-512 kernels;
only the flags line above says which backend was under test.

So this stays open, and the hardware it wants is now specific: Zen 4 or
later on AMD, Ice Lake or Sapphire Rapids or later on Intel. A larger
hosted runner class would also reach it, at cost.

## Metal: it works, and it loses (2026-07-28)

`src/metal.m` implements `mvq_rows_f32` — the quantized matvec every trunk
projection and the output head go through. Built with
`make WASTE_ENABLE_METAL=1`; the shader is compiled from source at first
use, because the offline Metal compiler ships with Xcode and not the
Command Line Tools.

The design point that makes it worth trying at all: **no copies**. The
engine moves ~17 GB of weights per token, so a backend that staged host
memory to the device would lose before it started. Apple Silicon has
unified memory, so trunk tensors are allocated page-aligned — the
`waste_dio_alloc` that O_DIRECT already needed, widened from 4 KiB to the
16 KiB page — and wrapped with `newBufferWithBytesNoCopy`. The GPU reads
the same physical pages the CPU does.

It is correct: logits match the CPU baseline to 8.1e-06, argmax and top-5
unchanged, on both Kimi-Linear and K3.

It is also slower, on K3, 5 decode steps:

| | total | kda | mla | lm_head |
|---|---|---|---|---|
| CPU/NEON | 15.93 s | 1.78 | 0.39 | 0.03 |
| Metal | 19.43 s | 3.56 | 0.67 | 0.11 |

**22% slower.** The first kernel put one thread on a whole row, which is
the obvious mapping and the wrong one — adjacent threads read addresses
`rowbytes` apart, so every load is its own cache line. Rewriting it as one
threadgroup per row with strided, coalesced loads and a threadgroup
reduction took the total from 21.28 s to 19.43 s. Still behind.

`lm_head` is the clearest case, because it is one dispatch per token over
1.17 GB of int8 weights: CPU 6 ms, GPU 22 ms. That is 195 GB/s against
53 GB/s — the CPU path is already running at the machine's memory
bandwidth, and this is a bandwidth-bound matvec. There is no headroom for
the GPU to take, and it pays a synchronous round-trip per call on top.

The conclusion is about shape, not about Metal. This engine issues several
hundred small dependent matvecs per token and waits for each; that is the
worst possible fit for an accelerator. Making the GPU pay would mean
moving the whole forward pass on-device so there is one dispatch per layer
instead of six, with the residual stream never returning to the host — a
different engine, not a backend. The code stays, off by default, because
it is correct and because that argument should be re-run if the CPU path
ever stops being bandwidth-bound.

## CI

`.github/workflows/ci.yml` builds on linux-x86_64, linux-arm64 and
macos-arm64, plus two jobs the matrix does not cover: a Metal build (off
by default, so otherwise never compiled anywhere) and a guards job that
checks the unimplemented accelerator flags still refuse with a message
and that every source file carries its SPDX header.

It exists because of what the first Linux run found. Both defects — the
Makefile missing `aarch64`, and a peak-RSS check using a BSD-only flag of
a tool debian does not ship — were invisible from macOS and would have
been caught by a single push.

**What it cannot do.** The end-to-end engine checks need a container, and
the smallest is 19 GB, so CI runs only what is independent of one: the
three builds, and the kernels against their PyTorch references — enough
to catch a numerical regression in SiTU, the decay gates, AttnRes or KDA
on a platform nobody develops on. Verified in a Linux arm64 container:
with `uv` present those two checks run and pass; `tests/run.sh` skips the
rest and exits 0.

**The gap is now closed**, by `tools/make_test_container.py`: a valid
1 MB container of deterministic noise, stdlib-only so it needs neither
torch nor the converter's C extension. `tests/run.sh` builds one whenever
no real container is given, which took the model-less run from 4 checks
to 13 at the time — chunked prefill against token-at-a-time, int8 storage
against f32, the SIMD backend against the CPU baseline, the expert cache
against no cache, session-state round-trip, the RAM plan and the
format-version guard all now run on every platform and in CI.

Counts drift as checks are added, so take them from a run rather than
from here. As of 2026-07-29 the suite is **31 checks**; without a container it is
**19 pass / 0 fail / 10 skip**.

Those seven are what still needs real weights: the oracle diff (those
logits belong to actual Kimi-Linear weights), the container round-trip
against the source shards, and anything that drives the CLI with text —
the synthetic container deliberately carries no tokenizer.

## What Windows actually costs (2026-07-29)

"The Windows branches are written" was generous: there is one, in
dot-product detection. So the question was asked properly — cross-compile
with MinGW-w64 and count what breaks:

| translation unit | for Windows |
|---|---|
| `backend.c` `ecache.c` `image.c` `kda.c` `kda_neon.c` | compiles |
| `tokenizer.c` `version.c` `vision.c` `waste.c` `cli/main.c` | compiles |
| `simd_avx2.c` `simd_avx512.c` | compiles with their ISA flags |
| `model.c` `vq.c` | **stop on one undeclared constant** |

Ten of thirteen build unchanged, and the gap is three POSIX calls:

| call | where | Windows equivalent |
|---|---|---|
| `sysconf(_SC_NPROCESSORS_ONLN)` | `threads.h` | `GetSystemInfo().dwNumberOfProcessors` |
| `posix_memalign` | `ecache.c` | `_aligned_malloc` — and `_aligned_free` on the way out, which `free()` cannot do |
| `pread` | `model.c` | `ReadFile` with `OVERLAPPED`, or the cache-bypass open below |

Two of those are only *warnings* under gcc 13, which is the trap: an
implicit declaration compiles and fails at link, so "it builds" would
have been the wrong question. MinGW supplies pthreads through
winpthreads, so the thread pool needs nothing.

That is a small port — three shims plus `FILE_FLAG_NO_BUFFERING` for the
page-cache bypass, which is where the real work is, since the whole
expert-streaming argument rests on it.

## The port (2026-07-29)

The estimate above was right about the three calls and wrong about the
size, because it counted what fails to compile. Two things that compile
cleanly were the actual work:

**`long` is 32 bits on Windows.** LLP64 keeps `int` and `long` at 32 and
widens only pointers, so every file offset in the engine — `pread_all`'s
argument, `waste_tensor.file_off`, the manifest's `off` and `scale_off`
through `js_int`, a bank's `bytes` before it is divided into records —
was a silent 2 GB truncation on a format whose small container is 17 GB.
None of it is a compile error and none of it is visible on a machine
where `long` is 64 bits. They are `int64_t` now.

**The archiver is part of the target.** `ar rcs` on macOS accepts PE
objects and writes a 96-byte archive without a word of complaint; the
link then fails with a page of undefined references naming every public
symbol, which reads like a source problem and is not one. `AR` follows
`CC`.

What the port is: [src/platform.h](../src/platform.h), which holds the
six calls that are not POSIX — positional read, aligned allocation, CPU
count, file size, cache-bypass open — with a Windows implementation and a
one-line POSIX one, so no call site branches. `pread` becomes `ReadFile`
with an `OVERLAPPED` offset, which does not move the shared file pointer
and so keeps the property the expert cache needs. `posix_memalign`
becomes `_aligned_malloc`, whose pointer must not be passed to `free()` —
that is heap corruption rather than a leak, which is why the allocation
and its release are a pair and neither is called directly. MinGW supplies
pthreads through winpthreads, so the thread pool needed nothing.

**CI builds it and then runs it**, in two jobs: cross-compile with
MinGW-w64 on a Linux runner, then execute the artifacts on
`windows-latest` against a synthetic container — the expert records
through the C structs, `waste info` and `waste plan`, and the forward
pass sequential against chunked. The old `windows-portability` tripwire
counted unresolved POSIX calls without linking; the build now subsumes
it, since a fourth dependency fails it outright.

What is still not claimed: MSVC (the sources are GNU C), ARM64 Windows,
and the bypass under load. CI reports whether Windows granted
`FILE_FLAG_NO_BUFFERING` on the runner's filesystem, which is a different
claim from a hit rate measured against a container that does not fit in
RAM.
