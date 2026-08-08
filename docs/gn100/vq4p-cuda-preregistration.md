# GB10 CUDA VQ4P crossover preregistration

Date: 2026-08-07  
Branch: `exp/cuda-vq4p-gb10`  
Vehicle: Kimi-Linear VQ4P on the Acer Veriton GN100 / NVIDIA GB10

This is a bounded qualification and crossover experiment, not a new VQ
format.  The manifest remains the source of truth for stages, vector width,
entries and index bits.  CUDA normalizes only two complete geometries — VQ3R
and VQ4P — and fails closed on every other tuple.  Host-side dispatch selects
separate compiled kernels; there is no branch-heavy generic device kernel.

## Numerical contract

The first gate is the little-endian 4x6-bit-to-3-byte unpack.  It is checked
exhaustively before any numerical or performance result is accepted.  This is
the remaining silent-wrongness trap because VQ3R and VQ4P carry the same three
index bytes per row/vector position.

VQ4P has a smaller exactness surface than VQ3R.  Within one 32-vector LUT
scale block, all gathered values and the accumulator are integers.  Integer
addition is associative here and the bound is 4 x 32 x 127 = 16,256, so the
kernel may reorder or parallel-reduce those additions without changing the
result.  The ordered obligations are only:

1. fold the scale blocks into fp32 in increasing vector-block order;
2. match the CPU reference's fused-versus-separate multiply/add choice; and
3. apply the final fp16 channel scale identically.

The preregistration inspection found that GCC 13 at the qualified `-O2`
profile emits scalar `FMADD` for obligation 2.  CUDA therefore uses an
explicit correctly-rounded FMA for each ordered block fold.

The strict comparison is CPU VQ4P against CUDA VQ4P on the same container.
VQ3R and VQ4P encode different approximations and are not expected to agree
with each other.

## Coherent-LUT arm and stop rule

The first CUDA arm uses the CPU's existing fp32 LUT build and VQ4P int8
quantization, then lets the GPU read the int8 table and block scales directly
from coherent host memory.  It makes no duplicate device LUT and introduces
no nominal H2D payload.  This is a candidate final GB10 configuration, not
merely scaffolding.

GPU-side LUT build/quantization is attempted only if the matched engine trace
shows CPU LUT build time greater than **10% of VQ apply time**.  At or below
10%, the extra kernel and its second exactness surface are rejected as having
insufficient recoverable work.  Above 10%, it becomes a separately gated
follow-on; the fp32 LUT, int8 LUT bytes and block scales must each match the
CPU reference before an end-to-end run.

## Gates and report shape

1. exhaustive packed-index round trip;
2. real-record gate/up/down output equality against CPU VQ4P;
3. short greedy causal capture: exact logits, routes and tokens, zero CUDA
   fallback, and exact semantic counters;
4. one matched performance crossover, without policy tuning.

The headline table reports scalar CPU, NEON CPU and CUDA on the same ARM
machine, both per apply shape and at engine level.  The success condition is
not a large token-rate win: it is a completed strict VQ4P CUDA contract and a
measured CPU/NEON/CUDA crossover.  Longer soaking and K3-scale conversion are
out of scope until the bounded run passes.

## Outcome

The bounded run passed. The CPU-LUT CUDA arm spent 1.205485 seconds building
LUTs against 0.194508 seconds applying them: **619.76%**, so the registered
10% trigger fired. The GPU builder subsequently matched the reference fp32
LUT, int8 bytes and scales exactly. Its 16-token median was 9.137839 tok/s,
versus 3.771248 for CPU-LUT CUDA, 2.293125 for NEON and 1.470860 for scalar.

All 17 compared causal states had byte-identical logits, routes and tokens,
and the selected mode reported zero fallback. VQ4P mode 2/group 1 is therefore
the selected configuration for this experiment; it remains on the `exp/*`
branch and is not a K3-profile promotion. See
[../GPU_VQ4P_GB10.md](../GPU_VQ4P_GB10.md) for the compact report.
