# GB10 CUDA provider qualification

Status: review recipe and short real-model smoke complete; full registered
qualification pending.

This is the release gate for the source-separated CUDA provider, not a claim
that CUDA ships in WARP. The engine owns the forward pass, routing, activation,
router weighting, and ordered accumulation. The provider owns only the
allowlisted one-token Q4G projections and VQ3R gathers. K2 and K3 are qualified
separately; an unlisted model, VQ4P, VQ2R, or a borrowed expert record whose
`record_scheme` is not VQ3R must be rejected before its indices are read.

## Registered gates

These gates are fixed before the GB10 run:

1. Build the engine and provider from recorded revisions. `test_backend` and
   the provider's host-only policy test pass. The device build retains
   `-fmad=false` and host `-ffp-contract=off`.
2. Device 0 reports coherent pageable-memory access with host page tables. A
   two-token combined-mode preflight completes with no backend error. Failure
   is terminal for that context; there is no mid-token CPU fallback.
3. Run three interleaved repeats of `cpu`, `vq`, `matvec`, and `combined` for
   each model, using the same 64-token greedy capture and the same recorded
   runtime controls. The runner records resolved memory/cache sizes and wraps
   the provider with semantic claim/call counters.
4. VQ-only is strict: every fp32 logit byte and the full route trace match the
   CPU capture; generated ids and ordered top-10 ids therefore match too. A
   trace must contain exactly one row for every expected position and every MoE
   layer, with no duplicates; two identically truncated traces fail.
5. MATVEC-only and combined are bounded: no non-finite logits; maximum absolute
   logit difference at most `1.0e-4`; generated ids, ordered top-10 ids, and
   ordered router selections identical at every step. Relative L2 is reported,
   not silently promoted to a new gate. The absolute bound is more than twice
   the previous observed K2 worst case (`4.9114e-5`).
6. Counter semantics match the selected mode: every claimed MATVEC tensor runs
   once per decode step; VQ begins once per MoE layer and applies gate/up and
   down once per selected expert. VQ-only makes no MATVEC calls, MATVEC-only
   makes no VQ calls, no callback reports failure, and every mode records zero
   provider execution calls during the CPU-owned prompt prefill.
7. Throughput is observational: report the median decode tok/s of the three
   repeats for all four modes. No percentage-advantage threshold is applied in
   this review run.

The comparator exits nonzero for every failed gate. Passing one model does not
qualify the other, and a release for which this recipe was not run carries no
current qualification claim.

## Build on the GB10 host

From the review checkout:

```bash
make libwaste.a test_backend
gate_dir=$(mktemp -d)
python3 tools/make_test_container.py "$gate_dir/tiny.waste"
./test_backend "$gate_dir/tiny.waste"
make -C backends/cuda test
make -C backends/cuda
cc -O2 -std=gnu11 -Wall -Wextra -Werror \
  -Isrc -Ibackends/cuda -c tools/cuda_qualify.c \
  -o tools/cuda_qualify.o
nvcc -o cuda-qualify tools/cuda_qualify.o libwaste.a \
  backends/cuda/libwaste_cuda_backend.a -lcudart -lpthread -lm
```

`cuda-qualify` takes:

```text
cuda-qualify MODEL MODE OUTDIR IDS N_GENERATE [RAM_BYTES [THREADS [CPU_LIST]]]
```

`MODE` is `cpu`, `vq`, `matvec`, or `combined`. `IDS` is an already-tokenized
comma-separated prompt. Each fresh output directory contains `run.json`, raw
little-endian `logits.f32`, `routes.txt`, and `steps.tsv`. Step zero is prompt
prefill; later frames are one-token decode, which is where the provider seam is
eligible. The runner creates the directory mode `0700` and refuses to overwrite
an existing directory. CPU mode opens a zero-capability metadata backend: all
math stays on CPU, while the public backend model view supplies `first_dense`
so route completeness can be checked without private engine headers.

Use an explicit, identical profile for every arm. This is the strict numerical
profile; change it only by recording a new matched profile, not for one arm:

```bash
profile=(env WASTE_Q8=1 WASTE_SDOT=0 WASTE_I8MM=0 \
  WASTE_CCR_LAMBDA=0 WASTE_XPAR=0 WASTE_LOOKAHEAD=6 \
  WASTE_IO_THREADS=2 WASTE_IO_DEPTH=2 WASTE_DIRECT=1)
```

Tokenize the repo-owned public fixture `studio-materials-v1` independently for
each model:

> A studio has one sheet of paper, one light, and ten minutes. Propose a small
> visual experiment.

```bash
public_prompt='A studio has one sheet of paper, one light, and ten minutes. Propose a small visual experiment.'
ids=$(./waste tokenize "$MODEL" "$public_prompt" --json | \
  python3 -c 'import json,sys; print(",".join(map(str,json.load(sys.stdin)["ids"])))')

run_root="private-evidence/gb10-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_root"
RAM_BYTES=0
THREADS=0
CPUS=-
"${profile[@]}" ./cuda-qualify "$MODEL" combined \
  "$run_root/preflight-combined" "$ids" 2 "$RAM_BYTES" "$THREADS" "$CPUS"
for repeat in 1 2 3; do
  case "$repeat" in
    1) modes=(cpu vq matvec combined) ;;
    2) modes=(matvec combined cpu vq) ;;
    3) modes=(vq cpu combined matvec) ;;
  esac
  for mode in "${modes[@]}"; do
    "${profile[@]}" ./cuda-qualify "$MODEL" "$mode" \
      "$run_root/r${repeat}-${mode}" "$ids" 64 "$RAM_BYTES" "$THREADS" "$CPUS"
  done
  python3 tools/compare_backend_runs.py \
    "$run_root/r${repeat}-cpu" "$run_root/r${repeat}-vq" \
    --contract exact-vq
  python3 tools/compare_backend_runs.py \
    "$run_root/r${repeat}-cpu" "$run_root/r${repeat}-matvec" \
    --contract dense --max-abs 1e-4
  python3 tools/compare_backend_runs.py \
    "$run_root/r${repeat}-cpu" "$run_root/r${repeat}-combined" \
    --contract dense --max-abs 1e-4
done
```

The runner opens a fresh context for each arm, including the two-token combined
preflight. Keep that directory with the private evidence rather than treating a
failed or partial warm-up as a measurement. The three fixed arm orders spread
machine drift instead of always charging the final mode for it.

## Evidence and publication

Raw captures are private by default. They contain exact prompt and generated
token ids, per-step logits, and routes; the absence of a model path or host name
does not make that usage data public. Do not attach or commit the raw capture
directories.

For the repo-owned fixture above only, the comparator can emit a redacted report
containing metrics and SHA-256 commitments instead of token arrays, logits, or
routes:

```bash
python3 tools/compare_backend_runs.py \
  "$run_root/r1-cpu" "$run_root/r1-combined" \
  --contract dense --max-abs 1e-4 \
  --public-fixture studio-materials-v1 \
  --public-report "$run_root/r1-combined-public.json"
```

The fixture label is an explicit assertion by the operator, not an automatic
privacy proof. Inspect the redacted JSON before publication and retain the raw
directories privately so their hashes remain checkable.

## Review-branch smoke

On 2026-08-27, the Layer 1/2 head `2991c43` was built on an NVIDIA GB10
(`sm_121`) with CUDA 13.0 and driver 580.159.03. One repo-owned fixed-ID
fixture ran a combined preflight followed by one `cpu`, `vq`, `matvec`, and
`combined` capture per model. Each capture included prompt prefill and two
one-token decode steps.

| model | VQ-only | MATVEC max abs | combined max abs | complete route rows | exact callback coverage |
|---|---|---:|---:|---:|---|
| Kimi K2 | byte-exact, 491,520 logits | `1.2398e-5` | `1.2398e-5` | 420 | yes |
| Kimi K3 | byte-exact, 491,520 logits | `1.0014e-5` | `1.0014e-5` | 644 | yes |

Both models kept generated ids, ordered top-10 ids, and ordered routes
identical. Every provider execution counter was zero after prefill, then
matched the registered per-decode formulas; callback failures were zero. Raw
captures and host provenance are retained privately outside the checkout.

This is a build, lifecycle, coverage, and numerical-contract smoke—not a
release qualification or a throughput measurement. The three-repeat,
64-token tables below remain pending, and the short-run tok/s values are
deliberately not promoted.

## Results

| model | engine revision | provider revision | VQ exact (3/3) | MATVEC bounded (3/3) | combined bounded (3/3) | counter semantics | result |
|---|---|---|---:|---:|---:|---|---|
| Kimi K2 | pending | pending | pending | pending | pending | pending | pending |
| Kimi K3 | pending | pending | pending | pending | pending | pending | pending |

| model | CPU median tok/s | VQ-only median tok/s | MATVEC-only median tok/s | combined median tok/s | qualified date |
|---|---:|---:|---:|---:|---|
| Kimi K2 | pending | pending | pending | pending | pending |
| Kimi K3 | pending | pending | pending | pending | pending |
