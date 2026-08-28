// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 SQLite Cloud, Inc.
/* Standalone correctness and timing gate for waste_cuda_q4_matvec2.
 *
 * This intentionally links the production src/cuda.cu implementation rather
 * than carrying a benchmark copy of q4_fast2.  It compares one fused T=2 call
 * bit-for-bit with two calls through the already-qualified mode-1 entry point,
 * then brackets fused timing with sequential timing on deterministic Q4G data.
 *
 *   nvcc -O3 -std=c++17 -arch=native -fmad=false \
 *     -Xcompiler=-ffp-contract=off -Xcompiler=-pthread \
 *     -DWASTE_ENABLE_CUDA=1 -DWASTE_CUDA_Q4_MATVEC2_TEST=1 \
 *     -I src -o cuda_q4_matvec2_test \
 *     tools/cuda_q4_matvec2_test.cu src/cuda.cu
 *   ./cuda_q4_matvec2_test
 *   ./cuda_q4_matvec2_test OUT IN [ITERATIONS] [WARMUP]
 *
 * With no shape arguments the gate covers every unique Q4 projection shape
 * in released GLM-5.3, plus the complete measured-positive allowlist.  A
 * single-shape invocation also reports bracketed timing.
 */

#include "model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

extern "C" int waste_cuda_q4_matvec(waste_model *, float *,
                                      const waste_tensor *, const float *,
                                      int, int, int);
extern "C" int waste_cuda_q4_matvec2_eligible(
    const waste_tensor *, int, int, int);
extern "C" int waste_cuda_q4_matvec2(
    waste_model *, float *, float *, const waste_tensor *,
    const float *, const float *, int, int, int);
extern "C" int waste_cuda_q4_matvec2_test_only(
    waste_model *, float *, float *, const waste_tensor *,
    const float *, const float *, int, int, int);
extern "C" void waste_cuda_kda_free(waste_model *);

namespace {

constexpr int kGroup = 128;

static int positive_arg(const char *text, const char *name)
{
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!text[0] || !end || *end || value < 1 ||
        value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "invalid %s: %s\n", name, text);
        std::exit(2);
    }
    return (int)value;
}

static uint32_t lcg(uint32_t &state)
{
    state = state * 1664525u + 1013904223u;
    return state;
}

static void deterministic_fixture(std::vector<int8_t> &weights,
                                  std::vector<uint16_t> &scales,
                                  std::vector<float> &x0,
                                  std::vector<float> &x1)
{
    uint32_t state = 0x4d545032u; /* "MTP2" */
    uint8_t *packed = reinterpret_cast<uint8_t *>(weights.data());
    for (size_t i = 0; i < weights.size(); i++)
        packed[i] = (uint8_t)(lcg(state) >> 24);

    /* Exact, finite binary16 powers of two: 2^-8 through 2^-4. */
    static constexpr uint16_t kHalfScales[] = {
        0x1c00u, 0x2000u, 0x2400u, 0x2800u, 0x2c00u
    };
    for (size_t i = 0; i < scales.size(); i++)
        scales[i] = kHalfScales[(lcg(state) >> 27) %
                                (sizeof kHalfScales / sizeof kHalfScales[0])];

    /* Values are exact multiples of 2^-15, avoiding a host-library-dependent
     * fixture before CUDA sees the two activation rows. */
    for (size_t i = 0; i < x0.size(); i++) {
        const int value = (int)((lcg(state) >> 20) & 0x0fffu) - 2048;
        x0[i] = (float)value * (1.0f / 32768.0f);
    }
    for (size_t i = 0; i < x1.size(); i++) {
        const int value = (int)((lcg(state) >> 20) & 0x0fffu) - 2048;
        x1[i] = (float)value * (1.0f / 32768.0f);
    }
}

static void scalar_pair(waste_model *model, const waste_tensor *tensor,
                        const float *x0, const float *x1,
                        float *y0, float *y1, int out, int in)
{
    if (waste_cuda_q4_matvec(model, y0, tensor, x0, out, in, 1) ||
        waste_cuda_q4_matvec(model, y1, tensor, x1, out, in, 1)) {
        std::fprintf(stderr, "mode-1 sequential projection failed\n");
        std::exit(1);
    }
}

static void fused_pair(waste_model *model, const waste_tensor *tensor,
                       const float *x0, const float *x1,
                       float *y0, float *y1, int out, int in,
                       bool test_only)
{
    const int rc = test_only
        ? waste_cuda_q4_matvec2_test_only(
              model, y0, y1, tensor, x0, x1, out, in, 1)
        : waste_cuda_q4_matvec2(
              model, y0, y1, tensor, x0, x1, out, in, 1);
    if (rc) {
        std::fprintf(stderr, "mode-1 fused2 projection failed\n");
        std::exit(1);
    }
}

template <typename Fn>
static double milliseconds_per_pair(int iterations, Fn &&fn)
{
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) fn();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count() /
           (double)iterations;
}

static int compare_bits(const char *row,
                        const std::vector<float> &reference,
                        const std::vector<float> &fused)
{
    size_t mismatches = 0, first = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < reference.size(); i++) {
        uint32_t a = 0, b = 0;
        std::memcpy(&a, &reference[i], sizeof a);
        std::memcpy(&b, &fused[i], sizeof b);
        if (a != b) {
            if (!mismatches) first = i;
            mismatches++;
        }
        max_abs = std::max(max_abs, std::fabs(reference[i] - fused[i]));
    }
    std::printf("correctness=fused2-vs-sequential row=%s elements=%zu "
                "byte_exact=%d "
                "mismatches=%zu max_abs=%.9g\n",
                row, reference.size(), mismatches == 0, mismatches, max_abs);
    if (mismatches) {
        uint32_t a = 0, b = 0;
        std::memcpy(&a, &reference[first], sizeof a);
        std::memcpy(&b, &fused[first], sizeof b);
        std::fprintf(stderr,
                     "first mismatch in %s at element %zu: "
                     "sequential=0x%08x fused2=0x%08x\n",
                     row, first, a, b);
        return -1;
    }
    return 0;
}

struct ShapeGate {
    int out, in;
    bool eligible;
    const char *role;
};

/* Unique Q4 matrix geometries reached by the released GLM-5.3 target and
 * recurrent MTP layer.  kv_b is consumed row-wise today, but including its
 * stored geometry makes this a complete arithmetic gate for a future batched
 * implementation rather than an accidental subset of the checkpoint. */
static constexpr ShapeGate kGlm53Shapes[] = {
    { 8192,  4096, true,  "KDA q/k/v" },
    {  128,  4096, false, "KDA f_a/g_a" },
    { 8192,   128, true,  "KDA f_b/g_b" },
    {   64,  4096, false, "KDA beta" },
    {  288,  4096, false, "MoE router" },
    { 4096,  8192, false, "KDA o and MTP eh" },
    { 1536,  4096, true,  "MLA q_a" },
    {16384,  1536, true,  "MLA q_b" },
    {  512,  4096, false, "MLA kv_a" },
    {32768,   512, true,  "MLA kv_b row store" },
    { 4096, 16384, true,  "MLA o" },
    {12288,  4096, true,  "dense gate/up" },
    { 4096, 12288, false, "dense down" },
    { 2048,  4096, true,  "shared gate/up" },
    { 4096,  2048, true,  "shared down" },
};

/* Measured-positive geometries not present as standalone projections in the
 * released GLM-5.3 graph.  They remain in the exact allowlist for reuse. */
static constexpr ShapeGate kMeasuredPositiveExtras[] = {
    { 4096, 1536, true, "measured-positive reuse" },
    { 4096, 4096, true, "measured-positive reuse" },
    {12288, 1536, true, "measured-positive reuse" },
};

static int run_shape(const ShapeGate &shape, int iterations, int warmup)
{
    const int out = shape.out, in = shape.in;
    const int groups = (in + kGroup - 1) / kGroup;
    const size_t rowbytes = ((size_t)in + 1) / 2;
    if ((size_t)out > std::numeric_limits<size_t>::max() / rowbytes ||
        (size_t)out > std::numeric_limits<size_t>::max() /
                      ((size_t)groups * sizeof(uint16_t))) {
        std::fprintf(stderr, "fixture size overflow for %dx%d\n", out, in);
        return 2;
    }

    std::vector<int8_t> weights((size_t)out * rowbytes);
    std::vector<uint16_t> scales((size_t)out * groups);
    std::vector<float> x0((size_t)in), x1((size_t)in);
    std::vector<float> sequential0((size_t)out), sequential1((size_t)out);
    std::vector<float> fused0((size_t)out), fused1((size_t)out);
    deterministic_fixture(weights, scales, x0, x1);

    waste_tensor tensor{};
    tensor.q = weights.data();
    tensor.qs = scales.data();
    tensor.group = kGroup;
    tensor.bits = 4;
    tensor.rowbytes = rowbytes;
    tensor.shape[0] = out;
    tensor.shape[1] = in;
    tensor.ndim = 2;
    tensor.n = (size_t)out * in;

    waste_model model{};
    model.cfg.hidden = std::max(out, in);
    const bool eligible = waste_cuda_q4_matvec2_eligible(
        &tensor, out, in, 1) != 0;
    std::printf("shape=%dx%d role=%s selector=%s expected=%s\n",
                out, in, shape.role, eligible ? "fused2" : "fallback",
                shape.eligible ? "fused2" : "fallback");
    if (eligible != shape.eligible) {
        std::fprintf(stderr, "selector mismatch for %dx%d\n", out, in);
        return 1;
    }
    if (waste_cuda_q4_matvec2_eligible(&tensor, out, in, 2)) {
        std::fprintf(stderr, "selector unexpectedly accepted mode 2\n");
        return 1;
    }
    if (waste_cuda_q4_matvec2(
            &model, fused0.data(), fused1.data(), &tensor,
            x0.data(), x1.data(), out, in, 2) != -1) {
        std::fprintf(stderr, "fused2 unexpectedly accepted mode 2\n");
        waste_cuda_kda_free(&model);
        return 1;
    }
    if (!eligible && waste_cuda_q4_matvec2(
            &model, fused0.data(), fused1.data(), &tensor,
            x0.data(), x1.data(), out, in, 1) != -1) {
        std::fprintf(stderr,
                     "production fused2 ignored fallback for %dx%d\n",
                     out, in);
        waste_cuda_kda_free(&model);
        return 1;
    }

    scalar_pair(&model, &tensor, x0.data(), x1.data(),
                sequential0.data(), sequential1.data(), out, in);
    fused_pair(&model, &tensor, x0.data(), x1.data(),
               fused0.data(), fused1.data(), out, in, !eligible);
    if (compare_bits("0", sequential0, fused0) ||
        compare_bits("1", sequential1, fused1)) {
        waste_cuda_kda_free(&model);
        return 1;
    }

    if (iterations > 0) {
        for (int i = 0; i < warmup; i++) {
            scalar_pair(&model, &tensor, x0.data(), x1.data(),
                        sequential0.data(), sequential1.data(), out, in);
            fused_pair(&model, &tensor, x0.data(), x1.data(),
                       fused0.data(), fused1.data(), out, in, !eligible);
        }
        const auto sequential_call = [&] {
            scalar_pair(&model, &tensor, x0.data(), x1.data(),
                        sequential0.data(), sequential1.data(), out, in);
        };
        const auto fused_call = [&] {
            fused_pair(&model, &tensor, x0.data(), x1.data(),
                       fused0.data(), fused1.data(), out, in, !eligible);
        };
        const double sequential_before =
            milliseconds_per_pair(iterations, sequential_call);
        const double fused_ms = milliseconds_per_pair(iterations, fused_call);
        const double sequential_after =
            milliseconds_per_pair(iterations, sequential_call);
        const double sequential_ms =
            0.5 * (sequential_before + sequential_after);
        const double speedup = sequential_ms / fused_ms;
        std::printf("group=%d iterations=%d warmup=%d\n",
                    kGroup, iterations, warmup);
        std::printf("path=sequential2-before wall_ms_per_pair=%.6f\n",
                    sequential_before);
        std::printf("path=%s wall_ms_per_pair=%.6f\n",
                    eligible ? "fused2" : "fused2-test-only", fused_ms);
        std::printf("path=sequential2-after wall_ms_per_pair=%.6f\n",
                    sequential_after);
        std::printf("comparison=bracketed speedup=%.4f "
                    "saved_ms_per_pair=%.6f\n",
                    speedup, sequential_ms - fused_ms);
    }

    waste_cuda_kda_free(&model);
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc == 1) {
        int passed = 0;
        for (const ShapeGate &shape : kGlm53Shapes) {
            if (run_shape(shape, 0, 0)) return 1;
            passed++;
        }
        for (const ShapeGate &shape : kMeasuredPositiveExtras) {
            if (run_shape(shape, 0, 0)) return 1;
            passed++;
        }
        std::printf("suite=glm53-q4x2 shapes=%d passed=%d failed=0\n",
                    passed, passed);
        return 0;
    }
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr,
                     "usage: %s [OUT IN [ITERATIONS] [WARMUP]]\n",
                     argv[0]);
        return 2;
    }
    const int out = positive_arg(argv[1], "OUT");
    const int in = positive_arg(argv[2], "IN");
    const int iterations = argc > 3
                         ? positive_arg(argv[3], "ITERATIONS") : 40;
    const int warmup = argc > 4 ? positive_arg(argv[4], "WARMUP") : 5;
    ShapeGate requested = { out, in, false, "requested" };

    /* CLI expectations come from the production predicate itself; the full
     * no-argument suite above is what pins that predicate to registered
     * release expectations. */
    int8_t q = 0;
    uint16_t qs = 0;
    waste_tensor t{};
    t.q = &q; t.qs = &qs; t.group = kGroup; t.bits = 4;
    t.rowbytes = ((size_t)in + 1) / 2;
    t.shape[0] = out; t.shape[1] = in; t.ndim = 2;
    requested.eligible = waste_cuda_q4_matvec2_eligible(
        &t, out, in, 1) != 0;
    return run_shape(requested, iterations, warmup);
}
