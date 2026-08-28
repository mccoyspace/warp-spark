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
 *     -DWASTE_ENABLE_CUDA=1 -I src -o cuda_q4_matvec2_test \
 *     tools/cuda_q4_matvec2_test.cu src/cuda.cu
 *   ./cuda_q4_matvec2_test [OUT] [IN] [ITERATIONS] [WARMUP]
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
extern "C" int waste_cuda_q4_matvec2(waste_model *, float *,
                                       const waste_tensor *, const float *,
                                       int, int, int);
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
                                  std::vector<float> &x)
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
    for (size_t i = 0; i < x.size(); i++) {
        const int value = (int)((lcg(state) >> 20) & 0x0fffu) - 2048;
        x[i] = (float)value * (1.0f / 32768.0f);
    }
}

static void scalar_pair(waste_model *model, const waste_tensor *tensor,
                        const float *x, float *y, int out, int in)
{
    if (waste_cuda_q4_matvec(model, y, tensor, x, out, in, 1) ||
        waste_cuda_q4_matvec(model, y + out, tensor, x + in, out, in, 1)) {
        std::fprintf(stderr, "mode-1 sequential projection failed\n");
        std::exit(1);
    }
}

static void fused_pair(waste_model *model, const waste_tensor *tensor,
                       const float *x, float *y, int out, int in)
{
    if (waste_cuda_q4_matvec2(model, y, tensor, x, out, in, 1)) {
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

static int compare_bits(const std::vector<float> &reference,
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
    std::printf("correctness=fused2-vs-sequential elements=%zu byte_exact=%d "
                "mismatches=%zu max_abs=%.9g\n",
                reference.size(), mismatches == 0, mismatches, max_abs);
    if (mismatches) {
        uint32_t a = 0, b = 0;
        std::memcpy(&a, &reference[first], sizeof a);
        std::memcpy(&b, &fused[first], sizeof b);
        std::fprintf(stderr,
                     "first mismatch at element %zu: sequential=0x%08x "
                     "fused2=0x%08x\n", first, a, b);
        return -1;
    }
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc > 5) {
        std::fprintf(stderr,
                     "usage: %s [OUT] [IN] [ITERATIONS] [WARMUP]\n",
                     argv[0]);
        return 2;
    }
    const int out = argc > 1 ? positive_arg(argv[1], "OUT") : 8192;
    const int in = argc > 2 ? positive_arg(argv[2], "IN") : 4096;
    const int iterations = argc > 3
                         ? positive_arg(argv[3], "ITERATIONS") : 40;
    const int warmup = argc > 4 ? positive_arg(argv[4], "WARMUP") : 5;
    const int groups = (in + kGroup - 1) / kGroup;
    const size_t rowbytes = ((size_t)in + 1) / 2;
    if ((size_t)out > std::numeric_limits<size_t>::max() / rowbytes ||
        (size_t)out > std::numeric_limits<size_t>::max() /
                      ((size_t)groups * sizeof(uint16_t))) {
        std::fprintf(stderr, "fixture size overflow\n");
        return 2;
    }

    std::vector<int8_t> weights((size_t)out * rowbytes);
    std::vector<uint16_t> scales((size_t)out * groups);
    std::vector<float> x((size_t)2 * in);
    std::vector<float> sequential((size_t)2 * out);
    std::vector<float> fused((size_t)2 * out);
    deterministic_fixture(weights, scales, x);

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

    /* The exported verifier primitive owns this fail-closed selector gate. */
    if (waste_cuda_q4_matvec2(&model, fused.data(), &tensor, x.data(),
                              out, in, 2) != -1) {
        std::fprintf(stderr, "fused2 unexpectedly accepted mode 2\n");
        waste_cuda_kda_free(&model);
        return 1;
    }

    scalar_pair(&model, &tensor, x.data(), sequential.data(), out, in);
    fused_pair(&model, &tensor, x.data(), fused.data(), out, in);
    if (compare_bits(sequential, fused)) {
        waste_cuda_kda_free(&model);
        return 1;
    }

    for (int i = 0; i < warmup; i++) {
        scalar_pair(&model, &tensor, x.data(), sequential.data(), out, in);
        fused_pair(&model, &tensor, x.data(), fused.data(), out, in);
    }

    const auto sequential_call = [&] {
        scalar_pair(&model, &tensor, x.data(), sequential.data(), out, in);
    };
    const auto fused_call = [&] {
        fused_pair(&model, &tensor, x.data(), fused.data(), out, in);
    };
    const double sequential_before =
        milliseconds_per_pair(iterations, sequential_call);
    const double fused_ms = milliseconds_per_pair(iterations, fused_call);
    const double sequential_after =
        milliseconds_per_pair(iterations, sequential_call);
    const double sequential_ms = 0.5 * (sequential_before + sequential_after);
    const double speedup = sequential_ms / fused_ms;

    std::printf("shape=%dx%d group=%d iterations=%d warmup=%d\n",
                out, in, kGroup, iterations, warmup);
    std::printf("path=sequential2-before wall_ms_per_pair=%.6f\n",
                sequential_before);
    std::printf("path=fused2 wall_ms_per_pair=%.6f\n", fused_ms);
    std::printf("path=sequential2-after wall_ms_per_pair=%.6f\n",
                sequential_after);
    std::printf("comparison=bracketed speedup=%.4f saved_ms_per_pair=%.6f\n",
                speedup, sequential_ms - fused_ms);

    waste_cuda_kda_free(&model);
    return 0;
}
