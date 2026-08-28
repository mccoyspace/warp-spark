// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 SQLite Cloud, Inc.
/* Standalone exactness and timing gate for the two-row VQ3R verifier path.
 *
 * It links the production CUDA implementation.  The reference is two
 * ordinary mode-2 rows, each with eight complete gate/up/down expert calls.
 * The candidate builds both row LUT pairs, hands all sixteen gate/up tasks to
 * one group, applies the same host SwiGLU, then reuses the existing sixteen-
 * slot grouped-down path.
 *
 *   nvcc -O3 -std=c++17 -arch=native -fmad=false \
 *     -Xcompiler=-ffp-contract=off -Xcompiler=-pthread \
 *     -DWASTE_ENABLE_CUDA=1 -I src -o cuda_vq_pair2_test \
 *     tools/cuda_vq_pair2_test.cu src/cuda.cu
 *   ./cuda_vq_pair2_test [ITERATIONS] [WARMUP]
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

extern "C" int waste_cuda_vq_init(waste_model *);
extern "C" int waste_cuda_vq_prepare_pair(
    waste_model *, int, const float *, const float *, const float *, int, int);
extern "C" int waste_cuda_vq_apply_pair(
    waste_model *, float *, float *, const uint8_t *, const uint8_t *,
    const uint16_t *, int, int);
extern "C" int waste_cuda_vq_apply_down(
    waste_model *, int, float *, const uint8_t *, const uint16_t *,
    const float *, const float *, int, int, int);
extern "C" int waste_cuda_vq_prepare_pair2(
    waste_model *, const float *, const float *, int, int);
extern "C" int waste_cuda_vq_group_pair2_enqueue(
    waste_model *, int, int, const uint8_t *, const uint8_t *,
    const uint16_t *, int, int);
extern "C" int waste_cuda_vq_group_pair_enqueue(
    waste_model *, int, const uint8_t *, const uint8_t *,
    const uint16_t *, int, int);
extern "C" int waste_cuda_vq_group_pair_finish(
    waste_model *, int, const float **);
extern "C" int waste_cuda_vq_group_down_enqueue(
    waste_model *, int, const uint8_t *, const uint16_t *, const float *,
    int, int, int);
extern "C" int waste_cuda_vq_group_down_finish(
    waste_model *, int, const float **);
extern "C" int waste_cuda_vq_group_drain(waste_model *);
extern "C" void waste_cuda_kda_free(waste_model *);

namespace {

constexpr int kStages = 3;
constexpr int kVec = 8;
constexpr int kEntries = 256;
constexpr int kBlock = 64;
constexpr int kRows = 2;
constexpr int kExpertsPerRow = 8;
constexpr int kTasks = kRows * kExpertsPerRow;
constexpr int kLat = 4096;
constexpr int kInter = 2048;
constexpr int kBooks = 9;

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

struct Fixture {
    size_t pair_index_bytes;
    size_t down_index_bytes;
    size_t scale_values;
    std::vector<float> books;
    std::vector<float> x;
    std::vector<uint8_t> gate_idx;
    std::vector<uint8_t> up_idx;
    std::vector<uint8_t> down_idx;
    std::vector<uint16_t> scale;

    Fixture()
        : pair_index_bytes((size_t)kInter * (kLat / kVec) * kStages),
          down_index_bytes((size_t)kLat * (kInter / kVec) * kStages),
          scale_values((size_t)2 * kInter + kLat),
          books((size_t)kBooks * kVec * kEntries),
          x((size_t)kRows * kLat),
          gate_idx((size_t)kTasks * pair_index_bytes),
          up_idx((size_t)kTasks * pair_index_bytes),
          down_idx((size_t)kTasks * down_index_bytes),
          scale((size_t)kTasks * scale_values)
    {
        uint32_t state = 0x56513232u; /* "VQ22" */
        for (float &v : books) {
            const int q = (int)((lcg(state) >> 20) & 0x7ffu) - 1024;
            v = (float)q * (1.0f / 16384.0f);
        }
        for (float &v : x) {
            const int q = (int)((lcg(state) >> 20) & 0xfffu) - 2048;
            v = (float)q * (1.0f / 32768.0f);
        }
        for (uint8_t &v : gate_idx) v = (uint8_t)(lcg(state) >> 24);
        for (uint8_t &v : up_idx) v = (uint8_t)(lcg(state) >> 24);
        for (uint8_t &v : down_idx) v = (uint8_t)(lcg(state) >> 24);
        static constexpr uint16_t half_scales[] = {
            0x2800u, 0x2a00u, 0x2c00u, 0x2e00u
        };
        for (uint16_t &v : scale)
            v = half_scales[(lcg(state) >> 30) & 3u];
    }

    const float *row_x(int row) const
    {
        return x.data() + (size_t)row * kLat;
    }
    const uint8_t *gate(int task) const
    {
        return gate_idx.data() + (size_t)task * pair_index_bytes;
    }
    const uint8_t *up(int task) const
    {
        return up_idx.data() + (size_t)task * pair_index_bytes;
    }
    const uint8_t *down(int task) const
    {
        return down_idx.data() + (size_t)task * down_index_bytes;
    }
    const uint16_t *scales(int task) const
    {
        return scale.data() + (size_t)task * scale_values;
    }
};

struct Work {
    std::vector<float> pair;
    std::vector<float> act;
    std::vector<float> down;

    Work()
        : pair((size_t)kTasks * 2 * kInter),
          act((size_t)kTasks * kInter),
          down((size_t)kTasks * kLat)
    {}
    float *pair_at(int task)
    {
        return pair.data() + (size_t)task * 2 * kInter;
    }
    float *act_at(int task)
    {
        return act.data() + (size_t)task * kInter;
    }
    float *down_at(int task)
    {
        return down.data() + (size_t)task * kLat;
    }
};

static void configure(waste_model &model, const Fixture &fixture)
{
    model = waste_model{};
    model.cfg.hidden = kLat;
    model.cfg.moe_inter = kInter;
    model.cfg.n_shared = 1;
    model.codebooksT = const_cast<float *>(fixture.books.data());
    model.n_books = kBooks;
    model.stages = kStages;
    model.vec_dim = kVec;
    model.cb_entries = kEntries;
    model.index_block = kBlock;
    model.index_bits = 8;
}

static void activate(const float *pair, float *out)
{
    const float *up = pair + kInter;
    for (int i = 0; i < kInter; i++) {
        float gate = pair[i], u = up[i];
        if (gate > 10.0f) gate = 10.0f;
        if (u > 10.0f) u = 10.0f;
        else if (u < -10.0f) u = -10.0f;
        out[i] = gate / (1.0f + expf(-gate)) * u;
    }
}

static int ordinary_run(waste_model &model, const Fixture &fixture,
                        Work &work)
{
    for (int row = 0; row < kRows; row++) {
        if (waste_cuda_vq_prepare_pair(
                &model, 2, fixture.row_x(row), nullptr, nullptr, 0, kLat))
            return -1;
        for (int expert = 0; expert < kExpertsPerRow; expert++) {
            const int task = row * kExpertsPerRow + expert;
            float *pair = work.pair_at(task);
            if (waste_cuda_vq_apply_pair(
                    &model, pair, pair + kInter,
                    fixture.gate(task), fixture.up(task),
                    fixture.scales(task), kInter, kLat))
                return -1;
            activate(pair, work.act_at(task));
            if (waste_cuda_vq_apply_down(
                    &model, 2, work.down_at(task), fixture.down(task),
                    fixture.scales(task) + 2 * kInter,
                    work.act_at(task), nullptr, 2 * kStages,
                    kLat, kInter))
                return -1;
        }
    }
    return 0;
}

static int pair2_run(waste_model &model, const Fixture &fixture,
                     Work &work, bool capture)
{
    if (waste_cuda_vq_prepare_pair2(
            &model, fixture.row_x(0), fixture.row_x(1), 0, kLat))
        return -1;
    for (int task = 0; task < kTasks; task++) {
        const int row = task / kExpertsPerRow;
        if (waste_cuda_vq_group_pair2_enqueue(
                &model, task, row, fixture.gate(task), fixture.up(task),
                fixture.scales(task), kInter, kLat))
            return -1;
    }
    const float *pair_out[kTasks] = {};
    if (waste_cuda_vq_group_pair_finish(&model, kTasks, pair_out)) return -1;
    for (int task = 0; task < kTasks; task++) {
        if (!pair_out[task]) return -1;
        if (capture)
            memcpy(work.pair_at(task), pair_out[task],
                   (size_t)2 * kInter * sizeof(float));
        activate(pair_out[task], work.act_at(task));
        if (waste_cuda_vq_group_down_enqueue(
                &model, task, fixture.down(task),
                fixture.scales(task) + 2 * kInter,
                work.act_at(task), 2 * kStages, kLat, kInter))
            return -1;
    }
    const float *down_out[kTasks] = {};
    if (waste_cuda_vq_group_down_finish(&model, kTasks, down_out)) return -1;
    if (capture)
        for (int task = 0; task < kTasks; task++) {
            if (!down_out[task]) return -1;
            memcpy(work.down_at(task), down_out[task],
                   (size_t)kLat * sizeof(float));
        }
    return 0;
}

static int compare_bits(const char *what, const std::vector<float> &a,
                        const std::vector<float> &b)
{
    size_t mismatches = 0, first = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        uint32_t ua = 0, ub = 0;
        memcpy(&ua, &a[i], sizeof ua);
        memcpy(&ub, &b[i], sizeof ub);
        if (ua != ub) {
            if (!mismatches) first = i;
            mismatches++;
        }
        max_abs = std::max(max_abs, std::fabs(a[i] - b[i]));
    }
    std::printf("correctness=%s elements=%zu byte_exact=%d mismatches=%zu "
                "max_abs=%.9g\n", what, a.size(), mismatches == 0,
                mismatches, max_abs);
    if (mismatches) {
        uint32_t ua = 0, ub = 0;
        memcpy(&ua, &a[first], sizeof ua);
        memcpy(&ub, &b[first], sizeof ub);
        std::fprintf(stderr,
                     "first %s mismatch at %zu: ordinary=0x%08x "
                     "pair2=0x%08x\n", what, first, ua, ub);
        return -1;
    }
    return 0;
}

template <typename Fn>
static double milliseconds_per(int iterations, Fn &&fn)
{
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
        if (fn()) {
            std::fprintf(stderr, "timed VQ execution failed\n");
            std::exit(1);
        }
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count() /
           (double)iterations;
}

static int established_group_reuse_gate(const Fixture &fixture)
{
    waste_model model{};
    configure(model, fixture);
    if (waste_cuda_vq_init(&model) || waste_cuda_vq_prepare_pair(
            &model, 2, fixture.row_x(0), nullptr, nullptr, 0, kLat))
        return -1;
    const float *out[1] = {};
    for (int group = 0; group < 2; group++) {
        if (waste_cuda_vq_group_pair_enqueue(
                &model, 0, fixture.gate(group), fixture.up(group),
                fixture.scales(group), kInter, kLat) ||
            waste_cuda_vq_group_pair_finish(&model, 1, out) || !out[0]) {
            waste_cuda_kda_free(&model);
            return -1;
        }
    }
    if (waste_cuda_vq_group_drain(&model)) {
        waste_cuda_kda_free(&model);
        return -1;
    }
    waste_cuda_kda_free(&model);
    return 0;
}

static int sticky_bounds_gate(const Fixture &fixture)
{
    waste_model model{};
    configure(model, fixture);
    if (waste_cuda_vq_init(&model)) return -1;
    if (waste_cuda_vq_prepare_pair2(
            &model, fixture.row_x(0), fixture.row_x(1), 0, kLat + kVec) != -1) {
        waste_cuda_kda_free(&model);
        return -1;
    }
    /* The invalid second-row bound must poison subsequent group work. */
    if (waste_cuda_vq_prepare_pair2(
            &model, fixture.row_x(0), fixture.row_x(1), 0, kLat) != -1) {
        waste_cuda_kda_free(&model);
        return -1;
    }
    waste_cuda_kda_free(&model);

    configure(model, fixture);
    if (waste_cuda_vq_init(&model) || waste_cuda_vq_prepare_pair2(
            &model, fixture.row_x(0), fixture.row_x(1), 0, kLat))
        return -1;
    for (int slot = 0; slot < kTasks; slot++)
        if (waste_cuda_vq_group_pair2_enqueue(
                &model, slot, slot / kExpertsPerRow,
                fixture.gate(slot), fixture.up(slot), fixture.scales(slot),
                kInter, kLat)) {
            waste_cuda_kda_free(&model);
            return -1;
        }
    if (waste_cuda_vq_group_pair2_enqueue(
            &model, kTasks, 0, fixture.gate(0), fixture.up(0),
            fixture.scales(0), kInter, kLat) != -1) {
        waste_cuda_kda_free(&model);
        return -1;
    }
    const float *out[kTasks] = {};
    if (waste_cuda_vq_group_pair_finish(&model, kTasks, out) != -1) {
        waste_cuda_kda_free(&model);
        return -1;
    }
    waste_cuda_kda_free(&model);
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc > 3) {
        std::fprintf(stderr, "usage: %s [ITERATIONS] [WARMUP]\n", argv[0]);
        return 2;
    }
    const int iterations = argc > 1
        ? positive_arg(argv[1], "ITERATIONS") : 3;
    const int warmup = argc > 2 ? positive_arg(argv[2], "WARMUP") : 1;
    Fixture fixture;
    Work ordinary, pair2;
    waste_model model{};
    configure(model, fixture);
    if (waste_cuda_vq_init(&model) ||
        ordinary_run(model, fixture, ordinary) ||
        pair2_run(model, fixture, pair2, true)) {
        std::fprintf(stderr, "initial VQ pair2 execution failed\n");
        waste_cuda_kda_free(&model);
        return 1;
    }
    if (compare_bits("gate-up", ordinary.pair, pair2.pair) ||
        compare_bits("activated", ordinary.act, pair2.act) ||
        compare_bits("down", ordinary.down, pair2.down)) {
        waste_cuda_kda_free(&model);
        return 1;
    }

    for (int i = 0; i < warmup; i++) {
        if (ordinary_run(model, fixture, ordinary) ||
            pair2_run(model, fixture, pair2, false)) {
            std::fprintf(stderr, "VQ pair2 warmup failed\n");
            waste_cuda_kda_free(&model);
            return 1;
        }
    }
    const auto ordinary_call = [&] {
        return ordinary_run(model, fixture, ordinary);
    };
    const auto pair2_call = [&] {
        return pair2_run(model, fixture, pair2, false);
    };
    const double ordinary_before = milliseconds_per(iterations, ordinary_call);
    const double pair2_ms = milliseconds_per(iterations, pair2_call);
    const double ordinary_after = milliseconds_per(iterations, ordinary_call);
    const double ordinary_ms = 0.5 * (ordinary_before + ordinary_after);
    std::printf("shape=rows2-top8 lat=%d inter=%d tasks=%d iterations=%d "
                "warmup=%d\n", kLat, kInter, kTasks, iterations, warmup);
    std::printf("path=ordinary2-before wall_ms=%.6f\n", ordinary_before);
    std::printf("path=pair2-group16 wall_ms=%.6f\n", pair2_ms);
    std::printf("path=ordinary2-after wall_ms=%.6f\n", ordinary_after);
    std::printf("comparison=bracketed speedup=%.4f saved_ms=%.6f\n",
                ordinary_ms / pair2_ms, ordinary_ms - pair2_ms);
    waste_cuda_kda_free(&model);

    if (established_group_reuse_gate(fixture) || sticky_bounds_gate(fixture)) {
        std::fprintf(stderr, "VQ pair2 state/bounds gate failed\n");
        return 1;
    }
    std::printf("state=existing-group-lut-reuse passed=1\n");
    std::printf("state=pair2-bounds-sticky passed=1\n");
    return 0;
}
