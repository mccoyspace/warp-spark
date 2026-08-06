// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 SQLite Cloud, Inc.
/* Standalone GB10 pilot for WASTE's blocked VQ3R expert matrices.
 *
 * The expert record remains ordinary 4 KiB-aligned pageable host memory.  The
 * CUDA kernels dereference its indices and fp16 row scales through GB10 HMM;
 * only the small transposed codebooks and activation-dependent LUTs are copied
 * to device memory.  This is deliberately the ownership model an integrated
 * path would use: no second expert cache and no cudaHostRegister of a cache
 * slot.
 *
 * The strict kernels give one CUDA thread ownership of one complete output
 * row.  LUT dimensions, VQ stages, vector positions, and the final row scale
 * are accumulated in the same order as src/model.c.  There are no atomics or
 * cross-row reductions.
 *
 * Build on the Spark:
 *
 *   nvcc -O3 -std=c++17 -arch=native -fmad=false \
 *     -Xcompiler=-ffp-contract=off -Xcompiler=-pthread \
 *     -Xcompiler=-mcpu=native -o cuda_vq_bench tools/cuda_vq_bench.cu
 *
 * Synthetic expert-style record (default):
 *
 *   ./cuda_vq_bench --iterations 12 --warmup 2 --threads 8
 *
 * Use one real K3 bank record's blocked indices and row scales (the benchmark
 * still uses deterministic synthetic codebooks, so it needs no model load):
 *
 *   ./cuda_vq_bench --record MODEL/experts-L1.bin --expert 17
 *
 * Rotate a larger, less cache-friendly host working set:
 *
 *   ./cuda_vq_bench --record MODEL/experts-L1.bin --expert 17 \
 *     --record-count 16
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <unistd.h>
#include <vector>

/* Use the engine's actual persistent fork-join pool.  The benchmark used to
 * create and join std::threads around every timed CPU call, which measured
 * thread lifecycle rather than WASTE's VQ path. */
#include "../src/threads.h"

namespace {

constexpr uint32_t MAGIC_EXPERT = 0x50584557u; // "WEXP"
constexpr uint8_t FMT_VQ3R = 4;
constexpr int STAGES = 3;
constexpr int VEC_DIM = 8;
constexpr int ENTRIES = 256;
constexpr int INDEX_BLOCK = 64;
constexpr int CUDA_THREADS = 256;
constexpr size_t RECORD_ALIGN = 4096;

struct Shape {
    int rows;
    int cols;
};

enum class ApplyVariant {
    StrictRow = 0,
    Tree = 1,
};

static const char *variant_name(ApplyVariant variant)
{
    return variant == ApplyVariant::StrictRow ? "strict-row" : "tree";
}

constexpr Shape GATE = {3072, 3584};
constexpr Shape UP = {3072, 3584};
constexpr Shape DOWN = {3584, 3072};

#pragma pack(push, 1)
struct ExpertHeader {
    uint32_t magic;
    uint16_t layer;
    uint16_t expert_id;
    uint8_t fmt;
    uint8_t flags;
    uint16_t codebook_id;
    uint16_t lowrank_id;
    uint16_t reserved0;
    uint32_t rec_4k_blocks;
    uint32_t gate_off;
    uint32_t up_off;
    uint32_t down_off;
    uint32_t chan_corr_off;
    uint32_t crc32;
    uint32_t reserved1[2];
};
#pragma pack(pop)

static_assert(sizeof(ExpertHeader) == 48, "WASTE expert header layout");

struct Options {
    const char *record_path = nullptr;
    uint64_t expert = 0;
    int record_count = 1;
    int iterations = 12;
    int warmup = 2;
    int threads = 8;
    uint32_t seed = 0x7f4a7c15u;
    double max_abs = 0.01;
    double min_speedup = 3.0;
};

struct Buffers {
    uint8_t *record = nullptr;
    size_t record_bytes = 0;             // stride of one expert record
    int record_count = 1;
    const uint8_t *idx[3] = {};
    const uint16_t *scale[3] = {};
    std::vector<float> books;
    std::vector<float> x_lat;
    std::vector<float> x_down;
    std::vector<float> cpu_lut[3];
    std::vector<float> cpu_y[3];
    std::vector<float> gpu_y[2][3];       // strict-row, tree
};

struct DeviceBuffers {
    cudaStream_t stream = nullptr;
    float *books = nullptr;
    float *x_lat = nullptr;
    float *x_down = nullptr;
    float *lut[3] = {};
    float *y[3] = {};
    float *host_y[3] = {};               // pinned, CPU-readable return path
};

static void cuda_ok(cudaError_t status, const char *what)
{
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    std::exit(1);
}

static uint64_t number(const char *text, const char *what)
{
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (!text[0] || !end || *end) {
        std::fprintf(stderr, "invalid %s: %s\n", what, text);
        std::exit(2);
    }
    return (uint64_t)value;
}

static void usage(const char *argv0)
{
    std::fprintf(stderr,
        "usage: %s [--record BANK --expert ID] [--record-count N] "
        "[--iterations N] "
        "[--warmup N] [--threads N] [--seed N] [--max-abs X] "
        "[--min-speedup X]\n", argv0);
}

static double real_number(const char *text, const char *what)
{
    char *end = nullptr;
    const double value = std::strtod(text, &end);
    if (!text[0] || !end || *end || !std::isfinite(value)) {
        std::fprintf(stderr, "invalid %s: %s\n", what, text);
        std::exit(2);
    }
    return value;
}

static Options parse_options(int argc, char **argv)
{
    Options o;
    for (int i = 1; i < argc; i++) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            usage(argv[0]);
            std::exit(2);
        }
        const char *value = argv[++i];
        if (arg == "--record") o.record_path = value;
        else if (arg == "--expert") o.expert = number(value, "expert id");
        else if (arg == "--record-count")
            o.record_count = (int)number(value, "record count");
        else if (arg == "--iterations") o.iterations = (int)number(value, "iterations");
        else if (arg == "--warmup") o.warmup = (int)number(value, "warmup");
        else if (arg == "--threads") o.threads = (int)number(value, "threads");
        else if (arg == "--seed") o.seed = (uint32_t)number(value, "seed");
        else if (arg == "--max-abs") o.max_abs = real_number(value, "max abs");
        else if (arg == "--min-speedup")
            o.min_speedup = real_number(value, "minimum speedup");
        else {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            usage(argv[0]);
            std::exit(2);
        }
    }
    if (o.iterations < 1 || o.warmup < 0 || o.threads < 1 || o.threads > 64 ||
        o.record_count < 1 || o.record_count > 256 ||
        o.max_abs < 0.0 || o.min_speedup < 0.0) {
        std::fprintf(stderr,
            "iterations, records and 1-64 threads must be positive; "
            "warmup may be zero\n");
        std::exit(2);
    }
    if (o.record_count > 1 && !o.record_path) {
        std::fprintf(stderr, "--record-count > 1 requires --record BANK\n");
        std::exit(2);
    }
    return o;
}

static size_t round_up(size_t n, size_t alignment)
{
    return (n + alignment - 1) / alignment * alignment;
}

static size_t index_bytes(Shape shape)
{
    const size_t blocks = ((size_t)shape.rows + INDEX_BLOCK - 1) / INDEX_BLOCK;
    return blocks * (size_t)(shape.cols / VEC_DIM) * INDEX_BLOCK * STAGES;
}

static size_t lut_values(Shape shape)
{
    return (size_t)(shape.cols / VEC_DIM) * STAGES * ENTRIES;
}

static void *pageable_aligned_alloc(size_t bytes)
{
    void *p = nullptr;
    if (posix_memalign(&p, RECORD_ALIGN, round_up(bytes, RECORD_ALIGN)) || !p) {
        std::fprintf(stderr, "could not allocate %zu pageable bytes\n", bytes);
        std::exit(1);
    }
    return p;
}

static void pread_all(int fd, void *dst, size_t bytes, uint64_t offset)
{
    uint8_t *p = static_cast<uint8_t *>(dst);
    while (bytes) {
        const ssize_t got = pread(fd, p, bytes, (off_t)offset);
        if (got <= 0) {
            std::perror("pread");
            std::exit(1);
        }
        p += got;
        bytes -= (size_t)got;
        offset += (uint64_t)got;
    }
}

static uint32_t rng_next(uint32_t &state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static float rng_float(uint32_t &state, float magnitude)
{
    const int32_t centered = (int32_t)(rng_next(state) & 0xffffu) - 32768;
    return magnitude * (float)centered / 32768.0f;
}

static void bind_record(Buffers &b, int record_index)
{
    if (record_index < 0 || record_index >= b.record_count) {
        std::fprintf(stderr, "record index %d is out of range\n", record_index);
        std::exit(1);
    }
    const uint8_t *record = b.record + (size_t)record_index * b.record_bytes;
    const ExpertHeader *h = reinterpret_cast<const ExpertHeader *>(record);
    const size_t gate_n = index_bytes(GATE);
    const size_t up_n = index_bytes(UP);
    const size_t down_n = index_bytes(DOWN);
    const size_t scale_n = 2u * (GATE.rows + UP.rows + DOWN.rows);
    const bool offsets_ok =
        h->gate_off >= sizeof *h && (size_t)h->gate_off + gate_n <= h->up_off &&
        (size_t)h->up_off + up_n <= h->down_off &&
        (size_t)h->down_off + down_n <= h->chan_corr_off &&
        (size_t)h->chan_corr_off + scale_n <= b.record_bytes;
    if (h->magic != MAGIC_EXPERT || h->fmt != FMT_VQ3R ||
        (size_t)h->rec_4k_blocks * RECORD_ALIGN != b.record_bytes || !offsets_ok) {
        std::fprintf(stderr, "record is not a K3-shaped WASTE VQ3R expert\n");
        std::exit(1);
    }
    b.idx[0] = record + h->gate_off;
    b.idx[1] = record + h->up_off;
    b.idx[2] = record + h->down_off;
    const uint16_t *sc = reinterpret_cast<const uint16_t *>(
        record + h->chan_corr_off);
    b.scale[0] = sc;
    b.scale[1] = sc + GATE.rows;
    b.scale[2] = sc + GATE.rows + UP.rows;
}

static void make_synthetic_record(Buffers &b, uint32_t seed)
{
    const size_t gate_n = index_bytes(GATE);
    const size_t up_n = index_bytes(UP);
    const size_t down_n = index_bytes(DOWN);
    const size_t corr_off = sizeof(ExpertHeader) + gate_n + up_n + down_n;
    const size_t unpadded = corr_off + 2u * (GATE.rows + UP.rows + DOWN.rows);
    b.record_bytes = round_up(unpadded, RECORD_ALIGN);
    b.record_count = 1;
    b.record = static_cast<uint8_t *>(pageable_aligned_alloc(b.record_bytes));
    std::memset(b.record, 0, b.record_bytes);

    ExpertHeader h = {};
    h.magic = MAGIC_EXPERT;
    h.layer = 1;
    h.expert_id = 0;
    h.fmt = FMT_VQ3R;
    h.rec_4k_blocks = (uint32_t)(b.record_bytes / RECORD_ALIGN);
    h.gate_off = sizeof h;
    h.up_off = (uint32_t)(h.gate_off + gate_n);
    h.down_off = (uint32_t)(h.up_off + up_n);
    h.chan_corr_off = (uint32_t)corr_off;
    std::memcpy(b.record, &h, sizeof h);

    uint32_t state = seed ? seed : 1u;
    for (size_t i = sizeof h; i < corr_off; i++)
        b.record[i] = (uint8_t)rng_next(state);

    uint16_t *sc = reinterpret_cast<uint16_t *>(b.record + corr_off);
    const size_t rows = (size_t)GATE.rows + UP.rows + DOWN.rows;
    for (size_t i = 0; i < rows; i++) sc[i] = 0x2400u; // 0.015625
    const uint16_t edge[] = {
        0x0000u, 0x8000u, 0x0001u, 0x8001u, 0x03ffu, 0x83ffu,
        0x0400u, 0x8400u, 0x3555u, 0xb555u, 0x3c00u, 0xbc00u
    };
    for (size_t i = 0; i < sizeof edge / sizeof edge[0]; i++) sc[i] = edge[i];
    bind_record(b, 0);
}

static void load_bank_records(Buffers &b, const char *path, uint64_t expert,
                              int record_count)
{
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        std::perror(path);
        std::exit(1);
    }
    ExpertHeader first = {};
    pread_all(fd, &first, sizeof first, 0);
    if (first.magic != MAGIC_EXPERT || !first.rec_4k_blocks) {
        std::fprintf(stderr, "%s does not begin with a WASTE expert record\n", path);
        std::exit(1);
    }
    b.record_bytes = (size_t)first.rec_4k_blocks * RECORD_ALIGN;
    b.record_count = record_count;
    if (expert > std::numeric_limits<uint64_t>::max() / b.record_bytes ||
        expert > std::numeric_limits<uint64_t>::max() -
                     (uint64_t)(record_count - 1)) {
        std::fprintf(stderr, "expert offset overflow\n");
        std::exit(1);
    }
    if ((size_t)record_count > std::numeric_limits<size_t>::max() /
                                   b.record_bytes) {
        std::fprintf(stderr, "record allocation overflow\n");
        std::exit(1);
    }
    const size_t total_bytes = (size_t)record_count * b.record_bytes;
    b.record = static_cast<uint8_t *>(pageable_aligned_alloc(total_bytes));
    pread_all(fd, b.record, total_bytes, expert * b.record_bytes);
    close(fd);
    for (int i = 0; i < record_count; i++) {
        bind_record(b, i);
        const ExpertHeader *h = reinterpret_cast<const ExpertHeader *>(
            b.record + (size_t)i * b.record_bytes);
        const uint64_t expected = expert + (uint64_t)i;
        if (expected <= UINT16_MAX && h->expert_id != (uint16_t)expected)
            std::fprintf(stderr,
                "warning: requested expert %llu, record names %u\n",
                (unsigned long long)expected, (unsigned)h->expert_id);
    }
    bind_record(b, 0);
}

static inline float half_to_float_cpu(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ff;
    if (exponent == 0) {
        const float value = (float)mantissa * 5.9604644775390625e-08f;
        return sign ? -value : value;
    }
    const uint32_t bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    float value;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

__device__ static float half_to_float_device(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ff;
    if (exponent == 0) {
        const float value = (float)mantissa * 5.9604644775390625e-08f;
        return sign ? -value : value;
    }
    return __uint_as_float(sign | ((exponent + 112u) << 23) |
                           (mantissa << 13));
}

struct CpuLutArgs {
    float *lut;
    const float *books;
    const float *x;
    Shape shape;
    int book_base;
};

/* Same [vector][stage][code] traversal and sixteen-code dependency groups as
 * waste_lutb_range.  Writing this without arm_neon.h keeps NVCC happy; its
 * host compiler can vectorize the sixteen independent explicit-FMA chains. */
static void cpu_lut_range(int begin, int end, void *opaque)
{
    const CpuLutArgs &a = *static_cast<const CpuLutArgs *>(opaque);
    for (int vector = begin; vector < end; vector++) {
        const float *xv = a.x + (size_t)vector * VEC_DIM;
        for (int stage = 0; stage < STAGES; stage++) {
            const float *book = a.books +
                (size_t)(a.book_base + stage) * VEC_DIM * ENTRIES;
            float *dst = a.lut +
                ((size_t)vector * STAGES + stage) * ENTRIES;
            for (int code0 = 0; code0 < ENTRIES; code0 += 16) {
                float acc[16] = {};
                for (int d = 0; d < VEC_DIM; d++) {
                    const float xd = xv[d];
                    const float *row = book + (size_t)d * ENTRIES + code0;
                    for (int lane = 0; lane < 16; lane++)
                        acc[lane] = std::fma(xd, row[lane], acc[lane]);
                }
                std::memcpy(dst + code0, acc, sizeof acc);
            }
        }
    }
}

static void cpu_build_lut(float *lut, const float *books, const float *x,
                          Shape shape, int book_base, int)
{
    CpuLutArgs args = {lut, books, x, shape, book_base};
    waste_parallel_for(shape.cols / VEC_DIM, 16, cpu_lut_range, &args);
}

constexpr int CPU_VQ_TILE = 64;
constexpr int CPU_VQ_SUPER = 2;

struct CpuApplyArgs {
    float *y;
    const uint8_t *idx;
    const uint16_t *scale;
    const float *lut;
    int nv;
};

/* This is the current engine's tiled/vector-outer VQ gather, including the
 * blocked index addressing and eight-row packed-index loop.  The old pilot's
 * row-outer reference defeated LUT reuse and made the CPU control much slower
 * than the code being replaced. */
static void cpu_apply_rows(int begin, int end, void *opaque)
{
    const CpuApplyArgs &a = *static_cast<const CpuApplyArgs *>(opaque);
    float acc[CPU_VQ_TILE * CPU_VQ_SUPER];

    for (int row0 = begin; row0 < end;
         row0 += CPU_VQ_TILE * CPU_VQ_SUPER) {
        const int rows = std::min(CPU_VQ_TILE * CPU_VQ_SUPER, end - row0);
        const int blocks = (rows + CPU_VQ_TILE - 1) / CPU_VQ_TILE;
        std::memset(acc, 0, (size_t)rows * sizeof(float));

        for (int vector = 0; vector < a.nv; vector++) {
            const float *table =
                a.lut + (size_t)vector * STAGES * ENTRIES;
            const float *stage1 = table + ENTRIES;
            const float *stage2 = table + 2 * ENTRIES;
            for (int block = 0; block < blocks; block++) {
                const int nr = std::min(CPU_VQ_TILE,
                                        rows - block * CPU_VQ_TILE);
                const uint8_t *ix = a.idx +
                    ((size_t)(row0 / CPU_VQ_TILE + block) * a.nv + vector) *
                    CPU_VQ_TILE * STAGES;
                float *out = acc + (size_t)block * CPU_VQ_TILE;
                int row = 0;
                for (; row + 8 <= nr; row += 8, ix += 8 * STAGES) {
                    uint32_t w0, w1, w2, w3, w4, w5;
                    std::memcpy(&w0, ix,      4);
                    std::memcpy(&w1, ix +  4, 4);
                    std::memcpy(&w2, ix +  8, 4);
                    std::memcpy(&w3, ix + 12, 4);
                    std::memcpy(&w4, ix + 16, 4);
                    std::memcpy(&w5, ix + 20, 4);
                    const float t0 = table[w0 & 0xff] +
                        stage1[(w0 >> 8) & 0xff] + stage2[(w0 >> 16) & 0xff];
                    const float t1 = table[w0 >> 24] + stage1[w1 & 0xff] +
                        stage2[(w1 >> 8) & 0xff];
                    const float t2 = table[(w1 >> 16) & 0xff] +
                        stage1[w1 >> 24] + stage2[w2 & 0xff];
                    const float t3 = table[(w2 >> 8) & 0xff] +
                        stage1[(w2 >> 16) & 0xff] + stage2[w2 >> 24];
                    const float t4 = table[w3 & 0xff] +
                        stage1[(w3 >> 8) & 0xff] + stage2[(w3 >> 16) & 0xff];
                    const float t5 = table[w3 >> 24] + stage1[w4 & 0xff] +
                        stage2[(w4 >> 8) & 0xff];
                    const float t6 = table[(w4 >> 16) & 0xff] +
                        stage1[w4 >> 24] + stage2[w5 & 0xff];
                    const float t7 = table[(w5 >> 8) & 0xff] +
                        stage1[(w5 >> 16) & 0xff] + stage2[w5 >> 24];
                    out[row] += t0;
                    out[row + 1] += t1;
                    out[row + 2] += t2;
                    out[row + 3] += t3;
                    out[row + 4] += t4;
                    out[row + 5] += t5;
                    out[row + 6] += t6;
                    out[row + 7] += t7;
                }
                for (; row < nr; row++, ix += STAGES) {
                    float term = table[ix[0]];
                    term = term + stage1[ix[1]];
                    term = term + stage2[ix[2]];
                    out[row] += term;
                }
            }
        }
        for (int row = 0; row < rows; row++)
            a.y[row0 + row] =
                acc[row] * half_to_float_cpu(a.scale[row0 + row]);
    }
}

static void cpu_apply(float *y, const uint8_t *idx, const uint16_t *scale,
                      const float *lut, Shape shape, int)
{
    CpuApplyArgs args = {
        y, idx, scale, lut, shape.cols / VEC_DIM
    };
    waste_parallel_for(shape.rows, CPU_VQ_TILE * CPU_VQ_SUPER,
                       cpu_apply_rows, &args);
}

__global__ static void build_lut_kernel(float *lut, const float *books,
                                        const float *x, int nv, int book_base)
{
    const int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = nv * STAGES * ENTRIES;
    if (p >= total) return;
    const int code = p % ENTRIES;
    const int q = p / ENTRIES;
    const int stage = q % STAGES;
    const int vector = q / STAGES;
    const float *book = books +
        (size_t)(book_base + stage) * VEC_DIM * ENTRIES;
    float sum = 0.0f;
#pragma unroll
    for (int d = 0; d < VEC_DIM; d++)
        sum = fmaf(x[(size_t)vector * VEC_DIM + d],
                   book[(size_t)d * ENTRIES + code], sum);
    lut[p] = sum;
}

__global__ static void apply_kernel(float *y, const uint8_t *idx,
                                    const uint16_t *scale, const float *lut,
                                    int rows, int nv)
{
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    float acc = 0.0f;
    for (int v = 0; v < nv; v++) {
        const size_t off =
            (((size_t)(row / INDEX_BLOCK) * nv + v) * INDEX_BLOCK +
             (row % INDEX_BLOCK)) * STAGES;
        const float *block = lut + (size_t)v * STAGES * ENTRIES;
        float term = block[idx[off]];
        term = __fadd_rn(term, block[ENTRIES + idx[off + 1]]);
        term = __fadd_rn(term, block[2 * ENTRIES + idx[off + 2]]);
        acc = __fadd_rn(acc, term);
    }
    y[row] = __fmul_rn(acc, half_to_float_device(scale[row]));
}

/* Integration topology for the two same-shaped matrices: one CTA owns one
 * 64-row index block from gate and one from up.  Each thread still owns a
 * complete row, so fusion changes launch geometry, not arithmetic order. */
__global__ static void apply_gate_up_strict_kernel(
    float *gate_y, float *up_y, const uint8_t *gate_idx,
    const uint8_t *up_idx, const uint16_t *gate_scale,
    const uint16_t *up_scale, const float *gate_lut, const float *up_lut,
    int rows, int nv)
{
    const int lane = (int)threadIdx.x;
    const bool is_up = lane >= INDEX_BLOCK;
    const int row = (int)blockIdx.x * INDEX_BLOCK +
                    (lane & (INDEX_BLOCK - 1));
    if (row >= rows || lane >= 2 * INDEX_BLOCK) return;
    const uint8_t *idx = is_up ? up_idx : gate_idx;
    const uint16_t *scale = is_up ? up_scale : gate_scale;
    const float *lut = is_up ? up_lut : gate_lut;
    float acc = 0.0f;
    for (int v = 0; v < nv; v++) {
        const size_t off =
            (((size_t)(row / INDEX_BLOCK) * nv + v) * INDEX_BLOCK +
             (row % INDEX_BLOCK)) * STAGES;
        const float *block = lut + (size_t)v * STAGES * ENTRIES;
        float term = block[idx[off]];
        term = __fadd_rn(term, block[ENTRIES + idx[off + 1]]);
        term = __fadd_rn(term, block[2 * ENTRIES + idx[off + 2]]);
        acc = __fadd_rn(acc, term);
    }
    float *y = is_up ? up_y : gate_y;
    y[row] = __fmul_rn(acc, half_to_float_device(scale[row]));
}

/* Mode 2 gives the vector positions of one row to 128 lanes.  Stage order is
 * still scalar, but the vector-position partials are tree-reduced.  It is the
 * explicitly non-bit-exact throughput comparison; strict-row above remains
 * the integration candidate unless this arm's bounded error is accepted. */
__global__ static void apply_tree_kernel(float *y, const uint8_t *idx,
                                         const uint16_t *scale,
                                         const float *lut, int rows, int nv)
{
    const int row = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    if (row >= rows || lane >= 128) return;
    float acc = 0.0f;
    for (int v = lane; v < nv; v += 128) {
        const size_t off =
            (((size_t)(row / INDEX_BLOCK) * nv + v) * INDEX_BLOCK +
             (row % INDEX_BLOCK)) * STAGES;
        const float *block = lut + (size_t)v * STAGES * ENTRIES;
        float term = block[idx[off]];
        term = __fadd_rn(term, block[ENTRIES + idx[off + 1]]);
        term = __fadd_rn(term, block[2 * ENTRIES + idx[off + 2]]);
        acc = __fadd_rn(acc, term);
    }
    __shared__ float partial[128];
    partial[lane] = acc;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (lane < stride)
            partial[lane] = __fadd_rn(partial[lane], partial[lane + stride]);
        __syncthreads();
    }
    if (lane == 0)
        y[row] = __fmul_rn(partial[0], half_to_float_device(scale[row]));
}

__global__ static void half_edge_kernel(const uint16_t *in, float *out, int n)
{
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = half_to_float_device(in[i]);
}

static void launch_build(DeviceBuffers &d, int kind, Shape shape, int base,
                         const float *x)
{
    const int total = (int)lut_values(shape);
    build_lut_kernel<<<(total + CUDA_THREADS - 1) / CUDA_THREADS,
                       CUDA_THREADS, 0, d.stream>>>(
        d.lut[kind], d.books, x, shape.cols / VEC_DIM, base);
    cuda_ok(cudaGetLastError(), "build_lut_kernel launch");
}

static void launch_apply(DeviceBuffers &d, int kind, Shape shape,
                         const uint8_t *idx, const uint16_t *scale,
                         ApplyVariant variant, int strict_threads = 128)
{
    if (variant == ApplyVariant::Tree) {
        apply_tree_kernel<<<shape.rows, 128, 0, d.stream>>>(
            d.y[kind], idx, scale, d.lut[kind], shape.rows,
            shape.cols / VEC_DIM);
        cuda_ok(cudaGetLastError(), "apply_tree_kernel launch");
    } else {
        apply_kernel<<<(shape.rows + strict_threads - 1) / strict_threads,
                       strict_threads, 0, d.stream>>>(
            d.y[kind], idx, scale, d.lut[kind], shape.rows,
            shape.cols / VEC_DIM);
        cuda_ok(cudaGetLastError(), "apply_kernel launch");
    }
}

static void cpu_pipeline(Buffers &b, int threads)
{
    cpu_build_lut(b.cpu_lut[0].data(), b.books.data(), b.x_lat.data(), GATE, 0, threads);
    cpu_build_lut(b.cpu_lut[1].data(), b.books.data(), b.x_lat.data(), UP, 3, threads);
    cpu_apply(b.cpu_y[0].data(), b.idx[0], b.scale[0], b.cpu_lut[0].data(), GATE, threads);
    cpu_apply(b.cpu_y[1].data(), b.idx[1], b.scale[1], b.cpu_lut[1].data(), UP, threads);
    cpu_build_lut(b.cpu_lut[2].data(), b.books.data(), b.x_down.data(), DOWN, 6, threads);
    cpu_apply(b.cpu_y[2].data(), b.idx[2], b.scale[2], b.cpu_lut[2].data(), DOWN, threads);
}

/* Two synchronizations mirror the first integration seam: gate/up return for
 * CPU SiTU, then down returns for CPU router-ordered expert accumulation. */
static void gpu_pipeline(const Buffers &b, DeviceBuffers &d,
                         ApplyVariant variant, bool return_to_cpu)
{
    launch_build(d, 0, GATE, 0, d.x_lat);
    launch_build(d, 1, UP, 3, d.x_lat);
    if (variant == ApplyVariant::StrictRow) {
        apply_gate_up_strict_kernel<<<GATE.rows / INDEX_BLOCK,
                                      2 * INDEX_BLOCK, 0, d.stream>>>(
            d.y[0], d.y[1], b.idx[0], b.idx[1], b.scale[0], b.scale[1],
            d.lut[0], d.lut[1], GATE.rows, GATE.cols / VEC_DIM);
        cuda_ok(cudaGetLastError(), "apply_gate_up_strict_kernel launch");
    } else {
        launch_apply(d, 0, GATE, b.idx[0], b.scale[0], variant);
        launch_apply(d, 1, UP, b.idx[1], b.scale[1], variant);
    }
    if (return_to_cpu) {
        cuda_ok(cudaMemcpyAsync(d.host_y[0], d.y[0],
                               (size_t)GATE.rows * sizeof(float),
                               cudaMemcpyDeviceToHost, d.stream),
                "return gate output");
        cuda_ok(cudaMemcpyAsync(d.host_y[1], d.y[1],
                               (size_t)UP.rows * sizeof(float),
                               cudaMemcpyDeviceToHost, d.stream),
                "return up output");
    }
    cuda_ok(cudaStreamSynchronize(d.stream), "gate/up kernel synchronization");

    launch_build(d, 2, DOWN, 6, d.x_down);
    launch_apply(d, 2, DOWN, b.idx[2], b.scale[2], variant);
    if (return_to_cpu)
        cuda_ok(cudaMemcpyAsync(d.host_y[2], d.y[2],
                               (size_t)DOWN.rows * sizeof(float),
                               cudaMemcpyDeviceToHost, d.stream),
                "return down output");
    cuda_ok(cudaStreamSynchronize(d.stream), "down kernel synchronization");
}

static DeviceBuffers make_device_buffers(const Buffers &b)
{
    DeviceBuffers d;
    cuda_ok(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking), "stream creation");
    cuda_ok(cudaMalloc((void **)&d.books, b.books.size() * sizeof(float)), "device codebooks");
    cuda_ok(cudaMalloc((void **)&d.x_lat, b.x_lat.size() * sizeof(float)), "device latent input");
    cuda_ok(cudaMalloc((void **)&d.x_down, b.x_down.size() * sizeof(float)), "device down input");
    cuda_ok(cudaMemcpy(d.books, b.books.data(), b.books.size() * sizeof(float),
                       cudaMemcpyHostToDevice), "copy codebooks");
    cuda_ok(cudaMemcpy(d.x_lat, b.x_lat.data(), b.x_lat.size() * sizeof(float),
                       cudaMemcpyHostToDevice), "copy latent input");
    cuda_ok(cudaMemcpy(d.x_down, b.x_down.data(), b.x_down.size() * sizeof(float),
                       cudaMemcpyHostToDevice), "copy down input");
    const Shape shape[3] = {GATE, UP, DOWN};
    for (int k = 0; k < 3; k++) {
        cuda_ok(cudaMalloc((void **)&d.lut[k], lut_values(shape[k]) * sizeof(float)),
                "device LUT");
        cuda_ok(cudaMalloc((void **)&d.y[k], (size_t)shape[k].rows * sizeof(float)),
                "device output");
        cuda_ok(cudaHostAlloc((void **)&d.host_y[k],
                              (size_t)shape[k].rows * sizeof(float),
                              cudaHostAllocDefault), "pinned return output");
    }
    return d;
}

static void free_device_buffers(DeviceBuffers &d)
{
    for (int k = 0; k < 3; k++) {
        if (d.host_y[k]) cudaFreeHost(d.host_y[k]);
        cudaFree(d.y[k]);
        cudaFree(d.lut[k]);
    }
    cudaFree(d.x_down);
    cudaFree(d.x_lat);
    cudaFree(d.books);
    if (d.stream) cudaStreamDestroy(d.stream);
}

static void initialize_numeric_inputs(Buffers &b, uint32_t seed)
{
    uint32_t state = seed ^ 0xa5a5a5a5u;
    b.books.resize((size_t)9 * VEC_DIM * ENTRIES);
    b.x_lat.resize(GATE.cols);
    b.x_down.resize(DOWN.cols);
    for (float &v : b.books) v = rng_float(state, 0.25f);
    for (float &v : b.x_lat) v = rng_float(state, 0.5f);
    for (float &v : b.x_down) v = rng_float(state, 0.5f);
    const Shape shape[3] = {GATE, UP, DOWN};
    for (int k = 0; k < 3; k++) {
        b.cpu_lut[k].resize(lut_values(shape[k]));
        b.cpu_y[k].resize(shape[k].rows);
        b.gpu_y[0][k].resize(shape[k].rows);
        b.gpu_y[1][k].resize(shape[k].rows);
    }
}

static uint32_t float_bits(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
}

static void check_half_edges()
{
    const uint16_t edge[] = {
        0x0000u, 0x8000u, 0x0001u, 0x8001u, 0x03ffu, 0x83ffu,
        0x0400u, 0x8400u, 0x3c00u, 0xbc00u, 0x7bffu, 0xfbffu,
        0x7c00u, 0xfc00u, 0x7e01u
    };
    constexpr int n = (int)(sizeof edge / sizeof edge[0]);
    uint16_t *device_in = nullptr;
    float *device_out = nullptr;
    float host[n] = {};
    cuda_ok(cudaMalloc((void **)&device_in, sizeof edge), "half edge input");
    cuda_ok(cudaMalloc((void **)&device_out, sizeof host), "half edge output");
    cuda_ok(cudaMemcpy(device_in, edge, sizeof edge, cudaMemcpyHostToDevice),
            "copy half edges");
    half_edge_kernel<<<1, 32>>>(device_in, device_out, n);
    cuda_ok(cudaGetLastError(), "half_edge_kernel launch");
    cuda_ok(cudaMemcpy(host, device_out, sizeof host, cudaMemcpyDeviceToHost),
            "copy half edge results");
    int mismatches = 0;
    for (int i = 0; i < n; i++)
        if (float_bits(host[i]) != float_bits(half_to_float_cpu(edge[i]))) mismatches++;
    cudaFree(device_out);
    cudaFree(device_in);
    std::printf("fp16 conversion edge cases: %d values, %d bit mismatches\n",
                n, mismatches);
    if (mismatches) std::exit(1);
}

struct CompareMetrics {
    double max_abs;
    double mean_abs;
    double max_rel;
    int bit_mismatches;
    int nonfinite;
};

static CompareMetrics compare_output(const char *name,
                                     const std::vector<float> &cpu,
                                     const std::vector<float> &gpu)
{
    double max_abs = 0.0;
    double sum_abs = 0.0;
    double max_rel = 0.0;
    int bit_mismatches = 0;
    int nonfinite = 0;
    size_t worst = 0;
    for (size_t i = 0; i < cpu.size(); i++) {
        if (!std::isfinite(cpu[i]) || !std::isfinite(gpu[i])) {
            nonfinite++;
            continue;
        }
        const double delta = std::fabs((double)cpu[i] - gpu[i]);
        sum_abs += delta;
        const double denom = std::max(1e-20, std::fabs((double)cpu[i]));
        if (delta > max_abs) { max_abs = delta; worst = i; }
        max_rel = std::max(max_rel, delta / denom);
        if (float_bits(cpu[i]) != float_bits(gpu[i])) bit_mismatches++;
    }
    const double mean_abs = sum_abs / (double)cpu.size();
    std::printf("%-10s: max_abs %.9g  mean_abs %.9g  max_rel %.9g  "
                "bit_mismatches %d/%zu"
                "  nonfinite %d  worst_row %zu\n",
                name, max_abs, mean_abs, max_rel, bit_mismatches, cpu.size(),
                nonfinite, worst);
    return {max_abs, mean_abs, max_rel, bit_mismatches, nonfinite};
}

static int repeat_mismatches(const std::vector<float> &a,
                             const std::vector<float> &b)
{
    int mismatches = 0;
    for (size_t i = 0; i < a.size(); i++)
        if (float_bits(a[i]) != float_bits(b[i])) mismatches++;
    return mismatches;
}

static double milliseconds(const std::chrono::steady_clock::time_point &begin,
                           const std::chrono::steady_clock::time_point &end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct TimingStats {
    double median;
    double mean;
    double minimum;
};

static TimingStats timing_stats(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    double sum = 0.0;
    for (double value : values) sum += value;
    const size_t n = values.size();
    const double median = n & 1 ? values[n / 2]
                                : 0.5 * (values[n / 2 - 1] + values[n / 2]);
    return {median, sum / (double)n, values.front()};
}

} // namespace

int main(int argc, char **argv)
{
    const Options options = parse_options(argc, argv);
    waste_pool_init(options.threads);

    /* This must precede the first CUDA context creation. */
    cuda_ok(cudaSetDeviceFlags(cudaDeviceMapHost), "mapped-host device flags");
    cuda_ok(cudaSetDevice(0), "select CUDA device 0");
    int pageable = 0;
    int host_tables = 0;
    cuda_ok(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0),
            "pageable-memory attribute");
    cuda_ok(cudaDeviceGetAttribute(&host_tables,
                                   cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0),
            "host-page-table attribute");
    if (!pageable || !host_tables) {
        std::fprintf(stderr,
            "this pilot requires pageable host-page-table access (got %d/%d)\n",
            pageable, host_tables);
        return 1;
    }

    cudaDeviceProp prop = {};
    cuda_ok(cudaGetDeviceProperties(&prop, 0), "CUDA device properties");

    Buffers buffers;
    if (options.record_path)
        load_bank_records(buffers, options.record_path, options.expert,
                          options.record_count);
    else
        make_synthetic_record(buffers, options.seed);
    initialize_numeric_inputs(buffers, options.seed);
    const auto setup_begin = std::chrono::steady_clock::now();
    DeviceBuffers device = make_device_buffers(buffers);
    const auto setup_end = std::chrono::steady_clock::now();
    const Shape shape[3] = {GATE, UP, DOWN};
    size_t device_bytes = buffers.books.size() * sizeof(float) +
                          (buffers.x_lat.size() + buffers.x_down.size()) * sizeof(float);
    for (int k = 0; k < 3; k++)
        device_bytes += lut_values(shape[k]) * sizeof(float) +
                        (size_t)shape[k].rows * sizeof(float);
    const size_t pinned_return_bytes =
        (size_t)(GATE.rows + UP.rows + DOWN.rows) * sizeof(float);

    std::printf("device: %s  cc %d.%d  pageable/host-page-table %d/%d\n",
                prop.name, prop.major, prop.minor, pageable, host_tables);
    const size_t total_record_bytes =
        buffers.record_bytes * (size_t)buffers.record_count;
    std::printf("record: %s  %d x %zu bytes (%.3f MiB total), "
                "ordinary pageable host memory\n",
                options.record_path ? options.record_path : "synthetic VQ3R",
                buffers.record_count, buffers.record_bytes,
                total_record_bytes / 1048576.0);
    if (buffers.record_count == 1)
        std::printf("record working set: one repeatedly reused expert; "
                    "arithmetic/cache-hot smoke only\n");
    else
        std::printf("record working set: rotating %d consecutive experts "
                    "outside timed regions\n", buffers.record_count);
    std::printf("shapes: gate/up [3072,3584], down [3584,3072], "
                "block 64, VQ3R 8x256\n");
    std::printf("device-resident: %.3f MiB codebooks, %.3f MiB dynamic LUTs\n",
                buffers.books.size() * sizeof(float) / 1048576.0,
                (lut_values(GATE) + lut_values(UP) + lut_values(DOWN)) *
                    sizeof(float) / 1048576.0);
    std::printf("CUDA setup: %zu device bytes (%.3f MiB) + %zu pinned return "
                "bytes, %.3f ms; expert bytes copied: 0\n",
                device_bytes, device_bytes / 1048576.0,
                pinned_return_bytes,
                milliseconds(setup_begin, setup_end));

    check_half_edges();

    /* Correctness is checked before timing, not inferred from a timed run. */
    bind_record(buffers, 0);
    cpu_pipeline(buffers, options.threads);
    const ApplyVariant variants[2] = {
        ApplyVariant::StrictRow, ApplyVariant::Tree
    };
    CompareMetrics metrics[2][3] = {};
    bool correctness_ok = true;
    for (int vi = 0; vi < 2; vi++) {
        gpu_pipeline(buffers, device, variants[vi], true);
        for (int k = 0; k < 3; k++)
            std::memcpy(buffers.gpu_y[vi][k].data(), device.host_y[k],
                        (size_t)shape[k].rows * sizeof(float));
        std::printf("%s correctness\n", variant_name(variants[vi]));
        const char *kind[3] = {"gate", "up", "down"};
        for (int k = 0; k < 3; k++) {
            char label[24];
            std::snprintf(label, sizeof label, "%s-%s",
                          variant_name(variants[vi]), kind[k]);
            metrics[vi][k] = compare_output(
                label, buffers.cpu_y[k], buffers.gpu_y[vi][k]);
            if (metrics[vi][k].nonfinite) correctness_ok = false;
            if (variants[vi] == ApplyVariant::StrictRow &&
                metrics[vi][k].bit_mismatches) correctness_ok = false;
            if (variants[vi] == ApplyVariant::Tree &&
                metrics[vi][k].max_abs > options.max_abs)
                correctness_ok = false;
        }
    }

    /* A second launch must reproduce every output bit for each mode. */
    int repeat_bad[2] = {};
    for (int vi = 0; vi < 2; vi++) {
        gpu_pipeline(buffers, device, variants[vi], true);
        for (int k = 0; k < 3; k++) {
            std::vector<float> repeat(device.host_y[k],
                                      device.host_y[k] + shape[k].rows);
            repeat_bad[vi] += repeat_mismatches(buffers.gpu_y[vi][k], repeat);
        }
        std::printf("%s deterministic repeat: %d bit mismatches\n",
                    variant_name(variants[vi]), repeat_bad[vi]);
        if (repeat_bad[vi]) correctness_ok = false;
    }

    /* Component timings answer two separate pilot questions: whether LUT
     * construction moves, and whether each real gather shape clears the
     * configured apply-only gate (3x by preregistered default). Apply timings
     * include an async return to pinned host memory followed by stream
     * synchronization, just like the integration pipeline. */
    const char *kind_name[3] = {"gate", "up", "down"};
    const int book_base[3] = {0, 3, 6};
    const float *device_x[3] = {device.x_lat, device.x_lat, device.x_down};
    const float *cpu_x[3] = {
        buffers.x_lat.data(), buffers.x_lat.data(), buffers.x_down.data()
    };
    bool component_performance_ok = true;
    std::printf("\nper-shape LUT-build timing (launch+sync, no apply)\n");
    for (int k = 0; k < 3; k++) {
        std::vector<double> cpu_build_ms, gpu_build_ms;
        for (int i = 0; i < options.iterations; i++) {
            auto time_cpu_build = [&]() {
                const auto begin = std::chrono::steady_clock::now();
                cpu_build_lut(buffers.cpu_lut[k].data(), buffers.books.data(),
                              cpu_x[k], shape[k], book_base[k], options.threads);
                cpu_build_ms.push_back(milliseconds(
                    begin, std::chrono::steady_clock::now()));
            };
            auto time_gpu_build = [&]() {
                const auto begin = std::chrono::steady_clock::now();
                launch_build(device, k, shape[k], book_base[k], device_x[k]);
                cuda_ok(cudaStreamSynchronize(device.stream),
                        "component LUT-build synchronization");
                gpu_build_ms.push_back(milliseconds(
                    begin, std::chrono::steady_clock::now()));
            };
            if (i & 1) { time_gpu_build(); time_cpu_build(); }
            else       { time_cpu_build(); time_gpu_build(); }
        }
        const TimingStats cb = timing_stats(cpu_build_ms);
        const TimingStats gb = timing_stats(gpu_build_ms);
        std::printf("%-4s CPU %.3f ms  GPU %.3f ms  speedup %.3fx\n",
                    kind_name[k], cb.median, gb.median, cb.median / gb.median);
    }

    std::printf("\nper-shape apply-only timing (pinned D2H + sync included)\n");
    for (int k = 0; k < 3; k++) {
        /* Leave both references with a fresh, matching LUT before apply-only. */
        cpu_build_lut(buffers.cpu_lut[k].data(), buffers.books.data(), cpu_x[k],
                      shape[k], book_base[k], options.threads);
        launch_build(device, k, shape[k], book_base[k], device_x[k]);
        cuda_ok(cudaStreamSynchronize(device.stream),
                "apply-only LUT preparation");

        std::vector<double> cpu_apply_ms;
        std::vector<double> strict_apply_ms[3];
        std::vector<double> tree_apply_ms;
        auto time_cpu_apply = [&]() {
            const auto begin = std::chrono::steady_clock::now();
            cpu_apply(buffers.cpu_y[k].data(), buffers.idx[k], buffers.scale[k],
                      buffers.cpu_lut[k].data(), shape[k], options.threads);
            cpu_apply_ms.push_back(milliseconds(
                begin, std::chrono::steady_clock::now()));
        };
        auto time_gpu_apply = [&](ApplyVariant variant, int block, int slot) {
            const auto begin = std::chrono::steady_clock::now();
            launch_apply(device, k, shape[k], buffers.idx[k], buffers.scale[k],
                         variant, block);
            cuda_ok(cudaMemcpyAsync(device.host_y[k], device.y[k],
                                    (size_t)shape[k].rows * sizeof(float),
                                    cudaMemcpyDeviceToHost, device.stream),
                    "return apply-only output");
            cuda_ok(cudaStreamSynchronize(device.stream),
                    "apply-only synchronization");
            const double elapsed = milliseconds(
                begin, std::chrono::steady_clock::now());
            if (variant == ApplyVariant::Tree) tree_apply_ms.push_back(elapsed);
            else strict_apply_ms[slot].push_back(elapsed);
        };
        for (int i = 0; i < options.iterations; i++) {
            bind_record(buffers, i % buffers.record_count);
            /* Rotate five tasks to make the CTA sweep insensitive to order. */
            for (int q = 0; q < 5; q++) {
                switch ((i + q) % 5) {
                    case 0: time_cpu_apply(); break;
                    case 1: time_gpu_apply(ApplyVariant::StrictRow, 64, 0); break;
                    case 2: time_gpu_apply(ApplyVariant::StrictRow, 128, 1); break;
                    case 3: time_gpu_apply(ApplyVariant::StrictRow, 256, 2); break;
                    default: time_gpu_apply(ApplyVariant::Tree, 128, 0); break;
                }
            }
        }
        const TimingStats ca = timing_stats(cpu_apply_ms);
        TimingStats sa[3] = {
            timing_stats(strict_apply_ms[0]), timing_stats(strict_apply_ms[1]),
            timing_stats(strict_apply_ms[2])
        };
        const TimingStats ta = timing_stats(tree_apply_ms);
        double best_strict = sa[0].median;
        for (int q = 1; q < 3; q++) best_strict = std::min(best_strict, sa[q].median);
        const double component_speedup = ca.median / best_strict;
        if (component_speedup < options.min_speedup)
            component_performance_ok = false;
        std::printf("%-4s CPU %.3f ms | strict-row CTA64 %.3f, CTA128 %.3f, "
                    "CTA256 %.3f | tree %.3f ms | best strict %.3fx %s\n",
                    kind_name[k], ca.median, sa[0].median, sa[1].median,
                    sa[2].median, ta.median, component_speedup,
                    component_speedup >= options.min_speedup ? "PASS" : "FAIL");
    }

    for (int i = 0; i < options.warmup; i++) {
        bind_record(buffers, i % buffers.record_count);
        cpu_pipeline(buffers, options.threads);
        gpu_pipeline(buffers, device, ApplyVariant::StrictRow, true);
        gpu_pipeline(buffers, device, ApplyVariant::Tree, true);
    }

    std::vector<double> cpu_ms;
    std::vector<double> gpu_ms[2];
    cpu_ms.reserve((size_t)options.iterations);
    gpu_ms[0].reserve((size_t)options.iterations);
    gpu_ms[1].reserve((size_t)options.iterations);
    auto time_cpu = [&]() {
        const auto begin = std::chrono::steady_clock::now();
        cpu_pipeline(buffers, options.threads);
        const auto end = std::chrono::steady_clock::now();
        cpu_ms.push_back(milliseconds(begin, end));
    };
    auto time_gpu = [&](ApplyVariant variant) {
        const auto begin = std::chrono::steady_clock::now();
        gpu_pipeline(buffers, device, variant, true);
        const auto end = std::chrono::steady_clock::now();
        gpu_ms[(int)variant].push_back(milliseconds(begin, end));
    };
    for (int i = 0; i < options.iterations; i++) {
        bind_record(buffers,
                    (options.warmup + i) % buffers.record_count);
        /* Rotate the order so neither GPU mode consistently inherits the
         * CPU's heat or the other mode's cache residency. */
        if (i % 3 == 0) { time_cpu(); time_gpu(ApplyVariant::StrictRow); time_gpu(ApplyVariant::Tree); }
        if (i % 3 == 1) { time_gpu(ApplyVariant::StrictRow); time_gpu(ApplyVariant::Tree); time_cpu(); }
        if (i % 3 == 2) { time_gpu(ApplyVariant::Tree); time_cpu(); time_gpu(ApplyVariant::StrictRow); }
    }

    const TimingStats cpu = timing_stats(cpu_ms);
    const TimingStats gpu1 = timing_stats(gpu_ms[0]);
    const TimingStats gpu2 = timing_stats(gpu_ms[1]);
    std::printf("\none-expert integration pipeline (2 synchronizations; pinned D2H included)\n");
    std::printf("CPU %d threads: median %.3f ms  mean %.3f  min %.3f\n",
                options.threads, cpu.median, cpu.mean, cpu.minimum);
    std::printf("GPU strict-row: median %.3f ms  mean %.3f  min %.3f  speedup %.3fx\n",
                gpu1.median, gpu1.mean, gpu1.minimum, cpu.median / gpu1.median);
    std::printf("GPU tree:       median %.3f ms  mean %.3f  min %.3f  speedup %.3fx\n",
                gpu2.median, gpu2.mean, gpu2.minimum, cpu.median / gpu2.median);
    std::printf("naive 1472-expert projection: CPU %.3f s, strict-row %.3f s, tree %.3f s\n",
                cpu.median * 1472.0 / 1000.0, gpu1.median * 1472.0 / 1000.0,
                gpu2.median * 1472.0 / 1000.0);
    std::printf("note: that projection rebuilds gate/up LUTs per expert; the engine "
                "builds them once per layer, so use it only as a pilot comparison\n");

    const double strict_speedup = cpu.median / gpu1.median;
    const bool performance_ok = strict_speedup >= options.min_speedup &&
                                component_performance_ok;
    std::printf("gates: correctness %s (tree max_abs <= %.6g), performance %s "
                "(full strict-row %.3fx and every apply shape >= %.3fx)\n",
                correctness_ok ? "PASS" : "FAIL", options.max_abs,
                performance_ok ? "PASS" : "FAIL", strict_speedup,
                options.min_speedup);

    free_device_buffers(device);
    waste_pool_shutdown();
    std::free(buffers.record);
    return correctness_ok && performance_ok ? 0 : 1;
}
