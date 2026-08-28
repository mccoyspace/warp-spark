// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 SQLite Cloud, Inc.
/* Experimental GB10 CUDA path: WASTE Q4G matvec against pageable trunk
 * storage and strict-order VQ3R expert gather. The 29 GiB trunk and bounded
 * expert cache stay in their existing host allocations, so CUDA does not
 * create a second weight cache outside the engine's memory budget.
 *
 * Selected decode-only KDA and dense projections call the Q4 operation
 * directly. The VQ arm is separately opt-in and keeps the router, SiTU and
 * expert reduction on the CPU. Prefill, absorbed MLA kv_b and the Q8 head
 * remain on the qualified CPU too.
 */

#include "model.h"
#include "waste_backend.h"
#include "waste_format.h"

#include <cuda_runtime.h>

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    Q4_GROUP = 128,
    Q4_THREADS = 128,
    VQ_STAGES = 3,
    VQ_VEC_DIM = 8,
    VQ_ENTRIES = 256,
    VQ_INDEX_BLOCK = WASTE_VQ_INDEX_BLOCK,
    VQ_BUILD_THREADS = 256,
    VQ_DOWN_THREADS = 256,
    VQ_GROUP_MAX = 16,
    VQ_GROUP_IDLE = 0,
    VQ_GROUP_PAIR = 1,
    VQ_GROUP_DOWN = 2,
    VQ_LUT_GATE0 = 0,
    VQ_LUT_UP0 = 1,
    VQ_LUT_DOWN = 2,
    VQ_LUT_GATE1 = 3,
    VQ_LUT_UP1 = 4,
    VQ_LUT_COUNT = 5
};

static_assert(Q4_THREADS > 0 &&
              (Q4_THREADS & (Q4_THREADS - 1)) == 0,
              "Q4 reduction requires a power-of-two thread count");

#if defined(WASTE_CUDA_VQ_FUSED_TEST)
/* Test-only descriptors for the true task-major VQ prototype.  Production
 * grouping deliberately remains untouched until the standalone gate says
 * whether exposing several expert grids at once buys useful occupancy. */
typedef struct {
    const uint8_t *gate_idx;
    const uint8_t *up_idx;
    const uint16_t *scale;
    int lut_row;
} vq_fused_pair_task;

typedef struct {
    const uint8_t *idx;
    const uint16_t *scale;
} vq_fused_down_task;
#endif

typedef struct {
    cudaStream_t stream;
    float *host_x, *host_y;
    float *device_x, *device_y;
    size_t capacity;
    float *vq_books, *vq_x, *vq_y;
    /* gate0, up0, down, gate1, up1.  The second gate/up pair is allocated
     * only once with the rest of the VQ context and is reused by verify2. */
    float *vq_lut[VQ_LUT_COUNT];
    size_t vq_lut_values[VQ_LUT_COUNT];
    size_t vq_y_capacity;
    float *vq_group_pair_host_y, *vq_group_pair_device_y;
    float *vq_group_down_host_x, *vq_group_down_device_x;
    float *vq_group_down_host_y, *vq_group_down_device_y;
    size_t vq_group_pair_slot_values;
    size_t vq_group_down_x_slot_values;
    size_t vq_group_down_y_slot_values;
    int vq_group_phase, vq_group_count;
    int vq_group_rows, vq_group_cols;
    int vq_group_pair_prepared, vq_group_failed;
    int vq_ready;
#if defined(WASTE_CUDA_VQ_FUSED_TEST)
    vq_fused_pair_task *vq_fused_pair_tasks;
    vq_fused_down_task *vq_fused_down_tasks;
    float *vq_fused_down_lut;
#endif
} waste_cuda_kda;

__device__ static float q4_half(uint16_t h)
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

__device__ static int q4_at(const uint8_t *row, int index)
{
    const uint8_t packed = row[index >> 1];
    return ((index & 1) ? (packed >> 4) : (packed & 0x0f)) - 8;
}

__global__ static void q4_fast(const uint8_t *weights,
                               const uint16_t *scales,
                               const float *x, float *y,
                               int out, int in, size_t rowbytes)
{
    const int row_index = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    if (row_index >= out || lane >= Q4_THREADS) return;
    const int groups = (in + Q4_GROUP - 1) / Q4_GROUP;
    const uint8_t *row = weights + (size_t)row_index * rowbytes;
    const uint16_t *row_scales = scales + (size_t)row_index * groups;
    float sum = 0.0f;
    for (int i = lane; i < in; i += Q4_THREADS)
        sum = fmaf(q4_half(row_scales[i / Q4_GROUP]) *
                   (float)q4_at(row, i), x[i], sum);
    __shared__ float partial[Q4_THREADS];
    partial[lane] = sum;
    __syncthreads();
    for (int stride = Q4_THREADS / 2; stride; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    if (lane == 0) y[row_index] = partial[0];
}

/* Two verifier positions against one projection. A CTA still owns one
 * output row, as in q4_fast, but every quantized weight is decoded once and
 * feeds two independent fp32 FMA chains. The lane assignment and reduction
 * tree of each chain are deliberately identical to q4_fast: batching must
 * not turn the qualified mode-1 target into a different numerical kernel. */
__global__ static void q4_fast2(const uint8_t *weights,
                                const uint16_t *scales,
                                const float *x0, const float *x1,
                                float *y0, float *y1,
                                int out, int in, size_t rowbytes)
{
    const int row_index = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    if (row_index >= out || lane >= Q4_THREADS) return;
    const int groups = (in + Q4_GROUP - 1) / Q4_GROUP;
    const uint8_t *row = weights + (size_t)row_index * rowbytes;
    const uint16_t *row_scales = scales + (size_t)row_index * groups;
    float sum0 = 0.0f, sum1 = 0.0f;
    for (int i = lane; i < in; i += Q4_THREADS) {
        const float w = q4_half(row_scales[i / Q4_GROUP]) *
                        (float)q4_at(row, i);
        sum0 = fmaf(w, x0[i], sum0);
        sum1 = fmaf(w, x1[i], sum1);
    }
    __shared__ float partial[2][Q4_THREADS];
    partial[0][lane] = sum0;
    partial[1][lane] = sum1;
    __syncthreads();
    for (int stride = Q4_THREADS / 2; stride; stride >>= 1) {
        if (lane < stride) {
            partial[0][lane] += partial[0][lane + stride];
            partial[1][lane] += partial[1][lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        y0[row_index] = partial[0][0];
        y1[row_index] = partial[1][0];
    }
}

__global__ static void q4_neon_order(const uint8_t *weights,
                                     const uint16_t *scales,
                                     const float *x, float *y,
                                     int out, int in, size_t rowbytes)
{
    const int row_index = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    if (row_index >= out || lane >= 4) return;
    const int groups = (in + Q4_GROUP - 1) / Q4_GROUP;
    const uint8_t *row = weights + (size_t)row_index * rowbytes;
    const uint16_t *row_scales = scales + (size_t)row_index * groups;
    __shared__ float partial[4];
    __shared__ float total;
    if (lane == 0) total = 0.0f;
    __syncwarp(0x0fu);
    for (int group = 0; group < groups; group++) {
        const int begin = group * Q4_GROUP;
        const int limit = min(Q4_GROUP, in - begin);
        float sum = 0.0f;
        for (int i = lane; i < limit; i += 4)
            sum = fmaf((float)q4_at(row, begin + i), x[begin + i], sum);
        partial[lane] = sum;
        __syncwarp(0x0fu);
        if (lane == 0) {
            const float part = (partial[0] + partial[1]) +
                               (partial[2] + partial[3]);
            total += q4_half(row_scales[group]) * part;
        }
        __syncwarp(0x0fu);
    }
    if (lane == 0) y[row_index] = total;
}

/* Build gate and up together so their shared input costs one physical
 * launch per MoE layer. The kind dimension selects consecutive groups of
 * three layer codebooks; dimensions are explicitly accumulated in the same
 * order as the ARM vfmaq loop in model.c. */
__global__ static void vq_build_pair(float *gate_lut, float *up_lut,
                                     const float *books, const float *x,
                                     int nv, int cb_base)
{
    const int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int one = nv * VQ_STAGES * VQ_ENTRIES;
    if (p >= 2 * one) return;
    const int kind = p / one;
    const int q = p - kind * one;
    const int code = q % VQ_ENTRIES;
    const int vs = q / VQ_ENTRIES;
    const int stage = vs % VQ_STAGES;
    const int vector = vs / VQ_STAGES;
    const float *book = books +
        (size_t)(cb_base + kind * VQ_STAGES + stage) *
        VQ_VEC_DIM * VQ_ENTRIES;
    float sum = 0.0f;
#pragma unroll
    for (int d = 0; d < VQ_VEC_DIM; d++)
        sum = fmaf(x[(size_t)vector * VQ_VEC_DIM + d],
                   book[(size_t)d * VQ_ENTRIES + code], sum);
    (kind ? up_lut : gate_lut)[q] = sum;
}

__global__ static void vq_build_one(float *lut, const float *books,
                                    const float *x, int nv, int cb_base)
{
    const int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = nv * VQ_STAGES * VQ_ENTRIES;
    if (p >= total) return;
    const int code = p % VQ_ENTRIES;
    const int vs = p / VQ_ENTRIES;
    const int stage = vs % VQ_STAGES;
    const int vector = vs / VQ_STAGES;
    const float *book = books +
        (size_t)(cb_base + stage) * VQ_VEC_DIM * VQ_ENTRIES;
    float sum = 0.0f;
#pragma unroll
    for (int d = 0; d < VQ_VEC_DIM; d++)
        sum = fmaf(x[(size_t)vector * VQ_VEC_DIM + d],
                   book[(size_t)d * VQ_ENTRIES + code], sum);
    lut[p] = sum;
}

/* A 128-thread CTA owns one 64-row blocked-index tile from each of gate and
 * up. Every thread owns a complete row: vector positions and stages retain
 * their scalar dependency chain, while the two matrices and rows run in
 * parallel. */
__global__ static void vq_apply_pair(float *y,
                                     const uint8_t *gate_idx,
                                     const uint8_t *up_idx,
                                     const uint16_t *scale,
                                     const float *gate_lut,
                                     const float *up_lut,
                                     int rows, int nv)
{
    const int kind = (int)threadIdx.x / VQ_INDEX_BLOCK;
    const int lane = (int)threadIdx.x % VQ_INDEX_BLOCK;
    const int row = (int)blockIdx.x * VQ_INDEX_BLOCK + lane;
    if (kind >= 2 || row >= rows) return;
    const uint8_t *idx = kind ? up_idx : gate_idx;
    const float *lut = kind ? up_lut : gate_lut;
    float acc = 0.0f;
    for (int v = 0; v < nv; v++) {
        const size_t off =
            (((size_t)blockIdx.x * nv + v) * VQ_INDEX_BLOCK + lane) *
            VQ_STAGES;
        const float *block = lut + (size_t)v * VQ_STAGES * VQ_ENTRIES;
        float term = block[idx[off]];
        term = __fadd_rn(term, block[VQ_ENTRIES + idx[off + 1]]);
        term = __fadd_rn(term, block[2 * VQ_ENTRIES + idx[off + 2]]);
        acc = __fadd_rn(acc, term);
    }
    y[(size_t)kind * rows + row] =
        __fmul_rn(acc, q4_half(scale[(size_t)kind * rows + row]));
}

__global__ static void vq_apply_one(float *y, const uint8_t *idx,
                                    const uint16_t *scale, const float *lut,
                                    int rows, int nv)
{
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    const int block_row = row / VQ_INDEX_BLOCK;
    const int lane = row % VQ_INDEX_BLOCK;
    float acc = 0.0f;
    for (int v = 0; v < nv; v++) {
        const size_t off =
            (((size_t)block_row * nv + v) * VQ_INDEX_BLOCK + lane) *
            VQ_STAGES;
        const float *block = lut + (size_t)v * VQ_STAGES * VQ_ENTRIES;
        float term = block[idx[off]];
        term = __fadd_rn(term, block[VQ_ENTRIES + idx[off + 1]]);
        term = __fadd_rn(term, block[2 * VQ_ENTRIES + idx[off + 2]]);
        acc = __fadd_rn(acc, term);
    }
    y[row] = __fmul_rn(acc, q4_half(scale[row]));
}

#if defined(WASTE_CUDA_VQ_FUSED_TEST)
/* Same arithmetic as vq_apply_pair, with blockIdx.y selecting an expert
 * task.  In particular each output owns the same thread, visits vector
 * positions in the same order and uses the same explicit round-to-nearest
 * additions.  The only experiment here is making all task blocks visible to
 * the scheduler in one launch. */
__global__ static void vq_apply_pair_fused_test(
    float *y, const vq_fused_pair_task *tasks,
    const float *gate0, const float *up0,
    const float *gate1, const float *up1,
    int rows, int nv, size_t output_stride)
{
    const int task_index = (int)blockIdx.y;
    const vq_fused_pair_task task = tasks[task_index];
    const int kind = (int)threadIdx.x / VQ_INDEX_BLOCK;
    const int lane = (int)threadIdx.x % VQ_INDEX_BLOCK;
    const int row = (int)blockIdx.x * VQ_INDEX_BLOCK + lane;
    if (kind >= 2 || row >= rows) return;
    const uint8_t *idx = kind ? task.up_idx : task.gate_idx;
    const float *lut = task.lut_row
        ? (kind ? up1 : gate1) : (kind ? up0 : gate0);
    float acc = 0.0f;
    for (int v = 0; v < nv; v++) {
        const size_t off =
            (((size_t)blockIdx.x * nv + v) * VQ_INDEX_BLOCK + lane) *
            VQ_STAGES;
        const float *block = lut + (size_t)v * VQ_STAGES * VQ_ENTRIES;
        float term = block[idx[off]];
        term = __fadd_rn(term, block[VQ_ENTRIES + idx[off + 1]]);
        term = __fadd_rn(term, block[2 * VQ_ENTRIES + idx[off + 2]]);
        acc = __fadd_rn(acc, term);
    }
    y[(size_t)task_index * output_stride + (size_t)kind * rows + row] =
        __fmul_rn(acc, q4_half(task.scale[(size_t)kind * rows + row]));
}

/* Task-major copies of vq_build_one and vq_apply_one.  Every LUT entry and
 * output row retains its original scalar dependency chain; blockIdx.y only
 * provides enough independent work to test the occupancy hypothesis. */
__global__ static void vq_build_one_fused_test(
    float *luts, const float *books, const float *xs,
    int nv, int cb_base, size_t lut_stride, size_t x_stride)
{
    const int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = nv * VQ_STAGES * VQ_ENTRIES;
    if (p >= total) return;
    const int task = (int)blockIdx.y;
    const int code = p % VQ_ENTRIES;
    const int vs = p / VQ_ENTRIES;
    const int stage = vs % VQ_STAGES;
    const int vector = vs / VQ_STAGES;
    const float *book = books +
        (size_t)(cb_base + stage) * VQ_VEC_DIM * VQ_ENTRIES;
    const float *x = xs + (size_t)task * x_stride;
    float sum = 0.0f;
#pragma unroll
    for (int d = 0; d < VQ_VEC_DIM; d++)
        sum = fmaf(x[(size_t)vector * VQ_VEC_DIM + d],
                   book[(size_t)d * VQ_ENTRIES + code], sum);
    luts[(size_t)task * lut_stride + p] = sum;
}

__global__ static void vq_apply_one_fused_test(
    float *y, const vq_fused_down_task *tasks, const float *luts,
    int rows, int nv, size_t output_stride, size_t lut_stride)
{
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    const int task_index = (int)blockIdx.y;
    const vq_fused_down_task task = tasks[task_index];
    const int block_row = row / VQ_INDEX_BLOCK;
    const int lane = row % VQ_INDEX_BLOCK;
    const float *lut = luts + (size_t)task_index * lut_stride;
    float acc = 0.0f;
    for (int v = 0; v < nv; v++) {
        const size_t off =
            (((size_t)block_row * nv + v) * VQ_INDEX_BLOCK + lane) *
            VQ_STAGES;
        const float *block = lut + (size_t)v * VQ_STAGES * VQ_ENTRIES;
        float term = block[task.idx[off]];
        term = __fadd_rn(term,
                         block[VQ_ENTRIES + task.idx[off + 1]]);
        term = __fadd_rn(term,
                         block[2 * VQ_ENTRIES + task.idx[off + 2]]);
        acc = __fadd_rn(acc, term);
    }
    y[(size_t)task_index * output_stride + row] =
        __fmul_rn(acc, q4_half(task.scale[row]));
}
#endif

static void cuda_problem(const char *where, cudaError_t status)
{
    fprintf(stderr, "waste: CUDA KDA %s: %s\n", where,
            cudaGetErrorString(status));
}

static void cuda_vq_group_reset(waste_cuda_kda *ctx)
{
    ctx->vq_group_phase = VQ_GROUP_IDLE;
    ctx->vq_group_count = 0;
    ctx->vq_group_rows = 0;
    ctx->vq_group_cols = 0;
}

/* A grouped-call error is sticky inside this backend as well as in model.c.
 * Synchronizing here is important: expert records are host pointers read by
 * the kernels, so the cache must not be allowed to recycle them while work
 * from a rejected group is still in flight. */
static int cuda_vq_group_abort(waste_cuda_kda *ctx, const char *where,
                               cudaError_t status)
{
    if (status != cudaSuccess) cuda_problem(where, status);
    if (ctx && ctx->stream) {
        const cudaError_t drained = cudaStreamSynchronize(ctx->stream);
        if (drained != cudaSuccess && drained != status)
            cuda_problem("VQ group error drain", drained);
    }
    if (ctx) {
        cuda_vq_group_reset(ctx);
        ctx->vq_group_pair_prepared = 0;
        ctx->vq_group_failed = 1;
    }
    return -1;
}

static pthread_once_t cuda_device_once = PTHREAD_ONCE_INIT;
static cudaError_t cuda_device_status = cudaSuccess;

static void cuda_device_init(void)
{
    /* cudaHostGetDevicePointer requires mapped-host support to be selected
     * before this module creates the device's primary context. */
    cuda_device_status = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (cuda_device_status == cudaSuccess)
        cuda_device_status = cudaSetDevice(0);
}

static waste_cuda_kda *cuda_create(const waste_model *m)
{
    int pageable = 0, host_tables = 0;
    pthread_once(&cuda_device_once, cuda_device_init);
    cudaError_t status = cuda_device_status;
    if (status != cudaSuccess) {
        cuda_problem("mapped-host device initialization", status);
        return NULL;
    }
    status = cudaDeviceGetAttribute(&pageable,
                                    cudaDevAttrPageableMemoryAccess, 0);
    if (status == cudaSuccess)
        status = cudaDeviceGetAttribute(
            &host_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0);
    if (status != cudaSuccess || !pageable || !host_tables) {
        if (status != cudaSuccess) cuda_problem("HMM probe", status);
        else fprintf(stderr, "waste: CUDA KDA needs pageable host-page-table access\n");
        return NULL;
    }

    waste_cuda_kda *ctx = (waste_cuda_kda *)calloc(1, sizeof *ctx);
    if (!ctx) return NULL;
    const size_t channels = (size_t)m->cfg.kda_heads * m->cfg.kda_dim;
    const size_t mla_q = (size_t)m->cfg.n_heads *
                         (size_t)(m->cfg.qk_nope + m->cfg.qk_rope);
    const size_t mla_v = (size_t)m->cfg.n_heads * (size_t)m->cfg.v_head;
    const size_t shared = (size_t)m->cfg.moe_inter *
                          (size_t)(m->cfg.n_shared ? m->cfg.n_shared : 1);
    ctx->capacity = (size_t)m->cfg.hidden;
#define GROW_CAPACITY(n) do { const size_t z = (size_t)(n); \
    if (z > ctx->capacity) ctx->capacity = z; } while (0)
    GROW_CAPACITY(channels);
    GROW_CAPACITY(mla_q);
    GROW_CAPACITY(mla_v);
    GROW_CAPACITY(m->cfg.q_lora);
    GROW_CAPACITY(m->cfg.kv_lora + m->cfg.qk_rope);
    GROW_CAPACITY(m->cfg.latent_dim);
    GROW_CAPACITY(shared);
    GROW_CAPACITY(m->cfg.dense_inter);
#undef GROW_CAPACITY
    if (ctx->capacity > SIZE_MAX / 2 / sizeof(float)) {
        free(ctx);
        return NULL;
    }
    status = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);
    /* Scalar decode uses row zero; the additive verifier primitive uses both
     * rows. Capacity is measured in floats per row. */
    if (status == cudaSuccess)
        status = cudaHostAlloc((void **)&ctx->host_x,
                               2 * ctx->capacity * sizeof(float),
                               cudaHostAllocMapped);
    if (status == cudaSuccess)
        status = cudaHostAlloc((void **)&ctx->host_y,
                               2 * ctx->capacity * sizeof(float),
                               cudaHostAllocMapped);
    if (status == cudaSuccess)
        status = cudaHostGetDevicePointer((void **)&ctx->device_x,
                                          ctx->host_x, 0);
    if (status == cudaSuccess)
        status = cudaHostGetDevicePointer((void **)&ctx->device_y,
                                          ctx->host_y, 0);
    if (status != cudaSuccess) {
        cuda_problem("staging allocation", status);
        if (ctx->host_y) cudaFreeHost(ctx->host_y);
        if (ctx->host_x) cudaFreeHost(ctx->host_x);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
        free(ctx);
        return NULL;
    }
    return ctx;
}

extern "C" int waste_cuda_q4_matvec(waste_model *m, float *y,
                                     const waste_tensor *tensor,
                                     const float *x, int out, int in,
                                     int mode)
{
    if (!m || !tensor || !tensor->q || !tensor->qs ||
        tensor->bits != 4 || tensor->group != Q4_GROUP ||
        out < 1 || in < 1 || (size_t)out > (size_t)INT32_MAX)
        return -1;
    waste_cuda_kda *ctx = (waste_cuda_kda *)m->cuda_kda_ctx;
    if (!ctx) {
        ctx = cuda_create(m);
        if (!ctx) return -1;
        m->cuda_kda_ctx = ctx;
    }
    if ((size_t)in > ctx->capacity || (size_t)out > ctx->capacity) return -1;
    memcpy(ctx->host_x, x, (size_t)in * sizeof(float));
    const size_t rowbytes = tensor->rowbytes;
    if (mode == 2)
        q4_neon_order<<<out, 32, 0, ctx->stream>>>(
            (const uint8_t *)tensor->q, tensor->qs,
            ctx->device_x, ctx->device_y, out, in, rowbytes);
    else
        q4_fast<<<out, Q4_THREADS, 0, ctx->stream>>>(
            (const uint8_t *)tensor->q, tensor->qs,
            ctx->device_x, ctx->device_y, out, in, rowbytes);
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) {
        cuda_problem("projection", status);
        return -1;
    }
    memcpy(y, ctx->host_y, (size_t)out * sizeof(float));
    return 0;
}

/* The two-row kernel is not automatically faster.  These are the shapes
 * whose bracketed GB10 screen beat two qualified mode-1 launches.  Keep this
 * an exact allowlist: small auxiliaries remain sequential until their saved
 * launch time is material.  Wide-to-hidden projections are shape-specific:
 * 4096x8192 regressed, 4096x12288 was only marginal, while 4096x16384 stayed
 * positive in three repeated clean brackets.
 *
 * The three entries not used by the released GLM-5.3 projection set remain
 * here because they were independently measured positive and make the
 * primitive reusable without turning this into a heuristic threshold. */
static int q4_fast2_shape_eligible(int out, int in)
{
    return (out == 8192  && in == 4096) ||
           (out == 8192  && in == 128)  ||
           (out == 1536  && in == 4096) ||
           (out == 16384 && in == 1536) ||
           (out == 4096  && in == 16384) ||
           (out == 2048  && in == 4096) ||
           (out == 4096  && in == 2048) ||
           (out == 4096  && in == 1536) ||
           (out == 4096  && in == 4096) ||
           (out == 12288 && in == 4096) ||
           (out == 12288 && in == 1536);
}

/* Public, side-effect-free dispatch predicate.  A zero is an ordinary
 * verifier fallback, not a CUDA failure.  Requiring the declared tensor
 * geometry here prevents an allowlisted pair of integers from being applied
 * to a differently shaped record. */
extern "C" int waste_cuda_q4_matvec2_eligible(
    const waste_tensor *tensor, int out, int in, int mode)
{
    if (!tensor || !tensor->q || !tensor->qs ||
        tensor->bits != 4 || tensor->group != Q4_GROUP || mode != 1 ||
        tensor->ndim != 2 || tensor->shape[0] != out ||
        tensor->shape[1] != in || out < 1 || in < 1 ||
        tensor->rowbytes < ((size_t)in + 1) / 2)
        return 0;
    return q4_fast2_shape_eligible(out, in);
}

static int q4_matvec2_run(waste_model *m, float *y0, float *y1,
                          const waste_tensor *tensor,
                          const float *x0, const float *x1,
                          int out, int in, int mode, int require_allowlist)
{
    if (!m || !tensor || !tensor->q || !tensor->qs ||
        !x0 || !x1 || !y0 || !y1 ||
        tensor->bits != 4 || tensor->group != Q4_GROUP || mode != 1 ||
        tensor->ndim != 2 || tensor->shape[0] != out ||
        tensor->shape[1] != in || out < 1 || in < 1 ||
        tensor->rowbytes < ((size_t)in + 1) / 2 ||
        (require_allowlist && !q4_fast2_shape_eligible(out, in)))
        return -1;
    waste_cuda_kda *ctx = (waste_cuda_kda *)m->cuda_kda_ctx;
    if (!ctx) {
        ctx = cuda_create(m);
        if (!ctx) return -1;
        m->cuda_kda_ctx = ctx;
    }
    if ((size_t)in > ctx->capacity || (size_t)out > ctx->capacity)
        return -1;
    memcpy(ctx->host_x, x0, (size_t)in * sizeof(float));
    memcpy(ctx->host_x + ctx->capacity, x1,
           (size_t)in * sizeof(float));
    q4_fast2<<<out, Q4_THREADS, 0, ctx->stream>>>(
        (const uint8_t *)tensor->q, tensor->qs,
        ctx->device_x, ctx->device_x + ctx->capacity,
        ctx->device_y, ctx->device_y + ctx->capacity,
        out, in, tensor->rowbytes);
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) {
        cuda_problem("two-row projection", status);
        return -1;
    }
    memcpy(y0, ctx->host_y, (size_t)out * sizeof(float));
    memcpy(y1, ctx->host_y + ctx->capacity,
           (size_t)out * sizeof(float));
    return 0;
}

/* Additive two-row entry point for the bounded MTP verifier. Only the
 * qualified mode-1 arithmetic is implemented: accepting another selector
 * here would silently give the target a prefill-specific numerical meaning.
 * Inputs and outputs are separate because verifier rows live in distinct
 * per-position workspaces.  Call waste_cuda_q4_matvec2_eligible first; this
 * entry also enforces the allowlist so bypassing the predicate fails closed. */
extern "C" int waste_cuda_q4_matvec2(
    waste_model *m, float *y0, float *y1, const waste_tensor *tensor,
    const float *x0, const float *x1, int out, int in, int mode)
{
    return q4_matvec2_run(m, y0, y1, tensor, x0, x1,
                          out, in, mode, 1);
}

#if defined(WASTE_CUDA_Q4_MATVEC2_TEST)
/* The standalone gate exercises arithmetic on every released GLM-5.3 Q4
 * geometry, including shapes that production deliberately routes to two
 * scalar launches.  It is absent from ordinary builds. */
extern "C" int waste_cuda_q4_matvec2_test_only(
    waste_model *m, float *y0, float *y1, const waste_tensor *tensor,
    const float *x0, const float *x1, int out, int in, int mode)
{
    return q4_matvec2_run(m, y0, y1, tensor, x0, x1,
                          out, in, mode, 0);
}
#endif

static void cuda_vq_release(waste_cuda_kda *ctx)
{
    if (!ctx) return;
    if (ctx->stream && ctx->vq_group_phase != VQ_GROUP_IDLE)
        cudaStreamSynchronize(ctx->stream);
#if defined(WASTE_CUDA_VQ_FUSED_TEST)
    if (ctx->vq_fused_down_lut) cudaFree(ctx->vq_fused_down_lut);
    if (ctx->vq_fused_down_tasks) cudaFree(ctx->vq_fused_down_tasks);
    if (ctx->vq_fused_pair_tasks) cudaFree(ctx->vq_fused_pair_tasks);
    ctx->vq_fused_down_lut = NULL;
    ctx->vq_fused_down_tasks = NULL;
    ctx->vq_fused_pair_tasks = NULL;
#endif
    if (ctx->vq_group_down_device_y)
        cudaFree(ctx->vq_group_down_device_y);
    if (ctx->vq_group_down_host_y)
        cudaFreeHost(ctx->vq_group_down_host_y);
    if (ctx->vq_group_down_device_x)
        cudaFree(ctx->vq_group_down_device_x);
    if (ctx->vq_group_down_host_x)
        cudaFreeHost(ctx->vq_group_down_host_x);
    if (ctx->vq_group_pair_device_y)
        cudaFree(ctx->vq_group_pair_device_y);
    if (ctx->vq_group_pair_host_y)
        cudaFreeHost(ctx->vq_group_pair_host_y);
    ctx->vq_group_down_device_y = NULL;
    ctx->vq_group_down_host_y = NULL;
    ctx->vq_group_down_device_x = NULL;
    ctx->vq_group_down_host_x = NULL;
    ctx->vq_group_pair_device_y = NULL;
    ctx->vq_group_pair_host_y = NULL;
    ctx->vq_group_pair_slot_values = 0;
    ctx->vq_group_down_x_slot_values = 0;
    ctx->vq_group_down_y_slot_values = 0;
    cuda_vq_group_reset(ctx);
    ctx->vq_group_pair_prepared = 0;
    ctx->vq_group_failed = 0;
    for (int i = 0; i < VQ_LUT_COUNT; i++) {
        if (ctx->vq_lut[i]) cudaFree(ctx->vq_lut[i]);
        ctx->vq_lut[i] = NULL;
        ctx->vq_lut_values[i] = 0;
    }
    if (ctx->vq_y) cudaFree(ctx->vq_y);
    if (ctx->vq_x) cudaFree(ctx->vq_x);
    if (ctx->vq_books) cudaFree(ctx->vq_books);
    ctx->vq_y = NULL;
    ctx->vq_x = NULL;
    ctx->vq_books = NULL;
    ctx->vq_y_capacity = 0;
    ctx->vq_ready = 0;
}

extern "C" int waste_cuda_vq_init(waste_model *m)
{
    if (!m || !m->codebooksT || m->n_books < 1 ||
        m->stages != VQ_STAGES || m->vec_dim != VQ_VEC_DIM ||
        m->cb_entries != VQ_ENTRIES || m->index_block != VQ_INDEX_BLOCK)
        return -1;
    waste_cuda_kda *ctx = (waste_cuda_kda *)m->cuda_kda_ctx;
    if (!ctx) {
        ctx = cuda_create(m);
        if (!ctx) return -1;
        m->cuda_kda_ctx = ctx;
    }
    if (ctx->vq_ready) return 0;

    const int lat = m->cfg.latent_dim ? m->cfg.latent_dim : m->cfg.hidden;
    const int inter = m->cfg.moe_inter;
    if (lat < 1 || inter < 1 || lat % VQ_VEC_DIM || inter % VQ_VEC_DIM ||
        lat % VQ_INDEX_BLOCK || inter % VQ_INDEX_BLOCK)
        return -1;
    ctx->vq_lut_values[VQ_LUT_GATE0] =
        (size_t)(lat / VQ_VEC_DIM) * VQ_STAGES * VQ_ENTRIES;
    ctx->vq_lut_values[VQ_LUT_UP0] =
        ctx->vq_lut_values[VQ_LUT_GATE0];
    ctx->vq_lut_values[VQ_LUT_DOWN] =
        (size_t)(inter / VQ_VEC_DIM) * VQ_STAGES * VQ_ENTRIES;
    ctx->vq_lut_values[VQ_LUT_GATE1] =
        ctx->vq_lut_values[VQ_LUT_GATE0];
    ctx->vq_lut_values[VQ_LUT_UP1] =
        ctx->vq_lut_values[VQ_LUT_GATE0];
    ctx->vq_y_capacity = (size_t)inter * 2;
    if ((size_t)lat > ctx->vq_y_capacity) ctx->vq_y_capacity = (size_t)lat;
    if (ctx->vq_y_capacity > ctx->capacity) return -1;
    ctx->vq_group_pair_slot_values = (size_t)inter * 2;
    ctx->vq_group_down_x_slot_values = (size_t)inter;
    ctx->vq_group_down_y_slot_values = (size_t)lat;
    if (ctx->vq_group_pair_slot_values >
            SIZE_MAX / VQ_GROUP_MAX / sizeof(float) ||
        ctx->vq_group_down_x_slot_values >
            SIZE_MAX / VQ_GROUP_MAX / sizeof(float) ||
        ctx->vq_group_down_y_slot_values >
            SIZE_MAX / VQ_GROUP_MAX / sizeof(float)
#if defined(WASTE_CUDA_VQ_FUSED_TEST)
        || ctx->vq_lut_values[VQ_LUT_DOWN] >
            SIZE_MAX / VQ_GROUP_MAX / sizeof(float)
#endif
        ) {
        cuda_vq_release(ctx);
        return -1;
    }

    const size_t book_values = (size_t)m->n_books * VQ_VEC_DIM * VQ_ENTRIES;
    const size_t group_pair_bytes = VQ_GROUP_MAX *
        ctx->vq_group_pair_slot_values * sizeof(float);
    const size_t group_down_x_bytes = VQ_GROUP_MAX *
        ctx->vq_group_down_x_slot_values * sizeof(float);
    const size_t group_down_y_bytes = VQ_GROUP_MAX *
        ctx->vq_group_down_y_slot_values * sizeof(float);
    cudaError_t status = cudaMalloc((void **)&ctx->vq_books,
                                    book_values * sizeof(float));
    if (status == cudaSuccess)
        status = cudaMalloc((void **)&ctx->vq_x,
                            2 * ctx->capacity * sizeof(float));
    if (status == cudaSuccess)
        status = cudaMalloc((void **)&ctx->vq_y,
                            ctx->vq_y_capacity * sizeof(float));
    for (int i = 0; status == cudaSuccess && i < VQ_LUT_COUNT; i++)
        status = cudaMalloc((void **)&ctx->vq_lut[i],
                            ctx->vq_lut_values[i] * sizeof(float));
    if (status == cudaSuccess)
        status = cudaHostAlloc((void **)&ctx->vq_group_pair_host_y,
                               group_pair_bytes, cudaHostAllocDefault);
    if (status == cudaSuccess)
        status = cudaMalloc((void **)&ctx->vq_group_pair_device_y,
                            group_pair_bytes);
    if (status == cudaSuccess)
        status = cudaHostAlloc((void **)&ctx->vq_group_down_host_x,
                               group_down_x_bytes, cudaHostAllocDefault);
    if (status == cudaSuccess)
        status = cudaMalloc((void **)&ctx->vq_group_down_device_x,
                            group_down_x_bytes);
    if (status == cudaSuccess)
        status = cudaHostAlloc((void **)&ctx->vq_group_down_host_y,
                               group_down_y_bytes, cudaHostAllocDefault);
    if (status == cudaSuccess)
        status = cudaMalloc((void **)&ctx->vq_group_down_device_y,
                            group_down_y_bytes);
#if defined(WASTE_CUDA_VQ_FUSED_TEST)
    if (status == cudaSuccess)
        status = cudaMalloc((void **)&ctx->vq_fused_pair_tasks,
                            VQ_GROUP_MAX * sizeof(vq_fused_pair_task));
    if (status == cudaSuccess)
        status = cudaMalloc((void **)&ctx->vq_fused_down_tasks,
                            VQ_GROUP_MAX * sizeof(vq_fused_down_task));
    if (status == cudaSuccess)
        status = cudaMalloc((void **)&ctx->vq_fused_down_lut,
                            VQ_GROUP_MAX *
                            ctx->vq_lut_values[VQ_LUT_DOWN] *
                            sizeof(float));
#endif
    if (status == cudaSuccess)
        status = cudaMemcpyAsync(ctx->vq_books, m->codebooksT,
                                 book_values * sizeof(float),
                                 cudaMemcpyHostToDevice, ctx->stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) {
        cuda_problem("VQ allocation/upload", status);
        cuda_vq_release(ctx);
        return -1;
    }
    ctx->vq_ready = 1;
    return 0;
}

extern "C" int waste_cuda_vq_prepare_pair(waste_model *m, int mode,
                                            const float *x,
                                            const float *gate_lut,
                                            const float *up_lut,
                                            int cb_base, int cols)
{
    waste_cuda_kda *ctx = m ? (waste_cuda_kda *)m->cuda_kda_ctx : NULL;
    if (!ctx || !ctx->vq_ready || (mode != 1 && mode != 2) ||
        ctx->vq_group_failed || ctx->vq_group_phase != VQ_GROUP_IDLE ||
        cols < 1 || cols % VQ_VEC_DIM || (size_t)cols > ctx->capacity)
        return -1;
    ctx->vq_group_pair_prepared = 0;
    const size_t values =
        (size_t)(cols / VQ_VEC_DIM) * VQ_STAGES * VQ_ENTRIES;
    if (values > ctx->vq_lut_values[VQ_LUT_GATE0] ||
        values > ctx->vq_lut_values[VQ_LUT_UP0])
        return -1;
    cudaError_t status = cudaSuccess;
    if (mode == 1) {
        if (!gate_lut || !up_lut) return -1;
        status = cudaMemcpyAsync(ctx->vq_lut[VQ_LUT_GATE0], gate_lut,
                                 values * sizeof(float),
                                 cudaMemcpyHostToDevice, ctx->stream);
        if (status == cudaSuccess)
            status = cudaMemcpyAsync(ctx->vq_lut[VQ_LUT_UP0], up_lut,
                                     values * sizeof(float),
                                     cudaMemcpyHostToDevice, ctx->stream);
    } else {
        if (!x || cb_base < 0 ||
            cb_base + 2 * VQ_STAGES > m->n_books)
            return -1;
        memcpy(ctx->host_x, x, (size_t)cols * sizeof(float));
        status = cudaMemcpyAsync(ctx->vq_x, ctx->host_x,
                                 (size_t)cols * sizeof(float),
                                 cudaMemcpyHostToDevice, ctx->stream);
        if (status == cudaSuccess) {
            const int total = (int)(2 * values);
            vq_build_pair<<<(total + VQ_BUILD_THREADS - 1) /
                                 VQ_BUILD_THREADS,
                             VQ_BUILD_THREADS, 0, ctx->stream>>>(
                ctx->vq_lut[VQ_LUT_GATE0], ctx->vq_lut[VQ_LUT_UP0],
                ctx->vq_books,
                ctx->vq_x, cols / VQ_VEC_DIM, cb_base);
            status = cudaGetLastError();
        }
    }
    if (status != cudaSuccess) {
        cuda_problem("VQ gate/up prepare", status);
        return -1;
    }
    if (mode == 2) ctx->vq_group_pair_prepared = 1;
    return 0;
}

/* Prepare the two verifier rows without an intervening synchronization.
 * Their activation vectors and LUTs occupy distinct storage, so the host may
 * enqueue row-0 and row-1 expert records in any interleaving afterward.  The
 * codebook range is shared because both positions traverse the same layer. */
extern "C" int waste_cuda_vq_prepare_pair2(
    waste_model *m, const float *x0, const float *x1,
    int cb_base, int cols)
{
    waste_cuda_kda *ctx = m ? (waste_cuda_kda *)m->cuda_kda_ctx : NULL;
    if (!ctx || !ctx->vq_ready || ctx->vq_group_failed) return -1;
    if (!x0 || !x1 || cb_base < 0 ||
        cb_base + 2 * VQ_STAGES > m->n_books ||
        ctx->vq_group_phase != VQ_GROUP_IDLE || cols < 1 ||
        cols % VQ_VEC_DIM || (size_t)cols > ctx->capacity)
        return cuda_vq_group_abort(ctx, "VQ pair2 prepare arguments",
                                   cudaErrorInvalidValue);
    const size_t values =
        (size_t)(cols / VQ_VEC_DIM) * VQ_STAGES * VQ_ENTRIES;
    if (values != ctx->vq_lut_values[VQ_LUT_GATE0] ||
        values != ctx->vq_lut_values[VQ_LUT_UP0] ||
        values != ctx->vq_lut_values[VQ_LUT_GATE1] ||
        values != ctx->vq_lut_values[VQ_LUT_UP1])
        return cuda_vq_group_abort(ctx, "VQ pair2 LUT geometry",
                                   cudaErrorInvalidValue);

    ctx->vq_group_pair_prepared = 0;
    float *host_x1 = ctx->host_x + ctx->capacity;
    float *device_x1 = ctx->vq_x + ctx->capacity;
    memcpy(ctx->host_x, x0, (size_t)cols * sizeof(float));
    memcpy(host_x1, x1, (size_t)cols * sizeof(float));
    cudaError_t status = cudaMemcpyAsync(
        ctx->vq_x, ctx->host_x, (size_t)cols * sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    if (status == cudaSuccess)
        status = cudaMemcpyAsync(
            device_x1, host_x1, (size_t)cols * sizeof(float),
            cudaMemcpyHostToDevice, ctx->stream);
    const int total = (int)(2 * values);
    if (status == cudaSuccess) {
        vq_build_pair<<<(total + VQ_BUILD_THREADS - 1) /
                            VQ_BUILD_THREADS,
                        VQ_BUILD_THREADS, 0, ctx->stream>>>(
            ctx->vq_lut[VQ_LUT_GATE0], ctx->vq_lut[VQ_LUT_UP0],
            ctx->vq_books, ctx->vq_x, cols / VQ_VEC_DIM, cb_base);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        vq_build_pair<<<(total + VQ_BUILD_THREADS - 1) /
                            VQ_BUILD_THREADS,
                        VQ_BUILD_THREADS, 0, ctx->stream>>>(
            ctx->vq_lut[VQ_LUT_GATE1], ctx->vq_lut[VQ_LUT_UP1],
            ctx->vq_books, device_x1, cols / VQ_VEC_DIM, cb_base);
        status = cudaGetLastError();
    }
    if (status != cudaSuccess)
        return cuda_vq_group_abort(ctx, "VQ pair2 prepare", status);
    ctx->vq_group_pair_prepared = 2;
    return 0;
}

extern "C" int waste_cuda_vq_apply_pair(waste_model *m,
                                          float *gate_y, float *up_y,
                                          const uint8_t *gate_idx,
                                          const uint8_t *up_idx,
                                          const uint16_t *scale,
                                          int rows, int cols)
{
    waste_cuda_kda *ctx = m ? (waste_cuda_kda *)m->cuda_kda_ctx : NULL;
    if (!ctx || !ctx->vq_ready || !gate_y || !up_y || !gate_idx ||
        !up_idx || !scale || ctx->vq_group_failed ||
        ctx->vq_group_phase != VQ_GROUP_IDLE || rows < 1 || cols < 1 ||
        rows % VQ_INDEX_BLOCK || cols % VQ_VEC_DIM ||
        (size_t)(2 * rows) > ctx->vq_y_capacity)
        return -1;
    vq_apply_pair<<<rows / VQ_INDEX_BLOCK, 2 * VQ_INDEX_BLOCK,
                    0, ctx->stream>>>(
        ctx->vq_y, gate_idx, up_idx, scale,
        ctx->vq_lut[VQ_LUT_GATE0], ctx->vq_lut[VQ_LUT_UP0],
        rows, cols / VQ_VEC_DIM);
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess)
        status = cudaMemcpyAsync(ctx->host_y, ctx->vq_y,
                                 (size_t)(2 * rows) * sizeof(float),
                                 cudaMemcpyDeviceToHost, ctx->stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) {
        cuda_problem("VQ gate/up apply", status);
        ctx->vq_group_pair_prepared = 0;
        return -1;
    }
    memcpy(gate_y, ctx->host_y, (size_t)rows * sizeof(float));
    memcpy(up_y, ctx->host_y + rows, (size_t)rows * sizeof(float));
    ctx->vq_group_pair_prepared = 0;
    return 0;
}

/* Grouped mode-2 handoff. The pair LUT is still prepared once with
 * waste_cuda_vq_prepare_pair(); these calls only defer the K per-expert
 * apply handoffs until one contiguous D2H and one stream synchronization.
 * After finish, host_outputs[slot] is valid until the next grouped pair and
 * has the layout [gate rows][up rows]. Slots must arrive as 0,1,...,count-1
 * so the returned storage is contiguous and count is bounded by top-k 16. */
static int cuda_vq_group_pair_enqueue_row(
    waste_model *m, int slot, int lut_row, int prepared_rows,
    const uint8_t *gate_idx, const uint8_t *up_idx,
    const uint16_t *scale, int rows, int cols)
{
    waste_cuda_kda *ctx = m ? (waste_cuda_kda *)m->cuda_kda_ctx : NULL;
    if (!ctx || !ctx->vq_ready || ctx->vq_group_failed) return -1;
    const size_t values = cols > 0 && cols % VQ_VEC_DIM == 0
        ? (size_t)(cols / VQ_VEC_DIM) * VQ_STAGES * VQ_ENTRIES : 0;
    if (!gate_idx || !up_idx || !scale || lut_row < 0 || lut_row > 1 ||
        ctx->vq_group_pair_prepared != prepared_rows ||
        slot < 0 || slot >= VQ_GROUP_MAX || slot != ctx->vq_group_count ||
        rows < 1 || rows % VQ_INDEX_BLOCK || cols < 1 ||
        cols % VQ_VEC_DIM || (size_t)rows * 2 !=
            ctx->vq_group_pair_slot_values ||
        values != ctx->vq_lut_values[VQ_LUT_GATE0] ||
        (ctx->vq_group_phase != VQ_GROUP_IDLE &&
         ctx->vq_group_phase != VQ_GROUP_PAIR) ||
        (ctx->vq_group_phase == VQ_GROUP_PAIR &&
         (rows != ctx->vq_group_rows || cols != ctx->vq_group_cols)))
        return cuda_vq_group_abort(ctx, "VQ pair group arguments",
                                   cudaErrorInvalidValue);
    if (ctx->vq_group_phase == VQ_GROUP_IDLE) {
        if (slot != 0)
            return cuda_vq_group_abort(ctx, "VQ pair group first slot",
                                       cudaErrorInvalidValue);
        ctx->vq_group_phase = VQ_GROUP_PAIR;
        ctx->vq_group_rows = rows;
        ctx->vq_group_cols = cols;
    }

    float *device_y = ctx->vq_group_pair_device_y +
        (size_t)slot * ctx->vq_group_pair_slot_values;
    const int gate_lut = lut_row ? VQ_LUT_GATE1 : VQ_LUT_GATE0;
    const int up_lut = lut_row ? VQ_LUT_UP1 : VQ_LUT_UP0;
    vq_apply_pair<<<rows / VQ_INDEX_BLOCK, 2 * VQ_INDEX_BLOCK,
                    0, ctx->stream>>>(
        device_y, gate_idx, up_idx, scale,
        ctx->vq_lut[gate_lut], ctx->vq_lut[up_lut],
        rows, cols / VQ_VEC_DIM);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return cuda_vq_group_abort(ctx, "VQ pair group enqueue", status);
    ctx->vq_group_count++;
    return 0;
}

extern "C" int waste_cuda_vq_group_pair_enqueue(
    waste_model *m, int slot, const uint8_t *gate_idx,
    const uint8_t *up_idx, const uint16_t *scale, int rows, int cols)
{
    return cuda_vq_group_pair_enqueue_row(
        m, slot, 0, 1, gate_idx, up_idx, scale, rows, cols);
}

/* One task is one verifier row plus one expert record.  Slots, rather than
 * rows, determine result order; callers may enqueue row 0/1 in any order as
 * long as slots themselves are dense and increasing.  Index/scale pointers
 * are coherent host expert-record views read asynchronously by the kernel;
 * their cache holds must remain live through group_pair_finish. */
extern "C" int waste_cuda_vq_group_pair2_enqueue(
    waste_model *m, int slot, int row, const uint8_t *gate_idx,
    const uint8_t *up_idx, const uint16_t *scale, int rows, int cols)
{
    return cuda_vq_group_pair_enqueue_row(
        m, slot, row, 2, gate_idx, up_idx, scale, rows, cols);
}

extern "C" int waste_cuda_vq_group_pair_finish(
    waste_model *m, int count, const float **host_outputs)
{
    waste_cuda_kda *ctx = m ? (waste_cuda_kda *)m->cuda_kda_ctx : NULL;
    if (!ctx || !ctx->vq_ready || ctx->vq_group_failed) return -1;
    if (!host_outputs || ctx->vq_group_phase != VQ_GROUP_PAIR ||
        count < 1 || count > VQ_GROUP_MAX || count != ctx->vq_group_count)
        return cuda_vq_group_abort(ctx, "VQ pair group finish arguments",
                                   cudaErrorInvalidValue);
    const size_t bytes = (size_t)count *
        ctx->vq_group_pair_slot_values * sizeof(float);
    cudaError_t status = cudaMemcpyAsync(
        ctx->vq_group_pair_host_y, ctx->vq_group_pair_device_y, bytes,
        cudaMemcpyDeviceToHost, ctx->stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess)
        return cuda_vq_group_abort(ctx, "VQ pair group finish", status);
    for (int slot = 0; slot < count; slot++)
        host_outputs[slot] = ctx->vq_group_pair_host_y +
            (size_t)slot * ctx->vq_group_pair_slot_values;
    cuda_vq_group_reset(ctx);
    /* Preserve both ordinary prepared=1 and pair2 prepared=2 LUTs across
     * bounded group finishes.  This lets either path split top-k without
     * rebuilding tables.  The next prepare replaces them, while group_drain
     * explicitly retires them after an aborted or caller-ended sequence. */
    return 0;
}

#if defined(WASTE_CUDA_VQ_FUSED_TEST)
/* Synchronous standalone-only gate for a real task-major launch.  Unlike the
 * production grouped API, this does not enqueue one kernel per expert: the
 * task is blockIdx.y in one grid. */
extern "C" int waste_cuda_vq_fused_pair2_test(
    waste_model *m, int count, const int *lut_rows,
    const uint8_t *const *gate_idx, const uint8_t *const *up_idx,
    const uint16_t *const *scale, int rows, int cols,
    const float **host_outputs)
{
    waste_cuda_kda *ctx = m ? (waste_cuda_kda *)m->cuda_kda_ctx : NULL;
    if (!ctx || !ctx->vq_ready || ctx->vq_group_failed) return -1;
    const size_t values = cols > 0 && cols % VQ_VEC_DIM == 0
        ? (size_t)(cols / VQ_VEC_DIM) * VQ_STAGES * VQ_ENTRIES : 0;
    if (!lut_rows || !gate_idx || !up_idx || !scale || !host_outputs ||
        ctx->vq_group_phase != VQ_GROUP_IDLE ||
        ctx->vq_group_pair_prepared != 2 ||
        count < 1 || count > VQ_GROUP_MAX || rows < 1 ||
        rows % VQ_INDEX_BLOCK || cols < 1 || cols % VQ_VEC_DIM ||
        (size_t)(2 * rows) != ctx->vq_group_pair_slot_values ||
        values != ctx->vq_lut_values[VQ_LUT_GATE0])
        return cuda_vq_group_abort(ctx, "VQ fused pair arguments",
                                   cudaErrorInvalidValue);

    vq_fused_pair_task tasks[VQ_GROUP_MAX];
    for (int slot = 0; slot < count; slot++) {
        if ((lut_rows[slot] != 0 && lut_rows[slot] != 1) ||
            !gate_idx[slot] || !up_idx[slot] || !scale[slot])
            return cuda_vq_group_abort(ctx, "VQ fused pair task",
                                       cudaErrorInvalidValue);
        tasks[slot].gate_idx = gate_idx[slot];
        tasks[slot].up_idx = up_idx[slot];
        tasks[slot].scale = scale[slot];
        tasks[slot].lut_row = lut_rows[slot];
    }

    cudaError_t status = cudaMemcpyAsync(
        ctx->vq_fused_pair_tasks, tasks,
        (size_t)count * sizeof *tasks, cudaMemcpyHostToDevice, ctx->stream);
    if (status == cudaSuccess) {
        const dim3 grid((unsigned)(rows / VQ_INDEX_BLOCK),
                        (unsigned)count);
        vq_apply_pair_fused_test<<<grid, 2 * VQ_INDEX_BLOCK,
                                   0, ctx->stream>>>(
            ctx->vq_group_pair_device_y, ctx->vq_fused_pair_tasks,
            ctx->vq_lut[VQ_LUT_GATE0], ctx->vq_lut[VQ_LUT_UP0],
            ctx->vq_lut[VQ_LUT_GATE1], ctx->vq_lut[VQ_LUT_UP1],
            rows, cols / VQ_VEC_DIM, ctx->vq_group_pair_slot_values);
        status = cudaGetLastError();
    }
    const size_t bytes = (size_t)count *
        ctx->vq_group_pair_slot_values * sizeof(float);
    if (status == cudaSuccess)
        status = cudaMemcpyAsync(
            ctx->vq_group_pair_host_y, ctx->vq_group_pair_device_y, bytes,
            cudaMemcpyDeviceToHost, ctx->stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess)
        return cuda_vq_group_abort(ctx, "VQ fused pair", status);
    for (int slot = 0; slot < count; slot++)
        host_outputs[slot] = ctx->vq_group_pair_host_y +
            (size_t)slot * ctx->vq_group_pair_slot_values;
    return 0;
}
#endif

/* Down uses one pinned and one device x slot per expert. This makes the
 * caller free to reuse or modify its activation as soon as enqueue returns.
 * The stream deliberately reuses the one down LUT in build/apply order:
 * H2D[x0], build0, apply0, H2D[x1], build1, apply1, ... . Outputs are
 * distinct, so finish can return them with one contiguous D2H+sync. */
extern "C" int waste_cuda_vq_group_down_enqueue(
    waste_model *m, int slot, const uint8_t *idx, const uint16_t *scale,
    const float *x, int cb_base, int rows, int cols)
{
    waste_cuda_kda *ctx = m ? (waste_cuda_kda *)m->cuda_kda_ctx : NULL;
    if (!ctx || !ctx->vq_ready || ctx->vq_group_failed) return -1;
    const size_t values = cols > 0 && cols % VQ_VEC_DIM == 0
        ? (size_t)(cols / VQ_VEC_DIM) * VQ_STAGES * VQ_ENTRIES : 0;
    if (!idx || !scale || !x || cb_base < 0 ||
        cb_base + VQ_STAGES > m->n_books ||
        slot < 0 || slot >= VQ_GROUP_MAX || slot != ctx->vq_group_count ||
        rows < 1 || rows % VQ_INDEX_BLOCK || cols < 1 ||
        cols % VQ_VEC_DIM || (size_t)rows !=
            ctx->vq_group_down_y_slot_values ||
        (size_t)cols != ctx->vq_group_down_x_slot_values ||
        values != ctx->vq_lut_values[VQ_LUT_DOWN] ||
        (ctx->vq_group_phase != VQ_GROUP_IDLE &&
         ctx->vq_group_phase != VQ_GROUP_DOWN) ||
        (ctx->vq_group_phase == VQ_GROUP_DOWN &&
         (rows != ctx->vq_group_rows || cols != ctx->vq_group_cols)))
        return cuda_vq_group_abort(ctx, "VQ down group arguments",
                                   cudaErrorInvalidValue);
    if (ctx->vq_group_phase == VQ_GROUP_IDLE) {
        if (slot != 0)
            return cuda_vq_group_abort(ctx, "VQ down group first slot",
                                       cudaErrorInvalidValue);
        ctx->vq_group_phase = VQ_GROUP_DOWN;
        ctx->vq_group_rows = rows;
        ctx->vq_group_cols = cols;
    }

    float *host_x = ctx->vq_group_down_host_x +
        (size_t)slot * ctx->vq_group_down_x_slot_values;
    float *device_x = ctx->vq_group_down_device_x +
        (size_t)slot * ctx->vq_group_down_x_slot_values;
    float *device_y = ctx->vq_group_down_device_y +
        (size_t)slot * ctx->vq_group_down_y_slot_values;
    memcpy(host_x, x, (size_t)cols * sizeof(float));
    cudaError_t status = cudaMemcpyAsync(
        device_x, host_x, (size_t)cols * sizeof(float),
        cudaMemcpyHostToDevice, ctx->stream);
    if (status == cudaSuccess) {
        const int total = (int)values;
        vq_build_one<<<(total + VQ_BUILD_THREADS - 1) / VQ_BUILD_THREADS,
                        VQ_BUILD_THREADS, 0, ctx->stream>>>(
            ctx->vq_lut[VQ_LUT_DOWN], ctx->vq_books, device_x,
            cols / VQ_VEC_DIM, cb_base);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        vq_apply_one<<<(rows + VQ_DOWN_THREADS - 1) / VQ_DOWN_THREADS,
                        VQ_DOWN_THREADS, 0, ctx->stream>>>(
            device_y, idx, scale, ctx->vq_lut[VQ_LUT_DOWN], rows,
            cols / VQ_VEC_DIM);
        status = cudaGetLastError();
    }
    if (status != cudaSuccess)
        return cuda_vq_group_abort(ctx, "VQ down group enqueue", status);
    ctx->vq_group_count++;
    return 0;
}

extern "C" int waste_cuda_vq_group_down_finish(
    waste_model *m, int count, const float **host_outputs)
{
    waste_cuda_kda *ctx = m ? (waste_cuda_kda *)m->cuda_kda_ctx : NULL;
    if (!ctx || !ctx->vq_ready || ctx->vq_group_failed) return -1;
    if (!host_outputs || ctx->vq_group_phase != VQ_GROUP_DOWN ||
        count < 1 || count > VQ_GROUP_MAX || count != ctx->vq_group_count)
        return cuda_vq_group_abort(ctx, "VQ down group finish arguments",
                                   cudaErrorInvalidValue);
    const size_t bytes = (size_t)count *
        ctx->vq_group_down_y_slot_values * sizeof(float);
    cudaError_t status = cudaMemcpyAsync(
        ctx->vq_group_down_host_y, ctx->vq_group_down_device_y, bytes,
        cudaMemcpyDeviceToHost, ctx->stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess)
        return cuda_vq_group_abort(ctx, "VQ down group finish", status);
    for (int slot = 0; slot < count; slot++)
        host_outputs[slot] = ctx->vq_group_down_host_y +
            (size_t)slot * ctx->vq_group_down_y_slot_values;
    cuda_vq_group_reset(ctx);
    return 0;
}

#if defined(WASTE_CUDA_VQ_FUSED_TEST)
extern "C" int waste_cuda_vq_fused_down_test(
    waste_model *m, int count, const uint8_t *const *idx,
    const uint16_t *const *scale, const float *const *x,
    int cb_base, int rows, int cols, const float **host_outputs)
{
    waste_cuda_kda *ctx = m ? (waste_cuda_kda *)m->cuda_kda_ctx : NULL;
    if (!ctx || !ctx->vq_ready || ctx->vq_group_failed) return -1;
    const size_t values = cols > 0 && cols % VQ_VEC_DIM == 0
        ? (size_t)(cols / VQ_VEC_DIM) * VQ_STAGES * VQ_ENTRIES : 0;
    if (!idx || !scale || !x || !host_outputs || cb_base < 0 ||
        cb_base + VQ_STAGES > m->n_books ||
        ctx->vq_group_phase != VQ_GROUP_IDLE ||
        count < 1 || count > VQ_GROUP_MAX || rows < 1 ||
        rows % VQ_INDEX_BLOCK || cols < 1 || cols % VQ_VEC_DIM ||
        (size_t)rows != ctx->vq_group_down_y_slot_values ||
        (size_t)cols != ctx->vq_group_down_x_slot_values ||
        values != ctx->vq_lut_values[VQ_LUT_DOWN])
        return cuda_vq_group_abort(ctx, "VQ fused down arguments",
                                   cudaErrorInvalidValue);

    vq_fused_down_task tasks[VQ_GROUP_MAX];
    for (int slot = 0; slot < count; slot++) {
        if (!idx[slot] || !scale[slot] || !x[slot])
            return cuda_vq_group_abort(ctx, "VQ fused down task",
                                       cudaErrorInvalidValue);
        tasks[slot].idx = idx[slot];
        tasks[slot].scale = scale[slot];
        memcpy(ctx->vq_group_down_host_x +
                   (size_t)slot * ctx->vq_group_down_x_slot_values,
               x[slot], (size_t)cols * sizeof(float));
    }

    cudaError_t status = cudaMemcpyAsync(
        ctx->vq_fused_down_tasks, tasks,
        (size_t)count * sizeof *tasks, cudaMemcpyHostToDevice, ctx->stream);
    if (status == cudaSuccess)
        status = cudaMemcpyAsync(
            ctx->vq_group_down_device_x, ctx->vq_group_down_host_x,
            (size_t)count * ctx->vq_group_down_x_slot_values * sizeof(float),
            cudaMemcpyHostToDevice, ctx->stream);
    if (status == cudaSuccess) {
        const dim3 grid((unsigned)((values + VQ_BUILD_THREADS - 1) /
                                   VQ_BUILD_THREADS),
                        (unsigned)count);
        vq_build_one_fused_test<<<grid, VQ_BUILD_THREADS, 0, ctx->stream>>>(
            ctx->vq_fused_down_lut, ctx->vq_books,
            ctx->vq_group_down_device_x, cols / VQ_VEC_DIM, cb_base,
            ctx->vq_lut_values[VQ_LUT_DOWN],
            ctx->vq_group_down_x_slot_values);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        const dim3 grid((unsigned)((rows + VQ_DOWN_THREADS - 1) /
                                   VQ_DOWN_THREADS),
                        (unsigned)count);
        vq_apply_one_fused_test<<<grid, VQ_DOWN_THREADS, 0, ctx->stream>>>(
            ctx->vq_group_down_device_y, ctx->vq_fused_down_tasks,
            ctx->vq_fused_down_lut, rows, cols / VQ_VEC_DIM,
            ctx->vq_group_down_y_slot_values,
            ctx->vq_lut_values[VQ_LUT_DOWN]);
        status = cudaGetLastError();
    }
    const size_t bytes = (size_t)count *
        ctx->vq_group_down_y_slot_values * sizeof(float);
    if (status == cudaSuccess)
        status = cudaMemcpyAsync(
            ctx->vq_group_down_host_y, ctx->vq_group_down_device_y, bytes,
            cudaMemcpyDeviceToHost, ctx->stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess)
        return cuda_vq_group_abort(ctx, "VQ fused down", status);
    for (int slot = 0; slot < count; slot++)
        host_outputs[slot] = ctx->vq_group_down_host_y +
            (size_t)slot * ctx->vq_group_down_y_slot_values;
    return 0;
}
#endif

/* Used when CPU-side collection fails after some kernels were enqueued.
 * Successful drain is reusable; a CUDA error remains sticky and is never
 * converted into a CPU fallback. */
extern "C" int waste_cuda_vq_group_drain(waste_model *m)
{
    waste_cuda_kda *ctx = m ? (waste_cuda_kda *)m->cuda_kda_ctx : NULL;
    if (!ctx) return m ? 0 : -1;
    const cudaError_t status = cudaStreamSynchronize(ctx->stream);
    cuda_vq_group_reset(ctx);
    ctx->vq_group_pair_prepared = 0;
    if (status != cudaSuccess) {
        cuda_problem("VQ group drain", status);
        ctx->vq_group_failed = 1;
        return -1;
    }
    return ctx->vq_group_failed ? -1 : 0;
}

extern "C" int waste_cuda_vq_apply_down(waste_model *m, int mode,
                                          float *y, const uint8_t *idx,
                                          const uint16_t *scale,
                                          const float *x,
                                          const float *cpu_lut,
                                          int cb_base, int rows, int cols)
{
    waste_cuda_kda *ctx = m ? (waste_cuda_kda *)m->cuda_kda_ctx : NULL;
    if (!ctx || !ctx->vq_ready || (mode != 1 && mode != 2) || !y || !idx ||
        !scale || ctx->vq_group_failed ||
        ctx->vq_group_phase != VQ_GROUP_IDLE ||
        rows < 1 || cols < 1 || rows % VQ_INDEX_BLOCK ||
        cols % VQ_VEC_DIM || (size_t)rows > ctx->vq_y_capacity ||
        (size_t)cols > ctx->capacity)
        return -1;
    const size_t values =
        (size_t)(cols / VQ_VEC_DIM) * VQ_STAGES * VQ_ENTRIES;
    if (values > ctx->vq_lut_values[VQ_LUT_DOWN]) return -1;
    cudaError_t status = cudaSuccess;
    if (mode == 1) {
        if (!cpu_lut) return -1;
        status = cudaMemcpyAsync(ctx->vq_lut[VQ_LUT_DOWN], cpu_lut,
                                 values * sizeof(float),
                                 cudaMemcpyHostToDevice, ctx->stream);
    } else {
        if (!x || cb_base < 0 || cb_base + VQ_STAGES > m->n_books)
            return -1;
        memcpy(ctx->host_x, x, (size_t)cols * sizeof(float));
        status = cudaMemcpyAsync(ctx->vq_x, ctx->host_x,
                                 (size_t)cols * sizeof(float),
                                 cudaMemcpyHostToDevice, ctx->stream);
        if (status == cudaSuccess) {
            const int total = (int)values;
            vq_build_one<<<(total + VQ_BUILD_THREADS - 1) /
                                VQ_BUILD_THREADS,
                            VQ_BUILD_THREADS, 0, ctx->stream>>>(
                ctx->vq_lut[VQ_LUT_DOWN], ctx->vq_books, ctx->vq_x,
                cols / VQ_VEC_DIM, cb_base);
            status = cudaGetLastError();
        }
    }
    if (status == cudaSuccess) {
        vq_apply_one<<<(rows + VQ_DOWN_THREADS - 1) / VQ_DOWN_THREADS,
                        VQ_DOWN_THREADS, 0, ctx->stream>>>(
            ctx->vq_y, idx, scale, ctx->vq_lut[VQ_LUT_DOWN], rows,
            cols / VQ_VEC_DIM);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess)
        status = cudaMemcpyAsync(ctx->host_y, ctx->vq_y,
                                 (size_t)rows * sizeof(float),
                                 cudaMemcpyDeviceToHost, ctx->stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) {
        cuda_problem("VQ down apply", status);
        return -1;
    }
    memcpy(y, ctx->host_y, (size_t)rows * sizeof(float));
    return 0;
}

extern "C" void waste_cuda_kda_free(waste_model *m)
{
    if (!m || !m->cuda_kda_ctx) return;
    waste_cuda_kda *ctx = (waste_cuda_kda *)m->cuda_kda_ctx;
    cudaStreamSynchronize(ctx->stream);
    cuda_vq_release(ctx);
    cudaFreeHost(ctx->host_y);
    cudaFreeHost(ctx->host_x);
    cudaStreamDestroy(ctx->stream);
    free(ctx);
    m->cuda_kda_ctx = NULL;
}

/* The generic dispatch table is intentionally untouched: this experiment is
 * per-model and decode-only, while waste_k is process-global. */
extern "C" const char *waste_register_cuda(waste_kernels *)
{
    return NULL;
}
