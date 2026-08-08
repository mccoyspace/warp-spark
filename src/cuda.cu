// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 SQLite Cloud, Inc.
/* Experimental GB10 CUDA path: WASTE Q4G matvec against pageable trunk
 * storage and scheme-specific VQ3R/VQ4P expert gather. The trunk and bounded
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
#include "vq_packed.h"

#include <cuda_runtime.h>

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    Q4_GROUP = 128,
    Q4_THREADS = 128,
    VQ3R_STAGES = 3,
    VQ3R_VEC_DIM = 8,
    VQ3R_ENTRIES = 256,
    VQ4P_STAGES = 4,
    VQ4P_VEC_DIM = 8,
    VQ4P_ENTRIES = 64,
    VQ_INDEX_BLOCK = WASTE_VQ_INDEX_BLOCK,
    VQ_BUILD_THREADS = 256,
    VQ_DOWN_THREADS = 256,
    VQ_GROUP_MAX = 16,
    VQ_GROUP_IDLE = 0,
    VQ_GROUP_PAIR = 1,
    VQ_GROUP_DOWN = 2
};

typedef struct {
    cudaStream_t stream;
    float *host_x, *host_y;
    float *device_x, *device_y;
    size_t capacity;
    float *vq_books, *vq_x, *vq_y;
    float *vq_lut[3];
    int8_t *vq_lut8[3];
    float *vq_lut_scale[3];
    const int8_t *vq_lut8_host[3];
    const float *vq_lut_scale_host[3];
    size_t vq_lut_values[3];
    size_t vq_lut_scale_values[3];
    int vq_lut_mode[3];
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
    waste_vq_scheme vq_scheme;
    int vq_ready;
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
    const int one = nv * VQ3R_STAGES * VQ3R_ENTRIES;
    if (p >= 2 * one) return;
    const int kind = p / one;
    const int q = p - kind * one;
    const int code = q % VQ3R_ENTRIES;
    const int vs = q / VQ3R_ENTRIES;
    const int stage = vs % VQ3R_STAGES;
    const int vector = vs / VQ3R_STAGES;
    const float *book = books +
        (size_t)(cb_base + kind * VQ3R_STAGES + stage) *
        VQ3R_VEC_DIM * VQ3R_ENTRIES;
    float sum = 0.0f;
#pragma unroll
    for (int d = 0; d < VQ3R_VEC_DIM; d++)
        sum = fmaf(x[(size_t)vector * VQ3R_VEC_DIM + d],
                   book[(size_t)d * VQ3R_ENTRIES + code], sum);
    (kind ? up_lut : gate_lut)[q] = sum;
}

__global__ static void vq_build_one(float *lut, const float *books,
                                    const float *x, int nv, int cb_base)
{
    const int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = nv * VQ3R_STAGES * VQ3R_ENTRIES;
    if (p >= total) return;
    const int code = p % VQ3R_ENTRIES;
    const int vs = p / VQ3R_ENTRIES;
    const int stage = vs % VQ3R_STAGES;
    const int vector = vs / VQ3R_STAGES;
    const float *book = books +
        (size_t)(cb_base + stage) * VQ3R_VEC_DIM * VQ3R_ENTRIES;
    float sum = 0.0f;
#pragma unroll
    for (int d = 0; d < VQ3R_VEC_DIM; d++)
        sum = fmaf(x[(size_t)vector * VQ3R_VEC_DIM + d],
                   book[(size_t)d * VQ3R_ENTRIES + code], sum);
    lut[p] = sum;
}

/* VQ4P mode 2: one CTA owns one 32-vector scale block.  It builds the
 * fp32 table, reduces max(abs(table)), publishes the block scale and then
 * quantizes every entry.  Max is order-independent for the finite tables
 * admitted by preflight, while each dot retains the CPU's eight-FMA order.
 * grid.y is one for down and two for the gate/up pair. */
__global__ static void vq4p_build_quant(
    float *lut0, float *lut1, int8_t *q0, int8_t *q1,
    float *scale0, float *scale1, const float *books, const float *x,
    int nv, int cb_base)
{
    const int kind = (int)blockIdx.y;
    float *lut = kind ? lut1 : lut0;
    int8_t *q = kind ? q1 : q0;
    float *scale = kind ? scale1 : scale0;
    const int v0 = (int)blockIdx.x * WASTE_VQ_LUT_BLK;
    const int vn = min(WASTE_VQ_LUT_BLK, nv - v0);
    const int n = vn * VQ4P_STAGES * VQ4P_ENTRIES;
    const size_t base = (size_t)v0 * VQ4P_STAGES * VQ4P_ENTRIES;
    float local_max = 0.0f;
    for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
        const int code = i % VQ4P_ENTRIES;
        const int vs = i / VQ4P_ENTRIES;
        const int stage = vs % VQ4P_STAGES;
        const int vector = v0 + vs / VQ4P_STAGES;
        const float *book = books +
            (size_t)(cb_base + kind * VQ4P_STAGES + stage) *
            VQ4P_VEC_DIM * VQ4P_ENTRIES;
        float sum = 0.0f;
#pragma unroll
        for (int d = 0; d < VQ4P_VEC_DIM; d++)
            sum = __fmaf_rn(x[(size_t)vector * VQ4P_VEC_DIM + d],
                            book[(size_t)d * VQ4P_ENTRIES + code], sum);
        lut[base + i] = sum;
        local_max = fmaxf(local_max, fabsf(sum));
    }
    __shared__ float maxima[VQ_BUILD_THREADS];
    __shared__ float inverse;
    maxima[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = VQ_BUILD_THREADS / 2; stride; stride >>= 1) {
        if ((int)threadIdx.x < stride)
            maxima[threadIdx.x] =
                fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scale[blockIdx.x] = __fdiv_rn(maxima[0], 127.0f);
        inverse = maxima[0] > 0.0f
            ? __fdiv_rn(127.0f, maxima[0]) : 0.0f;
    }
    __syncthreads();
    for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
        int value = __float2int_rn(__fmul_rn(lut[base + i], inverse));
        value = max(-127, min(127, value));
        q[base + i] = (int8_t)value;
    }
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
            VQ3R_STAGES;
        const float *block = lut + (size_t)v * VQ3R_STAGES * VQ3R_ENTRIES;
        float term = block[idx[off]];
        term = __fadd_rn(term, block[VQ3R_ENTRIES + idx[off + 1]]);
        term = __fadd_rn(term, block[2 * VQ3R_ENTRIES + idx[off + 2]]);
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
            VQ3R_STAGES;
        const float *block = lut + (size_t)v * VQ3R_STAGES * VQ3R_ENTRIES;
        float term = block[idx[off]];
        term = __fadd_rn(term, block[VQ3R_ENTRIES + idx[off + 1]]);
        term = __fadd_rn(term, block[2 * VQ3R_ENTRIES + idx[off + 2]]);
        acc = __fadd_rn(acc, term);
    }
    y[row] = __fmul_rn(acc, q4_half(scale[row]));
}

/* VQ4P spends precision only at the 32-vector block folds.  Everything
 * inside one block is signed-integer addition, so threads may reorder it
 * without changing a bit; this first kernel keeps one row per thread and
 * the transparent scalar order while the crossover is established.  The
 * only ordered obligations are the fp32 folds and final channel scale. */
__global__ static void vq4p_apply_pair(float *y,
                                       const uint8_t *gate_idx,
                                       const uint8_t *up_idx,
                                       const uint16_t *scale,
                                       const int8_t *gate_lut,
                                       const int8_t *up_lut,
                                       const float *gate_lscale,
                                       const float *up_lscale,
                                       int rows, int nv)
{
    const int kind = (int)threadIdx.x / VQ_INDEX_BLOCK;
    const int lane = (int)threadIdx.x % VQ_INDEX_BLOCK;
    const int row = (int)blockIdx.x * VQ_INDEX_BLOCK + lane;
    if (kind >= 2 || row >= rows) return;
    const uint8_t *idx = kind ? up_idx : gate_idx;
    const int8_t *lut = kind ? up_lut : gate_lut;
    const float *lscale = kind ? up_lscale : gate_lscale;
    float acc = 0.0f;
    for (int v0 = 0; v0 < nv; v0 += WASTE_VQ_LUT_BLK) {
        const int v1 = min(v0 + WASTE_VQ_LUT_BLK, nv);
        int sum = 0;
        for (int v = v0; v < v1; v++) {
            const size_t off =
                (((size_t)blockIdx.x * nv + v) * VQ_INDEX_BLOCK + lane) * 3;
            const unsigned b0 = idx[off];
            const unsigned b1 = idx[off + 1];
            const unsigned b2 = idx[off + 2];
            const int8_t *table =
                lut + (size_t)v * VQ4P_STAGES * VQ4P_ENTRIES;
            sum += table[WASTE_P6_J0(b0, b1, b2)];
            sum += table[VQ4P_ENTRIES + WASTE_P6_J1(b0, b1, b2)];
            sum += table[2 * VQ4P_ENTRIES + WASTE_P6_J2(b0, b1, b2)];
            sum += table[3 * VQ4P_ENTRIES + WASTE_P6_J3(b0, b1, b2)];
        }
        /* GCC 13 emits scalar FMADD for the CPU fold at -O2.  Spell it out:
         * CUDA is otherwise built with contraction disabled, and a separate
         * multiply/add would miss the strict contract at rounding edges. */
        acc = __fmaf_rn((float)sum,
                        lscale[v0 / WASTE_VQ_LUT_BLK], acc);
    }
    y[(size_t)kind * rows + row] =
        __fmul_rn(acc, q4_half(scale[(size_t)kind * rows + row]));
}

__global__ static void vq4p_apply_one(float *y, const uint8_t *idx,
                                      const uint16_t *scale,
                                      const int8_t *lut,
                                      const float *lscale,
                                      int rows, int nv)
{
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;
    const int block_row = row / VQ_INDEX_BLOCK;
    const int lane = row % VQ_INDEX_BLOCK;
    float acc = 0.0f;
    for (int v0 = 0; v0 < nv; v0 += WASTE_VQ_LUT_BLK) {
        const int v1 = min(v0 + WASTE_VQ_LUT_BLK, nv);
        int sum = 0;
        for (int v = v0; v < v1; v++) {
            const size_t off =
                (((size_t)block_row * nv + v) * VQ_INDEX_BLOCK + lane) * 3;
            const unsigned b0 = idx[off];
            const unsigned b1 = idx[off + 1];
            const unsigned b2 = idx[off + 2];
            const int8_t *table =
                lut + (size_t)v * VQ4P_STAGES * VQ4P_ENTRIES;
            sum += table[WASTE_P6_J0(b0, b1, b2)];
            sum += table[VQ4P_ENTRIES + WASTE_P6_J1(b0, b1, b2)];
            sum += table[2 * VQ4P_ENTRIES + WASTE_P6_J2(b0, b1, b2)];
            sum += table[3 * VQ4P_ENTRIES + WASTE_P6_J3(b0, b1, b2)];
        }
        acc = __fmaf_rn((float)sum,
                        lscale[v0 / WASTE_VQ_LUT_BLK], acc);
    }
    y[row] = __fmul_rn(acc, q4_half(scale[row]));
}

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
    status = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);
    if (status == cudaSuccess)
        status = cudaHostAlloc((void **)&ctx->host_x,
                               ctx->capacity * sizeof(float),
                               cudaHostAllocMapped);
    if (status == cudaSuccess)
        status = cudaHostAlloc((void **)&ctx->host_y,
                               ctx->capacity * sizeof(float),
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

static void cuda_vq_release(waste_cuda_kda *ctx)
{
    if (!ctx) return;
    if (ctx->stream && ctx->vq_group_phase != VQ_GROUP_IDLE)
        cudaStreamSynchronize(ctx->stream);
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
    for (int i = 0; i < 3; i++) {
        if (ctx->vq_lut[i]) cudaFree(ctx->vq_lut[i]);
        if (ctx->vq_lut8[i]) cudaFree(ctx->vq_lut8[i]);
        if (ctx->vq_lut_scale[i]) cudaFree(ctx->vq_lut_scale[i]);
        ctx->vq_lut[i] = NULL;
        ctx->vq_lut8[i] = NULL;
        ctx->vq_lut_scale[i] = NULL;
        ctx->vq_lut_values[i] = 0;
        ctx->vq_lut_scale_values[i] = 0;
        ctx->vq_lut_mode[i] = 0;
        ctx->vq_lut8_host[i] = NULL;
        ctx->vq_lut_scale_host[i] = NULL;
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
        (m->vq_scheme != WASTE_VQ_SCHEME_VQ3R &&
         m->vq_scheme != WASTE_VQ_SCHEME_VQ4P))
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
    if (lat < 1 || inter < 1 || lat % m->vec_dim || inter % m->vec_dim ||
        lat % VQ_INDEX_BLOCK || inter % VQ_INDEX_BLOCK)
        return -1;
    ctx->vq_lut_values[0] =
        (size_t)(lat / m->vec_dim) * m->stages * m->cb_entries;
    ctx->vq_lut_values[1] = ctx->vq_lut_values[0];
    ctx->vq_lut_values[2] =
        (size_t)(inter / m->vec_dim) * m->stages * m->cb_entries;
    ctx->vq_lut_scale_values[0] = ctx->vq_lut_scale_values[1] =
        (size_t)((lat / m->vec_dim + WASTE_VQ_LUT_BLK - 1) /
                 WASTE_VQ_LUT_BLK);
    ctx->vq_lut_scale_values[2] =
        (size_t)((inter / m->vec_dim + WASTE_VQ_LUT_BLK - 1) /
                 WASTE_VQ_LUT_BLK);
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
            SIZE_MAX / VQ_GROUP_MAX / sizeof(float)) {
        cuda_vq_release(ctx);
        return -1;
    }

    const size_t book_values =
        (size_t)m->n_books * m->vec_dim * m->cb_entries;
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
                            ctx->capacity * sizeof(float));
    if (status == cudaSuccess)
        status = cudaMalloc((void **)&ctx->vq_y,
                            ctx->vq_y_capacity * sizeof(float));
    for (int i = 0; status == cudaSuccess && i < 3; i++)
        status = cudaMalloc((void **)&ctx->vq_lut[i],
                            ctx->vq_lut_values[i] * sizeof(float));
    if (m->vq_scheme == WASTE_VQ_SCHEME_VQ4P)
        for (int i = 0; status == cudaSuccess && i < 3; i++) {
            status = cudaMalloc((void **)&ctx->vq_lut8[i],
                                ctx->vq_lut_values[i]);
            if (status == cudaSuccess)
                status = cudaMalloc((void **)&ctx->vq_lut_scale[i],
                                    ctx->vq_lut_scale_values[i] *
                                    sizeof(float));
        }
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
    ctx->vq_scheme = m->vq_scheme;
    return 0;
}

extern "C" int waste_cuda_vq_prepare_pair(waste_model *m, int mode,
                                            const float *x,
                                            waste_vq_lut_view gate_lut,
                                            waste_vq_lut_view up_lut,
                                            int cb_base, int cols)
{
    waste_cuda_kda *ctx = m ? (waste_cuda_kda *)m->cuda_kda_ctx : NULL;
    if (!ctx || !ctx->vq_ready || (mode != 1 && mode != 2) ||
        ctx->vq_group_failed || ctx->vq_group_phase != VQ_GROUP_IDLE ||
        cols < 1 || cols % m->vec_dim || (size_t)cols > ctx->capacity)
        return -1;
    ctx->vq_group_pair_prepared = 0;
    const size_t values =
        (size_t)(cols / m->vec_dim) * m->stages * m->cb_entries;
    if (values > ctx->vq_lut_values[0] || values > ctx->vq_lut_values[1])
        return -1;
    if (ctx->vq_scheme == WASTE_VQ_SCHEME_VQ4P) {
        if (mode == 1) {
            if (!gate_lut.i8 || !up_lut.i8 ||
                !gate_lut.block_scale || !up_lut.block_scale)
                return -1;
            /* GB10 HMM makes these ordinary CPU allocations directly
             * visible to the GPU, just like expert-record pointers. */
            ctx->vq_lut8_host[0] = gate_lut.i8;
            ctx->vq_lut8_host[1] = up_lut.i8;
            ctx->vq_lut_scale_host[0] = gate_lut.block_scale;
            ctx->vq_lut_scale_host[1] = up_lut.block_scale;
        } else {
            if (!x || cb_base < 0 ||
                cb_base + 2 * VQ4P_STAGES > m->n_books)
                return -1;
            memcpy(ctx->host_x, x, (size_t)cols * sizeof(float));
            cudaError_t status = cudaMemcpyAsync(
                ctx->vq_x, ctx->host_x, (size_t)cols * sizeof(float),
                cudaMemcpyHostToDevice, ctx->stream);
            if (status == cudaSuccess) {
                const dim3 grid(
                    (unsigned)((cols / m->vec_dim + WASTE_VQ_LUT_BLK - 1) /
                               WASTE_VQ_LUT_BLK), 2, 1);
                vq4p_build_quant<<<grid, VQ_BUILD_THREADS, 0, ctx->stream>>>(
                    ctx->vq_lut[0], ctx->vq_lut[1],
                    ctx->vq_lut8[0], ctx->vq_lut8[1],
                    ctx->vq_lut_scale[0], ctx->vq_lut_scale[1],
                    ctx->vq_books, ctx->vq_x, cols / m->vec_dim, cb_base);
                status = cudaGetLastError();
            }
            if (status != cudaSuccess) {
                cuda_problem("VQ4P gate/up build", status);
                return -1;
            }
        }
        ctx->vq_lut_mode[0] = ctx->vq_lut_mode[1] = mode;
        ctx->vq_group_pair_prepared = 1;
        return 0;
    }
    if (ctx->vq_scheme != WASTE_VQ_SCHEME_VQ3R) return -1;
    cudaError_t status = cudaSuccess;
    if (mode == 1) {
        if (!gate_lut.f32 || !up_lut.f32) return -1;
        status = cudaMemcpyAsync(ctx->vq_lut[0], gate_lut.f32,
                                 values * sizeof(float),
                                 cudaMemcpyHostToDevice, ctx->stream);
        if (status == cudaSuccess)
            status = cudaMemcpyAsync(ctx->vq_lut[1], up_lut.f32,
                                     values * sizeof(float),
                                     cudaMemcpyHostToDevice, ctx->stream);
    } else {
        if (!x || cb_base < 0 ||
            cb_base + 2 * m->stages > m->n_books)
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
                ctx->vq_lut[0], ctx->vq_lut[1], ctx->vq_books,
                ctx->vq_x, cols / m->vec_dim, cb_base);
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
        rows % VQ_INDEX_BLOCK || cols % m->vec_dim ||
        (size_t)(2 * rows) > ctx->vq_y_capacity)
        return -1;
    if (ctx->vq_scheme == WASTE_VQ_SCHEME_VQ3R) {
        vq_apply_pair<<<rows / VQ_INDEX_BLOCK, 2 * VQ_INDEX_BLOCK,
                        0, ctx->stream>>>(
            ctx->vq_y, gate_idx, up_idx, scale,
            ctx->vq_lut[0], ctx->vq_lut[1], rows, cols / m->vec_dim);
    } else if (ctx->vq_scheme == WASTE_VQ_SCHEME_VQ4P &&
               ctx->vq_group_pair_prepared && ctx->vq_lut_mode[0] &&
               ctx->vq_lut_mode[0] == ctx->vq_lut_mode[1]) {
        const int mode = ctx->vq_lut_mode[0];
        const int8_t *gate_lut = mode == 1
            ? ctx->vq_lut8_host[0] : ctx->vq_lut8[0];
        const int8_t *up_lut = mode == 1
            ? ctx->vq_lut8_host[1] : ctx->vq_lut8[1];
        const float *gate_scale = mode == 1
            ? ctx->vq_lut_scale_host[0] : ctx->vq_lut_scale[0];
        const float *up_scale = mode == 1
            ? ctx->vq_lut_scale_host[1] : ctx->vq_lut_scale[1];
        if (!gate_lut || !up_lut || !gate_scale || !up_scale) return -1;
        vq4p_apply_pair<<<rows / VQ_INDEX_BLOCK, 2 * VQ_INDEX_BLOCK,
                          0, ctx->stream>>>(
            ctx->vq_y, gate_idx, up_idx, scale,
            gate_lut, up_lut, gate_scale, up_scale,
            rows, cols / m->vec_dim);
    } else {
        return -1;
    }
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
    /* VQ4P mode 1 borrows the CPU-built pair once per MoE layer and applies
     * it to every routed expert.  The storage remains owned and unchanged
     * until the next prepare call; clearing this after expert zero made the
     * second expert fail despite a successful exactness preflight. */
    if (ctx->vq_scheme != WASTE_VQ_SCHEME_VQ4P)
        ctx->vq_group_pair_prepared = 0;
    return 0;
}

/* Preflight-only inspection of mode-2's three intermediate contracts.
 * End-to-end equality would catch most errors, but comparing these directly
 * makes a failure local: dot order, max/scale, or byte quantization. */
extern "C" int waste_cuda_vq_check_lut(waste_model *m, int slot,
                                         waste_vq_lut_view reference,
                                         int cols)
{
    waste_cuda_kda *ctx = m ? (waste_cuda_kda *)m->cuda_kda_ctx : NULL;
    if (!ctx || ctx->vq_scheme != WASTE_VQ_SCHEME_VQ4P ||
        slot < 0 || slot > 2 || ctx->vq_lut_mode[slot] != 2 ||
        !reference.f32 || !reference.i8 || !reference.block_scale ||
        cols < 1 || cols % m->vec_dim)
        return -1;
    const size_t values =
        (size_t)(cols / m->vec_dim) * m->stages * m->cb_entries;
    const size_t scales =
        (size_t)((cols / m->vec_dim + WASTE_VQ_LUT_BLK - 1) /
                 WASTE_VQ_LUT_BLK);
    if (values > ctx->vq_lut_values[slot] ||
        scales > ctx->vq_lut_scale_values[slot])
        return -1;
    float *f32 = (float *)malloc(values * sizeof(float));
    int8_t *i8 = (int8_t *)malloc(values);
    float *block_scale = (float *)malloc(scales * sizeof(float));
    if (!f32 || !i8 || !block_scale) {
        free(f32); free(i8); free(block_scale);
        return -1;
    }
    cudaError_t status = cudaStreamSynchronize(ctx->stream);
    if (status == cudaSuccess)
        status = cudaMemcpy(f32, ctx->vq_lut[slot], values * sizeof(float),
                            cudaMemcpyDeviceToHost);
    if (status == cudaSuccess)
        status = cudaMemcpy(i8, ctx->vq_lut8[slot], values,
                            cudaMemcpyDeviceToHost);
    if (status == cudaSuccess)
        status = cudaMemcpy(block_scale, ctx->vq_lut_scale[slot],
                            scales * sizeof(float), cudaMemcpyDeviceToHost);
    const int different = status != cudaSuccess ||
        memcmp(f32, reference.f32, values * sizeof(float)) ||
        memcmp(i8, reference.i8, values) ||
        memcmp(block_scale, reference.block_scale,
               scales * sizeof(float));
    if (status != cudaSuccess)
        cuda_problem("VQ4P LUT verification", status);
    else if (different)
        fprintf(stderr,
                "waste: CUDA VQ4P mode-2 LUT contract failed at slot %d\n",
                slot);
    free(f32); free(i8); free(block_scale);
    return different ? -1 : 0;
}

/* Grouped mode-2 handoff. The pair LUT is still prepared once with
 * waste_cuda_vq_prepare_pair(); these calls only defer the K per-expert
 * apply handoffs until one contiguous D2H and one stream synchronization.
 * After finish, host_outputs[slot] is valid until the next grouped pair and
 * has the layout [gate rows][up rows]. Slots must arrive as 0,1,...,count-1
 * so the returned storage is contiguous and count is bounded by top-k 16. */
extern "C" int waste_cuda_vq_group_pair_enqueue(
    waste_model *m, int slot, const uint8_t *gate_idx,
    const uint8_t *up_idx, const uint16_t *scale, int rows, int cols)
{
    waste_cuda_kda *ctx = m ? (waste_cuda_kda *)m->cuda_kda_ctx : NULL;
    if (!ctx || !ctx->vq_ready || ctx->vq_group_failed ||
        ctx->vq_scheme != WASTE_VQ_SCHEME_VQ3R) return -1;
    const size_t values = cols > 0 && cols % VQ3R_VEC_DIM == 0
        ? (size_t)(cols / VQ3R_VEC_DIM) * VQ3R_STAGES * VQ3R_ENTRIES : 0;
    if (!gate_idx || !up_idx || !scale || !ctx->vq_group_pair_prepared ||
        slot < 0 || slot >= VQ_GROUP_MAX || slot != ctx->vq_group_count ||
        rows < 1 || rows % VQ_INDEX_BLOCK || cols < 1 ||
        cols % VQ3R_VEC_DIM || (size_t)rows * 2 !=
            ctx->vq_group_pair_slot_values ||
        values != ctx->vq_lut_values[0] ||
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
    vq_apply_pair<<<rows / VQ_INDEX_BLOCK, 2 * VQ_INDEX_BLOCK,
                    0, ctx->stream>>>(
        device_y, gate_idx, up_idx, scale,
        ctx->vq_lut[0], ctx->vq_lut[1], rows, cols / VQ3R_VEC_DIM);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return cuda_vq_group_abort(ctx, "VQ pair group enqueue", status);
    ctx->vq_group_count++;
    return 0;
}

extern "C" int waste_cuda_vq_group_pair_finish(
    waste_model *m, int count, const float **host_outputs)
{
    waste_cuda_kda *ctx = m ? (waste_cuda_kda *)m->cuda_kda_ctx : NULL;
    if (!ctx || !ctx->vq_ready || ctx->vq_group_failed ||
        ctx->vq_scheme != WASTE_VQ_SCHEME_VQ3R) return -1;
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
    return 0;
}

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
    if (!ctx || !ctx->vq_ready || ctx->vq_group_failed ||
        ctx->vq_scheme != WASTE_VQ_SCHEME_VQ3R) return -1;
    const size_t values = cols > 0 && cols % VQ3R_VEC_DIM == 0
        ? (size_t)(cols / VQ3R_VEC_DIM) * VQ3R_STAGES * VQ3R_ENTRIES : 0;
    if (!idx || !scale || !x || cb_base < 0 ||
        cb_base + VQ3R_STAGES > m->n_books ||
        slot < 0 || slot >= VQ_GROUP_MAX || slot != ctx->vq_group_count ||
        rows < 1 || rows % VQ_INDEX_BLOCK || cols < 1 ||
        cols % VQ3R_VEC_DIM || (size_t)rows !=
            ctx->vq_group_down_y_slot_values ||
        (size_t)cols != ctx->vq_group_down_x_slot_values ||
        values != ctx->vq_lut_values[2] ||
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
            ctx->vq_lut[2], ctx->vq_books, device_x,
            cols / VQ3R_VEC_DIM, cb_base);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        vq_apply_one<<<(rows + VQ_DOWN_THREADS - 1) / VQ_DOWN_THREADS,
                        VQ_DOWN_THREADS, 0, ctx->stream>>>(
            device_y, idx, scale, ctx->vq_lut[2], rows,
            cols / VQ3R_VEC_DIM);
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
    if (!ctx || !ctx->vq_ready || ctx->vq_group_failed ||
        ctx->vq_scheme != WASTE_VQ_SCHEME_VQ3R) return -1;
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
                                          waste_vq_lut_view cpu_lut,
                                          int cb_base, int rows, int cols)
{
    waste_cuda_kda *ctx = m ? (waste_cuda_kda *)m->cuda_kda_ctx : NULL;
    if (!ctx || !ctx->vq_ready || (mode != 1 && mode != 2) || !y || !idx ||
        !scale || ctx->vq_group_failed ||
        ctx->vq_group_phase != VQ_GROUP_IDLE ||
        rows < 1 || cols < 1 || rows % VQ_INDEX_BLOCK ||
        cols % m->vec_dim || (size_t)rows > ctx->vq_y_capacity ||
        (size_t)cols > ctx->capacity)
        return -1;
    const size_t values =
        (size_t)(cols / m->vec_dim) * m->stages * m->cb_entries;
    if (values > ctx->vq_lut_values[2]) return -1;
    if (ctx->vq_scheme == WASTE_VQ_SCHEME_VQ4P) {
        const int8_t *lut8 = NULL;
        const float *lscale = NULL;
        cudaError_t status = cudaSuccess;
        if (mode == 1) {
            if (!cpu_lut.i8 || !cpu_lut.block_scale) return -1;
            lut8 = cpu_lut.i8;
            lscale = cpu_lut.block_scale;
        } else {
            if (!x || cb_base < 0 ||
                cb_base + VQ4P_STAGES > m->n_books)
                return -1;
            memcpy(ctx->host_x, x, (size_t)cols * sizeof(float));
            status = cudaMemcpyAsync(
                ctx->vq_x, ctx->host_x, (size_t)cols * sizeof(float),
                cudaMemcpyHostToDevice, ctx->stream);
            if (status == cudaSuccess) {
                const dim3 grid(
                    (unsigned)((cols / m->vec_dim + WASTE_VQ_LUT_BLK - 1) /
                               WASTE_VQ_LUT_BLK), 1, 1);
                vq4p_build_quant<<<grid, VQ_BUILD_THREADS, 0, ctx->stream>>>(
                    ctx->vq_lut[2], NULL, ctx->vq_lut8[2], NULL,
                    ctx->vq_lut_scale[2], NULL, ctx->vq_books, ctx->vq_x,
                    cols / m->vec_dim, cb_base);
                status = cudaGetLastError();
            }
            if (status != cudaSuccess) {
                cuda_problem("VQ4P down build", status);
                return -1;
            }
            lut8 = ctx->vq_lut8[2];
            lscale = ctx->vq_lut_scale[2];
        }
        ctx->vq_lut_mode[2] = mode;
        vq4p_apply_one<<<(rows + VQ_DOWN_THREADS - 1) / VQ_DOWN_THREADS,
                          VQ_DOWN_THREADS, 0, ctx->stream>>>(
            ctx->vq_y, idx, scale, lut8, lscale,
            rows, cols / m->vec_dim);
        status = cudaGetLastError();
        if (status == cudaSuccess)
            status = cudaMemcpyAsync(ctx->host_y, ctx->vq_y,
                                     (size_t)rows * sizeof(float),
                                     cudaMemcpyDeviceToHost, ctx->stream);
        if (status == cudaSuccess) status = cudaStreamSynchronize(ctx->stream);
        if (status != cudaSuccess) {
            cuda_problem("VQ4P down apply", status);
            return -1;
        }
        memcpy(y, ctx->host_y, (size_t)rows * sizeof(float));
        return 0;
    }
    if (ctx->vq_scheme != WASTE_VQ_SCHEME_VQ3R) return -1;
    cudaError_t status = cudaSuccess;
    if (mode == 1) {
        if (!cpu_lut.f32) return -1;
        status = cudaMemcpyAsync(ctx->vq_lut[2], cpu_lut.f32,
                                 values * sizeof(float),
                                 cudaMemcpyHostToDevice, ctx->stream);
    } else {
        if (!x || cb_base < 0 || cb_base + m->stages > m->n_books)
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
                ctx->vq_lut[2], ctx->vq_books, ctx->vq_x,
                cols / m->vec_dim, cb_base);
            status = cudaGetLastError();
        }
    }
    if (status == cudaSuccess) {
        vq_apply_one<<<(rows + VQ_DOWN_THREADS - 1) / VQ_DOWN_THREADS,
                        VQ_DOWN_THREADS, 0, ctx->stream>>>(
            ctx->vq_y, idx, scale, ctx->vq_lut[2], rows,
            cols / m->vec_dim);
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
