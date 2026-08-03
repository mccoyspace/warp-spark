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
    VQ_INDEX_BLOCK = 64,
    VQ_BUILD_THREADS = 256,
    VQ_DOWN_THREADS = 256
};

typedef struct {
    cudaStream_t stream;
    float *host_x, *host_y;
    float *device_x, *device_y;
    size_t capacity;
    float *vq_books, *vq_x, *vq_y;
    float *vq_lut[3];
    size_t vq_lut_values[3];
    size_t vq_y_capacity;
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

static void cuda_problem(const char *where, cudaError_t status)
{
    fprintf(stderr, "waste: CUDA KDA %s: %s\n", where,
            cudaGetErrorString(status));
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
    for (int i = 0; i < 3; i++) {
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
        m->cb_entries != VQ_ENTRIES)
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
    ctx->vq_lut_values[0] =
        (size_t)(lat / VQ_VEC_DIM) * VQ_STAGES * VQ_ENTRIES;
    ctx->vq_lut_values[1] = ctx->vq_lut_values[0];
    ctx->vq_lut_values[2] =
        (size_t)(inter / VQ_VEC_DIM) * VQ_STAGES * VQ_ENTRIES;
    ctx->vq_y_capacity = (size_t)inter * 2;
    if ((size_t)lat > ctx->vq_y_capacity) ctx->vq_y_capacity = (size_t)lat;
    if (ctx->vq_y_capacity > ctx->capacity) return -1;

    const size_t book_values = (size_t)m->n_books * VQ_VEC_DIM * VQ_ENTRIES;
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
        cols < 1 || cols % VQ_VEC_DIM || (size_t)cols > ctx->capacity)
        return -1;
    const size_t values =
        (size_t)(cols / VQ_VEC_DIM) * VQ_STAGES * VQ_ENTRIES;
    if (values > ctx->vq_lut_values[0] || values > ctx->vq_lut_values[1])
        return -1;
    cudaError_t status = cudaSuccess;
    if (mode == 1) {
        if (!gate_lut || !up_lut) return -1;
        status = cudaMemcpyAsync(ctx->vq_lut[0], gate_lut,
                                 values * sizeof(float),
                                 cudaMemcpyHostToDevice, ctx->stream);
        if (status == cudaSuccess)
            status = cudaMemcpyAsync(ctx->vq_lut[1], up_lut,
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
                ctx->vq_lut[0], ctx->vq_lut[1], ctx->vq_books,
                ctx->vq_x, cols / VQ_VEC_DIM, cb_base);
            status = cudaGetLastError();
        }
    }
    if (status != cudaSuccess) {
        cuda_problem("VQ gate/up prepare", status);
        return -1;
    }
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
        !up_idx || !scale || rows < 1 || cols < 1 ||
        rows % VQ_INDEX_BLOCK || cols % VQ_VEC_DIM ||
        (size_t)(2 * rows) > ctx->vq_y_capacity)
        return -1;
    vq_apply_pair<<<rows / VQ_INDEX_BLOCK, 2 * VQ_INDEX_BLOCK,
                    0, ctx->stream>>>(
        ctx->vq_y, gate_idx, up_idx, scale,
        ctx->vq_lut[0], ctx->vq_lut[1], rows, cols / VQ_VEC_DIM);
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess)
        status = cudaMemcpyAsync(ctx->host_y, ctx->vq_y,
                                 (size_t)(2 * rows) * sizeof(float),
                                 cudaMemcpyDeviceToHost, ctx->stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(ctx->stream);
    if (status != cudaSuccess) {
        cuda_problem("VQ gate/up apply", status);
        return -1;
    }
    memcpy(gate_y, ctx->host_y, (size_t)rows * sizeof(float));
    memcpy(up_y, ctx->host_y + rows, (size_t)rows * sizeof(float));
    return 0;
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
        !scale || rows < 1 || cols < 1 || rows % VQ_INDEX_BLOCK ||
        cols % VQ_VEC_DIM || (size_t)rows > ctx->vq_y_capacity ||
        (size_t)cols > ctx->capacity)
        return -1;
    const size_t values =
        (size_t)(cols / VQ_VEC_DIM) * VQ_STAGES * VQ_ENTRIES;
    if (values > ctx->vq_lut_values[2]) return -1;
    cudaError_t status = cudaSuccess;
    if (mode == 1) {
        if (!cpu_lut) return -1;
        status = cudaMemcpyAsync(ctx->vq_lut[2], cpu_lut,
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
                ctx->vq_lut[2], ctx->vq_books, ctx->vq_x,
                cols / VQ_VEC_DIM, cb_base);
            status = cudaGetLastError();
        }
    }
    if (status == cudaSuccess) {
        vq_apply_one<<<(rows + VQ_DOWN_THREADS - 1) / VQ_DOWN_THREADS,
                        VQ_DOWN_THREADS, 0, ctx->stream>>>(
            ctx->vq_y, idx, scale, ctx->vq_lut[2], rows,
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
