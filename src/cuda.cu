// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 SQLite Cloud, Inc.
/* Experimental GB10 CUDA path: WASTE Q4G matvec against pageable trunk
 * storage. The 29 GiB trunk stays in its existing host allocation, so the
 * expert cache and CUDA do not discover one another through reclaim.
 *
 * This module deliberately implements one operation. Decode KDA calls it
 * directly; prefill and every non-KDA path remain on the qualified CPU.
 */

#include "model.h"
#include "waste_backend.h"

#include <cuda_runtime.h>

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { Q4_GROUP = 128, Q4_THREADS = 128 };

typedef struct {
    cudaStream_t stream;
    float *host_x, *host_y;
    float *device_x, *device_y;
    size_t capacity;
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
    ctx->capacity = channels > (size_t)m->cfg.hidden
                  ? channels : (size_t)m->cfg.hidden;
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

extern "C" void waste_cuda_kda_free(waste_model *m)
{
    if (!m || !m->cuda_kda_ctx) return;
    waste_cuda_kda *ctx = (waste_cuda_kda *)m->cuda_kda_ctx;
    cudaStreamSynchronize(ctx->stream);
    cudaFreeHost(ctx->host_y);
    cudaFreeHost(ctx->host_x);
    cudaStreamDestroy(ctx->stream);
    free(ctx);
    m->cuda_kda_ctx = NULL;
}

/* The generic dispatch table is intentionally untouched: this experiment is
 * per-model and KDA-only, while waste_k is process-global. */
extern "C" const char *waste_register_cuda(waste_kernels *)
{
    return NULL;
}
