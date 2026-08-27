// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 SQLite Cloud, Inc.
/* Source-separated GB10 provider: promoted group-one Q4G projections and
 * strict-order VQ3R gather through waste_backend_v1. */

#include "waste_cuda_policy.h"

#include <cuda_runtime.h>

#include <pthread.h>
#include <stdarg.h>
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
    VQ_DOWN_THREADS = 256,
};

typedef struct {
    waste_cuda_policy policy;
    const waste_backend_model *model;
    const waste_backend_tensor **claimed;
    size_t n_claimed;

    cudaStream_t stream;
    float *host_x, *host_y;
    float *device_x, *device_y;
    float *vq_books, *vq_x, *vq_y;
    float *vq_lut[3];

    uint32_t prepared_cols;
    uint32_t prepared_gate_base;
    uint32_t prepared_up_base;
    int vq_prepared;
    char error[256];
} waste_cuda_ctx;

__device__ static float fp16_to_float(uint16_t h)
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
        sum = fmaf(fp16_to_float(row_scales[i / Q4_GROUP]) *
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

__global__ static void q4_cpu_order(const uint8_t *weights,
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
            total += fp16_to_float(row_scales[group]) * part;
        }
        __syncwarp(0x0fu);
    }
    if (lane == 0) y[row_index] = total;
}

/* Gate and up share an input, but their public matrix views do not promise
 * adjacent codebook ranges or adjacent scale arrays. */
__global__ static void vq_build_pair(float *gate_lut, float *up_lut,
                                     const float *books, const float *x,
                                     int nv, int gate_base, int up_base)
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
    const int base = kind ? up_base : gate_base;
    const float *book = books +
        (size_t)(base + stage) * VQ_VEC_DIM * VQ_ENTRIES;
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

/* Each thread owns a complete row. Vector positions and stages retain the
 * ordered fp32 chain; only independent rows and matrices run in parallel. */
__global__ static void vq_apply_pair(float *y,
                                     const uint8_t *gate_idx,
                                     const uint8_t *up_idx,
                                     const uint16_t *gate_scale,
                                     const uint16_t *up_scale,
                                     const float *gate_lut,
                                     const float *up_lut,
                                     int rows, int nv)
{
    const int kind = (int)threadIdx.x / VQ_INDEX_BLOCK;
    const int lane = (int)threadIdx.x % VQ_INDEX_BLOCK;
    const int row = (int)blockIdx.x * VQ_INDEX_BLOCK + lane;
    if (kind >= 2 || row >= rows) return;
    const uint8_t *idx = kind ? up_idx : gate_idx;
    const uint16_t *scale = kind ? up_scale : gate_scale;
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
        __fmul_rn(acc, fp16_to_float(scale[row]));
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
    y[row] = __fmul_rn(acc, fp16_to_float(scale[row]));
}

static void set_error(waste_cuda_ctx *ctx, const char *fmt, ...)
{
    if (!ctx) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ctx->error, sizeof ctx->error, fmt, ap);
    va_end(ap);
}

static waste_status cuda_error(waste_cuda_ctx *ctx, const char *where,
                               cudaError_t status)
{
    set_error(ctx, "%s: %s", where, cudaGetErrorString(status));
    return status == cudaErrorMemoryAllocation ? WASTE_E_OOM : WASTE_E_BACKEND;
}

static int ends_with(const char *text, const char *suffix)
{
    if (!text || !suffix) return 0;
    const size_t nt = strlen(text), ns = strlen(suffix);
    return nt >= ns && !memcmp(text + nt - ns, suffix, ns + 1);
}

static const waste_backend_tensor *find_suffix(
    const waste_backend_model *model, const char *suffix)
{
    const waste_backend_tensor *found = NULL;
    for (size_t i = 0; i < model->n_tensors; i++) {
        const waste_backend_tensor *t = &model->tensors[i];
        if (!ends_with(t->name, suffix)) continue;
        if (found) return NULL; /* an ambiguous suffix is never safe */
        found = t;
    }
    return found;
}

static int layer_suffix(char *dst, size_t cap, uint32_t layer,
                        const char *tail)
{
    const int n = snprintf(dst, cap, "model.layers.%u.%s", layer, tail);
    return n > 0 && (size_t)n < cap;
}

static int append_target(waste_cuda_ctx *ctx,
                         const waste_backend_tensor *tensor,
                         size_t rows, size_t cols, const char *suffix)
{
    if (!tensor || !waste_cuda_policy_q4_tensor(tensor) ||
        tensor->rows != rows || tensor->cols != cols ||
        rows > ctx->policy.capacity || cols > ctx->policy.capacity) {
        set_error(ctx, "missing or incompatible Q4G target: %s", suffix);
        return 0;
    }
    for (size_t i = 0; i < ctx->n_claimed; i++)
        if (ctx->claimed[i] == tensor) return 1;
    ctx->claimed[ctx->n_claimed++] = tensor;
    return 1;
}

static int require_target(waste_cuda_ctx *ctx, uint32_t layer,
                          const char *tail, size_t rows, size_t cols)
{
    char suffix[192];
    if (!layer_suffix(suffix, sizeof suffix, layer, tail)) return 0;
    return append_target(ctx, find_suffix(ctx->model, suffix),
                         rows, cols, suffix);
}

static const waste_backend_tensor *optional_target(waste_cuda_ctx *ctx,
                                                    uint32_t layer,
                                                    const char *tail,
                                                    char suffix[192])
{
    if (!layer_suffix(suffix, 192, layer, tail)) return NULL;
    return find_suffix(ctx->model, suffix);
}

static int build_target_set(waste_cuda_ctx *ctx)
{
    const waste_backend_model_info *m = &ctx->model->info;
    ctx->claimed = (const waste_backend_tensor **)calloc(
        ctx->model->n_tensors ? ctx->model->n_tensors : 1,
        sizeof *ctx->claimed);
    if (!ctx->claimed) {
        set_error(ctx, "Q4G claim table allocation failed");
        return 0;
    }

    uint32_t kda_count = 0, moe_count = 0, dense_count = 0;
    for (uint32_t L = 0; L < m->n_layers; L++) {
        char suffix[192];
        const waste_backend_tensor *fa = optional_target(
            ctx, L, "self_attn.f_a_proj.weight", suffix);
        if (fa) {
            /* Exact K3 recurrent-attention geometry. */
            kda_count++;
            if (!require_target(ctx, L, "self_attn.q_proj.weight", 12288, 7168) ||
                !require_target(ctx, L, "self_attn.k_proj.weight", 12288, 7168) ||
                !require_target(ctx, L, "self_attn.v_proj.weight", 12288, 7168) ||
                !append_target(ctx, fa, 128, 7168, suffix) ||
                !require_target(ctx, L, "self_attn.f_b_proj.weight", 12288, 128) ||
                !require_target(ctx, L, "self_attn.b_proj.weight", 96, 7168) ||
                !require_target(ctx, L, "self_attn.o_proj.weight", 7168, 12288))
                return 0;
            char full_name[192], low_a_name[192], low_b_name[192];
            const waste_backend_tensor *full = optional_target(
                ctx, L, "self_attn.g_proj.weight", full_name);
            const waste_backend_tensor *low_a = optional_target(
                ctx, L, "self_attn.g_a_proj.weight", low_a_name);
            const waste_backend_tensor *low_b = optional_target(
                ctx, L, "self_attn.g_b_proj.weight", low_b_name);
            if (full && !low_a && !low_b) {
                if (!append_target(ctx, full, 12288, 7168, full_name)) return 0;
            } else if (!full && low_a && low_b) {
                if (!append_target(ctx, low_a, 128, 7168, low_a_name) ||
                    !append_target(ctx, low_b, 12288, 128, low_b_name)) return 0;
            } else {
                set_error(ctx, "ambiguous or incomplete KDA gate at layer %u", L);
                return 0;
            }
        } else {
            /* MLA scope 2: absorbed kv_b remains CPU. */
            const size_t q_rows = (size_t)m->n_heads *
                                  (m->qk_nope + m->qk_rope);
            const size_t o_cols = (size_t)m->n_heads * m->v_head;
            if (!require_target(ctx, L, "self_attn.q_a_proj.weight",
                                m->q_lora, m->hidden) ||
                !require_target(ctx, L, "self_attn.q_b_proj.weight",
                                q_rows, m->q_lora) ||
                !require_target(ctx, L, "self_attn.kv_a_proj_with_mqa.weight",
                                m->kv_lora + m->qk_rope, m->hidden) ||
                !require_target(ctx, L, "self_attn.o_proj.weight",
                                m->hidden, o_cols))
                return 0;
            char gate_name[192];
            const waste_backend_tensor *gate = optional_target(
                ctx, L, "self_attn.g_proj.weight", gate_name);
            if (gate && !append_target(ctx, gate, o_cols, m->hidden,
                                       gate_name)) return 0;
        }

        char router_name[192];
        const waste_backend_tensor *router = optional_target(
            ctx, L, "block_sparse_moe.gate.weight", router_name);
        if (router) {
            moe_count++;
            const size_t shared = (size_t)m->moe_inter * m->n_shared;
            if (m->latent_dim &&
                (!require_target(ctx, L,
                    "block_sparse_moe.routed_expert_down_proj.weight",
                    m->latent_dim, m->hidden) ||
                 !require_target(ctx, L,
                    "block_sparse_moe.routed_expert_up_proj.weight",
                    m->hidden, m->latent_dim)))
                return 0;
            if (!require_target(ctx, L,
                    "block_sparse_moe.shared_experts.gate_proj.weight",
                    shared, m->hidden) ||
                !require_target(ctx, L,
                    "block_sparse_moe.shared_experts.up_proj.weight",
                    shared, m->hidden) ||
                !require_target(ctx, L,
                    "block_sparse_moe.shared_experts.down_proj.weight",
                    m->hidden, shared))
                return 0;
        } else {
            dense_count++;
            /* K2's promoted scope 3 includes its non-MoE dense FFN. K3's
             * promoted scope 2 deliberately leaves that FFN on CPU. */
            if (ctx->policy.model_kind == WASTE_CUDA_MODEL_K2 &&
                (!require_target(ctx, L, "mlp.gate_proj.weight",
                                 m->dense_inter, m->hidden) ||
                 !require_target(ctx, L, "mlp.up_proj.weight",
                                 m->dense_inter, m->hidden) ||
                 !require_target(ctx, L, "mlp.down_proj.weight",
                                 m->hidden, m->dense_inter)))
                return 0;
        }
    }
    if (kda_count != m->n_kda_layers ||
        moe_count != m->n_layers - m->first_dense ||
        dense_count != m->first_dense || !ctx->n_claimed) {
        set_error(ctx,
                  "incomplete projection set (KDA %u/%u, MoE %u, dense %u)",
                  kda_count, m->n_kda_layers, moe_count, dense_count);
        return 0;
    }
    return 1;
}

static pthread_once_t cuda_device_once = PTHREAD_ONCE_INIT;
static cudaError_t cuda_device_status = cudaSuccess;

static void cuda_device_init(void)
{
    /* cudaSetDeviceFlags applies to the calling thread's current device.
     * Select the only device v1 accepts before setting process-wide primary
     * context policy; every operational callback selects it again below. */
    cuda_device_status = cudaSetDevice(0);
    if (cuda_device_status == cudaSuccess)
        cuda_device_status = cudaSetDeviceFlags(cudaDeviceMapHost);
}

static waste_status activate_device(waste_cuda_ctx *ctx,
                                    const char *operation)
{
    if (!ctx) return WASTE_E_ARG;
    const cudaError_t s = cudaSetDevice(ctx->policy.device_ordinal);
    return s == cudaSuccess ? WASTE_OK : cuda_error(ctx, operation, s);
}

static waste_status initialize_device(waste_cuda_ctx *ctx)
{
    int pageable = 0, host_tables = 0;
    pthread_once(&cuda_device_once, cuda_device_init);
    if (cuda_device_status != cudaSuccess)
        return cuda_error(ctx, "mapped-host device initialization",
                          cuda_device_status);
    waste_status active = activate_device(ctx, "CUDA device activation");
    if (active != WASTE_OK) return active;
    cudaError_t s = cudaDeviceGetAttribute(
        &pageable, cudaDevAttrPageableMemoryAccess, 0);
    if (s == cudaSuccess)
        s = cudaDeviceGetAttribute(
            &host_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0);
    if (s != cudaSuccess) return cuda_error(ctx, "coherent-memory probe", s);
    if (!pageable || !host_tables) {
        set_error(ctx, "device 0 lacks coherent pageable host-page-table access");
        return WASTE_E_UNSUPPORTED;
    }
    s = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);
    if (s == cudaSuccess)
        s = cudaHostAlloc((void **)&ctx->host_x,
                          ctx->policy.capacity * sizeof(float),
                          cudaHostAllocMapped);
    if (s == cudaSuccess)
        s = cudaHostAlloc((void **)&ctx->host_y,
                          ctx->policy.capacity * sizeof(float),
                          cudaHostAllocMapped);
    if (s == cudaSuccess)
        s = cudaHostGetDevicePointer((void **)&ctx->device_x, ctx->host_x, 0);
    if (s == cudaSuccess)
        s = cudaHostGetDevicePointer((void **)&ctx->device_y, ctx->host_y, 0);
    if (s != cudaSuccess) return cuda_error(ctx, "staging allocation", s);
    return WASTE_OK;
}

static waste_status initialize_vq(waste_cuda_ctx *ctx)
{
    const waste_backend_model *m = ctx->model;
    if (!m->codebooks_t ||
        m->n_codebook_values != ctx->policy.book_values) {
        set_error(ctx, "VQ3R codebook view has the wrong size");
        return WASTE_E_UNSUPPORTED;
    }
    cudaError_t s = cudaMalloc((void **)&ctx->vq_books,
                               ctx->policy.book_values * sizeof(float));
    if (s == cudaSuccess)
        s = cudaMalloc((void **)&ctx->vq_x,
                       ctx->policy.capacity * sizeof(float));
    if (s == cudaSuccess)
        s = cudaMalloc((void **)&ctx->vq_y,
                       ctx->policy.vq_y_capacity * sizeof(float));
    for (int i = 0; s == cudaSuccess && i < 3; i++)
        s = cudaMalloc((void **)&ctx->vq_lut[i],
                       ctx->policy.vq_lut_values[i] * sizeof(float));
    if (s == cudaSuccess)
        s = cudaMemcpyAsync(ctx->vq_books, m->codebooks_t,
                            ctx->policy.book_values * sizeof(float),
                            cudaMemcpyHostToDevice, ctx->stream);
    if (s == cudaSuccess) s = cudaStreamSynchronize(ctx->stream);
    if (s != cudaSuccess) return cuda_error(ctx, "VQ allocation/upload", s);
    return WASTE_OK;
}

static void release_context(waste_cuda_ctx *ctx)
{
    if (!ctx) return;
    const int has_cuda = ctx->stream || ctx->host_x || ctx->host_y ||
                         ctx->vq_books || ctx->vq_x || ctx->vq_y ||
                         ctx->vq_lut[0] || ctx->vq_lut[1] || ctx->vq_lut[2];
    if (!has_cuda || cudaSetDevice(ctx->policy.device_ordinal) == cudaSuccess) {
        if (ctx->stream) cudaStreamSynchronize(ctx->stream);
        for (int i = 0; i < 3; i++)
            if (ctx->vq_lut[i]) cudaFree(ctx->vq_lut[i]);
        if (ctx->vq_y) cudaFree(ctx->vq_y);
        if (ctx->vq_x) cudaFree(ctx->vq_x);
        if (ctx->vq_books) cudaFree(ctx->vq_books);
        if (ctx->host_y) cudaFreeHost(ctx->host_y);
        if (ctx->host_x) cudaFreeHost(ctx->host_x);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
    }
    free(ctx->claimed);
    free(ctx);
}

static waste_status provider_plan(const waste_backend_model_info *model,
                                  const void *cfg, uint64_t *reserved_bytes)
{
    if (!reserved_bytes) return WASTE_E_ARG;
    waste_cuda_policy p;
    const waste_status s = waste_cuda_policy_resolve(model, cfg, &p);
    if (s == WASTE_OK) *reserved_bytes = p.reserved_bytes;
    return s;
}

static waste_status provider_open(const waste_backend_model *model,
                                  const void *cfg, void **backend_ctx,
                                  uint32_t *capabilities)
{
    if (!model || model->struct_size < sizeof *model || !backend_ctx ||
        !capabilities || (model->n_tensors && !model->tensors))
        return WASTE_E_ARG;
    *backend_ctx = NULL;
    *capabilities = 0;

    waste_cuda_ctx *ctx = (waste_cuda_ctx *)calloc(1, sizeof *ctx);
    if (!ctx) return WASTE_E_OOM;
    ctx->model = model;
    waste_status s = waste_cuda_policy_resolve(&model->info, cfg,
                                                &ctx->policy);
    if (s != WASTE_OK) {
        release_context(ctx);
        return s;
    }
    if ((ctx->policy.capabilities & WASTE_BACKEND_CAP_MATVEC) &&
        !build_target_set(ctx)) {
        release_context(ctx);
        return WASTE_E_UNSUPPORTED;
    }
    s = initialize_device(ctx);
    if (s == WASTE_OK &&
        (ctx->policy.capabilities & WASTE_BACKEND_CAP_VQ))
        s = initialize_vq(ctx);
    if (s != WASTE_OK) {
        release_context(ctx);
        return s;
    }
    *backend_ctx = ctx;
    *capabilities = ctx->policy.capabilities;
    return WASTE_OK;
}

static int provider_claim_matvec(void *opaque,
                                 const waste_backend_tensor *tensor)
{
    waste_cuda_ctx *ctx = (waste_cuda_ctx *)opaque;
    if (!ctx || !tensor) return -1;
    for (size_t i = 0; i < ctx->n_claimed; i++)
        if (ctx->claimed[i] == tensor) return 1;
    return 0;
}

static waste_status provider_matvec(void *opaque,
                                    const waste_backend_tensor *tensor,
                                    const float *input, float *output)
{
    waste_cuda_ctx *ctx = (waste_cuda_ctx *)opaque;
    if (!ctx || !tensor || !input || !output ||
        !waste_cuda_policy_q4_tensor(tensor) ||
        tensor->rows > ctx->policy.capacity ||
        tensor->cols > ctx->policy.capacity)
        return WASTE_E_ARG;
    waste_status active = activate_device(ctx, "Q4G device activation");
    if (active != WASTE_OK) return active;
    int claimed = 0;
    for (size_t i = 0; i < ctx->n_claimed; i++)
        claimed |= ctx->claimed[i] == tensor;
    if (!claimed) return WASTE_E_UNSUPPORTED;

    memcpy(ctx->host_x, input, tensor->cols * sizeof(float));
    if (ctx->policy.q4_mode == WASTE_CUDA_Q4_CPU_ORDER)
        q4_cpu_order<<<(unsigned)tensor->rows, 32, 0, ctx->stream>>>(
            (const uint8_t *)tensor->weights, tensor->scales,
            ctx->device_x, ctx->device_y, (int)tensor->rows,
            (int)tensor->cols, tensor->row_stride);
    else
        q4_fast<<<(unsigned)tensor->rows, Q4_THREADS, 0, ctx->stream>>>(
            (const uint8_t *)tensor->weights, tensor->scales,
            ctx->device_x, ctx->device_y, (int)tensor->rows,
            (int)tensor->cols, tensor->row_stride);
    cudaError_t s = cudaGetLastError();
    if (s == cudaSuccess) s = cudaStreamSynchronize(ctx->stream);
    if (s != cudaSuccess) return cuda_error(ctx, "Q4G projection", s);
    memcpy(output, ctx->host_y, tensor->rows * sizeof(float));
    return WASTE_OK;
}

static waste_status provider_vq_begin(void *opaque, const float *input,
                                      uint32_t cols,
                                      uint32_t gate_codebook_base,
                                      uint32_t up_codebook_base)
{
    waste_cuda_ctx *ctx = (waste_cuda_ctx *)opaque;
    const uint32_t n_books = ctx ? ctx->model->info.n_codebooks : 0;
    if (!ctx || !input ||
        !(ctx->policy.capabilities & WASTE_BACKEND_CAP_VQ) ||
        cols != ctx->model->info.expert_cols[0] || cols % VQ_VEC_DIM ||
        cols > ctx->policy.capacity || n_books < VQ_STAGES ||
        gate_codebook_base > n_books - VQ_STAGES ||
        up_codebook_base > n_books - VQ_STAGES)
        return WASTE_E_ARG;
    waste_status active = activate_device(ctx, "VQ device activation");
    if (active != WASTE_OK) return active;
    const size_t values =
        (size_t)(cols / VQ_VEC_DIM) * VQ_STAGES * VQ_ENTRIES;
    if (values > ctx->policy.vq_lut_values[0] ||
        values > ctx->policy.vq_lut_values[1])
        return WASTE_E_UNSUPPORTED;

    ctx->vq_prepared = 0;
    memcpy(ctx->host_x, input, (size_t)cols * sizeof(float));
    cudaError_t s = cudaMemcpyAsync(ctx->vq_x, ctx->host_x,
                                    (size_t)cols * sizeof(float),
                                    cudaMemcpyHostToDevice, ctx->stream);
    if (s == cudaSuccess) {
        const int total = (int)(2 * values);
        vq_build_pair<<<(total + VQ_BUILD_THREADS - 1) / VQ_BUILD_THREADS,
                         VQ_BUILD_THREADS, 0, ctx->stream>>>(
            ctx->vq_lut[0], ctx->vq_lut[1], ctx->vq_books, ctx->vq_x,
            cols / VQ_VEC_DIM, (int)gate_codebook_base,
            (int)up_codebook_base);
        s = cudaGetLastError();
    }
    if (s != cudaSuccess) return cuda_error(ctx, "VQ gate/up LUT build", s);
    ctx->prepared_cols = cols;
    ctx->prepared_gate_base = gate_codebook_base;
    ctx->prepared_up_base = up_codebook_base;
    ctx->vq_prepared = 1;
    return WASTE_OK;
}

static int valid_matrix(const waste_backend_vq_matrix *matrix)
{
    if (!matrix || matrix->struct_size < sizeof *matrix ||
        !matrix->indices || !matrix->channel_scales ||
        matrix->record_scheme != WASTE_BACKEND_VQ3R ||
        !matrix->rows || !matrix->cols || matrix->cols % VQ_VEC_DIM)
        return 0;
    const size_t row_blocks =
        ((size_t)matrix->rows + VQ_INDEX_BLOCK - 1) / VQ_INDEX_BLOCK;
    const size_t vectors = (size_t)matrix->cols / VQ_VEC_DIM;
    if (row_blocks > SIZE_MAX / VQ_INDEX_BLOCK ||
        row_blocks * VQ_INDEX_BLOCK > SIZE_MAX / vectors ||
        row_blocks * VQ_INDEX_BLOCK * vectors > SIZE_MAX / VQ_STAGES)
        return 0;
    const size_t needed =
        row_blocks * VQ_INDEX_BLOCK * vectors * VQ_STAGES;
    return matrix->indices_bytes >= needed &&
           matrix->n_channel_scales >= matrix->rows;
}

static waste_status provider_vq_gate_up(
    void *opaque, const waste_backend_vq_matrix *gate,
    const waste_backend_vq_matrix *up, float *gate_output, float *up_output)
{
    waste_cuda_ctx *ctx = (waste_cuda_ctx *)opaque;
    if (!ctx || !ctx->vq_prepared || !valid_matrix(gate) ||
        !valid_matrix(up) || !gate_output || !up_output ||
        gate->rows != up->rows || gate->cols != up->cols ||
        gate->rows != ctx->model->info.expert_rows[0] ||
        gate->cols != ctx->prepared_cols ||
        gate->codebook_base != ctx->prepared_gate_base ||
        up->codebook_base != ctx->prepared_up_base ||
        gate->rows % VQ_INDEX_BLOCK || gate->cols % VQ_VEC_DIM ||
        (size_t)gate->rows * 2 > ctx->policy.vq_y_capacity)
        return WASTE_E_ARG;
    waste_status active = activate_device(ctx, "VQ device activation");
    if (active != WASTE_OK) return active;

    vq_apply_pair<<<gate->rows / VQ_INDEX_BLOCK, 2 * VQ_INDEX_BLOCK,
                    0, ctx->stream>>>(
        ctx->vq_y, gate->indices, up->indices,
        gate->channel_scales, up->channel_scales,
        ctx->vq_lut[0], ctx->vq_lut[1],
        (int)gate->rows, (int)(gate->cols / VQ_VEC_DIM));
    cudaError_t s = cudaGetLastError();
    if (s == cudaSuccess)
        s = cudaMemcpyAsync(ctx->host_y, ctx->vq_y,
                            (size_t)(2 * gate->rows) * sizeof(float),
                            cudaMemcpyDeviceToHost, ctx->stream);
    if (s == cudaSuccess) s = cudaStreamSynchronize(ctx->stream);
    if (s != cudaSuccess) return cuda_error(ctx, "VQ gate/up gather", s);
    memcpy(gate_output, ctx->host_y, (size_t)gate->rows * sizeof(float));
    memcpy(up_output, ctx->host_y + gate->rows,
           (size_t)up->rows * sizeof(float));
    return WASTE_OK;
}

static waste_status provider_vq_down(void *opaque, const float *input,
                                     const waste_backend_vq_matrix *down,
                                     float *output)
{
    waste_cuda_ctx *ctx = (waste_cuda_ctx *)opaque;
    const uint32_t n_books = ctx ? ctx->model->info.n_codebooks : 0;
    if (!ctx || !ctx->vq_prepared || !input || !valid_matrix(down) ||
        !output || down->rows != ctx->model->info.expert_rows[2] ||
        down->cols != ctx->model->info.expert_cols[2] ||
        down->rows % VQ_INDEX_BLOCK || down->cols % VQ_VEC_DIM ||
        down->rows > ctx->policy.vq_y_capacity ||
        down->cols > ctx->policy.capacity ||
        n_books < VQ_STAGES || down->codebook_base > n_books - VQ_STAGES)
        return WASTE_E_ARG;
    waste_status active = activate_device(ctx, "VQ device activation");
    if (active != WASTE_OK) return active;
    const size_t values =
        (size_t)(down->cols / VQ_VEC_DIM) * VQ_STAGES * VQ_ENTRIES;
    if (values > ctx->policy.vq_lut_values[2]) return WASTE_E_UNSUPPORTED;

    memcpy(ctx->host_x, input, (size_t)down->cols * sizeof(float));
    cudaError_t s = cudaMemcpyAsync(ctx->vq_x, ctx->host_x,
                                    (size_t)down->cols * sizeof(float),
                                    cudaMemcpyHostToDevice, ctx->stream);
    if (s == cudaSuccess) {
        vq_build_one<<<(values + VQ_BUILD_THREADS - 1) / VQ_BUILD_THREADS,
                        VQ_BUILD_THREADS, 0, ctx->stream>>>(
            ctx->vq_lut[2], ctx->vq_books, ctx->vq_x,
            (int)(down->cols / VQ_VEC_DIM), (int)down->codebook_base);
        s = cudaGetLastError();
    }
    if (s == cudaSuccess) {
        vq_apply_one<<<(down->rows + VQ_DOWN_THREADS - 1) / VQ_DOWN_THREADS,
                        VQ_DOWN_THREADS, 0, ctx->stream>>>(
            ctx->vq_y, down->indices, down->channel_scales,
            ctx->vq_lut[2], (int)down->rows,
            (int)(down->cols / VQ_VEC_DIM));
        s = cudaGetLastError();
    }
    if (s == cudaSuccess)
        s = cudaMemcpyAsync(ctx->host_y, ctx->vq_y,
                            (size_t)down->rows * sizeof(float),
                            cudaMemcpyDeviceToHost, ctx->stream);
    if (s == cudaSuccess) s = cudaStreamSynchronize(ctx->stream);
    if (s != cudaSuccess) return cuda_error(ctx, "VQ down gather", s);
    memcpy(output, ctx->host_y, (size_t)down->rows * sizeof(float));
    return WASTE_OK;
}

static const char *provider_error_detail(void *opaque)
{
    const waste_cuda_ctx *ctx = (const waste_cuda_ctx *)opaque;
    return ctx && ctx->error[0] ? ctx->error : "CUDA backend error";
}

static void provider_close(void *opaque)
{
    release_context((waste_cuda_ctx *)opaque);
}

extern "C" const waste_backend_v1 waste_cuda_backend_provider = {
    WASTE_BACKEND_API_VERSION,
    sizeof(waste_backend_v1),
    "cuda-gb10-q4g-vq3r",
    provider_plan,
    provider_open,
    provider_claim_matvec,
    provider_matvec,
    provider_vq_begin,
    provider_vq_gate_up,
    provider_vq_down,
    provider_error_detail,
    provider_close,
};
