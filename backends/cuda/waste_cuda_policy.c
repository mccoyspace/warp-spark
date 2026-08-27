/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#include "waste_cuda_policy.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

enum {
    CUDA_Q4_GROUP = 128,
    CUDA_VQ_STAGES = 3,
    CUDA_VQ_VEC_DIM = 8,
    CUDA_VQ_ENTRIES = 256,
    CUDA_VQ_INDEX_BLOCK = 64,
    CUDA_RUNTIME_RESERVE = 64 * 1024 * 1024,
};

static int same3(const uint32_t a[3], uint32_t a0, uint32_t a1, uint32_t a2)
{
    return a[0] == a0 && a[1] == a1 && a[2] == a2;
}

static int is_vq3r(const waste_backend_model_info *m)
{
    return m->vq_scheme == WASTE_BACKEND_VQ3R &&
           m->vq_stages == CUDA_VQ_STAGES &&
           m->vq_entries == CUDA_VQ_ENTRIES &&
           m->vq_vec_dim == CUDA_VQ_VEC_DIM &&
           m->vq_index_bits == 8 &&
           m->vq_index_block == CUDA_VQ_INDEX_BLOCK &&
           m->vq_lut_block == 32 && m->n_codebooks >= 9;
}

static int model_kind(const waste_backend_model_info *m)
{
    if (!m || m->struct_size < sizeof *m || !m->arch || !is_vq3r(m))
        return WASTE_CUDA_MODEL_NONE;

    if (!strcmp(m->arch, "DeepseekV3ForCausalLM") &&
        m->n_layers == 61 && m->n_kda_layers == 0 &&
        m->hidden == 7168 && m->n_experts == 384 && m->top_k == 8 &&
        m->moe_inter == 2048 && m->dense_inter == 18432 &&
        m->latent_dim == 0 && m->n_shared == 1 && m->first_dense == 1 &&
        m->n_heads == 64 && m->kv_lora == 512 && m->q_lora == 1536 &&
        m->qk_nope == 128 && m->qk_rope == 64 && m->v_head == 128 &&
        same3(m->expert_rows, 2048, 2048, 7168) &&
        same3(m->expert_cols, 7168, 7168, 2048))
        return WASTE_CUDA_MODEL_K2;

    if (!strcmp(m->arch, "KimiK3ForConditionalGeneration") &&
        m->n_layers == 93 && m->n_kda_layers == 69 &&
        m->hidden == 7168 && m->n_experts == 896 && m->top_k == 16 &&
        m->moe_inter == 3072 && m->dense_inter == 33792 &&
        m->latent_dim == 3584 && m->n_shared == 2 &&
        m->first_dense == 1 && m->n_heads == 96 &&
        m->kv_lora == 512 && m->q_lora == 1536 &&
        m->qk_nope == 128 && m->qk_rope == 64 && m->v_head == 128 &&
        same3(m->expert_rows, 3072, 3072, 3584) &&
        same3(m->expert_cols, 3584, 3584, 3072))
        return WASTE_CUDA_MODEL_K3;

    return WASTE_CUDA_MODEL_NONE;
}

static int add_u64(uint64_t *sum, uint64_t value)
{
    if (UINT64_MAX - *sum < value) return 0;
    *sum += value;
    return 1;
}

static int mul_u64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (a && b > UINT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static size_t max_size(size_t a, size_t b)
{
    return a > b ? a : b;
}

static waste_status resolve_config(const void *opaque, waste_cuda_policy *p)
{
    p->device_ordinal = 0;
    p->capabilities = WASTE_BACKEND_CAP_MATVEC | WASTE_BACKEND_CAP_VQ;
    p->q4_mode = WASTE_CUDA_Q4_FAST;
    if (!opaque) return WASTE_OK;

    const waste_cuda_backend_config *c =
        (const waste_cuda_backend_config *)opaque;
    if (c->struct_size < sizeof *c ||
        c->config_version != WASTE_CUDA_BACKEND_CONFIG_VERSION ||
        c->device_ordinal != 0 ||
        (c->capabilities & ~(WASTE_BACKEND_CAP_MATVEC |
                             WASTE_BACKEND_CAP_VQ)))
        return WASTE_E_ARG;
    if (c->capabilities) p->capabilities = c->capabilities;
    if (c->q4_mode) p->q4_mode = c->q4_mode;
    if (p->q4_mode != WASTE_CUDA_Q4_FAST &&
        p->q4_mode != WASTE_CUDA_Q4_CPU_ORDER)
        return WASTE_E_ARG;
    return WASTE_OK;
}

waste_status waste_cuda_policy_resolve(
    const waste_backend_model_info *m, const void *backend_cfg,
    waste_cuda_policy *out)
{
    if (!m || !out || m->struct_size < sizeof *m) return WASTE_E_ARG;
    memset(out, 0, sizeof *out);
    waste_status s = resolve_config(backend_cfg, out);
    if (s != WASTE_OK) return s;
    out->model_kind = model_kind(m);
    if (!out->model_kind) return WASTE_E_UNSUPPORTED;

    size_t cap = m->hidden;
    cap = max_size(cap, (size_t)m->n_heads * (m->qk_nope + m->qk_rope));
    cap = max_size(cap, (size_t)m->n_heads * m->v_head);
    cap = max_size(cap, m->q_lora);
    cap = max_size(cap, m->kv_lora + m->qk_rope);
    cap = max_size(cap, m->latent_dim);
    cap = max_size(cap, (size_t)m->moe_inter *
                        (m->n_shared ? m->n_shared : 1));
    cap = max_size(cap, m->dense_inter);
    out->capacity = cap;

    const size_t lat = m->latent_dim ? m->latent_dim : m->hidden;
    if (!lat || !m->moe_inter || lat % CUDA_VQ_VEC_DIM ||
        m->moe_inter % CUDA_VQ_VEC_DIM || lat % CUDA_VQ_INDEX_BLOCK ||
        m->moe_inter % CUDA_VQ_INDEX_BLOCK)
        return WASTE_E_UNSUPPORTED;
    out->vq_lut_values[0] =
        (lat / CUDA_VQ_VEC_DIM) * CUDA_VQ_STAGES * CUDA_VQ_ENTRIES;
    out->vq_lut_values[1] = out->vq_lut_values[0];
    out->vq_lut_values[2] =
        ((size_t)m->moe_inter / CUDA_VQ_VEC_DIM) *
        CUDA_VQ_STAGES * CUDA_VQ_ENTRIES;
    out->vq_y_capacity = max_size((size_t)m->moe_inter * 2, lat);
    out->book_values =
        (size_t)m->n_codebooks * CUDA_VQ_VEC_DIM * CUDA_VQ_ENTRIES;
    if (out->vq_y_capacity > cap) return WASTE_E_UNSUPPORTED;

    /* The 64 MiB term is deliberately visible policy, not a claim that the
     * CUDA driver's process/context allocation is byte-stable. It keeps the
     * opaque runtime overhead inside WASTE's budget on the qualified GB10. */
    uint64_t bytes = CUDA_RUNTIME_RESERVE, part;
    if (!mul_u64((uint64_t)cap, 2u * sizeof(float), &part) ||
        !add_u64(&bytes, part)) return WASTE_E_OOM;
    if (out->capabilities & WASTE_BACKEND_CAP_VQ) {
        if (!mul_u64((uint64_t)cap, sizeof(float), &part) ||
            !add_u64(&bytes, part) ||
            !mul_u64((uint64_t)out->vq_y_capacity, sizeof(float), &part) ||
            !add_u64(&bytes, part) ||
            !mul_u64((uint64_t)out->book_values, sizeof(float), &part) ||
            !add_u64(&bytes, part)) return WASTE_E_OOM;
        for (int i = 0; i < 3; i++)
            if (!mul_u64((uint64_t)out->vq_lut_values[i], sizeof(float),
                         &part) || !add_u64(&bytes, part))
                return WASTE_E_OOM;
    }
    out->reserved_bytes = bytes;
    return WASTE_OK;
}

int waste_cuda_policy_q4_tensor(const waste_backend_tensor *t)
{
    if (!t || t->struct_size < sizeof *t || !t->name || !t->weights ||
        !t->scales || !t->rows || !t->cols || t->bits != 4 ||
        t->group != CUDA_Q4_GROUP)
        return 0;
    return t->row_stride >= (t->cols + 1) / 2;
}
