/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* Model-free tests for accelerator allowlists and attention-format gates. */

#include <stdio.h>
#include <string.h>

#include "../src/model.h"
#include "../src/waste_format.h"

static int bad;

#define CHECK(expr) do {                                                    \
    if (!(expr)) {                                                          \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);    \
        bad++;                                                              \
    }                                                                       \
} while (0)

static waste_model k2(void)
{
    waste_model m;
    memset(&m, 0, sizeof m);
    strcpy(m.cfg.arch, "DeepseekV3ForCausalLM");
    m.cfg.n_layers = 61;
    m.cfg.hidden = 7168;
    m.cfg.n_experts = 384;
    m.cfg.top_k = 8;
    m.cfg.moe_inter = 2048;
    m.cfg.dense_inter = 18432;
    m.cfg.n_shared = 1;
    m.cfg.first_dense = 1;
    m.cfg.n_heads = 64;
    m.cfg.kv_lora = 512;
    m.cfg.q_lora = 1536;
    m.cfg.qk_nope = 128;
    m.cfg.qk_rope = 64;
    m.cfg.v_head = 128;
    m.expert_m[0] = m.expert_m[1] = 2048;
    m.expert_m[2] = 7168;
    m.expert_n[0] = m.expert_n[1] = 7168;
    m.expert_n[2] = 2048;
    m.index_bits = 8;
    m.stages = 3;
    m.vec_dim = 8;
    m.cb_entries = 256;
    m.index_block = WASTE_VQ_INDEX_BLOCK;
    return m;
}

static waste_model glm47_flash(void)
{
    waste_model m;
    memset(&m, 0, sizeof m);
    strcpy(m.cfg.arch, "Glm4MoeLiteForCausalLM");
    m.cfg.n_layers = 47;
    m.cfg.hidden = 2048;
    m.cfg.n_experts = 64;
    m.cfg.top_k = 4;
    m.cfg.moe_inter = 1536;
    m.cfg.dense_inter = 10240;
    m.cfg.n_shared = 1;
    m.cfg.first_dense = 1;
    m.cfg.n_heads = 20;
    m.cfg.kv_lora = 512;
    m.cfg.q_lora = 768;
    m.cfg.qk_nope = 192;
    m.cfg.qk_rope = 64;
    m.cfg.v_head = 256;
    m.cfg.rope_interleave = 1;
    m.expert_m[0] = m.expert_m[1] = 1536;
    m.expert_m[2] = 2048;
    m.expert_n[0] = m.expert_n[1] = 2048;
    m.expert_n[2] = 1536;
    m.index_bits = 8;
    m.stages = 3;
    m.vec_dim = 8;
    m.cb_entries = 256;
    m.index_block = WASTE_VQ_INDEX_BLOCK;
    return m;
}

static waste_model glm47_full_gqa(void)
{
    waste_model m;
    memset(&m, 0, sizeof m);
    strcpy(m.cfg.arch, "Glm4MoeForCausalLM");
    strcpy(m.cfg.model_type, "glm4_moe");
    strcpy(m.cfg.hidden_act, "silu");
    strcpy(m.cfg.topk_method, "noaux_tc");
    strcpy(m.cfg.router_activation, "sigmoid");
    m.cfg.n_layers = 4;
    m.cfg.hidden = 96;
    m.cfg.n_experts = 8;
    m.cfg.top_k = 2;
    m.cfg.attention_kind = WASTE_ATTN_GQA;
    m.cfg.n_heads = 12;
    m.cfg.n_kv_heads = 1;
    m.cfg.head_dim = 8;
    m.cfg.qk_nope = 4;
    m.cfg.qk_rope = 4;
    m.cfg.v_head = 8;
    m.cfg.partial_rotary_factor = 0.5f;
    m.cfg.qkv_bias = 1;
    m.cfg.qk_norm = 1;
    m.cfg.router_n_group = 1;
    m.cfg.router_topk_group = 1;
    m.cfg.max_position_embeddings = 202752;
    m.cfg.renorm = 1;
    m.cfg.routed_scale = 2.5f;
    return m;
}

#define REJECT_DENSE(field, value) do {                                     \
    waste_model changed = k2();                                             \
    changed.field = (value);                                                \
    CHECK(!waste_model_cuda_k2_dense_compatible(&changed));                 \
} while (0)

#define REJECT_VQ(field, value) do {                                        \
    waste_model changed = k2();                                             \
    changed.field = (value);                                                \
    CHECK(waste_model_cuda_k2_dense_compatible(&changed));                  \
    CHECK(!waste_model_cuda_k2_vq3r_compatible(&changed));                  \
    CHECK(!waste_model_cuda_vq_dense_scope_compatible(&changed, 3));        \
} while (0)

#define REJECT_FLASH_DENSE(field, value) do {                               \
    waste_model changed = glm47_flash();                                    \
    changed.field = (value);                                                \
    CHECK(!waste_model_cuda_glm47_flash_dense_compatible(&changed));        \
} while (0)

#define REJECT_FLASH_VQ(field, value) do {                                  \
    waste_model changed = glm47_flash();                                    \
    changed.field = (value);                                                \
    CHECK(waste_model_cuda_glm47_flash_dense_compatible(&changed));         \
    CHECK(!waste_model_cuda_glm47_flash_vq3r_compatible(&changed));         \
    CHECK(!waste_model_cuda_vq_dense_scope_compatible(&changed, 3));        \
} while (0)

int main(void)
{
    waste_model exact = k2();
    CHECK(WASTE_VQ_INDEX_BLOCK == 64);
    CHECK(!waste_model_cuda_k2_dense_compatible(NULL));
    CHECK(!waste_model_cuda_glm47_flash_dense_compatible(NULL));
    CHECK(waste_model_cuda_k2_dense_compatible(&exact));
    CHECK(waste_model_cuda_k2_vq3r_compatible(&exact));
    CHECK(!waste_model_cuda_glm47_flash_dense_compatible(&exact));
    CHECK(waste_model_cuda_vq_dense_scope_compatible(&exact, 2));
    CHECK(waste_model_cuda_vq_dense_scope_compatible(&exact, 3));
    CHECK(!waste_model_cuda_vq_dense_scope_compatible(&exact, 1));
    CHECK(!waste_model_cuda_vq_dense_scope_compatible(NULL, 2));

    {
        waste_model gqa = glm47_full_gqa();
        CHECK(!waste_model_glm47_gqa_compatible(NULL));
        CHECK(waste_model_glm47_gqa_compatible(&gqa));
        gqa.cfg.n_layers = WASTE_MAX_LAYERS + 1;
        CHECK(!waste_model_glm47_gqa_compatible(&gqa));
        gqa = glm47_full_gqa();
        gqa.cfg.n_kv_heads = 5;
        CHECK(!waste_model_glm47_gqa_compatible(&gqa));
        gqa = glm47_full_gqa(); gqa.cfg.qk_rope = 3;
        CHECK(!waste_model_glm47_gqa_compatible(&gqa));
        gqa = glm47_full_gqa(); gqa.cfg.partial_rotary_factor = 0.51f;
        CHECK(!waste_model_glm47_gqa_compatible(&gqa));
        gqa = glm47_full_gqa(); gqa.cfg.qkv_bias = 0;
        CHECK(!waste_model_glm47_gqa_compatible(&gqa));
        gqa = glm47_full_gqa(); gqa.cfg.qk_norm = 0;
        CHECK(!waste_model_glm47_gqa_compatible(&gqa));
        gqa = glm47_full_gqa(); gqa.cfg.rope_interleave = 1;
        CHECK(!waste_model_glm47_gqa_compatible(&gqa));
        gqa = glm47_full_gqa(); gqa.cfg.router_n_group = 2;
        CHECK(!waste_model_glm47_gqa_compatible(&gqa));
        gqa = glm47_full_gqa(); strcpy(gqa.cfg.topk_method, "greedy");
        CHECK(!waste_model_glm47_gqa_compatible(&gqa));
        gqa = glm47_full_gqa(); strcpy(gqa.cfg.router_activation, "softmax");
        CHECK(!waste_model_glm47_gqa_compatible(&gqa));
        gqa = glm47_full_gqa(); gqa.cfg.renorm = 0;
        CHECK(!waste_model_glm47_gqa_compatible(&gqa));
        gqa = glm47_full_gqa(); gqa.cfg.routed_scale = 1.0f;
        CHECK(!waste_model_glm47_gqa_compatible(&gqa));
        gqa = glm47_full_gqa(); gqa.cfg.kda_layer[0] = 1;
        CHECK(!waste_model_glm47_gqa_compatible(&gqa));

        CHECK(waste_model_gqa_kv_head(0, 12, 3) == 0);
        CHECK(waste_model_gqa_kv_head(3, 12, 3) == 0);
        CHECK(waste_model_gqa_kv_head(4, 12, 3) == 1);
        CHECK(waste_model_gqa_kv_head(11, 12, 3) == 2);
        CHECK(waste_model_gqa_kv_head(12, 12, 3) == -1);
        CHECK(waste_model_gqa_kv_head(0, 11, 3) == -1);

        float x[8] = { 1, 2, 3, 4, 50, 60, 70, 80 };
        const float cs[2] = { 0, 1 }, sn[2] = { 1, 0 };
        waste_model_rope_half(x, 4, cs, sn);
        CHECK(x[0] == -3 && x[1] == 2 && x[2] == 1 && x[3] == 4);
        CHECK(x[4] == 50 && x[5] == 60 && x[6] == 70 && x[7] == 80);

        size_t state_bytes = 0;
        gqa = glm47_full_gqa();
        CHECK(waste_model_state_size(&gqa, 0, &state_bytes) == -1);
    }

    /* K2 is qualified for decode but never for this GLM-only pilot. */
    exact.cuda_prefill_vq = 1;
    exact.cuda_vq_mode = 2;
    exact.cuda_vq_preflight_modes = 1 << 2;
    CHECK(!waste_model_cuda_prefill_vq_compatible(&exact));

    waste_model changed = k2();
    strcpy(changed.cfg.arch, "KimiK3ForConditionalGeneration");
    CHECK(!waste_model_cuda_k2_dense_compatible(&changed));
    changed = k2(); changed.cfg.kda_layer[17] = 1;
    CHECK(!waste_model_cuda_k2_dense_compatible(&changed));
    CHECK(!waste_model_cuda_vq_dense_scope_compatible(&changed, 3));

    REJECT_DENSE(cfg.n_layers, 60);
    REJECT_DENSE(cfg.hidden, 7169);
    REJECT_DENSE(cfg.n_experts, 256);
    REJECT_DENSE(cfg.top_k, 16);
    REJECT_DENSE(cfg.moe_inter, 3072);
    REJECT_DENSE(cfg.dense_inter, 16384);
    REJECT_DENSE(cfg.n_shared, 2);
    REJECT_DENSE(cfg.first_dense, 0);
    REJECT_DENSE(cfg.n_heads, 32);
    REJECT_DENSE(cfg.kv_lora, 256);
    REJECT_DENSE(cfg.q_lora, 0);
    REJECT_DENSE(cfg.qk_nope, 64);
    REJECT_DENSE(cfg.qk_rope, 128);
    REJECT_DENSE(cfg.v_head, 64);
    REJECT_DENSE(cfg.latent_dim, 3584);
    REJECT_DENSE(cfg.kda_heads, 1);
    REJECT_DENSE(cfg.kda_dim, 128);
    REJECT_DENSE(expert_m[2], 7104);
    REJECT_DENSE(expert_n[0], 7104);

    REJECT_VQ(index_bits, 6);
    REJECT_VQ(stages, 2);
    REJECT_VQ(vec_dim, 4);
    REJECT_VQ(cb_entries, 64);
    REJECT_VQ(index_block, 32);

    exact = glm47_flash();
    CHECK(waste_model_cuda_glm47_flash_dense_compatible(&exact));
    CHECK(waste_model_cuda_glm47_flash_vq3r_compatible(&exact));
    CHECK(!waste_model_cuda_k2_dense_compatible(&exact));
    CHECK(waste_model_cuda_vq_dense_scope_compatible(&exact, 2));
    CHECK(waste_model_cuda_vq_dense_scope_compatible(&exact, 3));
    exact.cuda_vq_mode = 2;
    exact.cuda_vq_preflight_modes = 1 << 2;
    CHECK(!waste_model_cuda_prefill_vq_compatible(&exact));
    exact.cuda_prefill_vq = 1;
    CHECK(waste_model_cuda_prefill_vq_compatible(&exact));
    exact.cuda_vq_mode = 1;
    CHECK(!waste_model_cuda_prefill_vq_compatible(&exact));
    exact.cuda_vq_mode = 2;
    exact.cuda_vq_preflight_modes = 0;
    CHECK(!waste_model_cuda_prefill_vq_compatible(&exact));

    {
        waste_model dense = glm47_flash();
        dense.cuda_kda_mode = 1;
        dense.cuda_dense_scope = 3;
        dense.cuda_dense_preflight_scope = 3;
        CHECK(!waste_model_cuda_prefill_dense_compatible(&dense));
        dense.cuda_prefill_dense = 1;
        dense.cuda_prefill_dense_preflight_mode = 1;
        CHECK(waste_model_cuda_prefill_dense_compatible(&dense));
        dense.cuda_prefill_dense = 2;
        CHECK(!waste_model_cuda_prefill_dense_compatible(&dense));
        dense.cuda_prefill_dense_preflight_mode = 2;
        CHECK(waste_model_cuda_prefill_dense_compatible(&dense));
        dense.cuda_prefill_dense = 3;
        dense.cuda_prefill_dense_preflight_mode = 3;
        CHECK(!waste_model_cuda_prefill_dense_compatible(&dense));
        dense.cuda_prefill_dense = 2;
        dense.cuda_prefill_dense_preflight_mode = 2;
        dense.cuda_dense_preflight_scope = 2;
        CHECK(!waste_model_cuda_prefill_dense_compatible(&dense));
        dense.cuda_dense_preflight_scope = 3;
        dense.cuda_dense_scope = 2;
        CHECK(!waste_model_cuda_prefill_dense_compatible(&dense));
        dense.cuda_dense_scope = 3;
        dense.cuda_kda_mode = 2;
        CHECK(!waste_model_cuda_prefill_dense_compatible(&dense));
        dense.cuda_kda_mode = 1;
        dense.cuda_kda_failed = 1;
        CHECK(!waste_model_cuda_prefill_dense_compatible(&dense));

        dense = k2();
        dense.cuda_prefill_dense = 1;
        dense.cuda_kda_mode = 1;
        dense.cuda_dense_scope = 3;
        dense.cuda_dense_preflight_scope = 3;
        CHECK(!waste_model_cuda_prefill_dense_compatible(&dense));
    }

    {
        const int src_ids[] = { 51, 3, 29, 11 };
        const float src_weights[] = { 0.51f, 0.03f, 0.29f, 0.11f };
        int ids[4] = { 0 };
        float weights[4] = { 0 };
        CHECK(!waste_model_sort_route_copy(
            src_ids, src_weights, 4, ids, weights));
        CHECK(ids[0] == 3 && ids[1] == 11 &&
              ids[2] == 29 && ids[3] == 51);
        CHECK(weights[0] == 0.03f && weights[1] == 0.11f &&
              weights[2] == 0.29f && weights[3] == 0.51f);
        CHECK(src_ids[0] == 51 && src_weights[0] == 0.51f);
        CHECK(waste_model_sort_route_copy(
            src_ids, src_weights, 65, ids, weights) == -1);
    }

    changed = glm47_flash();
    strcpy(changed.cfg.arch, "DeepseekV3ForCausalLM");
    CHECK(!waste_model_cuda_glm47_flash_dense_compatible(&changed));
    changed = glm47_flash(); changed.cfg.kda_layer[17] = 1;
    CHECK(!waste_model_cuda_glm47_flash_dense_compatible(&changed));
    CHECK(!waste_model_cuda_vq_dense_scope_compatible(&changed, 3));

    REJECT_FLASH_DENSE(cfg.n_layers, 46);
    REJECT_FLASH_DENSE(cfg.hidden, 2049);
    REJECT_FLASH_DENSE(cfg.n_experts, 63);
    REJECT_FLASH_DENSE(cfg.top_k, 8);
    REJECT_FLASH_DENSE(cfg.moe_inter, 2048);
    REJECT_FLASH_DENSE(cfg.dense_inter, 18432);
    REJECT_FLASH_DENSE(cfg.n_shared, 2);
    REJECT_FLASH_DENSE(cfg.first_dense, 0);
    REJECT_FLASH_DENSE(cfg.n_heads, 16);
    REJECT_FLASH_DENSE(cfg.kv_lora, 256);
    REJECT_FLASH_DENSE(cfg.q_lora, 1536);
    REJECT_FLASH_DENSE(cfg.qk_nope, 128);
    REJECT_FLASH_DENSE(cfg.qk_rope, 128);
    REJECT_FLASH_DENSE(cfg.v_head, 128);
    REJECT_FLASH_DENSE(cfg.rope_interleave, 0);
    REJECT_FLASH_DENSE(cfg.latent_dim, 3584);
    REJECT_FLASH_DENSE(cfg.kda_heads, 1);
    REJECT_FLASH_DENSE(cfg.kda_dim, 128);
    REJECT_FLASH_DENSE(expert_m[0], 1472);
    REJECT_FLASH_DENSE(expert_m[1], 1472);
    REJECT_FLASH_DENSE(expert_m[2], 1984);
    REJECT_FLASH_DENSE(expert_n[0], 1984);
    REJECT_FLASH_DENSE(expert_n[1], 1984);
    REJECT_FLASH_DENSE(expert_n[2], 1472);

    REJECT_FLASH_VQ(index_bits, 6);
    REJECT_FLASH_VQ(stages, 2);
    REJECT_FLASH_VQ(vec_dim, 4);
    REJECT_FLASH_VQ(cb_entries, 64);
    REJECT_FLASH_VQ(index_block, 32);

    if (bad) return 1;
    puts("CUDA GEOMETRY OK");
    return 0;
}
