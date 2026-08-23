/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* Model-free tests for the exact CUDA model allowlists. */

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

static waste_model k3(void)
{
    waste_model m;
    memset(&m, 0, sizeof m);
    strcpy(m.cfg.arch, "KimiK3ForConditionalGeneration");
    strcpy(m.cfg.prefix, "language_model.");
    m.cfg.n_layers = 93;
    m.cfg.hidden = 7168;
    m.cfg.vocab = 163840;
    m.cfg.n_experts = 896;
    m.cfg.top_k = 16;
    m.manifest_top_k = 16;
    m.cfg.moe_inter = 3072;
    m.cfg.dense_inter = 33792;
    m.cfg.n_shared = 2;
    m.cfg.first_dense = 1;
    m.cfg.n_heads = 96;
    m.cfg.kv_lora = 512;
    m.cfg.q_lora = 1536;
    m.cfg.qk_nope = 128;
    m.cfg.qk_rope = 64;
    m.cfg.v_head = 128;
    m.cfg.latent_dim = 3584;
    m.cfg.latent_norm = 1;
    m.cfg.kda_heads = 96;
    m.cfg.kda_dim = 128;
    m.cfg.conv_k = 4;
    m.cfg.full_rank_gate = 1;
    m.cfg.attn_res_block = 12;
    m.cfg.mla_output_gate = 1;
    m.cfg.mla_nope = 1;
    m.cfg.act_situ = 1;
    for (int L = 0; L < m.cfg.n_layers; L++)
        m.cfg.kda_layer[L] = L < 92 && L % 4 != 3;
    m.expert_m[0] = m.expert_m[1] = 3072;
    m.expert_m[2] = 3584;
    m.expert_n[0] = m.expert_n[1] = 3584;
    m.expert_n[2] = 3072;
    m.index_bits = 8;
    m.stages = 3;
    m.vec_dim = 8;
    m.cb_entries = 256;
    m.index_block = WASTE_VQ_INDEX_BLOCK;
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

#define REJECT_K3_DENSE(field, value) do {                                  \
    waste_model changed = k3();                                             \
    changed.field = (value);                                                \
    CHECK(!waste_model_cuda_k3_dense_compatible(&changed));                 \
} while (0)

#define REJECT_K3_VQ(field, value) do {                                     \
    waste_model changed = k3();                                             \
    changed.field = (value);                                                \
    CHECK(waste_model_cuda_k3_dense_compatible(&changed));                  \
    CHECK(!waste_model_cuda_k3_vq3r_compatible(&changed));                  \
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

    /* K2 enters the prefill pilot only after its exact decode preflight. */
    exact.cuda_prefill_vq = 1;
    exact.cuda_vq_mode = 2;
    exact.cuda_vq_preflight_modes = 1 << 2;
    CHECK(waste_model_cuda_prefill_vq_compatible(&exact));
    exact.cuda_vq_preflight_modes = 0;
    CHECK(!waste_model_cuda_prefill_vq_compatible(&exact));
    exact.cuda_vq_preflight_modes = 1 << 2;

    {
        waste_model dense = k2();
        dense.cuda_kda_mode = 1;
        dense.cuda_dense_scope = 3;
        dense.cuda_dense_preflight_scope = 3;
        dense.cuda_prefill_dense = 1;
        dense.cuda_prefill_dense_preflight_mode = 1;
        CHECK(waste_model_cuda_prefill_dense_compatible(&dense));
        dense.cuda_prefill_dense = 2;
        CHECK(!waste_model_cuda_prefill_dense_compatible(&dense));
        dense.cuda_prefill_dense_preflight_mode = 2;
        CHECK(waste_model_cuda_prefill_dense_compatible(&dense));
        dense.cfg.hidden++;
        CHECK(!waste_model_cuda_prefill_dense_compatible(&dense));
    }

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

    {
        waste_model exact_k3 = k3();
        CHECK(!waste_model_cuda_k3_dense_compatible(NULL));
        CHECK(!waste_model_k3_routing_compatible(NULL));
        CHECK(waste_model_k3_routing_compatible(&exact_k3));
        CHECK(waste_model_cuda_k3_dense_compatible(&exact_k3));
        CHECK(waste_model_cuda_k3_vq3r_compatible(&exact_k3));
        CHECK(waste_model_cuda_vq_dense_scope_compatible(&exact_k3, 2));
        CHECK(!waste_model_cuda_vq_dense_scope_compatible(&exact_k3, 3));

        exact_k3.cuda_prefill_vq = 1;
        exact_k3.cuda_vq_mode = 2;
        exact_k3.cuda_vq_preflight_modes = 1 << 2;
        CHECK(waste_model_cuda_prefill_vq_compatible(&exact_k3));
        exact_k3.cuda_vq_preflight_modes = 0;
        CHECK(!waste_model_cuda_prefill_vq_compatible(&exact_k3));
        exact_k3.cuda_vq_preflight_modes = 1 << 2;

        exact_k3.cuda_kda_mode = 1;
        exact_k3.cuda_dense_scope = 2;
        exact_k3.cuda_dense_preflight_scope = 2;
        exact_k3.cuda_prefill_dense = 1;
        exact_k3.cuda_prefill_dense_preflight_mode = 1;
        CHECK(waste_model_cuda_prefill_dense_compatible(&exact_k3));
        exact_k3.cuda_dense_scope = 3;
        CHECK(!waste_model_cuda_prefill_dense_compatible(&exact_k3));
        exact_k3.cuda_dense_scope = 2;
        exact_k3.cuda_dense_preflight_scope = 3;
        CHECK(!waste_model_cuda_prefill_dense_compatible(&exact_k3));
        exact_k3.cuda_dense_preflight_scope = 2;
        exact_k3.cuda_kda_mode = 2;
        CHECK(!waste_model_cuda_prefill_dense_compatible(&exact_k3));
        exact_k3.cuda_kda_mode = 1;
        exact_k3.cuda_kda_failed = 1;
        CHECK(!waste_model_cuda_prefill_dense_compatible(&exact_k3));
        exact_k3.cuda_kda_failed = 0;
        exact_k3.cuda_prefill_dense = 2;
        exact_k3.cuda_prefill_dense_preflight_mode = 2;
        CHECK(!waste_model_cuda_prefill_dense_compatible(&exact_k3));

        exact_k3 = k3(); exact_k3.cfg.kda_layer[3] = 1;
        CHECK(!waste_model_cuda_k3_dense_compatible(&exact_k3));
        exact_k3 = k3(); exact_k3.cfg.kda_layer[90] = 0;
        CHECK(!waste_model_cuda_k3_dense_compatible(&exact_k3));
        exact_k3 = k3(); strcpy(exact_k3.cfg.arch, "KimiLinearForCausalLM");
        CHECK(!waste_model_cuda_k3_dense_compatible(&exact_k3));
        exact_k3 = k3(); strcpy(exact_k3.cfg.prefix, "");
        CHECK(!waste_model_cuda_k3_dense_compatible(&exact_k3));

        /* Approximate K3 routing is accepted only with trained top-16
         * provenance and an explicit startup selector matching cfg.top_k. */
        for (int i = 0; i < 2; i++) {
            const int top_k = i ? 8 : 12;
            exact_k3 = k3();
            exact_k3.cfg.top_k = top_k;
            exact_k3.k3_approx_top_k = top_k;
            CHECK(waste_model_k3_routing_compatible(&exact_k3));
            CHECK(waste_model_cuda_k3_dense_compatible(&exact_k3));
            CHECK(waste_model_cuda_k3_vq3r_compatible(&exact_k3));
            exact_k3.cuda_prefill_vq = 1;
            exact_k3.cuda_vq_mode = 2;
            exact_k3.cuda_vq_preflight_modes = 1 << 2;
            CHECK(waste_model_cuda_prefill_vq_compatible(&exact_k3));
            exact_k3.cuda_kda_mode = 1;
            exact_k3.cuda_dense_scope = 2;
            exact_k3.cuda_dense_preflight_scope = 2;
            exact_k3.cuda_prefill_dense = 1;
            exact_k3.cuda_prefill_dense_preflight_mode = 1;
            CHECK(waste_model_cuda_prefill_dense_compatible(&exact_k3));
        }
        exact_k3 = k3(); exact_k3.cfg.top_k = 8;
        CHECK(!waste_model_k3_routing_compatible(&exact_k3));
        exact_k3 = k3(); exact_k3.k3_approx_top_k = 8;
        CHECK(!waste_model_k3_routing_compatible(&exact_k3));
        exact_k3 = k3(); exact_k3.cfg.top_k = 8;
        exact_k3.k3_approx_top_k = 12;
        CHECK(!waste_model_k3_routing_compatible(&exact_k3));
        for (int top_k = 4; top_k <= 17; top_k += top_k == 4 ? 11 : 2) {
            exact_k3 = k3();
            exact_k3.cfg.top_k = top_k;
            exact_k3.k3_approx_top_k = top_k;
            CHECK(!waste_model_k3_routing_compatible(&exact_k3));
            CHECK(!waste_model_cuda_k3_dense_compatible(&exact_k3));
        }
        exact_k3 = k3(); exact_k3.manifest_top_k = 12;
        exact_k3.cfg.top_k = 12;
        CHECK(!waste_model_k3_routing_compatible(&exact_k3));
        CHECK(!waste_model_cuda_k3_dense_compatible(&exact_k3));
        exact_k3 = k3(); exact_k3.manifest_top_k = 8;
        exact_k3.cfg.top_k = 8;
        CHECK(!waste_model_k3_routing_compatible(&exact_k3));
        CHECK(!waste_model_cuda_k3_dense_compatible(&exact_k3));
    }

    REJECT_K3_DENSE(cfg.n_layers, 92);
    REJECT_K3_DENSE(cfg.hidden, 7169);
    REJECT_K3_DENSE(cfg.vocab, 163839);
    REJECT_K3_DENSE(cfg.n_experts, 895);
    REJECT_K3_DENSE(cfg.top_k, 8);
    REJECT_K3_DENSE(cfg.moe_inter, 2048);
    REJECT_K3_DENSE(cfg.dense_inter, 33791);
    REJECT_K3_DENSE(cfg.n_shared, 1);
    REJECT_K3_DENSE(cfg.first_dense, 0);
    REJECT_K3_DENSE(cfg.n_heads, 64);
    REJECT_K3_DENSE(cfg.kv_lora, 256);
    REJECT_K3_DENSE(cfg.q_lora, 0);
    REJECT_K3_DENSE(cfg.qk_nope, 64);
    REJECT_K3_DENSE(cfg.qk_rope, 128);
    REJECT_K3_DENSE(cfg.v_head, 64);
    REJECT_K3_DENSE(cfg.latent_dim, 0);
    REJECT_K3_DENSE(cfg.latent_norm, 0);
    REJECT_K3_DENSE(cfg.kda_heads, 64);
    REJECT_K3_DENSE(cfg.kda_dim, 64);
    REJECT_K3_DENSE(cfg.conv_k, 3);
    REJECT_K3_DENSE(cfg.full_rank_gate, 0);
    REJECT_K3_DENSE(cfg.attn_res_block, 0);
    REJECT_K3_DENSE(cfg.mla_output_gate, 0);
    REJECT_K3_DENSE(cfg.mla_nope, 0);
    REJECT_K3_DENSE(cfg.act_situ, 0);
    REJECT_K3_DENSE(expert_m[0], 3008);
    REJECT_K3_DENSE(expert_m[2], 3520);
    REJECT_K3_DENSE(expert_n[0], 3520);
    REJECT_K3_DENSE(expert_n[2], 3008);
    REJECT_K3_VQ(index_bits, 6);
    REJECT_K3_VQ(stages, 4);
    REJECT_K3_VQ(vec_dim, 4);
    REJECT_K3_VQ(cb_entries, 64);
    REJECT_K3_VQ(index_block, 32);

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
