/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* Model-free tests for the exact all-MLA K2 CUDA allowlist. */

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
} while (0)

int main(void)
{
    waste_model exact = k2();
    CHECK(WASTE_VQ_INDEX_BLOCK == 64);
    CHECK(waste_model_cuda_k2_dense_compatible(&exact));
    CHECK(waste_model_cuda_k2_vq3r_compatible(&exact));

    waste_model changed = k2();
    strcpy(changed.cfg.arch, "KimiK3ForConditionalGeneration");
    CHECK(!waste_model_cuda_k2_dense_compatible(&changed));
    changed = k2(); changed.cfg.kda_layer[17] = 1;
    CHECK(!waste_model_cuda_k2_dense_compatible(&changed));

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

    if (bad) return 1;
    puts("CUDA GEOMETRY OK");
    return 0;
}
