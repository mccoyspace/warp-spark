/* SPDX-License-Identifier: Apache-2.0 */
#include "waste_cuda_policy.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static waste_backend_model_info k2(void)
{
    waste_backend_model_info m;
    memset(&m, 0, sizeof m);
    m.struct_size = sizeof m;
    m.arch = "DeepseekV3ForCausalLM";
    m.n_layers = 61; m.hidden = 7168; m.n_experts = 384; m.top_k = 8;
    m.moe_inter = 2048; m.dense_inter = 18432;
    m.n_shared = 1; m.first_dense = 1;
    m.n_heads = 64; m.kv_lora = 512; m.q_lora = 1536;
    m.qk_nope = 128; m.qk_rope = 64; m.v_head = 128;
    m.vq_scheme = WASTE_BACKEND_VQ3R; m.vq_stages = 3;
    m.vq_entries = 256; m.vq_vec_dim = 8; m.vq_index_bits = 8;
    m.vq_index_block = 64; m.vq_lut_block = 32; m.n_codebooks = 9;
    m.expert_rows[0] = m.expert_rows[1] = 2048;
    m.expert_rows[2] = 7168;
    m.expert_cols[0] = m.expert_cols[1] = 7168;
    m.expert_cols[2] = 2048;
    return m;
}

static waste_backend_model_info k3(void)
{
    waste_backend_model_info m = k2();
    m.arch = "KimiK3ForConditionalGeneration";
    m.n_layers = 93; m.n_kda_layers = 69;
    m.n_experts = 896; m.top_k = 16;
    m.moe_inter = 3072; m.dense_inter = 33792; m.latent_dim = 3584;
    m.n_shared = 2; m.n_heads = 96;
    m.expert_rows[0] = m.expert_rows[1] = 3072;
    m.expert_rows[2] = 3584;
    m.expert_cols[0] = m.expert_cols[1] = 3584;
    m.expert_cols[2] = 3072;
    return m;
}

int main(void)
{
    waste_cuda_policy p;
    waste_backend_model_info m = k2();
    assert(waste_cuda_policy_resolve(&m, NULL, &p) == WASTE_OK);
    assert(p.model_kind == WASTE_CUDA_MODEL_K2);
    assert(p.q4_mode == WASTE_CUDA_Q4_FAST);
    assert(p.capabilities ==
           (WASTE_BACKEND_CAP_MATVEC | WASTE_BACKEND_CAP_VQ));
    assert(p.capacity == 18432 && p.reserved_bytes > 64u * 1024u * 1024u);

    m = k3();
    assert(waste_cuda_policy_resolve(&m, NULL, &p) == WASTE_OK);
    assert(p.model_kind == WASTE_CUDA_MODEL_K3 && p.capacity == 33792);

    m.vq_scheme = WASTE_BACKEND_VQ4P;
    assert(waste_cuda_policy_resolve(&m, NULL, &p) == WASTE_E_UNSUPPORTED);
    m = k3(); m.n_kda_layers--;
    assert(waste_cuda_policy_resolve(&m, NULL, &p) == WASTE_E_UNSUPPORTED);

    waste_cuda_backend_config c = {
        sizeof c, WASTE_CUDA_BACKEND_CONFIG_VERSION, 0,
        WASTE_BACKEND_CAP_MATVEC, WASTE_CUDA_Q4_CPU_ORDER
    };
    m = k2();
    assert(waste_cuda_policy_resolve(&m, &c, &p) == WASTE_OK);
    assert(p.capabilities == WASTE_BACKEND_CAP_MATVEC);
    assert(p.q4_mode == WASTE_CUDA_Q4_CPU_ORDER);
    m.vq_scheme = WASTE_BACKEND_VQ4P;
    assert(waste_cuda_policy_resolve(&m, &c, &p) == WASTE_E_UNSUPPORTED);
    c.device_ordinal = 1;
    m = k2();
    assert(waste_cuda_policy_resolve(&m, &c, &p) == WASTE_E_ARG);

    uint8_t weights[64] = {0};
    uint16_t scales[1] = {0};
    waste_backend_tensor t = {
        sizeof t, "q4", weights, scales, 1, 128, 64, 4, 128
    };
    assert(waste_cuda_policy_q4_tensor(&t));
    t.row_stride = 63;
    assert(!waste_cuda_policy_q4_tensor(&t));
    puts("cuda policy: ok");
    return 0;
}
