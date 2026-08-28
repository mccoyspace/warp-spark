#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Cheap GLM source-boundary checks; no model shards required."""

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_convert():
    missing = object()
    old_torch = sys.modules.get("torch", missing)
    old_mxfp4 = sys.modules.get("mxfp4", missing)
    torch = types.ModuleType("torch")
    torch.device = lambda value: value
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False))
    sys.modules["torch"] = torch
    mx = types.ModuleType("mxfp4")
    mx.ST = object
    mx.unblock_scale = lambda value, scale, block: value
    sys.modules["mxfp4"] = mx
    sys.path.insert(0, os.path.join(REPO, "tools"))
    try:
        import convert
        return convert
    finally:
        for name, old in (("torch", old_torch), ("mxfp4", old_mxfp4)):
            if old is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


CONVERT = load_convert()


FULL_CFG = {
    "architectures": ["Glm4MoeForCausalLM"],
    "attention_bias": True,
    "eos_token_id": [151329, 151336, 151338],
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "partial_rotary_factor": 0.5,
    "intermediate_size": 12288,
    "max_position_embeddings": 202752,
    "model_type": "glm4_moe",
    "moe_intermediate_size": 1536,
    "norm_topk_prob": True,
    "num_attention_heads": 96,
    "n_group": 1,
    "topk_group": 1,
    "n_routed_experts": 160,
    "n_shared_experts": 1,
    "routed_scaling_factor": 2.5,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 3,
    "num_hidden_layers": 92,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-5,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "num_nextn_predict_layers": 1,
    "tie_word_embeddings": False,
    "use_qk_norm": True,
    "vocab_size": 151552,
}


def glm53_config():
    kda = [i for i in range(45) if i % 4 != 3]
    full = [i for i in range(45) if i % 4 == 3]
    return {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "image_start_token_id": 154830,
        "image_end_token_id": 154831,
        "video_start_token_id": 154832,
        "video_end_token_id": 154833,
        "image_token_id": 154854,
        "video_token_id": 154855,
        "quantization_config": {
            "quant_method": "fp8", "fmt": "e4m3",
            "activation_scheme": "dynamic", "weight_block_size": [128, 128],
        },
        "vision_config": {
            "model_type": "glm5_next_vision", "depth": 24,
            "hidden_size": 1024, "intermediate_size": 4096,
            "out_hidden_size": 4096, "num_heads": 16,
            "image_size": 448, "patch_size": 14,
            "temporal_patch_size": 2, "spatial_merge_size": 2,
            "in_channels": 3, "rms_norm_eps": 1e-5,
            "hidden_act": "silu", "swiglu_limit": 10.0,
            "attention_bias": True,
        },
        "text_config": {
            "model_type": "glm5_next_text",
            "eos_token_id": [154820, 154827, 154829],
            "num_hidden_layers": 45, "hidden_size": 4096,
            "vocab_size": 154880, "n_routed_experts": 288,
            "num_experts_per_tok": 8, "n_shared_experts": 1,
            "first_k_dense_replace": 3, "intermediate_size": 12288,
            "moe_intermediate_size": 2048, "num_attention_heads": 64,
            "num_key_value_heads": 64, "q_lora_rank": 1536,
            "kv_lora_rank": 512, "qk_head_dim": 256,
            "qk_nope_head_dim": 256, "qk_rope_head_dim": 0,
            "v_head_dim": 256, "index_topk": 2048,
            "index_head_dim": 128, "index_n_heads": 32,
            "index_kpool": 4, "index_kpool_compress": True,
            "index_kpool_always_select_tail": True,
            "indexer_rope_interleave": True,
            "max_position_embeddings": 1048576, "rms_norm_eps": 1e-5,
            "hidden_act": "silu", "swiglu_limit": 10.0,
            "routed_scaling_factor": 2.5, "topk_method": "noaux_tc",
            "scoring_func": "sigmoid", "norm_topk_prob": True,
            "n_group": 1, "topk_group": 1, "attention_bias": False,
            "tie_word_embeddings": False, "mhc": True, "hc_mult": 4,
            "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
            "mla_use_nope": True, "moe_router_dtype": "float32",
            "num_nextn_predict_layers": 1,
            "layer_types": [
                "linear_attention" if i in kda
                else "deepseek_sparse_attention" for i in range(45)
            ],
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
            "indexer_types": ["full"] * 45,
            "linear_attn_config": {
                "kda_layers": kda, "full_attn_layers": full,
                "num_heads": 64, "head_dim": 128,
                "short_conv_kernel_size": 4, "gate_lower_bound": -5.0,
            },
        },
    }


def flattened_glm53_config():
    raw = glm53_config()
    return {**raw["text_config"],
            "_outer": {k: v for k, v in raw.items() if k != "text_config"}}


def glm53_source_meta():
    """The complete positive text-header contract, without model bytes."""
    meta = {}

    def add(name, shape, dtype="BF16"):
        meta[name] = {"shape": list(shape), "dtype": dtype}

    def fp8(name, shape):
        add(name, shape, "F8_E4M3")
        add(name + "_scale_inv", [(int(d) + 127) // 128 for d in shape],
            "F32")

    p, H, heads, kd, qa, kv, qd, vh = (
        "model.language_model.", 4096, 64, 128, 1536, 512, 256, 256)
    add(p + "embed_tokens.weight", (154880, H))
    add(p + "norm.weight", (H,))
    add("lm_head.weight", (154880, H))
    kda_layers = {i for i in range(45) if i % 4 != 3}
    for layer in range(45):
        base = f"{p}layers.{layer}."
        for branch in ("attn", "ffn"):
            add(base + f"hc_{branch}_base", (24,), "F32")
            add(base + f"hc_{branch}_fn", (24, 4 * H))
            add(base + f"hc_{branch}_scale", (3,), "F32")
        add(base + "input_layernorm.weight", (H,))
        add(base + "post_attention_layernorm.weight", (H,))
        attn = base + "self_attn."
        if layer in kda_layers:
            add(attn + "A_log", (heads,), "F32")
            add(attn + "dt_bias", (heads * kd,), "F32")
            add(attn + "b_proj.weight", (heads, H))
            for projection in ("q", "k", "v"):
                add(attn + f"{projection}_proj.weight", (heads * kd, H))
                add(attn + f"{projection}_conv1d.weight", (heads * kd, 1, 4))
            for projection in ("f", "g"):
                add(attn + f"{projection}_a_proj.weight", (kd, H))
                add(attn + f"{projection}_b_proj.weight", (heads * kd, kd))
            add(attn + "o_norm.weight", (kd,))
            add(attn + "o_proj.weight", (H, heads * kd))
        else:
            fp8(attn + "q_a_proj.weight", (qa, H))
            add(attn + "q_a_layernorm.weight", (qa,))
            fp8(attn + "q_b_proj.weight", (heads * qd, qa))
            fp8(attn + "kv_a_proj_with_mqa.weight", (kv, H))
            add(attn + "kv_a_layernorm.weight", (kv,))
            add(attn + "kv_b_proj.weight", (heads * (qd + vh), kv))
            fp8(attn + "o_proj.weight", (H, heads * vh))
            ix = attn + "indexer."
            add(ix + "wq_b.weight", (32 * 128, qa))
            add(ix + "wk.weight", (128, H))
            add(ix + "k_norm.weight", (128,))
            add(ix + "k_norm.bias", (128,))
            add(ix + "weights_proj.weight", (32, H))
            add(ix + "index_kpool_compress_ape", (4, 128))
            add(ix + "index_kpool_compress_gate", (128, H))
        mlp = base + "mlp."
        if layer < 3:
            fp8(mlp + "gate_proj.weight", (12288, H))
            fp8(mlp + "up_proj.weight", (12288, H))
            fp8(mlp + "down_proj.weight", (H, 12288))
        else:
            add(mlp + "gate.weight", (288, H))
            add(mlp + "gate.e_score_correction_bias", (288,), "F32")
            fp8(mlp + "shared_experts.gate_proj.weight", (2048, H))
            fp8(mlp + "shared_experts.up_proj.weight", (2048, H))
            fp8(mlp + "shared_experts.down_proj.weight", (H, 2048))
            for expert in range(288):
                ep = mlp + f"experts.{expert}."
                fp8(ep + "gate_proj.weight", (2048, H))
                fp8(ep + "up_proj.weight", (2048, H))
                fp8(ep + "down_proj.weight", (H, 2048))
    mtp = p + "layers.45."
    add(mtp + "shared_head.norm.weight", (H,))
    add(mtp + "hnorm.weight", (H,))
    add(mtp + "enorm.weight", (H,))
    add(mtp + "eh_proj.weight", (H, 2 * H))
    add("model.visual.patch_embed.proj.weight", (1024, 3, 2, 14, 14))
    add("model.visual.post_layernorm.weight", (1024,))
    return meta


def full_source_meta():
    """A header-only official-layout fixture: exact names/shapes, no weights."""
    meta = {}

    def add(name, shape, dtype="BF16"):
        meta[name] = {"shape": list(shape), "dtype": dtype}

    H, hd, qrows, kvrows = 5120, 128, 96 * 128, 8 * 128
    add("model.embed_tokens.weight", (151552, H))
    add("model.norm.weight", (H,))
    add("lm_head.weight", (151552, H))
    for layer in (0, 91):
        base = f"model.layers.{layer}."
        add(base + "input_layernorm.weight", (H,))
        add(base + "post_attention_layernorm.weight", (H,))
        attn = base + "self_attn."
        add(attn + "q_proj.weight", (qrows, H))
        add(attn + "q_proj.bias", (qrows,))
        add(attn + "k_proj.weight", (kvrows, H))
        add(attn + "k_proj.bias", (kvrows,))
        add(attn + "v_proj.weight", (kvrows, H))
        add(attn + "v_proj.bias", (kvrows,))
        add(attn + "o_proj.weight", (H, qrows))
        add(attn + "q_norm.weight", (hd,))
        add(attn + "k_norm.weight", (hd,))
    for layer in (0, 2):
        mlp = f"model.layers.{layer}.mlp."
        add(mlp + "gate_proj.weight", (12288, H))
        add(mlp + "up_proj.weight", (12288, H))
        add(mlp + "down_proj.weight", (H, 12288))
    for layer in range(3, 92):
        mlp = f"model.layers.{layer}.mlp."
        if layer in (3, 91):
            add(mlp + "gate.weight", (160, H))
            add(mlp + "gate.e_score_correction_bias", (160,), "F32")
            add(mlp + "shared_experts.gate_proj.weight", (1536, H))
            add(mlp + "shared_experts.up_proj.weight", (1536, H))
            add(mlp + "shared_experts.down_proj.weight", (H, 1536))
        for expert in range(160):
            ep = mlp + f"experts.{expert}."
            add(ep + "gate_proj.weight", (1536, H))
            add(ep + "up_proj.weight", (1536, H))
            add(ep + "down_proj.weight", (H, 1536))
    # One name is enough to bind the declared optional MTP head to layer 92.
    add("model.layers.92.shared_head.norm.weight", (H,))
    return meta


class HeaderFixture:
    def __init__(self, meta):
        self.meta = meta

    def names(self):
        return self.meta.keys()

    def tensor_meta(self, name):
        return self.meta[name]

    def have(self, name):
        return name in self.meta


class GlmConversionBoundaryTest(unittest.TestCase):
    def test_glm53_text_contract_is_bounded_explicit_and_prefix_safe(self):
        source_cfg = flattened_glm53_config()
        cfg = CONVERT.normalise_cfg(source_cfg)
        self.assertEqual(cfg["source_max_position_embeddings"], 1048576)
        self.assertEqual(cfg["max_position_embeddings"], 2048)
        self.assertEqual(cfg["dsa_dense_context_limit"], 2048)
        self.assertEqual(cfg["kda_qk_l2_norm_eps"], 1e-6)
        self.assertEqual(cfg["swiglu_limit"], 10.0)
        self.assertEqual(cfg["hc_mult"], 4)
        self.assertEqual(cfg["hc_sinkhorn_iters"], 20)
        self.assertEqual(
            cfg["linear_attn_config"]["kda_layer_index_base"], 0)
        self.assertEqual(cfg["linear_attn_config"]["kda_layers"][:4],
                         [0, 1, 2, 4])

        self.assertEqual(CONVERT.source_model_prefix(source_cfg,
                                                     "language_model."),
                         "model.language_model.")
        self.assertEqual(CONVERT.runtime_tensor_name(
            "model.language_model.layers.4.hc_attn_fn", source_cfg,
            "language_model."),
            "language_model.model.layers.4.hc_attn_fn")
        self.assertEqual(CONVERT.runtime_tensor_name(
            "lm_head.weight", source_cfg, "language_model."),
            "language_model.lm_head.weight")
        self.assertTrue(CONVERT.is_f32_trunk_tensor(
            "language_model.model.layers.4.hc_attn_fn", source_cfg))
        self.assertFalse(CONVERT.is_f32_trunk_tensor(
            "language_model.model.layers.4.self_attn.q_proj.weight",
            source_cfg))
        self.assertFalse(CONVERT.emits_kimi_vision_json(source_cfg))
        self.assertTrue(CONVERT.emits_kimi_vision_json({
            "_outer": {"vision_config": {"hidden_size": 1024}},
            "model_type": "kimi_linear",
        }))

        self.assertTrue(CONVERT.is_omitted_source_tensor(
            "model.language_model.layers.3.self_attn.indexer.wk.weight",
            source_cfg, 45))
        self.assertTrue(CONVERT.is_omitted_source_tensor(
            "model.language_model.layers.45.shared_head.norm.weight",
            source_cfg, 45))
        self.assertTrue(CONVERT.is_omitted_source_tensor(
            "model.visual.patch_embed.proj.weight", source_cfg, 45))
        self.assertEqual(CONVERT.source_layer_index(
            "model.language_model.layers.45.shared_head.norm.weight"), 45)
        features = CONVERT.unsupported_source_features(source_cfg)
        self.assertEqual([f["name"] for f in features], [
            "deepseek_sparse_attention_indexer",
            "multi_token_prediction",
            "vision_tower",
        ])
        self.assertIn("text-only", features[-1]["reason"])

    def test_glm53_header_contract_passes_and_rejects_drift(self):
        cfg = flattened_glm53_config()
        meta = glm53_source_meta()
        source = HeaderFixture(meta)
        CONVERT.validate_glm53_source(cfg, source, "language_model.")
        layout, segment, kinds = CONVERT.moe_layout_at(
            source, "model.language_model.", 3)
        self.assertEqual((layout, segment), ("deepseek", "mlp"))
        self.assertEqual([tag for _kind, tag in kinds],
                         ["gate_proj", "up_proj", "down_proj"])

        name = "model.language_model.layers.44.hc_ffn_fn"
        meta[name] = {"shape": [24, 16383], "dtype": "BF16"}
        with self.assertRaisesRegex(ValueError, "layers.44.hc_ffn_fn shape"):
            CONVERT.validate_glm53_source(cfg, source, "language_model.")

    def test_glm53_trunk_reads_source_prefix_maps_runtime_and_drops_omissions(self):
        cfg = flattened_glm53_config()
        kept = ["model.language_model.layers.0.hc_attn_fn",
                "model.language_model.layers.0.hc_attn_scale"]
        omitted = [
            "model.language_model.layers.3.self_attn.indexer.wk.weight",
            "model.language_model.layers.45.shared_head.norm.weight",
            "model.visual.patch_embed.proj.weight",
        ]

        class FakeTensor:
            def __init__(self, shape):
                self.shape = shape

            def dim(self):
                return len(self.shape)

            def numel(self):
                total = 1
                for dim in self.shape:
                    total *= dim
                return total

            def float(self):
                return self

        class FakeSource:
            def names(self):
                return kept + omitted

        class FakeWeights:
            def __init__(self):
                self.read = []

            def have(self, _name):
                return True

            def tensor(self, name):
                self.read.append(name)
                return FakeTensor((3,) if name.endswith("_scale")
                                  else (24, 16384))

        args = types.SimpleNamespace(skip_trunk=False, trunk8=False,
                                     trunk_bits=4)
        weights = FakeWeights()
        old_raw = CONVERT.raw_bytes
        CONVERT.raw_bytes = lambda _tensor: b"hc-f32"
        try:
            with tempfile.TemporaryDirectory(prefix="glm53-trunk-") as tmp:
                args.out = tmp
                index = CONVERT.build_trunk(
                    args, FakeSource(), weights, None,
                    os.path.join(tmp, "manifest.json"), 45, cfg)
        finally:
            CONVERT.raw_bytes = old_raw
        self.assertEqual(weights.read, kept)
        self.assertEqual([entry["name"] for entry in index], [
            "language_model.model.layers.0.hc_attn_fn",
            "language_model.model.layers.0.hc_attn_scale",
        ])
        self.assertEqual([entry["fmt"] for entry in index],
                         [CONVERT.FMT_F32, CONVERT.FMT_F32])

    def test_glm53_contradictory_implicit_math_fails_closed(self):
        cfg = flattened_glm53_config()
        cfg["kda_qk_l2_norm_eps"] = 1e-5
        with self.assertRaisesRegex(ValueError, "L2 epsilon"):
            CONVERT.normalise_cfg(cfg)

        cfg = flattened_glm53_config()
        cfg["linear_attn_config"] = dict(cfg["linear_attn_config"],
                                          kda_layer_index_base=1)
        with self.assertRaisesRegex(ValueError, "zero-based"):
            CONVERT.normalise_cfg(cfg)

    def test_glm52_dense_equivalent_contract_is_explicit_and_bounded(self):
        source_cfg = {
            "architectures": ["GlmMoeDsaForCausalLM"],
            "model_type": "glm_moe_dsa",
            "max_position_embeddings": 1048576,
            "norm_topk_prob": True,
            "eos_token_id": [154820, 154827, 154829],
            "rope_interleave": True,
            "rope_parameters": {
                "rope_type": "default", "rope_theta": 8000000,
            },
        }
        cfg = CONVERT.normalise_cfg(source_cfg)
        self.assertEqual(cfg["source_max_position_embeddings"], 1048576)
        self.assertEqual(cfg["max_position_embeddings"], 2048)
        self.assertEqual(cfg["dsa_dense_context_limit"], 2048)
        self.assertEqual(cfg["rope_theta"], 8000000)
        self.assertEqual(cfg["mla_rms_norm_eps"], 1e-6)
        self.assertEqual(cfg["eos_token_ids"], [154820, 154827, 154829])
        self.assertIs(cfg["moe_renormalize"], True)
        self.assertTrue(CONVERT.is_omitted_source_tensor(
            "model.layers.6.self_attn.indexer.wk.weight", source_cfg, 78))
        self.assertFalse(CONVERT.is_omitted_source_tensor(
            "model.layers.6.self_attn.kv_b_proj.weight", source_cfg, 78))
        features = CONVERT.unsupported_source_features(
            dict(source_cfg, num_hidden_layers=78,
                 num_nextn_predict_layers=1))
        self.assertEqual(features[0]["name"],
                         "deepseek_sparse_attention_indexer")
        self.assertEqual(features[0]["action"], "dense_equivalent")
        self.assertEqual(features[1]["source_layers"], [78])

    def test_glm52_nondefault_rope_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "default rope_parameters"):
            CONVERT.normalise_cfg({
                "architectures": ["GlmMoeDsaForCausalLM"],
                "model_type": "glm_moe_dsa",
                "rope_parameters": {"rope_type": "yarn", "rope_theta": 1},
            })

    def test_glm_lite_implicit_contract_is_made_explicit(self):
        cfg = CONVERT.normalise_cfg({
            "model_type": "glm4_moe_lite",
            "eos_token_id": [154820, 154827, 154829],
        })
        self.assertEqual(cfg["eos_token_id"], 154820)
        self.assertEqual(cfg["eos_token_ids"], [154820, 154827, 154829])
        self.assertEqual(cfg["mla_rms_norm_eps"], 1e-6)
        self.assertIs(cfg["rope_interleave"], True)

    def test_explicit_rope_layout_is_never_overwritten(self):
        cfg = CONVERT.normalise_cfg({
            "model_type": "glm4_moe_lite", "rope_interleave": False,
        })
        self.assertIs(cfg["rope_interleave"], False)

    def test_appended_mtp_layer_is_not_base_trunk_or_expert_work(self):
        base = "model.layers.46.mlp.experts.0.gate_proj.weight"
        mtp = "model.layers.47.mlp.experts.0.gate_proj.weight"
        wrapped = "language_model.model.layers.47.shared_head.norm.weight"
        vision = "vision_tower.encoder.layers.47.mlp.weight"
        self.assertFalse(CONVERT.is_source_only_layer(base, 47))
        self.assertTrue(CONVERT.is_source_only_layer(mtp, 47))
        self.assertTrue(CONVERT.is_source_only_layer(wrapped, 47))
        self.assertFalse(CONVERT.is_source_only_layer(vision, 47))
        self.assertEqual(
            CONVERT.ShardDebt.consumer(mtp, 47),
            CONVERT.ShardDebt.DROP)
        self.assertEqual(
            CONVERT.ShardDebt.consumer(base, 47),
            ("layer", 46))

    def test_official_full_config_and_separate_experts_pass(self):
        source = HeaderFixture(full_source_meta())
        CONVERT.validate_glm47_full_source(FULL_CFG, source)
        layout, segment, kinds = CONVERT.moe_layout(source, "", 3)
        self.assertEqual(layout, "deepseek")
        self.assertEqual(segment, "mlp")
        self.assertEqual([tag for _kind, tag in kinds],
                         ["gate_proj", "up_proj", "down_proj"])

        cfg = CONVERT.normalise_cfg(FULL_CFG)
        self.assertEqual(cfg["num_experts"], 160)
        self.assertEqual(cfg["num_experts_per_token"], 8)
        self.assertEqual(cfg["num_shared_experts"], 1)
        self.assertIs(cfg["moe_renormalize"], True)
        self.assertEqual(cfg["eos_token_id"], 151329)
        self.assertEqual(cfg["eos_token_ids"], [151329, 151336, 151338])
        self.assertEqual(cfg["topk_method"], "noaux_tc")
        self.assertEqual(cfg["moe_router_activation_func"], "sigmoid")
        self.assertEqual(CONVERT.unsupported_source_features(FULL_CFG), [{
            "name": "multi_token_prediction",
            "source_layers": [92],
            "action": "omitted",
            "reason": "unsupported",
        }])

    def test_packed_transformers_expert_representation_is_rejected(self):
        meta = full_source_meta()
        meta["model.layers.3.mlp.experts.gate_up_proj"] = {
            "shape": [160, 3072, 5120], "dtype": "BF16"}
        with self.assertRaisesRegex(ValueError, "packed 3-D expert storage"):
            CONVERT.validate_glm47_full_source(FULL_CFG, HeaderFixture(meta))

    def test_conflicting_explicit_router_semantics_are_rejected(self):
        source = HeaderFixture(full_source_meta())
        for key, value in (("topk_method", "greedy"),
                           ("moe_router_activation_func", "softmax")):
            with self.subTest(key=key):
                cfg = dict(FULL_CFG, **{key: value})
                with self.assertRaisesRegex(ValueError, key):
                    CONVERT.validate_glm47_full_source(cfg, source)

    def test_every_separate_expert_shape_is_checked(self):
        meta = full_source_meta()
        name = "model.layers.50.mlp.experts.73.down_proj.weight"
        meta[name] = {"shape": [5119, 1536], "dtype": "BF16"}
        with self.assertRaisesRegex(ValueError, "layers.50.*experts.73"):
            CONVERT.validate_glm47_full_source(FULL_CFG, HeaderFixture(meta))

    def test_declared_mtp_layer_must_be_exactly_layer_92(self):
        meta = full_source_meta()
        del meta["model.layers.92.shared_head.norm.weight"]
        meta["model.layers.93.shared_head.norm.weight"] = {
            "shape": [5120], "dtype": "BF16"}
        with self.assertRaisesRegex(ValueError, "appended source layers"):
            CONVERT.validate_glm47_full_source(FULL_CFG, HeaderFixture(meta))

    def test_tiny_full_container_exercises_gqa_and_omits_mtp(self):
        with tempfile.TemporaryDirectory(prefix="tiny-glm47-") as tmp:
            out = os.path.join(tmp, "tiny.waste")
            subprocess.run([
                sys.executable,
                os.path.join(REPO, "tools", "make_test_container.py"),
                out, "--glm47-full",
            ], check=True, stdout=subprocess.PIPE, text=True)
            with open(os.path.join(out, "manifest.json")) as inp:
                man = json.load(inp)
            self.assertEqual(man["arch"], "Glm4MoeForCausalLM")
            self.assertEqual(list(man["layers"]), ["3"])
            self.assertEqual(man["source_ignored_layers"], [4])
            self.assertEqual(man["unsupported_features"][0]["name"],
                             "multi_token_prediction")
            names = {entry["name"] for entry in man["trunk"]}
            for tail in ("q_proj.bias", "k_proj.bias", "v_proj.bias",
                         "q_norm.weight", "k_norm.weight"):
                self.assertIn("model.layers.0.self_attn." + tail, names)
            self.assertNotIn("model.layers.4.input_layernorm.weight", names)


if __name__ == "__main__":
    unittest.main()
