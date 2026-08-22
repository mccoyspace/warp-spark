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
    torch = types.ModuleType("torch")
    torch.device = lambda value: value
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False))
    sys.modules.setdefault("torch", torch)
    mx = types.ModuleType("mxfp4")
    mx.ST = object
    mx.unblock_scale = lambda value, scale, block: value
    sys.modules.setdefault("mxfp4", mx)
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import convert
    return convert


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
