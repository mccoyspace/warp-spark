#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Cheap GLM source-boundary checks; no model shards required."""

import os
import sys
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


if __name__ == "__main__":
    unittest.main()
