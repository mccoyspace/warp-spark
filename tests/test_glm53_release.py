#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.

import importlib.util
import os
import sys
import unittest


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "check_glm53_release",
    os.path.join(REPO, "tools", "check_glm53_release.py"))
CHECK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = CHECK
SPEC.loader.exec_module(CHECK)


def valid_config():
    text = dict(CHECK.TEXT_EXPECTED)
    text["linear_attn_config"] = {
        **CHECK.LINEAR_EXPECTED,
        "kda_layers": list(CHECK.KDA_LAYERS),
        "full_attn_layers": list(CHECK.FULL_ATTN_LAYERS),
    }
    text["layer_types"] = [
        "linear_attention" if i in CHECK.KDA_LAYERS
        else "deepseek_sparse_attention" for i in range(45)]
    text["mlp_layer_types"] = ["dense"] * 3 + ["sparse"] * 42
    text["indexer_types"] = ["full"] * 45
    return {
        **CHECK.OUTER_EXPECTED,
        "text_config": text,
        "vision_config": dict(CHECK.VISION_EXPECTED),
        "quantization_config": dict(CHECK.QUANT_EXPECTED),
        "transformers_version": "post-training metadata may move",
    }


class Glm53ReleaseCheckTest(unittest.TestCase):
    def test_exact_execution_contract_passes(self):
        self.assertEqual(CHECK.config_differences(valid_config()), [])

    def test_metadata_can_change(self):
        cfg = valid_config()
        cfg["transformers_version"] = "future"
        cfg["chat_template"] = "new post-training contract"
        self.assertEqual(CHECK.config_differences(cfg), [])

    def test_mhc_change_fails_closed(self):
        cfg = valid_config()
        cfg["text_config"]["hc_mult"] = 8
        self.assertIn("text_config.hc_mult", CHECK.config_differences(cfg)[0])

    def test_hybrid_schedule_change_fails_closed(self):
        cfg = valid_config()
        cfg["text_config"]["linear_attn_config"]["kda_layers"] = []
        self.assertTrue(any("kda_layers" in item
                            for item in CHECK.config_differences(cfg)))

    def test_index_requires_every_referenced_shard(self):
        index = {
            "metadata": {"total_size": 123},
            "weight_map": {"a": "one.safetensors", "b": "two.safetensors"},
        }
        with self.assertRaisesRegex(CHECK.ReleaseCheckError, "absent shard"):
            CHECK.validate_index(index, frozenset({"one.safetensors"}))
        self.assertEqual(
            CHECK.validate_index(
                index, frozenset({"one.safetensors", "two.safetensors"})),
            (2, 123))

    def test_header_gate_binds_names_bytes_and_shards(self):
        index = {
            "metadata": {"total_size": 12},
            "weight_map": {"a": "one.safetensors", "b": "two.safetensors"},
        }
        headers = {
            "one.safetensors": ({"a": {
                "dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
                64, 16),
            "two.safetensors": ({"b": {
                "dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}},
                64, 16),
        }
        result = CHECK.validate_header_set(index, headers)
        self.assertEqual(result["tensors"], 2)
        self.assertEqual(result["payload_bytes"], 12)
        headers["two.safetensors"][0]["b"]["data_offsets"] = [0, 2]
        with self.assertRaisesRegex(CHECK.ReleaseCheckError, "byte range"):
            CHECK.validate_header_set(index, headers)


if __name__ == "__main__":
    unittest.main()
