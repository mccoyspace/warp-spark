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


def config():
    value = {field: f"value:{field}" for field in CHECK.STRUCTURAL_FIELDS}
    value["quantization_config"] = {
        field: f"value:{field}" for field in CHECK.QUANT_FIELDS}
    return value


class Glm53ReleaseCheckTest(unittest.TestCase):
    def test_post_training_metadata_can_change(self):
        base = config()
        candidate = dict(base)
        candidate.update({
            "transformers_version": "future",
            "chat_template": "new post-training contract",
        })
        self.assertEqual(CHECK.structural_differences(base, candidate), [])

    def test_execution_geometry_change_fails_closed(self):
        base = config()
        candidate = config()
        candidate["index_topk"] = 1024
        self.assertEqual(
            CHECK.structural_differences(base, candidate),
            ["index_topk: baseline='value:index_topk', candidate=1024"])

    def test_fp8_recipe_change_fails_closed(self):
        base = config()
        candidate = config()
        candidate["quantization_config"] = dict(
            candidate["quantization_config"], weight_block_size=[256, 128])
        self.assertIn(
            "quantization_config.weight_block_size",
            CHECK.structural_differences(base, candidate)[0])

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


if __name__ == "__main__":
    unittest.main()
