#!/usr/bin/env python3
"""Regression checks for GLM-5.3 source names used by container verification."""

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_container", ROOT / "tools" / "verify_container.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class _Source:
    def __init__(self, present):
        self.present = set(present)
        self.probed = []

    def have(self, name):
        self.probed.append(name)
        return name in self.present


class Glm53VerifierPrefixTest(unittest.TestCase):
    def test_runtime_prefix_resolves_to_published_expert_namespace(self):
        cfg = {
            "model_type": "glm5_next_text",
            "_outer": {
                "architectures": ["Glm5NextForConditionalGeneration"],
            },
        }
        expert = (
            "model.language_model.layers.3.mlp.experts.0."
            "gate_proj.weight"
        )
        source = _Source({expert})

        model_prefix, layout, segment, kinds = VERIFY.source_moe_layout(
            source, cfg, "language_model.", 3)

        self.assertEqual(model_prefix, "model.language_model.")
        self.assertIsNotNone(layout)
        self.assertEqual(segment, "mlp")
        self.assertEqual(kinds[0], ("gate", "gate_proj"))
        self.assertIn(expert, source.probed)
        self.assertNotIn(
            "language_model.model.layers.3.mlp.experts.0.gate_proj.weight",
            source.probed,
        )


if __name__ == "__main__":
    unittest.main()
