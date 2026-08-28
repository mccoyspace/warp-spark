#!/usr/bin/env python3
"""Regression checks for GLM-5.3 source names used by container verification."""

import importlib.util
import os
import pathlib
import tempfile
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

    def names(self):
        return iter(self.present)


class Glm53VerifierPrefixTest(unittest.TestCase):
    @staticmethod
    def _mtp_fixture(container):
        cfg = {
            "model_type": "glm5_next_text",
            "_outer": {
                "architectures": ["Glm5NextForConditionalGeneration"],
            },
            "num_hidden_layers": 45,
            "num_nextn_predict_layers": 1,
            "n_routed_experts": 288,
            "mtp_layers": 1,
            "mtp_source_layer": 45,
        }
        source_prefix = "model.language_model.layers.45."
        source = _Source({
            source_prefix + "enorm.weight",
            source_prefix + "eh_proj.weight",
            source_prefix + "mlp.gate.e_score_correction_bias",
            source_prefix + "mlp.experts.0.gate_proj.weight",
            source_prefix + "self_attn.indexer.wk.weight",
            source_prefix + "self_attn.q_a_proj.weight_scale_inv",
        })
        bank = {"file": "experts-L45.bin", "experts": 288,
                "bytes": 4, "codebook_base": 378}
        with open(os.path.join(container, bank["file"]), "wb") as out:
            out.write(b"bank")
        man = {
            "config": cfg,
            "tensor_prefix": "language_model.",
            "layers": {},
            "trunk": [
                {"name": "language_model.model.layers.45.enorm.weight"},
                {"name": "language_model.model.layers.45.eh_proj.weight"},
                {"name": "language_model.model.layers.45.block_sparse_moe."
                         "gate.e_score_correction_bias"},
            ],
            "source_ignored_layers": [],
            "unsupported_features": VERIFY.unsupported_source_features(
                cfg, mtp=True),
            "mtp": VERIFY.glm53_mtp_manifest(cfg, bank),
        }
        return man, source

    def _validate_without_full_header_fixture(self, man, source, container,
                                               bank_sound=True):
        old_validate = VERIFY.validate_glm53_source
        old_bank = VERIFY.bank_is_sound
        VERIFY.validate_glm53_source = lambda *_args, **_kwargs: None
        VERIFY.bank_is_sound = lambda *_args, **_kwargs: bank_sound
        try:
            return VERIFY.validate_manifest_source_contract(
                man, source, container)
        finally:
            VERIFY.validate_glm53_source = old_validate
            VERIFY.bank_is_sound = old_bank

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

    def test_mtp_contract_checks_distinct_bank_trunk_and_ignored_layer(self):
        with tempfile.TemporaryDirectory(prefix="verify-glm53-mtp-") as tmp:
            man, source = self._mtp_fixture(tmp)
            banks = self._validate_without_full_header_fixture(
                man, source, tmp)
        self.assertEqual(banks, [("45", man["mtp"]["bank"])])

    def test_non_mtp_contract_still_requires_layer_45_to_be_ignored(self):
        with tempfile.TemporaryDirectory(prefix="verify-glm53-base-") as tmp:
            man, source = self._mtp_fixture(tmp)
            del man["mtp"]
            del man["config"]["mtp_layers"]
            del man["config"]["mtp_source_layer"]
            man["trunk"] = []
            man["source_ignored_layers"] = [45]
            man["unsupported_features"] = VERIFY.unsupported_source_features(
                man["config"])
            banks = self._validate_without_full_header_fixture(
                man, source, tmp)
        self.assertEqual(banks, [])

    def test_mtp_contract_fails_on_ignored_layer_missing_trunk_or_bad_bank(self):
        with tempfile.TemporaryDirectory(prefix="verify-glm53-mtp-bad-") as tmp:
            man, source = self._mtp_fixture(tmp)
            man["source_ignored_layers"] = [45]
            with self.assertRaisesRegex(AssertionError, "ignored-layer"):
                self._validate_without_full_header_fixture(man, source, tmp)

            man, source = self._mtp_fixture(tmp)
            man["trunk"] = man["trunk"][:-1]
            with self.assertRaisesRegex(AssertionError, "MTP trunk tensors missing"):
                self._validate_without_full_header_fixture(man, source, tmp)

            man, source = self._mtp_fixture(tmp)
            with self.assertRaisesRegex(AssertionError, "invalid record geometry"):
                self._validate_without_full_header_fixture(
                    man, source, tmp, bank_sound=False)


if __name__ == "__main__":
    unittest.main()
