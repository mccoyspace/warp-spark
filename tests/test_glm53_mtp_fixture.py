#!/usr/bin/env python3
"""Structural gates for the opt-in tiny GLM-5.3 MTP container."""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "make_glm53_fixture.py"
SPEC = importlib.util.spec_from_file_location("make_glm53_fixture", FIXTURE_PATH)
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)


class Glm53MtpFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="glm53-mtp-fixture-")
        root = Path(cls.temp.name)
        cls.base = root / "base.waste"
        cls.mtp = root / "mtp.waste"
        FIXTURE.build(cls.base, 17)
        FIXTURE.build(cls.mtp, 17, mtp=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def manifest(path):
        return json.loads((path / "manifest.json").read_text())

    def test_default_fixture_still_omits_mtp_completely(self):
        manifest = self.manifest(self.base)
        self.assertNotIn("mtp", manifest)
        self.assertNotIn("mtp_layers", manifest["config"])
        self.assertNotIn("mtp_source_layer", manifest["config"])
        self.assertEqual(manifest["source_ignored_layers"], [45])
        self.assertIn("multi_token_prediction", {
            item["name"] for item in manifest["unsupported_features"]})
        self.assertNotIn("45", manifest["layers"])
        self.assertFalse((self.base / "experts-L45.bin").exists())
        self.assertFalse(any(
            ".layers.45." in entry["name"] for entry in manifest["trunk"]))

    def test_opt_in_fixture_has_complete_separate_standard_residual_mtp(self):
        manifest = self.manifest(self.mtp)
        config = manifest["config"]
        self.assertEqual(config["mtp_layers"], 1)
        self.assertEqual(config["mtp_source_layer"], 45)
        self.assertEqual(config["dsa_dense_context_limit"], 2048)
        self.assertEqual(config["max_position_embeddings"], 2048)
        self.assertNotIn("source_ignored_layers", manifest)
        self.assertNotIn("multi_token_prediction", {
            item["name"] for item in manifest["unsupported_features"]})
        self.assertEqual(sorted(map(int, manifest["layers"])),
                         list(range(3, 45)))

        contract = manifest["mtp"]
        bank = contract["bank"]
        self.assertEqual({key: contract[key] for key in (
            "version", "num_layers", "source_layer", "context_limit",
            "attention", "recurrent")}, {
                "version": 1,
                "num_layers": 1,
                "source_layer": 45,
                "context_limit": 2048,
                "attention": "dense_equivalent_dsa",
                "recurrent": True,
            })
        self.assertEqual(bank["file"], "experts-L45.bin")
        self.assertEqual(bank["experts"], 8)
        self.assertEqual(bank["codebook_base"], 378)
        self.assertEqual((self.mtp / bank["file"]).stat().st_size,
                         bank["bytes"])

        names = {entry["name"] for entry in manifest["trunk"]}
        p = "language_model.model.layers.45."
        required = {
            p + "enorm.weight",
            p + "hnorm.weight",
            p + "eh_proj.weight",
            p + "shared_head.norm.weight",
            p + "input_layernorm.weight",
            p + "post_attention_layernorm.weight",
            p + "self_attn.q_a_proj.weight",
            p + "self_attn.q_a_layernorm.weight",
            p + "self_attn.q_b_proj.weight",
            p + "self_attn.kv_a_proj_with_mqa.weight",
            p + "self_attn.kv_a_layernorm.weight",
            p + "self_attn.kv_b_proj.weight",
            p + "self_attn.o_proj.weight",
            p + "block_sparse_moe.gate.weight",
            p + "block_sparse_moe.gate.e_score_correction_bias",
            p + "block_sparse_moe.shared_experts.gate_proj.weight",
            p + "block_sparse_moe.shared_experts.up_proj.weight",
            p + "block_sparse_moe.shared_experts.down_proj.weight",
        }
        self.assertTrue(required <= names)
        mtp_names = {name for name in names if name.startswith(p)}
        self.assertEqual(mtp_names, required)
        self.assertFalse(any(".hc_" in name for name in mtp_names))
        self.assertFalse(any(".indexer." in name for name in mtp_names))

        codebook_record = 16 + 256 * 8 * 2
        n_books, rem = divmod(
            (self.mtp / "codebooks.bin").stat().st_size, codebook_record)
        self.assertEqual(rem, 0)
        self.assertEqual(bank["codebook_base"] + 9, n_books)

        data = (self.mtp / bank["file"]).read_bytes()
        offset, records = 0, []
        while offset < len(data):
            header = struct.unpack_from("<IHHBBHHHIIIIIIII", data, offset)
            records.append((header[0], header[1], header[2], header[5]))
            offset += header[8] * 4096
        self.assertEqual(offset, len(data))
        self.assertEqual([record[2] for record in records], list(range(8)))
        self.assertTrue(all(record[0] == 0x50584557 and record[1] == 45 and
                            record[3] == bank["codebook_base"]
                            for record in records))


if __name__ == "__main__":
    unittest.main()
