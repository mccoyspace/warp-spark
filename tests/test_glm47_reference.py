#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Tiny full-GLM differential against an independent PyTorch forward.

Run after ``make test_forward``:

  python3 tests/test_glm47_reference.py

The fixture is regenerated from seed zero. No model download or golden file is
needed, and negative controls prove the chosen prompt actually distinguishes
the semantics this test claims to cover.
"""

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

try:
    import torch
except ImportError:                                             # pragma: no cover
    torch = None


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDS = [3, 7, 11, 5, 9, 13, 2]

if torch is not None:
    sys.path.insert(0, os.path.join(REPO, "tools"))
    from glm47_ref import Glm47Ref                               # noqa: E402
    from kimi_ref import Container                               # noqa: E402


@unittest.skipIf(torch is None, "PyTorch is required for the independent oracle")
class Glm47ReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.mkdtemp(prefix="glm47-ref-")
        cls.container_path = os.path.join(cls.tmp, "tiny.waste")
        subprocess.run([
            sys.executable,
            os.path.join(REPO, "tools", "make_test_container.py"),
            cls.container_path, "--glm47-full", "--seed", "0",
        ], check=True, stdout=subprocess.PIPE, text=True)
        cls.container = Container(cls.container_path)
        with torch.no_grad():
            cls.logits, cls.routes = Glm47Ref(cls.container).forward(IDS)

    @classmethod
    def tearDownClass(cls):
        for bank, _meta in cls.container.banks.values():
            bank.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def variant(self, **kwargs):
        with torch.no_grad():
            return Glm47Ref(self.container, **kwargs).forward(IDS)

    def test_fixture_distinguishes_each_attention_obligation(self):
        canonical = self.logits[-1]
        variants = {
            "QKV bias": {"qkv_bias": False},
            "Q/K norm": {"qk_norm": False},
            "half-split partial RoPE": {"rope_layout": "interleaved"},
            "12:1 grouped KV mapping": {"kv_map": "first"},
        }
        for name, kwargs in variants.items():
            with self.subTest(name=name):
                logits, _routes = self.variant(**kwargs)
                delta = float((canonical - logits[-1]).abs().max())
                self.assertGreater(delta, 1e-3, name + " negative control is inert")

    def test_fixture_distinguishes_correction_biased_selection(self):
        logits, routes = self.variant(correction=False)
        self.assertNotEqual(
            [row["experts"] for row in routes],
            [row["experts"] for row in self.routes],
            "correction-bias negative control did not change any route")
        self.assertGreater(
            float((self.logits[-1] - logits[-1]).abs().max()), 1e-5)

    def test_engine_matches_logits_routes_and_router_weights(self):
        engine = os.path.join(REPO, "test_forward")
        if not os.path.isfile(engine):
            self.skipTest("make test_forward has not been run")
        actual_path = os.path.join(self.tmp, "engine.bin")
        route_path = os.path.join(self.tmp, "engine.routes")
        env = dict(os.environ, WASTE_DUMP_ROUTE=route_path)
        env.update({
            "WASTE_BACKEND": "cpu",
            "WASTE_CHUNK": "0",
            "WASTE_CUDA_KDA": "0",
            "WASTE_CUDA_DENSE": "0",
            "WASTE_CUDA_VQ": "0",
            "WASTE_SDOT": "0",
            "WASTE_I8MM": "0",
        })
        run = subprocess.run([
            engine, self.container_path, ",".join(map(str, IDS)),
            actual_path, "0",
        ], cwd=REPO, env=env, text=True, stdout=subprocess.PIPE,
           stderr=subprocess.PIPE)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

        with open(actual_path, "rb") as inp:
            raw = inp.read()
        actual = torch.tensor(struct.unpack(f"<{len(raw) // 4}f", raw))
        expected = self.logits[-1].float()
        self.assertEqual(actual.numel(), expected.numel())
        self.assertEqual(int(actual.argmax()), int(expected.argmax()))
        self.assertLess(float((actual - expected).abs().max()), 1e-3)

        engine_routes = []
        with open(route_path) as inp:
            for line in inp:
                fields = line.split()
                top_k = self.container.cfg["num_experts_per_token"]
                engine_routes.append({
                    "position": int(fields[0]),
                    "layer": int(fields[1]),
                    "experts": [int(v) for v in fields[2:2 + top_k]],
                    "weights": [float(v) for v in
                                fields[2 + top_k:2 + 2 * top_k]],
                })
        self.assertEqual(len(engine_routes), len(self.routes))
        for actual_route, expected_route in zip(engine_routes, self.routes):
            self.assertEqual(actual_route["position"], expected_route["position"])
            self.assertEqual(actual_route["layer"], expected_route["layer"])
            self.assertEqual(actual_route["experts"], expected_route["experts"])
            for actual_weight, expected_weight in zip(
                    actual_route["weights"], expected_route["weights"]):
                self.assertAlmostEqual(actual_weight, expected_weight, places=4)


if __name__ == "__main__":
    unittest.main()
