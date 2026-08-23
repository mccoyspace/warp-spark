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

    def run_engine(self, label, *, chunked, gqa_chunk=None, reset=False,
                   dump_routes=True, ids=IDS):
        engine = os.path.join(REPO, "test_forward")
        if not os.path.isfile(engine):
            self.skipTest("make test_forward has not been run")
        actual_path = os.path.join(self.tmp, label + ".bin")
        route_path = os.path.join(self.tmp, label + ".routes")
        env = dict(os.environ)
        env.pop("WASTE_GQA_CHUNK_PREFILL", None)
        env.update({
            "WASTE_BACKEND": "cpu",
            "WASTE_CHUNK": "1" if chunked else "0",
            "WASTE_CUDA_KDA": "0",
            "WASTE_CUDA_DENSE": "0",
            "WASTE_CUDA_VQ": "0",
            "WASTE_SDOT": "0",
            "WASTE_I8MM": "0",
        })
        if gqa_chunk is not None:
            env["WASTE_GQA_CHUNK_PREFILL"] = str(gqa_chunk)
        if reset:
            env["WASTE_TEST_RESET_REPLAY"] = "1"
        if dump_routes:
            env["WASTE_DUMP_ROUTE"] = route_path
        else:
            env.pop("WASTE_DUMP_ROUTE", None)
        run = subprocess.run([
            engine, self.container_path, ",".join(map(str, ids)),
            actual_path, "0",
        ], cwd=REPO, env=env, text=True, stdout=subprocess.PIPE,
           stderr=subprocess.PIPE)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        with open(actual_path, "rb") as inp:
            raw = inp.read()
        actual = torch.tensor(struct.unpack(f"<{len(raw) // 4}f", raw))
        routes = {}
        if dump_routes:
            top_k = self.container.cfg["num_experts_per_token"]
            with open(route_path) as inp:
                for line in inp:
                    fields = line.split()
                    key = (int(fields[0]), int(fields[1]))
                    self.assertNotIn(key, routes)
                    routes[key] = {
                        "experts": [int(v) for v in fields[2:2 + top_k]],
                        "weights": [float(v) for v in
                                    fields[2 + top_k:2 + 2 * top_k]],
                    }
        return raw, actual, routes, run

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

    def test_layer_major_chunk_matches_sequential_contract(self):
        seq_raw, seq, seq_routes, _ = self.run_engine(
            "sequential", chunked=False)
        chunk_raw, chunk, chunk_routes, _ = self.run_engine(
            "layer-major", chunked=True, gqa_chunk=1)

        self.assertEqual(int(seq.argmax()), int(chunk.argmax()))
        self.assertEqual(
            torch.topk(seq, 10).indices.tolist(),
            torch.topk(chunk, 10).indices.tolist())
        self.assertLess(float((seq - chunk).abs().max()), 1e-4)
        self.assertEqual(seq_routes.keys(), chunk_routes.keys())
        for key in seq_routes:
            self.assertEqual(seq_routes[key]["experts"],
                             chunk_routes[key]["experts"])
            self.assertEqual(seq_routes[key]["weights"],
                             chunk_routes[key]["weights"])

        repeat_raw, _repeat, repeat_routes, _ = self.run_engine(
            "layer-major-repeat", chunked=True, gqa_chunk=1)
        self.assertEqual(chunk_raw, repeat_raw)
        self.assertEqual(chunk_routes, repeat_routes)
        self.assertNotEqual(seq_raw, chunk_raw,
                            "fixture no longer exercises fold-order drift")

    def test_layer_major_reset_replay_is_byte_exact(self):
        _raw, _actual, _routes, run = self.run_engine(
            "layer-major-reset", chunked=True, gqa_chunk=1,
            reset=True, dump_routes=False)
        self.assertIn("reset replay exact", run.stdout)

    def test_layer_major_64_plus_1_tail_matches_sequential(self):
        ids = (IDS * 10)[:65]
        _seq_raw, seq, seq_routes, _ = self.run_engine(
            "tail-sequential", chunked=False, ids=ids)
        chunk_raw, chunk, chunk_routes, _ = self.run_engine(
            "tail-layer-major", chunked=True, gqa_chunk=1, ids=ids)
        self.assertEqual(int(seq.argmax()), int(chunk.argmax()))
        self.assertEqual(torch.topk(seq, 10).indices.tolist(),
                         torch.topk(chunk, 10).indices.tolist())
        self.assertLess(float((seq - chunk).abs().max()), 1e-4)
        self.assertEqual(seq_routes, chunk_routes)

        replay_raw, _actual, _routes, run = self.run_engine(
            "tail-layer-major-reset", chunked=True, gqa_chunk=1,
            reset=True, dump_routes=False, ids=ids)
        self.assertEqual(chunk_raw, replay_raw)
        self.assertIn("reset replay exact", run.stdout)

    def test_layer_major_env_rejects_non_boolean_value(self):
        engine = os.path.join(REPO, "test_forward")
        env = dict(os.environ, WASTE_BACKEND="cpu",
                   WASTE_GQA_CHUNK_PREFILL="yes")
        run = subprocess.run([
            engine, self.container_path, ",".join(map(str, IDS)), "", "0",
        ], cwd=REPO, env=env, text=True, stdout=subprocess.PIPE,
           stderr=subprocess.PIPE)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("WASTE_GQA_CHUNK_PREFILL must be 0 or 1", run.stderr)


if __name__ == "__main__":
    unittest.main()
