# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.

import contextlib
import io
import json
import math
import pathlib
import struct
import tempfile
import unittest

from tools.compare_backend_runs import (
    ComparisonError,
    compare,
    load_run,
    main,
    write_public_report,
)


class BackendComparatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.frames = ([5.0, 4.0, 3.0, 2.0], [6.0, 3.0, 2.0, 1.0])
        self.cpu = self.capture("cpu", self.frames)

    def tearDown(self):
        self.temp.cleanup()

    def capture(self, mode, frames, *, generated=(0, 0), routes=None,
                controls=None):
        path = self.root / f"{mode}-{len(list(self.root.iterdir()))}"
        path.mkdir()
        wants_matvec = mode in {"matvec", "combined"}
        wants_vq = mode in {"vq", "combined"}
        manifest = {
            "format": "waste-backend-qualification-v1",
            "mode": mode,
            "provider": None if mode == "cpu" else "test-provider",
            "build_info": "test build",
            "model": {
                "arch": "test", "quant_summary": "VQ3R", "n_layers": 2,
                "n_experts": 4, "top_k": 2, "first_dense": 1, "hidden": 8,
                "vq_scheme": 1, "vq_stages": 3, "vq_entries": 256,
                "vq_vec_dim": 8, "vq_index_bits": 8,
                "params_total": 1, "params_active": 1,
            },
            "capture": {
                "prompt_tokens": [10, 11],
                "generated_tokens": list(generated),
                "vocab": 4,
                "logit_steps": 2,
            },
            "config": {
                "ctx_tokens": 18, "ram_budget_bytes": 0, "n_threads": 0,
                "cpu_list": None, "use_direct_io": 1,
            },
            "controls": controls or {"WASTE_LOOKAHEAD": "0"},
            "resolved_memory": {"allocated_bytes": 1},
            "backend_calls": {
                "effective_capabilities": {
                    "cpu": 0, "matvec": 1, "vq": 2, "combined": 3,
                }[mode],
                "claim_queries": 2 if wants_matvec else 0,
                "claimed_tensors": 1 if wants_matvec else 0,
                "matvec_calls": 1 if wants_matvec else 0,
                "vq_begin_calls": 1 if wants_vq else 0,
                "vq_gate_up_calls": 2 if wants_vq else 0,
                "vq_down_calls": 2 if wants_vq else 0,
                "callback_failures": 0,
                "prefill_matvec_calls": 0,
                "prefill_vq_begin_calls": 0,
                "prefill_vq_gate_up_calls": 0,
                "prefill_vq_down_calls": 0,
                "prefill_callback_failures": 0,
            },
            "timing": {"decode_tok_s": 1.0},
            "stats": {"bytes_read": 0},
        }
        (path / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
        with (path / "logits.f32").open("wb") as stream:
            for frame in frames:
                stream.write(struct.pack("<4f", *frame))
        route_text = routes or (
            "0 1 2 3 0.6 0.4 -1 -1\n"
            "1 1 3 2 0.7 0.3 -1 -1\n"
            "2 1 2 3 0.6 0.4 -1 -1\n"
        )
        (path / "routes.txt").write_text(route_text, encoding="ascii")
        (path / "steps.tsv").write_text("step\n", encoding="ascii")
        return path

    def test_exact_vq_passes(self):
        vq = self.capture("vq", self.frames)
        result = compare(load_run(self.cpu), load_run(vq), "exact-vq",
                         1e-4, None, 10)
        self.assertTrue(result["byte_exact"])
        self.assertEqual(result["max_abs"], 0.0)

    def test_exact_vq_rejects_one_changed_float(self):
        changed = ([5.0, 4.0, 3.0, 2.0], [6.0, 3.0, 2.0, 1.000001])
        vq = self.capture("vq", changed)
        with self.assertRaisesRegex(ComparisonError, "not byte-exact"):
            compare(load_run(self.cpu), load_run(vq), "exact-vq",
                    1e-4, None, 10)

    def test_dense_bounded_passes(self):
        changed = ([5.00001, 4.0, 3.0, 2.0], [6.0, 3.00001, 2.0, 1.0])
        dense = self.capture("matvec", changed)
        result = compare(load_run(self.cpu), load_run(dense), "dense",
                         1e-4, None, 10)
        self.assertLess(result["max_abs"], 1e-4)

    def test_dense_bound_is_enforced(self):
        changed = ([5.001, 4.0, 3.0, 2.0], [6.0, 3.0, 2.0, 1.0])
        dense = self.capture("combined", changed)
        with self.assertRaisesRegex(ComparisonError, "exceeds"):
            compare(load_run(self.cpu), load_run(dense), "dense",
                    1e-4, None, 10)

    def test_dense_top_order_is_enforced_below_numeric_bound(self):
        cpu_frames = ([5.0, 4.00002, 4.0, 2.0], self.frames[1])
        cpu = self.capture("cpu", cpu_frames)
        changed = ([5.0, 3.99998, 4.00004, 2.0], self.frames[1])
        dense = self.capture("matvec", changed)
        with self.assertRaisesRegex(ComparisonError, "top-4 ordering"):
            compare(load_run(cpu), load_run(dense), "dense", 1e-4, None, 10)

    def test_routes_are_enforced(self):
        routes = (
            "0 1 3 2 0.6 0.4 -1 -1\n"
            "1 1 3 2 0.7 0.3 -1 -1\n"
            "2 1 2 3 0.6 0.4 -1 -1\n"
        )
        dense = self.capture("matvec", self.frames, routes=routes)
        with self.assertRaisesRegex(ComparisonError, "router selections"):
            compare(load_run(self.cpu), load_run(dense), "dense",
                    1e-4, None, 10)

    def test_nonfinite_is_enforced(self):
        changed = ([5.0, 4.0, math.nan, 2.0], self.frames[1])
        dense = self.capture("matvec", changed)
        with self.assertRaisesRegex(ComparisonError, "non-finite"):
            compare(load_run(self.cpu), load_run(dense), "dense",
                    1e-4, None, 10)

    def test_identically_truncated_routes_are_rejected(self):
        truncated = (
            "0 1 2 3 0.6 0.4 -1 -1\n"
            "1 1 3 2 0.7 0.3 -1 -1\n"
        )
        cpu = self.capture("cpu", self.frames, routes=truncated)
        vq = self.capture("vq", self.frames, routes=truncated)
        with self.assertRaisesRegex(ComparisonError, "incomplete route coverage"):
            compare(load_run(cpu), load_run(vq), "exact-vq", 1e-4, None, 10)

    def test_incomplete_vq_callback_coverage_is_rejected(self):
        vq = self.capture("vq", self.frames)
        manifest_path = vq / "run.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["backend_calls"]["vq_down_calls"] = 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ComparisonError, "coverage is incomplete"):
            compare(load_run(self.cpu), load_run(vq), "exact-vq",
                    1e-4, None, 10)

    def test_incomplete_matvec_callback_coverage_is_rejected(self):
        dense = self.capture("matvec", self.frames)
        manifest_path = dense / "run.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["backend_calls"]["matvec_calls"] = 0
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ComparisonError, "callback count"):
            compare(load_run(self.cpu), load_run(dense), "dense",
                    1e-4, None, 10)

    def test_capture_without_a_decode_step_is_rejected(self):
        manifest_path = self.cpu / "run.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["capture"]["generated_tokens"] = [0]
        manifest["capture"]["logit_steps"] = 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (self.cpu / "logits.f32").write_bytes(
            struct.pack("<4f", *self.frames[0])
        )
        with self.assertRaisesRegex(ComparisonError, "invalid capture dimensions"):
            load_run(self.cpu)

    def test_nonfinite_numeric_gate_is_rejected(self):
        dense = self.capture("matvec", self.frames)
        with self.assertRaisesRegex(ComparisonError, "finite and non-negative"):
            compare(load_run(self.cpu), load_run(dense), "dense",
                    math.nan, None, 10)

    def test_cli_returns_nonzero_on_failure(self):
        changed = ([5.001, 4.0, 3.0, 2.0], self.frames[1])
        dense = self.capture("matvec", changed)
        with contextlib.redirect_stderr(io.StringIO()):
            status = main([
                str(self.cpu), str(dense), "--contract", "dense",
                "--max-abs", "0.0001",
            ])
        self.assertEqual(status, 1)

    def test_public_report_contains_hashes_not_token_arrays(self):
        vq = self.capture("vq", self.frames)
        reference, candidate = load_run(self.cpu), load_run(vq)
        result = compare(reference, candidate, "exact-vq", 1e-4, None, 10)
        report_path = self.root / "public.json"
        write_public_report(str(report_path), "repo-public-v1",
                            reference, candidate, result)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["public_fixture"], "repo-public-v1")
        self.assertIn("prompt_tokens_sha256", report["reference"])
        self.assertNotIn("capture", report["reference"])


if __name__ == "__main__":
    unittest.main()
