# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Command-line wiring that must happen before the engine opens."""

import importlib
import io
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.serve.fake_engine import FakeEngine             # noqa: E402

MAIN = importlib.import_module("serve.__main__")


class _StoppedServer:
    def __init__(self, tmpdir: str):
        self.tmpdir = tmpdir
        self.shutdown_called = False
        self.close_called = False

    def serve_forever(self):
        raise KeyboardInterrupt()

    def shutdown(self):
        self.shutdown_called = True

    def server_close(self):
        self.close_called = True


class TestMain(unittest.TestCase):
    def test_plan_labels_physical_capacity_and_current_ceiling(self):
        tmp = tempfile.mkdtemp(prefix="waste-main-test-")
        model = Path(tmp) / "model.waste"
        model.mkdir()
        output = io.StringIO()
        plan = SimpleNamespace(
            trunk_bytes=1 << 20, state_bytes=2 << 20,
            scratch_bytes=3 << 20, min_expert_cache=4 << 20,
            floor_bytes=10 << 20, recommended_bytes=22 << 20,
            vision_bytes=0)
        try:
            with (patch.object(MAIN, "Engine") as open_,
                  patch.object(MAIN, "plan_memory", return_value=plan),
                  patch.object(MAIN, "build_info", return_value="WASTE test"),
                  patch.object(MAIN, "physical_ram", return_value=128 << 30),
                  patch.object(MAIN, "usable_ram", return_value=6 << 30),
                  patch.object(MAIN, "memory_ceiling", return_value=5 << 30),
                  redirect_stdout(output), redirect_stderr(output)):
                status = MAIN.main([str(model), "--plan"])
            text = output.getvalue()
            self.assertEqual(status, 0, text)
            self.assertIn("host physical RAM           128.0 GB", text)
            self.assertIn("stable process capacity     6.0 GB", text)
            self.assertIn("automatic-open ceiling now  5.0 GB (snapshot)", text)
            open_.assert_not_called()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_conversation_head_is_rejected_before_engine_open(self):
        tmp = tempfile.mkdtemp(prefix="waste-main-test-")
        model = Path(tmp) / "model.waste"
        model.mkdir()
        output = io.StringIO()
        try:
            with (patch.object(MAIN, "Engine") as open_,
                  redirect_stdout(output), redirect_stderr(output)):
                status = MAIN.main([str(model), "--conversation-head"])
            self.assertEqual(status, 2, output.getvalue())
            self.assertIn("requires --prefix-cache", output.getvalue())
            open_.assert_not_called()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_too_small_prefix_cache_is_rejected_before_engine_open(self):
        tmp = tempfile.mkdtemp(prefix="waste-main-test-")
        model = Path(tmp) / "model.waste"
        model.mkdir()
        output = io.StringIO()
        try:
            with (patch.object(MAIN, "Engine") as open_,
                  redirect_stdout(output), redirect_stderr(output)):
                status = MAIN.main([
                    str(model), "--prefix-cache", "4095",
                ])
            self.assertEqual(status, 2, output.getvalue())
            self.assertIn("at least 4096 bytes", output.getvalue())
            open_.assert_not_called()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_prefix_bytes_are_reserved_before_open_and_reused_as_limit(self):
        tmp = tempfile.mkdtemp(prefix="waste-main-test-")
        model = Path(tmp) / "model.waste"
        model.mkdir()
        server_tmp = tempfile.mkdtemp(prefix="waste-main-server-test-")
        engine = FakeEngine(host_reserved_bytes=4096)
        stopped = _StoppedServer(server_tmp)
        output = io.StringIO()
        try:
            with (patch.object(MAIN, "Engine", return_value=engine) as open_,
                  patch.object(MAIN, "serve", return_value=stopped) as serve_,
                  patch.object(MAIN, "build_info", return_value="WASTE test"),
                  redirect_stdout(output), redirect_stderr(output)):
                status = MAIN.main([
                    str(model), "--port", "0",
                    "--prefix-cache", "4K",
                    "--prefix-cache-entries", "3",
                ])
            self.assertEqual(status, 0, output.getvalue())
            self.assertEqual(open_.call_args.kwargs["host_reserved_bytes"],
                             4096)
            self.assertEqual(serve_.call_args.kwargs["prefix_cache_bytes"],
                             4096)
            self.assertEqual(serve_.call_args.kwargs["prefix_cache_entries"],
                             3)
            self.assertFalse(serve_.call_args.kwargs["conversation_head"])
            self.assertFalse(serve_.call_args.kwargs["semantic_anchors"])
            self.assertTrue(engine.closed)
            self.assertTrue(stopped.shutdown_called)
            self.assertTrue(stopped.close_called)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(server_tmp, ignore_errors=True)

    def test_semantic_anchors_are_validated_and_wired_before_open(self):
        tmp = tempfile.mkdtemp(prefix="waste-main-test-")
        model = Path(tmp) / "model.waste"
        model.mkdir()
        server_tmp = tempfile.mkdtemp(prefix="waste-main-server-test-")
        engine = FakeEngine(host_reserved_bytes=1 << 20)
        stopped = _StoppedServer(server_tmp)
        output = io.StringIO()
        try:
            with (patch.object(MAIN, "Engine", return_value=engine),
                  patch.object(MAIN, "serve", return_value=stopped) as serve_,
                  patch.object(MAIN, "build_info", return_value="WASTE test"),
                  redirect_stdout(output), redirect_stderr(output)):
                status = MAIN.main([
                    str(model), "--port", "0", "--prefix-cache", "1M",
                    "--prefix-cache-entries", "4", "--semantic-anchors",
                ])
            self.assertEqual(status, 0, output.getvalue())
            self.assertTrue(serve_.call_args.kwargs["semantic_anchors"])
            self.assertFalse(serve_.call_args.kwargs["conversation_head"])
            self.assertIn("exact completed-message checkpoints", output.getvalue())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.rmtree(server_tmp, ignore_errors=True)

    def test_semantic_anchors_reject_missing_cache_and_head_combination(self):
        tmp = tempfile.mkdtemp(prefix="waste-main-test-")
        model = Path(tmp) / "model.waste"
        model.mkdir()
        output = io.StringIO()
        try:
            with (patch.object(MAIN, "Engine") as open_,
                  redirect_stdout(output), redirect_stderr(output)):
                missing = MAIN.main([str(model), "--semantic-anchors"])
                combined = MAIN.main([
                    str(model), "--prefix-cache", "1M",
                    "--prefix-cache-entries", "4",
                    "--conversation-head", "--semantic-anchors",
                ])
            self.assertEqual(missing, 2)
            self.assertEqual(combined, 2)
            open_.assert_not_called()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_strict_profile_rejects_effective_reader_mismatch(self):
        tmp = tempfile.mkdtemp(prefix="waste-main-test-")
        model = Path(tmp) / "model.waste"
        model.mkdir()
        engine = FakeEngine()
        engine.stats = lambda: {
            "tokens_generated": 0, "experts_hit": 0, "experts_missed": 0,
            "bytes_read": 0, "sec_total": 0.0, "sec_io": 0.0,
            "direct_io": 1, "read_ahead_threads": 1,
            "read_ahead_depth": 2,
        }
        output = io.StringIO()
        try:
            with (patch.dict(MAIN.os.environ, {}, clear=True),
                  patch.object(MAIN, "Engine", return_value=engine),
                  patch.object(MAIN.RequestQos, "from_env",
                               return_value=MAIN.DisabledRequestQos()),
                  patch.object(MAIN, "serve") as serve_,
                  redirect_stdout(output), redirect_stderr(output)):
                status = MAIN.main([
                    str(model), "--performance-profile", "spark-q0",
                ])
            self.assertEqual(status, 1, output.getvalue())
            self.assertIn("engine reported 1/2", output.getvalue())
            serve_.assert_not_called()
            self.assertTrue(engine.closed)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
