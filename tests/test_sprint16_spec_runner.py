#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sprint16_runner", ROOT / "tools/run_sprint16_spec_probe.py"
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Sprint16RunnerTest(unittest.TestCase):
    def test_preinference_amendment_freezes_actual_state_probe(self) -> None:
        amendment = runner.read_json(
            ROOT / "docs/gn100/sprint16-preinference-amendment.json"
        )
        runner.check_amendment(amendment)
        amendment["state_costs"]["actual_state_probe"][
            "actual_mla_latent_rows"
        ] = False
        with self.assertRaisesRegex(runner.CampaignError, "state-cost contract drift"):
            runner.check_amendment(amendment)

    def test_joint_budget_is_exact_not_merely_positive(self) -> None:
        args = Namespace(
            target_cache_mb=40502, draft_cache_mb=16926,
            target_rollback_bytes=536870912, draft_rollback_bytes=134217728,
        )
        runner.check_registered_budget(args)
        args.target_cache_mb += 1
        with self.assertRaisesRegex(runner.CampaignError, "registered value 40502"):
            runner.check_registered_budget(args)

    def test_sealed_manifest_rejects_evidence_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            evidence = output / "load-only.json"
            evidence.write_text("{}\n", encoding="utf-8")
            runner.manifest(output)
            result = runner.verify_manifest(output)
            self.assertRegex(result["manifest_sha256"], r"^[0-9a-f]{64}$")
            evidence.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(runner.CampaignError, "evidence drift"):
                runner.verify_manifest(output)

    def test_capture_preflight_failure_never_reseals_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "sealed"
            output.mkdir()
            evidence = output / "load-only.json"
            evidence.write_text("{}\n", encoding="utf-8")
            runner.manifest(output)
            manifest_hash = runner.sha(output / "manifest.json")
            evidence.write_text("tampered\n", encoding="utf-8")
            args = Namespace(stage="capture", output=output,
                             _stage_output_owned=False,
                             _inference_started_this_invocation=False)
            failure = runner.record_failure(args, runner.CampaignError("drift"))
            self.assertNotEqual(failure.parent, output)
            self.assertTrue(failure.is_file())
            self.assertEqual(runner.sha(output / "manifest.json"), manifest_hash)
            with self.assertRaisesRegex(runner.CampaignError, "evidence drift"):
                runner.verify_manifest(output)

    def test_post_inference_failures_are_unique_and_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "inference-start.json").write_text("{}\n", encoding="utf-8")
            args = Namespace(stage="capture", output=output,
                             _stage_output_owned=False,
                             _inference_started_this_invocation=True)
            first = runner.record_failure(args, runner.CampaignError("first"))
            second = runner.record_failure(args, runner.CampaignError("second"))
            self.assertNotEqual(first, second)
            self.assertIn("first", first.read_text())
            self.assertIn("second", second.read_text())
            runner.verify_manifest(output)

    def test_owned_capture_start_failure_is_sealed_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "pre-inference-seal.json").write_text("{}\n", encoding="utf-8")
            args = Namespace(stage="capture", output=output,
                             _stage_output_owned=False,
                             _inference_started_this_invocation=True)
            failure = runner.record_failure(args, OSError("start marker fsync"))
            self.assertEqual(failure.parent, output)
            self.assertIn("start marker fsync", failure.read_text())
            runner.verify_manifest(output)

    def test_reruns_leave_existing_campaign_tree_byte_identical(self) -> None:
        for stage in ("seal", "capture"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "campaign"
                output.mkdir()
                (output / "evidence.json").write_text("{}\n", encoding="utf-8")
                if stage == "capture":
                    (output / "inference-start.json").write_text("{}\n", encoding="utf-8")
                runner.manifest(output)
                before = {
                    path.relative_to(output).as_posix(): (runner.sha(path), path.stat().st_size)
                    for path in output.rglob("*") if path.is_file()
                }
                args = Namespace(stage=stage, output=output,
                                 _stage_output_owned=False,
                                 _inference_started_this_invocation=False)
                failure = runner.record_failure(args, runner.CampaignError("rerun"))
                after = {
                    path.relative_to(output).as_posix(): (runner.sha(path), path.stat().st_size)
                    for path in output.rglob("*") if path.is_file()
                }
                self.assertEqual(after, before)
                self.assertNotEqual(failure.parent, output)

    def test_h2_is_rejected_before_case_parsing(self) -> None:
        path = ROOT / "docs/gn100/sprint15-h2-corpus.json"
        with self.assertRaisesRegex(runner.CampaignError, "H2 embargo"):
            runner.check_corpus(path)

    def test_full_usage_accepts_root_level_kimi_geometry(self) -> None:
        manifest = {
            "num_hidden_layers": 27,
            "first_k_dense_replace": 1,
            "num_experts": 256,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "usage.waste"
            count = runner.write_full_usage(path, manifest)
            self.assertEqual(count, 26 * 256)
            self.assertEqual(
                path.stat().st_size,
                runner.USAGE_HEADER.size + count * runner.USAGE_ENTRY.size,
            )

    def test_legacy_a_replay_uses_sixteen_token_prefixes(self) -> None:
        expected = runner.LEGACY_A_HASHES["composition_a"]
        value = {
            "token_prefix_hashes": ["unused"] * 15 + [expected["token"]],
            "logit_prefix_hashes": ["unused"] * 16 + [expected["logit"]],
            "route_prefix_hashes": ["unused"] * 15 + [expected["route"]],
        }
        result = runner.check_a("composition_a", value)
        self.assertTrue(result["pass"])
        value["route_prefix_hashes"][15] = "0x0"
        with self.assertRaisesRegex(runner.CampaignError, "legacy A replay failed"):
            runner.check_a("composition_a", value)

    def test_teacher_branch_dimensions_and_padding(self) -> None:
        targets = [10, 11, 12]
        rows = [
            [10, 99, 98, -1, -1, -1, -1, -1],
            [11, 12, -1, -1, -1, -1, -1, -1],
            [12, -1, -1, -1, -1, -1, -1, -1],
        ]
        value = {
            "schema": "waste.gn100.spec_teacher.v1",
            "target_tokens": 3,
            "targets": targets,
            "branch_width": 8,
            "branch_widths": [3, 2, 1],
            "branch_predictions": rows,
            "branch_logit_row_hashes": [
                ["0x0000000000000001"] * width
                + ["0x0000000000000000"] * (8 - width)
                for width in (3, 2, 1)
            ],
            "branch_route_row_hashes": [
                ["0x0000000000000002"] * max(0, width - 1)
                + ["0x0000000000000000"] * (8 - max(0, width - 1))
                for width in (3, 2, 1)
            ],
            "prefix_match_lengths": [1, 2, 1],
            "branch_step_seconds": [
                [0.1, 0.1, 0.0, 0, 0, 0, 0, 0],
                [0.1, 0.0, 0, 0, 0, 0, 0, 0],
                [0.0, 0, 0, 0, 0, 0, 0, 0],
            ],
            "predictions": [10, 11, 12],
            "matches": [1, 1, 1],
            "logit_row_hashes": ["a", "b", "c"],
            "step_seconds": [0.1, 0.1, 0.1],
            "state_bytes": 1,
            "state_hash": "0x1",
            "snapshot_count": 3,
            "restore_count": 3,
            "process_safety": {"vmswap_kib": 0, "timed_major_faults_delta": 0,
                               "timed_scope": "prefill_plus_teacher_branches_snapshots_restores"},
            "io": {"direct": 1, "readers": 2, "depth": 2},
            "cache": {
                "fully_warm_at_start": True,
                "warm_ready": 4,
                "routed_records": 4,
            },
            "cuda": {"kda": 1, "dense": 2, "vq": 0, "fallbacks": 0},
        }
        runner.check_teacher(value, targets)
        value["branch_predictions"][1][2] = 42
        with self.assertRaisesRegex(runner.CampaignError, "padding drift"):
            runner.check_teacher(value, targets)

    def test_post_prompt_state_cost_requires_actual_mla_state(self) -> None:
        value = {
            "schema": "waste.gn100.spec_state.v1",
            "actual_model_state": True,
            "synthetic_mla_rows": False,
            "canonical_continuation_tokens": 32,
            "continuation_tokens_applied": 31,
            "state_position": 123,
            "state_bytes": 1024,
            "state_hash": "0x0000000000000001",
            "prompt_tokens": 92,
            "mla_latent_bytes": 256,
            "mla_layers": 1,
            "roundtrip_replay": {
                "logits_byte_identical": True,
                "ordered_routes_byte_identical": True,
                "post_state_byte_identical": True,
                "root_restored_after_check": True,
                "post_state_bytes": 1032,
            },
            "in_memory_state_serialization": {
                "bytes": 1024, "warmup_roundtrips": 1,
                "export_seconds": 0.01, "import_seconds": 0.02,
            },
            "shadow_copy_floor": {
                "bytes": 1024, "optimistic": True,
                "is_pointer_swap": False, "is_durable_file_io": False,
                "pages_pretouched": True,
                "thread_creation_timed": False,
                "worker_dispatch_timed": True,
                "repeats": 7,
                "threads_1": {"threads": 1, "best_seconds": 0.01,
                              "median_seconds": 0.02,
                              "best_gib_s": 1.0, "median_gib_s": 0.5},
                "threads_10": {"threads": 10, "best_seconds": 0.005,
                               "median_seconds": 0.006,
                               "best_gib_s": 2.0, "median_gib_s": 1.5},
            },
            "process_safety": {"vmswap_kib": 0, "timed_major_faults_delta": 0,
                               "timed_scope": "prefill_continuation_export_import_shadow_copy"},
            "io": {"direct": 1, "readers": 2, "depth": 2},
            "cache": {"slots": 4, "warm_ready": 4, "routed_records": 8},
            "cuda": {"kda": 1, "dense": 2, "vq": 2, "fallbacks": 0},
        }
        runner.check_state(value, 123)
        value["synthetic_mla_rows"] = True
        with self.assertRaisesRegex(runner.CampaignError, "post-prompt.*state"):
            runner.check_state(value, 123)

    def test_load_state_costs_keep_root_floor_separate(self) -> None:
        value = {
            "target": {"state0_bytes": 1024},
            "draft": {"state0_bytes": 256},
            "in_memory_state_serialization": {
                "warmup_roundtrips": 1, "rollback_pages_pretouched": True,
                "target": {"bytes": 1024, "export_seconds": 0.1,
                           "import_seconds": 0.2},
                "draft": {"bytes": 256, "export_seconds": 0.01,
                          "import_seconds": 0.02},
            },
            "target_shadow_copy_floor": {
                "bytes": 1024, "optimistic": True,
                "is_pointer_swap": False, "is_durable_file_io": False,
                "temporary_mapping_released_before_memory_snapshot": True,
                "repeats": 7,
                "threads_1": {"threads": 1, "best_seconds": 0.1,
                              "median_seconds": 0.2},
                "threads_10": {"threads": 10, "best_seconds": 0.03,
                               "median_seconds": 0.04},
            },
        }
        runner.check_load_state_costs(value)
        value["target_shadow_copy_floor"]["is_pointer_swap"] = True
        with self.assertRaisesRegex(runner.CampaignError, "shadow-copy floor"):
            runner.check_load_state_costs(value)


if __name__ == "__main__":
    unittest.main()
