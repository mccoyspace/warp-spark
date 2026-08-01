# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the balanced Spark whole-expert acceptance campaign."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools import analyze_spark_whole_expert as analyzer
from tools import pm_qos_exec
from tools import spark_whole_expert_campaign as campaign


FIXTURE = (Path(__file__).resolve().parents[1] / "fixtures" /
           "spark_whole_expert_factorial.json")


class SparkCampaignFixture:
    def __init__(self, root: Path):
        self.root = root
        self.spec = json.loads(FIXTURE.read_text())
        self.identity = {"binary_sha256": "binary",
                         "manifest_sha256": "manifest",
                         "model_geometry": {
                             "moe_layers": self.spec["moe_layers"]}}
        self.campaign = {
            "schema": "waste.spark_whole_expert_campaign.v1",
            "blocks": 4,
            "orders": [list(row) for row in campaign.BASE_ORDERS],
            "decoded_tokens": 1,
            "io_backend": "io_uring", "io_queue_depth": 4,
            "direct_io": True, "identity": self.identity,
            "limits": {"cpu_temp_millic": 52000,
                       "nvme_temp_millic": 55000,
                       "gpu_util_percent": 5.0},
        }
        self.records = self._records()

    def _records(self) -> list[dict]:
        records = []
        for block, order in enumerate(campaign.BASE_ORDERS, 1):
            base = self.spec["block_baselines_ms"][block - 1]
            for position, treatment in enumerate(order, 1):
                treatment_spec = campaign.TREATMENTS[treatment]
                ratio = self.spec["treatment_ratios"][treatment]
                value = base * ratio
                decode = {}
                for metric in campaign.TRACE_METRICS:
                    decode[metric] = (value if metric.endswith("_ms") else
                                      {"cache_hits": 10, "cache_misses": 20,
                                       "bytes_read": 30_000}[metric])
                decode["expert_compute_work_ms"] = value * 1.4
                if treatment_spec["qos_us"] is None:
                    qos = {"enabled": False}
                else:
                    qos = {
                        "holder_euid": 0, "latency_us": 0,
                        "fd_opened": True, "fd_closed": True,
                        "fd_scoped": True, "self_cleaning": True,
                        "sysfs_modified": False, "child_returncode": 0,
                    }
                run_id = (f"b{block:02d}-p{position}-{treatment}-"
                          f"{treatment_spec['name']}")
                records.append({
                    "schema": "waste.spark_whole_expert_run.v1",
                    "run_id": run_id, "block": block, "position": position,
                    "treatment": treatment,
                    "treatment_name": treatment_spec["name"],
                    "schedule": treatment_spec["schedule"],
                    "qos_us": treatment_spec["qos_us"],
                    "returncode": 0, "wall_seconds": value / 10.0,
                    "stdout_sha256": "same-answer", "identity_after": self.identity,
                    "cooldown_reasons": [], "pm_qos": qos,
                    "vm_delta": {"pswpin": 0, "pswpout": 0},
                    "telemetry": {
                        "peak_swap_bytes": 0,
                        "memory_psi_full_total_delta_us": 0,
                        "max_processor_cooling_state": 0,
                        "any_cppc_perf_limited": False,
                        "cpu_peak_millic": 50000,
                        "nvme_peak_millic": 45000,
                        "gpu": {"available_all_samples": True,
                                "compute_apps": [],
                                "max_utilization_percent": 0.0},
                    },
                    "trace": {
                        "backend": "io_uring", "queue_depth": 4,
                        "io_fallback": False,
                        "expert_schedule_requested": treatment_spec["schedule"],
                        "expert_schedule_selected": treatment_spec["schedule"],
                        "decode_layer_schedules": [treatment_spec["schedule"]],
                        "event_counts": {"prefill_layer": self.spec["moe_layers"],
                                         "decode_layer": self.spec["moe_layers"],
                                         "decode_token": 1},
                        "all_direct_io": True, "any_read_error": False,
                        "route_sha256": "same-routes",
                        "traffic_sha256": "same-traffic",
                        "layer_signature_sha256": "same-layer-signature",
                        "decode": decode,
                    },
                    "valid": True, "invalid_reasons": [],
                })
        return records

    def write(self, mutate=None) -> None:
        records = copy.deepcopy(self.records)
        if mutate:
            mutate(records)
        (self.root / "campaign.json").write_text(
            json.dumps(self.campaign) + "\n")
        (self.root / "runs.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records))


class TestSparkWholeExpertCampaign(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_williams_square_is_position_balanced(self):
        generated = campaign.orders(4)
        self.assertEqual(generated, list(campaign.BASE_ORDERS))
        for position in range(4):
            self.assertEqual({row[position] for row in generated},
                             set(campaign.TREATMENTS))
        with self.assertRaises(ValueError):
            campaign.orders(3)

    def test_commands_change_only_the_selected_factors(self):
        target = campaign.target_command(
            binary=Path("/repo/waste"), model=Path("/model/k3"),
            trace=Path("/out/trace.jsonl"), budget=80_000_000_000,
            threads=8, cpus="5-9,15-19", schedule="whole")
        self.assertIn("WASTE_EXPERT_SCHED=whole", target)
        self.assertIn("WASTE_DIRECT=1", target)
        self.assertEqual(target[target.index("--io-backend") + 1], "io_uring")
        self.assertEqual(target[target.index("--io-queue-depth") + 1], "4")
        self.assertEqual(target[target.index("-n") + 1], "1")
        ordinary = campaign.wrapped_command(
            target, treatment=campaign.TREATMENTS["WN"], user="nvidia",
            qos_helper=Path("/repo/tools/pm_qos_exec.py"),
            qos_status=Path("/out/qos.json"), python=Path("/usr/bin/python3"))
        qos = campaign.wrapped_command(
            target, treatment=campaign.TREATMENTS["WQ"], user="nvidia",
            qos_helper=Path("/repo/tools/pm_qos_exec.py"),
            qos_status=Path("/out/qos.json"), python=Path("/usr/bin/python3"))
        self.assertNotIn("pm_qos_exec.py", " ".join(ordinary))
        self.assertIn("pm_qos_exec.py", " ".join(qos))
        self.assertIn("--latency-us", qos)
        self.assertEqual(qos[qos.index("--latency-us") + 1], "0")
        self.assertNotIn("/sys", " ".join(target + ordinary + qos))

    def test_trace_parser_preserves_schedule_routes_and_traffic(self):
        trace = self.root / "trace.jsonl"
        rows = [
            {"event": "meta", "io_backend": "io_uring", "queue_depth": 4,
             "io_fallback": False, "expert_schedule_requested": "whole",
             "expert_schedule": "whole", "start_monotonic_ns": 1},
            {"event": "prefill_layer", "layer": 1, "direct_io": True,
             "read_error": False},
            {"event": "prefill_chunk", "direct_io": True, "read_error": False,
             "end_monotonic_ns": 2},
            {"event": "decode_layer", "position": 64, "layer": 1,
             "experts": [2, 7], "cache_hits": 1, "cache_misses": 1,
             "bytes_read": 4096, "expert_schedule": "whole",
             "expert_compute_ms": 10, "expert_compute_work_ms": 18,
             "direct_io": True, "read_error": False},
            {"event": "decode_token", "total_ms": 20,
             "expert_compute_ms": 10, "direct_io": True, "read_error": False,
             "end_monotonic_ns": 3},
        ]
        trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary = campaign.trace_summary(trace)
        self.assertEqual(summary["backend"], "io_uring")
        self.assertEqual(summary["expert_schedule_selected"], "whole")
        self.assertEqual(summary["decode_layer_schedules"], ["whole"])
        self.assertEqual(summary["decode"]["expert_compute_work_ms"], 18)
        self.assertTrue(summary["all_direct_io"])

    def test_pm_qos_cli_is_zero_only_and_fd_scoped(self):
        parsed = pm_qos_exec.parse_args([
            "--latency-us", "0", "--status", "/tmp/qos.json",
            "--user", "nvidia", "--", "/bin/true"])
        self.assertEqual(parsed.latency_us, 0)
        self.assertEqual(parsed.command, ["/bin/true"])
        with self.assertRaises(SystemExit):
            pm_qos_exec.parse_args([
                "--latency-us", "1", "--status", "/tmp/qos.json",
                "--user", "nvidia", "--", "/bin/true"])

    def test_fixture_factorial_effects_and_exact_gates(self):
        fixture = SparkCampaignFixture(self.root)
        fixture.write()
        result = analyzer.analyze(self.root)
        self.assertTrue(result["exact_gates_pass"])
        self.assertTrue(result["exact_gates"]["identical_routes"]["pass"])
        self.assertTrue(result["traffic_comparability"]["global_match"])
        self.assertEqual(result["eligible_blocks"], [1, 2, 3, 4])
        expected = fixture.spec["expected_effect_percent"]
        effects = result["factorial_effects"]
        for name, value in expected.items():
            self.assertAlmostEqual(
                effects[name]["total_ms"]["ratio"]["percent_change"],
                value, places=9)
        self.assertLess(result["performance"]["whole_schedule_total_percent"], 0)

    def test_traffic_drift_is_reported_without_invalidating_campaign(self):
        fixture = SparkCampaignFixture(self.root)

        def drift(records):
            records[0]["trace"]["traffic_sha256"] = "different"

        fixture.write(drift)
        result = analyzer.analyze(self.root)
        self.assertTrue(result["exact_gates_pass"])
        self.assertTrue(result["exact_gates"]["identical_routes"]["pass"])
        self.assertFalse(result["traffic_comparability"]["global_match"])
        self.assertEqual(
            result["traffic_comparability"]["by_treatment"]["RN"]
            ["decode_traffic"]["bytes_read"]["mean"], 30_000)

    def test_route_drift_is_a_hard_gate(self):
        fixture = SparkCampaignFixture(self.root)

        def drift(records):
            records[0]["trace"]["route_sha256"] = "different"

        fixture.write(drift)
        result = analyzer.analyze(self.root)
        self.assertFalse(result["exact_gates_pass"])
        self.assertFalse(result["exact_gates"]["identical_routes"]["pass"])

    def test_schedule_mismatch_fails_run_and_schedule_gates(self):
        fixture = SparkCampaignFixture(self.root)

        def mismatch(records):
            records[1]["trace"]["expert_schedule_selected"] = "row"

        fixture.write(mismatch)
        result = analyzer.analyze(self.root)
        self.assertFalse(result["exact_gates_pass"])
        self.assertFalse(result["exact_gates"]["run_records_valid"]["pass"])
        self.assertFalse(result["exact_gates"][
            "schedule_selected_by_environment"]["pass"])

    def test_incomplete_trace_reports_failed_gates_instead_of_crashing(self):
        fixture = SparkCampaignFixture(self.root)

        def truncate(records):
            records[0]["trace"] = {"parse_error": "truncated JSONL"}
            records[0]["valid"] = False
            records[0]["invalid_reasons"] = ["trace_parse_error"]

        fixture.write(truncate)
        result = analyzer.analyze(self.root)
        self.assertFalse(result["exact_gates_pass"])
        self.assertFalse(result["exact_gates"]["identical_routes"]["pass"])
        self.assertFalse(result["traffic_comparability"]["global_match"])
        self.assertEqual(result["eligible_blocks"], [2, 3, 4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
