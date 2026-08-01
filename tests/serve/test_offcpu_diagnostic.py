# SPDX-License-Identifier: Apache-2.0
"""Deterministic parser/classifier checks for the Linux sidecar."""

import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.offcpu_diagnostic import analyze, load_windows, parse_cpu_set  # noqa: E402
from tools.analyze_offcpu_pair import compare as compare_pair    # noqa: E402


class TestOffcpuDiagnostic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.trace = self.root / "layers.jsonl"
        self.events = self.root / "events.tsv"
        self.trace.write_text(json.dumps({
            "schema": "waste.layer_trace.v2",
            "event": "decode_layer", "position": 7, "layer": 3,
            "expert_compute_intervals_monotonic_ns": [
                [1_000_000, 1_400_000], [1_450_000, 2_000_000]],
        }) + "\n")
        rows = [
            "WASTE_OFFCPU\tMETA\t900000\t100",
            "WASTE_OFFCPU\tTHREAD\t910000\t100\t101",
            "WASTE_OFFCPU\tFUTEX_ENTER\t950000\t5\t101\t0\t1234",
            "WASTE_OFFCPU\tCPU_IDLE\t1000000\t5\t3",
            "WASTE_OFFCPU\tWAKEUP\t1100000\t0\t101\t100\t5",
            "WASTE_OFFCPU\tCPU_IDLE\t1110000\t5\t4294967295",
            "WASTE_OFFCPU\tSWITCH_IN\t1120000\t5\t101\t44",
            "WASTE_OFFCPU\tFUTEX_EXIT\t1130000\t5\t101\t0\t1234\t0\t180000",
            "WASTE_OFFCPU\tFUTEX_ENTER\t1450000\t5\t101\t0\t1234",
            "WASTE_OFFCPU\tSWITCH_OUT\t1500000\t5\t101\t1\t44",
            "WASTE_OFFCPU\tWAKEUP\t1600000\t0\t101\t100\t5",
            "WASTE_OFFCPU\tSWITCH_IN\t1620000\t5\t101\t44",
            "WASTE_OFFCPU\tFUTEX_EXIT\t1630000\t5\t101\t0\t1234\t0\t180000",
            "WASTE_OFFCPU\tTHREAD_EXIT\t2100000\t101",
        ]
        self.events.write_text("\n".join(rows) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_barrier_idle_and_active_wakes_are_separated(self):
        result = analyze(self.trace, self.events, idle_lookback_us=50)
        cats = result["wake_to_run"]["categories"]
        self.assertEqual(cats["barrier_idle_exit"]["count"], 1)
        self.assertEqual(cats["barrier_no_idle_exit"]["count"], 1)
        self.assertEqual(result["wake_to_run"]["all_workers"]["p50_us"], 20.0)
        self.assertEqual(result["idle_exit"]["states"]["3"]["count"], 1)
        self.assertEqual(
            result["idle_exit"]["states"]["3"]["residency"]["p50_us"],
            110.0)
        self.assertEqual(result["offcpu"]["blocked_worker_ms"], 0.12)
        self.assertEqual(result["coverage"]["routed_intervals"], 2)
        self.assertTrue(result["coverage"]["exact_intervals"])

    def test_capture_must_cover_trace_windows(self):
        rows = self.events.read_text().splitlines()
        self.events.write_text("\n".join(rows[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "does not cover"):
            analyze(self.trace, self.events)

    def test_parallel_expert_windows_are_coalesced_without_double_counting(self):
        self.trace.write_text(json.dumps({
            "schema": "waste.layer_trace.v2",
            "event": "decode_layer", "position": 7, "layer": 3,
            "expert_schedule": "whole",
            "routed_loop_start_monotonic_ns": 900_000,
            "routed_loop_end_monotonic_ns": 2_100_000,
            "expert_compute_intervals_monotonic_ns": [
                [1_000_000, 1_600_000], [1_200_000, 2_000_000]],
        }) + "\n")
        windows, meta = load_windows(self.trace)
        self.assertEqual([(w.start, w.end) for w in windows],
                         [(900_000, 2_100_000)])
        self.assertEqual(meta["raw_expert_intervals"], 2)
        self.assertEqual(meta["overlapping_interval_rows"], 1)
        self.assertEqual(meta["whole_schedule_rows"], 1)
        self.assertEqual(meta["raw_expert_work_ms"], 1.4)
        self.assertEqual(meta["raw_expert_union_ms"], 1.0)
        self.assertTrue(meta["exact_intervals"])

    def test_caller_futex_wait_is_not_attributed_to_workers(self):
        with self.events.open("a") as f:
            f.write("WASTE_OFFCPU\tFUTEX_EXIT\t1300000\t5\t100\t0\t9999\t0\t200000\n")
        result = analyze(self.trace, self.events, idle_lookback_us=50)
        self.assertEqual(result["futex"]["wait_calls_overlapping_compute"], 2)
        self.assertEqual(result["futex"]["wait_worker_ms"], 0.31)
        self.assertEqual(
            result["futex"]["all_thread_wait_calls_overlapping_compute"], 3)
        self.assertEqual(result["futex"]["all_thread_wait_ms"], 0.51)

    def test_cpu_set_parser(self):
        self.assertEqual(parse_cpu_set("5-7,10,12-13"), [5, 6, 7, 10, 12, 13])
        with self.assertRaises(ValueError):
            parse_cpu_set("7-5")

    def test_matched_pair_reports_mechanism_deltas(self):
        def summary(blocked, wait, p50, p95, nonzero, state0, state3):
            return {
                "coverage": {"exact_intervals": True}, "worker_threads": 7,
                "offcpu": {"blocked_worker_ms": blocked},
                "futex": {"wait_worker_ms": wait,
                           "wait_calls_overlapping_compute": 100},
                "wake_to_run": {"all_workers": {"p50_us": p50,
                                                  "p95_us": p95}},
                "idle_exit": {
                    "nonzero_state_fraction_of_barrier_wakes": nonzero,
                    "states": {
                        "0": {"fraction_of_barrier_wakes": state0,
                              "residency": {"p95_us": 100}},
                        "3": {"fraction_of_barrier_wakes": state3,
                              "residency": {"p95_us": 1000}},
                    },
                },
            }

        p = summary(1000, 900, 4, 12, .8, .2, .6)
        q = summary(500, 450, 2, 6, .4, .6, .2)
        trace = {
            "layer_rows": 92, "route_traffic_sha256": "same",
            "direct_io": True, "read_error": False,
            "backends": ["pread_sync"], "queue_depths": [1],
            "meta": {"io_fallback": False},
            "token": {"total_ms": 6, "expert_io_ms": 2,
                      "expert_compute_ms": 3},
        }
        qtrace = copy.deepcopy(trace)
        qtrace.update({"backends": ["io_uring"], "queue_depths": [4],
                       "token": {"total_ms": 4, "expert_io_ms": 1,
                                 "expert_compute_ms": 2}})
        result = compare_pair(p, q, trace, qtrace)
        self.assertTrue(result["valid"])
        self.assertEqual(
            result["metrics"]["blocked_worker_ms"]["qd4_reduction_fraction"],
            .5)
        self.assertTrue(result["gates"]["same_routes_and_traffic"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
