# SPDX-License-Identifier: Apache-2.0
"""Unit checks for the real-K3 family-root acceptance analyzer."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.analyze_family_prefix import (                 # noqa: E402
    canonical_sha256, compare, markdown,
)
from tools.family_prefix_acceptance import trace_summary  # noqa: E402


def output(text="same output"):
    value = {
        "message": {"role": "assistant", "content": text},
        "finish_reason": "length",
        "completion_tokens": 2,
    }
    return value, canonical_sha256(value)


def trace(*, route="route", rows=184, bytes_read=1000,
          prefill_tokens=150, position=0, replay=False):
    positions = {
        "150": {"count": 92, "sha256": route + "-150"},
        "151": {"count": 92, "sha256": route + "-151"},
    }
    if replay:
        positions["149"] = {"count": 92, "sha256": route + "-149"}
        rows += 92
    return {
        "decode_route_sha256": route,
        "decode_route_count": rows,
        "decode_route_positions": positions,
        "prefill_tokens": prefill_tokens,
        "prefill_position_start": position,
        "expert_bytes_read": bytes_read,
        "expert_cache_hits": 200,
        "expert_cache_misses": 50,
        "transport_ok": True,
        "boundary_complete": True,
        "read_error": False,
    }


def req(name, prefix, *, tool="stable", trace_row=None, text="same output",
        wall_seconds=100.0):
    stable, digest = output(text)
    return {
        "name": name,
        "request_sha256": ("b-request" if name in ("cold_b", "family_b")
                           else name + "-request"),
        "tool_sha256": tool,
        "wall_seconds": wall_seconds,
        "stable_output": stable,
        "stable_output_sha256": digest,
        "response_waste": {
            "direct_io": True,
            "io_backend": "io_uring",
            "io_queue_depth": 4,
            "io_fallback": False,
            "prefix_cache": prefix,
        },
        "usage": {"prompt_tokens": 150, "completion_tokens": 2},
        "trace": trace_row or trace(),
    }


def records():
    common = {
        "campaign_id": "campaign",
        "server_returncode": 0,
        "error": None,
        "manifest_sha256": "manifest",
        "usage_sha256": "usage",
        "library_sha256": "library",
        "budget_bytes": 86_583_021_568,
        "rss_limit_bytes": 86_583_021_568,
        "max_memory_psi_us": 0,
        "threads": 8,
        "cpu_set": "performance",
        "resolved_cpu_set": "5-9,15-19",
        "io_backend_requested": "io_uring",
        "io_queue_depth_requested": 4,
        "direct_io_requested": True,
        "engine_environment": {
            "WASTE_EXPERT_SCHED": "row", "WASTE_PROFILE": "0",
            "WASTE_Q8": "1", "WASTE_SDOT": "0", "WASTE_I8MM": "0",
            "WASTE_VERIFY": "0",
        },
        "tokens_requested": 2,
        "system_sha256": "system",
        "stable_tool_sha256": "stable",
        "changed_tool_sha256": "changed",
        "prompt_b_sha256": "prompt-b",
        "peak_rss_bytes": 85_700_000_000,
        "peak_process_swap_bytes": 0,
        "min_mem_available_bytes": 39_000_000_000,
        "vm_delta": {
            "pswpin": 0, "pswpout": 0,
            "pgscan_direct": 0, "pgscan_kswapd": 0,
            "pgsteal_direct": 0, "pgsteal_kswapd": 0,
            "pgmajfault": 4,
        },
        "memory_psi_total_us_delta": {"some": 0, "full": 0},
        "trace_meta": {"io_backend": "io_uring", "queue_depth": 4,
                       "io_fallback": False,
                       "expert_schedule_requested": "row",
                       "expert_schedule": "row"},
    }
    control = copy.deepcopy(common)
    control.update({"role": "control", "run_id": "control",
                    "prefix_cache_bytes": 0})
    control["requests"] = [
        req("seed_a", {"status": "bypass_disabled", "key_tokens": 150},
            trace_row=trace(route="seed", bytes_read=900,
                            prefill_tokens=150, position=0)),
        req("cold_b", {"status": "bypass_disabled", "key_tokens": 150},
            trace_row=trace(bytes_read=1000, prefill_tokens=150, position=0)),
    ]

    treatment = copy.deepcopy(common)
    treatment.update({"role": "prefix", "run_id": "prefix",
                      "prefix_cache_bytes": 1 << 30})
    treatment["requests"] = [
        req("seed_a", {
            "status": "miss_stored_family", "hit": False,
            "hit_kind": "none", "family_root_tokens": 120,
        }, trace_row=trace(route="seed", bytes_read=900, replay=True)),
        req("family_b", {
            "status": "hit", "hit": True, "hit_kind": "family",
            "family_root_tokens": 120, "reused_tokens": 120,
            "checkpoint_tokens": 120, "key_tokens": 150,
            "prompt_tokens_evaluated": 30, "replayed_tokens": 30,
            "snapshot_bytes": 466_000_000,
            "resident_bytes": 466_000_856,
            "capacity_bytes": 1 << 30,
            "family_entries": 1, "exact_entries": 0,
        }, trace_row=trace(bytes_read=400, prefill_tokens=29, position=120,
                           replay=True), wall_seconds=40.0),
        req("changed_tool_b", {
            "status": "miss_stored_family", "hit": False,
            "hit_kind": "none", "family_root_tokens": 121,
            "family_snapshot_bytes": 466_100_000,
            "resident_bytes": 932_100_856,
            "capacity_bytes": 1 << 30,
            "family_entries": 2, "exact_entries": 0,
        }, tool="changed", trace_row=trace(bytes_read=950)),
    ]
    return control, treatment


class TestAnalyzeFamilyPrefix(unittest.TestCase):
    def test_accepts_complete_matched_evidence(self):
        control, treatment = records()
        result = compare(control, treatment)
        self.assertTrue(result["valid"])
        self.assertTrue(result["accepted"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["family_prefix"]["root_tokens"], 120)
        self.assertEqual(result["expert_io"]["matched_generated_positions"], 2)
        self.assertAlmostEqual(
            result["expert_io"]["bytes_reduction_fraction"], 0.6)
        self.assertEqual(result["latency"]["speedup"], 2.5)
        report = markdown(result)
        self.assertIn("Accepted: **yes**", report)
        self.assertIn("60.00%", report)

    def test_rejects_changed_output_or_corrupt_output_digest(self):
        control, treatment = records()
        treatment["requests"][1]["stable_output"]["message"]["content"] = "other"
        result = compare(control, treatment)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["checks"]["output_artifacts_valid"])
        self.assertFalse(result["checks"]["output_canonical_identical"])

    def test_rejects_wrong_root_depth_or_non_replayed_suffix(self):
        control, treatment = records()
        prefix = treatment["requests"][1]["response_waste"]["prefix_cache"]
        prefix["checkpoint_tokens"] = 119
        result = compare(control, treatment)
        self.assertFalse(result["checks"]["family_root_replay"])
        self.assertFalse(result["accepted"])

    def test_rejects_nonmatching_seed_workload(self):
        control, treatment = records()
        treatment["requests"][0]["request_sha256"] = "different-seed"
        result = compare(control, treatment)
        self.assertFalse(result["checks"]["matched_seed_a"])
        self.assertFalse(result["accepted"])

    def test_rejects_nonidentical_b_request(self):
        control, treatment = records()
        treatment["requests"][1]["request_sha256"] = "different-b"
        result = compare(control, treatment)
        self.assertFalse(result["checks"]["matched_prompt_b"])
        self.assertFalse(result["accepted"])

    def test_rejects_matched_but_nonbaseline_arithmetic_environment(self):
        control, treatment = records()
        control["engine_environment"]["WASTE_Q8"] = "0"
        treatment["engine_environment"]["WASTE_Q8"] = "0"
        result = compare(control, treatment)
        self.assertTrue(result["valid"])
        self.assertFalse(result["checks"]["stable_row_arithmetic"])
        self.assertFalse(result["accepted"])

    def test_rejects_route_drift_or_no_byte_reduction(self):
        control, treatment = records()
        hit_trace = treatment["requests"][1]["trace"]
        hit_trace["decode_route_positions"]["150"]["sha256"] = "different"
        hit_trace["expert_bytes_read"] = 1000
        result = compare(control, treatment)
        self.assertFalse(result["checks"]["decode_routes_identical"])
        self.assertFalse(result["checks"]["expert_bytes_reduced"])

    def test_rejects_trace_rows_buffered_past_request_boundary(self):
        control, treatment = records()
        treatment["requests"][1]["trace"]["boundary_complete"] = False
        result = compare(control, treatment)
        self.assertFalse(result["checks"]["direct_io_qd4_no_fallback"])
        self.assertFalse(result["accepted"])

    def test_rejects_transport_memory_pressure_and_changed_tool_hit(self):
        control, treatment = records()
        treatment["requests"][1]["response_waste"]["io_fallback"] = True
        treatment["requests"][1]["trace"]["boundary_complete"] = False
        treatment["peak_process_swap_bytes"] = 4096
        treatment["vm_delta"]["pgscan_direct"] = 2
        treatment["memory_psi_total_us_delta"]["some"] = 1
        changed = treatment["requests"][2]["response_waste"]["prefix_cache"]
        changed.update({"status": "hit", "hit": True, "hit_kind": "family"})
        result = compare(control, treatment)
        self.assertFalse(result["checks"]["direct_io_qd4_no_fallback"])
        self.assertFalse(result["checks"]["no_process_swap"])
        self.assertFalse(result["checks"]["no_reclaim"])
        self.assertFalse(result["checks"]["memory_psi_within_limit"])
        self.assertFalse(result["checks"]["changed_tool_miss"])

    def test_rejects_changed_tool_root_that_cannot_be_admitted(self):
        control, treatment = records()
        changed = treatment["requests"][2]["response_waste"]["prefix_cache"]
        changed.update({"status": "miss_too_large",
                        "family_snapshot_bytes": 0,
                        "family_entries": 1})
        result = compare(control, treatment)
        self.assertFalse(result["checks"]["changed_tool_miss"])
        self.assertFalse(result["accepted"])

    def test_configuration_mismatch_is_invalid(self):
        control, treatment = records()
        treatment["budget_bytes"] -= 1
        result = compare(control, treatment)
        self.assertFalse(result["valid"])
        self.assertIn("budget_bytes", result["mismatches"])


class TestFamilyTraceSummary(unittest.TestCase):
    def test_routes_and_expert_bytes_are_grouped_per_request(self):
        common = {"io_backend": "io_uring", "queue_depth": 4,
                  "direct_io": True, "read_error": False}
        rows = [
            {"event": "prefill_chunk", "position_start": 5,
             "position_end": 7, "tokens": 3, "bytes_read": 100,
             "cache_hits": 2, "cache_misses": 3, **common},
            {"event": "decode_layer", "position": 8, "layer": 1,
             "experts": [3, 4], **common},
            {"event": "decode_layer", "position": 9, "layer": 1,
             "experts": [5, 6], **common},
            {"event": "decode_token", "position": 8, "bytes_read": 20,
             "cache_hits": 4, "cache_misses": 1, **common},
            {"event": "decode_token", "position": 9, "bytes_read": 30,
             "cache_hits": 5, "cache_misses": 1, **common},
        ]
        result = trace_summary(rows)
        self.assertTrue(result["transport_ok"])
        self.assertEqual(result["prefill_tokens"], 3)
        self.assertEqual(result["expert_bytes_read"], 150)
        self.assertEqual(set(result["decode_route_positions"]), {"8", "9"})

    def test_fallback_transport_is_not_accepted(self):
        result = trace_summary([{
            "event": "decode_token", "position": 8, "bytes_read": 1,
            "io_backend": "pread", "queue_depth": 1,
            "direct_io": True, "read_error": False,
        }])
        self.assertFalse(result["transport_ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
