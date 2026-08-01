# SPDX-License-Identifier: Apache-2.0
"""Checks for the exact-prefix acceptance comparator."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.analyze_prefix_cache import compare, markdown       # noqa: E402


def run_record(run_id, request_seconds, *, prefix=False):
    common = {
        "run_id": run_id,
        "server_returncode": 0,
        "manifest_sha256": "manifest",
        "budget_bytes": 86_583_021_568,
        "threads": 8,
        "cpu_set": "performance",
        "resolved_cpu_set": "5-9,15-19",
        "io_backend_requested": "io_uring",
        "io_queue_depth_requested": 4,
        "tokens_requested": 4,
        "prompt_sha256": "prompt",
        "peak_process_swap_bytes": 0,
        "peak_rss_bytes": 85_000_000_000,
        "min_mem_available_bytes": 39_000_000_000,
        "vm_delta": {
            "pswpout": 0, "pgscan_direct": 0, "pgscan_kswapd": 0,
            "pgsteal_direct": 0, "pgsteal_kswapd": 0,
        },
        "memory_psi_total_us_delta": {"some": 0, "full": 0},
    }
    if prefix:
        prefixes = [
            {"status": "miss_stored", "key_tokens": 10,
             "prompt_tokens_evaluated": 10, "reused_tokens": 0,
             "snapshot_bytes": 466_000_000, "capacity_bytes": 1 << 30,
             "resident_bytes": 466_000_400, "export_ms": 150.0,
             "prepare_ms": 130_000.0},
            {"status": "hit", "key_tokens": 10,
             "prompt_tokens_evaluated": 1, "reused_tokens": 9,
             "snapshot_bytes": 466_000_000, "capacity_bytes": 1 << 30,
             "resident_bytes": 466_000_400, "restore_ms": 20.0},
        ]
        totals = [(100, 10, 5), (140, 14, 12)]
    else:
        prefixes = [{"status": "disabled"}, {"status": "disabled"}]
        totals = [(100, 10, 5), (300, 30, 15)]
    common["requests"] = [
        {
            "wall_seconds": seconds,
            "answer_sha256": "answer",
            "waste": {
                "direct_io": True, "io_fallback": False,
                "bytes_read": totals[i][0],
                "experts_missed": totals[i][1],
                "experts_hit": totals[i][2],
                "prefix_cache": prefixes[i],
            },
        }
        for i, seconds in enumerate(request_seconds)
    ]
    return common


class TestAnalyzePrefixCache(unittest.TestCase):
    def setUp(self):
        self.control = run_record("control", [10.0, 12.0])
        self.treatment = run_record("treatment", [10.5, 2.0], prefix=True)

    def test_accepts_matching_bit_exact_hit(self):
        result = compare(self.control, self.treatment)
        self.assertTrue(result["valid"])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["latency"]["hit_speedup_vs_control_mean"], 5.5)
        self.assertAlmostEqual(
            result["request_2_engine_delta"]["bytes_reduction_fraction"], .8)
        rendered = markdown(result)
        self.assertIn("Accepted: **yes**", rendered)
        self.assertIn("80.00% less", rendered)

    def test_rejects_noncomparable_configuration(self):
        treatment = copy.deepcopy(self.treatment)
        treatment["budget_bytes"] -= 1
        result = compare(self.control, treatment)
        self.assertFalse(result["valid"])
        self.assertIn("budget_bytes", result["mismatches"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
