# SPDX-License-Identifier: Apache-2.0
"""Exact-prefix policy tests, independent of model numerics."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from serve.prefix_cache import ENTRY_OVERHEAD, PrefixCache       # noqa: E402
from tests.serve.fake_engine import FakeEngine                   # noqa: E402


class TestPrefixCache(unittest.TestCase):
    def test_exact_hit_restores_n_minus_one_and_replays_last(self):
        engine = FakeEngine(prefix_cache_bytes=4096)
        cache = PrefixCache(engine)
        tokens = [10, 20, 30, 40]

        tail, first = cache.prepare(tokens)
        self.assertEqual(tail, [40])
        self.assertEqual(first["status"], "miss_stored")
        self.assertEqual(engine.state_tokens, tokens[:-1])

        tail, second = cache.prepare(tokens)
        self.assertEqual(tail, [40])
        self.assertTrue(second["hit"])
        self.assertEqual(second["reused_tokens"], 3)
        self.assertEqual(second["prompt_tokens_evaluated"], 1)
        self.assertEqual(second["replayed_tokens"], 1)
        self.assertEqual(engine.state_tokens, tokens[:-1])

    def test_match_is_full_token_sequence_not_hash_or_lcp(self):
        engine = FakeEngine(prefix_cache_bytes=4096)
        cache = PrefixCache(engine)
        cache.prepare([1, 2, 3, 4])
        _, report = cache.prepare([1, 2, 3, 5])
        self.assertFalse(report["hit"])
        self.assertEqual(report["status"], "miss_stored")
        self.assertEqual(report["misses"], 2)

    def test_media_bypasses_even_when_token_ids_match(self):
        engine = FakeEngine(prefix_cache_bytes=4096)
        cache = PrefixCache(engine)
        tokens = [1, 2, 3]
        cache.prepare(tokens)
        tail, report = cache.prepare(tokens, n_images=1)
        self.assertEqual(tail, tokens)
        self.assertEqual(report["status"], "bypass_media")
        self.assertFalse(report["hit"])

    def test_lru_never_exceeds_reservation(self):
        # Three-token entry: 12 key + 8-byte snapshot + fixed overhead.
        one = 12 + 8 + ENTRY_OVERHEAD
        engine = FakeEngine(prefix_cache_bytes=one * 2)
        cache = PrefixCache(engine)
        cache.prepare([1, 2, 3])
        cache.prepare([4, 5, 6])
        cache.prepare([7, 8, 9])
        self.assertLessEqual(cache.resident_bytes, cache.capacity_bytes)
        self.assertEqual(cache.stats()["entries"], 2)
        self.assertEqual(cache.stats()["evictions"], 1)
        _, report = cache.prepare([1, 2, 3])
        self.assertFalse(report["hit"], "oldest entry should have been evicted")

    def test_failed_import_is_invalidated_and_rebuilt_cold(self):
        engine = FakeEngine(prefix_cache_bytes=4096)
        cache = PrefixCache(engine)
        tokens = [1, 2, 3]
        cache.prepare(tokens)
        engine.fail_next_import = True
        tail, report = cache.prepare(tokens)
        self.assertEqual(tail, [3])
        self.assertEqual(report["status"], "miss_invalidated")
        self.assertEqual(report["invalidations"], 1)
        self.assertEqual(engine.state_tokens, tokens[:-1])

    def test_snapshot_larger_than_budget_is_not_retained(self):
        engine = FakeEngine(prefix_cache_bytes=ENTRY_OVERHEAD)
        cache = PrefixCache(engine)
        _, report = cache.prepare([1, 2, 3])
        self.assertEqual(report["status"], "miss_too_large")
        self.assertEqual(report["entries"], 0)
        self.assertEqual(report["resident_bytes"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
