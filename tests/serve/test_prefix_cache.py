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

    def test_family_root_replays_changed_suffix_bit_exactly(self):
        engine = FakeEngine(prefix_cache_bytes=4096)
        cache = PrefixCache(engine)
        root = [10, 20]
        first_tokens = root + [30, 40, 50]
        second_tokens = root + [31, 41, 51]

        tail, first = cache.prepare(
            first_tokens, family_root_tokens=root)
        self.assertEqual(tail, [50])
        self.assertEqual(first["status"], "miss_stored_family")
        self.assertEqual(engine.state_tokens, first_tokens[:-1])
        self.assertEqual(first["family_entries"], 1)
        self.assertEqual(first["exact_entries"], 0)

        tail, second = cache.prepare(
            second_tokens, family_root_tokens=root)
        self.assertEqual(tail, [51])
        self.assertTrue(second["hit"])
        self.assertEqual(second["hit_kind"], "family")
        self.assertEqual(second["checkpoint_tokens"], len(root))
        self.assertEqual(second["reused_tokens"], len(root))
        self.assertEqual(second["prompt_tokens_evaluated"], 3)
        self.assertEqual(engine.state_tokens, second_tokens[:-1])

        # Reconstructing the same state cold gives the identical evaluated
        # token sequence; the real-engine suite covers identical logits.
        cold = FakeEngine(prefix_cache_bytes=0)
        cold.state_reset()
        cold.prefill(second_tokens[:-1])
        self.assertEqual(engine.state_export(), cold.state_export())

    def test_changed_semantic_root_is_an_exact_miss(self):
        engine = FakeEngine(prefix_cache_bytes=4096)
        cache = PrefixCache(engine)
        cache.prepare([1, 2, 30, 40], family_root_tokens=[1, 2])
        _, report = cache.prepare(
            [1, 3, 30, 40], family_root_tokens=[1, 3])
        self.assertFalse(report["hit"])
        self.assertEqual(report["hit_kind"], "none")

    def test_bad_family_root_is_invalidated_and_rebuilt_cold(self):
        engine = FakeEngine(prefix_cache_bytes=4096)
        cache = PrefixCache(engine)
        tokens = [7, 8, 9, 10, 11]
        root = tokens[:2]
        cache.prepare(tokens, family_root_tokens=root)
        engine.fail_next_import = True
        _, report = cache.prepare(tokens, family_root_tokens=root)
        self.assertFalse(report["hit"])
        self.assertEqual(report["status"], "miss_invalidated")
        self.assertEqual(report["invalidations"], 1)
        self.assertEqual(engine.state_tokens, tokens[:-1])

    def test_two_family_roots_coexist_without_exact_leaf_duplication(self):
        # Two-token root: 8-byte key + 8-byte state + fixed overhead.
        one = 8 + 8 + ENTRY_OVERHEAD
        engine = FakeEngine(prefix_cache_bytes=one * 2)
        cache = PrefixCache(engine)
        cache.prepare([1, 2, 10, 11], family_root_tokens=[1, 2])
        cache.prepare([3, 4, 20, 21], family_root_tokens=[3, 4])
        self.assertEqual(cache.stats()["family_entries"], 2)
        self.assertEqual(cache.stats()["exact_entries"], 0)
        self.assertLessEqual(cache.resident_bytes, cache.capacity_bytes)

        _, report = cache.prepare(
            [1, 2, 12, 13], family_root_tokens=[1, 2])
        self.assertTrue(report["hit"])
        self.assertEqual(report["hit_kind"], "family")


if __name__ == "__main__":
    unittest.main(verbosity=2)
