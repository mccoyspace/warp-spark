# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Policy checks for the exact in-process prefix cache."""

import platform
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from serve.prefix_cache import (CONTROLLER_OVERHEAD_BYTES,               # noqa: E402
                                ENTRY_OVERHEAD_BYTES, PrefixCache, _Entry)
from tests.serve.fake_engine import FakeEngine             # noqa: E402


class TestPrefixCachePolicy(unittest.TestCase):
    @staticmethod
    def finish(engine, suffix):
        output = []
        engine.generate(suffix,
                        lambda token, *_: output.append(token) or True,
                        max_tokens=4)
        return output, bytes(engine.state_export())

    def test_deepest_exact_ancestor_never_passes_stable_boundary(self):
        identity = ("one engine",)
        cache = PrefixCache(1 << 20, 4, identity)
        engine = FakeEngine(host_reserved_bytes=1 << 20, prefill_chunk=4)
        tokens = list(range(1, 21))

        engine.state_reset()
        _, first = cache.prepare(engine, tokens, [12], identity)
        self.assertEqual(first.status, "miss")
        self.assertEqual(first.cached_tokens, 12)

        # The twelve-token entry is an exact prefix, but this request proves
        # only four tokens are a stable family root. Restoring past that
        # boundary would turn a prior request tail into shared state.
        engine.state_reset()
        _, shorter = cache.prepare(engine, tokens, [4], identity)
        self.assertEqual(shorter.status, "miss")
        self.assertEqual(engine.imports, 0)
        self.assertEqual(shorter.cached_tokens, 4)

        # Both exact ancestors now exist. The deepest allowed one wins.
        engine.state_reset()
        suffix, deepest = cache.prepare(engine, tokens, [12], identity)
        self.assertEqual(deepest.status, "hit")
        self.assertEqual(deepest.restored_tokens, 12)
        self.assertEqual(suffix, tokens[12:])

    def test_token_mismatch_is_never_a_hash_style_hit(self):
        identity = ("one engine",)
        cache = PrefixCache(1 << 20, 4, identity)
        engine = FakeEngine(host_reserved_bytes=1 << 20, prefill_chunk=4)
        original = list(range(1, 21))
        changed = list(original)
        changed[5] = 999

        engine.state_reset()
        cache.prepare(engine, original, [12], identity)
        engine.state_reset()
        _, use = cache.prepare(engine, changed, [12], identity)
        self.assertEqual(use.status, "miss")
        self.assertEqual(engine.imports, 0)

    def test_mutable_head_restores_exact_history_and_replaces_one_branch(self):
        identity = ("one engine",)
        cache = PrefixCache(1 << 20, 2, identity, conversation_head=True)
        engine = FakeEngine(host_reserved_bytes=1 << 20, prefill_chunk=4)
        first = list(range(1, 22))

        engine.state_reset()
        suffix, cold = cache.prepare(engine, first, [8], identity)
        _, cold_state = self.finish(engine, suffix)
        self.assertEqual(cold.status, "miss")
        self.assertEqual(cold.cached_tokens, 20)
        self.assertEqual(cold.head_cached_tokens, 20)
        self.assertEqual(cache.stats()["root_entries"], 1)
        self.assertEqual(cache.stats()["head_entries"], 1)

        # A real next chat request contains the exact old request followed by
        # the assistant response and next user turn. The mutable checkpoint
        # should win over the stable family root, then move forward once.
        extended = first + list(range(101, 113))
        engine.state_reset()
        suffix, warm = cache.prepare(engine, extended, [8], identity)
        _, warm_state = self.finish(engine, suffix)
        self.assertEqual(warm.status, "hit")
        self.assertEqual(warm.restored_kind, "head")
        self.assertEqual(warm.restored_tokens, 20)
        self.assertEqual(warm.head_cached_tokens, 32)
        self.assertEqual(suffix, extended[32:])
        stats = cache.stats()
        self.assertEqual(stats["root_entries"], 1)
        self.assertEqual(stats["head_entries"], 1)
        self.assertEqual(stats["head_hits"], 1)
        self.assertEqual(stats["head_replacements"], 1)

        baseline = FakeEngine(prefill_chunk=4)
        baseline.state_reset()
        _, expected_state = self.finish(baseline, extended)
        self.assertNotEqual(cold_state, warm_state)
        self.assertEqual(warm_state, expected_state)

    def test_divergent_history_falls_back_to_root_without_losing_it(self):
        identity = ("one engine",)
        cache = PrefixCache(1 << 20, 2, identity, conversation_head=True)
        engine = FakeEngine(host_reserved_bytes=1 << 20, prefill_chunk=4)
        original = list(range(1, 22))
        changed = list(original)
        changed[12] = 999

        engine.state_reset()
        cache.prepare(engine, original, [8], identity)
        engine.state_reset()
        suffix, use = cache.prepare(engine, changed, [8], identity)
        self.assertEqual(use.status, "hit")
        self.assertEqual(use.restored_kind, "root")
        self.assertEqual(use.restored_tokens, 8)
        self.assertEqual(suffix, changed[20:])
        stats = cache.stats()
        self.assertEqual(stats["root_entries"], 1)
        self.assertEqual(stats["head_entries"], 1)
        self.assertEqual(stats["root_hits"], 1)

    def test_head_budget_rejection_preserves_stable_root(self):
        identity = ("one engine",)
        engine = FakeEngine(prefill_chunk=4)
        tokens = list(range(1, 22))
        # The root snapshot fits exactly; a second snapshot cannot. Head
        # acceleration is optional and must never evict this stable root.
        root_snapshot = 8 + 4 * 8
        root_charge = root_snapshot + 4 * 8 + ENTRY_OVERHEAD_BYTES
        limit = CONTROLLER_OVERHEAD_BYTES + root_charge
        engine.host_reserved_bytes = limit
        cache = PrefixCache(limit, 2, identity, conversation_head=True)

        engine.state_reset()
        suffix, use = cache.prepare(engine, tokens, [8], identity)
        self.assertEqual(suffix, tokens[20:])
        self.assertEqual(use.cached_tokens, 8)
        self.assertEqual(use.head_cached_tokens, 0)
        stats = cache.stats()
        self.assertEqual(stats["root_entries"], 1)
        self.assertEqual(stats["head_entries"], 0)
        self.assertEqual(stats["admission_rejects"], 1)

    def test_semantic_lru_retains_multiple_exact_turn_ancestors(self):
        identity = ("one engine",)
        cache = PrefixCache(1 << 20, 4, identity, semantic_anchors=True)
        engine = FakeEngine(host_reserved_bytes=1 << 20, prefill_chunk=4)
        first = list(range(1, 22))

        engine.state_reset()
        suffix, cold = cache.prepare(
            engine, first, [8], identity, [17])
        self.assertEqual(cold.status, "miss")
        self.assertEqual(cold.anchor_cached_tokens, 16)
        self.assertEqual(suffix, first[16:])

        extended = first + list(range(101, 110))
        engine.state_reset()
        suffix, warm = cache.prepare(
            engine, extended, [8], identity, [25])
        self.assertEqual(warm.status, "hit")
        self.assertEqual(warm.restored_kind, "anchor")
        self.assertEqual(warm.restored_tokens, 16)
        self.assertEqual(warm.anchor_cached_tokens, 24)
        self.assertEqual(suffix, extended[24:])

        changed = list(extended)
        changed[20] = 999
        engine.state_reset()
        suffix, fallback = cache.prepare(
            engine, changed, [8], identity, [25])
        self.assertEqual(fallback.status, "hit")
        self.assertEqual(fallback.restored_kind, "anchor")
        self.assertEqual(fallback.restored_tokens, 16)
        self.assertEqual(suffix, changed[24:])
        stats = cache.stats()
        self.assertEqual(stats["root_entries"], 1)
        self.assertEqual(stats["anchor_entries"], 3)
        self.assertEqual(stats["anchor_hits"], 2)

    def test_semantic_anchor_pressure_evicts_anchor_not_root(self):
        identity = ("one engine",)
        cache = PrefixCache(1 << 20, 2, identity, semantic_anchors=True)
        engine = FakeEngine(host_reserved_bytes=1 << 20, prefill_chunk=4)
        first = list(range(1, 22))
        second = first + list(range(101, 110))

        engine.state_reset()
        cache.prepare(engine, first, [8], identity, [17])
        old_anchor = next(e.token_bytes for e in cache._entries
                          if e.kind == "anchor")
        engine.state_reset()
        cache.prepare(engine, second, [8], identity, [25])

        stats = cache.stats()
        self.assertEqual(stats["root_entries"], 1)
        self.assertEqual(stats["anchor_entries"], 1)
        self.assertEqual(stats["evictions"], 1)
        self.assertNotIn(old_anchor,
                         [e.token_bytes for e in cache._entries])

    def test_oversized_later_anchor_preserves_smaller_old_anchor(self):
        identity = ("one engine",)
        engine = FakeEngine(prefill_chunk=4)
        first = list(range(1, 18))
        root_snapshot = 8 + 4 * 8
        root_charge = root_snapshot + 4 * 8 + ENTRY_OVERHEAD_BYTES
        old_snapshot = 8 + 4 * 12
        old_charge = old_snapshot + 4 * 12 + ENTRY_OVERHEAD_BYTES
        limit = CONTROLLER_OVERHEAD_BYTES + root_charge + old_charge
        engine.host_reserved_bytes = limit
        cache = PrefixCache(limit, 3, identity, semantic_anchors=True)

        engine.state_reset()
        cache.prepare(engine, first, [8], identity, [13])
        old_anchor = next(e for e in cache._entries if e.kind == "anchor")
        old_key = old_anchor.token_bytes
        old_blob = bytes(old_anchor.blob)

        extended = first + list(range(101, 110))
        engine.state_reset()
        _, use = cache.prepare(engine, extended, [8], identity, [21])
        self.assertEqual(use.status, "hit")
        self.assertEqual(use.restored_tokens, 12)
        anchors = [e for e in cache._entries if e.kind == "anchor"]
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0].token_bytes, old_key)
        self.assertEqual(bytes(anchors[0].blob), old_blob)
        self.assertEqual(cache.stats()["admission_rejects"], 1)

    def test_semantic_anchor_works_without_a_family_root(self):
        identity = ("one engine",)
        cache = PrefixCache(1 << 20, 2, identity, semantic_anchors=True)
        engine = FakeEngine(host_reserved_bytes=1 << 20, prefill_chunk=4)
        tokens = list(range(1, 22))

        engine.state_reset()
        cache.prepare(engine, tokens, [], identity, [13])
        engine.state_reset()
        suffix, use = cache.prepare(engine, tokens, [], identity, [13])
        self.assertEqual(use.status, "hit")
        self.assertEqual(use.restored_kind, "anchor")
        self.assertEqual(use.restored_tokens, 12)
        self.assertEqual(suffix, tokens[12:])

    def test_shallow_hit_promotes_deeper_root_and_preserves_full_state(self):
        identity = ("one engine",)
        cache = PrefixCache(1 << 20, 4, identity)
        engine = FakeEngine(host_reserved_bytes=1 << 20, prefill_chunk=4)
        tokens = list(range(1, 25))

        engine.state_reset()
        suffix, shallow = cache.prepare(engine, tokens, [8], identity)
        shallow_output, shallow_state = self.finish(engine, suffix)
        self.assertEqual(shallow.status, "miss")
        self.assertEqual(shallow.cached_tokens, 8)

        engine.state_reset()
        suffix, promoted = cache.prepare(engine, tokens, [16], identity)
        promoted_output, promoted_state = self.finish(engine, suffix)
        self.assertEqual(promoted.status, "hit")
        self.assertEqual(promoted.restored_tokens, 8)
        self.assertEqual(promoted.promoted_tokens, 8)
        self.assertEqual(promoted.cached_tokens, 16)
        self.assertEqual(engine.prefills[-1], tokens[8:16])

        engine.state_reset()
        suffix, deepest = cache.prepare(engine, tokens, [16], identity)
        deepest_output, deepest_state = self.finish(engine, suffix)
        self.assertEqual(deepest.status, "hit")
        self.assertEqual(deepest.restored_tokens, 16)
        self.assertEqual(deepest.promoted_tokens, 0)
        self.assertEqual(shallow_output, promoted_output)
        self.assertEqual(shallow_output, deepest_output)
        self.assertEqual(shallow_state, promoted_state)
        self.assertEqual(shallow_state, deepest_state)

        stats = cache.stats()
        self.assertEqual(stats["admissions"], 2)
        self.assertEqual(stats["entries"], 2)
        self.assertEqual(stats["promotions"], 1)
        self.assertEqual(stats["promoted_tokens"], 8)
        self.assertLessEqual(stats["bytes_used"], stats["max_bytes"])
        self.assertEqual(stats["controller_bytes"], CONTROLLER_OVERHEAD_BYTES)

    def test_failed_restore_releases_old_blob_before_replacement_export(self):
        identity = ("one engine",)
        cache = PrefixCache(1 << 20, 1, identity)
        engine = FakeEngine(host_reserved_bytes=1 << 20, prefill_chunk=4)
        tokens = list(range(1, 21))
        engine.state_reset()
        cache.prepare(engine, tokens, [12], identity)
        old_blob = cache._entries[0].blob
        real_export = engine.state_export
        lengths_at_export = []

        def checked_export():
            lengths_at_export.append(len(old_blob))
            return real_export()

        engine.state_export = checked_export
        engine.fail_state_import_once = True
        engine.state_reset()
        _, use = cache.prepare(engine, tokens, [12], identity)
        self.assertEqual(use.status, "restore_failed")
        self.assertEqual(lengths_at_export, [0])
        self.assertEqual(len(old_blob), 0)
        self.assertEqual(cache.stats()["entries"], 1)

    def test_one_entry_promotion_releases_shallow_blob_before_export(self):
        identity = ("one engine",)
        cache = PrefixCache(1 << 20, 1, identity)
        engine = FakeEngine(host_reserved_bytes=1 << 20, prefill_chunk=4)
        tokens = list(range(1, 25))

        engine.state_reset()
        cache.prepare(engine, tokens, [8], identity)
        old_blob = cache._entries[0].blob
        real_export = engine.state_export
        lengths_at_export = []

        def checked_export():
            lengths_at_export.append(len(old_blob))
            return real_export()

        engine.state_export = checked_export
        engine.state_reset()
        suffix, use = cache.prepare(engine, tokens, [16], identity)
        self.assertEqual(use.status, "hit")
        self.assertEqual(use.restored_tokens, 8)
        self.assertEqual(use.cached_tokens, 16)
        self.assertEqual(use.promoted_tokens, 8)
        self.assertEqual(suffix, tokens[16:])
        self.assertEqual(lengths_at_export, [0])
        self.assertEqual(len(old_blob), 0)
        stats = cache.stats()
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(stats["evictions"], 1)
        self.assertLessEqual(stats["bytes_used"], stats["max_bytes"])

    def test_insert_allocation_failures_reset_for_unsplit_fallback(self):
        identity = ("one engine",)
        tokens = list(range(1, 21))

        baseline = FakeEngine(prefill_chunk=4)
        baseline.state_reset()
        expected_output, expected_state = self.finish(baseline, tokens)

        for error in (MemoryError, OverflowError):
            with self.subTest(error=error.__name__):
                cache = PrefixCache(1 << 20, 2, identity)
                engine = FakeEngine(
                    host_reserved_bytes=1 << 20, prefill_chunk=4)
                engine.state_reset()
                with patch.object(cache, "_insert", side_effect=error):
                    suffix, use = cache.prepare(
                        engine, tokens, [12], identity)
                self.assertEqual(use.status, "allocation_failed")
                self.assertEqual(suffix, tokens)
                output, state = self.finish(engine, suffix)
                self.assertEqual(output, expected_output)
                self.assertEqual(state, expected_state)
                stats = cache.stats()
                self.assertEqual(stats["allocation_failures"], 1)
                self.assertEqual(stats["entries"], 0)
                self.assertEqual(stats["bytes_used"],
                                 CONTROLLER_OVERHEAD_BYTES)

    @unittest.skipUnless(platform.python_implementation() == "CPython",
                         "shallow allocation accounting is CPython-specific")
    def test_metadata_charges_cover_retained_cpython_objects(self):
        key = bytes(17)
        blob = bytearray(19)
        entry = _Entry(key, 12345, "root", blob, 67890, 2, 3)
        entry_meta = (sys.getsizeof(entry)
                      + sys.getsizeof(key) - len(key)
                      + sys.getsizeof(blob) - len(blob)
                      + sys.getsizeof([entry]) - sys.getsizeof([]))
        entry_meta += sys.getsizeof(entry.kind)
        for value in (entry.n_tokens, entry.charge,
                      entry.created, entry.last_used):
            entry_meta += sys.getsizeof(value)
        self.assertLessEqual(entry_meta, ENTRY_OVERHEAD_BYTES)

        cache = PrefixCache(1 << 20, 8, ("identity",))
        controller = (sys.getsizeof(cache) + sys.getsizeof(cache.__dict__)
                      + sys.getsizeof(cache._entries)
                      + sys.getsizeof(cache._lock)
                      + sys.getsizeof(cache._counters))
        controller += sum(sys.getsizeof(key) + sys.getsizeof(value)
                          for key, value in cache._counters.items())
        self.assertLessEqual(controller, CONTROLLER_OVERHEAD_BYTES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
