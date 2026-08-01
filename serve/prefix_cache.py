# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Exact and semantic-family prompt checkpoints inside one RAM reservation.

Exact entries retain the Sprint 4 contract: the complete prompt is their
lookup key and the checkpoint is taken after all but its final token.  Chat
rendering may additionally provide a stable leading semantic root (tools,
thinking-effort note and leading system turns).  Family entries are keyed by
the *evaluated int32 root tokens*.  A request restores its deepest matching
checkpoint, evaluates the exact suffix through the ordinary engine path, and
replays the final prompt token to regenerate logits.

This is deliberately a linear ancestor lookup, not a block-snapshot trie.
Images still bypass because placeholder ids do not identify pixels.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import struct
import time
from typing import Optional

from .engine import Engine, EngineError


ENTRY_OVERHEAD = 256
_EntryKey = tuple[str, bytes]


@dataclass
class _Entry:
    state: bytearray
    tokens: int                 # evaluated positions represented by state
    charge: int
    kind: str                   # "exact" or "family"


class PrefixCache:
    """Budgeted LRU with exact leaves and selective semantic family roots."""

    def __init__(self, engine: Engine, capacity_bytes: Optional[int] = None):
        self.engine = engine
        if capacity_bytes is None:
            capacity_bytes = int(
                engine.memory_used().get("prefix_cache_bytes", 0))
        if capacity_bytes < 0:
            raise ValueError("prefix cache capacity must be non-negative")
        self.capacity_bytes = int(capacity_bytes)
        self._entries: OrderedDict[_EntryKey, _Entry] = OrderedDict()
        self.resident_bytes = 0
        self.requests = 0
        self.hits = 0
        self.exact_hits = 0
        self.family_hits = 0
        self.misses = 0
        self.bypasses = 0
        self.evictions = 0
        self.invalidations = 0
        self.snapshot_errors = 0
        self.family_admissions = 0
        self.exact_admissions = 0
        self.admission_rejects = 0
        self.promotions = 0
        self.root_mismatches = 0
        self.tokens_reused = 0
        self.tokens_evaluated = 0

    @staticmethod
    def _key(tokens: list[int]) -> bytes:
        # Fixed-width explicit endian makes equality collision-free and
        # independent of Python hash randomisation.
        return struct.pack(f"<{len(tokens)}i", *tokens)

    def _drop(self, key: _EntryKey, *, eviction: bool = False) -> None:
        entry = self._entries.pop(key)
        self.resident_bytes -= entry.charge
        if eviction:
            self.evictions += 1

    def _make_room(self, charge: int, *, kind: str,
                   protect: frozenset[_EntryKey] = frozenset()) -> bool:
        """Evict before allocation, preserving a family root for exact leaves."""
        while self.resident_bytes + charge > self.capacity_bytes:
            candidates = [
                key for key, entry in self._entries.items()
                if key not in protect and (kind == "family"
                                           or entry.kind == "exact")]
            if not candidates:
                return False
            # Family admission first displaces exact leaves, then the oldest
            # unrelated family. Exact admission never displaces a root.
            if kind == "family":
                exact = [key for key in candidates
                         if self._entries[key].kind == "exact"]
                victim = exact[0] if exact else candidates[0]
            else:
                victim = candidates[0]
            self._drop(victim, eviction=True)
        return True

    def _store_current(self, key: _EntryKey, *, kind: str, tokens: int,
                       protect: frozenset[_EntryKey] = frozenset()
                       ) -> tuple[int, float, str]:
        """Store the engine's current state; return bytes, ms, disposition."""
        existing = self._entries.get(key)
        if existing is not None:
            self._entries.move_to_end(key)
            return len(existing.state), 0.0, "existing"

        try:
            size = self.engine.state_size()
            charge = len(key[1]) + size + ENTRY_OVERHEAD
            if charge > self.capacity_bytes:
                self.admission_rejects += 1
                return 0, 0.0, "too_large"
            if not self._make_room(charge, kind=kind, protect=protect):
                self.admission_rejects += 1
                return 0, 0.0, "no_room"

            started = time.monotonic()
            blob = self.engine.state_export()
            elapsed = (time.monotonic() - started) * 1000
            actual = len(key[1]) + len(blob) + ENTRY_OVERHEAD
            if actual > self.capacity_bytes:
                self.admission_rejects += 1
                return 0, elapsed, "too_large"
            # state_size is the canonical exact size, but retain the check so
            # a faulty Engine implementation cannot violate the reservation.
            if not self._make_room(actual, kind=kind, protect=protect):
                self.admission_rejects += 1
                return 0, elapsed, "no_room"
            self._entries[key] = _Entry(blob, tokens, actual, kind)
            self.resident_bytes += actual
            if kind == "family":
                self.family_admissions += 1
            else:
                self.exact_admissions += 1
            return len(blob), elapsed, "stored"
        except (EngineError, MemoryError):
            self.snapshot_errors += 1
            return 0, 0.0, "error"

    def _candidates(self, prompt_key: bytes, prompt_tokens: int
                    ) -> list[tuple[_EntryKey, _Entry]]:
        out = []
        for key, entry in self._entries.items():
            kind, token_key = key
            if entry.tokens > prompt_tokens - 1:
                continue
            if ((kind == "exact" and token_key == prompt_key)
                    or (kind == "family" and prompt_key.startswith(token_key))):
                out.append((key, entry))
        # An exact entry normally wins at n-1; otherwise choose the deepest
        # semantic ancestor. Stable sort keeps LRU order for equal depths.
        out.sort(key=lambda item: item[1].tokens, reverse=True)
        return out

    def clear(self) -> None:
        self._entries.clear()
        self.resident_bytes = 0

    def stats(self) -> dict:
        family_entries = sum(e.kind == "family"
                             for e in self._entries.values())
        return {
            "enabled": self.capacity_bytes > 0,
            "mode": "exact_family_root",
            "capacity_bytes": self.capacity_bytes,
            "resident_bytes": self.resident_bytes,
            "entries": len(self._entries),
            "exact_entries": len(self._entries) - family_entries,
            "family_entries": family_entries,
            "requests": self.requests,
            "hits": self.hits,
            "exact_hits": self.exact_hits,
            "family_hits": self.family_hits,
            "misses": self.misses,
            "bypasses": self.bypasses,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "snapshot_errors": self.snapshot_errors,
            "family_admissions": self.family_admissions,
            "exact_admissions": self.exact_admissions,
            "admission_rejects": self.admission_rejects,
            "promotions": self.promotions,
            "root_mismatches": self.root_mismatches,
            "tokens_reused": self.tokens_reused,
            "tokens_evaluated": self.tokens_evaluated,
        }

    def _report(self, *, status: str, hit: bool, tokens: int,
                hit_kind: str = "none", reused: int = 0,
                evaluated: Optional[int] = None, checkpoint_tokens: int = 0,
                family_root_tokens: int = 0, snapshot_bytes: int = 0,
                family_snapshot_bytes: int = 0,
                exact_snapshot_bytes: int = 0,
                prepare_ms: float = 0.0, restore_ms: float = 0.0,
                suffix_prefill_ms: float = 0.0, export_ms: float = 0.0,
                admission: str = "none") -> dict:
        evaluated = tokens if evaluated is None else evaluated
        out = self.stats()
        out.update({
            "status": status,
            "hit": hit,
            "hit_kind": hit_kind,
            "key_tokens": tokens,
            "checkpoint_tokens": checkpoint_tokens,
            "family_root_tokens": family_root_tokens,
            "reused_tokens": reused,
            "prompt_tokens_evaluated": evaluated,
            "suffix_tokens_evaluated": evaluated,
            "replayed_tokens": evaluated if hit else 0,
            "snapshot_bytes": snapshot_bytes,
            "family_snapshot_bytes": family_snapshot_bytes,
            "exact_snapshot_bytes": exact_snapshot_bytes,
            "admission": admission,
            "prepare_ms": round(prepare_ms, 3),
            "restore_ms": round(restore_ms, 3),
            "suffix_prefill_ms": round(suffix_prefill_ms, 3),
            "export_ms": round(export_ms, 3),
        })
        return out

    def prepare(self, tokens: list[int], *, n_images: int = 0,
                family_root_tokens: Optional[list[int]] = None
                ) -> tuple[list[int], dict]:
        """Reset, restore the deepest ancestor, and evaluate through n-1.

        The caller holds the engine lock across this method and the following
        generate call.  Returning the final token preserves the canonical
        logits-replay path used by exact Sprint 4 entries.
        """
        self.requests += 1
        started = time.monotonic()
        self.engine.state_reset()

        def bypass(status: str):
            self.bypasses += 1
            self.tokens_evaluated += len(tokens)
            return list(tokens), self._report(
                status=status, hit=False, tokens=len(tokens),
                prepare_ms=(time.monotonic() - started) * 1000)

        if self.capacity_bytes == 0:
            return bypass("bypass_disabled")
        if n_images:
            return bypass("bypass_media")
        if len(tokens) < 2:
            return bypass("bypass_short_prompt")

        prompt_key = self._key(tokens)
        exact_depth = len(tokens) - 1
        root = list(family_root_tokens or [])
        if root and (len(root) > exact_depth
                     or tokens[:len(root)] != root):
            self.root_mismatches += 1
            root = []
        root_key: Optional[_EntryKey] = (
            ("family", self._key(root)) if root else None)

        restored_entry: Optional[_Entry] = None
        restore_ms = 0.0
        invalidated = False
        for key, entry in self._candidates(prompt_key, len(tokens)):
            restore_started = time.monotonic()
            try:
                self.engine.state_import(entry.state)
            except EngineError:
                # Full snapshots are independent: invalidate only the bad
                # blob and fall back to the next exact ancestor or cold state.
                self._drop(key)
                self.invalidations += 1
                invalidated = True
                self.engine.state_reset()
                continue
            restore_ms += (time.monotonic() - restore_started) * 1000
            restored_entry = entry
            self._entries.move_to_end(key)
            break

        hit = restored_entry is not None
        if hit:
            self.hits += 1
            if restored_entry.kind == "family":
                self.family_hits += 1
            else:
                self.exact_hits += 1
        else:
            self.misses += 1

        reused = restored_entry.tokens if restored_entry else 0
        current_depth = reused
        suffix_prefill_ms = 0.0
        export_ms = 0.0
        family_snapshot_bytes = 0
        exact_snapshot_bytes = 0
        dispositions: list[str] = []

        # Admit exactly one semantic root supplied by rendering. If a
        # shallower checkpoint was restored, this is an explicit promotion.
        if (root_key is not None and root_key not in self._entries
                and current_depth <= len(root)):
            if current_depth < len(root):
                t0 = time.monotonic()
                self.engine.prefill(tokens[current_depth:len(root)])
                suffix_prefill_ms += (time.monotonic() - t0) * 1000
                current_depth = len(root)
            family_snapshot_bytes, elapsed, disposition = self._store_current(
                root_key, kind="family", tokens=len(root))
            export_ms += elapsed
            dispositions.append("family_" + disposition)
            if disposition == "stored" and hit:
                self.promotions += 1

        # Replay the exact suffix through waste_eval in one chunked call.
        # A rooted request deliberately does not add an exact leaf: every K3
        # checkpoint duplicates the ~443 MiB fixed recurrent base, so a root
        # plus one leaf would let a single family consume a 1 GiB reservation.
        if current_depth < exact_depth:
            t0 = time.monotonic()
            self.engine.prefill(tokens[current_depth:exact_depth])
            suffix_prefill_ms += (time.monotonic() - t0) * 1000
            current_depth = exact_depth

        exact_key: _EntryKey = ("exact", prompt_key)
        if root_key is None and exact_key not in self._entries:
            exact_snapshot_bytes, elapsed, disposition = self._store_current(
                exact_key, kind="exact", tokens=exact_depth)
            export_ms += elapsed
            dispositions.append("exact_" + disposition)

        evaluated = len(tokens) - reused
        self.tokens_reused += reused
        self.tokens_evaluated += evaluated
        if hit:
            status = "hit"
        elif invalidated:
            status = "miss_invalidated"
        elif family_snapshot_bytes:
            status = "miss_stored_family"
        elif exact_snapshot_bytes:
            status = "miss_stored"
        elif any(x.endswith("too_large") for x in dispositions):
            status = "miss_too_large"
        elif any(x.endswith("error") for x in dispositions):
            status = "miss_snapshot_error"
        else:
            status = "miss_not_admitted"

        snapshot_bytes = (len(restored_entry.state) if restored_entry
                          else exact_snapshot_bytes or family_snapshot_bytes)
        return [tokens[-1]], self._report(
            status=status, hit=hit, tokens=len(tokens),
            hit_kind=restored_entry.kind if restored_entry else "none",
            reused=reused, evaluated=evaluated,
            checkpoint_tokens=reused,
            family_root_tokens=len(root), snapshot_bytes=snapshot_bytes,
            family_snapshot_bytes=family_snapshot_bytes,
            exact_snapshot_bytes=exact_snapshot_bytes,
            admission=",".join(dispositions) or "none",
            prepare_ms=(time.monotonic() - started) * 1000,
            restore_ms=restore_ms, suffix_prefill_ms=suffix_prefill_ms,
            export_ms=export_ms)
