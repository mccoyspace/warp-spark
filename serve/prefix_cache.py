# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Exact prompt-prefix snapshots, budgeted against the expert cache.

The first implementation is intentionally narrow: a key is the complete
int32 token sequence and a hit must be exact.  It checkpoints the engine
after all but the final prompt token.  Restoring that state and replaying the
last token regenerates the exact logits through the normal forward path, so
the cache does not need to retain logits or depend on sampling parameters.

Images bypass the cache.  Equal media placeholder ids do not prove equal
pixels; content-addressed media keying belongs in a later design.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import struct
import time
from typing import Optional

from .engine import Engine, EngineError


ENTRY_OVERHEAD = 256


@dataclass
class _Entry:
    state: bytearray
    tokens: int
    charge: int


class PrefixCache:
    """LRU of exact-prefix state snapshots owned by one Engine."""

    def __init__(self, engine: Engine, capacity_bytes: Optional[int] = None):
        self.engine = engine
        if capacity_bytes is None:
            capacity_bytes = int(
                engine.memory_used().get("prefix_cache_bytes", 0))
        if capacity_bytes < 0:
            raise ValueError("prefix cache capacity must be non-negative")
        self.capacity_bytes = int(capacity_bytes)
        self._entries: OrderedDict[bytes, _Entry] = OrderedDict()
        self.resident_bytes = 0
        self.requests = 0
        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.evictions = 0
        self.invalidations = 0
        self.snapshot_errors = 0

    @staticmethod
    def _key(tokens: list[int]) -> bytes:
        # Fixed-width, explicit-endian token ids make equality collision-free
        # and independent of Python object identity or hash randomisation.
        return struct.pack(f"<{len(tokens)}i", *tokens)

    def _drop(self, key: bytes, *, eviction: bool = False) -> None:
        entry = self._entries.pop(key)
        self.resident_bytes -= entry.charge
        if eviction:
            self.evictions += 1

    def _make_room(self, charge: int) -> None:
        while self._entries and self.resident_bytes + charge > self.capacity_bytes:
            key = next(iter(self._entries))
            self._drop(key, eviction=True)

    def clear(self) -> None:
        self._entries.clear()
        self.resident_bytes = 0

    def stats(self) -> dict:
        return {
            "enabled": self.capacity_bytes > 0,
            "mode": "exact",
            "capacity_bytes": self.capacity_bytes,
            "resident_bytes": self.resident_bytes,
            "entries": len(self._entries),
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "bypasses": self.bypasses,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "snapshot_errors": self.snapshot_errors,
        }

    def _report(self, *, status: str, hit: bool, tokens: int,
                reused: int = 0, evaluated: Optional[int] = None,
                snapshot_bytes: int = 0, prepare_ms: float = 0.0,
                restore_ms: float = 0.0, export_ms: float = 0.0) -> dict:
        out = self.stats()
        out.update({
            "status": status,
            "hit": hit,
            "key_tokens": tokens,
            "reused_tokens": reused,
            "prompt_tokens_evaluated": tokens if evaluated is None else evaluated,
            "replayed_tokens": 1 if tokens > 1 and evaluated == 1 else 0,
            "snapshot_bytes": snapshot_bytes,
            "prepare_ms": round(prepare_ms, 3),
            "restore_ms": round(restore_ms, 3),
            "export_ms": round(export_ms, 3),
        })
        return out

    def prepare(self, tokens: list[int], *, n_images: int = 0
                ) -> tuple[list[int], dict]:
        """Reset and prepare one standalone request.

        Returns the prompt suffix the ordinary generation path must evaluate
        plus per-request/cumulative observability.  The caller must already
        hold the engine lock across this method and the following generate.
        """
        self.requests += 1
        started = time.monotonic()
        self.engine.state_reset()

        if self.capacity_bytes == 0:
            self.bypasses += 1
            return list(tokens), self._report(
                status="bypass_disabled", hit=False, tokens=len(tokens),
                prepare_ms=(time.monotonic() - started) * 1000)
        if n_images:
            self.bypasses += 1
            return list(tokens), self._report(
                status="bypass_media", hit=False, tokens=len(tokens),
                prepare_ms=(time.monotonic() - started) * 1000)
        if len(tokens) < 2:
            self.bypasses += 1
            return list(tokens), self._report(
                status="bypass_short_prompt", hit=False, tokens=len(tokens),
                prepare_ms=(time.monotonic() - started) * 1000)

        key = self._key(tokens)
        entry = self._entries.get(key)
        invalidated = False
        if entry is not None:
            restore_started = time.monotonic()
            try:
                self.engine.state_import(entry.state)
            except EngineError:
                # A corrupt/stale blob is never allowed to poison a request.
                # Import is transactional, then the cold path reconstructs it.
                self._drop(key)
                self.invalidations += 1
                invalidated = True
                self.engine.state_reset()
            else:
                restore_ms = (time.monotonic() - restore_started) * 1000
                self._entries.move_to_end(key)
                self.hits += 1
                return [tokens[-1]], self._report(
                    status="hit", hit=True, tokens=len(tokens),
                    reused=len(tokens) - 1, evaluated=1,
                    snapshot_bytes=len(entry.state),
                    prepare_ms=(time.monotonic() - started) * 1000,
                    restore_ms=restore_ms)

        self.misses += 1
        self.engine.prefill(tokens[:-1])
        status = "miss_invalidated" if invalidated else "miss_stored"
        snapshot_bytes = 0
        export_ms = 0.0
        try:
            size = self.engine.state_size()
            charge = len(key) + size + ENTRY_OVERHEAD
            if charge > self.capacity_bytes:
                status = "miss_too_large"
            else:
                # Evict before allocating the blob: resident prefix memory
                # never transiently exceeds the reservation.
                self._make_room(charge)
                export_started = time.monotonic()
                blob = self.engine.state_export()
                export_ms = (time.monotonic() - export_started) * 1000
                actual = len(key) + len(blob) + ENTRY_OVERHEAD
                if actual > self.capacity_bytes:
                    status = "miss_too_large"
                else:
                    self._make_room(actual)
                    self._entries[key] = _Entry(blob, len(tokens) - 1, actual)
                    self.resident_bytes += actual
                    snapshot_bytes = len(blob)
        except (EngineError, MemoryError):
            # Snapshotting is an optimisation. The already-prefilled live
            # state is valid, so continue the request through its final token.
            self.snapshot_errors += 1
            status = "miss_snapshot_error"

        return [tokens[-1]], self._report(
            status=status, hit=False, tokens=len(tokens), evaluated=len(tokens),
            snapshot_bytes=snapshot_bytes,
            prepare_ms=(time.monotonic() - started) * 1000,
            export_ms=export_ms)
