# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Bounded, process-local exact-prefix state cache.

The cache owns policy and bytes; libwaste owns state validation. An entry is
admitted only at a caller-proven family-root boundary aligned to the engine's
prefill chunk size. That alignment is what makes a cold split execute the
same model calls as an unsplit prompt, rather than merely producing close
floating-point values.

Keys carry the exact int32 token bytes and the cache is tied to one explicit
process-local identity. There are no hash-only matches and no persistence.
"""

from __future__ import annotations

import threading
from array import array
from dataclasses import dataclass
from typing import Any, Iterable

from .engine import EngineError, WASTE_E_IO

_TOKEN_BYTES = array("i").itemsize
if _TOKEN_BYTES != 4:  # waste tokens are public int32_t values
    raise RuntimeError("Python array('i') is not 32 bits on this platform")

# Blob and key lengths are exact. These conservative charges cover the
# retained Python objects around them too. Measured on CPython 3.13, an entry
# with slots retains under 400 bytes beyond its blob/key payloads; 1 KiB leaves
# room for allocator rounding and other supported CPython versions. The 4 KiB
# controller charge covers the cache object, lock, counters and entry list.
# Request-local token arrays are ordinary server working memory, not retained
# cache state; see docs/PREFIX_CACHE.md for the accounting boundary.
CONTROLLER_OVERHEAD_BYTES = 4096
ENTRY_OVERHEAD_BYTES = 1024


def _pack(tokens: Iterable[int]) -> bytes:
    return array("i", tokens).tobytes()


@dataclass(slots=True)
class CacheUse:
    """What this request did; emitted in its response."""

    status: str
    restored_tokens: int
    replayed_tokens: int
    cached_tokens: int = 0
    promoted_tokens: int = 0
    restored_kind: str | None = None
    head_cached_tokens: int = 0

    def response(self) -> dict[str, Any]:
        hit = self.status == "hit"
        miss = self.status in (
            "miss", "restore_failed", "snapshot_failed", "allocation_failed")
        return {
            "status": self.status,
            "hits": 1 if hit else 0,
            "misses": 1 if miss else 0,
            "restored_tokens": self.restored_tokens,
            "replayed_tokens": self.replayed_tokens,
            "cached_tokens": self.cached_tokens,
            "promoted_tokens": self.promoted_tokens,
            "restored_kind": self.restored_kind,
            "head_cached_tokens": self.head_cached_tokens,
        }


@dataclass(slots=True, eq=False)
class _Entry:
    token_bytes: bytes
    n_tokens: int
    kind: str
    blob: bytearray
    charge: int
    created: int
    last_used: int


class PrefixCache:
    """Deterministic LRU over exact roots and one optional mutable head."""

    def __init__(self, max_bytes: int, max_entries: int, identity: Any,
                 *, conversation_head: bool = False):
        if max_bytes < 0 or max_entries < 0:
            raise ValueError("prefix cache limits must be non-negative")
        if (max_bytes > 0 and max_entries > 0 and
                max_bytes < CONTROLLER_OVERHEAD_BYTES):
            raise ValueError(
                f"an enabled prefix cache needs at least "
                f"{CONTROLLER_OVERHEAD_BYTES} bytes for its controller")
        self.max_bytes = int(max_bytes)
        self.max_entries = int(max_entries)
        self.conversation_head = bool(conversation_head)
        self._controller_bytes = (CONTROLLER_OVERHEAD_BYTES
                                  if max_bytes > 0 and max_entries > 0 else 0)
        self._identity = identity
        self._entries: list[_Entry] = []
        self._bytes_used = self._controller_bytes
        self._clock = 0
        self._lock = threading.RLock()
        self._counters = {
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "bypasses": 0,
            "restores": 0,
            "root_hits": 0,
            "head_hits": 0,
            "restore_failures": 0,
            "restored_tokens": 0,
            "replayed_tokens": 0,
            "admissions": 0,
            "root_admissions": 0,
            "head_admissions": 0,
            "head_replacements": 0,
            "promotions": 0,
            "promoted_tokens": 0,
            "admission_rejects": 0,
            "snapshot_failures": 0,
            "allocation_failures": 0,
            "evictions": 0,
            "invalidations": 0,
        }

    @property
    def enabled(self) -> bool:
        return self.max_bytes > 0 and self.max_entries > 0

    def ensure_identity(self, identity: Any) -> None:
        """Drop every entry before accepting a different runtime identity."""
        with self._lock:
            if identity == self._identity:
                return
            for entry in self._entries:
                entry.blob.clear()
            self._entries.clear()
            self._bytes_used = self._controller_bytes
            self._identity = identity
            self._counters["invalidations"] += 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "max_bytes": self.max_bytes,
                "max_entries": self.max_entries,
                "bytes_used": self._bytes_used,
                "controller_bytes": self._controller_bytes,
                "entry_bytes_used": self._bytes_used - self._controller_bytes,
                "entries": len(self._entries),
                "root_entries": sum(e.kind == "root" for e in self._entries),
                "head_entries": sum(e.kind == "head" for e in self._entries),
                "conversation_head": self.conversation_head,
                **self._counters,
            }

    def _record_use(self, use: CacheUse) -> CacheUse:
        with self._lock:
            self._counters["requests"] += 1
            self._counters["replayed_tokens"] += use.replayed_tokens
            if use.status == "hit":
                self._counters["hits"] += 1
                self._counters["restores"] += 1
                self._counters["restored_tokens"] += use.restored_tokens
                if use.restored_kind == "head":
                    self._counters["head_hits"] += 1
                else:
                    self._counters["root_hits"] += 1
                if use.promoted_tokens:
                    self._counters["promotions"] += 1
                    self._counters["promoted_tokens"] += use.promoted_tokens
            elif use.status in (
                    "miss", "restore_failed", "snapshot_failed",
                    "allocation_failed"):
                self._counters["misses"] += 1
            else:
                self._counters["bypasses"] += 1
        return use

    def _remove(self, entry: _Entry) -> None:
        self._entries.remove(entry)
        self._bytes_used -= entry.charge
        # A caller may still have a local reference to an evicted entry.
        # Clear the bytearray here so that reference cannot pin a K3-sized
        # allocation while its replacement snapshot is exported.
        entry.blob.clear()

    def _evict_one(self) -> None:
        victim = min(self._entries,
                     key=lambda e: (e.last_used, e.created, e.token_bytes))
        self._remove(victim)
        self._counters["evictions"] += 1

    def _lookup(self, prompt_bytes: bytes, prompt_tokens: int,
                max_tokens: int) -> _Entry | None:
        with self._lock:
            best = None
            for candidate in self._entries:
                if (candidate.n_tokens <= max_tokens and
                        candidate.n_tokens < prompt_tokens and
                        prompt_bytes.startswith(candidate.token_bytes) and
                        (best is None or
                         (candidate.n_tokens, -candidate.created) >
                         (best.n_tokens, -best.created))):
                    best = candidate
            return best

    def _mark_restored(self, entry: _Entry) -> None:
        with self._lock:
            self._clock += 1
            entry.last_used = self._clock

    def _drop_failed_restore(self, entry: _Entry) -> None:
        with self._lock:
            if entry in self._entries:
                self._remove(entry)
            else:
                entry.blob.clear()
            self._counters["restore_failures"] += 1

    def _reserve_admission(self, key: bytes, snapshot_bytes: int,
                           kind: str) -> int:
        charge = snapshot_bytes + len(key) + ENTRY_OVERHEAD_BYTES
        with self._lock:
            # Replacement never holds two copies of the same boundary. A
            # mutable head is singular by design: release the old branch
            # before allocating its successor's model-sized blob.
            for entry in list(self._entries):
                if (entry.kind == kind and entry.token_bytes == key) or (
                        kind == "head" and entry.kind == "head"):
                    self._remove(entry)
                    if kind == "head":
                        self._counters["head_replacements"] += 1
                    break
            if (charge > self.max_bytes or self.max_entries < 1):
                self._counters["admission_rejects"] += 1
                return 0
            # A head is an acceleration of one branch, never a reason to
            # discard the stable family root that makes divergent requests
            # cheap and safe. If both do not fit, retain the root.
            if kind == "head" and (
                    self._bytes_used + charge > self.max_bytes or
                    len(self._entries) >= self.max_entries):
                self._counters["admission_rejects"] += 1
                return 0
            while (self._entries and
                   (self._bytes_used + charge > self.max_bytes or
                    len(self._entries) >= self.max_entries)):
                heads = [e for e in self._entries if e.kind == "head"]
                if heads:
                    self._remove(min(heads,
                                     key=lambda e: (e.last_used, e.created,
                                                    e.token_bytes)))
                    self._counters["evictions"] += 1
                else:
                    self._evict_one()
            if (self._bytes_used + charge > self.max_bytes or
                    len(self._entries) >= self.max_entries):
                self._counters["admission_rejects"] += 1
                return 0
            return charge

    def _insert(self, key: bytes, n_tokens: int, kind: str, blob: bytearray,
                charge: int) -> bool:
        with self._lock:
            # prepare() is serialized by the engine lock. Keep the bound
            # defensive anyway, in case a different host calls this class.
            while (self._entries and
                   (self._bytes_used + charge > self.max_bytes or
                    len(self._entries) >= self.max_entries)):
                self._evict_one()
            if (self._bytes_used + charge > self.max_bytes or
                    len(self._entries) >= self.max_entries):
                self._counters["admission_rejects"] += 1
                return False
            # Prepare every allocating value before append. list.append is
            # atomic on allocation failure; after it succeeds, only pointer
            # assignments remain, so accounting cannot be left half-updated.
            new_clock = self._clock + 1
            new_bytes = self._bytes_used + charge
            new_admissions = self._counters["admissions"] + 1
            new_entry = _Entry(
                key, n_tokens, kind, blob, charge, new_clock, new_clock)
            self._entries.append(new_entry)
            self._clock = new_clock
            self._bytes_used = new_bytes
            self._counters["admissions"] = new_admissions
            self._counters[f"{kind}_admissions"] += 1
            return True

    def _has_entry(self, key: bytes, kind: str) -> bool:
        with self._lock:
            return any(e.kind == kind and e.token_bytes == key
                       for e in self._entries)

    def _retained_tokens(self, entry: _Entry | None) -> int:
        if entry is None:
            return 0
        with self._lock:
            return entry.n_tokens if entry in self._entries else 0

    def _prepare_enabled(self, engine, tokens: list[int],
                         boundaries: Iterable[int]
                         ) -> tuple[list[int], CacheUse]:
        """Allocation-fallible cache path, isolated for clean unwinding."""
        try:
            chunk = int(engine.prefill_chunk_size())
        except EngineError:
            chunk = 0
        aligned = sorted({(int(b) // chunk) * chunk for b in boundaries
                          if chunk > 0 and isinstance(b, int)})
        aligned = [b for b in aligned if 0 < b < len(tokens)]
        if not aligned:
            use = CacheUse("bypass_no_root", 0, len(tokens))
            return tokens, self._record_use(use)

        root_tokens = aligned[-1]
        # Keep generate() a non-empty call. If the prompt length itself is a
        # chunk multiple, the previous boundary is the deepest state that can
        # be restored while preserving the ordinary chunked-prefill sequence.
        head_tokens = ((len(tokens) - 1) // chunk) * chunk
        if not self.conversation_head or head_tokens <= root_tokens:
            head_tokens = root_tokens
        targets = [("root", root_tokens)]
        if head_tokens > root_tokens:
            targets.append(("head", head_tokens))

        prompt_bytes = _pack(tokens)
        entry = self._lookup(prompt_bytes, len(tokens), head_tokens)
        restore_failed = False
        restored_tokens = 0
        restored_kind = None
        if entry is not None:
            try:
                engine.state_import(entry.blob)
            except EngineError:
                # Import is transactional, and resetting again makes the
                # fallback independent even if a future engine regresses.
                engine.state_reset()
                self._drop_failed_restore(entry)
                entry = None       # release the old blob before replacement
                restore_failed = True
            else:
                self._mark_restored(entry)
                restored_tokens = entry.n_tokens
                restored_kind = entry.kind

        # A shallow exact ancestor is useful, but not terminal. Evaluate each
        # aligned gap in the same chunk order as an unsplit prompt. The stable
        # family root is retained first; an optional singular conversation
        # head may then cover the exact prior turns without becoming a block
        # tree or a hash-only prefix match.
        start = restored_tokens
        cached_tokens = self._retained_tokens(entry)
        head_cached_tokens = (cached_tokens
                              if entry is not None and entry.kind == "head"
                              else 0)
        try:
            root_key = prompt_bytes[:root_tokens * _TOKEN_BYTES]
            for kind, target in targets:
                if target <= start:
                    continue
                engine.prefill(tokens[start:target])
                start = target

                # A mutable head accelerates one exact branch but never exists
                # at the expense of the stable family root.
                if kind == "head" and not self._has_entry(root_key, "root"):
                    continue

                key = prompt_bytes[:target * _TOKEN_BYTES]
                snapshot_bytes = int(engine.state_size())
                if snapshot_bytes <= 0:
                    raise EngineError("state_size", WASTE_E_IO,
                                      "snapshot size is not positive")
                charge = self._reserve_admission(key, snapshot_bytes, kind)
                # Reservation may evict the restored shallow ancestor or old
                # mutable head. Drop the local reference before allocating a
                # replacement model-sized blob.
                cached_tokens = max(cached_tokens,
                                    self._retained_tokens(entry))
                entry = None
                if not charge:
                    continue
                blob = engine.state_export()
                if len(blob) != snapshot_bytes:
                    raise EngineError("state_export", WASTE_E_IO,
                                      "snapshot size changed")
                if self._insert(key, target, kind, blob, charge):
                    cached_tokens = max(cached_tokens, target)
                    if kind == "head":
                        head_cached_tokens = target
        except EngineError:
            # We cannot prove the checkpoint is sound. Re-evaluate from a
            # clean state rather than continuing from an uncertain prefix.
            engine.state_reset()
            with self._lock:
                self._counters["snapshot_failures"] += 1
            use = CacheUse("snapshot_failed", 0, len(tokens))
            return tokens, self._record_use(use)

        if restored_tokens:
            promoted = (cached_tokens - restored_tokens
                        if cached_tokens > restored_tokens else 0)
            use = CacheUse("hit", restored_tokens,
                           len(tokens) - restored_tokens,
                           cached_tokens=cached_tokens,
                           promoted_tokens=promoted,
                           restored_kind=restored_kind,
                           head_cached_tokens=head_cached_tokens)
        else:
            status = "restore_failed" if restore_failed else "miss"
            use = CacheUse(status, 0, len(tokens),
                           cached_tokens=cached_tokens,
                           head_cached_tokens=head_cached_tokens)
        return tokens[start:], self._record_use(use)

    def prepare(self, engine, tokens: list[int], boundaries: Iterable[int],
                identity: Any) -> tuple[list[int], CacheUse]:
        """Restore the deepest ancestor or build one cold family root.

        The caller must hold the engine lock and reset its state first.
        Returned tokens are the exact unmatched suffix to hand to generate.
        """
        self.ensure_identity(identity)
        if not self.enabled:
            use = CacheUse("disabled", 0, len(tokens))
            return tokens, self._record_use(use)
        if not tokens:
            use = CacheUse("bypass_no_root", 0, 0)
            return tokens, self._record_use(use)

        try:
            return self._prepare_enabled(engine, tokens, boundaries)
        except (MemoryError, OverflowError):
            # Snapshot allocation can be hundreds of MiB on K3. Allocation
            # failure is a cache miss, not a partially-prefilled HTTP failure:
            # unwind the helper frame (dropping its large locals), reset, and
            # let generate evaluate the original unsplit prompt.
            engine.state_reset()
            with self._lock:
                self._counters["allocation_failures"] += 1
            use = CacheUse("allocation_failed", 0, len(tokens))
            return tokens, self._record_use(use)
