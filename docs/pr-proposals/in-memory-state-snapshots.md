# Add transactional in-memory state snapshots

Candidate commit: `7cbb12def90922dc378768be711092c242e2eab3`

## Problem

Kimi K3 is KDA-heavy and interleaves compressed MLA state, so a useful prompt
checkpoint is comparatively compact.  The existing state API nevertheless
requires a filesystem round trip, and a server retaining snapshots has no way
to account that caller-owned memory against the engine's hard RAM ceiling.
That prevents a correct, budgeted prefix cache from being built above the
public API.

## Change

- Add `waste_state_size`, `waste_state_export`, and `waste_state_import` using
  the existing versioned state bytes in caller-owned memory.
- Size only live MLA rows rather than the full context allocation.
- Preflight the complete header, dimensions, per-layer lengths, position, and
  trailing length before replacing any live state.
- Report the required size without partially writing a short export buffer.
- Add generic `waste_cfg.host_reserved_bytes` accounting so retained host
  caches displace expert cache inside the same hard budget.
- Report that reservation through `waste_memory_used` and mirror both the API
  and struct additions in the Python binding.

This candidate deliberately provides a primitive rather than a cache policy.
Exact-prefix matching, admission, eviction, and model-identity keying remain
server responsibilities.

## Correctness

- Restore plus replay produces bit-identical next-token logits and an
  identical complete post-step state, covering the deterministic routes that
  produced it.
- Memory and file snapshots are byte-identical.
- Truncated and structurally corrupt imports leave the live context unchanged.
- A short export reports the exact required size and leaves its destination
  untouched.
- Reservation tests cover expert-cache displacement, explicit under-floor
  refusal, and unsigned-addition overflow.

## Performance and safety evidence

- Portable suite: 28 passed, 0 failed, 12 skipped.
- ASan/UBSan suite: 27 passed, 0 failed, 13 skipped.
- Focused Python binding tests: 21 passed.
- GN100 Linux ARM64 native acceptance: the C state test and all 48 Python
  binding tests passed.
- The combined integration branch, after resolving interactions with usable-
  memory budgeting and model ownership, passed 30 portable checks and all 173
  server checks.  That merge additionally refuses an automatic reservation
  that would cross Linux's live-memory/cgroup ceiling and releases ownership
  on every new early-return path.

The API copy is not performance-sensitive.  Its purpose is to enable a
separate measured prefix-cache change without filesystem I/O.

## Compatibility

The snapshot bytes remain state format version 1; file save/load behavior and
model arithmetic do not change.  Appended public struct fields and new symbols
require binary clients to rebuild, and the in-tree `ctypes` layout changes
atomically.  Version 1 has no model/manifest fingerprint or payload checksum:
snapshots are opaque, context/model-local values.  Structural corruption is
rejected, but arbitrary payload bit flips are not authenticated.

## Rollback

Hosts that do not call the new functions or set `host_reserved_bytes` retain
the previous behavior.  Reverting the single candidate commit removes the API
and accounting fields.
