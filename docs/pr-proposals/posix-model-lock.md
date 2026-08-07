# Prevent competing processes from opening one container

Candidate commit: `293d06e4db23b4674ce3431ccc47c67218253425`

## Problem

Two WASTE processes can open the same large container and each begin
model-sized allocation before either discovers the resulting memory pressure.
On a unified-memory workstation that failure mode is both slow and disruptive.
Multiple contexts deliberately opened by one embedding process must continue
to work.

## Change

- Take a non-blocking advisory POSIX `flock` on the container directory before
  planning or model-sized allocation.
- Key an in-process registry by device/inode so aliases share one
  reference-counted ownership entry.
- Return `WASTE_E_BUSY` to a competing process while allowing multiple contexts
  in the owner process.
- Release ownership after the last close and on every planning, budget, and
  partial-load error path.
- Close descriptors across exec and clear inherited registry state after
  fork.
- Add `waste_cfg.allow_concurrent_open` and matching CLI/server opt-outs for
  hosts that deliberately accept competing loads.
- Leave Windows lifecycle behavior unchanged.

The mechanism is advisory and depends on filesystem `flock` support.  It
coordinates cooperating WASTE processes; it is not a filesystem lease or a
security boundary.

## Correctness

- A same-process second context succeeds and keeps ownership after the first
  context closes.
- An exec'd competing process receives `WASTE_E_BUSY` before budget handling.
- The explicit opt-out reaches the ordinary budget result instead of `busy`.
- Normal close, under-budget refusal, malformed planning input, and partial
  model-load failure all release the raw OS lock.
- The directory descriptor is close-on-exec and fork registry handling is
  covered by the process tests.

## Performance and safety evidence

- Portable suite: 29 passed, 0 failed, 12 skipped; server: 167 passed.
- ASan/UBSan suite: 28 passed, 0 failed, 13 skipped.
- The combined integration branch on the GN100 passed 28 Linux ARM64 native
  engine checks and 168 server checks; skips were limited to missing `uv` and
  external model/reference fixtures.
- The lock is acquired once per context lifetime and is not present in the
  token path; no throughput claim is needed.

## Compatibility

No container, arithmetic, routing, state, or I/O format changes.  The new
status and appended `waste_cfg` field require binary clients to rebuild; the
in-tree CLI and Python `ctypes` layout change atomically.  Windows ignores the
new opt-out because the ownership mechanism is POSIX-only.

## Rollback

`--allow-concurrent-open` disables the behavior for an intentional multi-process
host.  Reverting the single candidate commit removes the lock and public status.
