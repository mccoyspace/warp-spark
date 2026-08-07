# Add an exact, budgeted server prefix cache

Implementation commit:
`2d6c74daa946681d388c7d00bd03285047872619`

Candidate status: implementation-complete, independently reviewed, and exercised
with real K3 on this project's Acer GN100. The measurement below is one
persistent-server acceptance sequence, not a general performance claim.

## Problem

Recurring agent requests often repeat a large system prompt and tool schema.
Without this opt-in cache, the server evaluates that shared prefix from scratch
on every request.
Kimi K3 is a particularly useful target for exact reuse: its KDA-heavy recurrent
state is independent of prompt length, while its interleaved MLA layers retain
compressed latent KV rather than a conventional full KV cache.

A correct cache cannot simply retain snapshots outside the engine's accounting.
Doing so would let prefix state and the expert cache compete only after the OS
begins reclaiming memory. It also cannot split a prompt at an arbitrary logical
boundary: changing the native prefill call sizes can change floating-point
arithmetic order.

## Dependency

This candidate is intentionally stacked on `pr/in-memory-state-snapshots`,
implementation commit `6fba880cad7accce8d06828f88b423a6a62a72e4`. It uses:

- `waste_state_size`, `waste_state_export`, and transactional
  `waste_state_import`;
- `waste_cfg.host_reserved_bytes`, so caller-retained snapshots displace expert
  cache inside the same hard RAM budget; and
- the existing opaque, context/model-local state format.

The dependency should land first. Rebase and rerun this proposal's acceptance
set if the state API or public struct layout changes during review.

## Change

- Add an opt-in, process-local exact-prefix cache to the OpenAI-compatible
  server, configured with `--prefix-cache SIZE` and
  `--prefix-cache-entries N`.
- Recognize a stable family root consisting of key-normalized tool
  declarations, an explicit thinking-effort note, and consecutive leading
  system/developer messages. User turns and request-tail controls remain outside
  the root.
- Bypass image requests because identical media marker tokens do not prove
  identical pixel embeddings; raw completion requests also bypass.
- Add `waste_prefill_chunk_size` and move the renderer-proven root backward to
  a native chunk boundary. Cold, promoted, and restored paths therefore retain
  the unsplit prefill call sizes and arithmetic order.
- Match exact int32 token bytes rather than hashes, restore the deepest eligible
  ancestor, replay an aligned gap, and promote it to the deeper stable root.
- Use deterministic LRU admission and expose per-request results plus cumulative
  health counters for hits, misses, restores, replay, promotions, failures,
  evictions, entries, and accounted bytes.
- Treat snapshot/import/allocation failures as cache misses: clear an invalid or
  evicted large blob before replacement export, reset the engine, and evaluate
  the original unsplit prompt.

## Correctness

- Exact matching never restores past the current request's renderer-proven
  stable boundary.
- A shallow hit is useful immediately and also admits the deeper root; the next
  matching request restores that deeper snapshot.
- Tool-schema-only roots and leading developer messages are cacheable, while
  image and rootless requests bypass.
- Concurrent HTTP requests with different roots remain isolated by the existing
  per-context engine lock, which spans prompt construction, cache preparation,
  and generation.
- Restore, export, insertion-allocation, overflow, and snapshot-validation
  failures unwind to an unsplit request rather than corrupting live state or
  turning an optional cache miss into an HTTP failure.
- Synthetic cold admission, shallow restore plus promotion, and deep restore
  produce byte-identical greedy output and complete post-generation state.

## Memory accounting

`--prefix-cache=N` is passed to the engine as `host_reserved_bytes=N` before it
sizes the expert cache. The server then refuses a retained-cache limit larger
than that reservation.

Within the cache, an enabled instance charges 4 KiB for its controller, lock,
counters, and entry list. Each slotted entry charges the exact snapshot length,
the exact token-key length, and a conservative 1 KiB metadata and allocator
allowance. Admission evicts before allocating a replacement snapshot, and
failed restore or one-entry promotion clears the old bytearray before export so
a stale local reference cannot pin a second K3-sized blob.

This is a hard bound on objects retained by the cache, not a promise that total
Python RSS grows by exactly N bytes. Request JSON, temporary token arrays,
interpreter state, and allocator fragmentation remain ordinary process working
memory and still require operating headroom. The metadata measurements and
tests are CPython-specific.

## Validation evidence

- Implementation commit `39a2334` received a fresh independent review after
  the failure, promotion, and accounting fixes; no actionable findings remained.
- Focused `39a2334` local server suite: 201 tests, OK (2 skipped).
- Focused `39a2334` GN100 Linux ARM64 native suite: 26 passed, 0 failed,
  13 skipped.
- Focused `39a2334` GN100 server suite: 201 tests, OK (2 skipped).
- The real synthetic-container acceptance test compares uncached generation,
  shallow restore plus promotion, and subsequent deep restore. Greedy output
  bytes and the complete post-generation state blob are identical on every
  path.
- Real K3 acceptance on integrated commit `5ba76d2` produced the intended
  miss-hit-hit sequence: one exact family root was admitted cold, then restored
  for both a divergent user tail and an exact repeated request.
- In that practical persistent-server sequence, restored requests were
  3.44-3.46x faster by request wall time and read about 70.5% fewer expert bytes
  than the cold request.

Skip counts are reported rather than treated as passes; they cover optional
environment-dependent checks, including unavailable external model/reference
fixtures. The K3 timing and I/O deltas are not an isolated causal estimate of
prefix caching: all three calls used one persistent process, so the expert cache
warmed alongside prefix-state reuse. Relative to the measured tree, final
integration code `9b5bfdb` changes only documentation, tests, and the PM-QoS
helper; the prefix-cache implementation measured here is unchanged.

## Compatibility and scope

The cache is disabled by default. Existing server behavior is unchanged unless
`--prefix-cache` is nonzero. Snapshot bytes remain state format version 1 and
are never persisted or shared across processes; that format has no model
fingerprint, so the cache identity includes the engine object, model path and
shape, model id, marker-token map, and prompt-template version.

The new public query symbol and the prerequisite's appended public struct field
require pre-1.0 binary clients to rebuild; the in-tree Python binding changes
atomically. This proposal is a family-root policy, not block-by-block KV caching,
longest-common-prefix indexing, or a claim that every workload will benefit.

## Rollback

Omitting `--prefix-cache` keeps the feature disabled. Reverting the candidate
commit removes the cache policy and `waste_prefill_chunk_size`; reverting the
snapshot prerequisite separately removes the underlying state and reservation
API.
