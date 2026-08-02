# Exact server prefix cache and conversation head

K3's KDA-heavy state makes a prompt checkpoint much smaller than a
conventional full KV cache. The server can keep a bounded number of those
checkpoints in process and restore one when requests share a stable prompt
family root:

```bash
python3 -m serve ~/models/k3.waste \
  --prefix-cache 2G --prefix-cache-entries 8
```

The feature is opt-in. `--prefix-cache` is both the cache's hard accounting
limit for retained cache state and the value passed to `waste_open` as
`host_reserved_bytes`. It is therefore taken out of the engine's RAM plan
before the expert cache is sized; the prefix cache and expert cache do not
discover each other through paging.
Programmatic users must likewise open `Engine(..., host_reserved_bytes=N)`
before calling `serve(..., prefix_cache_bytes=N)`. The server refuses a cache
limit larger than the engine reservation.

## What is cached

Version 1 recognizes the stable beginning of a rendered chat prompt:

- the key-normalized tool declaration (tool-list order is preserved);
- an explicit thinking-effort note; and
- consecutive leading `system` / `developer` messages.

It deliberately excludes user turns and request-tail controls such as
`tool_choice` and `response_format`. A request with no such root bypasses the
cache. Raw `/v1/completions` requests bypass it too. Image requests bypass
because equal media marker tokens do not prove equal pixel embeddings.

The renderer reports a logical boundary. The cache moves it backward to a
multiple of `waste_prefill_chunk_size(ctx)`, then stores the exact int32 token
bytes and the engine snapshot at that point. This alignment preserves the
same prefill call sizes and arithmetic order as an unsplit prompt. A lookup is
not hash-only: it compares the exact token prefix and restores the deepest
stored ancestor that does not pass the current request's stable boundary.
If that ancestor is shallower than the current stable root, the server
prefills the aligned gap and admits the deeper root before replaying the
request tail. Thus a useful shallow checkpoint does not permanently prevent
the cache from learning a higher-fanout descendant.

This is a family-root policy, not block-by-block checkpointing. It aims at the
high-fanout boundary common in agent traffic — system prompt plus tool schema —
without retaining one large recurrent-state copy at every prompt block.

## Optional mutable conversation head

For one active chat branch, the server can retain one additional exact
checkpoint near the end of the previous request prompt:

```bash
python3 -m serve ~/models/k3.waste \
  --prefix-cache 2G --prefix-cache-entries 2 --conversation-head
```

The stable family root is admitted first. The head is the deepest aligned
boundary that still leaves a non-empty suffix for ordinary generation. A next
request containing the complete prior message history can restore that head
and replay only the prior request tail, assistant turn, and new user turn. The
snapshot is still taken before generation; model output is never guessed or
implicitly carried between requests.

There is exactly one head, not a block tree. Moving forward releases the old
head before exporting its successor, so two model-sized head blobs are never
retained at once. A head cannot evict its stable family root. If the byte or
two-entry limit cannot hold both, the root remains and the head admission is
rejected. An altered prior turn fails the exact-token match and falls back to
the family root. Images, raw completions, and prompts without a stable root
retain the existing bypass behavior.

## Bounds, eviction, and identity

Both retained-cache accounting limits are hard. An enabled cache first charges
4 KiB for its controller, lock, counters, and entry list. Each slotted entry
then charges its exact snapshot bytes, exact token-key bytes, and a conservative
1 KiB metadata/allocator allowance (measured retained metadata is under 400
bytes on CPython 3.13). Admission evicts before allocating the new snapshot,
and an entry larger than the remaining byte limit is not retained. LRU ordering
uses a monotonic local counter with a deterministic creation/key tie-break.

This is not a claim that `--prefix-cache=N` makes total Python process RSS
exactly N bytes larger. Request JSON, temporary token arrays, interpreter
state, allocator fragmentation, and the rest of the HTTP server are ordinary
host working memory. The bound covers objects retained by the prefix cache,
with conservative metadata charges; operators should retain normal process
headroom in addition to the engine's hard budget.

Entries are process-local. Their identity includes the engine object, model
path and reported shape, model id, marker-token mapping, and prompt-template
version. An identity change drops every entry. The state format currently has
no model fingerprint, so the server never persists or shares these snapshots
between processes or contexts.

Import failure invalidates the entry and falls back safely to uncached
execution: the bad entry is removed, its large blob is released before
replacement export, the engine is reset, and the complete prompt is evaluated
normally. Snapshot validation or an allocation exception, including one raised
while inserting, likewise resets and evaluates the unsplit prompt. An ordinary
admission rejection retains no new entry but can safely continue from the valid
prefilled root. None of these paths may change generated tokens or logits or
turn an optional cache miss into an HTTP failure.

## Observability

Every chat response carries its request result under
`waste.prefix_cache`:

```json
{
  "status": "hit",
  "hits": 1,
  "misses": 0,
  "restored_tokens": 640,
  "replayed_tokens": 37,
  "cached_tokens": 704,
  "promoted_tokens": 64,
  "restored_kind": "head",
  "head_cached_tokens": 704
}
```

Statuses are `disabled`, `bypass_no_root`, `miss`, `hit`, `restore_failed`,
`snapshot_failed`, or `allocation_failed`. `/health` exposes cumulative
requests, hits, misses, bypasses, restored, replayed, and promoted tokens,
admissions, promotions, rejects, evictions, invalidations, restore failures,
snapshot/allocation failures, entries, controller bytes, entry bytes, and
total accounted bytes. With conversation heads enabled it also separates root
and head entries, hits, admissions, and replacements.

## Upstream dependency

This change is intentionally stacked on the transactional in-memory state
API and host-memory reservation change (`pr/in-memory-state-snapshots`, commit
`7cbb12d`). It requires `waste_state_size`, `waste_state_export`,
`waste_state_import`, and `waste_cfg.host_reserved_bytes`; it should be
reviewed or rebased only after that prerequisite. `spark/integration` already
contains both changes; this ordering describes the focused upstream proposals.

The acceptance tests compare shallow restore plus promotion, deep restore,
conversation-head restore plus tail replay, and ordinary unsplit generation
on a real synthetic container. Greedy output tokens and the entire
post-generation state blob must be byte-identical. Policy tests additionally
cover altered history, head replacement, and preserving the root when the
second snapshot does not fit.

## GN100 qualification

On the project's Acer GN100, one persistent K3 server retained a 256-token
system/tool root in 479,053,984 accounted bytes within a one-GiB reservation.
Two requests restoring that root took 249.029 and 247.828 seconds of model work
versus 857.309 seconds for the cold request, and read about 70.5% fewer bytes.
All three requests used the same total engine budget and effective direct I/O.

This is a practical combined result, not an isolated cache-only benchmark:
the expert cache warmed in the persistent process. The run's single generated
token checked cold/repeat output stability; the synthetic acceptance above is
what establishes byte-identical state and logits.
