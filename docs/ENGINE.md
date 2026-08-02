# Engine shape: embeddable library + CLI, under a hard RAM ceiling

Three scope requirements (Marco, 2026-07-27):

1. the C engine is **embeddable** — a library any host program can link;
2. it ships a **fully featured CLI** that is itself a *client* of that
   library, with no private back door;
3. the engine runs under a **configured maximum RAM**, which first requires
   knowing the **minimum RAM** the model needs at all.

## 1. Library first

The public surface is [src/waste.h](../src/waste.h): opaque `waste_ctx`,
no global state (several models can be open at once), errors returned
never printed, nothing calls `exit()`, no dependencies beyond C11 + libc.

Capability set exposed to hosts: memory planning before load, open/close,
tokenize/detokenize, `waste_generate` with a per-token callback (carrying
cache hit/miss and I/O timing so a host can draw a real progress UI),
lower-level `waste_eval` for hosts that do their own sampling, session
state save/load, model introspection and aggregate stats.

Deliberately *not* in the API: logging to stdout, signal handlers, config
files, argument parsing. Those belong to the host — the CLI included.

### Container ownership

On POSIX hosts, the first `waste_open` takes a non-blocking advisory lock on
the container directory before memory planning or model-sized allocation. A
different process opening the same container receives `WASTE_E_BUSY`; it does
not wait while both processes allocate the model and discover the collision
through memory pressure. Paths are matched by device and inode, so aliases of
one directory do not evade the check.

Contexts in one process remain independent as documented: they share a
reference-counted ownership entry, and the last `waste_close` releases it.
Failures during planning, budget validation, or partial model loading release
it as well. Lock descriptors are close-on-exec, and a forked child is treated
as a different process rather than inheriting the parent's registry.

An embedding host that deliberately accepts competing model loads can set
`waste_cfg.allow_concurrent_open`. The CLI and server expose the same opt-out
as `--allow-concurrent-open`. This is an advisory lock between cooperating
WASTE processes and depends on the filesystem's `flock` support; Windows keeps
its existing lifecycle behavior and ignores the setting.

## 2. CLI as a first-class client

`cli/` links the library and adds only host concerns: argv parsing, a
terminal renderer for the token callback, REPL/history, and file I/O.
**Rule: if the CLI needs a capability, it goes into `waste.h` first.**
That keeps the embedded path honest — anything a user can do from the
shell, a host program can do from C.

Shipped in 0.5.0, nine commands: `run`, `chat` (state kept across turns,
`/reset`, `/stats`, `/save`, `/load`, `/image`), `eval`, `tokenize`,
`detokenize`, `bench`, `plan`, `info`, `version` — the full surface is
tabulated at the end of this document, and it is still those nine in the
0.7.0-spark.1 integration.
Still to come as *subcommands*: `serve` and `convert`.

Both exist; neither is a `waste` subcommand, and for related reasons. The
OpenAI-compatible server shipped in 0.6.0 as `serve/`
(`python3 -m serve MODEL`) — stdlib-only Python reaching this same header
through ctypes, and the second client the rule above is meant to produce.
It lives there because the parts that change are the chat format and the
OpenAI schema, neither of which belongs in a C engine trying to stay small.
[SERVE.md](SERVE.md). The converter is Python for the same shape of reason:
it needs torch and safetensors, which the inference path never does.

Every one of them goes through `waste.h`. `plan` is the CLI face of
`waste_plan_memory`, `bench` of `waste_get_stats`, `run`/`chat` of
`waste_generate` and its per-token callback — the same callback an
embedding host would use to draw a progress UI.

```
$ waste plan kimi-linear.waste --budget 4G
  resident trunk              988 MB
  KDA state + KV cache        106 MB
  scratch                     178 MB
  minimum expert cache         41 MB
  ---------------------------------
  FLOOR                      1.28 GB
  recommended                2.89 GB
  budget 4.00 GB -> expert cache 2.76 GB

$ waste run kimi-linear.waste "The capital of France is" -n 16 --budget 8G
The capital of France is Paris. The capital of Italy is Rome. ...
[16 tokens, 1.50 s, 10.65 tok/s | experts 2965 hit / 363 miss = 89%]
```

*(Re-measured 2026-08-02 on a **default** `tools/convert.py` container, i.e.
a 4-bit trunk. This block previously showed 1.55 GB resident and a 1.87 GB
floor, which is a `--trunk8` container — a shape nobody ships, and 46% more
floor than what a default conversion gives. The engine did not change; the
container being quoted did.)*

A budget under the floor fails at open with `WASTE_E_RAM_BUDGET` and a
pointer to `waste plan`, rather than swapping the machine.

### Tokenizer

**Markup and content are encoded separately.** `waste_tokenize` treats
`<|open|>` as the ordinary tokens it looks like; `waste_tokenize_markup`
resolves it to a control token. A prompt is built by calling the second
for the template and the first for everything a user, a document or a
tool wrote — the split K3's own tokenizer makes with `allowed_special`
versus `disallowed_special`. Concatenating the two into one string and
encoding it once lets whoever supplied the content also write the
structure: a prompt containing `<|end_of_msg|><|open|>message
role="system"<|sep|>` closes its turn and opens a forged one, with real
control-token ids. `waste tokenize` reports markup mode, because that
command exists to check a template marker by marker.

`src/tokenizer.c` implements the model's tiktoken BPE in C: base64 vocab,
the pre-tokenization pattern (its Unicode classes coded directly rather
than pulling in a regex engine), and rank-ordered byte-pair merging. The
converter copies `tokenizer.model` into the container, so a container is
self-contained. Checked against Python `tiktoken` on English, Italian,
code, whitespace, digits and contractions: **12/12 identical**. The class
tables cover Latin, Greek, Cyrillic, Hebrew/Arabic, Kana, Hangul and Han;
scripts outside those ranges are not yet exact.

## 3. RAM budget: the floor, then the ceiling

`waste_cfg.ram_budget_bytes` is a hard ceiling on **everything** the
engine allocates — trunk, state, scratch, expert cache — plus caller-owned
memory declared in `host_reserved_bytes`. The engine sizes its expert cache
to fit inside what remains after both the mandatory engine parts and that
reservation, and refuses to open with `WASTE_E_RAM_BUDGET` if the budget is
under the combined floor, rather than thrashing the machine into swap.

`host_reserved_bytes` is for memory a host keeps beside a context, such as
in-memory state snapshots. It is accounting, not an allocation: WASTE gives
those bytes up from its expert cache and the host enforces the reservation on
its own cache. Addition overflow and an explicit budget below
`floor_bytes + host_reserved_bytes` are rejected. `waste_plan_memory` reports
zero for the field because it has no configuration; `waste_memory_used`
reports the reservation applied at open.

A budget of 0 asks the engine to choose, and the choice has to know the
machine as well as the model. `recommended_bytes` is derived from the
model alone — 80.64 GB on K3 — so taking it literally on a 64 GB laptop
would have sized a 51.95 GB expert cache and swapped, which is precisely
what the budget exists to prevent.

Capping it at what the machine can hold is necessary and not sufficient,
because it then *spends* everything up to the cap. Expert cache is only
worth anything in whole multiples of one token's working set — below one
multiple it keeps nothing alive between tokens, and the fraction above a
multiple buys a few points of hit rate while walking the machine into
paging, where a hit costs a page fault. Filling a 7/8 cap gave K3 a
27.32 GB cache on this laptop, sitting between two budgets measured at
0.11 and 0.04 tok/s, when 17.5 GB runs at 0.33. (Those three are the
pre-read-ahead sweep; the ratios between them are the point and read-ahead
does not change them — see [EFFICIENCY.md](EFFICIENCY.md).)

The default has two memory layers. Stable capacity is
`waste_usable_ram()`: physical RAM or, when smaller, the least finite cgroup-v2
`memory.max` or `memory.high` over the process and every ancestor. This fixes
the container failure in upstream 0.6.3. Linux `sysconf(_SC_PHYS_PAGES)` reads
host RAM from inside a cgroup, so without this layer a 32 GiB group on a large
host could resolve K3 at roughly 80 GB and be killed rather than merely slowed.

The Spark integration then takes one current-pressure snapshot at open.
`waste_memory_ceiling()` is the smallest of 7/8 stable usable capacity,
`MemAvailable` minus the same proportional reserve, and finite cgroup
headroom (`limit - memory.current`) minus that limit's reserve. Stable capacity
still binds when a current file is missing or malformed. These are deliberately
separate APIs: capacity describes the process's machine; the ceiling describes
what one automatic open can safely add now.

Within that ceiling the resolver steps down a whole working set at a time and
takes the largest of `floor + 3x`, `2x`, `1x`, or the floor. A host reservation
displaces expert cache inside the selected total. If it is larger than all
optional cache, the total rises only to the combined floor and only when that
floor still fits the known Linux current ceiling. K3 lands on `floor + 1x` on
the measured 64 GB Mac; an idle 128 GB GN100 still gets the full `3x`.

When the engine floor plus caller reservation is above a known Linux ceiling,
automatic sizing returns `WASTE_E_MEMORY` before a model-sized allocation. An
explicit nonzero budget remains the caller's contract and is honored, with a
warning when it exceeds current safe headroom.

### What the floor is made of

| part | scales with | can it shrink? |
|---|---|---|
| trunk (embeddings, LM head, attention, routers, shared experts, norms) | model, trunk bit-width | only by quantizing harder |
| KDA recurrent state | layers × heads × d_state² — **not** context | no |
| MLA latent KV | context length (compressed) | shorter ctx |
| scratch / activations | threads, hidden, vocab | fewer threads |
| minimum expert cache | top_k × expert record × 2 (double-buffered) | no |

Everything above the floor is expert cache, and that is the only knob
that buys speed.

> **Superseded in part (2026-08-01).** That was true of a cache which only
> fills on demand. The router lookahead fetches a layer ahead, so a record
> has to survive one attention rather than one token, and a 3.32 GB cache
> now measures 29.1% hit against 0.0% without it — within 10% of a 17.32 GB
> one on throughput. The floor is still exactly where §4 put it; it is no
> longer what limits the engine. [LEARNED.md](LEARNED.md) §41.

### Estimated for a K3-shaped config (60L, H=7168, 896 experts, top_k=16,
### experts @2.12 bit, trunk @4.25 bit, ctx 32k — `tools/memplan.py`)

> **Superseded.** This is the analytic estimate made before the weights
> dropped, kept because the reasoning is the record. K3 turned out to be
> 93 layers with a *latent* MoE at 3 bits, and the real figures are in
> [K3.md](K3.md) and [LEARNED.md](LEARNED.md) §12: the floor is 29.05 GB
> at 4K rather than 11.44, and decode is ~0.3 tok/s rather than the
> 1.75 projected below. The shape of the curve held; the level did not.
> Do not quote the numbers in this subsection.

```
trunk                     10.32 GB   (attention 6.25, routers 1.41,
                                      shared 1.29, emb+head 1.16)
KDA state (O(1) in ctx)    0.18 GB
MLA latent KV @32k         0.53 GB
scratch                    0.07 GB
min expert cache           0.35 GB   (16 x 11.1 MB x 2)
-------------------------------------------------
RAM FLOOR                 11.44 GB
```

**So the model can technically run in ~12 GB of RAM** — it just reads
~10.3 GB from disk per token. RAM above the floor converts to hit rate:

| budget | expert cache | cache fraction | hit rate* | GB read/token | tok/s* |
|---|---|---|---|---|---|
| 16 GB | 4.9 GB | 0.9% | 3% | 9.9 | 1.29 |
| 32 GB | 20.9 GB | 3.6% | 14% | 8.8 | 1.45 |
| **64 GB** | **52.9 GB** | **9.2%** | **29%** | **7.3** | **1.75** |
| 128 GB | 116.9 GB | 20.3% | 46% | 5.5 | 2.32 |

\* hit rate interpolated from the Gate 0 OLMoE curve; tok/s counts disk
I/O only (12.78 GB/s internal SSD, Gate H) and ignores compute. Both get
replaced by measurements at Gate 2.

Two things this changes:

- **64 GB is not a cliff, it is a point on a gentle curve.** Going from
  16 GB to 128 GB moves throughput less than 2×, because with 575 GB of
  experts even 117 GB of cache holds only a fifth of them. The dominant
  lever is not RAM, it is *bytes read per token* — i.e. bit-width and
  expert pruning.
- Numbers assume the analytic trunk estimate. Gate 1 replaces it with
  exact tensor sizes from the shard headers.

## Chunked prefill

Decoding is one token at a time by nature, but prefill is not, and for a
streaming engine the difference is not compute — it is disk. Tokens in a
chunk route to *overlapping* expert sets, so the number of distinct
experts is far below `n_tokens * top_k`, and each one need only be read
once.

`waste_model_prefill()` processes up to 64 tokens together. Measured on a
32-token prompt against the same prompt fed one token at a time:

| | expert reads | bytes read | time |
|---|---|---|---|
| one token at a time | 6656 | 16.53 GB | 5.03 s |
| **chunked** | **2032** | **5.05 GB** | **3.85 s** |

**3.3x fewer reads**, identical logits (max abs difference 6.7e-06 on
values of magnitude ~18, i.e. float noise). On a machine where the whole
container fits in page cache this shows up as a modest 1.3x; on K3, where
the experts do not fit in anything, it is the difference between reading
17 GB per prompt token and reading 5.

One design note worth recording, because the first attempt got it wrong.
The obvious approach is to expand each expert once and run GEMMs, on the
theory that the cost amortizes over the chunk. Measured, it does not: 16
tokens spread across ~1200 distinct experts, under 3 tokens each, so
expanding 7 M weights to serve 3 vectors is far worse than the LUT. The
version that shipped keeps the decode-style LUT arithmetic and reorganizes
*only* to read each expert once — gate/up tables depend on the token but
not the expert, so they are built once per token and reused across every
expert that token routes to.

## Session state

`waste_state_save` / `waste_state_load` persist the whole session: KDA
recurrent state, the short-conv rings, the MLA KV cache, the AttnRes block
history and the position. The file records every shape it depends on, so
a state built for a different model is rejected with `WASTE_E_FORMAT`
rather than silently producing nonsense.

`waste_state_size` / `waste_state_export` / `waste_state_import` expose the
same versioned bytes without a filesystem round trip. Size is exact for the
live prefix (MLA contributes only populated latent rows), a short export does
not touch its destination, and import preflights the complete blob before it
modifies the context. Structurally corrupt and truncated blobs therefore
leave the live conversation unchanged. (Version 1 has no payload checksum.)
This is the primitive a server can use for an exact prefix cache; cache policy
and prefix matching deliberately remain host concerns.

`waste_prefill_chunk_size` reports the grouping used by `waste_eval` and
`waste_generate`. A host that splits a prompt around a checkpoint must split
on a multiple of this value to preserve the unsplit call boundaries and
floating-point arithmetic order exactly.

Version 1 validates layout and bounds but has no model/manifest identity
fingerprint. Snapshots are opaque, context/model-local values. A host must not
reuse one across containers merely because their shapes match; durable or
cross-model caches need an identity digest in a later format version.

This matters more here than in a conventional engine. At K3's streaming
speeds re-prefilling a long agent transcript is minutes; restoring it is a
file read. `waste chat` exposes it as `/save FILE` and `/load FILE`, and
`tests/test_state.c` asserts that restore and replay reproduce logits and the
complete post-step state bit-for-bit, that malformed imports are
transactional, and that file and memory snapshots use identical bytes.

## Learned hotlist

The cache records which experts a workload actually touches, and
`--learn` writes that to `usage.waste` beside the container. The next open
preloads the hottest ones, so a run starts warm instead of empty.

Measured on Kimi-Linear, same prompt twice at a 5 GB budget:

| | expert misses | hit rate |
|---|---|---|
| cold | 1602 | 61% |
| warm | 1175 | 72% |

This does not move the floor Gate 5 established — a cache below one
token's working set still keeps nothing alive — it removes the ramp at the
start of every run.

## Verification bar

The CLI must never be able to do something the library cannot, and
`waste_plan_memory` must agree with `waste_memory_used` after open. Any
allocation path that can exceed the budget is a bug, not a tuning issue.

`make check` runs everything: kernels against their reference
implementations, the shard downloader's resume and skip paths against a
local server, the image loader, the container round-trip and its damaged-
record paths, chunked prefill against token-at-a-time, int8 storage against
f32, the SIMD backend against the CPU baseline, the cache against no cache,
read-ahead against synchronous reads, the router lookahead against no
lookahead, purgeable slots against ordinary ones, the trace simulator
against the engine's own cache, the engine against the PyTorch oracle,
session round-trip, hotlist effect, budget enforcement including peak RSS
on both models, the derived `info` and parameter counts, the converter's
resume, image normalization against the release's own preprocessor config,
the markup/content split against a forged control token, the tokenizer
against Python tiktoken, and the server suite.

Four of those are bit-identity checks — cache, read-ahead, lookahead,
purgeable — and they are the reason each of those mechanisms could ship:
every one changes *when* bytes move and none may change what comes out.

**45 checks** as of 2026-08-02. With both containers on disk: **44 pass, 0
fail, 1 SKIP** — the one skip is image normalization against the release,
which Kimi-Linear has no vision tower for. With no container at all the same
run is **33 pass / 0 fail / 12 SKIP**: the synthetic container carries the
engine checks, and everything needing real weights or a tokenizer says SKIP
rather than passing quietly. Take the numbers from a run, not from here:
they move as checks are added.

It exists because this project twice lost hours to checks that silently
did not run — once to objects compiled against a stale header, once to a
stale test binary. So it rebuilds first, and a missing prerequisite is
reported as SKIP, never as a pass.


## CLI surface (2026-08-02)

Nine commands. This table maps the CLI-facing library calls; it is not a
claim that every embedding API needs a shell command. The server also uses
`waste_state_size/export/import` for in-memory prefix snapshots and
`waste_prefill_chunk_size` for safe chunk boundaries. Version, API and
structure-size functions exist for host compatibility checks.

| command | uses |
|---|---|
| `run` | `waste_generate`, `waste_tokenize`(+`_markup`), `waste_save_usage` |
| `chat` | plus `waste_state_save/load/reset` |
| `eval` | `waste_eval`, `waste_detokenize` |
| `tokenize` / `detokenize` | `waste_tokenize`, `waste_detokenize` |
| `bench` | `waste_get_stats` |
| `plan` | `waste_plan_memory`, `waste_physical_ram`, `waste_usable_ram`, `waste_memory_ceiling` |
| `info` | `waste_model_get_info`, `waste_memory_used` |

Images add four more, reachable from `run`, `chat` and `eval` via
`--image`: `waste_image_add`, `waste_image_dimensions`,
`waste_image_expand`, `waste_image_clear`.

`eval` is the one worth knowing about: it runs the prompt and prints the
next-token distribution without generating, which is how you get a logit
or a log-probability out of the engine.

The prompt is an argument, a file (`--file`), or stdin — either as `-` or
simply piped, so `echo hi | waste run M` does the obvious thing. `--json`
makes `eval`, `tokenize`, `plan`, `info` and `bench` machine-readable.
`--stop STR` ends generation when the text appears, matched against a
rolling tail so a stop string split across two tokens still fires.

**The parser used to be wrong in a way that produced no error.** The
prompt was `argv[3]` unconditionally and options were only looked for
after it, so `waste run M --temp 0 "hi"` generated from the string
"--temp", `waste run M -n 3` generated from "-n", and a second positional
was dropped silently. Positionals are collected properly now, an
unexpected one is an error, a value that looks like an option is a
missing value, and the sampling parameters are range-checked instead of
producing empty output when set to nonsense.
