# serve — the OpenAI-compatible server

```bash
make libwaste.dylib          # or libwaste.so on Linux
python3 -m serve ~/models/k3.waste --port 8000
```

```bash
curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"k3","messages":[{"role":"user","content":"Why is the sky blue?"}]}'
```

Stdlib only. No package index, no virtualenv, no framework: a server that
needs a dependency resolver to start is one more thing between a downloaded
model and an answer.

## Why Python, and why ctypes

`waste.h` opens by saying the engine is a library first and the CLI is one
of its clients. This is the second client. It does not reimplement any
inference — every model operation is a call into `libwaste` through ctypes,
and `serve/engine.py` mirrors the header struct for struct.

What is left for Python is everything that is *not* arithmetic: K3's prompt
format, the parser that reads its replies back, request validation, SSE
framing. That code changes with the OpenAI API and with each model's chat
format, neither of which belongs in a C engine that is trying to stay small
and dependency-free.

## What the model actually needs

Kimi K3 ships **no Jinja template**. It builds prompts with a Python
program, `encoding_k3.py`, which emits a token sequence directly in XTML —
an XML-like markup whose angle brackets are reserved tokens:

| in this repo | token | role |
|---|---|---|
| `[open]` | `<\|open\|>` | starts a tag |
| `[sep]` | `<\|sep\|>` | ends a tag header |
| `[close]` | `<\|close\|>` | starts a closing tag |
| `[end_of_msg]` | `<\|end_of_msg\|>` | ends a message |

A turn:

```
<|open|>message role="user"<|sep|>What is the weather?<|close|>message<|sep|><|end_of_msg|>
```

and the model is handed the floor with an unclosed assistant message:

```
<|open|>message role="assistant"<|sep|><|open|>think<|sep|>
```

`examples/chat-k3.json` covers the text conversation in four
prefix/suffix strings, which is all the C CLI can carry. It explicitly does
not cover tool definitions, tool results, JSON schemas, the think channel,
or parsing the reply back. Those are what `serve/` adds.

### serve/xtml.py — the prompt

A port of `encoding_k3.py`, checked against it. It renders:

- **tool declarations** — a system message carrying compact JSON Schema,
  with a separate lazy-loading variant for tools introduced mid-conversation
- **tool calls** — `call` elements with typed `argument` children
  (`string`, `number`, `boolean`, `null`, `object`, `array`), or a raw
  `json` element when the model's arguments did not parse
- **tool results** — `message role="tool"` numbered by position, with
  out-of-order OpenAI `tool_call_id` results re-sorted to match the calls
- **response_format** — `json_object` and `json_schema`, injected as
  synthetic system messages, since K3 has no request field for them
- **tool_choice** — `required` and `none`, likewise
- **the think channel**, and `thinking_effort`
- **images** — `<|media_begin|>image WxH<|media_content|><|media_pad|><|media_end|>`

It returns **segments**, not a string:

```python
Segment('<|open|>', markup=True), Segment('message', markup=False), ...
```

because the two halves go to different tokenizer entry points —
`waste_tokenize_markup` for structure, `waste_tokenize` for anything a
user, document or tool wrote. That is what stops pasted text from closing a
turn or opening a forged system message. Upstream draws the same line with
`allowed_special` against `disallowed_special`.

Two rules in `tokenize_segments` are load-bearing and easy to "optimize"
into bugs:

1. **Never concatenate a prompt and encode it once.** That hands whoever
   wrote the content the ability to write the structure.
2. **Never merge adjacent same-mode segments either.** Upstream encodes one
   segment at a time, and BPE is not associative: ` role` + `="` + `user`
   encoded apart is a different token sequence than ` role="user"` encoded
   whole. Merging is a cheap win and a wrong prompt.
   `tests/serve/test_engine.py::test_segments_are_encoded_separately`
   demonstrates the difference on a real tokenizer.

### serve/regions.py — the reply

The half `encoding_k3.py` does not have. It reads the model's XTML back
into `reasoning_content`, `content` and OpenAI `tool_calls`, incrementally,
so SSE deltas can go out while the model is still talking.

There are two ways to feed it, and they are not equally good:

- **`feed_token(id, piece)`** — what the server uses. Structure is decided
  by the token id the engine reports. A model that writes the *characters*
  `<|sep|>` — because a user asked what the markup looks like — emits
  ordinary text tokens, and the element stays open. This is the output-side
  twin of the tokenize/tokenize_markup split.
- **`feed(text)`** — for hosts that only have text. It finds markers by
  scanning, so it cannot tell a real `<|sep|>` from one the model spelled
  out.

Malformed output is expected, not exceptional: an unterminated element, a
`<|close|>` for something never opened, a reply cut off mid-marker by the
token limit. Every one ends as text or a dropped element. A truncated
answer beats no answer.

## HTTP

| endpoint | notes |
|---|---|
| `GET /health` | liveness; never requires the API key |
| `GET /v1/models`, `GET /v1/models/{id}` | reports the container's real shape under a `waste` key |
| `POST /v1/chat/completions` | streaming and not, tools, images |
| `POST /v1/completions` | raw continuation, no chat template |

Supported request fields: `messages`, `tools`, `tool_choice`,
`response_format`, `temperature`, `top_p`, `top_k`, `seed`, `max_tokens` /
`max_completion_tokens`, `stop`, `stream`, `stream_options.include_usage`,
`reasoning_effort`.

Responses carry an extra `waste` object with the numbers that actually
matter for an expert-streaming engine — hit rate, bytes read, whether the
page cache was bypassed — because the OpenAI schema has nowhere to put them.
It also names the selected performance profile and reports whether its
request-scoped PM-QoS lease was acquired and released. `GET /health` exposes
the same profile plus aggregate control-channel state.

### reasoning_effort

K3's encoder accepts `low`, `high`, `max`. Its own system message
advertises a fourth value, `medium`, and its assert then rejects it; the
port reproduces the refusal rather than the documentation, and the server
returns a 400 that says so instead of quietly substituting `high`.

`none`, `minimal` and `off` turn the think channel off entirely.

**The default is thinking on**, which is what the model was trained for.
The technical report measures reasoning at up to 73% of the tokens in a
request, and at this engine's speeds that is a long wait before the first
word of the answer. `--no-thinking` flips the default; a request can
override either way.

### Statelessness

Each HTTP request resets the engine's conversation state before it is
prefilled. A `waste_ctx` keeps its KDA state and MLA KV across calls —
that is what makes `waste chat` a conversation — and carrying that into a
stateless server means request N is prefilled on top of request N-1: the
same request gets different answers depending on what came before, and one
client's turn conditions another's. The lock spans prompt building *and*
generation, so the image queue cannot be crossed between requests either.

### Concurrency

`waste.h`: a `waste_ctx` is not thread-safe. So generations serialize on
one lock, and requests queue. On a model streaming experts off an SSD at a
few tokens a second, the wait for the lock is small next to the wait for
the answer.

Streaming is written straight from the token callback, on the thread
holding the lock. A client hanging up propagates back as a return value the
engine understands — the callback says stop, `waste_generate` unwinds, the
next request starts. A disconnected client stops costing tokens
immediately, which on a model this slow is the difference between a wasted
minute and a wasted hour.

### Acer GN100 `spark-q0` profile

`--performance-profile spark-q0` is an explicit, strict profile for the Acer
Veriton GN100 / NVIDIA GB10 work. It fixes the in-process settings from the
winning current-upstream qualification: eight compute threads, LFRU, effective
direct I/O, stable arithmetic, two upstream read-ahead threads at depth two,
router lookahead disabled, ordinary pageable/non-purgeable storage, and a
zero-microsecond Linux CPU DMA-latency request around `engine.generate()`
prefill and decode. For streaming, acquisition
happens immediately before the SSE headers so the first callback can write
safely. The profile does not change the default server.

The server is never privileged and never receives the
`/dev/cpu_dma_latency` descriptor. Launch it through the root holder, which
opens that exact character device on `BEGIN`, closes it on `END`, and drops the
child to the named ordinary account:

```bash
sudo python3 tools/pm_qos_exec.py \
  --scope requests --max-hold-seconds 1800 \
  --status /var/lib/waste-qos/status.json \
  --events /var/lib/waste-qos/events.jsonl \
  --user "$USER" -- \
  taskset -c 5-9,15-19 python3 -m serve /path/to/k3.waste \
    --performance-profile spark-q0 --port 8000
```

All accepted GN100 Q0 measurements used CPU set `5-9,15-19`. `taskset` above
reproduces that external condition, but this profile neither sets nor verifies
affinity on the current upstream base. Omitting it is a different run condition
and health/response profile evidence does not certify the CPU set.

The holder's artifacts are mode `0600`; use `sudo` to inspect them. It filters
the root environment before dropping privileges. If a command deliberately
needs a credential such as `HF_TOKEN`, put `--pass-env HF_TOKEN` before `--`.
The status record stores only that variable's name, never its value, and
redacts common secret-valued command options.

The status path, optional event path, and singleton lock path must identify
three distinct files. The holder rejects normalized-path, resolved-parent, and
existing-inode aliases before creating or replacing any artifact, so a status
rewrite cannot unlink the locked inode or erase the event stream.

The profile refuses to start if its private root control socket is missing,
the peer is not root, a fixed setting conflicts, or direct I/O fell back. A
pre-existing engine/debug `WASTE_*` selector not fixed by the profile is also
an error; this includes dump/instrumentation switches and `WASTE_LIB`, whose
effective binary identity current upstream cannot attest. `WASTE_API_KEY` and
the private control descriptor are service plumbing, remain allowed, and are
never included in public profile evidence.

A lost or malformed acknowledgement permanently closes the channel so EOF
makes the root holder release Q0; it is never guessed successful. The holder also
releases on inference errors, disconnects, child or holder death, signals,
protocol errors, and a bounded hold timeout. Timeout or loss of the required
control plane terminates the strict-profile child rather than leaving a server
running under a false label. Root-to-child acknowledgements are nonblocking and
bounded by both a control timeout and, while Q0 is active, the lease deadline.

Status and event files are delayed, best-effort telemetry, outside the Q0
safety boundary: the holder records the true boundary timestamp and actual
open/close counters in memory, closes the device, acknowledges `END`, and only
then performs potentially blocking filesystem writes. Event records include
both occurrence and delivery monotonic timestamps. Consequently the status
file may lag an in-flight request, but it is never written as active after the
descriptor has actually been released. On fatal acknowledgement, EOF, or
protocol/control failure, strict-child termination is also issued before those
writes. Response lease evidence comes from the acknowledged control protocol
rather than the files.

The current server streams socket writes synchronously from the engine's token
callback. Consequently a slow streaming client can extend the lease even
though the intended boundary is model work; the maximum hold is the safety
bound. CPU affinity, whole-expert scheduling, effective read-ahead reporting,
and prefix snapshots do not exist on this upstream base and are not implied by
the profile name. They remain separate measured integration work. When the
prefix-cache branch is integrated, the same lease must cover
`PrefixCache.prepare()` (including cold-miss prefill) and `engine.generate()`,
in that order, with acquisition still preceding streaming headers. Its
integration test must assert all three boundaries explicitly.

### Images

`--vision` loads the tower (434 MB of weights on K3, and 1.12 GB reserved
once the bounded source decode, the tower's activations and the queued image
embeddings are counted — out of the same budget the expert cache draws on).
Images arrive as base64 `data:` URLs.

`http://` and `https://` URLs are **not fetched**. Doing so would make the
server issue requests to addresses its clients choose, which is a
server-side request forgery in any deployment where the server can reach
more of the network than the client can. Local filesystem paths are off by
default too, behind `--allow-local-images`, since they let any client read
files the server can reach.

## Security

- `--host` defaults to `127.0.0.1`. Binding anywhere else without
  `--api-key` prints a warning.
- `--api-key` (or `$WASTE_API_KEY`) requires a bearer token, compared in
  constant time.
- Request bodies are capped at 64 MB, refused on the declared
  Content-Length before anything is read.
- Prompt injection through message content is structurally prevented, not
  filtered: content never reaches the markup tokenizer. Checked end to end
  in `test_server.py` and against the real tokenizer in
  `test_integration.py`.

## Tests

```bash
make serve-check                                  # everything
K3_DIR=/Volumes/WasteDisk/k3 make serve-check     # plus the differential
```

Six suites, in order of what they prove:

| file | what it checks | needs |
|---|---|---|
| `test_xtml.py` | every corpus case rendered **segment for segment against the release's own `encoding_k3.py`**, plus frozen goldens | the release, for the differential |
| `test_regions.py` | round trip: anything the encoder can express, the parser reads back; every chunk split; malformed output | — |
| `test_engine.py` | the ctypes binding against a **real engine** and a synthetic container | `libwaste` |
| `test_qos.py` | exact profile resolution; fake-device/real-socket PM-QoS protocol, timeout, EOF, and privileged-artifact safety | — |
| `test_server.py` | HTTP over real sockets against a scripted engine | — |
| `test_integration.py` | the whole stack, no fakes | `libwaste` |

The goldens in `tests/serve/fixtures/` record whether the release was
present when they were generated. Goldens produced by our own renderer
would lock in whatever it currently does, bugs included, so
`test_goldens_were_generated_from_upstream` fails rather than let that pass
as evidence.

Regenerate them on a machine that has the weights:

```bash
K3_DIR=/Volumes/WasteDisk/k3 python3 tools/gen_xtml_goldens.py
```

## Flags

```
python3 -m serve MODEL [options]

  --host, --port, --model-id, --api-key
  --budget SIZE      hard RAM ceiling, e.g. 48G (0 = the engine chooses)
  --ctx N            context tokens
  --threads N        compute threads (0 = one per core)
  --cache {lfru,lru} expert-cache eviction policy
  --no-direct-io     keep the page cache in the way (the bypass is on)
  --vision           load the vision tower
  --verify           check every expert record's crc32 as it is read
  --usage PATH       learned hotlist (default <model>/usage.waste)
  --performance-profile spark-q0
                     strict GN100 request-scoped Q0 profile
  --max-tokens N     default cap when a request does not set one
  --no-thinking      answer without the think channel unless asked
  --allow-local-images
  --plan             print the memory plan and exit
```
