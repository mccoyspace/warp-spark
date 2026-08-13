# Examples

This directory contains runnable examples for the CLI, the public C API, and
the OpenAI-compatible server. Commands below assume they are run from the
repository root after `make`.

## CLI examples

### Text

```bash
# Inspect the model and the memory budget before loading it.
./waste info ~/models/k3.waste
./waste plan ~/models/k3.waste

# Generate a completion from an argument, stdin, or a file.
./waste run ~/models/k3.waste "Write a small C function that swaps two ints" -n 64
printf '%s\n' "The capital of Italy is" | ./waste run ~/models/k3.waste - -n 16
./waste run ~/models/k3.waste --file prompt.txt -n 64

# Inspect tokenization and the next-token distribution.
./waste tokenize ~/models/k3.waste "Hello, world"
./waste eval ~/models/k3.waste "The capital of France is" --top-k 5

# Keep a conversation state and inspect its cumulative I/O statistics.
./waste chat ~/models/k3.waste
> /stats
> /save session.waste
> /load session.waste
```

`-n` limits generated tokens. `--budget` is normally unnecessary because the
engine selects a conservative default. Use `--verify` after copying or
downloading a container to check expert-record checksums as they are read.

### Multimodal

Kimi K3 can combine text with one or more images:

```bash
./waste run ~/models/k3.waste "Describe this photograph" \
    --image photo.jpg -n 64

./waste run ~/models/k3.waste "List the differences" \
    --image before.png --image after.png -n 96

./waste eval ~/models/k3.waste "A photograph of a" \
    --image coast.jpg --top-k 10
```

Images are placed before the text in CLI prompts. In a chat, attach an image
to the next turn and then ask the question:

```text
$ ./waste chat ~/models/k3.waste
> /image diagram.png
(diagram.png attached to the next message)
> Explain the data flow in this diagram.
```

Repeat `/image` before the message to attach several images. Once an image has
been consumed, later turns can discuss it without encoding it again because
its positions remain in the conversation state. `/reset` clears both text and
image state.

An image expands into many prompt positions. At K3's default patch budget, an
896×896 image becomes 256 positions, and each costs approximately as much as a
text position during prefill. See [K3.md](../docs/K3.md) and
[TECHNICAL.md](../docs/TECHNICAL.md) for measurements.

## C API examples

The examples include only the public header, [waste.h](../src/waste.h):

- [api_plan.c](api_plan.c) reads the memory plan without loading weights;
- [api_text.c](api_text.c) tokenizes a raw prompt and generates text;
- [api_vision.c](api_vision.c) builds a K3 image turn, expands the image
  placeholder, and generates a response.

Build and run them against the static library:

```bash
make libwaste.a

cc -O2 -std=gnu11 -Isrc examples/api_plan.c libwaste.a \
    -lm -lpthread -o example-plan
cc -O2 -std=gnu11 -Isrc examples/api_text.c libwaste.a \
    -lm -lpthread -o example-text
cc -O2 -std=gnu11 -Isrc examples/api_vision.c libwaste.a \
    -lm -lpthread -o example-vision

./example-plan ~/models/k3.waste 4096
./example-text ~/models/k3.waste "The capital of Italy is"
./example-vision ~/models/k3.waste photo.jpg "What is in this image?"
```

The vision example sets `cfg.vision = 1`, calls `waste_image_add`, keeps K3
markup separate from untrusted user text, expands `<|media_pad|>` with
`waste_image_expand`, and only then calls `waste_generate`. That ordering is
required: one placeholder represents every embedding produced by the tower.

Conversation state can be persisted independently of generation:

```c
waste_state_save(ctx, "session.waste");
waste_state_reset(ctx);
waste_state_load(ctx, "session.waste");
```

See [ENGINE.md](../docs/ENGINE.md) for lifecycle and threading details.

## Server examples

Start the server with the vision tower when image requests are needed:

```bash
make libwaste.dylib                     # libwaste.so on Linux
python3 -m serve ~/models/k3.waste --port 8000 --vision
```

### Chat and streaming

```bash
curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"k3","messages":[{"role":"user","content":"Why is the sky blue?"}],"reasoning_effort":"off"}'

curl -N localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"k3","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"Write a haiku about local inference"}]}'

curl localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"k3","prompt":"The capital of Italy is","max_tokens":16}'
```

The last request is a raw continuation and does not use the chat renderer.

### Images

The server accepts base64 `data:` URLs. It deliberately does not fetch remote
HTTP URLs. Local paths require the explicit `--allow-local-images` server flag.

```bash
IMAGE_B64="$(base64 < photo.jpg | tr -d '\n')"

curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary "{\"model\":\"k3\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,${IMAGE_B64}\"}},{\"type\":\"text\",\"text\":\"Describe this image\"}]}],\"reasoning_effort\":\"off\"}"
```

### Tools and structured output

```bash
curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"k3","messages":[{"role":"user","content":"What is the weather in Rome?"}],"tools":[{"type":"function","function":{"name":"weather","description":"Get current weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"required"}'

curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"k3","messages":[{"role":"user","content":"Return the capital of Italy"}],"response_format":{"type":"json_schema","json_schema":{"name":"capital","schema":{"type":"object","properties":{"country":{"type":"string"},"capital":{"type":"string"}},"required":["country","capital"],"additionalProperties":false}}}}'
```

## How server prompts are rendered

The HTTP server is stateless between requests. Under one engine lock it resets
the conversation, resolves and encodes images, renders the request into XTML
segments, tokenizes those segments, expands image placeholders, and starts
generation. Requests queue because a `waste_ctx` is not thread-safe.

Prompt structure and user content take different tokenizer paths. XTML control
segments use `waste_tokenize_markup`; message text uses `waste_tokenize`, where
control-looking strings remain ordinary text. Segments are never concatenated
or merged before tokenization: doing so would both permit prompt-structure
injection and change BPE boundaries relative to K3's reference encoder.

The renderer emits, in order, tool declarations, thinking controls, messages,
tool-choice and response-format instructions, then an open `think` or
`response` element. Images become XTML media blocks whose `<|media_pad|>` token
expands to the number of embeddings produced by the tower.

On output, `serve/regions.py` parses structural token IDs rather than scanning
text. It incrementally separates `reasoning_content`, answer content, and tool
calls, which allows the same parser to produce blocking responses and SSE
deltas. A disconnected streaming client cancels generation immediately.

See [SERVE.md](../docs/SERVE.md) for supported fields, endpoint behavior,
security constraints, differential tests, and the full rendering protocol.

## chat.json — the conversation format

`waste chat` addresses an instruct model in the format it was trained on.
That format is read from `chat.json` **inside the container**, next to
`manifest.json`. Without one the CLI says so and continues raw, which is
deliberate: a guessed format is worse than a visible absence.

[chat.json](chat.json) here is the ChatML layout, which is what a large
part of the instruct ecosystem uses. Copy it into a container and edit the
strings:

```bash
cp examples/chat.json ~/models/some-model.waste/chat.json
```

Every field is optional. Each role is a `[prefix, suffix]` pair, and
`open` is what is appended after the last user turn to hand the floor to
the model:

```json
{"system":    ["<prefix>", "<suffix>"],
 "user":      ["<prefix>", "<suffix>"],
 "assistant": ["<prefix>", "<suffix>"],
 "open":      "<what starts the model's turn>"}
```

`\n` and `\t` are the escapes the reader understands. Whatever markup you
put in these strings has to exist in the tokenizer as a *single* token, or
it will be split into ordinary text and the model will not recognize it —
`waste tokenize MODEL "<|im_start|>"` is the check, and the container's
`specials.json` is the list of what is available. Neither Kimi release
uses ChatML's markers, so this file is a starting point for other models
and the two below are the ones to copy for those.

## chat-kimi-linear.json — Kimi-Linear

[chat-kimi-linear.json](chat-kimi-linear.json) is Kimi-Linear's format.
`tools/convert.py` installs it when it recognises the architecture; copy
it by hand into a container converted before that:

```bash
cp examples/chat-kimi-linear.json ~/models/kimi-linear.waste/chat.json
```

The markup is Moonshot's own, not ChatML — five control tokens, one per
role plus a separator and a terminator:

| token | id | role |
|---|---|---|
| `<\|im_system\|>` | 163594 | opens a system turn |
| `<\|im_user\|>` | 163587 | opens a user turn |
| `<\|im_assistant\|>` | 163588 | opens an assistant turn |
| `<\|im_middle\|>` | 163601 | ends the turn header |
| `<\|im_end\|>` | 163586 | ends a turn |

A turn is the opener, the role name as ordinary text, `<|im_middle|>`, the
content, and `<|im_end|>`; the model is handed the floor with an assistant
opener and no terminator. Note that `<|im_start|>` — ChatML's, and what
[chat.json](chat.json) carries — is **not** in this vocabulary: it encodes
as six ordinary tokens, so a container given the generic template answers
as if it were continuing prose. `waste tokenize` is the check.

`serve/` serves this format too, from this same file: a container whose
tokenizer has no XTML markers is chatted with through its `chat.json`.
Plain conversation only — no tools, no reasoning channel, no images, each
refused with a 400 naming the field rather than dropped. See
[SERVE.md](../docs/SERVE.md), "serve/chatfmt.py".

## chat-k3.json — Kimi K3

[chat-k3.json](chat-k3.json) is K3's own format, transcribed from
`encoding_k3.py` in the release. **`tools/convert.py` installs it for you**
when it recognises K3, so a fresh container already answers questions
instead of continuing text. It is here for containers converted before
that, and for editing:

```bash
cp examples/chat-k3.json ~/models/k3.waste/chat.json
```

The converter never overwrites a `chat.json` that is already there, so an
edited one survives a re-conversion.

**Neither Kimi release ships a Jinja template**, which is why the
converter has nothing to copy: K3 does not have one. It builds the prompt
with a Python program (`encoding_k3.py`) that emits a token sequence
directly, in **XTML** — an XML-like markup where the angle brackets are
three reserved special tokens, with a fourth as the stop marker:

| in the report | token | id |
|---|---|---|
| `[open]` | `<\|open\|>` | 163587 |
| `[sep]` | `<\|sep\|>` | 163589 |
| `[close]` | `<\|close\|>` | 163588 |
| `[end_of_msg]` | `<\|end_of_msg\|>` | 163586 |

A turn is a `message` element with a `role` attribute:

```
<|open|>message role="user"<|sep|> …content… <|close|>message<|sep|><|end_of_msg|>
```

and the model is handed the floor with an unclosed assistant message:

```
<|open|>message role="assistant"<|sep|><|open|>response<|sep|>
```

Everything that is not a control token — `message`, `role="user"`, the
tag names — is ordinary text, which is what lets a format this structured
fit four prefix/suffix strings at all.

### What this template leaves out

`encoding_k3.py` is 647 lines and this file covers the text conversation.
Not covered: **tool definitions and tool results**, which are their own
XTML elements with typed `argument` children; `response_format` and JSON
schemas, which the program injects as synthetic system messages; and the
**think channel**. K3 opens `<|open|>think<|sep|>` before the response
when thinking is enabled, and the technical report measures reasoning at
up to 73% of the tokens in a request — at this engine's speeds that is
hours before an answer starts, so this template asks for the `response`
channel directly. Set the `open` field to `…<|open|>think<|sep|>` if you
want the reasoning, and expect to wait for it.

**All of it lives in [`serve/`](../serve/)**, which is where a format this
big belongs: four prefix/suffix strings cannot express a typed argument
list, and a C engine should not be growing a JSON Schema renderer.
`serve/xtml.py` is a port of `encoding_k3.py`, checked against it segment
for segment, and `serve/regions.py` is the other half of that program —
reading the reply back into reasoning, answer and tool calls. Both are
reachable over HTTP:

```bash
python3 -m serve ~/models/k3.waste
```

See [docs/SERVE.md](../docs/SERVE.md). This file remains the answer for
`waste chat` and `waste run`, which carry no Python.
