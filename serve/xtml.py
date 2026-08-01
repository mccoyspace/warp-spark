# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
xtml.py — Kimi K3's prompt format, as segments the engine can tokenize.

K3 ships no Jinja template. It builds prompts with a Python program,
`encoding_k3.py` in the release, which emits a token sequence directly in
XTML: an XML-like markup whose angle brackets are three reserved special
tokens, with a fourth ending the message.

    <|open|>message role="user"<|sep|> …content… <|close|>message<|sep|><|end_of_msg|>

This module is a port of that program. It is deliberately a *port* and not
an interpretation: where upstream is surprising, this file reproduces the
surprise and says so in a comment, because a prompt that differs from the
one the model was trained on is wrong in ways no test here can see. The
differential test in tests/serve/test_xtml.py checks it against the real
encoding_k3.py whenever the release is on disk.

What it emits is a list of Segment, not a string:

    Segment("<|open|>", markup=True), Segment("message role=", markup=False), ...

because the two halves go to different tokenizer entry points. Markup goes
to waste_tokenize_markup, where `<|open|>` is the one control token it
looks like. Everything a user, a document or a tool wrote goes to
waste_tokenize, where the same bytes are ordinary text and cannot close a
turn or forge a system message. Concatenating the two into one string and
encoding it once would hand whoever wrote the content the ability to write
the structure too — upstream draws the same line, with allowed_special
against disallowed_special, and draws it for the same reason.

The engine consumes the list through serve.engine.tokenize_segments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional

# The four reserved tokens the markup is made of. Ids are the container's
# business, not this module's — these are the strings the tokenizer maps.
OPEN_TOKEN = "<|open|>"
CLOSE_TOKEN = "<|close|>"
SEP_TOKEN = "<|sep|>"
END_OF_MSG_TOKEN = "<|end_of_msg|>"

# The needle a caller leaves in the text for each image. The vision
# processor replaces it with
#   <|media_begin|>image WxH<|media_content|><|media_pad|><|media_end|>
# so the model is told the resolution it is looking at (see waste.h).
IMAGE_PLACEHOLDER = "<|kimi_image_placeholder|>"

# Upstream asserts against this set. Note that the system message it emits
# advertises `low`, `medium`, `high`, `max` — four values — while the set
# accepts three: `medium` is documented and rejected. That inconsistency is
# upstream's and is reproduced here rather than repaired, so a request that
# K3's own encoder refuses is refused identically. See
# tests/serve/test_xtml.py::test_thinking_effort_medium_is_rejected_upstream.
_VALID_THINKING_EFFORTS = {"low", "high", "max"}


class XTMLError(ValueError):
    """A conversation that cannot be rendered in K3's format."""


@dataclass(frozen=True)
class Segment:
    """A run of prompt text and the tokenizer entry point it belongs to.

    `markup=True` is upstream's `allow_special=True`.
    """

    text: str
    markup: bool = False


class ChatSegments(list[Segment]):
    """Rendered segments plus the stable leading-family boundary.

    The boundary follows top-level tool declarations, the optional thinking
    effort note, and contiguous leading system messages.  Everything after
    it is request/conversation-specific.  It is segment-based because BPE is
    deliberately run once per Segment; callers can therefore tokenize the
    two slices independently and concatenate them without changing a token.
    """

    family_root_segments: int = 0


def image_prompt(width: int, height: int) -> str:
    """The media block K3 wraps an image in, at its source resolution.

    Mirrors KimiK3VisionProcessor.make_image_prompt. The dimensions are the
    *original* ones, not the resized patch grid — waste_image_dimensions
    reads them from the file header without decoding it.
    """
    return (f"<|media_begin|>image {width}x{height}"
            f"<|media_content|><|media_pad|><|media_end|>")


# ---- segment construction ------------------------------------------------


class _ImagePromptState:
    """Hands out one media block per placeholder, in order.

    `image_prompts=None` means the caller is rendering without images, and
    the placeholder is left standing as itself.
    """

    def __init__(self, image_prompts: Optional[list[str]] = None):
        self.image_prompts = image_prompts
        self.index = 0

    def next_prompt(self) -> str:
        if self.image_prompts is None:
            return IMAGE_PLACEHOLDER
        if self.index >= len(self.image_prompts):
            raise XTMLError("More image placeholders than image prompts.")
        prompt = self.image_prompts[self.index]
        self.index += 1
        return prompt

    def assert_consumed(self) -> None:
        if self.image_prompts is None:
            return
        if self.index != len(self.image_prompts):
            raise XTMLError(
                f"image prompt count {len(self.image_prompts)} != "
                f"consumed placeholder count {self.index}")


def _segment(text: Any, *, markup: bool = False) -> list[Segment]:
    text = str(text)
    if not text:
        return []
    return [Segment(text, markup=markup)]


def _control(text: str) -> list[Segment]:
    return _segment(text, markup=True)


def _text(text: Any) -> list[Segment]:
    return _segment(text, markup=False)


def _append_text(segments: list[Segment], text: Any,
                 image_state: _ImagePromptState) -> None:
    """Append caller text, splicing a media block at each placeholder.

    The media block is markup — it is made of reserved tokens — but the text
    around it is not, which is the whole reason this splits rather than
    encoding the string once.
    """
    text = str(text)
    if text == "":
        return
    if image_state.image_prompts is None or IMAGE_PLACEHOLDER not in text:
        segments.extend(_text(text))
        return

    parts = text.split(IMAGE_PLACEHOLDER)
    for i, part in enumerate(parts):
        segments.extend(_text(part))
        if i < len(parts) - 1:
            segments.extend(_segment(image_state.next_prompt(), markup=True))


def _escape_attr_value(value: Any) -> str:
    """`&` then `"` — that order, or the ampersand of `&quot;` is escaped again.

    Note what is *not* escaped: `<` and `>` are literal in an XTML attribute,
    because the brackets of this markup are tokens, not characters.
    """
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


def _attr(key: str, value: Any) -> list[Segment]:
    return (_text(f" {key}") + _text('="')
            + _text(_escape_attr_value(value)) + _text('"'))


def _open_tag(tag: str, attrs: Iterable[tuple[str, Any]] = ()) -> list[Segment]:
    segments: list[Segment] = []
    segments.extend(_control(OPEN_TOKEN))
    segments.extend(_text(tag))
    for key, value in attrs:
        segments.extend(_attr(key, value))
    segments.extend(_control(SEP_TOKEN))
    return segments


def _close_tag(tag: str) -> list[Segment]:
    segments: list[Segment] = []
    segments.extend(_control(CLOSE_TOKEN))
    segments.extend(_text(tag))
    segments.extend(_control(SEP_TOKEN))
    return segments


def _end_of_msg() -> list[Segment]:
    return _control(END_OF_MSG_TOKEN)


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _xtml_type(value: Any) -> str:
    """The `type` attribute of an `argument` element.

    bool is checked before int because in Python it *is* an int, and a
    tool argument of True would otherwise be declared a number.
    """
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        return "string"
    if _is_mapping(value):
        return "object"
    return "array"


def _xtml_value(value: Any) -> str:
    """A string argument is written raw; everything else as JSON.

    So the string "hi" renders as hi, not "hi" — the `type` attribute
    already said which it is, and the model was trained on the bare form.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---- normalization -------------------------------------------------------


def extract_response_schema(response_format: Any) -> Any:
    """Dig the JSON Schema out of an OpenAI `response_format`.

    Tolerant of the three shapes clients actually send: the schema under
    `json_schema.schema`, under `json_schema.json_schema`, or the
    `json_schema` object being the schema itself.
    """
    if response_format is None:
        return None

    json_schema = _get_value(response_format, "json_schema")
    if json_schema is None:
        return None

    if isinstance(json_schema, dict):
        return json_schema.get(
            "schema", json_schema.get("json_schema", json_schema))

    schema = _get_value(json_schema, "schema")
    if schema is not None:
        return schema

    schema = _get_value(json_schema, "json_schema")
    if schema is not None:
        return schema

    return json_schema


def deep_sort_dict(obj: Any) -> Any:
    """Sort every mapping by key, recursively.

    Tool declarations and response schemas go through this so that two
    requests carrying the same tools in different key order produce the
    same prompt bytes — which is what makes a prefix cache hit.
    """
    if isinstance(obj, dict):
        return {k: deep_sort_dict(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [deep_sort_dict(item) for item in obj]
    return obj


def normalize_tool_arguments(arguments: Any) -> tuple[dict[str, Any], Optional[str]]:
    """Split a tool call's arguments into (mapping, raw JSON block).

    OpenAI sends arguments as a JSON *string*. When it parses to an object
    we render typed `argument` children; when it does not parse at all we
    keep the original text and render it inside a `json` element, because
    a model that emitted broken JSON should see back what it emitted rather
    than a silent repair. A string that parses to a non-object is an error:
    there is no XTML shape for it.
    """
    if arguments is None:
        return {}, None
    if isinstance(arguments, dict):
        return arguments, None
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}, None
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}, arguments
        if not isinstance(parsed, dict):
            raise XTMLError("Kimi K3 tool call arguments must be a JSON object.")
        return parsed, None
    raise XTMLError(
        "Kimi K3 tool call arguments must be a dict or a JSON object string.")


def normalize_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message

    normalized = dict(message)

    tools = normalized.get("tools")
    if tools is not None:
        normalized["tools"] = deep_sort_dict(tools)

    tool_calls = normalized.get("tool_calls")
    if not tool_calls:
        return normalized

    normalized_calls = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            normalized_calls.append(tool_call)
            continue

        tc = dict(tool_call)
        function = tc.get("function")
        if isinstance(function, dict):
            fn = dict(function)
            arguments, json_block = normalize_tool_arguments(fn.get("arguments"))
            fn["arguments"] = arguments
            if json_block is None:
                fn.pop("_xtml_json_block", None)
            else:
                fn["_xtml_json_block"] = json_block
            tc["function"] = fn
        else:
            arguments, json_block = normalize_tool_arguments(tc.get("arguments"))
            tc["arguments"] = arguments
            if json_block is None:
                tc.pop("_xtml_json_block", None)
            else:
                tc["_xtml_json_block"] = json_block
        normalized_calls.append(tc)

    normalized["tool_calls"] = normalized_calls
    return normalized


def normalize_conversation(conversation: Any) -> Any:
    if not isinstance(conversation, list):
        return conversation

    def normalize_messages(messages: list[Any]) -> list[Any]:
        return [normalize_message(message) for message in messages]

    if conversation and isinstance(conversation[0], list):
        return [normalize_messages(messages) for messages in conversation]
    return normalize_messages(conversation)


def is_batched_conversation(conversation: Any) -> bool:
    return (isinstance(conversation, list) and bool(conversation)
            and isinstance(conversation[0], list))


def _tool_call_id_index(tool_calls: Any) -> dict:
    """Map assistant `tool_calls[].id` to (1-based position, function name).

    The position mirrors the enumeration over `tool_calls`: every entry
    advances it, even an id-less one. Duplicate ids keep their first
    occurrence.
    """
    index: dict = {}
    if not isinstance(tool_calls, list):
        return index
    for position, tool_call in enumerate(tool_calls, start=1):
        if not isinstance(tool_call, dict):
            continue
        call_id = tool_call.get("id")
        if call_id is None:
            continue
        key = str(call_id)
        if key in index:
            continue
        function = tool_call.get("function")
        name = (function.get("name") if isinstance(function, dict)
                else tool_call.get("name"))
        index[key] = (position, name)
    return index


def normalize_xtml_tool_result_messages(messages: list[Any]) -> list[Any]:
    """Re-sort tool results into the order the assistant asked for them.

    XTML numbers tool results by rendered position, so their order in the
    list *is* the binding to the calls. An OpenAI client is under no
    obligation to send them in call order — it binds by `tool_call_id` —
    so each run of consecutive tool messages is matched against the most
    recent preceding assistant `tool_calls` and sorted by matched position.

    The matched call is authoritative: the message's `tool` is set to that
    call's function name, so an explicit and possibly stale `tool`/`name`
    cannot drift out of sync with the reordered position. A run that cannot
    be fully matched is left untouched rather than half-sorted. Re-running
    is idempotent.

    Side-effect free: matched messages are shallow-copied before rewriting,
    everything else is passed through by reference. The caller's list and
    message objects are never mutated.
    """
    if not isinstance(messages, list):
        return messages

    output: list[Any] = []
    current_index: dict = {}
    i = 0
    n = len(messages)

    while i < n:
        message = messages[i]

        if isinstance(message, dict) and message.get("role") == "assistant":
            tool_calls = message.get("tool_calls")
            current_index = _tool_call_id_index(tool_calls) if tool_calls else {}
            output.append(message)
            i += 1
            continue

        if not isinstance(message, dict) or message.get("role") != "tool":
            output.append(message)
            i += 1
            continue

        run: list[tuple] = []  # (position, original_offset, message, name)
        unresolved = False
        offset = 0
        while (i < n and isinstance(messages[i], dict)
               and messages[i].get("role") == "tool"):
            tool_message = messages[i]
            call_id = tool_message.get("tool_call_id", tool_message.get("id"))
            matched = current_index.get(str(call_id)) if call_id is not None else None
            if matched is None:
                unresolved = True
                run.append((None, offset, tool_message, None))
            else:
                position, name = matched
                run.append((position, offset, tool_message, name))
            offset += 1
            i += 1

        if unresolved:
            output.extend(item[2] for item in run)
        else:
            run.sort(key=lambda item: (item[0], item[1]))
            for _, _, tool_message, name in run:
                if name is None:
                    output.append(tool_message)
                    continue
                resolved = dict(tool_message)
                resolved["tool"] = name
                if "name" in resolved:
                    resolved["name"] = name
                output.append(resolved)

    return output


# ---- rendering -----------------------------------------------------------


def _render_content_segments(content: Any,
                             image_state: _ImagePromptState) -> list[Segment]:
    """A message's content: a plain string, or OpenAI's list of parts."""
    segments: list[Segment] = []
    if isinstance(content, str):
        _append_text(segments, content, image_state)
    elif content is not None:
        for part in content:
            if part["type"] in ["image", "image_url"]:
                segments.extend(_segment(image_state.next_prompt(), markup=True))
            else:
                _append_text(segments, part["text"], image_state)
    return segments


def _internal_system_message(message_type: str, body: str) -> list[Segment]:
    """A system turn the server synthesizes, not one the caller wrote.

    K3 has no request fields for tool_choice or response_format: they are
    delivered as system messages carrying a `type` attribute, which is how
    the model was trained to receive them.
    """
    segments: list[Segment] = []
    segments.extend(_open_tag("message", [("role", "system"),
                                          ("type", message_type)]))
    segments.extend(_text(body.strip()))
    segments.extend(_close_tag("message"))
    segments.extend(_end_of_msg())
    return segments


def _render_assistant_segments(message: dict[str, Any],
                               image_state: _ImagePromptState,
                               thinking: bool = True) -> list[Segment]:
    segments: list[Segment] = []
    # The think channel is structural: with thinking on, every assistant
    # message carries the open/close pair even when there is no reasoning to
    # put in it. With thinking off the channel is absent entirely — not
    # empty, absent.
    if thinking:
        reasoning_content = (message.get("reasoning_content")
                             or message.get("reasoning"))
        segments.extend(_open_tag("think"))
        if reasoning_content is not None and str(reasoning_content).strip():
            _append_text(segments, reasoning_content, image_state)
        segments.extend(_close_tag("think"))

    segments.extend(_open_tag("response"))
    segments.extend(_render_content_segments(message.get("content"), image_state))
    segments.extend(_close_tag("response"))

    tool_calls = message.get("tool_calls")
    if tool_calls:
        segments.extend(_open_tag("tools"))
        for index, tool_call in enumerate(tool_calls, start=1):
            if not isinstance(tool_call, dict):
                raise XTMLError("Kimi K3 tool calls must be objects.")
            fn = tool_call.get("function", tool_call)
            if (not isinstance(fn, dict) or
                    not isinstance(fn.get("name"), str) or not fn["name"]):
                raise XTMLError(
                    "Kimi K3 tool calls require a non-empty function name.")
            segments.extend(_open_tag("call", [("tool", fn["name"]),
                                               ("index", index)]))
            args = fn.get("arguments", {})
            json_block = fn.get("_xtml_json_block")
            if json_block is not None:
                segments.extend(_open_tag("json", [("type", "object")]))
                _append_text(segments, json_block, image_state)
                segments.extend(_close_tag("json"))
            elif _is_mapping(args):
                for key, value in args.items():
                    segments.extend(_open_tag(
                        "argument", [("key", key), ("type", _xtml_type(value))]))
                    _append_text(segments, _xtml_value(value), image_state)
                    segments.extend(_close_tag("argument"))
            segments.extend(_close_tag("call"))
        segments.extend(_close_tag("tools"))

    return segments


def _render_tool_declare(tools: Any, *, dynamic: bool = False) -> list[Segment]:
    """Tool definitions, as a system message carrying compact JSON Schema.

    `dynamic=True` is the lazy-loading variant: a mid-conversation system
    message announcing tools that were not in the opening declaration.
    """
    if dynamic:
        body = ("## New Tools Available\n"
                "The system dynamically extends the toolset via lazy-loading.\n"
                "You have access to all existing and extended tools.\n"
                "Here are the specs for the extended tools.\n\n"
                "```json\n"
                f"{_json_compact(tools)}\n"
                "```")
    else:
        body = ("# Tools\n"
                "Here are the available tools, described in JSONSchema.\n\n"
                "```json\n"
                f"{_json_compact(tools)}\n"
                "```")
    segments: list[Segment] = []
    segments.extend(_open_tag("message", [("role", "system"),
                                          ("type", "tool-declare")]))
    segments.extend(_text(body))
    segments.extend(_close_tag("message"))
    segments.extend(_end_of_msg())
    return segments


def build_chat_segments(messages: list[Any],
                        tools: Optional[list[dict]] = None,
                        *,
                        add_generation_prompt: bool = True,
                        thinking: bool = True,
                        image_prompts: Optional[list[str]] = None,
                        **kwargs: Any) -> ChatSegments:
    """Render a conversation into segments, in K3's own order.

    That order is: tool declarations, then a thinking-effort note, then the
    conversation, then tool_choice and response_format notes, then the
    opening of the assistant's turn. The synthesized notes come *after* the
    conversation deliberately — they describe this request, so they are the
    last thing the model reads before it answers.

    Keyword arguments beyond the named ones mirror upstream's `**kwargs`:
    `thinking_effort`, `tool_choice`, `response_format`, `response_schema`.
    """
    # Re-sort tool results at the lowest layer, so every caller gets
    # correctly ordered XTML whether it came through the server or not.
    messages = normalize_xtml_tool_result_messages(messages)
    messages = normalize_conversation(messages)
    tools = deep_sort_dict(tools)

    kwargs = dict(kwargs)
    response_format = kwargs.get("response_format")
    if "response_schema" not in kwargs:
        response_schema = extract_response_schema(response_format)
        if response_schema is not None:
            kwargs["response_schema"] = response_schema
    if kwargs.get("response_schema") is not None:
        kwargs["response_schema"] = deep_sort_dict(kwargs["response_schema"])

    image_state = _ImagePromptState(image_prompts)
    segments = ChatSegments()

    tool_calls = None
    tool_index = 0

    if tools:
        segments.extend(_render_tool_declare(tools))

    thinking_effort = kwargs.get("thinking_effort")
    if thinking and thinking_effort is not None:
        if thinking_effort not in _VALID_THINKING_EFFORTS:
            raise XTMLError(
                f"Unsupported thinking_effort={thinking_effort!r}; "
                f"supported values are {sorted(_VALID_THINKING_EFFORTS)}.")
    if thinking and thinking_effort in _VALID_THINKING_EFFORTS:
        segments.extend(_internal_system_message(
            "thinking-effort",
            "`thinking_effort` guides on how much to think in your "
            "thinking channel (not including the response channel), "
            "supported values include `low`, `medium`, `high`, and `max`.\n"
            f"Now the system is invoked with `thinking_effort={thinking_effort}`."))

    # Tools and the thinking-effort note are synthesized ahead of the
    # conversation and are stable across a prompt family.  Contiguous system
    # turns at the start of the caller's conversation extend the same root.
    # A later system turn is conversation history and must stay in the suffix.
    family_root_segments = len(segments)
    leading_system = True

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message["role"]
        if role != "system":
            leading_system = False
        if role == "user":
            attrs = [("role", "user")]
            if message.get("name"):
                attrs.append(("name", message["name"]))
            segments.extend(_open_tag("message", attrs))
            segments.extend(_render_content_segments(message.get("content"),
                                                     image_state))
            segments.extend(_close_tag("message"))
            segments.extend(_end_of_msg())
        elif role == "system" and message.get("tools"):
            segments.extend(_render_tool_declare(message["tools"], dynamic=True))
        elif role == "system":
            attrs = [("role", "system")]
            if message.get("name"):
                attrs.append(("name", message["name"]))
            segments.extend(_open_tag("message", attrs))
            segments.extend(_render_content_segments(message.get("content"),
                                                     image_state))
            segments.extend(_close_tag("message"))
            segments.extend(_end_of_msg())
        elif role == "tool":
            tool_index += 1
            tool_name = message.get("tool", message.get("name"))
            if (tool_name is None and tool_calls is not None
                    and tool_index <= len(tool_calls)):
                tc = tool_calls[tool_index - 1]
                fn = tc.get("function", tc)
                tool_name = fn["name"]
            if tool_name is None:
                raise XTMLError(
                    "Kimi K3 tool messages need a resolvable tool name: "
                    "carry `tool`/`name`, or match a preceding assistant "
                    "tool_call by order.")
            segments.extend(_open_tag("message", [("role", "tool"),
                                                  ("tool", tool_name),
                                                  ("index", tool_index)]))
            segments.extend(_render_content_segments(message.get("content"),
                                                     image_state))
            segments.extend(_close_tag("message"))
            segments.extend(_end_of_msg())
        elif role == "assistant":
            tool_calls = message.get("tool_calls")
            tool_index = 0
            attrs = [("role", "assistant")]
            if message.get("name"):
                attrs.append(("name", message["name"]))
            segments.extend(_open_tag("message", attrs))
            segments.extend(_render_assistant_segments(message, image_state,
                                                       thinking))
            segments.extend(_close_tag("message"))
            segments.extend(_end_of_msg())

        if leading_system:
            family_root_segments = len(segments)

    tool_choice = kwargs.get("tool_choice")
    if tool_choice == "required":
        segments.extend(_internal_system_message(
            "tool-choice",
            "The system is invoked with `tool_choice=required`.\n"
            "You MUST call tools in the next message."))
    elif tool_choice == "none":
        segments.extend(_internal_system_message(
            "tool-choice",
            "The system is invoked with `tool_choice=none`.\n"
            "You MUST NOT call any tools in the next message."))

    rf = kwargs.get("response_format")
    rf_type = _get_value(rf, "type", rf) if isinstance(rf, dict) else rf
    if rf_type == "json_object":
        segments.extend(_internal_system_message(
            "response-format",
            "The system is invoked with `response_format=json_object`.\n"
            "Your response must be raw JSON data without markdown code "
            "blocks (```json) or any additional formatting."))
    elif rf_type == "json_schema":
        schema = _json_compact(kwargs.get("response_schema"))
        segments.extend(_internal_system_message(
            "response-format",
            "The system is invoked with `response_format=json_schema`.\n"
            "Your response must be raw JSON data without markdown code "
            "blocks (```json) or any additional formatting.\n"
            "The JSON data must match the following schema:\n"
            f"```json\n{schema}\n```"))

    if add_generation_prompt:
        # The floor is handed over with an *unclosed* assistant message: the
        # model continues it rather than starting one. Which channel opens
        # decides whether it reasons first or answers first.
        segments.extend(_open_tag("message", [("role", "assistant")]))
        segments.extend(_open_tag("think" if thinking else "response"))

    image_state.assert_consumed()
    segments.family_root_segments = family_root_segments
    return segments


# ---- helpers for callers -------------------------------------------------


def render_text(segments: list[Segment]) -> str:
    """The prompt as one string. For debugging and goldens, not for the engine.

    Two separate reasons this is not how a prompt gets built:

    1. Encoding it in one pass is the mistake the segment list exists to
       prevent — text that happens to contain `<|sep|>` would become
       structure.
    2. Even encoding it in two passes, one per mode, would be wrong.
       Segments are BPE'd *one at a time*, never concatenated first:
       upstream's `_encode_chat_segments` calls the encoder per segment, so
       ` role`, `="`, `user`, `"` are four separate encodes and merging them
       into ` role="user"` yields different token ids — a prompt the model
       was not trained on. Tempting, since rendering emits one segment per
       attribute fragment, and wrong.

    serve.engine.tokenize_segments is the one that does it properly.
    """
    return "".join(s.text for s in segments)
