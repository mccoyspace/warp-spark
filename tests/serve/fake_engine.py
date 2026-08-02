# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
fake_engine.py — an engine that says what the test told it to.

The HTTP layer has to be testable without a model. Not because a real one
is slow — the synthetic container in test_engine.py is milliseconds — but
because its weights are noise, so it cannot be made to emit a tool call, a
reasoning channel, or any other reply a test wants to assert about.

So this replays a script. It implements the surface serve/server.py and
serve/api.py actually use, with the same types, and it tokenizes the way
the real one does in the way that matters: markers are their own ids and
ordinary text is not. test_engine.py is what checks the real binding;
this checks everything above it.
"""

from __future__ import annotations

import struct
import threading
from array import array
from dataclasses import dataclass, field
from typing import Callable, Optional

from serve import xtml
from serve.engine import EngineError, WASTE_E_FORMAT, WASTE_E_UNSUPPORTED

MARKERS = {1: xtml.OPEN_TOKEN, 2: xtml.CLOSE_TOKEN,
           3: xtml.SEP_TOKEN, 4: xtml.END_OF_MSG_TOKEN}
_BY_TEXT = {v: k for k, v in MARKERS.items()}

# Ordinary text starts here, so no piece of content can collide with a
# marker id — the same separation the real container gets from its
# reserved block.
_TEXT_BASE = 1000


def split_markers(text: str) -> list[tuple[int, str]]:
    """Cut a reply into (token_id, piece), markers as their own tokens.

    Everything else becomes one token per character, which is not how BPE
    works but is the most demanding shape for a streaming parser: every
    boundary is a chunk boundary.
    """
    out: list[tuple[int, str]] = []
    rest = text
    while rest:
        best, marker = -1, ""
        for m in _BY_TEXT:
            p = rest.find(m)
            if p >= 0 and (best < 0 or p < best):
                best, marker = p, m
        if best < 0:
            out.extend((_TEXT_BASE + ord(c), c) for c in rest)
            break
        out.extend((_TEXT_BASE + ord(c), c) for c in rest[:best])
        out.append((_BY_TEXT[marker], ""))
        rest = rest[best + len(marker):]
    return out


@dataclass
class FakeEngine:
    """A scripted engine. Set `reply` to what the model should say."""

    reply: str = "Hello!"
    vision: bool = False
    ctx_max: int = 4096
    # Raised by generate(), to exercise the error paths.
    fail_with: Optional[Exception] = None
    # Sleep this long before each token, to test client disconnects.
    delay: float = 0.0
    host_reserved_bytes: int = 0
    prefill_chunk: int = 4
    fail_state_import_once: bool = False
    fail_state_export_once: bool = False
    fail_state_export_memory_once: bool = False
    model_path: str = "fake.waste"

    prompts: list[list[int]] = field(default_factory=list)
    run_prompts: list[list[int]] = field(default_factory=list)
    prefills: list[list[int]] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    resets: int = 0
    imports: int = 0
    exports: int = 0
    closed: bool = False
    _state_tokens: list[int] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # ---- what api.py and server.py use ----------------------------------

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def marker_ids(self) -> dict[int, str]:
        return dict(MARKERS)

    def model_info(self) -> dict:
        return {"n_layers": 4, "n_experts": 8, "top_k": 2, "hidden": 128,
                "ctx_max": self.ctx_max, "params_total": 1000,
                "params_active": 100, "arch": "kimi-k3",
                "quant_summary": "experts VQ2R, trunk Q4G"}

    def stats(self) -> dict:
        return {"tokens_generated": 0, "experts_hit": 3, "experts_missed": 1,
                "bytes_read": 4096, "sec_total": 0.1, "sec_io": 0.01,
                "direct_io": 0}

    def memory_used(self) -> dict:
        return {"trunk_bytes": 1, "state_bytes": 1, "scratch_bytes": 1,
                "min_expert_cache": 1, "floor_bytes": 4,
                "recommended_bytes": 8, "vision_bytes": 0,
                "host_reserved_bytes": self.host_reserved_bytes}

    def tokenize(self, text: str, *, markup: bool = False,
                 add_bos: bool = False) -> list[int]:
        if markup:
            return [tid for tid, _ in split_markers(text)]
        # Text mode: markers are ordinary characters, never control ids.
        return [_TEXT_BASE + ord(c) for c in text]

    def tokenize_segments(self, segments, *, add_bos: bool = False) -> list[int]:
        out: list[int] = []
        for s in segments:
            if s.text:
                out.extend(self.tokenize(s.text, markup=s.markup))
        return out

    def detokenize(self, tokens: list[int]) -> str:
        out = []
        for t in tokens:
            if t in MARKERS:
                out.append(MARKERS[t])
            elif t >= _TEXT_BASE:
                out.append(chr(t - _TEXT_BASE))
        return "".join(out)

    def image_clear(self) -> None:
        self.images.clear()

    def image_add(self, path: str) -> int:
        if not self.vision:
            raise EngineError("image_add", WASTE_E_UNSUPPORTED)
        self.images.append(path)
        return 4

    @staticmethod
    def image_dimensions(path: str) -> tuple[int, int]:
        return 640, 480

    def image_expand(self, tokens: list[int]) -> list[int]:
        return list(tokens)

    def state_reset(self) -> None:
        self.resets += 1
        self._state_tokens.clear()

    def prefill_chunk_size(self) -> int:
        return self.prefill_chunk

    def prefill(self, tokens: list[int]) -> None:
        if not tokens:
            raise EngineError("prefill", WASTE_E_FORMAT, "empty prompt")
        self.prefills.append(list(tokens))
        self._state_tokens.extend(tokens)

    def state_size(self) -> int:
        return 8 + 4 * len(self._state_tokens)

    def state_export(self) -> bytearray:
        self.exports += 1
        if self.fail_state_export_memory_once:
            self.fail_state_export_memory_once = False
            raise MemoryError("injected snapshot allocation failure")
        if self.fail_state_export_once:
            self.fail_state_export_once = False
            raise EngineError("state_export", WASTE_E_FORMAT,
                              "injected export failure")
        payload = array("i", self._state_tokens).tobytes()
        return bytearray(struct.pack("<4sI", b"FAKE", len(self._state_tokens))
                         + payload)

    def state_import(self, blob) -> None:
        self.imports += 1
        if self.fail_state_import_once:
            self.fail_state_import_once = False
            raise EngineError("state_import", WASTE_E_FORMAT,
                              "injected mismatch")
        raw = bytes(blob)
        if len(raw) < 8:
            raise EngineError("state_import", WASTE_E_FORMAT,
                              "short snapshot")
        magic, count = struct.unpack("<4sI", raw[:8])
        if magic != b"FAKE" or len(raw) != 8 + count * 4:
            raise EngineError("state_import", WASTE_E_FORMAT,
                              "invalid snapshot")
        restored = array("i")
        restored.frombytes(raw[8:])
        self._state_tokens = list(restored)

    def close(self) -> None:
        self.closed = True

    # ---- generation ------------------------------------------------------

    def generate(self, prompt: list[int], on_token: Callable, *,
                 temperature: float = 0.0, top_p=None, top_k=None, seed=None,
                 max_tokens: int = 512, grammar=None,
                 stop_tokens=None) -> bool:
        import time

        with self._lock:
            if self.fail_with is not None:
                raise self.fail_with
            run_prompt = list(prompt)
            self.run_prompts.append(run_prompt)
            self.prompts.append(self._state_tokens + run_prompt)
            self._state_tokens.extend(run_prompt)
            self.calls.append({"temperature": temperature, "top_p": top_p,
                               "top_k": top_k, "seed": seed,
                               "max_tokens": max_tokens,
                               "stop_tokens": list(stop_tokens or [])})

            tokens = split_markers(self.reply)[:max_tokens]
            for i, (tid, piece) in enumerate(tokens):
                if self.delay:
                    time.sleep(self.delay)
                info = _Info(i, tid)
                if not on_token(tid, piece, info):
                    return False
                if stop_tokens and tid in stop_tokens:
                    return True
            return True


@dataclass
class _Info:
    """Stands in for waste_token_info."""

    token_index: int
    token: int
    logprob: float = 0.0
    experts_hit: int = 0
    experts_missed: int = 0
    bytes_read: int = 0
    ms_total: float = 0.0
    ms_io: float = 0.0


# ---- reply scripts -------------------------------------------------------

def reply_plain(answer: str, *, thinking: bool = True,
                reasoning: str = "") -> str:
    """What the model emits after the generation prompt opened a channel."""
    out = []
    if thinking:
        out.append(reasoning)
        out.append(f"{xtml.CLOSE_TOKEN}think{xtml.SEP_TOKEN}")
        out.append(f"{xtml.OPEN_TOKEN}response{xtml.SEP_TOKEN}")
    out.append(answer)
    out.append(f"{xtml.CLOSE_TOKEN}response{xtml.SEP_TOKEN}")
    out.append(f"{xtml.CLOSE_TOKEN}message{xtml.SEP_TOKEN}")
    out.append(xtml.END_OF_MSG_TOKEN)
    return "".join(out)


def reply_tool_call(name: str, arguments: dict, *, answer: str = "",
                    thinking: bool = True, reasoning: str = "") -> str:
    """A reply that calls one tool, rendered the way K3 renders one."""
    out = []
    if thinking:
        out.append(reasoning)
        out.append(f"{xtml.CLOSE_TOKEN}think{xtml.SEP_TOKEN}")
        out.append(f"{xtml.OPEN_TOKEN}response{xtml.SEP_TOKEN}")
    out.append(answer)
    out.append(f"{xtml.CLOSE_TOKEN}response{xtml.SEP_TOKEN}")
    out.append(f"{xtml.OPEN_TOKEN}tools{xtml.SEP_TOKEN}")
    out.append(f'{xtml.OPEN_TOKEN}call tool="{name}" index="1"{xtml.SEP_TOKEN}')
    for key, value in arguments.items():
        kind = xtml._xtml_type(value)
        out.append(f'{xtml.OPEN_TOKEN}argument key="{key}" '
                   f'type="{kind}"{xtml.SEP_TOKEN}')
        out.append(xtml._xtml_value(value))
        out.append(f"{xtml.CLOSE_TOKEN}argument{xtml.SEP_TOKEN}")
    out.append(f"{xtml.CLOSE_TOKEN}call{xtml.SEP_TOKEN}")
    out.append(f"{xtml.CLOSE_TOKEN}tools{xtml.SEP_TOKEN}")
    out.append(f"{xtml.CLOSE_TOKEN}message{xtml.SEP_TOKEN}")
    out.append(xtml.END_OF_MSG_TOKEN)
    return "".join(out)
