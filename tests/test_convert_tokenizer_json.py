#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Focused checks for tokenizer.json -> WARP rank-file conversion.

The quantizer is irrelevant here, so torch/mxfp4 are stubbed before importing
convert.py.  The release-scale differential lives outside this unit test; these
checks make the format boundary and native-rank precedence cheap in CI.
"""

import base64
import json
import os
import sys
import tempfile
import types
import unittest


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_convert():
    torch = types.ModuleType("torch")
    torch.device = lambda value: value
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False))
    sys.modules.setdefault("torch", torch)
    mx = types.ModuleType("mxfp4")
    mx.ST = object
    mx.unblock_scale = lambda value, scale, block: value
    sys.modules.setdefault("mxfp4", mx)
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import convert
    return convert


CONVERT = load_convert()


def tokenizer_doc():
    # ByteLevel alphabet: Ġ is space, Ċ is newline, and Ã© is UTF-8 "é".
    return {
        "version": "1.0",
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split",
                 "pattern": {"Regex": CONVERT._HF_BYTELEVEL_PATTERN},
                 "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False,
                 "trim_offsets": True, "use_regex": False},
            ],
        },
        "decoder": {"type": "ByteLevel", "add_prefix_space": True,
                    "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None,
            "continuing_subword_prefix": None, "end_of_word_suffix": None,
            "fuse_unk": False, "byte_fallback": False,
            "ignore_merges": True,
            "vocab": {"!": 0, "Ġ": 1, "Ċ": 2, "Ã©": 3},
            "merges": [],
        },
        "added_tokens": [
            {"id": 4, "content": "<|assistant|>", "special": True},
            {"id": 5, "content": "/nothink", "special": True},
        ],
    }


def read_ranks(path):
    rows = []
    with open(path, "rb") as inp:
        for line in inp:
            token, rank = line.split()
            rows.append((base64.b64decode(token), int(rank)))
    return rows


class TokenizerJsonConversionTest(unittest.TestCase):
    def test_bytelevel_vocab_and_added_tokens_are_published(self):
        with tempfile.TemporaryDirectory(prefix="warp-hf-tok-") as tmp:
            src, out = os.path.join(tmp, "src"), os.path.join(tmp, "out")
            os.mkdir(src)
            os.mkdir(out)
            with open(os.path.join(src, "tokenizer.json"), "w") as f:
                json.dump(tokenizer_doc(), f, ensure_ascii=False)

            texts = CONVERT.install_tokenizer(src, out)

            self.assertEqual(
                read_ranks(os.path.join(out, "tokenizer.model")),
                [(b"!", 0), (b" ", 1), (b"\n", 2),
                 ("é".encode(), 3)])
            self.assertEqual(texts, {"<|assistant|>", "/nothink"})
            with open(os.path.join(out, "specials.json")) as f:
                self.assertEqual(json.load(f), [
                    {"id": 4, "text": "<|assistant|>"},
                    {"id": 5, "text": "/nothink"},
                ])

    def test_native_rank_file_keeps_precedence(self):
        with tempfile.TemporaryDirectory(prefix="warp-native-tok-") as tmp:
            src, out = os.path.join(tmp, "src"), os.path.join(tmp, "out")
            os.mkdir(src)
            os.mkdir(out)
            native = b"IQ== 0\n"
            with open(os.path.join(src, "tiktoken.model"), "wb") as f:
                f.write(native)
            # Deliberately unsupported: it must not reinterpret a release
            # that already supplied WARP's native rank format.
            with open(os.path.join(src, "tokenizer.json"), "w") as f:
                json.dump({"model": {"type": "WordPiece"}}, f)

            self.assertEqual(CONVERT.install_tokenizer(src, out), set())
            with open(os.path.join(out, "tokenizer.model"), "rb") as f:
                self.assertEqual(f.read(), native)

    def test_unsupported_shape_fails_without_partial_output(self):
        with tempfile.TemporaryDirectory(prefix="warp-bad-tok-") as tmp:
            src = os.path.join(tmp, "tokenizer.json")
            dst = os.path.join(tmp, "tokenizer.model")
            doc = tokenizer_doc()
            doc["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = ".+"
            with open(src, "w") as f:
                json.dump(doc, f)

            with self.assertRaisesRegex(ValueError, "not the qualified"):
                CONVERT.convert_tokenizer_json(src, dst)
            self.assertFalse(os.path.exists(dst))
            self.assertFalse(os.path.exists(dst + ".tmp"))


if __name__ == "__main__":
    unittest.main()
