#!/usr/bin/env python3
"""Freeze XTML segments and exact Kimi token IDs into a prompt corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve import xtml


def canonical_segments(segments: list[xtml.Segment]) -> bytes:
    payload = [{"text": segment.text, "markup": segment.markup}
               for segment in segments]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                      sort_keys=True).encode()


def render(corpus: dict, case: dict) -> list[xtml.Segment]:
    rendering = corpus["rendering"]
    if rendering != {
        "function": "serve.xtml.build_chat_segments",
        "add_generation_prompt": True,
        "thinking": False,
        "tools": None,
        "add_bos": False,
        "tokenization": "segment-by-segment markup/plain",
    }:
        raise ValueError("unsupported rendering contract")
    return xtml.build_chat_segments(
        messages=[
            {"role": "system", "content": corpus["system"]},
            {"role": "user", "content": case["user"]},
        ],
        tools=None,
        add_generation_prompt=True,
        thinking=False,
    )


def tokenize(binary: Path, model: Path,
             texts: list[str], *, plain: bool) -> list[list[int]]:
    if not texts:
        return []
    env = dict(os.environ)
    if plain:
        env["WASTE_TOK_PLAIN"] = "1"
    else:
        env.pop("WASTE_TOK_PLAIN", None)
    proc = subprocess.run(
        [str(binary), str(model), *texts],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    lines = proc.stdout.splitlines()
    if len(lines) != len(texts):
        raise RuntimeError(
            f"tokenizer returned {len(lines)} rows for {len(texts)} segments")
    rows: list[list[int]] = []
    for line in lines:
        values = [int(value) for value in line.split()]
        if not values or values[0] != len(values) - 1:
            raise RuntimeError(f"malformed tokenizer row: {line!r}")
        rows.append(values[1:])
    return rows


def freeze(corpus: dict, binary: Path, model: Path) -> dict:
    rendered = [render(corpus, case) for case in corpus["cases"]]
    markup_texts = [segment.text for segments in rendered for segment in segments
                    if segment.text and segment.markup]
    plain_texts = [segment.text for segments in rendered for segment in segments
                   if segment.text and not segment.markup]
    markup_rows = iter(tokenize(binary, model, markup_texts, plain=False))
    plain_rows = iter(tokenize(binary, model, plain_texts, plain=True))

    for case, segments in zip(corpus["cases"], rendered):
        ids: list[int] = []
        for segment in segments:
            if not segment.text:
                continue
            ids.extend(next(markup_rows) if segment.markup else next(plain_rows))
        case["token_ids"] = ids
        case["token_count"] = len(ids)
        case["segments_sha256"] = hashlib.sha256(
            canonical_segments(segments)).hexdigest()

    try:
        next(markup_rows)
        raise RuntimeError("unused markup tokenizer row")
    except StopIteration:
        pass
    try:
        next(plain_rows)
        raise RuntimeError("unused plain tokenizer row")
    except StopIteration:
        pass
    return corpus


def verify(expected_path: Path, binary: Path, model: Path) -> None:
    expected = json.loads(expected_path.read_text())
    actual = freeze(json.loads(expected_path.read_text()), binary, model)
    for expected_case, actual_case in zip(expected["cases"], actual["cases"]):
        for field in ("token_ids", "token_count", "segments_sha256"):
            if actual_case[field] != expected_case[field]:
                raise RuntimeError(
                    f"verification failed for {expected_case['id']}:{field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()

    if args.verify:
        verify(args.verify, args.tokenizer, args.model)
    corpus = freeze(json.loads(args.source.read_text()),
                    args.tokenizer, args.model)
    args.output.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
