#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Build a learned expert hotlist from deterministic route captures.

The input is one or more ``waste.gpu_capture.v1`` manifests written by the
sweep harness through ``WASTE_CAPTURE_JSON``.  Only the embedded route rows
are needed; the companion logit files are deliberately not read.  The output
is a little-endian ``usage.waste`` file accepted by ``waste_ecache_warm``.

Use calibration traffic that represents the prompts the hotlist will serve.
A hotlist built from the benchmark it is tested on is useful diagnostically,
but its hit rate is an in-sample result and should be labelled accordingly.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


CAPTURE_SCHEMA = "waste.gpu_capture.v1"
USAGE_MAGIC = 0x47535557
USAGE_VERSION = 1
USAGE_HEADER = struct.Struct("<IIQ")
USAGE_ENTRY = struct.Struct("<HHIII")
UINT16_MAX = (1 << 16) - 1
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1


class CaptureError(ValueError):
    """A route capture cannot be represented as a usage hotlist."""


@dataclass(frozen=True)
class UsageEntry:
    layer: int
    expert_id: int
    hits: int
    last_seen: int
    next_layer_top: int = 0


@dataclass(frozen=True)
class UsageData:
    total_tokens: int
    selections: int
    entries: tuple[UsageEntry, ...]


def _integer(value: Any, where: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaptureError(f"{where} must be an integer")
    if value < 0 or value > maximum:
        raise CaptureError(f"{where} must be in [0, {maximum}]")
    return value


def _load_manifest(path: os.PathLike[str] | str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"{path}: could not read capture: {exc}") from exc
    if not isinstance(raw, dict):
        raise CaptureError(f"{path}: capture root must be an object")
    if raw.get("schema") != CAPTURE_SCHEMA:
        raise CaptureError(
            f"{path}: schema must be {CAPTURE_SCHEMA!r}, got {raw.get('schema')!r}"
        )
    return raw


def build_usage(paths: Iterable[os.PathLike[str] | str]) -> UsageData:
    """Aggregate route frequency and recency from capture manifests."""

    counts: collections.Counter[tuple[int, int]] = collections.Counter()
    last_seen: dict[tuple[int, int], int] = {}
    total_tokens = 0
    selection_clock = 0
    capture_count = 0

    for path in paths:
        capture_count += 1
        raw = _load_manifest(path)
        top_k = _integer(raw.get("top_k"), f"{path}: top_k", maximum=UINT16_MAX)
        if top_k == 0:
            raise CaptureError(f"{path}: top_k must be positive")
        steps = raw.get("steps")
        if not isinstance(steps, list) or not steps:
            raise CaptureError(f"{path}: steps must be a non-empty list")

        for expected_index, step in enumerate(steps):
            where = f"{path}: steps[{expected_index}]"
            if not isinstance(step, dict):
                raise CaptureError(f"{where} must be an object")
            index = _integer(
                step.get("index"), f"{where}.index", maximum=UINT64_MAX
            )
            if index != expected_index:
                raise CaptureError(
                    f"{where}.index must be contiguous (expected {expected_index})"
                )
            routes = step.get("routes", [])
            if not isinstance(routes, list):
                raise CaptureError(f"{where}.routes must be a list")
            if routes:
                total_tokens += 1
                if total_tokens > UINT64_MAX:
                    raise CaptureError("routed token count exceeds usage.waste format")

            previous_layer = -1
            for route_index, route in enumerate(routes):
                rwhere = f"{where}.routes[{route_index}]"
                if not isinstance(route, dict):
                    raise CaptureError(f"{rwhere} must be an object")
                layer = _integer(
                    route.get("layer"), f"{rwhere}.layer", maximum=UINT16_MAX
                )
                if layer <= previous_layer:
                    raise CaptureError(
                        f"{where}.routes must have unique ascending layers"
                    )
                previous_layer = layer
                experts = route.get("experts")
                if not isinstance(experts, list) or len(experts) != top_k:
                    raise CaptureError(
                        f"{rwhere}.experts must contain exactly top_k={top_k} ids"
                    )
                expert_ids = [
                    _integer(
                        expert,
                        f"{rwhere}.experts[{expert_index}]",
                        maximum=UINT16_MAX,
                    )
                    for expert_index, expert in enumerate(experts)
                ]
                if len(set(expert_ids)) != len(expert_ids):
                    raise CaptureError(f"{rwhere}.experts contains duplicate ids")

                for expert_id in expert_ids:
                    selection_clock += 1
                    if selection_clock > UINT32_MAX:
                        raise CaptureError(
                            "expert selection count exceeds usage.waste recency field"
                        )
                    key = (layer, expert_id)
                    counts[key] += 1
                    if counts[key] > UINT32_MAX:
                        raise CaptureError(
                            f"hit count for layer {layer}, expert {expert_id} "
                            "exceeds usage.waste format"
                        )
                    last_seen[key] = selection_clock

    if capture_count == 0:
        raise CaptureError("at least one capture is required")
    if selection_clock == 0:
        raise CaptureError("captures contain no routed expert selections")

    entries = tuple(
        UsageEntry(layer, expert_id, hits, last_seen[(layer, expert_id)])
        for (layer, expert_id), hits in sorted(
            counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )
    )
    return UsageData(total_tokens, selection_clock, entries)


def encode_usage(data: UsageData) -> bytes:
    """Encode ``UsageData`` using the packed structs in waste_format.h."""

    chunks = [USAGE_HEADER.pack(USAGE_MAGIC, USAGE_VERSION, data.total_tokens)]
    chunks.extend(
        USAGE_ENTRY.pack(
            entry.layer,
            entry.expert_id,
            entry.hits,
            entry.last_seen,
            entry.next_layer_top,
        )
        for entry in data.entries
    )
    return b"".join(chunks)


def write_usage(
    path: os.PathLike[str] | str, data: UsageData, *, force: bool = False
) -> None:
    """Write one complete usage file, refusing overwrite unless requested."""

    mode = "wb" if force else "xb"
    try:
        with open(path, mode) as stream:
            stream.write(encode_usage(data))
    except FileExistsError as exc:
        raise CaptureError(f"{path}: already exists (use --force to replace it)") from exc
    except OSError as exc:
        raise CaptureError(f"{path}: could not write usage file: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="convert WASTE route capture JSON to usage.waste"
    )
    parser.add_argument(
        "captures",
        nargs="+",
        metavar="CAPTURE.json",
        help="waste.gpu_capture.v1 manifest (repeat captures may be combined)",
    )
    parser.add_argument("-o", "--output", required=True, help="output usage.waste")
    parser.add_argument(
        "--force", action="store_true", help="replace an existing output file"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        data = build_usage(Path(path) for path in args.captures)
        write_usage(args.output, data, force=args.force)
    except CaptureError as exc:
        print(f"capture_to_usage: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {args.output}: {len(data.entries)} experts, "
        f"{data.total_tokens} routed tokens, {data.selections} selections"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
