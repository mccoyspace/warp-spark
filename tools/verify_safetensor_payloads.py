#!/usr/bin/env python3
"""Verify every indexed safetensors file is present and structurally whole.

This deliberately uses only the Python standard library so a large source
checkpoint can be gated before the converter environment is started.  It
checks index/header membership, non-overlapping contiguous payload offsets,
and exact file length; tensor dtype/shape semantics remain the converter's
architecture-specific responsibility.
"""

import argparse
import json
import os
import struct


def verify(root, expected_shards=0):
    index_path = os.path.join(root, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as stream:
        index = json.load(stream)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("index has no non-empty weight_map")

    mapped = {}
    for tensor, shard in weight_map.items():
        if not isinstance(tensor, str) or not isinstance(shard, str):
            raise ValueError("weight_map keys and values must be strings")
        mapped.setdefault(shard, set()).add(tensor)
    if expected_shards and len(mapped) != expected_shards:
        raise ValueError(
            f"index names {len(mapped)} shards, expected {expected_shards}")

    total = 0
    tensors = 0
    for shard, expected in sorted(mapped.items()):
        path = os.path.join(root, shard)
        size = os.path.getsize(path)
        with open(path, "rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"{shard}: truncated header length")
            header_size = struct.unpack("<Q", raw)[0]
            if header_size < 2 or header_size > size - 8:
                raise ValueError(f"{shard}: invalid header size {header_size}")
            header = json.loads(stream.read(header_size))

        actual = set(header) - {"__metadata__"}
        if actual != expected:
            missing = sorted(expected - actual)[:3]
            extra = sorted(actual - expected)[:3]
            raise ValueError(
                f"{shard}: index/header mismatch; missing={missing}, extra={extra}")

        intervals = []
        for tensor in actual:
            entry = header[tensor]
            offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
            if (not isinstance(offsets, list) or len(offsets) != 2 or
                    not all(isinstance(value, int) for value in offsets)):
                raise ValueError(f"{shard}: {tensor} has invalid data_offsets")
            begin, end = offsets
            if begin < 0 or end <= begin:
                raise ValueError(f"{shard}: {tensor} has invalid interval {offsets}")
            intervals.append((begin, end, tensor))
        intervals.sort()
        cursor = 0
        for begin, end, tensor in intervals:
            if begin != cursor:
                raise ValueError(
                    f"{shard}: payload discontinuity before {tensor}: "
                    f"{begin} != {cursor}")
            cursor = end
        expected_size = 8 + header_size + cursor
        if size != expected_size:
            raise ValueError(
                f"{shard}: file size {size}, expected {expected_size}")
        total += size
        tensors += len(actual)

    return len(mapped), tensors, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--expected-shards", type=int, default=0)
    args = parser.parse_args()
    shards, tensors, total = verify(args.root, args.expected_shards)
    print(f"SAFETENSORS_PAYLOADS_OK shards={shards} tensors={tensors} bytes={total}")


if __name__ == "__main__":
    main()
