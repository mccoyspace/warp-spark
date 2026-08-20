#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Regression checks for fp8 block-scale dequantization.

These tests use small synthetic tensors so they do not need model weights. A
missing torch installation is an explicit skip, matching tests/run.sh's rule
that unavailable prerequisites must never look like a pass.

The one case nothing here can catch, stated because it is the reason the tile
size is read from the checkpoint's config rather than inferred from the two
shapes: a *compatible but wrong* block size. 300 rows against 3 scale rows
admits both 128 (the truth, with a partial last tile) and 100 (a clean split).
Both satisfy the shape check below, both produce a tensor of the right size,
and the wrong one applies every scale to the wrong rows. No assertion over
shapes can separate them, which is why `unblock_scale` takes `block` as an
argument instead of deriving it — the check that matters happened before this
file was reached.
"""

import json
import os
import struct
import sys
import tempfile

try:
    import torch
except ImportError:
    print("SKIP: torch is not installed")
    raise SystemExit(77)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from convert import ShardReader
from mxfp4 import ST, unblock_scale


def expected_dequant(q, scale, block):
    """Reference the tile lookup with explicit row and column indices."""
    bm, bn = block
    rows = torch.arange(q.shape[0]) // bm
    cols = torch.arange(q.shape[1]) // bn
    return q.float() * scale[rows[:, None], cols[None, :]]


def write_safetensors_model(root, block, include_scale=True):
    """Write the smallest safetensors model both readers can consume."""
    q = torch.tensor([[1.0, -2.0, 3.0, -4.0, 0.5],
                      [-1.0, 2.0, -3.0, 4.0, -0.5]], dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    tensors = {"weight": ("F8_E4M3", list(q.shape),
                           bytes(q.contiguous().view(torch.uint8).flatten().tolist()))}
    if include_scale:
        tensors["weight_scale_inv"] = ("F32", list(scale.shape),
                                         bytes(scale.contiguous().view(torch.uint8).flatten().tolist()))

    header = {}
    payload = bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        start = len(payload)
        payload.extend(raw)
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [start, len(payload)]}
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    while (8 + len(header_bytes)) % 8:
        header_bytes += b" "
    with open(os.path.join(root, "shard.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)

    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {name: "shard.safetensors" for name in tensors}}, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({"quantization_config": {"weight_block_size": list(block)}}, f)
    return q.float(), scale


def test_aligned_tiles_apply_each_scale():
    q = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    scale = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    result = unblock_scale(q, scale, (2, 2))
    assert torch.equal(result, expected_dequant(q, scale, (2, 2)))


def test_partial_last_row_and_column_are_cropped_after_mapping():
    q = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    scale = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    result = unblock_scale(q, scale, (2, 3))
    assert torch.equal(result, expected_dequant(q, scale, (2, 3)))


def test_missing_scale_companion_is_rejected():
    """Both readers refuse, not just the one that is easy to remember.

    ST and ShardReader each carry their own copy of this guard, and covering
    only ST left the convert.py one free to return the tensor unscaled: a
    mutation that deleted its `raise` kept the whole suite green. That is the
    silent-wrong-answer this file exists to prevent, in the reader #26's own
    description called out as "a second reader and was easy to miss".
    """
    for label, read in (("mxfp4.ST", lambda r: ST(r).tensor("weight")),
                        ("convert.ShardReader", lambda r: ShardReader(r).get("weight"))):
        with tempfile.TemporaryDirectory() as root:
            write_safetensors_model(root, (2, 3), include_scale=False)
            try:
                read(root)
            except KeyError as exc:
                assert "weight_scale_inv" in str(exc), f"{label}: {exc}"
            else:
                raise AssertionError(
                    f"{label} accepted an fp8 tensor with no scale companion")


def test_gross_scale_shape_mismatch_is_rejected():
    q = torch.zeros(3, 5)
    scale = torch.ones(1, 1)
    try:
        unblock_scale(q, scale, (2, 3))
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("a gross fp8 scale shape mismatch was accepted")


def test_safetensor_readers_agree_on_configured_block_size():
    with tempfile.TemporaryDirectory() as root:
        expected_q, expected_scale = write_safetensors_model(root, (2, 3))
        from_st = ST(root).tensor("weight")
        from_converter = ShardReader(root).get("weight")
        expected = expected_dequant(expected_q, expected_scale, (2, 3))
        assert torch.equal(from_st, expected)
        assert torch.equal(from_converter, expected)
        assert torch.equal(from_st, from_converter)


def main():
    tests = [
        test_aligned_tiles_apply_each_scale,
        test_partial_last_row_and_column_are_cropped_after_mapping,
        test_missing_scale_companion_is_rejected,
        test_gross_scale_shape_mismatch_is_rejected,
        test_safetensor_readers_agree_on_configured_block_size,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"PASS: {len(tests)} fp8 block-scale checks")


if __name__ == "__main__":
    main()
