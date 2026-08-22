#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
verify_container.py — read a WASTE container back and check it against the
source weights. This is the test that the format actually works: it parses
records with the exact byte layout of src/waste_format.h, so a mismatch in
header size, offsets, alignment or codebook indexing shows up here.

  uv run --with torch python tools/verify_container.py \
      --container /path/model.waste --src /Volumes/WasteDisk/kimi-linear
"""

import argparse
import json
import os
import struct
import sys
import zlib

import torch

MAGIC_EXPERT = 0x50584557
MAGIC_CODEBOOK = 0x4B424357
ALIGN = 4096
VEC_DIM = 8
CB_ENTRIES = 256
IDX_BLOCK = 64
HDR = "<IHHBBHHHIIIIIIII"        # must match waste_expert_hdr
HDR_SIZE = 48
KINDS = (("gate", "w1"), ("up", "w3"), ("down", "w2"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxfp4 import ST                                              # noqa: E402
from convert import moe_layout                                    # noqa: E402


def load_codebooks(path):
    """codebooks.bin -> list of [CB_ENTRIES, VEC_DIM] tensors, in file order."""
    books, data = [], open(path, "rb").read()
    rec = 16 + CB_ENTRIES * VEC_DIM * 2
    for off in range(0, len(data), rec):
        magic, _cid, _fmt, vdim, n, _r = struct.unpack("<IHBBII", data[off:off + 16])
        assert magic == MAGIC_CODEBOOK, f"bad codebook magic at {off}"
        assert vdim == VEC_DIM and n == CB_ENTRIES
        t = torch.frombuffer(bytearray(data[off + 16:off + rec]),
                             dtype=torch.float16).view(n, vdim).float()
        books.append(t)
    return books


def read_expert(bank_bytes, rec_off, books, cb_base, stages, shapes, block=0):
    h = struct.unpack(HDR, bank_bytes[rec_off:rec_off + HDR_SIZE])
    (magic, layer, eid, fmt, flags, cb_id, lowrank_id, _r0,
     blocks, g_off, u_off, d_off, corr_off, crc, _r1, _r2) = h
    assert magic == MAGIC_EXPERT, f"bad expert magic at {rec_off:#x}"
    assert lowrank_id == 0, "v0 requires lowrank_id == 0"
    assert cb_id == cb_base, f"codebook base mismatch {cb_id} != {cb_base}"

    end = rec_off + blocks * ALIGN
    body = bank_bytes[rec_off + HDR_SIZE:end]
    # crc covers the body up to the padding
    payload_len = corr_off - HDR_SIZE + sum(s[0] for s in shapes) * 2
    assert zlib.crc32(bytes(body[:payload_len])) & 0xFFFFFFFF == crc, "CRC mismatch"

    out, scale_cursor = {}, corr_off - HDR_SIZE
    offs = {"gate": g_off, "up": u_off, "down": d_off}
    for i, (kind, _tag) in enumerate(KINDS):
        M, N = shapes[i]
        nvec = M * N // VEC_DIM
        beg = offs[kind] - HDR_SIZE
        raw = torch.frombuffer(bytearray(body[beg:beg + nvec * stages]),
                               dtype=torch.uint8)
        if block:                       # [M/B][nvr][B][stage] -> [nvec][stage]
            nvr = N // VEC_DIM
            nb = (M + block - 1) // block
            idx = (raw.view(nb, nvr, block, stages).permute(0, 2, 1, 3)
                      .reshape(nb * block, nvr, stages)[:M]
                      .reshape(nvec, stages).long())
        else:
            idx = raw.view(nvec, stages).long()
        recon = torch.zeros(nvec, VEC_DIM)
        for s in range(stages):
            recon += books[cb_base + i * stages + s][idx[:, s]]
        sc = torch.frombuffer(bytearray(body[scale_cursor:scale_cursor + M * 2]),
                              dtype=torch.float16).float().view(M, 1)
        scale_cursor += M * 2
        out[kind] = recon.view(M, N) * sc
    return out, blocks, eid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--src", default="/Volumes/WasteDisk/kimi-linear")
    ap.add_argument("--experts", type=int, default=4, help="how many to check")
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.container, "manifest.json")))
    stages = man["expert_quant"]["stages"]
    books = load_codebooks(os.path.join(args.container, "codebooks.bin"))
    print(f"container: {man['expert_quant']['fmt']}, {len(books)} codebooks, "
          f"layers {list(man['layers'])}")

    sr = ST(args.src)
    prefix = man.get("tensor_prefix", "")
    ok = True
    for lstr, meta in man["layers"].items():
        L = int(lstr)
        bank = open(os.path.join(args.container, meta["file"]), "rb").read()
        assert len(bank) == meta["bytes"]
        layout, segment, source_kinds = moe_layout(sr, prefix, L)
        assert layout is not None, f"no supported source expert layout at layer {L}"
        shapes = []
        for _kind, tag in source_kinds:
            t = sr.tensor(
                f"{prefix}model.layers.{L}.{segment}.experts.0.{tag}.weight")
            shapes.append(tuple(t.shape))

        off, checked = 0, 0
        while off < len(bank) and checked < args.experts:
            rec, blocks, eid = read_expert(bank, off, books,
                                           meta["codebook_base"], stages, shapes,
                                           man["expert_quant"].get("index_block", 0))
            assert off % ALIGN == 0, f"record {eid} not 4 KiB aligned"
            for i, (kind, tag) in enumerate(source_kinds):
                W = sr.tensor(
                    f"{prefix}model.layers.{L}.{segment}.experts.{eid}.{tag}.weight")
                err = (W - rec[kind]).norm() / W.norm()
                flag = "ok " if err < 0.30 else "BAD"
                if err >= 0.30:
                    ok = False
                print(f"  L{L} e{eid:<3} {kind:<5} {tuple(W.shape)} "
                      f"rel err {err:>6.2%}  {flag}")
            off += blocks * ALIGN
            checked += 1
        print(f"  layer {L}: {len(bank)//ALIGN} blocks, "
              f"{meta['experts']} experts, {len(bank)/2**20:.1f} MB, "
              f"{len(bank)/meta['experts']/2**20:.2f} MB/expert")

    print("\nPASS — container round-trips" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
