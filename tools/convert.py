#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
convert.py — safetensors (Kimi family) -> WASTE container.

Writes the layout in docs/FORMAT.md with records binary-compatible with
src/waste_format.h: 4 KiB-aligned expert records holding gate/up/down
adjacently so one pread() yields a whole expert.

Experts use VQ3R by default: 3-stage residual VQ over 8-dim vectors,
256 entries per stage (3.0 bits/weight), plus one FP16 scale per output
channel. Codebooks are trained per (layer, matrix kind) on a sample and
stored once. Gate 3 measured this recipe on real Kimi experts.

Reads a Kimi checkpoint as published — the 1.42 TB of moonshotai/Kimi-K3
that tools/fetch_weights.sh leaves on the staging disk, or any other
member of the family (Kimi-Linear) by pointing --src elsewhere.

  uv run --with torch python tools/convert.py \
      --src /path/to/hf-checkpoint \
      --out model.waste \
      --layers 1,2                       # subset for a fast first pass

Resumable: a layer whose bank file already exists is skipped. Never holds
more than one layer of experts in memory.
"""

import argparse
import io
import json
import os
import shutil
import struct
import sys
import time
import zlib

import torch


def atomic_copyfile(src, dst):
    """Copy a published sidecar without exposing a partially written file."""
    tmp = dst + ".tmp"
    with open(src, "rb") as inp, open(tmp, "wb") as out:
        while True:
            chunk = inp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, dst)


def atomic_json(path, value):
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as out:
        json.dump(value, out, indent=1)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def atomic_text(path, value):
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as out:
        out.write(value)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxfp4 import ST                                            # noqa: E402

# --- native VQ encoder (optional; ~15x the torch path) --------------------
_VQ = None


def _load_vq():
    """libwastevq: the assign fused with the argmin, so the [n, 256] distance
    matrix never exists. torch has to materialize it — 1.4 GB per stage for
    one expert matrix — which is what made conversion memory-bound."""
    global _VQ
    if _VQ is not None:
        return _VQ or None
    import ctypes
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("libwastevq.dylib", "libwastevq.so", "libwastevq.dll"):
        path = os.path.join(here, name)
        if os.path.exists(path):
            lib = ctypes.CDLL(path)
            lib.waste_vq_encode.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
            lib.waste_vq_encode.restype = None
            _VQ = (lib, ctypes)
            return _VQ
    _VQ = False
    return None

MAGIC_EXPERT = 0x50584557        # 'WEXP'
MAGIC_CODEBOOK = 0x4B424357      # 'WCBK'
ALIGN = 4096
FMT_F32, FMT_F16, FMT_Q8G, FMT_Q4G, FMT_VQ3R, FMT_VQ2R = 0, 1, 2, 3, 4, 5
FMT_Q3G = 7
VEC_DIM = 8
CB_ENTRIES = 256
TRAIN_VECTORS = 300000       # vectors k-means sees per (layer, matrix kind)
IDX_BLOCK = 64          # rows per index block; matches VQ_TILE in the engine
KINDS = (("gate", "w1"), ("up", "w3"), ("down", "w2"))   # Mixtral naming


# ---------------------------------------------------------------- reading --

class ShardReader:
    """Lazy safetensors reader over a sharded model directory."""

    def __init__(self, model_dir):
        self.dir = model_dir
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        self.wm = idx["weight_map"]
        self._hdr = {}

    def _header(self, fn):
        if fn not in self._hdr:
            with open(os.path.join(self.dir, fn), "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                self._hdr[fn] = (json.loads(f.read(hlen)), 8 + hlen)
        return self._hdr[fn]

    def names(self):
        return self.wm.keys()

    def get(self, name):
        fn = self.wm[name]
        hdr, base = self._header(fn)
        meta = hdr[name]
        beg, end = meta["data_offsets"]
        with open(os.path.join(self.dir, fn), "rb") as f:
            f.seek(base + beg)
            raw = f.read(end - beg)
        dt = {"BF16": torch.bfloat16, "F16": torch.float16,
              "F32": torch.float32}[meta["dtype"]]
        return torch.frombuffer(bytearray(raw), dtype=dt).view(*meta["shape"]).float()


# ------------------------------------------------------------ quantizers --

def train_codebooks(X, n_stages, dev, iters=10, sample=300000, seed=0):
    """Residual k-means: one codebook per stage, fitted on the running
    residual. X is [n, VEC_DIM] on `dev`."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    books = []
    resid = X[torch.randperm(X.shape[0], generator=g)[:sample]].to(dev)
    for _ in range(n_stages):
        C = resid[torch.randperm(resid.shape[0], generator=g)[:CB_ENTRIES]].clone()
        for _ in range(iters):
            idx = assign(resid, C)
            for j in range(CB_ENTRIES):
                m = idx == j
                if m.any():
                    C[j] = resid[m].mean(0)
        books.append(C)
        resid = resid - C[assign(resid, C)]
    return books


def assign(X, C, chunk=1 << 20):
    """Nearest centroid via the ||x||^2 - 2x.c + ||c||^2 expansion (a GEMM)."""
    cn = (C * C).sum(1)
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk]
        d = cn.unsqueeze(0) - 2.0 * (x @ C.T)
        out[s:s + chunk] = d.argmin(1)
    return out


def block_indices(idx, M, N, stages):
    """Reorder [stages, nvec] -> [M/B][v][row_in_block][stage].

    The engine walks a tile of rows for one vector position at a time; in
    row-major order those rows sit N/8*stages bytes apart, so each one is a
    separate cache line. Blocked, a tile's indices for a given position are
    contiguous. Measured 1.44x on the gather loop.
    """
    nvr = N // VEC_DIM
    t = idx.view(stages, M, nvr).permute(1, 2, 0)            # [M, nvr, st]
    pad = (-M) % IDX_BLOCK
    if pad:
        t = torch.cat([t, torch.zeros(pad, nvr, stages, dtype=t.dtype)], 0)
    nb = t.shape[0] // IDX_BLOCK
    t = t.view(nb, IDX_BLOCK, nvr, stages).permute(0, 2, 1, 3)
    return t.contiguous()


def quantize_vq(W, books, dev):
    """W [out, in] -> (indices uint8 [stages, nvec], per-channel fp16 scale)."""
    M, N = W.shape
    scale = W.abs().amax(-1, keepdim=True).clamp(min=1e-8)

    vq = _load_vq()
    if vq is not None:
        lib, ctypes = vq
        X = (W / scale).reshape(-1, VEC_DIM).contiguous().float()
        B = torch.stack([C.detach().cpu().float() for C in books]).contiguous()
        n, st = X.shape[0], len(books)
        out = torch.empty(n * st, dtype=torch.uint8)
        fp = ctypes.POINTER(ctypes.c_float)
        # nthreads=0 means "every core", capped at 64 — right on a laptop,
        # thrash on a many-core box with --jobs > 1. Size the native pool to
        # the same per-worker share as torch's intra-op pool (set in main()).
        nthreads = int(os.environ.get("OMP_NUM_THREADS") or 0)
        lib.waste_vq_encode(
            ctypes.cast(X.data_ptr(), fp), n,
            ctypes.cast(B.data_ptr(), fp), st, CB_ENTRIES, VEC_DIM,
            ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_uint8)), nthreads)
        return out.view(n, st).T.contiguous(), scale.half().flatten().cpu()

    X = (W / scale).to(dev).reshape(-1, VEC_DIM)
    idxs, resid = [], X
    for C in books:
        i = assign(resid, C)
        idxs.append(i.to(torch.uint8).cpu())
        resid = resid - C[i]
    return torch.stack(idxs), scale.half().flatten().cpu()


def quantize_q3g(W, group=128):
    """3 bits per weight, packed per row, fp16 scale per group.

    Values live in [-4, 3] biased by +4 into 0..7, written LSB-first at bit
    offset 3*i within the row, and each row is padded to a fixed stride
    that includes one guard byte so the decoder's two-byte read is always
    in bounds. The engine uses the identical indexing.

    Packing the whole tensor as one stream instead of per row is the
    obvious shortcut and it is wrong: the decoder addresses rows by a fixed
    stride, and with a guard byte in that stride every row after the first
    lands one byte off. It is invisible at 4 bits, where a row is a whole
    number of bytes with no guard needed.
    """
    orig = W.shape
    X = W.reshape(-1, orig[-1])
    rows, N = X.shape
    pad = (-N) % group
    if pad:
        X = torch.nn.functional.pad(X, (0, pad))
    ng = X.shape[1] // group
    n = ng * group
    Xg = X.view(rows, ng, group)
    scale = Xg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 3.0
    u = (torch.clamp(torch.round(Xg / scale), -4, 3).to(torch.int32) + 4).view(rows, n)

    rowbytes = (n * 3 + 7) // 8 + 1                   # + guard
    buf = torch.zeros(rows, rowbytes, dtype=torch.int32)
    off = torch.arange(n) * 3
    byte, shift = off // 8, off % 8
    idx = byte.unsqueeze(0).expand(rows, n)
    lo = ((u << shift) & 0xFF).to(torch.int32).contiguous()
    buf.scatter_add_(1, idx.contiguous(), lo)
    hi = torch.where(shift > 5, (u >> (8 - shift)) & 0xFF,
                     torch.zeros_like(u)).to(torch.int32).contiguous()
    buf.scatter_add_(1, (byte + 1).unsqueeze(0).expand(rows, n).contiguous(), hi)
    return buf.to(torch.uint8).flatten(), scale.half().flatten(), list(orig)


def quantize_q4g(W, group=128):
    """int4 packed two per byte (low nibble first), fp16 scale per group.

    The trunk is the RAM floor and K3's is 54 B parameters — at 8 bits that
    is 54 GB, over the budget of the machine this is meant to run on. At
    4 bits it is 27."""
    orig = W.shape
    X = W.reshape(-1, orig[-1])
    N = X.shape[-1]
    pad = (-N) % group
    if pad:
        X = torch.nn.functional.pad(X, (0, pad))
    Xg = X.view(X.shape[0], -1, group)
    scale = Xg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7.0
    Q = torch.clamp(torch.round(Xg / scale), -8, 7).to(torch.int32).flatten()
    nib = (Q + 8).to(torch.uint8) & 0x0F                 # 0..15, biased
    packed = (nib[0::2] | (nib[1::2] << 4)).contiguous()
    return packed, scale.half().flatten(), list(orig)


def quantize_q8g(W, group=128):
    """int8 + fp16 scale per group of `group` inputs. Returns (bytes, meta)."""
    orig = W.shape
    X = W.reshape(-1, orig[-1])
    N = X.shape[-1]
    pad = (-N) % group
    if pad:
        X = torch.nn.functional.pad(X, (0, pad))
    Xg = X.view(X.shape[0], -1, group)
    scale = Xg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
    Q = torch.clamp(torch.round(Xg / scale), -127, 127).to(torch.int8)
    return Q.flatten(), scale.half().flatten(), list(orig)


# ---------------------------------------------------------------- writing --

def raw_bytes(t):
    """Contiguous little-endian bytes of a tensor (torch has no .tobytes())."""
    t = t.detach().cpu().contiguous()
    n = t.numel() * t.element_size()
    buf = torch.empty(n, dtype=torch.uint8)
    buf.view(t.dtype)[:t.numel()] = t.flatten()
    return bytes(memoryview(buf.numpy() if False else bytearray(buf.tolist())))

def write_expert_record(f, layer, eid, cb_base, payloads, scales, shapes):
    """One 4 KiB-aligned WEXP record: header, then gate|up|down indices,
    then the per-channel scales for all three."""
    hdr_size = 48
    off = hdr_size
    offsets = []
    body = bytearray()
    for i, p in enumerate(payloads):          # [stages, nvec] uint8
        offsets.append(off)
        M, N = shapes[i]
        b = raw_bytes(block_indices(p, M, N, p.shape[0]))
        body += b
        off += len(b)
    corr_off = off
    for s in scales:
        b = raw_bytes(s)
        body += b
        off += len(b)

    total = hdr_size + len(body)
    blocks = (total + ALIGN - 1) // ALIGN
    pad = blocks * ALIGN - total
    crc = zlib.crc32(bytes(body)) & 0xFFFFFFFF

    hdr = struct.pack("<IHHBBHHHIIIIIIII",
                      MAGIC_EXPERT, layer, eid, FMT_VQ3R, 0, cb_base, 0, 0,
                      blocks, offsets[0], offsets[1], offsets[2], corr_off,
                      crc, 0, 0)
    assert len(hdr) == hdr_size, len(hdr)
    f.write(hdr)
    f.write(body)
    f.write(b"\0" * pad)
    return blocks


def bank_codebook_base(path):
    """The codebook base a finished bank was written with, or -1.

    Every record carries it absolutely — the engine reads codebook_id from
    the record, not from the manifest — so a bank that already exists has a
    base that is not ours to reassign. Reading it back is what lets a run
    interrupted before it published a manifest still resume.
    """
    try:
        with open(path, "rb") as f:
            hdr = f.read(12)
    except OSError:
        return -1
    if len(hdr) < 12 or struct.unpack_from("<I", hdr, 0)[0] != MAGIC_EXPERT:
        return -1
    return struct.unpack_from("<H", hdr, 10)[0]


# ------------------------------------------------------------- worker ----

def convert_layer(job):
    """One layer, in its own process. Layers share nothing: separate bank
    file and separate codebook file. The parent decides the base — from the
    published manifest, from an existing bank's own records, or new after
    the published record count — and never renumbers a bank that exists."""
    (L, src, out, prefix, n_exp, stages, device, cb_sample, cb_base,
     cached_ok) = job
    import time as _t
    bank = os.path.join(out, f"experts-L{L}.bin")
    cbf = os.path.join(out, f"codebooks-L{L}.bin")
    # Whether a finished bank counts as done is the parent's call: it is the
    # only side that can see the manifest, the merged codebooks and this
    # bank's own base together. Deciding it here — from the bank's existence
    # alone — is what used to renumber a resumed layer's codebooks.
    if cached_ok and os.path.exists(bank):
        return (L, os.path.getsize(bank), cb_base, "cached")

    st = ST(src)
    dev = torch.device(device)

    def ename(e, tag):
        return f"{prefix}model.layers.{L}.block_sparse_moe.experts.{e}.{tag}.weight"

    if not (st.have(ename(0, "w1")) or st.have(ename(0, "w1") + "_packed")):
        return (L, 0, cb_base, "missing")

    t0 = _t.time()
    shapes = [tuple(st.tensor(ename(0, tag)).shape) for _, tag in KINDS]

    books, sample_ids = {}, list(range(0, n_exp, max(1, n_exp // cb_sample)))[:cb_sample]
    per = max(1, TRAIN_VECTORS // len(sample_ids))
    with open(cbf + ".tmp", "wb") as cf:
        for ki, (kind, tag) in enumerate(KINDS):
            chunks = []
            for e in sample_ids:
                W = st.tensor(ename(e, tag))
                sc = W.abs().amax(-1, keepdim=True).clamp(min=1e-8)
                V = (W / sc).reshape(-1, VEC_DIM)
                g = torch.Generator().manual_seed(1234 + e)
                chunks.append(V[torch.randperm(V.shape[0], generator=g)[:per]])
                del W, V
            X = torch.cat(chunks); del chunks
            books[kind] = train_codebooks(X, stages, dev, sample=TRAIN_VECTORS)
            del X
            for si, C in enumerate(books[kind]):
                cid = cb_base + ki * stages + si
                cf.write(struct.pack("<IHBBII", MAGIC_CODEBOOK, cid & 0xFFFF,
                                     FMT_VQ3R, VEC_DIM, CB_ENTRIES, 0))
                cf.write(raw_bytes(C.cpu().half()))
    os.replace(cbf + ".tmp", cbf)

    with open(bank + ".tmp", "wb") as f:
        for e in range(n_exp):
            payloads, scales = [], []
            for kind, tag in KINDS:
                W = st.tensor(ename(e, tag))
                idx, sc = quantize_vq(W, books[kind], dev)
                payloads.append(idx); scales.append(sc)
                del W
            write_expert_record(f, L, e, cb_base, payloads, scales, shapes)
    os.replace(bank + ".tmp", bank)
    return (L, os.path.getsize(bank), cb_base, f"{_t.time()-t0:.0f}s")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="HF checkpoint directory, as published")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="", help="comma list; default = all MoE layers")
    ap.add_argument("--stages", type=int, choices=(2, 3), default=3,
                    help="3 = VQ3R, 2 = VQ2R")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--experts", type=int, default=0, help="limit experts (debug)")
    ap.add_argument("--trunk-bits", type=int, default=4, choices=(3, 4, 8),
                    help="bit width for the bulk of the trunk; it is the RAM "
                         "floor, so this is the main lever on cache size")
    ap.add_argument("--trunk8", action="store_true",
                    help="keep the whole trunk at 8 bits (needs the RAM)")
    ap.add_argument("--skip-trunk", action="store_true",
                    help="experts only; keeps the trunk and the trunk index "
                         "the existing manifest describes, and republishes "
                         "the manifest with the layers this run converted")
    ap.add_argument("--jobs", type=int, default=3,
                    help="layers converted in parallel; measured sweet spot "
                         "is 3 — beyond that the native encoder is already "
                         "using every core")
    ap.add_argument("--cb-sample", type=int, default=12,
                    help="experts sampled per layer to fit the codebooks")
    args = ap.parse_args()
    if args.jobs < 1:
        ap.error("--jobs must be at least 1")
    if args.cb_sample < 1:
        ap.error("--cb-sample must be at least 1")

    # torch sizes its intra-op pool from os.cpu_count() by default, so N
    # worker processes would otherwise spawn N*cpus threads and thrash each
    # other off the machine. Cap each worker at its fair share of the cores;
    # the env is inherited by the spawn'd workers before they import torch,
    # which is the only point the setting is read. setdefault leaves an
    # explicit OMP_NUM_THREADS the caller's to control.
    per = max(1, (os.cpu_count() or 1) // args.jobs)
    os.environ.setdefault("OMP_NUM_THREADS", str(per))
    os.environ.setdefault("MKL_NUM_THREADS", str(per))

    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    prefix = ""
    if "text_config" in cfg:                     # K3 nests the text model
        cfg = {**cfg["text_config"], "_outer": {k: v for k, v in cfg.items()
                                                if k != "text_config"}}
        prefix = "language_model."
    st = ST(args.src)
    sr = ShardReader(args.src)
    dev = torch.device(args.device)
    print(f"device={dev}  stages={args.stages}")

    n_layers = cfg["num_hidden_layers"]
    n_exp = cfg.get("num_experts") or cfg.get("n_routed_experts")
    print(f"prefix {prefix!r}  layers {n_layers}  experts {n_exp}")
    first_dense = cfg.get("first_k_dense_replace", 0)
    if args.experts:
        n_exp = min(n_exp, args.experts)
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(first_dense, n_layers)))

    # ---- expert banks, one layer at a time ------------------------------
    manifest_path = os.path.join(args.out, "manifest.json")
    existing = None
    if os.path.exists(manifest_path):
        try:
            existing = json.load(open(manifest_path))
        except (OSError, ValueError):
            existing = None

    cb_record_bytes = struct.calcsize("<IHBBII") + CB_ENTRIES * VEC_DIM * 2
    merged = os.path.join(args.out, "codebooks.bin")
    old_books = 0
    old_quant = existing.get("expert_quant", {}) if existing else {}
    compatible = bool(
        existing and os.path.exists(merged) and
        old_quant.get("stages") == args.stages and
        old_quant.get("vec_dim") == VEC_DIM and
        old_quant.get("entries") == CB_ENTRIES and
        os.path.getsize(merged) % cb_record_bytes == 0)
    if compatible:
        old_books = os.path.getsize(merged) // cb_record_bytes
        manifest_layers = dict(existing.get("layers", {}))
    else:
        manifest_layers = {}


    def ename(L, e, tag):
        return f"{prefix}model.layers.{L}.block_sparse_moe.experts.{e}.{tag}.weight"

    n_cb_per_layer = 3 * args.stages
    next_base = old_books
    jobs = []
    for L in layers:
        meta = manifest_layers.get(str(L), {})
        bank = os.path.join(args.out, f"experts-L{L}.bin")
        part = os.path.join(args.out, f"codebooks-L{L}.bin")
        base = meta.get("codebook_base", -1)
        cached_ok = bool(
            compatible and isinstance(base, int) and base >= 0 and
            base + n_cb_per_layer <= old_books and
            meta.get("experts") == n_exp and os.path.exists(bank) and
            meta.get("bytes") == os.path.getsize(bank))
        if not cached_ok:
            # A run interrupted before it published anything leaves no
            # manifest and no codebooks.bin, so there is nothing to read a
            # base from — and re-deriving one positionally is the bug this
            # scheme exists to avoid. But a finished bank names its own
            # base in every record it holds, and its unmerged codebook part
            # is still on disk beside it. Together those are a complete,
            # self-describing layer, so honour them.
            recovered = bank_codebook_base(bank)
            # Bounded by the whole model, not by this invocation's --layers:
            # the base in the bank reflects the run that wrote it, which may
            # have been converting far more layers than this one is.
            if (old_books <= recovered <=
                    old_books + n_layers * n_cb_per_layer and
                    os.path.exists(part) and
                    os.path.getsize(part) == n_cb_per_layer * cb_record_bytes
                    and os.path.getsize(bank) > 0):
                base = recovered
                cached_ok = True
        source_ok = (st.have(ename(L, 0, "w1")) or
                     st.have(ename(L, 0, "w1") + "_packed"))
        if not cached_ok:
            base = next_base
            if source_ok:
                next_base += n_cb_per_layer
        elif base + n_cb_per_layer > next_base:
            next_base = base + n_cb_per_layer
        jobs.append((L, args.src, args.out, prefix, n_exp, args.stages,
                     str(dev), args.cb_sample, base, cached_ok))

    if args.jobs > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")     # torch/MPS is not fork-safe
        print(f"converting {len(jobs)} layers with {args.jobs} processes", flush=True)
        with ctx.Pool(args.jobs) as pool:
            results = []
            for res in pool.imap_unordered(convert_layer, jobs):
                results.append(res)
                Lr, sz, base, how = res
                print(f"  layer {Lr}: {sz/2**20:.0f} MB, cb base {base} [{how}] "
                      f"({len(results)}/{len(jobs)})", flush=True)
    else:
        results = []
        for j in jobs:
            res = convert_layer(j)
            results.append(res)
            print(f"  layer {res[0]}: {res[1]/2**20:.0f} MB [{res[3]}]", flush=True)

    for L, sz, base, how in sorted(results):
        if sz:
            manifest_layers[str(L)] = {"file": f"experts-L{L}.bin",
                                       "experts": n_exp, "bytes": sz,
                                       "codebook_base": base}
        else:
            manifest_layers.pop(str(L), None)

    # Append new records after the already-published books. A cached base
    # comes from the old manifest or from the bank itself; a new one starts
    # at the old record count. Either way the base is fixed before we get
    # here, so this writes each part *at* its base and never renumbers.
    parts = [(res[2], os.path.join(args.out, f"codebooks-L{res[0]}.bin"))
             for res in results]
    if any(os.path.exists(p) for _, p in parts):
        with open(merged + ".tmp", "wb") as cb_out:
            if compatible:
                with open(merged, "rb") as old:
                    shutil.copyfileobj(old, cb_out)
            for base, part in sorted(parts):
                if os.path.exists(part):
                    if os.path.getsize(part) != n_cb_per_layer * cb_record_bytes:
                        raise RuntimeError(f"malformed codebook part: {part}")
                    expected = cb_out.tell() // cb_record_bytes
                    if base < expected:
                        raise RuntimeError(
                            f"codebook base {base} overlaps the {expected} "
                            f"records already written")
                    if base > expected:
                        # A resume can recover a bank whose base sits past
                        # the end of what is being written, because an
                        # earlier run finished a layer this invocation is
                        # not redoing. No record names the ids in between,
                        # so pad them: the bases inside the banks are the
                        # engine's truth and cannot be moved.
                        cb_out.write(b"\0" * ((base - expected) *
                                              cb_record_bytes))
                    with open(part, "rb") as pf:
                        shutil.copyfileobj(pf, cb_out)
                    os.remove(part)
            cb_out.flush()
            os.fsync(cb_out.fileno())
        os.replace(merged + ".tmp", merged)
    else:
        print("codebooks: all layers cached, keeping existing codebooks.bin")


    # ---- tokenizer: copy it in so the container is self-contained -------
    for name in ("tiktoken.model", "tokenizer.model"):
        src_tok = os.path.join(args.src, name)
        if os.path.exists(src_tok):
            atomic_copyfile(src_tok, os.path.join(args.out, "tokenizer.model"))
            print(f"tokenizer: copied {name}")
            break

    # ---- special tokens --------------------------------------------------
    # tiktoken's rank file holds only ordinary merges; the markup tokens live
    # in tokenizer_config.json. Without them the engine splits <|open|> into
    # six ordinary tokens, which silently destroys any chat template and the
    # media markers an image would be wrapped in.
    p_cfg = os.path.join(args.src, "tokenizer_config.json")
    if os.path.exists(p_cfg):
        dec = json.load(open(p_cfg)).get("added_tokens_decoder", {})
        specials = sorted(((int(i), v["content"]) for i, v in dec.items()),
                          key=lambda x: x[0])
        if specials:
            atomic_json(os.path.join(args.out, "specials.json"),
                        [{"id": i, "text": t} for i, t in specials])
            print(f"special tokens: {len(specials)} written")

    # ---- chat template ---------------------------------------------------
    # HF keeps it either as a field in tokenizer_config.json or, more
    # recently, as its own .jinja file. Copy whichever exists so `waste chat`
    # can address an instruct model in the format it was trained on instead
    # of as raw continuation. Not every release ships one — K3 describes its
    # template in the technical report without distributing it — so a
    # container without a template is normal and the CLI falls back.
    tpl = None
    for name in ("chat_template.jinja", "chat_template.json"):
        p_tpl = os.path.join(args.src, name)
        if os.path.exists(p_tpl):
            tpl = io.open(p_tpl, encoding="utf-8").read()
            break
    if tpl is None:
        p_cfg = os.path.join(args.src, "tokenizer_config.json")
        if os.path.exists(p_cfg):
            tpl = json.load(open(p_cfg)).get("chat_template")
    if tpl:
        atomic_text(os.path.join(args.out, "chat_template.jinja"), tpl)
        print(f"chat template: copied ({len(tpl)} bytes)")

    # ---- chat.json -------------------------------------------------------
    # The declarative format the CLI reads. Neither Kimi release ships a
    # template, so there is nothing to convert from — but for the models we
    # have transcribed from the reference encoder, examples/ holds one, and
    # a container that has to be finished by hand is a container that will
    # be used unfinished. Copied from examples/ rather than embedded here,
    # so there is one copy of each template and not two that drift.
    #
    # Only for an architecture we actually have: for anything else the CLI
    # keeps saying so and falling back, which is better than a guessed
    # format that produces plausible wrong answers. And never over an
    # existing file — a hand-edited chat.json outranks the shipped one.
    _tmpl_for = {"kimi-k3": "chat-k3.json"}
    _hf0 = ((cfg.get("_outer", {}).get("architectures")
             or cfg.get("architectures") or [""]))[0]
    _arch0 = ("kimi-k3" if "KimiK3" in _hf0 else
              "kimi-linear" if "KimiLinear" in _hf0 else "")
    _dst = os.path.join(args.out, "chat.json")
    _name = _tmpl_for.get(_arch0)
    if _name and not os.path.exists(_dst):
        _src_tmpl = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "examples", _name)
        if os.path.exists(_src_tmpl):
            atomic_copyfile(_src_tmpl, _dst)
            print(f"chat.json: {_name} — `waste chat` will use it")
        else:
            print(f"chat.json: examples/{_name} not found; the CLI will "
                  f"fall back to raw continuation")
    elif os.path.exists(_dst):
        print("chat.json: already present, left alone")
    elif _arch0:
        print(f"chat.json: none known for {_arch0}; the CLI will fall back "
              f"to raw continuation (see examples/README.md)")

    # ---- vision config ---------------------------------------------------
    # The tower's shape lives in the release's nested vision_config, and the
    # engine reads it from vision.json — only from there. Without this file
    # waste_image_add returns WASTE_E_UNSUPPORTED and a multimodal container
    # silently has no images, which is exactly what happened before it was
    # written: the file had to be produced by hand after every conversion.
    #
    # Two keys are ours rather than the release's:
    #   vt_rms_eps    the tower's RMSNorms are nn.RMSNorm(dim) with no eps,
    #                 so PyTorch uses finfo(float32).eps. Written explicitly
    #                 so the oracle in tools/vision_ref.py reads the same
    #                 number the engine compiles in.
    #   max_patches   our patch budget, i.e. how much of the context an
    #                 image costs. The release's own limit is far higher
    #                 (in_patch_limit 65536, 512 patches on a side); an
    #                 image is priced as text of the same length, so this
    #                 is a deliberate cap and halving it halves the prompt.
    #
    # image_mean/std are the release's, read from preprocessor_config.json,
    # where `media_proc_cfg` is what kimi_k3_vision_processing.py actually
    # applies. They were hardcoded to the CLIP constants here for exactly
    # one day, on the belief that K3 shipped no preprocessor config — it
    # does, and K3 normalizes to [-1, 1] with mean = std = 0.5, which is
    # not what CLIP does. Guess nothing that the release states.
    vc = cfg.get("_outer", {}).get("vision_config")
    if vc:
        vj = {k: v for k, v in vc.items() if not k.startswith("_")}
        vj["vt_rms_eps"] = 1.1920928955078125e-07
        mp = cfg.get("_outer", {}).get("media_placeholder_token_id")
        if mp is not None:
            vj["media_placeholder_token_id"] = mp
        vj["max_patches"] = 1024

        p_pre = os.path.join(args.src, "preprocessor_config.json")
        if os.path.exists(p_pre):
            mpc = json.load(open(p_pre)).get("media_proc_cfg", {})
            for k in ("image_mean", "image_std"):
                if mpc.get(k) is not None:
                    vj[k] = mpc[k]
            for k in ("patch_size", "in_patch_limit", "patch_limit_on_one_side"):
                if mpc.get(k) is not None:
                    vj[k] = mpc[k]
            print(f"vision: normalization from preprocessor_config.json, "
                  f"mean {vj.get('image_mean')}")
        else:
            # No config to read: say so loudly rather than invent constants.
            # A tower fed the wrong normalization still matches its oracle,
            # because the oracle reads this same file — the error hides.
            vj["image_mean"] = [0.5, 0.5, 0.5]
            vj["image_std"] = [0.5, 0.5, 0.5]
            print("vision: WARNING no preprocessor_config.json in --src; "
                  "assuming mean=std=0.5. Check it against the release.")

        atomic_json(os.path.join(args.out, "vision.json"), vj)
        print(f"vision: {vj.get('vt_num_hidden_layers', '?')}-layer tower, "
              f"patch {vj.get('patch_size', '?')}, "
              f"max_patches {vj['max_patches']}")

    # ---- trunk ----------------------------------------------------------
    trunk_path = os.path.join(args.out, "trunk.bin")
    trunk_tmp = trunk_path + ".tmp"
    tindex = []
    if args.skip_trunk:
        # "The trunk is unchanged between runs" — so carry its published
        # index forward and still publish. Returning here instead left the
        # banks this run had just converted unreferenced by any manifest,
        # and the codebooks.bin the merge had already replaced described by
        # an old one: a container that reports itself as fine and whose
        # layer bases point into records that are no longer there.
        tindex = (existing or {}).get("trunk")
        if not isinstance(tindex, list) or not tindex:
            print(f"--skip-trunk keeps the trunk {manifest_path} describes, "
                  "and it describes none — convert once without it",
                  file=sys.stderr)
            return 1
        if not os.path.exists(trunk_path):
            print(f"--skip-trunk needs {trunk_path}, which is not there",
                  file=sys.stderr)
            return 1
        print(f"skipping trunk: keeping the published {len(tindex)} tensors")
    else:
        # Keep the currently published trunk intact for the entire
        # conversion. A K3 trunk takes hours to rebuild; opening the final
        # path with "wb" destroyed a working container at the start of that
        # interval.
        with open(trunk_tmp, "wb") as tf:
            for name in sorted(sr.names()):
                if ".experts." in name or name.endswith(("_packed", "_scale")):
                    continue
                if not st.have(name):
                    continue                  # shard not downloaded yet
                t = st.tensor(name)
                off = tf.tell()
                if t.dim() == 1 or t.numel() < 1 << 16:
                    tf.write(raw_bytes(t.float()))
                    tindex.append({"name": name, "fmt": FMT_F32, "off": off,
                                   "shape": list(t.shape),
                                   "bytes": tf.tell() - off})
                else:
                    # 4 bits for the bulk; the embedding table and the output
                    # head keep 8, they are small and sit at both ends of the
                    # network where error is least forgiving
                    big = not (name.endswith("embed_tokens.weight")
                               or name.endswith("lm_head.weight"))
                    bits = 8 if (args.trunk8 or not big) else args.trunk_bits
                    if bits == 3:
                        q, sc, shape = quantize_q3g(t); fmt = FMT_Q3G
                    elif bits == 4:
                        q, sc, shape = quantize_q4g(t); fmt = FMT_Q4G
                    else:
                        q, sc, shape = quantize_q8g(t); fmt = FMT_Q8G
                    tf.write(raw_bytes(q))
                    sc_off = tf.tell()
                    tf.write(raw_bytes(sc))
                    tindex.append({"name": name, "fmt": fmt,
                                   "off": off, "shape": shape, "group": 128,
                                   "scale_off": sc_off,
                                   "bytes": tf.tell() - off})
            tf.flush()
            os.fsync(tf.fileno())
        print(f"trunk: {os.path.getsize(trunk_tmp)/2**20:.0f} MB, "
              f"{len(tindex)} tensors")

    # `arch` is descriptive only — the engine derives its own from
    # config._outer.architectures, because the *inner* architectures says
    # KimiLinearForCausalLM on both models. It used to be written from
    # model_type, which for K3 is the text model's and reads "kimi_linear"
    # in a K3 container. Write what `waste info` reports, so a container
    # inspected by hand and a container opened by the engine agree.
    _hf = ((cfg.get("_outer", {}).get("architectures")
            or cfg.get("architectures") or [""]))[0]
    arch = ("kimi-k3" if "KimiK3" in _hf else
            "kimi-linear" if "KimiLinear" in _hf else
            _hf or cfg.get("model_type", "unknown"))

    manifest = {
        "format_version": 0,
        "arch": arch,
        "tensor_prefix": prefix,
        "config": cfg,
        "expert_quant": {"fmt": "VQ3R" if args.stages == 3 else "VQ2R",
                         "stages": args.stages, "vec_dim": VEC_DIM,
                         "entries": CB_ENTRIES, "index_block": IDX_BLOCK,
                         "bits_per_weight": args.stages},
        "layers": manifest_layers,
        "trunk": tindex,
    }
    # A manifest that lists fewer expert layers than the one it replaces
    # publishes a container the engine will refuse to open, and the banks it
    # drops are still on disk taking up room. Never intended; say so.
    dropped = sorted(set((existing or {}).get("layers", {})) -
                     set(manifest_layers), key=int)
    if dropped:
        print(f"WARNING dropping {len(dropped)} expert layer(s) the previous "
              f"manifest listed: {', '.join(dropped)}", file=sys.stderr)

    manifest_tmp = manifest_path + ".tmp"
    with open(manifest_tmp, "w") as f:
        json.dump(manifest, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    # The manifest is the commit record and is always published last.  An
    # exception before these replaces leaves the old container untouched.
    if not args.skip_trunk:
        os.replace(trunk_tmp, trunk_path)
    os.replace(manifest_tmp, manifest_path)
    print(f"wrote {args.out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
