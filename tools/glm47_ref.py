#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Independent PyTorch oracle for full GLM-4.7 WARP containers.

The first full-GLM runtime path is intentionally small and auditable, but its
attention contract has several ordered obligations that can each produce
plausible logits when wrong: add Q/K/V bias, normalize Q/K per head, rotate
only the first partial-RoPE slice in LLaMA half-split order, then map each
query head onto its grouped KV head. The router likewise selects on
``sigmoid(logit) + correction`` but weights the selected experts with the
uncorrected sigmoid values.

This file reimplements that sequence from the converted container rather than
calling engine helpers. On the tiny ``--glm47-full`` fixture it gives a cheap
end-to-end differential:

  python3 tools/make_test_container.py /tmp/glm47.waste --glm47-full
  uv run --no-project --with torch python tools/glm47_ref.py \
      --container /tmp/glm47.waste --ids 3,7,11,5,9 --dump /tmp/ref.bin
  ./test_forward /tmp/glm47.waste 3,7,11,5,9 /tmp/c.bin 0

Weights are read from the same container on both sides, so the comparison is
about runtime arithmetic and state, not conversion loss.
"""

import argparse
import json
import math
import os
import struct
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kimi_ref import Container, rms_norm                         # noqa: E402


def apply_partial_rope(x, pos, rotary_dim, theta, layout="half"):
    """Rotate the first ``rotary_dim`` of ``x[T,H,D]``; pass the rest.

    ``half`` is the LLaMA layout used by full GLM. ``interleaved`` exists only
    as a negative control for the differential test: it is the tempting GLM
    Flash/MLA layout and still yields finite, weight-shaped output.
    """
    if rotary_dim < 2 or rotary_dim & 1 or rotary_dim > x.shape[-1]:
        raise ValueError(f"invalid partial rotary width {rotary_dim}")
    inv = 1.0 / (float(theta) **
                 (torch.arange(0, rotary_dim, 2, dtype=x.dtype,
                               device=x.device) / rotary_dim))
    ang = pos.to(dtype=x.dtype, device=x.device).unsqueeze(-1) * inv
    cos = ang.cos().unsqueeze(1)
    sin = ang.sin().unsqueeze(1)
    rot = x[..., :rotary_dim]
    out = x.clone()
    if layout == "half":
        half = rotary_dim // 2
        a, b = rot[..., :half], rot[..., half:]
        out[..., :half] = a * cos - b * sin
        out[..., half:rotary_dim] = a * sin + b * cos
    elif layout == "interleaved":
        a, b = rot[..., 0::2], rot[..., 1::2]
        out[..., 0:rotary_dim:2] = a * cos - b * sin
        out[..., 1:rotary_dim:2] = a * sin + b * cos
    else:
        raise ValueError(f"unknown RoPE layout {layout!r}")
    return out


def select_routes(scores, correction, top_k, use_correction=True):
    """C's ordered top-k: first maximum wins; correction is selection-only."""
    choice = scores + correction.unsqueeze(0) if use_correction else scores
    work = choice.clone()
    picked = []
    for _ in range(top_k):
        best = work.argmax(-1)
        picked.append(best)
        work.scatter_(1, best.unsqueeze(1), float("-inf"))
    ids = torch.stack(picked, -1)
    return ids, scores.gather(1, ids)


class Glm47Ref:
    def __init__(self, container, *, qkv_bias=True, qk_norm=True,
                 rope_layout="half", kv_map="grouped", correction=True):
        self.c = container
        self.t = container.t
        self.cfg = container.cfg
        self.p = container.prefix
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.rope_layout = rope_layout
        self.kv_map = kv_map
        self.correction = correction
        if ((self.cfg.get("architectures") or [""])[0] !=
                "Glm4MoeForCausalLM"):
            raise ValueError("container is not Glm4MoeForCausalLM")

    def attention(self, layer, x, pos):
        cfg, T = self.cfg, x.shape[0]
        p = f"{self.p}model.layers.{layer}.self_attn."
        qh = cfg["num_attention_heads"]
        kh = cfg["num_key_value_heads"]
        D = cfg["head_dim"]
        if qh % kh:
            raise ValueError("query heads are not divisible by KV heads")

        def project(kind, heads):
            y = x @ self.t[p + kind + "_proj.weight"].T
            if self.qkv_bias:
                y = y + self.t[p + kind + "_proj.bias"]
            return y.view(T, heads, D)

        q = project("q", qh)
        k = project("k", kh)
        v = project("v", kh)
        if self.qk_norm:
            q = rms_norm(q, self.t[p + "q_norm.weight"],
                         cfg["rms_norm_eps"])
            k = rms_norm(k, self.t[p + "k_norm.weight"],
                         cfg["rms_norm_eps"])

        rotary = int(round(D * float(cfg["partial_rotary_factor"])))
        q = apply_partial_rope(q, pos, rotary, cfg["rope_theta"],
                               self.rope_layout)
        k = apply_partial_rope(k, pos, rotary, cfg["rope_theta"],
                               self.rope_layout)

        if self.kv_map == "grouped":
            kv_for_q = torch.arange(qh, device=x.device) // (qh // kh)
        elif self.kv_map == "first":
            kv_for_q = torch.zeros(qh, dtype=torch.long, device=x.device)
        else:
            raise ValueError(f"unknown KV mapping {self.kv_map!r}")
        kg, vg = k[:, kv_for_q], v[:, kv_for_q]
        score = torch.einsum("thd,shd->hts", q, kg) / math.sqrt(D)
        score = score + torch.full((T, T), float("-inf"),
                                   device=x.device).triu(1)
        prob = score.softmax(-1)
        out = torch.einsum("hts,shd->thd", prob, vg).reshape(T, qh * D)
        return out @ self.t[p + "o_proj.weight"].T

    def dense(self, layer, x):
        p = f"{self.p}model.layers.{layer}.mlp."
        gate = x @ self.t[p + "gate_proj.weight"].T
        up = x @ self.t[p + "up_proj.weight"].T
        return (F.silu(gate) * up) @ self.t[p + "down_proj.weight"].T

    def moe(self, layer, x, routes):
        p = f"{self.p}model.layers.{layer}.block_sparse_moe."
        cfg, T = self.cfg, x.shape[0]
        scores = torch.sigmoid(
            x.float() @ self.t[p + "gate.weight"].float().T)
        correction = self.t[p + "gate.e_score_correction_bias"].float()
        ids, weights = select_routes(
            scores, correction, cfg["num_experts_per_token"],
            use_correction=self.correction)
        if cfg.get("moe_renormalize"):
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * cfg["routed_scaling_factor"]

        out = torch.zeros_like(x)
        for token in range(T):
            routes.append({
                "position": token,
                "layer": layer,
                "experts": [int(i) for i in ids[token]],
                "weights": [float(w) for w in weights[token]],
            })
            for slot in range(ids.shape[1]):
                expert = self.c.expert(layer, int(ids[token, slot]))
                gate = x[token] @ expert["gate"].T
                up = x[token] @ expert["up"].T
                out[token] += weights[token, slot] * (
                    (F.silu(gate) * up) @ expert["down"].T)

        gate = self.t[p + "shared_experts.gate_proj.weight"]
        up = self.t[p + "shared_experts.up_proj.weight"]
        down = self.t[p + "shared_experts.down_proj.weight"]
        return out + (F.silu(x @ gate.T) * (x @ up.T)) @ down.T

    def forward(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long)
        pos = torch.arange(ids.numel(), dtype=torch.long)
        x = self.t[self.p + "model.embed_tokens.weight"][ids]
        routes = []
        for layer in range(self.cfg["num_hidden_layers"]):
            p = f"{self.p}model.layers.{layer}."
            h = rms_norm(x, self.t[p + "input_layernorm.weight"],
                         self.cfg["rms_norm_eps"])
            x = x + self.attention(layer, h, pos)
            h = rms_norm(x, self.t[p + "post_attention_layernorm.weight"],
                         self.cfg["rms_norm_eps"])
            x = x + (self.dense(layer, h)
                     if layer < self.cfg["first_k_dense_replace"]
                     else self.moe(layer, h, routes))
        x = rms_norm(x, self.t[self.p + "model.norm.weight"],
                     self.cfg["rms_norm_eps"])
        return x @ self.t[self.p + "lm_head.weight"].T, routes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--ids", required=True, help="comma-separated token ids")
    ap.add_argument("--dump", help="last-position logits as little-endian f32")
    ap.add_argument("--trace", help="JSON route trace")
    ap.add_argument("--no-bias", action="store_true", help="negative control")
    ap.add_argument("--no-qk-norm", action="store_true", help="negative control")
    ap.add_argument("--rope-layout", choices=("half", "interleaved"),
                    default="half")
    ap.add_argument("--kv-map", choices=("grouped", "first"),
                    default="grouped")
    ap.add_argument("--no-correction", action="store_true",
                    help="negative control: select on raw sigmoid scores")
    args = ap.parse_args()

    ids = [int(i) for i in args.ids.replace(" ", ",").split(",") if i]
    ref = Glm47Ref(Container(args.container),
                   qkv_bias=not args.no_bias,
                   qk_norm=not args.no_qk_norm,
                   rope_layout=args.rope_layout,
                   kv_map=args.kv_map,
                   correction=not args.no_correction)
    with torch.no_grad():
        logits, routes = ref.forward(ids)
    last = logits[-1].contiguous().float()
    if args.dump:
        with open(args.dump, "wb") as out:
            out.write(struct.pack(f"<{last.numel()}f", *last.tolist()))
    if args.trace:
        with open(args.trace, "w") as out:
            json.dump(routes, out, indent=1)
    top = torch.topk(last, min(10, last.numel()))
    print(json.dumps({
        "prompt_tokens": len(ids),
        "argmax": int(last.argmax()),
        "top": [{"id": int(i), "logit": float(v)}
                for v, i in zip(top.values, top.indices)],
        "routes": routes,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
