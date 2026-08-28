#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Build a small, integrated GLM-5.3 runtime container.

This intentionally reuses ``tools/make_test_container.py`` for the WASTE
trunk, codebook and expert-record layouts.  The release loader requires the
real 45-layer 3-KDA/1-MLA schedule, so only the tensor dimensions, vocabulary
and DSA-equivalent context bound are reduced.  The resulting fixture reaches
the public model loader and the ordinary step/prefill/reset paths; it is not a
numerical oracle for the released weights.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "tools" / "make_test_container.py"


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "waste_make_test_container", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def glm53_config(*, generation_schedule):
    """A shape-scaled text config with the release's semantic contracts.

    The shared helper historically converts its KDA list from one-based to
    zero-based while deciding which tensors to emit.  ``generation_schedule``
    is therefore one-based during construction and is replaced by the exact
    zero-based release schedule in the published manifest below.
    """
    return {
        "model_type": "glm5_next_text",
        "_outer": {
            "architectures": ["Glm5NextForConditionalGeneration"],
            "model_type": "glm5_next",
        },
        "hidden_size": 128,
        "num_hidden_layers": 45,
        "first_k_dense_replace": 3,
        "intermediate_size": 256,
        "moe_intermediate_size": 64,
        "num_experts": 8,
        "num_experts_per_token": 2,
        "num_shared_experts": 1,
        "moe_renormalize": True,
        "norm_topk_prob": True,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "moe_router_activation_func": "sigmoid",
        "routed_scaling_factor": 2.5,
        "n_group": 1,
        "topk_group": 1,
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "q_lora_rank": 64,
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 0,
        "v_head_dim": 16,
        "mla_use_nope": True,
        "mla_rms_norm_eps": 1e-5,
        "rms_norm_eps": 1e-5,
        "source_max_position_embeddings": 1048576,
        "dsa_dense_context_limit": 8,
        "max_position_embeddings": 8,
        "index_topk": 8,
        "vocab_size": 256,
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "eos_token_ids": [2, 3, 4],
        "mhc": True,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20,
        "kda_qk_l2_norm_eps": 1e-6,
        "num_nextn_predict_layers": 1,
        "layer_types": [
            "linear_attention" if layer % 4 != 3
            else "deepseek_sparse_attention"
            for layer in range(45)
        ],
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
        "indexer_types": ["full"] * 45,
        "linear_attn_config": {
            "num_heads": 4,
            "head_dim": 32,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layer_index_base": 0,
            "kda_layers": list(generation_schedule),
            "full_attn_layers": [layer + 1 for layer in range(45)
                                 if layer % 4 == 3],
        },
    }


def append_mhc(helper, out: Path, *, seed: int, prefix: str):
    """Append the six resident fp32 mHC tensors used at every layer."""
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    trunk_path = out / "trunk.bin"
    old_size = trunk_path.stat().st_size

    extra = helper.Trunk(random.Random(seed ^ 0x53A11CE), prefix)
    hc, hidden = 4, 128
    mix = (2 + hc) * hc
    for layer in range(45):
        stem = f"model.layers.{layer}."
        for site in ("attn", "ffn"):
            extra.f32(stem + f"hc_{site}_fn", [mix, hc * hidden])
            extra.f32(stem + f"hc_{site}_base", [mix])
            extra.f32(stem + f"hc_{site}_scale", [3])

    with trunk_path.open("ab") as stream:
        stream.write(extra.buf)
    for entry in extra.index:
        entry["off"] += old_size
        manifest["trunk"].append(entry)

    # Publish the exact zero-based schedule the runtime contract parses.
    kda = [layer for layer in range(45) if layer % 4 != 3]
    full = [layer for layer in range(45) if layer % 4 == 3]
    config = manifest["config"]
    config["linear_attn_config"]["kda_layers"] = kda
    config["linear_attn_config"]["full_attn_layers"] = full
    manifest["arch"] = "Glm5NextForConditionalGeneration"
    manifest["source_ignored_layers"] = [45]
    manifest["unsupported_features"] = [
        {
            "name": "deepseek_sparse_attention_indexer",
            "action": "dense_equivalent",
            "context_limit": 8,
            "reason": "index_topk covers the complete causal history",
        },
        {
            "name": "multi_token_prediction",
            "source_layers": [45],
            "action": "omitted",
            "reason": "unsupported",
        },
        {
            "name": "vision_tower",
            "action": "omitted",
            "reason": "initial GLM-5.3 contract is text-only",
        },
    ]
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")


def build(out: Path, seed: int):
    helper = load_helper()
    kda = [layer for layer in range(45) if layer % 4 != 3]

    # See glm53_config(): the helper's existing KDA fixture generator expects
    # a one-based list.  The manifest is corrected before it is observable to
    # the runtime.
    helper.CFG = glm53_config(generation_schedule=[layer + 1 for layer in kda])
    prefix = "language_model."
    old_argv = sys.argv
    try:
        sys.argv = [str(HELPER_PATH), str(out), "--seed", str(seed),
                    "--prefix", prefix]
        rc = helper.main()
    finally:
        sys.argv = old_argv
    if rc:
        raise RuntimeError(f"base fixture generator exited {rc}")
    append_mhc(helper, out, seed=seed, prefix=prefix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"output directory is not empty: {args.out}")
    os.makedirs(args.out, exist_ok=True)
    build(args.out, args.seed)
    size = sum(path.stat().st_size for path in args.out.iterdir())
    print(f"wrote {args.out}: integrated GLM-5.3 fixture, "
          f"45 layers, {size / (1 << 20):.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
