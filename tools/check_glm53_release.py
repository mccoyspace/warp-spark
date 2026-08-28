#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Gate the pinned GLM-5.3-Flash source before a large download.

The default check reads only repository metadata, config.json, and the weight
index.  ``--headers`` additionally range-reads every safetensor header and
binds all tensor names, dtypes, byte ranges, and shard assignments to that
index without downloading any tensor payload.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any


CANDIDATE_REPO = "zai-org/GLM-5.3-Flash"
PINNED_REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
INDEX_NAME = "model.safetensors.index.json"

OUTER_EXPECTED = {
    "architectures": ["Glm5NextForConditionalGeneration"],
    "model_type": "glm5_next",
    "tie_word_embeddings": False,
}
TEXT_EXPECTED = {
    "model_type": "glm5_next_text",
    "num_hidden_layers": 45,
    "hidden_size": 4096,
    "vocab_size": 154880,
    "n_routed_experts": 288,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "first_k_dense_replace": 3,
    "intermediate_size": 12288,
    "moe_intermediate_size": 2048,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 0,
    "v_head_dim": 256,
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_kpool": 4,
    "num_nextn_predict_layers": 1,
    "max_position_embeddings": 1048576,
    "rms_norm_eps": 1e-5,
    "routed_scaling_factor": 2.5,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "n_group": 1,
    "topk_group": 1,
    "mhc": True,
    "hc_mult": 4,
    "hc_eps": 1e-6,
    "hc_sinkhorn_iters": 20,
    "mla_use_nope": True,
    "swiglu_limit": 10.0,
}
LINEAR_EXPECTED = {
    "num_heads": 64,
    "head_dim": 128,
    "short_conv_kernel_size": 4,
    "gate_lower_bound": -5.0,
}
VISION_EXPECTED = {
    "model_type": "glm5_next_vision",
    "depth": 24,
    "hidden_size": 1024,
    "num_heads": 16,
    "intermediate_size": 4096,
    "out_hidden_size": 4096,
    "image_size": 448,
    "patch_size": 14,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
}
QUANT_EXPECTED = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
}
KDA_LAYERS = [i for i in range(45) if i % 4 != 3]
FULL_ATTN_LAYERS = [i for i in range(45) if i % 4 == 3]
DTYPE_BYTES = {"F8_E4M3": 1, "BF16": 2, "F16": 2,
               "F32": 4, "I32": 4, "I64": 8, "U8": 1}


class ReleaseCheckError(RuntimeError):
    """The source is incomplete or outside the registered contract."""


@dataclass(frozen=True)
class RepoMetadata:
    repo: str
    revision: str
    files: frozenset[str]


def _request(url: str, *, byte_range: tuple[int, int] | None = None):
    headers = {"User-Agent": "warp-glm53-release-check/2"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)
    try:
        return urllib.request.urlopen(request, timeout=180)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 404):
            raise ReleaseCheckError(f"not publicly available: {url}") from exc
        raise ReleaseCheckError(f"HTTP {exc.code}: {url}") from exc
    except OSError as exc:
        raise ReleaseCheckError(f"cannot read {url}: {exc}") from exc


def _json_url(url: str) -> Any:
    try:
        with _request(url) as response:
            return json.load(response)
    except json.JSONDecodeError as exc:
        raise ReleaseCheckError(f"invalid JSON: {url}: {exc}") from exc


def repo_metadata(repo: str, revision: str | None = None) -> RepoMetadata:
    quoted = "/".join(urllib.parse.quote(part, safe="") for part in repo.split("/"))
    url = f"https://huggingface.co/api/models/{quoted}"
    if revision:
        url += "/revision/" + urllib.parse.quote(revision, safe="")
    payload = _json_url(url)
    sha = payload.get("sha")
    if not isinstance(sha, str) or len(sha) != 40:
        raise ReleaseCheckError(f"{repo}: API did not return an immutable revision")
    files = frozenset(
        row.get("rfilename") for row in payload.get("siblings", [])
        if isinstance(row, dict) and isinstance(row.get("rfilename"), str)
    )
    return RepoMetadata(repo, sha, files)


def repo_json(repo: str, revision: str, filename: str) -> Any:
    quoted_repo = "/".join(
        urllib.parse.quote(part, safe="") for part in repo.split("/"))
    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_file = "/".join(
        urllib.parse.quote(part, safe="") for part in filename.split("/"))
    return _json_url(
        f"https://huggingface.co/{quoted_repo}/resolve/"
        f"{quoted_revision}/{quoted_file}")


def _compare_fields(where: str, value: dict[str, Any],
                    expected: dict[str, Any]) -> list[str]:
    return [f"{where}{key}: got={value.get(key)!r}, expected={want!r}"
            for key, want in expected.items() if value.get(key) != want]


def config_differences(config: dict[str, Any]) -> list[str]:
    """Return every execution-contract difference from the pinned release."""
    differences = _compare_fields("", config, OUTER_EXPECTED)
    text = config.get("text_config")
    vision = config.get("vision_config")
    quant = config.get("quantization_config")
    if not isinstance(text, dict):
        differences.append("text_config: missing object")
        return differences
    if not isinstance(vision, dict):
        differences.append("vision_config: missing object")
    else:
        differences += _compare_fields("vision_config.", vision, VISION_EXPECTED)
    if not isinstance(quant, dict):
        differences.append("quantization_config: missing object")
    else:
        differences += _compare_fields("quantization_config.", quant,
                                       QUANT_EXPECTED)
    differences += _compare_fields("text_config.", text, TEXT_EXPECTED)

    linear = text.get("linear_attn_config")
    if not isinstance(linear, dict):
        differences.append("text_config.linear_attn_config: missing object")
    else:
        differences += _compare_fields("text_config.linear_attn_config.",
                                       linear, LINEAR_EXPECTED)
        if linear.get("kda_layers") != KDA_LAYERS:
            differences.append("text_config.linear_attn_config.kda_layers: "
                               "does not match the 34-layer schedule")
        if linear.get("full_attn_layers") != FULL_ATTN_LAYERS:
            differences.append("text_config.linear_attn_config.full_attn_layers: "
                               "does not match the 11-layer schedule")
    layer_types = ["linear_attention" if i in KDA_LAYERS
                   else "deepseek_sparse_attention" for i in range(45)]
    if text.get("layer_types") != layer_types:
        differences.append("text_config.layer_types: hybrid schedule changed")
    if text.get("mlp_layer_types") != ["dense"] * 3 + ["sparse"] * 42:
        differences.append("text_config.mlp_layer_types: expected 3 dense + 42 sparse")
    if text.get("indexer_types") != ["full"] * 45:
        differences.append("text_config.indexer_types: expected independent full indexers")
    return differences


def validate_index(index: dict[str, Any], files: frozenset[str]) -> tuple[int, int]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ReleaseCheckError("weight index has no non-empty weight_map")
    if any(not isinstance(name, str) for name in weight_map.values()):
        raise ReleaseCheckError("weight index contains a non-string shard name")
    shards = set(weight_map.values())
    missing = sorted(shards - files)
    if missing:
        raise ReleaseCheckError(
            f"weight index names {len(missing)} absent shard(s): "
            + ", ".join(missing[:5]))
    total_size = (index.get("metadata") or {}).get("total_size")
    if not isinstance(total_size, int) or total_size <= 0:
        raise ReleaseCheckError("weight index has no positive metadata.total_size")
    return len(shards), total_size


def _shard_url(repo: str, revision: str, shard: str) -> str:
    return (f"https://huggingface.co/{repo}/resolve/{revision}/"
            + urllib.parse.quote(shard, safe="/"))


def fetch_safetensor_header(repo: str, revision: str, shard: str):
    """Return ``(header, shard_bytes, header_bytes)`` using two range reads."""
    url = _shard_url(repo, revision, shard)
    with _request(url, byte_range=(0, 7)) as response:
        first = response.read()
        content_range = response.headers.get("Content-Range", "")
    if len(first) != 8:
        raise ReleaseCheckError(f"{shard}: first range returned {len(first)} bytes")
    try:
        shard_bytes = int(content_range.rsplit("/", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ReleaseCheckError(f"{shard}: no total size in Content-Range") from exc
    header_bytes = struct.unpack("<Q", first)[0]
    if header_bytes < 2 or header_bytes > 128 * 1024 * 1024:
        raise ReleaseCheckError(f"{shard}: implausible header length {header_bytes}")
    with _request(url, byte_range=(8, 7 + header_bytes)) as response:
        raw = response.read()
    if len(raw) != header_bytes:
        raise ReleaseCheckError(
            f"{shard}: header range returned {len(raw)}, expected {header_bytes}")
    try:
        return json.loads(raw), shard_bytes, header_bytes
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseCheckError(f"{shard}: invalid safetensor header: {exc}") from exc


def validate_header_set(index: dict[str, Any], headers: dict[str, tuple]):
    """Bind all header records to the index and validate their byte ranges."""
    weight_map = index["weight_map"]
    seen: dict[str, str] = {}
    dtypes: Counter[str] = Counter()
    payload_bytes = 0
    header_bytes = 0
    for shard, (doc, shard_size, hlen) in headers.items():
        header_bytes += hlen
        for name, meta in doc.items():
            if name == "__metadata__":
                continue
            if name in seen:
                raise ReleaseCheckError(f"tensor occurs in two headers: {name}")
            if not isinstance(meta, dict):
                raise ReleaseCheckError(f"{shard}:{name}: metadata is not an object")
            dtype = meta.get("dtype")
            shape = meta.get("shape")
            offsets = meta.get("data_offsets")
            if (dtype not in DTYPE_BYTES or not isinstance(shape, list) or
                    not isinstance(offsets, list) or len(offsets) != 2):
                raise ReleaseCheckError(f"{shard}:{name}: unsupported metadata")
            if any(not isinstance(dim, int) or dim < 0 for dim in shape):
                raise ReleaseCheckError(f"{shard}:{name}: malformed shape")
            begin, end = offsets
            expected = math.prod(shape) * DTYPE_BYTES[dtype]
            if (not isinstance(begin, int) or not isinstance(end, int) or
                    begin < 0 or end < begin or end - begin != expected or
                    8 + hlen + end > shard_size):
                raise ReleaseCheckError(f"{shard}:{name}: invalid data byte range")
            seen[name] = shard
            dtypes[dtype] += 1
            payload_bytes += expected
    absent = sorted(set(weight_map) - set(seen))
    extra = sorted(set(seen) - set(weight_map))
    wrong = sorted(name for name, shard in seen.items()
                   if weight_map.get(name) != shard)
    if absent or extra or wrong:
        raise ReleaseCheckError(
            "header/index mismatch: "
            f"{len(absent)} absent, {len(extra)} extra, {len(wrong)} wrong shard")
    indexed_bytes = (index.get("metadata") or {}).get("total_size")
    if payload_bytes != indexed_bytes:
        raise ReleaseCheckError(
            f"header payload bytes {payload_bytes} != index {indexed_bytes}")
    return {
        "tensors": len(seen),
        "payload_bytes": payload_bytes,
        "header_bytes": header_bytes,
        "dtypes": dict(sorted(dtypes.items())),
    }


def remote_header_gate(repo: str, revision: str, index: dict[str, Any],
                       jobs: int) -> dict[str, Any]:
    shards = sorted(set(index["weight_map"].values()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = pool.map(
            lambda shard: (shard, fetch_safetensor_header(repo, revision, shard)),
            shards)
        headers = dict(rows)
    return validate_header_set(index, headers)


def gib(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=CANDIDATE_REPO)
    parser.add_argument("--candidate-revision", default=PINNED_REVISION,
                        help="immutable commit to verify (defaults to the qualified pin)")
    parser.add_argument("--headers", action="store_true",
                        help="range-read and validate every safetensor header")
    parser.add_argument("--jobs", type=int, default=8,
                        help="parallel header probes (default: 8)")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.jobs > 32:
        parser.error("--jobs must be between 1 and 32")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate = repo_metadata(args.candidate, args.candidate_revision)
    if candidate.revision != args.candidate_revision:
        raise ReleaseCheckError(
            f"requested {args.candidate_revision}, API resolved {candidate.revision}")
    required = {"config.json", INDEX_NAME, "tokenizer_config.json",
                "tokenizer.json", "chat_template.jinja", "processor_config.json"}
    missing = sorted(required - candidate.files)
    if missing:
        raise ReleaseCheckError(
            f"{candidate.repo}@{candidate.revision}: missing {', '.join(missing)}")

    config = repo_json(candidate.repo, candidate.revision, "config.json")
    differences = config_differences(config)
    if differences:
        raise ReleaseCheckError(
            "candidate is outside the registered GLM-5.3-Flash contract:\n  "
            + "\n  ".join(differences))
    index = repo_json(candidate.repo, candidate.revision, INDEX_NAME)
    shard_count, total_size = validate_index(index, candidate.files)
    result = {
        "status": "READY_FOR_HEADER_GATE" if not args.headers else "HEADER_GATE_PASSED",
        "candidate": candidate.repo,
        "revision": candidate.revision,
        "shards": shard_count,
        "weight_bytes": total_size,
        "weight_gib": round(total_size / (1024 ** 3), 2),
        "architecture": config.get("architectures"),
        "model_type": config.get("model_type"),
        "scope": "text-only, context <= 2048; MTP and vision omitted",
    }
    if args.headers:
        result["header_gate"] = remote_header_gate(
            candidate.repo, candidate.revision, index, args.jobs)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = run(args)
    except ReleaseCheckError as exc:
        if args.as_json:
            print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, indent=2))
        else:
            print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"{result['status']}: {result['candidate']}@{result['revision']}")
        print(f"weights: {result['shards']} shards, {gib(result['weight_bytes'])}")
        if "header_gate" in result:
            gate = result["header_gate"]
            print(f"headers: {gate['tensors']} tensors, {gate['header_bytes']} bytes")
        print(result["scope"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
