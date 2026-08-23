#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Check the public GLM-5.3 checkpoint before any large download.

The check is deliberately metadata-only.  It compares the candidate FP8
configuration with the immutable GLM-5.2-FP8 base, validates the advertised
weight index, and prints the exact candidate revision to pin.  It does not
download model shards.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


BASELINE_REPO = "zai-org/GLM-5.2-FP8"
BASELINE_REVISION = "ba978f7d347eaf65d22f1a86833408afdb953541"
CANDIDATE_REPO = "zai-org/GLM-5.3-FP8"

# These fields define the execution contract WARP would have to implement.
# Post-training metadata and chat/reasoning templates are intentionally not
# included: GLM-5.3 is expected to change those while retaining the same base.
STRUCTURAL_FIELDS = (
    "architectures",
    "model_type",
    "hidden_size",
    "intermediate_size",
    "moe_intermediate_size",
    "num_hidden_layers",
    "first_k_dense_replace",
    "mlp_layer_types",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_head_dim",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "topk_method",
    "scoring_func",
    "norm_topk_prob",
    "routed_scaling_factor",
    "moe_router_dtype",
    "index_head_dim",
    "index_n_heads",
    "index_topk",
    "index_topk_freq",
    "index_skip_topk_offset",
    "indexer_types",
    "indexer_rope_interleave",
    "rope_interleave",
    "rope_parameters",
    "rms_norm_eps",
    "vocab_size",
    "eos_token_id",
    "num_nextn_predict_layers",
)

QUANT_FIELDS = ("quant_method", "fmt", "weight_block_size", "activation_scheme")
INDEX_NAME = "model.safetensors.index.json"


class ReleaseCheckError(RuntimeError):
    """A release is absent, incomplete, or outside the registered contract."""


@dataclass(frozen=True)
class RepoMetadata:
    repo: str
    revision: str
    files: frozenset[str]


def _json_url(url: str) -> Any:
    request = urllib.request.Request(
        url, headers={"User-Agent": "warp-glm53-release-check/1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 404):
            raise ReleaseCheckError(f"not publicly available: {url}") from exc
        raise ReleaseCheckError(f"HTTP {exc.code}: {url}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseCheckError(f"cannot read metadata: {url}: {exc}") from exc


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


def structural_differences(base: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    differences = []
    for field in STRUCTURAL_FIELDS:
        if base.get(field) != candidate.get(field):
            differences.append(
                f"{field}: baseline={base.get(field)!r}, "
                f"candidate={candidate.get(field)!r}")
    base_quant = base.get("quantization_config") or {}
    candidate_quant = candidate.get("quantization_config") or {}
    for field in QUANT_FIELDS:
        if base_quant.get(field) != candidate_quant.get(field):
            differences.append(
                f"quantization_config.{field}: "
                f"baseline={base_quant.get(field)!r}, "
                f"candidate={candidate_quant.get(field)!r}")
    return differences


def validate_index(index: dict[str, Any], files: frozenset[str]) -> tuple[int, int]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ReleaseCheckError("weight index has no non-empty weight_map")
    shards = {name for name in weight_map.values() if isinstance(name, str)}
    if len(shards) != len(set(weight_map.values())):
        raise ReleaseCheckError("weight index contains a non-string shard name")
    missing = sorted(shards - files)
    if missing:
        preview = ", ".join(missing[:5])
        raise ReleaseCheckError(
            f"weight index names {len(missing)} absent shard(s): {preview}")
    total_size = (index.get("metadata") or {}).get("total_size")
    if not isinstance(total_size, int) or total_size <= 0:
        raise ReleaseCheckError("weight index has no positive metadata.total_size")
    return len(shards), total_size


def gib(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=CANDIDATE_REPO)
    parser.add_argument(
        "--candidate-revision",
        help="candidate commit to verify; default resolves current public HEAD")
    parser.add_argument("--baseline", default=BASELINE_REPO)
    parser.add_argument("--baseline-revision", default=BASELINE_REVISION)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    baseline = repo_metadata(args.baseline, args.baseline_revision)
    candidate = repo_metadata(args.candidate, args.candidate_revision)

    required = {"config.json", INDEX_NAME, "tokenizer_config.json"}
    missing = sorted(required - candidate.files)
    if missing:
        raise ReleaseCheckError(
            f"{candidate.repo}@{candidate.revision}: missing {', '.join(missing)}")

    base_cfg = repo_json(baseline.repo, baseline.revision, "config.json")
    candidate_cfg = repo_json(candidate.repo, candidate.revision, "config.json")
    differences = structural_differences(base_cfg, candidate_cfg)
    if differences:
        detail = "\n  ".join(differences)
        raise ReleaseCheckError(
            "candidate is not structurally identical to the registered "
            f"GLM-5.2 base:\n  {detail}")

    index = repo_json(candidate.repo, candidate.revision, INDEX_NAME)
    shard_count, total_size = validate_index(index, candidate.files)
    return {
        "status": "READY_FOR_HEADER_GATE",
        "candidate": candidate.repo,
        "revision": candidate.revision,
        "shards": shard_count,
        "weight_bytes": total_size,
        "weight_gib": round(total_size / (1024 ** 3), 2),
        "architecture": candidate_cfg.get("architectures"),
        "model_type": candidate_cfg.get("model_type"),
        "note": "Metadata only; safetensor header/source gates still precede conversion.",
    }


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
        print(result["note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
