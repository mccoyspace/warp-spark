#!/usr/bin/env python3
"""Run the frozen A+B reduced-expert K3 self-draft screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


CASES = (
    "composition_a", "revision_a", "color_value_a", "material_a",
    "composition_b", "revision_b", "color_value_b", "material_b",
)
CORPUS_SHA256 = "4fb4ca60d82ed521c0d59732c748a5f99b22f720d35e19d5590b41bc01c423fe"
BASELINE_SECONDS = 302.954579022
MIN_GAIN = 0.15
OPTIMISTIC_VERIFIER_SECONDS = 1.5
PROFILE = {
    "WASTE_BACKEND": "auto", "WASTE_CUDA_KDA": "1",
    "WASTE_CUDA_DENSE": "2", "WASTE_CUDA_VQ": "2",
    "WASTE_CUDA_VQ_GROUP": "1", "WASTE_THREADS": "10",
    "WASTE_Q8": "1", "WASTE_SDOT": "0", "WASTE_I8MM": "0",
    "WASTE_LOOKAHEAD": "0", "WASTE_IO_THREADS": "2",
    "WASTE_IO_DEPTH": "2", "WASTE_LFRU_AGE_TOKENS": "0",
    "WASTE_LFRU_PRIOR_LOG2": "0", "WASTE_MLOCK": "0",
    "WASTE_PURGEABLE": "0", "WASTE_VERIFY": "0", "WASTE_DIRECT": "1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--usage", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-mb", default="59340")
    args = parser.parse_args()

    for path in (args.binary, args.model, args.usage, args.corpus):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
    if sha256(args.corpus) != CORPUS_SHA256:
        raise SystemExit("frozen corpus SHA-256 mismatch")
    if args.out.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out}")
    args.out.mkdir(parents=True)

    corpus = json.loads(args.corpus.read_text())
    by_id = {case["id"]: case for case in corpus["cases"]}
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("WASTE_")}
    env.update(PROFILE)
    rows = []

    for case_id in CASES:
        case = by_id[case_id]
        target_path = args.targets / f"{case_id}.json"
        target = json.loads(target_path.read_text())
        prompt_ids = ",".join(map(str, case["token_ids"]))
        target_ids = ",".join(map(str, target["tokens"]))
        command = [str(args.binary), "selfdraft", str(args.model),
                   args.cache_mb, str(args.usage), prompt_ids, target_ids]
        started = time.monotonic()
        process = subprocess.run(command, env=env, text=True,
                                 capture_output=True)
        elapsed = time.monotonic() - started
        stdout_path = args.out / f"{case_id}.json"
        stderr_path = args.out / f"{case_id}.stderr.txt"
        stdout_path.write_text(process.stdout)
        stderr_path.write_text(process.stderr)
        write_json(args.out / f"{case_id}.run.json", {
            "command": command[:5] + ["<prompt-ids>", "<target-ids>"],
            "elapsed_seconds": elapsed,
            "returncode": process.returncode,
            "stdout_sha256": sha256(stdout_path),
        })
        if process.returncode:
            raise SystemExit(f"{case_id} failed; evidence retained in {args.out}")
        value = json.loads(process.stdout)
        safety = value.get("process_safety", {})
        profile = value.get("profile", {})
        exact = value.get("final_state", {}).get("hash") == target.get("state_hash")
        safe = (safety.get("vmswap_kib") == 0 and
                safety.get("timed_major_faults_delta") == 0 and
                value.get("draft_cuda_delta", {}).get("fallbacks") == 0 and
                profile.get("fallbacks") == 0 and
                profile.get("full_top_k_restored") is True)
        rows.append({
            "case": case_id,
            "family": case["family"],
            "blocks": value["blocks"],
            "committed_tokens": value["agreement"]["committed_tokens"],
            "committed_per_block": value["agreement"]["committed_per_block"],
            "marginal_agreement": value["agreement"]["marginal"],
            "draft_branch_seconds": value["timing"]["draft_branch_seconds"],
            "draft_branch_bytes": sum(value["branch_bytes"]),
            "canonical_final_state_exact": exact,
            "safety_and_fallback_gate": safe,
            "elapsed_seconds": elapsed,
        })
        print(f"{case_id}: {value['blocks']} blocks, "
              f"{value['agreement']['committed_per_block']:.3f} committed/block, "
              f"{value['timing']['draft_branch_seconds']:.3f}s draft", flush=True)

    total_blocks = sum(row["blocks"] for row in rows)
    draft_seconds = sum(row["draft_branch_seconds"] for row in rows)
    optimistic_seconds = draft_seconds + total_blocks * OPTIMISTIC_VERIFIER_SECONDS
    maximum_seconds = BASELINE_SECONDS / (1.0 + MIN_GAIN)
    gate = (optimistic_seconds <= maximum_seconds and
            all(row["canonical_final_state_exact"] and
                row["safety_and_fallback_gate"] for row in rows))
    summary = {
        "schema": "waste.gn100.k3_selfdraft_campaign.v1",
        "status": "optimistic_viability_pass" if gate else "viability_miss",
        "candidate": {"draft_experts": 4, "target_experts": 16, "width": 4,
                      "first_proposal_reuses_exact_root_logits": True},
        "profile_environment": PROFILE,
        "rows": rows,
        "totals": {
            "tokens": sum(row["committed_tokens"] for row in rows),
            "blocks": total_blocks,
            "draft_branch_seconds": draft_seconds,
            "optimistic_verifier_seconds_per_block": OPTIMISTIC_VERIFIER_SECONDS,
            "optimistic_candidate_seconds": optimistic_seconds,
            "full_cache_baseline_seconds": BASELINE_SECONDS,
            "maximum_seconds_for_15pct_throughput_gain": maximum_seconds,
            "optimistic_projected_gain": BASELINE_SECONDS / optimistic_seconds - 1.0,
        },
        "gate": {
            "all_final_state_hashes_exact": all(
                row["canonical_final_state_exact"] for row in rows),
            "all_safety_and_fallback_checks_pass": all(
                row["safety_and_fallback_gate"] for row in rows),
            "optimistic_time_within_budget": optimistic_seconds <= maximum_seconds,
            "pass": gate,
        },
        "scope": {
            "integrated_decoder": False, "h1_model_steps": 0,
            "h2_model_steps": 0, "nvme_of_hardware": "deferred",
        },
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary["totals"], indent=2, sort_keys=True))
    print(f"decision: {summary['status']}")
    return 0 if gate else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"run_k3_selfdraft: {exc}", file=sys.stderr)
        raise SystemExit(1)
