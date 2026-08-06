#!/usr/bin/env python3
"""Run the frozen A+B K3 self-draft plus exact-verifier campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


CASES = ("composition_a", "revision_a", "color_value_a", "material_a",
         "composition_b", "revision_b", "color_value_b", "material_b")
CORPUS_SHA256 = "4fb4ca60d82ed521c0d59732c748a5f99b22f720d35e19d5590b41bc01c423fe"
BASELINE_SECONDS = 302.954579022
MAXIMUM_SECONDS = BASELINE_SECONDS / 1.15
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
        target = json.loads((args.targets / f"{case_id}.json").read_text())
        command = [str(args.binary), "selfdraft-exact", str(args.model),
                   args.cache_mb, str(args.usage),
                   ",".join(map(str, case["token_ids"])),
                   ",".join(map(str, target["tokens"]))]
        started = time.monotonic()
        process = subprocess.run(command, env=env, text=True,
                                 capture_output=True)
        elapsed = time.monotonic() - started
        output = args.out / f"{case_id}.json"
        output.write_text(process.stdout)
        output.with_suffix(".stderr.txt").write_text(process.stderr)
        write_json(output.with_suffix(".run.json"), {
            "command": command[:5] + ["<prompt-ids>", "<target-ids>"],
            "elapsed_seconds": elapsed, "returncode": process.returncode,
            "stdout_sha256": sha256(output),
        })
        if process.returncode:
            raise SystemExit(f"{case_id} failed; evidence retained")
        value = json.loads(process.stdout)
        verifier = value["exact_verifier"]
        exact_state = value["final_state"]["hash"] == target["state_hash"]
        safe = (value["process_safety"]["vmswap_kib"] == 0 and
                verifier["timed_major_faults"] == 0 and
                value["draft_cuda_delta"]["fallbacks"] == 0 and
                value["exact_verifier_cuda_delta"]["fallbacks"] == 0 and
                value["profile"]["fallbacks"] == 0)
        row = {
            "case": case_id, "family": case["family"],
            "blocks": value["blocks"],
            "committed_tokens": value["agreement"]["committed_tokens"],
            "draft_seconds": value["timing"]["draft_branch_seconds"],
            "verifier_seconds": verifier["seconds"],
            "verifier_misses": verifier["misses"],
            "verifier_bytes": verifier["bytes"],
            "verifier_union": verifier["expert_union"],
            "snapshot_seconds": value["timing"]["snapshot_seconds"],
            "pre_verifier_restore_seconds": value["timing"]["restore_seconds"],
            "post_verifier_restore_seconds": verifier["post_verifier_restore_seconds"],
            "canonical_final_state_exact": exact_state,
            "safety_and_fallback_gate": safe,
            "elapsed_seconds": elapsed,
        }
        rows.append(row)
        print(f"{case_id}: {row['blocks']} blocks, "
              f"{row['verifier_seconds']:.3f}s verifier, "
              f"{row['verifier_bytes'] / 2**30:.3f} GiB", flush=True)

    blocks = sum(row["blocks"] for row in rows)
    draft = sum(row["draft_seconds"] for row in rows)
    verifier = sum(row["verifier_seconds"] for row in rows)
    lower_bound = draft + verifier
    transaction = sum(row["snapshot_seconds"] +
                      row["pre_verifier_restore_seconds"] +
                      row["post_verifier_restore_seconds"] for row in rows)
    exact_safe = all(row["canonical_final_state_exact"] and
                     row["safety_and_fallback_gate"] for row in rows)
    pass_gate = exact_safe and lower_bound + transaction <= MAXIMUM_SECONDS
    summary = {
        "schema": "waste.gn100.exact_verifier_campaign.v1",
        "status": "pass" if pass_gate else "verifier_viability_miss",
        "rows": rows,
        "totals": {
            "tokens": sum(row["committed_tokens"] for row in rows),
            "blocks": blocks, "draft_seconds": draft,
            "verifier_seconds": verifier,
            "verifier_seconds_per_block": verifier / blocks,
            "verifier_misses": sum(row["verifier_misses"] for row in rows),
            "verifier_bytes": sum(row["verifier_bytes"] for row in rows),
            "verifier_union": sum(row["verifier_union"] for row in rows),
            "draft_plus_verifier_lower_bound_seconds": lower_bound,
            "measured_snapshot_and_restore_seconds": transaction,
            "lower_bound_plus_transaction_seconds": lower_bound + transaction,
            "baseline_seconds": BASELINE_SECONDS,
            "maximum_seconds_for_15pct_throughput_gain": MAXIMUM_SECONDS,
            "projected_gain_before_unimplemented_replay_and_sync":
                BASELINE_SECONDS / (lower_bound + transaction) - 1.0,
        },
        "gate": {
            "all_canonical_final_states_exact": all(
                row["canonical_final_state_exact"] for row in rows),
            "all_safety_and_fallback_checks_pass": all(
                row["safety_and_fallback_gate"] for row in rows),
            "time_within_budget": lower_bound + transaction <= MAXIMUM_SECONDS,
            "pass": pass_gate,
        },
        "scope": {"integrated_decoder": False, "h1_model_steps": 0,
                  "h2_model_steps": 0, "nvme_of_hardware": "deferred"},
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary["totals"], indent=2, sort_keys=True))
    print(f"decision: {summary['status']}")
    return 0 if pass_gate else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"run_exact_verifier_campaign: {exc}", file=sys.stderr)
        raise SystemExit(1)
