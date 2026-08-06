#!/usr/bin/env python3
"""Run one frozen Sprint 17 block through a verify4 diagnostic arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


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


def ids(values: list[int]) -> str:
    return ",".join(map(str, values)) if values else "-"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("serial", "exact"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--usage", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--selfdraft", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-mb", default="59340")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    for path in (args.binary, args.model, args.usage, args.corpus,
                 args.selfdraft, args.targets):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    corpus = json.loads(args.corpus.read_text())
    case = next((item for item in corpus["cases"]
                 if item["id"] == args.case), None)
    if case is None:
        raise SystemExit(f"case not found: {args.case}")
    trace = json.loads((args.selfdraft / f"{args.case}.json").read_text())
    target = json.loads((args.targets / f"{args.case}.json").read_text())
    if args.block < 0 or args.block >= trace["blocks"]:
        raise SystemExit("block index out of range")
    if trace["block_widths"][args.block] != 4:
        raise SystemExit("pilot requires a complete width-four block")
    root = trace["block_roots"][args.block]
    proposal = trace["proposals"][args.block][:4]
    command = [str(args.binary), "verify4", args.arm, str(args.model),
               args.cache_mb, str(args.usage), ids(case["token_ids"]),
               ids(target["tokens"][:root]), ids(proposal)]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("WASTE_")}
    env.update(PROFILE)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = subprocess.run(command, env=env, text=True, capture_output=True)
    elapsed = time.monotonic() - started
    args.out.write_text(process.stdout)
    args.out.with_suffix(".stderr.txt").write_text(process.stderr)
    run = {
        "schema": "waste.gn100.exact_verifier_block_run.v1",
        "arm": args.arm, "case": args.case, "block": args.block,
        "root": root, "accepted_prefix": trace["accepted_prefix"][args.block],
        "command": command[:6] + ["<prompt-ids>", "<root-ids>",
                                  "<proposal-ids>"],
        "profile_environment": PROFILE,
        "elapsed_seconds": elapsed, "returncode": process.returncode,
        "stdout_sha256": sha256(args.out),
    }
    args.out.with_suffix(".run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n")
    if process.returncode:
        raise SystemExit(process.returncode)
    value = json.loads(process.stdout)
    print(json.dumps({
        "arm": args.arm, "case": args.case, "block": args.block,
        "accepted_prefix": run["accepted_prefix"],
        "verifier_seconds": value["timing"]["verifier_seconds"],
        "misses": value["cache_delta"]["misses"],
        "bytes": value["cache_delta"]["bytes"],
        "union": value["cache_delta"]["chunk_expert_union"],
        "final_state_hash": value["final_state"]["hash"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, StopIteration, ValueError,
            json.JSONDecodeError) as exc:
        print(f"run_exact_verifier_block: {exc}", file=sys.stderr)
        raise SystemExit(1)
