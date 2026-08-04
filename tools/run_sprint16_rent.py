#!/usr/bin/env python3
"""Run the small paired F/R cache-rent screen for Sprint 16."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

import run_sprint16_spec_probe as shared


F_CACHE_MIB = 59340
R_CACHE_MIB = 40502
DRAFT_CACHE_MIB = 16926
TARGET_ROLLBACK_BYTES = 536870912
DRAFT_ROLLBACK_BYTES = 134217728


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    value = shared.read_json(path)
    rows = value.get("cases")
    if not isinstance(rows, list):
        raise shared.CampaignError("corpus has no case list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise shared.CampaignError("invalid corpus case")
        tokens = row.get("token_ids")
        if not isinstance(tokens, list) or not tokens or not all(
            isinstance(token, int) and token >= 0 for token in tokens
        ):
            raise shared.CampaignError(f"invalid prompt tokens for {row['id']}")
        result[row["id"]] = row
    return result


def validate_target(value: Mapping[str, Any], *, resident: bool) -> None:
    if value.get("schema") != "waste.gn100.spec_target.v1":
        raise shared.CampaignError("invalid target schema")
    if value.get("generated") != shared.TOKENS:
        raise shared.CampaignError("target token count drift")
    for field, size in (
        ("tokens", shared.TOKENS),
        ("token_prefix_hashes", shared.TOKENS),
        ("route_prefix_hashes", shared.TOKENS),
        ("logit_prefix_hashes", shared.TOKENS + 1),
    ):
        if not isinstance(value.get(field), list) or len(value[field]) != size:
            raise shared.CampaignError(f"target {field} dimension drift")
    cache = value.get("cache")
    expected_slots = 5015 if not resident else 3423
    if (
        not isinstance(cache, dict)
        or cache.get("slots") != expected_slots
        or cache.get("warm_ready") != expected_slots
    ):
        raise shared.CampaignError(f"target cache drift: {cache}")
    shared.check_io_cuda(value, draft=False)
    shared.check_process_safety(
        value,
        "resident target" if resident else "full target",
        "draft_prefill_plus_target_prefill_decode" if resident else "prefill_plus_decode",
    )
    if not resident:
        return
    control = value.get("resident_control")
    if not isinstance(control, dict):
        raise shared.CampaignError("resident control evidence is missing")
    required = {
        "enabled": True,
        "speculation_enabled": False,
        "draft_kept_loaded_during_target_decode": True,
        "target_cache_mib": R_CACHE_MIB,
        "draft_cache_mib": DRAFT_CACHE_MIB,
        "target_rollback_bytes": TARGET_ROLLBACK_BYTES,
        "draft_rollback_bytes": DRAFT_ROLLBACK_BYTES,
        "draft_fully_resident": True,
        "draft_cuda_fallbacks": 0,
    }
    if any(control.get(key) != expected for key, expected in required.items()):
        raise shared.CampaignError(f"resident control drift: {control}")
    if control.get("memavailable_after_prompt_snapshots_kib", 0) < shared.MIN_MEMAVAILABLE_KIB:
        raise shared.CampaignError("resident control lost the 24 GiB memory floor")


def same_trajectory(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = (
        "tokens",
        "token_hash",
        "logit_row_hashes",
        "logit_hash",
        "route_row_hashes",
        "route_hash",
    )
    return all(left.get(field) == right.get(field) for field in fields)


def row(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decode_seconds": value["decode_seconds"],
        "tok_s": value["tok_s"],
        "decode_hits": value["decode_hits"],
        "decode_misses": value["decode_misses"],
        "decode_bytes": value["decode_bytes"],
    }


def run(args: argparse.Namespace) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise shared.CampaignError("output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = shared.check_capture_runtime()
    cases = load_cases(args.corpus)
    selected = args.case or list(shared.A_CASES + shared.B_CASES)
    if len(selected) != len(set(selected)) or any(case_id not in cases for case_id in selected):
        raise shared.CampaignError("unknown or repeated case selection")
    shared.write_json(args.output / "host-profile.json", shared.host_profile())
    shared.write_json(args.output / "run-contract.json", {
        "schema": "waste.gn100.spec_rent_contract.v1",
        "runtime": runtime,
        "cases": selected,
        "arm_order": "F,R on even case index; R,F on odd case index",
        "F_target_cache_mib": F_CACHE_MIB,
        "R_target_cache_mib": R_CACHE_MIB,
        "R_draft_cache_mib": DRAFT_CACHE_MIB,
        "target_rollback_bytes": TARGET_ROLLBACK_BYTES,
        "draft_rollback_bytes": DRAFT_ROLLBACK_BYTES,
        "speculation_enabled": False,
        "h2": "embargoed_unspent",
    })

    summaries: list[dict[str, Any]] = []
    for index, case_id in enumerate(selected):
        prompt = shared.ids(cases[case_id]["token_ids"])
        reference = shared.read_json(args.capture / "target" / f"{case_id}.json")
        commands = {
            "F": [
                str(args.probe), "target", str(args.target_model), str(F_CACHE_MIB),
                str(args.target_usage), prompt, str(shared.TOKENS),
            ],
            "R": [
                str(args.probe), "resident-target", str(args.target_model),
                str(args.target_usage), str(args.draft_model), str(args.draft_usage),
                prompt, prompt, str(shared.TOKENS),
            ],
        }
        values: dict[str, dict[str, Any]] = {}
        order = ("F", "R") if index % 2 == 0 else ("R", "F")
        for arm in order:
            path = args.output / "rows" / case_id / f"{arm}.json"
            values[arm] = shared.run_probe(commands[arm], path)
            validate_target(values[arm], resident=arm == "R")
            if not same_trajectory(values[arm], reference):
                raise shared.CampaignError(f"{case_id} {arm} trajectory differs from capture")
        if not same_trajectory(values["F"], values["R"]):
            raise shared.CampaignError(f"{case_id} F/R trajectory mismatch")
        f, r = values["F"], values["R"]
        summaries.append({
            "case_id": case_id,
            "F": row(f),
            "R": row(r),
            "R_over_F_decode_seconds": r["decode_seconds"] / f["decode_seconds"],
            "R_vs_F_gain": f["decode_seconds"] / r["decode_seconds"] - 1.0,
            "trajectory_exact": True,
        })
        shared.write_json(args.output / "progress.json", {
            "completed": [entry["case_id"] for entry in summaries],
            "h2": "embargoed_unspent",
        })

    ratios = [entry["R_over_F_decode_seconds"] for entry in summaries]
    gains = [entry["R_vs_F_gain"] for entry in summaries]
    shared.write_json(args.output / "summary.json", {
        "schema": "waste.gn100.spec_rent_summary.v1",
        "status": "valid_paired_rent_screen",
        "cases": summaries,
        "median_R_over_F_decode_seconds": statistics.median(ratios),
        "median_R_vs_F_gain": statistics.median(gains),
        "h2": {"status": "embargoed_unspent", "model_steps": 0},
    })
    shared.manifest(args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--draft-model", type=Path, required=True)
    parser.add_argument("--target-usage", type=Path, required=True)
    parser.add_argument("--draft-usage", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append")
    return parser.parse_args()


def main() -> int:
    try:
        run(parse_args())
    except (OSError, ValueError, KeyError, TypeError, shared.CampaignError) as exc:
        print(f"run_sprint16_rent: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
