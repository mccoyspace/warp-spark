#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Compare exact-prefix and no-prefix server acceptance records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


def load(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n}: {e}") from e
    return rows


def choose(rows: list[dict], run_id: str) -> dict:
    found = [r for r in rows if r.get("run_id") == run_id]
    if len(found) != 1:
        raise ValueError(f"expected one {run_id!r} record, found {len(found)}")
    return found[0]


def incremental(requests: list[dict], key: str, index: int) -> int | None:
    try:
        now = int(requests[index]["waste"][key])
        before = int(requests[index - 1]["waste"][key]) if index else 0
        return now - before
    except (KeyError, TypeError, ValueError):
        return None


def safe_ratio(a, b):
    return a / b if a is not None and b else None


def compare(control: dict, treatment: dict) -> dict:
    errors: list[str] = []
    for name, run in (("control", control), ("treatment", treatment)):
        if run.get("error"):
            errors.append(f"{name} error: {run['error']}")
        if run.get("server_returncode") != 0:
            errors.append(f"{name} server return code {run.get('server_returncode')}")
        if len(run.get("requests", [])) < 2:
            errors.append(f"{name} has fewer than two requests")
    comparable = ("manifest_sha256", "budget_bytes", "threads", "cpu_set",
                  "resolved_cpu_set", "io_backend_requested",
                  "io_queue_depth_requested", "tokens_requested")
    mismatches = {k: [control.get(k), treatment.get(k)] for k in comparable
                  if control.get(k) != treatment.get(k)}
    if (control.get("prompt_sha256") and treatment.get("prompt_sha256") and
            control["prompt_sha256"] != treatment["prompt_sha256"]):
        mismatches["prompt_sha256"] = [control["prompt_sha256"],
                                       treatment["prompt_sha256"]]
    if mismatches:
        errors.append(f"non-comparable configuration: {mismatches}")
    if errors:
        return {"schema": "waste.prefix_compare.v1", "valid": False,
                "errors": errors, "mismatches": mismatches}

    cr, tr = control["requests"], treatment["requests"]
    hashes = [r.get("answer_sha256") for r in cr + tr]
    outputs_identical = bool(hashes) and None not in hashes and len(set(hashes)) == 1
    control_s = [float(r["wall_seconds"]) for r in cr]
    treatment_s = [float(r["wall_seconds"]) for r in tr]
    control_reference = statistics.mean(control_s)
    hit_s = treatment_s[1]
    prefix = tr[1].get("waste", {}).get("prefix_cache", {})
    first_prefix = tr[0].get("waste", {}).get("prefix_cache", {})

    control_bytes = incremental(cr, "bytes_read", 1)
    hit_bytes = incremental(tr, "bytes_read", 1)
    control_misses = incremental(cr, "experts_missed", 1)
    hit_misses = incremental(tr, "experts_missed", 1)
    control_hits = incremental(cr, "experts_hit", 1)
    hit_hits = incremental(tr, "experts_hit", 1)

    vm_keys = ("pswpout", "pgscan_direct", "pgscan_kswapd",
               "pgsteal_direct", "pgsteal_kswapd")
    memory_safe = all(
        run.get("peak_process_swap_bytes", 0) == 0 and
        all(run.get("vm_delta", {}).get(k, 0) == 0 for k in vm_keys) and
        all(v == 0 for v in run.get("memory_psi_total_us_delta", {}).values())
        for run in (control, treatment))
    cache_contract = (
        first_prefix.get("status") == "miss_stored" and
        prefix.get("status") == "hit" and
        prefix.get("prompt_tokens_evaluated") == 1 and
        prefix.get("reused_tokens") == prefix.get("key_tokens", 0) - 1)
    direct_io = all(bool(r.get("waste", {}).get("direct_io"))
                    and not r.get("waste", {}).get("io_fallback")
                    for r in cr + tr)

    return {
        "schema": "waste.prefix_compare.v1",
        "valid": True,
        "accepted": outputs_identical and memory_safe and cache_contract and direct_io,
        "control_run": control["run_id"],
        "treatment_run": treatment["run_id"],
        "configuration": {k: control.get(k) for k in comparable},
        "outputs_identical": outputs_identical,
        "answer_sha256": hashes[0] if outputs_identical else None,
        "direct_io_no_fallback": direct_io,
        "memory_safe": memory_safe,
        "cache_contract": cache_contract,
        "latency": {
            "control_requests_s": control_s,
            "control_mean_s": control_reference,
            "treatment_requests_s": treatment_s,
            "hit_s": hit_s,
            "hit_speedup_vs_control_mean": safe_ratio(control_reference, hit_s),
            "hit_latency_reduction_fraction": 1 - hit_s / control_reference,
            "cold_overhead_fraction_vs_control_first":
                treatment_s[0] / control_s[0] - 1,
        },
        "prefix": {
            "snapshot_bytes": prefix.get("snapshot_bytes"),
            "capacity_bytes": prefix.get("capacity_bytes"),
            "resident_bytes": prefix.get("resident_bytes"),
            "reused_tokens": prefix.get("reused_tokens"),
            "key_tokens": prefix.get("key_tokens"),
            "restore_ms": prefix.get("restore_ms"),
            "export_ms": first_prefix.get("export_ms"),
            "cold_prefill_prepare_ms": first_prefix.get("prepare_ms"),
        },
        "request_2_engine_delta": {
            "control_bytes_read": control_bytes,
            "hit_bytes_read": hit_bytes,
            "bytes_reduction_fraction":
                1 - hit_bytes / control_bytes
                if control_bytes and hit_bytes is not None else None,
            "control_expert_misses": control_misses,
            "hit_expert_misses": hit_misses,
            "control_expert_hits": control_hits,
            "hit_expert_hits": hit_hits,
        },
        "memory": {
            "control_peak_rss_bytes": control.get("peak_rss_bytes"),
            "treatment_peak_rss_bytes": treatment.get("peak_rss_bytes"),
            "control_min_mem_available_bytes": control.get("min_mem_available_bytes"),
            "treatment_min_mem_available_bytes": treatment.get("min_mem_available_bytes"),
            "control_vm_delta": control.get("vm_delta"),
            "treatment_vm_delta": treatment.get("vm_delta"),
        },
    }


def markdown(result: dict) -> str:
    if not result.get("valid"):
        return "# Prefix-cache comparison\n\nINVALID: " + "; ".join(result["errors"]) + "\n"
    lat, pfx, io = result["latency"], result["prefix"], result["request_2_engine_delta"]
    return f"""# Exact-prefix cache acceptance

Accepted: **{'yes' if result['accepted'] else 'no'}**. Outputs identical:
**{'yes' if result['outputs_identical'] else 'no'}**. Direct I/O/no fallback:
**{'yes' if result['direct_io_no_fallback'] else 'no'}**. Memory gates:
**{'pass' if result['memory_safe'] else 'fail'}**.

| Metric | Control | Exact-prefix hit | Change |
|---|---:|---:|---:|
| Request latency | {lat['control_mean_s']:.3f} s mean | {lat['hit_s']:.3f} s | {lat['hit_speedup_vs_control_mean']:.2f}x faster |
| Expert bytes read, request 2 | {io['control_bytes_read']} | {io['hit_bytes_read']} | {100 * io['bytes_reduction_fraction']:.2f}% less |
| Expert misses, request 2 | {io['control_expert_misses']} | {io['hit_expert_misses']} | — |

Snapshot: {pfx['snapshot_bytes']} bytes in a {pfx['capacity_bytes']} byte
reservation; {pfx['reused_tokens']} of {pfx['key_tokens']} prompt tokens
reused. Export took {pfx['export_ms']} ms and restore took {pfx['restore_ms']}
ms. The enabled cold request overhead versus control request 1 was
{100 * lat['cold_overhead_fraction_vs_control_first']:.2f}%.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("records", type=Path)
    ap.add_argument("--control", required=True)
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args(argv)
    try:
        rows = load(args.records)
        result = compare(choose(rows, args.control), choose(rows, args.treatment))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        text = markdown(result)
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(text)
        print(text, end="")
        return 0 if result.get("accepted") else 1
    except (OSError, ValueError) as e:
        print(f"analyze_prefix_cache: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
