#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Gate a matched real-K3 semantic-family prefix-cache campaign.

The runner keeps raw responses, layer traces and Linux telemetry.  This
analyzer consumes its two append-only JSONL records and makes every acceptance
decision explicit; a Markdown report is a rendering of the JSON result, not a
second source of truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


SCHEMA = "waste.family_prefix_analysis.v1"
SWAP_KEYS = ("pswpin", "pswpout")
RECLAIM_KEYS = ("pgscan_direct", "pgscan_kswapd",
                "pgsteal_direct", "pgsteal_kswapd")
EXPECTED_ENGINE_ENV = {
    "WASTE_EXPERT_SCHED": "row", "WASTE_PROFILE": "0",
    "WASTE_Q8": "1", "WASTE_SDOT": "0", "WASTE_I8MM": "0",
    "WASTE_VERIFY": "0",
}
COMPARABLE = (
    "campaign_id", "manifest_sha256", "usage_sha256", "library_sha256",
    "budget_bytes", "rss_limit_bytes", "max_memory_psi_us", "threads",
    "cpu_set", "resolved_cpu_set", "io_backend_requested",
    "io_queue_depth_requested", "direct_io_requested", "tokens_requested",
    "engine_environment", "system_sha256", "stable_tool_sha256",
    "changed_tool_sha256", "prompt_b_sha256",
)


def load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{lineno}: record is not an object")
        rows.append(row)
    return rows


def choose(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
    found = [row for row in rows if row.get("role") == role]
    if len(found) != 1:
        raise ValueError(
            f"expected one {role!r} record, found {len(found)}")
    return found[0]


def request(run: dict[str, Any], name: str) -> dict[str, Any] | None:
    found = [row for row in run.get("requests", [])
             if row.get("name") == name]
    return found[0] if len(found) == 1 else None


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _output_valid(row: dict[str, Any]) -> bool:
    value = row.get("stable_output")
    digest = row.get("stable_output_sha256")
    return value is not None and digest == canonical_sha256(value)


def _prefix(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get("response_waste") or {}).get("prefix_cache") or {}


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _transport(run: dict[str, Any]) -> bool:
    meta = run.get("trace_meta") or {}
    if (meta.get("io_backend") != "io_uring" or
            _int(meta.get("queue_depth")) != 4 or
            bool(meta.get("io_fallback"))):
        return False
    rows = run.get("requests") or []
    if not rows:
        return False
    for row in rows:
        waste = row.get("response_waste") or {}
        trace = row.get("trace") or {}
        if (not bool(waste.get("direct_io")) or
                waste.get("io_backend") != "io_uring" or
                _int(waste.get("io_queue_depth")) != 4 or
                bool(waste.get("io_fallback")) or
                not bool(trace.get("transport_ok")) or
                not bool(trace.get("boundary_complete")) or
                bool(trace.get("read_error"))):
            return False
    return True


def _all_vm_zero(run: dict[str, Any], keys: tuple[str, ...]) -> bool:
    vm = run.get("vm_delta") or {}
    return all(_int(vm.get(key)) == 0 for key in keys)


def _psi_safe(run: dict[str, Any]) -> bool:
    limit = _int(run.get("max_memory_psi_us"))
    values = run.get("memory_psi_total_us_delta") or {}
    if limit is None or any(key not in values for key in ("some", "full")):
        return False
    return all(0 <= int(values[key]) <= limit for key in ("some", "full"))


def _routes_match(cold: dict[str, Any], hit: dict[str, Any]) -> bool:
    """Match generated-token routes, allowing hit's final-prompt replay.

    A cold multi-token prompt is handled by the chunked prefill path.  After
    restoring n-1 state, the cache path intentionally evaluates the final
    prompt token through the one-token path to regenerate logits.  Its trace
    therefore has exactly one additional routed position immediately before
    the generated-token positions shared with control.
    """
    cold_positions = cold.get("decode_route_positions") or {}
    hit_positions = hit.get("decode_route_positions") or {}
    if not cold_positions or not hit_positions:
        return False
    try:
        cold_keys = {int(key) for key in cold_positions}
        hit_keys = {int(key) for key in hit_positions}
    except (TypeError, ValueError):
        return False
    if not cold_keys <= hit_keys:
        return False
    extras = hit_keys - cold_keys
    if len(extras) != 1 or next(iter(extras)) != min(cold_keys) - 1:
        return False
    return all(hit_positions[str(position)] == cold_positions[str(position)]
               for position in cold_keys)


def compare(control: dict[str, Any], treatment: dict[str, Any]
            ) -> dict[str, Any]:
    """Return the complete machine-readable verdict for one matched pair."""
    errors: list[str] = []
    if control.get("role") != "control":
        errors.append("control record has the wrong role")
    if treatment.get("role") != "prefix":
        errors.append("prefix record has the wrong role")
    for name, run in (("control", control), ("prefix", treatment)):
        if run.get("error"):
            errors.append(f"{name} error: {run['error']}")
        if run.get("server_returncode") != 0:
            errors.append(
                f"{name} server return code {run.get('server_returncode')}")

    control_seed = request(control, "seed_a")
    cold = request(control, "cold_b")
    seed = request(treatment, "seed_a")
    hit = request(treatment, "family_b")
    changed = request(treatment, "changed_tool_b")
    for name, row in (("control seed_a", control_seed),
                      ("control cold_b", cold), ("prefix seed_a", seed),
                      ("prefix family_b", hit),
                      ("prefix changed_tool_b", changed)):
        if row is None:
            errors.append(f"missing or duplicate request {name}")

    mismatches = {
        key: [control.get(key), treatment.get(key)]
        for key in COMPARABLE if control.get(key) != treatment.get(key)
    }
    if mismatches:
        errors.append(f"non-comparable configuration: {mismatches}")
    if errors:
        return {"schema": SCHEMA, "valid": False, "accepted": False,
                "errors": errors, "mismatches": mismatches, "checks": {}}

    assert control_seed is not None and cold is not None and seed is not None
    assert hit is not None and changed is not None
    cold_prefix, seed_prefix = _prefix(cold), _prefix(seed)
    hit_prefix, changed_prefix = _prefix(hit), _prefix(changed)
    cold_trace, hit_trace = cold.get("trace") or {}, hit.get("trace") or {}

    output_artifacts_valid = all(
        _output_valid(row) for row in (control_seed, cold, seed, hit))
    output_canonical_identical = (
        output_artifacts_valid and
        cold["stable_output_sha256"] == hit["stable_output_sha256"])
    seed_request_matched = (
        control_seed.get("request_sha256") == seed.get("request_sha256") and
        control_seed.get("tool_sha256") == seed.get("tool_sha256") and
        control_seed.get("stable_output_sha256") ==
        seed.get("stable_output_sha256") and
        _routes_match(control_seed.get("trace") or {},
                      seed.get("trace") or {}))
    prompt_b_matched = (
        cold.get("request_sha256") == hit.get("request_sha256") and
        cold.get("tool_sha256") == treatment.get("stable_tool_sha256") and
        hit.get("tool_sha256") == treatment.get("stable_tool_sha256") and
        _int((cold.get("usage") or {}).get("prompt_tokens")) ==
        _int((hit.get("usage") or {}).get("prompt_tokens")))

    root_depth = _int(hit_prefix.get("family_root_tokens"))
    reused = _int(hit_prefix.get("reused_tokens"))
    checkpoint = _int(hit_prefix.get("checkpoint_tokens"))
    key_tokens = _int(hit_prefix.get("key_tokens"))
    evaluated = _int(hit_prefix.get("prompt_tokens_evaluated"))
    replayed = _int(hit_prefix.get("replayed_tokens"))
    prefill_tokens = _int(hit_trace.get("prefill_tokens"))
    family_root_replay = (
        seed_prefix.get("status") == "miss_stored_family" and
        not bool(seed_prefix.get("hit")) and
        hit_prefix.get("status") == "hit" and
        bool(hit_prefix.get("hit")) and
        hit_prefix.get("hit_kind") == "family" and
        root_depth is not None and root_depth > 0 and
        _int(seed_prefix.get("family_root_tokens")) == root_depth and
        reused == root_depth and checkpoint == root_depth and
        key_tokens is not None and evaluated is not None and
        evaluated >= 2 and evaluated == key_tokens - root_depth and
        replayed == evaluated and
        prefill_tokens == evaluated - 1 and
        (hit_trace.get("prefill_position_start") == root_depth) and
        cold_prefix.get("status") == "bypass_disabled" and
        _int(cold_prefix.get("key_tokens")) == key_tokens)

    routes_identical = _routes_match(cold_trace, hit_trace)
    cold_bytes = _int(cold_trace.get("expert_bytes_read"))
    hit_bytes = _int(hit_trace.get("expert_bytes_read"))
    bytes_reduced = (
        cold_bytes is not None and cold_bytes > 0 and
        hit_bytes is not None and 0 <= hit_bytes < cold_bytes)

    capacity = _int(hit_prefix.get("capacity_bytes"))
    resident = _int(hit_prefix.get("resident_bytes"))
    snapshot = _int(hit_prefix.get("snapshot_bytes"))
    reserve = _int(treatment.get("prefix_cache_bytes"))
    budget = _int(treatment.get("budget_bytes"))
    snapshot_within_reservation = (
        reserve is not None and reserve > 0 and capacity == reserve and
        snapshot is not None and 0 < snapshot <= capacity and
        resident is not None and snapshot <= resident <= capacity and
        _int(hit_prefix.get("family_entries")) == 1 and
        _int(hit_prefix.get("exact_entries")) == 0)
    fixed_budget = (
        budget is not None and budget > 0 and
        _int(control.get("budget_bytes")) == budget and
        _int(control.get("prefix_cache_bytes")) == 0 and
        reserve is not None and 0 < reserve < budget)

    rss_safe = all(
        _int(run.get("peak_rss_bytes")) is not None and
        _int(run.get("rss_limit_bytes")) is not None and
        0 < int(run["peak_rss_bytes"]) <= int(run["rss_limit_bytes"])
        for run in (control, treatment))
    memavailable_observed = all(
        (_int(run.get("min_mem_available_bytes")) or 0) > 0
        for run in (control, treatment))
    no_process_swap = all(
        _int(run.get("peak_process_swap_bytes")) == 0
        for run in (control, treatment))
    no_swap_io = all(_all_vm_zero(run, SWAP_KEYS)
                     for run in (control, treatment))
    no_reclaim = all(_all_vm_zero(run, RECLAIM_KEYS)
                     for run in (control, treatment))
    psi_safe = all(_psi_safe(run) for run in (control, treatment))

    changed_snapshot = _int(changed_prefix.get("family_snapshot_bytes"))
    changed_resident = _int(changed_prefix.get("resident_bytes"))
    changed_tool_miss = (
        treatment.get("stable_tool_sha256") !=
        treatment.get("changed_tool_sha256") and
        changed.get("tool_sha256") == treatment.get("changed_tool_sha256") and
        not bool(changed_prefix.get("hit")) and
        changed_prefix.get("hit_kind") in (None, "none") and
        changed_prefix.get("status") == "miss_stored_family" and
        _int(changed_prefix.get("family_entries")) == 2 and
        _int(changed_prefix.get("exact_entries")) == 0 and
        changed_snapshot is not None and changed_snapshot > 0 and
        reserve is not None and reserve > 0 and
        _int(changed_prefix.get("capacity_bytes")) == reserve and
        changed_resident is not None and
        changed_snapshot <= changed_resident <= reserve)

    direct_io_no_fallback = _transport(control) and _transport(treatment)
    stable_row_arithmetic = (
        control.get("engine_environment") == EXPECTED_ENGINE_ENV and
        treatment.get("engine_environment") == EXPECTED_ENGINE_ENV and
        all((run.get("trace_meta") or {}).get(
                "expert_schedule_requested") == "row" and
            (run.get("trace_meta") or {}).get("expert_schedule") == "row"
            for run in (control, treatment)))
    checks = {
        "output_artifacts_valid": output_artifacts_valid,
        "output_canonical_identical": output_canonical_identical,
        "matched_seed_a": seed_request_matched,
        "matched_prompt_b": prompt_b_matched,
        "stable_row_arithmetic": stable_row_arithmetic,
        "direct_io_qd4_no_fallback": direct_io_no_fallback,
        "family_root_replay": family_root_replay,
        "decode_routes_identical": routes_identical,
        "expert_bytes_reduced": bytes_reduced,
        "snapshot_within_reservation": snapshot_within_reservation,
        "fixed_total_budget": fixed_budget,
        "rss_within_limit": rss_safe,
        "memavailable_observed": memavailable_observed,
        "no_process_swap": no_process_swap,
        "no_swap_io": no_swap_io,
        "no_reclaim": no_reclaim,
        "memory_psi_within_limit": psi_safe,
        "changed_tool_miss": changed_tool_miss,
    }
    reduction = (1.0 - hit_bytes / cold_bytes
                 if bytes_reduced and cold_bytes else None)
    cold_seconds = _float(cold.get("wall_seconds"))
    hit_seconds = _float(hit.get("wall_seconds"))
    latency_speedup = (cold_seconds / hit_seconds
                       if cold_seconds is not None and hit_seconds else None)
    return {
        "schema": SCHEMA,
        "valid": True,
        "accepted": all(checks.values()),
        "errors": [],
        "mismatches": {},
        "control_run": control.get("run_id"),
        "prefix_run": treatment.get("run_id"),
        "checks": checks,
        "output": {
            "stable_output_sha256": (hit.get("stable_output_sha256")
                                      if output_canonical_identical else None),
        },
        "family_prefix": {
            "root_tokens": root_depth,
            "prompt_tokens": key_tokens,
            "suffix_tokens_replayed": replayed,
            "suffix_prefill_tokens": prefill_tokens,
            "snapshot_bytes": snapshot,
            "resident_bytes": resident,
            "capacity_bytes": capacity,
        },
        "latency": {
            "cold_b_seconds": cold_seconds,
            "family_b_seconds": hit_seconds,
            "speedup": latency_speedup,
            "reduction_fraction": (
                1.0 - hit_seconds / cold_seconds
                if cold_seconds and hit_seconds is not None else None),
        },
        "expert_io": {
            "cold_decode_route_count": cold_trace.get("decode_route_count"),
            "family_hit_decode_route_count":
                hit_trace.get("decode_route_count"),
            "matched_generated_positions": (
                len(cold_trace.get("decode_route_positions") or {})
                if routes_identical else 0),
            "cold_bytes_read": cold_bytes,
            "family_hit_bytes_read": hit_bytes,
            "bytes_reduction_fraction": reduction,
            "cold_cache_misses": cold_trace.get("expert_cache_misses"),
            "family_hit_cache_misses": hit_trace.get("expert_cache_misses"),
        },
        "memory": {
            "budget_bytes": budget,
            "rss_limit_bytes": treatment.get("rss_limit_bytes"),
            "control_peak_rss_bytes": control.get("peak_rss_bytes"),
            "prefix_peak_rss_bytes": treatment.get("peak_rss_bytes"),
            "control_min_mem_available_bytes":
                control.get("min_mem_available_bytes"),
            "prefix_min_mem_available_bytes":
                treatment.get("min_mem_available_bytes"),
            "control_peak_swap_bytes":
                control.get("peak_process_swap_bytes"),
            "prefix_peak_swap_bytes":
                treatment.get("peak_process_swap_bytes"),
            "control_vm_delta": control.get("vm_delta"),
            "prefix_vm_delta": treatment.get("vm_delta"),
            "control_memory_psi_total_us_delta":
                control.get("memory_psi_total_us_delta"),
            "prefix_memory_psi_total_us_delta":
                treatment.get("memory_psi_total_us_delta"),
        },
    }


def _bytes(value: Any) -> str:
    n = _int(value)
    if n is None:
        return "n/a"
    return f"{n / (1024 ** 3):.3f} GiB"


def markdown(result: dict[str, Any]) -> str:
    if not result.get("valid"):
        details = "\n".join(f"- {error}" for error in result.get("errors", []))
        return f"# K3 family-root prefix acceptance\n\nINVALID\n\n{details}\n"

    checks = result["checks"]
    prefix = result["family_prefix"]
    latency = result["latency"]
    expert = result["expert_io"]
    memory = result["memory"]
    gate_rows = "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in checks.items())
    reduction = expert.get("bytes_reduction_fraction")
    reduction_text = (f"{100 * reduction:.2f}%" if reduction is not None
                      else "n/a")
    return f"""# K3 family-root prefix acceptance

Accepted: **{'yes' if result['accepted'] else 'no'}**.

## Gates

| Gate | Result |
|---|---:|
{gate_rows}

## Matched cold B versus family-root B

| Metric | Cold control | Family-root hit |
|---|---:|---:|
| Stable output SHA-256 | `{result['output'].get('stable_output_sha256') or 'mismatch'}` | same gate |
| Wall time | {latency.get('cold_b_seconds')} s | {latency.get('family_b_seconds')} s |
| Wall-time speedup | — | {latency.get('speedup')}x |
| Routed decode rows | {expert.get('cold_decode_route_count')} | {expert.get('family_hit_decode_route_count')} |
| Expert bytes read | {expert.get('cold_bytes_read')} | {expert.get('family_hit_bytes_read')} |
| Expert-byte reduction | — | {reduction_text} |
| Expert cache misses | {expert.get('cold_cache_misses')} | {expert.get('family_hit_cache_misses')} |

The semantic root reused **{prefix.get('root_tokens')}** tokens from a
{prefix.get('prompt_tokens')}-token prompt and replayed
**{prefix.get('suffix_tokens_replayed')}** suffix tokens. Its checkpoint is
{_bytes(prefix.get('snapshot_bytes'))}; total prefix-cache residency is
{_bytes(prefix.get('resident_bytes'))} inside a
{_bytes(prefix.get('capacity_bytes'))} reservation.

## Memory evidence

| Metric | Control | Prefix-cache server |
|---|---:|---:|
| Fixed engine budget | {_bytes(memory.get('budget_bytes'))} | {_bytes(memory.get('budget_bytes'))} |
| Peak RSS | {_bytes(memory.get('control_peak_rss_bytes'))} | {_bytes(memory.get('prefix_peak_rss_bytes'))} |
| Minimum MemAvailable | {_bytes(memory.get('control_min_mem_available_bytes'))} | {_bytes(memory.get('prefix_min_mem_available_bytes'))} |
| Peak process swap | {_bytes(memory.get('control_peak_swap_bytes'))} | {_bytes(memory.get('prefix_peak_swap_bytes'))} |
| Memory PSI total deltas | `{memory.get('control_memory_psi_total_us_delta')}` | `{memory.get('prefix_memory_psi_total_us_delta')}` |
| VM deltas | `{memory.get('control_vm_delta')}` | `{memory.get('prefix_vm_delta')}` |
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("records", type=Path,
                    help="family_prefix_acceptance.py runs.jsonl")
    ap.add_argument("--output", type=Path, required=True,
                    help="machine-readable analysis JSON")
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        rows = load(args.records)
        result = compare(choose(rows, "control"), choose(rows, "prefix"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2,
                                          sort_keys=True) + "\n")
        rendered = markdown(result)
        args.markdown.write_text(rendered)
        print(rendered, end="")
        return 0 if result.get("accepted") else 1
    except (OSError, ValueError) as exc:
        print(f"analyze_family_prefix: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
