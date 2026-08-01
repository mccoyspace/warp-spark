#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Compare matched pread and QD4 phase-gated off-CPU captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read {path}: {e}") from e


def trace_facts(path: Path) -> dict:
    meta = []
    layers = []
    tokens = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: {e}") from e
        event = row.get("event")
        if event == "meta":
            meta.append(row)
        elif event == "decode_layer":
            layers.append(row)
        elif event == "decode_token":
            tokens.append(row)
    if len(meta) != 1 or not layers or len(tokens) != 1:
        raise ValueError(
            f"{path}: expected one meta, decode layers, and one decode token")
    signature = [
        [r.get("position"), r.get("layer"), r.get("attention_kind"),
         r.get("top_k"), r.get("experts"), r.get("cache_hits"),
         r.get("cache_misses"), r.get("bytes_read")]
        for r in layers
    ]
    encoded = json.dumps(signature, separators=(",", ":")).encode()
    return {
        "path": str(path),
        "meta": meta[0],
        "layer_rows": len(layers),
        "route_traffic_sha256": hashlib.sha256(encoded).hexdigest(),
        "direct_io": all(r.get("direct_io") is True for r in layers),
        "read_error": any(r.get("read_error") is True for r in layers),
        "backends": sorted({r.get("io_backend") for r in layers}),
        "queue_depths": sorted({r.get("queue_depth") for r in layers}),
        "token": tokens[0],
    }


def reduction(before, after):
    return 1 - float(after) / float(before) if before else None


def state_metric(summary: dict, state: int, key: str):
    state_row = summary.get("idle_exit", {}).get("states", {}).get(str(state), {})
    if key == "fraction":
        return state_row.get("fraction_of_barrier_wakes", 0)
    return state_row.get("residency", {}).get(key)


def compare(pread: dict, qd4: dict, pread_trace: dict,
            qd4_trace: dict) -> dict:
    gates = {
        "exact_intervals": bool(
            pread.get("coverage", {}).get("exact_intervals") and
            qd4.get("coverage", {}).get("exact_intervals")),
        "same_worker_count": pread.get("worker_threads") == qd4.get("worker_threads"),
        "same_layer_count": pread_trace["layer_rows"] == qd4_trace["layer_rows"],
        "same_routes_and_traffic": (
            pread_trace["route_traffic_sha256"] ==
            qd4_trace["route_traffic_sha256"]),
        "direct_io_no_errors": bool(
            pread_trace["direct_io"] and qd4_trace["direct_io"] and
            not pread_trace["read_error"] and not qd4_trace["read_error"]),
        "transport_identity": bool(
            pread_trace["backends"] == ["pread_sync"] and
            pread_trace["queue_depths"] == [1] and
            qd4_trace["backends"] == ["io_uring"] and
            qd4_trace["queue_depths"] == [4] and
            not pread_trace["meta"].get("io_fallback") and
            not qd4_trace["meta"].get("io_fallback")),
    }

    def pair(before, after):
        return {"pread": before, "qd4": after,
                "qd4_change": float(after) - float(before),
                "qd4_change_fraction":
                    float(after) / float(before) - 1 if before else None,
                "qd4_reduction_fraction": reduction(before, after)}

    pt, qt = pread_trace["token"], qd4_trace["token"]
    po, qo = pread["offcpu"], qd4["offcpu"]
    pf, qf = pread["futex"], qd4["futex"]
    pw = pread["wake_to_run"]["all_workers"]
    qw = qd4["wake_to_run"]["all_workers"]
    pi, qi = pread["idle_exit"], qd4["idle_exit"]
    metrics = {
        "decode_total_ms": pair(pt["total_ms"], qt["total_ms"]),
        "expert_io_ms": pair(pt["expert_io_ms"], qt["expert_io_ms"]),
        "routed_compute_ms": pair(pt["expert_compute_ms"],
                                  qt["expert_compute_ms"]),
        "blocked_worker_ms": pair(po["blocked_worker_ms"],
                                  qo["blocked_worker_ms"]),
        "futex_wait_worker_ms": pair(pf["wait_worker_ms"],
                                     qf["wait_worker_ms"]),
        "futex_wait_calls": pair(pf["wait_calls_overlapping_compute"],
                                 qf["wait_calls_overlapping_compute"]),
        "wake_to_run_p50_us": pair(pw["p50_us"], qw["p50_us"]),
        "wake_to_run_p95_us": pair(pw["p95_us"], qw["p95_us"]),
        "nonzero_idle_fraction": pair(
            pi["nonzero_state_fraction_of_barrier_wakes"],
            qi["nonzero_state_fraction_of_barrier_wakes"]),
        "state0_fraction": pair(state_metric(pread, 0, "fraction"),
                                state_metric(qd4, 0, "fraction")),
        "state3_fraction": pair(state_metric(pread, 3, "fraction"),
                                state_metric(qd4, 3, "fraction")),
        "state3_residency_p95_us": pair(state_metric(pread, 3, "p95_us"),
                                       state_metric(qd4, 3, "p95_us")),
    }
    return {
        "schema": "waste.offcpu_pair.v1",
        "valid": all(gates.values()),
        "gates": gates,
        "route_traffic_sha256": (pread_trace["route_traffic_sha256"]
                                 if gates["same_routes_and_traffic"] else None),
        "layer_rows": pread_trace["layer_rows"],
        "worker_threads": pread.get("worker_threads"),
        "metrics": metrics,
        "note": ("One phase-gated diagnostic capture per backend. Use the "
                 "untraced campaign for the primary performance estimate; "
                 "this pair diagnoses the off-CPU mechanism."),
    }


def signed_pct(value) -> str:
    return "—" if value is None else f"{100 * value:+.2f}%"


def signed_pp(value) -> str:
    return "—" if value is None else f"{100 * value:+.2f} pp"


def markdown(result: dict) -> str:
    status = "pass" if result["valid"] else "fail"
    lines = [
        "# Matched off-CPU diagnostic",
        "",
        f"Comparability gates: **{status}**. This is one instrumented capture "
        "per backend; the untraced campaign remains the primary timing result.",
        "",
        "| Metric | pread QD1 | io_uring QD4 | QD4 change |",
        "|---|---:|---:|---:|",
    ]
    labels = (
        ("Decode total, ms", "decode_total_ms", False),
        ("Expert I/O, ms", "expert_io_ms", False),
        ("Routed compute, ms", "routed_compute_ms", False),
        ("Blocked worker time, ms", "blocked_worker_ms", False),
        ("Futex wait worker-time, ms", "futex_wait_worker_ms", False),
        ("Futex waits", "futex_wait_calls", False),
        ("Wake-to-run p50, us", "wake_to_run_p50_us", False),
        ("Wake-to-run p95, us", "wake_to_run_p95_us", False),
        ("Nonzero idle-state fraction", "nonzero_idle_fraction", True),
        ("State 0 fraction", "state0_fraction", True),
        ("State 3 fraction", "state3_fraction", True),
        ("State 3 residency p95, us", "state3_residency_p95_us", False),
    )
    for label, key, is_fraction in labels:
        row = result["metrics"][key]
        before = f"{100 * row['pread']:.2f}%" if is_fraction else row["pread"]
        after = f"{100 * row['qd4']:.2f}%" if is_fraction else row["qd4"]
        change = (signed_pp(row["qd4_change"]) if is_fraction else
                  signed_pct(row["qd4_change_fraction"]))
        lines.append(f"| {label} | {before} | {after} | {change} |")
    lines += [
        "",
        f"All {result['layer_rows']} layer routes, hit/miss counts, and bytes "
        f"match (`{result['route_traffic_sha256']}`).",
        "",
        result["note"],
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pread-summary", type=Path, required=True)
    ap.add_argument("--qd4-summary", type=Path, required=True)
    ap.add_argument("--pread-trace", type=Path, required=True)
    ap.add_argument("--qd4-trace", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args(argv)
    try:
        result = compare(load_json(args.pread_summary), load_json(args.qd4_summary),
                         trace_facts(args.pread_trace), trace_facts(args.qd4_trace))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        rendered = markdown(result)
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(rendered)
        print(rendered, end="")
        return 0 if result["valid"] else 1
    except (OSError, KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        print(f"analyze_offcpu_pair: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
