#!/usr/bin/env python3
"""Summarize an io_compute_diagnostic.py A/B/C campaign.

The diagnostic has one observation per treatment in each balanced block.  This
analyzer therefore reports both ordinary per-treatment summaries and paired
within-block estimates.  Percentage confidence intervals are Student-t
intervals over paired log ratios; millisecond intervals are Student-t
intervals over paired arithmetic differences.

Only Python's standard library is required.  Raw run records and layer traces
remain the source of truth; the generated CSV/JSON files are disposable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


TREATMENT_ORDER = ("A", "B", "C")
TREATMENT_NAMES = {
    "A": "pread_interleaved",
    "B": "pread_batch",
    "C": "io_uring_qd4",
}
COMPARISONS = (("B", "A"), ("C", "B"), ("C", "A"))
METRICS = (
    "total_ms",
    "attention_ms",
    "router_ms",
    "expert_io_ms",
    "expert_compute_ms",
    "shared_compute_ms",
    "moe_total_ms",
    "other_ms",
)
PERF_EVENTS = {
    "cycles": "armv8_pmuv3_1/cpu_cycles/u",
    "instructions": "armv8_pmuv3_1/inst_retired/u",
    "backend_stalls": "armv8_pmuv3_1/stall_backend/u",
    "memory_stalls": "armv8_pmuv3_1/stall_backend_mem/u",
    "l1d_refills": "armv8_pmuv3_1/l1d_cache_refill/u",
    "l2d_refills": "armv8_pmuv3_1/l2d_cache_refill/u",
    "task_clock_ns": "task-clock",
    "context_switches": "context-switches",
    "sched_switches": "sched:sched_switch",
    "sched_wakeup": "sched:sched_wakeup",
    "sched_waking": "sched:sched_waking",
    "futex_calls": "syscalls:sys_enter_futex",
    "pread64_calls": "syscalls:sys_enter_pread64",
    "io_uring_enter_calls": "syscalls:sys_enter_io_uring_enter",
}
PERF_DERIVED = (
    "wall_seconds",
    "cycles",
    "instructions",
    "task_clock_ns",
    "ipc",
    "ghz_proxy",
    "active_cpus",
    "backend_stall_percent",
    "memory_stall_percent",
    "l1d_refills_per_kinst",
    "l2d_refills_per_kinst",
)

# Two-sided 95% Student-t critical values.  Campaigns normally have six
# blocks, but retaining a small table makes resumed/extended campaigns useful.
T95 = {
    1: 12.706205,
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    6: 2.446912,
    7: 2.364624,
    8: 2.306004,
    9: 2.262157,
    10: 2.228139,
    11: 2.200985,
    12: 2.178813,
    13: 2.160369,
    14: 2.144787,
    15: 2.131450,
    16: 2.119905,
    17: 2.109816,
    18: 2.100922,
    19: 2.093024,
    20: 2.085963,
    21: 2.079614,
    22: 2.073873,
    23: 2.068658,
    24: 2.063899,
    25: 2.059539,
    26: 2.055529,
    27: 2.051831,
    28: 2.048407,
    29: 2.045230,
    30: 2.042272,
}


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    """Return mean and two-sided 95% Student-t confidence interval."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    critical = T95.get(len(values) - 1, 1.959964)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def distribution(values: list[float]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "sd": sd,
        "cv_percent": 100.0 * sd / mean if mean else 0.0,
        "min": min(values),
        "max": max(values),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def read_perf(path: Path) -> tuple[dict[str, list[tuple[float, float]]],
                                   dict[str, float]]:
    """Parse perf stat -I -x output into intervals and exact summaries."""
    intervals: dict[str, list[tuple[float, float]]] = {}
    summaries: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        stamp, value, event = fields[0], fields[1], fields[3]
        try:
            parsed = float(value)
        except ValueError:
            continue
        if stamp == "summary":
            summaries[event] = parsed
            continue
        try:
            end = float(stamp)
        except ValueError:
            continue
        intervals.setdefault(event, []).append((end, parsed))
    return intervals, summaries


def integrate_perf(intervals: dict[str, list[tuple[float, float]]],
                   start: float, end: float) -> dict[str, float]:
    """Boundary-weight perf intervals into a monotonic phase window.

    The runner records monotonic boundaries immediately around the perf
    subprocess.  Counts in either boundary's at-most-500 ms interval are
    prorated by temporal overlap, which assumes events are uniform inside
    that interval.  Interior intervals are retained exactly.
    """
    totals: dict[str, float] = {}
    for event, samples in intervals.items():
        previous = 0.0
        total = 0.0
        for interval_end, value in samples:
            duration = interval_end - previous
            overlap = max(0.0, min(interval_end, end) - max(previous, start))
            if duration > 0.0 and overlap > 0.0:
                total += value * overlap / duration
            previous = interval_end
        totals[event] = total
    return totals


def derive_perf(events: dict[str, float], wall_seconds: float) -> dict[str, float]:
    row = {name: events.get(event, 0.0) for name, event in PERF_EVENTS.items()}
    cycles = row["cycles"]
    instructions = row["instructions"]
    task_clock_ns = row["task_clock_ns"]
    row.update({
        "wall_seconds": wall_seconds,
        "ipc": instructions / cycles if cycles else 0.0,
        # cycles / task-clock nanoseconds is numerically GHz.
        "ghz_proxy": cycles / task_clock_ns if task_clock_ns else 0.0,
        "active_cpus": task_clock_ns / 1e9 / wall_seconds if wall_seconds else 0.0,
        "backend_stall_percent": 100.0 * row["backend_stalls"] / cycles
        if cycles else 0.0,
        "memory_stall_percent": 100.0 * row["memory_stalls"] / cycles
        if cycles else 0.0,
        "l1d_refills_per_kinst": 1000.0 * row["l1d_refills"] / instructions
        if instructions else 0.0,
        "l2d_refills_per_kinst": 1000.0 * row["l2d_refills"] / instructions
        if instructions else 0.0,
    })
    return row


def layer_signature(rows: Iterable[dict[str, Any]]) -> str:
    payload = []
    for row in rows:
        if row.get("event") not in ("prefill_layer", "decode_layer"):
            continue
        payload.append({
            "event": row.get("event"),
            "position": row.get("position"),
            "position_start": row.get("position_start"),
            "position_end": row.get("position_end"),
            "layer": row.get("layer"),
            "experts": row.get("experts"),
            "cache_hits": row.get("cache_hits"),
            "cache_misses": row.get("cache_misses"),
            "bytes_read": row.get("bytes_read"),
        })
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def miss_bin(misses: int) -> str:
    if misses == 0:
        return "0"
    if misses <= 4:
        return "1-4"
    if misses <= 8:
        return "5-8"
    if misses <= 12:
        return "9-12"
    return "13-16"


def ratio_estimate(numerator: list[float], denominator: list[float]) -> dict[str, Any]:
    logs = [math.log(a / b) for a, b in zip(numerator, denominator, strict=True)]
    log_mean, log_lo, log_hi = mean_ci(logs)
    ratios = [a / b for a, b in zip(numerator, denominator, strict=True)]
    positive = sum(ratio > 1.0 for ratio in ratios)
    negative = sum(ratio < 1.0 for ratio in ratios)
    sign_n = positive + negative
    tail = min(positive, negative)
    sign_p = min(1.0, 2.0 * sum(math.comb(sign_n, k) for k in range(tail + 1))
                 / (2 ** sign_n)) if sign_n else 1.0
    return {
        "n_pairs": len(ratios),
        "block_ratios": ratios,
        "geometric_ratio": math.exp(log_mean),
        "percent_change": 100.0 * (math.exp(log_mean) - 1.0),
        "percent_change_ci95": [
            100.0 * (math.exp(log_lo) - 1.0),
            100.0 * (math.exp(log_hi) - 1.0),
        ],
        "same_direction_pairs": sum((ratio < 1.0) == (statistics.fmean(logs) < 0)
                                    for ratio in ratios),
        "two_sided_sign_p": sign_p,
    }


def difference_estimate(numerator: list[float], denominator: list[float]) -> dict[str, Any]:
    diffs = [a - b for a, b in zip(numerator, denominator, strict=True)]
    mean, lo, hi = mean_ci(diffs)
    return {"mean": mean, "ci95": [lo, hi], "block_differences": diffs}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path,
                        help="directory containing runs.jsonl and per-run traces")
    parser.add_argument("--out", type=Path,
                        help="output directory (default: CAMPAIGN/analysis)")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    out = (args.out or campaign / "analysis").resolve()
    out.mkdir(parents=True, exist_ok=True)

    runs = read_jsonl(campaign / "runs.jsonl")
    valid = [row for row in runs if row.get("valid")]
    if not valid:
        raise SystemExit("campaign contains no valid runs")

    by_treatment: dict[str, list[dict[str, Any]]] = {key: [] for key in TREATMENT_ORDER}
    by_block: dict[int, dict[str, dict[str, Any]]] = {}
    trace_rows: dict[str, list[dict[str, Any]]] = {}
    perf_rows: dict[str, dict[str, float]] = {}
    perf_summaries: dict[str, dict[str, float]] = {}
    for run in valid:
        treatment = run["treatment"]
        by_treatment.setdefault(treatment, []).append(run)
        by_block.setdefault(int(run["block"]), {})[treatment] = run
        trace = read_jsonl(campaign / run["run_id"] / "trace.jsonl")
        trace_rows[run["run_id"]] = trace
        run["layer_signature_sha256"] = layer_signature(trace)
        intervals, perf_summary = read_perf(campaign / run["run_id"] / "perf.tsv")
        phase_start = ((int(run["trace"]["prefill_end_monotonic_ns"])
                        - int(run["started_monotonic_ns"])) / 1e9)
        phase_end = ((int(run["trace"]["decode_end_monotonic_ns"])
                      - int(run["started_monotonic_ns"])) / 1e9)
        perf_rows[run["run_id"]] = derive_perf(
            integrate_perf(intervals, phase_start, phase_end),
            phase_end - phase_start)
        perf_summaries[run["run_id"]] = perf_summary
    for treatment in by_treatment:
        by_treatment[treatment].sort(key=lambda row: int(row["block"]))

    incomplete = {block: sorted(set(TREATMENT_ORDER) - set(group))
                  for block, group in by_block.items()
                  if set(group) != set(TREATMENT_ORDER)}
    if incomplete:
        raise SystemExit(f"incomplete blocks: {incomplete}")

    per_mode: dict[str, Any] = {}
    for treatment in TREATMENT_ORDER:
        group = by_treatment[treatment]
        per_mode[treatment] = {
            "name": TREATMENT_NAMES[treatment],
            "wall_seconds": distribution([float(row["wall_seconds"]) for row in group]),
            "decode": {
                metric: distribution([float(row["trace"]["decode"][metric]) for row in group])
                for metric in METRICS
            },
        }

    paired: dict[str, Any] = {}
    blocks = sorted(by_block)
    for numerator, denominator in COMPARISONS:
        key = f"{numerator}_vs_{denominator}"
        paired[key] = {}
        for metric in METRICS:
            num = [float(by_block[block][numerator]["trace"]["decode"][metric])
                   for block in blocks]
            den = [float(by_block[block][denominator]["trace"]["decode"][metric])
                   for block in blocks]
            paired[key][metric] = {
                **ratio_estimate(num, den),
                "difference_ms": difference_estimate(num, den),
            }

    positions: dict[str, Any] = {}
    for position in (1, 2, 3):
        rows = [row for row in valid if int(row["position"]) == position]
        positions[str(position)] = {
            metric: distribution([float(row["trace"]["decode"][metric]) for row in rows])
            for metric in ("total_ms", "expert_io_ms", "expert_compute_ms")
        }

    perf_by_mode: dict[str, Any] = {}
    for treatment in TREATMENT_ORDER:
        group = by_treatment[treatment]
        derived = [perf_rows[run["run_id"]] for run in group]
        sums = {name: sum(float(row[name]) for row in derived)
                for name in PERF_EVENTS}
        pooled_events = {event: sums[name] for name, event in PERF_EVENTS.items()}
        pooled = derive_perf(
            pooled_events, sum(float(row["wall_seconds"]) for row in derived))
        pooled.update({
            "context_switches_per_wall_second":
                sums["context_switches"] / pooled["wall_seconds"],
            "futex_calls_per_wall_second": sums["futex_calls"] / pooled["wall_seconds"],
            "whole_process_instructions": sum(
                perf_summaries[run["run_id"]].get(PERF_EVENTS["instructions"], 0.0)
                for run in group),
        })
        perf_by_mode[treatment] = {
            "pooled": pooled,
            "per_run": {name: distribution([float(row[name]) for row in derived])
                        for name in PERF_DERIVED},
        }

    perf_paired: dict[str, Any] = {}
    for numerator, denominator in COMPARISONS:
        comparison: dict[str, Any] = {}
        for metric in PERF_DERIVED:
            num = [float(perf_rows[by_block[block][numerator]["run_id"]][metric])
                   for block in blocks]
            den = [float(perf_rows[by_block[block][denominator]["run_id"]][metric])
                   for block in blocks]
            comparison[metric] = ratio_estimate(num, den)
        perf_paired[f"{numerator}_vs_{denominator}"] = comparison

    # Layer-stratified routed compute.  Cache/miss signatures must match before
    # a mode comparison is meaningful, so validate each block/key explicitly.
    strata_by_block: dict[int, dict[str, dict[str, float]]] = {}
    count_per_block: dict[int, dict[str, int]] = {}
    for block in blocks:
        keyed: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
        for treatment in TREATMENT_ORDER:
            run = by_block[block][treatment]
            keyed[treatment] = {
                (int(row["position"]), int(row["layer"])): row
                for row in trace_rows[run["run_id"]]
                if row.get("event") == "decode_layer"
            }
        if not all(set(keyed[t]) == set(keyed["A"]) for t in TREATMENT_ORDER):
            raise SystemExit(f"decode layer keys differ in block {block}")
        strata_by_block[block] = {t: {} for t in TREATMENT_ORDER}
        count_per_block[block] = {}
        for key, arow in keyed["A"].items():
            signatures = {
                (int(keyed[t][key]["cache_hits"]),
                 int(keyed[t][key]["cache_misses"]),
                 int(keyed[t][key]["bytes_read"]))
                for t in TREATMENT_ORDER
            }
            if len(signatures) != 1:
                raise SystemExit(f"cache signature differs in block {block}, layer {key}")
            bucket = miss_bin(int(arow["cache_misses"]))
            count_per_block[block][bucket] = count_per_block[block].get(bucket, 0) + 1
            for treatment in TREATMENT_ORDER:
                value = float(keyed[treatment][key]["expert_compute_ms"])
                target = strata_by_block[block][treatment]
                target[bucket] = target.get(bucket, 0.0) + value

    strata_rows: list[dict[str, Any]] = []
    for bucket in ("0", "1-4", "5-8", "9-12", "13-16"):
        row: dict[str, Any] = {
            "misses": bucket,
            "layers_per_run": statistics.fmean(
                [count_per_block[block].get(bucket, 0) for block in blocks]),
        }
        values: dict[str, list[float]] = {}
        for treatment in TREATMENT_ORDER:
            values[treatment] = [strata_by_block[block][treatment].get(bucket, 0.0)
                                 for block in blocks]
            row[f"{treatment}_mean_compute_ms"] = statistics.fmean(values[treatment])
        for numerator, denominator in COMPARISONS:
            if all(values[denominator]) and all(values[numerator]):
                estimate = ratio_estimate(values[numerator], values[denominator])
                row[f"{numerator}_vs_{denominator}_percent"] = estimate["percent_change"]
                row[f"{numerator}_vs_{denominator}_ci95_low"] = estimate["percent_change_ci95"][0]
                row[f"{numerator}_vs_{denominator}_ci95_high"] = estimate["percent_change_ci95"][1]
            else:
                row[f"{numerator}_vs_{denominator}_percent"] = None
                row[f"{numerator}_vs_{denominator}_ci95_low"] = None
                row[f"{numerator}_vs_{denominator}_ci95_high"] = None
        strata_rows.append(row)

    prefill_totals: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    perf_csv_rows: list[dict[str, Any]] = []
    for run in sorted(valid, key=lambda row: (int(row["block"]), int(row["position"]))):
        trace = trace_rows[run["run_id"]]
        prefill = next(row for row in trace if row.get("event") == "prefill_chunk")
        prefill_totals.append({key: prefill.get(key) for key in
                               ("cache_hits", "cache_misses", "bytes_read")})
        timing_rows.append({
            "run_id": run["run_id"],
            "block": run["block"],
            "position": run["position"],
            "treatment": run["treatment"],
            "treatment_name": run["treatment_name"],
            "valid": run["valid"],
            "wall_seconds": run["wall_seconds"],
            **{metric: run["trace"]["decode"][metric] for metric in METRICS},
            "cache_hits": run["trace"]["decode"]["cache_hits"],
            "cache_misses": run["trace"]["decode"]["cache_misses"],
            "bytes_read": run["trace"]["decode"]["bytes_read"],
            "peak_rss_bytes": run["peak_rss_bytes"],
            "peak_swap_bytes": run["peak_swap_bytes"],
            "stdout_sha256": run["stdout_sha256"],
            "route_sha256": run["trace"]["route_sha256"],
            "usage_sha256_after": run["usage_sha256_after"],
            "layer_signature_sha256": run["layer_signature_sha256"],
        })
        perf_csv_rows.append({
            "run_id": run["run_id"],
            "block": run["block"],
            "position": run["position"],
            "treatment": run["treatment"],
            **perf_rows[run["run_id"]],
            "whole_process_instructions": perf_summaries[run["run_id"]].get(
                PERF_EVENTS["instructions"]),
        })

    determinism = {
        "measured_runs": len(runs),
        "valid_runs": len(valid),
        "unique_stdout_sha256": sorted({row["stdout_sha256"] for row in valid}),
        "unique_route_sha256": sorted({row["trace"]["route_sha256"] for row in valid}),
        "unique_usage_sha256": sorted({row["usage_sha256_after"] for row in valid}),
        "unique_layer_signature_sha256": sorted(
            {row["layer_signature_sha256"] for row in valid}),
        "decode_cache_signatures": sorted({
            (int(row["trace"]["decode"]["cache_hits"]),
             int(row["trace"]["decode"]["cache_misses"]),
             int(row["trace"]["decode"]["bytes_read"]))
            for row in valid
        }),
        "prefill_cache_signatures": sorted({
            (int(row["cache_hits"]), int(row["cache_misses"]), int(row["bytes_read"]))
            for row in prefill_totals
        }),
        "all_direct_io": all(row["trace"]["all_direct_io"] for row in valid),
        "any_read_error": any(row["trace"]["any_read_error"] for row in valid),
        "any_io_fallback": any(row["trace"]["io_fallback"] for row in valid),
        "any_process_swap": any(int(row["peak_swap_bytes"]) != 0 for row in valid),
    }

    summary = {
        "schema": "waste.io_compute_analysis.v1",
        "campaign": str(campaign),
        "blocks": blocks,
        "treatments": TREATMENT_NAMES,
        "determinism": determinism,
        "per_mode": per_mode,
        "paired": paired,
        "position": positions,
        "miss_strata": strata_rows,
        "perf_decode": {
            "method": "500 ms perf intervals prorated at monotonic decode boundaries",
            "boundary_interval_seconds": 0.5,
            "per_mode": perf_by_mode,
            "paired": perf_paired,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(out / "runs.csv", timing_rows, list(timing_rows[0]))
    write_csv(out / "miss-strata.csv", strata_rows, list(strata_rows[0]))
    write_csv(out / "perf-decode.csv", perf_csv_rows, list(perf_csv_rows[0]))

    print(f"valid runs: {len(valid)}/{len(runs)}; complete blocks: {len(blocks)}")
    print("\nDecode timing medians (ms)")
    print("mode\ttotal\texpert_io\trouted_compute\tCV routed")
    for treatment in TREATMENT_ORDER:
        decode = per_mode[treatment]["decode"]
        print(f"{treatment}\t{decode['total_ms']['median']:.3f}"
              f"\t{decode['expert_io_ms']['median']:.3f}"
              f"\t{decode['expert_compute_ms']['median']:.3f}"
              f"\t{decode['expert_compute_ms']['cv_percent']:.2f}%")
    print("\nPaired geometric changes (95% CI)")
    print("comparison\ttotal\texpert_io\trouted_compute")
    for numerator, denominator in COMPARISONS:
        comparison = paired[f"{numerator}_vs_{denominator}"]
        cells = []
        for metric in ("total_ms", "expert_io_ms", "expert_compute_ms"):
            estimate = comparison[metric]
            lo, hi = estimate["percent_change_ci95"]
            cells.append(f"{estimate['percent_change']:+.2f}% [{lo:+.2f}, {hi:+.2f}]")
        print(f"{numerator}/{denominator}\t" + "\t".join(cells))
    print(f"\nWrote {out / 'summary.json'}, {out / 'runs.csv'}, "
          f"{out / 'miss-strata.csv'}, and {out / 'perf-decode.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
