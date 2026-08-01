#!/usr/bin/env python3
"""Analyze the Spark whole-expert 2x2 campaign with paired block effects.

Only the Python standard library is used.  The analyzer recomputes exact
experimental gates from each durable run record, then estimates schedule,
PM-QoS, and interaction effects within each Williams-balanced block.  Ratio
effects use paired log contrasts; millisecond effects use paired arithmetic
contrasts.  Confidence intervals are Student-t intervals across blocks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Sequence

try:
    from spark_whole_expert_campaign import (
        BASE_ORDERS, TREATMENTS, TRACE_METRICS, exact_run_reasons, orders,
    )
except ModuleNotFoundError:  # pragma: no cover - invocation dependent
    from tools.spark_whole_expert_campaign import (
        BASE_ORDERS, TREATMENTS, TRACE_METRICS, exact_run_reasons, orders,
    )


TREATMENT_ORDER = ("RN", "WN", "RQ", "WQ")
EFFECTS: dict[str, dict[str, float]] = {
    "whole_at_no_qos": {"WN": 1.0, "RN": -1.0},
    "whole_at_qos0": {"WQ": 1.0, "RQ": -1.0},
    "qos0_at_row": {"RQ": 1.0, "RN": -1.0},
    "qos0_at_whole": {"WQ": 1.0, "WN": -1.0},
    "schedule_main": {"WN": 0.5, "WQ": 0.5,
                      "RN": -0.5, "RQ": -0.5},
    "qos_main": {"RQ": 0.5, "WQ": 0.5,
                 "RN": -0.5, "WN": -0.5},
    "schedule_x_qos": {"WQ": 1.0, "RQ": -1.0,
                       "WN": -1.0, "RN": 1.0},
}
METRICS = tuple(TRACE_METRICS) + ("expert_compute_work_ms", "wall_seconds")

# Two-sided 95% Student-t critical values.  Four blocks => df=3, but the
# table supports replicated four-block squares as well.
T95 = {
    1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445,
    5: 2.570582, 6: 2.446912, 7: 2.364624, 8: 2.306004,
    9: 2.262157, 10: 2.228139, 11: 2.200985, 12: 2.178813,
    13: 2.160369, 14: 2.144787, 15: 2.131450, 16: 2.119905,
    17: 2.109816, 18: 2.100922, 19: 2.093024, 20: 2.085963,
    21: 2.079614, 22: 2.073873, 23: 2.068658, 24: 2.063899,
    25: 2.059539, 26: 2.055529, 27: 2.051831, 28: 2.048407,
    29: 2.045230, 30: 2.042272,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def mean_ci(values: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"n": len(values), "mean": mean, "ci95": [mean, mean],
                "block_values": values}
    critical = T95.get(len(values) - 1, 1.959964)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return {"n": len(values), "mean": mean,
            "ci95": [mean - half, mean + half], "block_values": values}


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean,
            "median": statistics.median(values), "sd": sd,
            "cv_percent": 100.0 * sd / mean if mean else 0.0,
            "min": min(values), "max": max(values)}


def metric_value(run: dict[str, Any], metric: str) -> float:
    if metric == "wall_seconds":
        return float(run[metric])
    return float(run["trace"]["decode"][metric])


def available_metric_values(runs: Sequence[dict[str, Any]],
                            metric: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        try:
            values.append(metric_value(run, metric))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def contrast(values: dict[str, float], coefficients: dict[str, float]) -> float:
    return sum(coefficients.get(treatment, 0.0) * values[treatment]
               for treatment in TREATMENT_ORDER)


def effect_estimate(block_rows: dict[int, dict[str, dict[str, Any]]],
                    metric: str, coefficients: dict[str, float]) -> dict[str, Any]:
    differences: list[float] = []
    log_effects: list[float] = []
    for block in sorted(block_rows):
        raw = {treatment: metric_value(block_rows[block][treatment], metric)
               for treatment in TREATMENT_ORDER}
        differences.append(contrast(raw, coefficients))
        if all(value > 0 for value in raw.values()):
            log_effects.append(contrast({key: math.log(value)
                                         for key, value in raw.items()},
                                        coefficients))
    result: dict[str, Any] = {"difference": mean_ci(differences)}
    if len(log_effects) == len(differences):
        estimate = mean_ci(log_effects)
        result["ratio"] = {
            **estimate,
            "geometric_ratio": math.exp(estimate["mean"]),
            "percent_change": 100.0 * (math.exp(estimate["mean"]) - 1.0),
            "percent_change_ci95": [
                100.0 * (math.exp(estimate["ci95"][0]) - 1.0),
                100.0 * (math.exp(estimate["ci95"][1]) - 1.0),
            ],
        }
    return result


def gate(passed: bool, detail: Any) -> dict[str, Any]:
    return {"pass": bool(passed), "detail": detail}


def expected_runs(campaign: dict[str, Any]) -> dict[str, tuple[int, int, str]]:
    result: dict[str, tuple[int, int, str]] = {}
    for block, order in enumerate(orders(int(campaign["blocks"])), 1):
        for position, treatment in enumerate(order, 1):
            name = TREATMENTS[treatment]["name"]
            run_id = f"b{block:02d}-p{position}-{treatment}-{name}"
            result[run_id] = (block, position, treatment)
    return result


def analyze(campaign_dir: Path) -> dict[str, Any]:
    campaign_dir = campaign_dir.resolve()
    campaign = json.loads((campaign_dir / "campaign.json").read_text())
    attempts = read_jsonl(campaign_dir / "runs.jsonl")
    latest: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for row in attempts:
        latest[row["run_id"]] = row
        counts[row["run_id"]] = counts.get(row["run_id"], 0) + 1

    expected = expected_runs(campaign)
    missing = sorted(set(expected) - set(latest))
    extra = sorted(set(latest) - set(expected))
    placement_errors: dict[str, Any] = {}
    for run_id in set(expected) & set(latest):
        block, position, treatment = expected[run_id]
        row = latest[run_id]
        got = (int(row.get("block", -1)), int(row.get("position", -1)),
               row.get("treatment"))
        if got != (block, position, treatment):
            placement_errors[run_id] = {"expected": [block, position, treatment],
                                        "actual": list(got)}

    identity = campaign["identity"]
    geometry = identity["model_geometry"]
    limits = campaign["limits"]
    recomputed: dict[str, list[str]] = {}
    for run_id, row in latest.items():
        if run_id not in expected:
            continue
        recomputed[run_id] = exact_run_reasons(
            record=row, expected=identity, geometry=geometry,
            gpu_limit=float(limits["gpu_util_percent"]),
            cpu_temp_limit=int(limits["cpu_temp_millic"]),
            nvme_temp_limit=int(limits["nvme_temp_millic"]))

    design_ok = (not missing and not extra and not placement_errors and
                 int(campaign.get("decoded_tokens", 0)) == 1 and
                 campaign.get("io_backend") == "io_uring" and
                 int(campaign.get("io_queue_depth", 0)) == 4 and
                 campaign.get("direct_io") is True and
                 campaign.get("orders") == [list(row) for row in
                                              orders(int(campaign["blocks"]))])
    run_validity = {run_id: {"record_valid": row.get("valid"),
                            "record_reasons": row.get("invalid_reasons", []),
                            "recomputed_reasons": recomputed.get(run_id, [])}
                    for run_id, row in latest.items() if run_id in expected}

    rows = [latest[run_id] for run_id in sorted(expected) if run_id in latest]
    hashes = lambda field: sorted({row[field] for row in rows
                                   if row.get(field) is not None})
    trace_hashes = lambda field: sorted({row["trace"].get(field) for row in rows
                                         if row.get("trace", {}).get(field) is not None})
    identity_ok = all(row.get("identity_after") == identity for row in rows)
    io_ok = all(row.get("trace", {}).get("backend") == "io_uring" and
                row["trace"].get("queue_depth") == 4 and
                row["trace"].get("io_fallback") is False and
                row["trace"].get("all_direct_io") is True and
                row["trace"].get("any_read_error") is False for row in rows)
    schedule_ok = all(
        row.get("trace", {}).get("expert_schedule_requested") ==
        TREATMENTS[row["treatment"]]["schedule"] and
        row["trace"].get("expert_schedule_selected") ==
        TREATMENTS[row["treatment"]]["schedule"] and
        row["trace"].get("decode_layer_schedules") ==
        [TREATMENTS[row["treatment"]]["schedule"]] for row in rows)
    qos_ok = all(
        (row.get("pm_qos") == {"enabled": False}
         if TREATMENTS[row["treatment"]]["qos_us"] is None else
         (row.get("pm_qos", {}).get("holder_euid") == 0 and
          row["pm_qos"].get("latency_us") == 0 and
          row["pm_qos"].get("fd_opened") is True and
          row["pm_qos"].get("fd_closed") is True and
          row["pm_qos"].get("sysfs_modified") is False)) for row in rows)
    no_pressure = all(
        int(row.get("vm_delta", {}).get("pswpin", 0)) == 0 and
        int(row.get("vm_delta", {}).get("pswpout", 0)) == 0 and
        int(row.get("telemetry", {}).get("peak_swap_bytes", 0)) == 0 and
        int(row.get("telemetry", {}).get(
            "memory_psi_full_total_delta_us", 0)) == 0 for row in rows)
    thermal_cppc = all(
        row.get("telemetry", {}).get("max_processor_cooling_state") == 0 and
        row.get("telemetry", {}).get("any_cppc_perf_limited") is False and
        (row["telemetry"].get("cpu_peak_millic") is None or
         row["telemetry"]["cpu_peak_millic"] <= limits["cpu_temp_millic"]) and
        (row["telemetry"].get("nvme_peak_millic") is None or
         row["telemetry"]["nvme_peak_millic"] <= limits["nvme_temp_millic"])
        for row in rows)
    gpu_ok = all(
        row.get("telemetry", {}).get("gpu", {}).get(
            "available_all_samples") is True and
        not row["telemetry"]["gpu"].get("compute_apps") and
        row["telemetry"]["gpu"].get("max_utilization_percent") is not None and
        float(row["telemetry"]["gpu"]["max_utilization_percent"]) <=
        float(limits["gpu_util_percent"]) for row in rows)

    route_hashes = trace_hashes("route_sha256")
    traffic_hashes = trace_hashes("traffic_sha256")
    layer_signature_hashes = trace_hashes("layer_signature_sha256")
    route_capture_count = sum(
        row.get("trace", {}).get("route_sha256") is not None for row in rows)
    traffic_capture_count = sum(
        row.get("trace", {}).get("traffic_sha256") is not None for row in rows)
    signature_capture_count = sum(
        row.get("trace", {}).get("layer_signature_sha256") is not None
        for row in rows)
    traffic_global_match = (
        len(rows) == len(expected) and
        traffic_capture_count == len(expected) and
        signature_capture_count == len(expected) and
        len(traffic_hashes) == 1 and
        len(layer_signature_hashes) == 1)
    traffic_by_treatment: dict[str, Any] = {}
    for treatment in TREATMENT_ORDER:
        treatment_rows = [row for row in rows
                          if row.get("treatment") == treatment]
        traffic_by_treatment[treatment] = {
            "traffic_sha256": sorted({
                row["trace"]["traffic_sha256"] for row in treatment_rows
                if row.get("trace", {}).get("traffic_sha256") is not None
            }),
            "layer_signature_sha256": sorted({
                row["trace"]["layer_signature_sha256"]
                for row in treatment_rows
                if row.get("trace", {}).get(
                    "layer_signature_sha256") is not None
            }),
            "decode_traffic": {
                metric: distribution(available_metric_values(
                    treatment_rows, metric))
                for metric in ("cache_hits", "cache_misses", "bytes_read")
            },
        }
    traffic_comparability = {
        "global_match": traffic_global_match,
        "traffic_sha256": traffic_hashes,
        "layer_signature_sha256": layer_signature_hashes,
        "captured_runs": {
            "expected": len(expected),
            "traffic": traffic_capture_count,
            "layer_signature": signature_capture_count,
        },
        "by_treatment": traffic_by_treatment,
        "interpretation": (
            "Traffic and per-layer signatures match across treatments; "
            "the total-latency and routed-compute effects therefore have "
            "the stronger identical-traffic interpretation."
            if traffic_global_match else
            "Traffic or per-layer signatures differ across treatments. "
            "Total latency intentionally retains the practical fixed-budget "
            "scratch/cache tradeoff; routed compute remains paired on the "
            "hard-gated identical routes."
        ),
    }

    gates = {
        "complete_balanced_design": gate(design_ok, {
            "missing": missing, "extra": extra,
            "placement_errors": placement_errors,
            "attempts_per_run": counts}),
        "run_records_valid": gate(
            len(rows) == len(expected) and
            all(row.get("valid") is True for row in rows) and
            all(not reasons for reasons in recomputed.values()), run_validity),
        "binary_model_identity": gate(identity_ok, {
            "expected": identity,
            "unique_after": len({json.dumps(row.get("identity_after"),
                                                   sort_keys=True)
                                 for row in rows})}),
        "io_uring_qd4_direct": gate(io_ok, None),
        "schedule_selected_by_environment": gate(schedule_ok, None),
        "pm_qos_fd_scope": gate(qos_ok, None),
        "deterministic_stdout": gate(len(hashes("stdout_sha256")) == 1,
                                     hashes("stdout_sha256")),
        "identical_routes": gate(
            len(rows) == len(expected) and
            route_capture_count == len(expected) and
            len(route_hashes) == 1,
            {"captured_runs": route_capture_count,
             "expected_runs": len(expected), "route_sha256": route_hashes}),
        "no_swap_or_full_memory_pressure": gate(no_pressure, None),
        "thermal_cppc_control": gate(thermal_cppc, None),
        "gpu_control": gate(gpu_ok, None),
    }
    exact_pass = all(value["pass"] for value in gates.values())

    eligible_blocks: dict[int, dict[str, dict[str, Any]]] = {}
    for block in range(1, int(campaign["blocks"]) + 1):
        group = {row["treatment"]: row for row in rows
                 if int(row["block"]) == block and row.get("valid") is True and
                 not recomputed.get(row["run_id"], [])}
        if set(group) == set(TREATMENT_ORDER):
            eligible_blocks[block] = group

    per_treatment: dict[str, Any] = {}
    for treatment in TREATMENT_ORDER:
        group = [row for block in eligible_blocks.values()
                 for key, row in block.items() if key == treatment]
        per_treatment[treatment] = {
            "name": TREATMENTS[treatment]["name"],
            "metrics": {metric: distribution(
                [metric_value(row, metric) for row in group]) for metric in METRICS},
        }

    effects: dict[str, Any] = {}
    if eligible_blocks:
        for effect_name, coefficients in EFFECTS.items():
            effects[effect_name] = {
                metric: effect_estimate(eligible_blocks, metric, coefficients)
                for metric in METRICS
            }

    block_effects: dict[str, Any] = {}
    for metric in METRICS:
        means = {str(block): statistics.fmean(
            metric_value(row, metric) for row in group.values())
                 for block, group in eligible_blocks.items()}
        overall = statistics.fmean(means.values()) if means else None
        block_effects[metric] = {
            "block_means": means, "grand_mean": overall,
            "relative_percent": ({block: 100.0 * (value / overall - 1.0)
                                  for block, value in means.items()}
                                 if overall else {}),
        }

    position_effects: dict[str, Any] = {}
    for position in range(1, 5):
        group = [row for block in eligible_blocks.values() for row in block.values()
                 if int(row["position"]) == position]
        position_effects[str(position)] = {
            metric: distribution([metric_value(row, metric) for row in group])
            for metric in ("total_ms", "expert_compute_ms", "wall_seconds")
        }

    performance: dict[str, Any] = {"descriptive_only": not exact_pass}
    if effects:
        schedule = effects["schedule_main"]
        performance.update({
            "whole_schedule_total_percent":
                schedule["total_ms"].get("ratio", {}).get("percent_change"),
            "whole_schedule_compute_percent":
                schedule["expert_compute_ms"].get("ratio", {}).get("percent_change"),
            "whole_total_ci_excludes_slowdown":
                schedule["total_ms"].get("ratio", {}).get(
                    "percent_change_ci95", [None, None])[1] is not None and
                schedule["total_ms"]["ratio"]["percent_change_ci95"][1] < 0,
            "whole_compute_ci_excludes_slowdown":
                schedule["expert_compute_ms"].get("ratio", {}).get(
                    "percent_change_ci95", [None, None])[1] is not None and
                schedule["expert_compute_ms"]["ratio"]["percent_change_ci95"][1] < 0,
            "qos_interaction_total_percent":
                effects["schedule_x_qos"]["total_ms"].get(
                    "ratio", {}).get("percent_change"),
        })

    return {
        "schema": "waste.spark_whole_expert_analysis.v1",
        "campaign": str(campaign_dir), "attempts": len(attempts),
        "latest_runs": len(latest), "eligible_blocks": sorted(eligible_blocks),
        "exact_gates": gates, "exact_gates_pass": exact_pass,
        "traffic_comparability": traffic_comparability,
        "performance": performance, "per_treatment": per_treatment,
        "factorial_effects": effects, "block_effects": block_effects,
        "position_effects": position_effects,
    }


def report_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Spark whole-expert acceptance", "",
             f"Exact gates: **{'PASS' if summary['exact_gates_pass'] else 'FAIL'}**. "
             f"Eligible balanced blocks: {len(summary['eligible_blocks'])}.", "",
             "## Exact gates", "", "| Gate | Result |", "|---|---:|"]
    for name, value in summary["exact_gates"].items():
        lines.append(f"| {name.replace('_', ' ')} | "
                     f"{'PASS' if value['pass'] else 'FAIL'} |")
    traffic = summary["traffic_comparability"]
    lines += ["", "## Traffic comparability", "",
              f"Global traffic match: **{'YES' if traffic['global_match'] else 'NO'}**.",
              "", traffic["interpretation"], "",
              "| Treatment | Traffic variants | Layer variants | "
              "Cache misses | Bytes read |",
              "|---|---:|---:|---:|---:|"]
    for treatment in TREATMENT_ORDER:
        row = traffic["by_treatment"][treatment]
        misses = row["decode_traffic"]["cache_misses"]
        read = row["decode_traffic"]["bytes_read"]
        lines.append(
            f"| {treatment} | {len(row['traffic_sha256'])} | "
            f"{len(row['layer_signature_sha256'])} | "
            f"{misses.get('mean', 'n/a')} | {read.get('mean', 'n/a')} |")
    if summary["factorial_effects"]:
        lines += ["", "## Paired factorial effects", "",
                  "Negative percentages are faster/lower.", "",
                  "| Effect | Decode total | Routed compute |", "|---|---:|---:|"]
        for effect in ("schedule_main", "whole_at_no_qos", "whole_at_qos0",
                       "qos_main", "schedule_x_qos"):
            row = summary["factorial_effects"][effect]
            cells = []
            for metric in ("total_ms", "expert_compute_ms"):
                ratio = row[metric].get("ratio")
                if not ratio:
                    cells.append("n/a")
                else:
                    lo, hi = ratio["percent_change_ci95"]
                    cells.append(f"{ratio['percent_change']:+.2f}% "
                                 f"[{lo:+.2f}, {hi:+.2f}]")
            lines.append(f"| {effect.replace('_', ' ')} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines) + "\n"


def write_runs_csv(path: Path, campaign_dir: Path) -> None:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(campaign_dir / "runs.jsonl"):
        latest[row["run_id"]] = row
    rows = []
    for run in sorted(latest.values(), key=lambda row: (int(row["block"]),
                                                        int(row["position"]))):
        rows.append({"run_id": run["run_id"], "block": run["block"],
                     "position": run["position"],
                     "treatment": run["treatment"], "valid": run["valid"],
                     "total_ms": run.get("trace", {}).get("decode", {}).get("total_ms"),
                     "expert_compute_ms": run.get("trace", {}).get("decode", {}).get(
                         "expert_compute_ms"),
                     "expert_compute_work_ms": run.get("trace", {}).get("decode", {}).get(
                         "expert_compute_work_ms"),
                     "expert_io_ms": run.get("trace", {}).get("decode", {}).get(
                         "expert_io_ms"),
                     "wall_seconds": run.get("wall_seconds"),
                     "route_sha256": run.get("trace", {}).get("route_sha256"),
                     "traffic_sha256": run.get("trace", {}).get("traffic_sha256")})
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    campaign = args.campaign.resolve()
    out = (args.out or campaign / "analysis").resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = analyze(campaign)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "report.md").write_text(report_markdown(summary))
    write_runs_csv(out / "runs.csv", campaign)
    print(f"exact gates: {'PASS' if summary['exact_gates_pass'] else 'FAIL'}; "
          f"eligible blocks: {len(summary['eligible_blocks'])}")
    if summary["performance"].get("whole_schedule_total_percent") is not None:
        print("whole schedule main effect: "
              f"total {summary['performance']['whole_schedule_total_percent']:+.2f}%, "
              f"routed compute "
              f"{summary['performance']['whole_schedule_compute_percent']:+.2f}%")
    print(f"wrote {out / 'summary.json'}, {out / 'report.md'}, and {out / 'runs.csv'}")
    return 0 if summary["exact_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
