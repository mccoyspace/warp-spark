#!/usr/bin/env python3
"""Summarize WASTE_DUMP_ROUTE_MARGIN CSV evidence.

The reported margin is the score of the selected Kth expert minus the best
unselected expert, after sigmoid and correction bias. If every router score
can move by at most epsilon, rows with margin <= 2*epsilon are the rows whose
route cannot be certified invariant from the margin alone.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict


DEFAULT_ERRORS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1)


def quantile(values, q):
    if not values:
        raise ValueError("empty margin trace")
    if len(values) == 1:
        return values[0]
    x = q * (len(values) - 1)
    lo = int(math.floor(x))
    hi = int(math.ceil(x))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - x) + values[hi] * (x - lo)


def read_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"pos", "layer", "kth_id", "next_id", "margin"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("not a WASTE route-margin CSV")
        rows = []
        for line, row in enumerate(reader, 2):
            try:
                margin = float(row["margin"])
                item = {
                    "pos": int(row["pos"]),
                    "layer": int(row["layer"]),
                    "kth_id": int(row["kth_id"]),
                    "next_id": int(row["next_id"]),
                    "margin": margin,
                    "phase": row.get("phase", "?"),
                    "attention": row.get("attention", "?"),
                    "boundary_ties": int(row.get("boundary_ties", "0")),
                    "nonfinite_scores": int(row.get("nonfinite_scores", "0")),
                    "score_std": float(row.get("score_std", "nan")),
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid row {line}: {exc}") from exc
            if not math.isfinite(margin) or margin < -1e-6:
                raise ValueError(f"invalid margin at row {line}: {margin}")
            # Tiny negatives are possible only from decimal round-tripping;
            # the selection algorithm itself guarantees a nonnegative gap.
            item["margin"] = max(0.0, margin)
            rows.append(item)
    if not rows:
        raise ValueError("empty margin trace")
    return rows


def stats(values):
    values = sorted(values)
    return {
        "min": values[0],
        "p0_1": quantile(values, 0.001),
        "p1": quantile(values, 0.01),
        "p5": quantile(values, 0.05),
        "p50": quantile(values, 0.50),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": values[-1],
    }


def summarize(rows, errors):
    margins = [row["margin"] for row in rows]
    by_layer = defaultdict(list)
    by_phase = defaultdict(list)
    by_attention = defaultdict(list)
    by_position = defaultdict(list)
    for row in rows:
        by_layer[row["layer"]].append(row["margin"])
        by_phase[row["phase"]].append(row["margin"])
        by_attention[row["attention"]].append(row["margin"])
        by_position[row["pos"]].append(row["margin"])
    risk = []
    for error in errors:
        at_risk = sum(margin <= 2.0 * error for margin in margins)
        risk.append({
            "max_score_error": error,
            "required_margin": 2.0 * error,
            "rows_not_certified": at_risk,
            "fraction_not_certified": at_risk / len(margins),
        })
    return {
        "rows": len(rows),
        "positions": len({row["pos"] for row in rows}),
        "layers": len(by_layer),
        "exact_boundary_ties": sum(row["margin"] == 0.0 for row in rows),
        "rows_with_nonfinite_scores": sum(row["nonfinite_scores"] > 0 for row in rows),
        "margin": stats(margins),
        "per_position_min_margin": stats([min(values) for values in by_position.values()]),
        "by_phase": {key: {"rows": len(values), **stats(values)}
                     for key, values in sorted(by_phase.items())},
        "by_attention": {key: {"rows": len(values), **stats(values)}
                         for key, values in sorted(by_attention.items())},
        "score_error_risk": risk,
        "per_layer": [
            {"layer": layer, "rows": len(values), **stats(values)}
            for layer, values in sorted(by_layer.items())
        ],
    }


def print_human(summary):
    print(f"rows={summary['rows']} positions={summary['positions']} "
          f"layers={summary['layers']} ties={summary['exact_boundary_ties']} "
          f"nonfinite={summary['rows_with_nonfinite_scores']}")
    s = summary["margin"]
    print("margin " + " ".join(f"{key}={value:.9g}" for key, value in s.items()))
    print("score-error bound (route invariance requires margin > 2*error):")
    for row in summary["score_error_risk"]:
        print(f"  error={row['max_score_error']:.6g} "
              f"not-certified={row['rows_not_certified']}/{summary['rows']} "
              f"({100.0 * row['fraction_not_certified']:.3f}%)")
    for name, groups in (("phase", summary["by_phase"]),
                         ("attention", summary["by_attention"])):
        print(f"by {name}:")
        for key, row in groups.items():
            print(f"  {key}: rows={row['rows']} min={row['min']:.9g} "
                  f"p1={row['p1']:.9g} p50={row['p50']:.9g}")
    weakest = sorted(summary["per_layer"], key=lambda row: (row["p1"], row["min"]))[:10]
    print("weakest layers by p1 margin:")
    for row in weakest:
        print(f"  layer={row['layer']} rows={row['rows']} min={row['min']:.9g} "
              f"p1={row['p1']:.9g} p50={row['p50']:.9g}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="CSV written by WASTE_DUMP_ROUTE_MARGIN")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--score-error", action="append", type=float,
                        help="per-score absolute error bound (repeatable)")
    args = parser.parse_args(argv)
    errors = args.score_error if args.score_error is not None else DEFAULT_ERRORS
    if any(not math.isfinite(x) or x < 0 for x in errors):
        parser.error("--score-error values must be finite and nonnegative")
    try:
        summary = summarize(read_rows(args.trace), sorted(set(errors)))
    except (OSError, ValueError) as exc:
        print(f"route_margins: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
