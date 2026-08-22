#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Validate and summarize WASTE_DUMP_PREFILL JSONL traces."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCHEMA = "waste-prefill-v1"
REQUIRED = {
    "schema", "pos", "tokens", "layer", "moe", "experts_total",
    "selected_pairs", "experts_unique", "expert_density",
    "expert_record_bytes", "logical_expert_bytes", "full_bank_bytes",
    "cache_hits", "cache_misses", "physical_bytes_read", "layer_ms",
    "attention_ms", "feed_forward_ms", "moe_ms", "expert_acquire_ms", "ok",
}
INTEGER_FIELDS = {
    "pos", "tokens", "layer", "experts_total", "selected_pairs",
    "experts_unique", "expert_record_bytes", "logical_expert_bytes",
    "full_bank_bytes", "cache_hits", "cache_misses",
    "physical_bytes_read",
}
FLOAT_FIELDS = {
    "expert_density", "layer_ms", "attention_ms", "feed_forward_ms",
    "moe_ms", "expert_acquire_ms",
}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def load(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}")
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{path}:{line_no}: row must be a JSON object")
                missing = REQUIRED - row.keys()
                if missing:
                    raise ValueError(
                        f"{path}:{line_no}: missing {sorted(missing)}")
                if row["schema"] != SCHEMA:
                    raise ValueError(
                        f"{path}:{line_no}: unsupported schema "
                        f"{row['schema']!r}")
                for name in ("moe", "ok"):
                    if not isinstance(row[name], bool):
                        raise ValueError(
                            f"{path}:{line_no}: invalid {name} flag")
                for name in INTEGER_FIELDS:
                    if (not isinstance(row[name], int) or
                            isinstance(row[name], bool)):
                        raise ValueError(
                            f"{path}:{line_no}: invalid integer {name}")
                    if row[name] < 0:
                        raise ValueError(
                            f"{path}:{line_no}: negative {name}")
                for name in FLOAT_FIELDS:
                    if (not isinstance(row[name], (int, float)) or
                            isinstance(row[name], bool) or
                            not math.isfinite(float(row[name])) or
                            row[name] < 0):
                        raise ValueError(
                            f"{path}:{line_no}: invalid {name}")
                if row["tokens"] < 2 or row["layer"] < 0:
                    raise ValueError(
                        f"{path}:{line_no}: invalid chunk coordinates")
                if row["expert_density"] > 1:
                    raise ValueError(
                        f"{path}:{line_no}: invalid expert_density")
                if row["attention_ms"] > row["layer_ms"] + 1e-6:
                    raise ValueError(
                        f"{path}:{line_no}: attention exceeds layer time")
                if row["feed_forward_ms"] > row["layer_ms"] + 1e-6:
                    raise ValueError(
                        f"{path}:{line_no}: feed-forward exceeds layer time")
                if row["expert_acquire_ms"] > row["feed_forward_ms"] + 1e-6:
                    raise ValueError(
                        f"{path}:{line_no}: expert acquire exceeds "
                        "feed-forward time")
                if row["moe_ms"] > row["layer_ms"] + 1e-6:
                    raise ValueError(
                        f"{path}:{line_no}: MoE exceeds layer time")
                if row["moe"]:
                    total = int(row["experts_total"])
                    unique = int(row["experts_unique"])
                    record = int(row["expert_record_bytes"])
                    if not (0 < unique <= total):
                        raise ValueError(
                            f"{path}:{line_no}: invalid expert union")
                    if row["logical_expert_bytes"] != unique * record:
                        raise ValueError(
                            f"{path}:{line_no}: logical byte mismatch")
                    if row["full_bank_bytes"] != total * record:
                        raise ValueError(
                            f"{path}:{line_no}: full-bank byte mismatch")
                    expected_density = unique / total
                    if not math.isclose(
                            float(row["expert_density"]), expected_density,
                            rel_tol=1e-7, abs_tol=1e-9):
                        raise ValueError(
                            f"{path}:{line_no}: expert density mismatch")
                rows.append(row)
    if not rows:
        raise ValueError("trace contains no rows")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    moe = [row for row in rows if row["moe"]]
    # Every buffered chunk starts with layer zero. Counting unique (pos,
    # tokens) coordinates would collapse repetitions and appended runs, which
    # commonly restart at position zero and are exactly what campaigns need
    # the summary to preserve.
    chunks = sum(int(row["layer"]) == 0 for row in rows)
    if not chunks:
        raise ValueError("trace contains no complete chunk start")
    densities = [float(row["expert_density"]) for row in moe]
    reuse = [float(row["selected_pairs"]) / row["experts_unique"]
             for row in moe]
    hits = sum(int(row["cache_hits"]) for row in moe)
    misses = sum(int(row["cache_misses"]) for row in moe)
    logical = sum(int(row["logical_expert_bytes"]) for row in moe)
    full = sum(int(row["full_bank_bytes"]) for row in moe)
    physical = sum(int(row["physical_bytes_read"]) for row in moe)
    failed = sum(not bool(row["ok"]) for row in rows)
    return {
        "schema": "waste-prefill-summary-v1",
        "rows": len(rows),
        "moe_rows": len(moe),
        "chunks": chunks,
        "failed_rows": failed,
        "layers": sorted({int(row["layer"]) for row in rows}),
        "chunk_tokens": sorted({int(row["tokens"]) for row in rows}),
        "expert_density": distribution(densities),
        "selected_pairs_per_unique_expert": distribution(reuse),
        "cache": {
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / (hits + misses) if hits + misses else None,
        },
        "bytes": {
            "logical_demand": logical,
            "physical_read": physical,
            "full_layer_stream": full,
            "full_layer_extra_over_logical": max(0, full - logical),
            "full_layer_to_logical_ratio": full / logical if logical else None,
        },
        "timing_ms": {
            "layers_total": sum(float(row["layer_ms"]) for row in rows),
            "attention_total": sum(float(row["attention_ms"])
                                   for row in rows),
            "feed_forward_total": sum(float(row["feed_forward_ms"])
                                      for row in rows),
            "expert_acquire_total": sum(float(row["expert_acquire_ms"])
                                        for row in rows),
            "layer": distribution([float(row["layer_ms"])
                                   for row in rows]),
        },
    }


def human(summary: dict[str, Any]) -> str:
    density = summary["expert_density"]
    byte = summary["bytes"]
    hit = summary["cache"]["hit_rate"]
    ratio = byte["full_layer_to_logical_ratio"]
    return "\n".join([
        f"rows {summary['rows']} ({summary['moe_rows']} MoE), "
        f"chunks {summary['chunks']}",
        "expert density "
        f"mean {density['mean']:.3f}, p50 {density['p50']:.3f}, "
        f"p95 {density['p95']:.3f}" if density["mean"] is not None
        else "expert density unavailable",
        f"cache hit rate {hit:.3f}" if hit is not None
        else "cache hit rate unavailable",
        f"logical expert bytes {byte['logical_demand'] / 2**30:.3f} GiB; "
        f"full-layer bytes {byte['full_layer_stream'] / 2**30:.3f} GiB; "
        f"ratio {ratio:.3f}x" if ratio is not None
        else "expert byte ratio unavailable",
        f"summed layer time {summary['timing_ms']['layers_total'] / 1000:.3f} s",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true",
                        help="write the machine-readable summary")
    args = parser.parse_args(argv)
    try:
        summary = summarize(load(args.trace))
    except (OSError, TypeError, ValueError) as exc:
        print(f"prefill_trace.py: {exc}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print(human(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
