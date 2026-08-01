#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Phase-gated scheduler/futex sidecar for WASTE routed compute.

Record (Linux, bpftrace, usually passwordless sudo):

  python3 tools/offcpu_diagnostic.py record \
    --cpus 5-9,15-19 --trace /tmp/layers.jsonl \
    --events /tmp/offcpu.tsv --summary /tmp/offcpu.json -- \
    ./waste run MODEL 'hello' -n 4 --trace-layers /tmp/layers.jsonl

Analyze an existing capture without tracing again:

  python3 tools/offcpu_diagnostic.py analyze --trace /tmp/layers.jsonl \
    --events /tmp/offcpu.tsv --summary /tmp/offcpu.json

The primary trace supplies exact expert-compute intervals in CLOCK_MONOTONIC
nanoseconds.  bpftrace's `nsecs` uses the same clock.  Scheduler, wakeup,
futex, and CPU-idle events are therefore filtered after capture to only the
routed arithmetic windows; this keeps the sidecar diagnostic separate from
the benchmark timing path.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
import getpass
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Iterable, Optional


PREFIX = "WASTE_OFFCPU\t"
IDLE_EXIT = (1 << 32) - 1
# pthread condition variables and mutexes normally use WAIT or WAIT_BITSET.
# Include the PI/requeue and futex2 wait opcodes so the classifier remains
# useful when libc changes implementation.
FUTEX_WAIT_OPS = {0, 6, 9, 11, 13, 31}


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    position: int
    layer: int
    expert_index: int
    exact: bool


def parse_cpu_set(text: str) -> list[int]:
    cpus: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo < 0 or hi < lo:
                raise ValueError(f"invalid CPU range: {part}")
            cpus.update(range(lo, hi + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError(f"invalid CPU: {part}")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU set is empty")
    return sorted(cpus)


def load_windows(path: Path) -> tuple[list[Window], dict]:
    windows: list[Window] = []
    layer_rows = 0
    exact_rows = 0
    raw_exact_intervals = 0
    overlapping_rows = 0
    whole_schedule_rows = 0
    raw_expert_work_ns = 0
    raw_expert_union_ns = 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            if row.get("event") != "decode_layer":
                continue
            layer_rows += 1
            raw = row.get("expert_compute_intervals_monotonic_ns")
            if raw:
                exact_rows += 1
                row_windows: list[tuple[int, int, int]] = []
                for j, pair in enumerate(raw):
                    if (not isinstance(pair, list) or len(pair) != 2 or
                            not all(isinstance(v, int) for v in pair)):
                        raise ValueError(
                            f"{path}:{lineno}: malformed compute interval")
                    start, end = pair
                    if end <= start:
                        raise ValueError(
                            f"{path}:{lineno}: non-positive compute interval")
                    row_windows.append((start, end, j))
                raw_exact_intervals += len(row_windows)
                raw_expert_work_ns += sum(end - start
                                          for start, end, _ in row_windows)
                row_windows.sort()
                merged: list[tuple[int, int, int]] = []
                row_overlaps = False
                for start, end, expert_index in row_windows:
                    if merged and start < merged[-1][1]:
                        row_overlaps = True
                        old_start, old_end, _ = merged[-1]
                        merged[-1] = (old_start, max(old_end, end), -1)
                    else:
                        merged.append((start, end, expert_index))
                overlapping_rows += int(row_overlaps)
                raw_expert_union_ns += sum(end - start
                                           for start, end, _ in merged)
                position = int(row.get("position", -1))
                layer = int(row.get("layer", -1))
                if row.get("expert_schedule") == "whole":
                    # Expert intervals overlap and intentionally exclude the
                    # two shared LUT pool jobs plus the ordered reduction.
                    # Gate scheduler/off-CPU accounting on the exact outer
                    # phase, while retaining interval work/union metadata.
                    start = row.get("routed_loop_start_monotonic_ns")
                    end = row.get("routed_loop_end_monotonic_ns")
                    if not (isinstance(start, int) and isinstance(end, int)
                            and end > start):
                        raise ValueError(
                            f"{path}:{lineno}: whole schedule lacks routed phase")
                    whole_schedule_rows += 1
                    windows.append(Window(start, end, position, layer, -1, True))
                else:
                    windows.extend(Window(start, end, position, layer,
                                          expert_index, True)
                                   for start, end, expert_index in merged)
            else:
                start = row.get("routed_loop_start_monotonic_ns")
                end = row.get("routed_loop_end_monotonic_ns")
                if isinstance(start, int) and isinstance(end, int) and end > start:
                    windows.append(Window(start, end,
                                          int(row.get("position", -1)),
                                          int(row.get("layer", -1)), 0, False))
    windows.sort(key=lambda w: (w.start, w.end))
    if not windows:
        raise ValueError(
            f"{path}: no decode routed-compute intervals; rebuild with the "
            "Sprint 4 trace fields and capture at least one decoded token")
    for a, b in zip(windows, windows[1:]):
        if b.start < a.end:
            raise ValueError(
                f"overlapping routed windows at positions {a.position}/{b.position}")
    return windows, {
        "decode_layer_rows": layer_rows,
        "exact_interval_rows": exact_rows,
        "exact_intervals": all(w.exact for w in windows),
        "raw_expert_intervals": raw_exact_intervals,
        "overlapping_interval_rows": overlapping_rows,
        "whole_schedule_rows": whole_schedule_rows,
        "raw_expert_work_ms": round(raw_expert_work_ns / 1e6, 3),
        "raw_expert_union_ms": round(raw_expert_union_ns / 1e6, 3),
        "coalesced_routed_intervals": len(windows),
    }


def load_events(path: Path) -> tuple[list[dict], dict]:
    events: list[dict] = []
    meta: dict = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.find(PREFIX)
            if p < 0:
                continue                         # workload stdout/stderr
            fields = line[p + len(PREFIX):].rstrip("\r\n").split("\t")
            kind = fields[0]
            try:
                if kind == "META":
                    event = {"kind": kind, "ts": int(fields[1]),
                             "cpid": int(fields[2])}
                    meta["cpid"] = event["cpid"]
                elif kind == "THREAD":
                    event = {"kind": kind, "ts": int(fields[1]),
                             "parent": int(fields[2]), "tid": int(fields[3])}
                elif kind == "THREAD_EXIT":
                    event = {"kind": kind, "ts": int(fields[1]),
                             "tid": int(fields[2])}
                elif kind == "SWITCH_OUT":
                    event = {"kind": kind, "ts": int(fields[1]),
                             "cpu": int(fields[2]), "tid": int(fields[3]),
                             "state": int(fields[4]), "next": int(fields[5])}
                elif kind == "SWITCH_IN":
                    event = {"kind": kind, "ts": int(fields[1]),
                             "cpu": int(fields[2]), "tid": int(fields[3]),
                             "prev": int(fields[4])}
                elif kind == "WAKEUP":
                    event = {"kind": kind, "ts": int(fields[1]),
                             "cpu": int(fields[2]), "tid": int(fields[3]),
                             "waker": int(fields[4]),
                             "target_cpu": int(fields[5])}
                elif kind == "FUTEX_ENTER":
                    event = {"kind": kind, "ts": int(fields[1]),
                             "cpu": int(fields[2]), "tid": int(fields[3]),
                             "op": int(fields[4]), "addr": int(fields[5])}
                elif kind == "FUTEX_EXIT":
                    duration = int(fields[7])
                    event = {"kind": kind, "ts": int(fields[1]),
                             "cpu": int(fields[2]), "tid": int(fields[3]),
                             "op": int(fields[4]), "addr": int(fields[5]),
                             "ret": int(fields[6]), "duration": duration,
                             "start": int(fields[1]) - duration}
                elif kind == "CPU_IDLE":
                    event = {"kind": kind, "ts": int(fields[1]),
                             "cpu": int(fields[2]), "state": int(fields[3])}
                else:
                    continue
            except (IndexError, ValueError) as e:
                raise ValueError(f"malformed {kind} line: {line.rstrip()}") from e
            events.append(event)
    events.sort(key=lambda e: e["ts"])
    if "cpid" not in meta:
        raise ValueError(f"{path}: no WASTE_OFFCPU META event")
    return events, meta


class WindowIndex:
    def __init__(self, windows: list[Window]):
        self.windows = windows
        self.starts = [w.start for w in windows]

    def containing(self, ts: int) -> Optional[Window]:
        i = bisect_right(self.starts, ts) - 1
        if i >= 0 and ts < self.windows[i].end:
            return self.windows[i]
        return None

    def overlaps(self, start: int, end: int) -> Iterable[tuple[Window, int]]:
        if end <= start:
            return
        i = max(0, bisect_right(self.starts, start) - 1)
        while i < len(self.windows) and self.windows[i].end <= start:
            i += 1
        while i < len(self.windows) and self.windows[i].start < end:
            w = self.windows[i]
            overlap = min(end, w.end) - max(start, w.start)
            if overlap > 0:
                yield w, overlap
            i += 1


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    p = (len(xs) - 1) * q
    lo = math.floor(p)
    hi = math.ceil(p)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - p) + xs[hi] * (p - lo)


def distribution_ns(values: list[int]) -> dict:
    us = [v / 1000.0 for v in values]
    return {
        "count": len(us),
        "mean_us": round(statistics.fmean(us), 3) if us else None,
        "p50_us": round(percentile(us, .50), 3) if us else None,
        "p95_us": round(percentile(us, .95), 3) if us else None,
        "max_us": round(max(us), 3) if us else None,
    }


def analyze(trace_path: Path, events_path: Path,
            idle_lookback_us: float = 50.0) -> dict:
    windows, trace_meta = load_windows(trace_path)
    events, event_meta = load_events(events_path)
    index = WindowIndex(windows)
    cpid = int(event_meta["cpid"])
    thread_ids = {cpid}
    for e in events:
        if e["kind"] == "THREAD":
            thread_ids.add(e["tid"])
    worker_ids = thread_ids - {cpid}

    first, last = windows[0].start, windows[-1].end
    event_first = events[0]["ts"] if events else 0
    event_last = events[-1]["ts"] if events else 0
    if event_first > first or event_last < last:
        raise ValueError(
            "sidecar does not cover all routed windows: "
            f"events={event_first}..{event_last}, trace={first}..{last}")

    futex_waits: list[dict] = []
    waits_by_tid: dict[int, list[dict]] = defaultdict(list)
    idle_start: dict[int, dict] = {}
    idle_intervals: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        if e["kind"] == "FUTEX_EXIT" and e["op"] in FUTEX_WAIT_OPS:
            futex_waits.append(e)
            waits_by_tid[e["tid"]].append(e)
        elif e["kind"] == "CPU_IDLE":
            cpu = e["cpu"]
            if e["state"] == IDLE_EXIT:
                start = idle_start.pop(cpu, None)
                if start is not None and start["ts"] <= e["ts"]:
                    idle_intervals[cpu].append({
                        "start": start["ts"], "end": e["ts"],
                        "state": start["state"],
                        "duration": e["ts"] - start["ts"],
                    })
            else:
                # A second entry means the first exit fell outside the gated
                # capture. Keep only the interval whose exit we can prove.
                idle_start[cpu] = e

    wait_ends_by_tid = {
        tid: [w["ts"] for w in waits]
        for tid, waits in waits_by_tid.items()
    }
    idle_ends_by_cpu = {
        cpu: [span["end"] for span in spans]
        for cpu, spans in idle_intervals.items()
    }

    def barrier_wait(tid: int, ts: int) -> bool:
        waits = waits_by_tid.get(tid, [])
        # Lists are naturally exit-time ordered. A wake/switch-in occurs
        # before sys_exit_futex, so the wait normally contains `ts`.
        i = bisect_left(wait_ends_by_tid.get(tid, []), ts - 50_000)
        return (i < len(waits) and waits[i]["start"] <= ts <=
                waits[i]["ts"] + 50_000)

    pending_out: dict[int, dict] = {}
    pending_wake: dict[int, dict] = {}
    offcpu: list[dict] = []
    switchins: list[dict] = []
    lookback_ns = int(idle_lookback_us * 1000)
    for e in events:
        kind = e["kind"]
        if kind == "SWITCH_OUT":
            pending_out[e["tid"]] = e
        elif kind == "WAKEUP":
            pending_wake[e["tid"]] = e
        elif kind == "SWITCH_IN":
            out = pending_out.pop(e["tid"], None)
            if out is not None and e["ts"] > out["ts"]:
                offcpu.append({"tid": e["tid"], "start": out["ts"],
                               "end": e["ts"], "state": out["state"],
                               "cpu_out": out["cpu"], "cpu_in": e["cpu"]})
            wake = pending_wake.pop(e["tid"], None)
            w = index.containing(e["ts"])
            if wake is None or w is None or wake["ts"] > e["ts"]:
                continue
            spans = idle_intervals.get(e["cpu"], [])
            j = bisect_right(idle_ends_by_cpu.get(e["cpu"], []), e["ts"]) - 1
            idle = spans[j] if j >= 0 else None
            # On this kernel the cpu_idle exit tracepoint can precede the
            # sched_wakeup tracepoint by a few microseconds: the target CPU
            # returns from firmware before the wake path emits its tracepoint.
            # Accept that narrow, configurable lead but reject the old 5 ms
            # heuristic, which could reuse one stale exit for many barriers.
            if (idle is not None and
                    not (wake["ts"] - lookback_ns <= idle["end"] <= e["ts"])):
                idle = None
            is_barrier = barrier_wait(e["tid"], e["ts"])
            is_worker = e["tid"] in worker_ids
            if is_barrier and idle is not None:
                category = "barrier_idle_exit"
            elif is_barrier:
                category = "barrier_no_idle_exit"
            elif idle is not None:
                category = "nonbarrier_idle_exit"
            else:
                category = "other"
            switchins.append({
                "tid": e["tid"], "cpu": e["cpu"], "ts": e["ts"],
                "wake_ts": wake["ts"], "latency": e["ts"] - wake["ts"],
                "idle_exit_ts": idle["end"] if idle else None,
                "idle_state": idle["state"] if idle else None,
                "idle_residency_ns": idle["duration"] if idle else None,
                "barrier": is_barrier,
                "worker": is_worker, "category": category, "window": w,
            })

    layer_metrics: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"compute_ns": 0, "blocked_ns": 0, "runnable_ns": 0,
                 "futex_wait_ns": 0, "wake_latencies": [],
                 "categories": defaultdict(int)})
    for w in windows:
        layer_metrics[(w.position, w.layer)]["compute_ns"] += w.end - w.start

    blocked_ns = runnable_ns = 0
    for span in offcpu:
        for w, overlap in index.overlaps(span["start"], span["end"]):
            key = (w.position, w.layer)
            if span["state"] == 0:
                runnable_ns += overlap
                layer_metrics[key]["runnable_ns"] += overlap
            else:
                blocked_ns += overlap
                layer_metrics[key]["blocked_ns"] += overlap

    futex_wait_ns = 0
    all_thread_futex_wait_ns = 0
    futex_wait_calls_in_window: set[tuple[int, int]] = set()
    all_thread_futex_wait_calls: set[tuple[int, int]] = set()
    for n, wait in enumerate(futex_waits):
        for w, overlap in index.overlaps(wait["start"], wait["ts"]):
            all_thread_futex_wait_ns += overlap
            all_thread_futex_wait_calls.add((wait["tid"], n))
            if wait["tid"] not in worker_ids:
                continue
            futex_wait_ns += overlap
            futex_wait_calls_in_window.add((wait["tid"], n))
            layer_metrics[(w.position, w.layer)]["futex_wait_ns"] += overlap

    worker_switchins = [s for s in switchins if s["worker"]]
    categories: dict[str, list[int]] = defaultdict(list)
    for s in worker_switchins:
        categories[s["category"]].append(s["latency"])
        key = (s["window"].position, s["window"].layer)
        layer_metrics[key]["wake_latencies"].append(s["latency"])
        layer_metrics[key]["categories"][s["category"]] += 1

    compute_ns = sum(w.end - w.start for w in windows)
    worker_capacity_ns = compute_ns * len(worker_ids)
    cat_summary = {name: distribution_ns(categories.get(name, []))
                   for name in ("barrier_idle_exit", "barrier_no_idle_exit",
                                "nonbarrier_idle_exit", "other")}
    barrier_n = (cat_summary["barrier_idle_exit"]["count"] +
                 cat_summary["barrier_no_idle_exit"]["count"])
    idle_barrier_n = cat_summary["barrier_idle_exit"]["count"]
    idle_by_state: dict[int, list[int]] = defaultdict(list)
    for s in worker_switchins:
        if s["barrier"] and s["idle_state"] is not None:
            idle_by_state[s["idle_state"]].append(s["idle_residency_ns"])
    idle_state_summary = {
        str(state): {
            "count": len(values),
            "fraction_of_barrier_wakes":
                round(len(values) / barrier_n, 6) if barrier_n else None,
            "residency": distribution_ns(values),
        }
        for state, values in sorted(idle_by_state.items())
    }
    nonzero_idle_n = sum(len(values) for state, values in idle_by_state.items()
                         if state != 0)

    per_layer = []
    for (position, layer), m in sorted(layer_metrics.items()):
        dist = distribution_ns(m["wake_latencies"])
        per_layer.append({
            "position": position, "layer": layer,
            "compute_ms": round(m["compute_ns"] / 1e6, 3),
            "blocked_worker_ms": round(m["blocked_ns"] / 1e6, 3),
            "runnable_worker_ms": round(m["runnable_ns"] / 1e6, 3),
            "futex_wait_worker_ms": round(m["futex_wait_ns"] / 1e6, 3),
            "wake_latency": dist,
            "wake_categories": dict(m["categories"]),
        })

    if not worker_switchins:
        interpretation = (
            "No worker wake-to-run events landed inside exact routed compute "
            "windows; verify TID discovery and sidecar coverage before drawing "
            "a scheduler conclusion.")
    elif barrier_n == 0:
        interpretation = (
            "Worker wakeups inside routed compute were not paired with futex "
            "waits. Scheduler contention or a non-futex wait path is more "
            "plausible than the thread-pool barrier hypothesis.")
    elif idle_barrier_n / barrier_n >= 0.5:
        interpretation = (
            "At least half of barrier wakeups have a tightly associated CPU "
            "idle exit. Compare idle-state residency and the matched versus "
            "unmatched wake latency; temporal association alone does not "
            "partition firmware exit cost from futex/barrier cost.")
    else:
        interpretation = (
            "Most futex/barrier wakeups have no CPU idle exit within the tight "
            "association window. Barrier wake latency is the stronger "
            "explanation in this capture; idle exit is present but not "
            "dominant.")

    return {
        "schema": "waste.offcpu.v1",
        "trace": str(trace_path),
        "events": str(events_path),
        "cpid": cpid,
        "thread_ids": sorted(thread_ids),
        "worker_threads": len(worker_ids),
        "coverage": {
            **trace_meta,
            "routed_intervals": len(windows),
            "first_monotonic_ns": first,
            "last_monotonic_ns": last,
            "compute_wall_ms": round(compute_ns / 1e6, 3),
            "event_first_monotonic_ns": event_first,
            "event_last_monotonic_ns": event_last,
        },
        "offcpu": {
            "blocked_worker_ms": round(blocked_ns / 1e6, 3),
            "runnable_worker_ms": round(runnable_ns / 1e6, 3),
            "blocked_fraction_of_worker_capacity":
                round(blocked_ns / worker_capacity_ns, 6)
                if worker_capacity_ns else None,
            "runnable_fraction_of_worker_capacity":
                round(runnable_ns / worker_capacity_ns, 6)
                if worker_capacity_ns else None,
        },
        "futex": {
            "wait_calls_overlapping_compute": len(futex_wait_calls_in_window),
            "wait_worker_ms": round(futex_wait_ns / 1e6, 3),
            "all_thread_wait_calls_overlapping_compute":
                len(all_thread_futex_wait_calls),
            "all_thread_wait_ms": round(all_thread_futex_wait_ns / 1e6, 3),
        },
        "wake_to_run": {
            "all_workers": distribution_ns(
                [s["latency"] for s in worker_switchins]),
            "categories": cat_summary,
            "barrier_idle_exit_fraction":
                round(idle_barrier_n / barrier_n, 6) if barrier_n else None,
        },
        "idle_exit": {
            "association_lead_us": idle_lookback_us,
            "matched_barrier_wakes": idle_barrier_n,
            "nonzero_state_matched_barrier_wakes": nonzero_idle_n,
            "nonzero_state_fraction_of_barrier_wakes":
                round(nonzero_idle_n / barrier_n, 6) if barrier_n else None,
            "states": idle_state_summary,
        },
        "interpretation": interpretation,
        "per_layer": per_layer,
    }


def markdown(summary: dict) -> str:
    cov = summary["coverage"]
    off = summary["offcpu"]
    futex = summary["futex"]
    wake = summary["wake_to_run"]
    idle = summary["idle_exit"]
    lines = [
        "# WASTE off-CPU diagnostic",
        "",
        f"Coverage: {cov['routed_intervals']} routed expert intervals, "
        f"{cov['compute_wall_ms']:.3f} ms of compute wall time, "
        f"{summary['worker_threads']} worker threads. Exact intervals: "
        f"{'yes' if cov['exact_intervals'] else 'no (routed-loop fallback)' }.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Blocked off-CPU worker time | {off['blocked_worker_ms']:.3f} ms |",
        f"| Runnable/preempted worker time | {off['runnable_worker_ms']:.3f} ms |",
        f"| Futex wait time overlapping compute | {futex['wait_worker_ms']:.3f} ms |",
        f"| Futex wait calls overlapping compute | {futex['wait_calls_overlapping_compute']} |",
        f"| Worker wake-to-run p50 | {wake['all_workers']['p50_us']} us |",
        f"| Worker wake-to-run p95 | {wake['all_workers']['p95_us']} us |",
        f"| Barrier wakes with a tightly matched idle exit | "
        f"{idle['matched_barrier_wakes']} "
        f"({100 * wake['barrier_idle_exit_fraction']:.2f}%) |",
        f"| Barrier wakes exiting nonzero idle states | "
        f"{idle['nonzero_state_matched_barrier_wakes']} "
        f"({100 * idle['nonzero_state_fraction_of_barrier_wakes']:.2f}%) |",
        "",
        "## Wake classification",
        "",
        "| Class | Count | p50 us | p95 us |",
        "|---|---:|---:|---:|",
    ]
    for name, d in wake["categories"].items():
        lines.append(f"| {name} | {d['count']} | {d['p50_us']} | {d['p95_us']} |")
    lines += ["", "## Matched idle states", "",
              "| State | Count | Barrier wakes | Residency p50 us | Residency p95 us |",
              "|---:|---:|---:|---:|---:|"]
    for state, d in idle["states"].items():
        lines.append(f"| {state} | {d['count']} | "
                     f"{100 * d['fraction_of_barrier_wakes']:.2f}% | "
                     f"{d['residency']['p50_us']} | {d['residency']['p95_us']} |")
    lines += ["", "## Interpretation", "", summary["interpretation"], "",
              "## Highest wake-latency layers", "",
              "| Position | Layer | Compute ms | Wake count | Wake p95 us |",
              "|---:|---:|---:|---:|---:|"]
    ranked = sorted(summary["per_layer"],
                    key=lambda r: (r["wake_latency"]["p95_us"] or -1),
                    reverse=True)[:20]
    for row in ranked:
        d = row["wake_latency"]
        lines.append(f"| {row['position']} | {row['layer']} | "
                     f"{row['compute_ms']:.3f} | {d['count']} | {d['p95_us']} |")
    return "\n".join(lines) + "\n"


def write_summary(summary: dict, output: Path,
                  markdown_path: Optional[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if markdown_path:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown(summary))


def render_bpftrace(cpus: list[int], probe_binary: Path) -> Path:
    template = Path(__file__).with_name("offcpu_trace.bt.in").read_text()
    cpu_map = "".join(f"    @perf_cpu[{cpu}] = 1;\n" for cpu in cpus)
    binary = str(probe_binary)
    if any(c in binary for c in (":", "\n")):
        raise ValueError(f"probe binary path is not bpftrace-safe: {binary}")
    uprobes = f"""uprobe:{binary}:moe_layer
/pid == cpid/
{{
    @phase[pid] = 1;
    printf(\"WASTE_OFFCPU\\tPHASE_BEGIN\\t%llu\\t%d\\n\", nsecs, pid);
}}

uretprobe:{binary}:moe_layer
/pid == cpid/
{{
    printf(\"WASTE_OFFCPU\\tPHASE_END\\t%llu\\t%d\\n\", nsecs, pid);
    delete(@phase[pid]);
}}
"""
    source = template.replace("/*CPU_MAP*/\n", cpu_map)
    source = source.replace("/*UPROBES*/\n", uprobes)
    f = tempfile.NamedTemporaryFile("w", suffix=".bt", delete=False)
    with f:
        f.write(source)
    return Path(f.name)


def render_launcher(command: list[str], *, drop_privileges: bool) -> Path:
    """Build a one-path exec launcher for bpftrace -c.

    bpftrace 0.20 splits its `-c` string itself and does not preserve a
    quoted argument containing spaces on every distro.  Passing one launcher
    path avoids that parser entirely; the launcher execs the requested argv
    in the same PID, which is also essential because `cpid` must remain the
    WASTE TGID. When bpftrace is elevated, it drops back to the invoking user
    immediately before exec.
    """
    env = {
        "HOME": str(Path.home()),
        "USER": getpass.getuser(),
        "LOGNAME": getpass.getuser(),
        "PATH": os.environ.get("PATH", ""),
    }
    waste_env = {
        "WASTE_ALLOW_CONCURRENT", "WASTE_BACKEND", "WASTE_CGROUP_DIR",
        "WASTE_CPU_SET", "WASTE_DIRECT", "WASTE_DUMP_HIDDEN",
        "WASTE_DUMP_LATENT", "WASTE_I8MM", "WASTE_IO_BACKEND",
        "WASTE_MEMINFO_PATH", "WASTE_PROFILE", "WASTE_Q8", "WASTE_SDOT",
        "WASTE_SYS_CPU_ROOT", "WASTE_THREADS", "WASTE_VERIFY",
        "WASTE_VIS_STAGE", "LD_LIBRARY_PATH",
    }
    for key in waste_env:
        if key in os.environ:
            env[key] = os.environ[key]
    privilege = ""
    if drop_privileges:
        privilege = (
            f"os.setgroups({os.getgroups()!r})\n"
            f"os.setgid({os.getgid()})\n"
            f"os.setuid({os.getuid()})\n")
    source = ("import os\n"
              f"argv = {command!r}\n"
              f"env = os.environ.copy()\nenv.update({env!r})\n"
              f"{privilege}"
              "os.execvpe(argv[0], argv, env)\n")
    f = tempfile.NamedTemporaryFile("w", suffix="-waste-offcpu-launch",
                                    delete=False)
    with f:
        f.write(source)
    path = Path(f.name)
    return path


def record(args) -> int:
    if sys.platform != "linux":
        raise RuntimeError("recording requires Linux; use analyze on other hosts")
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("record requires a command after --")
    cpus = parse_cpu_set(args.cpus)
    bpftrace = shutil.which("bpftrace")
    if not bpftrace:
        raise RuntimeError("bpftrace is not installed")

    probe_binary = (args.probe_binary or Path(command[0])).expanduser()
    if not probe_binary.is_absolute():
        probe_binary = (Path.cwd() / probe_binary).resolve()
    if not probe_binary.is_file():
        raise ValueError(
            f"cannot infer probe binary from {command[0]!r}; use --probe-binary")
    program = render_bpftrace(cpus, probe_binary)
    args.events.parent.mkdir(parents=True, exist_ok=True)
    # A failed workload must not be analyzed against an older trace that
    # happens to have the requested name.
    args.trace.unlink(missing_ok=True)
    use_sudo = os.geteuid() != 0 and not args.no_sudo
    launcher_script = render_launcher(command, drop_privileges=use_sudo)
    launcher = [bpftrace, "-q", "-c"]
    if use_sudo:
        sudo = shutil.which("sudo")
        if not sudo:
            raise RuntimeError("recording as non-root requires sudo")
        launcher = [sudo, "-n", *launcher]
    cmd = [*launcher, f"{sys.executable} {launcher_script}", str(program)]
    try:
        with args.events.open("w", encoding="utf-8") as capture:
            result = subprocess.run(cmd, stdout=capture)
    finally:
        program.unlink(missing_ok=True)
        launcher_script.unlink(missing_ok=True)
    if result.returncode:
        raise RuntimeError(f"bpftrace/workload exited with {result.returncode}")
    summary = analyze(args.trace, args.events, args.idle_lookback_us)
    summary["command"] = command
    summary["cpus"] = cpus
    write_summary(summary, args.summary, args.markdown)
    print(markdown(summary), end="")
    return 0


def analyze_command(args) -> int:
    summary = analyze(args.trace, args.events, args.idle_lookback_us)
    # A post-capture reanalysis may improve classification without changing
    # the experiment. Preserve the original launcher provenance when the
    # output is being regenerated in place.
    if args.summary.exists():
        try:
            previous = json.loads(args.summary.read_text())
            for key in ("command", "cpus"):
                if key in previous:
                    summary[key] = previous[key]
        except (OSError, json.JSONDecodeError):
            pass
    write_summary(summary, args.summary, args.markdown)
    print(markdown(summary), end="")
    return 0


def add_analysis_args(p) -> None:
    p.add_argument("--trace", type=Path, required=True,
                   help="waste.layer_trace.v2 JSONL")
    p.add_argument("--events", type=Path, required=True,
                   help="raw WASTE_OFFCPU/bpftrace capture")
    p.add_argument("--summary", type=Path, required=True,
                   help="output JSON summary")
    p.add_argument("--markdown", type=Path,
                   help="optional human-readable report")
    p.add_argument("--idle-lookback-us", type=float, default=50.0,
                   help="maximum idle-exit lead before a worker wake (default: 50)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    rec = sub.add_parser("record", help="run a command under the sidecar")
    add_analysis_args(rec)
    rec.add_argument("--cpus", required=True,
                     help="performance CPU set, e.g. 5-9,15-19")
    rec.add_argument("--no-sudo", action="store_true",
                     help="invoke bpftrace directly (already root/capable)")
    rec.add_argument("--probe-binary", type=Path,
                     help="ELF containing local moe_layer (default command[0])")
    rec.add_argument("command", nargs=argparse.REMAINDER)
    ana = sub.add_parser("analyze", help="analyze an existing capture")
    add_analysis_args(ana)
    args = ap.parse_args(argv)
    if args.idle_lookback_us <= 0:
        ap.error("--idle-lookback-us must be positive")
    try:
        return record(args) if args.action == "record" else analyze_command(args)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"offcpu_diagnostic: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
