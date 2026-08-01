#!/usr/bin/env python3
"""Controlled Acer A/B/C campaign for expert-I/O versus compute phasing.

A: interleaved synchronous pread
B: exact decoded-layer batch, sequential pread (physical QD1)
C: exact decoded-layer batch, io_uring QD4

The script is intentionally Linux/sysfs specific.  It preserves raw evidence
per run and writes an append-only runs.jsonl even when a run is invalid.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
from typing import Any


ORDERS = (("A", "B", "C"), ("B", "C", "A"), ("C", "A", "B"),
          ("A", "C", "B"), ("C", "B", "A"), ("B", "A", "C"))
TREATMENTS = {
    "A": {"name": "pread_interleaved", "backend": "pread", "qd": 1},
    "B": {"name": "pread_batch", "backend": "pread_batch", "qd": 1},
    "C": {"name": "io_uring_qd4", "backend": "io_uring", "qd": 4},
}
VM_KEYS = ("pswpin", "pswpout", "pgmajfault", "pgscan_direct",
           "pgscan_kswapd", "pgsteal_direct", "pgsteal_kswapd")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    # Some Arm CPPC sysfs attributes return EAGAIN while a core is idle.
    # pathlib normally exposes that as OSError, but Python's text wrapper
    # can surface an empty kernel read as TypeError during decoding.
    except (OSError, UnicodeError, TypeError):
        return None


def read_int(path: Path) -> int | None:
    text = read_text(path)
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def parse_cpu_list(spec: str) -> list[int]:
    cpus: set[int] = set()
    for part in spec.split(","):
        lo_hi = part.strip().split("-", 1)
        lo = int(lo_hi[0])
        hi = int(lo_hi[1]) if len(lo_hi) == 2 else lo
        cpus.update(range(lo, hi + 1))
    return sorted(cpus)


def meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    text = read_text(Path("/proc/meminfo")) or ""
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key in ("MemAvailable", "MemFree", "SwapTotal", "SwapFree"):
            m = re.search(r"\d+", value)
            if m:
                out[key] = int(m.group()) * 1024
    return out


def vmstat() -> dict[str, int]:
    values: dict[str, int] = {}
    text = read_text(Path("/proc/vmstat")) or ""
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2 and any(fields[0].startswith(k) for k in VM_KEYS):
            values[fields[0]] = int(fields[1])
    return values


def cpuidle(cpus: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cpu in cpus:
        states = []
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpuidle")
        for state in sorted(base.glob("state*")):
            row: dict[str, Any] = {"state": state.name,
                                   "name": read_text(state / "name")}
            for field in ("usage", "time", "above", "below", "rejected",
                          "latency", "residency", "disable"):
                row[field] = read_int(state / field)
            states.append(row)
        result[str(cpu)] = states
    return result


def frequencies(cpus: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cpu in cpus:
        policy = Path(f"/sys/devices/system/cpu/cpufreq/policy{cpu}")
        cppc = Path(f"/sys/devices/system/cpu/cpu{cpu}/acpi_cppc")
        result[str(cpu)] = {
            "avg_khz": read_int(policy / "cpuinfo_avg_freq"),
            "requested_khz": read_int(policy / "scaling_cur_freq"),
            "governor": read_text(policy / "scaling_governor"),
            "driver": read_text(policy / "scaling_driver"),
            "perf_limited": read_int(cppc / "perf_limited"),
            "feedback_ctrs": read_text(cppc / "feedback_ctrs"),
        }
    return result


def temperatures() -> dict[str, Any]:
    acpi: list[int] = []
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        if read_text(zone / "type") == "acpitz":
            value = read_int(zone / "temp")
            if value is not None:
                acpi.append(value)
    nvme: list[int] = []
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        if read_text(hwmon / "name") == "nvme":
            for item in hwmon.glob("temp*_input"):
                value = read_int(item)
                if value is not None:
                    nvme.append(value)
    processor_cooling: list[int] = []
    for device in Path("/sys/class/thermal").glob("cooling_device*"):
        if read_text(device / "type") == "Processor":
            value = read_int(device / "cur_state")
            if value is not None:
                processor_cooling.append(value)
    return {"acpi_millic": acpi, "nvme_millic": nvme,
            "processor_cooling": processor_cooling,
            "acpi_max_millic": max(acpi) if acpi else None,
            "nvme_max_millic": max(nvme) if nvme else None}


def snapshot(cpus: list[int]) -> dict[str, Any]:
    return {
        "utc": utc_now(), "monotonic_ns": time.monotonic_ns(),
        "meminfo": meminfo(), "vmstat": vmstat(),
        "memory_pressure": read_text(Path("/proc/pressure/memory")),
        "cpuidle": cpuidle(cpus), "frequency": frequencies(cpus),
        "temperature": temperatures(),
    }


def pressure_avg10(text: str | None) -> float | None:
    m = re.search(r"some avg10=([0-9.]+)", text or "")
    return float(m.group(1)) if m else None


def ready(snap: dict[str, Any], cpu_limit: int, nvme_limit: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    temp = snap["temperature"]
    if temp["acpi_max_millic"] is not None and temp["acpi_max_millic"] > cpu_limit:
        reasons.append("cpu_temperature")
    if temp["nvme_max_millic"] is not None and temp["nvme_max_millic"] > nvme_limit:
        reasons.append("nvme_temperature")
    if any(v != 0 for v in temp["processor_cooling"]):
        reasons.append("processor_cooling")
    limited = [v.get("perf_limited") for v in snap["frequency"].values()]
    if any(v not in (None, 0) for v in limited):
        reasons.append("performance_limited")
    psi = pressure_avg10(snap["memory_pressure"])
    if psi is not None and psi > 0:
        reasons.append("memory_pressure")
    return not reasons, reasons


def wait_ready(cpus: list[int], timeout: int, cpu_limit: int,
               nvme_limit: int) -> tuple[dict[str, Any], list[str]]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    reasons: list[str] = []
    while True:
        last = snapshot(cpus)
        ok, reasons = ready(last, cpu_limit, nvme_limit)
        if ok or time.monotonic() >= deadline:
            return last, reasons
        time.sleep(5)


def descendants(pid: int) -> set[int]:
    found = {pid}
    changed = True
    while changed:
        changed = False
        for proc in Path("/proc").glob("[0-9]*"):
            status = read_text(proc / "status") or ""
            m = re.search(r"^PPid:\s+(\d+)", status, re.MULTILINE)
            if m and int(m.group(1)) in found and int(proc.name) not in found:
                found.add(int(proc.name))
                changed = True
    return found


def process_memory(root_pid: int) -> tuple[int, int]:
    peak_rss = peak_swap = 0
    for pid in descendants(root_pid):
        status = read_text(Path(f"/proc/{pid}/status")) or ""
        for key, target in (("VmRSS", "rss"), ("VmSwap", "swap")):
            m = re.search(rf"^{key}:\s+(\d+)", status, re.MULTILINE)
            if m:
                value = int(m.group(1)) * 1024
                if target == "rss":
                    peak_rss = max(peak_rss, value)
                else:
                    peak_swap = max(peak_swap, value)
    return peak_rss, peak_swap


def telemetry(path: Path, stop: threading.Event, cpus: list[int],
              root_pid: int, totals: dict[str, int]) -> None:
    with path.open("w") as f:
        while not stop.wait(1.0):
            rss, swap = process_memory(root_pid)
            totals["peak_rss_bytes"] = max(totals.get("peak_rss_bytes", 0), rss)
            totals["peak_swap_bytes"] = max(totals.get("peak_swap_bytes", 0), swap)
            row = {"utc": utc_now(), "monotonic_ns": time.monotonic_ns(),
                   "frequency": frequencies(cpus),
                   "temperature": temperatures(), "meminfo": meminfo(),
                   "memory_pressure": read_text(Path("/proc/pressure/memory")),
                   "process_rss_bytes": rss, "process_swap_bytes": swap}
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()


def trace_summary(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    by_event: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_event.setdefault(row.get("event", "unknown"), []).append(row)
    decoded = by_event.get("decode_token", [])
    layers = by_event.get("decode_layer", [])
    routes = [(r.get("position"), r.get("layer"), r.get("experts")) for r in layers]
    route_hash = hashlib.sha256(json.dumps(routes, separators=(",", ":")).encode()).hexdigest()
    fields = ("total_ms", "attention_ms", "router_ms", "expert_io_ms",
              "expert_compute_ms", "shared_compute_ms", "moe_total_ms",
              "other_ms", "cache_hits", "cache_misses", "bytes_read")
    totals = {field: sum(float(r.get(field, 0)) for r in decoded) for field in fields}
    meta = by_event.get("meta", [{}])[0]
    return {
        "event_counts": {key: len(value) for key, value in by_event.items()},
        "backend": meta.get("io_backend"), "queue_depth": meta.get("queue_depth"),
        "io_fallback": meta.get("io_fallback"), "route_sha256": route_hash,
        "decode": totals,
        "all_direct_io": all(r.get("direct_io", True) for r in rows),
        "any_read_error": any(r.get("read_error", False) for r in rows),
        "trace_start_monotonic_ns": meta.get("start_monotonic_ns"),
        "prefill_end_monotonic_ns": (by_event.get("prefill_chunk", [{}])[-1]
                                      .get("end_monotonic_ns")),
        "decode_end_monotonic_ns": (decoded[-1].get("end_monotonic_ns")
                                     if decoded else None),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("./waste"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--blocks", type=int, choices=range(1, 7), default=3)
    ap.add_argument("--start-block", type=int, choices=range(1, 7), default=1,
                    help="resume at this 1-based Williams block")
    ap.add_argument("--cpus", default="5-9,15-19")
    ap.add_argument("--sampler-cpus", default="0-4,10-14")
    ap.add_argument("--user", default=getpass.getuser())
    ap.add_argument("--cooldown-timeout", type=int, default=600)
    ap.add_argument("--cpu-temp-limit", type=int, default=52000)
    ap.add_argument("--nvme-temp-limit", type=int, default=55000)
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    if args.start_block > args.blocks:
        ap.error("--start-block cannot exceed --blocks")

    if os.uname().sysname != "Linux":
        ap.error("this diagnostic requires Linux perf and sysfs")
    model, binary, out = args.model.resolve(), args.binary.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    perf_cpus = parse_cpu_list(args.cpus)
    sampler_cpus = parse_cpu_list(args.sampler_cpus)
    try:
        os.sched_setaffinity(0, sampler_cpus)
    except (AttributeError, OSError):
        pass

    manifest = model / "manifest.json"
    usage = model / "usage.waste"
    identity = {"binary_sha256": sha256(binary),
                "manifest_sha256": sha256(manifest),
                "usage_sha256": sha256(usage)}
    campaign = {
        "schema": "waste.io_compute_campaign.v1", "created_utc": utc_now(),
        "host": socket.gethostname(), "kernel": os.uname().release,
        "model": str(model), "binary": str(binary), "out": str(out),
        "budget_bytes": args.budget, "threads": args.threads,
        "tokens": args.tokens, "blocks": args.blocks, "orders": ORDERS[:args.blocks],
        "performance_cpus": args.cpus, "sampler_cpus": args.sampler_cpus,
        "treatments": TREATMENTS, **identity,
    }
    (out / "campaign.json").write_text(json.dumps(campaign, indent=2,
                                                    sort_keys=True) + "\n")

    prompt = "Write a C function that parses a JSON array of integers and explain its edge cases."
    common = [str(binary), "run", str(model), prompt, "--raw", "-n", str(args.tokens),
              "--temp", "0", "--seed", "1", "--threads", str(args.threads),
              "--budget", str(args.budget), "--cpu-set", "performance"]

    if not args.skip_warmup:
        warm = out / "warmup"
        warm.mkdir(exist_ok=True)
        cmd = ["taskset", "-c", args.cpus, *common, "--io-backend", "pread",
               "--io-queue-depth", "1"]
        with (warm / "stdout.txt").open("w") as stdout, \
             (warm / "stderr.txt").open("w") as stderr:
            warm_rc = subprocess.run(cmd, stdout=stdout, stderr=stderr).returncode
        if warm_rc:
            return warm_rc

    events = [
        "{armv8_pmuv3_1/cpu_cycles/u,armv8_pmuv3_1/inst_retired/u,"
        "armv8_pmuv3_1/stall_backend/u,armv8_pmuv3_1/stall_backend_mem/u,"
        "armv8_pmuv3_1/l1d_cache_refill/u,armv8_pmuv3_1/l2d_cache_refill/u}",
        "task-clock:u,task-clock,context-switches,cpu-migrations,major-faults,minor-faults",
        "sched:sched_switch,sched:sched_wakeup,sched:sched_waking",
        "syscalls:sys_enter_futex,syscalls:sys_enter_pread64,"
        "syscalls:sys_enter_io_uring_enter",
    ]

    for block in range(args.start_block, args.blocks + 1):
        order = ORDERS[block - 1]
        for position, code in enumerate(order, 1):
            treatment = TREATMENTS[code]
            run_id = f"b{block:02d}-p{position}-{code}-{treatment['name']}"
            run_dir = out / run_id
            run_dir.mkdir(exist_ok=False)
            before, cooldown_reasons = wait_ready(
                perf_cpus, args.cooldown_timeout, args.cpu_temp_limit,
                args.nvme_temp_limit)
            (run_dir / "before.json").write_text(
                json.dumps(before, indent=2, sort_keys=True) + "\n")
            trace_path = run_dir / "trace.jsonl"
            time_path = run_dir / "time.txt"
            perf_path = run_dir / "perf.tsv"
            target = [*common, "--io-backend", treatment["backend"],
                      "--io-queue-depth", str(treatment["qd"]),
                      "--trace-layers", str(trace_path)]
            cmd = ["sudo", "-n", "perf", "stat", "--no-big-num", "-I", "500",
                   "--summary", "-x", "\t", "-o", str(perf_path)]
            for event in events:
                cmd += ["-e", event]
            cmd += ["--", "taskset", "-c", args.cpus, "sudo", "-n", "-u",
                    args.user, "-H", "--", "/usr/bin/time", "-v", "-o",
                    str(time_path), *target]

            started_utc, started_mono = utc_now(), time.monotonic_ns()
            totals: dict[str, int] = {}
            with (run_dir / "stdout.txt").open("w") as stdout, \
                 (run_dir / "stderr.txt").open("w") as stderr:
                proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
                stop = threading.Event()
                watcher = threading.Thread(
                    target=telemetry,
                    args=(run_dir / "telemetry.jsonl", stop, perf_cpus,
                          proc.pid, totals), daemon=True)
                watcher.start()
                try:
                    rc = proc.wait()
                except KeyboardInterrupt:
                    proc.send_signal(signal.SIGINT)
                    rc = proc.wait()
                    raise
                finally:
                    stop.set()
                    watcher.join(timeout=3)
            ended_mono = time.monotonic_ns()
            after = snapshot(perf_cpus)
            (run_dir / "after.json").write_text(
                json.dumps(after, indent=2, sort_keys=True) + "\n")

            reasons = list(cooldown_reasons)
            summary: dict[str, Any] | None = None
            if rc:
                reasons.append(f"returncode_{rc}")
            if trace_path.exists():
                try:
                    summary = trace_summary(trace_path)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    reasons.append(f"trace_parse:{type(exc).__name__}")
            else:
                reasons.append("trace_missing")
            expected_backend = {"A": "pread_sync", "B": "pread_batch",
                                "C": "io_uring"}[code]
            if summary:
                if summary["backend"] != expected_backend:
                    reasons.append("backend_mismatch")
                if summary["queue_depth"] != treatment["qd"]:
                    reasons.append("queue_depth_mismatch")
                if summary["io_fallback"]:
                    reasons.append("io_fallback")
                counts = summary["event_counts"]
                if counts.get("prefill_layer") != 92 or counts.get("decode_layer") != 368:
                    reasons.append("trace_layer_count")
                if counts.get("prefill_chunk") != 1 or counts.get("decode_token") != 4:
                    reasons.append("trace_total_count")
                if not summary["all_direct_io"]:
                    reasons.append("direct_io_disabled")
                if summary["any_read_error"]:
                    reasons.append("read_error")
            if sha256(usage) != identity["usage_sha256"]:
                reasons.append("usage_changed")
            vm_before, vm_after = before["vmstat"], after["vmstat"]
            vm_delta = {key: vm_after.get(key, 0) - vm_before.get(key, 0)
                        for key in set(vm_before) | set(vm_after)}
            if vm_delta.get("pswpout", 0) or totals.get("peak_swap_bytes", 0):
                reasons.append("paging")

            stdout_hash = sha256(run_dir / "stdout.txt")
            record = {
                "schema": "waste.io_compute_run.v1", "run_id": run_id,
                "block": block, "position": position, "treatment": code,
                "treatment_name": treatment["name"], "command": cmd,
                "started_utc": started_utc, "started_monotonic_ns": started_mono,
                "wall_seconds": (ended_mono - started_mono) / 1e9,
                "returncode": rc, "stdout_sha256": stdout_hash,
                "usage_sha256_after": sha256(usage), "trace": summary,
                "vm_delta": vm_delta, **totals,
                "start_temperature": before["temperature"],
                "end_temperature": after["temperature"],
                "valid": not reasons, "invalid_reasons": reasons,
            }
            (run_dir / "record.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n")
            append_jsonl(runs_path, record)
            print(json.dumps({"run_id": run_id, "valid": not reasons,
                              "reasons": reasons, "wall_seconds": record["wall_seconds"],
                              "decode": summary.get("decode") if summary else None},
                             sort_keys=True), flush=True)
            if rc:
                return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
