#!/usr/bin/env python3
"""Reproducible 2x2 Spark campaign for whole-expert scheduling.

Factors:

* routed compute: ``row`` versus ``whole`` (selected only through
  ``WASTE_EXPERT_SCHED``);
* CPU idle control: no PM-QoS request versus a zero-microsecond, FD-scoped
  request held by :mod:`pm_qos_exec`.

Every measured command otherwise uses the same K3 prompt, one decoded token,
io_uring QD4, direct I/O, budget, CPU placement, and deterministic sampling.
Four Williams-balanced blocks (16 runs) are the default.  Run directories are
durable and ``runs.jsonl`` is append-only, so rerunning the command resumes at
the first unrecorded treatment.

This tool reads sysfs telemetry but never writes sysfs.  The only power-policy
operation is the file-descriptor request to ``/dev/cpu_dma_latency``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Sequence

try:  # direct execution puts tools/, tests import through the repo root
    from io_compute_diagnostic import (
        frequencies, meminfo, process_memory, parse_cpu_list, read_text,
        snapshot, temperatures, wait_ready,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on invocation style
    from tools.io_compute_diagnostic import (
        frequencies, meminfo, process_memory, parse_cpu_list, read_text,
        snapshot, temperatures, wait_ready,
    )


TREATMENTS: dict[str, dict[str, Any]] = {
    "RN": {"schedule": "row", "qos_us": None,
           "name": "row_no_qos"},
    "WN": {"schedule": "whole", "qos_us": None,
           "name": "whole_no_qos"},
    "RQ": {"schedule": "row", "qos_us": 0,
           "name": "row_qos0"},
    "WQ": {"schedule": "whole", "qos_us": 0,
           "name": "whole_qos0"},
}

# Even-order Williams square: each treatment occurs once at every position,
# and every ordered first-period carryover pair occurs once across four blocks.
BASE_ORDERS = (
    ("RN", "WN", "WQ", "RQ"),
    ("WN", "RQ", "RN", "WQ"),
    ("RQ", "WQ", "WN", "RN"),
    ("WQ", "RN", "RQ", "WN"),
)

PROMPT = ("Write a C function that parses a JSON array of integers and "
          "explain its edge cases.")
TRACE_METRICS = (
    "total_ms", "attention_ms", "router_ms", "expert_io_ms",
    "expert_compute_ms", "shared_compute_ms", "moe_total_ms", "other_ms",
    "cache_hits", "cache_misses", "bytes_read",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_sha256(path: Path) -> str | None:
    try:
        return sha256(path)
    except OSError:
        return None


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"),
                         sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def orders(blocks: int) -> list[tuple[str, ...]]:
    if blocks <= 0 or blocks % len(BASE_ORDERS):
        raise ValueError("blocks must be a positive multiple of four")
    return [BASE_ORDERS[i % len(BASE_ORDERS)] for i in range(blocks)]


def model_inventory(model: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(model.iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
        rows.append({"name": path.name, "size": stat.st_size,
                     "mtime_ns": stat.st_mtime_ns,
                     "inode": stat.st_ino, "device": stat.st_dev})
    return {"files": rows, "sha256": canonical_sha256(rows)}


def model_geometry(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    config = manifest.get("config", {})
    outer = config.get("_outer", {}) if isinstance(config, dict) else {}
    architectures = (outer.get("architectures") or
                     config.get("architectures") or [])
    return {
        "architectures": architectures,
        "layers": int(config.get("num_hidden_layers", 0)),
        "moe_layers": len(manifest.get("layers", {})),
        "experts": int(config.get("num_experts", 0)),
        "top_k": int(config.get("num_experts_per_token", 0)),
        "hidden": int(config.get("hidden_size", 0)),
        "latent": int(config.get("routed_expert_hidden_size", 0)),
        "format_version": manifest.get("format_version"),
    }


def capture_identity(binary: Path, model: Path) -> dict[str, Any]:
    manifest = model / "manifest.json"
    codebooks = model / "codebooks.bin"
    usage = model / "usage.waste"
    library = next((binary.parent / name for name in
                    ("libwaste.so", "libwaste.dylib", "libwaste.dll")
                    if (binary.parent / name).exists()), None)
    return {
        "binary_sha256": optional_sha256(binary),
        "library_sha256": optional_sha256(library) if library else None,
        "manifest_sha256": optional_sha256(manifest),
        "codebooks_sha256": optional_sha256(codebooks),
        "usage_sha256": optional_sha256(usage),
        "model_inventory": model_inventory(model),
        "model_geometry": model_geometry(manifest),
    }


def parse_csv_rows(text: str, fields: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for values in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(values) != len(fields):
            continue
        rows.append({key: value.strip() for key, value in zip(fields, values)})
    return rows


GPU_FIELDS = ("index", "uuid", "name", "driver_version", "pstate",
              "temperature.gpu", "power.draw", "clocks.sm", "clocks.mem",
              "utilization.gpu", "memory.used")
GPU_APP_FIELDS = ("pid", "process_name", "used_memory")


def nvidia_query() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "error": "nvidia-smi not found",
                "gpus": [], "compute_apps": []}
    command = [executable, f"--query-gpu={','.join(GPU_FIELDS)}",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc), "gpus": [],
                "compute_apps": []}
    try:
        apps = subprocess.run(
            [executable, f"--query-compute-apps={','.join(GPU_APP_FIELDS)}",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": result.returncode == 0,
                "returncode": result.returncode,
                "stderr": result.stderr.strip(),
                "gpus": parse_csv_rows(result.stdout, GPU_FIELDS),
                "compute_apps": [], "apps_returncode": None,
                "apps_stderr": str(exc)}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
        "gpus": parse_csv_rows(result.stdout, GPU_FIELDS),
        "compute_apps": (parse_csv_rows(apps.stdout, GPU_APP_FIELDS)
                         if apps.returncode == 0 else []),
        "apps_returncode": apps.returncode,
        "apps_stderr": apps.stderr.strip(),
    }


def pressure_totals(value: str | None) -> dict[str, int]:
    totals: dict[str, int] = {}
    for line in (value or "").splitlines():
        fields = line.split()
        if not fields:
            continue
        for field in fields[1:]:
            key, separator, number = field.partition("=")
            if key == "total" and separator:
                try:
                    totals[fields[0]] = int(number)
                except ValueError:
                    pass
    return totals


def parse_feedback(value: str | None) -> tuple[int, int] | None:
    found: dict[str, int] = {}
    for part in (value or "").split():
        key, separator, number = part.partition(":")
        if separator:
            try:
                found[key] = int(number)
            except ValueError:
                return None
    if "ref" not in found or "del" not in found:
        return None
    return found["ref"], found["del"]


def cpuidle_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    elapsed = (int(after["monotonic_ns"]) - int(before["monotonic_ns"])) / 1e9
    cpus = sorted(set(before.get("cpuidle", {})) &
                  set(after.get("cpuidle", {})), key=int)
    states: dict[str, dict[str, int]] = {}
    for cpu in cpus:
        old = {row.get("name"): row for row in before["cpuidle"][cpu]}
        new = {row.get("name"): row for row in after["cpuidle"][cpu]}
        for name in set(old) & set(new):
            if not name:
                continue
            target = states.setdefault(str(name), {"usage_delta": 0,
                                                   "time_us_delta": 0})
            target["usage_delta"] += int(new[name].get("usage") or 0) - \
                                     int(old[name].get("usage") or 0)
            target["time_us_delta"] += int(new[name].get("time") or 0) - \
                                       int(old[name].get("time") or 0)
    return {"scope": "whole workload including model open/prefill/decode",
            "elapsed_seconds": elapsed, "cpu_count": len(cpus),
            "states": states}


def cppc_ratio(rows: list[dict[str, Any]]) -> float | None:
    reference = delivered = 0
    for old, new in zip(rows, rows[1:]):
        for cpu in set(old.get("frequency", {})) & set(new.get("frequency", {})):
            a = parse_feedback(old["frequency"][cpu].get("feedback_ctrs"))
            b = parse_feedback(new["frequency"][cpu].get("feedback_ctrs"))
            if a is None or b is None:
                continue
            dr, dd = b[0] - a[0], b[1] - a[1]
            if dr >= 0 and dd >= 0:
                reference += dr
                delivered += dd
    return delivered / reference if reference else None


def telemetry_sample(cpus: list[int], root_pid: int) -> dict[str, Any]:
    rss, swap = process_memory(root_pid)
    return {
        "utc": utc_now(), "monotonic_ns": time.monotonic_ns(),
        "frequency": frequencies(cpus), "temperature": temperatures(),
        "meminfo": meminfo(),
        "memory_pressure": read_text(Path("/proc/pressure/memory")),
        "process_rss_bytes": rss, "process_swap_bytes": swap,
        "gpu": nvidia_query(),
    }


def telemetry_loop(path: Path, stop: threading.Event, cpus: list[int],
                   root_pid: int, totals: dict[str, Any], interval: float) -> None:
    with path.open("w") as handle:
        while True:
            row = telemetry_sample(cpus, root_pid)
            totals["peak_rss_bytes"] = max(
                int(totals.get("peak_rss_bytes", 0)),
                int(row["process_rss_bytes"]))
            totals["peak_swap_bytes"] = max(
                int(totals.get("peak_swap_bytes", 0)),
                int(row["process_swap_bytes"]))
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            if stop.wait(interval):
                break


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def trace_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    by_event: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_event.setdefault(str(row.get("event", "unknown")), []).append(row)
    meta = by_event.get("meta", [{}])[0]
    layers = by_event.get("decode_layer", [])
    tokens = by_event.get("decode_token", [])
    routes = [(row.get("position"), row.get("layer"), row.get("experts"))
              for row in layers]
    traffic = [(row.get("position"), row.get("layer"),
                row.get("cache_hits"), row.get("cache_misses"),
                row.get("bytes_read")) for row in layers]
    signature = [(row.get("position"), row.get("layer"), row.get("experts"),
                  row.get("cache_hits"), row.get("cache_misses"),
                  row.get("bytes_read")) for row in layers]
    decoded = {metric: sum(float(row.get(metric, 0)) for row in tokens)
               for metric in TRACE_METRICS}
    decoded["expert_compute_work_ms"] = sum(
        float(row.get("expert_compute_work_ms", row.get("expert_compute_ms", 0)))
        for row in layers)
    timed_rows = [row for row in rows if row.get("event") in
                  ("prefill_layer", "prefill_chunk", "decode_layer", "decode_token")]
    return {
        "event_counts": {key: len(value) for key, value in by_event.items()},
        "backend": meta.get("io_backend"),
        "queue_depth": meta.get("queue_depth"),
        "io_fallback": meta.get("io_fallback"),
        "expert_schedule_requested": meta.get("expert_schedule_requested"),
        "expert_schedule_selected": meta.get("expert_schedule"),
        "decode_layer_schedules": sorted({row.get("expert_schedule")
                                           for row in layers}),
        "route_sha256": canonical_sha256(routes),
        "traffic_sha256": canonical_sha256(traffic),
        "layer_signature_sha256": canonical_sha256(signature),
        "decode": decoded,
        "all_direct_io": bool(timed_rows) and
                         all(row.get("direct_io") is True for row in timed_rows),
        "any_read_error": any(row.get("read_error") is True for row in timed_rows),
        "trace_start_monotonic_ns": meta.get("start_monotonic_ns"),
        "prefill_end_monotonic_ns": (by_event.get("prefill_chunk", [{}])[-1]
                                      .get("end_monotonic_ns")),
        "decode_end_monotonic_ns": (tokens[-1].get("end_monotonic_ns")
                                     if tokens else None),
    }


def target_command(*, binary: Path, model: Path, trace: Path, budget: int,
                   threads: int, cpus: str, schedule: str,
                   prompt: str = PROMPT) -> list[str]:
    if schedule not in ("row", "whole"):
        raise ValueError("schedule must be row or whole")
    return [
        "taskset", "-c", cpus, "/usr/bin/env",
        "-u", "WASTE_DUMP_HIDDEN", "-u", "WASTE_DUMP_LATENT",
        f"WASTE_EXPERT_SCHED={schedule}", "WASTE_DIRECT=1",
        "WASTE_BACKEND=auto", "WASTE_PROFILE=0", "WASTE_Q8=1",
        "WASTE_SDOT=0", "WASTE_I8MM=0", "WASTE_VERIFY=0",
        "WASTE_ALLOW_CONCURRENT=0", "WASTE_IO_BACKEND=io_uring",
        str(binary), "run", str(model), prompt, "--raw", "-n", "1",
        "--temp", "0", "--seed", "1", "--threads", str(threads),
        "--budget", str(budget), "--cpu-set", "performance",
        "--io-backend", "io_uring", "--io-queue-depth", "4",
        "--trace-layers", str(trace),
    ]


def wrapped_command(target: list[str], *, treatment: dict[str, Any],
                    user: str, qos_helper: Path, qos_status: Path,
                    python: Path) -> list[str]:
    if treatment["qos_us"] is None:
        return ["sudo", "-n", "-u", user, "-H", "--", *target]
    return ["sudo", "-n", str(python), str(qos_helper),
            "--latency-us", str(treatment["qos_us"]),
            "--status", str(qos_status), "--user", user, "--", *target]


def gpu_summary(before: dict[str, Any], after: dict[str, Any],
                samples: list[dict[str, Any]]) -> dict[str, Any]:
    gpu_rows = [before.get("gpu", {}),
                *[row.get("gpu", {}) for row in samples], after.get("gpu", {})]
    utilization: list[float] = []
    apps: list[dict[str, str]] = []
    for gpu in gpu_rows:
        apps.extend(gpu.get("compute_apps", []))
        for row in gpu.get("gpus", []):
            try:
                utilization.append(float(row.get("utilization.gpu", "")))
            except ValueError:
                pass
    return {"available_all_samples": bool(gpu_rows) and
            all(row.get("available") is True for row in gpu_rows),
            "max_utilization_percent": max(utilization, default=None),
            "compute_apps": apps}


def run_telemetry_summary(before: dict[str, Any], after: dict[str, Any],
                          samples: list[dict[str, Any]], totals: dict[str, Any]) -> dict[str, Any]:
    pressures_before = pressure_totals(before.get("memory_pressure"))
    pressures_after = pressure_totals(after.get("memory_pressure"))
    temps = [before.get("temperature", {}),
             *[row.get("temperature", {}) for row in samples],
             after.get("temperature", {})]
    acpi = [int(row["acpi_max_millic"]) for row in temps
            if row.get("acpi_max_millic") is not None]
    nvme = [int(row["nvme_max_millic"]) for row in temps
            if row.get("nvme_max_millic") is not None]
    cooling = [int(state) for row in temps
               for state in row.get("processor_cooling", [])]
    available = [int(row["meminfo"]["MemAvailable"]) for row in samples
                 if row.get("meminfo", {}).get("MemAvailable") is not None]
    limited = [value for row in samples
               for value in (entry.get("perf_limited")
                             for entry in row.get("frequency", {}).values())
               if value is not None]
    return {
        "peak_rss_bytes": int(totals.get("peak_rss_bytes", 0)),
        "peak_swap_bytes": int(totals.get("peak_swap_bytes", 0)),
        "min_memavailable_bytes": min(available, default=None),
        "memory_psi_some_total_delta_us":
            pressures_after.get("some", 0) - pressures_before.get("some", 0),
        "memory_psi_full_total_delta_us":
            pressures_after.get("full", 0) - pressures_before.get("full", 0),
        "cpu_peak_millic": max(acpi, default=None),
        "nvme_peak_millic": max(nvme, default=None),
        "max_processor_cooling_state": max(cooling, default=0),
        "any_cppc_perf_limited": any(value != 0 for value in limited),
        "cppc_delivered_reference_ratio": cppc_ratio(samples),
        "cpuidle": cpuidle_summary(before, after),
        "gpu": gpu_summary(before, after, samples),
    }


def exact_run_reasons(*, record: dict[str, Any], expected: dict[str, Any],
                      geometry: dict[str, Any], gpu_limit: float,
                      cpu_temp_limit: int, nvme_temp_limit: int) -> list[str]:
    reasons = list(record.get("cooldown_reasons", []))
    trace = record.get("trace") or {}
    treatment = TREATMENTS[record["treatment"]]
    expected_schedule = treatment["schedule"]
    if record.get("returncode") != 0:
        reasons.append(f"returncode_{record.get('returncode')}")
    if trace.get("backend") != "io_uring": reasons.append("backend_mismatch")
    if trace.get("queue_depth") != 4: reasons.append("queue_depth_mismatch")
    if trace.get("io_fallback") is not False: reasons.append("io_fallback")
    if trace.get("expert_schedule_requested") != expected_schedule:
        reasons.append("schedule_request_mismatch")
    if trace.get("expert_schedule_selected") != expected_schedule:
        reasons.append("schedule_selection_mismatch")
    if trace.get("decode_layer_schedules") != [expected_schedule]:
        reasons.append("layer_schedule_mismatch")
    counts = trace.get("event_counts", {})
    if counts.get("decode_token") != 1: reasons.append("decode_token_count")
    if counts.get("decode_layer") != geometry.get("moe_layers"):
        reasons.append("decode_layer_count")
    if counts.get("prefill_layer") != geometry.get("moe_layers"):
        reasons.append("prefill_layer_count")
    if not trace.get("all_direct_io"): reasons.append("direct_io_disabled")
    if trace.get("any_read_error"): reasons.append("read_error")
    if record.get("identity_after") != expected: reasons.append("identity_changed")

    qos = record.get("pm_qos", {})
    if treatment["qos_us"] is None:
        if qos != {"enabled": False}: reasons.append("unexpected_pm_qos")
    else:
        required = (qos.get("holder_euid") == 0 and qos.get("latency_us") == 0
                    and qos.get("fd_opened") is True
                    and qos.get("fd_closed") is True
                    and qos.get("fd_scoped") is True
                    and qos.get("self_cleaning") is True
                    and qos.get("sysfs_modified") is False
                    and qos.get("child_returncode") == 0)
        if not required: reasons.append("pm_qos_evidence")

    vm = record.get("vm_delta", {})
    telemetry = record.get("telemetry", {})
    if int(vm.get("pswpin", 0)) or int(vm.get("pswpout", 0)):
        reasons.append("system_swap_io")
    if int(telemetry.get("peak_swap_bytes", 0)):
        reasons.append("process_swap")
    if int(telemetry.get("memory_psi_full_total_delta_us", 0)):
        reasons.append("memory_psi_full")
    if telemetry.get("max_processor_cooling_state", 0) != 0:
        reasons.append("processor_cooling")
    if telemetry.get("any_cppc_perf_limited") is True:
        reasons.append("cppc_perf_limited")
    if (telemetry.get("cpu_peak_millic") is not None and
            telemetry["cpu_peak_millic"] > cpu_temp_limit):
        reasons.append("cpu_temperature")
    if (telemetry.get("nvme_peak_millic") is not None and
            telemetry["nvme_peak_millic"] > nvme_temp_limit):
        reasons.append("nvme_temperature")
    gpu = telemetry.get("gpu", {})
    if gpu.get("available_all_samples") is not True:
        reasons.append("gpu_telemetry_missing")
    if gpu.get("compute_apps"):
        reasons.append("gpu_compute_process")
    if (gpu.get("max_utilization_percent") is None or
            float(gpu["max_utilization_percent"]) > gpu_limit):
        reasons.append("gpu_utilization")
    return sorted(set(reasons))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("./waste"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=4,
                    help="positive multiple of 4 (default: 4)")
    ap.add_argument("--cpus", default="5-9,15-19")
    ap.add_argument("--sampler-cpus", default="0-4,10-14")
    ap.add_argument("--user", default=os.environ.get("SUDO_USER") or
                    os.environ.get("USER") or "nvidia")
    ap.add_argument("--root-python", type=Path,
                    default=Path("/usr/bin/python3"),
                    help="system Python used by sudo for the QoS holder")
    ap.add_argument("--telemetry-interval", type=float, default=1.0)
    ap.add_argument("--cooldown-timeout", type=int, default=600)
    ap.add_argument("--cpu-temp-limit", type=int, default=52000)
    ap.add_argument("--nvme-temp-limit", type=int, default=55000)
    ap.add_argument("--gpu-util-limit", type=float, default=5.0)
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--retry-invalid", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the balanced commands; do not create or run")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = parser()
    args = ap.parse_args(argv)
    try:
        campaign_orders = orders(args.blocks)
    except ValueError as exc:
        ap.error(str(exc))
    if args.telemetry_interval <= 0 or args.gpu_util_limit < 0:
        ap.error("telemetry interval must be positive and GPU limit nonnegative")

    model, binary, out = args.model.resolve(), args.binary.resolve(), args.out.resolve()
    helper = Path(__file__).with_name("pm_qos_exec.py").resolve()
    if args.dry_run:
        for block, order in enumerate(campaign_orders, 1):
            for position, code in enumerate(order, 1):
                treatment = TREATMENTS[code]
                run_id = f"b{block:02d}-p{position}-{code}-{treatment['name']}"
                target = target_command(binary=binary, model=model,
                                        trace=out / run_id / "trace.jsonl",
                                        budget=args.budget, threads=args.threads,
                                        cpus=args.cpus,
                                        schedule=treatment["schedule"])
                command = wrapped_command(
                    target, treatment=treatment, user=args.user,
                    qos_helper=helper, qos_status=out / run_id / "pm_qos.json",
                    python=args.root_python)
                print(json.dumps({"block": block, "position": position,
                                  "treatment": code, "command": command}))
        return 0

    if not sys.platform.startswith("linux"):
        ap.error("the Spark campaign requires Linux")
    if not binary.is_file() or not (model / "manifest.json").is_file():
        ap.error("binary or model manifest is missing")
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"

    identity = capture_identity(binary, model)
    geometry = identity["model_geometry"]
    if not any("KimiK3" in str(name) for name in geometry["architectures"]):
        ap.error(f"campaign requires K3; architectures={geometry['architectures']}")
    version = subprocess.run([str(binary), "version"], capture_output=True,
                             text=True, check=False)
    campaign = {
        "schema": "waste.spark_whole_expert_campaign.v1",
        "created_utc": utc_now(), "host": socket.gethostname(),
        "kernel": os.uname().release, "model": str(model),
        "binary": str(binary), "out": str(out), "budget_bytes": args.budget,
        "threads": args.threads, "decoded_tokens": 1, "blocks": args.blocks,
        "orders": [list(order) for order in campaign_orders],
        "treatments": TREATMENTS,
        "performance_cpus": args.cpus, "sampler_cpus": args.sampler_cpus,
        "prompt": PROMPT, "prompt_sha256": canonical_sha256(PROMPT),
        "io_backend": "io_uring", "io_queue_depth": 4, "direct_io": True,
        "identity": identity, "binary_version_stdout": version.stdout.strip(),
        "binary_version_stderr": version.stderr.strip(),
        "qos_helper_sha256": sha256(helper),
        "limits": {"cpu_temp_millic": args.cpu_temp_limit,
                   "nvme_temp_millic": args.nvme_temp_limit,
                   "gpu_util_percent": args.gpu_util_limit},
        "gpu_identity": nvidia_query(),
    }
    campaign_path = out / "campaign.json"
    if campaign_path.exists():
        previous = json.loads(campaign_path.read_text())
        comparable = {key: value for key, value in campaign.items()
                      if key not in ("created_utc", "gpu_identity")}
        old = {key: value for key, value in previous.items()
               if key not in ("created_utc", "gpu_identity")}
        if comparable != old:
            raise SystemExit("existing campaign.json does not match this run")
        campaign = previous
    else:
        atomic_json(campaign_path, campaign)

    sampler = parse_cpu_list(args.sampler_cpus)
    perf_cpus = parse_cpu_list(args.cpus)
    try:
        os.sched_setaffinity(0, sampler)
    except (AttributeError, OSError):
        pass

    latest = {row["run_id"]: row for row in read_jsonl(runs_path)}
    if not args.skip_warmup and not (out / "warmup.json").exists():
        warm_trace = out / "warmup.trace.jsonl"
        warm_target = target_command(
            binary=binary, model=model, trace=warm_trace, budget=args.budget,
            threads=args.threads, cpus=args.cpus, schedule="row")
        warm_cmd = ["sudo", "-n", "-u", args.user, "-H", "--", *warm_target]
        warm_rc = subprocess.run(warm_cmd, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL).returncode
        atomic_json(out / "warmup.json", {"command": warm_cmd,
                                           "returncode": warm_rc})
        if warm_rc:
            return warm_rc

    for block, order in enumerate(campaign_orders, 1):
        for position, code in enumerate(order, 1):
            treatment = TREATMENTS[code]
            run_id = f"b{block:02d}-p{position}-{code}-{treatment['name']}"
            old = latest.get(run_id)
            if old and (old.get("valid") or not args.retry_invalid):
                print(json.dumps({"run_id": run_id, "resume": "skipped",
                                  "valid": old.get("valid")}), flush=True)
                continue
            run_dir = out / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            before, cooldown_reasons = wait_ready(
                perf_cpus, args.cooldown_timeout, args.cpu_temp_limit,
                args.nvme_temp_limit)
            before["gpu"] = nvidia_query()
            atomic_json(run_dir / "before.json", before)

            trace_path = run_dir / "trace.jsonl"
            qos_path = run_dir / "pm_qos.json"
            if treatment["qos_us"] is None:
                atomic_json(qos_path, {"enabled": False})
            target = target_command(
                binary=binary, model=model, trace=trace_path,
                budget=args.budget, threads=args.threads, cpus=args.cpus,
                schedule=treatment["schedule"])
            command = wrapped_command(
                target, treatment=treatment, user=args.user,
                qos_helper=helper, qos_status=qos_path,
                python=args.root_python)
            started_utc, started_ns = utc_now(), time.monotonic_ns()
            totals: dict[str, Any] = {}
            with (run_dir / "stdout.txt").open("w") as stdout, \
                 (run_dir / "stderr.txt").open("w") as stderr:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
                stop = threading.Event()
                watcher = threading.Thread(
                    target=telemetry_loop,
                    args=(run_dir / "telemetry.jsonl", stop, perf_cpus,
                          process.pid, totals, args.telemetry_interval),
                    daemon=True)
                watcher.start()
                try:
                    returncode = process.wait()
                except KeyboardInterrupt:
                    process.send_signal(signal.SIGINT)
                    returncode = process.wait()
                    raise
                finally:
                    stop.set()
                    watcher.join(timeout=max(3.0, args.telemetry_interval * 2))
            ended_ns = time.monotonic_ns()
            after = snapshot(perf_cpus)
            after["gpu"] = nvidia_query()
            atomic_json(run_dir / "after.json", after)
            samples = read_jsonl(run_dir / "telemetry.jsonl")
            telemetry = run_telemetry_summary(before, after, samples, totals)
            before_vm, after_vm = before["vmstat"], after["vmstat"]
            vm_delta = {key: int(after_vm.get(key, 0)) - int(before_vm.get(key, 0))
                        for key in set(before_vm) | set(after_vm)}
            try:
                trace = trace_summary(trace_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                trace = {"parse_error": f"{type(exc).__name__}: {exc}"}
            try:
                qos = json.loads(qos_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                qos = {"parse_error": f"{type(exc).__name__}: {exc}"}
            identity_after = capture_identity(binary, model)
            record = {
                "schema": "waste.spark_whole_expert_run.v1",
                "run_id": run_id, "block": block, "position": position,
                "treatment": code, "treatment_name": treatment["name"],
                "schedule": treatment["schedule"], "qos_us": treatment["qos_us"],
                "command": command, "started_utc": started_utc,
                "started_monotonic_ns": started_ns,
                "ended_monotonic_ns": ended_ns,
                "wall_seconds": (ended_ns - started_ns) / 1e9,
                "returncode": returncode,
                "stdout_sha256": optional_sha256(run_dir / "stdout.txt"),
                "stderr_sha256": optional_sha256(run_dir / "stderr.txt"),
                "trace": trace, "pm_qos": qos, "telemetry": telemetry,
                "vm_delta": vm_delta, "identity_after": identity_after,
                "cooldown_reasons": cooldown_reasons,
            }
            reasons = exact_run_reasons(
                record=record, expected=identity, geometry=geometry,
                gpu_limit=args.gpu_util_limit,
                cpu_temp_limit=args.cpu_temp_limit,
                nvme_temp_limit=args.nvme_temp_limit)
            record.update({"valid": not reasons, "invalid_reasons": reasons})
            atomic_json(run_dir / "record.json", record)
            append_jsonl(runs_path, record)
            latest[run_id] = record
            print(json.dumps({"run_id": run_id, "valid": not reasons,
                              "reasons": reasons,
                              "decode": trace.get("decode")}), flush=True)
            if returncode:
                return returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
