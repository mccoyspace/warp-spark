#!/usr/bin/env python3
"""Run a WASTE acceptance workload and append one self-describing JSONL row.

Linux adds per-process RSS/swap, MemAvailable, VM counters and pressure-stall
deltas. The command output is preserved beside the JSONL so a summary never
becomes the only copy of a result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time


VM_KEYS = ("pswpin", "pswpout", "pgmajfault", "pgscan_kswapd",
           "pgscan_direct", "pgsteal_kswapd", "pgsteal_direct")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def key_values(path: str) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path(path).read_text().splitlines():
            fields = line.replace(":", "").split()
            if len(fields) >= 2 and fields[1].isdigit():
                out[fields[0]] = int(fields[1])
    except OSError:
        pass
    return out


def vmstat() -> dict[str, int]:
    values = key_values("/proc/vmstat")
    return {k: values.get(k, 0) for k in VM_KEYS}


def pressure() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/pressure/memory").read_text().splitlines():
            fields = line.split()
            total = next((x for x in fields if x.startswith("total=")), None)
            if total:
                out[fields[0]] = int(total.split("=", 1)[1])
    except OSError:
        pass
    return out


def diff(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {k: after.get(k, 0) - before.get(k, 0)
            for k in set(before) | set(after)}


def proc_sample(pid: int) -> tuple[int | None, int | None]:
    values = key_values(f"/proc/{pid}/status")
    rss = values.get("VmRSS")
    swap = values.get("VmSwap")
    return (rss * 1024 if rss is not None else None,
            swap * 1024 if swap is not None else None)


def mem_available() -> int | None:
    values = key_values("/proc/meminfo")
    v = values.get("MemAvailable")
    return v * 1024 if v is not None else None


def parse_result(workload: str, stdout: str, stderr: str) -> dict:
    if workload == "bench":
        for line in reversed(stdout.splitlines()):
            if line.lstrip().startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return {}
    m = re.search(
        r"\[(\d+) tokens, ([0-9.]+) s, ([0-9.]+) tok/s \| "
        r"experts (\d+) hit / (\d+) miss = (\d+)%\]", stderr)
    if not m:
        return {}
    tokens = int(m[1])
    decode_seconds = float(m[2])
    return {"tokens": tokens, "decode_seconds": decode_seconds,
            "tok_per_s": float(m[3]),
            "tok_per_s_from_reported_seconds":
                tokens / decode_seconds if decode_seconds else None,
            "experts_hit": int(m[4]), "experts_missed": int(m[5]),
            "hit_rate_rounded_pct": int(m[6])}


def resolved_cpu_set(stderr: str) -> str | None:
    m = re.search(r"^waste: CPU set (.+)$", stderr, re.MULTILINE)
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("./waste"))
    ap.add_argument("--out", type=Path, required=True,
                    help="append-only JSONL result file")
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--workload", choices=("bench", "run"), default="bench")
    ap.add_argument("--prompt", default="Explain NVMe clearly.")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--repetitions", type=int, default=2)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cpu-set", default="performance")
    ap.add_argument("--io-backend",
                    choices=("pread", "pread_batch", "io_uring"),
                    default="pread")
    ap.add_argument("--io-queue-depth", type=int, default=1)
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    model = args.model.resolve()

    for rep in range(1, args.repetitions + 1):
        run_id = f"{args.label}-r{rep}"
        cmd = [str(binary), args.workload, str(model)]
        if args.workload == "run":
            cmd.append(args.prompt)
            cmd += ["--temp", "0"]
        cmd += ["-n", str(args.tokens), "--budget", str(args.budget),
                "--threads", str(args.threads), "--cpu-set", args.cpu_set,
                "--io-backend", args.io_backend,
                "--io-queue-depth", str(args.io_queue_depth)]
        if args.workload == "bench":
            # Bench already makes its generated text quiet.  Do not pass the
            # CLI-wide quiet flag: it would also hide the resolved CPU set,
            # which is part of the acceptance evidence.
            cmd += ["--json"]
        trace_path = args.artifacts / f"{run_id}-layers.jsonl"
        if args.trace:
            cmd += ["--trace-layers", str(trace_path)]

        before_vm, before_psi = vmstat(), pressure()
        started = time.time_ns()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        peak_rss = peak_swap = 0
        min_avail: int | None = None
        while proc.poll() is None:
            rss, swap = proc_sample(proc.pid)
            if rss is not None:
                peak_rss = max(peak_rss, rss)
            if swap is not None:
                peak_swap = max(peak_swap, swap)
            available = mem_available()
            if available is not None:
                min_avail = available if min_avail is None else min(min_avail, available)
            time.sleep(0.5)
        stdout, stderr = proc.communicate()
        ended = time.time_ns()
        (args.artifacts / f"{run_id}-stdout.txt").write_text(stdout)
        (args.artifacts / f"{run_id}-stderr.txt").write_text(stderr)

        record = {
            "schema": "waste.acceptance.v1",
            "run_id": run_id,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "host": socket.gethostname(),
            "workload": args.workload,
            "command": cmd,
            "returncode": proc.returncode,
            "wall_seconds": (ended - started) / 1e9,
            "binary_sha256": sha256(binary),
            "manifest_sha256": sha256(model / "manifest.json"),
            "usage_sha256": sha256(model / "usage.waste"),
            "budget_bytes": args.budget,
            "threads": args.threads,
            "cpu_set": args.cpu_set,
            "resolved_cpu_set": resolved_cpu_set(stderr),
            "io_backend_requested": args.io_backend,
            "io_queue_depth_requested": args.io_queue_depth,
            "tokens_requested": args.tokens,
            "peak_rss_bytes": peak_rss or None,
            "peak_process_swap_bytes": peak_swap,
            "min_mem_available_bytes": min_avail,
            "vm_delta": diff(vmstat(), before_vm),
            "memory_psi_total_us_delta": diff(pressure(), before_psi),
            "trace_path": str(trace_path) if args.trace else None,
            "result": parse_result(args.workload, stdout, stderr),
        }
        with args.out.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps(record, sort_keys=True))
        if proc.returncode:
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
