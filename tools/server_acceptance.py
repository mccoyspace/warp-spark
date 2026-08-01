#!/usr/bin/env python3
"""Measure repeated requests to one persistent WASTE server process."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


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


def delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {k: after.get(k, 0) - before.get(k, 0)
            for k in set(before) | set(after)}


def mem_available() -> int | None:
    value = key_values("/proc/meminfo").get("MemAvailable")
    return value * 1024 if value is not None else None


def proc_memory(pid: int) -> tuple[int | None, int | None]:
    values = key_values(f"/proc/{pid}/status")
    rss, swap = values.get("VmRSS"), values.get("VmSwap")
    return (rss * 1024 if rss is not None else None,
            swap * 1024 if swap is not None else None)


def request_json(url: str, body: dict | None = None,
                 timeout: float = 30.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--binary", type=Path, default=Path("./waste"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cpu-set", default="performance")
    ap.add_argument("--io-backend",
                    choices=("pread", "pread_batch", "io_uring"),
                    default="pread")
    ap.add_argument("--io-queue-depth", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--requests", type=int, default=2)
    ap.add_argument("--prompt", default="Explain NVMe clearly.")
    ap.add_argument("--startup-timeout", type=float, default=600.0)
    ap.add_argument("--request-timeout", type=float, default=900.0)
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    repo, model = args.repo.resolve(), args.model.resolve()
    binary = args.binary.resolve()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = args.artifacts / f"{args.label}-server-stdout.txt"
    stderr_path = args.artifacts / f"{args.label}-server-stderr.txt"
    trace_path = args.artifacts / f"{args.label}-layers.jsonl"
    cmd = [sys.executable, "-m", "serve", str(model),
           "--host", "127.0.0.1", "--port", str(args.port),
           "--model-id", args.label, "--budget", str(args.budget),
           "--threads", str(args.threads), "--cpu-set", args.cpu_set,
           "--io-backend", args.io_backend,
           "--io-queue-depth", str(args.io_queue_depth),
           "--max-tokens", str(args.tokens), "--no-thinking"]
    if args.trace:
        cmd += ["--trace-layers", str(trace_path)]

    before_vm, before_psi = vmstat(), pressure()
    peak_rss = peak_swap = 0
    min_available: int | None = None
    monitoring = threading.Event()
    started_ns = time.time_ns()
    stdout_file = stdout_path.open("w")
    stderr_file = stderr_path.open("w")
    proc = subprocess.Popen(cmd, cwd=repo, stdout=stdout_file,
                            stderr=stderr_file, text=True)

    def monitor() -> None:
        nonlocal peak_rss, peak_swap, min_available
        while not monitoring.wait(0.5):
            rss, swap = proc_memory(proc.pid)
            if rss is not None:
                peak_rss = max(peak_rss, rss)
            if swap is not None:
                peak_swap = max(peak_swap, swap)
            available = mem_available()
            if available is not None:
                min_available = (available if min_available is None else
                                 min(min_available, available))

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    ready_ns: int | None = None
    responses: list[dict] = []
    lock_probe: dict = {}
    error: str | None = None
    base = f"http://127.0.0.1:{args.port}"
    try:
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with {proc.returncode}")
            try:
                request_json(base + "/health", timeout=2.0)
                ready_ns = time.time_ns()
                break
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(1.0)
        if ready_ns is None:
            raise RuntimeError("server startup timed out")

        probe_started = time.time_ns()
        probe = subprocess.run(
            [str(binary), "info", str(model), "--json"],
            text=True, capture_output=True, timeout=30.0)
        lock_probe = {
            "returncode": probe.returncode,
            "wall_seconds": (time.time_ns() - probe_started) / 1e9,
            "stdout": probe.stdout,
            "stderr": probe.stderr,
        }

        body = {
            "model": args.label,
            "messages": [{"role": "user", "content": args.prompt}],
            "temperature": 0,
            "seed": 1,
            "max_tokens": args.tokens,
            "stream": False,
            "reasoning_effort": "none",
        }
        for index in range(1, args.requests + 1):
            request_started = time.time_ns()
            response = request_json(base + "/v1/chat/completions", body,
                                    timeout=args.request_timeout)
            elapsed = (time.time_ns() - request_started) / 1e9
            response_path = args.artifacts / f"{args.label}-response-{index}.json"
            response_path.write_text(json.dumps(response, indent=2) + "\n")
            responses.append({"index": index, "wall_seconds": elapsed,
                              "response_path": str(response_path),
                              "usage": response.get("usage"),
                              "waste": response.get("waste")})
    except Exception as exc:  # preserve partial evidence before returning
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        monitoring.set()
        watcher.join(timeout=2)
        stdout_file.close()
        stderr_file.close()

    stderr_text = stderr_path.read_text(errors="replace")
    cpu_match = re.search(r"^CPU set\s+(.+)$", stderr_text, re.MULTILINE)
    if cpu_match is None:
        cpu_match = re.search(r"^CPU set\s+(.+)$",
                              stdout_path.read_text(errors="replace"),
                              re.MULTILINE)
    ended_ns = time.time_ns()
    record = {
        "schema": "waste.server_acceptance.v1",
        "run_id": args.label,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "command": cmd,
        "server_returncode": proc.returncode,
        "error": error,
        "startup_seconds": ((ready_ns - started_ns) / 1e9
                            if ready_ns is not None else None),
        "total_wall_seconds": (ended_ns - started_ns) / 1e9,
        "binary_sha256": sha256(binary),
        "library_sha256": sha256(repo / "libwaste.so"),
        "manifest_sha256": sha256(model / "manifest.json"),
        "usage_sha256": sha256(model / "usage.waste"),
        "budget_bytes": args.budget,
        "threads": args.threads,
        "cpu_set": args.cpu_set,
        "resolved_cpu_set": cpu_match.group(1) if cpu_match else None,
        "io_backend_requested": args.io_backend,
        "io_queue_depth_requested": args.io_queue_depth,
        "tokens_requested": args.tokens,
        "requests": responses,
        "lock_probe": lock_probe,
        "peak_rss_bytes": peak_rss or None,
        "peak_process_swap_bytes": peak_swap,
        "min_mem_available_bytes": min_available,
        "vm_delta": delta(vmstat(), before_vm),
        "memory_psi_total_us_delta": delta(pressure(), before_psi),
        "trace_path": str(trace_path) if args.trace else None,
    }
    with args.out.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(record, sort_keys=True))
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
