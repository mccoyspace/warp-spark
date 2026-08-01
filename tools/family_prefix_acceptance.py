#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Run the compact matched real-K3 semantic-family prefix acceptance.

The control server receives prompt A and then prompt B with caching disabled.
A separate server with the same total RAM budget receives the same A/B pair
with prefix caching, then prompt B with a changed tool declaration.  Both
servers are direct-I/O io_uring QD4.  Raw responses, per-
request layer traces, server logs and Linux memory telemetry are retained.

This is intentionally a one-shot acceptance campaign, not a throughput sweep.
Two generated tokens per request keep the five-request Spark run compact.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request


VM_KEYS = ("pswpin", "pswpout", "pgmajfault", "pgscan_kswapd",
           "pgscan_direct", "pgsteal_kswapd", "pgsteal_direct")
COUNTER_KEYS = ("tokens_generated", "experts_hit", "experts_missed",
                "bytes_read")
ENGINE_ENV = {
    "WASTE_EXPERT_SCHED": "row",
    "WASTE_PROFILE": "0",
    "WASTE_Q8": "1",
    "WASTE_SDOT": "0",
    "WASTE_I8MM": "0",
    "WASTE_VERIFY": "0",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_text(path: Path) -> str | None:
    try:
        return path.read_text()
    except (OSError, UnicodeError):
        return None


def key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in (read_text(path) or "").splitlines():
        fields = line.replace(":", "").split()
        if len(fields) >= 2:
            try:
                values[fields[0]] = int(fields[1])
            except ValueError:
                pass
    return values


def vmstat() -> dict[str, int]:
    values = key_values(Path("/proc/vmstat"))
    return {key: values.get(key, 0) for key in VM_KEYS}


def pressure() -> dict[str, int]:
    totals: dict[str, int] = {}
    for line in (read_text(Path("/proc/pressure/memory")) or "").splitlines():
        fields = line.split()
        total = next((field for field in fields
                      if field.startswith("total=")), None)
        if fields and total:
            try:
                totals[fields[0]] = int(total.split("=", 1)[1])
            except ValueError:
                pass
    return totals


def mem_available() -> int | None:
    value = key_values(Path("/proc/meminfo")).get("MemAvailable")
    return value * 1024 if value is not None else None


def process_memory(pid: int) -> tuple[int | None, int | None, int | None]:
    values = key_values(Path(f"/proc/{pid}/status"))
    rss, high, swap = (values.get("VmRSS"), values.get("VmHWM"),
                       values.get("VmSwap"))
    return (rss * 1024 if rss is not None else None,
            high * 1024 if high is not None else None,
            swap * 1024 if swap is not None else None)


def delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: after.get(key, 0) - before.get(key, 0)
            for key in set(before) | set(after)}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as target:
        target.write(json.dumps(row, sort_keys=True) + "\n")
        target.flush()
        os.fsync(target.fileno())


def request_json(url: str, body: dict[str, Any] | None = None,
                 timeout: float = 30.0) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("server response is not a JSON object")
    return value


def read_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (read_text(path) or "").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def trace_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    routes = [
        {"position": row.get("position"), "layer": row.get("layer"),
         "experts": row.get("experts")}
        for row in rows if row.get("event") == "decode_layer"
    ]
    aggregates = [row for row in rows
                  if row.get("event") in ("prefill_chunk", "decode_token")]
    chunks = [row for row in rows if row.get("event") == "prefill_chunk"]
    traced = [row for row in rows if "direct_io" in row]
    route_payload = json.dumps(routes, sort_keys=True,
                               separators=(",", ":")).encode()
    by_position: dict[str, list[dict[str, Any]]] = {}
    for route in routes:
        by_position.setdefault(str(route["position"]), []).append({
            "layer": route["layer"], "experts": route["experts"]})
    route_positions = {
        position: {
            "count": len(position_routes),
            "sha256": canonical_sha256(position_routes),
        }
        for position, position_routes in by_position.items()
    }
    return {
        "event_counts": {
            event: sum(row.get("event") == event for row in rows)
            for event in ("prefill_layer", "prefill_chunk", "decode_layer",
                          "decode_token")
        },
        "decode_route_count": len(routes),
        "decode_route_sha256": hashlib.sha256(route_payload).hexdigest(),
        "decode_route_positions": route_positions,
        "prefill_tokens": sum(int(row.get("tokens", 0)) for row in chunks),
        "prefill_position_start": (chunks[0].get("position_start")
                                   if chunks else None),
        "prefill_chunks": [
            {key: row.get(key) for key in
             ("position_start", "position_end", "tokens", "bytes_read")}
            for row in chunks
        ],
        "expert_bytes_read": sum(int(row.get("bytes_read", 0))
                                 for row in aggregates),
        "expert_cache_hits": sum(int(row.get("cache_hits", 0))
                                 for row in aggregates),
        "expert_cache_misses": sum(int(row.get("cache_misses", 0))
                                   for row in aggregates),
        "transport_ok": bool(traced) and all(
            bool(row.get("direct_io")) and
            row.get("io_backend") == "io_uring" and
            int(row.get("queue_depth", 0)) == 4 and
            not bool(row.get("read_error")) for row in traced),
        "read_error": any(bool(row.get("read_error")) for row in traced),
    }


def trace_slice(source: Path, offset: int, destination: Path
                ) -> tuple[int, dict[str, Any]]:
    with source.open("rb") as trace:
        trace.seek(offset)
        payload = trace.read()
        end = trace.tell()
    destination.write_bytes(payload)
    rows = [json.loads(line) for line in payload.decode().splitlines()
            if line.strip()]
    summary = trace_summary(rows)
    summary["newline_terminated"] = bool(payload) and payload.endswith(b"\n")
    return end, summary


def stable_output(response: dict[str, Any]) -> dict[str, Any]:
    choice = response.get("choices", [{}])[0]
    usage = response.get("usage") or {}
    return {
        "message": choice.get("message"),
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def counter_delta(current: dict[str, Any], previous: dict[str, int]
                  ) -> dict[str, int]:
    result: dict[str, int] = {}
    for key in COUNTER_KEYS:
        now = int(current.get(key, 0))
        result[key] = now - previous.get(key, 0)
        previous[key] = now
    return result


def resolved_cpu_set(stdout: str, stderr: str) -> str | None:
    match = re.search(r"^CPU set\s+(.+)$", stdout + "\n" + stderr,
                      re.MULTILINE)
    return match.group(1) if match else None


def git_commit(repo: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def monitor_process(proc: subprocess.Popen, path: Path,
                    stop: threading.Event, totals: dict[str, Any]) -> None:
    with path.open("w") as target:
        while not stop.is_set():
            rss, high, swap = process_memory(proc.pid)
            available = mem_available()
            if rss is not None:
                totals["peak_rss_bytes"] = max(
                    totals.get("peak_rss_bytes", 0), rss, high or 0)
            if swap is not None:
                totals["peak_process_swap_bytes"] = max(
                    totals.get("peak_process_swap_bytes", 0), swap)
            if available is not None:
                old = totals.get("min_mem_available_bytes")
                totals["min_mem_available_bytes"] = (
                    available if old is None else min(old, available))
            row = {
                "timestamp_utc": utc_now(),
                "monotonic_ns": time.monotonic_ns(),
                "process_rss_bytes": rss,
                "process_high_water_rss_bytes": high,
                "process_swap_bytes": swap,
                "mem_available_bytes": available,
                "vmstat": vmstat(),
                "memory_psi_totals_us": pressure(),
            }
            target.write(json.dumps(row, sort_keys=True) + "\n")
            target.flush()
            stop.wait(0.5)


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def tool_schema(name: str) -> list[dict[str, Any]]:
    return [{
        "type": "function",
        "function": {
            "name": name,
            "description": "Return one named system metric.",
            "parameters": {
                "type": "object",
                "properties": {"metric": {"type": "string"}},
                "required": ["metric"],
                "additionalProperties": False,
            },
        },
    }]


def chat_body(model_id: str, system: str, user: str,
              tools: list[dict[str, Any]], tokens: int) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "tools": tools,
        "temperature": 0,
        "seed": 1,
        "max_tokens": tokens,
        "stream": False,
        "reasoning_effort": "none",
    }


def run_server(*, role: str, port: int, requests: list[dict[str, Any]],
               repo: Path, model: Path, out: Path, campaign: dict[str, Any],
               prefix_cache_bytes: int, startup_timeout: float,
               request_timeout: float) -> dict[str, Any]:
    run_dir = out / role
    run_dir.mkdir(parents=True, exist_ok=False)
    trace_path = run_dir / "trace.jsonl"
    stdout_path = run_dir / "server-stdout.txt"
    stderr_path = run_dir / "server-stderr.txt"
    telemetry_path = run_dir / "telemetry.jsonl"
    model_id = campaign["model_id"]
    cmd = [
        sys.executable, "-m", "serve", str(model),
        "--host", "127.0.0.1", "--port", str(port),
        "--model-id", model_id,
        "--budget", str(campaign["budget_bytes"]),
        "--threads", str(campaign["threads"]),
        "--cpu-set", campaign["cpu_set"],
        "--cache", "lfru", "--io-backend", "io_uring",
        "--io-queue-depth", "4",
        "--max-tokens", str(campaign["tokens_requested"]),
        "--no-thinking", "--trace-layers", str(trace_path),
    ]
    if prefix_cache_bytes:
        cmd += ["--prefix-cache", str(prefix_cache_bytes)]
    environment_overrides = dict(ENGINE_ENV)
    environment_overrides["WASTE_LIB"] = campaign["library_path"]
    (run_dir / "command.json").write_text(
        json.dumps({"cwd": str(repo), "argv": cmd,
                    "environment_overrides": environment_overrides},
                   indent=2, sort_keys=True) + "\n")

    before_vm, before_psi = vmstat(), pressure()
    started_ns = time.time_ns()
    ready_ns: int | None = None
    response_rows: list[dict[str, Any]] = []
    error: str | None = None
    totals: dict[str, Any] = {"peak_rss_bytes": 0,
                              "peak_process_swap_bytes": 0}
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment.update(environment_overrides)
    stdout_file = stdout_path.open("w")
    stderr_file = stderr_path.open("w")
    proc = subprocess.Popen(cmd, cwd=repo, stdout=stdout_file,
                            stderr=stderr_file, text=True, env=environment)
    stop = threading.Event()
    watcher = threading.Thread(target=monitor_process,
                               args=(proc, telemetry_path, stop, totals),
                               daemon=True)
    watcher.start()
    base = f"http://127.0.0.1:{port}"
    trace_offset = 0
    previous_counters: dict[str, int] = {}
    try:
        deadline = time.monotonic() + startup_timeout
        health: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with {proc.returncode}")
            try:
                health = request_json(base + "/health", timeout=2.0)
                ready_ns = time.time_ns()
                break
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(1.0)
        if health is None:
            raise RuntimeError("server startup timed out")
        (run_dir / "health-ready.json").write_text(
            json.dumps(health, indent=2, sort_keys=True) + "\n")
        trace_offset = trace_path.stat().st_size

        for index, spec in enumerate(requests, 1):
            started = time.time_ns()
            response = request_json(base + "/v1/chat/completions",
                                    spec["body"], timeout=request_timeout)
            wall_seconds = (time.time_ns() - started) / 1e9
            response_path = run_dir / f"response-{index}-{spec['name']}.json"
            response_path.write_text(
                json.dumps(response, indent=2, ensure_ascii=False,
                           sort_keys=True) + "\n")
            request_trace_path = run_dir / f"trace-{index}-{spec['name']}.jsonl"
            trace_offset, summary = trace_slice(
                trace_path, trace_offset, request_trace_path)
            output = stable_output(response)
            waste = response.get("waste") or {}
            prefix = waste.get("prefix_cache") or {}
            completion_tokens = int(
                (response.get("usage") or {}).get("completion_tokens", 0))
            # Rooted requests hand the final prompt token back to generate()
            # for canonical logits replay, so their trace contains one more
            # decode_token than the number delivered to the HTTP client.
            expected_decode = completion_tokens + (
                1 if int(prefix.get("family_root_tokens", 0)) > 0 else 0)
            summary["expected_decode_tokens"] = expected_decode
            summary["boundary_complete"] = (
                summary["newline_terminated"] and
                summary["event_counts"].get("decode_token") ==
                expected_decode)
            response_rows.append({
                "index": index,
                "name": spec["name"],
                "wall_seconds": wall_seconds,
                "request_sha256": canonical_sha256(spec["body"]),
                "tool_sha256": spec["tool_sha256"],
                "response_path": str(response_path),
                "response_sha256": sha256(response_path),
                "trace_path": str(request_trace_path),
                "trace_sha256": sha256(request_trace_path),
                "trace": summary,
                "stable_output": output,
                "stable_output_sha256": canonical_sha256(output),
                "usage": response.get("usage"),
                "response_waste": waste,
                "engine_counter_delta": counter_delta(
                    waste, previous_counters),
            })
    except Exception as exc:  # preserve partial evidence
        error = f"{type(exc).__name__}: {exc}"
    finally:
        stop_server(proc)
        stop.set()
        watcher.join(timeout=2)
        stdout_file.close()
        stderr_file.close()

    stdout = read_text(stdout_path) or ""
    stderr = read_text(stderr_path) or ""
    trace_rows = read_trace(trace_path) if trace_path.exists() else []
    meta = next((row for row in trace_rows if row.get("event") == "meta"), {})
    ended_ns = time.time_ns()
    return {
        "schema": "waste.family_prefix_run.v1",
        "campaign_id": campaign["campaign_id"],
        "run_id": f"{campaign['campaign_id']}-{role}",
        "role": role,
        "timestamp_utc": utc_now(),
        "host": socket.gethostname(),
        "kernel": platform.release(),
        "command": cmd,
        "server_returncode": proc.returncode,
        "error": error,
        "startup_seconds": ((ready_ns - started_ns) / 1e9
                            if ready_ns is not None else None),
        "total_wall_seconds": (ended_ns - started_ns) / 1e9,
        "git_commit": campaign["git_commit"],
        "library_path": campaign["library_path"],
        "library_sha256": campaign["library_sha256"],
        "manifest_sha256": campaign["manifest_sha256"],
        "usage_sha256": campaign["usage_sha256"],
        "budget_bytes": campaign["budget_bytes"],
        "rss_limit_bytes": campaign["rss_limit_bytes"],
        "max_memory_psi_us": campaign["max_memory_psi_us"],
        "prefix_cache_bytes": prefix_cache_bytes,
        "threads": campaign["threads"],
        "cpu_set": campaign["cpu_set"],
        "resolved_cpu_set": resolved_cpu_set(stdout, stderr),
        "io_backend_requested": "io_uring",
        "io_queue_depth_requested": 4,
        "direct_io_requested": True,
        "engine_environment": dict(ENGINE_ENV),
        "tokens_requested": campaign["tokens_requested"],
        "system_sha256": campaign["system_sha256"],
        "stable_tool_sha256": campaign["stable_tool_sha256"],
        "changed_tool_sha256": campaign["changed_tool_sha256"],
        "prompt_b_sha256": campaign["prompt_b_sha256"],
        "requests": response_rows,
        "peak_rss_bytes": totals.get("peak_rss_bytes") or None,
        "peak_process_swap_bytes": totals.get("peak_process_swap_bytes", 0),
        "min_mem_available_bytes": totals.get("min_mem_available_bytes"),
        "vm_delta": delta(vmstat(), before_vm),
        "memory_psi_total_us_delta": delta(pressure(), before_psi),
        "trace_path": str(trace_path),
        "trace_sha256": sha256(trace_path),
        "trace_meta": meta,
        "telemetry_path": str(telemetry_path),
        "telemetry_sha256": sha256(telemetry_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--library", type=Path,
                    help="libwaste.so to force in both servers "
                         "(default: REPO/libwaste.so)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="k3-family-root-qd4")
    ap.add_argument("--budget", type=int, required=True,
                    help="fixed total RAM budget in bytes for both servers")
    ap.add_argument("--prefix-cache", type=int, default=1 << 30,
                    help="treatment reservation inside the fixed budget")
    ap.add_argument("--rss-limit", type=int,
                    help="acceptance ceiling; default is --budget")
    ap.add_argument("--max-memory-psi-us", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cpu-set", default="performance")
    ap.add_argument("--tokens", type=int, default=2)
    ap.add_argument("--control-port", type=int, default=18080)
    ap.add_argument("--prefix-port", type=int, default=18081)
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--request-timeout", type=float, default=1200.0)
    ap.add_argument("--system", default=(
        "You are a concise systems assistant. Answer in one short sentence."))
    ap.add_argument("--prompt-a", default="Define queue depth.")
    ap.add_argument("--prompt-b", default="Define direct I/O.")
    args = ap.parse_args(argv)

    if platform.system() != "Linux":
        ap.error("the real-K3 acceptance requires Linux /proc telemetry")
    if args.budget <= 0:
        ap.error("--budget must be positive")
    if not 0 < args.prefix_cache < args.budget:
        ap.error("--prefix-cache must be positive and smaller than --budget")
    if args.tokens < 1 or args.threads < 1:
        ap.error("--tokens and --threads must be positive")
    if args.max_memory_psi_us < 0:
        ap.error("--max-memory-psi-us cannot be negative")
    if args.rss_limit is not None and args.rss_limit <= 0:
        ap.error("--rss-limit must be positive")
    if args.control_port == args.prefix_port:
        ap.error("control and prefix ports must differ")

    repo, model, out = (args.repo.resolve(), args.model.resolve(),
                        args.out.resolve())
    if not (model / "manifest.json").is_file():
        ap.error(f"not a WASTE container: {model}")
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    if runs_path.exists() or (out / "control").exists() or (out / "prefix").exists():
        ap.error(f"output already contains a campaign: {out}")

    stable_tools = tool_schema("read_metric")
    changed_tools = tool_schema("read_metric_v2")
    model_id = "k3-family-root"
    body_a = chat_body(model_id, args.system, args.prompt_a,
                       stable_tools, args.tokens)
    body_b = chat_body(model_id, args.system, args.prompt_b,
                       stable_tools, args.tokens)
    body_changed = chat_body(model_id, args.system, args.prompt_b,
                             changed_tools, args.tokens)
    library = (args.library or (repo / "libwaste.so")).resolve()
    if not library.is_file():
        ap.error(f"build or select the shared library first: {library}")
    rss_limit = args.rss_limit if args.rss_limit is not None else args.budget
    campaign = {
        "schema": "waste.family_prefix_campaign.v1",
        "campaign_id": args.label,
        "created_utc": utc_now(),
        "host": socket.gethostname(),
        "kernel": platform.release(),
        "repo": str(repo),
        "model": str(model),
        "model_id": model_id,
        "git_commit": git_commit(repo),
        "library_path": str(library),
        "library_sha256": sha256(library),
        "manifest_sha256": sha256(model / "manifest.json"),
        "usage_sha256": sha256(model / "usage.waste"),
        "budget_bytes": args.budget,
        "rss_limit_bytes": rss_limit,
        "prefix_cache_bytes": args.prefix_cache,
        "max_memory_psi_us": args.max_memory_psi_us,
        "threads": args.threads,
        "cpu_set": args.cpu_set,
        "io_backend": "io_uring",
        "io_queue_depth": 4,
        "direct_io": True,
        "engine_environment": dict(ENGINE_ENV),
        "tokens_requested": args.tokens,
        "system": args.system,
        "system_sha256": hashlib.sha256(args.system.encode()).hexdigest(),
        "prompt_a": args.prompt_a,
        "prompt_b": args.prompt_b,
        "prompt_b_sha256": hashlib.sha256(args.prompt_b.encode()).hexdigest(),
        "stable_tool_schema": stable_tools,
        "stable_tool_sha256": canonical_sha256(stable_tools),
        "changed_tool_schema": changed_tools,
        "changed_tool_sha256": canonical_sha256(changed_tools),
        "request_b_sha256": canonical_sha256(body_b),
    }
    (out / "campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n")

    common = dict(repo=repo, model=model, out=out, campaign=campaign,
                  startup_timeout=args.startup_timeout,
                  request_timeout=args.request_timeout)
    control = run_server(
        role="control", port=args.control_port,
        requests=[
            {"name": "seed_a", "body": body_a,
             "tool_sha256": campaign["stable_tool_sha256"]},
            {"name": "cold_b", "body": body_b,
             "tool_sha256": campaign["stable_tool_sha256"]},
        ],
        prefix_cache_bytes=0, **common)
    append_jsonl(runs_path, control)
    prefix = run_server(
        role="prefix", port=args.prefix_port,
        requests=[
            {"name": "seed_a", "body": body_a,
             "tool_sha256": campaign["stable_tool_sha256"]},
            {"name": "family_b", "body": body_b,
             "tool_sha256": campaign["stable_tool_sha256"]},
            {"name": "changed_tool_b", "body": body_changed,
             "tool_sha256": campaign["changed_tool_sha256"]},
        ], prefix_cache_bytes=args.prefix_cache, **common)
    append_jsonl(runs_path, prefix)

    analyzer = repo / "tools" / "analyze_family_prefix.py"
    analyze_cmd = [sys.executable, str(analyzer), str(runs_path),
                   "--output", str(out / "analysis.json"),
                   "--markdown", str(out / "analysis.md")]
    analysis = subprocess.run(analyze_cmd, cwd=repo, text=True,
                              capture_output=True)
    (out / "analysis-stdout.txt").write_text(analysis.stdout)
    (out / "analysis-stderr.txt").write_text(analysis.stderr)
    print(analysis.stdout, end="")
    if analysis.stderr:
        print(analysis.stderr, end="", file=sys.stderr)
    return analysis.returncode


if __name__ == "__main__":
    raise SystemExit(main())
