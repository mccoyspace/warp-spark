#!/usr/bin/env python3
"""Run one command while holding an FD-scoped Linux PM-QoS request.

This helper is deliberately narrow: the Spark campaign needs a zero-
microsecond CPU DMA-latency request, held only for the lifetime of one child.
It opens ``/dev/cpu_dma_latency``, writes one signed 32-bit value, retains the
descriptor in the root parent, and closes it in ``finally``.  The kernel also
closes the descriptor if this process is killed, so the request is inherently
self-cleaning.  Nothing under sysfs is read or written by this helper.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import pwd
import signal
import stat
import struct
import subprocess
import sys
import time
from typing import Any, Sequence


QOS_DEVICE = Path("/dev/cpu_dma_latency")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def child_identity(user: str) -> tuple[pwd.struct_passwd, dict[str, str]]:
    account = pwd.getpwnam(user)
    env = os.environ.copy()
    env.update({"HOME": account.pw_dir, "USER": account.pw_name,
                "LOGNAME": account.pw_name})
    return account, env


def privilege_drop(account: pwd.struct_passwd):
    def drop() -> None:
        os.initgroups(account.pw_name, account.pw_gid)
        os.setgid(account.pw_gid)
        os.setuid(account.pw_uid)
    return drop


def run_scoped(command: Sequence[str], *, user: str, latency_us: int,
               device: Path, status_path: Path) -> int:
    status: dict[str, Any] = {
        "schema": "waste.pm_qos_exec.v1",
        "started_utc": utc_now(),
        "started_monotonic_ns": time.monotonic_ns(),
        "holder_pid": os.getpid(),
        "holder_euid": os.geteuid(),
        "target_user": user,
        "device": str(device),
        "latency_us": latency_us,
        "fd_scoped": True,
        "self_cleaning": True,
        "sysfs_modified": False,
        "command": list(command),
        "fd_opened": False,
        "fd_closed": False,
        "child_started": False,
    }
    fd = -1
    child: subprocess.Popen[bytes] | None = None
    forwarded_signal: int | None = None

    def forward(signum: int, _frame: object) -> None:
        nonlocal forwarded_signal
        forwarded_signal = signum
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    old_handlers = {sig: signal.getsignal(sig)
                    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        if os.geteuid() != 0:
            raise PermissionError("PM-QoS holder must run as root")
        if latency_us != 0:
            raise ValueError("this acceptance helper permits only 0 us")
        if device != QOS_DEVICE:
            raise ValueError(f"device must be exactly {QOS_DEVICE}")
        if not command:
            raise ValueError("no child command")
        account, child_env = child_identity(user)
        if not stat.S_ISCHR(os.stat(device, follow_symlinks=False).st_mode):
            raise ValueError(f"{device} is not a character device")
        fd = os.open(device, os.O_RDWR | getattr(os, "O_CLOEXEC", 0) |
                     getattr(os, "O_NOFOLLOW", 0))
        payload = struct.pack("=i", latency_us)
        if os.write(fd, payload) != len(payload):
            raise OSError("short write to PM-QoS device")
        status.update({"fd_opened": True,
                       "fd_opened_monotonic_ns": time.monotonic_ns()})
        atomic_json(status_path, status)
        for sig in old_handlers:
            signal.signal(sig, forward)
        child = subprocess.Popen(
            list(command), env=child_env,
            preexec_fn=privilege_drop(account), start_new_session=True)
        status.update({"child_started": True, "child_pid": child.pid,
                       "child_started_monotonic_ns": time.monotonic_ns()})
        returncode = child.wait()
        status.update({"child_returncode": returncode,
                       "child_ended_monotonic_ns": time.monotonic_ns()})
        if forwarded_signal is not None and returncode < 0:
            return 128 + forwarded_signal
        return returncode
    except (KeyError, OSError, PermissionError, ValueError) as exc:
        status.update({"error": f"{type(exc).__name__}: {exc}",
                       "child_returncode": None})
        return 1
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if fd >= 0:
            os.close(fd)
            status.update({"fd_closed": True,
                           "fd_closed_monotonic_ns": time.monotonic_ns()})
        status.update({"ended_utc": utc_now(),
                       "ended_monotonic_ns": time.monotonic_ns()})
        try:
            atomic_json(status_path, status)
        except OSError as exc:
            print(f"pm_qos_exec: cannot write status: {exc}", file=sys.stderr)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-us", type=int, choices=(0,), default=0)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_scoped(args.command, user=args.user,
                      latency_us=args.latency_us, device=QOS_DEVICE,
                      status_path=args.status)


if __name__ == "__main__":
    raise SystemExit(main())
