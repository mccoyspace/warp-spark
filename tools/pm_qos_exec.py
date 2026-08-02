#!/usr/bin/env python3
"""Run a command with a root-held Linux CPU DMA-latency request.

``--scope child`` preserves the simple acceptance-run behavior: Q0 is held for
the complete child lifetime.  ``--scope requests`` passes the unprivileged
child only one end of a private socketpair.  The root parent opens
``/dev/cpu_dma_latency`` on BEGIN and closes it on END, EOF, child death,
signal, protocol failure, or the configured maximum hold time.  It never
writes sysfs and never passes the PM-QoS descriptor to the child.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import pwd
import re
import secrets
import select
import signal
import socket
import stat
import struct
import subprocess
import sys
import time
from typing import Any, Callable, Optional, Sequence


QOS_DEVICE = Path("/dev/cpu_dma_latency")
CONTROL_FD_ENV = "WASTE_PM_QOS_CONTROL_FD"
PROTOCOL = "waste.pm_qos.requests.v1"
MAX_FRAME_BYTES = 16 * 1024
MAX_HOLD_SECONDS = 4 * 60 * 60
CONTROL_SEND_TIMEOUT_SECONDS = 5.0
DEFAULT_LOCK = Path("/run/lock/waste-pm-qos.lock")
SAFE_CHILD_ENVIRONMENT = {
    "LANG", "LANGUAGE", "PATH", "PYTHONPATH", "TZ", "VIRTUAL_ENV",
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
}
SAFE_CHILD_ENV_PREFIXES = ("LC_", "OMP_", "OPENBLAS_", "WASTE_")
SECRET_OPTIONS = {
    "--api-key", "--password", "--secret", "--token", "--access-token",
}
HOLDER_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
FATAL_CONTROL_REASONS = {
    "control_send_timeout", "eof", "protocol_error", "control_error",
    "holder_exit",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _open_output_directory(path: Path) -> int:
    """Open/create a directory path without traversing a symlink.

    The holder runs as root and its artifact locations are CLI input.  Walking
    one component at a time with openat semantics prevents a writable parent
    from redirecting a status or event write into an arbitrary root path.
    """
    path = Path(path)
    flags = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
             getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    if path.is_absolute():
        fd = os.open("/", flags)
        parts = path.parts[1:]
    else:
        fd = os.open(".", flags)
        parts = path.parts
    try:
        for part in parts:
            if part in ("", "."):
                continue
            if part == "..":
                raise ValueError("output paths may not contain '..'")
            try:
                os.mkdir(part, 0o700, dir_fd=fd)
            except FileExistsError:
                pass
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short artifact write")
        view = view[written:]


def _require_private_regular(fd: int, description: str) -> None:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{description} is not a regular file")
    if info.st_uid != os.geteuid() or info.st_nlink != 1:
        raise PermissionError(
            f"{description} must be owned by the holder with one link")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    if not path.name:
        raise ValueError("status path must name a file")
    directory_fd = _open_output_directory(path.parent)
    temporary: Optional[str] = None
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        for _ in range(128):
            candidate = f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
            try:
                fd = os.open(
                    candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                    getattr(os, "O_CLOEXEC", 0) |
                    getattr(os, "O_NOFOLLOW", 0), 0o600,
                    dir_fd=directory_fd)
                temporary = candidate
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError("cannot allocate a secure status temp file")
        try:
            _require_private_regular(fd, "status temp file")
            _write_all(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, path.name, src_dir_fd=directory_fd,
                   dst_dir_fd=directory_fd)
        temporary = None
        os.fsync(directory_fd)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def append_jsonl(path: Optional[Path], value: dict[str, Any]) -> None:
    if path is None:
        return
    path = Path(path)
    if not path.name:
        raise ValueError("event path must name a file")
    directory_fd = _open_output_directory(path.parent)
    fd = -1
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_APPEND | os.O_CREAT |
                     getattr(os, "O_CLOEXEC", 0) |
                     getattr(os, "O_NOFOLLOW", 0) |
                     getattr(os, "O_NONBLOCK", 0), 0o600,
                     dir_fd=directory_fd)
        _require_private_regular(fd, "PM-QoS event log")
        os.fchmod(fd, 0o600)
        payload = (json.dumps(value, sort_keys=True) + "\n").encode()
        _write_all(fd, payload)
        os.fsync(fd)
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(directory_fd)


class PMQosDevice:
    """The privileged descriptor, with injectable syscalls for unit tests."""

    def __init__(self, path: Path = QOS_DEVICE, *,
                 stat_fn: Callable[..., os.stat_result] = os.stat,
                 open_fn: Callable[..., int] = os.open,
                 write_fn: Callable[[int, bytes], int] = os.write,
                 close_fn: Callable[[int], None] = os.close,
                 require_exact_path: bool = True):
        self.path = path
        self._stat = stat_fn
        self._open = open_fn
        self._write = write_fn
        self._close = close_fn
        self._require_exact_path = require_exact_path
        self.fd = -1
        self.opened_monotonic_ns: Optional[int] = None

    @property
    def active(self) -> bool:
        return self.fd >= 0

    def validate(self) -> None:
        if self._require_exact_path and self.path != QOS_DEVICE:
            raise ValueError(f"device must be exactly {QOS_DEVICE}")
        mode = self._stat(self.path, follow_symlinks=False).st_mode
        if not stat.S_ISCHR(mode):
            raise ValueError(f"{self.path} is not a character device")

    def acquire(self, latency_us: int) -> int:
        if latency_us != 0:
            raise ValueError("this helper permits only a 0 us request")
        if self.active:
            raise RuntimeError("PM-QoS descriptor is already active")
        self.validate()
        flags = (os.O_RDWR | getattr(os, "O_CLOEXEC", 0) |
                 getattr(os, "O_NOFOLLOW", 0))
        fd = self._open(self.path, flags)
        try:
            payload = struct.pack("=i", latency_us)
            if self._write(fd, payload) != len(payload):
                raise OSError("short write to PM-QoS device")
        except BaseException:
            self._close(fd)
            raise
        self.fd = fd
        self.opened_monotonic_ns = time.monotonic_ns()
        return self.opened_monotonic_ns

    def release(self) -> Optional[int]:
        if not self.active:
            return None
        fd, self.fd = self.fd, -1
        self._close(fd)
        return time.monotonic_ns()


class RequestSupervisor:
    """Serve the BEGIN/END protocol and enforce a bounded active lease."""

    def __init__(self, control: socket.socket, device: Any, *, latency_us: int,
                 max_hold_seconds: float,
                 status_callback: Optional[Callable[[str, dict], None]] = None,
                 expiration_callback: Optional[
                     Callable[[str, dict], None]] = None,
                 fatal_control_callback: Optional[
                     Callable[[str], None]] = None,
                 send_timeout_seconds: float = CONTROL_SEND_TIMEOUT_SECONDS,
                 should_stop: Optional[Callable[[], bool]] = None):
        if latency_us != 0:
            raise ValueError("request supervisor permits only Q0")
        if max_hold_seconds <= 0:
            raise ValueError("maximum PM-QoS hold must be positive")
        if (not math.isfinite(max_hold_seconds) or
                max_hold_seconds > MAX_HOLD_SECONDS):
            raise ValueError(
                f"maximum PM-QoS hold may not exceed {MAX_HOLD_SECONDS}s")
        if not math.isfinite(send_timeout_seconds) or \
                send_timeout_seconds <= 0:
            raise ValueError("PM-QoS control send timeout must be positive")
        self.control = control
        self.device = device
        self.latency_us = latency_us
        self.max_hold_seconds = float(max_hold_seconds)
        self.send_timeout_seconds = float(send_timeout_seconds)
        self.status_callback = status_callback or (lambda event, data: None)
        self.expiration_callback = expiration_callback or \
            (lambda request_id, data: None)
        self.fatal_control_callback = fatal_control_callback or \
            (lambda reason: None)
        self.should_stop = should_stop or (lambda: False)
        self.buffer = bytearray()
        self.active_request: Optional[str] = None
        self.acquired_ns: Optional[int] = None
        self.deadline_ns: Optional[int] = None
        self.last_expired_request: Optional[str] = None
        self.leases_started = 0
        self.leases_completed = 0
        self.lease_expirations = 0
        self.fd_open_count = 0
        self.fd_close_count = 0
        self.protocol_errors = 0
        self.last_error: Optional[str] = None
        self.closed_reason: Optional[str] = None
        self.hello_complete = False
        self.telemetry_errors = 0
        self.telemetry_last_error: Optional[str] = None
        self._pending_telemetry: list[tuple[str, dict]] = []
        self.fatal_control_reason: Optional[str] = None

    def snapshot(self) -> dict:
        return {
            "protocol": PROTOCOL,
            "scope": "requests",
            "latency_us": self.latency_us,
            "max_hold_seconds": self.max_hold_seconds,
            "active": self.active_request is not None,
            "active_request": self.active_request,
            "acquired_monotonic_ns": self.acquired_ns,
            "fd_active": bool(self.device.active),
            "fd_open_count": self.fd_open_count,
            "fd_close_count": self.fd_close_count,
            "leases_started": self.leases_started,
            "leases_completed": self.leases_completed,
            "lease_expirations": self.lease_expirations,
            "protocol_errors": self.protocol_errors,
            "last_error": self.last_error,
            "closed_reason": self.closed_reason,
            "telemetry_errors": self.telemetry_errors,
            "telemetry_last_error": self.telemetry_last_error,
            "telemetry_pending": len(self._pending_telemetry),
            "fatal_control_reason": self.fatal_control_reason,
        }

    def _queue_telemetry(self, event: str, **detail: Any) -> None:
        """Record an event now for best-effort delivery after Q0 release."""
        payload = {"event": event, "monotonic_ns": time.monotonic_ns(),
                   "delivery": "deferred_best_effort", **detail,
                   "supervisor": self.snapshot()}
        self._pending_telemetry.append((event, payload))

    def _flush_telemetry(self) -> None:
        """Deliver recorded events only while the privileged fd is closed."""
        if self.device.active:
            raise RuntimeError("refusing telemetry while PM-QoS is active")
        pending, self._pending_telemetry = self._pending_telemetry, []
        for event, original in pending:
            payload = {**original,
                       "delivery_monotonic_ns": time.monotonic_ns()}
            try:
                self.status_callback(event, payload)
            except Exception as exc:
                # Telemetry is deliberately outside the safety boundary.  A
                # later event/final status write gets another chance and
                # exposes the failure without changing device accounting.
                self.telemetry_errors += 1
                self.telemetry_last_error = f"{type(exc).__name__}: {exc}"

    def _notify(self, event: str, **detail: Any) -> None:
        self._queue_telemetry(event, **detail)
        self._flush_telemetry()

    def _terminate_on_fatal_control(self, reason: str) -> None:
        """Stop a strict child after fd release and before any telemetry."""
        if reason not in FATAL_CONTROL_REASONS or \
                self.fatal_control_reason is not None:
            return
        self.fatal_control_reason = reason
        try:
            self.fatal_control_callback(reason)
        except Exception as exc:
            self.last_error = (
                f"fatal PM-QoS control loss; child cleanup failed: "
                f"{type(exc).__name__}: {exc}")

    def _send(self, value: dict) -> None:
        data = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        if len(data) + 1 > MAX_FRAME_BYTES:
            raise ValueError("PM-QoS response frame is too large")
        view = memoryview(data + b"\n")
        now = time.monotonic_ns()
        ordinary_deadline = now + int(self.send_timeout_seconds * 1e9)
        send_deadline = ordinary_deadline
        if self.deadline_ns is not None:
            send_deadline = min(send_deadline, self.deadline_ns)
        previous_timeout = self.control.gettimeout()
        try:
            self.control.setblocking(False)
            while view:
                now = time.monotonic_ns()
                if now >= send_deadline:
                    if self.active_request is not None:
                        expired = (self.deadline_ns is not None and
                                   now >= self.deadline_ns)
                        self._release(
                            "max_hold" if expired else "control_send_timeout",
                            expired=expired, notify=False)
                        if not expired:
                            self._terminate_on_fatal_control(
                                "control_send_timeout")
                    else:
                        self._terminate_on_fatal_control(
                            "control_send_timeout")
                    self._flush_telemetry()
                    raise TimeoutError(
                        "PM-QoS control acknowledgement send timed out")
                try:
                    written = self.control.send(view)
                except (BlockingIOError, InterruptedError):
                    written = 0
                else:
                    if written <= 0:
                        raise BrokenPipeError(
                            "PM-QoS control acknowledgement channel closed")
                    view = view[written:]
                    continue
                timeout = min(0.1, (send_deadline - now) / 1e9)
                select.select([], [self.control], [], max(0.0, timeout))
        finally:
            try:
                self.control.settimeout(previous_timeout)
            except OSError:
                pass

    def _acquire(self) -> int:
        """Acquire once and account only observed descriptor transitions."""
        was_active = bool(self.device.active)
        try:
            raw_acquired_ns = self.device.acquire(self.latency_us)
        except (OSError, RuntimeError, ValueError):
            # A defensive cleanup covers an injected/partial implementation
            # that opens successfully and then raises before returning.
            if not was_active and self.device.active:
                self.fd_open_count += 1
                self.device.release()
                if not self.device.active:
                    self.fd_close_count += 1
            raise
        if was_active or not self.device.active:
            raise RuntimeError(
                "PM-QoS acquire did not create one active descriptor")
        self.fd_open_count += 1
        try:
            return int(raw_acquired_ns)
        except (TypeError, ValueError, OverflowError):
            self.device.release()
            if not self.device.active:
                self.fd_close_count += 1
            raise ValueError("PM-QoS acquire returned an invalid timestamp")

    def _release(self, reason: str, *, expired: bool = False,
                 notify: bool = True) -> dict:
        request_id = self.active_request
        acquired_ns = self.acquired_ns
        was_active = bool(self.device.active)
        released_ns = self.device.release()
        if was_active and not self.device.active:
            self.fd_close_count += 1
        if released_ns is None:
            released_ns = time.monotonic_ns()
        hold_ms = ((released_ns - acquired_ns) / 1e6
                   if acquired_ns is not None else 0.0)
        self.active_request = None
        self.acquired_ns = None
        self.deadline_ns = None
        if expired:
            self.lease_expirations += 1
            self.last_expired_request = request_id
            self.last_error = "lease expired at maximum hold"
        elif reason == "end":
            self.leases_completed += 1
        released = {"request_id": request_id,
                    "released_monotonic_ns": released_ns, "hold_ms": hold_ms}
        self._queue_telemetry(
            "release", request_id=request_id, reason=reason,
            expired=expired, released_monotonic_ns=released_ns,
            hold_ms=hold_ms)
        if expired:
            try:
                self.expiration_callback(request_id or "", released)
            except Exception as exc:
                self.last_error = (
                    f"lease expired; child cleanup failed: "
                    f"{type(exc).__name__}: {exc}")
        if notify:
            self._flush_telemetry()
        return released

    def abort(self, reason: str) -> None:
        self.closed_reason = reason
        if self.active_request is not None:
            self._release(reason, notify=False)
        elif self.device.active:
            # Outer holder cleanup is the final safety net.  Account an
            # orphaned descriptor here as well so a later final persist cannot
            # overwrite the real close with a stale supervisor count.
            released_ns = self.device.release()
            if not self.device.active:
                self.fd_close_count += 1
            self._queue_telemetry(
                "release", request_id=None, reason=reason, expired=False,
                released_monotonic_ns=(released_ns or time.monotonic_ns()),
                hold_ms=0.0)
        self._terminate_on_fatal_control(reason)
        self._flush_telemetry()

    def _protocol_failure(self, message: str) -> bool:
        self.protocol_errors += 1
        self.last_error = message
        if self.active_request is not None:
            self._release("protocol_error", notify=False)
        self._terminate_on_fatal_control("protocol_error")
        self._queue_telemetry("protocol_error", error=message)
        try:
            self._send({"ok": False, "error": message})
        except OSError:
            pass
        self._flush_telemetry()
        return False

    def _handle(self, message: dict) -> bool:
        op = message.get("op")
        if op == "hello":
            if self.hello_complete:
                return self._protocol_failure("duplicate PM-QoS hello")
            if message.get("protocol") != PROTOCOL:
                return self._protocol_failure("incompatible PM-QoS protocol")
            self.hello_complete = True
            self._send({"ok": True, "op": "hello", "protocol": PROTOCOL,
                        "scope": "requests", "latency_us": self.latency_us,
                        "max_hold_seconds": self.max_hold_seconds})
            self._notify("hello")
            return True
        if not self.hello_complete:
            return self._protocol_failure("PM-QoS hello is required first")

        request_id = message.get("request_id")
        if (not isinstance(request_id, str) or not request_id or
                len(request_id) > 128):
            return self._protocol_failure("invalid PM-QoS request id")

        if op == "begin":
            if self.active_request is not None:
                return self._protocol_failure("overlapping PM-QoS lease")
            try:
                acquired_ns = self._acquire()
            except (OSError, RuntimeError, ValueError) as exc:
                self.last_error = f"cannot acquire PM-QoS: {exc}"
                self._send({"ok": False, "op": "begin",
                            "request_id": request_id,
                            "error": self.last_error})
                self._notify("acquire_error", request_id=request_id,
                             error=self.last_error)
                return True
            self.active_request = request_id
            self.acquired_ns = acquired_ns
            self.deadline_ns = acquired_ns + int(self.max_hold_seconds * 1e9)
            self.leases_started += 1
            self._queue_telemetry(
                "acquire", request_id=request_id,
                acquired_monotonic_ns=acquired_ns)
            self._send({"ok": True, "op": "begin",
                        "request_id": request_id,
                        "acquired_monotonic_ns": acquired_ns})
            return True

        if op == "end":
            if self.active_request is None:
                if request_id == self.last_expired_request:
                    self._send({"ok": False, "op": "end",
                                "request_id": request_id,
                                "error": "PM-QoS lease expired at maximum hold"})
                    self.last_expired_request = None
                    return True
                self._send({"ok": False, "op": "end",
                            "request_id": request_id,
                            "error": "no active PM-QoS lease"})
                return True
            if request_id != self.active_request:
                return self._protocol_failure("PM-QoS END request id mismatch")
            released = self._release("end", notify=False)
            self._send({"ok": True, "op": "end", **released})
            self._flush_telemetry()
            return True

        return self._protocol_failure(f"unknown PM-QoS operation: {op!r}")

    def serve(self) -> str:
        """Run until EOF, child exit, stop request, or a fatal protocol error."""
        reason = "eof"
        try:
            while True:
                if self.should_stop():
                    reason = "child_exit"
                    break
                now = time.monotonic_ns()
                if self.deadline_ns is not None and now >= self.deadline_ns:
                    self._release("max_hold", expired=True)
                timeout = 0.1
                if self.deadline_ns is not None:
                    timeout = min(timeout, max(0.0,
                                  (self.deadline_ns - now) / 1e9))
                ready, _, _ = select.select([self.control], [], [], timeout)
                if not ready:
                    continue
                chunk = self.control.recv(4096)
                if not chunk:
                    reason = "eof"
                    break
                self.buffer.extend(chunk)
                if len(self.buffer) > MAX_FRAME_BYTES:
                    self._protocol_failure("PM-QoS control frame exceeds limit")
                    reason = "protocol_error"
                    break
                while b"\n" in self.buffer:
                    raw, _, remainder = self.buffer.partition(b"\n")
                    self.buffer[:] = remainder
                    try:
                        message = json.loads(raw)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        self._protocol_failure("invalid PM-QoS JSON frame")
                        reason = "protocol_error"
                        return reason
                    if not isinstance(message, dict):
                        self._protocol_failure("PM-QoS frame is not an object")
                        reason = "protocol_error"
                        return reason
                    if not self._handle(message):
                        reason = "protocol_error"
                        return reason
        except OSError as exc:
            self.last_error = f"PM-QoS control I/O failed: {exc}"
            reason = "control_error"
        finally:
            self.abort(reason)
            self._notify("closed", reason=reason)
        return reason


def redacted_command(command: Sequence[str]) -> list[str]:
    """Retain reproducibility without writing common CLI secrets to status."""
    result: list[str] = []
    redact_next = False
    for argument in command:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        option, separator, _value = argument.partition("=")
        if option in SECRET_OPTIONS:
            if separator:
                result.append(f"{option}=<redacted>")
            else:
                result.append(argument)
                redact_next = True
            continue
        result.append(argument)
    return result


def child_identity(user: str, *, pass_env: Sequence[str] = ()) \
        -> tuple[pwd.struct_passwd, dict[str, str]]:
    account = pwd.getpwnam(user)
    if account.pw_uid == 0:
        raise PermissionError("the PM-QoS child must be an unprivileged user")
    environment = {
        name: value for name, value in os.environ.items()
        if (name in SAFE_CHILD_ENVIRONMENT or
            name.startswith(SAFE_CHILD_ENV_PREFIXES)) and
        name != CONTROL_FD_ENV
    }
    for name in pass_env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid environment variable name: {name!r}")
        if name == CONTROL_FD_ENV:
            raise ValueError(f"{CONTROL_FD_ENV} is reserved by the holder")
        if name not in os.environ:
            raise ValueError(f"environment variable is not set: {name}")
        environment[name] = os.environ[name]
    environment.update({"HOME": account.pw_dir, "USER": account.pw_name,
                        "LOGNAME": account.pw_name})
    return account, environment


def restore_child_signal_state(mask: set[signal.Signals]) -> None:
    """Undo holder handlers and its setup-time block before child exec."""
    # Python-level handlers survive fork until exec. If PDEATHSIG becomes
    # pending while TERM is blocked, unblocking first would run the holder's
    # nonterminating forwarding closure inside the child and consume the only
    # parent-death notification. Restore dispositions before the mask.
    for signum in HOLDER_SIGNALS:
        signal.signal(signum, signal.SIG_DFL)
    # Managed termination signals must be usable even if the shell which
    # launched the holder already had one blocked.  Preserve unrelated mask
    # policy, but never pass a mask to the strict child that defeats
    # PDEATHSIG or the holder's forwarded cleanup.
    child_mask = set(mask) - set(HOLDER_SIGNALS)
    signal.pthread_sigmask(signal.SIG_SETMASK, child_mask)


def child_setup(account: pwd.struct_passwd, expected_parent_pid: int,
                inherited_mask: set[signal.Signals]):
    def setup() -> None:
        # If the privileged supervisor disappears, do not leave a server alive
        # that still claims the required profile.  The PM-QoS fd itself is also
        # closed by the kernel with its owner.
        if sys.platform.startswith("linux"):
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
                raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG)")
        # Popen inherits the parent's blocked mask. Leaving TERM blocked would
        # defeat both PDEATHSIG and the holder's forwarded cleanup.
        restore_child_signal_state(inherited_mask)
        if sys.platform.startswith("linux") and \
                os.getppid() != expected_parent_pid:
            # A subreaper can adopt this process without giving it ppid 1.
            os.kill(os.getpid(), signal.SIGTERM)
        os.initgroups(account.pw_name, account.pw_gid)
        os.setgid(account.pw_gid)
        os.setuid(account.pw_uid)
    return setup


def acquire_lock(path: Path) -> int:
    path = Path(path)
    if not path.name:
        raise ValueError("lock path must name a file")
    directory_fd = _open_output_directory(path.parent)
    try:
        fd = os.open(path.name, os.O_RDWR | os.O_CREAT |
                     getattr(os, "O_CLOEXEC", 0) |
                     getattr(os, "O_NOFOLLOW", 0) |
                     getattr(os, "O_NONBLOCK", 0), 0o600,
                     dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    try:
        _require_private_regular(fd, "PM-QoS holder lock")
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise
    return fd


def run_scoped(command: Sequence[str], *, user: str, latency_us: int,
               scope: str, max_hold_seconds: float, device: PMQosDevice,
               status_path: Path, events_path: Optional[Path],
               lock_path: Path = DEFAULT_LOCK,
               pass_env: Sequence[str] = ()) -> int:
    status: dict[str, Any] = {
        "schema": "waste.pm_qos_exec.v2", "started_utc": utc_now(),
        "started_monotonic_ns": time.monotonic_ns(),
        "holder_pid": os.getpid(), "holder_euid": os.geteuid(),
        "target_user": user, "device": str(device.path),
        "latency_us": latency_us, "scope": scope, "fd_scoped": True,
        "self_cleaning": True, "sysfs_modified": False,
        "command": redacted_command(command),
        "passed_environment_names": sorted(set(pass_env)),
        "child_started": False,
        "fd_active": False, "fd_open_count": 0, "fd_close_count": 0,
    }
    child: Optional[subprocess.Popen[bytes]] = None
    lock_fd = -1
    supervisor: Optional[RequestSupervisor] = None
    parent_control: Optional[socket.socket] = None
    child_control: Optional[socket.socket] = None
    forwarded_signal: Optional[int] = None
    stopping = False
    forced_child_stop = False
    previous_signal_mask = None
    signals_blocked = False
    handlers_installed = False

    def persist(event: Optional[dict] = None) -> None:
        if supervisor is not None:
            snapshot = supervisor.snapshot()
            status["supervisor"] = snapshot
            status["fd_active"] = bool(device.active)
            # These counters follow actual successful device transitions,
            # independent of delayed, failed, or lost telemetry callbacks.
            status["fd_open_count"] = snapshot["fd_open_count"]
            status["fd_close_count"] = snapshot["fd_close_count"]
        atomic_json(status_path, status)
        if event is not None:
            append_jsonl(events_path, {
                "schema": "waste.pm_qos_event.v1", "timestamp_utc": utc_now(),
                **event})

    def on_event(_name: str, event: dict) -> None:
        persist(event)

    def on_expiration(_request_id: str, _released: dict) -> None:
        nonlocal forced_child_stop
        forced_child_stop = True
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def on_fatal_control(_reason: str) -> None:
        nonlocal forced_child_stop
        forced_child_stop = True
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def forward(signum: int, _frame: object) -> None:
        nonlocal forwarded_signal, stopping
        forwarded_signal = signum
        stopping = True
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    def wait_for_child(*, bounded: bool) -> int:
        assert child is not None
        deadline: Optional[float] = None
        while True:
            if bounded or stopping or forced_child_stop:
                if deadline is None:
                    deadline = time.monotonic() + 5.0
                if time.monotonic() >= deadline:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return child.wait(timeout=5.0)
            try:
                return child.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                continue

    old_handlers = {sig: signal.getsignal(sig) for sig in HOLDER_SIGNALS}
    try:
        if os.geteuid() != 0:
            raise PermissionError("PM-QoS holder must run as root")
        if latency_us != 0:
            raise ValueError("this helper permits only 0 us")
        if scope not in ("child", "requests"):
            raise ValueError("scope must be child or requests")
        if not command:
            raise ValueError("no child command")
        if (not math.isfinite(max_hold_seconds) or
                not 0 < max_hold_seconds <= MAX_HOLD_SECONDS):
            raise ValueError(
                f"maximum PM-QoS hold must be in (0, {MAX_HOLD_SECONDS}]")
        device.validate()
        account, child_env = child_identity(user, pass_env=pass_env)
        lock_fd = acquire_lock(lock_path)
        status["lock_path"] = str(lock_path)
        persist()
        previous_signal_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, set(old_handlers))
        signals_blocked = True
        for sig in old_handlers:
            signal.signal(sig, forward)
        handlers_installed = True

        pass_fds: tuple[int, ...] = ()
        if scope == "child":
            opened = device.acquire(latency_us)
            status.update({"fd_active": True, "fd_open_count": 1,
                           "fd_opened_monotonic_ns": opened})
        else:
            parent_control, child_control = socket.socketpair()
            child_fd = child_control.fileno()
            child_env[CONTROL_FD_ENV] = str(child_fd)
            pass_fds = (child_fd,)

        child = subprocess.Popen(
            list(command), env=child_env, pass_fds=pass_fds,
            preexec_fn=child_setup(account, os.getpid(), previous_signal_mask),
            start_new_session=True)
        status.update({"child_started": True, "child_pid": child.pid,
                       "child_started_monotonic_ns": time.monotonic_ns()})
        if child_control is not None:
            child_control.close()
            child_control = None
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_signal_mask)
        signals_blocked = False
        persist()

        if scope == "requests":
            assert parent_control is not None
            supervisor = RequestSupervisor(
                parent_control, device, latency_us=latency_us,
                max_hold_seconds=max_hold_seconds, status_callback=on_event,
                expiration_callback=on_expiration,
                fatal_control_callback=on_fatal_control,
                should_stop=lambda: (stopping or forced_child_stop or
                                     child.poll() is not None))
            reason = supervisor.serve()
            status["supervisor_exit_reason"] = reason
            if reason in ("eof", "protocol_error", "control_error") and \
                    child.poll() is None:
                # The strict-profile server must not outlive its required
                # control plane and continue under a misleading profile name.
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                forced_child_stop = True

        returncode = wait_for_child(
            bounded=stopping or forced_child_stop)
        status.update({"child_returncode": returncode,
                       "child_ended_monotonic_ns": time.monotonic_ns()})
        if forwarded_signal is not None:
            return 128 + forwarded_signal
        return 128 - returncode if returncode < 0 else returncode
    except Exception as exc:
        status.update({"error": f"{type(exc).__name__}: {exc}",
                       "child_returncode": None})
        return 1
    finally:
        if supervisor is not None and \
                (supervisor.closed_reason is None or device.active):
            supervisor.abort("holder_exit")
        if device.active:
            was_active = bool(device.active)
            device.release()
            if was_active and not device.active:
                if supervisor is not None:
                    supervisor.fd_close_count += 1
                else:
                    status["fd_close_count"] += 1
        status["fd_active"] = False
        if parent_control is not None:
            parent_control.close()
        if child_control is not None:
            child_control.close()
        if child is not None:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    wait_for_child(bounded=True)
                except (OSError, subprocess.SubprocessError):
                    pass
            if child.poll() is not None and "child_ended_monotonic_ns" not in status:
                status.update({"child_returncode": child.returncode,
                               "child_ended_monotonic_ns": time.monotonic_ns()})
        if lock_fd >= 0:
            os.close(lock_fd)
        if signals_blocked:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_signal_mask)
            signals_blocked = False
        if forwarded_signal is not None:
            status["forwarded_signal"] = forwarded_signal
        status.update({"ended_utc": utc_now(),
                       "ended_monotonic_ns": time.monotonic_ns()})
        try:
            persist()
        except Exception as exc:
            print(f"pm_qos_exec: cannot write status: {exc}", file=sys.stderr)
        if handlers_installed:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency-us", type=int, choices=(0,), default=0)
    parser.add_argument("--scope", choices=("child", "requests"),
                        default="child")
    parser.add_argument("--max-hold-seconds", type=float, default=1800.0,
                        help="requests scope safety limit (default: 1800)")
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--events", type=Path,
                        help="optional append-only boundary event log")
    parser.add_argument("--user", required=True)
    parser.add_argument(
        "--pass-env", action="append", default=[], metavar="NAME",
        help="explicitly pass one otherwise-filtered variable (repeatable; "
             "for example --pass-env HF_TOKEN)")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if (not math.isfinite(args.max_hold_seconds) or
            not 0 < args.max_hold_seconds <= MAX_HOLD_SECONDS):
        parser.error(
            f"--max-hold-seconds must be in (0, {MAX_HOLD_SECONDS}]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_scoped(
        args.command, user=args.user, latency_us=args.latency_us,
        scope=args.scope, max_hold_seconds=args.max_hold_seconds,
        device=PMQosDevice(), status_path=args.status,
        events_path=args.events, pass_env=args.pass_env)


if __name__ == "__main__":
    raise SystemExit(main())
