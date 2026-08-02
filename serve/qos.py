# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Client for a root-held, request-scoped Linux PM-QoS lease.

The privileged holder creates a private socketpair before dropping the server
to its ordinary account.  Only the control socket reaches this process.  A
BEGIN/END pair brackets engine work; the ``/dev/cpu_dma_latency`` descriptor
always remains in the root supervisor.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
import socket
import struct
import sys
import threading
from typing import Iterator, Mapping, MutableMapping, Optional


CONTROL_FD_ENV = "WASTE_PM_QOS_CONTROL_FD"
PROTOCOL = "waste.pm_qos.requests.v1"
MAX_FRAME_BYTES = 16 * 1024
MAX_HOLD_SECONDS = 4 * 60 * 60


class QosError(RuntimeError):
    """The required PM-QoS supervisor is missing or rejected a lease."""


class _JsonSocket:
    def __init__(self, sock: socket.socket, timeout: float):
        self.sock = sock
        self.sock.settimeout(timeout)
        self.buffer = bytearray()

    def send(self, value: dict) -> None:
        data = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        if len(data) + 1 > MAX_FRAME_BYTES:
            raise QosError("PM-QoS control frame is too large")
        try:
            self.sock.sendall(data + b"\n")
        except OSError as exc:
            raise QosError(f"PM-QoS control send failed: {exc}") from exc

    def receive(self) -> dict:
        while b"\n" not in self.buffer:
            try:
                chunk = self.sock.recv(4096)
            except (OSError, TimeoutError) as exc:
                raise QosError(f"PM-QoS control receive failed: {exc}") from exc
            if not chunk:
                raise QosError("PM-QoS supervisor closed the control channel")
            self.buffer.extend(chunk)
            if len(self.buffer) > MAX_FRAME_BYTES:
                raise QosError("PM-QoS control frame exceeds the limit")
        raw, _, remainder = self.buffer.partition(b"\n")
        self.buffer[:] = remainder
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise QosError(f"invalid PM-QoS supervisor response: {exc}") from exc
        if not isinstance(value, dict):
            raise QosError("PM-QoS supervisor response is not an object")
        return value


@dataclass
class QosLease:
    request_id: str
    requested: bool = True
    acquired: bool = False
    released: bool = False
    expired: bool = False
    latency_us: Optional[int] = None
    acquired_monotonic_ns: Optional[int] = None
    released_monotonic_ns: Optional[int] = None
    hold_ms: Optional[float] = None
    error: Optional[str] = None

    def report(self) -> dict:
        return {
            "requested": self.requested,
            "acquired": self.acquired,
            "released": self.released,
            "expired": self.expired,
            "latency_us": self.latency_us,
            "acquired_monotonic_ns": self.acquired_monotonic_ns,
            "released_monotonic_ns": self.released_monotonic_ns,
            "hold_ms": self.hold_ms,
            "error": self.error,
        }


class DisabledRequestQos:
    """The default server policy: no PM-QoS request and no IPC."""

    @contextmanager
    def scope(self, request_id: str) -> Iterator[QosLease]:
        del request_id
        yield QosLease(request_id="", requested=False)

    def stats(self) -> dict:
        return {
            "enabled": False, "required": False, "connected": False,
            "active": False, "latency_us": None, "leases_started": 0,
            "leases_completed": 0, "lease_errors": 0, "last_error": None,
        }

    def close(self) -> None:
        pass


class RequestQos:
    """Synchronous request lease client over a private inherited socket."""

    def __init__(self, sock: socket.socket, *, timeout: float = 5.0,
                 required: bool = True, require_root_peer: bool = True):
        if timeout <= 0:
            raise ValueError("PM-QoS control timeout must be positive")
        os.set_inheritable(sock.fileno(), False)
        self._channel = _JsonSocket(sock, timeout)
        self._required = required
        self._scope_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._closed = False
        self._active = False
        self._latency_us: Optional[int] = None
        self._max_hold_seconds: Optional[float] = None
        self._started = 0
        self._completed = 0
        self._errors = 0
        self._last_error: Optional[str] = None
        self._last_lease: Optional[dict] = None
        if require_root_peer and sys.platform.startswith("linux"):
            try:
                credentials = sock.getsockopt(
                    socket.SOL_SOCKET, socket.SO_PEERCRED,
                    struct.calcsize("=3i"))
                _pid, peer_uid, _gid = struct.unpack("=3i", credentials)
            except (AttributeError, OSError, struct.error) as exc:
                self._poison("cannot authenticate PM-QoS supervisor")
                raise QosError(
                    "cannot authenticate PM-QoS supervisor as root") from exc
            if peer_uid != 0:
                self._poison("PM-QoS supervisor peer is not root")
                raise QosError("PM-QoS supervisor peer is not root")
        reply = self._round_trip({"op": "hello", "protocol": PROTOCOL})
        if reply.get("protocol") != PROTOCOL or reply.get("scope") != "requests":
            self._poison("PM-QoS supervisor speaks an incompatible protocol")
            raise QosError("PM-QoS supervisor speaks an incompatible protocol")
        try:
            latency = reply.get("latency_us")
            if type(latency) is not int:
                raise TypeError
            self._latency_us = latency
            self._max_hold_seconds = float(reply.get("max_hold_seconds", 0))
        except (TypeError, ValueError, OverflowError) as exc:
            self._poison("PM-QoS supervisor returned invalid lease limits")
            raise QosError(
                "PM-QoS supervisor returned invalid lease limits") from exc
        if (self._latency_us != 0 or
                not math.isfinite(self._max_hold_seconds) or
                not 0 < self._max_hold_seconds <= MAX_HOLD_SECONDS):
            self._poison("PM-QoS supervisor did not offer bounded Q0 leases")
            raise QosError("PM-QoS supervisor did not offer bounded Q0 leases")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ, *,
                 required: bool = False, timeout: float = 5.0):
        value = environ.get(CONTROL_FD_ENV)
        if value is None:
            if required:
                raise QosError(
                    f"required request-scoped PM-QoS control is missing; "
                    f"launch through tools/pm_qos_exec.py --scope requests")
            return DisabledRequestQos()
        if isinstance(environ, MutableMapping):
            # The control descriptor must not become ambient authority in a
            # subprocess created after the server has adopted it.
            environ.pop(CONTROL_FD_ENV, None)
        try:
            fd = int(value, 10)
            if fd < 0:
                raise ValueError
        except ValueError:
            raise QosError(f"{CONTROL_FD_ENV} is not a valid descriptor") from None
        try:
            sock = socket.socket(fileno=fd)
        except OSError as exc:
            raise QosError(f"cannot adopt PM-QoS control descriptor: {exc}") from exc
        try:
            return cls(sock, timeout=timeout, required=required)
        except BaseException:
            sock.close()
            raise

    def _record_error(self, message: str) -> None:
        with self._state_lock:
            self._errors += 1
            self._last_error = message

    def _poison(self, message: str) -> None:
        """Close an ambiguous channel so the root holder releases on EOF.

        A timeout after BEGIN may mean the device was acquired and only the
        acknowledgement was lost.  The client must not guess.  Closing the
        socket is the protocol's rollback signal and also prevents a strict
        profile from continuing over a desynchronised request/reply stream.
        This deliberately does not take ``_scope_lock``: it is called while a
        scope already owns that non-reentrant lock.
        """
        with self._state_lock:
            self._last_error = message
            if self._closed:
                return
            self._closed = True
        try:
            self._channel.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._channel.sock.close()

    def _round_trip(self, request: dict) -> dict:
        if self._closed:
            raise QosError("PM-QoS control channel is closed")
        try:
            self._channel.send(request)
            reply = self._channel.receive()
        except QosError as exc:
            self._poison(str(exc))
            raise
        expected_op = request.get("op")
        expected_request_id = request.get("request_id")
        if reply.get("op") != expected_op:
            message = "PM-QoS acknowledgement operation mismatch"
            self._poison(message)
            raise QosError(message)
        if (expected_request_id is not None and
                reply.get("request_id") != expected_request_id):
            message = "PM-QoS acknowledgement request id mismatch"
            self._poison(message)
            raise QosError(message)
        if reply.get("ok") is False:
            message = str(reply.get("error") or "PM-QoS request rejected")
            raise QosError(message)
        if reply.get("ok") is not True:
            message = "malformed PM-QoS supervisor acknowledgement"
            self._poison(message)
            raise QosError(message)
        return reply

    @contextmanager
    def scope(self, request_id: str) -> Iterator[QosLease]:
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise QosError("PM-QoS request id must be 1..128 characters")
        lease = QosLease(request_id=request_id, latency_us=self._latency_us)
        with self._scope_lock:
            try:
                reply = self._round_trip({"op": "begin", "request_id": request_id})
                acquired_ns = reply.get("acquired_monotonic_ns")
                if type(acquired_ns) is not int or acquired_ns <= 0:
                    message = "invalid PM-QoS BEGIN timestamp"
                    self._poison(message)
                    raise QosError(message)
                lease.acquired = True
                lease.acquired_monotonic_ns = acquired_ns
                with self._state_lock:
                    self._active = True
                    self._started += 1
                yield lease
            except QosError as exc:
                lease.error = str(exc)
                self._record_error(lease.error)
                raise
            finally:
                if lease.acquired:
                    active_exception = sys.exc_info()[0] is not None
                    try:
                        reply = self._round_trip(
                            {"op": "end", "request_id": request_id})
                        released_ns = reply.get("released_monotonic_ns")
                        hold_ms = reply.get("hold_ms")
                        if (type(released_ns) is not int or released_ns <= 0 or
                                released_ns < lease.acquired_monotonic_ns or
                                isinstance(hold_ms, bool) or
                                not isinstance(hold_ms, (int, float)) or
                                not math.isfinite(float(hold_ms)) or
                                float(hold_ms) < 0):
                            message = "invalid PM-QoS END evidence"
                            self._poison(message)
                            raise QosError(message)
                        lease.released = True
                        lease.released_monotonic_ns = released_ns
                        lease.hold_ms = float(hold_ms)
                        with self._state_lock:
                            self._completed += 1
                    except QosError as exc:
                        lease.error = str(exc)
                        lease.expired = "expired" in lease.error.lower()
                        self._record_error(lease.error)
                        if not active_exception:
                            raise
                    finally:
                        with self._state_lock:
                            self._active = False
                            self._last_lease = lease.report()

    def stats(self) -> dict:
        with self._state_lock:
            return {
                "enabled": True,
                "required": self._required,
                "connected": not self._closed,
                "active": self._active,
                "latency_us": self._latency_us,
                "max_hold_seconds": self._max_hold_seconds,
                "leases_started": self._started,
                "leases_completed": self._completed,
                "lease_errors": self._errors,
                "last_error": self._last_error,
                "last_lease": dict(self._last_lease) if self._last_lease else None,
            }

    def close(self) -> None:
        self._poison("PM-QoS client closed")


def discard_control_from_env(
        environ: MutableMapping[str, str] = os.environ) -> None:
    """Remove and close an ambient control fd for a non-profile server."""
    value = environ.pop(CONTROL_FD_ENV, None)
    if value is None:
        return
    try:
        fd = int(value, 10)
        if fd < 0:
            return
        sock = socket.socket(fileno=fd)
    except (OSError, ValueError):
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()
