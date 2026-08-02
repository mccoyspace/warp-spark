# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Profile resolution and request-scoped PM-QoS protocol tests.

No test opens the host's real latency device.  The device syscalls and the
root-side lease object are injected; the protocol still runs over a real
socketpair so framing, EOF cleanup, exceptions, and the hold deadline are
exercised end to end.
"""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
import os
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from serve.profiles import (ProfileError, SPARK_Q0_ENVIRONMENT,  # noqa: E402
                            resolve_profile)
from serve.qos import (PROTOCOL, QosError, RequestQos,            # noqa: E402
                       discard_control_from_env)
from tools.pm_qos_exec import (HOLDER_SIGNALS, PMQosDevice,       # noqa: E402
                               RequestSupervisor, atomic_json, child_identity,
                               parse_args, redacted_command,
                               restore_child_signal_state, run_scoped,
                               validate_artifact_paths)


class FakeDevice:
    def __init__(self, *, acquire_error=None, acquire_delay=0.0):
        self.path = Path("/dev/fake-cpu-dma-latency")
        self.active = False
        self.acquire_error = acquire_error
        self.acquire_delay = acquire_delay
        self.acquires = 0
        self.releases = 0
        self.released = threading.Event()

    def validate(self):
        pass

    def acquire(self, latency_us):
        if self.acquire_delay:
            time.sleep(self.acquire_delay)
        if self.acquire_error is not None:
            raise self.acquire_error
        if self.active:
            raise RuntimeError("already active")
        self.active = True
        self.acquires += 1
        return time.monotonic_ns()

    def release(self):
        if not self.active:
            return None
        self.active = False
        self.releases += 1
        self.released.set()
        return time.monotonic_ns()


def close_error_device(message="injected close failure after close"):
    closed = threading.Event()

    def fail_after_close(_fd):
        closed.set()
        raise OSError(message)

    device = PMQosDevice(
        stat_fn=lambda path, follow_symlinks: SimpleNamespace(
            st_mode=stat.S_IFCHR | 0o600),
        open_fn=lambda path, flags: 73,
        write_fn=lambda fd, value: len(value),
        close_fn=fail_after_close)
    return device, closed


class BlockingAckSocket:
    """Real control socket whose selected response can never make progress."""

    def __init__(self, sock, operation):
        self.sock = sock
        self.marker = f'"op":"{operation}"'.encode()
        self.blocked = threading.Event()

    def send(self, payload):
        if self.marker in bytes(payload):
            self.blocked.set()
            raise BlockingIOError("injected acknowledgement backpressure")
        return self.sock.send(payload)

    def __getattr__(self, name):
        return getattr(self.sock, name)


class BrokenAckSocket(BlockingAckSocket):
    """Real control socket with an immediately lost selected ACK."""

    def send(self, payload):
        if self.marker in bytes(payload):
            self.blocked.set()
            raise BrokenPipeError("injected acknowledgement channel loss")
        return self.sock.send(payload)


class ProtocolHarness:
    def __init__(self, *, max_hold=1.0, device=None, timeout=2.0,
                 supervisor_class=RequestSupervisor, status_callback=None,
                 expiration_callback=None, fatal_control_callback=None):
        self.device = device or FakeDevice()
        self.supervisor_socket, client_socket = socket.socketpair()
        self.supervisor = supervisor_class(
            self.supervisor_socket, self.device, latency_us=0,
            max_hold_seconds=max_hold, status_callback=status_callback,
            expiration_callback=expiration_callback,
            fatal_control_callback=fatal_control_callback)
        self.thread = threading.Thread(target=self.supervisor.serve,
                                       daemon=True)
        self.thread.start()
        self.client = RequestQos(
            client_socket, timeout=timeout, require_root_peer=False)

    def close(self):
        self.client.close()
        self.thread.join(timeout=2)
        self.supervisor_socket.close()
        if self.thread.is_alive():
            raise AssertionError("PM-QoS supervisor did not stop at EOF")


class TestProfiles(unittest.TestCase):
    def test_default_profile_preserves_cli_values_and_environment(self):
        environment = {
            "WASTE_IO_THREADS": "3", "WASTE_IO_DEPTH": "7",
            "WASTE_LOOKAHEAD": "17", "UNRELATED": "kept",
        }
        profile = resolve_profile(
            None, threads=None, cache="lru", no_direct_io=True, verify=True,
            environ=environment)
        self.assertEqual(profile.name, "default")
        self.assertEqual(profile.threads, 0)
        self.assertEqual(profile.cache, "lru")
        self.assertFalse(profile.direct_io)
        self.assertTrue(profile.verify_records)
        self.assertEqual(environment, {
            "WASTE_IO_THREADS": "3", "WASTE_IO_DEPTH": "7",
            "WASTE_LOOKAHEAD": "17", "UNRELATED": "kept",
        })
        self.assertEqual(profile.public()["storage"], {
            "requested_read_ahead_threads": 3,
            "requested_read_ahead_depth": 7,
            "requested_router_lookahead": 17,
            "effective_configuration_reported": False})

    def test_default_profile_reports_malformed_and_clamped_storage_env(self):
        profile = resolve_profile(
            None, threads=None, cache="lfru", no_direct_io=False,
            verify=False,
            environ={"WASTE_IO_THREADS": "abc",
                     "WASTE_IO_DEPTH": "-5 trailing",
                     "WASTE_LOOKAHEAD": "-9"})
        self.assertEqual(profile.public()["storage"], {
            "requested_read_ahead_threads": 0,
            "requested_read_ahead_depth": 0,
            "requested_router_lookahead": 0,
            "effective_configuration_reported": False})

        clamped = resolve_profile(
            None, threads=None, cache="lfru", no_direct_io=False,
            verify=False,
            environ={"WASTE_IO_THREADS": "7",
                     "WASTE_IO_DEPTH": "2",
                     "WASTE_LOOKAHEAD": "999"})
        self.assertEqual(clamped.public()["storage"], {
            "requested_read_ahead_threads": 7,
            "requested_read_ahead_depth": 7,
            "requested_router_lookahead": 64,
            "effective_configuration_reported": False})

    def test_spark_q0_is_an_exact_current_upstream_configuration(self):
        environment = {"UNRELATED": "kept"}
        profile = resolve_profile(
            "spark-q0", threads=8, cache="lfru", no_direct_io=False,
            verify=False, environ=environment)
        self.assertEqual(profile.threads, 8)
        self.assertTrue(profile.request_qos_required)
        self.assertEqual(profile.request_qos_us, 0)
        self.assertEqual(profile.environment, SPARK_Q0_ENVIRONMENT)
        self.assertEqual(environment["UNRELATED"], "kept")
        for key, value in SPARK_Q0_ENVIRONMENT.items():
            self.assertEqual(environment[key], value)
        public = profile.public()
        self.assertEqual(public["storage"], {
            "requested_read_ahead_threads": 2,
            "requested_read_ahead_depth": 2,
            "requested_router_lookahead": 0,
            "effective_configuration_reported": False})
        self.assertEqual(public["cpu_affinity"], {
            "managed": False, "effective_cpu_list": None})

    def test_spark_q0_rejects_cli_or_environment_drift_atomically(self):
        environment = {"WASTE_IO_DEPTH": "4"}
        with self.assertRaisesRegex(ProfileError, "WASTE_IO_DEPTH"):
            resolve_profile(
                "spark-q0", threads=6, cache="lfru", no_direct_io=False,
                verify=False, environ=environment)
        self.assertEqual(environment, {"WASTE_IO_DEPTH": "4"})

    def test_spark_q0_rejects_undeclared_engine_or_debug_environment(self):
        for key in ("WASTE_MLOCK", "WASTE_DUMP_HIDDEN", "WASTE_VIS_STAGE",
                    "WASTE_LIB"):
            with self.subTest(key=key), self.assertRaisesRegex(
                    ProfileError, key):
                resolve_profile(
                    "spark-q0", threads=8, cache="lfru",
                    no_direct_io=False, verify=False, environ={key: "1"})

    def test_spark_q0_allows_hidden_service_environment(self):
        environment = {
            "WASTE_API_KEY": "do-not-report",
            "WASTE_PM_QOS_CONTROL_FD": "17",
        }
        profile = resolve_profile(
            "spark-q0", threads=8, cache="lfru", no_direct_io=False,
            verify=False, environ=environment)
        public = profile.public()
        self.assertNotIn("do-not-report", repr(public))
        self.assertNotIn("17", repr(public))
        self.assertEqual(environment["WASTE_API_KEY"], "do-not-report")
        self.assertEqual(environment["WASTE_PM_QOS_CONTROL_FD"], "17")

    def test_required_control_channel_cannot_be_omitted(self):
        with self.assertRaisesRegex(QosError, "required.*missing"):
            RequestQos.from_env({}, required=True)

    def test_default_profile_discards_an_ambient_control_descriptor(self):
        supervisor, inherited = socket.socketpair()
        descriptor = inherited.detach()
        environment = {"WASTE_PM_QOS_CONTROL_FD": str(descriptor)}
        try:
            discard_control_from_env(environment)
            self.assertNotIn("WASTE_PM_QOS_CONTROL_FD", environment)
            self.assertEqual(supervisor.recv(1), b"")
        finally:
            supervisor.close()


class TestInjectedDevice(unittest.TestCase):
    def test_exact_device_open_write_and_close_are_scoped(self):
        calls = []

        def fake_open(path, flags):
            calls.append(("open", path, flags))
            return 41

        def fake_write(fd, value):
            calls.append(("write", fd, value))
            return len(value)

        def fake_close(fd):
            calls.append(("close", fd))

        device = PMQosDevice(
            stat_fn=lambda path, follow_symlinks: SimpleNamespace(
                st_mode=stat.S_IFCHR | 0o600),
            open_fn=fake_open, write_fn=fake_write, close_fn=fake_close)
        device.acquire(0)
        self.assertTrue(device.active)
        self.assertEqual(calls[1], ("write", 41, struct.pack("=i", 0)))
        device.release()
        self.assertFalse(device.active)
        self.assertEqual(calls[-1], ("close", 41))

    def test_short_device_write_closes_the_partial_acquisition(self):
        closed = []
        device = PMQosDevice(
            stat_fn=lambda path, follow_symlinks: SimpleNamespace(
                st_mode=stat.S_IFCHR | 0o600),
            open_fn=lambda path, flags: 42,
            write_fn=lambda fd, value: len(value) - 1,
            close_fn=closed.append)
        with self.assertRaisesRegex(OSError, "short write"):
            device.acquire(0)
        self.assertEqual(closed, [42])
        self.assertFalse(device.active)

    def test_release_reports_close_error_after_logical_close(self):
        device, closed = close_error_device()
        device.acquire(0)
        released_ns = device.release()
        self.assertTrue(closed.is_set())
        self.assertFalse(device.active)
        self.assertIsInstance(released_ns, int)
        self.assertIn("injected close failure", device.take_release_error())
        self.assertIsNone(device.take_release_error())

    def test_status_write_does_not_follow_predictable_or_final_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            victim = directory / "victim"
            victim.write_text("must survive\n")
            status_path = directory / "status.json"
            old_temporary = directory / f".status.json.{os.getpid()}.tmp"
            old_temporary.symlink_to(victim)
            status_path.symlink_to(victim)

            atomic_json(status_path, {"safe": True})

            self.assertEqual(victim.read_text(), "must survive\n")
            self.assertFalse(status_path.is_symlink())
            self.assertEqual(json.loads(status_path.read_text()), {"safe": True})
            self.assertEqual(stat.S_IMODE(status_path.stat().st_mode), 0o600)

    def test_artifact_paths_reject_normalized_and_inode_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            status_path = directory / "status.json"
            events_path = directory / "events.jsonl"
            lock_path = directory / "holder.lock"

            for status, events, lock in (
                    (status_path, events_path, status_path),
                    (status_path, status_path, lock_path),
                    (status_path, directory / "nested" / ".." /
                     "status.json", lock_path)):
                with self.subTest(status=status, events=events, lock=lock), \
                        self.assertRaisesRegex(ValueError, "collide"):
                    validate_artifact_paths(status, events, lock)

            status_path.write_text("sentinel\n")
            os.link(status_path, events_path)
            with self.assertRaisesRegex(ValueError, "share an inode"):
                validate_artifact_paths(status_path, events_path, lock_path)
            self.assertEqual(status_path.read_text(), "sentinel\n")
            self.assertEqual(events_path.read_text(), "sentinel\n")

    def test_artifact_paths_canonicalize_leading_slashes_and_missing_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            missing = directory / "not-created" / "status.json"
            single = Path("/" + os.fspath(missing).lstrip("/"))
            for prefix in ("//", "///", "////"):
                alias = Path(prefix + os.fspath(missing).lstrip("/"))
                with self.subTest(alias=alias), self.assertRaisesRegex(
                        ValueError, "collide"):
                    validate_artifact_paths(single, None, alias)

            real_parent = directory / "real"
            real_parent.mkdir()
            alias_parent = directory / "alias"
            alias_parent.symlink_to(real_parent, target_is_directory=True)
            status = real_parent / "missing" / "same.json"
            lock = alias_parent / "missing" / "same.json"
            with self.assertRaisesRegex(ValueError, "collide"):
                validate_artifact_paths(status, None, lock)

    def test_colliding_artifacts_fail_before_any_path_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            shared = directory / "shared"
            with redirect_stderr(io.StringIO()):
                result = run_scoped(
                    ["must-not-start"], user="unused", latency_us=0,
                    scope="requests", max_hold_seconds=1,
                    device=FakeDevice(), status_path=shared,
                    events_path=None, lock_path=shared)
            self.assertEqual(result, 1)
            self.assertFalse(shared.exists())

            status_path = directory / "status.json"
            lock_path = directory / "holder.lock"
            with redirect_stderr(io.StringIO()):
                result = run_scoped(
                    ["must-not-start"], user="unused", latency_us=0,
                    scope="requests", max_hold_seconds=1,
                    device=FakeDevice(), status_path=status_path,
                    events_path=status_path, lock_path=lock_path)
            self.assertEqual(result, 1)
            self.assertFalse(status_path.exists())
            self.assertFalse(lock_path.exists())


class TestHelperInputs(unittest.TestCase):
    def test_child_environment_is_allowlisted_with_explicit_secret_opt_in(self):
        account = SimpleNamespace(
            pw_uid=1000, pw_gid=1000, pw_name="spark", pw_dir="/home/spark")
        source = {
            "PATH": "/usr/bin", "WASTE_Q8": "1", "HF_TOKEN": "wanted",
            "AWS_SECRET_ACCESS_KEY": "must-not-leak",
            "SSH_AUTH_SOCK": "/tmp/agent",
            "WASTE_PM_QOS_CONTROL_FD": "99",
        }
        with patch("tools.pm_qos_exec.pwd.getpwnam", return_value=account), \
                patch.dict(os.environ, source, clear=True):
            _, environment = child_identity("spark", pass_env=["HF_TOKEN"])
        self.assertEqual(environment["HF_TOKEN"], "wanted")
        self.assertEqual(environment["WASTE_Q8"], "1")
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", environment)
        self.assertNotIn("SSH_AUTH_SOCK", environment)
        self.assertNotIn("WASTE_PM_QOS_CONTROL_FD", environment)
        self.assertEqual(environment["HOME"], "/home/spark")

    def test_root_child_is_rejected(self):
        account = SimpleNamespace(
            pw_uid=0, pw_gid=0, pw_name="root", pw_dir="/root")
        with patch("tools.pm_qos_exec.pwd.getpwnam", return_value=account):
            with self.assertRaisesRegex(PermissionError, "unprivileged"):
                child_identity("root")

    def test_status_command_redacts_common_secret_options(self):
        self.assertEqual(
            redacted_command([
                "python3", "-m", "serve", "model", "--api-key", "secret",
                "--token=also-secret", "--port", "8000"]),
            ["python3", "-m", "serve", "model", "--api-key", "<redacted>",
             "--token=<redacted>", "--port", "8000"])

    def test_nonfinite_or_excessive_hold_is_rejected_by_cli(self):
        base = ["--status", "status.json", "--user", "spark", "--",
                "true"]
        for value in ("nan", "inf", "14401"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                parse_args(["--max-hold-seconds", value, *base])

    def test_child_restores_mask_and_terminates_on_forwarded_term(self):
        managed = {signal.SIGHUP, signal.SIGINT, signal.SIGTERM}
        original = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        baseline = original - managed
        child = None
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, baseline)
            previous = signal.pthread_sigmask(signal.SIG_BLOCK, managed)
            child = subprocess.Popen(
                [sys.executable, "-c",
                 "import json,signal,time; "
                 "print(json.dumps(sorted(int(s) for s in "
                "signal.pthread_sigmask(signal.SIG_BLOCK,set()))),flush=True); "
                 "time.sleep(10)"],
                stdout=subprocess.PIPE, text=True, start_new_session=True,
                preexec_fn=lambda: restore_child_signal_state(previous))
            reported = set(json.loads(child.stdout.readline()))
            self.assertEqual(reported, {int(item) for item in baseline})
            os.killpg(child.pid, signal.SIGTERM)
            self.assertEqual(child.wait(timeout=2), -signal.SIGTERM)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, original)
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=2)
            if child is not None and child.stdout is not None:
                child.stdout.close()

    def test_child_resets_inherited_term_handler_before_unblocking(self):
        managed = {signal.SIGHUP, signal.SIGINT, signal.SIGTERM}
        original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        original_handler = signal.getsignal(signal.SIGTERM)
        child = None

        def inherited_holder_handler(signum, frame):
            del signum, frame

        try:
            signal.signal(signal.SIGTERM, inherited_holder_handler)
            baseline = original_mask - managed
            signal.pthread_sigmask(signal.SIG_SETMASK, baseline)
            previous = signal.pthread_sigmask(signal.SIG_BLOCK, managed)

            def die_before_exec():
                restore_child_signal_state(previous)
                os.kill(os.getpid(), signal.SIGTERM)

            child = subprocess.Popen(
                [sys.executable, "-c", "raise SystemExit(99)"],
                preexec_fn=die_before_exec)
            self.assertEqual(child.wait(timeout=2), -signal.SIGTERM)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)
            signal.signal(signal.SIGTERM, original_handler)
            if child is not None and child.poll() is None:
                child.kill()
                child.wait(timeout=2)

    def test_child_unblocks_managed_signals_preblocked_before_holder(self):
        managed = set(HOLDER_SIGNALS)
        preserved = {signal.SIGUSR1}
        inherited = managed | preserved
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import json,signal,time; "
             "print(json.dumps(sorted(int(s) for s in "
             "signal.pthread_sigmask(signal.SIG_BLOCK,set()))),flush=True); "
             "time.sleep(10)"],
            stdout=subprocess.PIPE, text=True, start_new_session=True,
            preexec_fn=lambda: restore_child_signal_state(inherited))
        try:
            reported = set(json.loads(child.stdout.readline()))
            self.assertTrue(reported.isdisjoint(
                {int(item) for item in managed}))
            self.assertIn(int(signal.SIGUSR1), reported)
            os.killpg(child.pid, signal.SIGTERM)
            self.assertEqual(child.wait(timeout=2), -signal.SIGTERM)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=2)
            if child.stdout is not None:
                child.stdout.close()

    def test_post_spawn_status_failure_terminates_and_reaps_child(self):
        class FakeChild:
            pid = 32123

            def __init__(self):
                self.returncode = None
                self.wait_calls = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                del timeout
                self.wait_calls += 1
                self.returncode = -signal.SIGTERM
                return self.returncode

        child = FakeChild()
        account = SimpleNamespace(
            pw_uid=1000, pw_gid=1000, pw_name="spark", pw_dir="/home/spark")
        writes = 0

        def status_write(path, value):
            nonlocal writes
            del path, value
            writes += 1
            if writes == 2:  # after Popen, before the supervisor starts
                raise OSError("injected status failure")

        lock_fd = os.open(os.devnull, os.O_RDONLY)
        with patch("tools.pm_qos_exec.os.geteuid", return_value=0), \
                patch("tools.pm_qos_exec.child_identity",
                      return_value=(account, {"PATH": "/usr/bin"})), \
                patch("tools.pm_qos_exec.acquire_lock", return_value=lock_fd), \
                patch("tools.pm_qos_exec.atomic_json", side_effect=status_write), \
                patch("tools.pm_qos_exec.subprocess.Popen", return_value=child), \
                patch("tools.pm_qos_exec.signal.pthread_sigmask",
                      return_value=set()), \
                patch("tools.pm_qos_exec.signal.signal"), \
                patch("tools.pm_qos_exec.os.killpg") as killpg:
            result = run_scoped(
                ["fake-server"], user="spark", latency_us=0,
                scope="requests", max_hold_seconds=1, device=FakeDevice(),
                status_path=Path("status.json"), events_path=None,
                lock_path=Path("holder.lock"))
        self.assertEqual(result, 1)
        killpg.assert_called_with(child.pid, signal.SIGTERM)
        self.assertGreaterEqual(child.wait_calls, 1)
        self.assertEqual(child.returncode, -signal.SIGTERM)


class TestRequestProtocol(unittest.TestCase):
    def _ack_loss(self, operation):
        device = FakeDevice()
        events = []
        parent_socket, client_socket = socket.socketpair()
        blocked_socket = BlockingAckSocket(parent_socket, operation)

        def observe(event, payload):
            # Disk/event telemetry must only run after the real fd is closed.
            events.append((event, device.active, payload))

        def terminate_child(reason):
            events.append(("fatal_control", device.active,
                           {"reason": reason}))

        supervisor = RequestSupervisor(
            blocked_socket, device, latency_us=0, max_hold_seconds=0.2,
            send_timeout_seconds=0.03, status_callback=observe,
            fatal_control_callback=terminate_child)
        thread = threading.Thread(target=supervisor.serve, daemon=True)
        thread.start()
        client = RequestQos(
            client_socket, timeout=0.15, require_root_peer=False)
        return (client, parent_socket, blocked_socket, device, supervisor,
                thread, events)

    def test_begin_end_acknowledges_a_released_lease(self):
        harness = ProtocolHarness()
        try:
            with harness.client.scope("chatcmpl-test") as lease:
                self.assertTrue(harness.device.active)
                self.assertTrue(lease.acquired)
            self.assertFalse(harness.device.active)
            self.assertTrue(lease.released)
            self.assertIsNotNone(lease.hold_ms)
            stats = harness.client.stats()
            self.assertEqual(stats["leases_started"], 1)
            self.assertEqual(stats["leases_completed"], 1)
        finally:
            harness.close()

    def test_body_exception_still_releases_the_device(self):
        harness = ProtocolHarness()
        try:
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                with harness.client.scope("chatcmpl-error"):
                    raise RuntimeError("inference failed")
            self.assertFalse(harness.device.active)
            self.assertEqual(harness.device.releases, 1)
        finally:
            harness.close()

    def test_max_hold_releases_and_marks_the_response_invalid(self):
        harness = ProtocolHarness(max_hold=0.02)
        try:
            with self.assertRaisesRegex(QosError, "expired"):
                with harness.client.scope("chatcmpl-slow"):
                    time.sleep(0.06)
            self.assertFalse(harness.device.active)
            self.assertEqual(harness.supervisor.lease_expirations, 1)
            self.assertEqual(harness.client.stats()["lease_errors"], 1)
        finally:
            harness.close()

    def test_blocked_acquire_telemetry_cannot_extend_maximum_hold(self):
        callback_started = threading.Event()
        child_cleanup_started = threading.Event()
        unblock_callback = threading.Event()
        callbacks = []
        ordering = []

        def block_acquire(event, payload):
            if event == "acquire":
                ordering.append("telemetry")
                callback_started.set()
                unblock_callback.wait(timeout=1)
            callbacks.append((event, payload))

        def terminate_child(_request_id, _released):
            ordering.append("child_cleanup")
            child_cleanup_started.set()

        harness = ProtocolHarness(
            max_hold=0.02, timeout=0.5, status_callback=block_acquire,
            expiration_callback=terminate_child)
        entered = threading.Event()
        finish = threading.Event()
        outcome = []

        def worker():
            try:
                with harness.client.scope("chatcmpl-blocked-telemetry"):
                    entered.set()
                    finish.wait(timeout=1)
            except QosError as exc:
                outcome.append(str(exc))

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        try:
            self.assertTrue(entered.wait(timeout=0.5))
            self.assertTrue(harness.device.released.wait(timeout=0.25))
            self.assertFalse(harness.device.active)
            self.assertTrue(child_cleanup_started.wait(timeout=0.1))
            self.assertTrue(callback_started.wait(timeout=0.1))
            self.assertEqual(ordering[:2], ["child_cleanup", "telemetry"])
            # The callback is blocked now, but only after the fd transition.
            self.assertEqual(harness.supervisor.fd_open_count, 1)
            self.assertEqual(harness.supervisor.fd_close_count, 1)
        finally:
            unblock_callback.set()
            finish.set()
            worker_thread.join(timeout=1)
            harness.close()
        self.assertTrue(outcome)
        boundaries = [event for event, _payload in callbacks
                      if event in ("acquire", "release")]
        self.assertEqual(boundaries, ["acquire", "release"])

    def test_holder_fallback_accounts_an_orphaned_active_descriptor(self):
        device = FakeDevice()
        supervisor_socket, client_socket = socket.socketpair()
        supervisor = RequestSupervisor(
            supervisor_socket, device, latency_us=0, max_hold_seconds=1)
        try:
            device.acquire(0)
            supervisor.fd_open_count = 1
            supervisor.closed_reason = "control_error"
            supervisor.abort("holder_exit")
            final = supervisor.snapshot()
            self.assertFalse(device.active)
            self.assertFalse(final["fd_active"])
            self.assertEqual(final["fd_open_count"], 1)
            self.assertEqual(final["fd_close_count"], 1)
        finally:
            client_socket.close()
            supervisor_socket.close()

    def test_fatal_abort_terminates_before_blocked_telemetry(self):
        for reason in ("eof", "protocol_error", "control_error",
                       "holder_exit"):
            with self.subTest(reason=reason):
                device = FakeDevice()
                supervisor_socket, client_socket = socket.socketpair()
                unblock = threading.Event()
                telemetry_started = threading.Event()
                fatal_started = threading.Event()
                ordering = []

                def block_telemetry(event, _payload):
                    if event == "acquire":
                        ordering.append(("telemetry", device.active))
                        telemetry_started.set()
                        unblock.wait(timeout=1)

                def terminate_child(fatal_reason):
                    ordering.append((f"fatal:{fatal_reason}", device.active))
                    fatal_started.set()

                supervisor = RequestSupervisor(
                    supervisor_socket, device, latency_us=0,
                    max_hold_seconds=1, status_callback=block_telemetry,
                    fatal_control_callback=terminate_child)
                acquired_ns = supervisor._acquire()
                supervisor.active_request = "cmpl-fatal-abort"
                supervisor.acquired_ns = acquired_ns
                supervisor.deadline_ns = acquired_ns + 1_000_000_000
                supervisor.leases_started = 1
                supervisor._queue_telemetry(
                    "acquire", request_id="cmpl-fatal-abort",
                    acquired_monotonic_ns=acquired_ns)
                worker = threading.Thread(
                    target=supervisor.abort, args=(reason,), daemon=True)
                worker.start()
                try:
                    self.assertTrue(device.released.wait(timeout=0.1))
                    self.assertTrue(fatal_started.wait(timeout=0.1))
                    self.assertTrue(telemetry_started.wait(timeout=0.1))
                    self.assertEqual(ordering[:2], [
                        (f"fatal:{reason}", False), ("telemetry", False)])
                    self.assertEqual(supervisor.fd_open_count, 1)
                    self.assertEqual(supervisor.fd_close_count, 1)
                finally:
                    unblock.set()
                    worker.join(timeout=1)
                    client_socket.close()
                    supervisor_socket.close()

    def test_immediate_end_ack_loss_terminates_before_telemetry(self):
        device = FakeDevice()
        events = []
        parent_socket, client_socket = socket.socketpair()
        broken_socket = BrokenAckSocket(parent_socket, "end")

        def observe(event, payload):
            events.append((event, device.active, payload))

        def terminate_child(reason):
            events.append(("fatal_control", device.active,
                           {"reason": reason}))

        supervisor = RequestSupervisor(
            broken_socket, device, latency_us=0, max_hold_seconds=1,
            status_callback=observe,
            fatal_control_callback=terminate_child)
        thread = threading.Thread(target=supervisor.serve, daemon=True)
        thread.start()
        client = RequestQos(
            client_socket, timeout=0.1, require_root_peer=False)
        try:
            with self.assertRaises(QosError):
                with client.scope("chatcmpl-broken-end"):
                    pass
            thread.join(timeout=0.5)
            self.assertFalse(thread.is_alive())
            self.assertFalse(device.active)
            names = [event for event, _active, _payload in events]
            self.assertLess(names.index("fatal_control"),
                            names.index("acquire"))
            fatal = next(payload for event, _active, payload in events
                         if event == "fatal_control")
            self.assertEqual(fatal["reason"], "control_error")
            self.assertEqual(supervisor.fd_open_count, 1)
            self.assertEqual(supervisor.fd_close_count, 1)
            self.assertTrue(all(not active for _event, active, _ in events))
        finally:
            client.close()
            parent_socket.close()

    def test_lost_begin_ack_releases_and_final_accounting_is_balanced(self):
        (client, parent_socket, blocked_socket, device, supervisor,
         thread, events) = self._ack_loss("begin")
        try:
            with self.assertRaises(QosError):
                with client.scope("chatcmpl-lost-begin"):
                    self.fail("body ran without a BEGIN acknowledgement")
            self.assertTrue(blocked_socket.blocked.wait(timeout=0.1))
            self.assertTrue(device.released.wait(timeout=0.25))
            thread.join(timeout=0.5)
            self.assertFalse(thread.is_alive())
            self.assertFalse(device.active)
            self.assertEqual(supervisor.fd_open_count, 1)
            self.assertEqual(supervisor.fd_close_count, 1)
            self.assertEqual(events[-1][0], "closed")
            self.assertFalse(events[-1][1])
            final = events[-1][2]["supervisor"]
            self.assertFalse(final["fd_active"])
            self.assertEqual(final["fd_open_count"], 1)
            self.assertEqual(final["fd_close_count"], 1)
            names = [event for event, _active, _payload in events]
            self.assertLess(names.index("fatal_control"),
                            names.index("acquire"))
            self.assertEqual(supervisor.fatal_control_reason,
                             "control_send_timeout")
            self.assertTrue(all(not active for _event, active, _ in events))
        finally:
            client.close()
            parent_socket.close()

    def test_lost_end_ack_releases_and_final_accounting_is_balanced(self):
        (client, parent_socket, blocked_socket, device, supervisor,
         thread, events) = self._ack_loss("end")
        try:
            with self.assertRaises(QosError):
                with client.scope("chatcmpl-lost-end"):
                    pass
            self.assertTrue(blocked_socket.blocked.wait(timeout=0.1))
            self.assertTrue(device.released.wait(timeout=0.1))
            thread.join(timeout=0.5)
            self.assertFalse(thread.is_alive())
            self.assertFalse(device.active)
            self.assertEqual(supervisor.fd_open_count, 1)
            self.assertEqual(supervisor.fd_close_count, 1)
            self.assertEqual(supervisor.leases_completed, 1)
            final = events[-1][2]["supervisor"]
            self.assertFalse(final["fd_active"])
            self.assertEqual(final["fd_open_count"], 1)
            self.assertEqual(final["fd_close_count"], 1)
            names = [event for event, _active, _payload in events]
            self.assertLess(names.index("fatal_control"),
                            names.index("acquire"))
            self.assertEqual(supervisor.fatal_control_reason,
                             "control_send_timeout")
            self.assertTrue(all(not active for _event, active, _ in events))
        finally:
            client.close()
            parent_socket.close()

    def test_eof_during_an_active_lease_releases_the_device(self):
        device = FakeDevice()
        fatal_reasons = []
        supervisor_socket, client = socket.socketpair()
        supervisor = RequestSupervisor(
            supervisor_socket, device, latency_us=0, max_hold_seconds=1,
            fatal_control_callback=fatal_reasons.append)
        thread = threading.Thread(target=supervisor.serve, daemon=True)
        thread.start()

        def request(value):
            client.sendall(json.dumps(value).encode() + b"\n")
            raw = b""
            while b"\n" not in raw:
                raw += client.recv(4096)
            return json.loads(raw.partition(b"\n")[0])

        try:
            hello = request({"op": "hello", "protocol": PROTOCOL})
            self.assertTrue(hello["ok"])
            begin = request({"op": "begin", "request_id": "cmpl-eof"})
            self.assertTrue(begin["ok"])
            self.assertTrue(device.active)
            client.close()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertFalse(device.active)
            self.assertEqual(supervisor.closed_reason, "eof")
            self.assertEqual(fatal_reasons, ["eof"])
        finally:
            client.close()
            supervisor_socket.close()

    def test_planned_stop_racing_with_eof_is_not_fatal(self):
        supervisor_socket, client = socket.socketpair()
        stop = threading.Event()
        first_check = threading.Event()
        checks = 0
        fatal_reasons = []
        result = []

        def should_stop():
            nonlocal checks
            checks += 1
            if checks == 1:
                first_check.set()
                return False
            return stop.is_set()

        supervisor = RequestSupervisor(
            supervisor_socket, FakeDevice(), latency_us=0,
            max_hold_seconds=1, should_stop=should_stop,
            fatal_control_callback=fatal_reasons.append)
        thread = threading.Thread(
            target=lambda: result.append(supervisor.serve()), daemon=True)
        thread.start()
        try:
            self.assertTrue(first_check.wait(timeout=0.5))
            stop.set()
            client.close()
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result, ["child_exit"])
            self.assertEqual(supervisor.closed_reason, "child_exit")
            self.assertEqual(fatal_reasons, [])
        finally:
            client.close()
            supervisor_socket.close()

    def test_active_eof_close_error_still_cleans_up_before_fatal_and_telemetry(self):
        device, closed = close_error_device("EOF close failed after close")
        events = []
        supervisor_socket, client = socket.socketpair()

        def observe(event, payload):
            events.append((event, device.active, payload))

        def terminate_child(reason):
            events.append(("fatal_control", device.active,
                           {"reason": reason}))

        supervisor = RequestSupervisor(
            supervisor_socket, device, latency_us=0, max_hold_seconds=1,
            status_callback=observe,
            fatal_control_callback=terminate_child)
        thread = threading.Thread(target=supervisor.serve, daemon=True)
        thread.start()

        def request(value):
            client.sendall(json.dumps(value).encode() + b"\n")
            raw = b""
            while b"\n" not in raw:
                raw += client.recv(4096)
            return json.loads(raw.partition(b"\n")[0])

        try:
            self.assertTrue(request({"op": "hello", "protocol": PROTOCOL})[
                "ok"])
            self.assertTrue(request({
                "op": "begin", "request_id": "cmpl-eof-close-error"})[
                    "ok"])
            self.assertTrue(device.active)
            client.close()
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            self.assertTrue(closed.is_set())
            self.assertFalse(device.active)
            self.assertIsNone(supervisor.active_request)
            self.assertEqual(supervisor.fd_open_count, 1)
            self.assertEqual(supervisor.fd_close_count, 1)
            self.assertEqual(supervisor.fd_close_errors, 1)
            self.assertIn("EOF close failed", supervisor.last_close_error)
            self.assertIn("EOF close failed", supervisor.last_error)
            names = [event for event, _active, _payload in events]
            self.assertLess(names.index("fatal_control"),
                            names.index("acquire"))
            fatal = next(payload for event, _active, payload in events
                         if event == "fatal_control")
            self.assertEqual(fatal["reason"], "eof")
            release = next(payload for event, _active, payload in events
                           if event == "release")
            self.assertIn("EOF close failed", release["close_error"])
            self.assertTrue(all(not active for _event, active, _ in events))
        finally:
            client.close()
            supervisor_socket.close()

    def test_ack_timeout_close_error_preserves_cleanup_order_and_both_errors(self):
        device, closed = close_error_device("ACK close failed after close")
        events = []
        parent_socket, client_socket = socket.socketpair()
        blocked_socket = BlockingAckSocket(parent_socket, "begin")

        def observe(event, payload):
            events.append((event, device.active, payload))

        def terminate_child(reason):
            events.append(("fatal_control", device.active,
                           {"reason": reason}))

        supervisor = RequestSupervisor(
            blocked_socket, device, latency_us=0, max_hold_seconds=0.2,
            send_timeout_seconds=0.03, status_callback=observe,
            fatal_control_callback=terminate_child)
        thread = threading.Thread(target=supervisor.serve, daemon=True)
        thread.start()
        client = RequestQos(
            client_socket, timeout=0.15, require_root_peer=False)
        try:
            with self.assertRaises(QosError):
                with client.scope("cmpl-ack-close-error"):
                    self.fail("body ran without a BEGIN acknowledgement")
            thread.join(timeout=0.5)
            self.assertFalse(thread.is_alive())
            self.assertTrue(blocked_socket.blocked.is_set())
            self.assertTrue(closed.is_set())
            self.assertFalse(device.active)
            self.assertIsNone(supervisor.active_request)
            self.assertEqual(supervisor.fd_open_count, 1)
            self.assertEqual(supervisor.fd_close_count, 1)
            self.assertEqual(supervisor.fd_close_errors, 1)
            self.assertIn("ACK close failed", supervisor.last_close_error)
            self.assertIn("ACK close failed", supervisor.last_error)
            self.assertIn("send timed out", supervisor.last_error)
            names = [event for event, _active, _payload in events]
            self.assertLess(names.index("fatal_control"),
                            names.index("acquire"))
            fatal = next(payload for event, _active, payload in events
                         if event == "fatal_control")
            self.assertEqual(fatal["reason"], "control_send_timeout")
            self.assertTrue(all(not active for _event, active, _ in events))
        finally:
            client.close()
            parent_socket.close()

    def test_normal_end_close_error_is_terminal(self):
        device, _closed = close_error_device("END close failed after close")
        fatal_reasons = []
        supervisor_socket, client_socket = socket.socketpair()
        supervisor = RequestSupervisor(
            supervisor_socket, device, latency_us=0, max_hold_seconds=1,
            fatal_control_callback=fatal_reasons.append)
        thread = threading.Thread(target=supervisor.serve, daemon=True)
        thread.start()
        client = RequestQos(
            client_socket, timeout=0.2, require_root_peer=False)
        try:
            with self.assertRaisesRegex(QosError, "END close failed"):
                with client.scope("cmpl-end-close-error"):
                    pass
            thread.join(timeout=0.5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(supervisor.closed_reason, "device_close_error")
            self.assertEqual(fatal_reasons, ["device_close_error"])
            self.assertFalse(device.active)
            self.assertEqual(supervisor.fd_open_count, 1)
            self.assertEqual(supervisor.fd_close_count, 1)
        finally:
            client.close()
            supervisor_socket.close()

    def test_immediate_ack_loss_and_close_error_preserve_both_causes(self):
        device, _closed = close_error_device(
            "broken-ACK close failed after close")
        parent_socket, client_socket = socket.socketpair()
        broken_socket = BrokenAckSocket(parent_socket, "begin")
        supervisor = RequestSupervisor(
            broken_socket, device, latency_us=0, max_hold_seconds=1)
        thread = threading.Thread(target=supervisor.serve, daemon=True)
        thread.start()
        client = RequestQos(
            client_socket, timeout=0.1, require_root_peer=False)
        try:
            with self.assertRaises(QosError):
                with client.scope("cmpl-broken-ack-close-error"):
                    self.fail("body ran without a BEGIN acknowledgement")
            thread.join(timeout=0.5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(supervisor.closed_reason, "control_error")
            self.assertIn("acknowledgement channel loss",
                          supervisor.last_error)
            self.assertIn("broken-ACK close failed", supervisor.last_error)
            self.assertEqual(supervisor.fd_open_count, 1)
            self.assertEqual(supervisor.fd_close_count, 1)
            self.assertEqual(supervisor.fd_close_errors, 1)
        finally:
            client.close()
            parent_socket.close()

    def test_injected_acquire_failure_is_visible_and_not_active(self):
        harness = ProtocolHarness(device=FakeDevice(
            acquire_error=OSError("fake device denied")))
        try:
            with self.assertRaisesRegex(QosError, "fake device denied"):
                with harness.client.scope("chatcmpl-denied"):
                    self.fail("scope body ran after an acquire failure")
            self.assertFalse(harness.device.active)
            self.assertEqual(harness.client.stats()["lease_errors"], 1)
        finally:
            harness.close()

    def test_ambiguous_begin_timeout_poison_closes_and_releases(self):
        harness = ProtocolHarness(
            device=FakeDevice(acquire_delay=0.06), timeout=0.02)
        try:
            with self.assertRaisesRegex(QosError, "timed out"):
                with harness.client.scope("chatcmpl-timeout"):
                    self.fail("scope body ran without a BEGIN acknowledgement")
            deadline = time.time() + 1
            while harness.device.active and time.time() < deadline:
                time.sleep(0.01)
            harness.thread.join(timeout=1)
            self.assertFalse(harness.device.active)
            self.assertFalse(harness.thread.is_alive())
            self.assertFalse(harness.client.stats()["connected"])
        finally:
            harness.close()

    def test_wrong_request_id_ack_poison_releases_the_lease(self):
        class WrongIdSupervisor(RequestSupervisor):
            def _send(self, value):
                if value.get("ok") and value.get("op") == "begin":
                    value = {**value, "request_id": "wrong-request"}
                super()._send(value)

        harness = ProtocolHarness(supervisor_class=WrongIdSupervisor)
        try:
            with self.assertRaisesRegex(QosError, "request id mismatch"):
                with harness.client.scope("chatcmpl-right"):
                    self.fail("body ran after a mismatched acknowledgement")
            harness.thread.join(timeout=1)
            self.assertFalse(harness.device.active)
            self.assertFalse(harness.client.stats()["connected"])
        finally:
            harness.close()

    def test_lost_end_ack_is_fatal_even_after_device_release(self):
        class SlowEndSupervisor(RequestSupervisor):
            def _send(self, value):
                if value.get("ok") and value.get("op") == "end":
                    time.sleep(0.06)
                super()._send(value)

        harness = ProtocolHarness(
            supervisor_class=SlowEndSupervisor, timeout=0.02)
        try:
            with self.assertRaisesRegex(QosError, "timed out"):
                with harness.client.scope("chatcmpl-end-timeout"):
                    pass
            self.assertFalse(harness.device.active)
            self.assertFalse(harness.client.stats()["connected"])
        finally:
            harness.close()

    def test_telemetry_failure_does_not_change_device_acknowledgements(self):
        def fail_telemetry(event, payload):
            del event, payload
            raise OSError("injected status disk failure")

        harness = ProtocolHarness(status_callback=fail_telemetry)
        try:
            with harness.client.scope("chatcmpl-telemetry") as lease:
                pass
            self.assertTrue(lease.released)
            self.assertFalse(harness.device.active)
            self.assertGreater(harness.supervisor.telemetry_errors, 0)
        finally:
            harness.close()

    def test_client_close_does_not_wait_for_a_hung_scope(self):
        harness = ProtocolHarness()
        entered = threading.Event()
        finish = threading.Event()
        outcome = []

        def worker():
            try:
                with harness.client.scope("chatcmpl-hung"):
                    entered.set()
                    finish.wait(timeout=2)
            except QosError as exc:
                outcome.append(str(exc))

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        self.assertTrue(entered.wait(timeout=1))
        started = time.monotonic()
        harness.client.close()
        self.assertLess(time.monotonic() - started, 0.2)
        harness.thread.join(timeout=1)
        self.assertFalse(harness.device.active)
        finish.set()
        worker_thread.join(timeout=1)
        self.assertTrue(outcome)
        harness.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
