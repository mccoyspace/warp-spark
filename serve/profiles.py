# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Explicit, reproducible server performance profiles.

Profiles live above the portable engine API.  They name a measured operating
configuration without changing the default server and without turning a
machine-wide policy into ambient process state.  In particular, ``spark-q0``
requires the private request-scoped PM-QoS control channel created by
``tools/pm_qos_exec.py``; the server never opens the privileged device itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import MutableMapping, Optional


DEFAULT_PROFILE = "default"
SPARK_Q0_PROFILE = "spark-q0"
PROFILE_NAMES = (SPARK_Q0_PROFILE,)


class ProfileError(ValueError):
    """A requested profile conflicts with explicit process configuration."""


SPARK_Q0_ENVIRONMENT = {
    # Preserve the arithmetic and instrumentation used for the accepted GB10
    # comparisons.  WASTE_PROFILE is phase tracing, not this profile's name.
    "WASTE_PROFILE": "0",
    "WASTE_Q8": "1",
    "WASTE_SDOT": "0",
    "WASTE_I8MM": "0",
    "WASTE_VERIFY": "0",
    "WASTE_BACKEND": "auto",
    "WASTE_DIRECT": "1",
    # The public engine option wins today, but pinning the ambient fallback
    # prevents contradictory evidence and future precedence drift.
    "WASTE_THREADS": "8",
    # The accepted Spark runs used ordinary pageable, non-purgeable storage.
    # Spell that out because both switches materially change residency.
    "WASTE_MLOCK": "0",
    "WASTE_PURGEABLE": "0",
    # Current-upstream storage pipeline.  These are defaults today, but making
    # them explicit keeps the profile honest if upstream defaults move later.
    "WASTE_IO_THREADS": "2",
    "WASTE_IO_DEPTH": "2",
    # On the GN100, Q0 plus the reader won its matched current-upstream
    # qualification without speculative router lookahead or its extra I/O.
    "WASTE_LOOKAHEAD": "0",
}

STORAGE_ENVIRONMENT = (
    "WASTE_IO_THREADS", "WASTE_IO_DEPTH", "WASTE_LOOKAHEAD",
)

# Service authentication and the private supervisor channel do not alter
# engine execution.  Their values are deliberately neither copied into the
# profile evidence nor exposed by ``public``.
SPARK_Q0_SERVICE_ENVIRONMENT = {
    "WASTE_API_KEY", "WASTE_PM_QOS_CONTROL_FD",
}


def _c_atoi(value: Optional[str], default: int) -> int:
    """Parse the ordinary C ``atoi`` inputs accepted by current upstream."""
    if value is None:
        return default
    match = re.match(r"\s*([+-]?[0-9]+)", value)
    return int(match.group(1)) if match else 0


@dataclass(frozen=True)
class ResolvedProfile:
    """Engine arguments and process settings after profile resolution."""

    name: str
    threads: int
    cache: str
    direct_io: bool
    verify_records: bool
    request_qos_required: bool
    request_qos_us: Optional[int]
    environment: dict[str, str]

    def public(self) -> dict:
        io_threads = _c_atoi(
            self.environment.get("WASTE_IO_THREADS"), 2)
        io_depth = _c_atoi(
            self.environment.get("WASTE_IO_DEPTH"), 2)
        # These are current upstream's configuration transforms, not a claim
        # that runtime-effective reader state can be queried from the ABI.
        if io_depth < io_threads:
            io_depth = io_threads
        lookahead = _c_atoi(
            self.environment.get("WASTE_LOOKAHEAD"), 6)
        lookahead = max(0, min(64, lookahead))
        return {
            "name": self.name,
            "engine": {
                "threads": self.threads,
                "cache": self.cache,
                "direct_io": self.direct_io,
                "verify_records": self.verify_records,
            },
            "storage": {
                # These are parsed requests. EC_MAXIO, cache-slot depth, and
                # worker creation can still change the effective reader.
                "requested_read_ahead_threads": io_threads,
                "requested_read_ahead_depth": io_depth,
                "requested_router_lookahead": lookahead,
                "effective_configuration_reported": False,
                "effective_read_ahead_threads": None,
                "effective_read_ahead_depth": None,
            },
            "cpu_affinity": {"managed": False, "effective_cpu_list": None},
            "pm_qos": {
                "required": self.request_qos_required,
                "scope": "request" if self.request_qos_required else "off",
                "latency_us": self.request_qos_us,
            },
        }


def resolve_profile(name: Optional[str], *, threads: Optional[int], cache: str,
                    no_direct_io: bool, verify: bool,
                    environ: MutableMapping[str, str]) -> ResolvedProfile:
    """Resolve one server profile and apply its exact environment.

    A profile is atomic: a caller may repeat one of its values explicitly, but
    a contradictory option or pre-existing environment variable is an error.
    That makes the name useful evidence rather than a best-effort hint.
    """
    if not name:
        return ResolvedProfile(
            name=DEFAULT_PROFILE, threads=0 if threads is None else threads,
            cache=cache, direct_io=not no_direct_io,
            verify_records=verify, request_qos_required=False,
            request_qos_us=None,
            environment={key: environ[key] for key in STORAGE_ENVIRONMENT
                         if key in environ})
    if name != SPARK_Q0_PROFILE:
        raise ProfileError(f"unknown performance profile: {name}")

    conflicts: list[str] = []
    if threads not in (None, 8):
        conflicts.append(f"--threads={threads} (spark-q0 requires 8)")
    if cache != "lfru":
        conflicts.append(f"--cache={cache} (spark-q0 requires lfru)")
    if no_direct_io:
        conflicts.append("--no-direct-io (spark-q0 requires direct I/O)")
    if verify:
        conflicts.append("--verify (spark-q0 fixes record verification off)")
    for key, expected in SPARK_Q0_ENVIRONMENT.items():
        actual = environ.get(key)
        if actual is not None and actual != expected:
            conflicts.append(f"{key}={actual!r} (requires {expected!r})")
    # A strict profile is evidence, not a partial overlay.  Reject unknown
    # WASTE_* engine/debug selectors rather than silently running a different
    # experiment under the same name.  WASTE_LIB is intentionally rejected:
    # it selects the engine binary and this upstream cannot yet report a
    # verified library identity.  API authentication and the private control
    # descriptor are service plumbing and remain allowed but hidden.
    for key in sorted(environ):
        if key.startswith("WASTE_") and \
                key not in SPARK_Q0_ENVIRONMENT and \
                key not in SPARK_Q0_SERVICE_ENVIRONMENT:
            conflicts.append(f"{key} is not permitted by spark-q0")
    if conflicts:
        raise ProfileError("spark-q0 conflicts with " + "; ".join(conflicts))

    environ.update(SPARK_Q0_ENVIRONMENT)
    return ResolvedProfile(
        name=SPARK_Q0_PROFILE, threads=8, cache="lfru", direct_io=True,
        verify_records=False, request_qos_required=True, request_qos_us=0,
        environment=dict(SPARK_Q0_ENVIRONMENT))
