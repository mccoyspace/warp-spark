#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
# Does the engine actually stay inside waste_cfg.ram_budget_bytes?
#
# waste.h calls the budget a hard ceiling on everything the engine
# allocates. That is a claim about peak RSS, so measure peak RSS.
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-$HOME/models/kimi-linear.waste}"
BUDGET_GB="${2:-6}"
# A short prompt never allocates the chunked-prefill scratch, which is the
# largest single thing the plan has to size. Pass "long" to force a chunk.
case "${3:-}" in
    long) PROMPT=$(python3 -c "print(' '.join(['token'] * 90))");;
    *)    PROMPT="hello";;
esac

# Peak RSS of the child, through getrusage. /usr/bin/time was the obvious
# tool and the wrong one: its flags differ between BSD and GNU, and it is
# not installed in a plain debian image at all — which is how this check
# first "failed" on Linux. ru_maxrss is bytes on macOS, kilobytes elsewhere.
#
# `resource` is Unix-only, and on Windows this printed nothing and exited 1,
# which run.sh read as "peak RSS exceeded the budget" — reporting a FAIL for
# a measurement that never happened, on the engine's most central claim
# (#36, gap 1). run.sh's own rule is that a missing prerequisite says SKIP
# loudly and is never a silent pass; it should equally never be a failure.
# So Windows gets the real measurement through PeakWorkingSetSize, and
# anything that still cannot measure says UNMEASURABLE and exits 77.
RSS=$(python3 -c '
import subprocess, sys

cmd = ["./waste", "run", sys.argv[1], sys.argv[2], "-n", "2",
       "--budget", sys.argv[3] + "G", "-q"]

if sys.platform == "win32":
    # PeakWorkingSetSize is still readable through a handle the parent
    # holds after the child exits, which is why Popen is used rather than
    # run(): subprocess.run would close it. Working set is not exactly
    # ru_maxrss, but it is the same question — the most physical memory
    # this process ever held — and it is what Windows has.
    import ctypes
    from ctypes import wintypes

    class COUNTERS(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    p.wait()
    c = COUNTERS()
    c.cb = ctypes.sizeof(c)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        int(p._handle), ctypes.byref(c), c.cb)
    if not ok:
        sys.exit(77)
    print(c.PeakWorkingSetSize)
else:
    import resource
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    r = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    print(r if sys.platform == "darwin" else r * 1024)
' "$MODEL" "$PROMPT" "$BUDGET_GB")
if [ -z "$RSS" ]; then
    echo "BUDGET UNMEASURABLE: no peak-RSS interface on this platform"
    exit 77
fi

python3 - "$RSS" "$BUDGET_GB" <<'PY'
import sys
rss = int(sys.argv[1]); budget = float(sys.argv[2]) * (1 << 30)
# Allow the process image itself (binary, libc, thread stacks) on top of
# what the engine allocates: 512 MB is generous and still catches a real
# overrun, which would be gigabytes.
slack = 512 << 20
print(f"peak RSS {rss/2**30:.2f} GB, budget {budget/2**30:.2f} GB, "
      f"slack {slack/2**20:.0f} MB")
print("BUDGET OK" if rss <= budget + slack else "BUDGET EXCEEDED")
sys.exit(0 if rss <= budget + slack else 1)
PY
