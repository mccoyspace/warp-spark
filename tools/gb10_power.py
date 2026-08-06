#!/usr/bin/env python3
"""Minimal Shelly wall-power logger and phase marker for GB10 campaigns."""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


STOP = False


def stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def status_url(base: str) -> str:
    return f"{base.rstrip('/')}/rpc/Switch.GetStatus?{urllib.parse.urlencode({'id': 0})}"


def log_power(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    deadline = time.monotonic()
    with args.output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("unix_ts", "monotonic_ts", "apower_w", "voltage",
                         "current", "aenergy_wh_total"))
        stream.flush()
        while not STOP:
            unix_ts = time.time()
            monotonic_ts = time.monotonic()
            try:
                with urllib.request.urlopen(status_url(args.shelly),
                                            timeout=args.timeout) as response:
                    value = json.load(response)
                writer.writerow((f"{unix_ts:.6f}", f"{monotonic_ts:.6f}",
                                 value["apower"], value["voltage"],
                                 value["current"], value["aenergy"]["total"]))
                stream.flush()
            except (OSError, KeyError, ValueError) as exc:
                print(f"gb10_power: poll failed: {exc}", file=sys.stderr,
                      flush=True)
            deadline += args.interval
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                deadline = time.monotonic()
    return 0


def mark(args: argparse.Namespace) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new = not args.output.exists()
    with args.output.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if new:
            writer.writerow(("unix_ts", "monotonic_ts", "label", "note"))
        writer.writerow((f"{time.time():.6f}", f"{time.monotonic():.6f}",
                         args.label, args.note))
        stream.flush()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    logger = sub.add_parser("log")
    logger.add_argument("--output", type=Path, required=True)
    logger.add_argument("--shelly", default="http://192.168.1.118")
    logger.add_argument("--interval", type=float, default=1.0)
    logger.add_argument("--timeout", type=float, default=2.0)
    logger.set_defaults(function=log_power)
    marker = sub.add_parser("mark")
    marker.add_argument("--output", type=Path, required=True)
    marker.add_argument("--label", required=True)
    marker.add_argument("--note", default="")
    marker.set_defaults(function=mark)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
