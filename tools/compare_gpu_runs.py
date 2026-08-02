#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Compare two deterministic CPU/GPU inference captures.

The capture is a small JSON manifest plus a row-major raw-logit file.  Routes
remain in JSON (a 64-token K3 capture is well under a megabyte); the much
larger vocabulary logits stay as little-endian float32 bytes.

Manifest schema::

    {
      "schema": "waste.gpu_capture.v1",
      "dtype": "float32-le",
      "logits_file": "control.logits.f32",
      "vocab": 163840,
      "top_k": 16,
      "greedy": true,
      "arm": {
        "key": "cuda", "value": 1, "effective": 1,
        "fallbacks": 0, "calls": 35328, "expected_calls": 35328
      },
      "steps": [
        {
          "index": 0,
          "position": 5,
          "input_token": 387,
          "routes": [
            {"layer": 4, "experts": [12, 31, 7, 88]}
          ]
        }
      ]
    }

There is exactly one logit row per step.  ``input_token`` is the token fed to
that step; it may be null for a chunked-prefill result.  When ``greedy`` is
true, each non-null input after the first must equal the previous row's
argmax.  Route rows are ordered by layer and expert ids retain router order.

Only steps reached with identical input tokens are causally comparable.  The
step that first produces different argmax tokens is still compared, but all
later steps are marked non-causal and excluded from aggregate route/logit
statistics.  This prevents a cascade over different inputs from being
misreported as an arithmetic error in the original GPU step.

Usage:

    python3 tools/compare_gpu_runs.py cpu.json gpu.json
    python3 tools/compare_gpu_runs.py --json cpu.json gpu.json

No third-party packages are required.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import mmap
import os
import struct
import sys
from dataclasses import dataclass
from typing import Any


SCHEMA = "waste.gpu_capture.v1"
DTYPE = "float32-le"


class CaptureError(ValueError):
    """A malformed or mutually incompatible capture."""


@dataclass(frozen=True)
class RouteRow:
    layer: int
    experts: tuple[int, ...]


@dataclass(frozen=True)
class Step:
    index: int
    position: int
    input_token: int | None
    routes: tuple[RouteRow, ...]


@dataclass(frozen=True)
class Arm:
    key: str
    value: int
    effective: int
    fallbacks: int
    calls: int | None
    expected_calls: int | None


@dataclass(frozen=True)
class Capture:
    manifest_path: str
    logits_path: str
    vocab: int
    top_k: int
    greedy: bool
    arm: Arm | None
    steps: tuple[Step, ...]


class LogitRows:
    """Memory-map a capture and materialize only one vocabulary row at once."""

    def __init__(self, capture: Capture):
        self.capture = capture
        self._file = None
        self._map = None
        self._fmt = struct.Struct(f"<{capture.vocab}f")

    def __enter__(self) -> "LogitRows":
        expected = len(self.capture.steps) * self._fmt.size
        try:
            actual = os.path.getsize(self.capture.logits_path)
        except OSError as exc:
            raise CaptureError(
                f"cannot stat logits file {self.capture.logits_path}: {exc}"
            ) from exc
        if actual != expected:
            raise CaptureError(
                f"{self.capture.logits_path}: expected {expected} logit bytes, "
                f"found {actual}"
            )
        try:
            self._file = open(self.capture.logits_path, "rb")
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except OSError as exc:
            self.close()
            raise CaptureError(
                f"cannot map logits file {self.capture.logits_path}: {exc}"
            ) from exc
        return self

    def close(self) -> None:
        if self._map is not None:
            self._map.close()
            self._map = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def row(self, index: int) -> tuple[float, ...]:
        if self._map is None:
            raise RuntimeError("logit file is not open")
        return self._fmt.unpack_from(self._map, index * self._fmt.size)

    def row_hash(self, index: int) -> str:
        if self._map is None:
            raise RuntimeError("logit file is not open")
        begin = index * self._fmt.size
        return hashlib.sha256(self._map[begin : begin + self._fmt.size]).hexdigest()


def _integer(
    value: Any,
    where: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        bound = f" between {minimum} and {maximum}" if maximum is not None else f" >= {minimum}"
        raise CaptureError(f"{where} must be an integer{bound}")
    return value


def load_capture(path: str) -> Capture:
    try:
        with open(path, encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot read capture {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise CaptureError(f"{path}: schema must be {SCHEMA!r}")
    if raw.get("dtype") != DTYPE:
        raise CaptureError(f"{path}: dtype must be {DTYPE!r}")

    vocab = _integer(raw.get("vocab"), f"{path}: vocab", minimum=1)
    top_k = _integer(raw.get("top_k"), f"{path}: top_k", minimum=1)
    greedy = raw.get("greedy", False)
    if not isinstance(greedy, bool):
        raise CaptureError(f"{path}: greedy must be boolean")
    logits_name = raw.get("logits_file")
    if not isinstance(logits_name, str) or not logits_name:
        raise CaptureError(f"{path}: logits_file must be a non-empty string")
    logits_path = (
        logits_name
        if os.path.isabs(logits_name)
        else os.path.join(os.path.dirname(os.path.abspath(path)), logits_name)
    )

    arm = None
    raw_arm = raw.get("arm")
    if raw_arm is not None:
        if not isinstance(raw_arm, dict):
            raise CaptureError(f"{path}: arm must be an object")
        key = raw_arm.get("key")
        if not isinstance(key, str) or not key:
            raise CaptureError(f"{path}: arm.key must be a non-empty string")
        calls = raw_arm.get("calls")
        expected_calls = raw_arm.get("expected_calls")
        arm = Arm(
            key=key,
            value=_integer(raw_arm.get("value"), f"{path}: arm.value"),
            effective=_integer(
                raw_arm.get("effective"), f"{path}: arm.effective"
            ),
            fallbacks=_integer(
                raw_arm.get("fallbacks"), f"{path}: arm.fallbacks"
            ),
            calls=(
                None
                if calls is None
                else _integer(calls, f"{path}: arm.calls")
            ),
            expected_calls=(
                None
                if expected_calls is None
                else _integer(
                    expected_calls, f"{path}: arm.expected_calls"
                )
            ),
        )

    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise CaptureError(f"{path}: steps must be a non-empty list")
    steps = []
    for expected_index, item in enumerate(raw_steps):
        where = f"{path}: steps[{expected_index}]"
        if not isinstance(item, dict):
            raise CaptureError(f"{where} must be an object")
        index = _integer(item.get("index"), f"{where}.index")
        if index != expected_index:
            raise CaptureError(
                f"{where}.index must be contiguous (expected {expected_index})"
            )
        position = _integer(item.get("position"), f"{where}.position")
        input_token = item.get("input_token")
        if input_token is not None:
            input_token = _integer(input_token, f"{where}.input_token")
        raw_routes = item.get("routes", [])
        if not isinstance(raw_routes, list):
            raise CaptureError(f"{where}.routes must be a list")
        routes = []
        last_layer = -1
        for route_index, route in enumerate(raw_routes):
            rwhere = f"{where}.routes[{route_index}]"
            if not isinstance(route, dict):
                raise CaptureError(f"{rwhere} must be an object")
            layer = _integer(
                route.get("layer"), f"{rwhere}.layer", maximum=0xFFFFFFFF
            )
            if layer <= last_layer:
                raise CaptureError(f"{where}.routes must have unique ascending layers")
            last_layer = layer
            experts = route.get("experts")
            if not isinstance(experts, list) or len(experts) != top_k:
                raise CaptureError(
                    f"{rwhere}.experts must contain exactly top_k={top_k} ids"
                )
            ids = tuple(
                _integer(
                    expert, f"{rwhere}.experts[{i}]", maximum=0xFFFFFFFF
                )
                for i, expert in enumerate(experts)
            )
            if len(set(ids)) != len(ids):
                raise CaptureError(f"{rwhere}.experts contains duplicate ids")
            routes.append(RouteRow(layer, ids))
        if arm is not None and arm.key == "cuda" and expected_index > 0 and not routes:
            raise CaptureError(f"{where}.routes must contain decode route rows")
        steps.append(Step(index, position, input_token, tuple(routes)))

    return Capture(
        manifest_path=os.path.abspath(path),
        logits_path=logits_path,
        vocab=vocab,
        top_k=top_k,
        greedy=greedy,
        arm=arm,
        steps=tuple(steps),
    )


def _arm_dict(arm: Arm) -> dict[str, Any]:
    return {
        "key": arm.key,
        "value": arm.value,
        "effective": arm.effective,
        "fallbacks": arm.fallbacks,
        "calls": arm.calls,
        "expected_calls": arm.expected_calls,
    }


def _validate_cuda_arms(cpu: Capture, gpu: Capture) -> dict[str, Any] | None:
    if cpu.arm is None or gpu.arm is None:
        raise CaptureError("both captures must have arm metadata")
    if cpu.arm.key != gpu.arm.key:
        raise CaptureError(
            f"captures have different arm keys: {cpu.arm.key!r} vs {gpu.arm.key!r}"
        )
    if cpu.arm.key != "cuda":
        return {"cpu": _arm_dict(cpu.arm), "gpu": _arm_dict(gpu.arm)}

    if cpu.arm.value != 0 or cpu.arm.effective != 0 or cpu.arm.fallbacks != 0:
        raise CaptureError(
            "CPU control arm must have value=0, effective=0, fallbacks=0"
        )
    if cpu.arm.calls != 0 or cpu.arm.expected_calls != 0:
        raise CaptureError("CPU control arm must declare calls=expected_calls=0")
    if gpu.arm.value not in (1, 2) or gpu.arm.effective != gpu.arm.value:
        raise CaptureError(
            "GPU arm must have value 1 or 2 and matching effective mode"
        )
    if gpu.arm.fallbacks != 0:
        raise CaptureError("GPU arm reported a CUDA fallback/failure")
    if (gpu.arm.calls is None or gpu.arm.expected_calls is None or
            gpu.arm.calls != gpu.arm.expected_calls or gpu.arm.calls == 0):
        raise CaptureError(
            f"GPU arm executed {gpu.arm.calls} CUDA calls; "
            f"expected {gpu.arm.expected_calls}"
        )
    return {"cpu": _arm_dict(cpu.arm), "gpu": _arm_dict(gpu.arm)}


def _ranked_tokens(logits: tuple[float, ...], count: int) -> list[int] | None:
    if any(not math.isfinite(value) for value in logits):
        return None
    # Lower token id wins exact ties, matching the engine's ascending scan
    # with a strict `>` comparison.
    return heapq.nsmallest(
        count, range(len(logits)), key=lambda token: (-logits[token], token)
    )


def _route_hash(routes: tuple[RouteRow, ...], *, canonical_sets: bool) -> str:
    digest = hashlib.sha256()
    for row in routes:
        ids = sorted(row.experts) if canonical_sets else row.experts
        digest.update(struct.pack("<II", row.layer, len(ids)))
        for expert in ids:
            digest.update(struct.pack("<I", expert))
    return digest.hexdigest()


def _compare_routes(
    cpu: Step,
    gpu: Step,
    top_k: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    cpu_layers = [row.layer for row in cpu.routes]
    gpu_layers = [row.layer for row in gpu.routes]
    if cpu_layers != gpu_layers:
        raise CaptureError(
            f"step {cpu.index}: route layer layouts differ "
            f"({cpu_layers!r} vs {gpu_layers!r})"
        )

    result = {
        "cpu_ordered_hash": _route_hash(cpu.routes, canonical_sets=False),
        "gpu_ordered_hash": _route_hash(gpu.routes, canonical_sets=False),
        "cpu_set_hash": _route_hash(cpu.routes, canonical_sets=True),
        "gpu_set_hash": _route_hash(gpu.routes, canonical_sets=True),
        "changed_rows": 0,
        "replacements": 0,
        "ordered_only_reorders": 0,
    }
    for cpu_row, gpu_row in zip(cpu.routes, gpu.routes):
        summary["compared_rows"] += 1
        summary["selected_slots"] += top_k
        if cpu_row.experts == gpu_row.experts:
            continue
        cpu_set, gpu_set = set(cpu_row.experts), set(gpu_row.experts)
        replacements = top_k - len(cpu_set & gpu_set)
        kind = "replacement" if replacements else "reorder"
        candidate = {
            "step": cpu.index,
            "position": cpu.position,
            "layer": cpu_row.layer,
            "kind": kind,
            "replacements": replacements,
            "cpu": list(cpu_row.experts),
            "gpu": list(gpu_row.experts),
        }
        summary["ordered_changed_rows"] += 1
        if summary["first_divergence"] is None:
            summary["first_divergence"] = candidate
        old_max = summary["max_divergence"]
        if old_max is None or replacements > old_max["replacements"]:
            summary["max_divergence"] = candidate
        if replacements:
            summary["changed_rows"] += 1
            summary["replacements"] += replacements
            result["changed_rows"] += 1
            result["replacements"] += replacements
        else:
            summary["ordered_only_reorders"] += 1
            result["ordered_only_reorders"] += 1
    return result


def _compare_logits(
    cpu: tuple[float, ...],
    gpu: tuple[float, ...],
    cpu_hash: str,
    gpu_hash: str,
) -> dict[str, Any]:
    cpu_nonfinite = sum(not math.isfinite(value) for value in cpu)
    gpu_nonfinite = sum(not math.isfinite(value) for value in gpu)
    finite_diffs = [
        abs(a - b)
        for a, b in zip(cpu, gpu)
        if math.isfinite(a) and math.isfinite(b)
    ]
    cpu_ranked = _ranked_tokens(cpu, min(10, len(cpu)))
    gpu_ranked = _ranked_tokens(gpu, min(10, len(gpu)))
    cpu_argmax = cpu_ranked[0] if cpu_ranked else None
    gpu_argmax = gpu_ranked[0] if gpu_ranked else None
    return {
        "cpu_hash": cpu_hash,
        "gpu_hash": gpu_hash,
        "byte_exact": cpu_hash == gpu_hash,
        "cpu_argmax": cpu_argmax,
        "gpu_argmax": gpu_argmax,
        "argmax_equal": cpu_argmax is not None and cpu_argmax == gpu_argmax,
        "cpu_top10": cpu_ranked,
        "gpu_top10": gpu_ranked,
        "top10_set_equal": (
            cpu_ranked is not None
            and gpu_ranked is not None
            and set(cpu_ranked) == set(gpu_ranked)
        ),
        "max_abs": max(finite_diffs) if finite_diffs else None,
        "mean_abs": (
            math.fsum(finite_diffs) / len(finite_diffs) if finite_diffs else None
        ),
        "finite_pairs": len(finite_diffs),
        "cpu_nonfinite": cpu_nonfinite,
        "gpu_nonfinite": gpu_nonfinite,
    }


def compare_captures(cpu_path: str, gpu_path: str) -> dict[str, Any]:
    cpu = load_capture(cpu_path)
    gpu = load_capture(gpu_path)
    for field in ("vocab", "top_k"):
        if getattr(cpu, field) != getattr(gpu, field):
            raise CaptureError(
                f"captures have different {field}: "
                f"{getattr(cpu, field)} vs {getattr(gpu, field)}"
            )
    if len(cpu.steps) != len(gpu.steps):
        raise CaptureError(
            f"captures have different step counts: {len(cpu.steps)} vs {len(gpu.steps)}"
        )
    if cpu.greedy != gpu.greedy:
        raise CaptureError(
            f"captures disagree on greedy mode: {cpu.greedy} vs {gpu.greedy}"
        )

    arms = _validate_cuda_arms(cpu, gpu)
    route_summary: dict[str, Any] = {
        "compared_rows": 0,
        "selected_slots": 0,
        "changed_rows": 0,
        "ordered_changed_rows": 0,
        "ordered_only_reorders": 0,
        "replacements": 0,
        "replacement_fraction": 0.0,
        "max_replacements_in_row": 0,
        "first_divergence": None,
        "max_divergence": None,
    }
    logit_summary: dict[str, Any] = {
        "compared_steps": 0,
        "byte_changed_steps": 0,
        "first_changed_step": None,
        "argmax_changed_steps": 0,
        "first_argmax_changed_step": None,
        "top10_changed_steps": 0,
        "max_abs": 0.0,
        "max_abs_step": None,
        "max_step_mean_abs": 0.0,
        "max_step_mean_abs_step": None,
        "mean_abs": 0.0,
        "finite_pairs": 0,
        "cpu_nonfinite": 0,
        "gpu_nonfinite": 0,
        "all_cpu_nonfinite": 0,
        "all_gpu_nonfinite": 0,
    }
    output: dict[str, Any] = {
        "schema": "waste.gpu_comparison.v1",
        "cpu_capture": cpu.manifest_path,
        "gpu_capture": gpu.manifest_path,
        "vocab": cpu.vocab,
        "top_k": cpu.top_k,
        "total_steps": len(cpu.steps),
        "causally_compared_steps": 0,
        "noncausal_steps": 0,
        "first_token_divergence": None,
        "arms": arms,
        "routes": route_summary,
        "logits": logit_summary,
        "steps": [],
    }

    causal = True
    previous_cpu_argmax = None
    previous_gpu_argmax = None
    total_abs = 0.0
    with LogitRows(cpu) as cpu_logits, LogitRows(gpu) as gpu_logits:
        for index, (cpu_step, gpu_step) in enumerate(zip(cpu.steps, gpu.steps)):
            if cpu_step.position != gpu_step.position:
                raise CaptureError(
                    f"step {index}: positions differ "
                    f"({cpu_step.position} vs {gpu_step.position})"
                )
            cpu_row = cpu_logits.row(index)
            gpu_row = gpu_logits.row(index)
            cpu_nonfinite = sum(not math.isfinite(value) for value in cpu_row)
            gpu_nonfinite = sum(not math.isfinite(value) for value in gpu_row)
            logit_summary["all_cpu_nonfinite"] += cpu_nonfinite
            logit_summary["all_gpu_nonfinite"] += gpu_nonfinite

            if cpu.greedy and index > 0 and cpu_step.input_token is not None:
                if (
                    previous_cpu_argmax is not None
                    and cpu_step.input_token != previous_cpu_argmax
                ):
                    raise CaptureError(
                        f"CPU step {index}: greedy input {cpu_step.input_token} "
                        f"does not match prior argmax {previous_cpu_argmax}"
                    )
            if gpu.greedy and index > 0 and gpu_step.input_token is not None:
                if (
                    previous_gpu_argmax is not None
                    and gpu_step.input_token != previous_gpu_argmax
                ):
                    raise CaptureError(
                        f"GPU step {index}: greedy input {gpu_step.input_token} "
                        f"does not match prior argmax {previous_gpu_argmax}"
                    )

            step_out: dict[str, Any] = {
                "step": index,
                "position": cpu_step.position,
                "cpu_input_token": cpu_step.input_token,
                "gpu_input_token": gpu_step.input_token,
                "causal_comparable": causal,
            }
            if causal and cpu_step.input_token != gpu_step.input_token:
                causal = False
                step_out["causal_comparable"] = False
                output["first_token_divergence"] = {
                    "step": index,
                    "kind": "input_token",
                    "cpu": cpu_step.input_token,
                    "gpu": gpu_step.input_token,
                }
            if not causal:
                output["noncausal_steps"] += 1
                step_out["skipped_reason"] = "different causal token history"
                step_out["nonfinite"] = {
                    "cpu": cpu_nonfinite,
                    "gpu": gpu_nonfinite,
                }
                output["steps"].append(step_out)
                # Preserve each capture's greedy validation for the next row.
                cpu_ranked = _ranked_tokens(cpu_row, 1)
                gpu_ranked = _ranked_tokens(gpu_row, 1)
                previous_cpu_argmax = cpu_ranked[0] if cpu_ranked else None
                previous_gpu_argmax = gpu_ranked[0] if gpu_ranked else None
                continue

            output["causally_compared_steps"] += 1
            route_result = _compare_routes(
                cpu_step, gpu_step, cpu.top_k, route_summary
            )
            logit_result = _compare_logits(
                cpu_row,
                gpu_row,
                cpu_logits.row_hash(index),
                gpu_logits.row_hash(index),
            )
            step_out["routes"] = route_result
            step_out["logits"] = logit_result
            output["steps"].append(step_out)

            logit_summary["compared_steps"] += 1
            if not logit_result["byte_exact"]:
                logit_summary["byte_changed_steps"] += 1
                if logit_summary["first_changed_step"] is None:
                    logit_summary["first_changed_step"] = index
            if not logit_result["argmax_equal"]:
                logit_summary["argmax_changed_steps"] += 1
                if logit_summary["first_argmax_changed_step"] is None:
                    logit_summary["first_argmax_changed_step"] = index
            if not logit_result["top10_set_equal"]:
                logit_summary["top10_changed_steps"] += 1
            logit_summary["cpu_nonfinite"] += logit_result["cpu_nonfinite"]
            logit_summary["gpu_nonfinite"] += logit_result["gpu_nonfinite"]
            finite_pairs = logit_result["finite_pairs"]
            logit_summary["finite_pairs"] += finite_pairs
            if logit_result["mean_abs"] is not None:
                total_abs += logit_result["mean_abs"] * finite_pairs
            if (
                logit_result["max_abs"] is not None
                and (
                    logit_summary["max_abs_step"] is None
                    or logit_result["max_abs"] > logit_summary["max_abs"]
                )
            ):
                logit_summary["max_abs"] = logit_result["max_abs"]
                logit_summary["max_abs_step"] = index
            if (
                logit_result["mean_abs"] is not None
                and (
                    logit_summary["max_step_mean_abs_step"] is None
                    or logit_result["mean_abs"] >
                    logit_summary["max_step_mean_abs"]
                )
            ):
                logit_summary["max_step_mean_abs"] = logit_result["mean_abs"]
                logit_summary["max_step_mean_abs_step"] = index

            previous_cpu_argmax = logit_result["cpu_argmax"]
            previous_gpu_argmax = logit_result["gpu_argmax"]
            if not logit_result["argmax_equal"]:
                output["first_token_divergence"] = {
                    "step": index,
                    "kind": "argmax",
                    "cpu": previous_cpu_argmax,
                    "gpu": previous_gpu_argmax,
                }
                causal = False

    slots = route_summary["selected_slots"]
    route_summary["replacement_fraction"] = (
        route_summary["replacements"] / slots if slots else 0.0
    )
    if route_summary["max_divergence"] is not None:
        route_summary["max_replacements_in_row"] = route_summary[
            "max_divergence"
        ]["replacements"]
    pairs = logit_summary["finite_pairs"]
    logit_summary["mean_abs"] = total_abs / pairs if pairs else None
    return output


def print_human(result: dict[str, Any]) -> None:
    routes = result["routes"]
    logits = result["logits"]
    arms = result.get("arms")
    if arms is not None:
        gpu = arms["gpu"]
        print(
            f"arm={gpu['key']} requested={gpu['value']} "
            f"effective={gpu['effective']} fallbacks={gpu['fallbacks']} "
            f"calls={gpu['calls']}/{gpu['expected_calls']}"
        )
    print(
        f"steps={result['total_steps']} causal={result['causally_compared_steps']} "
        f"noncausal={result['noncausal_steps']}"
    )
    print(
        f"routes rows={routes['compared_rows']} changed={routes['changed_rows']} "
        f"reorders={routes['ordered_only_reorders']} "
        f"replacements={routes['replacements']}/{routes['selected_slots']} "
        f"({100.0 * routes['replacement_fraction']:.6f}%) "
        f"max-row={routes['max_replacements_in_row']}"
    )
    max_abs = logits["max_abs"]
    mean_abs = logits["mean_abs"]
    print(
        f"logits steps={logits['compared_steps']} byte-changed={logits['byte_changed_steps']} "
        f"argmax-changed={logits['argmax_changed_steps']} "
        f"top10-changed={logits['top10_changed_steps']} "
        f"max-abs={max_abs if max_abs is not None else 'n/a'} "
        f"max-step-mean={logits['max_step_mean_abs']} "
        f"mean-abs={mean_abs if mean_abs is not None else 'n/a'} "
        f"nonfinite={logits['cpu_nonfinite']}/{logits['gpu_nonfinite']} "
        f"all={logits['all_cpu_nonfinite']}/{logits['all_gpu_nonfinite']}"
    )
    if result["first_token_divergence"] is not None:
        print("token divergence: " + json.dumps(result["first_token_divergence"], sort_keys=True))
    if routes["first_divergence"] is not None:
        print("first route divergence: " + json.dumps(routes["first_divergence"], sort_keys=True))
    if routes["max_divergence"] is not None:
        print("max route divergence: " + json.dumps(routes["max_divergence"], sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare WASTE CPU/GPU route and logit capture manifests."
    )
    parser.add_argument("cpu_capture")
    parser.add_argument("gpu_capture")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    try:
        result = compare_captures(args.cpu_capture, args.gpu_capture)
    except CaptureError as exc:
        print(f"compare_gpu_runs: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
