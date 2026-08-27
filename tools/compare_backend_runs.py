#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Compare public-backend qualification captures and enforce their contract."""

from __future__ import annotations

import argparse
import array
import heapq
import hashlib
import json
import math
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import BinaryIO


FORMAT = "waste-backend-qualification-v1"


class ComparisonError(Exception):
    """A malformed capture or failed qualification gate."""


@dataclass(frozen=True)
class Run:
    directory: pathlib.Path
    manifest: dict
    mode: str
    vocab: int
    steps: int
    prompt: tuple[int, ...]
    generated: tuple[int, ...]
    top_k: int
    n_layers: int
    first_dense: int
    n_experts: int

    @property
    def logits_path(self) -> pathlib.Path:
        return self.directory / "logits.f32"

    @property
    def routes_path(self) -> pathlib.Path:
        return self.directory / "routes.txt"


def load_run(directory: str) -> Run:
    root = pathlib.Path(directory)
    try:
        manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"{root}: cannot read run.json: {exc}") from exc
    if manifest.get("format") != FORMAT:
        raise ComparisonError(f"{root}: unsupported capture format")
    try:
        mode = str(manifest["mode"])
        capture = manifest["capture"]
        prompt = tuple(int(v) for v in capture["prompt_tokens"])
        generated = tuple(int(v) for v in capture["generated_tokens"])
        vocab = int(capture["vocab"])
        steps = int(capture["logit_steps"])
        top_k = int(manifest["model"]["top_k"])
        n_layers = int(manifest["model"]["n_layers"])
        first_dense = int(manifest["model"]["first_dense"])
        n_experts = int(manifest["model"]["n_experts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ComparisonError(f"{root}: malformed run.json") from exc
    if mode not in {"cpu", "vq", "matvec", "combined"}:
        raise ComparisonError(f"{root}: invalid mode {mode!r}")
    if (not prompt or steps < 2 or vocab < 1 or top_k < 1 or
            n_layers < 1 or first_dense < 0 or first_dense >= n_layers or
            n_experts < top_k):
        raise ComparisonError(f"{root}: empty or invalid capture dimensions")
    if len(generated) != steps:
        raise ComparisonError(
            f"{root}: {len(generated)} generated ids for {steps} logit steps"
        )
    expected_bytes = vocab * steps * 4
    try:
        actual_bytes = (root / "logits.f32").stat().st_size
    except OSError as exc:
        raise ComparisonError(f"{root}: cannot stat logits.f32: {exc}") from exc
    if actual_bytes != expected_bytes:
        raise ComparisonError(
            f"{root}: logits.f32 is {actual_bytes} bytes; expected {expected_bytes}"
        )
    return Run(root, manifest, mode, vocab, steps, prompt, generated, top_k,
               n_layers, first_dense, n_experts)


def require_matched_run_shape(reference: Run, candidate: Run) -> None:
    if reference.mode != "cpu":
        raise ComparisonError("reference capture must use cpu mode")
    for key in ("build_info", "model", "config", "controls"):
        if reference.manifest.get(key) != candidate.manifest.get(key):
            raise ComparisonError(f"unmatched {key} between captures")
    if reference.prompt != candidate.prompt:
        raise ComparisonError("prompt token ids differ")
    if reference.vocab != candidate.vocab or reference.steps != candidate.steps:
        raise ComparisonError("logit capture dimensions differ")
    if reference.top_k != candidate.top_k:
        raise ComparisonError("router top-k differs")


def require_backend_semantics(run: Run) -> None:
    try:
        calls = {key: int(value) for key, value in
                 run.manifest["backend_calls"].items()}
        caps = calls["effective_capabilities"]
        failures = calls["callback_failures"]
        queries = calls["claim_queries"]
        claimed = calls["claimed_tensors"]
        matvec = calls["matvec_calls"]
        begin = calls["vq_begin_calls"]
        gate_up = calls["vq_gate_up_calls"]
        down = calls["vq_down_calls"]
        prefill_execution = sum(calls[key] for key in (
            "prefill_matvec_calls", "prefill_vq_begin_calls",
            "prefill_vq_gate_up_calls", "prefill_vq_down_calls",
            "prefill_callback_failures",
        ))
    except (KeyError, TypeError, ValueError) as exc:
        raise ComparisonError(f"{run.directory}: malformed backend call counters") from exc
    if failures:
        raise ComparisonError(f"{run.directory}: backend callback failures were recorded")
    if prefill_execution:
        raise ComparisonError(f"{run.directory}: backend ran during CPU-owned prefill")
    if claimed > queries:
        raise ComparisonError(f"{run.directory}: impossible MATVEC claim counters")
    expected_caps = {"cpu": 0, "matvec": 1, "vq": 2, "combined": 3}[run.mode]
    if caps != expected_caps:
        raise ComparisonError(
            f"{run.directory}: effective capabilities {caps} != {expected_caps}"
        )
    wants_matvec = run.mode in {"matvec", "combined"}
    wants_vq = run.mode in {"vq", "combined"}
    decode_steps = run.steps - 1
    expected_matvec = claimed * decode_steps
    expected_begin = decode_steps * (run.n_layers - run.first_dense)
    expected_expert_apply = expected_begin * run.top_k
    if wants_matvec and (queries < 1 or claimed < 1 or
                         matvec != expected_matvec):
        raise ComparisonError(
            f"{run.directory}: MATVEC callback count {matvec} != "
            f"{claimed} claimed tensors x {decode_steps} decode steps"
        )
    if not wants_matvec and (queries or claimed or matvec):
        raise ComparisonError(f"{run.directory}: unexpected MATVEC activity")
    if wants_vq and (begin != expected_begin or
                     gate_up != expected_expert_apply or
                     down != expected_expert_apply):
        raise ComparisonError(
            f"{run.directory}: VQ callback coverage is incomplete "
            f"(begin {begin}/{expected_begin}, gate/up "
            f"{gate_up}/{expected_expert_apply}, down "
            f"{down}/{expected_expert_apply})"
        )
    if not wants_vq and (begin or gate_up or down):
        raise ComparisonError(f"{run.directory}: unexpected VQ activity")
    if wants_vq:
        model = run.manifest["model"]
        vq_tuple = tuple(int(model.get(key, -1)) for key in (
            "vq_scheme", "vq_stages", "vq_entries", "vq_vec_dim",
            "vq_index_bits",
        ))
        if vq_tuple != (1, 3, 256, 8, 8):
            raise ComparisonError(f"{run.directory}: VQ mode is not complete VQ3R")


def read_frame(stream: BinaryIO, vocab: int, label: str, step: int) -> tuple[bytes, array.array]:
    raw = stream.read(vocab * 4)
    if len(raw) != vocab * 4:
        raise ComparisonError(f"{label}: short logit frame at step {step}")
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return raw, values


def top_ids(values: array.array, n: int) -> tuple[int, ...]:
    # The id is the deterministic tiebreak, matching the engine's first-max
    # greedy rule and the existing oracle comparator.
    return tuple(
        heapq.nlargest(min(n, len(values)), range(len(values)),
                       key=lambda i: (values[i], -i))
    )


def parse_routes(run: Run) -> tuple[bytes, tuple[tuple[int, int, tuple[int, ...]], ...]]:
    try:
        raw = run.routes_path.read_bytes()
    except OSError as exc:
        raise ComparisonError(f"{run.directory}: cannot read routes.txt: {exc}") from exc
    rows: list[tuple[int, int, tuple[int, ...]]] = []
    seen: set[tuple[int, int]] = set()
    try:
        text = raw.decode("ascii")
        for line_number, line in enumerate(text.splitlines(), 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2 + 3 * run.top_k:
                raise ValueError(f"line {line_number} has the wrong field count")
            pos, layer = int(fields[0]), int(fields[1])
            ids = tuple(int(v) for v in fields[2:2 + run.top_k])
            key = (pos, layer)
            if key in seen:
                raise ValueError(f"line {line_number} duplicates position/layer {key}")
            if len(set(ids)) != run.top_k or any(
                    expert < 0 or expert >= run.n_experts for expert in ids):
                raise ValueError(f"line {line_number} has invalid expert ids")
            seen.add(key)
            rows.append((pos, layer, ids))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ComparisonError(f"{run.directory}: malformed routes.txt: {exc}") from exc
    n_positions = len(run.prompt) + run.steps - 1
    expected = {
        (position, layer)
        for position in range(n_positions)
        for layer in range(run.first_dense, run.n_layers)
    }
    if seen != expected:
        missing = len(expected - seen)
        extra = len(seen - expected)
        raise ComparisonError(
            f"{run.directory}: incomplete route coverage "
            f"({missing} missing, {extra} unexpected position/layer rows)"
        )
    return raw, tuple(rows)


def compare(reference: Run, candidate: Run, contract: str,
            max_abs_gate: float, max_rel_l2_gate: float | None,
            top_n: int) -> dict:
    require_matched_run_shape(reference, candidate)
    require_backend_semantics(reference)
    require_backend_semantics(candidate)
    if contract == "exact-vq" and candidate.mode != "vq":
        raise ComparisonError("exact-vq contract requires a vq candidate")
    if contract == "dense" and candidate.mode not in {"matvec", "combined"}:
        raise ComparisonError("dense contract requires matvec or combined candidate")
    if top_n < 1:
        raise ComparisonError("top-n must be positive")
    effective_top_n = min(top_n, reference.vocab)
    if (not math.isfinite(max_abs_gate) or max_abs_gate < 0 or
            (max_rel_l2_gate is not None and
             (not math.isfinite(max_rel_l2_gate) or max_rel_l2_gate < 0))):
        raise ComparisonError("numeric gates must be finite and non-negative")

    if reference.generated != candidate.generated:
        raise ComparisonError("generated token ids differ")

    max_abs = 0.0
    diff_sq = 0.0
    ref_sq = 0.0
    compared = 0
    with reference.logits_path.open("rb") as ref_file, \
            candidate.logits_path.open("rb") as candidate_file:
        for step in range(reference.steps):
            ref_raw, ref_values = read_frame(ref_file, reference.vocab,
                                             str(reference.directory), step)
            cand_raw, cand_values = read_frame(candidate_file, candidate.vocab,
                                               str(candidate.directory), step)
            if contract == "exact-vq" and ref_raw != cand_raw:
                raise ComparisonError(f"logits are not byte-exact at step {step}")
            if any(not math.isfinite(v) for v in ref_values):
                raise ComparisonError(f"reference has a non-finite logit at step {step}")
            if any(not math.isfinite(v) for v in cand_values):
                raise ComparisonError(f"candidate has a non-finite logit at step {step}")
            ref_top = top_ids(ref_values, effective_top_n)
            cand_top = top_ids(cand_values, effective_top_n)
            if ref_top != cand_top:
                raise ComparisonError(
                    f"top-{effective_top_n} ordering differs at step {step}"
                )
            if ref_top[0] != reference.generated[step]:
                raise ComparisonError(
                    f"reference token id does not match its logits at step {step}"
                )
            if cand_top[0] != candidate.generated[step]:
                raise ComparisonError(
                    f"candidate token id does not match its logits at step {step}"
                )
            for ref_value, cand_value in zip(ref_values, cand_values):
                delta = float(cand_value) - float(ref_value)
                absolute = abs(delta)
                if absolute > max_abs:
                    max_abs = absolute
                diff_sq += delta * delta
                ref_sq += float(ref_value) * float(ref_value)
            compared += len(ref_values)

    ref_routes_raw, ref_routes = parse_routes(reference)
    cand_routes_raw, cand_routes = parse_routes(candidate)
    if ref_routes != cand_routes:
        raise ComparisonError("ordered router selections differ")
    if contract == "exact-vq" and ref_routes_raw != cand_routes_raw:
        raise ComparisonError("VQ route trace is not byte-exact")

    rel_l2 = math.sqrt(diff_sq / ref_sq) if ref_sq else math.sqrt(diff_sq)
    if contract == "dense" and max_abs > max_abs_gate:
        raise ComparisonError(
            f"max abs logit diff {max_abs:.9g} exceeds {max_abs_gate:.9g}"
        )
    if (contract == "dense" and max_rel_l2_gate is not None and
            rel_l2 > max_rel_l2_gate):
        raise ComparisonError(
            f"relative L2 {rel_l2:.9g} exceeds {max_rel_l2_gate:.9g}"
        )
    return {
        "contract": contract,
        "candidate_mode": candidate.mode,
        "steps": reference.steps,
        "logits_compared": compared,
        "route_rows": len(ref_routes),
        "max_abs": max_abs,
        "relative_l2": rel_l2,
        "top_n": effective_top_n,
        "generated_tokens_identical": True,
        "ordered_routes_identical": True,
        "byte_exact": contract == "exact-vq",
    }


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise ComparisonError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def token_hash(tokens: tuple[int, ...]) -> str:
    raw = json.dumps(list(tokens), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def public_run_summary(run: Run) -> dict:
    artifacts = {}
    for name in ("run.json", "logits.f32", "routes.txt", "steps.tsv"):
        artifacts[name] = sha256_file(run.directory / name)
    return {
        "mode": run.mode,
        "provider": run.manifest.get("provider"),
        "build_info": run.manifest.get("build_info"),
        "model": run.manifest.get("model"),
        "config": run.manifest.get("config"),
        "controls": run.manifest.get("controls"),
        "resolved_memory": run.manifest.get("resolved_memory"),
        "backend_calls": run.manifest.get("backend_calls"),
        "timing": run.manifest.get("timing"),
        "stats": run.manifest.get("stats"),
        "prompt_tokens_sha256": token_hash(run.prompt),
        "generated_tokens_sha256": token_hash(run.generated),
        "raw_artifact_sha256": artifacts,
    }


def write_public_report(path_text: str, fixture: str, reference: Run,
                        candidate: Run, result: dict) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", fixture):
        raise ComparisonError(
            "public fixture label must be 1-80 letters, digits, '.', '_' or '-'"
        )
    path = pathlib.Path(path_text)
    if path.exists():
        raise ComparisonError(f"refusing to overwrite public report {path}")
    report = {
        "format": "waste-backend-qualification-public-v1",
        "public_fixture": fixture,
        "raw_capture_policy": (
            "private; hashes only--raw prompt ids, generated ids, logits and "
            "routes are intentionally omitted"
        ),
        "comparison": result,
        "reference": public_run_summary(reference),
        "candidate": public_run_summary(candidate),
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise ComparisonError(f"cannot write public report {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="CPU capture directory")
    parser.add_argument("candidate", help="provider capture directory")
    parser.add_argument("--contract", required=True,
                        choices=("exact-vq", "dense"))
    parser.add_argument("--max-abs", type=float, default=1.0e-4,
                        help="dense max-absolute-logit gate (default: 1e-4)")
    parser.add_argument("--max-rel-l2", type=float,
                        help="optional additional dense relative-L2 gate")
    parser.add_argument("--top-n", type=int, default=10,
                        help="ordered top-token gate (default: 10)")
    parser.add_argument("--json", action="store_true",
                        help="print the result as JSON")
    parser.add_argument("--public-report",
                        help="write a redacted, hash-backed JSON report")
    parser.add_argument("--public-fixture",
                        help="explicit repo-owned public fixture label")
    args = parser.parse_args(argv)
    if bool(args.public_report) != bool(args.public_fixture):
        parser.error("--public-report and --public-fixture must be used together")
    try:
        reference = load_run(args.reference)
        candidate = load_run(args.candidate)
        result = compare(reference, candidate, args.contract, args.max_abs,
                         args.max_rel_l2, args.top_n)
        if args.public_report:
            write_public_report(args.public_report, args.public_fixture,
                                reference, candidate, result)
    except (ComparisonError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"status": "PASS", **result}, sort_keys=True))
    else:
        exact = "byte-exact; " if result["byte_exact"] else ""
        print(
            f"PASS {result['contract']} ({result['candidate_mode']}): "
            f"{exact}{result['logits_compared']:,} logits, "
            f"max abs {result['max_abs']:.9g}, "
            f"rel L2 {result['relative_l2']:.9g}; "
            f"top-{result['top_n']}, generated tokens and "
            f"{result['route_rows']:,} ordered route rows unchanged"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
