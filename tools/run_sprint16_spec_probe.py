#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sealed two-stage runner for the Sprint 16 GN100 agreement probe.

``seal`` verifies the frozen inputs, writes a complete Kimi-Linear expert
usage file, and runs the load-only co-residency check. ``capture`` requires
that seal, reproduces all four legacy A hashes, and only then touches B/H1.
This runner has no H2 mode and accepts only the frozen A+B/H1 corpus.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


DEV_CORPUS_SHA = "4fb4ca60d82ed521c0d59732c748a5f99b22f720d35e19d5590b41bc01c423fe"
H2_CORPUS_SHA = "5c5b806358a26bb8d9ce782aab190b7e10bd83f00e445d1bab7367333cee5bee"
TARGET_USAGE_SHA = "6ef8b0e752b4c5e369eae1e088010bb47243f6ee4d1d9470053067c4e85f693e"
DRAFT_TEMPLATE_SHA = "75e3cd76bd26b77c02507eb35073b66c8ffd0c88eab185ebc6f8c6e425161809"
TRACE_WIDTH = 8
TOKENS = 32
MIN_MEMAVAILABLE_KIB = 24 * 1024 * 1024
PERFORMANCE_CPUS = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]
TARGET_CACHE_MB = 40502
DRAFT_CACHE_MB = 16926
TARGET_ROLLBACK_BYTES = 536870912
DRAFT_ROLLBACK_BYTES = 134217728

A_CASES = ("composition_a", "revision_a", "color_value_a", "material_a")
B_CASES = ("composition_b", "revision_b", "color_value_b", "material_b")
H1_CASES = ("ambiguity_h", "history_h", "series_h", "display_h")
CASE_ORDER = A_CASES + B_CASES + H1_CASES

LEGACY_A_HASHES: Mapping[str, Mapping[str, str]] = {
    "composition_a": {"token": "0x09b57e1f6747aa4e", "logit": "0xb604dd7e1454c3b8", "route": "0xfffe67b059bddac0"},
    "revision_a": {"token": "0x9e7adcd1e1fd34e0", "logit": "0x261c8770708bfb7c", "route": "0x4e118156bc09da35"},
    "color_value_a": {"token": "0xd8bcff529ed734e9", "logit": "0xe10f2189581546cb", "route": "0xf48ae283c806806e"},
    "material_a": {"token": "0x4162138f88aa6d80", "logit": "0x0204f5edfccfe132", "route": "0x493e1714514071e2"},
}

PROFILE = {
    "WASTE_BACKEND": "auto", "WASTE_CUDA_KDA": "1",
    "WASTE_CUDA_DENSE": "2", "WASTE_CUDA_VQ": "2",
    "WASTE_CUDA_VQ_GROUP": "1", "WASTE_THREADS": "10",
    "WASTE_Q8": "1", "WASTE_SDOT": "0", "WASTE_I8MM": "0",
    "WASTE_LOOKAHEAD": "0", "WASTE_IO_THREADS": "2",
    "WASTE_IO_DEPTH": "2", "WASTE_LFRU_AGE_TOKENS": "0",
    "WASTE_LFRU_PRIOR_LOG2": "0", "WASTE_MLOCK": "0",
    "WASTE_PURGEABLE": "0", "WASTE_VERIFY": "0", "WASTE_DIRECT": "1",
}

USAGE_HEADER = struct.Struct("<IIQ")
USAGE_ENTRY = struct.Struct("<HHIII")
USAGE_MAGIC = 0x47535557
class CampaignError(RuntimeError):
    pass
def positive_finite(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and
            math.isfinite(float(value)) and value > 0)
def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignError(f"{path} is not a JSON object")
    return value
def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
def file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise CampaignError(f"missing {label}: {path}")
    return path
def tokenizer(model: Path) -> Path:
    for name in ("tokenizer.model", "tiktoken.model"):
        if (model / name).is_file():
            return model / name
    raise CampaignError(f"no tokenizer model in {model}")
def check_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema") != "waste.gn100.speculative_contract.v1":
        raise CampaignError("wrong Sprint 16 contract schema")
    corpora = contract.get("corpora", {})
    dev, h2 = corpora.get("development_veto", {}), corpora.get("h2", {})
    grid = contract.get("candidate_grid", {})
    if dev.get("sha256") != DEV_CORPUS_SHA or dev.get("continuation_tokens") != TOKENS:
        raise CampaignError("development corpus contract drift")
    if h2.get("sha256") != H2_CORPUS_SHA or h2.get("inference_status") != "embargoed_unspent":
        raise CampaignError("H2 is not sealed and unspent")
    if (grid.get("trace_width") != TRACE_WIDTH or
            grid.get("prompt_formats") != ["target_ids", "kimi_native"] or
            grid.get("block_widths") != [2, 4, 8]):
        raise CampaignError("candidate grid contract drift")
def check_amendment(amendment: Mapping[str, Any]) -> None:
    if (amendment.get("schema") !=
            "waste.gn100.speculative_preinference_amendment.v1" or
            amendment.get("status") != "frozen_before_new_model_inference"):
        raise CampaignError("wrong Sprint 16 pre-inference amendment")
    h2, grid, lookup, state_costs = (
        amendment.get("h2", {}), amendment.get("candidate_grid", {}),
        amendment.get("prompt_lookup", {}), amendment.get("state_costs", {}),
    )
    if (h2.get("sha256") != H2_CORPUS_SHA or
            h2.get("status") != "embargoed_unspent" or h2.get("model_steps") != 0):
        raise CampaignError("amendment does not preserve the H2 embargo")
    if (grid.get("candidate_cells") != 9 or grid.get("block_widths") != [2, 4, 8] or
            grid.get("mtp") != "excluded_unavailable"):
        raise CampaignError("amended candidate grid drift")
    if (lookup.get("max_ngram") != 4 or lookup.get("max_proposals") != 8 or
            lookup.get("occurrence_tie_break") != "most_recent"):
        raise CampaignError("prompt-lookup contract drift")
    actual_state = state_costs.get("actual_state_probe", {})
    if (state_costs.get("load_only_position_zero_role") !=
            "inference_free_fixed_kda_conv_floor_without_mla_latent_rows" or
            actual_state.get("selection") !=
            "longest_frozen_development_prompt_by_token_count" or
            actual_state.get("position") !=
            "prompt_tokens_plus_31_last_root_of_32_token_trace" or
            actual_state.get("actual_mla_latent_rows") is not True or
            actual_state.get("rollback_check") !=
            "advance_restore_replay_requires_byte_identical_logits_ordered_routes_and_post_state" or
            actual_state.get("candidate_selection_role") != "none"):
        raise CampaignError("post-prompt state-cost contract drift")
    budget = amendment.get("joint_budget", {})
    if budget != {
            "F_target_cache_mib": 59340,
            "R_S_target_cache_mib": TARGET_CACHE_MB,
            "R_S_draft_cache_mib": DRAFT_CACHE_MB,
            "target_rollback_bytes": TARGET_ROLLBACK_BYTES,
            "draft_rollback_bytes": DRAFT_ROLLBACK_BYTES,
        }:
        raise CampaignError("joint memory budget contract drift")
    safety = amendment.get("safety_measurement", {})
    if (safety.get("process_swap_scope") != "complete_model_subprocess" or
            safety.get("major_fault_scope") != "named_timed_inference_or_state_copy_window" or
            safety.get("load_startup_major_faults") != "recorded_informational_not_gated" or
            safety.get("capture_affinity") != PERFORMANCE_CPUS or
            safety.get("pm_qos") != {"scope": "child", "latency_us": 0, "holder_uid": 0}):
        raise CampaignError("capture safety-measurement contract drift")
def check_corpus(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    digest = sha(path)
    if digest == H2_CORPUS_SHA:
        raise CampaignError("H2 embargo: this runner cannot consume H2")
    if digest != DEV_CORPUS_SHA:
        raise CampaignError(f"only the frozen A+B/H1 corpus is accepted; got {digest}")
    corpus = read_json(path)
    cases = corpus.get("cases")
    if not isinstance(corpus.get("system"), str) or not isinstance(cases, list):
        raise CampaignError("invalid development corpus")
    if tuple(case.get("id") for case in cases if isinstance(case, dict)) != CASE_ORDER:
        raise CampaignError("development case set/order drift")
    by_id = {case["id"]: case for case in cases}
    for case_id in CASE_ORDER:
        ids = by_id[case_id].get("token_ids")
        if not isinstance(ids, list) or not ids or by_id[case_id].get("token_count") != len(ids):
            raise CampaignError(f"invalid prompt tokens for {case_id}")
    return corpus, by_id
def model_config(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    config = manifest.get("config")
    return config if isinstance(config, dict) else manifest
def routed_geometry(manifest: Mapping[str, Any]) -> tuple[list[int], int]:
    config = model_config(manifest)
    n_layers = config.get("num_hidden_layers")
    first_dense = config.get("first_k_dense_replace", 0)
    experts = config.get("num_experts") or config.get("n_routed_experts")
    if not all(isinstance(value, int) and not isinstance(value, bool)
               for value in (n_layers, first_dense, experts)):
        raise CampaignError("draft manifest lacks integer routed geometry")
    if n_layers <= first_dense or first_dense < 0 or experts <= 0 or experts > 65535:
        raise CampaignError("invalid draft routed geometry")
    layers_obj = manifest.get("layers")
    if isinstance(layers_obj, dict) and layers_obj:
        layers = sorted(int(layer) for layer in layers_obj)
    else:
        layers = list(range(first_dense, n_layers))
    if layers != list(range(first_dense, n_layers)):
        raise CampaignError("draft manifest does not contain every routed layer")
    return layers, experts
def write_full_usage(path: Path, manifest: Mapping[str, Any]) -> int:
    layers, experts = routed_geometry(manifest)
    count = len(layers) * experts
    if count > 0xFFFFFFFF:
        raise CampaignError("draft expert count exceeds usage format")
    chunks = [USAGE_HEADER.pack(USAGE_MAGIC, 1, 0)]
    clock = 0
    for layer in layers:
        for expert in range(experts):
            clock += 1
            chunks.append(USAGE_ENTRY.pack(layer, expert, 1, clock, 0))
    path.write_bytes(b"".join(chunks))
    return count
def git(repo: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=repo, text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise CampaignError(f"git {' '.join(arguments)} failed") from exc
def check_repo(repo: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    if git(repo, "status", "--porcelain"):
        raise CampaignError("repository must be clean")
    branch, head = git(repo, "branch", "--show-current"), git(repo, "rev-parse", "HEAD")
    if branch != contract.get("branch"):
        raise CampaignError(f"expected branch {contract.get('branch')}, got {branch}")
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(contract.get("parent_commit")), head],
            cwd=repo, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise CampaignError("contract parent is not an ancestor of HEAD") from exc
    return {"branch": branch, "head": head, "dirty": False}
def environment(draft: bool = False) -> dict[str, str]:
    result = dict(os.environ)
    for key in list(result):
        if key.startswith("WASTE_"):
            del result[key]
    result.update(PROFILE)
    if draft:
        result.update(WASTE_CUDA_VQ="0", WASTE_CUDA_VQ_GROUP="1")
    return result
def host_profile() -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    meminfo = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.split(":", 1)[0] in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "CmaTotal", "CmaFree"}:
                meminfo[line.split(":", 1)[0]] = int(line.split()[1])
    except OSError:
        pass
    return {
        "captured_utc": now(), "uname": list(os.uname()), "affinity": affinity,
        "profile_environment": PROFILE, "meminfo_kib": meminfo,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
def check_affinity() -> list[int]:
    if not hasattr(os, "sched_getaffinity"):
        raise CampaignError("Linux CPU affinity is required for the Spark profile")
    affinity = sorted(os.sched_getaffinity(0))
    if affinity != PERFORMANCE_CPUS:
        raise CampaignError(
            f"capture affinity {affinity} != registered {PERFORMANCE_CPUS}"
        )
    return affinity
def check_capture_runtime() -> dict[str, Any]:
    affinity = check_affinity()
    ppid = os.getppid()
    try:
        command = [part.decode(errors="replace") for part in
                   Path(f"/proc/{ppid}/cmdline").read_bytes().split(b"\0") if part]
        status_lines = Path(f"/proc/{ppid}/status").read_text().splitlines()
    except OSError as exc:
        raise CampaignError(f"cannot verify Q0 holder parent {ppid}: {exc}") from exc
    uid_line = next((line for line in status_lines if line.startswith("Uid:")), "")
    parent_uid = int(uid_line.split()[1]) if len(uid_line.split()) >= 2 else -1
    def option(name: str) -> str | None:
        for index, item in enumerate(command):
            if item == name and index + 1 < len(command):
                return command[index + 1]
            if item.startswith(name + "="):
                return item.split("=", 1)[1]
        return None
    if (parent_uid != 0 or not any(Path(item).name == "pm_qos_exec.py" for item in command) or
            option("--scope") != "child" or option("--latency-us") != "0"):
        raise CampaignError("capture is not a child of the registered root-held Q0 wrapper")
    return {
        "affinity": affinity, "q0_parent_pid": ppid, "q0_parent_uid": parent_uid,
        "pm_qos_scope": "child", "pm_qos_latency_us": 0,
        "parent_command_sha256": hashlib.sha256(b"\0".join(
            item.encode() for item in command)).hexdigest(),
    }
def ids(values: Sequence[int]) -> str:
    return ",".join(map(str, values))
def run_probe(command: Sequence[str], path: Path, *, draft: bool = False) -> dict[str, Any]:
    if path.exists():
        raise CampaignError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = subprocess.run(command, text=True, capture_output=True, env=environment(draft))
    path.write_text(process.stdout, encoding="utf-8")
    path.with_suffix(".stderr.txt").write_text(process.stderr, encoding="utf-8")
    write_json(path.with_suffix(".run.json"), {
        "command": ["<token-ids>" if argument.count(",") > 8 else argument for argument in command],
        "returncode": process.returncode, "elapsed_seconds": time.monotonic() - started,
        "stdout_sha256": sha(path), "profile": "draft" if draft else "target",
    })
    if process.returncode:
        raise CampaignError(f"probe failed ({process.returncode}): {' '.join(command[:2])}")
    return read_json(path)
def check_io_cuda(value: Mapping[str, Any], *, draft: bool) -> None:
    if value.get("io") != {"direct": 1, "readers": 2, "depth": 2}:
        raise CampaignError(f"I/O profile drift: {value.get('io')}")
    cuda = value.get("cuda")
    expected = {"kda": 1, "dense": 2, "vq": 0 if draft else 2, "fallbacks": 0}
    if not isinstance(cuda, dict) or any(cuda.get(key) != wanted for key, wanted in expected.items()):
        raise CampaignError(f"CUDA profile drift: {cuda}")
def check_process_safety(value: Mapping[str, Any], label: str, scope: str) -> None:
    safety = value.get("process_safety")
    if (not isinstance(safety, dict) or safety.get("vmswap_kib") != 0 or
            safety.get("timed_major_faults_delta") != 0 or
            safety.get("timed_scope") != scope):
        raise CampaignError(f"{label} process swap/major-fault gate failed: {safety}")
def check_target(value: Mapping[str, Any]) -> list[int]:
    tokens = value.get("tokens")
    if (value.get("schema") != "waste.gn100.spec_target.v1" or
            value.get("generated") != TOKENS or not isinstance(tokens, list) or len(tokens) != TOKENS):
        raise CampaignError("invalid target capture")
    for field, size in (("token_prefix_hashes", TOKENS), ("route_prefix_hashes", TOKENS),
                        ("logit_prefix_hashes", TOKENS + 1)):
        if not isinstance(value.get(field), list) or len(value[field]) != size:
            raise CampaignError(f"target {field} dimension drift")
    cache = value.get("cache")
    if not isinstance(cache, dict) or cache.get("warm_ready") != cache.get("slots"):
        raise CampaignError("target warm-cache gate failed")
    check_io_cuda(value, draft=False)
    check_process_safety(value, "target", "prefill_plus_decode")
    return tokens
def check_a(case_id: str, value: Mapping[str, Any]) -> dict[str, Any]:
    observed = {
        "token": value["token_prefix_hashes"][15],
        "logit": value["logit_prefix_hashes"][16],
        "route": value["route_prefix_hashes"][15],
    }
    expected = dict(LEGACY_A_HASHES[case_id])
    if observed != expected:
        raise CampaignError(f"legacy A replay failed for {case_id}: {observed} != {expected}")
    return {"case": case_id, "expected": expected, "observed": observed, "pass": True}
def check_prompt(value: Mapping[str, Any]) -> list[int]:
    tokens = value.get("tokens")
    if (value.get("schema") != "waste.gn100.spec_prompt.v1" or
            not isinstance(tokens, list) or value.get("token_count") != len(tokens)):
        raise CampaignError("invalid native prompt")
    return tokens
def check_state(value: Mapping[str, Any], expected_position: int) -> None:
    serialization, shadow = (
        value.get("in_memory_state_serialization"), value.get("shadow_copy_floor")
    )
    replay = value.get("roundtrip_replay")
    state_hash = value.get("state_hash")
    cache = value.get("cache")
    if (value.get("schema") != "waste.gn100.spec_state.v1" or
            value.get("actual_model_state") is not True or
            value.get("synthetic_mla_rows") is not False or
            value.get("canonical_continuation_tokens") != TOKENS or
            value.get("continuation_tokens_applied") != TOKENS - 1 or
            value.get("state_position") != expected_position or
            not isinstance(value.get("prompt_tokens"), int) or
            value["prompt_tokens"] + TOKENS - 1 != expected_position or
            not isinstance(value.get("state_bytes"), int) or value["state_bytes"] <= 0 or
            not isinstance(state_hash, str) or
            not re.fullmatch(r"0x[0-9a-f]{16}", state_hash) or
            state_hash == "0x0000000000000000" or
            not isinstance(value.get("mla_layers"), int) or value["mla_layers"] <= 0 or
            not isinstance(value.get("mla_latent_bytes"), int) or
            value["mla_latent_bytes"] <= 0):
        raise CampaignError("invalid post-prompt target state capture")
    if (not isinstance(replay, dict) or
            replay.get("logits_byte_identical") is not True or
            replay.get("ordered_routes_byte_identical") is not True or
            replay.get("post_state_byte_identical") is not True or
            replay.get("root_restored_after_check") is not True or
            not isinstance(replay.get("post_state_bytes"), int) or
            replay["post_state_bytes"] < value["state_bytes"]):
        raise CampaignError("post-prompt rollback replay is not bit-exact")
    if (not isinstance(serialization, dict) or serialization.get("bytes") != value["state_bytes"] or
            serialization.get("warmup_roundtrips") != 1 or
            not positive_finite(serialization.get("export_seconds")) or
            not positive_finite(serialization.get("import_seconds"))):
        raise CampaignError("invalid post-prompt state serialization timing")
    if (not isinstance(shadow, dict) or shadow.get("bytes") != value["state_bytes"] or
            shadow.get("optimistic") is not True or shadow.get("is_pointer_swap") is not False or
            shadow.get("is_durable_file_io") is not False or
            shadow.get("pages_pretouched") is not True or
            shadow.get("thread_creation_timed") is not False or
            shadow.get("worker_dispatch_timed") is not True or
            shadow.get("repeats") != 7):
        raise CampaignError("invalid post-prompt shadow-copy timing")
    for name, threads in (("threads_1", 1), ("threads_10", 10)):
        row = shadow.get(name)
        if (not isinstance(row, dict) or row.get("threads") != threads or
                not positive_finite(row.get("best_seconds")) or
                not positive_finite(row.get("median_seconds")) or
                not positive_finite(row.get("best_gib_s")) or
                not positive_finite(row.get("median_gib_s")) or
                row["median_seconds"] < row["best_seconds"] or
                row["best_gib_s"] < row["median_gib_s"]):
            raise CampaignError(f"invalid post-prompt {name} copy timing")
    if (not isinstance(cache, dict) or cache.get("warm_ready") != cache.get("slots") or
            not isinstance(cache.get("routed_records"), int)):
        raise CampaignError("post-prompt target warm-cache gate failed")
    check_io_cuda(value, draft=False)
    check_process_safety(
        value, "post-prompt state",
        "prefill_continuation_export_import_shadow_copy",
    )
def check_load_state_costs(value: Mapping[str, Any]) -> None:
    serialization = value.get("in_memory_state_serialization")
    shadow = value.get("target_shadow_copy_floor")
    target, draft = value.get("target"), value.get("draft")
    if not all(isinstance(item, dict) for item in (serialization, shadow, target, draft)):
        raise CampaignError("load-only state-cost rows are missing")
    if (serialization.get("warmup_roundtrips") != 1 or
            serialization.get("rollback_pages_pretouched") is not True):
        raise CampaignError("invalid load-only in-memory state warmup")
    for name, model in (("target", target), ("draft", draft)):
        row = serialization.get(name)
        if (not isinstance(row, dict) or row.get("bytes") != model.get("state0_bytes") or
                not positive_finite(row.get("export_seconds")) or
                not positive_finite(row.get("import_seconds"))):
            raise CampaignError(f"invalid load-only {name} state timing")
    if (shadow.get("bytes") != target.get("state0_bytes") or
            shadow.get("optimistic") is not True or
            shadow.get("is_pointer_swap") is not False or
            shadow.get("is_durable_file_io") is not False or
            shadow.get("temporary_mapping_released_before_memory_snapshot") is not True or
            shadow.get("repeats") != 7):
        raise CampaignError("invalid load-only target shadow-copy floor")
    for name, threads in (("threads_1", 1), ("threads_10", 10)):
        row = shadow.get(name)
        if (not isinstance(row, dict) or row.get("threads") != threads or
                not positive_finite(row.get("best_seconds")) or
                not positive_finite(row.get("median_seconds")) or
                row["median_seconds"] < row["best_seconds"]):
            raise CampaignError(f"invalid load-only {name} copy timing")
def check_teacher(value: Mapping[str, Any], targets: Sequence[int]) -> None:
    n = len(targets)
    rows, widths, matches = (
        value.get("branch_predictions"), value.get("branch_widths"),
        value.get("prefix_match_lengths"),
    )
    logit_hashes = value.get("branch_logit_row_hashes")
    route_hashes = value.get("branch_route_row_hashes")
    if (value.get("schema") != "waste.gn100.spec_teacher.v1" or
            value.get("targets") != list(targets) or value.get("branch_width") != TRACE_WIDTH or
            not all(isinstance(item, list) and len(item) == n for item in
                    (rows, widths, matches, logit_hashes, route_hashes))):
        raise CampaignError("invalid draft branch capture")
    zero_hash = "0x0000000000000000"
    valid_hash = re.compile(r"0x[0-9a-f]{16}").fullmatch
    for index, (row, width, match, logits, routes) in enumerate(
            zip(rows, widths, matches, logit_hashes, route_hashes)):
        expected_width = min(TRACE_WIDTH, n - index)
        if (width != expected_width or len(row) != TRACE_WIDTH or
                len(logits) != TRACE_WIDTH or len(routes) != TRACE_WIDTH or
                any(token != -1 for token in row[width:])):
            raise CampaignError(f"branch dimension/padding drift at prefix {index}")
        if (any(not isinstance(item, str) or not valid_hash(item) for item in logits + routes) or
                any(item == zero_hash for item in logits[:width]) or
                any(item != zero_hash for item in logits[width:]) or
                any(item == zero_hash for item in routes[:max(0, width - 1)]) or
                any(item != zero_hash for item in routes[max(0, width - 1):])):
            raise CampaignError(f"branch hash/padding drift at prefix {index}")
        derived = 0
        while derived < width and row[derived] == targets[index + derived]:
            derived += 1
        if match != derived:
            raise CampaignError(f"prefix match accounting drift at {index}")
    cache = value.get("cache")
    if not isinstance(cache, dict) or cache.get("fully_warm_at_start") is not True:
        raise CampaignError("draft was not fully resident")
    if value.get("snapshot_count") != n or value.get("restore_count") != n:
        raise CampaignError("draft snapshot/restore accounting drift")
    check_io_cuda(value, draft=True)
    check_process_safety(
        value, "draft teacher",
        "prefill_plus_teacher_branches_snapshots_restores",
    )
def manifest(output: Path) -> None:
    manifest_path, sums_path = output / "manifest.json", output / "MANIFEST.sha256"
    files = sorted(path for path in output.rglob("*") if path.is_file() and path not in (manifest_path, sums_path))
    write_json(manifest_path, {"schema": "waste.gn100.spec_probe_manifest.v1", "files": [
        {"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}
        for path in files
    ]})
    files.append(manifest_path)
    sums_path.write_text("".join(f"{sha(path)}  {path.relative_to(output).as_posix()}\n" for path in files))
def verify_manifest(output: Path) -> dict[str, str]:
    manifest_path, sums_path = output / "manifest.json", output / "MANIFEST.sha256"
    value = read_json(file(manifest_path, "sealed manifest"))
    entries = value.get("files")
    if value.get("schema") != "waste.gn100.spec_probe_manifest.v1" or not isinstance(entries, list):
        raise CampaignError("invalid sealed evidence manifest")
    listed: list[Path] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise CampaignError(f"invalid sealed manifest entry {index}")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in {
                "manifest.json", "MANIFEST.sha256"}:
            raise CampaignError(f"unsafe sealed manifest path {relative}")
        path = output / relative
        if (not path.is_file() or path.stat().st_size != entry.get("bytes") or
                sha(path) != entry.get("sha256")):
            raise CampaignError(f"sealed evidence drift: {relative}")
        listed.append(path)
    observed = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path not in (manifest_path, sums_path)
    )
    if listed != observed:
        raise CampaignError("sealed evidence file set drift")
    expected_sums = "".join(
        f"{sha(path)}  {path.relative_to(output).as_posix()}\n"
        for path in [*listed, manifest_path]
    )
    if sums_path.read_text(encoding="utf-8") != expected_sums:
        raise CampaignError("sealed checksum ledger drift")
    return {"manifest_sha256": sha(manifest_path), "ledger_sha256": sha(sums_path)}
def fixed_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    contract = read_json(file(args.contract, "contract"))
    check_contract(contract)
    amendment = read_json(file(args.amendment, "pre-inference amendment"))
    check_amendment(amendment)
    corpus, cases = check_corpus(file(args.corpus, "development corpus"))
    repo = check_repo(args.repo, contract)
    target_manifest_path = file(args.target_model / "manifest.json", "target manifest")
    draft_manifest_path = file(args.draft_model / "manifest.json", "draft manifest")
    artifacts = {
        "contract": args.contract, "amendment": args.amendment,
        "corpus": args.corpus, "probe": args.probe,
        "target_manifest": target_manifest_path, "draft_manifest": draft_manifest_path,
        "target_tokenizer": file(tokenizer(args.target_model), "target tokenizer"),
        "draft_tokenizer": file(tokenizer(args.draft_model), "draft tokenizer"),
        "target_specials": file(args.target_model / "specials.json", "target specials"),
        "draft_specials": file(args.draft_model / "specials.json", "draft specials"),
        "draft_template": file(args.draft_model / "chat_template.jinja", "draft template"),
        "target_usage": args.target_usage,
    }
    expected = contract["models"]
    expected_hashes = {
        "target_manifest": expected["target_manifest_sha256"],
        "draft_manifest": expected["draft_manifest_sha256"],
        "target_tokenizer": expected["tokenizer_sha256"],
        "draft_tokenizer": expected["tokenizer_sha256"],
        "target_specials": expected["target_specials_sha256"],
        "draft_specials": expected["draft_specials_sha256"],
        "draft_template": DRAFT_TEMPLATE_SHA,
        "target_usage": TARGET_USAGE_SHA,
    }
    hashes = {name: sha(file(path, name)) for name, path in artifacts.items()}
    for name, wanted in expected_hashes.items():
        if hashes[name] != wanted:
            raise CampaignError(f"{name} hash drift: {hashes[name]} != {wanted}")
    return contract, corpus, cases, {"repository": repo, "hashes": hashes}
def fingerprint(args: argparse.Namespace, provenance: Mapping[str, Any], draft_usage_sha: str) -> str:
    value = {
        "provenance": provenance, "target_cache_mb": args.target_cache_mb,
        "draft_cache_mb": args.draft_cache_mb,
        "target_rollback_bytes": args.target_rollback_bytes,
        "draft_rollback_bytes": args.draft_rollback_bytes,
        "draft_usage_sha256": draft_usage_sha, "profile": PROFILE,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
def check_registered_budget(args: argparse.Namespace) -> None:
    registered = {
        "target_cache_mb": TARGET_CACHE_MB,
        "draft_cache_mb": DRAFT_CACHE_MB,
        "target_rollback_bytes": TARGET_ROLLBACK_BYTES,
        "draft_rollback_bytes": DRAFT_ROLLBACK_BYTES,
    }
    for name, expected in registered.items():
        if getattr(args, name) != expected:
            raise CampaignError(
                f"--{name.replace('_', '-')} must equal registered value {expected}"
            )
def record_failure(args: argparse.Namespace, exc: Exception) -> Path:
    payload = {
        "stage": args.stage, "failed_utc": now(), "error": str(exc),
        "h2_model_steps": 0,
    }
    owns_output = (
        (args.stage == "seal" and getattr(args, "_stage_output_owned", False)) or
        (args.stage == "capture" and
         getattr(args, "_inference_started_this_invocation", False))
    )
    if not owns_output:
        # Existing evidence is immutable. Preflight errors and accidental
        # reruns are archived beside it, never added to or re-manifested into
        # the campaign they failed to start.
        directory = args.output.parent / f"{args.output.name}-{args.stage}-failures"
        path = directory / f"attempt-{time.time_ns()}.json"
        write_json(path, {**payload, "sealed_output_unchanged": True})
        return path
    base = args.output / "failure.json"
    path = base if not base.exists() else args.output / f"failure-{time.time_ns()}.json"
    write_json(path, {**payload, "sealed_output_unchanged": False})
    # Seal partial load evidence or every post-inference failure. The prior
    # pre-inference manifest hash remains embedded in inference-start.json.
    manifest(args.output)
    return path
def seal(args: argparse.Namespace) -> None:
    if args.output.exists() and any(args.output.iterdir()):
        raise CampaignError("seal output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    args._stage_output_owned = True
    check_affinity()
    contract, _corpus, _cases, provenance = fixed_inputs(args)
    draft_usage = args.output / "draft-full-usage.waste"
    expected_records = write_full_usage(draft_usage, read_json(args.draft_model / "manifest.json"))
    digest = fingerprint(args, provenance, sha(draft_usage))
    write_json(args.output / "host-profile.json", host_profile())
    load = run_probe([
        str(args.probe), "load", str(args.target_model), str(args.target_cache_mb),
        str(args.target_usage), str(args.draft_model), str(args.draft_cache_mb),
        str(draft_usage), str(args.target_rollback_bytes), str(args.draft_rollback_bytes),
    ], args.output / "load-only.json")
    target, draft = load.get("target", {}), load.get("draft", {})
    if (load.get("schema") != "waste.gn100.spec_load.v1" or load.get("vmswap_kib") != 0 or
            load.get("memavailable_after_touch_kib", 0) < MIN_MEMAVAILABLE_KIB):
        raise CampaignError("load-only memory/swap gate failed")
    for record, vq in ((target, 2), (draft, 0)):
        if (record.get("direct_io") != 1 or record.get("readers") != 2 or record.get("depth") != 2 or
                record.get("cuda_kda_requested") != 1 or record.get("cuda_dense_requested") != 2 or
                record.get("cuda_vq_requested") != vq):
            raise CampaignError("load-only profile drift")
    if (draft.get("routed_records") != expected_records or draft.get("warm_ready") != expected_records or
            draft.get("cache_covers_all_records") is not True):
        raise CampaignError("draft is not fully resident")
    if (load.get("target_rollback_bytes") != args.target_rollback_bytes or
            load.get("draft_rollback_bytes") != args.draft_rollback_bytes):
        raise CampaignError("rollback-byte contract drift")
    check_load_state_costs(load)
    write_json(args.output / "pre-inference-seal.json", {
        "schema": "waste.gn100.spec_pre_inference_seal.v1", "sealed_utc": now(),
        "configuration_sha256": digest, "provenance": provenance,
        "draft_usage_sha256": sha(draft_usage), "draft_usage_entries": expected_records,
        "load_only_sha256": sha(args.output / "load-only.json"),
        "h2": {"sha256": H2_CORPUS_SHA, "status": "embargoed_unspent", "model_steps": 0},
        "inference_started": False, "contract_parent": contract["parent_commit"],
    })
    manifest(args.output)
def capture(args: argparse.Namespace) -> None:
    seal_path, started_path = args.output / "pre-inference-seal.json", args.output / "inference-start.json"
    if not seal_path.is_file() or started_path.exists():
        raise CampaignError("capture requires one unused pre-inference seal")
    _contract, corpus, cases, provenance = fixed_inputs(args)
    draft_usage = file(args.output / "draft-full-usage.waste", "sealed draft usage")
    sealed = read_json(seal_path)
    sealed_manifest = verify_manifest(args.output)
    digest = fingerprint(args, provenance, sha(draft_usage))
    if sealed.get("configuration_sha256") != digest or sealed.get("inference_started") is not False:
        raise CampaignError("current inputs do not match the load-only seal")
    if sealed.get("load_only_sha256") != sha(file(args.output / "load-only.json", "load-only evidence")):
        raise CampaignError("load-only evidence no longer matches its seal")
    runtime = check_capture_runtime()
    # Every preflight check is complete. From this point the invocation owns
    # any mutation and must seal even a start-marker write failure as a
    # partial attempt; no model call occurs before the marker is durable.
    args._inference_started_this_invocation = True
    write_json(started_path, {"started_utc": now(), "seal_sha256": sha(seal_path),
                              "sealed_manifest": sealed_manifest, "runtime": runtime,
                              "h2_model_steps": 0})

    targets: dict[str, list[int]] = {}
    replay = []
    for case_id in A_CASES:
        case = cases[case_id]
        value = run_probe([
            str(args.probe), "target", str(args.target_model), str(args.target_cache_mb),
            str(args.target_usage), ids(case["token_ids"]), str(TOKENS),
        ], args.output / "target" / f"{case_id}.json")
        targets[case_id] = check_target(value)
        replay.append(check_a(case_id, value))
    write_json(args.output / "a-replay-gate.json", {
        "schema": "waste.gn100.spec_a_replay_gate.v1", "pass": True,
        "cases": replay, "b_h1_model_steps_before_pass": 0,
    })

    summary: dict[str, Any] = {}
    tiers = {**{case: "A" for case in A_CASES}, **{case: "B" for case in B_CASES},
             **{case: "H1" for case in H1_CASES}}
    for case_id in CASE_ORDER:
        case = cases[case_id]
        if case_id not in targets:
            value = run_probe([
                str(args.probe), "target", str(args.target_model), str(args.target_cache_mb),
                str(args.target_usage), ids(case["token_ids"]), str(TOKENS),
            ], args.output / "target" / f"{case_id}.json")
            targets[case_id] = check_target(value)
        native = run_probe([
            str(args.probe), "prompt", str(args.draft_model), corpus["system"], case["user"],
        ], args.output / "prompts" / f"{case_id}-kimi_native.json", draft=True)
        prompts = {"target_ids": case["token_ids"], "kimi_native": check_prompt(native)}
        summary[case_id] = {}
        for prompt_format, prompt_ids in prompts.items():
            path = args.output / "teacher" / prompt_format / f"{case_id}.json"
            value = run_probe([
                str(args.probe), "teacher", str(args.draft_model), str(args.draft_cache_mb),
                str(draft_usage), ids(prompt_ids), ids(targets[case_id]),
            ], path, draft=True)
            check_teacher(value, targets[case_id])
            labels = {"case_id": case_id, "tier": tiers[case_id],
                      "family": case["family"], "format": prompt_format}
            append_jsonl(args.output / f"target-{prompt_format}.jsonl",
                         {**read_json(args.output / "target" / f"{case_id}.json"), **labels})
            append_jsonl(args.output / f"teacher-{prompt_format}.jsonl", {**value, **labels})
            summary[case_id][prompt_format] = {
                "marginal_agreement": value.get("marginal_agreement"),
                "prefix_match_lengths": value["prefix_match_lengths"], "sha256": sha(path),
            }
    state_case_id = max(CASE_ORDER, key=lambda item: len(cases[item]["token_ids"]))
    state_case = cases[state_case_id]
    state_value = run_probe([
        str(args.probe), "state", str(args.target_model), str(args.target_cache_mb),
        str(args.target_usage), ids(state_case["token_ids"]), ids(targets[state_case_id]),
    ], args.output / "target-state-cost.json")
    expected_position = len(state_case["token_ids"]) + TOKENS - 1
    check_state(state_value, expected_position)
    write_json(args.output / "campaign-summary.json", {
        "schema": "waste.gn100.spec_probe_campaign.v1", "completed_utc": now(),
        "status": "valid_development_agreement_capture", "case_order": CASE_ORDER,
        "selection": A_CASES + B_CASES, "veto_only": H1_CASES,
        "trace_width": TRACE_WIDTH, "derived_widths_only": [2, 4, 8],
        "candidate_selection_status": "pending_registered_F_R_and_verifier_cost_calibration",
        "a_replay_gate": "pass", "teacher": summary,
        "post_prompt_state_cost": {
            "case_id": state_case_id, "family": state_case["family"],
            "state_position": expected_position,
            "sha256": sha(args.output / "target-state-cost.json"),
        },
        "h2": {"status": "embargoed_unspent", "model_steps": 0},
    })
    manifest(args.output)
def make_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", type=Path, default=root)
    common.add_argument("--contract", type=Path, default=root / "docs/gn100/sprint16-speculative-contract.json")
    common.add_argument("--amendment", type=Path,
                        default=root / "docs/gn100/sprint16-preinference-amendment.json")
    common.add_argument("--corpus", type=Path, default=root / "docs/gn100/sprint14-heldout-corpus.json")
    common.add_argument("--probe", type=Path, required=True)
    common.add_argument("--target-model", type=Path, required=True)
    common.add_argument("--draft-model", type=Path, required=True)
    common.add_argument("--target-usage", type=Path, required=True)
    common.add_argument("--output", type=Path, required=True)
    common.add_argument("--target-cache-mb", type=int, required=True)
    common.add_argument("--draft-cache-mb", type=int, required=True)
    common.add_argument("--target-rollback-bytes", type=int, required=True)
    common.add_argument("--draft-rollback-bytes", type=int, required=True)
    parser = argparse.ArgumentParser(description="sealed Sprint 16 agreement probe runner")
    commands = parser.add_subparsers(dest="stage", required=True)
    commands.add_parser("seal", parents=[common], help="load-only check and seal")
    commands.add_parser("capture", parents=[common], help="A gate then A+B/H1 trace")
    return parser
def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    args.output = args.output.resolve()
    args._stage_output_owned = False
    args._inference_started_this_invocation = False
    try:
        for name in ("target_cache_mb", "draft_cache_mb", "target_rollback_bytes", "draft_rollback_bytes"):
            if getattr(args, name) <= 0:
                raise CampaignError(f"--{name.replace('_', '-')} must be positive")
        check_registered_budget(args)
        for name in ("repo", "contract", "amendment", "corpus", "probe",
                     "target_model", "draft_model", "target_usage"):
            setattr(args, name, getattr(args, name).resolve())
        (seal if args.stage == "seal" else capture)(args)
    except (CampaignError, OSError, KeyError, ValueError) as exc:
        record_failure(args, exc)
        print(f"run_sprint16_spec_probe: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
