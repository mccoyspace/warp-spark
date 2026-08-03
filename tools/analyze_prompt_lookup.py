#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure a frozen prompt-lookup speculative baseline.

At every target root, the predictor finds the longest (at most four-token)
suffix that occurred earlier in the canonical context, preferring the most
recent occurrence, and copies up to eight already-seen following tokens.
Lookup timing is descriptive and is never an acceptance statistic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEV_CORPUS_SHA = "4fb4ca60d82ed521c0d59732c748a5f99b22f720d35e19d5590b41bc01c423fe"
H2_CORPUS_SHA = "5c5b806358a26bb8d9ce782aab190b7e10bd83f00e445d1bab7367333cee5bee"
TARGET_SCHEMA = "waste.gn100.spec_target.v1"
OUTPUT_SCHEMA = "waste.gn100.prompt_lookup.v1"
MAX_NGRAM = 4
MAX_PROPOSALS = 8
WIDTHS = (2, 4, 8)
EXPECTED_TOKENS = 32
SPLIT_TIERS = {
    "calibration_train": "A",
    "within_family_holdout": "B",
    "unseen_family_holdout": "H1",
}


class LookupError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer_list(value: Any, where: str) -> list[int]:
    if not isinstance(value, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value):
        raise LookupError(f"{where} must be an array of nonnegative integers")
    return value


def load_corpus(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    digest = sha256(path)
    if digest == H2_CORPUS_SHA:
        raise LookupError("H2 inference embargo: prompt lookup cannot consume H2")
    if digest != DEV_CORPUS_SHA:
        raise LookupError(f"only the frozen Sprint 14 corpus is accepted; got {digest}")
    try:
        corpus = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LookupError(f"cannot read corpus {path}: {exc}") from exc
    if not isinstance(corpus, dict) or not isinstance(corpus.get("cases"), list):
        raise LookupError("corpus must contain cases")
    cases = corpus["cases"]
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise LookupError(f"corpus case {index} is not an object")
        for field in ("id", "family", "split"):
            if not isinstance(case.get(field), str) or not case[field]:
                raise LookupError(f"corpus case {index} lacks {field}")
        prompt = integer_list(case.get("token_ids"), f"corpus {case['id']}.token_ids")
        if case.get("token_count") != len(prompt) or case["split"] not in SPLIT_TIERS:
            raise LookupError(f"corpus case {case['id']} metadata drift")
    return corpus, cases


def load_targets(path: Path, cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise LookupError(f"{path}:{line_number}: row is not an object")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise LookupError(f"cannot read target JSONL {path}: {exc}") from exc
    if len(rows) != len(cases):
        raise LookupError(f"target rows {len(rows)} != corpus cases {len(cases)}")
    for index, (row, case) in enumerate(zip(rows, cases), 1):
        expected_tier = SPLIT_TIERS[case["split"]]
        if (row.get("schema") != TARGET_SCHEMA or row.get("case_id") != case["id"] or
                row.get("tier") != expected_tier or row.get("family") != case["family"]):
            raise LookupError(f"target row {index} identity/tier/family drift")
        tokens = integer_list(row.get("tokens"), f"target row {index}.tokens")
        if row.get("generated") != len(tokens) or len(tokens) != EXPECTED_TOKENS:
            raise LookupError(f"target row {index} must contain {EXPECTED_TOKENS} tokens")
    return rows


def lookup(context: Sequence[int]) -> tuple[int, int | None, list[int]]:
    """Return n-gram length, most-recent occurrence, and copied continuation."""
    for ngram in range(min(MAX_NGRAM, len(context)), 0, -1):
        suffix = context[-ngram:]
        occurrence = None
        # Excludes the suffix itself and requires at least one known follower.
        for start in range(0, len(context) - ngram):
            if list(context[start:start + ngram]) == list(suffix):
                occurrence = start
        if occurrence is not None:
            following = occurrence + ngram
            return ngram, occurrence, list(context[following:following + MAX_PROPOSALS])
    return 0, None, []


def build_roots(prompt: Sequence[int], targets: Sequence[int]) -> tuple[list[dict[str, Any]], int]:
    roots: list[dict[str, Any]] = []
    lookup_ns = 0
    for root in range(len(targets)):
        context = [*prompt, *targets[:root]]
        started = time.perf_counter_ns()
        ngram, occurrence, proposals = lookup(context)
        elapsed = time.perf_counter_ns() - started
        lookup_ns += elapsed
        compare = min(len(proposals), len(targets) - root)
        prefix = 0
        while prefix < compare and proposals[prefix] == targets[root + prefix]:
            prefix += 1
        first_mismatch = None if prefix == compare else prefix + 1
        rejection_positions = (
            [] if first_mismatch is None else list(range(first_mismatch, compare + 1))
        )
        roots.append({
            "root": root,
            "context_tokens": len(context),
            "remaining": len(targets) - root,
            "ngram": ngram,
            "occurrence": occurrence,
            "proposals_available": len(proposals),
            "proposals": proposals,
            "prefix_match_length": prefix,
            "first_mismatch": first_mismatch,
            "rejection_positions": rejection_positions,
            "lookup_cpu_ns": elapsed,
        })
    return roots, lookup_ns


def curve(roots: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(roots)
    survival: dict[str, Any] = {}
    for width in range(1, MAX_PROPOSALS + 1):
        eligible = [row for row in rows if row["remaining"] >= width]
        survived = sum(
            row["proposals_available"] >= width and row["prefix_match_length"] >= width
            for row in eligible
        )
        missing = sum(row["proposals_available"] < width for row in eligible)
        survival[f"Q{width}"] = {
            "survived": survived,
            "eligible_roots": len(eligible),
            "missing_proposals": missing,
            "rate": survived / len(eligible) if eligible else None,
        }
    mismatch = Counter(
        "no_proposal" if not row["proposals_available"] else
        "none" if row["first_mismatch"] is None else str(row["first_mismatch"])
        for row in rows
    )
    rejected = Counter(
        position for row in rows for position in row["rejection_positions"]
    )
    return {
        "roots": len(rows),
        "roots_with_proposals": sum(bool(row["proposals_available"]) for row in rows),
        "prefix_survival": survival,
        "first_mismatch_histogram": {
            **{str(position): mismatch[str(position)] for position in range(1, 9)},
            "none": mismatch["none"], "no_proposal": mismatch["no_proposal"],
        },
        "rejection_position_histogram": {
            str(position): rejected[position] for position in range(1, 9)
        },
    }


def simulate(targets: Sequence[int], roots: Sequence[Mapping[str, Any]], width: int) -> dict[str, Any]:
    position = proposals = accepted = rejected = corrections = bonuses = direct = 0
    emitted: list[int] = []
    mismatch, rejection_positions = Counter(), Counter()
    blocks: list[dict[str, Any]] = []
    while position < len(targets):
        remaining = len(targets) - position
        available = roots[position]["proposals_available"]
        block_width = min(width, available, remaining - 1)
        if block_width == 0:
            emitted.append(targets[position])
            blocks.append({"root": position, "proposals": 0, "accepted": 0,
                           "terminal": "direct", "committed": 1})
            direct += 1
            position += 1
            continue
        candidate = roots[position]["proposals"][:block_width]
        block_accepted = 0
        while (block_accepted < block_width and
               candidate[block_accepted] == targets[position + block_accepted]):
            block_accepted += 1
        proposals += block_width
        accepted += block_accepted
        rejected += block_width - block_accepted
        emitted.extend(candidate[:block_accepted])
        if block_accepted < block_width:
            first = block_accepted + 1
            mismatch[str(first)] += 1
            for rejected_position in range(first, block_width + 1):
                rejection_positions[str(rejected_position)] += 1
            terminal = "correction"
            corrections += 1
        else:
            mismatch["none"] += 1
            terminal = "bonus"
            bonuses += 1
        emitted.append(targets[position + block_accepted])
        committed = block_accepted + 1
        blocks.append({
            "root": position, "proposals": block_width, "accepted": block_accepted,
            "rejected": block_width - block_accepted, "terminal": terminal,
            "first_mismatch": None if terminal == "bonus" else block_accepted + 1,
            "committed": committed,
        })
        position += committed
    verifier_blocks = corrections + bonuses
    accounting = {
        "committed_eq_A_plus_M_plus_U_plus_D": (
            len(emitted) == accepted + corrections + bonuses + direct),
        "Q_eq_M_plus_U": verifier_blocks == corrections + bonuses,
        "P_eq_A_plus_rejected": proposals == accepted + rejected,
        "emitted_equals_target": emitted == list(targets),
    }
    if not all(accounting.values()):
        raise LookupError(f"k={width} trajectory accounting failed")
    return {
        "k": width, "target_tokens": len(targets), "verifier_blocks": verifier_blocks,
        "proposals": proposals, "accepted": accepted, "rejected": rejected,
        "corrections": corrections, "bonuses": bonuses, "direct": direct,
        "committed": len(emitted), "mean_committed_per_verifier": (
            None if not verifier_blocks else (len(emitted) - direct) / verifier_blocks),
        "first_mismatch_histogram": {
            **{str(position): mismatch[str(position)] for position in range(1, width + 1)},
            "none": mismatch["none"],
        },
        "rejection_position_histogram": {
            str(position): rejection_positions[str(position)]
            for position in range(1, width + 1)
        },
        "accounting": accounting, "blocks": blocks,
    }


def merge_trajectories(rows: Sequence[Mapping[str, Any]], width: int) -> dict[str, Any]:
    keys = ("target_tokens", "verifier_blocks", "proposals", "accepted", "rejected",
            "corrections", "bonuses", "direct", "committed")
    merged = {key: sum(row[key] for row in rows) for key in keys}
    mismatches, rejections = Counter(), Counter()
    for row in rows:
        mismatches.update(row["first_mismatch_histogram"])
        rejections.update(row["rejection_position_histogram"])
    merged.update({
        "k": width,
        "mean_committed_per_verifier": (
            None if not merged["verifier_blocks"] else
            (merged["committed"] - merged["direct"]) / merged["verifier_blocks"]),
        "first_mismatch_histogram": {
            **{str(position): mismatches[str(position)] for position in range(1, width + 1)},
            "none": mismatches["none"],
        },
        "rejection_position_histogram": {
            str(position): rejections[str(position)] for position in range(1, width + 1)
        },
        "accounting": {
            "committed_eq_A_plus_M_plus_U_plus_D": (
                merged["committed"] == merged["accepted"] + merged["corrections"]
                + merged["bonuses"] + merged["direct"]),
            "Q_eq_M_plus_U": merged["verifier_blocks"] == merged["corrections"] + merged["bonuses"],
            "P_eq_A_plus_rejected": merged["proposals"] == merged["accepted"] + merged["rejected"],
            "committed_equals_target_tokens": merged["committed"] == merged["target_tokens"],
        },
    })
    if not all(merged["accounting"].values()):
        raise LookupError(f"aggregate k={width} accounting failed")
    return merged


def analyze(cases: Sequence[Mapping[str, Any]], targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    family_roots: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    tier_roots: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    family_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    tier_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    total_ns = 0
    for case, target in zip(cases, targets):
        roots, lookup_ns = build_roots(case["token_ids"], target["tokens"])
        tier = SPLIT_TIERS[case["split"]]
        trajectories = {str(width): simulate(target["tokens"], roots, width) for width in WIDTHS}
        output = {
            "case_id": case["id"], "family": case["family"], "tier": tier,
            "tokens": len(target["tokens"]), "lookup_cpu_seconds": lookup_ns / 1e9,
            "curve": curve(roots), "trajectories": trajectories, "roots": roots,
        }
        outputs.append(output)
        family_roots[case["family"]].extend(roots)
        tier_roots[tier].extend(roots)
        family_rows[case["family"]].append(output)
        tier_rows[tier].append(output)
        total_ns += lookup_ns
    all_roots = [root for row in outputs for root in row["roots"]]
    def group(rows: Sequence[Mapping[str, Any]], roots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "case_ids": [row["case_id"] for row in rows],
            "curve": curve(roots),
            "trajectories": {
                str(width): merge_trajectories(
                    [row["trajectories"][str(width)] for row in rows], width
                ) for width in WIDTHS
            },
        }
    return {
        "schema": OUTPUT_SCHEMA,
        "algorithm": {"max_ngram": MAX_NGRAM, "max_proposals": MAX_PROPOSALS,
                      "tie_break": "most_recent", "sampling": "greedy"},
        "timing": {"role": "descriptive_only", "lookup_cpu_seconds": total_ns / 1e9},
        "cases": outputs,
        "families": {family: group(family_rows[family], roots)
                     for family, roots in family_roots.items()},
        "tiers": {tier: group(tier_rows[tier], roots)
                  for tier, roots in tier_roots.items()},
        "aggregate": {
            "curve": curve(all_roots),
            "trajectories": {str(width): merge_trajectories(
                [row["trajectories"][str(width)] for row in outputs], width
            ) for width in WIDTHS},
        },
    }


def print_human(result: Mapping[str, Any]) -> None:
    aggregate = result["aggregate"]
    rates = aggregate["curve"]["prefix_survival"]
    print(f"{len(result['cases'])} cases; lookup CPU {result['timing']['lookup_cpu_seconds']:.6f}s")
    print("  " + "  ".join(
        f"Q{width}={rates[f'Q{width}']['rate']:.3%}" for width in range(1, 9)
    ))
    for width in WIDTHS:
        row = aggregate["trajectories"][str(width)]
        print(f"k={width}: proposals={row['proposals']} accepted={row['accepted']} "
              f"blocks={row['verifier_blocks']} direct={row['direct']} "
              f"committed/block={row['mean_committed_per_verifier']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("target", type=Path, help="enriched target JSONL")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        _corpus, cases = load_corpus(args.corpus)
        result = analyze(cases, load_targets(args.target, cases))
    except (LookupError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"analyze_prompt_lookup: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
