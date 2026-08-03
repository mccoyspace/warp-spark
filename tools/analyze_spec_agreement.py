#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Analyze exact greedy draft/target agreement traces.

The Sprint 16 probe writes one target object and one teacher-forced draft
object per JSONL row.  Rows are paired in file order.  The draft trace is
teacher-forced on the target tokens, so it is sufficient to find the first
mismatch in a speculative block: every proposal before that mismatch has the
same prefix as the target.  Predictions after the first mismatch are not used
as if they came from the rejected branch.

Prefix survival uses the retained autoregressive branch at every within-case
root eligible for j proposals, so Qj has denominator N-j+1.  It never joins
two prompt rows.  Trajectory simulations are separate and exact for k=2,4,8.
At each root they issue ``min(k, remaining - 1)`` proposals, preserving one
target token for correction or bonus.  Only a final single token can take the
direct tail.  The same row-preserving summaries are reported for each family,
tier, and prompt format; grouping never creates a branch across case rows.

No independent-agreement or p**j approximation is computed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from typing import Any, Iterable


TARGET_SCHEMA = "waste.gn100.spec_target.v1"
TEACHER_SCHEMA = "waste.gn100.spec_teacher.v1"
OUTPUT_SCHEMA = "waste.gn100.spec_agreement.v1"
MAX_WIDTH = 8
WIDTHS = (2, 4, 8)


class AgreementError(ValueError):
    """A malformed or mutually incompatible agreement trace."""


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AgreementError(f"{where} must be an integer >= {minimum}")
    return value


def _integer_list(
    value: Any, where: str, *, binary: bool = False, minimum: int = 0
) -> list[int]:
    if not isinstance(value, list):
        raise AgreementError(f"{where} must be an array")
    out = []
    for index, item in enumerate(value):
        item = _integer(item, f"{where}[{index}]", minimum=minimum)
        if binary and item not in (0, 1):
            raise AgreementError(f"{where}[{index}] must be 0 or 1")
        out.append(item)
    return out


def _integer_matrix(
    value: Any, where: str, rows: int, columns: int, *, minimum: int = 0
) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != rows:
        raise AgreementError(f"{where} must contain {rows} rows")
    out = []
    for index, row in enumerate(value):
        parsed = _integer_list(row, f"{where}[{index}]", minimum=minimum)
        if len(parsed) != columns:
            raise AgreementError(
                f"{where}[{index}] must contain {columns} entries"
            )
        out.append(parsed)
    return out


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgreementError(f"{where} must be a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise AgreementError(f"{where} must be a finite nonnegative number")
    return value


def _number_list(value: Any, where: str) -> list[float]:
    if not isinstance(value, list):
        raise AgreementError(f"{where} must be an array")
    return [_number(item, f"{where}[{index}]")
            for index, item in enumerate(value)]


def _number_matrix(
    value: Any, where: str, rows: int, columns: int
) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise AgreementError(f"{where} must contain {rows} rows")
    out = []
    for index, row in enumerate(value):
        parsed = _number_list(row, f"{where}[{index}]")
        if len(parsed) != columns:
            raise AgreementError(
                f"{where}[{index}] must contain {columns} entries"
            )
        out.append(parsed)
    return out


def load_jsonl(path: str, schema: str) -> list[dict[str, Any]]:
    """Load non-empty JSONL and require the registered object schema."""

    rows: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AgreementError(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(row, dict):
                    raise AgreementError(f"{path}:{line_number}: row must be an object")
                if row.get("schema") != schema:
                    raise AgreementError(
                        f"{path}:{line_number}: schema must be {schema!r}"
                    )
                rows.append(row)
    except OSError as exc:
        raise AgreementError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise AgreementError(f"{path}: no JSONL rows")
    return rows


def _optional_identity(row: dict[str, Any]) -> str | None:
    for key in ("case_id", "case", "id"):
        value = row.get(key)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise AgreementError(f"{key} must be a non-empty string")
            return value
    return None


def _shared_label(
    target: dict[str, Any], teacher: dict[str, Any], key: str, row_number: int
) -> str | None:
    left = target.get(key)
    right = teacher.get(key)
    for value in (left, right):
        if value is not None and (not isinstance(value, str) or not value):
            raise AgreementError(f"row {row_number}: {key} must be a non-empty string")
    if left is not None and right is not None and left != right:
        raise AgreementError(
            f"row {row_number}: target {key} {left!r} != teacher {key} {right!r}"
        )
    return right if right is not None else left


def _required_shared_label(
    target: dict[str, Any], teacher: dict[str, Any], key: str, row_number: int
) -> str:
    """Require the same non-empty metadata label on both paired records."""

    value = _shared_label(target, teacher, key, row_number)
    if target.get(key) is None or teacher.get(key) is None:
        raise AgreementError(
            f"row {row_number}: target and teacher {key} are required"
        )
    assert value is not None
    return value


def _contiguous_match_runs(bits: Iterable[int]) -> list[int]:
    runs: list[int] = []
    current = 0
    for bit in bits:
        if bit:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def _prefix_summary(
    branch_width_rows: list[list[int]],
    prefix_length_rows: list[list[int]],
    bitmaps: list[list[int]],
    width: int = MAX_WIDTH,
) -> dict[str, Any]:
    accepted_runs = [run for rows in prefix_length_rows for run in rows]
    widths = [value for rows in branch_width_rows for value in rows]
    if len(accepted_runs) != len(widths):
        raise AgreementError("branch width and prefix-length counts differ")

    prefix: dict[str, Any] = {}
    conditional: dict[str, Any] = {}
    for j in range(1, width + 1):
        eligible = sum(branch_width >= j for branch_width in widths)
        numerator = sum(
            branch_width >= j and run >= j
            for branch_width, run in zip(widths, accepted_runs)
        )
        prefix[f"Q{j}"] = {
            "survived": numerator,
            "eligible_roots": eligible,
            "rate": numerator / eligible,
        }
        denominator = sum(
            branch_width >= j and run >= j - 1
            for branch_width, run in zip(widths, accepted_runs)
        )
        conditional[f"C{j}"] = {
            "matched": numerator,
            "eligible_prefixes": denominator,
            "rate": None if denominator == 0 else numerator / denominator,
        }

    accepted_histogram = Counter(accepted_runs)
    mismatch_histogram = Counter(
        "none" if run == branch_width else str(run + 1)
        for branch_width, run in zip(widths, accepted_runs)
    )
    rejection_histogram = {
        **{str(position): mismatch_histogram.get(str(position), 0)
           for position in range(1, width + 1)},
        "no_rejection": mismatch_histogram.get("none", 0),
    }
    contiguous = [run for bits in bitmaps for run in _contiguous_match_runs(bits)]
    return {
        "maximum_width": width,
        "roots": len(accepted_runs),
        "prefix_survival": prefix,
        "conditional_survival": conditional,
        "full_block_acceptance": prefix[f"Q{width}"],
        "accepted_run_lengths": accepted_runs,
        "accepted_run_histogram": {
            str(run): accepted_histogram.get(run, 0) for run in range(width + 1)
        },
        "first_mismatch_histogram": {
            **{str(position): mismatch_histogram.get(str(position), 0)
               for position in range(1, width + 1)},
            "none": mismatch_histogram.get("none", 0),
        },
        "rejection_position_histogram": rejection_histogram,
        "contiguous_match_run_lengths": contiguous,
        "contiguous_match_run_histogram": {
            str(run): count for run, count in sorted(Counter(contiguous).items())
        },
    }


def _branch_cost_summary(
    widths: list[int],
    prefix_lengths: list[int],
    step_seconds: list[list[float]],
    hits: list[int],
    misses: list[int],
    byte_counts: list[int],
) -> dict[str, Any]:
    roots: list[dict[str, Any]] = []
    by_width: dict[str, dict[str, Any]] = {}
    for root, branch_width in enumerate(widths):
        steps = max(0, branch_width - 1)
        seconds = sum(step_seconds[root][:steps])
        row = {
            "root": root,
            "width": branch_width,
            "prefix_match_length": prefix_lengths[root],
            "branch_steps": steps,
            "seconds": seconds,
            "hits": hits[root],
            "misses": misses[root],
            "bytes": byte_counts[root],
        }
        roots.append(row)
        bucket = by_width.setdefault(
            str(branch_width),
            {"roots": 0, "branch_steps": 0, "seconds": 0.0,
             "hits": 0, "misses": 0, "bytes": 0},
        )
        bucket["roots"] += 1
        for key in ("branch_steps", "seconds", "hits", "misses", "bytes"):
            bucket[key] += row[key]

    totals = {
        "roots": len(roots),
        "branch_steps": sum(row["branch_steps"] for row in roots),
        "seconds": sum(row["seconds"] for row in roots),
        "hits": sum(row["hits"] for row in roots),
        "misses": sum(row["misses"] for row in roots),
        "bytes": sum(row["bytes"] for row in roots),
    }
    totals["seconds_per_branch_step"] = (
        None if not totals["branch_steps"]
        else totals["seconds"] / totals["branch_steps"]
    )
    return {"totals": totals, "by_width": by_width, "roots": roots}


def simulate_trajectory(
    tokens: list[int],
    branch_widths: list[int],
    prefix_match_lengths: list[int],
    branch_step_seconds: list[list[float]],
    width: int,
) -> dict[str, Any]:
    """Simulate fixed-width greedy correction/bonus semantics exactly."""

    if width <= 0:
        raise AgreementError("trajectory width must be positive")
    if not (
        len(tokens) == len(branch_widths)
        == len(prefix_match_lengths) == len(branch_step_seconds)
    ):
        raise AgreementError("trajectory branch arrays differ in length")

    position = 0
    blocks: list[dict[str, Any]] = []
    emitted: list[int] = []
    accepted = corrections = bonuses = 0
    draft_branch_steps = 0
    draft_branch_seconds = 0.0
    accepted_histogram: Counter[int] = Counter()
    mismatch_histogram: Counter[str] = Counter()

    while len(tokens) - position > 1:
        remaining = len(tokens) - position
        proposal_width = min(width, remaining - 1)
        if branch_widths[position] < proposal_width:
            raise AgreementError(
                f"root {position}: branch width {branch_widths[position]} "
                f"cannot simulate k={width} width {proposal_width}"
            )
        matched_prefix = min(prefix_match_lengths[position], proposal_width)
        first_mismatch = (
            None if matched_prefix == proposal_width else matched_prefix
        )

        if first_mismatch is None:
            block_accepted = proposal_width
            bonus_index = position + proposal_width
            committed = proposal_width + 1
            correction_index = None
            bonuses += 1
            mismatch_histogram["none"] += 1
        else:
            block_accepted = first_mismatch
            correction_index = position + first_mismatch
            bonus_index = None
            committed = first_mismatch + 1
            corrections += 1
            mismatch_histogram[str(first_mismatch + 1)] += 1

        accepted += block_accepted
        accepted_histogram[block_accepted] += 1
        branch_steps = max(0, proposal_width - 1)
        branch_seconds = sum(branch_step_seconds[position][:branch_steps])
        draft_branch_steps += branch_steps
        draft_branch_seconds += branch_seconds
        accepted_tokens = tokens[position : position + block_accepted]
        if correction_index is not None:
            terminal_token = tokens[correction_index]
        else:
            assert bonus_index is not None
            terminal_token = tokens[bonus_index]
        block_tokens = [*accepted_tokens, terminal_token]
        emitted.extend(block_tokens)
        blocks.append(
            {
                "root": position,
                "remaining": remaining,
                "proposals": proposal_width,
                "recorded_branch_width": branch_widths[position],
                "recorded_prefix_match_length": prefix_match_lengths[position],
                "draft_branch_steps": branch_steps,
                "draft_branch_seconds": branch_seconds,
                "accepted": block_accepted,
                "accepted_tokens": accepted_tokens,
                "rejected": proposal_width - block_accepted,
                "first_mismatch": (
                    None if first_mismatch is None else first_mismatch + 1
                ),
                "correction_index": correction_index,
                "correction_token": (
                    None if correction_index is None else tokens[correction_index]
                ),
                "bonus_index": bonus_index,
                "bonus_token": None if bonus_index is None else tokens[bonus_index],
                "committed": committed,
                "committed_tokens": block_tokens,
            }
        )
        position += committed

    direct_tail = len(tokens) - position
    direct_tail_tokens = tokens[position:]
    if direct_tail not in (0, 1):
        raise AgreementError(
            f"k={width} simulation left an invalid {direct_tail}-token direct tail"
        )
    emitted.extend(direct_tail_tokens)
    proposals = sum(block["proposals"] for block in blocks)
    rejected = proposals - accepted
    block_committed = accepted + corrections + bonuses
    committed = block_committed + direct_tail
    accounting = {
        "committed_eq_A_plus_M_plus_U_plus_D": (
            committed == accepted + corrections + bonuses + direct_tail
        ),
        "Q_eq_M_plus_U": len(blocks) == corrections + bonuses,
        "P_eq_A_plus_rejected": proposals == accepted + rejected,
        "committed_equals_target_tokens": committed == len(tokens),
    }
    return {
        "k": width,
        "target_tokens": len(tokens),
        "verifier_blocks": len(blocks),
        "proposals": proposals,
        "accepted": accepted,
        "rejected": rejected,
        "corrections": corrections,
        "bonuses": bonuses,
        "direct_tail": direct_tail,
        "direct_tail_tokens": direct_tail_tokens,
        "draft_branch_steps": draft_branch_steps,
        "draft_branch_seconds": draft_branch_seconds,
        "block_committed": block_committed,
        "committed": committed,
        "emitted_tokens": emitted,
        "emitted_equals_target": emitted == tokens,
        "mean_committed_per_block": (
            None if not blocks else block_committed / len(blocks)
        ),
        "accepted_histogram": {
            str(run): accepted_histogram.get(run, 0) for run in range(width + 1)
        },
        "first_mismatch_histogram": {
            **{str(position): mismatch_histogram.get(str(position), 0)
               for position in range(1, width + 1)},
            "none": mismatch_histogram.get("none", 0),
        },
        "rejection_position_histogram": {
            **{str(position): mismatch_histogram.get(str(position), 0)
               for position in range(1, width + 1)},
            "no_rejection": mismatch_histogram.get("none", 0),
        },
        "accounting": accounting,
        "blocks": blocks,
    }


def _merge_trajectories(trajectories: list[dict[str, Any]], width: int) -> dict[str, Any]:
    scalar_keys = (
        "target_tokens",
        "verifier_blocks",
        "proposals",
        "accepted",
        "rejected",
        "corrections",
        "bonuses",
        "direct_tail",
        "block_committed",
        "committed",
        "draft_branch_steps",
        "draft_branch_seconds",
    )
    merged = {key: sum(row[key] for row in trajectories) for key in scalar_keys}
    accepted_histogram = Counter()
    mismatch_histogram = Counter()
    for row in trajectories:
        accepted_histogram.update(row["accepted_histogram"])
        mismatch_histogram.update(row["first_mismatch_histogram"])
    blocks = merged["verifier_blocks"]
    merged.update(
        {
            "k": width,
            "mean_committed_per_block": (
                None if not blocks else merged["block_committed"] / blocks
            ),
            "accepted_histogram": {
                str(run): accepted_histogram.get(str(run), 0)
                for run in range(width + 1)
            },
            "first_mismatch_histogram": {
                **{str(position): mismatch_histogram.get(str(position), 0)
                   for position in range(1, width + 1)},
                "none": mismatch_histogram.get("none", 0),
            },
            "rejection_position_histogram": {
                **{str(position): mismatch_histogram.get(str(position), 0)
                   for position in range(1, width + 1)},
                "no_rejection": mismatch_histogram.get("none", 0),
            },
            "accounting": {
                "committed_eq_A_plus_M_plus_U_plus_D": (
                    merged["committed"]
                    == merged["accepted"] + merged["corrections"]
                    + merged["bonuses"] + merged["direct_tail"]
                ),
                "Q_eq_M_plus_U": (
                    merged["verifier_blocks"]
                    == merged["corrections"] + merged["bonuses"]
                ),
                "P_eq_A_plus_rejected": (
                    merged["proposals"]
                    == merged["accepted"] + merged["rejected"]
                ),
                "committed_equals_target_tokens": (
                    merged["committed"] == merged["target_tokens"]
                ),
            },
        }
    )
    return merged


def _merge_branch_costs(costs: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "roots": 0,
        "branch_steps": 0,
        "seconds": 0.0,
        "hits": 0,
        "misses": 0,
        "bytes": 0,
    }
    by_width: dict[str, dict[str, Any]] = {}
    for cost in costs:
        for key in totals:
            totals[key] += cost["totals"][key]
        for width, row in cost["by_width"].items():
            bucket = by_width.setdefault(
                width,
                {"roots": 0, "branch_steps": 0, "seconds": 0.0,
                 "hits": 0, "misses": 0, "bytes": 0},
            )
            for key in bucket:
                bucket[key] += row[key]
    totals["seconds_per_branch_step"] = (
        None if not totals["branch_steps"]
        else totals["seconds"] / totals["branch_steps"]
    )
    return {"totals": totals, "by_width": by_width}


def _pair_rows(
    target: dict[str, Any], teacher: dict[str, Any], row_number: int
) -> dict[str, Any]:
    tokens = _integer_list(target.get("tokens"), f"target row {row_number}.tokens")
    generated = _integer(
        target.get("generated"), f"target row {row_number}.generated", minimum=1
    )
    if generated != len(tokens):
        raise AgreementError(
            f"target row {row_number}: generated {generated} != {len(tokens)} tokens"
        )

    targets = _integer_list(
        teacher.get("targets"), f"teacher row {row_number}.targets"
    )
    predictions = _integer_list(
        teacher.get("predictions"), f"teacher row {row_number}.predictions"
    )
    target_tokens = _integer(
        teacher.get("target_tokens"),
        f"teacher row {row_number}.target_tokens",
        minimum=1,
    )
    if targets != tokens:
        raise AgreementError(f"row {row_number}: teacher targets differ from target tokens")
    if target_tokens != len(tokens) or len(predictions) != len(tokens):
        raise AgreementError(f"row {row_number}: teacher array/count lengths differ")

    computed = [int(prediction == expected) for prediction, expected in zip(predictions, tokens)]
    recorded = _integer_list(
        teacher.get("matches"), f"teacher row {row_number}.matches", binary=True
    )
    if recorded != computed:
        raise AgreementError(f"row {row_number}: recorded matches disagree with token IDs")
    if len(computed) < MAX_WIDTH:
        raise AgreementError(
            f"row {row_number}: {len(computed)} tokens cannot measure Q1..Q{MAX_WIDTH}"
        )

    branch_width = _integer(
        teacher.get("branch_width"), f"teacher row {row_number}.branch_width",
        minimum=1,
    )
    if branch_width != MAX_WIDTH:
        raise AgreementError(
            f"row {row_number}: branch_width {branch_width} != {MAX_WIDTH}"
        )
    branch_widths = _integer_list(
        teacher.get("branch_widths"),
        f"teacher row {row_number}.branch_widths",
    )
    prefix_match_lengths = _integer_list(
        teacher.get("prefix_match_lengths"),
        f"teacher row {row_number}.prefix_match_lengths",
    )
    for name, values in (
        ("branch_widths", branch_widths),
        ("prefix_match_lengths", prefix_match_lengths),
    ):
        if len(values) != len(tokens):
            raise AgreementError(
                f"row {row_number}: {name} has {len(values)} entries, "
                f"expected {len(tokens)}"
            )
    branch_predictions = _integer_matrix(
        teacher.get("branch_predictions"),
        f"teacher row {row_number}.branch_predictions",
        len(tokens), MAX_WIDTH, minimum=-1,
    )
    branch_step_seconds = _number_matrix(
        teacher.get("branch_step_seconds"),
        f"teacher row {row_number}.branch_step_seconds",
        len(tokens), MAX_WIDTH,
    )
    branch_hits = _integer_list(
        teacher.get("branch_hits"), f"teacher row {row_number}.branch_hits"
    )
    branch_misses = _integer_list(
        teacher.get("branch_misses"), f"teacher row {row_number}.branch_misses"
    )
    branch_bytes = _integer_list(
        teacher.get("branch_bytes"), f"teacher row {row_number}.branch_bytes"
    )
    for name, values in (
        ("branch_hits", branch_hits),
        ("branch_misses", branch_misses),
        ("branch_bytes", branch_bytes),
    ):
        if len(values) != len(tokens):
            raise AgreementError(
                f"row {row_number}: {name} has {len(values)} entries, "
                f"expected {len(tokens)}"
            )

    for root in range(len(tokens)):
        expected_width = min(MAX_WIDTH, len(tokens) - root)
        width_at_root = branch_widths[root]
        if width_at_root != expected_width:
            raise AgreementError(
                f"row {row_number} root {root}: branch width {width_at_root} "
                f"!= expected {expected_width}"
            )
        predictions_at_root = branch_predictions[root]
        if predictions_at_root[0] != predictions[root]:
            raise AgreementError(
                f"row {row_number} root {root}: branch root prediction "
                "differs from teacher prediction"
            )
        if any(value < 0 for value in predictions_at_root[:width_at_root]):
            raise AgreementError(
                f"row {row_number} root {root}: active branch prediction is negative"
            )
        if any(value != -1 for value in predictions_at_root[width_at_root:]):
            raise AgreementError(
                f"row {row_number} root {root}: unused branch predictions must be -1"
            )
        computed_prefix = 0
        while (
            computed_prefix < width_at_root
            and predictions_at_root[computed_prefix]
            == tokens[root + computed_prefix]
        ):
            computed_prefix += 1
        if prefix_match_lengths[root] != computed_prefix:
            raise AgreementError(
                f"row {row_number} root {root}: prefix_match_length "
                f"{prefix_match_lengths[root]} != computed {computed_prefix}"
            )
        # Producing w proposals consumes w-1 model steps because the root
        # logits already provide the first proposal.  The probe zero-pads the
        # rest of each fixed-width timing row.
        if any(branch_step_seconds[root][width_at_root - 1:]):
            raise AgreementError(
                f"row {row_number} root {root}: unused branch timings must be zero"
            )

    target_id = _optional_identity(target)
    teacher_id = _optional_identity(teacher)
    if target_id is not None and teacher_id is not None and target_id != teacher_id:
        raise AgreementError(
            f"row {row_number}: target case {target_id!r} != teacher case {teacher_id!r}"
        )
    case_id = teacher_id or target_id or f"row_{row_number}"
    family = _required_shared_label(target, teacher, "family", row_number)
    tier = _shared_label(target, teacher, "tier", row_number)
    prompt_format = _shared_label(target, teacher, "format", row_number)

    row: dict[str, Any] = {
        "case_id": case_id,
        "family": family,
        "tokens": len(tokens),
        "target_tokens": tokens,
        "draft_predictions": predictions,
        "match_bitmap": computed,
        "branch_widths": branch_widths,
        "branch_predictions": branch_predictions,
        "prefix_match_lengths": prefix_match_lengths,
        "matched": sum(computed),
        "marginal_agreement": sum(computed) / len(computed),
        "agreement": _prefix_summary(
            [branch_widths], [prefix_match_lengths], [computed]
        ),
        "branch_costs": _branch_cost_summary(
            branch_widths, prefix_match_lengths, branch_step_seconds,
            branch_hits, branch_misses, branch_bytes,
        ),
        "trajectories": {
            str(width): simulate_trajectory(
                tokens, branch_widths, prefix_match_lengths,
                branch_step_seconds, width,
            )
            for width in WIDTHS
        },
    }
    if tier is not None:
        row["tier"] = tier
    if prompt_format is not None:
        row["format"] = prompt_format
    return row


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge complete case rows without constructing cross-case branches."""

    if not rows:
        raise AgreementError("cannot summarize an empty row group")
    total_tokens = sum(row["tokens"] for row in rows)
    total_matched = sum(row["matched"] for row in rows)
    bitmaps = [row["match_bitmap"] for row in rows]
    branch_width_rows = [row["branch_widths"] for row in rows]
    prefix_length_rows = [row["prefix_match_lengths"] for row in rows]
    return {
        "tokens": total_tokens,
        "matched": total_matched,
        "marginal_agreement": total_matched / total_tokens,
        "match_bitmaps": [
            {
                "case_id": row["case_id"],
                "family": row["family"],
                **({"tier": row["tier"]} if "tier" in row else {}),
                **({"format": row["format"]} if "format" in row else {}),
                "bitmap": row["match_bitmap"],
            }
            for row in rows
        ],
        "agreement": _prefix_summary(
            branch_width_rows, prefix_length_rows, bitmaps
        ),
        "branch_costs": _merge_branch_costs(
            [row["branch_costs"] for row in rows]
        ),
        "trajectories": {
            str(width): _merge_trajectories(
                [row["trajectories"][str(width)] for row in rows], width
            )
            for width in WIDTHS
        },
    }


def _group_summaries(
    rows: list[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    """Return deterministic aggregate summaries for every present label."""

    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get(label)
        if value is not None:
            buckets.setdefault(value, []).append(row)
    return {
        value: {
            "row_count": len(group),
            "case_ids": [row["case_id"] for row in group],
            **_summarize_rows(group),
        }
        for value, group in sorted(buckets.items())
    }


def analyze(
    targets: list[dict[str, Any]], teachers: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate paired rows and return exact per-row and aggregate metrics."""

    if len(targets) != len(teachers):
        raise AgreementError(
            f"target/teacher row count differs: {len(targets)} != {len(teachers)}"
        )
    if not targets:
        raise AgreementError("target/teacher inputs are empty")

    rows = [_pair_rows(target, teacher, index)
            for index, (target, teacher) in enumerate(zip(targets, teachers), 1)]
    return {
        "schema": OUTPUT_SCHEMA,
        "max_width": MAX_WIDTH,
        "trajectory_widths": list(WIDTHS),
        "row_count": len(rows),
        "rows": rows,
        "families": _group_summaries(rows, "family"),
        "tiers": _group_summaries(rows, "tier"),
        "formats": _group_summaries(rows, "format"),
        "aggregate": _summarize_rows(rows),
    }


def print_human(summary: dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    print(
        f"{summary['row_count']} rows, {aggregate['tokens']} tokens, "
        f"marginal agreement {aggregate['marginal_agreement']:.3%}"
    )
    prefix = aggregate["agreement"]["prefix_survival"]
    print("prefix survival:")
    print("  " + "  ".join(
        f"Q{j}={prefix[f'Q{j}']['rate']:.3%}" for j in range(1, MAX_WIDTH + 1)
    ))
    branch = aggregate["branch_costs"]["totals"]
    print(
        f"recorded branches: roots={branch['roots']} "
        f"steps={branch['branch_steps']} seconds={branch['seconds']:.6f} "
        f"hits={branch['hits']} misses={branch['misses']} bytes={branch['bytes']}"
    )
    for width in WIDTHS:
        trajectory = aggregate["trajectories"][str(width)]
        print(
            f"k={width}: blocks={trajectory['verifier_blocks']} "
            f"accepted={trajectory['accepted']} "
            f"corrections={trajectory['corrections']} "
            f"bonuses={trajectory['bonuses']} "
            f"direct_tail={trajectory['direct_tail']} "
            f"committed/block={trajectory['mean_committed_per_block']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help=f"JSONL rows with schema {TARGET_SCHEMA}")
    parser.add_argument("teacher", help=f"JSONL rows with schema {TEACHER_SCHEMA}")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    try:
        summary = analyze(
            load_jsonl(args.target, TARGET_SCHEMA),
            load_jsonl(args.teacher, TEACHER_SCHEMA),
        )
    except AgreementError as exc:
        print(f"analyze_spec_agreement: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
