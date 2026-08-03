#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Focused tests for exact speculative-agreement analysis."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import analyze_spec_agreement as AGREEMENT  # noqa: E402


def paired_rows(
    bits, *, case_id="case_a", family="studio", tier="A",
    prompt_format="target_ids",
):
    tokens = list(range(100, 100 + len(bits)))
    predictions = [token if bit else token + 1000 for token, bit in zip(tokens, bits)]
    branch_widths = []
    branch_predictions = []
    prefix_match_lengths = []
    branch_step_seconds = []
    branch_hits = []
    branch_misses = []
    branch_bytes = []
    for root in range(len(tokens)):
        width = min(AGREEMENT.MAX_WIDTH, len(tokens) - root)
        branch_widths.append(width)
        prefix = 0
        row = []
        mismatched = False
        for offset in range(width):
            bit = bits[root + offset]
            if not mismatched and bit:
                row.append(tokens[root + offset])
                prefix += 1
            else:
                row.append(tokens[root + offset] + 1000)
                mismatched = True
        row.extend([-1] * (AGREEMENT.MAX_WIDTH - width))
        branch_predictions.append(row)
        prefix_match_lengths.append(prefix)
        branch_steps = width - 1
        branch_step_seconds.append(
            [0.01 * (offset + 1) for offset in range(branch_steps)]
            + [0.0] * (AGREEMENT.MAX_WIDTH - branch_steps)
        )
        branch_hits.append(2 * branch_steps)
        branch_misses.append(branch_steps)
        branch_bytes.append(100 * branch_steps)
    target = {
        "schema": AGREEMENT.TARGET_SCHEMA,
        "case_id": case_id,
        "family": family,
        "tier": tier,
        "format": prompt_format,
        "generated": len(tokens),
        "tokens": tokens,
    }
    teacher = {
        "schema": AGREEMENT.TEACHER_SCHEMA,
        "case_id": case_id,
        "family": family,
        "tier": tier,
        "format": prompt_format,
        "target_tokens": len(tokens),
        "targets": list(tokens),
        "predictions": predictions,
        "matches": list(bits),
        "branch_width": AGREEMENT.MAX_WIDTH,
        "branch_widths": branch_widths,
        "branch_predictions": branch_predictions,
        "prefix_match_lengths": prefix_match_lengths,
        "branch_step_seconds": branch_step_seconds,
        "branch_hits": branch_hits,
        "branch_misses": branch_misses,
        "branch_bytes": branch_bytes,
    }
    return target, teacher


class AnalyzeSpecAgreementTest(unittest.TestCase):
    def test_prefix_survival_uses_actual_branches_and_per_j_eligible_roots(self):
        bits = [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
        target, teacher = paired_rows(bits)
        summary = AGREEMENT.analyze([target], [teacher])
        row = summary["rows"][0]
        agreement = row["agreement"]

        self.assertEqual(row["match_bitmap"], bits)
        self.assertEqual(row["matched"], 10)
        self.assertAlmostEqual(row["marginal_agreement"], 10 / 12)
        self.assertEqual(agreement["roots"], 12)
        self.assertEqual(
            agreement["accepted_run_lengths"],
            [2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1],
        )

        expected_survival = [10, 7, 4, 2, 0, 0, 0, 0]
        expected_eligible = [12, 11, 10, 9, 8, 7, 6, 5]
        for j, (survived, eligible) in enumerate(
            zip(expected_survival, expected_eligible), 1
        ):
            q = agreement["prefix_survival"][f"Q{j}"]
            self.assertEqual(q["eligible_roots"], eligible)
            self.assertEqual(q["survived"], survived)
            self.assertAlmostEqual(q["rate"], survived / eligible)

        expected_conditionals = [10 / 12, 7 / 9, 4 / 6, 2 / 3, 0.0]
        for j, rate in enumerate(expected_conditionals, 1):
            self.assertAlmostEqual(
                agreement["conditional_survival"][f"C{j}"]["rate"], rate
            )
        for j in range(6, 9):
            self.assertIsNone(
                agreement["conditional_survival"][f"C{j}"]["rate"]
            )

        self.assertEqual(
            agreement["accepted_run_histogram"],
            {"0": 2, "1": 3, "2": 3, "3": 2, "4": 2,
             "5": 0, "6": 0, "7": 0, "8": 0},
        )
        self.assertEqual(
            agreement["rejection_position_histogram"],
            {"1": 2, "2": 2, "3": 2, "4": 1, "5": 1,
             "6": 0, "7": 0, "8": 0, "no_rejection": 4},
        )
        self.assertEqual(agreement["contiguous_match_run_lengths"], [2, 4, 4])
        # Q2 is the observed correlated prefix frequency, not p**2.
        self.assertNotAlmostEqual(
            agreement["prefix_survival"]["Q2"]["rate"],
            row["marginal_agreement"] ** 2,
        )

    def test_exact_correction_and_bonus_trajectories(self):
        bits = [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
        target, teacher = paired_rows(bits)
        row = AGREEMENT.analyze([target], [teacher])["rows"][0]

        k2 = row["trajectories"]["2"]
        self.assertEqual(
            (k2["verifier_blocks"], k2["proposals"], k2["accepted"],
             k2["rejected"], k2["corrections"], k2["bonuses"],
             k2["direct_tail"], k2["committed"]),
            (4, 8, 7, 1, 1, 3, 1, 12),
        )
        self.assertEqual([block["root"] for block in k2["blocks"]], [0, 3, 6, 8])
        self.assertEqual(
            [block["first_mismatch"] for block in k2["blocks"]],
            [None, None, 2, None],
        )

        k4 = row["trajectories"]["4"]
        self.assertEqual(
            (k4["verifier_blocks"], k4["proposals"], k4["accepted"],
             k4["rejected"], k4["corrections"], k4["bonuses"],
             k4["direct_tail"], k4["committed"]),
            (3, 11, 9, 2, 1, 2, 0, 12),
        )
        self.assertEqual([block["root"] for block in k4["blocks"]], [0, 3, 8])
        self.assertEqual([block["proposals"] for block in k4["blocks"]], [4, 4, 3])

        k8 = row["trajectories"]["8"]
        self.assertEqual(
            (k8["verifier_blocks"], k8["proposals"], k8["accepted"],
             k8["rejected"], k8["corrections"], k8["bonuses"],
             k8["direct_tail"], k8["committed"]),
            (3, 19, 9, 10, 2, 1, 0, 12),
        )

        self.assertEqual(k2["draft_branch_steps"], 4)
        self.assertAlmostEqual(k2["draft_branch_seconds"], 0.04)
        self.assertEqual(k4["draft_branch_steps"], 8)
        self.assertAlmostEqual(k4["draft_branch_seconds"], 0.15)
        self.assertEqual(k8["draft_branch_steps"], 16)
        self.assertAlmostEqual(k8["draft_branch_seconds"], 0.59)

        for trajectory in (k2, k4, k8):
            self.assertEqual(trajectory["emitted_tokens"], target["tokens"])
            self.assertTrue(trajectory["emitted_equals_target"])
            self.assertTrue(all(trajectory["accounting"].values()))

    def test_all_mismatches_leave_undefined_later_conditionals(self):
        target, teacher = paired_rows([0] * 10)
        row = AGREEMENT.analyze([target], [teacher])["rows"][0]
        agreement = row["agreement"]
        self.assertEqual(agreement["prefix_survival"]["Q1"]["rate"], 0.0)
        for j in range(2, 9):
            self.assertIsNone(
                agreement["conditional_survival"][f"C{j}"]["rate"]
            )
        k2 = row["trajectories"]["2"]
        self.assertEqual(k2["verifier_blocks"], 9)
        self.assertEqual(k2["proposals"], 17)
        self.assertEqual(k2["corrections"], 9)
        self.assertEqual(k2["accepted"], 0)
        self.assertEqual(k2["direct_tail"], 1)
        self.assertEqual(k2["committed"], 10)

    def test_branch_timing_and_io_costs_are_summarized_without_gain_claims(self):
        bits = [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
        target, teacher = paired_rows(bits)
        summary = AGREEMENT.analyze([target], [teacher])
        costs = summary["rows"][0]["branch_costs"]
        totals = costs["totals"]
        self.assertEqual(totals["roots"], 12)
        self.assertEqual(totals["branch_steps"], 56)
        self.assertAlmostEqual(totals["seconds"], 1.96)
        self.assertEqual(totals["hits"], 112)
        self.assertEqual(totals["misses"], 56)
        self.assertEqual(totals["bytes"], 5600)
        self.assertAlmostEqual(totals["seconds_per_branch_step"], 0.035)
        self.assertEqual(costs["by_width"]["8"]["roots"], 5)
        self.assertEqual(costs["by_width"]["8"]["branch_steps"], 35)
        self.assertEqual(summary["aggregate"]["branch_costs"]["totals"], totals)
        self.assertNotIn("gain", json.dumps(costs).lower())

    def test_aggregate_never_builds_windows_across_prompt_rows(self):
        target_a, teacher_a = paired_rows([0, 0, 0, 0, 1, 1, 1, 1], case_id="a")
        target_b, teacher_b = paired_rows([1, 1, 1, 1, 0, 0, 0, 0], case_id="b")
        aggregate = AGREEMENT.analyze(
            [target_a, target_b], [teacher_a, teacher_b]
        )["aggregate"]
        # Q8 has one eligible root in each row.  Concatenating the rows would
        # invent an eight-token success across their boundary.
        self.assertEqual(
            aggregate["agreement"]["prefix_survival"]["Q8"],
            {"survived": 0, "eligible_roots": 2, "rate": 0.0},
        )
        self.assertEqual(aggregate["agreement"]["roots"], 16)
        for width in AGREEMENT.WIDTHS:
            self.assertTrue(
                all(aggregate["trajectories"][str(width)]["accounting"].values())
            )

    def test_grouped_summaries_preserve_row_boundaries_and_metadata(self):
        target_a, teacher_a = paired_rows(
            [1] * 8, case_id="a", family="studio", tier="A",
            prompt_format="target_ids",
        )
        target_b, teacher_b = paired_rows(
            [0] * 8, case_id="b", family="studio", tier="B",
            prompt_format="kimi_native",
        )
        target_c, teacher_c = paired_rows(
            [1, 0] * 4, case_id="c", family="code", tier="A",
            prompt_format="target_ids",
        )
        summary = AGREEMENT.analyze(
            [target_a, target_b, target_c],
            [teacher_a, teacher_b, teacher_c],
        )

        self.assertEqual(summary["rows"][0]["family"], "studio")
        self.assertEqual(list(summary["families"]), ["code", "studio"])
        self.assertEqual(list(summary["tiers"]), ["A", "B"])
        self.assertEqual(list(summary["formats"]), ["kimi_native", "target_ids"])

        studio = summary["families"]["studio"]
        self.assertEqual(studio["row_count"], 2)
        self.assertEqual(studio["case_ids"], ["a", "b"])
        self.assertEqual(
            studio["agreement"]["first_mismatch_histogram"],
            {"1": 8, "2": 0, "3": 0, "4": 0, "5": 0,
             "6": 0, "7": 0, "8": 0, "none": 8},
        )
        # There is one Q8-eligible root per case.  The grouping must merge the
        # two case summaries, not concatenate their token streams.
        self.assertEqual(
            studio["agreement"]["prefix_survival"]["Q8"],
            {"survived": 1, "eligible_roots": 2, "rate": 0.5},
        )
        self.assertEqual(
            studio["trajectories"]["4"]["first_mismatch_histogram"],
            {"1": 7, "2": 0, "3": 0, "4": 0, "none": 2},
        )
        for dimension, label, expected_ids in (
            ("tiers", "A", ["a", "c"]),
            ("formats", "target_ids", ["a", "c"]),
        ):
            group = summary[dimension][label]
            self.assertEqual(group["case_ids"], expected_ids)
            self.assertIn("prefix_survival", group["agreement"])
            self.assertIn("first_mismatch_histogram", group["agreement"])
            self.assertEqual(set(group["trajectories"]), {"2", "4", "8"})
            for trajectory in group["trajectories"].values():
                self.assertTrue(all(trajectory["accounting"].values()))

    def test_family_is_required_on_both_rows_and_must_match(self):
        target, teacher = paired_rows([1] * 8)
        del target["family"]
        with self.assertRaisesRegex(
            AGREEMENT.AgreementError, "target and teacher family are required"
        ):
            AGREEMENT.analyze([target], [teacher])

        target, teacher = paired_rows([1] * 8)
        teacher["family"] = "different"
        with self.assertRaisesRegex(AGREEMENT.AgreementError, "target family"):
            AGREEMENT.analyze([target], [teacher])

    def test_jsonl_loader_and_json_cli(self):
        target, teacher = paired_rows([1, 0, 1, 1, 0, 1, 1, 1])
        with tempfile.TemporaryDirectory() as directory:
            target_path = os.path.join(directory, "target.jsonl")
            teacher_path = os.path.join(directory, "teacher.jsonl")
            with open(target_path, "w", encoding="utf-8") as stream:
                stream.write("\n" + json.dumps(target) + "\n")
            with open(teacher_path, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(teacher) + "\n\n")

            self.assertEqual(
                AGREEMENT.load_jsonl(target_path, AGREEMENT.TARGET_SCHEMA),
                [target],
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = AGREEMENT.main([target_path, teacher_path, "--json"])
            self.assertEqual(rc, 0)
            decoded = json.loads(output.getvalue())
            self.assertEqual(decoded["schema"], AGREEMENT.OUTPUT_SCHEMA)
            self.assertEqual(decoded["rows"][0]["case_id"], "case_a")
            self.assertEqual(decoded["rows"][0]["family"], "studio")
            self.assertEqual(decoded["families"]["studio"]["case_ids"], ["case_a"])

    def test_inconsistent_recorded_bitmap_is_rejected(self):
        target, teacher = paired_rows([1] * 8)
        teacher["matches"][3] = 0
        with self.assertRaisesRegex(
            AGREEMENT.AgreementError, "recorded matches disagree"
        ):
            AGREEMENT.analyze([target], [teacher])

    def test_branch_predictions_are_checked_against_canonical_targets(self):
        target, teacher = paired_rows([1] * 8)
        teacher["branch_predictions"][0][1] += 1000
        with self.assertRaisesRegex(
            AGREEMENT.AgreementError, "prefix_match_length"
        ):
            AGREEMENT.analyze([target], [teacher])

        target, teacher = paired_rows([1] * 8)
        teacher["branch_widths"][-1] = 2
        with self.assertRaisesRegex(AGREEMENT.AgreementError, "branch width"):
            AGREEMENT.analyze([target], [teacher])

    def test_row_counts_and_target_tokens_must_match(self):
        target, teacher = paired_rows([1] * 8)
        with self.assertRaisesRegex(AGREEMENT.AgreementError, "row count differs"):
            AGREEMENT.analyze([target], [])
        teacher["targets"][0] += 1
        with self.assertRaisesRegex(AGREEMENT.AgreementError, "targets differ"):
            AGREEMENT.analyze([target], [teacher])


if __name__ == "__main__":
    unittest.main()
