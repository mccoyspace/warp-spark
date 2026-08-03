#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prompt_lookup", ROOT / "tools/analyze_prompt_lookup.py"
)
assert SPEC and SPEC.loader
lookup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lookup)


class PromptLookupTest(unittest.TestCase):
    def test_longest_suffix_then_most_recent_occurrence(self) -> None:
        context = [1, 2, 3, 4, 5, 9, 2, 3, 4, 6, 2, 3, 4]
        ngram, occurrence, proposals = lookup.lookup(context)
        self.assertEqual((ngram, occurrence), (3, 6))
        self.assertEqual(proposals, [6, 2, 3, 4])
        self.assertEqual(lookup.lookup([1, 2, 3]), (0, None, []))

    def test_missing_proposals_are_prefix_non_survival(self) -> None:
        roots = [
            {"remaining": 3, "proposals_available": 0, "prefix_match_length": 0,
             "first_mismatch": None, "rejection_positions": []},
            {"remaining": 2, "proposals_available": 2, "prefix_match_length": 2,
             "first_mismatch": None, "rejection_positions": []},
            {"remaining": 1, "proposals_available": 1, "prefix_match_length": 0,
             "first_mismatch": 1, "rejection_positions": [1]},
        ]
        curve = lookup.curve(roots)["prefix_survival"]
        self.assertEqual(curve["Q1"], {
            "survived": 1, "eligible_roots": 3, "missing_proposals": 1,
            "rate": 1 / 3,
        })
        self.assertEqual(curve["Q2"]["survived"], 1)
        self.assertEqual(curve["Q2"]["eligible_roots"], 2)
        self.assertEqual(curve["Q3"]["rate"], 0)

    def test_greedy_trajectory_correction_bonus_and_direct(self) -> None:
        targets = [1, 2, 3, 4, 5]
        roots = [
            {"proposals_available": 2, "proposals": [1, 2]},
            {"proposals_available": 0, "proposals": []},
            {"proposals_available": 0, "proposals": []},
            {"proposals_available": 1, "proposals": [99]},
            {"proposals_available": 0, "proposals": []},
        ]
        result = lookup.simulate(targets, roots, 2)
        self.assertEqual(
            (result["proposals"], result["accepted"], result["rejected"]),
            (3, 2, 1),
        )
        self.assertEqual(
            (result["bonuses"], result["corrections"], result["direct"]),
            (1, 1, 1),
        )
        self.assertTrue(all(result["accounting"].values()))
        self.assertEqual(result["rejection_position_histogram"], {"1": 1, "2": 0})

    def test_h2_is_explicitly_rejected(self) -> None:
        with self.assertRaisesRegex(lookup.LookupError, "H2 inference embargo"):
            lookup.load_corpus(ROOT / "docs/gn100/sprint15-h2-corpus.json")

    def test_family_and_tier_groups_include_exact_trajectories(self) -> None:
        cases = [
            {"id": "a", "family": "shared", "split": "calibration_train",
             "token_ids": [1, 2]},
            {"id": "b", "family": "shared", "split": "within_family_holdout",
             "token_ids": [3, 4]},
        ]
        targets = [{"tokens": list(range(10, 42))}, {"tokens": list(range(50, 82))}]
        result = lookup.analyze(cases, targets)
        family = result["families"]["shared"]
        self.assertEqual(family["case_ids"], ["a", "b"])
        self.assertEqual(set(family["trajectories"]), {"2", "4", "8"})
        for trajectory in family["trajectories"].values():
            self.assertEqual(trajectory["target_tokens"], 64)
            self.assertTrue(all(trajectory["accounting"].values()))
        self.assertEqual(result["tiers"]["A"]["case_ids"], ["a"])
        self.assertEqual(result["tiers"]["B"]["case_ids"], ["b"])


if __name__ == "__main__":
    unittest.main()
