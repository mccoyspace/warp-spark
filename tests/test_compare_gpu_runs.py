#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Focused tests for the dependency-free CPU/GPU capture comparator."""

import copy
import json
import os
import struct
import sys
import tempfile
import unittest


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import compare_gpu_runs as COMPARE  # noqa: E402


VOCAB = 12
TOP_K = 2


def base_logits():
    return [
        [float(i) for i in range(VOCAB)],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 20.0, 11.0],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 30.0, 10.0, 11.0],
    ]


def base_steps():
    # Argmax sequence is 11, 10, 9, so inputs after the first establish that
    # these are greedy captures rather than teacher-forced rows.
    return [
        {
            "index": 0,
            "position": 5,
            "input_token": 5,
            "routes": [
                {"layer": 4, "experts": [1, 2]},
                {"layer": 5, "experts": [3, 4]},
            ],
        },
        {
            "index": 1,
            "position": 6,
            "input_token": 11,
            "routes": [
                {"layer": 4, "experts": [2, 5]},
                {"layer": 5, "experts": [3, 4]},
            ],
        },
        {
            "index": 2,
            "position": 7,
            "input_token": 10,
            "routes": [
                {"layer": 4, "experts": [6, 7]},
                {"layer": 5, "experts": [8, 9]},
            ],
        },
    ]


def write_capture(directory, name, logits, steps, *, greedy=True, arm=None):
    raw_name = name + ".logits.f32"
    raw_path = os.path.join(directory, raw_name)
    with open(raw_path, "wb") as stream:
        for row in logits:
            if len(row) != VOCAB:
                raise AssertionError("bad fixture vocabulary")
            stream.write(struct.pack(f"<{VOCAB}f", *row))
    manifest = {
        "schema": COMPARE.SCHEMA,
        "dtype": COMPARE.DTYPE,
        "logits_file": raw_name,
        "vocab": VOCAB,
        "top_k": TOP_K,
        "greedy": greedy,
        "steps": steps,
    }
    if arm is not None:
        manifest["arm"] = arm
    manifest_path = os.path.join(directory, name + ".json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream)
    return manifest_path


class CompareGpuRunsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def compare(
        self,
        cpu_logits=None,
        gpu_logits=None,
        cpu_steps=None,
        gpu_steps=None,
        cpu_arm=None,
        gpu_arm=None,
    ):
        cpu_logits = copy.deepcopy(cpu_logits if cpu_logits is not None else base_logits())
        gpu_logits = copy.deepcopy(gpu_logits if gpu_logits is not None else base_logits())
        cpu_steps = copy.deepcopy(cpu_steps if cpu_steps is not None else base_steps())
        gpu_steps = copy.deepcopy(gpu_steps if gpu_steps is not None else base_steps())
        cpu = write_capture(
            self.temp.name, "cpu", cpu_logits, cpu_steps, arm=cpu_arm
        )
        gpu = write_capture(
            self.temp.name, "gpu", gpu_logits, gpu_steps, arm=gpu_arm
        )
        return COMPARE.compare_captures(cpu, gpu)

    @staticmethod
    def arms():
        cpu = {
            "key": "cuda", "value": 0, "effective": 0,
            "fallbacks": 0, "calls": 0, "expected_calls": 0,
        }
        gpu = {
            "key": "cuda", "value": 1, "effective": 1,
            "fallbacks": 0, "calls": 24, "expected_calls": 24,
        }
        return cpu, gpu

    def test_self_compare_is_exact(self):
        result = self.compare()
        self.assertEqual(result["causally_compared_steps"], 3)
        self.assertEqual(result["noncausal_steps"], 0)
        self.assertIsNone(result["first_token_divergence"])
        self.assertEqual(result["routes"]["compared_rows"], 6)
        self.assertEqual(result["routes"]["selected_slots"], 12)
        self.assertEqual(result["routes"]["changed_rows"], 0)
        self.assertEqual(result["routes"]["ordered_only_reorders"], 0)
        self.assertEqual(result["routes"]["replacements"], 0)
        self.assertEqual(result["logits"]["byte_changed_steps"], 0)
        self.assertEqual(result["logits"]["argmax_changed_steps"], 0)
        self.assertEqual(result["logits"]["top10_changed_steps"], 0)
        self.assertEqual(result["logits"]["max_abs"], 0.0)
        self.assertEqual(result["logits"]["mean_abs"], 0.0)

    def test_order_only_reorder_is_not_a_replacement(self):
        gpu_steps = base_steps()
        gpu_steps[1]["routes"][0]["experts"] = [5, 2]
        result = self.compare(gpu_steps=gpu_steps)
        routes = result["routes"]
        self.assertEqual(routes["changed_rows"], 0)
        self.assertEqual(routes["replacements"], 0)
        self.assertEqual(routes["ordered_only_reorders"], 1)
        self.assertEqual(routes["ordered_changed_rows"], 1)
        self.assertEqual(routes["first_divergence"]["kind"], "reorder")
        self.assertEqual(routes["first_divergence"]["step"], 1)
        self.assertEqual(routes["first_divergence"]["layer"], 4)
        step = result["steps"][1]["routes"]
        self.assertNotEqual(step["cpu_ordered_hash"], step["gpu_ordered_hash"])
        self.assertEqual(step["cpu_set_hash"], step["gpu_set_hash"])

    def test_one_membership_replacement_has_exact_denominator(self):
        gpu_steps = base_steps()
        gpu_steps[1]["routes"][1]["experts"] = [3, 12]
        result = self.compare(gpu_steps=gpu_steps)
        routes = result["routes"]
        self.assertEqual(routes["changed_rows"], 1)
        self.assertEqual(routes["ordered_only_reorders"], 0)
        self.assertEqual(routes["replacements"], 1)
        self.assertEqual(routes["max_replacements_in_row"], 1)
        self.assertAlmostEqual(routes["replacement_fraction"], 1 / 12)
        self.assertEqual(routes["first_divergence"]["kind"], "replacement")
        self.assertEqual(routes["max_divergence"]["layer"], 5)

    def test_logit_mutation_reports_per_step_and_global_error(self):
        gpu_logits = base_logits()
        gpu_logits[1][0] += 0.25
        result = self.compare(gpu_logits=gpu_logits)
        logits = result["logits"]
        self.assertEqual(logits["byte_changed_steps"], 1)
        self.assertEqual(logits["first_changed_step"], 1)
        self.assertEqual(logits["argmax_changed_steps"], 0)
        self.assertIsNone(logits["first_argmax_changed_step"])
        self.assertEqual(logits["top10_changed_steps"], 0)
        self.assertEqual(logits["max_abs_step"], 1)
        self.assertAlmostEqual(logits["max_abs"], 0.25)
        self.assertAlmostEqual(logits["mean_abs"], 0.25 / (3 * VOCAB))
        step = result["steps"][1]["logits"]
        self.assertFalse(step["byte_exact"])
        self.assertTrue(step["argmax_equal"])
        self.assertTrue(step["top10_set_equal"])
        self.assertAlmostEqual(step["max_abs"], 0.25)
        self.assertAlmostEqual(step["mean_abs"], 0.25 / VOCAB)

    def test_argmax_divergence_marks_later_steps_noncausal(self):
        gpu_logits = base_logits()
        gpu_logits[1][9] = 21.0  # token 9 replaces token 10 as argmax
        gpu_steps = base_steps()
        gpu_steps[2]["input_token"] = 9
        result = self.compare(gpu_logits=gpu_logits, gpu_steps=gpu_steps)
        self.assertEqual(result["causally_compared_steps"], 2)
        self.assertEqual(result["noncausal_steps"], 1)
        self.assertEqual(result["first_token_divergence"], {
            "step": 1, "kind": "argmax", "cpu": 10, "gpu": 9,
        })
        self.assertEqual(result["logits"]["first_argmax_changed_step"], 1)
        self.assertTrue(result["steps"][1]["causal_comparable"])
        self.assertFalse(result["steps"][2]["causal_comparable"])
        self.assertNotIn("logits", result["steps"][2])
        self.assertEqual(result["routes"]["compared_rows"], 4)
        self.assertEqual(result["logits"]["compared_steps"], 2)

    def test_nonfinite_logits_are_counted(self):
        gpu_logits = base_logits()
        gpu_logits[2][0] = float("nan")
        result = self.compare(gpu_logits=gpu_logits)
        self.assertEqual(result["logits"]["cpu_nonfinite"], 0)
        self.assertEqual(result["logits"]["gpu_nonfinite"], 1)
        self.assertIsNone(result["steps"][2]["logits"]["gpu_argmax"])
        self.assertFalse(result["steps"][2]["logits"]["top10_set_equal"])

    def test_valid_cuda_arm_metadata_is_retained(self):
        cpu_arm, gpu_arm = self.arms()
        result = self.compare(cpu_arm=cpu_arm, gpu_arm=gpu_arm)
        self.assertEqual(result["arms"]["cpu"]["calls"], 0)
        self.assertEqual(result["arms"]["gpu"]["calls"], 24)

    def test_cuda_fallback_or_call_shortfall_is_rejected(self):
        cpu_arm, gpu_arm = self.arms()
        for mutation in ({"fallbacks": 1}, {"calls": 23}):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(gpu_arm)
                bad.update(mutation)
                with self.assertRaises(COMPARE.CaptureError):
                    self.compare(cpu_arm=cpu_arm, gpu_arm=bad)


if __name__ == "__main__":
    unittest.main()
