#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Focused tests for route-capture to usage.waste conversion."""

import json
import os
import struct
import sys
import tempfile
import unittest


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import capture_to_usage as CONVERT  # noqa: E402


def capture(steps, *, top_k=2, schema=CONVERT.CAPTURE_SCHEMA):
    return {"schema": schema, "top_k": top_k, "steps": steps}


def step(index, routes):
    return {"index": index, "routes": routes}


class CaptureToUsageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def write_capture(self, name, value):
        path = os.path.join(self.temp.name, name)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(value, stream)
        return path

    def test_frequency_order_recency_and_binary_layout(self):
        path = self.write_capture(
            "capture.json",
            capture(
                [
                    step(0, []),
                    step(
                        1,
                        [
                            {"layer": 4, "experts": [2, 1]},
                            {"layer": 5, "experts": [3, 4]},
                        ],
                    ),
                    step(
                        2,
                        [
                            {"layer": 4, "experts": [1, 5]},
                            {"layer": 5, "experts": [4, 3]},
                        ],
                    ),
                ]
            ),
        )
        data = CONVERT.build_usage([path])
        self.assertEqual(data.total_tokens, 2)
        self.assertEqual(data.selections, 8)
        self.assertEqual(
            [(e.layer, e.expert_id, e.hits, e.last_seen) for e in data.entries],
            [
                (4, 1, 2, 5),
                (5, 3, 2, 8),
                (5, 4, 2, 7),
                (4, 2, 1, 1),
                (4, 5, 1, 6),
            ],
        )

        raw = CONVERT.encode_usage(data)
        self.assertEqual(len(raw), 16 + 5 * 16)
        self.assertEqual(
            CONVERT.USAGE_HEADER.unpack_from(raw),
            (CONVERT.USAGE_MAGIC, CONVERT.USAGE_VERSION, 2),
        )
        self.assertEqual(struct.unpack_from("<HHIII", raw, 16), (4, 1, 2, 5, 0))

    def test_multiple_captures_are_aggregated(self):
        first = self.write_capture(
            "first.json",
            capture([step(0, [{"layer": 4, "experts": [1, 2]}])]),
        )
        second = self.write_capture(
            "second.json",
            capture([step(0, [{"layer": 4, "experts": [2, 3]}])]),
        )
        data = CONVERT.build_usage([first, second])
        self.assertEqual((data.total_tokens, data.selections), (2, 4))
        self.assertEqual(
            [(e.layer, e.expert_id, e.hits, e.last_seen) for e in data.entries],
            [(4, 2, 2, 3), (4, 1, 1, 1), (4, 3, 1, 4)],
        )

    def test_malformed_routes_are_rejected(self):
        cases = {
            "wrong-schema.json": capture(
                [step(0, [{"layer": 4, "experts": [1, 2]}])], schema="other"
            ),
            "duplicate-layer.json": capture(
                [
                    step(
                        0,
                        [
                            {"layer": 4, "experts": [1, 2]},
                            {"layer": 4, "experts": [3, 4]},
                        ],
                    )
                ]
            ),
            "duplicate-expert.json": capture(
                [step(0, [{"layer": 4, "experts": [1, 1]}])]
            ),
            "wide-expert.json": capture(
                [step(0, [{"layer": 4, "experts": [1, 65536]}])]
            ),
        }
        for name, value in cases.items():
            with self.subTest(name=name):
                path = self.write_capture(name, value)
                with self.assertRaises(CONVERT.CaptureError):
                    CONVERT.build_usage([path])

    def test_route_free_capture_is_rejected(self):
        path = self.write_capture("empty.json", capture([step(0, [])]))
        with self.assertRaisesRegex(CONVERT.CaptureError, "no routed"):
            CONVERT.build_usage([path])

    def test_write_refuses_accidental_overwrite(self):
        path = self.write_capture(
            "capture.json",
            capture([step(0, [{"layer": 4, "experts": [1, 2]}])]),
        )
        data = CONVERT.build_usage([path])
        output = os.path.join(self.temp.name, "usage.waste")
        CONVERT.write_usage(output, data)
        with self.assertRaisesRegex(CONVERT.CaptureError, "already exists"):
            CONVERT.write_usage(output, data)
        CONVERT.write_usage(output, data, force=True)
        with open(output, "rb") as stream:
            self.assertEqual(stream.read(), CONVERT.encode_usage(data))


if __name__ == "__main__":
    unittest.main()
