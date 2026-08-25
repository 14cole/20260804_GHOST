#!/usr/bin/env python3
"""Focused regressions for backend trust, frame, and unit invariants."""

from __future__ import annotations

import ctypes
import importlib.machinery
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Backend"
sys.path.insert(0, str(BACKEND))

import feature_sum  # noqa: E402
import fmm_helmholtz_2d  # noqa: E402
import rcs_solver  # noqa: E402


class NativeLoaderTrustTests(unittest.TestCase):
    def test_native_candidates_never_come_from_working_directory(self) -> None:
        loaded = []
        fake_library = SimpleNamespace(compute_sk_blocks_batch_q=object())
        with tempfile.TemporaryDirectory() as directory:
            untrusted = Path(directory)
            for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                (untrusted / ("fmm_near_cy" + suffix)).write_bytes(b"untrusted")
            for name in (
                "fmm_near.so",
                "fmm_near.dll",
                "fmm_near.dylib",
            ):
                (untrusted / name).write_bytes(b"untrusted")

            previous = Path.cwd()
            os.chdir(untrusted)
            try:
                with mock.patch.object(
                    ctypes,
                    "CDLL",
                    side_effect=lambda path: (
                        loaded.append(Path(path).resolve()) or fake_library
                    ),
                ):
                    kind, library = fmm_helmholtz_2d.FMMOperator._load_native()
            finally:
                os.chdir(previous)

        self.assertEqual(kind, "ctypes")
        self.assertIs(library, fake_library)
        self.assertTrue(loaded)
        self.assertTrue(
            all(candidate.parent == BACKEND.resolve() for candidate in loaded),
            loaded,
        )


class BodyFrameExportTests(unittest.TestCase):
    class _ReachedPostAxisValidation(RuntimeError):
        pass

    def _validate_axis_without_export(self, axis) -> None:
        with mock.patch.object(
            feature_sum,
            "surface_of_revolution_normal",
            side_effect=self._ReachedPostAxisValidation,
        ):
            with self.assertRaises(self._ReachedPostAxisValidation):
                feature_sum.export_signature_grim(
                    "unused.grim",
                    bor_result=None,
                    placements=[],
                    generatrix=np.asarray([[1.0, 1.0], [1.0, -1.0]]),
                    frequencies_ghz=[1.0],
                    aspects_deg=[0.0],
                    axis=axis,
                )

    def test_legacy_export_does_not_mutate_float_axis_array(self) -> None:
        axis = np.asarray([0.0, 0.0, 2.0], dtype=float)
        original = axis.copy()
        self._validate_axis_without_export(axis)
        np.testing.assert_array_equal(axis, original)

    def test_legacy_export_accepts_read_only_canonical_axis(self) -> None:
        axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
        axis.setflags(write=False)
        self._validate_axis_without_export(axis)
        self.assertFalse(axis.flags.writeable)

    def test_legacy_body_frame_export_rejects_non_z_axis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "signature.grim"
            with self.assertRaisesRegex(ValueError, r"canonical \+z BoR axis"):
                feature_sum.export_signature_grim(
                    str(destination),
                    bor_result=None,
                    placements=[],
                    generatrix=np.asarray([[1.0, 1.0], [1.0, -1.0]]),
                    frequencies_ghz=[1.0],
                    aspects_deg=[0.0],
                    axis=(1.0, 0.0, 0.0),
                )
            self.assertEqual(list(Path(directory).iterdir()), [])


def _open_chain_snapshot(points_m, meters_scale):
    pairs = []
    for start, end in points_m:
        pairs.append(
            {
                "x1": start[0] / meters_scale,
                "y1": start[1] / meters_scale,
                "x2": end[0] / meters_scale,
                "y2": end[1] / meters_scale,
            }
        )
    return {
        "segments": [
            {
                "name": "sheet",
                "seg_type": 1,
                "properties": ["1", "0", "0", "0", "0"],
                "point_pairs": pairs,
            }
        ],
        "ibcs": [],
        "dielectrics": [],
    }


class GeometryPreflightUnitTests(unittest.TestCase):
    def test_equivalent_meter_and_inch_geometry_returns_same_report(self) -> None:
        physical = [
            ((0.0, 0.0), (0.01, 0.0)),
            ((0.01, 0.0), (0.02, 0.0)),
        ]
        reports = []
        for scale in (1.0, 0.0254):
            reports.append(
                rcs_solver.validate_geometry_snapshot_for_solver(
                    _open_chain_snapshot(physical, scale),
                    base_dir=".",
                    meters_scale=scale,
                )
            )
        self.assertEqual(reports[0], reports[1])

    def test_equivalent_submicron_crack_has_same_meter_space_diagnosis(self) -> None:
        physical = [
            ((0.0, 0.0), (0.01, 0.0)),
            ((0.0100001, 0.0), (0.02, 0.0)),
        ]
        messages = []
        for scale in (1.0, 0.0254):
            with self.assertRaisesRegex(ValueError, "Geometry crack") as raised:
                rcs_solver.validate_geometry_snapshot_for_solver(
                    _open_chain_snapshot(physical, scale),
                    base_dir=".",
                    meters_scale=scale,
                )
            messages.append(str(raised.exception).split(" -- ", 1)[1])
        self.assertEqual(messages[0], messages[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
