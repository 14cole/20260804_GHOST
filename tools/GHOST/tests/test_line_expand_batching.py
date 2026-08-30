"""Bounded look-batch regressions for 3-D line expansion."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


BACKEND = Path(__file__).resolve().parents[1] / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from line_expand import (  # noqa: E402
    SeamCoefficients,
    expand_perimeter,
    prepare_perimeter_frame,
)
from occluder import PackedVisibility  # noqa: E402


def _fixture():
    angles = np.asarray([0.0, 20.0, 65.0, 110.0, 160.0, 180.0])
    coefficients = SeamCoefficients(
        1.0,
        angles,
        np.asarray([
            0.20 + 0.40j,
            0.75 - 0.15j,
            1.10 + 0.25j,
            0.55 + 0.80j,
            -0.10 + 0.45j,
            0.30 - 0.20j,
        ]),
        np.asarray([
            0.90 - 0.30j,
            0.35 + 0.55j,
            -0.20 + 0.70j,
            0.80 - 0.10j,
            0.45 + 0.20j,
            -0.25 + 0.60j,
        ]),
    )
    tangent = np.asarray([0.30, 0.50, np.sqrt(0.66)])
    tangent /= np.linalg.norm(tangent)
    normal = np.asarray([1.0, -0.60, 0.0])
    normal /= np.linalg.norm(normal)
    start = np.asarray([0.03, -0.235, -0.08])
    segments = np.asarray([[start, start + 0.47 * tangent]])
    normals = np.asarray([[normal, normal]])

    azimuth = np.radians(np.linspace(-79.0, 79.0, 37))
    elevation = np.radians(17.0 * np.sin(np.linspace(0.0, 3.0 * np.pi, 37)))
    directions = np.column_stack((
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ))
    # Exercise both sides of the local seam-basis degeneracy threshold.  The
    # small positive x component makes these technically lit, while the raised
    # cosine correctly drives their physical contribution toward zero.
    near_parallel = np.asarray([
        tangent + 5.0e-13 * normal,
        tangent + 2.0e-12 * normal,
        -tangent + 7.0e-13 * normal,
        -tangent + 3.0e-12 * normal,
    ])
    directions = np.vstack((directions, near_parallel))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    common = dict(
        segments=segments,
        coefficients=coefficients,
        normal_fn=None,
        directions=directions,
        frequency_ghz=1.0,
        psi_tm_deg=37.0,
        psi_te_deg=-23.0,
        max_piece_length_m=0.005,
        segment_normals=normals,
    )
    return common


def _assert_fields_close(testcase, actual, expected):
    for channel in ("F_vv", "F_hh", "F_vh"):
        testcase.assertTrue(np.all(np.isfinite(actual[channel])))
        np.testing.assert_allclose(
            actual[channel], expected[channel], rtol=8.0e-13, atol=3.0e-15
        )


class LineLookBatchingTests(unittest.TestCase):
    def test_batched_cross_pol_and_near_degenerate_bases_match_scalar(self):
        common = _fixture()
        scalar = expand_perimeter(_look_batch_size=1, **common)
        batched = expand_perimeter(_look_batch_size=7, **common)
        automatic = expand_perimeter(**common)

        _assert_fields_close(self, batched, scalar)
        _assert_fields_close(self, automatic, scalar)
        self.assertGreater(float(np.max(np.abs(scalar["F_vh"]))), 1.0e-5)

    def test_small_default_path_retains_scalar_bit_pattern(self):
        common = _fixture()
        common["directions"] = common["directions"][:8]
        reference = expand_perimeter(_look_batch_size=1, **common)
        default = expand_perimeter(**common)
        for channel in ("F_vv", "F_hh", "F_vh"):
            np.testing.assert_array_equal(default[channel], reference[channel])

    def test_very_fine_perimeter_avoids_counterproductive_batch(self):
        common = _fixture()
        common["directions"] = common["directions"][:16]
        common["max_piece_length_m"] = 0.0002
        reference = expand_perimeter(_look_batch_size=1, **common)
        automatic = expand_perimeter(**common)
        for channel in ("F_vv", "F_hh", "F_vh"):
            np.testing.assert_array_equal(automatic[channel], reference[channel])

    def test_dense_and_packed_shadow_batches_match_scalar(self):
        common = _fixture()
        piece_count = len(prepare_perimeter_frame(
            common["segments"],
            common["max_piece_length_m"],
            segment_normals=common["segment_normals"],
        )[3])
        direction_count = len(common["directions"])
        piece_index = np.arange(piece_count)[:, None]
        direction_index = np.arange(direction_count)[None, :]
        dense = ((3 * piece_index + 5 * direction_index) % 7) != 0
        packed = PackedVisibility(
            np.packbits(dense, axis=1, bitorder="little"),
            n_points=piece_count,
            n_directions=direction_count,
        )

        scalar = expand_perimeter(
            _look_batch_size=1, _shadow_visibility=dense, **common
        )
        dense_batch = expand_perimeter(
            _look_batch_size=9, _shadow_visibility=dense, **common
        )
        packed_batch = expand_perimeter(
            _look_batch_size=9, _shadow_visibility=packed, **common
        )
        _assert_fields_close(self, dense_batch, scalar)
        _assert_fields_close(self, packed_batch, scalar)

    def test_live_shadow_keeps_per_look_query_semantics(self):
        common = _fixture()

        class LiveShadow:
            def __init__(self):
                self.calls = []

            def visible(self, points, direction, cancel_check=None):
                self.calls.append(np.asarray(direction).copy())
                if cancel_check is not None and cancel_check():
                    raise InterruptedError("Feature assembly cancelled.")
                index = np.arange(len(points))
                return ((index + int(direction[1] > 0.0)) % 4) != 0

        scalar_shadow = LiveShadow()
        requested_batch_shadow = LiveShadow()
        scalar = expand_perimeter(
            occluder=scalar_shadow, _look_batch_size=1, **common
        )
        requested_batch = expand_perimeter(
            occluder=requested_batch_shadow, _look_batch_size=11, **common
        )
        for channel in ("F_vv", "F_hh", "F_vh"):
            np.testing.assert_array_equal(
                requested_batch[channel], scalar[channel]
            )
        outward = common["segment_normals"][0, 0]
        expected_queries = int(np.count_nonzero(
            common["directions"] @ outward > 0.0
        ))
        self.assertEqual(len(scalar_shadow.calls), expected_queries)
        self.assertEqual(len(requested_batch_shadow.calls), expected_queries)

    def test_batch_progress_and_cancellation_remain_per_direction(self):
        common = _fixture()
        common["directions"] = common["directions"][:11]
        checks = 0
        progress = []

        def cancel_on_third_direction():
            nonlocal checks
            checks += 1
            return checks >= 3

        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            expand_perimeter(
                _look_batch_size=5,
                cancel_check=cancel_on_third_direction,
                progress_callback=lambda done, total: progress.append(
                    (done, total)
                ),
                **common,
            )
        self.assertEqual(checks, 3)
        self.assertEqual(progress, [(1, 11), (2, 11)])

        complete_progress = []
        expand_perimeter(
            _look_batch_size=5,
            progress_callback=lambda done, total: complete_progress.append(
                (done, total)
            ),
            **common,
        )
        self.assertEqual(
            complete_progress,
            [(index, 11) for index in range(1, 12)],
        )


if __name__ == "__main__":
    unittest.main()
