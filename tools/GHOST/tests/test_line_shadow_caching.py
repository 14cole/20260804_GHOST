"""Reusable packed line body-shadow correctness and job-control tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


BACKEND = Path(__file__).resolve().parents[1] / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import feature_sum  # noqa: E402
from line_expand import (  # noqa: E402
    SeamCoefficients,
    expand_perimeter,
    prepare_perimeter_frame,
)
from occluder import Occluder, PackedVisibility  # noqa: E402


def _box_triangles() -> np.ndarray:
    corners = np.asarray([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=float)
    faces = (
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    )
    return np.asarray([[corners[index] for index in face]
                       for face in faces], dtype=float)


def _coefficient(frequency: float) -> SeamCoefficients:
    angles = np.asarray([0.0, 90.0, 180.0])
    return SeamCoefficients(
        float(frequency),
        angles,
        np.asarray([1.0 + 0.25j] * len(angles)),
        np.asarray([0.7 - 0.1j] * len(angles)),
    )


def _line_geometry():
    segments = np.asarray([[[-2.0, -0.4, 0.0], [-2.0, 0.4, 0.0]]])
    normals = np.asarray([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    return segments, normals


class _LegacyOccluder:
    """Historical occluder-like surface exposing only ``visible``."""

    def __init__(self):
        self.calls = 0

    def visible(self, points, _direction):
        self.calls += 1
        values = np.ones(len(np.atleast_2d(points)), dtype=bool)
        values[1::2] = False
        return values


class LineShadowCachingTests(unittest.TestCase):
    def test_expand_perimeter_packed_shadow_matches_scalar_reference(self):
        blocker = Occluder(_box_triangles(), bias=1e-6)
        segments, normals = _line_geometry()
        directions = np.asarray([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
        piece_length = 0.1
        common = dict(
            segments=segments,
            coefficients=_coefficient(1.0),
            normal_fn=None,
            directions=directions,
            frequency_ghz=1.0,
            max_piece_length_m=piece_length,
            segment_normals=normals,
        )
        reference = expand_perimeter(occluder=blocker, **common)
        (
            _starts,
            _tangents,
            _lengths,
            midpoints,
            sampled_normals,
            _frame_tangents,
        ) = prepare_perimeter_frame(
            segments, piece_length, segment_normals=normals
        )
        packed = blocker.visible_many_packed(
            midpoints,
            directions,
            facing_normals=sampled_normals,
        )

        with mock.patch.object(
            blocker,
            "visible",
            side_effect=AssertionError("cached expansion retraced the body"),
        ):
            cached = expand_perimeter(
                occluder=blocker,
                _shadow_visibility=packed,
                **common,
            )

        for channel in ("F_vv", "F_hh", "F_vh"):
            np.testing.assert_allclose(
                cached[channel], reference[channel], rtol=0.0, atol=0.0
            )

        wrong_shape = PackedVisibility(
            np.zeros((len(midpoints) - 1, 1), dtype=np.uint8),
            n_points=len(midpoints) - 1,
            n_directions=len(directions),
        )
        with self.assertRaisesRegex(ValueError, "n_solver_pieces"):
            expand_perimeter(
                _shadow_visibility=wrong_shape,
                **common,
            )

    def test_export_caches_one_line_query_across_all_frequencies(self):
        blocker = Occluder(_box_triangles(), bias=1e-6)
        # Put this line on the radar-facing side so equivalence includes a
        # nonzero, unshadowed complex field rather than only zeros.
        # Vehicle +z maps to earth +x at the default horizontal body attitude.
        segments = np.asarray([[[0.0, -0.4, 2.0], [0.0, 0.4, 2.0]]])
        normals = np.asarray([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
        piece_length = 0.1
        midpoints = prepare_perimeter_frame(
            segments, piece_length, segment_normals=normals
        )[3]
        placement = {
            "delta": object(),
            "perimeter": segments,
            "segment_normals": normals,
            "kind": "delta",
            "line_id": "door-gap",
            "max_piece_length_m": piece_length,
            "shadow_points": midpoints,
        }
        frequencies = [1.0, 2.0, 3.0]
        grid = dict(
            frequencies_ghz=frequencies,
            azimuths_deg=[0.0, 120.0, 180.0, 240.0],
            elevations_deg=[0.0],
        )
        updates = []

        def resolve(_placements, frequency, _cache):
            resolved = dict(placement)
            resolved["delta"] = _coefficient(float(frequency))
            return [resolved]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cached_path = root / "cached.grim"
            scalar_path = root / "scalar.grim"
            with (
                mock.patch.object(
                    feature_sum,
                    "_prepared_line_placements_at_frequency",
                    side_effect=resolve,
                ),
                mock.patch.object(
                    blocker,
                    "visible_many_packed",
                    wraps=blocker.visible_many_packed,
                ) as packed_query,
                mock.patch.object(
                    blocker, "visible", wraps=blocker.visible
                ) as cached_visible,
            ):
                feature_sum.export_radar_grim(
                    str(cached_path),
                    bor_result=None,
                    placements=[placement],
                    occluder=blocker,
                    progress_callback=lambda done, total, message: updates.append(
                        (int(done), int(total), str(message))
                    ),
                    **grid,
                )
            packed_query.assert_called_once()
            cached_visible_calls = cached_visible.call_count
            self.assertGreater(cached_visible_calls, 0)

            # Disable only cache eligibility; expansion retains the identical
            # explicit piece length and therefore supplies a scalar oracle.
            with (
                mock.patch.object(
                    feature_sum,
                    "_prepared_line_placements_at_frequency",
                    side_effect=resolve,
                ),
                mock.patch.object(
                    feature_sum,
                    "_reusable_line_shadow_inputs",
                    return_value=(None,),
                ),
                mock.patch.object(
                    blocker, "visible", wraps=blocker.visible
                ) as scalar_visible,
            ):
                feature_sum.export_radar_grim(
                    str(scalar_path),
                    bor_result=None,
                    placements=[placement],
                    occluder=blocker,
                    **grid,
                )

            self.assertEqual(
                scalar_visible.call_count,
                cached_visible_calls * len(frequencies),
            )
            with np.load(cached_path, allow_pickle=False) as cached, np.load(
                scalar_path, allow_pickle=False
            ) as scalar:
                np.testing.assert_array_equal(
                    cached["rcs_amp_real"], scalar["rcs_amp_real"]
                )
                np.testing.assert_array_equal(
                    cached["rcs_amp_imag"], scalar["rcs_amp_imag"]
                )
                self.assertGreater(
                    float(np.max(np.abs(cached["rcs_amp_real"]))), 0.0
                )

        fractions = [done / total for done, total, _message in updates]
        self.assertEqual(fractions[0], 0.0)
        self.assertEqual(fractions[-1], 1.0)
        self.assertEqual(fractions, sorted(fractions))
        self.assertTrue(any(
            "line door-gap body-shadow" in message
            for _done, _total, message in updates
        ))

    def test_legacy_occluder_fallback_is_packed_progressive_and_cancellable(self):
        legacy = _LegacyOccluder()
        origins = np.asarray([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
        ])
        normals = np.repeat([[1.0, 0.0, 0.0]], len(origins), axis=0)
        directions = np.repeat([[1.0, 0.0, 0.0]], 5, axis=0)
        progress = []
        cached = feature_sum._precompute_line_shadow_visibility(
            ((origins, normals, "seal"),),
            directions,
            legacy,
            progress_callback=lambda done, total, message: progress.append(
                (int(done), int(total), str(message))
            ),
        )
        self.assertEqual(legacy.calls, len(directions))
        self.assertIsInstance(cached[0], PackedVisibility)
        np.testing.assert_array_equal(
            cached[0].to_dense(),
            np.repeat([[True], [False], [True]], len(directions), axis=1),
        )
        self.assertEqual(
            [(done, total) for done, total, _message in progress],
            [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)],
        )

        checks = 0
        cancelled_progress = []
        back_normals = -normals

        def cancel_on_third_look():
            nonlocal checks
            checks += 1
            # One setup check precedes the per-look checks.
            return checks >= 4

        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            feature_sum._precompute_line_shadow_visibility(
                ((origins, back_normals, "seal"),),
                directions,
                legacy,
                cancel_check=cancel_on_third_look,
                progress_callback=lambda done, total, message: (
                    cancelled_progress.append((int(done), int(total)))
                ),
            )
        self.assertEqual(cancelled_progress, [(1, 5), (2, 5)])

    def test_body_frame_export_also_reuses_frozen_line_visibility(self):
        blocker = Occluder(_box_triangles(), bias=1e-6)
        segments = np.asarray([[[0.0, -0.4, 2.0], [0.0, 0.4, 2.0]]])
        normals = np.asarray([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
        placement = {
            "delta": object(),
            "perimeter": segments,
            "segment_normals": normals,
            "kind": "delta",
            "max_piece_length_m": 0.1,
        }

        def resolve(_placements, frequency, _cache):
            resolved = dict(placement)
            resolved["delta"] = _coefficient(float(frequency))
            return [resolved]

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(
                    feature_sum,
                    "_prepared_line_placements_at_frequency",
                    side_effect=resolve,
                ),
                mock.patch.object(
                    blocker,
                    "visible_many_packed",
                    wraps=blocker.visible_many_packed,
                ) as packed_query,
            ):
                written = feature_sum.export_signature_grim(
                    str(Path(directory) / "body-frame.grim"),
                    bor_result=None,
                    placements=[placement],
                    generatrix=np.asarray([[1.0, 1.0], [1.0, -1.0]]),
                    frequencies_ghz=[1.0, 2.0],
                    aspects_deg=[0.0, 60.0],
                    rolls_deg=[0.0],
                    occluder=blocker,
                )

        packed_query.assert_called_once()
        self.assertEqual(len(written), 3)


if __name__ == "__main__":
    unittest.main()
