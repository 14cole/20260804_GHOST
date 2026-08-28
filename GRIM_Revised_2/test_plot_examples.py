"""Focused tests for the runnable folder-to-plot examples."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from examples import plot_folder_azimuth_sweeps as azimuth_example
from examples import plot_folder_frequency_sweeps as frequency_example
from grim_dataset import RcsGrid


def _grid(*, db_offset: float = 0.0) -> RcsGrid:
    db_values = np.asarray(
        [[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]],
        dtype=float,
    ) + float(db_offset)
    return RcsGrid(
        azimuths=(-10.0, 0.0, 10.0),
        elevations=(0.0,),
        frequencies=(1.0, 2.0),
        polarizations=("VV",),
        rcs_power=np.power(10.0, db_values / 10.0).reshape(3, 1, 2, 1),
        rcs_phase=np.zeros((3, 1, 2, 1), dtype=float),
        units={
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
            "angular_coordinate_system": "conic",
        },
        extra={"phase_reference": "origin=(0,0,0); exp(-jkr)"},
    )


def _fake_renderer(captured):
    def render(spec, output_path, **_kwargs):
        captured.append(spec)
        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake png")
        return destination

    return render


class FolderPlotExampleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        _grid().save(self.root / "first.grim")
        _grid(db_offset=10.0).save(self.root / "second.grim")
        (self.root / "ignore.png").write_bytes(b"not a dataset")

    def tearDown(self):
        self.temporary.cleanup()

    def test_discovery_is_sorted_and_filters_unsupported_files(self):
        paths = azimuth_example.discover_dataset_paths(self.root)
        self.assertEqual([path.name for path in paths], ["first.grim", "second.grim"])
        with self.assertRaisesRegex(ValueError, "stays inside"):
            frequency_example.discover_dataset_paths(self.root, pattern="../*")

    def test_cartesian_azimuth_example_builds_overlay_and_avoids_overwrite(self):
        output_dir = self.root / "azimuth plots"
        parser = azimuth_example.build_parser()
        arguments = [
            str(self.root),
            "--output-dir",
            str(output_dir),
            "--frequency",
            "2",
            "--elevation",
            "0",
            "--polarization",
            "VV",
            "--dpi",
            "72",
        ]
        captured = []
        with mock.patch.object(
            azimuth_example,
            "render_plot_png",
            side_effect=_fake_renderer(captured),
        ), redirect_stdout(io.StringIO()):
            first_paths = azimuth_example.run(
                parser.parse_args(arguments), parser=parser
            )
            second_paths = azimuth_example.run(
                parser.parse_args(arguments), parser=parser
            )

        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0].kind, "azimuth_rect")
        self.assertEqual(len(captured[0].series), 2)
        self.assertEqual(captured[0].series[0].x, (-10.0, 0.0, 10.0))
        self.assertTrue(first_paths[0].is_file())
        self.assertTrue(second_paths[0].is_file())
        self.assertNotEqual(first_paths[0], second_paths[0])
        self.assertTrue(second_paths[0].stem.endswith("_2"))

    def test_frequency_example_calculates_optional_display_domain_percentile(self):
        output_dir = self.root / "frequency plots"
        parser = frequency_example.build_parser()
        args = parser.parse_args(
            [
                str(self.root),
                "--output-dir",
                str(output_dir),
                "--azimuth-band",
                "-10",
                "10",
                "--percentile",
                "90",
                "--elevation",
                "0",
                "--polarization",
                "VV",
                "--dpi",
                "72",
            ]
        )
        captured = []
        with mock.patch.object(
            frequency_example,
            "render_plot_png",
            side_effect=_fake_renderer(captured),
        ), redirect_stdout(io.StringIO()):
            rendered = frequency_example.run(args, parser=parser)

        self.assertEqual(len(rendered), 1)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].kind, "frequency")
        self.assertEqual(len(captured[0].series), 2)
        np.testing.assert_allclose(captured[0].series[0].y, (18.0, 28.0))
        np.testing.assert_allclose(captured[0].series[1].y, (28.0, 38.0))
        self.assertIn("P90 across Azimuth [-10, 10] deg", captured[0].title)

    def test_cartesian_example_runs_through_real_headless_png_renderer(self):
        output_dir = self.root / "rendered"
        parser = azimuth_example.build_parser()
        args = parser.parse_args(
            [
                str(self.root),
                "--output-dir",
                str(output_dir),
                "--frequency",
                "1",
                "--dpi",
                "72",
                "--width",
                "3",
                "--height",
                "2",
            ]
        )
        with redirect_stdout(io.StringIO()):
            rendered = azimuth_example.run(args, parser=parser)
        self.assertEqual(len(rendered), 1)
        self.assertEqual(rendered[0].read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_percentile_without_band_exits_with_actionable_error(self):
        with redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as raised:
                frequency_example.main([str(self.root), "--percentile", "90"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--percentile requires --azimuth-band", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
