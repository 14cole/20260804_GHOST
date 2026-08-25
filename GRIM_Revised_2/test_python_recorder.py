from __future__ import annotations

import os
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from grim_dataset import RcsGrid
from grim_python import (
    DatasetReference,
    PythonScriptRecorder,
    join_datasets,
    plot_datasets,
    save_dataset_batch,
)


class PythonRecorderTests(unittest.TestCase):
    def _grid(self) -> RcsGrid:
        power = np.arange(12, dtype=float).reshape(3, 1, 2, 2) + 1.0
        phase = np.zeros_like(power)
        return RcsGrid(
            np.asarray([-180.0, 0.0, 180.0]),
            np.asarray([0.0]),
            np.asarray([9.0, 10.0]),
            np.asarray(["HH", "VV"]),
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain="power_phase",
            units={"frequency": "GHz", "rcs_log_unit": "dBsm"},
        )

    @staticmethod
    def _run_script(script_path: Path, cwd: Path) -> subprocess.CompletedProcess:
        environment = dict(os.environ)
        # Do not let the generated recipe inherit GRIM itself through
        # PYTHONPATH; it must use the embedded GRIM_MODULE_DIR bootstrap.
        # Preserve unrelated dependency locations so this test also works in
        # isolated/disposable Python environments.
        project_dir = Path(__file__).resolve().parent
        python_paths = []
        for entry in environment.get("PYTHONPATH", "").split(os.pathsep):
            if not entry:
                continue
            try:
                if Path(entry).resolve() == project_dir:
                    continue
            except OSError:
                pass
            python_paths.append(entry)
        if python_paths:
            environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        else:
            environment.pop("PYTHONPATH", None)
        environment["MPLBACKEND"] = "Agg"
        return subprocess.run(
            [sys.executable, str(script_path)],
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_generated_dataset_script_runs_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            source = root / "source.grim"
            output = root / "result.grim"
            script_path = root / "recipe.py"
            run_directory = root / "elsewhere"
            run_directory.mkdir()
            self._grid().save(source)

            recorder = PythonScriptRecorder()
            original = DatasetReference("stable-source-id", "Editable display", str(source))
            cropped = DatasetReference("stable-crop-id", "Cropped")
            wrapped = DatasetReference("stable-wrap-id", "Wrapped")
            recorder.bind_loaded(original)
            recorder.record_method(
                cropped,
                original,
                "axis_crop",
                kwargs={
                    "azimuths": [-180.0, 0.0],
                    "elevations": [0.0],
                    "frequencies": [9.0],
                    "polarizations": ["HH"],
                },
                comment="Resolved crop",
            )
            recorder.record_method(
                wrapped,
                cropped,
                "wrap_azimuth",
                args=("0_360",),
                comment="Resolved azimuth wrap",
            )
            recorder.record_save(wrapped, str(output))
            script_path.write_text(recorder.script, encoding="utf-8")

            completed = self._run_script(script_path, run_directory)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())
            replayed = RcsGrid.load(output)
            np.testing.assert_allclose(replayed.azimuths, [0.0, 180.0])
            self.assertEqual(replayed.rcs_power.shape, (2, 1, 1, 1))
            self.assertNotIn("PySide6", recorder.script)
            self.assertIn("GRIM_MODULE_DIR", recorder.script)

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None,
        "Matplotlib is not installed in this runtime",
    )
    def test_generated_plot_script_uses_agg_and_writes_png(self):
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            source = root / "source.grim"
            image = root / "azimuth.png"
            script_path = root / "plot_recipe.py"
            run_directory = root / "elsewhere"
            run_directory.mkdir()
            self._grid().save(source)

            recorder = PythonScriptRecorder()
            original = DatasetReference("stable-source-id", "Dataset A", str(source))
            recorder.bind_loaded(original)
            recorder.record_plot(
                [original],
                names=["Dataset A"],
                mode="azimuth_rect",
                parameters={
                    "azimuths": [-180.0, 0.0, 180.0],
                    "elevations": [0.0],
                    "frequencies": [9.0],
                    "polarization": "HH",
                    "phase": False,
                    "scale": "dbsm",
                    "colormap": "viridis",
                    "show_grid": True,
                    "show_legend": True,
                    "polar_zero": "N",
                },
            )
            recorder.record_plot_save(str(image), dpi=100)
            script_path.write_text(recorder.script, encoding="utf-8")

            completed = self._run_script(script_path, run_directory)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(image.is_file())
            self.assertGreater(image.stat().st_size, 100)

    def test_clear_resets_bindings_and_variable_names(self):
        recorder = PythonScriptRecorder()
        source = DatasetReference("id", "Data", "input.grim")
        recorder.bind_loaded(source)
        self.assertIn("dataset_1", recorder.script)
        recorder.clear()
        recorder.bind_loaded(source)
        self.assertEqual(recorder.script.count("dataset_1 ="), 1)
        self.assertNotIn("dataset_2 =", recorder.script)

    def test_identical_plot_specs_are_deduplicated(self):
        recorder = PythonScriptRecorder()
        source = DatasetReference("id", "Data", "input.grim")
        parameters = {
            "azimuths": [-180.0, 0.0],
            "elevations": [0.0],
            "frequencies": [9.0],
            "polarization": "HH",
        }
        first = recorder.record_plot(
            [source], names=["Data"], mode="azimuth_rect", parameters=parameters
        )
        second = recorder.record_plot(
            [source], names=["Data"], mode="azimuth_rect", parameters=parameters
        )
        self.assertEqual(first, second)
        self.assertEqual(recorder.script.count("plot_1 ="), 1)
        self.assertNotIn("plot_2 =", recorder.script)

    def test_changed_plot_is_the_export_target(self):
        recorder = PythonScriptRecorder()
        source = DatasetReference("id", "Data", "input.grim")
        base = {
            "azimuths": [-180.0, 0.0],
            "elevations": [0.0],
            "polarization": "HH",
        }
        recorder.record_plot(
            [source],
            names=["Data"],
            mode="azimuth_rect",
            parameters={**base, "frequencies": [9.0]},
        )
        recorder.record_plot(
            [source],
            names=["Data"],
            mode="azimuth_rect",
            parameters={**base, "frequencies": [10.0]},
        )
        recorder.record_plot_save("newest.png")
        self.assertIn("plot_2.savefig(", recorder.script)
        self.assertNotIn("plot_1.savefig(", recorder.script)

    def test_unsupported_plot_cannot_export_stale_supported_plot(self):
        recorder = PythonScriptRecorder()
        source = DatasetReference("id", "Data", "input.grim")
        recorder.record_plot(
            [source],
            names=["Data"],
            mode="azimuth_rect",
            parameters={
                "azimuths": [-180.0, 0.0],
                "elevations": [0.0],
                "frequencies": [9.0],
                "polarization": "HH",
            },
        )
        recorder.record_unsupported_plot("waterfall", "not supported")
        self.assertFalse(recorder.record_plot_save("stale.png"))
        self.assertNotIn("plot_1.savefig(", recorder.script)

    def test_save_batch_is_one_ordered_transactional_call(self):
        recorder = PythonScriptRecorder()
        first = DatasetReference("first-id", "First", "first-input.grim")
        second = DatasetReference("second-id", "Second", "second-input.grim")

        self.assertTrue(
            recorder.record_save_batch(
                [(first, "first-output.grim"), (second, "second-output.grim")]
            )
        )

        script = recorder.script
        self.assertEqual(script.count("save_dataset_batch(["), 1)
        first_entry = script.index("    (dataset_1, Path(")
        second_entry = script.index("    (dataset_2, Path(")
        self.assertLess(first_entry, second_entry)

    def test_save_batch_restores_existing_outputs_when_later_publish_fails(self):
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            first = root / "first.grim"
            second = root / "second.grim"
            first.write_bytes(b"original first")
            second.write_bytes(b"original second")
            real_replace = os.replace

            def fail_second_stage(source, destination):
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    source_path.name.startswith(".grim-stage-")
                    and destination_path.resolve() == second.resolve()
                ):
                    raise OSError("injected second publication failure")
                real_replace(source, destination)

            with mock.patch("grim_python._replace_file", side_effect=fail_second_stage):
                with self.assertRaisesRegex(OSError, "injected second publication"):
                    save_dataset_batch(
                        [(self._grid(), first), (self._grid(), second)]
                    )

            self.assertEqual(first.read_bytes(), b"original first")
            self.assertEqual(second.read_bytes(), b"original second")
            self.assertEqual(list(root.glob(".grim-stage-*")), [])
            self.assertEqual(list(root.glob(".grim-backup-*")), [])

    def test_join_helper_uses_half_of_available_memory(self):
        available = 10_000
        fake_psutil = mock.Mock()
        fake_psutil.virtual_memory.return_value.available = available
        expected = object()
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            with mock.patch.object(
                RcsGrid, "join_many", return_value=expected
            ) as join_many:
                result = join_datasets(self._grid(), self._grid(), tol=2.0e-5)

        self.assertIs(result, expected)
        args, kwargs = join_many.call_args
        self.assertEqual(len(args), 2)
        self.assertEqual(kwargs["tol"], 2.0e-5)
        self.assertEqual(kwargs["overlap"], "error")
        self.assertEqual(kwargs["max_output_bytes"], available // 2)

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None,
        "Matplotlib is not installed in this runtime",
    )
    def test_headless_plot_skips_incompatible_dataset_and_requires_one_match(self):
        compatible = self._grid()
        incompatible = compatible.axis_crop(frequencies=[10.0])
        parameters = {
            "mode": "azimuth_rect",
            "azimuths": [-180.0, 0.0, 180.0],
            "elevations": [0.0],
            "frequencies": [9.0],
            "polarization": "HH",
        }
        figure = plot_datasets(
            [("compatible", compatible), ("missing frequency", incompatible)],
            **parameters,
        )
        self.assertEqual(len(figure.axes[0].lines), 1)
        figure.clear()
        with self.assertRaisesRegex(ValueError, "None of the selected datasets"):
            plot_datasets([("missing frequency", incompatible)], **parameters)


if __name__ == "__main__":
    unittest.main()
