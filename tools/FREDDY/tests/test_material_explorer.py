from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
import unittest

from ibc.io import MATERIAL_HEADER
from ibc.material_explorer import (
    MaterialExplorerModel,
    extrema_preserving_indices,
    loss_tangent,
)


def _write_material(path: Path, rows: list[tuple[float, float, float, float, float]]) -> None:
    body = "\n".join(",".join(str(value) for value in row) for row in rows)
    path.write_text(MATERIAL_HEADER + "\n" + body + "\n", encoding="utf-8")


class MaterialExplorerModelTests(unittest.TestCase):
    def test_direct_properties_summary_and_loss_tangent(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "absorber.csv"
            _write_material(
                path,
                [
                    (1.0e9, 4.0, -0.4, 2.0, -0.1),
                    (2.0e9, 5.0, -1.0, 3.0, -0.3),
                ],
            )
            model = MaterialExplorerModel()
            source = model.add_path(path)

            self.assertIsNotNone(source)
            assert source is not None
            summary = model.summary(source)
            self.assertEqual(summary.sample_count, 2)
            self.assertEqual((summary.frequency_min_ghz, summary.frequency_max_ghz), (1.0, 2.0))
            self.assertEqual(summary.eps_real_range, (4.0, 5.0))
            self.assertEqual(summary.eps_imag_range, (-1.0, -0.4))
            self.assertEqual(summary.mu_real_range, (2.0, 3.0))
            self.assertEqual(summary.mu_imag_range, (-0.3, -0.1))
            self.assertEqual(summary.electric_loss_tangent_range, (0.1, 0.2))
            self.assertAlmostEqual(summary.magnetic_loss_tangent_range[0], 0.05)
            self.assertAlmostEqual(summary.magnetic_loss_tangent_range[1], 0.1)

            sample = model.sample_at(0, 0)
            self.assertEqual(sample.frequency_ghz, 1.0)
            self.assertEqual(sample.eps_imag, -0.4)
            self.assertEqual(sample.mu_imag, -0.1)
            self.assertAlmostEqual(sample.electric_loss_tangent, 0.1)

    def test_nonpositive_real_part_has_no_reported_loss_tangent(self) -> None:
        self.assertIsNone(loss_tangent(complex(0.0, -1.0)))
        self.assertIsNone(loss_tangent(complex(-2.0, -1.0)))
        self.assertAlmostEqual(loss_tangent(complex(4.0, -0.4)), 0.1)
        zero = loss_tangent(complex(1.0, 0.0))
        self.assertEqual(zero, 0.0)
        self.assertEqual(math.copysign(1.0, zero), 1.0)

    def test_same_frequency_comparison_interpolates_without_extrapolation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "material.csv"
            _write_material(
                path,
                [
                    (1.0e9, 2.0, -0.2, 1.0, 0.0),
                    (3.0e9, 4.0, -0.6, 2.0, -0.2),
                ],
            )
            model = MaterialExplorerModel()
            source = model.add_path(path)
            assert source is not None

            sample = model.sample_at_frequency(source, 2.0)
            assert sample is not None
            self.assertEqual(sample.eps_real, 3.0)
            self.assertEqual(sample.eps_imag, -0.4)
            self.assertEqual(sample.mu_real, 1.5)
            self.assertEqual(sample.mu_imag, -0.1)
            self.assertIsNone(model.sample_at_frequency(source, 0.9))
            self.assertIsNone(model.sample_at_frequency(source, 3.1))

    def test_extrema_preserving_decimation_keeps_narrow_resonance(self) -> None:
        values = [0.0] * 5001
        values[2501] = 100.0

        indices = extrema_preserving_indices(values, limit=100)

        self.assertLessEqual(len(indices), 100)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], len(values) - 1)
        self.assertIn(2501, indices)

    def test_resolved_path_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "material.csv"
            _write_material(path, [(1.0e9, 2.0, -0.1, 1.0, 0.0)])
            model = MaterialExplorerModel()

            self.assertIsNotNone(model.add_path(path))
            self.assertIsNone(model.add_path(path.parent / "." / path.name))
            self.assertEqual(len(model), 1)

    @unittest.skipIf(
        os.path.normcase("A.csv") == os.path.normcase("a.csv"),
        "host path rules are case-insensitive",
    )
    def test_case_distinct_files_remain_distinct_on_case_sensitive_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            upper = Path(folder) / "A.csv"
            lower = Path(folder) / "a.csv"
            _write_material(upper, [(1.0e9, 2.0, 0.0, 1.0, 0.0)])
            _write_material(lower, [(1.0e9, 3.0, 0.0, 1.0, 0.0)])
            if upper.samefile(lower):
                self.skipTest("temporary filesystem is case-insensitive")
            model = MaterialExplorerModel()
            self.assertIsNotNone(model.add_path(upper))
            self.assertIsNotNone(model.add_path(lower))
            self.assertEqual(len(model), 2)

    def test_color_identity_survives_removing_an_earlier_source(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            model = MaterialExplorerModel()
            sources = []
            for index in range(3):
                path = Path(folder) / f"material_{index}.csv"
                _write_material(path, [(1.0e9, 2.0 + index, 0.0, 1.0, 0.0)])
                source = model.add_path(path)
                assert source is not None
                sources.append(source)

            model.remove_keys([sources[0].key])

            self.assertEqual(model.sources[0].color_index, sources[1].color_index)
            self.assertEqual(model.sources[1].color_index, sources[2].color_index)

    def test_batch_keeps_valid_files_when_another_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            valid = Path(folder) / "valid.csv"
            invalid = Path(folder) / "ibc.csv"
            _write_material(valid, [(1.0e9, 2.0, -0.1, 1.0, 0.0)])
            invalid.write_text(
                "frequency_hz,resistance_ohm,reactance_ohm\n1000000000,50,0\n",
                encoding="utf-8",
            )
            model = MaterialExplorerModel()

            added, duplicates, errors = model.add_requests([valid, invalid, valid])

            self.assertEqual(len(added), 1)
            self.assertEqual(duplicates, [valid.resolve()])
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0][0], invalid.resolve())
            self.assertEqual(len(model), 1)

    def test_common_frequency_range_uses_overlap_without_extrapolation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "first.csv"
            second = Path(folder) / "second.csv"
            third = Path(folder) / "third.csv"
            _write_material(
                first,
                [(1.0e9, 2.0, 0.0, 1.0, 0.0), (3.0e9, 2.0, 0.0, 1.0, 0.0)],
            )
            _write_material(
                second,
                [(2.0e9, 3.0, 0.0, 1.0, 0.0), (4.0e9, 3.0, 0.0, 1.0, 0.0)],
            )
            _write_material(third, [(8.0e9, 4.0, 0.0, 1.0, 0.0)])
            model = MaterialExplorerModel()
            model.add_path(first)
            model.add_path(second)
            self.assertEqual(model.common_frequency_range(), (2.0, 3.0))
            model.add_path(third)
            self.assertIsNone(model.common_frequency_range())

    def test_failed_reload_retains_last_valid_cached_table(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "material.csv"
            _write_material(path, [(1.0e9, 2.0, -0.1, 1.0, 0.0)])
            model = MaterialExplorerModel()
            original = model.add_path(path)
            assert original is not None

            path.write_text(
                MATERIAL_HEADER + "\n1000000000,2.0,0.25,1.0,0.0\n# changed\n",
                encoding="utf-8",
            )
            self.assertEqual(model.source_state(original), "changed")
            with self.assertRaisesRegex(ValueError, "gain-sign imaginary"):
                model.reload_key(original.key)

            retained = model.source_for_key(original.key)
            self.assertIs(retained, original)
            self.assertEqual(retained.table.eps_r, [complex(2.0, -0.1)])

    def test_inactive_material_mix_target_is_not_offered_to_explorer(self) -> None:
        from ibc.ui import ImpedanceGui

        class Value:
            def __init__(self, value: str) -> None:
                self.value = value

            def get(self) -> str:
                return self.value

        class MixState:
            mix_components = [{"file": "component.csv"}]
            mix_prop_file_var = Value("stale_target.csv")
            mix_prop_source_var = Value("Material file")
            property_objective = False

            def _mix_objective_is_property(self) -> bool:
                return self.property_objective

        state = MixState()
        requests = ImpedanceGui._material_explorer_mix_sources(state)
        self.assertEqual([str(request[0]) for request in requests], ["component.csv"])

        state.property_objective = True
        state.mix_prop_source_var.value = "Constant values"
        requests = ImpedanceGui._material_explorer_mix_sources(state)
        self.assertEqual([str(request[0]) for request in requests], ["component.csv"])

        state.mix_prop_source_var.value = "Material file"
        requests = ImpedanceGui._material_explorer_mix_sources(state)
        self.assertEqual(
            [str(request[0]) for request in requests],
            ["component.csv", "stale_target.csv"],
        )


try:
    from PySide6.QtWidgets import QApplication

    from ibc.ui import ImpedanceGui

    UI_AVAILABLE = True
except Exception:
    QApplication = None  # type: ignore[assignment]
    ImpedanceGui = None  # type: ignore[assignment]
    UI_AVAILABLE = False


@unittest.skipUnless(UI_AVAILABLE, "GUI dependencies are unavailable")
class MaterialExplorerUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_mode_uses_full_read_only_workspace(self) -> None:
        workspace = ImpedanceGui()
        try:
            self.assertIn("Material Explorer", workspace._mode_labels)
            index = workspace._mode_labels.index("Material Explorer")
            workspace.mode_stack.setCurrentIndex(index)
            self.app.processEvents()

            self.assertTrue(workspace.layers_group.isHidden())
            self.assertTrue(workspace.results_pane.isHidden())

            workspace.mode_stack.setCurrentIndex(0)
            self.app.processEvents()
            self.assertFalse(workspace.layers_group.isHidden())
            self.assertFalse(workspace.results_pane.isHidden())
        finally:
            workspace.deleteLater()
            self.app.processEvents()

    def test_loading_explorer_file_does_not_dirty_project_or_change_stack(self) -> None:
        workspace = ImpedanceGui()
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "material.csv"
                _write_material(path, [(1.0e9, 2.0, -0.1, 1.0, 0.0)])
                before = list(workspace.layers)
                exported: list[tuple[str, str]] = []
                cleared: list[bool] = []
                workspace.nominal_artifact_exported.connect(
                    lambda kind, output: exported.append((kind, output))
                )
                workspace.nominal_artifact_cleared.connect(
                    lambda: cleared.append(True)
                )

                added, duplicates, errors = workspace.material_explorer.add_requests([path])

                self.assertEqual((added, duplicates, errors), (1, 0, []))
                self.assertEqual(workspace.layers, before)
                self.assertFalse(workspace.is_dirty())
                self.assertEqual(exported, [])
                self.assertEqual(cleared, [])

                added, duplicates, errors = workspace.material_explorer.add_requests([path])
                self.assertEqual((added, duplicates, errors), (0, 1, []))
                self.assertIn(
                    "1 already loaded", workspace.material_explorer.status_label.text()
                )
                self.assertEqual(len(workspace.material_explorer.source_list.selectedItems()), 1)
        finally:
            workspace.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
