"""Preview-visibility regressions for the Assembly tree."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from assembly_tree import (
    AssemblyTree,
    AssemblyTreePanel,
    _COLUMN_INCLUDED,
    _COLUMN_VISIBILITY,
    _ROLE_GRID,
    _TYPE_BRANCH,
    _TYPE_ROOT,
    _attach,
    _align_grids_for_assembly,
    _grid_to_b64,
    _interp_target_axes,
    _dict_to_item,
    _item_to_dict,
    build_assembly_grid,
)
from grim_dataset import RcsGrid


def _grid(amplitude: float) -> RcsGrid:
    field = np.asarray([[[[complex(amplitude)]]]], dtype=np.complex128)
    return RcsGrid(
        [0.0],
        [0.0],
        [10.0],
        ["VV"],
        rcs=field,
        units={
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
        },
    )


class AssemblyVisibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _tree_with_two_leaves(self):
        tree = AssemblyTree()
        root = tree._make_node("Vehicle", _TYPE_ROOT, edit=False)
        branch = tree._make_node("Fasteners", _TYPE_BRANCH, parent=root, edit=False)
        first = tree._make_leaf("A", _grid(1.0))
        second = tree._make_leaf("B", _grid(2.0))
        _attach(tree, first, branch)
        _attach(tree, second, branch)
        return tree, root, branch, first, second

    def test_interp_subsets_polarizations_before_numeric_resampling(self):
        units = {
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
        }
        first = RcsGrid(
            [0.0, 2.0], [0.0], [10.0], ["VV", "HH"],
            rcs=np.ones((2, 1, 1, 2), dtype=np.complex128),
            units=units,
        )
        second = RcsGrid(
            [0.0, 1.0, 2.0], [0.0], [10.0], ["VV"],
            rcs=2.0 * np.ones((3, 1, 1, 1), dtype=np.complex128),
            units=units,
        )

        aligned = _align_grids_for_assembly([first, second], "interp")

        for grid in aligned:
            np.testing.assert_array_equal(grid.azimuths, [0.0, 2.0])
            np.testing.assert_array_equal(grid.polarizations, ["VV"])
        np.testing.assert_allclose(aligned[0].rcs, 1.0)
        np.testing.assert_allclose(aligned[1].rcs, 2.0)

    def test_interp_roundoff_endpoints_are_clamped_inside_common_support(self):
        first = RcsGrid(
            [-5e-10, 1.0, 2.0], [0.0], [10.0], ["VV"],
            rcs=np.ones((3, 1, 1, 1), dtype=np.complex128),
            units={"frequency": "GHz"},
        )
        second = RcsGrid(
            [0.0, 1.0, 2.0], [0.0], [10.0], ["VV"],
            rcs=np.ones((3, 1, 1, 1), dtype=np.complex128),
            units={"frequency": "GHz"},
        )
        azimuths, _elevations, _frequencies, _polarizations = (
            _interp_target_axes([first, second])
        )
        np.testing.assert_array_equal(azimuths, [0.0, 1.0, 2.0])
        aligned = _align_grids_for_assembly([first, second], "interp")
        for grid in aligned:
            np.testing.assert_array_equal(grid.azimuths, [0.0, 1.0, 2.0])

    def test_asy_embedding_rejects_unloadable_object_metadata(self):
        grid = _grid(1.0)
        grid.extra["placement"] = {"offset": [1.0, 2.0, 3.0]}
        with self.assertRaisesRegex(ValueError, "pickle-free .asy.*placement"):
            _grid_to_b64(grid)

    def test_visibility_column_defaults_checked_and_cascades(self) -> None:
        tree, root, branch, first, second = self._tree_with_two_leaves()
        self.assertEqual(tree.columnCount(), 4)
        self.assertEqual(tree.headerItem().text(_COLUMN_INCLUDED), "Use")
        self.assertEqual(tree.headerItem().text(_COLUMN_VISIBILITY), "Show")
        self.assertEqual(first.checkState(_COLUMN_VISIBILITY), Qt.Checked)

        changes = []
        tree.visibility_changed.connect(
            lambda item, effective: changes.append((item, effective))
        )
        tree.set_item_preview_visible(first, False)

        self.assertEqual(first.checkState(_COLUMN_VISIBILITY), Qt.Unchecked)
        self.assertEqual(second.checkState(_COLUMN_VISIBILITY), Qt.Checked)
        self.assertEqual(branch.checkState(_COLUMN_VISIBILITY), Qt.PartiallyChecked)
        self.assertEqual(root.checkState(_COLUMN_VISIBILITY), Qt.PartiallyChecked)
        self.assertFalse(tree.item_visible(first))
        self.assertTrue(tree.item_visible(second))
        self.assertEqual(changes[-1], (first, False))

        changes.clear()
        tree.set_item_preview_visible(branch, False)
        self.assertEqual(branch.checkState(_COLUMN_VISIBILITY), Qt.Unchecked)
        self.assertEqual(first.checkState(_COLUMN_VISIBILITY), Qt.Unchecked)
        self.assertEqual(second.checkState(_COLUMN_VISIBILITY), Qt.Unchecked)
        self.assertFalse(tree.item_visible(second))
        self.assertCountEqual(
            [item for item, _effective in changes], [branch, first, second]
        )
        self.assertTrue(all(not effective for _item, effective in changes))

        tree.set_item_preview_visible(branch, True)
        self.assertEqual(branch.checkState(_COLUMN_VISIBILITY), Qt.Checked)
        self.assertEqual(first.checkState(_COLUMN_VISIBILITY), Qt.Checked)
        self.assertEqual(second.checkState(_COLUMN_VISIBILITY), Qt.Checked)

        # Direct check-state changes exercise the same path as a user click.
        branch.setCheckState(_COLUMN_VISIBILITY, Qt.Unchecked)
        self.assertEqual(first.checkState(_COLUMN_VISIBILITY), Qt.Unchecked)
        self.assertEqual(second.checkState(_COLUMN_VISIBILITY), Qt.Unchecked)

    def test_preview_visibility_does_not_change_assembly_physics(self) -> None:
        tree, root, _branch, first, _second = self._tree_with_two_leaves()
        tree.set_item_preview_visible(first, False)

        result, _history = build_assembly_grid(root, axis_mode="strict")

        self.assertIsNotNone(result)
        self.assertAlmostEqual(float(result.rcs_power.item()), 9.0, places=12)

    def test_use_column_controls_build_without_changing_preview(self) -> None:
        tree, root, branch, first, second = self._tree_with_two_leaves()
        self.assertEqual(first.checkState(_COLUMN_INCLUDED), Qt.Checked)
        self.assertTrue(tree.item_included(first))

        tree.set_item_included(first, False)

        self.assertEqual(first.checkState(_COLUMN_INCLUDED), Qt.Unchecked)
        self.assertEqual(second.checkState(_COLUMN_INCLUDED), Qt.Checked)
        self.assertEqual(branch.checkState(_COLUMN_INCLUDED), Qt.PartiallyChecked)
        self.assertEqual(root.checkState(_COLUMN_INCLUDED), Qt.PartiallyChecked)
        self.assertTrue(tree.item_visible(first))
        self.assertFalse(tree.item_included(first))
        self.assertTrue(tree.item_included(second))

        result, history = build_assembly_grid(root, axis_mode="strict")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(float(result.rcs_power.item()), 4.0, places=12)
        self.assertNotIn("A", history)
        self.assertIn("B", history)

        # Show remains preview-only even after solve membership is changed.
        tree.set_item_preview_visible(second, False)
        result, _history = build_assembly_grid(root, axis_mode="strict")
        self.assertAlmostEqual(float(result.rcs_power.item()), 4.0, places=12)

    def test_use_cascades_and_round_trips_with_legacy_default(self) -> None:
        tree, root, branch, first, second = self._tree_with_two_leaves()
        branch.setCheckState(_COLUMN_INCLUDED, Qt.Unchecked)
        self.assertEqual(first.checkState(_COLUMN_INCLUDED), Qt.Unchecked)
        self.assertEqual(second.checkState(_COLUMN_INCLUDED), Qt.Unchecked)

        added_while_off = tree._make_leaf("C", _grid(3.0))
        _attach(tree, added_while_off, branch)
        self.assertEqual(
            added_while_off.checkState(_COLUMN_INCLUDED), Qt.Unchecked
        )
        self.assertIsNone(build_assembly_grid(root, axis_mode="strict")[0])

        payload = _item_to_dict(root)
        loaded = _dict_to_item(payload)
        self.assertEqual(
            loaded.child(0).checkState(_COLUMN_INCLUDED), Qt.Unchecked
        )
        self.assertFalse(AssemblyTree.item_included(loaded.child(0).child(0)))

        def strip_inclusion(node):
            node.pop("included", None)
            for child in node.get("children", []):
                strip_inclusion(child)

        strip_inclusion(payload)
        legacy = _dict_to_item(payload)
        self.assertEqual(legacy.checkState(_COLUMN_INCLUDED), Qt.Checked)
        self.assertEqual(
            legacy.child(0).child(0).checkState(_COLUMN_INCLUDED), Qt.Checked
        )

        invalid = _item_to_dict(root)
        invalid["included"] = "false"
        with self.assertRaisesRegex(ValueError, "non-boolean 'included'"):
            _dict_to_item(invalid)

    def test_duplicate_subtree_deep_copies_grid_and_trade_state(self) -> None:
        tree, _root, branch, first, _second = self._tree_with_two_leaves()
        original_grid = first.data(0, _ROLE_GRID)
        original_grid.extra["placement"] = {
            "offset": np.asarray([1.0, 2.0, 3.0])
        }
        tree.set_item_included(first, False)

        duplicate = tree.duplicate_response_subtree(branch)
        duplicate_first = duplicate.child(0)
        duplicate_grid = duplicate_first.data(0, _ROLE_GRID)

        self.assertEqual(duplicate.text(0), "Fasteners Copy")
        self.assertIsNot(duplicate_grid, original_grid)
        self.assertFalse(
            np.shares_memory(duplicate_grid.rcs_power, original_grid.rcs_power)
        )
        self.assertFalse(
            np.shares_memory(
                duplicate_grid.extra["placement"]["offset"],
                original_grid.extra["placement"]["offset"],
            )
        )
        self.assertEqual(
            duplicate_first.checkState(_COLUMN_INCLUDED), Qt.Unchecked
        )

        duplicate_grid.rcs_power[...] = 81.0
        duplicate_grid.extra["placement"]["offset"][0] = 99.0
        self.assertAlmostEqual(float(original_grid.rcs_power.item()), 1.0)
        self.assertEqual(original_grid.extra["placement"]["offset"][0], 1.0)

        tree.set_item_included(duplicate, False)
        self.assertEqual(branch.checkState(_COLUMN_INCLUDED), Qt.PartiallyChecked)
        self.assertEqual(duplicate.checkState(_COLUMN_INCLUDED), Qt.Unchecked)

    def test_show_all_checkbox_controls_every_top_level_subtree(self) -> None:
        panel = AssemblyTreePanel()
        first = panel.tree._make_node("First", _TYPE_ROOT, edit=False)
        second = panel.tree._make_node("Second", _TYPE_ROOT, edit=False)

        panel.tree.set_item_preview_visible(first, False)
        self.assertEqual(panel.chk_show_all.checkState(), Qt.PartiallyChecked)

        panel.chk_show_all.setChecked(False)
        self.assertEqual(first.checkState(_COLUMN_VISIBILITY), Qt.Unchecked)
        self.assertEqual(second.checkState(_COLUMN_VISIBILITY), Qt.Unchecked)
        self.assertEqual(panel.chk_show_all.checkState(), Qt.Unchecked)

        panel.chk_show_all.setChecked(True)
        self.assertEqual(first.checkState(_COLUMN_VISIBILITY), Qt.Checked)
        self.assertEqual(second.checkState(_COLUMN_VISIBILITY), Qt.Checked)
        self.assertEqual(panel.chk_show_all.checkState(), Qt.Checked)

    def test_visibility_round_trip_and_legacy_default(self) -> None:
        tree, root, branch, first, second = self._tree_with_two_leaves()
        tree.set_item_preview_visible(first, False)
        payload = _item_to_dict(root)

        loaded = _dict_to_item(payload)
        loaded_branch = loaded.child(0)
        self.assertEqual(
            loaded_branch.checkState(_COLUMN_VISIBILITY), Qt.PartiallyChecked
        )
        self.assertEqual(
            loaded_branch.child(0).checkState(_COLUMN_VISIBILITY), Qt.Unchecked
        )
        self.assertEqual(
            loaded_branch.child(1).checkState(_COLUMN_VISIBILITY), Qt.Checked
        )

        def strip_visibility(node):
            node.pop("visible", None)
            for child in node.get("children", []):
                strip_visibility(child)

        strip_visibility(payload)
        legacy = _dict_to_item(payload)
        legacy_branch = legacy.child(0)
        self.assertEqual(legacy.checkState(_COLUMN_VISIBILITY), Qt.Checked)
        self.assertEqual(
            legacy_branch.checkState(_COLUMN_VISIBILITY), Qt.Checked
        )
        self.assertEqual(
            legacy_branch.child(0).checkState(_COLUMN_VISIBILITY), Qt.Checked
        )
        self.assertEqual(
            legacy_branch.child(1).checkState(_COLUMN_VISIBILITY), Qt.Checked
        )


if __name__ == "__main__":
    unittest.main()
