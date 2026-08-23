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
    _COLUMN_VISIBILITY,
    _TYPE_BRANCH,
    _TYPE_ROOT,
    _attach,
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

    def test_visibility_column_defaults_checked_and_cascades(self) -> None:
        tree, root, branch, first, second = self._tree_with_two_leaves()
        self.assertEqual(tree.columnCount(), 3)
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
