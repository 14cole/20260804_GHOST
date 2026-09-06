"""Regression checks for headless services and inherited format entrypoints."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from grim_dataset import RcsGrid


class ModuleBoundaryTests(unittest.TestCase):
    def test_numerical_and_form_services_import_without_qt(self):
        root = Path(__file__).resolve().parents[1]
        script = """
import importlib.abc
from pathlib import Path
import sys
root = Path(sys.argv[1])
sys.path[:0] = [str(root / 'GRIM_Revised_2'),
               str(root / 'tools' / 'GHOST' / 'Backend'),
               str(root / 'tools' / 'FREDDY')]
class RejectQt(importlib.abc.MetaPathFinder):
    attempts = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('PySide6', 'PyQt')):
            self.attempts.append(fullname)
            raise ImportError('Qt is deliberately unavailable in this process')
guard = RejectQt()
sys.meta_path.insert(0, guard)
import grim_dataset
import feature_assembly_model
import feature_assembly_recipe
import ibc.design_search
import rcs_solver
from plot_modes import isar_mode
assert not guard.attempts, guard.attempts
assert 'ibc.ui' not in sys.modules
assert 'feature_assembly_panel' not in sys.modules
# Public annotation introspection must work after implementation movement.
import typing
from feature_assembly_values import FeatureAssemblyValues, LoadedFeatureAssemblyRecipe
assert typing.get_type_hints(feature_assembly_recipe.feature_assembly_recipe_payload)['values'] is FeatureAssemblyValues
assert typing.get_type_hints(feature_assembly_recipe.read_feature_assembly_recipe)['return'] is LoadedFeatureAssemblyRecipe
assert typing.get_type_hints(rcs_solver._assemble_linear_operator_matrices)['mesh'] is rcs_solver.LinearMesh
"""
        result = subprocess.run(
            [sys.executable, '-c', script, str(root)],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_format_adapter_preserves_subclass_and_complex_data(self):
        class SpecializedGrid(RcsGrid):
            pass

        grid = SpecializedGrid(
            [0.0], [0.0], [1.0], ['VV'],
            rcs_power=np.full((1, 1, 1, 1), 4.0),
            rcs_phase=np.full((1, 1, 1, 1), 0.3),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = str(Path(temporary) / 'roundtrip.grim')
            grid.save(path)
            loaded = SpecializedGrid.load(path)
        self.assertIsInstance(loaded, SpecializedGrid)
        np.testing.assert_allclose(loaded.rcs_power, grid.rcs_power)
        np.testing.assert_allclose(loaded.rcs_phase, grid.rcs_phase)


if __name__ == '__main__':
    unittest.main()
