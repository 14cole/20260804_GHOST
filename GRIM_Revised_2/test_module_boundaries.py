"""Regression checks for headless services and inherited format entrypoints."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from grim_backend.datasets.grid import RcsGrid


class ModuleBoundaryTests(unittest.TestCase):
    def test_public_modules_share_the_backend_model_and_operations(self):
        import grim_dataset
        import grim_headless
        import grim_python
        from grim_backend.datasets import transforms
        from grim_backend.io import batch, loaders
        from grim_backend.scripting import plotting, recorder

        self.assertIs(grim_dataset.RcsGrid, RcsGrid)
        self.assertIs(grim_headless.load_dataset, loaders.load_dataset)
        self.assertIs(grim_headless.load_folder, loaders.load_folder)
        self.assertIs(grim_python.save_dataset_batch, batch.save_dataset_batch)
        self.assertIs(grim_python.plot_datasets, plotting.plot_datasets)
        self.assertIs(grim_python.PythonScriptRecorder, recorder.PythonScriptRecorder)
        for name in (
            'coherent_divide', 'convert_extrusion', 'crop_dataset', 'decimate_axis',
            'duplicate_dataset', 'join_datasets', 'medianize_azimuth', 'offset_db',
            'regrid_axis', 'shift_dataset', 'stitch_datasets', 'wedge_to_conic',
            'wrap_phase',
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(grim_python, name), getattr(transforms, name))

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
import grim_backend.datasets.audit
import grim_backend
import pkgutil
for module in pkgutil.walk_packages(grim_backend.__path__, grim_backend.__name__ + '.'):
    importlib.import_module(module.name)
import feature_assembly_model
import feature_assembly_recipe
import ibc.design_search
import ibc.mix_analysis
import ibc.search_checkpoint
import ghost_backend.twod.solver as rcs_solver
import ghost_backend.runs.config as driver_config
import ghost_backend.assembly.preparation as feature_preparation
import ghost_backend.assembly.contracts as feature_library_contracts
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
assert typing.get_type_hints(feature_preparation.capture_assembly_sources)['return'] is feature_preparation.AssemblySources
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
