"""Exercise the shipped editor without editable-checkout import fallbacks."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest


class InstalledWheelTests(unittest.TestCase):
    def test_installed_wheel_opens_and_edits_point_placements(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, installed, wheels = root/"source", root/"installed", root/"wheels"
            source.mkdir()
            for name in ("pyproject.toml", "README.md"):
                shutil.copy2(repo/name, source/name)
            shutil.copytree(repo/"GRIM_Revised_2", source/"GRIM_Revised_2",
                ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "*.grim", "*.png", "*.pdf", "*.mp4"))
            def run(args):
                result = subprocess.run(args, cwd=root, text=True, capture_output=True,
                                        timeout=120, env=dict(os.environ, QT_QPA_PLATFORM="offscreen"))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result
            run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                 "--no-index", "--wheel-dir", str(wheels), str(source)])
            wheel = next(wheels.glob("*.whl"))
            run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile",
                 "--no-index", "--target", str(installed), str(wheel)])
            # -I alone still executes editable-install .pth files via site.
            # -S prevents that; add dependency directories explicitly so Qt is
            # available while no repository or editable import hook is loaded.
            dependencies = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
            script = '''
import json, sys
from pathlib import Path
installed = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(installed)] + json.loads(sys.argv[2])
import assembly_placement_editor as editor_module
from feature_assembly_panel import POINT_PLACEMENT_COLUMNS
from PySide6.QtWidgets import QApplication
assert Path(editor_module.__file__).resolve().parent == installed
# Exercise extracted modules from the wheel, with checkout imports disabled.
import importlib
for name in ('grim_backend.datasets.audit', 'grim_backend.io.native', 'grim_backend.io.cst', 'grim_backend.io.sentri', 'grim_backend.io.pioneer',
             'grim_backend.io.out', 'grim_backend.io.ptm', 'grim_backend.io.xpatch', 'dataset_jobs', 'dataset_dialogs',
             'grim_backend.io.batch', 'feature_assembly_model',
             'feature_assembly_recipe', 'feature_assembly_values', 'plot_modes.isar_render'):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(installed), name

import pkgutil
import grim_backend
assert (Path(grim_backend.__file__).parent / 'README.md').is_file()
import grim_dataset
import grim_headless
import grim_python
from grim_backend.datasets.grid import RcsGrid
from grim_backend.io.loaders import load_dataset
for item in pkgutil.walk_packages(grim_backend.__path__, 'grim_backend.'):
    module = importlib.import_module(item.name)
    assert Path(module.__file__).resolve().is_relative_to(installed), item.name
assert grim_dataset.RcsGrid is RcsGrid
assert grim_headless.load_dataset is load_dataset
import numpy as np
grid = RcsGrid([0.0], [0.0], [1.0], ['VV'],
               rcs_power=np.ones((1, 1, 1, 1)), rcs_phase=np.zeros((1, 1, 1, 1)))
import tempfile
with tempfile.TemporaryDirectory() as temp:
    path = Path(temp) / 'roundtrip.grim'
    grim_python.save_dataset_batch([(grid, path)])
    loaded = load_dataset(path)
    np.testing.assert_array_equal(loaded.rcs, grid.rcs)

app = QApplication([])
editor = editor_module.PlacementEditor('point', columns=POINT_PLACEMENT_COLUMNS, units='meters')
editor.change([['p','f','0','0','0','0','0','1','1','0','0']])
editor.table.selectRow(0)
editor.duplicate()
assert len(editor.rows()) == 2
assert editor.rows()[0][0] != editor.rows()[1][0]
editor.undo()
assert len(editor.rows()) == 1
editor._saved_rows = editor.rows()
editor.reject()
editor.deleteLater()
app.processEvents()
'''
            run([sys.executable, "-I", "-S", "-c", script, str(installed), json.dumps(dependencies)])


if __name__ == "__main__":
    unittest.main()
