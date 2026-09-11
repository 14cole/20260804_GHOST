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
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, installed, wheels = root/"source", root/"installed", root/"wheels"
            source.mkdir()
            for name in ("pyproject.toml", "README.md"):
                shutil.copy2(repo/name, source/name)
            shutil.copytree(repo/"GRIM_Backend", source/"GRIM_Backend",
                ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "*.grim", "*.pdf", "*.mp4"))
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
import GRIM_Backend.assembly.placement_editor as editor_module
from GRIM_Backend.assembly.panel import POINT_PLACEMENT_COLUMNS
from PySide6.QtWidgets import QApplication
assert Path(editor_module.__file__).resolve() == installed / "GRIM_Backend/assembly/placement_editor.py"
# Exercise extracted modules from the wheel, with checkout imports disabled.
import importlib
for name in ('GRIM_Backend.datasets.audit', 'GRIM_Backend.io.native', 'GRIM_Backend.io.cst', 'GRIM_Backend.io.sentri', 'GRIM_Backend.io.pioneer',
             'GRIM_Backend.io.out', 'GRIM_Backend.io.ptm', 'GRIM_Backend.io.xpatch', 'GRIM_Backend.execution.dataset_jobs', 'GRIM_Backend.ui.dataset_dialogs',
             'GRIM_Backend.io.batch', 'GRIM_Backend.assembly.model',
             'GRIM_Backend.assembly.recipe', 'GRIM_Backend.assembly.values', 'GRIM_Backend.plotting.modes.isar_render'):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(installed), name

import pkgutil
import GRIM_Backend
assert (Path(next(iter(GRIM_Backend.__path__))) / 'docs/BACKEND.md').is_file()
import GRIM_Backend.datasets.api as grim_dataset
import GRIM_Backend.scripting.api as grim_headless
import GRIM_Backend.scripting.workspace as grim_python
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.io.loaders import load_dataset
for item in pkgutil.walk_packages(GRIM_Backend.__path__, 'GRIM_Backend.'):
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
