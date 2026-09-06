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
for name in ('grim_dataset_audit', 'grim_format_io', 'grim_cst_io', 'grim_sentri_io', 'grim_pio_io',
             'grim_legacy_io', 'dataset_jobs', 'dataset_dialogs',
             'dataset_publication', 'feature_assembly_model',
             'feature_assembly_recipe', 'feature_assembly_values', 'plot_modes.isar_render'):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(installed), name

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
