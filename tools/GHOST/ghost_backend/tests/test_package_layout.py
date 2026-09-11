"""Package relocation, source identity, and standalone deployment checks."""
import os
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND.parent))
from ghost_backend.execution.provenance import backend_source_inventory, backend_source_fingerprint


class PackageLayoutTests(unittest.TestCase):
    def test_top_level_contains_launcher_guides_and_categorized_backend(self):
        ghost = BACKEND.parent
        entries = list(ghost.iterdir())
        self.assertEqual({p.name for p in entries if p.is_dir()}, {'ghost_backend'})
        self.assertTrue(all(p.suffix in ('.bat', '.md') for p in entries if p.is_file()))
        self.assertFalse((ghost / '1c_build_deltas').exists())
        self.assertFalse((BACKEND / 'ghost_backend').exists())
        self.assertFalse((BACKEND / 'solver').exists())
        self.assertEqual({p.name for p in BACKEND.iterdir() if p.is_file()}, {
            'run_gui.py', 'run_local_monostatic.py', 'run_local_bor.py',
            'run_hpc_monostatic.py', 'run_hpc_bor_monostatic.py'})
        for category in ('twod', 'bor', 'data_tools', 'geometry', 'validation', 'tests'):
            self.assertTrue((BACKEND / category).is_dir())
        launcher = (ghost / 'Launch_GHOST_GUI.bat').read_text()
        self.assertIn('set \"GHOST_GUI=%~dp0ghost_backend\\run_gui.py\"', launcher)
        for line in launcher.splitlines():
            if '--check' in line or line.strip().startswith('start '):
                self.assertIn('\"%GHOST_GUI%\"', line)

    @unittest.skipUnless(importlib.util.find_spec('PySide6'), 'PySide6 desktop dependency unavailable')
    def test_dataset_cli_and_gui_check_run_from_an_unrelated_directory(self):
        ghost = BACKEND.parent
        with tempfile.TemporaryDirectory() as temporary:
            for script, args in ((BACKEND / 'run_gui.py', ['--check']),
                                 (ghost / 'ghost_backend/data_tools/run_cli.py', ['subtract', '--help'])):
                result = subprocess.run([sys.executable, str(script), *args], cwd=temporary,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_nested_sources_are_distinct_and_invalidate_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative, value in [('driver.py', 'driver'),
                                    ('twod/solver.py', 'two'),
                                    ('bor/solver.py', 'bor')]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
            records = backend_source_inventory(str(root))
            self.assertEqual(set(records), {
                'ghost_backend/driver.py', 'ghost_backend/twod/solver.py',
                'ghost_backend/bor/solver.py',
            })
            before = backend_source_fingerprint(str(root))
            (root / 'bor/solver.py').write_text('changed')
            self.assertNotEqual(before, backend_source_fingerprint(str(root)))
            before = backend_source_fingerprint(str(root))
            for relative in ('tests/check.py', 'data_tools/tool.py', 'results/output.py',
                             'geometry/geometries/example.py', 'validation/study/generate.py'):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('support data')
            self.assertEqual(before, backend_source_fingerprint(str(root)))

    def test_copied_backend_imports_without_checkout_or_unused_lu_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "ghost_backend"
            shutil.copytree(str(BACKEND), str(copied),
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            self.assertFalse((copied / 'cpu_checked_lu.py').exists())
            script = '''
import importlib, pathlib, pkgutil, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root.parent))
class NoGui:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('PySide6','PySide2','PyQt5','PyQt6'):
            raise RuntimeError('Unexpected GUI dependency: ' + fullname)
sys.meta_path.insert(0, NoGui())
import ghost_backend
for info in pkgutil.walk_packages(ghost_backend.__path__, 'ghost_backend.'):
    name = info.name
    if name == 'ghost_backend.ui' or name.startswith('ghost_backend.ui.') or name == 'ghost_backend.run_gui':
        continue
    module = importlib.import_module(name)
    assert root in pathlib.Path(module.__file__).resolve().parents, name
import ghost_backend.twod.solver as rcs_solver
import ghost_backend.bor.solver as bor_solver
import ghost_backend.assembly.workflow as feature_workflow
from ghost_backend.twod import solver
from ghost_backend.bor import solver as bor
from ghost_backend.assembly import workflow
assert rcs_solver is solver
assert bor_solver is bor
assert feature_workflow is workflow
import run_local_monostatic, run_local_bor, run_hpc_monostatic, run_hpc_bor_monostatic
assert run_hpc_monostatic._solver_source_records()[0] == str(root)
assert 'cpu_checked_lu' not in sys.modules
'''
            result = subprocess.run([sys.executable, '-I', '-c', script, str(copied)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, timeout=60,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2'))
            self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == '__main__':
    unittest.main()
