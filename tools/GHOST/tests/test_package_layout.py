"""Package relocation, source identity, and standalone deployment checks."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

BACKEND = Path(__file__).resolve().parents[1] / 'Backend'
sys.path.insert(0, str(BACKEND))
from ghost_backend.execution.provenance import backend_source_inventory, backend_source_fingerprint


class PackageLayoutTests(unittest.TestCase):
    def test_nested_sources_are_distinct_and_invalidate_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative, value in [('driver.py', 'driver'),
                                    ('ghost_backend/twod/solver.py', 'two'),
                                    ('ghost_backend/bor/solver.py', 'bor')]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
            records = backend_source_inventory(str(root))
            self.assertEqual(set(records), {
                'Backend/driver.py', 'Backend/ghost_backend/twod/solver.py',
                'Backend/ghost_backend/bor/solver.py',
            })
            before = backend_source_fingerprint(str(root))
            (root / 'ghost_backend/bor/solver.py').write_text('changed')
            self.assertNotEqual(before, backend_source_fingerprint(str(root)))

    def test_copied_backend_imports_without_checkout_or_unused_lu_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / 'Backend'
            shutil.copytree(str(BACKEND), str(copied),
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            # Test removal only in the owned temporary copy.
            (copied / 'cpu_checked_lu.py').unlink()
            script = '''
import importlib, pathlib, pkgutil, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
class NoGui:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('PySide6','PySide2','PyQt5','PyQt6'):
            raise RuntimeError('Unexpected GUI dependency: ' + fullname)
sys.meta_path.insert(0, NoGui())
import ghost_backend
for info in pkgutil.walk_packages(ghost_backend.__path__, 'ghost_backend.'):
    name = info.name
    if name == 'ghost_backend.ui' or name.startswith('ghost_backend.ui.'):
        continue
    module = importlib.import_module(name)
    assert root in pathlib.Path(module.__file__).resolve().parents, name
import rcs_solver, bor_solver, feature_workflow
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
