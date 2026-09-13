"""Exercise native thread limits with the installed package unavailable."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BACKEND = Path(__file__).resolve().parents[1]
ISOLATE = '''
import importlib.abc
from pathlib import Path
import sys
backend = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(backend.parent), str(backend / 'tests')]
class RejectInstalledThreadControl(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'threadpoolctl' or fullname.startswith('threadpoolctl.'):
            raise ModuleNotFoundError('Installed threadpoolctl is unavailable')
for name in tuple(sys.modules):
    if name == 'threadpoolctl' or name.startswith('threadpoolctl.'):
        del sys.modules[name]
sys.meta_path.insert(0, RejectInstalledThreadControl())
try:
    import threadpoolctl
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('The installed package must be unavailable')
'''


class BundledThreadControlTests(unittest.TestCase):
    def run_isolated(self, script):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, '-I', '-c', ISOLATE + script, str(BACKEND)],
                cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, timeout=120,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_native_limits_and_restoration_without_installed_package(self):
        output = self.run_isolated('''
import json
import numpy as np
import scipy.linalg as la
from ghost_backend.execution import thread_control
from ghost_backend.execution.options import execution_scope
assert Path(thread_control.implementation.__file__).resolve().parent == backend / 'execution/thread_control'
def counts():
    return {row['filepath']: row['num_threads'] for row in thread_control.threadpool_info()
            if row['user_api'] == 'blas'}
before = counts()
assert before, 'No BLAS runtime was detected'
a = np.array([[3+1j, 1-2j], [2+0j, 5-1j]])
b = np.array([1+2j, -3+1j])
with execution_scope({'blas_threads': 2}, limit_blas=True):
    assert set(counts().values()) == {2}, counts()
    with execution_scope({'blas_threads': 1}, limit_blas=True):
        assert set(counts().values()) == {1}, counts()
        x = la.lu_solve(la.lu_factor(a), b)
        np.testing.assert_allclose(a @ x, b, atol=1e-12)
    assert set(counts().values()) == {2}
assert counts() == before
try:
    with execution_scope({'blas_threads': 1}, limit_blas=True):
        raise RuntimeError('abort')
except RuntimeError:
    pass
assert counts() == before
assert 'threadpoolctl' not in sys.modules
print(json.dumps({'version': thread_control.__version__, 'libraries': len(before)}))
''')
        expected = '3.6.0' if sys.version_info >= (3, 9) else '2.2.0'
        self.assertEqual(json.loads(output)['version'], expected)

    def test_pec_and_mixed_sweeps_without_installed_package(self):
        self.run_isolated('''
import numpy as np
from ghost_backend.twod import solver
from test_experimental_cpu import fixture, fields
for material in ('pec', 'mixed'):
    results = []
    for mode in ('dense', 'compressed'):
        result = solver.solve_monostatic_rcs_2d_survey(
            fixture(material, 12), [.6], [0., 90., 180., 270., 360.],
            geometry_units='meters', solver_method='experimental_cpu',
            execution_options={'factorization': mode, 'blas_threads': 2,
                               'assembly_threads': 2, 'compressed_storage_mib': 128})
        results.append(result)
        assert result['metadata']['execution_threads']['blas'] == 2
    for pol in ('VV', 'HH'):
        reference, actual = fields(results[0], pol), fields(results[1], pol)
        np.testing.assert_allclose(actual, reference, rtol=1e-8, atol=1e-10)
assert 'threadpoolctl' not in sys.modules
''')


if __name__ == '__main__':
    unittest.main()
