"""Headless runtime checks that run unchanged on Python 3.6.8 and desktop Python."""
import os
from pathlib import Path
import pickle
import sys
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

BACKEND = Path(__file__).resolve().parents[1] / 'Backend'
sys.path.insert(0, str(BACKEND))
import ghost_runtime


class HpcRuntimeTests(unittest.TestCase):
    def test_headless_modules_import_without_gui(self):
        modules = (
            'run_hpc_monostatic', 'run_hpc_bor_monostatic',
            'run_local_monostatic', 'run_local_bor', 'hpc_bundle',
            'rcs_solver', 'bor_solver', 'bor_dispatch', 'bor_streaming',
            'thin_sheet', 'feature_workflow', 'feature_preparation',
            'feature_library_contracts', 'assembly_workload', 'mesh_quality',
            'create_feature_manifest', 'build_bor_stream_kernel',
        )
        script = (
            "import sys, importlib\n"
            "class NoGui:\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname.split('.')[0] in ('PySide6', 'PyQt5', 'PyQt6'):\n"
            "            raise RuntimeError('GUI import: ' + fullname)\n"
            "sys.meta_path.insert(0, NoGui())\n"
            "for name in {!r}: importlib.import_module(name)\n".format(modules)
        )
        result = subprocess.run(
            [sys.executable, '-c', script],
            env={**os.environ, 'PYTHONPATH': str(BACKEND)},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_dataclasses_defaults_frozen_replace_asdict_and_pickle(self):
        from driver_config import LoadedConfiguration
        from assembly_workload import AssemblyWorkload
        if sys.version_info < (3, 7):
            self.assertEqual(ghost_runtime.dataclass.__module__, '_ghost_dataclasses')
        record = LoadedConfiguration(Path('config.json'), 'abc')
        with self.assertRaises(AttributeError):
            record.sha256 = 'changed'
        replacement = ghost_runtime.replace(record, sha256='def')
        self.assertEqual(ghost_runtime.asdict(replacement)['sha256'], 'def')
        self.assertEqual(pickle.loads(pickle.dumps(replacement)), replacement)
        workload = AssemblyWorkload(available=True)
        self.assertEqual(workload.as_dict()['review_reasons'], ())

    def test_lu_without_new_scipy_warning_export_preserves_double_fallback(self):
        # A fresh process reproduces the import failure even if the main test
        # process already imported refined_lu through another solver module.
        script = """
import numpy as np
import scipy.linalg as linalg
if hasattr(linalg, 'LinAlgWarning'):
    del linalg.LinAlgWarning
from refined_lu import RefinedLU, linear_precision
import rcs_solver as rcs
a = np.array([[3+1j, 1-2j], [2+0j, 5-1j]])
b = np.array([1+2j, -3+1j])
np.testing.assert_allclose(RefinedLU(a).solve(b), np.linalg.solve(a, b),
                           rtol=1e-10, atol=1e-11)
# This matrix becomes singular in complex64 but remains solvable in double.
a = np.array([[1, 1], [1, 1+1e-10]], dtype=complex)
b = np.array([2, 2+1e-10], dtype=complex)
rcs._reset_dense_backend_telemetry()
with linear_precision('mixed'):
    actual = rcs._solve_dense_system(a, b)
np.testing.assert_allclose(actual, np.linalg.solve(a, b), rtol=1e-10)
assert any('fell back' in reason for reason in
           rcs._dense_backend_summary()['dense_fallback_reasons'])
"""
        result = subprocess.run(
            [sys.executable, '-c', script],
            env={**os.environ, 'PYTHONPATH': str(BACKEND)},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_lu_still_rejects_old_scipy_runtime_warning(self):
        import numpy as np
        import refined_lu
        import warnings
        def old_factor(_matrix):
            warnings.warn('Singular matrix.', RuntimeWarning)
            self.fail('A singular single-precision factorization was accepted')
        with mock.patch.object(refined_lu, 'lu_factor', side_effect=old_factor):
            with self.assertRaisesRegex(RuntimeWarning, 'Singular matrix'):
                refined_lu.RefinedLU(np.eye(2))

    def test_solver_scope_restores_exceptions_and_isolates_threads(self):
        # Exercise both the current interpreter's implementation and the
        # fallback, so desktop CI can protect the cluster's thread semantics.
        for context_type in (ghost_runtime.ContextVar, None):
            with self.subTest(context=context_type), mock.patch.object(
                    ghost_runtime, 'ContextVar', context_type):
                setting = ghost_runtime.ScopedValue('test', 'default')
                values = []
                with setting.override('outer'):
                    with self.assertRaisesRegex(ValueError, 'nested'):
                        with setting.override('inner'):
                            self.assertEqual(setting.get(), 'inner')
                            raise ValueError('nested')
                    self.assertEqual(setting.get(), 'outer')
                    def worker():
                        values.append(setting.get())
                        with setting.override('thread'):
                            values.append(setting.get())
                        values.append(setting.get())
                    thread = threading.Thread(target=worker)
                    thread.start()
                    thread.join(timeout=5)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(setting.get(), 'outer')
                self.assertEqual(setting.get(), 'default')
                self.assertEqual(values, ['default', 'thread', 'default'])

    def test_failed_profiled_solve_restores_precision_and_metrics(self):
        from refined_lu import linear_precision, requested_precision
        from solver_metrics import active_metrics, profiled_solve
        @profiled_solve
        def failed_solve():
            self.assertIsNotNone(active_metrics())
            with linear_precision('mixed'):
                self.assertEqual(requested_precision(), 'mixed')
                raise ValueError('solve failed')
        with self.assertRaisesRegex(ValueError, 'solve failed'):
            failed_solve()
        self.assertIsNone(active_metrics())
        self.assertEqual(requested_precision(), 'double')

    def test_file_adapters_preserve_errors_and_lf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'text.txt'
            ghost_runtime.write_text_lf(path, 'one\ntwo\n')
            self.assertEqual(path.read_bytes(), b'one\ntwo\n')
            ghost_runtime.unlink_if_exists(path)
            ghost_runtime.unlink_if_exists(path)
            with mock.patch.object(Path, 'unlink', side_effect=PermissionError('denied')):
                with self.assertRaises(PermissionError):
                    ghost_runtime.unlink_if_exists(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
