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

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND.parent))
import ghost_backend.execution.runtime as ghost_runtime


class HpcRuntimeTests(unittest.TestCase):
    def test_environment_checker_accepts_the_installed_test_stack(self):
        result = subprocess.run(
            [sys.executable, str(BACKEND / 'hpc/check_environment.py')],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('PASS: driver/solver imports and complex LU', result.stdout)

    def test_packed_visibility_preserves_every_bit_and_empty_shapes(self):
        import numpy as np
        from ghost_backend.geometry.occlusion import PackedVisibility
        packed = np.arange(256, dtype=np.uint8).reshape(256, 1)
        visibility = PackedVisibility(packed, n_points=256, n_directions=8)
        expected = np.array([[bool(byte & (1 << bit)) for bit in range(8)]
                             for byte in range(256)])
        np.testing.assert_array_equal(visibility.to_dense(), expected)
        for byte in range(256):
            np.testing.assert_array_equal(visibility.row(byte).to_dense(), expected[byte])
        partial = PackedVisibility(np.array([[129, 1]], dtype=np.uint8),
                                   n_points=1, n_directions=9)
        np.testing.assert_array_equal(partial.to_dense(),
                                      [[True, False, False, False, False, False, False, True, True]])
        for points, directions in ((0, 9), (2, 0), (0, 0)):
            empty = PackedVisibility(np.zeros((points, (directions + 7) // 8), dtype=np.uint8),
                                     n_points=points, n_directions=directions)
            self.assertEqual(empty.to_dense().shape, (points, directions))
            if points:
                self.assertEqual(empty.row(0).to_dense().shape, (directions,))

    def test_packed_visibility_cannot_be_changed_after_review(self):
        import numpy as np
        from ghost_backend.geometry.occlusion import PackedVisibility
        source = np.array([[129, 1]], dtype=np.uint8)
        visibility = PackedVisibility(source, n_points=1, n_directions=9)
        source[:] = 0
        self.assertTrue(visibility.row(0)[8])
        for array in (visibility._packed, visibility.row(0)._packed):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)
            with self.assertRaises(ValueError):
                array[...] = 0

    def test_headless_modules_import_without_gui(self):
        modules = (
            'ghost_backend.run_hpc_monostatic', 'ghost_backend.run_hpc_bor_monostatic',
            'ghost_backend.run_local_monostatic', 'ghost_backend.run_local_bor', 'ghost_backend.hpc.bundle',
            'ghost_backend.twod.solver', 'ghost_backend.bor.solver', 'ghost_backend.bor.dispatch', 'ghost_backend.bor.streaming',
            'ghost_backend.twod.formulations.thin_layer', 'ghost_backend.assembly.workflow', 'ghost_backend.assembly.preparation',
            'ghost_backend.linalg.dense', 'ghost_backend.twod.fields', 'ghost_backend.twod.assembly.session', 'ghost_backend.twod.assembly.mass',
            'ghost_backend.twod.formulations.dielectric', 'ghost_backend.twod.formulations.robin', 'ghost_backend.twod.formulations.sheet', 'ghost_backend.twod.formulations.regions',
            'ghost_backend.assembly.contracts', 'ghost_backend.assembly.workload', 'ghost_backend.geometry.quality',
            'ghost_backend.assembly.create_feature_manifest', 'ghost_backend.bor.native.build_kernel',
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
            env={**os.environ, 'PYTHONPATH': str(BACKEND.parent)},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_dataclasses_defaults_frozen_replace_asdict_and_pickle(self):
        from ghost_backend.runs.config import LoadedConfiguration
        from ghost_backend.assembly.workload import AssemblyWorkload
        if sys.version_info < (3, 7):
            self.assertEqual(ghost_runtime.dataclass.__module__, 'ghost_backend.execution._dataclasses')
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
from ghost_backend.linalg.refined_lu import RefinedLU, linear_precision
import ghost_backend.twod.solver as rcs
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
            env={**os.environ, 'PYTHONPATH': str(BACKEND.parent)},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_lu_still_rejects_old_scipy_runtime_warning(self):
        import numpy as np
        import ghost_backend.linalg.refined_lu as refined_lu
        import warnings
        def old_factor(_matrix, overwrite_a=False, check_finite=True):
            self.assertTrue(overwrite_a)
            self.assertFalse(check_finite)
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
        from ghost_backend.linalg.refined_lu import linear_precision, requested_precision
        from ghost_backend.execution.metrics import active_metrics, profiled_solve
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
