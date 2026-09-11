"""Development checks catch incomplete source copies and failed subprocesses."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import verify_project
import GRIM_Backend.execution.diagnostics as grim_diagnostics


class VerificationTests(unittest.TestCase):
    def test_real_source_and_wheel_inventories_cover_imports(self):
        verify_project.check_source_inventories(Path(__file__).resolve().parents[2])

    def test_import_closure_handles_relative_deferred_and_package_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'pkg').mkdir()
            for name, source in {
                'main.py': 'import pkg.view\n', 'pkg/__init__.py': '',
                'pkg/view.py': 'from . import values\ndef run():\n from .compute import solve\n',
                'pkg/values.py': '', 'pkg/compute.py': 'def solve(): pass\n',
            }.items():
                (root / name).write_text(source, encoding='utf-8')
            self.assertEqual(verify_project.local_import_closure(root, ['main.py']),
                             {Path(name) for name in ('main.py', 'pkg/__init__.py',
                              'pkg/view.py', 'pkg/values.py', 'pkg/compute.py')})

    def test_missing_workflow_module_is_reported_by_file_check(self):
        root = Path(__file__).resolve().parents[2] / 'tools/FREDDY'
        original = Path.is_file
        with mock.patch.object(Path, 'is_file', lambda p: False if p.name == 'inverse_workflow.py' else original(p)):
            self.assertIn('ibc/inverse_workflow.py', [Path(p).as_posix() for p in
                grim_diagnostics._missing_files(root, grim_diagnostics.FREDDY_SENTINELS)])

    def test_failed_check_propagates_exit_and_output(self):
        with mock.patch.object(verify_project.subprocess, 'run', return_value=
                               SimpleNamespace(returncode=2, stdout='FAIL: important_case\n' + 'x' * 20000 + '\nspecific failure')):
            with self.assertRaisesRegex(RuntimeError, 'specific failure') as raised:
                verify_project.run_suite('example', Path('.'), ('check.py',))
            self.assertIn('important_case', str(raised.exception))

    def test_full_mode_includes_standalone_integrations(self):
        commands = [args for _, _, args in verify_project.development_suites(Path('.'), 'full')]
        for name in ('test_hpc_scheduling.py', 'test_local_drivers.py', 'test_source_is_ascii.py'):
            self.assertIn((f'tests/{name}',), commands)


if __name__ == '__main__':
    unittest.main()
