"""Development checks shared with the mandatory release acceptance gate.

Run ``python verify_project.py --mode quick`` during development, or use
``--mode full`` for every acceptance suite without building a release.
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def local_import_closure(root: Path, entrypoints) -> set[Path]:
    """Find existing local Python dependencies without importing application code.

    Include deferred and optional imports conservatively. Dynamic imports and
    data assets still need explicit inventory entries.
    """
    root = root.resolve()
    pending = [Path(path) for path in entrypoints]
    found = set()

    def candidates(parts):
        if parts and parts[0] == root.name:
            parts = parts[1:]
        for end in range(1, len(parts) + 1):
            package = Path(*parts[:end]) / '__init__.py'
            if (root / package).is_file():
                pending.append(package)
        if parts:
            module = Path(*parts).with_suffix('.py')
            if (root / module).is_file():
                pending.append(module)

    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError(f'Import path escapes source root: {relative}')
        found.add(relative)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    candidates(alias.name.split('.'))
            elif isinstance(node, ast.ImportFrom):
                package = list(relative.parent.parts)
                if node.level:
                    if node.level > len(package) + 1:
                        continue
                    package = package[:len(package) - node.level + 1]
                else:
                    package = []
                parts = package + (node.module.split('.') if node.module else [])
                candidates(parts)
                for alias in node.names:
                    if alias.name != '*':
                        candidates(parts + [alias.name])
    return found


def check_source_inventories(root: Path) -> None:
    from GRIM_Backend.execution import diagnostics
    import tomllib

    groups = (
        ('GRIM', root / 'GRIM_Backend', diagnostics.GRIM_STARTUP_FILES),
        ('GHOST', root / 'tools/GHOST/ghost_backend', diagnostics.GHOST_SENTINELS),
        ('FREDDY', root / 'tools/FREDDY', diagnostics.FREDDY_SENTINELS),
    )
    failures = []
    for name, directory, inventory in groups:
        declared = {Path(path) for path in inventory}
        if len(declared) != len(inventory):
            failures.append(f'{name}: duplicate inventory entries')
        absent = sorted(str(path) for path in declared if not (directory / path).is_file())
        if absent:
            failures.append(f'{name}: missing source files: {", ".join(absent)}')
            continue
        omitted = local_import_closure(directory, declared) - declared
        if omitted:
            failures.append(f'{name}: imports absent from inventory: ' +
                            ', '.join(sorted(path.as_posix() for path in omitted)))
    project = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))
    from setuptools import find_namespace_packages
    config = project['tool']['setuptools']['packages']['find']
    packages = set(find_namespace_packages(str(root), include=config['include'], exclude=config['exclude']))
    for relative in diagnostics.GRIM_STARTUP_FILES:
        package = '.'.join(('GRIM_Backend', *Path(relative).parent.parts))
        if package not in packages:
            failures.append('Packaging: runtime package absent from wheel: ' + package)
    if failures:
        raise ValueError('\n'.join(failures))


def acceptance_suites(source_root: Path):
    """One suite inventory used by both development and release verification."""
    discover = ('-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test*.py', '-v')
    return (
        ('source inventory checks', source_root, ('verify_project.py', '--inventory-only')),
        ('UTF-8 cleaner tests', source_root,
         ('-W', 'error', '-m', 'unittest', '-v', 'test_clean_utf8.py')),
        ('offline wheelhouse tests', source_root / 'requirements',
         ('-m', 'unittest', 'discover', '-s', '.', '-p', 'test*.py', '-v')),
        ('GRIM tests', source_root,
         ('-m', 'unittest', 'discover', '-s', 'GRIM_Backend/tests', '-p', 'test*.py', '-v')),
        ('GHOST tests', source_root / 'tools/GHOST/ghost_backend', discover),
        ('GHOST CEM tools tests', source_root / 'tools/GHOST/ghost_backend/data_tools', discover),
        ('GHOST HPC scheduling integration', source_root / 'tools/GHOST/ghost_backend',
         ('tests/test_hpc_scheduling.py',)),
        ('GHOST local-driver integration', source_root / 'tools/GHOST/ghost_backend',
         ('tests/test_local_drivers.py',)),
        ('GHOST ASCII-transfer compatibility', source_root / 'tools/GHOST/ghost_backend',
         ('tests/test_source_is_ascii.py',)),
        ('FREDDY tests', source_root / 'tools/FREDDY', discover),
    )


def development_suites(root: Path, mode: str):
    startup = ('startup diagnostics', root, ('-m', 'GRIM_Backend.execution.diagnostics'))
    if mode == 'full':
        return (startup, *acceptance_suites(root))
    return (
        startup,
        acceptance_suites(root)[0],
        ('development smoke tests', root,
         ('-m', 'unittest', '-v', 'GRIM_Backend.tests.test_module_boundaries', 'GRIM_Backend.tests.test_verification',
          'GRIM_Backend.tests.test_grim_diagnostics', 'GRIM_Backend.tests.test_wheel_installation', 'GRIM_Backend.tests.test_release_builder')),
        ('GHOST ASCII-transfer compatibility', root / 'tools/GHOST/ghost_backend',
         ('tests/test_source_is_ascii.py',)),
    )


def run_suite(name, cwd, arguments, *, error_type=RuntimeError):
    print(f'Checking {name} ...', flush=True)
    started = time.monotonic()
    try:
        result = subprocess.run(
            (sys.executable, *arguments), cwd=cwd, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', timeout=60 * 60,
            env=dict(os.environ, PYTHONIOENCODING='utf-8', QT_QPA_PLATFORM='offscreen'),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise error_type(f'Cannot run {name}: {exc}') from exc
    if result.returncode:
        failure_cases = [line for line in result.stdout.splitlines()
                         if line.startswith(('FAIL:', 'ERROR:'))]
        # Source-string assertions can put an entire module on one line and
        # otherwise crowd the actual failing test names out of the report.
        lines = [line if len(line) <= 1000 else line[:1000] + ' ... [line shortened]'
                 for line in result.stdout.splitlines()]
        tail = '\n'.join(lines)[-12000:].strip()
        raise error_type(f'{name} failed with exit code {result.returncode}. '
                         + '\n'.join(failure_cases) + f'\nLast output:\n{tail}')
    print(f'{name}: passed ({time.monotonic() - started:.1f}s)', flush=True)
    for line in result.stdout.splitlines():
        if re.fullmatch(r'Ran \d+ tests? in .+|OK(?: \(skipped=\d+\))?', line):
            print(f'  {line}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('quick', 'full'), default='quick')
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    try:
        if args.inventory_only:
            check_source_inventories(root)
            print('Source and wheel inventories cover local imports.')
        else:
            for name, cwd, command in development_suites(root, args.mode):
                run_suite(name, cwd, command)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
