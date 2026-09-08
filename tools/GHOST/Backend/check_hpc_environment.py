#!/usr/bin/env python3
"""Check the headless GHOST runtime without submitting jobs or writing results."""
import importlib
from pathlib import Path
import platform
import sys


def main():
    print('Python: {} ({})'.format(platform.python_version(), sys.executable))
    print('Compiler: {}'.format(platform.python_compiler()))
    print('Backend: {}'.format(Path(__file__).resolve().parent))
    if sys.version_info < (3, 6, 8):
        print('FAIL: the headless backend requires Python 3.6.8 or newer.')
        return 1
    try:
        import numpy as np
        import scipy
        from scipy.linalg import lu_factor, lu_solve
        import ghost_runtime
        print('NumPy: {}'.format(np.__version__))
        print('SciPy: {}'.format(scipy.__version__))
        for name, version, minimum in (
            ('NumPy', np.__version__, (1, 14, 3)),
            ('SciPy', scipy.__version__, (1, 0, 0)),
        ):
            numbers = tuple(int(part) for part in version.split('.')[:3])
            if numbers < minimum:
                raise RuntimeError('{} {} is older than the tested HPC minimum {}'.format(
                    name, version, '.'.join(str(part) for part in minimum)))
        print('dataclasses: {}'.format(ghost_runtime.dataclass.__module__))
        try:
            import psutil
            print('psutil: {} (process RSS: {} bytes)'.format(
                psutil.__version__, psutil.Process().memory_info().rss))
        except ImportError:
            print('psutil: absent (optional memory sampling unavailable)')
        for name in ('run_hpc_monostatic', 'run_hpc_bor_monostatic',
                     'rcs_solver', 'bor_dispatch', 'hpc_bundle'):
            importlib.import_module(name)
        matrix = np.array([[3+1j, 1-2j], [2+0j, 5-1j]], dtype=np.complex128)
        rhs = np.array([1+2j, -3+1j], dtype=np.complex128)
        result = lu_solve(lu_factor(matrix), rhs)
        residual = float(np.max(np.abs(matrix @ result - rhs)))
        if not np.isfinite(residual) or residual > 1e-12:
            raise RuntimeError('Complex LU check failed: residual {}'.format(residual))
        print('PASS: driver/solver imports and complex LU (residual {:.3g}).'.format(residual))
    except Exception as exc:
        print('FAIL: {}: {}'.format(type(exc).__name__, exc))
        return 1
    print('Next: run a small sweep inside a compute allocation; SLURM is not checked here.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
