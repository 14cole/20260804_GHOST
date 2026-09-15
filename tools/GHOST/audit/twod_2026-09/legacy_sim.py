"""Run the 2-D solver with every post-SciPy-1.0 API made unavailable.

Each shim reproduces the older signature exactly, so `inspect.signature`
capability checks see what an old SciPy would show and the try/except paths
raise the same TypeError. Covers both discretizations on all three backends.
"""
import sys, importlib, inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np, scipy.integrate
calls = dict(rmatmat_refused=0, gmres=0, lgmres=0, ball_fallback=0)
from scipy.spatial import cKDTree
from scipy.sparse.linalg import LinearOperator as _LinearOperator
from scipy.sparse.linalg import gmres as _gmres, lgmres as _lgmres


class LegacyTree(cKDTree):
    """SciPy < 1.9: query_ball_point coerces r with float()."""
    def query_ball_point(self, x, r, *a, **k):
        float(r)
        return super().query_ball_point(x, r, *a, **k)


def legacy_linear_operator(shape, matvec=None, rmatvec=None, matmat=None, dtype=None, **rest):
    """SciPy < 1.4: no rmatmat keyword."""
    if rest:
        calls['rmatmat_refused'] += 1
        raise TypeError("__init__() got an unexpected keyword argument %r" % sorted(rest)[0])
    return _LinearOperator(shape, matvec=matvec, rmatvec=rmatvec, matmat=matmat, dtype=dtype)


def legacy_gmres(A, b, x0=None, tol=1e-5, restart=None, maxiter=None, M=None,
                 callback=None, restrt=None):
    """SciPy 1.0.0 signature: no atol, no callback_type, no rtol."""
    calls['gmres'] += 1
    return _gmres(A, b, x0=x0, rtol=tol, atol=0., restart=restart, maxiter=maxiter,
                  M=M, callback=callback, callback_type='legacy')


def legacy_lgmres(A, b, x0=None, tol=1e-5, maxiter=1000, M=None, callback=None,
                  inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True):
    """SciPy 1.0.0 signature: no atol, no rtol, no prepend_outer_v."""
    calls['lgmres'] += 1
    return _lgmres(A, b, x0=x0, rtol=tol, atol=0., maxiter=maxiter, M=M, callback=callback,
                   inner_m=inner_m, outer_k=outer_k, outer_v=outer_v,
                   store_outer_Av=store_outer_Av)


# SciPy < 1.4: scipy.integrate has no quad_vec. Remove it before any import.
had_quad_vec = hasattr(scipy.integrate, 'quad_vec')
if had_quad_vec: del scipy.integrate.quad_vec
for name in [m for m in list(sys.modules) if m.startswith('ghost_backend')]:
    del sys.modules[name]

import ghost_backend.twod.fmm.galerkin as G
import ghost_backend.twod.fmm.factor as F
import ghost_backend.compressed.factor as CF
import ghost_backend.twod.pulse.coefficients as C
G.cKDTree = LegacyTree; G._VECTOR_RADIUS[0] = True
F.LinearOperator = legacy_linear_operator
F.gmres, F.lgmres = legacy_gmres, legacy_lgmres
F._SIGNATURES.clear()
CF.gmres = legacy_gmres
print('pulse.coefficients imports with no scipy.integrate.quad_vec present')
print('  -> integrates with scipy.integrate.%s' % C.quad.__name__)
for name, function in (('gmres', legacy_gmres), ('lgmres', legacy_lgmres)):
    seen = set(inspect.signature(function).parameters)
    assert 'rtol' not in seen and 'atol' not in seen, name
print('  -> legacy %s/%s signatures expose tol only' % ('gmres', 'lgmres'))

from ghost_backend.execution.policy import native_fmm_available
from common import run, fields, fixture
modes = ['dense', 'compressed'] + (['fmm'] if native_fmm_available() else [])
reference = None
# 'pec' TM is a fused combined-field system and uses ordinary GMRES; 'ibc'
# keeps LGMRES augmentation, so both Krylov branches get exercised.
for case in ('pec', 'ibc'):
    for disc in ('galerkin', 'pulse'):
        for mode in modes:
            amplitude = fields(run(fixture(case, 96), 1.0, [0., 37.], disc=disc, mode=mode), 'HH')
            print('%-4s %-9s %-11s OK  %s' % (case, disc, mode, np.round(amplitude, 5)))
assert not G._VECTOR_RADIUS[0], 'the legacy KD-tree fallback was never exercised'
for name in ('rmatmat_refused', 'gmres', 'lgmres') if 'fmm' in modes else ('rmatmat_refused',):
    assert calls[name], 'the %s legacy path was never exercised' % name
print('legacy paths exercised: KD-tree fallback yes, %s'
      % ', '.join('%s=%d' % item for item in sorted(calls.items()) if item[0] != 'ball_fallback'))
if had_quad_vec: importlib.reload(scipy.integrate)
