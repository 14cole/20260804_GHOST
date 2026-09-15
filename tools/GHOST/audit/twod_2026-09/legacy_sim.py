"""Reproduce the two reported SciPy-1.0 failures, then show them gone."""
import sys, importlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np, scipy.integrate
from scipy.spatial import cKDTree

# ---- 1. SciPy < 1.9: query_ball_point coerces r with float() ----
class LegacyTree(cKDTree):
    def query_ball_point(self, x, r, *a, **k):
        float(r)                      # TypeError for a multi-element array
        return super().query_ball_point(x, r, *a, **k)

# ---- 2. SciPy < 1.4: scipy.integrate has no quad_vec ----
had_quad_vec = hasattr(scipy.integrate, 'quad_vec')
if had_quad_vec: del scipy.integrate.quad_vec
for name in [m for m in list(sys.modules) if m.startswith('ghost_backend')]:
    del sys.modules[name]

import ghost_backend.twod.fmm.galerkin as G
G.cKDTree = LegacyTree
G._VECTOR_RADIUS[0] = True
import ghost_backend.twod.pulse.coefficients as C
print('pulse.coefficients imports with no scipy.integrate.quad_vec present')
print('  -> uses', C.quad.__name__, 'from scipy.integrate')

from common import run, fields, fixture
for disc in ('galerkin','pulse'):
    r = run(fixture('pec',96), 1.0, [0.,37.], disc=disc)
    print('%-9s solve OK  amplitudes %s  (legacy tree used: %s)'
          % (disc, np.round(fields(r,'HH'),4), not G._VECTOR_RADIUS[0]))
assert not G._VECTOR_RADIUS[0], 'the legacy fallback was never exercised'
if had_quad_vec: importlib.reload(scipy.integrate)
