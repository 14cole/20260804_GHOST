"""Guard the solver against re-acquiring SciPy/NumPy APIs it works around.

Each check reproduces an older signature exactly, so `inspect.signature`
capability tests see what the older release would show and the try/except
fallbacks meet the same TypeError. `audit/twod_2026-09/legacy_sim.py` drives
the same shims through complete dense/compressed/FMM solves; these are the
fast unit-level guards.
"""
from pathlib import Path
import importlib
import sys
import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, gmres, lgmres
from scipy.spatial import cKDTree

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
from ghost_backend.twod.fmm import galerkin, factor
from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry


def legacy_gmres(A, b, x0=None, tol=1e-5, restart=None, maxiter=None, M=None,
                 callback=None, restrt=None):
    """SciPy 1.0 signature: no rtol, no atol, no callback_type."""


def legacy_lgmres(A, b, x0=None, tol=1e-5, maxiter=1000, M=None, callback=None,
                  inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True):
    """SciPy 1.0 signature: no rtol, no atol, no prepend_outer_v."""


def test_krylov_keywords_follow_the_installed_signature():
    modern = factor._krylov_kwargs(gmres, 1e-9, callback_type='legacy')
    assert modern == {'rtol': 1e-9, 'atol': 0., 'callback_type': 'legacy'}
    assert factor._krylov_kwargs(lgmres, 1e-9, prepend_outer_v=True) == {
        'rtol': 1e-9, 'atol': 0., 'prepend_outer_v': True}
    # A legacy solver keeps only tol, which it measures against norm(b) --
    # exactly what rtol with atol=0 means.
    assert factor._krylov_kwargs(legacy_gmres, 1e-9, callback_type='legacy') == {'tol': 1e-9}
    assert factor._krylov_kwargs(legacy_lgmres, 1e-9, prepend_outer_v=True) == {'tol': 1e-9}


def test_equation_operator_drops_rmatmat_when_scipy_refuses_it(monkeypatch):
    refused = []

    def legacy_linear_operator(shape, matvec=None, rmatvec=None, matmat=None, dtype=None, **rest):
        if rest:
            refused.append(sorted(rest))
            raise TypeError('unexpected keyword argument {!r}'.format(sorted(rest)[0]))
        return LinearOperator(shape, matvec=matvec, rmatvec=rmatvec, matmat=matmat, dtype=dtype)

    matrix = np.array([[2+1j, 0.5], [-1j, 3.]], dtype=complex)
    build = lambda: factor._equation_operator((2, 2), lambda x: matrix@x,
        lambda x: matrix.conj().T@x, lambda x: matrix@x, lambda x: matrix.conj().T@x)
    x = np.array([[1+2j, -3.], [0.5j, 4.]], dtype=complex)
    expected = build()@x
    monkeypatch.setattr(factor, 'LinearOperator', legacy_linear_operator)
    np.testing.assert_allclose(build()@x, expected)
    np.testing.assert_allclose(build().H@x, matrix.conj().T@x)
    assert refused and all(names == ['rmatmat'] for names in refused)


def test_near_pairs_survive_a_scalar_only_query_ball_point(monkeypatch):
    from test_fmm_efficiency import mesh_for
    geometry = AssemblyGeometry(mesh_for('reentrant', 48, 3.)[0])
    galerkin._VECTOR_RADIUS[0] = True
    expected = galerkin.near_pairs(geometry, float('inf'))

    class LegacyTree(cKDTree):
        """SciPy < 1.9 coerces the radius with float()."""
        def query_ball_point(self, x, r, *args, **kwargs):
            float(r)
            return super().query_ball_point(x, r, *args, **kwargs)

    monkeypatch.setattr(galerkin, 'cKDTree', LegacyTree)
    galerkin._VECTOR_RADIUS[0] = True
    try:
        assert galerkin.near_pairs(geometry, float('inf')) == expected
        assert not galerkin._VECTOR_RADIUS[0], 'the scalar-radius fallback was never taken'
    finally:
        galerkin._VECTOR_RADIUS[0] = True


def test_pulse_coefficients_import_without_quad_vec(monkeypatch):
    import scipy.integrate
    monkeypatch.delattr(scipy.integrate, 'quad_vec', raising=False)
    monkeypatch.delitem(sys.modules, 'ghost_backend.twod.pulse.coefficients', raising=False)
    module = importlib.import_module('ghost_backend.twod.pulse.coefficients')
    reloaded = importlib.reload(module)
    assert reloaded.self_single_layer(1.3, .25) == pytest.approx(module.self_single_layer(1.3, .25))


def test_stable_sorts_use_a_kind_every_numpy_accepts():
    sources = [Path(factor.__file__), Path(importlib.import_module(
        'ghost_backend.linalg.hierarchical').__file__)]
    for path in sources:
        text = path.read_text(encoding='utf-8')
        assert "kind='stable'" not in text, path
        assert "kind='mergesort'" in text, path
