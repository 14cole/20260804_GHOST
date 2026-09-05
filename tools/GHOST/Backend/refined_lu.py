"""Opt-in single-precision factorization with double-precision residuals.

The assembled operator stays complex128. This reduces LU storage and cubic
factorization work, not the quadratic operator-assembly cost. Any stalled or
nonfinite correction must trigger a full double-precision factorization.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import warnings
import numpy as np
from scipy.linalg import lu_factor, lu_solve, LinAlgWarning
from solver_metrics import timed_stage

_PRECISION = ContextVar("ghost_lu_precision", default="double")


@contextmanager
def linear_precision(value):
    if value not in {"double", "mixed"}:
        raise ValueError("LU precision must be double or mixed.")
    token = _PRECISION.set(value)
    try:
        yield
    finally:
        _PRECISION.reset(token)


def requested_precision():
    return _PRECISION.get()


class RefinedLU:
    @timed_stage("mixed_factorization")
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.complex128)
        self.max_corrections = 0
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            warnings.simplefilter("error", LinAlgWarning)
            self.lu, self.piv = lu_factor(self.matrix.astype(np.complex64))
        if not np.all(np.isfinite(self.lu)):
            raise np.linalg.LinAlgError("Single-precision LU was nonfinite.")

    @timed_stage("mixed_rhs_and_refinement")
    def solve(self, rhs, trans=0):
        b = np.asarray(rhs, dtype=np.complex128)
        op = self.matrix if trans == 0 else self.matrix.T if trans == 1 else self.matrix.conj().T
        x = lu_solve((self.lu, self.piv), b.astype(np.complex64), trans=trans).astype(np.complex128)
        norm_b = np.max(np.abs(b), axis=0)
        scale = np.where(norm_b > 0, norm_b, 1.)
        previous = float("inf")
        for step in range(10):
            residual = b - op @ x
            error = float(np.max(np.max(np.abs(residual), axis=0) / scale))
            if np.isfinite(error) and error <= 2e-12:
                self.max_corrections = max(self.max_corrections, step)
                return x
            if not np.isfinite(error) or error >= previous * .98:
                raise np.linalg.LinAlgError("Mixed-precision refinement stalled; double LU required.")
            previous = error
            x += lu_solve((self.lu, self.piv), residual.astype(np.complex64), trans=trans).astype(np.complex128)
        raise np.linalg.LinAlgError("Mixed-precision refinement exceeded its correction limit.")
