"""Opt-in single-precision factorization with double-precision residuals."""
from contextlib import contextmanager
from ghost_backend.execution.runtime import ScopedValue
import warnings
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from ghost_backend.execution.metrics import timed_stage
from ghost_backend.linalg.workspace import first_nonfinite

_PRECISION = ScopedValue("ghost_lu_precision", default="double")


@contextmanager
def linear_precision(value):
    if value not in {"double", "mixed"}:
        raise ValueError("LU precision must be double or mixed.")
    with _PRECISION.override(value):
        yield


def requested_precision():
    return _PRECISION.get()


class RefinedLU:
    @timed_stage("mixed_factorization")
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=np.complex128)
        self.max_corrections = 0
        with warnings.catch_warnings():


            warnings.simplefilter("error", RuntimeWarning)
            single = np.array(self.matrix, dtype=np.complex64, order='F')
            if first_nonfinite(single) is not None:
                raise np.linalg.LinAlgError("Single-precision matrix was nonfinite.")
            self.lu, self.piv = lu_factor(single, overwrite_a=True, check_finite=False)
        if first_nonfinite(self.lu) is not None:
            raise np.linalg.LinAlgError("Single-precision LU was nonfinite.")

    @timed_stage("mixed_rhs_and_refinement")
    def solve(self, rhs, trans=0, return_residual=False):
        b = np.asarray(rhs, dtype=np.complex128)
        op = self.matrix if trans == 0 else self.matrix.T
        x = lu_solve((self.lu, self.piv), b.astype(np.complex64), trans=trans, check_finite=False).astype(np.complex128)
        norm_b = np.max(np.abs(b), axis=0)
        scale = np.where(norm_b > 0, norm_b, 1.)
        previous = float("inf")
        for step in range(10):

            applied = np.conjugate(op @ np.conjugate(x)) if trans == 2 else op @ x
            residual = b - applied
            error = float(np.max(np.max(np.abs(residual), axis=0) / scale))
            if np.isfinite(error) and error <= 2e-12:
                self.max_corrections = max(self.max_corrections, step)
                return (x, residual) if return_residual else x
            if not np.isfinite(error) or error >= previous * .98:
                raise np.linalg.LinAlgError("Mixed-precision refinement stalled; double LU required.")
            previous = error
            x += lu_solve((self.lu, self.piv), residual.astype(np.complex64), trans=trans, check_finite=False).astype(np.complex128)
        raise np.linalg.LinAlgError("Mixed-precision refinement exceeded its correction limit.")
