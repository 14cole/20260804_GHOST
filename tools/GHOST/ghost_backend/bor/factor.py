"""One owned LU per BOR mode with bounded physical-RHS validation."""
import math
import numpy as np
from scipy.linalg import get_lapack_funcs
from ghost_backend.execution.metrics import timed_stage
from ghost_backend.linalg.workspace import matrix_inf_norm, first_nonfinite


class ModalFactor:
    def __init__(self, matrix, mode, monitor_cond, checkpoint=None):
        from ghost_backend.bor.solver import BOR_CONDITION_EST_MAX
        self.a = matrix
        self.mode = mode
        self.checkpoint = checkpoint or (lambda: None)
        self.diagnostics = None
        self.checkpoint()
        if first_nonfinite(matrix) is not None:
            raise RuntimeError('BoR mode m={} produced a non-finite system matrix.'.format(mode))
        self.matrix_inf = matrix_inf_norm(matrix)
        self.event = dict(factorizations=1, rhs_batches=0, max_rhs_columns=0,
                          max_backward_error=0., max_relative_residual=0., refinement_steps=0)
        getrf, self.getrs, gecon = get_lapack_funcs(('getrf', 'getrs', 'gecon'), (matrix,))
        # Keep A for original-coefficient residuals; LAPACK owns just one F-order copy.
        self.lu, self.piv, info = timed_stage('factorization')(getrf)(
            np.array(matrix, dtype=complex, order='F', copy=True), overwrite_a=True)
        if info:
            raise RuntimeError('BoR mode m={} LU factorization failed (LAPACK info={}).'.format(mode, info))
        self.condition = math.nan
        if monitor_cond:
            # Row-stripped norm avoids an extra full real matrix from abs(A).
            sums = np.zeros(len(matrix))
            for start in range(0, len(matrix), 64):
                self.checkpoint()
                sums += np.sum(abs(matrix[start:start + 64]), axis=0)
            norm = float(np.max(sums))
            reciprocal, info = timed_stage('condition_estimate')(gecon)(self.lu, norm)
            if info or not math.isfinite(float(reciprocal)) or reciprocal <= 0 or norm <= 0:
                raise RuntimeError('BoR mode m={} condition estimation failed.'.format(mode))
            self.condition = 1. / float(reciprocal)
            if not math.isfinite(self.condition) or self.condition > BOR_CONDITION_EST_MAX:
                raise RuntimeError('BoR mode m={} estimated 1-norm condition {} exceeds the release limit {}.'.format(
                    mode, self.condition, BOR_CONDITION_EST_MAX))

    def inverse(self, rhs):
        self.checkpoint()
        value, info = timed_stage('rhs_solve')(self.getrs)(self.lu, self.piv, rhs)
        if info:
            raise RuntimeError('BoR mode m={} LU solve failed (LAPACK info={}).'.format(self.mode, info))
        return value

    def errors(self, x, b):
        residual = self.a @ x - b
        norms = np.linalg.norm(residual, axis=0)
        bnorms = np.linalg.norm(b, axis=0)
        relative = norms / np.where(bnorms > 0., bnorms, 1.)
        numerator = np.max(abs(residual), axis=0)
        denominator = self.matrix_inf * np.max(abs(x), axis=0) + np.max(abs(b), axis=0)
        backward = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.)
        backward[(denominator <= 0.) & (numerator > 0.)] = np.inf
        return residual, relative, backward

    def solve(self, rhs):
        from ghost_backend.bor.solver import BOR_LINEAR_BACKWARD_ERROR_MAX, BOR_LINEAR_RESIDUAL_MAX
        b = np.asarray(rhs, complex)
        if b.ndim != 2 or b.shape[0] != len(self.a) or not b.shape[1] or first_nonfinite(b) is not None:
            raise RuntimeError('BoR mode m={} produced an invalid or non-finite excitation.'.format(self.mode))
        x = self.inverse(b)
        residual, relative, backward = self.errors(x, b)
        for attempt in range(2):
            if np.max(relative) <= BOR_LINEAR_RESIDUAL_MAX and np.max(backward) <= BOR_LINEAR_BACKWARD_ERROR_MAX:
                break
            candidate = x + self.inverse(-residual)
            updated = self.errors(candidate, b)
            if np.max(updated[1]) >= np.max(relative) and np.max(updated[2]) >= np.max(backward):
                break
            x = candidate
            residual, relative, backward = updated
            self.event['refinement_steps'] += 1
        if (first_nonfinite(x) is not None or not np.all(np.isfinite(relative))
                or not np.all(np.isfinite(backward)) or np.max(backward) > BOR_LINEAR_BACKWARD_ERROR_MAX):
            raise RuntimeError('BoR mode m={} normwise linear backward error {} exceeds the release limit {}.'.format(
                self.mode, float(np.max(backward)), BOR_LINEAR_BACKWARD_ERROR_MAX))
        self.relative_residual = relative
        self.event['rhs_batches'] += 1
        self.event['max_rhs_columns'] = max(self.event['max_rhs_columns'], b.shape[1])
        self.event['max_backward_error'] = max(self.event['max_backward_error'], float(np.max(backward)))
        self.event['max_relative_residual'] = max(self.event['max_relative_residual'], float(np.max(relative)))
        return x


def compressed_factor(oracle, mode, monitor_cond, options, workers, checkpoint=None):
    from scipy.sparse.linalg import LinearOperator, onenormest
    from ghost_backend.compressed.operator import StreamedOperator
    from ghost_backend.compressed.factor import CompressedFactor
    from ghost_backend.bor.solver import BOR_CONDITION_EST_MAX
    budget = options['compressed_storage_mib'] * 1024**2 // workers
    coordinates = oracle.row_coordinates
    if coordinates is None:
        coordinates = np.arange(oracle.n, dtype=float)[:, None]
    operator = timed_stage('modal_compressed_assembly')(StreamedOperator)(oracle, coordinates,
        tile=options['compression_tile'], budget=budget, checkpoint=checkpoint)
    factor = CompressedFactor(operator, label='BoR mode m={}'.format(mode), checkpoint=checkpoint,
                              storage_budget_bytes=budget, check_precision=False)
    factor.event['refinement_steps'] = 0
    factor.condition = math.nan
    if monitor_cond:
        inverse = LinearOperator(operator.shape, matvec=lambda z: factor.inverse(z),
                                 rmatvec=lambda z: factor.inverse(z, trans=2), dtype=complex)
        # Preserve BOR's unscaled 1-norm condition criterion.
        factor.condition = float(np.max(operator.column_norm + operator.column_error)) * float(onenormest(inverse))
        if not math.isfinite(factor.condition) or factor.condition > BOR_CONDITION_EST_MAX:
            raise RuntimeError('BoR mode m={} estimated 1-norm condition {} exceeds the release limit {}.'.format(
                mode, factor.condition, BOR_CONDITION_EST_MAX))
    return factor
