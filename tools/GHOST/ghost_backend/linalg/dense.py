"""Reusable CPU factorization with bounded solves and shared residual evidence."""
import numpy as np
from ghost_backend.execution.metrics import timed_stage
from ghost_backend.linalg.workspace import matrix_inf_norm, first_nonfinite
from ghost_backend.linalg.refined_lu import requested_precision


class DenseFactor:
    def __init__(self, matrix, diagnostics=None, label='dense system', evidence=None,
                 checkpoint=None, force_double=False, coordinates=None):
        import ghost_backend.twod.solver as rcs
        self.a = np.asarray(matrix, dtype=np.complex128)
        self.diagnostics, self.label = diagnostics, label
        self.checkpoint = checkpoint or (lambda: None)
        self.checkpoint()
        rcs._ensure_finite_linear_system(self.a, label=label)
        self.matrix_inf = matrix_inf_norm(self.a)
        self.lu = self.piv = self.mixed = None
        self.hierarchical = None
        self.fallback_reason = ''
        requested, threshold = rcs._requested_dense_backend()
        if requested != 'cpu' and diagnostics is not None:
            self.fallback_reason = 'CPU LU required for certified condition estimation'
        self.relative_residual = np.empty(0)
        self.event = dict(unknowns=len(self.a), factorizations=0, rhs_batches=0,
                          max_rhs_columns=0, max_backward_error=0., max_relative_residual=0.)
        self.reported_factorizations = 0
        self.reported_hierarchical_builds = 0
        if evidence is not None:
            evidence.append(self.event)
        from ghost_backend.linalg.hierarchical import (
            factor_mode,
            HierarchicalFactor,
            HierarchicalRejected,
        )
        self.factor_mode = factor_mode()
        if self.factor_mode == 'compressed':
            raise ValueError('Compressed factorization requires a geometry-built compressed operator.')
        if self.factor_mode != 'dense' and requested_precision() == 'mixed':
            raise ValueError('Hierarchical CPU factorization requires double precision.')
        if self.factor_mode == 'hierarchical' or self.factor_mode == 'auto' and len(self.a) >= 2048:
            try:
                self.hierarchical = timed_stage('factorization')(HierarchicalFactor)(
                    self.a, coordinates, self.checkpoint, self.matrix_inf)
                self._sync_hierarchical_builds()
                self.event['hierarchical'] = self.hierarchical.evidence
            except (HierarchicalRejected, np.linalg.LinAlgError, ValueError, RuntimeWarning) as exc:
                if self.factor_mode == 'hierarchical':
                    raise RuntimeError('Hierarchical factorization rejected this system: {}'.format(exc)) from exc
                self.fallback_reason = 'Hierarchical factorization fell back to dense LU: {}'.format(exc)
        if self.hierarchical is None and requested_precision() == 'mixed' and not force_double and rcs._SCIPY_LINALG is not None:
            try:
                self.mixed = rcs.RefinedLU(self.a)
                self.event['factorizations'] += 1
            except (np.linalg.LinAlgError, ValueError, FloatingPointError, RuntimeWarning) as exc:
                self.fallback_reason = 'Mixed precision fell back to double LU: {}'.format(exc)
        if self.mixed is None and self.hierarchical is None:
            self._factor_double()
        if diagnostics is not None:
            self._condition()

    def _sync_hierarchical_builds(self):
        if self.hierarchical is not None:
            builds = self.hierarchical.evidence.get('builds', 1)
            self.event['factorizations'] += builds-self.reported_hierarchical_builds
            self.reported_hierarchical_builds = builds

    def _factor_double(self):
        import ghost_backend.twod.solver as rcs
        self._sync_hierarchical_builds()
        self.mixed = None
        self.hierarchical = None
        if rcs._SCIPY_LINALG is not None:
            self.lu, self.piv = timed_stage('factorization')(rcs._SCIPY_LINALG.lu_factor)(
                self.a, check_finite=False)
            self.event['factorizations'] += 1

    def _condition(self):
        import ghost_backend.twod.solver as rcs
        failed = False
        if self.hierarchical is not None:
            try:
                estimate = rcs._equilibrated_condition_from_lu(self.a, None, None, self.hierarchical.solve)
                method = 'equilibrated_1norm_hierarchical_refined_inverse'
            except (RuntimeError, np.linalg.LinAlgError, FloatingPointError, RuntimeWarning) as exc:
                if self.factor_mode != 'auto':
                    raise
                self.fallback_reason = 'Hierarchical condition check fell back to dense LU: {}'.format(exc)
                failed = True
        elif self.mixed is not None:
            try:
                estimate = rcs._equilibrated_condition_from_lu(
                    self.a, self.mixed.lu, self.mixed.piv, self.mixed.solve)
                method = 'equilibrated_1norm_refined_inverse'
            except (np.linalg.LinAlgError, ValueError, FloatingPointError, RuntimeWarning) as exc:
                self.fallback_reason = 'Mixed precision fell back to double LU: {}'.format(exc)
                failed = True
        elif self.lu is not None:
            estimate = rcs._equilibrated_condition_from_lu(self.a, self.lu, self.piv)
            method = 'equilibrated_1norm_lu_onenormest'
        else:

            row = np.maximum(np.max(np.abs(self.a), axis=1), 1e-300)
            eq = self.a / row[:, None]
            col = np.maximum(np.max(np.abs(eq), axis=0), 1e-300)
            estimate = float(np.linalg.cond(eq / col[None, :], p=1))
            method = 'equilibrated_1norm_numpy_fallback'
        if failed:


            self._factor_double()
            return self._condition()
        self._sync_hierarchical_builds()
        self.diagnostics.update(condition_est=estimate, condition_method=method,
                                condition_label=self.label)

    def _apply_inverse(self, b, return_residual=False):
        import ghost_backend.twod.solver as rcs
        if self.hierarchical is not None:
            try:
                result = timed_stage('rhs_solve')(self.hierarchical.solve)(b, return_residual=return_residual)
                self._sync_hierarchical_builds()
                return result
            except (RuntimeError, np.linalg.LinAlgError, FloatingPointError, RuntimeWarning) as exc:
                if self.factor_mode != 'auto':
                    raise
                self.fallback_reason = 'Hierarchical residual check fell back to dense LU: {}'.format(exc)


            self._factor_double()
            if self.diagnostics is not None:
                self._condition()
        if self.mixed is not None:
            try:
                return self.mixed.solve(b, return_residual=return_residual)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError, RuntimeWarning) as exc:
                self.fallback_reason = 'Mixed precision fell back to double LU: {}'.format(exc)
            self._factor_double()
            if self.diagnostics is not None:
                self._condition()
        if self.lu is None:
            self.event['factorizations'] += 1
            x = np.linalg.solve(self.a, b)
        else:
            x = timed_stage('rhs_solve')(rcs._SCIPY_LINALG.lu_solve)(
                (self.lu, self.piv), b, check_finite=False)
        return (x, None) if return_residual else x

    @timed_stage('linear_solve')
    def solve(self, rhs):
        import ghost_backend.twod.solver as rcs
        self.checkpoint()
        b = np.asarray(rhs, dtype=np.complex128)
        vector = b.ndim == 1
        if vector:
            b = b[:, None]
        if b.ndim != 2 or b.shape[0] != len(self.a) or not b.shape[1]:
            raise ValueError('RHS must have one or more columns and match the system.')
        if first_nonfinite(b) is not None:
            raise ValueError('Nonfinite RHS')
        x, inner_residual = self._apply_inverse(b, return_residual=True)
        if inner_residual is not None:


            np.negative(inner_residual, out=inner_residual)
            self.event['residual_products_reused'] = self.event.get('residual_products_reused', 0) + 1
        def metrics(candidate, residual=None):
            if residual is None:
                residual = self.a @ candidate - b
            ri = np.max(np.abs(residual), axis=0)
            den = self.matrix_inf * np.max(np.abs(candidate), axis=0) + np.max(np.abs(b), axis=0)
            errors = np.divide(ri, den, out=np.zeros_like(ri), where=den > 0)
            errors[(den <= 0) & (ri > 0)] = np.inf
            return residual, float(np.max(errors))
        residual, error = metrics(x, inner_residual)
        inner_residual = None
        steps = 0
        for attempt in range(2):
            if error <= rcs.DENSE_LINEAR_BACKWARD_ERROR_MAX:
                break
            candidate = x + self._apply_inverse(-residual)
            updated, candidate_error = metrics(candidate)
            if candidate_error >= error:
                break
            x, residual, error = candidate, updated, candidate_error
            steps += 1
        if not np.isfinite(error) or error > rcs.DENSE_LINEAR_BACKWARD_ERROR_MAX:
            raise RuntimeError('{} normwise backward error {} exceeds the release limit {}.'.format(
                self.label, error, rcs.DENSE_LINEAR_BACKWARD_ERROR_MAX))
        norm = np.linalg.norm(b, axis=0)
        self.relative_residual = np.linalg.norm(residual, axis=0) / np.where(norm <= rcs.EPS, 1., norm)
        self.event['rhs_batches'] += 1
        self.event['max_rhs_columns'] = max(self.event['max_rhs_columns'], b.shape[1])
        self.event['max_backward_error'] = max(self.event['max_backward_error'], error)
        self.event['max_relative_residual'] = max(self.event['max_relative_residual'], float(np.max(self.relative_residual)))
        if self.diagnostics is not None:
            self.diagnostics.update(linear_backward_error=self.event['max_backward_error'],
                                    linear_backward_error_limit=rcs.DENSE_LINEAR_BACKWARD_ERROR_MAX,
                                    linear_refinement_steps=max(self.diagnostics.get('linear_refinement_steps', 0), steps))
            if self.mixed is not None:
                self.diagnostics['mixed_precision_corrections'] = self.mixed.max_corrections
        requested, _ = rcs._requested_dense_backend()
        rcs._record_dense_backend_event(requested=requested, used='cpu_hierarchical' if self.hierarchical is not None else 'cpu_mixed_lu' if self.mixed is not None else 'cpu',
            n=len(self.a), label=self.label, fallback_reason=self.fallback_reason, gpu_device='',
            refinement_steps=steps, factorizations=self.event['factorizations'] - self.reported_factorizations,
            rhs_columns=b.shape[1], hierarchical=self.event.get('hierarchical'),
            sweep_compression=self.event.get('sweep_compression'),
            condition_method=(self.diagnostics or {}).get('condition_method'))
        self.reported_factorizations = self.event['factorizations']
        return x[:, 0] if vector else x
