"""Bounded, incrementally shared QR incident basis with original-RHS checks."""
from ghost_backend.execution.options import environment_value
import os
import weakref
import numpy as np
import scipy.linalg as la
from ghost_backend.execution.metrics import timed_stage
from ghost_backend.linalg.workspace import first_nonfinite


def _frobenius_norm(value):


    flat = value.ravel(order='K')
    return float(np.sqrt(np.vdot(flat, flat).real))


def mode():
    value = environment_value('GHOST_CPU_RHS_COMPRESSION', 'auto').strip().lower()
    if value not in ('off', 'auto', 'on'):
        raise ValueError('GHOST_CPU_RHS_COMPRESSION must be off, auto, or on.')
    return value


def _qr_basis(value, threshold):
    """Form only selected Q columns, avoiding a full D-by-batch Q."""
    (raw, tau), r, piv = la.qr(value, mode='raw', pivoting=True, check_finite=False)
    tail = np.sqrt(np.cumsum(np.sum(abs(r)**2, axis=1)[::-1])[::-1])
    rank = int(np.count_nonzero(tail > threshold))
    if not rank:
        return np.empty((len(value), 0), complex), np.empty((0, value.shape[1]), complex)
    packed = np.array(raw[:, :rank], order='F', copy=True)
    ungqr = la.get_lapack_funcs('ungqr', (packed,))
    q, work, info = ungqr(packed, tau[:rank], overwrite_a=True)
    if info:
        raise la.LinAlgError('QR basis construction failed (ungqr {}).'.format(info))
    recovery = np.empty((rank, value.shape[1]), complex)
    recovery[:, piv] = r[:rank]
    return q, recovery


class SweepBasis:
    """Per-factor lifetime; retained basis and solutions never exceed one batch."""
    def __init__(self, capacity):
        self.capacity = max(1, min(256, int(capacity)))
        self.scale = self.q = self.x = None
        self.owner = None

    def bind(self, factor):

        if self.owner is None or self.owner() is not factor:
            self.owner = weakref.ref(factor)
            self.scale = self.q = self.x = None


def _evidence(factor, setting):
    return factor.event.setdefault('sweep_compression', dict(requested=setting,
        method='incremental_pivoted_qr', input_columns=0, solved_columns=0,
        accepted_batches=0, fallback_batches=0, fallback_columns=0,
        max_rhs_error=0., max_reconstructed_backward_error=0.,
        retained_basis_columns=0, reused_basis_batches=0))


@timed_stage('rhs_compression')
def solve(factor, rhs, basis_state=None):
    setting = mode()
    rhs = np.asarray(rhs, complex)
    if rhs.ndim != 2 or rhs.shape[0] != len(factor.a) or not rhs.shape[1]:
        raise ValueError('Sweep RHS must have one or more columns and match the system.')
    if first_nonfinite(rhs) is not None:
        raise ValueError('Nonfinite RHS')
    factor.checkpoint()
    n, count = rhs.shape
    evidence = _evidence(factor, setting)
    evidence['input_columns'] += count
    if setting == 'off' or (count < 32 and basis_state is None) or (setting == 'auto' and n < 512):
        evidence['solved_columns'] += count
        return factor.solve(rhs)
    state = basis_state if basis_state is not None else SweepBasis(count)
    state.bind(factor)
    if state.scale is None:
        state.scale = np.max(abs(rhs), axis=1)
        state.scale[state.scale == 0] = 1.
        state.q = np.empty((n, 0), complex)
        state.x = np.empty((n, 0), complex)
    with np.errstate(over='ignore', invalid='ignore'):
        scaled = rhs / state.scale[:, None]
        norm = max(_frobenius_norm(scaled), 1e-300)
    if not np.isfinite(norm):


        state.scale = state.q = state.x = None
        evidence['fallback_batches'] += 1
        evidence['solved_columns'] += count
        scaled = None
        return factor.solve(rhs)
    existing = state.q.shape[1]
    recovery = state.q.conj().T @ scaled
    remainder = scaled - state.q @ recovery

    correction = state.q.conj().T @ remainder
    recovery += correction
    remainder -= state.q @ correction
    try:
        if existing and _frobenius_norm(remainder) <= 2e-15*norm:
            q, extension = np.empty((n, 0), complex), np.empty((0, count), complex)
        else:
            q, extension = _qr_basis(remainder, 2e-15*norm)
    except (la.LinAlgError, ValueError):
        q = None
    if q is None or q.shape[1] + existing > state.capacity or q.shape[1] >= .7*count:
        evidence['fallback_batches'] += 1
        evidence['solved_columns'] += count
        scaled = remainder = correction = recovery = q = extension = None
        return factor.solve(rhs)
    rank = q.shape[1]


    remainder -= q @ extension
    error = _frobenius_norm(remainder)/norm
    evidence['max_rhs_error'] = max(evidence['max_rhs_error'], error)
    if not np.isfinite(error) or error > 1e-14:
        evidence['fallback_batches'] += 1
        evidence['solved_columns'] += count
        scaled = remainder = correction = recovery = q = extension = None
        return factor.solve(rhs)
    if not existing and not rank:

        evidence['solved_columns'] += count
        scaled = remainder = correction = recovery = q = extension = None
        return factor.solve(rhs)
    scaled = remainder = correction = None
    new_solution = factor.solve(q*state.scale[:, None]) if rank else np.empty((n, 0), complex)
    result = state.x @ recovery + new_solution @ extension
    evidence['solved_columns'] += rank

    zero = np.all(rhs == 0, axis=0)
    result[:, zero] = 0
    residual = factor.a @ result - rhs
    den = factor.matrix_inf*np.max(abs(result), axis=0) + np.max(abs(rhs), axis=0)
    errors = (factor.physical_errors(result,rhs,residual) if hasattr(factor,'physical_errors') else
              np.max(abs(residual), axis=0)/np.maximum(den, 1e-300))
    from ghost_backend.twod.constants import DENSE_LINEAR_BACKWARD_ERROR_MAX, EPS
    failed = ~np.isfinite(errors) | (errors > DENSE_LINEAR_BACKWARD_ERROR_MAX)
    backward = float(np.max(errors))
    evidence['max_reconstructed_backward_error'] = max(evidence['max_reconstructed_backward_error'], backward)

    if np.any(failed):
        evidence['fallback_batches'] += 1
        evidence['fallback_columns'] += int(np.sum(failed))
        evidence['solved_columns'] += int(np.sum(failed))
        result[:, failed] = factor.solve(rhs[:, failed])
        residual[:, failed] = factor.a @ result[:, failed] - rhs[:, failed]
        den = factor.matrix_inf*np.max(abs(result[:, failed]), axis=0)+np.max(abs(rhs[:, failed]), axis=0)
        errors[failed] = (factor.physical_errors(result[:,failed],rhs[:,failed],residual[:,failed])
                          if hasattr(factor,'physical_errors') else
                          np.max(abs(residual[:, failed]), axis=0)/np.maximum(den, 1e-300))
    if not np.all(np.isfinite(errors)) or np.max(errors) > DENSE_LINEAR_BACKWARD_ERROR_MAX:
        raise RuntimeError('Recovered sweep failed the original-matrix backward-error check.')
    if not np.any(failed):
        if rank and basis_state is not None:
            state.q = np.column_stack((state.q, q))
            state.x = np.column_stack((state.x, new_solution))
        evidence['accepted_batches'] += 1
        evidence['reused_basis_batches'] += int(existing > 0)
    evidence['retained_basis_columns'] = state.q.shape[1]
    rhs_norm = np.linalg.norm(rhs, axis=0)
    factor.relative_residual = (factor.relative_errors(result,rhs,residual) if hasattr(factor,'relative_errors') else
        np.linalg.norm(residual, axis=0)/np.where(rhs_norm <= EPS, 1., rhs_norm))
    factor.event['max_backward_error'] = max(factor.event['max_backward_error'], float(np.max(errors)))
    factor.event['max_relative_residual'] = max(factor.event['max_relative_residual'], float(np.max(factor.relative_residual)))
    if factor.diagnostics is not None:
        factor.diagnostics['linear_backward_error'] = factor.event['max_backward_error']
    return result
