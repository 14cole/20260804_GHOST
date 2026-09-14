"""Polynomial Galerkin near blocks with analytic logarithmic self terms."""
from functools import lru_cache
import numpy as np
from scipy.special import hankel2e
from ghost_backend.twod.basis import values, coefficients, derivative_matrix


@lru_cache(maxsize=4)
def log_moments(degree):
    # Integral x**n y**m log|x-y| on the unit square, split at y=x.
    result = np.zeros((degree + 1, degree + 1))
    harmonic = [0.] + [sum(1. / j for j in range(1, i + 1)) for i in range(1, degree + 2)]
    for n in range(degree + 1):
        for m in range(degree + 1):
            total = n + m + 2
            result[n, m] = (-1. / ((m + 1) * total**2) - harmonic[m + 1] / ((m + 1) * total)
                            -1. / ((n + 1) * total**2) - harmonic[n + 1] / ((n + 1) * total))
    c = coefficients(degree)
    return c.T @ result @ c


def hypersingular(s, k, normal_o, normal_s, length_o, length_s):
    do = derivative_matrix(s.shape[0] - 1)
    ds = derivative_matrix(s.shape[1] - 1)
    return -complex(k)**2 * np.dot(normal_o, normal_s) * s + do.T @ s @ ds / (length_o * length_s)


def block(obs, src, k, obs_derivative=True, order=24, intervals=((0., 1.), (0., 1.))):
    from ghost_backend.twod.operators import _get_quadrature
    qo, qs = len(obs.node_ids) - 1, len(src.node_ids) - 1
    oa, ob = intervals[0]; sa, sb = intervals[1]
    t, w = _get_quadrature(order)
    same = obs.panel_index == src.panel_index and intervals[0] == intervals[1]
    shared = None
    if not same:
        for a in (0, 1):
            for b in (0, 1):
                po = obs.p0 + (oa if a == 0 else ob) * (obs.p1 - obs.p0)
                ps = src.p0 + (sa if b == 0 else sb) * (src.p1 - src.p0)
                if np.linalg.norm(po - ps) <= 1e-12:
                    shared = (a == 0, b == 0)
    stable_difference = None
    if same or shared is not None:
        # A fourth-power radial substitution regularizes endpoint log terms.
        u = t[:, None]**4
        v = t[None, :]
        weights = (4 * t[:, None]**7 * w[:, None] * w[None, :]).ravel()
        x = np.concatenate((np.broadcast_to(u, (order, order)).ravel(), (u*v).ravel()))
        y = np.concatenate(((u*v).ravel(), np.broadcast_to(u, (order, order)).ravel()))
        weight = np.tile(weights, 2)
        if same:
            stable_difference = ((ob-oa)*(x-y))[:, None]*(obs.p1-obs.p0)
        if shared is not None:
            eo = obs.p0 + (oa if shared[0] else ob)*(obs.p1-obs.p0)
            es = src.p0 + (sa if shared[1] else sb)*(src.p1-src.p0)
            origin = eo-es
            roundoff = 64*np.finfo(float).eps*max(np.linalg.norm(eo), np.linalg.norm(es), obs.length, src.length)
            if np.linalg.norm(origin) <= roundoff: origin = np.zeros(2)
            do = (ob-oa)*(obs.p1-obs.p0)*(1 if shared[0] else -1)
            ds = (sb-sa)*(src.p1-src.p0)*(1 if shared[1] else -1)
            stable_difference = origin + x[:, None]*do - y[:, None]*ds
            if not shared[0]: x = 1 - x
            if not shared[1]: y = 1 - y
    else:
        x, y = np.meshgrid(t, t, indexing='ij')
        x, y = x.ravel(), y.ravel()
        weight = np.outer(w, w).ravel()
    x = oa + (ob - oa) * x
    y = sa + (sb - sa) * y
    diff = ((obs.p0-src.p0) + x[:, None]*(obs.p1-obs.p0) - y[:, None]*(src.p1-src.p0)
            if stable_difference is None else stable_difference)
    distance = np.linalg.norm(diff, axis=1)
    phi_o, phi_s = values(x, qo), values(y, qs)
    if np.any(distance <= 0):
        raise ValueError('Polynomial quadrature encountered intersecting integration points.')
    argument = complex(k)*distance
    phase = np.exp(-1j*argument)
    # Duffy nodes approach a corner without reaching it. A fixed distance cutoff
    # would remove a changing portion of the integrable kernel as order rises.
    green = .25j*hankel2e(0, argument)*phase
    exact_log = same and intervals == ((0., 1.), (0., 1.))
    if exact_log:
        green -= np.log(abs(x - y)) / (2 * np.pi)
    sbk = phi_o.T @ ((weight * green)[:, None] * phi_s)
    if exact_log: sbk += log_moments(qo) / (2 * np.pi)
    if same:
        kbk = np.zeros_like(sbk)
    else:
        normal = obs.normal if obs_derivative else src.normal
        derivative = ((-.25j if obs_derivative else .25j)*complex(k)*hankel2e(1, argument)*phase
                      * (diff @ normal)/distance)
        kbk = phi_o.T @ ((weight * derivative)[:, None] * phi_s)
    scale = obs.length * src.length * (ob - oa) * (sb - sa)
    if not np.all(np.isfinite(sbk)) or not np.all(np.isfinite(kbk)):
        raise ValueError('Polynomial near quadrature did not converge: non-finite kernel block.')
    return sbk * scale, kbk * scale


def near_block(obs, src, k, obs_derivative=True, depth=0, intervals=((0., 1.), (0., 1.))):
    # The absolute scale prevents relative tests on symmetry-zero K blocks.
    from ghost_backend.twod.operators import NEAR_PAIR_QUADRATURE_RTOL, NEAR_PAIR_QUADRATURE_MAX_DEPTH
    from ghost_backend.execution.cpu import current_state
    state = current_state()
    if state is not None: state.checkpoint()
    low = block(obs, src, k, obs_derivative, 20, intervals)
    high = block(obs, src, k, obs_derivative, 36, intervals)
    scale = max(np.max(abs(high[0])), np.max(abs(high[1])), obs.length * src.length * 1e-10)
    error = max(np.max(abs(a-b)) for a, b in zip(low, high))
    if error <= NEAR_PAIR_QUADRATURE_RTOL * scale:
        return high
    if obs.panel_index == src.panel_index or depth >= 2*NEAR_PAIR_QUADRATURE_MAX_DEPTH:
        finest = block(obs, src, k, obs_derivative, 72, intervals)
        error = max(np.max(abs(a-b)) for a, b in zip(high, finest))
        if error > NEAR_PAIR_QUADRATURE_RTOL * scale:
            extra = block(obs, src, k, obs_derivative, 144, intervals)
            error = max(np.max(abs(a-b)) for a, b in zip(finest, extra))
            finest = extra
        if error > NEAR_PAIR_QUADRATURE_RTOL * scale:
            raise ValueError('Polynomial near quadrature did not converge; refine the mesh. '
                             'Panels {} / {}, k={}, intervals={}, relative change={:.3g}.'.format(
                                 obs.panel_index, src.panel_index, k, intervals, error/scale))
        return finest
    oi, si = intervals
    if (oi[1]-oi[0])*obs.length >= (si[1]-si[0])*src.length:
        mid = sum(oi) / 2
        children = [((oi[0], mid), si), ((mid, oi[1]), si)]
    else:
        mid = sum(si) / 2
        children = [(oi, (si[0], mid)), (oi, (mid, si[1]))]
    parts = [near_block(obs, src, k, obs_derivative, depth + 1, child) for child in children]
    return tuple(a + b for a, b in zip(*parts))
