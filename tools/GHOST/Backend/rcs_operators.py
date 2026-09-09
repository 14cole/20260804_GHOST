"""2-D quadrature, boundary operators, and linear-mesh field evaluation."""

import cmath
import math
import os
import threading
import numpy as np

# Keep string annotations resolvable for callers inspecting the public facade.
from rcs_geometry import (
    ComplexTable,
    ImpedanceTaper,
    LinearElement,
    LinearMesh,
    LinearNode,
    MaterialLibrary,
    MediumTable,
    Panel,
    PanelCoupledInfo,
)
from solver_metrics import timed_stage
from cpu_execution import current_state, cached_operator
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from rcs_constants import (
    EPS,
    EULER_GAMMA,
)
from rcs_special import (
    _SCIPY_SPECIAL,
    _hankel2_0,
    _hankel2_1,
)
from rcs_geometry import (
    _linear_shape_values,
    _surface_robin_alpha,
)


def _linear_param_to_point(elem: 'LinearElement', xi: 'float') -> 'np.ndarray':
    return elem.p0 + float(xi) * (elem.p1 - elem.p0)

def _linear_interval_point(elem: 'LinearElement', interval: 'Tuple[float, float]', use_start: 'bool') -> 'np.ndarray':
    a, b = float(interval[0]), float(interval[1])
    return _linear_param_to_point(elem, a if use_start else b)

def _linear_interval_length(elem: 'LinearElement', interval: 'Tuple[float, float]') -> 'float':
    a, b = float(interval[0]), float(interval[1])
    return max(abs(b - a) * float(elem.length), 0.0)

def _linear_interval_midpoint(elem: 'LinearElement', interval: 'Tuple[float, float]') -> 'np.ndarray':
    a, b = float(interval[0]), float(interval[1])
    return _linear_param_to_point(elem, 0.5 * (a + b))

def _linear_map_local_to_parent(interval: 'Tuple[float, float]', local_xi: 'float', start_is_shared: 'bool') -> 'float':
    a, b = float(interval[0]), float(interval[1])
    h = b - a
    x = float(local_xi)
    return (a + h * x) if start_is_shared else (b - h * x)

def _linear_shared_interval_endpoint_info(
    obs_elem: 'LinearElement',
    obs_interval: 'Tuple[float, float]',
    src_elem: 'LinearElement',
    src_interval: 'Tuple[float, float]',
    tol: 'float' = 1.0e-12,
) -> 'Optional[Tuple[bool, bool]]':
    obs_pts = [
        _linear_interval_point(obs_elem, obs_interval, True),
        _linear_interval_point(obs_elem, obs_interval, False),
    ]
    src_pts = [
        _linear_interval_point(src_elem, src_interval, True),
        _linear_interval_point(src_elem, src_interval, False),
    ]
    for obs_is_start, op in enumerate(obs_pts):
        for src_is_start, sp in enumerate(src_pts):
            if float(np.linalg.norm(op - sp)) <= float(tol):
                return bool(obs_is_start == 0), bool(src_is_start == 0)
    return None

def _integrate_linear_pair_box(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    kernel_eval: 'Callable[[np.ndarray, np.ndarray], complex]',
    obs_interval: 'Tuple[float, float]',
    src_interval: 'Tuple[float, float]',
    obs_order: 'int',
    src_order: 'int',
) -> 'np.ndarray':
    qt_obs, qw_obs = _get_quadrature(max(2, int(obs_order)))
    qt_src, qw_src = _get_quadrature(max(2, int(src_order)))
    obs_scale = max(float(obs_interval[1]) - float(obs_interval[0]), 0.0)
    src_scale = max(float(src_interval[1]) - float(src_interval[0]), 0.0)
    obs_len = float(obs_elem.length) * obs_scale
    src_len = float(src_elem.length) * src_scale
    block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return block

    for tobs, wobs in zip(qt_obs, qw_obs):
        xi_obs = float(obs_interval[0]) + obs_scale * float(tobs)
        phi_obs = _linear_shape_values(xi_obs)
        robs = _linear_param_to_point(obs_elem, xi_obs)
        for tsrc, wsrc in zip(qt_src, qw_src):
            xi_src = float(src_interval[0]) + src_scale * float(tsrc)
            phi_src = _linear_shape_values(xi_src)
            rsrc = _linear_param_to_point(src_elem, xi_src)
            kval = complex(kernel_eval(robs, rsrc))
            block += (float(wobs) * float(wsrc) * kval) * np.outer(phi_obs, phi_src)

    return block * obs_len * src_len

def _integrate_linear_pair_box_sk_vectorized(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    obs_interval: 'Tuple[float, float]',
    src_interval: 'Tuple[float, float]',
    obs_order: 'int',
    src_order: 'int',
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """
    Vectorized tensor-Gauss 2x2 S and K block assembly for one element pair.

    Evaluates all quadrature point pairs at once using array Hankel functions,
    avoiding per-point Python-loop overhead.  Returns (S_block, K_block).
    """

    qt_obs, qw_obs = _get_quadrature(max(2, int(obs_order)))
    qt_src, qw_src = _get_quadrature(max(2, int(src_order)))
    oa, ob = float(obs_interval[0]), float(obs_interval[1])
    sa, sb = float(src_interval[0]), float(src_interval[1])
    obs_scale = max(ob - oa, 0.0)
    src_scale = max(sb - sa, 0.0)
    obs_len = float(obs_elem.length) * obs_scale
    src_len = float(src_elem.length) * src_scale
    s_block = np.zeros((2, 2), dtype=np.complex128)
    k_block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return s_block, k_block

    nobs = len(qt_obs)
    nsrc = len(qt_src)

    # Precompute all parametric coordinates and physical points.
    xi_obs_all = oa + obs_scale * np.asarray(qt_obs, dtype=float)          # (nobs,)
    xi_src_all = sa + src_scale * np.asarray(qt_src, dtype=float)          # (nsrc,)
    phi_obs_all = np.column_stack([1.0 - xi_obs_all, xi_obs_all])          # (nobs, 2)
    phi_src_all = np.column_stack([1.0 - xi_src_all, xi_src_all])          # (nsrc, 2)

    obs_seg = obs_elem.p1 - obs_elem.p0
    src_seg = src_elem.p1 - src_elem.p0
    robs_all = obs_elem.p0[None, :] + xi_obs_all[:, None] * obs_seg[None, :]  # (nobs, 2)
    rsrc_all = src_elem.p0[None, :] + xi_src_all[:, None] * src_seg[None, :]  # (nsrc, 2)

    # All pairwise differences: (nobs, nsrc, 2)
    diff = robs_all[:, None, :] - rsrc_all[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))   # (nobs, nsrc)
    dist_safe = np.maximum(dist, EPS)

    if not bool(compute_single_layer) and not bool(compute_double_layer):
        raise ValueError("At least one element-pair operator must be requested.")

    kr = np.asarray(complex(k0) * dist_safe, dtype=np.complex128)
    kr[np.abs(kr) <= 1e-12] = 1e-12 + 0.0j
    if compute_single_layer:
        # Green's function: G = j/4 * H_0^(2)(k*r)
        h0 = _hankel2_0_array(kr.ravel()).reshape(nobs, nsrc)
        g_vals = 0.25j * h0   # (nobs, nsrc)

    if compute_double_layer:
        h1 = _hankel2_1_array(kr.ravel()).reshape(nobs, nsrc)
        if obs_normal_deriv:
            # dG/dn_obs
            proj = np.sum(diff * obs_elem.normal[None, None, :], axis=2) / dist_safe
            dk_vals = (-0.25j * complex(k0)) * h1 * proj
        else:
            # dG/dn_src (diff = robs-rsrc; source normal flips sign)
            proj = np.sum(src_elem.normal[None, None, :] * diff, axis=2) / dist_safe
            dk_vals = (0.25j * complex(k0)) * h1 * proj
        dk_vals[dist <= EPS] = 0.0

    # Weight tensor: (nobs, nsrc)
    w_outer = np.outer(np.asarray(qw_obs, dtype=float), np.asarray(qw_src, dtype=float))

    # Accumulate 2x2 blocks using einsum.
    # weighted_g = w * G, shape (nobs, nsrc)
    if compute_single_layer:
        weighted_g = w_outer * g_vals
        # s_block[a,b] = sum weighted_g[i,j]*phi_obs[i,a]*phi_src[j,b]
        s_block = np.einsum(
            'ij,ia,jb->ab', weighted_g, phi_obs_all, phi_src_all
        )
    if compute_double_layer:
        weighted_k = w_outer * dk_vals
        k_block = np.einsum(
            'ij,ia,jb->ab', weighted_k, phi_obs_all, phi_src_all
        )

    scale = obs_len * src_len
    return s_block * scale, k_block * scale


def _integrate_linear_pairs_box_sk_batched(
    elements: 'Sequence[LinearElement]',
    obs_indices: 'np.ndarray',
    src_indices: 'np.ndarray',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    order: 'int',
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """Tensor-Gauss S/K blocks for many full-interval element pairs.

    This is the same calculation as `_integrate_linear_pair_box_sk_vectorized`
    with an additional leading pair axis. It is used only for separated near
    pairs whose adaptive classifier selected a fixed tensor rule; singular,
    touching, and recursively adaptive pairs retain their dedicated paths.
    """

    obs_ids = np.asarray(obs_indices, dtype=np.int64).reshape(-1)
    src_ids = np.asarray(src_indices, dtype=np.int64).reshape(-1)
    if obs_ids.size != src_ids.size:
        raise ValueError("Batched near-pair index arrays must have equal length.")
    npairs = int(obs_ids.size)
    zero = np.zeros((npairs, 2, 2), dtype=np.complex128)
    if npairs == 0:
        return zero.copy(), zero.copy()
    if not bool(compute_single_layer) and not bool(compute_double_layer):
        raise ValueError("At least one batched near-pair operator is required.")

    qt, qw = _get_quadrature(max(2, int(order)))
    q = np.asarray(qt, dtype=float)
    weights = np.asarray(qw, dtype=float)
    phi = np.column_stack((1.0 - q, q))
    obs_elems = [elements[int(index)] for index in obs_ids]
    src_elems = [elements[int(index)] for index in src_ids]
    obs_p0 = np.asarray([elem.p0 for elem in obs_elems], dtype=float)
    src_p0 = np.asarray([elem.p0 for elem in src_elems], dtype=float)
    obs_seg = np.asarray([elem.p1 - elem.p0 for elem in obs_elems], dtype=float)
    src_seg = np.asarray([elem.p1 - elem.p0 for elem in src_elems], dtype=float)
    obs_pts = obs_p0[:, None, :] + q[None, :, None] * obs_seg[:, None, :]
    src_pts = src_p0[:, None, :] + q[None, :, None] * src_seg[:, None, :]
    diff = obs_pts[:, :, None, :] - src_pts[:, None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=3))
    dist_safe = np.maximum(dist, EPS)
    kr = np.asarray(complex(k0) * dist_safe, dtype=np.complex128)
    kr[np.abs(kr) <= 1e-12] = 1e-12 + 0.0j
    w_outer = np.outer(weights, weights)

    if compute_single_layer:
        h0 = _hankel2_0_array(kr.reshape(-1)).reshape(dist.shape)
        weighted_g = w_outer[None, :, :] * (0.25j * h0)
        s_blocks = np.einsum(
            'pij,ia,jb->pab', weighted_g, phi, phi
        )
    else:
        s_blocks = zero.copy()

    if compute_double_layer:
        h1 = _hankel2_1_array(kr.reshape(-1)).reshape(dist.shape)
        if obs_normal_deriv:
            normals = np.asarray(
                [elem.normal for elem in obs_elems], dtype=float
            )
            proj = np.sum(
                diff * normals[:, None, None, :], axis=3
            ) / dist_safe
            dk_vals = (-0.25j * complex(k0)) * h1 * proj
        else:
            normals = np.asarray(
                [elem.normal for elem in src_elems], dtype=float
            )
            proj = np.sum(
                diff * normals[:, None, None, :], axis=3
            ) / dist_safe
            dk_vals = (0.25j * complex(k0)) * h1 * proj
        dk_vals[dist <= EPS] = 0.0
        k_blocks = np.einsum(
            'pij,ia,jb->pab', w_outer[None, :, :] * dk_vals, phi, phi
        )
    else:
        k_blocks = zero.copy()

    scales = np.asarray(
        [obs.length * src.length for obs, src in zip(obs_elems, src_elems)],
        dtype=float,
    )[:, None, None]
    return s_blocks * scales, k_blocks * scales

def _single_layer_self_block_exact(
    elem: 'LinearElement',
    k0: 'Union[complex, float]',
    interval: 'Tuple[float, float]' = (0.0, 1.0),
) -> 'Optional[np.ndarray]':
    """
    Closed-form linear-Galerkin single-layer self block for a straight element.

    On a straight element the kernel depends only on u = |t - s|, so

        B_ij = l^2 * (j/4) * Int_0^1 H0^(2)(k*l*u) * C_ij(u) du

    with shape-pair weights (phi0 = 1-t, phi1 = t):

        C_diag(u) = (2 - 3u + u^3)/3      C_off(u) = (1 - u^3)/3

    (their sum reproduces the constant-basis weight 2(1-u) used by the
    exact `_single_layer_self_term`).  Substituting the small-argument
    series H0^(2)(x) = J0(x)[1 - j(2/pi)(ln(x/2)+gamma)] - j*R(x) turns the
    u-integral into exact moments:

        Int u^p du = 1/(p+1)        Int u^p ln(u) du = -1/(p+1)^2

    so the whole block is a rapidly convergent series -- machine precision,
    unlike the (u, uv) "Duffy" map, whose unresolved log singularity along
    the diagonal capped the self block at ~0.1-1% error.

    Returns None when |k*l| is too large for the series to be well
    conditioned (caller falls back to numeric quadrature).
    """

    a, b = float(interval[0]), float(interval[1])
    h = b - a
    ell = float(elem.length) * h
    if ell <= 0.0:
        return np.zeros((2, 2), dtype=np.complex128)
    z = complex(k0) * ell
    if abs(z) > 8.0:
        return None  # series cancellation grows beyond ~kl=8; use quadrature
    if abs(z) <= 1e-30:
        return None

    c_diag = (2.0 / 3.0, -1.0, 0.0, 1.0 / 3.0)   # coefficients of u^0..u^3
    c_off = (1.0 / 3.0, 0.0, 0.0, -1.0 / 3.0)

    def moment(p: 'int', coeffs) -> 'float':
        return sum(c / (p + q + 1) for q, c in enumerate(coeffs))

    def log_moment(p: 'int', coeffs) -> 'float':
        return -sum(c / (p + q + 1) ** 2 for q, c in enumerate(coeffs))

    two_over_pi = 2.0 / math.pi
    log_term = cmath.log(z / 2.0) + EULER_GAMMA
    z_quarter_sq = (z / 2.0) ** 2

    b_diag = 0.0 + 0.0j
    b_off = 0.0 + 0.0j
    alpha = 1.0 + 0.0j        # (-1)^m (z/2)^(2m) / (m!)^2
    harmonic = 0.0            # H_m
    m = 0
    while True:
        # H0^(2)(z*u) = sum_m alpha_m [1 - j(2/pi)(log_term - H_m)] u^(2m)
        #             - j(2/pi) ln(u) * sum_m alpha_m u^(2m)
        a_m = alpha * (1.0 - 1j * two_over_pi * (log_term - harmonic))
        p = 2 * m
        b_diag += a_m * moment(p, c_diag) - 1j * two_over_pi * alpha * log_moment(p, c_diag)
        b_off += a_m * moment(p, c_off) - 1j * two_over_pi * alpha * log_moment(p, c_off)
        m += 1
        alpha *= -z_quarter_sq / (m * m)
        harmonic += 1.0 / m
        if m > 60:
            return None
        if abs(alpha) < 1e-18 * max(1.0, abs(b_diag)):
            break

    block_local = (0.25j * ell * ell) * np.array(
        [[b_diag, b_off], [b_off, b_diag]], dtype=np.complex128,
    )
    if a == 0.0 and b == 1.0:
        return block_local
    # Sub-interval: parent shape functions restricted to [a, b] are linear
    # combinations of the local interval basis: phi_parent = T @ psi_local.
    t_mat = np.array([[1.0 - a, 1.0 - b], [a, b]], dtype=np.complex128)
    return t_mat @ block_local @ t_mat.T


def _integrate_linear_self_duffy(
    elem: 'LinearElement',
    kernel_eval: 'Callable[[np.ndarray, np.ndarray], complex]',
    interval: 'Tuple[float, float]',
    order: 'int' = 20,
) -> 'np.ndarray':
    qt, qw = _get_quadrature(max(4, int(order)))
    a, b = float(interval[0]), float(interval[1])
    h = max(b - a, 0.0)
    elem_len = float(elem.length) * h
    block = np.zeros((2, 2), dtype=np.complex128)
    if elem_len <= 0.0:
        return block

    for u, wu in zip(qt, qw):
        uu = float(u)
        jac_outer = float(wu) * uu
        t_major = a + h * uu
        s_major = t_major
        robs_major = _linear_param_to_point(elem, t_major)
        rsrc_major = _linear_param_to_point(elem, s_major)
        phi_t_major = _linear_shape_values(t_major)
        phi_s_major = _linear_shape_values(s_major)
        for v, wv in zip(qt, qw):
            vv = float(v)
            weight = jac_outer * float(wv)
            # Triangle: s <= t
            xi_t = a + h * uu
            xi_s = a + h * (uu * vv)
            phi_t = _linear_shape_values(xi_t)
            phi_s = _linear_shape_values(xi_s)
            robs = _linear_param_to_point(elem, xi_t)
            rsrc = _linear_param_to_point(elem, xi_s)
            block += weight * complex(kernel_eval(robs, rsrc)) * np.outer(phi_t, phi_s)
            # Triangle: t <= s
            xi_t2 = a + h * (uu * vv)
            xi_s2 = a + h * uu
            phi_t2 = _linear_shape_values(xi_t2)
            phi_s2 = _linear_shape_values(xi_s2)
            robs2 = _linear_param_to_point(elem, xi_t2)
            rsrc2 = _linear_param_to_point(elem, xi_s2)
            block += weight * complex(kernel_eval(robs2, rsrc2)) * np.outer(phi_t2, phi_s2)

    return block * (elem_len * elem_len)

def _integrate_linear_touching_duffy(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    kernel_eval: 'Callable[[np.ndarray, np.ndarray], complex]',
    obs_interval: 'Tuple[float, float]',
    src_interval: 'Tuple[float, float]',
    obs_start_is_shared: 'bool',
    src_start_is_shared: 'bool',
    order: 'int' = 20,
) -> 'np.ndarray':
    qt, qw = _get_quadrature(max(4, int(order)))
    obs_len = _linear_interval_length(obs_elem, obs_interval)
    src_len = _linear_interval_length(src_elem, src_interval)
    block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return block

    for u, wu in zip(qt, qw):
        uu = float(u)
        jac_outer = float(wu) * uu
        for v, wv in zip(qt, qw):
            vv = float(v)
            weight = jac_outer * float(wv)
            # Triangle 1: source local distance <= observation local distance
            xi_obs = _linear_map_local_to_parent(obs_interval, uu, obs_start_is_shared)
            xi_src = _linear_map_local_to_parent(src_interval, uu * vv, src_start_is_shared)
            phi_obs = _linear_shape_values(xi_obs)
            phi_src = _linear_shape_values(xi_src)
            robs = _linear_param_to_point(obs_elem, xi_obs)
            rsrc = _linear_param_to_point(src_elem, xi_src)
            block += weight * complex(kernel_eval(robs, rsrc)) * np.outer(phi_obs, phi_src)
            # Triangle 2: observation local distance <= source local distance
            xi_obs2 = _linear_map_local_to_parent(obs_interval, uu * vv, obs_start_is_shared)
            xi_src2 = _linear_map_local_to_parent(src_interval, uu, src_start_is_shared)
            phi_obs2 = _linear_shape_values(xi_obs2)
            phi_src2 = _linear_shape_values(xi_src2)
            robs2 = _linear_param_to_point(obs_elem, xi_obs2)
            rsrc2 = _linear_param_to_point(src_elem, xi_src2)
            block += weight * complex(kernel_eval(robs2, rsrc2)) * np.outer(phi_obs2, phi_src2)

    return block * (obs_len * src_len)


def _integrate_linear_touching_duffy_sk_vectorized(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    obs_interval: 'Tuple[float, float]',
    src_interval: 'Tuple[float, float]',
    obs_start_is_shared: 'bool',
    src_start_is_shared: 'bool',
    order: 'int' = 20,
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """Vectorized two-triangle Duffy rule for endpoint-touching panels."""

    if not bool(compute_single_layer) and not bool(compute_double_layer):
        raise ValueError("At least one touching-pair operator must be requested.")

    qt, qw = _get_quadrature(max(4, int(order)))
    obs_len = _linear_interval_length(obs_elem, obs_interval)
    src_len = _linear_interval_length(src_elem, src_interval)
    s_block = np.zeros((2, 2), dtype=np.complex128)
    k_block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return s_block, k_block

    u = np.asarray(qt, dtype=float)[:, None]
    v = np.asarray(qt, dtype=float)[None, :]
    weights = (
        np.asarray(qw, dtype=float)[:, None]
        * u
        * np.asarray(qw, dtype=float)[None, :]
    ).reshape(-1)
    uv = u * v

    def map_many(interval, local, start_is_shared):
        a, b = float(interval[0]), float(interval[1])
        h = b - a
        return (
            a + h * local
            if start_is_shared
            else b - h * local
        )

    # Match the scalar routine's summation domain exactly: triangle 1 uses
    # (obs=u, src=u*v), triangle 2 uses (obs=u*v, src=u).
    xi_obs = np.concatenate((
        np.broadcast_to(
            map_many(obs_interval, u, obs_start_is_shared), uv.shape
        ).reshape(-1),
        map_many(obs_interval, uv, obs_start_is_shared).reshape(-1),
    ))
    xi_src = np.concatenate((
        map_many(src_interval, uv, src_start_is_shared).reshape(-1),
        np.broadcast_to(
            map_many(src_interval, u, src_start_is_shared), uv.shape
        ).reshape(-1),
    ))
    weights = np.concatenate((weights, weights))

    phi_obs = np.column_stack((1.0 - xi_obs, xi_obs))
    phi_src = np.column_stack((1.0 - xi_src, xi_src))
    obs_seg = obs_elem.p1 - obs_elem.p0
    src_seg = src_elem.p1 - src_elem.p0
    robs = obs_elem.p0[None, :] + xi_obs[:, None] * obs_seg[None, :]
    rsrc = src_elem.p0[None, :] + xi_src[:, None] * src_seg[None, :]
    diff = robs - rsrc

    if compute_single_layer:
        g_vals = _green_2d_array(k0, np.linalg.norm(diff, axis=1))
        s_block = np.einsum(
            'q,q,qa,qb->ab', weights, g_vals, phi_obs, phi_src
        )

    if compute_double_layer:
        if obs_normal_deriv:
            dk_vals = _dgreen_dn_obs_array(k0, diff, obs_elem.normal)
        else:
            src_normals = np.broadcast_to(src_elem.normal, diff.shape)
            dk_vals = _dgreen_dn_src_array(k0, diff, src_normals)
        k_block = np.einsum(
            'q,q,qa,qb->ab', weights, dk_vals, phi_obs, phi_src
        )

    scale = obs_len * src_len
    return s_block * scale, k_block * scale


def _integrate_linear_pair_recursive(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    kernel_eval: 'Callable[[np.ndarray, np.ndarray], complex]',
    obs_interval: 'Tuple[float, float]',
    src_interval: 'Tuple[float, float]',
    obs_order: 'int',
    src_order: 'int',
    depth: 'int' = 0,
    max_depth: 'int' = 3,
) -> 'np.ndarray':
    obs_len = _linear_interval_length(obs_elem, obs_interval)
    src_len = _linear_interval_length(src_elem, src_interval)
    block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return block

    same_elem_same_interval = (
        obs_elem.panel_index == src_elem.panel_index
        and abs(float(obs_interval[0]) - float(src_interval[0])) <= 1.0e-15
        and abs(float(obs_interval[1]) - float(src_interval[1])) <= 1.0e-15
    )
    if same_elem_same_interval:
        order = max(6, int(max(obs_order, src_order)) + 1)
        return _integrate_linear_self_duffy(
            obs_elem,
            kernel_eval,
            interval=obs_interval,
            order=order,
        )

    shared = _linear_shared_interval_endpoint_info(obs_elem, obs_interval, src_elem, src_interval)
    if shared is not None:
        order = max(6, int(max(obs_order, src_order)) + 1)
        return _integrate_linear_touching_duffy(
            obs_elem,
            src_elem,
            kernel_eval,
            obs_interval=obs_interval,
            src_interval=src_interval,
            obs_start_is_shared=bool(shared[0]),
            src_start_is_shared=bool(shared[1]),
            order=order,
        )

    obs_mid = _linear_interval_midpoint(obs_elem, obs_interval)
    src_mid = _linear_interval_midpoint(src_elem, src_interval)
    distance = float(np.linalg.norm(obs_mid - src_mid))
    scale = max(obs_len, src_len, EPS)
    ratio = distance / scale

    # Refine near-singular element pairs adaptively before falling back to tensor Gauss.
    if depth < max_depth and ratio < 0.95:
        oa, ob = float(obs_interval[0]), float(obs_interval[1])
        sa, sb = float(src_interval[0]), float(src_interval[1])
        if ratio < 0.16:
            om = 0.5 * (oa + ob)
            sm = 0.5 * (sa + sb)
            sub_obs = [(oa, om), (om, ob)]
            sub_src = [(sa, sm), (sm, sb)]
            for oi in sub_obs:
                for si in sub_src:
                    block += _integrate_linear_pair_recursive(
                        obs_elem,
                        src_elem,
                        kernel_eval,
                        oi,
                        si,
                        obs_order=obs_order,
                        src_order=src_order,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
            return block
        if obs_len >= src_len:
            om = 0.5 * (oa + ob)
            return (
                _integrate_linear_pair_recursive(
                    obs_elem, src_elem, kernel_eval, (oa, om), src_interval,
                    obs_order=obs_order, src_order=src_order, depth=depth + 1, max_depth=max_depth,
                )
                + _integrate_linear_pair_recursive(
                    obs_elem, src_elem, kernel_eval, (om, ob), src_interval,
                    obs_order=obs_order, src_order=src_order, depth=depth + 1, max_depth=max_depth,
                )
            )
        sm = 0.5 * (sa + sb)
        return (
            _integrate_linear_pair_recursive(
                obs_elem, src_elem, kernel_eval, obs_interval, (sa, sm),
                obs_order=obs_order, src_order=src_order, depth=depth + 1, max_depth=max_depth,
            )
            + _integrate_linear_pair_recursive(
                obs_elem, src_elem, kernel_eval, obs_interval, (sm, sb),
                obs_order=obs_order, src_order=src_order, depth=depth + 1, max_depth=max_depth,
            )
        )

    adapt_order, _ = _near_singular_scheme(distance, scale)
    tensor_order = max(int(max(obs_order, src_order)), min(16, int(max(5, adapt_order))))
    return _integrate_linear_pair_box(
        obs_elem,
        src_elem,
        kernel_eval,
        obs_interval=obs_interval,
        src_interval=src_interval,
        obs_order=tensor_order,
        src_order=tensor_order,
    )

def _integrate_linear_pair_generic(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    kernel_eval: 'Callable[[np.ndarray, np.ndarray], complex]',
    obs_order: 'int' = 6,
    src_order: 'int' = 6,
) -> 'np.ndarray':
    """
    Assemble a 2x2 Galerkin block for one observation/source element pair.

    This upgraded implementation keeps the straight-element tensor-Gauss backbone but
    adds two accuracy-critical improvements for the experimental linear/Galerkin path:
    - Duffy-type quadrature for same-element and endpoint-touching singular pairs
    - adaptive recursive interval subdivision for near-singular pairs
    """

    return _integrate_linear_pair_recursive(
        obs_elem,
        src_elem,
        kernel_eval,
        obs_interval=(0.0, 1.0),
        src_interval=(0.0, 1.0),
        obs_order=obs_order,
        src_order=src_order,
        depth=0,
        max_depth=6,
    )

def _stable_hankel2_array(order: 'int', x: 'np.ndarray') -> 'np.ndarray':
    """Robust array Hankel evaluator for real and complex arguments.

    Uses scaled SciPy Hankel for complex arguments when available, then repairs
    any remaining non-finite entries with the existing scalar helpers.
    """

    z = np.asarray(x, dtype=np.complex128)
    out: 'Optional[np.ndarray]' = None
    if _SCIPY_SPECIAL is not None:
        try:
            # Real fast path when possible.
            if np.all(np.abs(z.imag) <= 1e-14) and np.all(z.real >= 0.0):
                xr = np.maximum(z.real.astype(float, copy=False), 1e-12)
                if order == 0:
                    out = np.asarray(_SCIPY_SPECIAL.j0(xr) - 1j * _SCIPY_SPECIAL.y0(xr), dtype=np.complex128)
                else:
                    out = np.asarray(_SCIPY_SPECIAL.j1(xr) - 1j * _SCIPY_SPECIAL.y1(xr), dtype=np.complex128)
            elif hasattr(_SCIPY_SPECIAL, 'hankel2e'):
                scaled = np.asarray(_SCIPY_SPECIAL.hankel2e(order, z), dtype=np.complex128)
                out = scaled * np.exp(-1j * z)
            else:
                out = np.asarray(_SCIPY_SPECIAL.hankel2(order, z), dtype=np.complex128)
        except Exception:
            out = None
    if out is None:
        vec = np.vectorize(_hankel2_0 if order == 0 else _hankel2_1, otypes=[np.complex128])
        return np.asarray(vec(z), dtype=np.complex128)

    finite = np.isfinite(out.real) & np.isfinite(out.imag)
    if not np.all(finite):
        vec = np.vectorize(_hankel2_0 if order == 0 else _hankel2_1, otypes=[np.complex128])
        repaired = np.asarray(vec(z[~finite]), dtype=np.complex128)
        out = np.asarray(out, dtype=np.complex128)
        out[~finite] = repaired
    return np.asarray(out, dtype=np.complex128)

def _hankel2_0_array(x: 'np.ndarray') -> 'np.ndarray':
    return _stable_hankel2_array(0, x)

def _hankel2_1_array(x: 'np.ndarray') -> 'np.ndarray':
    return _stable_hankel2_array(1, x)

def _green_2d_array(k0: 'Union[complex, float]', r: 'np.ndarray') -> 'np.ndarray':
    rr = np.maximum(np.asarray(r, dtype=float), EPS)
    x = np.asarray(complex(k0) * rr, dtype=np.complex128)
    x[np.abs(x) <= 1e-12] = 1e-12 + 0.0j
    return 0.25j * _hankel2_0_array(x)

def _dgreen_dn_obs_array(k0: 'Union[complex, float]', r_vec: 'np.ndarray', n_obs: 'np.ndarray') -> 'np.ndarray':
    rr = np.linalg.norm(r_vec, axis=1)
    out = np.zeros(rr.shape[0], dtype=np.complex128)
    mask = rr > EPS
    if not np.any(mask):
        return out
    rrm = rr[mask]
    x = np.asarray(complex(k0) * rrm, dtype=np.complex128)
    x[np.abs(x) <= 1e-12] = 1e-12 + 0.0j
    h1 = _hankel2_1_array(x)
    projection = (r_vec[mask] @ np.asarray(n_obs, dtype=float)) / rrm
    out[mask] = (-0.25j * complex(k0)) * h1 * projection
    return out

def _dgreen_dn_src_array(k0: 'Union[complex, float]', r_vec: 'np.ndarray', n_src: 'np.ndarray') -> 'np.ndarray':
    rr = np.linalg.norm(r_vec, axis=1)
    out = np.zeros(rr.shape[0], dtype=np.complex128)
    mask = rr > EPS
    if not np.any(mask):
        return out
    rrm = rr[mask]
    x = np.asarray(complex(k0) * rrm, dtype=np.complex128)
    x[np.abs(x) <= 1e-12] = 1e-12 + 0.0j
    h1 = _hankel2_1_array(x)
    projection = np.sum(np.asarray(n_src, dtype=float)[mask] * r_vec[mask], axis=1) / rrm
    out[mask] = (0.25j * complex(k0)) * h1 * projection
    return out



def _single_layer_block_linear(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    k0: 'Union[complex, float]',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
) -> 'np.ndarray':
    if current_state() is not None and obs_elem.panel_index != src_elem.panel_index:
        shared = _linear_shared_interval_endpoint_info(obs_elem, (0., 1.), src_elem, (0., 1.))
        if shared is not None:
            return _integrate_linear_touching_duffy_sk_vectorized(
                obs_elem, src_elem, k0, False, (0., 1.), (0., 1.), shared[0], shared[1],
                order=max(6, max(int(obs_order), int(src_order)) + 1),
                compute_single_layer=True, compute_double_layer=False)[0]
    if obs_elem.panel_index == src_elem.panel_index:
        exact = _single_layer_self_block_exact(obs_elem, k0)
        if exact is not None:
            return exact
    return _integrate_linear_pair_generic(
        obs_elem,
        src_elem,
        lambda robs, rsrc: _green_2d(k0, max(float(np.linalg.norm(robs - rsrc)), EPS)),
        obs_order=obs_order,
        src_order=src_order,
    )


def _sk_blocks_near_linear(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """
    Compute S and K 2x2 blocks for a near element pair.

    Uses Duffy transforms for self and touching pairs (via the existing recursive
    path), and the vectorized tensor-Gauss path for separated-near pairs.
    """

    same_elem = obs_elem.panel_index == src_elem.panel_index
    # Singularity classification is geometric, not algebraic.  The
    # interface-aware mesh deliberately gives coincident endpoints distinct
    # node IDs when their boundary/material signatures differ.  Such panels
    # still share a physical endpoint and require touching-pair Duffy
    # quadrature; treating them as separated makes tensor subdivision chase
    # the endpoint singularity until max_depth without converging.
    shared = (
        None
        if same_elem
        else _linear_shared_interval_endpoint_info(
            obs_elem, (0.0, 1.0), src_elem, (0.0, 1.0), tol=1.0e-9
        )
    )

    zero = np.zeros((2, 2), dtype=np.complex128)
    if same_elem:
        # A straight panel's normal derivative of G against itself is exactly
        # zero: every source-observation displacement is tangential.  The jump
        # term is represented separately by the mass matrix, so no principal-
        # value work belongs in K/K' here.
        s_blk = (
            _single_layer_block_linear(
                obs_elem, src_elem, k0, obs_order, src_order
            ) if compute_single_layer else zero
        )
        return s_blk, zero

    if shared is not None:
        order = max(6, int(max(obs_order, src_order)) + 1)
        return _integrate_linear_touching_duffy_sk_vectorized(
            obs_elem=obs_elem,
            src_elem=src_elem,
            k0=k0,
            obs_normal_deriv=obs_normal_deriv,
            obs_interval=(0.0, 1.0),
            src_interval=(0.0, 1.0),
            obs_start_is_shared=bool(shared[0]),
            src_start_is_shared=bool(shared[1]),
            order=order,
            compute_single_layer=compute_single_layer,
            compute_double_layer=compute_double_layer,
        )

    # Separated-near pairs: a fixed tensor rule is not sufficient when two
    # panels are almost parallel and their gap is small compared with their
    # length.  In that limit the logarithmic S kernel (and the sharply peaked
    # normal-derivative kernel) is concentrated in a narrow band around the
    # projected diagonal.  Increasing the rule from 8 to 16 points without
    # subdividing still leaves percent-to-order-one block errors for
    # gap/length below a few percent.
    obs_mid = obs_elem.center
    src_mid = src_elem.center
    distance = float(np.linalg.norm(obs_mid - src_mid))
    scale = max(obs_elem.length, src_elem.length, EPS)
    adapt_order, _ = _near_singular_scheme(distance, scale)
    tensor_order = max(int(max(obs_order, src_order)), min(16, int(max(5, adapt_order))))

    if distance / scale < 0.75:
        return _integrate_linear_pair_adaptive_sk(
            obs_elem=obs_elem,
            src_elem=src_elem,
            k0=k0,
            obs_normal_deriv=obs_normal_deriv,
            obs_order=tensor_order,
            src_order=tensor_order,
            compute_single_layer=compute_single_layer,
            compute_double_layer=compute_double_layer,
        )

    return _integrate_linear_pair_box_sk_vectorized(
        obs_elem, src_elem, k0, obs_normal_deriv,
        obs_interval=(0.0, 1.0), src_interval=(0.0, 1.0),
        obs_order=tensor_order, src_order=tensor_order,
        compute_single_layer=compute_single_layer,
        compute_double_layer=compute_double_layer,
    )


NEAR_PAIR_QUADRATURE_RTOL = 1.0e-9
NEAR_PAIR_QUADRATURE_MAX_DEPTH = 12


def _integrate_linear_pair_adaptive_sk(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    obs_order: 'int',
    src_order: 'int',
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
    rtol: 'float' = NEAR_PAIR_QUADRATURE_RTOL,
    max_depth: 'int' = NEAR_PAIR_QUADRATURE_MAX_DEPTH,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """Converged S/K quadrature for separated, nearly singular panel pairs.

    Each box is compared with its four-way bisection.  Only children whose
    parent comparison has not converged are refined further, so a narrow
    diagonal interaction costs O(2**depth), rather than uniformly applying a
    very high tensor rule to the whole pair.  A child block already computed
    for its parent's error estimate is reused as its own coarse estimate.

    The error denominator is the sum of child-block norms, rather than the
    norm of their possibly cancelling sum.  This prevents a physical null in
    one 2x2 block from making the convergence test spuriously permissive.
    Failure at ``max_depth`` is explicit: silently accepting an unresolved
    close-gap interaction can produce a small linear-system residual for the
    wrong discrete operator.
    """

    zero = np.zeros((2, 2), dtype=np.complex128)

    def evaluate(
        obs_interval: 'Tuple[float, float]',
        src_interval: 'Tuple[float, float]',
    ) -> 'Tuple[np.ndarray, np.ndarray]':
        return _integrate_linear_pair_box_sk_vectorized(
            obs_elem=obs_elem,
            src_elem=src_elem,
            k0=k0,
            obs_normal_deriv=obs_normal_deriv,
            obs_interval=obs_interval,
            src_interval=src_interval,
            obs_order=obs_order,
            src_order=src_order,
            compute_single_layer=compute_single_layer,
            compute_double_layer=compute_double_layer,
        )

    def relative_error(
        coarse: 'np.ndarray',
        children: 'List[np.ndarray]',
    ) -> 'float':
        # Do not pass ``start`` by keyword to the built-in sum.  Some Python
        # versions deployed on HPC systems expose the second argument only as
        # positional and raise ``TypeError: sum() takes no keyword arguments``
        # when a close panel pair reaches this adaptive path.
        fine = sum(children, zero.copy())
        scale_norm = sum(float(np.linalg.norm(block)) for block in children)
        floor = np.finfo(float).eps * max(
            1.0,
            float(obs_elem.length) * float(src_elem.length),
        )
        return float(np.linalg.norm(fine - coarse)) / max(scale_norm, floor)

    def recurse(
        obs_interval: 'Tuple[float, float]',
        src_interval: 'Tuple[float, float]',
        depth: 'int',
        coarse: 'Optional[Tuple[np.ndarray, np.ndarray]]' = None,
    ) -> 'Tuple[np.ndarray, np.ndarray]':
        coarse_s, coarse_k = coarse if coarse is not None else evaluate(
            obs_interval, src_interval
        )
        oa, ob = map(float, obs_interval)
        sa, sb = map(float, src_interval)
        om = 0.5 * (oa + ob)
        sm = 0.5 * (sa + sb)
        child_intervals = [
            ((oa, om), (sa, sm)),
            ((oa, om), (sm, sb)),
            ((om, ob), (sa, sm)),
            ((om, ob), (sm, sb)),
        ]
        child_blocks = [evaluate(oi, si) for oi, si in child_intervals]
        s_children = [block[0] for block in child_blocks]
        k_children = [block[1] for block in child_blocks]
        err_s = (
            relative_error(coarse_s, s_children)
            if compute_single_layer else 0.0
        )
        err_k = (
            relative_error(coarse_k, k_children)
            if compute_double_layer else 0.0
        )
        error = max(err_s, err_k)
        if error <= float(rtol):
            return (
                sum(s_children, zero.copy()),
                sum(k_children, zero.copy()),
            )
        if depth >= int(max_depth):
            gap_ratio = float(np.linalg.norm(obs_elem.center - src_elem.center)) / max(
                float(obs_elem.length), float(src_elem.length), EPS
            )
            raise FloatingPointError(
                "Separated-near Galerkin quadrature did not converge: "
                f"panel pair ({obs_elem.panel_index}, {src_elem.panel_index}), "
                f"center-gap/length={gap_ratio:.6g}, estimated relative block "
                f"error={error:.3e} after depth {depth}. Refine the boundary "
                "mesh or increase NEAR_PAIR_QUADRATURE_MAX_DEPTH."
            )

        s_total = zero.copy()
        k_total = zero.copy()
        for (oi, si), child in zip(child_intervals, child_blocks):
            child_s, child_k = recurse(
                oi, si, depth + 1, coarse=child
            )
            s_total += child_s
            k_total += child_k
        return s_total, k_total

    return recurse((0.0, 1.0), (0.0, 1.0), depth=0)

_TANGENT_OUTER = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.complex128)

def _hypersingular_block_from_s_block(
    s_block: 'np.ndarray',
    k0: 'Union[complex, float]',
    n_obs: 'np.ndarray',
    n_src: 'np.ndarray',
    obs_length: 'float',
    src_length: 'float',
) -> 'np.ndarray':
    """
    Compute the 2x2 hypersingular D block from the single-layer S block via Maue identity.

    The Maue regularisation recasts the hypersingular kernel integral as:
        D_ij = -k^2 (n_obs . n_src) S_ij
             + (1/(L_obs*L_src)) * tangent_outer_ij * sum(S_block)

    where tangent_outer = [[1,-1],[-1,1]] encodes the linear shape-function
    tangential derivatives.  This avoids all hypersingular quadrature.
    """

    k2 = complex(k0) ** 2
    n_dot_n = float(np.dot(n_obs, n_src))
    raw_integral = complex(np.sum(s_block))
    denom = max(float(obs_length) * float(src_length), EPS * EPS)
    return -k2 * n_dot_n * s_block + _TANGENT_OUTER * (raw_integral / denom)

# --- Batched far-pair quadrature engine -------------------------------------
#
# Assembling the dense boundary-integral operators is the whole cost of a large
# 2-D solve: the linear solve is O(N^3) with threaded BLAS behind it, while the
# assembly is O(N^2 Q^2) evaluations of a Hankel function.  The engine below
# keeps the mathematics of the original element-pair quadrature intact -- same
# quadrature rule, same far/near split, same near-pair Duffy treatment -- and
# only changes how the work is staged:
#
#   * Element pairs are processed in cache-sized tiles instead of as whole
#     N_elem x N_elem arrays.  The previous formulation held eight complex
#     N_elem^2 accumulators plus roughly six N_elem^2 temporaries live at once
#     (3.2 GB + 1.2 GB at 5 000 elements).  That capped how many solves fit on
#     a node far more tightly than the matrices themselves do, and made every
#     quadrature point a DRAM round trip.
#   * The kernel is evaluated once per unordered element pair.  G is symmetric
#     and the Galerkin block of the transposed pair is the transpose of the
#     block already computed, so the strictly-upper tiles cover everything.
#     The double-layer kernels are not symmetric, but they share the same
#     H_1^(2)(kr), which is what the time actually goes into -- so the
#     transposed projection is formed from the same Bessel evaluation.
#     This halves the dominant cost outright.
#   * The Hankel evaluation takes a real-argument fast path.  G = (j/4)H_0^(2)
#     and its normal derivative have closed real/imaginary parts in terms of
#     J_n/Y_n, so a real wavenumber needs no complex argument array and none of
#     the all-finite / all-real guard scans `_stable_hankel2_array` must run to
#     stay safe for arbitrary complex input.
#   * The far mask is applied once to the finished tile accumulators instead of
#     to every kernel evaluation.  Masked-out entries stay finite (distances
#     are floored at EPS), so accumulating and then discarding them is exact.
#
# Tile size and thread count are tunable because the right values depend on how
# the node is packed: one solve per core wants small tiles because the L3 slice
# is shared, while a handful of large solves wants big tiles and threads.

_ASSEMBLY_TILE_TARGET_BYTES = 24 * 1024 * 1024


def _env_positive_int(name: 'str', default: 'int') -> 'int':
    try:
        value = int(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


_ASSEMBLY_THREADS = _env_positive_int("GHOST_ASSEMBLY_THREADS", 1)
_ASSEMBLY_TILE = _env_positive_int("GHOST_ASSEMBLY_TILE", 0)

# Opt-in override for the FAR single-/double-layer quadrature order only.
#
# Roughly half of a large assembly is Hankel evaluations, and their count is
# exactly N_elem^2 * Q^2.  The default Q = 8 is inherited from the near-field
# rule, where it is needed; across a far pair -- separated by at least
# `far_ratio` element lengths, so with a smooth non-singular integrand -- it is
# heavily over-resolved, and a lower order costs (Q_new/8)^2 of the time.
#
# This is left OFF by default because it changes computed values: it is a
# different quadrature rule, not a faster evaluation of the same one.  Turning
# it on is a legitimate engineering trade, but it has to be validated for the
# geometries in question -- run tests/measure_far_quadrature.py, which reports
# the RCS shift against the default rule, and keep the mesh-convergence gate
# in the loop.  A solve that used the override says so in its warnings, so a
# published field always carries the fact.
_FAR_QUAD_ORDER = _env_positive_int("GHOST_FAR_QUAD_ORDER", 0)

# Compact the source axis when the active masks select less than this fraction
# of the mesh.  Compaction and the transposed-pair shortcut are mutually
# exclusive (one needs the axes to differ, the other needs them to match), and
# they break even at one half.  Settable to 0 to force the full-width path,
# which must produce identical matrices -- tests/test_assembly_equivalence.py
# checks exactly that.
_ASSEMBLY_COMPACT_BELOW = 0.5


def set_assembly_compaction(fraction: 'float') -> 'None':
    """Set the active-fraction threshold below which the source axis compacts."""

    global _ASSEMBLY_COMPACT_BELOW
    _ASSEMBLY_COMPACT_BELOW = float(fraction)


def set_far_quadrature_order(order: 'int') -> 'None':
    """Override the far-pair quadrature order (0 restores the default rule)."""

    global _FAR_QUAD_ORDER
    _FAR_QUAD_ORDER = max(0, int(order))


def set_assembly_threads(count: 'int') -> 'None':
    """
    Set how many threads tiled operator assembly may use (1 = serial).

    Assembly tiles are independent and the heavy numpy/SciPy ufuncs inside them
    release the GIL, so this scales usefully when a node has more cores than
    concurrent solves.  When a run has at least one unit per core, leave it at
    1 and let the process pool own the parallelism -- threads and processes
    competing for the same cores is strictly worse than either alone.
    """

    global _ASSEMBLY_THREADS
    _ASSEMBLY_THREADS = max(1, int(count))


def get_assembly_threads() -> 'int':
    """Current tiled-assembly thread count."""

    return int(_ASSEMBLY_THREADS)


def _assembly_tile_size(nelems: 'int', bytes_per_entry: 'int') -> 'int':
    """Pick an element-tile edge so one tile's working set stays cache-sized.

    When assembly threads are enabled the tile is also capped so there are
    several observation blocks per thread: blocks are the unit of parallelism,
    and the symmetric traversal makes the first block the most expensive (it
    pairs with every later one), so a handful of coarse blocks would both
    starve threads and hand them wildly unequal work.
    """

    if _ASSEMBLY_TILE > 0:
        return max(1, min(int(nelems), _ASSEMBLY_TILE))
    if nelems <= 192:
        return int(nelems)
    entries = float(_ASSEMBLY_TILE_TARGET_BYTES) / float(max(1, int(bytes_per_entry)))
    tile = int(math.sqrt(max(1.0, entries)))
    tile = max(128, min(1024, min(int(nelems), tile)))
    if _ASSEMBLY_THREADS > 1:
        per_thread_blocks = 4
        tile = min(
            tile,
            max(64, int(math.ceil(nelems / (per_thread_blocks * _ASSEMBLY_THREADS)))),
        )
    return max(1, min(int(nelems), tile))


# Distance- and wavelength-graded far quadrature.
#
# The far pass used a fixed order (8) for every well-separated pair.  Measuring
# the order actually needed to hold a Galerkin block to 1e-12 relative --
# against an order-24 reference, worst case over collinear, parallel,
# perpendicular, and oblique pair orientations -- gives:
#
#     kL \ r      3    4    5    6    8   10   14   20   30   50
#     0.10        6    6    5    5    5    5    4    4    4    4
#     0.31        6    6    5    5    5    5    5    5    5    5
#     1.00        7    6    6    6    6    6    6    6    6    6
#     2.00        7    7    7    7    7    7    7    7    7    7
#     4.00        9    9    9    9    9    9    9    9    9    9
#
# where r is the centre separation in element lengths and kL is the element
# length in radians.  Two things fall out.  Separation barely matters once a
# pair is far at all -- what sets the order is how many wavelengths the element
# spans, because that is what the integrand oscillates over.  And at kL >= 4
# the fixed order 8 is *under*-resolved: a mesh that coarse needs 9.
#
# The table below carries a +1 margin over those measurements, and the result
# is clamped to the caller's configured order, so grading only ever reduces
# work.  Raising the order past what the caller asked for would change
# published values on coarse meshes -- worth doing, but not silently.
#
# The order is chosen per tile from the tile's *worst* pair, so it costs two
# reductions per tile rather than a branch per element pair.

# Each row is calibrated at its band's UPPER kL edge -- the worst case inside
# the band -- over five pair orientations, so the band itself is the margin.
# The r < 5 column carries one extra order as a hedge: centre separation can
# understate how close two elements actually come when a mesh is irregular.
_FAR_ORDER_TABLE = (
    # (kL upper bound, order for r < 5, for r < 10, for r >= 10)
    (0.15, 6, 5, 5),
    (0.50, 7, 6, 5),
    (1.50, 7, 6, 6),
    (3.00, 8, 8, 8),
    (float("inf"), 10, 10, 10),
)

_FAR_GRADED = _env_positive_int("GHOST_FAR_GRADED", 1) != 0


def set_far_quadrature_grading(enabled: 'bool') -> 'None':
    """Enable/disable per-tile far-quadrature grading (default on).

    Grading only reduces the order below what the caller configured, and only
    where a calibrated table says the reduction costs under 1e-12 relative on
    the element-pair block, so turning it off should change nothing that
    matters.  The switch exists to make that testable.
    """

    global _FAR_GRADED
    _FAR_GRADED = bool(enabled)


def _graded_far_order(kl_max: 'float', ratio_min: 'float', cap: 'int') -> 'int':
    """Quadrature order for a tile whose worst far pair has these parameters."""

    if not _FAR_GRADED:
        return int(cap)
    for bound, near, mid, far in _FAR_ORDER_TABLE:
        if kl_max <= bound:
            if ratio_min < 5.0:
                order = near
            elif ratio_min < 10.0:
                order = mid
            else:
                order = far
            return max(2, min(int(cap), int(order)))
    return int(cap)


def _wavenumber_is_real(k0: 'Union[complex, float]') -> 'bool':
    value = complex(k0)
    return value.imag == 0.0 and value.real > 0.0


def _axpy_into(acc: 'np.ndarray', src: 'np.ndarray', coeff: 'float',
               scratch: 'np.ndarray') -> 'None':
    """acc += coeff * src, in place, without allocating a temporary.

    Deliberately not scipy's BLAS axpy: its f2py wrapper carries a fixed
    per-call cost of several milliseconds, which swamps a cache-sized tile and
    only pays for itself on arrays of millions of elements.
    """

    np.multiply(src, coeff, out=scratch)
    np.add(acc, scratch, out=acc)


def _far_kernel_argument(k0: 'Union[complex, float]', dist: 'np.ndarray',
                         out: 'np.ndarray') -> 'None':
    """kr = k0 * dist on the real fast path, floored where the scalar
    evaluators floor it so the two agree entry for entry."""

    np.multiply(dist, complex(k0).real, out=out)
    np.maximum(out, 1e-12, out=out)


def _far_green_into(
    k0: 'Union[complex, float]',
    real_k: 'bool',
    dist: 'np.ndarray',
    kr: 'np.ndarray',
    scratch: 'np.ndarray',
    out: 'np.ndarray',
) -> 'None':
    """out <- (j/4) H_0^(2)(k0 r) over a whole tile.

    For real k0 this is (1/4)(Y_0(kr) + j J_0(kr)), so the two real Bessel
    evaluations write straight into the halves of the complex output buffer.
    """

    if real_k and _SCIPY_SPECIAL is not None:
        _SCIPY_SPECIAL.y0(kr, out=scratch)
        np.multiply(scratch, 0.25, out=out.real)
        _SCIPY_SPECIAL.j0(kr, out=scratch)
        np.multiply(scratch, 0.25, out=out.imag)
        return
    arg = np.asarray(complex(k0) * dist, dtype=np.complex128)
    np.multiply(_hankel2_0_array(arg), 0.25j, out=out)


def _far_hankel1_into(
    k0: 'Union[complex, float]',
    real_k: 'bool',
    dist: 'np.ndarray',
    kr: 'np.ndarray',
    scratch: 'np.ndarray',
    out: 'np.ndarray',
) -> 'None':
    """out <- (j/4) k0 H_1^(2)(k0 r) over a whole tile.

    Kept separate from the projection so both orientations of an element pair
    can reuse one Bessel evaluation -- the normal-derivative kernels differ
    only by which normal the displacement is projected onto.
    """

    if real_k and _SCIPY_SPECIAL is not None:
        coeff = 0.25 * complex(k0).real
        _SCIPY_SPECIAL.y1(kr, out=scratch)
        np.multiply(scratch, coeff, out=out.real)
        _SCIPY_SPECIAL.j1(kr, out=scratch)
        np.multiply(scratch, coeff, out=out.imag)
        return
    arg = np.asarray(complex(k0) * dist, dtype=np.complex128)
    np.multiply(_hankel2_1_array(arg), 0.25j * complex(k0), out=out)


def _run_tiled_obs_blocks(
    nelems: 'int',
    tile: 'int',
    body: 'Callable[[int, int], None]',
) -> 'None':
    """Run ``body(i0, i1)`` over every observation tile, threaded when asked."""

    starts = list(range(0, nelems, tile))
    workers = min(int(_ASSEMBLY_THREADS), len(starts))
    if workers <= 1:
        for i0 in starts:
            body(i0, min(i0 + tile, nelems))
        return
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda i0: body(i0, min(i0 + tile, nelems)), starts))


def _expand_near_chunks(
    chunks: 'List[Tuple[int, np.ndarray, int, np.ndarray, bool]]',
) -> 'Tuple[np.ndarray, np.ndarray]':
    """Flatten recorded near-pair tiles into ascending (obs, src) index arrays.

    Each chunk carries the tile's global source-element ids, because a masked
    assembly compacts the source axis and the recorded column is an index into
    that compacted axis rather than into the mesh.

    Sorting here is what keeps the near-field accumulation order independent of
    how tiles were scheduled, so a threaded assembly reproduces a serial one
    exactly rather than only to within floating-point reassociation.
    """

    if not chunks:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    obs_parts: 'List[np.ndarray]' = []
    src_parts: 'List[np.ndarray]' = []
    for row_base, src_global, ncols, flat, transposed in chunks:
        rows = flat // ncols
        cols = src_global[flat % ncols]
        if transposed:
            obs_parts.append(cols)
            src_parts.append(row_base + rows)
        else:
            obs_parts.append(row_base + rows)
            src_parts.append(cols)
    obs_idx = np.concatenate(obs_parts)
    src_idx = np.concatenate(src_parts)
    order = np.lexsort((src_idx, obs_idx))
    return obs_idx[order], src_idx[order]


def _warn_far_quadrature_override(materials: 'MaterialLibrary') -> 'None':
    """Record a far-quadrature override in the solve's own warnings.

    The override changes computed values, so a field produced under it must
    say so wherever it travels; the warning list is copied into the .grim
    audit, which is the record that survives the run directory.
    """

    if _FAR_QUAD_ORDER <= 0:
        return
    materials.warn_once(
        f"Far-pair quadrature order overridden to {_FAR_QUAD_ORDER} "
        "(default 8) via GHOST_FAR_QUAD_ORDER / set_far_quadrature_order. "
        "Well-separated element pairs are integrated with a coarser rule "
        "than the shipped default, so these values are NOT bit-comparable "
        "with default-rule results. The mesh-convergence certificate still "
        "applies to the discretization, not to this quadrature choice."
    )


@timed_stage("operators")
@cached_operator("SK")
def _assemble_linear_operator_matrices_multi(
    mesh: 'LinearMesh',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    source_element_masks: 'Sequence[Optional[np.ndarray]]',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    far_ratio: 'float' = 3.0,
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
    compute_double_layer_many: 'Optional[Sequence[bool]]' = None,
    single_layer_observation_coefficients: 'Optional[np.ndarray]' = None,
    single_layer_observation_coefficients_many: 'Optional[Sequence[Optional[np.ndarray]]]' = None,
) -> 'List[Tuple[np.ndarray, np.ndarray]]':
    """
    Assemble S and K/K' for several source-element masks in ONE traversal.

    The masks select which source elements contribute, but they are applied to
    the finished tile accumulators -- the quadrature itself does not depend on
    them.  Assembling each mask separately therefore repeats every Hankel
    evaluation, which is most of the cost of a solve.  The multi-region
    formulation asks for exactly this: one operator per (region, interface
    side), where both sides of a region share a wavenumber and differ only in
    which elements are active.

    Returns one (S, K) pair per mask, in the order given.  A mask selecting no
    element gets zero matrices without costing anything.  When
    ``single_layer_observation_coefficients`` is supplied, every returned S
    matrix uses that piecewise-constant coefficient inside the observation
    integral. ``single_layer_observation_coefficients_many`` instead supplies
    one coefficient vector per source mask.  This permits several differently
    weighted S matrices to share exactly the same kernel/quadrature traversal.

        S_c[i,j] = sum_e integral_e phi_i(x) c_e (S phi_j)(x) ds,

    while K/K' remains unweighted.  This is required for spatially varying
    Robin and sheet coefficients: multiplying completed rows by a nodal
    average is not the same weak form.

    Two passes, as before:
    1. Far interactions: batched numpy quadrature over cache-sized element
       tiles, one kernel evaluation per unordered pair.
    2. Near interactions: per-element-pair recursive/Duffy quadrature over the
       union of all masks, followed by deterministic distribution to every
       requested output. Overlapping weighted/unweighted masks therefore do
       not duplicate singular-kernel evaluation.
    """
    far_green, far_hankel = _far_green_into, _far_hankel1_into
    if current_state() is not None:
        from cpu_kernels import select_far_kernels
        far_green, far_hankel = select_far_kernels(mesh, k0, far_green, far_hankel)

    nnodes = len(mesh.nodes)
    elements = list(mesh.elements)
    nelems = len(elements)
    n_masks = len(source_element_masks)
    if n_masks == 0:
        raise ValueError("At least one source-element mask must be requested.")
    if compute_double_layer_many is None:
        want_k_masks = [bool(compute_double_layer)] * n_masks
    else:
        want_k_masks = [bool(value) for value in compute_double_layer_many]
        if len(want_k_masks) != n_masks:
            raise ValueError(
                "compute_double_layer_many length must match "
                "source_element_masks."
            )
    if not bool(compute_single_layer) and not any(want_k_masks):
        raise ValueError("At least one linear operator must be requested.")
    # Preserve the long-standing shaped-zero return contract for a skipped
    # operator without allocating and zero-filling another dense N-by-N
    # array.  The zero-stride views are read-only, which is intentional: a
    # disabled output is a result placeholder, never an assembly target.
    zero_view = np.broadcast_to(
        np.zeros((), dtype=np.complex128), (nnodes, nnodes)
    )
    s_mats = [
        np.zeros((nnodes, nnodes), dtype=np.complex128)
        if compute_single_layer else zero_view
        for _ in range(n_masks)
    ]
    k_mats = [
        np.zeros((nnodes, nnodes), dtype=np.complex128)
        if want_k_masks[index] else zero_view
        for index in range(n_masks)
    ]
    if not elements:
        return list(zip(s_mats, k_mats))

    src_masks = []
    for mask in source_element_masks:
        if mask is None:
            src_masks.append(np.ones(nelems, dtype=bool))
            continue
        resolved = np.asarray(mask, dtype=bool).reshape(-1)
        if resolved.size != nelems:
            raise ValueError("source_element_mask length must match mesh element count.")
        src_masks.append(resolved)

    if (
        single_layer_observation_coefficients is not None
        and single_layer_observation_coefficients_many is not None
    ):
        raise ValueError(
            "Supply either shared or per-mask single-layer observation "
            "coefficients, not both."
        )

    def _validated_slp_coeff(value):
        if value is None:
            return None
        resolved = np.asarray(value, dtype=np.complex128).reshape(-1)
        if resolved.size != nelems:
            raise ValueError(
                "single_layer_observation_coefficients length must match "
                "mesh element count."
            )
        if not np.all(
            np.isfinite(resolved.real) & np.isfinite(resolved.imag)
        ):
            raise ValueError(
                "single-layer observation coefficients must all be finite."
            )
        return resolved

    if single_layer_observation_coefficients_many is not None:
        supplied_coeffs = list(single_layer_observation_coefficients_many)
        if len(supplied_coeffs) != n_masks:
            raise ValueError(
                "single_layer_observation_coefficients_many length must "
                "match source_element_masks."
            )
        slp_obs_coeffs = [
            _validated_slp_coeff(value) for value in supplied_coeffs
        ]
    else:
        shared_coeff = _validated_slp_coeff(
            single_layer_observation_coefficients
        )
        slp_obs_coeffs = [shared_coeff] * n_masks
    # An empty mask contributes nothing; drop it from the traversal entirely
    # rather than paying a tile sweep to produce zeros.
    active = [index for index, mask in enumerate(src_masks) if bool(np.any(mask))]
    if not active:
        return list(zip(s_mats, k_mats))

    centers = np.stack([e.center for e in elements], axis=0)
    lengths = np.asarray([e.length for e in elements], dtype=float)
    node_ids = np.asarray([e.node_ids for e in elements], dtype=int)  # (nelems, 2)
    p0_arr = np.stack([e.p0 for e in elements], axis=0)               # (nelems, 2)
    seg_arr = np.stack([e.p1 - e.p0 for e in elements], axis=0)       # (nelems, 2)
    normals_arr = np.stack([e.normal for e in elements], axis=0)      # (nelems, 2)

    want_s = bool(compute_single_layer)
    want_k = any(want_k_masks)

    # The near pass keeps the requested orders; only the far quadrature honours
    # the override, because that is the only place the integrand is smooth
    # enough for a lower order to be defensible.
    far_obs_order = _FAR_QUAD_ORDER or int(obs_order)
    far_src_order = _FAR_QUAD_ORDER or int(src_order)
    # Grading picks a per-tile order, so the rule is cached rather than
    # precomputed once; tile quadrature points are built from p0/segment inside
    # the tile, which is cheap and avoids holding an (nelems, Q, 2) array per
    # order that might be used.
    _rule_cache: 'Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]' = {}

    def _rule(order: 'int') -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]':
        cached = _rule_cache.get(order)
        if cached is None:
            nodes, weights = _get_quadrature(max(2, int(order)))
            phi = np.array([_linear_shape_values(float(t)) for t in nodes])
            cached = (np.asarray(nodes, dtype=float),
                      np.asarray(weights, dtype=float), phi)
            _rule_cache[order] = cached
        return cached

    # Source-axis compaction.  A mask that selects a minority of the elements
    # -- normal for a coating or a small dielectric region -- otherwise pays a
    # full-width sweep whose result is then masked away.  Restricting the
    # source axis to the union of the active masks makes the work proportional
    # to what is actually wanted.
    #
    # It costs the transposed-pair shortcut, which needs both axes to index the
    # same set, so it is only worth it when the active set is under half the
    # mesh: compacted work is nelems * n_active, symmetric work nelems^2 / 2.
    union_mask = src_masks[active[0]].copy()
    for index in active[1:]:
        union_mask |= src_masks[index]
    n_active = int(np.count_nonzero(union_mask))
    compact = n_active < _ASSEMBLY_COMPACT_BELOW * nelems
    src_sel = (
        np.flatnonzero(union_mask) if compact
        else np.arange(nelems, dtype=np.int64)
    )
    n_src = int(src_sel.size)
    src_centers = centers[src_sel]
    src_lengths = lengths[src_sel]
    src_node_ids = node_ids[src_sel]
    src_normals = normals_arr[src_sel]

    # The transposed-pair shortcut needs both directions to share one
    # quadrature rule, and both axes to index the same elements.
    symmetric = (
        int(max(2, far_obs_order)) == int(max(2, far_src_order))
        and not compact
    )
    abs_k = abs(complex(k0))

    n_acc = 4 * (int(want_s) + (2 if want_k else 0))
    n_kernel = int(want_s) + (2 if want_k else 0)
    tile = _assembly_tile_size(nelems, 16 * (n_acc + n_kernel + 1) + 8 * 7)

    real_k = _wavenumber_is_real(k0)
    # `_dgreen_dn_obs` carries the minus sign; `_dgreen_dn_src` does not.
    dgreen_sign = -1.0 if obs_normal_deriv else 1.0
    # Near-pair geometry is independent of the output mask. Record the union
    # once, evaluate each singular/near-singular block once, then distribute
    # that block to every output whose source mask contains the source element.
    # This matters when weighted and unweighted outputs intentionally overlap.
    near_chunks: 'List[Tuple[int, np.ndarray, int, np.ndarray, bool]]' = []
    write_lock = threading.Lock()

    def _far_pass(i0: 'int', i1: 'int') -> 'None':
        mb = i1 - i0
        obs_slice = slice(i0, i1)
        obs_nid = node_ids[obs_slice]
        obs_norm = normals_arr[obs_slice]
        obs_len = lengths[obs_slice]
        obs_p0 = p0_arr[obs_slice]
        obs_seg = seg_arr[obs_slice]
        obs_ctr = centers[obs_slice]
        obs_pts_cache: 'Dict[int, np.ndarray]' = {}
        local_near: 'List[Tuple[int, np.ndarray, int, np.ndarray, bool]]' = []

        for j0 in range(i0 if symmetric else 0, n_src, tile):
            j1 = min(j0 + tile, n_src)
            nb = j1 - j0
            mirrored = symmetric and j0 > i0
            src_slice = slice(j0, j1)
            src_global = src_sel[src_slice]
            src_nid = src_node_ids[src_slice]
            src_len = src_lengths[src_slice]
            src_norm = src_normals[src_slice]
            # Orientation-independent part of the far predicate.
            mdx = obs_ctr[:, 0][:, None] - src_centers[src_slice, 0][None, :]
            mdy = obs_ctr[:, 1][:, None] - src_centers[src_slice, 1][None, :]
            centre_dist = np.sqrt(mdx * mdx + mdy * mdy)
            scale = np.maximum(np.maximum(obs_len[:, None], src_len[None, :]), EPS)
            far_sym = (centre_dist / scale) >= float(far_ratio)
            far_sym &= ~(
                (obs_nid[:, 0][:, None] == src_nid[None, :, 0])
                | (obs_nid[:, 0][:, None] == src_nid[None, :, 1])
                | (obs_nid[:, 1][:, None] == src_nid[None, :, 0])
                | (obs_nid[:, 1][:, None] == src_nid[None, :, 1])
            )
            # Self pairs are never far. With a compacted source axis the two
            # axes no longer share an index, so match on the global element id.
            np.logical_and(
                far_sym,
                np.arange(i0, i1)[:, None] != src_global[None, :],
                out=far_sym,
            )

            # Which elements are active is the only mask-dependent term, and
            # the only orientation-dependent one: everything above is shared.
            far_ij = {}
            far_ji = {}
            any_ij = False
            any_ji = False
            for mi in active:
                mask = src_masks[mi]
                src_msk = mask[src_global]
                fij = far_sym & src_msk[None, :]
                far_ij[mi] = fij
                any_ij = any_ij or bool(fij.any())
                if mirrored:
                    obs_msk = mask[obs_slice]
                    fji = far_sym & obs_msk[:, None]
                    far_ji[mi] = fji
                    any_ji = any_ji or bool(fji.any())

            near_union = (~far_sym) & union_mask[src_global][None, :]
            flat = np.flatnonzero(near_union.ravel())
            if flat.size:
                local_near.append((i0, src_global, nb, flat, False))
            if mirrored:
                near_union_t = (~far_sym) & union_mask[obs_slice][:, None]
                flat_t = np.flatnonzero(near_union_t.ravel())
                if flat_t.size:
                    local_near.append((i0, src_global, nb, flat_t, True))

            if not (any_ij or any_ji):
                continue

            # Worst far pair in this tile decides the order for all of it.
            any_far = far_sym
            ratio_min = float(np.min(
                np.where(any_far, centre_dist / scale, np.inf)
            )) if any_far.any() else float("inf")
            kl_max = abs_k * float(max(obs_len.max(), src_len.max()))
            tile_order = _graded_far_order(kl_max, ratio_min, far_obs_order)
            t_obs_f, qw_obs, phi_obs_arr = _rule(tile_order)
            t_src_f, qw_src, phi_src_arr = _rule(tile_order)

            obs_pts = obs_pts_cache.get(tile_order)
            if obs_pts is None:
                obs_pts = (obs_p0[:, None, :]
                           + t_obs_f[None, :, None] * obs_seg[:, None, :])
                obs_pts_cache[tile_order] = obs_pts
            src_p0 = p0_arr[src_global]
            src_seg = seg_arr[src_global]
            src_pts = src_p0[:, None, :] + t_src_f[None, :, None] * src_seg[:, None, :]
            acc_s = (
                [np.zeros((mb, nb), dtype=np.complex128) for _ in range(4)]
                if want_s else None
            )
            acc_k = (
                [np.zeros((mb, nb), dtype=np.complex128) for _ in range(4)]
                if want_k else None
            )
            acc_kt = (
                [np.zeros((mb, nb), dtype=np.complex128) for _ in range(4)]
                if (want_k and mirrored) else None
            )
            g_buf = np.empty((mb, nb), dtype=np.complex128) if want_s else None
            h1_buf = np.empty((mb, nb), dtype=np.complex128) if want_k else None
            dk_buf = np.empty((mb, nb), dtype=np.complex128) if want_k else None
            cscratch = np.empty((mb, nb), dtype=np.complex128)
            dx = np.empty((mb, nb), dtype=float)
            dy = np.empty((mb, nb), dtype=float)
            dist = np.empty((mb, nb), dtype=float)
            krbuf = np.empty((mb, nb), dtype=float)
            work = np.empty((mb, nb), dtype=float)
            proj = np.empty((mb, nb), dtype=float) if want_k else None

            # Which normals each orientation projects the displacement onto.
            if obs_normal_deriv:
                n_ij = (obs_norm[:, 0][:, None], obs_norm[:, 1][:, None])
                n_ji = (src_norm[None, :, 0], src_norm[None, :, 1])
            else:
                n_ij = (src_norm[None, :, 0], src_norm[None, :, 1])
                n_ji = (obs_norm[:, 0][:, None], obs_norm[:, 1][:, None])

            for qi in range(t_obs_f.size):
                r_obs = obs_pts[:, qi, :]
                w_obs_qi = float(qw_obs[qi])
                phi_o = phi_obs_arr[qi]

                for qj in range(t_src_f.size):
                    r_src = src_pts[:, qj, :]
                    w_src_qj = float(qw_src[qj])
                    phi_s = phi_src_arr[qj]

                    np.subtract(r_obs[:, 0][:, None], r_src[None, :, 0], out=dx)
                    np.subtract(r_obs[:, 1][:, None], r_src[None, :, 1], out=dy)
                    np.multiply(dx, dx, out=dist)
                    np.multiply(dy, dy, out=work)
                    np.add(dist, work, out=dist)
                    np.sqrt(dist, out=dist)
                    np.maximum(dist, EPS, out=dist)
                    if real_k:
                        _far_kernel_argument(k0, dist, krbuf)

                    if want_s:
                        far_green(k0, real_k, dist, krbuf, work, g_buf)
                    if want_k:
                        far_hankel(k0, real_k, dist, krbuf, work, h1_buf)

                    w = w_obs_qi * w_src_qj
                    if want_k:
                        np.multiply(dx, n_ij[0], out=proj)
                        np.multiply(dy, n_ij[1], out=work)
                        np.add(proj, work, out=proj)
                        np.divide(proj, dist, out=proj)
                        np.multiply(h1_buf, proj, out=dk_buf)
                        if dgreen_sign < 0.0:
                            np.negative(dk_buf, out=dk_buf)
                    for a in range(2):
                        coeff_a = w * float(phi_o[a])
                        for b in range(2):
                            coeff = coeff_a * float(phi_s[b])
                            if acc_s is not None:
                                _axpy_into(acc_s[2 * a + b], g_buf, coeff, cscratch)
                            if acc_k is not None:
                                _axpy_into(acc_k[2 * a + b], dk_buf, coeff, cscratch)

                    if acc_kt is not None:
                        # Same point pair, roles swapped: the displacement
                        # reverses and the other element's normal is used, so
                        # only the projection is recomputed.
                        np.multiply(dx, n_ji[0], out=proj)
                        np.multiply(dy, n_ji[1], out=work)
                        np.add(proj, work, out=proj)
                        np.divide(proj, dist, out=proj)
                        np.multiply(h1_buf, proj, out=dk_buf)
                        if dgreen_sign > 0.0:
                            np.negative(dk_buf, out=dk_buf)
                        for a in range(2):
                            coeff_a = w * float(phi_s[a])
                            for b in range(2):
                                _axpy_into(
                                    acc_kt[2 * a + b], dk_buf,
                                    coeff_a * float(phi_o[b]), cscratch,
                                )

            len_prod = obs_len[:, None] * src_len[None, :]
            with write_lock:
                for mi in active:
                    fij = far_ij[mi]
                    if fij.any():
                        scale_ij = len_prod * fij
                        slp_obs_coeff = slp_obs_coeffs[mi]
                        if slp_obs_coeff is not None:
                            scale_s_ij = (
                                scale_ij
                                * slp_obs_coeff[obs_slice][:, None]
                            )
                        else:
                            scale_s_ij = scale_ij
                        for a in range(2):
                            rows = obs_nid[:, a][:, None]
                            for b in range(2):
                                cols = src_nid[None, :, b]
                                if acc_s is not None:
                                    np.add.at(s_mats[mi], (rows, cols),
                                              acc_s[2 * a + b] * scale_s_ij)
                                if acc_k is not None and want_k_masks[mi]:
                                    np.add.at(k_mats[mi], (rows, cols),
                                              acc_k[2 * a + b] * scale_ij)
                    fji = far_ji.get(mi)
                    if fji is not None and fji.any():
                        scale_ji = len_prod * fji
                        slp_obs_coeff = slp_obs_coeffs[mi]
                        if slp_obs_coeff is not None:
                            scale_s_ji = (
                                scale_ji
                                * slp_obs_coeff[src_global][None, :]
                            )
                        else:
                            scale_s_ji = scale_ji
                        for a in range(2):
                            rows = src_nid[:, a][:, None]
                            for b in range(2):
                                cols = obs_nid[None, :, b]
                                if acc_s is not None:
                                    # S is symmetric: block (j,i)[a][b] is the
                                    # transpose of block (i,j)[b][a].
                                    np.add.at(s_mats[mi], (rows, cols),
                                              (acc_s[2 * b + a] * scale_s_ji).T)
                                if acc_kt is not None and want_k_masks[mi]:
                                    np.add.at(k_mats[mi], (rows, cols),
                                              (acc_kt[2 * a + b] * scale_ji).T)

        if local_near:
            with write_lock:
                near_chunks.extend(local_near)

    _run_tiled_obs_blocks(nelems, tile, _far_pass)

    # --- Pass 2: Near interactions (self, touching, close pairs) ---
    obs_idx, src_idx = _expand_near_chunks(near_chunks)

    # Fixed-rule separated-near pairs have identical tensor-Gauss structure.
    # Batch their Hankel calls by rule order, but retain the original sorted
    # scatter order below so global matrix accumulation stays deterministic.
    fixed_positions_by_order: 'Dict[int, List[int]]' = {}
    for pos in range(obs_idx.size):
        obs_elem_eval = elements[int(obs_idx[pos])]
        src_elem_eval = elements[int(src_idx[pos])]
        if obs_elem_eval.panel_index == src_elem_eval.panel_index:
            continue
        shared = _linear_shared_interval_endpoint_info(
            obs_elem_eval,
            (0.0, 1.0),
            src_elem_eval,
            (0.0, 1.0),
            tol=1.0e-9,
        )
        if shared is not None:
            continue
        distance = float(np.linalg.norm(obs_elem_eval.center - src_elem_eval.center))
        scale = max(obs_elem_eval.length, src_elem_eval.length, EPS)
        if distance / scale < 0.75:
            continue
        adapt_order, _ = _near_singular_scheme(distance, scale)
        tensor_order = max(
            int(max(obs_order, src_order)),
            min(16, int(max(5, adapt_order))),
        )
        fixed_positions_by_order.setdefault(tensor_order, []).append(pos)

    fixed_blocks: 'Dict[int, Tuple[np.ndarray, np.ndarray]]' = {}
    for tensor_order, positions in fixed_positions_by_order.items():
        # Bound the pair x Q x Q kernel arrays to roughly one million entries.
        batch_pairs = max(1, 1_000_000 // max(1, tensor_order * tensor_order))
        for start in range(0, len(positions), batch_pairs):
            selected = positions[start:start + batch_pairs]
            selected_arr = np.asarray(selected, dtype=np.int64)
            s_batch, k_batch = _integrate_linear_pairs_box_sk_batched(
                elements,
                obs_idx[selected_arr],
                src_idx[selected_arr],
                k0,
                obs_normal_deriv,
                tensor_order,
                compute_single_layer=compute_single_layer,
                compute_double_layer=want_k,
            )
            for local, original_pos in enumerate(selected):
                fixed_blocks[int(original_pos)] = (
                    s_batch[local], k_batch[local]
                )

    last_obs = -1
    obs_elem = None
    obs_ids = None
    for pos in range(obs_idx.size):
        obs_index = int(obs_idx[pos])
        src_index = int(src_idx[pos])
        if obs_index != last_obs:
            last_obs = obs_index
            obs_elem = elements[obs_index]
            obs_ids = np.asarray(obs_elem.node_ids, dtype=int)
        src_elem = elements[src_index]
        src_ids = src_elem.node_ids
        precomputed = fixed_blocks.get(pos)
        if precomputed is None:
            s_blk, k_blk = _sk_blocks_near_linear(
                obs_elem=obs_elem,
                src_elem=src_elem,
                k0=k0,
                obs_normal_deriv=obs_normal_deriv,
                obs_order=obs_order,
                src_order=src_order,
                compute_single_layer=compute_single_layer,
                compute_double_layer=want_k,
            )
        else:
            s_blk, k_blk = precomputed
        for mi in active:
            if not bool(src_masks[mi][src_index]):
                continue
            if compute_single_layer:
                slp_obs_coeff = slp_obs_coeffs[mi]
                coeff = (
                    complex(slp_obs_coeff[obs_index])
                    if slp_obs_coeff is not None else 1.0 + 0.0j
                )
                s_mats[mi][np.ix_(obs_ids, src_ids)] += coeff * s_blk
            if want_k_masks[mi]:
                k_mats[mi][np.ix_(obs_ids, src_ids)] += k_blk
    return list(zip(s_mats, k_mats))


def _assemble_linear_operator_matrices(
    mesh: 'LinearMesh',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    far_ratio: 'float' = 3.0,
    source_element_mask: 'Optional[np.ndarray]' = None,
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
    single_layer_observation_coefficients: 'Optional[np.ndarray]' = None,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """
    Assemble dense linear-Galerkin S and K/K' matrices on global nodal DOFs.

    ``compute_single_layer`` and ``compute_double_layer`` let formulation
    callers skip an operator they do not consume.  A zero matrix is returned
    for a skipped operator so the long-standing two-array return contract is
    preserved.

    Single-mask front end for `_assemble_linear_operator_matrices_multi`; a
    caller wanting several masks at one wavenumber should use that directly
    so the quadrature is shared.
    """

    return _assemble_linear_operator_matrices_multi(
        mesh=mesh,
        k0=k0,
        obs_normal_deriv=obs_normal_deriv,
        source_element_masks=[source_element_mask],
        obs_order=obs_order,
        src_order=src_order,
        far_ratio=far_ratio,
        compute_single_layer=compute_single_layer,
        compute_double_layer=compute_double_layer,
        single_layer_observation_coefficients=(
            single_layer_observation_coefficients
        ),
    )[0]

@timed_stage("near_and_hypersingular")
@cached_operator("D")
def _assemble_linear_hypersingular_matrix(
    mesh: 'LinearMesh',
    k0: 'Union[complex, float]',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    far_ratio: 'float' = 3.0,
    source_element_mask: 'Optional[np.ndarray]' = None,
) -> 'np.ndarray':
    """
    Assemble the hypersingular D operator via the Maue identity.

    D is computed element-by-element from single-layer S blocks:
        D_block = -k^2 (n_obs . n_src) S_block
                + tangent_outer / (L_obs * L_src) * sum(S_block)

    This avoids all hypersingular quadrature; the log singularity in S is handled
    by the existing Duffy transforms.

    The S blocks come from `_integrate_linear_pair_generic`, whose recursion
    bottoms out in a plain tensor-Gauss box for every pair that is neither
    singular, endpoint-touching, nor near-singular (centre separation under
    0.95 element lengths) -- all but O(N) of the N^2 pairs.  Evaluating those
    one Python call at a time made this the only genuinely O(N^2)-interpreted
    routine in the solver; they are batched here over the same tiles the
    single-layer assembly uses, at the same quadrature order the recursion
    would have chosen, leaving the per-pair path only the singular work.

    Both the S blocks and the Maue combination are symmetric under swapping the
    element pair, so only the upper tiles are evaluated.
    """
    far_green, far_hankel = _far_green_into, _far_hankel1_into
    if current_state() is not None:
        from cpu_kernels import select_far_kernels
        far_green, far_hankel = select_far_kernels(mesh, k0, far_green, far_hankel)

    nnodes = len(mesh.nodes)
    d_mat = np.zeros((nnodes, nnodes), dtype=np.complex128)
    elements = list(mesh.elements)
    nelems = len(elements)
    if not elements:
        return d_mat

    if source_element_mask is None:
        src_mask = np.ones(nelems, dtype=bool)
    else:
        src_mask = np.asarray(source_element_mask, dtype=bool).reshape(-1)
        if src_mask.size != nelems:
            raise ValueError("source_element_mask length must match mesh element count.")
    if not np.any(src_mask):
        return d_mat

    lengths = np.asarray([e.length for e in elements], dtype=float)
    node_ids = np.asarray([e.node_ids for e in elements], dtype=int)
    panel_index = np.asarray([e.panel_index for e in elements], dtype=int)
    p0_arr = np.stack([e.p0 for e in elements], axis=0)
    p1_arr = np.stack([e.p1 for e in elements], axis=0)
    seg_arr = p1_arr - p0_arr
    mid_arr = 0.5 * (p0_arr + p1_arr)
    normals_arr = np.stack([e.normal for e in elements], axis=0)

    # `_integrate_linear_pair_recursive` stops subdividing at ratio >= 0.95 and
    # then picks tensor_order = max(obs_order, src_order, min(16, adaptive)),
    # which saturates at 16 everywhere in that regime.
    box_order = max(int(obs_order), int(src_order), 16)
    qt, qw = _get_quadrature(max(2, box_order))
    phi_arr = np.array([_linear_shape_values(float(t)) for t in qt])
    t_f = np.asarray(qt, dtype=float)
    quad_pts = p0_arr[:, None, :] + t_f[None, :, None] * seg_arr[:, None, :]

    tile = _assembly_tile_size(nelems, 16 * 7 + 8 * 5)
    real_k = _wavenumber_is_real(k0)
    k2 = complex(k0) ** 2
    touch_tol_sq = 1.0e-12 ** 2
    scalar_chunks: 'List[Tuple[int, int, int, np.ndarray, bool]]' = []
    write_lock = threading.Lock()

    def _batch_pass(i0: 'int', i1: 'int') -> 'None':
        mb = i1 - i0
        obs_slice = slice(i0, i1)
        obs_nid = node_ids[obs_slice]
        obs_len = lengths[obs_slice]
        obs_norm = normals_arr[obs_slice]
        obs_msk = src_mask[obs_slice]
        obs_pts = quad_pts[obs_slice]
        local_scalar: 'List[Tuple[int, int, int, np.ndarray, bool]]' = []

        for j0 in range(i0, nelems, tile):
            j1 = min(j0 + tile, nelems)
            nb = j1 - j0
            mirrored = j0 > i0
            src_slice = slice(j0, j1)
            src_nid = node_ids[src_slice]
            src_len = lengths[src_slice]
            src_norm = normals_arr[src_slice]
            src_msk = src_mask[src_slice]

            mdx = mid_arr[obs_slice, 0][:, None] - mid_arr[src_slice, 0][None, :]
            mdy = mid_arr[obs_slice, 1][:, None] - mid_arr[src_slice, 1][None, :]
            centre_dist = np.sqrt(mdx * mdx + mdy * mdy)
            scale = np.maximum(np.maximum(obs_len[:, None], src_len[None, :]), EPS)
            batch_sym = (centre_dist / scale) >= 0.95
            batch_sym &= (
                panel_index[obs_slice][:, None] != panel_index[src_slice][None, :]
            )
            if batch_sym.any():
                # Endpoint coincidence routes a pair to the touching-Duffy
                # branch, and node snapping (1e-9) is coarser than that 1e-12
                # test, so the geometric predicate is evaluated directly.
                touching = np.zeros((mb, nb), dtype=bool)
                for ends_o in (p0_arr, p1_arr):
                    for ends_s in (p0_arr, p1_arr):
                        edx = ends_o[obs_slice, 0][:, None] - ends_s[src_slice, 0][None, :]
                        edy = ends_o[obs_slice, 1][:, None] - ends_s[src_slice, 1][None, :]
                        touching |= (edx * edx + edy * edy) <= touch_tol_sq
                batch_sym &= ~touching

            batch_ij = batch_sym & src_msk[None, :]
            scalar = ~batch_ij
            scalar &= src_msk[None, :]
            flat = np.flatnonzero(scalar.ravel())
            if flat.size:
                local_scalar.append((i0, np.arange(j0, j1), nb, flat, False))
            batch_ji = None
            if mirrored:
                batch_ji = batch_sym & obs_msk[:, None]
                scalar_t = ~batch_ji
                scalar_t &= obs_msk[:, None]
                flat_t = np.flatnonzero(scalar_t.ravel())
                if flat_t.size:
                    local_scalar.append((i0, np.arange(j0, j1), nb, flat_t, True))

            any_ij = bool(batch_ij.any())
            any_ji = bool(batch_ji.any()) if mirrored else False
            if not (any_ij or any_ji):
                continue

            src_pts = quad_pts[src_slice]
            acc = [np.zeros((mb, nb), dtype=np.complex128) for _ in range(4)]
            g_buf = np.empty((mb, nb), dtype=np.complex128)
            cscratch = np.empty((mb, nb), dtype=np.complex128)
            dx = np.empty((mb, nb), dtype=float)
            dy = np.empty((mb, nb), dtype=float)
            dist = np.empty((mb, nb), dtype=float)
            krbuf = np.empty((mb, nb), dtype=float)
            work = np.empty((mb, nb), dtype=float)

            for qi in range(t_f.size):
                r_obs = obs_pts[:, qi, :]
                w_obs_qi = float(qw[qi])
                phi_o = phi_arr[qi]
                for qj in range(t_f.size):
                    r_src = src_pts[:, qj, :]
                    np.subtract(r_obs[:, 0][:, None], r_src[None, :, 0], out=dx)
                    np.subtract(r_obs[:, 1][:, None], r_src[None, :, 1], out=dy)
                    np.multiply(dx, dx, out=dist)
                    np.multiply(dy, dy, out=work)
                    np.add(dist, work, out=dist)
                    np.sqrt(dist, out=dist)
                    np.maximum(dist, EPS, out=dist)
                    if real_k:
                        _far_kernel_argument(k0, dist, krbuf)
                    far_green(k0, real_k, dist, krbuf, work, g_buf)

                    w = w_obs_qi * float(qw[qj])
                    phi_s = phi_arr[qj]
                    for a in range(2):
                        coeff_a = w * float(phi_o[a])
                        for b in range(2):
                            _axpy_into(
                                acc[2 * a + b], g_buf,
                                coeff_a * float(phi_s[b]), cscratch,
                            )

            # Maue identity, applied to the whole tile at once.  n . n, the
            # length product, and the block sum are all symmetric under
            # swapping the pair, so the mirrored block reuses them with only
            # the (a, b) indices exchanged.
            len_prod = obs_len[:, None] * src_len[None, :]
            n_dot_n = (
                obs_norm[:, 0][:, None] * src_norm[None, :, 0]
                + obs_norm[:, 1][:, None] * src_norm[None, :, 1]
            )
            factor = -k2 * n_dot_n
            denom = np.maximum(len_prod, EPS * EPS)

            def _emit(mask: 'np.ndarray', swap: 'bool', transpose: 'bool') -> 'None':
                scale_mat = len_prod * mask
                scaled = [entry * scale_mat for entry in acc]
                block_sum = scaled[0] + scaled[1] + scaled[2] + scaled[3]
                block_sum /= denom
                rows_src = src_nid if transpose else obs_nid
                cols_src = obs_nid if transpose else src_nid
                for a in range(2):
                    rows = rows_src[:, a][:, None]
                    for b in range(2):
                        cols = cols_src[None, :, b]
                        index = (2 * b + a) if swap else (2 * a + b)
                        contrib = factor * scaled[index]
                        contrib += _TANGENT_OUTER[a, b] * block_sum
                        np.add.at(
                            d_mat, (rows, cols),
                            contrib.T if transpose else contrib,
                        )

            with write_lock:
                if any_ij:
                    _emit(batch_ij, swap=False, transpose=False)
                if any_ji:
                    _emit(batch_ji, swap=True, transpose=True)

        if local_scalar:
            with write_lock:
                scalar_chunks.extend(local_scalar)

    _run_tiled_obs_blocks(nelems, tile, _batch_pass)

    obs_idx, src_idx = _expand_near_chunks(scalar_chunks)
    last_obs = -1
    obs_elem = None
    obs_ids = None
    for pos in range(obs_idx.size):
        obs_index = int(obs_idx[pos])
        if obs_index != last_obs:
            last_obs = obs_index
            obs_elem = elements[obs_index]
            obs_ids = np.asarray(obs_elem.node_ids, dtype=int)
        src_elem = elements[int(src_idx[pos])]
        src_ids = np.asarray(src_elem.node_ids, dtype=int)
        s_blk = _single_layer_block_linear(
            obs_elem=obs_elem,
            src_elem=src_elem,
            k0=k0,
            obs_order=obs_order,
            src_order=src_order,
        )
        d_blk = _hypersingular_block_from_s_block(
            s_blk, k0, obs_elem.normal, src_elem.normal,
            obs_elem.length, src_elem.length,
        )
        d_mat[np.ix_(obs_ids, src_ids)] += d_blk
    return d_mat


@timed_stage("excitation")
def _linear_element_incident_load_many(
    elem: 'LinearElement',
    k_air: 'float',
    elevations_deg: 'np.ndarray',
    order: 'int' = 8,
) -> 'np.ndarray':
    if current_state() is not None:
        from cpu_kernels import incident
        return incident(elem, k_air, elevations_deg, order)
    qt, qw = _get_quadrature(max(2, int(order)))
    seg = elem.p1 - elem.p0
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    phi = np.deg2rad(elev)
    dirs = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    out = np.zeros((2, elev.size), dtype=np.complex128)
    for t, w in zip(qt, qw):
        shape = _linear_shape_values(float(t))[:, None]
        rp = elem.p0 + float(t) * seg
        phase = np.exp((1j * k_air) * (dirs @ rp))
        out += float(w) * shape * phase[None, :]
    return out * float(elem.length)

@timed_stage("excitation")
def _linear_element_incident_dn_load_many(
    elem: 'LinearElement',
    k_air: 'float',
    elevations_deg: 'np.ndarray',
    order: 'int' = 8,
) -> 'np.ndarray':
    """
    Galerkin-tested normal derivative of the incident plane wave on one element.

    du_inc/dn = j*k*(d_inc . n) * exp(j*k*d_inc . r)

    Used by TE sheet and impedance/flux right-hand sides.
    """
    if current_state() is not None:
        from cpu_kernels import incident_dn
        return incident_dn(elem, k_air, elevations_deg, order)

    qt, qw = _get_quadrature(max(2, int(order)))
    seg = elem.p1 - elem.p0
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    phi = np.deg2rad(elev)
    dirs = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    # d_inc . n for each elevation angle
    d_dot_n = dirs @ np.asarray(elem.normal, dtype=float)  # shape (nelevations,)
    out = np.zeros((2, elev.size), dtype=np.complex128)
    for t, w in zip(qt, qw):
        shape = _linear_shape_values(float(t))[:, None]
        rp = elem.p0 + float(t) * seg
        phase = np.exp((1j * k_air) * (dirs @ rp))
        out += float(w) * shape * (1j * k_air * d_dot_n * phase)[None, :]
    return out * float(elem.length)




@timed_stage("far_field")
def _farfield_linear_density_many(
    mesh: 'LinearMesh',
    density: 'np.ndarray',
    k_air: 'float',
    observation_angles_deg: 'np.ndarray',
    potential: 'str',
    order: 'int' = 8,
    element_mask: 'Optional[np.ndarray]' = None,
    projection: 'str' = "matched",
) -> 'np.ndarray':
    """Vectorized SLP/DLP far field for matched or rectangular projections.

    ``density`` may contain one column (one incidence projected at every
    observation angle) or one column per observation angle (the monostatic
    batched-solve case). With ``projection='grid'``, every density column is
    projected at every observation angle and the result has shape
    ``(density_columns, observation_angles)``. Element tiling bounds the
    temporary phase matrix in either mode.
    """
    if current_state() is not None:
        from cpu_kernels import farfield
        return farfield(mesh, density, k_air, observation_angles_deg, potential, order, element_mask, projection)

    obs = np.asarray(observation_angles_deg, dtype=float).reshape(-1)
    rho = np.asarray(density, dtype=np.complex128)
    if rho.ndim == 1:
        rho = rho[:, None]
    if rho.shape[0] != len(mesh.nodes):
        raise ValueError("Far-field density height must match mesh node count.")
    projection_mode = str(projection).strip().lower()
    if projection_mode not in {"matched", "grid"}:
        raise ValueError("Far-field projection must be 'matched' or 'grid'.")
    if projection_mode == "matched" and rho.shape[1] not in (1, obs.size):
        raise ValueError(
            "Far-field density must have one column or one per observation angle."
        )
    kind = str(potential).strip().upper()
    if kind not in {"SLP", "DLP"}:
        raise ValueError("Far-field potential must be 'SLP' or 'DLP'.")

    if element_mask is None:
        elements = list(mesh.elements)
    else:
        mask = np.asarray(element_mask, dtype=bool).reshape(-1)
        if mask.size != len(mesh.elements):
            raise ValueError("Far-field element mask must match mesh element count.")
        elements = [elem for elem, keep in zip(mesh.elements, mask) if keep]
    if not elements:
        shape = (rho.shape[1], obs.size) if projection_mode == "grid" else (obs.size,)
        return np.zeros(shape, dtype=np.complex128)
    qt, qw = _get_quadrature(max(2, int(order)))
    q = np.asarray(qt, dtype=float)
    wq = np.asarray(qw, dtype=float)
    phi_q = np.column_stack((1.0 - q, q))
    dirs = np.column_stack((
        np.cos(np.deg2rad(obs)), np.sin(np.deg2rad(obs))
    ))
    node_ids = np.asarray([elem.node_ids for elem in elements], dtype=int)
    p0 = np.asarray([elem.p0 for elem in elements], dtype=float)
    seg = np.asarray([elem.p1 - elem.p0 for elem in elements], dtype=float)
    lengths = np.asarray([elem.length for elem in elements], dtype=float)
    normals = np.asarray([elem.normal for elem in elements], dtype=float)

    amp = (
        np.zeros((rho.shape[1], obs.size), dtype=np.complex128)
        if projection_mode == "grid"
        else np.zeros(obs.size, dtype=np.complex128)
    )
    phase_entries = 2_000_000  # about 32 MB of complex128 phase data
    tile = max(1, min(
        len(elements), phase_entries // max(1, obs.size * q.size)
    ))
    for start in range(0, len(elements), tile):
        stop = min(start + tile, len(elements))
        pts = (
            p0[start:stop, None, :]
            + q[None, :, None] * seg[start:stop, None, :]
        )
        phase = np.exp(
            1j * float(k_air) * np.einsum('ad,eqd->aeq', dirs, pts)
        )
        local = rho[node_ids[start:stop], :]
        rho_q = np.einsum('qi,eic->eqc', phi_q, local)
        weights = lengths[start:stop, None] * wq[None, :]
        if kind == "DLP":
            dot_n = dirs @ normals[start:stop].T
            phase *= (1j * float(k_air)) * dot_n[:, :, None]
        if projection_mode == "grid":
            amp += np.einsum(
                'aeq,eqc,eq->ca', phase, rho_q, weights
            )
        elif rho.shape[1] == 1:
            amp += np.einsum(
                'aeq,eq,eq->a', phase, rho_q[:, :, 0], weights
            )
        else:
            amp += np.einsum(
                'aeq,eqa,eq->a', phase, rho_q, weights
            )
    return amp

def _linear_mass_block(elem: 'LinearElement') -> 'np.ndarray':
    """Consistent 2-node boundary mass matrix on one straight element."""

    l = float(elem.length)
    return l * np.asarray([[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]], dtype=np.complex128)

def _linear_coupled_interface_signature(elem: 'LinearElement', info: 'PanelCoupledInfo') -> 'Tuple[Any, ...]':
    return (
        int(elem.seg_type),
        int(elem.ibc_flag),
        int(elem.pos_mat),
        int(elem.neg_mat),
        int(info.minus_region),
        int(info.plus_region),
        str(info.bc_kind),
    )

def _linear_coupled_node_report(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
) -> 'Dict[str, int]':
    """
    Summarize node configurations for the nodal coupled solve.

    The linear/Galerkin path handles shared geometric junctions by
    augmenting the nodal system with trace-continuity and region-wise flux-balance rows.
    Branching and mixed-interface node counts are reported for diagnostics; they are
    not automatic blockers by themselves.
    """

    incident: 'Dict[int, List[int]]' = {}
    for eidx, elem in enumerate(mesh.elements):
        for nid in elem.node_ids:
            incident.setdefault(int(nid), []).append(int(eidx))

    branching_nodes = 0
    mixed_interface_nodes = 0
    for nid, elem_ids in incident.items():
        unique = sorted(set(int(v) for v in elem_ids))
        if len(unique) <= 1:
            continue
        sigs = {
            _linear_coupled_interface_signature(mesh.elements[eidx], infos[eidx])
            for eidx in unique
        }
        if len(unique) > 2:
            branching_nodes += 1
        if len(sigs) > 1:
            mixed_interface_nodes += 1

    return {
        "linear_node_count": int(len(mesh.nodes)),
        "linear_element_count": int(len(mesh.elements)),
        "linear_branching_nodes": int(branching_nodes),
        "linear_mixed_interface_nodes": int(mixed_interface_nodes),
        "linear_unsupported_nodes": 0,
    }

def _build_linear_junction_constraints(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    materialize: 'bool' = True,
) -> 'Tuple[np.ndarray, Dict[str, int]]':
    """
    Build nodal junction constraints for the linear/Galerkin coupled solve.

    The linear trace unknown is continuous only across explicitly shared nodes. When the
    interface-aware mesh intentionally splits nodes at the same geometric coordinate, we
    restore pointwise continuity at true shared geometric junctions with explicit trace
    constraints. We also add region-wise flux-balance constraints using the endpoint sign
    convention. When ``materialize`` is false, compute the same candidate counts and
    orientation diagnostics without allocating the dense trace/flux matrix.
    """

    nnodes = len(mesh.nodes)
    grouped: 'Dict[Tuple[int, int], List[Tuple[int, int, int]]]' = {}
    for eidx, elem in enumerate(mesh.elements):
        n0, n1 = (int(v) for v in elem.node_ids)
        grouped.setdefault(mesh.nodes[n0].key, []).append((int(eidx), 0, n0))
        grouped.setdefault(mesh.nodes[n1].key, []).append((int(eidx), 1, n1))

    rows: 'List[np.ndarray]' = []
    trace_count = 0
    flux_count = 0
    junction_nodes = 0
    orientation_conflict_nodes = 0
    constrained_nodes: 'Set[int]' = set()
    constrained_elems: 'Set[int]' = set()

    for entries in grouped.values():
        unique_elems = sorted({int(eidx) for eidx, _, _ in entries})
        unique_nodes = sorted({int(nid) for _, _, nid in entries})
        if len(unique_elems) < 2 and len(unique_nodes) < 2:
            continue

        by_elem_sign: 'Dict[int, int]' = {}
        seg_names: 'Set[str]' = set()
        region_set: 'Set[int]' = set()
        for eidx, local_end, nid in entries:
            endpoint_sign = +1 if int(local_end) == 0 else -1
            by_elem_sign[int(eidx)] = by_elem_sign.get(int(eidx), 0) + endpoint_sign
            seg_names.add(mesh.elements[int(eidx)].name)
            info = infos[int(eidx)]
            if info.minus_region >= 0:
                region_set.add(int(info.minus_region))
            if info.plus_region >= 0:
                region_set.add(int(info.plus_region))

        if len(seg_names) >= 2:
            signs = [int(np.sign(by_elem_sign.get(eidx, 0))) for eidx in unique_elems]
            has_pos = any(s > 0 for s in signs)
            has_neg = any(s < 0 for s in signs)
            if not (has_pos and has_neg):
                orientation_conflict_nodes += 1

        if len(unique_nodes) > 1:
            ref_nid = unique_nodes[0]
            for other_nid in unique_nodes[1:]:
                if materialize:
                    row = np.zeros(2 * nnodes, dtype=np.complex128)
                    row[ref_nid] = 1.0 + 0.0j
                    row[other_nid] = -1.0 + 0.0j
                    rows.append(row)
                trace_count += 1
                constrained_nodes.add(ref_nid)
                constrained_nodes.add(other_nid)

        for region in sorted(region_set):
            sparse_row: 'Dict[int, complex]' = {}
            terms = 0
            for eidx, local_end, nid in entries:
                endpoint_sign = +1 if int(local_end) == 0 else -1
                info = infos[int(eidx)]
                coeff_u = 0.0 + 0.0j
                coeff_q = 0.0 + 0.0j
                participates = False
                if info.minus_region == region:
                    coeff_q += 1.0 + 0.0j
                    participates = True
                if info.plus_region == region:
                    coeff_u += complex(info.q_plus_gamma)
                    coeff_q += complex(info.q_plus_beta)
                    participates = True
                if not participates:
                    continue

                w = complex(float(endpoint_sign), 0.0)
                nid_i = int(nid)
                sparse_row[nid_i] = sparse_row.get(nid_i, 0.0j) + w * coeff_u
                flux_index = nnodes + nid_i
                sparse_row[flux_index] = (
                    sparse_row.get(flux_index, 0.0j) + w * coeff_q
                )
                terms += 1
                constrained_nodes.add(nid_i)
                constrained_elems.add(int(eidx))

            row_norm_sq = sum(abs(value) ** 2 for value in sparse_row.values())
            if terms >= 2 and row_norm_sq > 0.0:
                if materialize:
                    row = np.zeros(2 * nnodes, dtype=np.complex128)
                    for index, value in sparse_row.items():
                        row[index] = value
                    rows.append(row)
                flux_count += 1

        junction_nodes += 1

    constraint_count = int(trace_count + flux_count)
    if constraint_count == 0:
        return np.zeros((0, 2 * nnodes), dtype=np.complex128), {
            "junction_nodes": 0,
            "junction_constraints": 0,
            "junction_panels": 0,
            "junction_trace_constraints": 0,
            "junction_flux_constraints": 0,
            "junction_orientation_conflict_nodes": int(orientation_conflict_nodes),
        }

    c_mat = (
        np.vstack(rows)
        if materialize
        else np.zeros((0, 0), dtype=np.complex128)
    )
    return c_mat, {
        "junction_nodes": int(junction_nodes),
        "junction_constraints": constraint_count,
        "junction_panels": int(len(constrained_elems)),
        "junction_trace_constraints": int(trace_count),
        "junction_flux_constraints": int(flux_count),
        "junction_orientation_conflict_nodes": int(orientation_conflict_nodes),
    }

def _ensure_finite_linear_system(a_mat: 'np.ndarray', rhs: 'Optional[np.ndarray]' = None, label: 'str' = "linear system") -> 'None':
    """Raise a clear error before calling LAPACK if the assembled system contains NaN/Inf."""

    a_eval = np.asarray(a_mat)
    if not np.all(np.isfinite(a_eval)):
        bad = np.argwhere(~np.isfinite(a_eval))
        first = tuple(int(v) for v in bad[0]) if bad.size else None
        raise ValueError(f"{label}: system matrix contains NaN/Inf at index {first}.")
    if rhs is None:
        return
    b_eval = np.asarray(rhs)
    if not np.all(np.isfinite(b_eval)):
        bad = np.argwhere(~np.isfinite(b_eval))
        first = tuple(int(v) for v in bad[0]) if bad.size else None
        raise ValueError(f"{label}: RHS contains NaN/Inf at index {first}.")

def _assemble_linear_mass_matrix(mesh: 'LinearMesh') -> 'np.ndarray':
    """Assemble the global consistent mass matrix for the linear boundary mesh."""

    nnodes = len(mesh.nodes)
    m_mat = np.zeros((nnodes, nnodes), dtype=np.complex128)
    for elem in mesh.elements:
        ids = np.asarray(elem.node_ids, dtype=int)
        m_mat[np.ix_(ids, ids)] += _linear_mass_block(elem)
    return m_mat




def _assemble_linear_weighted_mass_matrix(
    mesh: 'LinearMesh',
    element_coefficients: 'np.ndarray',
) -> 'np.ndarray':
    """Assemble ``integral phi_i c_h phi_j ds`` for elementwise-constant c.

    Material tables and impedance tapers are sampled at element centers, so
    the discrete coefficient represented by ``PanelCoupledInfo`` is naturally
    piecewise constant. Keeping it inside each element weak integral is exact
    for that discrete material model and avoids unweighted node averaging on
    nonuniform meshes and at taper endpoints.
    """

    coeff = np.asarray(element_coefficients, dtype=np.complex128).reshape(-1)
    if coeff.size != len(mesh.elements):
        raise ValueError(
            "Weighted mass coefficient count must match mesh element count."
        )
    if not np.all(np.isfinite(coeff.real) & np.isfinite(coeff.imag)):
        raise ValueError("Weighted mass coefficients must all be finite.")
    nnodes = len(mesh.nodes)
    weighted = np.zeros((nnodes, nnodes), dtype=np.complex128)
    for eidx, elem in enumerate(mesh.elements):
        ids = np.asarray(elem.node_ids, dtype=int)
        weighted[np.ix_(ids, ids)] += (
            complex(coeff[eidx]) * _linear_mass_block(elem)
        )
    return weighted


def _robin_alpha_elements(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
) -> 'Tuple[np.ndarray, np.ndarray]':
    """Return per-element Robin alpha and the PEC-element mask."""

    if len(mesh.elements) != len(infos):
        raise ValueError(
            "Robin coefficient construction requires matching elements and infos."
        )
    alpha = np.zeros(len(mesh.elements), dtype=np.complex128)
    pec = np.zeros(len(mesh.elements), dtype=bool)
    for eidx, info in enumerate(infos):
        z_surf = complex(info.robin_impedance)
        if abs(z_surf) <= EPS:
            pec[eidx] = True
            continue
        eps_m = info.eps_minus if info.minus_region >= 0 else info.eps_plus
        mu_m = info.mu_minus if info.minus_region >= 0 else info.mu_plus
        k_m = info.k_minus if info.minus_region >= 0 else info.k_plus
        alpha[eidx] = _surface_robin_alpha(
            pol, eps_m, mu_m, k_m, z_surf
        )
    return alpha, pec


def _green_2d(k0: 'Union[complex, float]', r: 'float') -> 'complex':
    """2D scalar Green's function G = j/4 * H0^(2)(k r)."""

    x = complex(k0) * max(r, EPS)
    if abs(x) <= 1e-12:
        x = 1e-12 + 0.0j
    return 0.25j * _hankel2_0(x)



def _quadrature_nodes(order: 'int' = 10) -> 'Tuple[np.ndarray, np.ndarray]':
    qx, qw = np.polynomial.legendre.leggauss(order)
    t = 0.5 * (qx + 1.0)
    w = 0.5 * qw
    return t, w

_QUAD_CACHE: 'Dict[int, Tuple[np.ndarray, np.ndarray]]' = {}
_QUAD_LOCK = threading.Lock()

def _get_quadrature(order: 'int') -> 'Tuple[np.ndarray, np.ndarray]':
    o = int(order)
    result = _QUAD_CACHE.get(o)
    if result is not None:
        return result
    with _QUAD_LOCK:
        # Double-check after acquiring lock.
        if o not in _QUAD_CACHE:
            _QUAD_CACHE[o] = _quadrature_nodes(o)
        return _QUAD_CACHE[o]

def _near_singular_scheme(distance: 'float', panel_length: 'float') -> 'Tuple[int, int]':
    """
    Choose quadrature order and source-panel subdivision count.

    This improves near-singular accuracy when observation points approach a panel.
    """

    ratio = float(distance) / max(float(panel_length), EPS)
    if ratio < 0.25:
        return 64, 16
    if ratio < 0.60:
        return 56, 10
    if ratio < 1.50:
        return 40, 6
    if ratio < 3.00:
        return 28, 3
    return 16, 1

