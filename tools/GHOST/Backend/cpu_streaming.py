"""Static streaming versions of the original dielectric BIE formulations.

Only RHS construction, solve/check workspace, and field projection are batched.
Matrix coefficients and quadrature retain the reference formulations. These
functions return monostatic fields, not full boundary-density diagnostics.
"""
import numpy as np
from cpu_execution import current_state
from cpu_checked_lu import CheckedLU
from cpu_kernels import incident_loads
from rcs_solver import (
    EPS,
    _assemble_robin_bie_system,
    _robin_alpha_elements,
    _robin_bie_rhs_many,
    _assemble_linear_hypersingular_matrix,
    _assemble_linear_mass_matrix,
    _assemble_linear_operator_matrices,
    _assemble_linear_operator_matrices_multi,
    _farfield_linear_density_many,
    _linear_element_incident_load_many,
    _make_elem_mask,
    _normalize_public_2d_solver_method,
    _rcs_sigma_from_amp,
    _surface_robin_alpha,
)


def _stream_robin_fields(mesh, matrix, k0, elevations_deg, rhs, diagnostics, label, order):
    state = current_state()
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    factor = CheckedLU(matrix, diagnostics, label, state.systems)
    amplitude = np.empty(len(elev), dtype=np.complex128)
    residual = 0.0
    for start in range(0, len(elev), state.batch_size):
        state.checkpoint()
        angles = elev[start:start + state.batch_size]
        solution = factor.solve(rhs(angles))
        amplitude[start:start + len(angles)] = _farfield_linear_density_many(
            mesh, solution, k0, angles, "SLP", order=order)
        residual = max(residual, float(np.max(factor.relative_residual)))
        state.checkpoint(start + len(angles), len(elev))
    return _rcs_sigma_from_amp(amplitude, k0), amplitude, residual


def _solve_te_robin_mfie(mesh, infos, pol, k0, elevations_deg,
                         obs_order=8, src_order=8, solver_method="auto",
                         condition_diagnostics=None, operator_cache=None):
    """Reference TE PEC/IBC MFIE matrix with bounded angle workspaces."""
    alpha, _ = _robin_alpha_elements(mesh, infos, pol)
    has_ibc = bool(np.any(np.abs(alpha) > EPS))
    s_alpha, kp = _assemble_linear_operator_matrices(
        mesh, k0, obs_normal_deriv=True, obs_order=obs_order,
        src_order=src_order, compute_single_layer=has_ibc,
        single_layer_observation_coefficients=alpha if has_ibc else None)
    matrix = -0.5 * _assemble_linear_mass_matrix(mesh) + kp
    if has_ibc:
        matrix += s_alpha
    # The bounded operator cache owns any remaining reusable references.
    s_alpha = kp = None

    def rhs(angles):
        if not has_ibc:
            return -incident_loads(mesh, k0, angles)[1]
        return _robin_bie_rhs_many(mesh, alpha, np.zeros(len(mesh.nodes), dtype=bool),
                                   pol, k0, angles)
    return _stream_robin_fields(mesh, matrix, k0, elevations_deg, rhs,
        condition_diagnostics, "TE Robin MFIE system", obs_order)


def _solve_robin_bie(mesh, infos, pol, k0, elevations_deg,
                     obs_order=8, src_order=8, condition_diagnostics=None,
                     operator_cache=None):
    """Reference element-weighted Robin/PEC EFIE matrix, streamed RHS."""
    # Avoid the older unbounded matrix cache; all underlying operators already
    # pass through the solve-local bounded LRU.
    matrix, alpha, pec_node = _assemble_robin_bie_system(
        mesh, infos, pol, k0, obs_order=obs_order, src_order=src_order,
        operator_cache=None)
    all_pec = pol == "TM" and bool(np.all(pec_node))

    def rhs(angles):
        if all_pec:
            return -incident_loads(mesh, k0, angles)[0]
        return _robin_bie_rhs_many(mesh, alpha, pec_node, pol, k0, angles)
    return _stream_robin_fields(mesh, matrix, k0, elevations_deg, rhs,
        condition_diagnostics, "Robin-BIE IBC system", obs_order)

def _solve_dielectric_indirect(mesh: 'LinearMesh', infos: 'List[PanelCoupledInfo]', pol: 'str', k0: 'float', elevations_deg: 'np.ndarray', obs_order: 'int'=8, src_order: 'int'=8, condition_diagnostics: 'Optional[Dict[str, Any]]'=None) -> 'Tuple[np.ndarray, np.ndarray, float]':
    """
    Solve dielectric scattering via the indirect two-density formulation.

    The coupled trace formulation degenerates for dielectrics because the exterior
    BIE alone determines the far-field amplitude regardless of the flux continuity
    parameter beta.  This indirect formulation uses separate densities:

        u_scat(r) = DL_0(mu)   (exterior, double-layer at k0)
        u_int(r)  = SL_1(sigma) (interior, single-layer at k1)

    Trace continuity:  u_inc + (mu/2 + K0*mu) = S1*sigma
    Flux continuity:   W0*mu + factor*(sigma/2 + K'1*sigma) = du_inc/dn

    The Maue matrix is W0 = -d_n DL_0. The interior derivative of SL_1
    is (1/2 + K'1). Thus both rows below follow directly from continuity;
    the physical exterior density is mu, without a far-field sign change.

    where factor = mu_ext/mu_int for E_z, eps_ext/eps_int for H_z.

    Far-field: A = integral jk0*(d.n)*mu * exp(jk0 d.r') ds'
    """
    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    k1_vals = {complex(info.k_plus) for info in infos if info.plus_region > 0}
    if not k1_vals:
        k1_vals = {complex(info.k_minus) for info in infos if info.minus_region > 0}
    if not k1_vals:
        raise ValueError('Dielectric indirect solver requires at least one dielectric region.')
    k1 = k1_vals.pop()
    info0 = infos[0]
    if pol == 'TM':
        factor = complex(info0.mu_minus / info0.mu_plus) if abs(info0.mu_plus) > EPS else 1.0
    else:
        factor = complex(info0.eps_minus / info0.eps_plus) if abs(info0.eps_plus) > EPS else 1.0
    _, K0 = _assemble_linear_operator_matrices(mesh, k0, obs_normal_deriv=False, obs_order=obs_order, src_order=src_order, compute_single_layer=False)
    S1, Kp1 = _assemble_linear_operator_matrices(mesh, k1, obs_normal_deriv=True, obs_order=obs_order, src_order=src_order, compute_single_layer=True)
    D0 = _assemble_linear_hypersingular_matrix(mesh, k0, obs_order=obs_order, src_order=src_order)
    M = _assemble_linear_mass_matrix(mesh)
    a_sys = np.zeros((2 * nnodes, 2 * nnodes), dtype=np.complex128)
    a_sys[:nnodes, :nnodes] = 0.5 * M + K0
    a_sys[:nnodes, nnodes:] = -S1
    a_sys[nnodes:, :nnodes] = D0
    a_sys[nnodes:, nnodes:] = factor * (0.5 * M + Kp1)
    K0 = S1 = Kp1 = D0 = M = _ = None
    _stream_all = elev
    _stream_amp = np.empty(elev.size, dtype=np.complex128)
    _stream_sigma = np.empty(elev.size, dtype=float)
    _stream_max_res = 0.0
    _stream_factor = CheckedLU(a_sys, condition_diagnostics, 'dielectric indirect system', current_state().systems)
    for _stream_start in range(0, len(_stream_all), current_state().batch_size):
        current_state().checkpoint()
        elev = _stream_all[_stream_start:_stream_start + current_state().batch_size]
        rhs_sys = np.zeros((2 * nnodes, elev.size), dtype=np.complex128)
        incident_loads(mesh, k0, elev, rhs_sys[:nnodes], rhs_sys[nnodes:])
        rhs_sys[:nnodes] *= -1
        sol = _stream_factor.solve(rhs_sys)
        if sol.ndim == 1:
            sol = sol.reshape(-1, 1)
        mu_mat = sol[:nnodes, :]
        residual_vec = _stream_factor.relative_residual
        amp = _farfield_linear_density_many(mesh, mu_mat, k0, elev, 'DLP', order=obs_order)
        rcs_lin = _rcs_sigma_from_amp(amp, k0)
        _stream_amp[_stream_start:_stream_start + len(elev)] = amp
        _stream_sigma[_stream_start:_stream_start + len(elev)] = rcs_lin
        _stream_max_res = max(_stream_max_res, float(np.max(residual_vec)))
        current_state().checkpoint(_stream_start + len(elev), len(_stream_all))
        sol = None
        mu_mat = rhs_sys = None
    return (_stream_sigma, _stream_amp, _stream_max_res)

def _solve_multi_region_indirect(mesh, infos, pol, k0, elevations_deg, obs_order=8, src_order=8, solver_method='auto', condition_diagnostics: 'Optional[Dict[str, Any]]'=None):
    """Multi-region indirect SLP formulation for layered dielectric coatings.

    BIE sign convention (validated against single-region solvers):
    - Element normal n points from minus_region toward plus_region.
    - Density on minus_region side: flux = (-1/2 M + K') * sigma
    - Density on plus_region side:  flux = (+1/2 M + K') * tau
    - Cross-interface operators (source != observer): no +/-1/2 jump.

    Uses dense LU for both solver_method="auto" and "direct".
    """
    _normalize_public_2d_solver_method(solver_method)
    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    elements = list(mesh.elements)
    nelems = len(elements)
    region_props = {}
    for info in infos:
        for rid, k, eps, mu, has_inc in [(info.minus_region, info.k_minus, info.eps_minus, info.mu_minus, info.minus_has_incident), (info.plus_region, info.k_plus, info.eps_plus, info.mu_plus, info.plus_has_incident)]:
            if rid >= 0 and rid not in region_props:
                region_props[rid] = {'k': complex(k), 'eps': complex(eps), 'mu': complex(mu), 'has_incident': bool(has_inc)}
    iface_elems = {}
    for eidx, info in enumerate(infos):
        iface_elems.setdefault((info.minus_region, info.plus_region), []).append(eidx)
    ifaces = []
    for (r_m, r_p), eids in sorted(iface_elems.items()):
        nodes = sorted({nid for ei in eids for nid in elements[ei].node_ids})
        pec_minus = r_m < 0
        pec_plus = r_p < 0
        robin_alpha = np.zeros(len(nodes), dtype=np.complex128)
        robin_alpha_elements = np.zeros(nelems, dtype=np.complex128)
        if pec_minus or pec_plus:
            diel_rid = r_p if pec_minus else r_m
            if diel_rid >= 0 and diel_rid in region_props:
                rp = region_props[diel_rid]
                side_sign = -1.0 if pec_minus else 1.0
                for ei in eids:
                    z_s = complex(infos[ei].robin_impedance)
                    if abs(z_s) > EPS:
                        robin_alpha_elements[ei] = side_sign * _surface_robin_alpha(pol, rp['eps'], rp['mu'], rp['k'], z_s)
                for ni, nid in enumerate(nodes):
                    incident = [robin_alpha_elements[ei] for ei in eids if nid in elements[ei].node_ids]
                    if incident:
                        robin_alpha[ni] = sum(incident) / len(incident)
        ifaces.append({'r_m': r_m, 'r_p': r_p, 'eids': eids, 'nodes': nodes, 'n': len(nodes), 'pec_minus': pec_minus, 'pec_plus': pec_plus, 'robin_alpha': robin_alpha, 'robin_alpha_elements': robin_alpha_elements, 'mask': _make_elem_mask(eids, nelems)})
    region_ifaces = {}
    for mi, ifc in enumerate(ifaces):
        for rid in [ifc['r_m'], ifc['r_p']]:
            if rid >= 0:
                region_ifaces.setdefault(rid, []).append(mi)
    dof_map = {}
    n_dof = 0
    for mi, ifc in enumerate(ifaces):
        if ifc['r_m'] >= 0:
            dof_map[mi, 'minus'] = (n_dof, ifc['n'])
            n_dof += ifc['n']
        if ifc['r_p'] >= 0:
            dof_map[mi, 'plus'] = (n_dof, ifc['n'])
            n_dof += ifc['n']
    M_global = _assemble_linear_mass_matrix(mesh)
    op_cache = {}

    def get_ops(k_val, src_mask):
        key = (complex(k_val), id(src_mask))
        if key not in op_cache:
            S, Kp = _assemble_linear_operator_matrices(mesh, k_val, True, obs_order, src_order, source_element_mask=src_mask)
            op_cache[key] = (S, Kp)
        return op_cache[key]
    weighted_s_cache = {}

    def get_weighted_s(k_val, src_mask, obs_coeff):
        coeff_eval = np.asarray(obs_coeff, dtype=np.complex128)
        key = (complex(k_val), id(src_mask), id(obs_coeff))
        if not np.any(np.abs(coeff_eval) > EPS):
            return np.broadcast_to(np.zeros((), dtype=np.complex128), (nnodes, nnodes))
        if key not in weighted_s_cache:
            S_alpha, _ = _assemble_linear_operator_matrices(mesh, k_val, True, obs_order, src_order, source_element_mask=src_mask, compute_double_layer=False, single_layer_observation_coefficients=coeff_eval)
            weighted_s_cache[key] = S_alpha
        return weighted_s_cache[key]
    _requests_by_k = {}
    _request_seen = {}
    for _rid, _mis in region_ifaces.items():
        _k = complex(region_props[_rid]['k'])
        _slot = _requests_by_k.setdefault(_k, [])
        _seen = _request_seen.setdefault(_k, set())
        for _mi in _mis:
            _mask = ifaces[_mi]['mask']
            _token = ('plain', id(_mask))
            if _token in _seen:
                continue
            _seen.add(_token)
            _slot.append(('plain', _mask, None))
    for _mi, _ifc in enumerate(ifaces):
        if not (_ifc['pec_minus'] or _ifc['pec_plus']):
            continue
        _rid = _ifc['r_p'] if _ifc['pec_minus'] else _ifc['r_m']
        _k = complex(region_props[_rid]['k'])
        _slot = _requests_by_k.setdefault(_k, [])
        _seen = _request_seen.setdefault(_k, set())
        for _mj in region_ifaces.get(_rid, []):
            _src_mask = ifaces[_mj]['mask']
            _obs_coeff = _ifc['robin_alpha_elements']
            if not np.any(np.abs(_obs_coeff) > EPS):
                continue
            _token = ('weighted', id(_src_mask), id(_obs_coeff))
            if _token in _seen:
                continue
            _seen.add(_token)
            _slot.append(('weighted', _src_mask, _obs_coeff))
    for _k, _requests in _requests_by_k.items():
        _masks = [request[1] for request in _requests]
        _coeffs = [request[2] for request in _requests]
        _want_k = [request[0] == 'plain' for request in _requests]
        _outputs = _assemble_linear_operator_matrices_multi(mesh=mesh, k0=_k, obs_normal_deriv=True, source_element_masks=_masks, obs_order=obs_order, src_order=src_order, compute_double_layer=True, compute_double_layer_many=_want_k, single_layer_observation_coefficients_many=_coeffs)
        for (_kind, _mask, _coeff), (_s_mat, _k_mat) in zip(_requests, _outputs):
            if _kind == 'plain':
                op_cache[_k, id(_mask)] = (_s_mat, _k_mat)
            else:
                weighted_s_cache[_k, id(_mask), id(_coeff)] = _s_mat
    Asys = np.zeros((n_dof, n_dof), dtype=np.complex128)

    def sub(mat, obs_n, src_n):
        return mat[np.ix_(obs_n, src_n)]

    def _add_robin_block_dense(mi, ifc, dof_side, region_id, jump_sign):
        dm = dof_map[mi, dof_side]
        obs_n = ifc['nodes']
        nm = ifc['n']
        k_d = region_props[region_id]['k']
        S_self, Kp_self = get_ops(k_d, ifc['mask'])
        S_alpha_self = get_weighted_s(k_d, ifc['mask'], ifc['robin_alpha_elements'])
        M_s = sub(M_global, obs_n, obs_n)
        alpha = ifc['robin_alpha']
        tm_pec_mask = np.abs(alpha) <= EPS if pol == 'TM' else np.zeros(nm, dtype=bool)
        S_sub = sub(S_self, obs_n, obs_n)
        Kp_sub = sub(Kp_self, obs_n, obs_n)
        block = jump_sign * 0.5 * M_s + Kp_sub + sub(S_alpha_self, obs_n, obs_n)
        if np.any(tm_pec_mask):
            block[tm_pec_mask, :] = S_sub[tm_pec_mask, :]
        Asys[dm[0]:dm[0] + nm, dm[0]:dm[0] + nm] += block
        for mj in region_ifaces.get(region_id, []):
            if mj == mi:
                continue
            ifj = ifaces[mj]
            side_j = 'minus' if ifj['r_m'] == region_id else 'plus'
            dj = dof_map.get((mj, side_j))
            if dj is None:
                continue
            S_x, Kp_x = get_ops(k_d, ifj['mask'])
            src_n = ifj['nodes']
            S_alpha_x = get_weighted_s(k_d, ifj['mask'], ifc['robin_alpha_elements'])
            S_x_sub = sub(S_x, obs_n, src_n)
            Kp_x_sub = sub(Kp_x, obs_n, src_n)
            cross_block = Kp_x_sub + sub(S_alpha_x, obs_n, src_n)
            if np.any(tm_pec_mask):
                cross_block[tm_pec_mask, :] = S_x_sub[tm_pec_mask, :]
            Asys[dm[0]:dm[0] + nm, dj[0]:dj[0] + dj[1]] += cross_block
    for mi, ifc in enumerate(ifaces):
        r_m, r_p = (ifc['r_m'], ifc['r_p'])
        if ifc['pec_minus']:
            _add_robin_block_dense(mi, ifc, 'plus', r_p, +1.0)
        elif ifc['pec_plus']:
            _add_robin_block_dense(mi, ifc, 'minus', r_m, -1.0)
        else:
            obs_n = ifc['nodes']
            nm = ifc['n']
            d_sigma = dof_map[mi, 'minus']
            d_tau = dof_map[mi, 'plus']
            k_m_val = region_props[r_m]['k']
            k_p_val = region_props[r_p]['k']
            if pol == 'TM':
                beta = complex(region_props[r_p]['mu'] / region_props[r_m]['mu']) if abs(region_props[r_m]['mu']) > EPS else 1.0 + 0j
            else:
                beta = complex(region_props[r_p]['eps'] / region_props[r_m]['eps']) if abs(region_props[r_m]['eps']) > EPS else 1.0 + 0j
            if abs(beta) <= EPS:
                beta = 1.0 + 0j
            inv_beta = 1.0 / beta
            S_m, Kp_m = get_ops(k_m_val, ifc['mask'])
            S_p, Kp_p = get_ops(k_p_val, ifc['mask'])
            M_s = sub(M_global, obs_n, obs_n)
            Asys[d_sigma[0]:d_sigma[0] + nm, d_sigma[0]:d_sigma[0] + nm] += -0.5 * M_s + sub(Kp_m, obs_n, obs_n)
            Asys[d_sigma[0]:d_sigma[0] + nm, d_tau[0]:d_tau[0] + nm] -= inv_beta * (0.5 * M_s + sub(Kp_p, obs_n, obs_n))
            Asys[d_tau[0]:d_tau[0] + nm, d_sigma[0]:d_sigma[0] + nm] += sub(S_m, obs_n, obs_n)
            Asys[d_tau[0]:d_tau[0] + nm, d_tau[0]:d_tau[0] + nm] -= sub(S_p, obs_n, obs_n)
            for mj in region_ifaces.get(r_m, []):
                if mj == mi:
                    continue
                ifj = ifaces[mj]
                side_j = 'minus' if ifj['r_m'] == r_m else 'plus'
                dj = dof_map.get((mj, side_j))
                if dj is None:
                    continue
                S_x, Kp_x = get_ops(k_m_val, ifj['mask'])
                src_n = ifj['nodes']
                Asys[d_sigma[0]:d_sigma[0] + nm, dj[0]:dj[0] + dj[1]] += sub(Kp_x, obs_n, src_n)
                Asys[d_tau[0]:d_tau[0] + nm, dj[0]:dj[0] + dj[1]] += sub(S_x, obs_n, src_n)
            for mj in region_ifaces.get(r_p, []):
                if mj == mi:
                    continue
                ifj = ifaces[mj]
                side_j = 'minus' if ifj['r_m'] == r_p else 'plus'
                dj = dof_map.get((mj, side_j))
                if dj is None:
                    continue
                S_x, Kp_x = get_ops(k_p_val, ifj['mask'])
                src_n = ifj['nodes']
                Asys[d_sigma[0]:d_sigma[0] + nm, dj[0]:dj[0] + dj[1]] -= inv_beta * sub(Kp_x, obs_n, src_n)
                Asys[d_tau[0]:d_tau[0] + nm, dj[0]:dj[0] + dj[1]] -= sub(S_x, obs_n, src_n)
    op_cache.clear()
    weighted_s_cache.clear()
    _outputs = []
    _s_mat = _k_mat = S_m = Kp_m = S_p = Kp_p = S_x = Kp_x = M_s = M_global = None
    _stream_all = elev
    _stream_amp = np.empty(elev.size, dtype=np.complex128)
    _stream_sigma = np.empty(elev.size, dtype=float)
    _stream_max_res = 0.0
    _stream_factor = CheckedLU(Asys, condition_diagnostics, 'multi-region indirect system', current_state().systems)
    for _stream_start in range(0, len(_stream_all), current_state().batch_size):
        current_state().checkpoint()
        elev = _stream_all[_stream_start:_stream_start + current_state().batch_size]
        bu, bdn = incident_loads(mesh, k0, elev)
        Brhs = np.zeros((n_dof, elev.size), dtype=np.complex128)
        for mi, ifc in enumerate(ifaces):
            obs_n = ifc['nodes']
            nm = ifc['n']
            r_m, r_p = (ifc['r_m'], ifc['r_p'])
            alpha = ifc['robin_alpha']
            if ifc['pec_minus'] or ifc['pec_plus']:
                dof_side = 'plus' if ifc['pec_minus'] else 'minus'
                dm = dof_map[mi, dof_side]
                rid = r_p if ifc['pec_minus'] else r_m
                tm_pec_mask = np.abs(alpha) <= EPS if pol == 'TM' else np.zeros(nm, dtype=bool)
                if region_props[rid].get('has_incident'):
                    alpha_bu_global = np.zeros_like(bu)
                    alpha_e = ifc['robin_alpha_elements']
                    for ei in ifc['eids']:
                        elem = elements[ei]
                        ids = np.asarray(elem.node_ids, dtype=int)
                        alpha_bu_global[ids] += complex(alpha_e[ei]) * _linear_element_incident_load_many(elem, k_air=k0, elevations_deg=elev)
                    alpha_bu = alpha_bu_global[obs_n]
                    rhs_block = bdn[obs_n] + alpha_bu
                    if np.any(tm_pec_mask):
                        rhs_block[tm_pec_mask] = bu[obs_n][tm_pec_mask]
                    Brhs[dm[0]:dm[0] + nm] -= rhs_block
            else:
                d_sigma = dof_map[mi, 'minus']
                d_tau = dof_map[mi, 'plus']
                if pol == 'TM':
                    beta = complex(region_props[r_p]['mu'] / region_props[r_m]['mu']) if abs(region_props[r_m]['mu']) > EPS else 1.0 + 0j
                else:
                    beta = complex(region_props[r_p]['eps'] / region_props[r_m]['eps']) if abs(region_props[r_m]['eps']) > EPS else 1.0 + 0j
                if abs(beta) <= EPS:
                    beta = 1.0 + 0j
                inv_beta = 1.0 / beta
                if region_props[r_m].get('has_incident'):
                    Brhs[d_sigma[0]:d_sigma[0] + nm] -= bdn[obs_n]
                    Brhs[d_tau[0]:d_tau[0] + nm] -= bu[obs_n]
                if region_props[r_p].get('has_incident'):
                    Brhs[d_sigma[0]:d_sigma[0] + nm] += inv_beta * bdn[obs_n]
                    Brhs[d_tau[0]:d_tau[0] + nm] += bu[obs_n]
        sol = _stream_factor.solve(Brhs)
        if sol.ndim == 1:
            sol = sol.reshape(-1, 1)
        max_res = float(np.max(_stream_factor.relative_residual))
        ext_rid = next((rid for rid, rp in region_props.items() if rp.get('has_incident')), 0)
        ext_density_global = np.zeros((nnodes, elev.size), dtype=np.complex128)
        ext_elem_mask = np.zeros(nelems, dtype=bool)
        for mi, ifc in enumerate(ifaces):
            if ifc['r_m'] == ext_rid:
                side = 'minus'
            elif ifc['r_p'] == ext_rid:
                side = 'plus'
            else:
                continue
            dm = dof_map.get((mi, side))
            if dm is None:
                continue
            density = sol[dm[0]:dm[0] + dm[1], :]
            for li, nid in enumerate(ifc['nodes']):
                ext_density_global[nid, :] += density[li, :]
            for eidx in ifc['eids']:
                ext_elem_mask[eidx] = True
        amp = _farfield_linear_density_many(mesh, ext_density_global, k0, elev, 'SLP', order=obs_order, element_mask=ext_elem_mask)
        rcs_lin = _rcs_sigma_from_amp(amp, k0)
        _stream_amp[_stream_start:_stream_start + len(elev)] = amp
        _stream_sigma[_stream_start:_stream_start + len(elev)] = rcs_lin
        _stream_max_res = max(_stream_max_res, max_res)
        current_state().checkpoint(_stream_start + len(elev), len(_stream_all))
        sol = None
        density = ext_density_global = bu = bdn = Brhs = alpha_bu_global = alpha_bu = rhs_block = None
    return (_stream_sigma, _stream_amp, _stream_max_res, None)
