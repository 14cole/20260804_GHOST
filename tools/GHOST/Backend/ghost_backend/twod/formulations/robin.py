"""Owned Robin system with only the S/K' rows used by the equations."""
import numpy as np
from ghost_backend.twod.assembly.mass import add_mass
from ghost_backend.twod.assembly.session import current_session, system_key


def assemble_system(mesh, infos, pol, k0, obs_order=8, src_order=8, operator_cache=None):
    import ghost_backend.twod.solver as rcs
    from ghost_backend.compressed.runtime import enabled, native
    if enabled():
        operator,oracle=native(mesh,infos,pol,k0,'robin',obs_order,src_order)
        return operator,oracle.alpha,oracle.pec_nodes
    n = len(mesh.nodes)
    alpha, pec_elements = rcs._robin_alpha_elements(mesh, infos, pol)
    pec_nodes = np.zeros(n, bool)
    for element, pec in zip(mesh.elements, pec_elements):
        if pec:
            pec_nodes[list(element.node_ids)] = True
    has_ibc = np.any(np.abs(alpha) > rcs.EPS)
    session = current_session()
    key = system_key(mesh, infos, 'robin', obs_order, src_order) if session is not None else None
    previous = session.take(key, pol) if session is not None else None
    if pol == 'TM' and np.all(pec_elements):
        s, _ = rcs._assemble_linear_operator_matrices(mesh, k0, True,
            obs_order=obs_order, src_order=src_order, compute_double_layer=False)
        return s, alpha, pec_nodes
    if not has_ibc and pol == 'TE':
        _, k = rcs._assemble_linear_operator_matrices(mesh, k0, True,
            obs_order=obs_order, src_order=src_order, compute_single_layer=False)
        matrix = k
        k = None
        add_mass(matrix, mesh, -.5)
        return matrix, alpha, pec_nodes
    pec_rows = np.flatnonzero(pec_nodes) if pol == 'TM' else np.empty(0, int)
    robin_rows = np.flatnonzero(~pec_nodes) if pol == 'TM' else np.arange(n)
    if previous is not None:
        matrix, old_alpha = previous
        weights = alpha - old_alpha
        if operator_cache is not None:
            operator_cache['_hits'] = int(operator_cache.get('_hits', 0)) + 1
    else:
        weights = alpha
        matrix = np.zeros((n, n), complex, order='F')
        add_mass(matrix, mesh, -.5)
    need_s = bool(np.any(weights != 0))
    need_k = previous is None
    requests = []
    if need_s or need_k:
        requests.append((robin_rows if need_s else [], robin_rows if need_k else [], weights))
    if len(pec_rows):
        requests.append((pec_rows, [], None))
    from ghost_backend.twod.assembly.scatter import robin_outputs
    destinations = robin_outputs(matrix, mesh, requests)
    rcs._assemble_linear_operator_matrices_multi(mesh, k0, True,
        [None]*len(requests), obs_order=obs_order, src_order=src_order,
        compute_double_layer_many=[bool(len(krows)) for _, krows, _ in requests],
        single_layer_observation_coefficients_many=[w for _, _, w in requests],
        output_node_ids_many=[(rows, np.arange(n)) for rows, _, _ in requests],
        double_layer_output_node_ids_many=[(rows, np.arange(n)) for _, rows, _ in requests],
        operator_outputs=destinations) if requests else None
    if session is not None and pol == 'TE':
        session.save(key, pol, (matrix, alpha))
        if operator_cache is not None:
            operator_cache['_stores'] = int(operator_cache.get('_stores', 0)) + 1
    return matrix, alpha, pec_nodes
