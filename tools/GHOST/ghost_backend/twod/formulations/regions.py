"""Shared multi-region layout, compact operator plan and bounded assembly."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from ghost_backend.twod.constants import EPS
from ghost_backend.twod.geometry import _surface_robin_alpha

BLOCK_ROWS = 64


def build_layout(mesh, infos, pol):
    elements = mesh.elements
    regions, interface_elements = {}, {}
    for ei, info in enumerate(infos):
        for rid, k, eps, mu, inc in (
            (info.minus_region, info.k_minus, info.eps_minus, info.mu_minus, info.minus_has_incident),
            (info.plus_region, info.k_plus, info.eps_plus, info.mu_plus, info.plus_has_incident)):
            if rid >= 0 and rid not in regions:
                regions[rid] = dict(k=complex(k), eps=complex(eps), mu=complex(mu), has_incident=bool(inc))
        interface_elements.setdefault((info.minus_region, info.plus_region), []).append(ei)
    ifaces = []
    region_ifaces = {}
    dof_map, n_dof = {}, 0
    for mi, ((rm, rp), eids) in enumerate(sorted(interface_elements.items())):
        nodes = sorted({n for ei in eids for n in elements[ei].node_ids})
        node_positions = {n: i for i, n in enumerate(nodes)}
        alpha = np.zeros(len(nodes), dtype=np.complex128)
        counts = np.zeros(len(nodes), dtype=int)
        alpha_elements = np.zeros(len(elements), dtype=np.complex128)
        if rm < 0 or rp < 0:
            region = regions[rp if rm < 0 else rm]
            sign = -1.0 if rm < 0 else 1.0
            for ei in eids:
                z = complex(infos[ei].robin_impedance)
                if abs(z) > EPS:
                    alpha_elements[ei] = sign * _surface_robin_alpha(
                        pol, region['eps'], region['mu'], region['k'], z)
                for n in elements[ei].node_ids:
                    j = node_positions[n]
                    alpha[j] += alpha_elements[ei]
                    counts[j] += 1
            alpha /= np.maximum(counts, 1)
        mask = np.zeros(len(elements), dtype=bool)
        mask[eids] = True
        ifaces.append(dict(r_m=rm, r_p=rp, eids=eids, nodes=nodes, n=len(nodes),
            pec_minus=rm < 0, pec_plus=rp < 0, robin_alpha=alpha,
            robin_alpha_elements=alpha_elements, mask=mask))
        for side, rid in (('minus', rm), ('plus', rp)):
            if rid >= 0:
                region_ifaces.setdefault(rid, []).append(mi)
                dof_map[mi, side] = (n_dof, len(nodes))
                n_dof += len(nodes)
    return dict(polarization=pol, region_props=regions, ifaces=ifaces, region_ifaces=region_ifaces,
                dof_map=dof_map, n_dof=n_dof)


def operator_plan(layout):
    """Coalesce equal-wavenumber requests while retaining every required row."""
    ifaces, regions = layout['ifaces'], layout['region_props']
    region_ifaces = layout['region_ifaces']
    by_k = {}
    for rid, mis in region_ifaces.items():
        k = regions[rid]['k']
        slot = by_k.setdefault(k, {})
        rows = {n for mi in mis for n in ifaces[mi]['nodes']}
        for mi in mis:


            request = slot.setdefault(('plain', mi), dict(kind='plain', source=mi, observer=None,
                                                         rows=set(), s_rows=set(), k_rows=set()))
            request['rows'].update(rows)
            for observer in mis:
                target = ifaces[observer]
                if target['r_m'] >= 0 and target['r_p'] >= 0:
                    request['s_rows'].update(target['nodes'])
                    request['k_rows'].update(target['nodes'])
                else:
                    pec = np.abs(target['robin_alpha']) <= EPS if layout['polarization'] == 'TM' else np.zeros(target['n'], bool)
                    request['s_rows'].update(n for n, keep in zip(target['nodes'], pec) if keep)
                    request['k_rows'].update(n for n, keep in zip(target['nodes'], pec) if not keep)
    for mi, ifc in enumerate(ifaces):
        if not (ifc['pec_minus'] or ifc['pec_plus']):
            continue
        if not np.any(np.abs(ifc['robin_alpha_elements']) > EPS):
            continue
        rid = ifc['r_p'] if ifc['pec_minus'] else ifc['r_m']
        slot = by_k.setdefault(regions[rid]['k'], {})
        for mj in region_ifaces[rid]:
            slot['weighted', mj, mi] = dict(kind='weighted', source=mj, observer=mi,
                rows=set(ifc['nodes']), s_rows=set(ifc['nodes']), k_rows=set())
    return [(k, list(slot.values())) for k, slot in by_k.items()]


def storage_resources(mesh, layout):
    entries = matrices = map_bytes = 0
    n = len(mesh.nodes)
    for k, requests in operator_plan(layout):
        for request in requests:
            rows = len(request['rows'])
            cols = layout['ifaces'][request['source']]['n']
            count = 2 if request['kind'] == 'plain' else 1
            matrices += count
            entries += (len(request['s_rows']) + len(request['k_rows'])) * cols


            map_bytes += 2 * 8 * (2*n + rows + cols)
            route_count = sum(1 for rid, sources in layout['region_ifaces'].items()
                if request['source'] in sources and layout['region_props'][rid]['k'] == k)
            map_bytes += 56*n*route_count

    mass_bytes = 160 * len(mesh.elements) + 8 * (n + 1)
    max_interface = max((i['n'] for i in layout['ifaces']), default=0)
    block_bytes = 16 * BLOCK_ROWS * max_interface * 12


    centers = np.asarray([e.center for e in mesh.elements])
    lengths = np.asarray([e.length for e in mesh.elements])
    near_pairs = 0
    if len(lengths):
        tree = cKDTree(centers)
        radius = 3.0 * float(lengths.max())
        for i, center in enumerate(centers):
            candidates = np.asarray(tree.query_ball_point(center, radius), dtype=int)
            distance = np.linalg.norm(centers[candidates]-center, axis=1)
            near_pairs += int(np.count_nonzero(distance <= 3.0*np.maximum(lengths[i], lengths[candidates])))
    import ghost_backend.twod.operators as ops
    tile = ops._assembly_tile_size(len(mesh.elements), 312)
    largest_group = max((len(requests) for _, requests in operator_plan(layout)), default=0)
    tile_bytes = ops.get_assembly_threads() * tile*tile * (1024 + 16*largest_group)


    near_batch_samples = min(ops._NEAR_BATCH_MAX_SAMPLES, near_pairs * 16 * 16)
    near_batch_bytes = near_batch_samples * 256
    assembly_workspace = 1024 * near_pairs + max(tile_bytes, near_batch_bytes)


    return dict(operator_matrices=matrices, operator_entries=entries,
                assembly_operator_entries=0,
                operator_map_bytes=map_bytes, mass_workspace_bytes=mass_bytes,
                block_workspace_bytes=block_bytes,
                assembly_workspace_bytes=assembly_workspace)


def _sparse_mass(mesh):
    ids = np.asarray([e.node_ids for e in mesh.elements], dtype=int)
    lengths = np.asarray([e.length for e in mesh.elements])
    rows = np.repeat(ids, 2, axis=1).reshape(-1)
    cols = np.tile(ids, (1, 2)).reshape(-1)
    values = (lengths[:, None] * np.asarray([1/3, 1/6, 1/6, 1/3])[None, :]).reshape(-1)
    return coo_matrix((values, (rows, cols)), shape=(len(mesh.nodes), len(mesh.nodes))).tocsr()


def _assemble_system_fresh(mesh, infos, pol, obs_order=8, src_order=8):

    import ghost_backend.twod.solver as rcs
    layout = build_layout(mesh, infos, pol)
    ifaces = layout['ifaces']
    regions = layout['region_props']
    region_ifaces = layout['region_ifaces']
    dofs = layout['dof_map']
    operators, weighted = {}, {}
    for k, requests in operator_plan(layout):
        masks = [ifaces[r['source']]['mask'] for r in requests]
        coefficients = [None if r['observer'] is None else
                        ifaces[r['observer']]['robin_alpha_elements'] for r in requests]
        outputs = rcs._assemble_linear_operator_matrices_multi(
            mesh=mesh, k0=k, obs_normal_deriv=True, source_element_masks=masks,
            obs_order=obs_order, src_order=src_order,
            compute_double_layer_many=[r['kind'] == 'plain' for r in requests],
            single_layer_observation_coefficients_many=coefficients,
            output_node_ids_many=[(sorted(r['s_rows']), ifaces[r['source']]['nodes']) for r in requests],
            double_layer_output_node_ids_many=[(sorted(r['k_rows']), ifaces[r['source']]['nodes']) for r in requests])
        for request, pair in zip(requests, outputs):
            if request['kind'] == 'plain':
                operators[k, request['source']] = pair
            else:
                weighted[k, request['source'], request['observer']] = pair[0]
    mass = _sparse_mass(mesh)
    matrix = np.zeros((layout['n_dof'], layout['n_dof']), dtype=np.complex128, order='F')

    def sub(op, rows, columns):
        return op.block(rows, columns) if hasattr(op, 'block') else op[np.ix_(rows, columns)]

    def alpha_s(k, source, observer, rows, columns):
        op = weighted.get((k, source, observer))
        return 0.0 if op is None else sub(op, rows, columns)

    for mi, ifc in enumerate(ifaces):
        rm, rp = ifc['r_m'], ifc['r_p']
        all_nodes, nm = ifc['nodes'], ifc['n']
        for start in range(0, nm, BLOCK_ROWS):
            stop = min(start + BLOCK_ROWS, nm)
            rows = all_nodes[start:stop]


            m = mass[rows, :][:, all_nodes].toarray()
            if rm < 0 or rp < 0:
                side = 'plus' if rm < 0 else 'minus'
                rid = rp if rm < 0 else rm
                k = regions[rid]['k']
                dm = dofs[mi, side][0]
                dest_rows = slice(dm + start, dm + stop)
                pec = np.abs(ifc['robin_alpha'][start:stop]) <= EPS if pol == 'TM' else np.zeros(stop-start, dtype=bool)
                s, kp = operators[k, mi]
                block = (0.5 if rm < 0 else -0.5) * m + sub(kp, rows, all_nodes) + alpha_s(k, mi, mi, rows, all_nodes)
                if np.any(pec):
                    block[pec] = sub(s, rows, all_nodes)[pec]
                matrix[dest_rows, dm:dm+nm] += block
                for mj in region_ifaces[rid]:
                    if mj == mi:
                        continue
                    src = ifaces[mj]
                    dj, nj = dofs[mj, 'minus' if src['r_m'] == rid else 'plus']
                    s, kp = operators[k, mj]
                    block = sub(kp, rows, src['nodes']) + alpha_s(k, mj, mi, rows, src['nodes'])
                    if np.any(pec):
                        block[pec] = sub(s, rows, src['nodes'])[pec]
                    matrix[dest_rows, dj:dj+nj] += block
            else:
                ds, dt = dofs[mi, 'minus'][0], dofs[mi, 'plus'][0]
                rs, rt = slice(ds+start, ds+stop), slice(dt+start, dt+stop)
                key = 'mu' if pol == 'TM' else 'eps'
                beta = regions[rp][key] / regions[rm][key] if abs(regions[rm][key]) > EPS else 1+0j
                inv = 1.0 / beta if abs(beta) > EPS else 1+0j
                sm, km = operators[regions[rm]['k'], mi]
                sp, kp = operators[regions[rp]['k'], mi]
                matrix[rs, ds:ds+nm] += -0.5*m + sub(km, rows, all_nodes)
                matrix[rs, dt:dt+nm] -= inv * (0.5*m + sub(kp, rows, all_nodes))
                matrix[rt, ds:ds+nm] += sub(sm, rows, all_nodes)
                matrix[rt, dt:dt+nm] -= sub(sp, rows, all_nodes)
                for rid, sign, flux in ((rm, 1.0, 1.0), (rp, -1.0, inv)):
                    for mj in region_ifaces[rid]:
                        if mj == mi:
                            continue
                        src = ifaces[mj]
                        dj, nj = dofs[mj, 'minus' if src['r_m'] == rid else 'plus']
                        s, kp = operators[regions[rid]['k'], mj]
                        if sign > 0:
                            matrix[rs, dj:dj+nj] += sub(kp, rows, src['nodes'])
                            matrix[rt, dj:dj+nj] += sub(s, rows, src['nodes'])
                        else:
                            matrix[rs, dj:dj+nj] -= flux * sub(kp, rows, src['nodes'])
                            matrix[rt, dj:dj+nj] -= sub(s, rows, src['nodes'])

    return matrix, layout


def _inverse_beta(layout, interface, pol):
    rm, rp = interface['r_m'], interface['r_p']
    key = 'mu' if pol == 'TM' else 'eps'
    regions = layout['region_props']
    beta = regions[rp][key] / regions[rm][key] if abs(regions[rm][key]) > EPS else 1+0j
    return 1.0 / beta if abs(beta) > EPS else 1+0j


def _reuse_te_system(matrix, old, layout, mesh, obs_order, src_order):
    """Convert owned TE coefficients to TM without retaining regional operators.

    Transmission trace rows contain S and remain unchanged. Reciprocity gives
    conductor-to-transmission S from those rows. Only conductor self/cross S
    and the change of weighted Robin S need new integration.
    """
    import ghost_backend.twod.solver as rcs
    interfaces, dofs = layout['ifaces'], layout['dof_map']
    for mi, target in enumerate(interfaces):
        if target['r_m'] < 0 or target['r_p'] < 0:
            continue
        row, n = dofs[mi, 'minus']
        ratio = _inverse_beta(layout, target, 'TM') / _inverse_beta(old, old['ifaces'][mi], 'TE')
        rid = target['r_p']
        for mj in layout['region_ifaces'][rid]:
            source = interfaces[mj]
            column, width = dofs[mj, 'minus' if source['r_m'] == rid else 'plus']
            matrix[row:row+n, column:column+width] *= ratio
    for mi, target in enumerate(interfaces):
        if target['r_m'] >= 0 and target['r_p'] >= 0:
            continue
        rid = target['r_p'] if target['r_m'] < 0 else target['r_m']
        offset, n = dofs[mi, 'plus' if target['r_m'] < 0 else 'minus']
        pec = np.abs(target['robin_alpha']) <= EPS
        pec_positions = np.flatnonzero(pec)
        other_positions = np.flatnonzero(~pec)
        delta = target['robin_alpha_elements'] - old['ifaces'][mi]['robin_alpha_elements']
        regional = layout['region_ifaces'][rid]
        conductors = [mj for mj in regional if interfaces[mj]['r_m'] < 0 or interfaces[mj]['r_p'] < 0]
        requests = []
        if len(pec_positions):
            requests.append(('pec', pec_positions, conductors, None))
        if len(other_positions) and np.any(delta != 0):
            requests.append(('robin', other_positions, regional, delta))
        masks, output_ids, coefficients = [], [], []
        for kind, positions, sources, weights in requests:
            mask = np.zeros(len(mesh.elements), bool)
            columns = set()
            for mj in sources:
                mask[interfaces[mj]['eids']] = True
                columns.update(interfaces[mj]['nodes'])
            masks.append(mask)
            output_ids.append((np.asarray(target['nodes'])[positions], sorted(columns)))
            coefficients.append(weights)
        from ghost_backend.twod.assembly.scatter import SystemScatter
        direct_outputs = []
        for (kind, positions, sources, weights), (rows, columns) in zip(requests, output_ids):
            row_map = np.full(len(mesh.nodes), -1, np.int64)
            row_map[rows] = offset + positions
            column_map = np.full(len(mesh.nodes), -1, np.int64)
            for mj in sources:
                source = interfaces[mj]
                column, width = dofs[mj, 'minus' if source['r_m'] == rid else 'plus']
                column_map[source['nodes']] = np.arange(column, column+width)
                if kind == 'pec':
                    for start in range(0, len(positions), BLOCK_ROWS):
                        matrix[np.ix_(offset+positions[start:start+BLOCK_ROWS],
                                      np.arange(column, column+width))] = 0
            direct_outputs.append((
                SystemScatter(matrix, len(mesh.nodes), rows, columns,
                    [(row_map, column_map, np.ones(len(mesh.nodes)))]),
                SystemScatter(matrix, len(mesh.nodes), [], columns, [])))
        rcs._assemble_linear_operator_matrices_multi(mesh,
            layout['region_props'][rid]['k'], True, masks, obs_order=obs_order, src_order=src_order,
            compute_double_layer=False, single_layer_observation_coefficients_many=coefficients,
            output_node_ids_many=output_ids, operator_outputs=direct_outputs) if masks else None
        direct_outputs = None


        for mj in regional:
            source = interfaces[mj]
            if mj in conductors or not len(pec_positions):
                continue
            trace, width = dofs[mj, 'plus']
            column, _ = dofs[mj, 'minus' if source['r_m'] == rid else 'plus']
            sign = 1.0 if source['r_m'] == rid else -1.0
            for start in range(0, len(pec_positions), BLOCK_ROWS):
                local = pec_positions[start:start+BLOCK_ROWS]
                block = matrix[np.ix_(np.arange(trace, trace+width), offset+local)].T
                matrix[np.ix_(offset+local, np.arange(column, column+width))] = sign * block
        block = None
    return matrix, layout


def assemble_system(mesh, infos, pol, obs_order=8, src_order=8):
    from ghost_backend.compressed.runtime import enabled, regional
    if enabled():return regional(mesh,infos,pol,obs_order,src_order)
    from ghost_backend.twod.assembly.session import current_session, system_key
    session = current_session()
    key = system_key(mesh, infos, 'multi_region', obs_order, src_order) if session is not None else None
    previous = session.take(key, pol) if session is not None else None
    if previous is not None and obs_order == src_order:
        matrix, old = previous
        layout = build_layout(mesh, infos, pol)
        return _reuse_te_system(matrix, old, layout, mesh, obs_order, src_order)
    previous = None
    from ghost_backend.twod.assembly.scatter import assemble_multi
    matrix, layout = assemble_multi(mesh, infos, pol, obs_order, src_order)
    if session is not None and obs_order == src_order:
        session.save(key, pol, (matrix, layout))
    return matrix, layout


def rhs_many(mesh, layout, k0, angles):
    """Integrate incident moments once, only on illuminated interfaces."""
    from ghost_backend.twod.assembly.kernels import incident_loads
    regions, interfaces, dofs = layout['region_props'], layout['ifaces'], layout['dof_map']
    mask = np.zeros(len(mesh.elements), bool)
    weights = np.zeros(len(mesh.elements), complex)
    for interface in interfaces:
        if any(regions.get(rid, {}).get('has_incident') for rid in (interface['r_m'], interface['r_p'])):
            mask[interface['eids']] = True
            weights[interface['eids']] = interface['robin_alpha_elements'][interface['eids']]
    bu, bdn = incident_loads(mesh, k0, angles, element_mask=mask,
                            observation_coefficients=weights)
    rhs = np.zeros((layout['n_dof'], len(angles)), complex)
    for mi, interface in enumerate(interfaces):
        nodes, n = interface['nodes'], interface['n']
        rm, rp = interface['r_m'], interface['r_p']
        if rm < 0 or rp < 0:
            rid, side = (rp, 'plus') if rm < 0 else (rm, 'minus')
            if regions[rid].get('has_incident'):
                offset, _ = dofs[mi, side]
                block = bdn[nodes].copy()
                if layout['polarization'] == 'TM':
                    pec = np.abs(interface['robin_alpha']) <= EPS
                    block[pec] = bu[nodes][pec]
                rhs[offset:offset+n] = -block
        else:
            flux, trace = dofs[mi, 'minus'][0], dofs[mi, 'plus'][0]
            if regions[rm].get('has_incident'):
                rhs[flux:flux+n] -= bdn[nodes]
                rhs[trace:trace+n] -= bu[nodes]
            if regions[rp].get('has_incident'):
                rhs[flux:flux+n] += _inverse_beta(layout, interface, layout['polarization']) * bdn[nodes]
                rhs[trace:trace+n] += bu[nodes]
    return rhs


def exterior_projection(mesh, layout):
    regions = layout['region_props']
    rid = next((rid for rid, region in regions.items() if region.get('has_incident')), 0)
    mask = np.zeros(len(mesh.elements), bool)
    mappings = []
    for mi, interface in enumerate(layout['ifaces']):
        side = 'minus' if interface['r_m'] == rid else 'plus' if interface['r_p'] == rid else None
        if side is not None:
            offset, count = layout['dof_map'][mi, side]
            mappings.append((interface['nodes'], offset, count))
            mask[interface['eids']] = True
    def density(solution):
        result = np.zeros((len(mesh.nodes), solution.shape[1]), complex)
        for nodes, offset, count in mappings:
            result[nodes] += solution[offset:offset+count]
        return result
    return mask, density


def dof_coordinates(mesh, layout):
    xy = np.zeros((len(mesh.nodes), 2))
    for element in mesh.elements:
        xy[element.node_ids[0]], xy[element.node_ids[1]] = element.p0, element.p1
    result = np.empty((layout['n_dof'], 2))
    for (mi, side), (offset, count) in layout['dof_map'].items():
        result[offset:offset+count] = xy[layout['ifaces'][mi]['nodes']]
    return result
