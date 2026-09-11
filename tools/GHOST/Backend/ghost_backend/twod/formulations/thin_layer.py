"""First-order isotropic thin-layer transmission model, including normal terms."""

import math
from ghost_backend.execution.runtime import dataclass, replace
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from ghost_backend.execution.metrics import timed_stage


@dataclass(frozen=True)
class ThinLayerDefinition:
    thickness_m: float
    dielectric_flag: int

    @classmethod
    def from_row(cls, row):
        if len(row) != 4 or str(row[1]).lower() != "thin_dielectric":
            raise ValueError("Thin layer requires: flag thin_dielectric thickness_m dielectric_flag")
        d, material = float(row[2]), float(row[3])
        if not math.isfinite(d) or d <= 0 or not math.isfinite(material) or material <= 0 or not material.is_integer():
            raise ValueError("Thin-layer thickness must be positive metres and dielectric flag a positive integer.")
        return cls(d, int(material))


def layer_for_mesh(mesh, materials, frequency_ghz):
    models = [materials.impedance_models.get(int(element.ibc_flag)) for element in mesh.elements]
    if not models or not all(isinstance(model, ThinLayerDefinition) for model in models):
        raise ValueError("Thin dielectric layers currently require an all-layer geometry; coupling to other boundary models is not implemented.")
    if len(set(models)) != 1:
        raise ValueError("All thin-layer segments must currently have the same thickness and dielectric material.")
    model = models[0]
    eps, mu = materials.get_medium(model.dielectric_flag, frequency_ghz)
    return eps, mu, model.thickness_m


def validate_thin_layer(epsilon, permeability, thickness_m, k0):
    from ghost_backend.twod.solver import _validate_passive_medium
    eps, mu = _validate_passive_medium(epsilon, permeability, "Thin dielectric layer")
    d, k = float(thickness_m), float(k0)
    if not math.isfinite(d) or d <= 0 or not math.isfinite(k) or k <= 0:
        raise ValueError("Thin-layer thickness and wavenumber must be positive and finite.")
    electrical_thickness = k * d * max(1.0, abs(complex(eps * mu) ** 0.5))
    if electrical_thickness > 0.15:
        raise ValueError(
            f"Thin-layer electrical thickness {electrical_thickness:.4g} exceeds "
            "k*d*max(1,|sqrt(epsilon*mu)|) <= 0.15. Use explicit bulk geometry."
        )
    return eps, mu, d, electrical_thickness


def _continuous_oriented_mesh(mesh):
    """Join geometric nodes and orient each nonbranching midsurface consistently.

    The general interface mesh splits traces by segment/material region. A
    homogeneous collapsed layer instead has one continuous trace at a shared
    geometric node, regardless of segment names or authoring direction. Work
    on copies so other formulations and cached geometry remain unchanged.
    """
    from ghost_backend.twod.solver import LinearMesh
    nodes_by_key = {}
    incident = {}
    ends = []
    for index, element in enumerate(mesh.elements):
        keys = tuple(mesh.nodes[node].key for node in element.node_ids)
        if keys[0] == keys[1]:
            raise ValueError("Thin-layer element collapses to one geometric node.")
        ends.append(keys)
        for key, node in zip(keys, element.node_ids):
            nodes_by_key.setdefault(key, mesh.nodes[node])
            incident.setdefault(key, []).append(index)
    if any(len(edges) > 2 for edges in incident.values()):
        raise ValueError("Thin-layer branching junctions require explicit bulk geometry.")
    keys = sorted(nodes_by_key)
    node_ids = {key: index for index, key in enumerate(keys)}
    pending = set(range(len(mesh.elements)))
    elements = []


    starts = sorted(key for key in keys if len(incident[key]) == 1) + keys
    for current in starts:
        while True:
            available = [edge for edge in incident[current] if edge in pending]
            if not available:
                break
            edge = min(available, key=lambda i: ends[i][1] if ends[i][0] == current else ends[i][0])
            pending.remove(edge)
            a, b = ends[edge]
            original = mesh.elements[edge]
            reverse = a != current
            following = a if reverse else b
            changes = {"node_ids": (node_ids[current], node_ids[following])}
            if reverse:
                changes.update(p0=original.p1, p1=original.p0,
                               tangent=-original.tangent, normal=-original.normal)
            elements.append(replace(original, **changes))
            current = following
    return LinearMesh(nodes=[nodes_by_key[key] for key in keys], elements=elements)


@timed_stage("thin_layer_operators_and_solve")
def solve_thin_layer_fields(mesh, k0, incidence_angles_deg, polarization,
                            epsilon, permeability, thickness_m, *,
                            observation_angles_deg=None,
                            condition_diagnostics=None, order=8):
    """Return width, stored complex field, residual and approximation evidence.

    The midsurface is a smooth closed curve or an open sheet. Junctions with
    other material models are deliberately not inferred from this API.
    """
    import ghost_backend.twod.solver as rcs
    eps, mu, d, electrical = validate_thin_layer(epsilon, permeability, thickness_m, k0)
    pol = str(polarization).upper()
    if pol not in {"TM", "TE"}:
        raise ValueError("Thin layer polarization must be TM or TE.")
    mesh = _continuous_oriented_mesh(mesh)
    n = len(mesh.nodes)
    alpha, beta = (eps, mu) if pol == "TM" else (mu, eps)
    B = d * (beta - 1.0)
    angles = np.asarray(incidence_angles_deg, float).reshape(-1)
    if n == 0 or not angles.size or not np.all(np.isfinite(angles)):
        raise ValueError("A thin-layer solve needs a mesh and finite incidence angles.")
    if observation_angles_deg is not None:
        observation_angles_deg = np.asarray(observation_angles_deg, float).reshape(-1)
        if not observation_angles_deg.size or not np.all(np.isfinite(observation_angles_deg)):
            raise ValueError('A thin-layer solve needs finite observation angles.')


    at_node = {}
    for element in mesh.elements:
        for node in element.node_ids:
            at_node.setdefault(mesh.nodes[node].key, []).append(element)
    max_curvature = 0.0
    for connected in at_node.values():
        if len(connected) > 2:
            raise ValueError("Thin-layer branching junctions require explicit bulk geometry.")
        if len(connected) == 2:
            a, b = connected
            turn = math.acos(float(np.clip(np.dot(a.tangent, b.tangent), -1, 1)))
            max_curvature = max(max_curvature, turn / (0.5 * (a.length + b.length)))
    curvature_ratio = d * max_curvature
    if curvature_ratio > 0.05:
        raise ValueError("Thin-layer thickness/curvature radius exceeds 0.05; use explicit bulk geometry.")

    zero_contrast = eps == 1 and mu == 1
    resources = dict(formulation='thin_dielectric_layer', analytic_zero=zero_contrast)
    from ghost_backend.compressed.runtime import enabled as compressed_enabled
    if compressed_enabled() and not zero_contrast:
        from ghost_backend.compressed.memory import geometry_storage
        resources['compressed_storage'] = geometry_storage(mesh, None, pol,
            'thin_dielectric_layer', k0, (eps,mu,d), n if B==0 else 2*n)
    required = rcs._estimate_memory_gb(n, False, system_dofs=n if B == 0 else 2*n,
        n_rhs=len(angles), dense_resources=resources)
    evidence = {
        'model': 'first_order_isotropic_transmitting_layer',
        'normal_polarization_terms': True,
        'thickness_m': d, 'electrical_thickness': electrical,
        'thickness_curvature_ratio': curvature_ratio,
        'unknowns': 0 if zero_contrast else n if B == 0 else 2*n,
        'estimated_peak_gib': required,
        'zero_field_jump_eliminated': B == 0,
        'zero_contrast_eliminated': zero_contrast,
        'approximation_error_certified': False,
        'limits': 'electrical thickness <= 0.15; thickness/radius <= 0.05; validate against bulk for application',
    }
    limit = rcs._solve_memory_limit_gb()
    if required > limit:
        raise MemoryError(rcs._memory_gate_message(required, limit, 'Thin-layer solve'))
    if zero_contrast:
        shape = len(angles) if observation_angles_deg is None else (len(angles), len(observation_angles_deg))
        field = np.zeros(shape, complex)
        if condition_diagnostics is not None:
            condition_diagnostics.update(condition_est=1., condition_method='analytic_zero_contrast',
                linear_backward_error=0., linear_backward_error_limit=rcs.DENSE_LINEAR_BACKWARD_ERROR_MAX)
        return rcs._rcs_sigma_from_amp(field, k0), field, 0., evidence

    from ghost_backend.twod.assembly.mass import sparse_mass, add_mass
    from ghost_backend.twod.assembly.kernels import incident_loads
    from ghost_backend.twod.fields import solve_fields
    from scipy.sparse import coo_matrix
    from ghost_backend.compressed.runtime import enabled, thin
    if enabled():
        matrix,rhs_builder=thin(mesh,k0,pol,eps,mu,d,order)
        _,field,residual,_=solve_fields(mesh,matrix,k0,angles,rhs_builder,condition_diagnostics,
            'thin dielectric layer',observation_angles=observation_angles_deg,order=order,
            second_potential='DLP' if B!=0 else None)
        return rcs._rcs_sigma_from_amp(field,k0),field,residual,evidence
    if B == 0:
        S, _ = rcs._assemble_linear_operator_matrices(mesh, k0, obs_normal_deriv=False,
            compute_double_layer=False, obs_order=order, src_order=order)
        coefficient = k0**2*d*(alpha-1.)
        matrix = S
        S = K = None
        matrix *= coefficient
        add_mass(matrix, mesh)
        def rhs_builder(batch):
            bu, _ = incident_loads(mesh, k0, batch, want_dn=False)
            bu *= -coefficient
            return bu
    else:
        matrix = np.zeros((2*n, 2*n), complex, order='F')
        S, K = rcs._assemble_linear_operator_matrices(mesh, k0, obs_normal_deriv=False,
            obs_order=order, src_order=order, single_layer_destination=matrix[:n, :n],
            double_layer_destination=matrix[:n, n:])
        mass = sparse_mass(mesh)
        C = k0**2 * d * (alpha - 1.0) * mass
        normal_term = d * (1.0 - 1.0 / beta)
        ids = np.asarray([element.node_ids for element in mesh.elements])
        lengths = np.asarray([element.length for element in mesh.elements])
        rows, columns = np.repeat(ids, 2, axis=1).ravel(), np.tile(ids, (1, 2)).ravel()
        values = (normal_term/lengths[:, None] * np.array([1., -1., -1., 1.])).ravel()
        C = (C + coo_matrix((values, (rows, columns)), shape=(n, n))).tocsr()
        mass_lu = splu(mass.tocsc())
        mass = None

        for start in range(0, n, 64):
            stop = min(start+64, n)
            matrix[n:, start:stop] = B * K.T[:, start:stop]


        for start in range(0, n, 64):
            stop = min(start+64, n)
            matrix[:n, start:stop] = C @ mass_lu.solve(S[:, start:stop])
            matrix[:n, n+start:n+stop] = C @ mass_lu.solve(K[:, start:stop])
        S = K = None
        add_mass(matrix[:n, :n], mesh)
        W = rcs._assemble_linear_hypersingular_matrix(mesh, k0, obs_order=order,
            src_order=order, destination=matrix[n:, n:])
        W *= -B
        W = None
        add_mass(matrix[n:, n:], mesh)
        endpoints = rcs._geometric_sheet_endpoint_nodes(mesh)
        matrix[n+endpoints, :] = 0
        matrix[n+endpoints, n+endpoints] = 1
        def rhs_builder(batch):
            bu, bq = incident_loads(mesh, k0, batch)
            rhs = np.empty((2*n, len(batch)), complex)
            rhs[:n] = -C @ mass_lu.solve(bu)
            rhs[n:] = -B * bq
            rhs[n+endpoints] = 0
            return rhs
    _, field, residual, _ = solve_fields(mesh, matrix, k0, angles,
        rhs_builder, condition_diagnostics, "thin dielectric layer",
        observation_angles=observation_angles_deg, order=order,
        second_potential='DLP' if B != 0 else None)
    return rcs._rcs_sigma_from_amp(field, k0), field, residual, evidence
