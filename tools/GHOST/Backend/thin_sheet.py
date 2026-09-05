"""First-order isotropic thin-layer transmission model, including normal terms.

The scalar field is E_z (TM) or H_z (TE). Across a collapsed layer in air,
with q = du/dn in air, alpha = epsilon (TM) / mu (TE), and beta its dual:

    [u] = d (beta - 1) q_average
    [q] = -k0^2 d (alpha - 1) u_average
          - d/ds [d (1/beta - 1) du_average/ds].

These follow by integrating the scalar Maxwell equation through a thin layer
and subtracting the air it replaces. Keeping the tangential derivative is
essential: a dielectric sheet is not just a tangential resistance at grazing
incidence. This is a first-order thickness approximation, not a bulk solver.
The existing G=+j H2/4 gives jumps [SLP q]=q and [DLP u]=-u.
"""

import math
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from solver_metrics import timed_stage


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
    from rcs_solver import _validate_passive_medium
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


@timed_stage("thin_layer_operators_and_solve")
def solve_thin_layer_fields(mesh, k0, incidence_angles_deg, polarization,
                            epsilon, permeability, thickness_m, *,
                            observation_angles_deg=None,
                            condition_diagnostics=None, order=8):
    """Return width, stored complex field, residual and approximation evidence.

    The midsurface is a smooth closed curve or an open sheet. Junctions with
    other material models are deliberately not inferred from this API.
    """
    import rcs_solver as rcs
    eps, mu, d, electrical = validate_thin_layer(epsilon, permeability, thickness_m, k0)
    pol = str(polarization).upper()
    if pol not in {"TM", "TE"}:
        raise ValueError("Thin layer polarization must be TM or TE.")
    n = len(mesh.nodes)
    alpha, beta = (eps, mu) if pol == "TM" else (mu, eps)
    B = d * (beta - 1.0)
    angles = np.asarray(incidence_angles_deg, float).reshape(-1)
    if n == 0 or not angles.size or not np.all(np.isfinite(angles)):
        raise ValueError("A thin-layer solve needs a mesh and finite incidence angles.")
    # Includes resident operators, coefficient products, the 2N system/LU,
    # sparse-mass solve scratch, all RHS and output density blocks.
    required = 16.0 * ((8 if B == 0 else 28) * n * n + 12 * n * len(angles)) / 1024**3
    limit = rcs._solve_memory_limit_gb()
    if required > limit:
        raise MemoryError(rcs._memory_gate_message(required, limit, "Thin-layer solve"))

    # Estimate curvature from adjacent straight-element tangents. Endpoints of
    # an open strip are not treated as turns. Sharp folds need bulk geometry.
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

    M = rcs._assemble_linear_mass_matrix(mesh)
    S, K = rcs._assemble_linear_operator_matrices(mesh, k0, obs_normal_deriv=False, compute_double_layer=(B != 0),
                                                 obs_order=order, src_order=order)
    bu, bq = np.zeros((n, len(angles)), complex), np.zeros((n, len(angles)), complex)
    for element in mesh.elements:
        ids = np.asarray(element.node_ids)
        bu[ids] += rcs._linear_element_incident_load_many(element, k_air=k0, elevations_deg=angles)
        bq[ids] += rcs._linear_element_incident_dn_load_many(element, k_air=k0, elevations_deg=angles)
    if B == 0:
        # Nonmagnetic TM layers have no field jump. Eliminate that density
        # exactly: one N system, no hypersingular or double-layer assembly.
        coefficient = k0**2*d*(alpha-1.)
        matrix = M + coefficient*S
        rhs = -coefficient*bu
    else:
        W = rcs._assemble_linear_hypersingular_matrix(mesh, k0, obs_order=order, src_order=order)
        C = k0**2 * d * (alpha - 1.0) * M
        normal_term = d * (1.0 - 1.0 / beta)
        for element in mesh.elements:
            ids = np.asarray(element.node_ids)
            C[np.ix_(ids, ids)] += normal_term / element.length * np.array([[1., -1.], [-1., 1.]])
        # Sparse mass factorization maps weak trace loads to nodal values.
        mass_lu = splu(csc_matrix(M))
        CM = mass_lu.solve(C.T, trans="T").T
        matrix = np.block([[M + CM @ S, CM @ K], [B * K.T, M - B * W]])
        rhs = np.vstack((-CM @ bu, -B * bq))
        for node in rcs._geometric_sheet_endpoint_nodes(mesh):
            matrix[n + node, :] = 0
            matrix[n + node, n + node] = 1
            rhs[n + node, :] = 0
    solution = rcs._solve_dense_system(matrix, rhs, condition_diagnostics, "thin dielectric layer")
    residuals = np.linalg.norm(matrix @ solution - rhs, axis=0)
    denominators = np.linalg.norm(rhs, axis=0)
    residual = float(np.max(residuals / np.where(denominators > 0, denominators, 1)))
    obs = angles if observation_angles_deg is None else np.asarray(observation_angles_deg, float)
    projection = "matched" if observation_angles_deg is None else "grid"
    field = rcs._farfield_linear_density_many(mesh, solution[:n], k0, obs, "SLP", projection=projection)
    if B != 0:
        field += rcs._farfield_linear_density_many(mesh, solution[n:], k0, obs, "DLP", projection=projection)
    evidence = {
        "model": "first_order_isotropic_transmitting_layer",
        "normal_polarization_terms": True,
        "thickness_m": d, "electrical_thickness": electrical,
        "thickness_curvature_ratio": curvature_ratio,
        "unknowns": int(matrix.shape[0]), "estimated_peak_gib": required,
        "zero_field_jump_eliminated": B == 0,
        "approximation_error_certified": False,
        "limits": "electrical thickness <= 0.15; thickness/radius <= 0.05; validate against bulk for application",
    }
    return rcs._rcs_sigma_from_amp(field, k0), field, residual, evidence
