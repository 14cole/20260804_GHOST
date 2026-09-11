"""Shared sheet and sheet/PEC matrix assembly, including open-edge conditions."""
import numpy as np
from ghost_backend.twod.assembly.mass import add_mass


def assemble_system(mesh, infos, pol, k0, obs_order=8, src_order=8):
    import ghost_backend.twod.solver as rcs
    from ghost_backend.compressed.runtime import enabled, native
    if enabled():
        operator,oracle=native(mesh,infos,pol,k0,'sheet',obs_order,src_order)
        return operator,oracle.endpoints
    z = np.asarray([complex(i.robin_impedance) if int(i.seg_type) == 1 else 0.0 for i in infos])
    endpoints = np.empty(0, dtype=int)
    if pol == 'TM':
        operator, _ = rcs._assemble_linear_operator_matrices(mesh, k0, False,
            obs_order=obs_order, src_order=src_order, compute_double_layer=False)
        coefficient = z / (1j * float(k0) * rcs.ETA0)
    else:
        destination = np.zeros((len(mesh.nodes), len(mesh.nodes)), complex, order='F')
        operator = rcs._assemble_linear_hypersingular_matrix(mesh, k0,
            obs_order=obs_order, src_order=src_order, destination=destination)
        coefficient = (1j * float(k0) / rcs.ETA0) * z
        endpoints = rcs._geometric_sheet_endpoint_nodes(mesh, infos)
    matrix = operator
    operator = None
    if np.any(coefficient != 0):
        add_mass(matrix, mesh, -1., coefficient)
    if endpoints.size:
        matrix[endpoints] = 0
        matrix[endpoints, endpoints] = 1
    return matrix, endpoints


def rhs_many(mesh, k0, angles, pol, endpoints):
    from ghost_backend.twod.assembly.kernels import incident_loads
    bu, bdn = incident_loads(mesh, k0, angles, want_u=pol == 'TM', want_dn=pol == 'TE')
    rhs = -bu if pol == 'TM' else bdn
    if endpoints.size:
        rhs[endpoints] = 0
    return rhs
