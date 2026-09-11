"""Owned two-density dielectric system; reuse its coefficients for TE/TM."""
import numpy as np
from ghost_backend.twod.assembly.mass import add_mass
from ghost_backend.twod.assembly.session import current_session, system_key


def assemble_system(mesh, infos, pol, k0, obs_order=8, src_order=8):
    import ghost_backend.twod.solver as rcs
    from ghost_backend.compressed.runtime import enabled, native
    if enabled():return native(mesh,infos,pol,k0,'dielectric',obs_order,src_order)[0]
    n = len(mesh.nodes)
    info = infos[0]
    numerator, denominator = ((info.mu_minus, info.mu_plus) if pol == 'TM'
                              else (info.eps_minus, info.eps_plus))
    factor = complex(numerator / denominator) if abs(denominator) > rcs.EPS else 1.0
    session = current_session()
    key = system_key(mesh, infos, 'dielectric', obs_order, src_order) if session is not None else None
    previous = session.take(key, pol) if session is not None else None
    if previous is not None:
        matrix, old_factor = previous
        if abs(old_factor) > rcs.EPS:
            matrix[n:, n:] *= factor / old_factor
            return matrix
        previous = matrix = None
    k1_vals = {complex(i.k_plus) for i in infos if i.plus_region > 0}
    if not k1_vals:
        k1_vals = {complex(i.k_minus) for i in infos if i.minus_region > 0}
    if not k1_vals:
        raise ValueError('Dielectric indirect solver requires a dielectric region.')
    k1 = k1_vals.pop()
    matrix = np.zeros((2*n, 2*n), complex, order='F')
    _, k = rcs._assemble_linear_operator_matrices(mesh, k0, False,
        obs_order=obs_order, src_order=src_order, compute_single_layer=False,
        double_layer_destination=matrix[:n, :n])
    k = None
    add_mass(matrix[:n, :n], mesh, .5)
    s, k = rcs._assemble_linear_operator_matrices(mesh, k1, True,
        obs_order=obs_order, src_order=src_order,
        single_layer_destination=matrix[:n, n:], double_layer_destination=matrix[n:, n:])
    matrix[:n, n:] *= -1
    s = k = None
    add_mass(matrix[n:, n:], mesh, .5)
    matrix[n:, n:] *= factor
    w = rcs._assemble_linear_hypersingular_matrix(mesh, k0, obs_order=obs_order,
        src_order=src_order, destination=matrix[n:, :n])
    w = None
    if session is not None:
        session.save(key, pol, (matrix, factor))
    return matrix


def rhs_many(mesh, k0, angles):
    from ghost_backend.twod.assembly.kernels import incident_loads
    n = len(mesh.nodes)
    rhs = np.zeros((2*n, len(angles)), complex)
    incident_loads(mesh, k0, angles, rhs[:n], rhs[n:])
    rhs[:n] *= -1
    return rhs
