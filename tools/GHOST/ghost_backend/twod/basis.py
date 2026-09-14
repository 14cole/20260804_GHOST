"""Continuous nodal polynomial bases on the unchanged straight input geometry.

Endpoints are always the first two local nodes. Interior nodes belong to one
element, so enrichment never joins distinct material interfaces at a junction.
"""
from functools import lru_cache
import numpy as np
from ghost_backend.execution.options import option


@lru_cache(maxsize=4)
def abscissae(degree):
    if degree not in (1, 2, 3):
        raise ValueError('Boundary polynomial degree must be 1, 2 or 3.')
    result = np.array([0., 1.] + [i / degree for i in range(1, degree)])
    result.flags.writeable = False
    return result


@lru_cache(maxsize=4)
def coefficients(degree):
    x = abscissae(degree)
    result = np.linalg.inv(np.vander(x, degree + 1, increasing=True))
    result.flags.writeable = False
    return result


def values(x, degree=1, derivative=False):
    x = np.asarray(x, float)
    c = coefficients(degree)
    if derivative:
        c = c[1:] * np.arange(1, degree + 1)[:, None]
    return np.polynomial.polynomial.polyval(x, c).T


@lru_cache(maxsize=4)
def derivative_matrix(degree):
    return values(abscissae(degree), degree, True)


def mesh_degree(mesh):
    return len(mesh.elements[0].node_ids) - 1 if mesh.elements else 1


def enrich(mesh, stats=None):
    """Add interior interpolation nodes while preserving endpoint signatures."""
    degree = option('basis_order', 1)
    if degree == 1:
        return mesh, stats
    from ghost_backend.twod.geometry import LinearNode, _linear_node_snap_key
    for element in mesh.elements:
        ids = list(element.node_ids)
        for t in abscissae(degree)[2:]:
            point = element.p0 + t * (element.p1 - element.p0)
            ids.append(len(mesh.nodes))
            mesh.nodes.append(LinearNode(point, _linear_node_snap_key(point)))
        element.node_ids = tuple(ids)
    if stats is not None:
        stats = dict(stats, polynomial_degree=degree, enriched_nodes=len(mesh.nodes))
    return mesh, stats


def mass_block(element):
    degree = len(element.node_ids) - 1
    q, w = np.polynomial.legendre.leggauss(degree + 1)
    phi = values((q + 1) / 2, degree)
    return element.length * (phi.T * (w / 2)) @ phi


def stiffness_block(element):
    degree = len(element.node_ids) - 1
    q, w = np.polynomial.legendre.leggauss(degree + 1)
    dphi = values((q + 1) / 2, degree, True)
    return (dphi.T * (w / 2)) @ dphi / element.length


def integral_bounds(element):
    """Conservative integral of |basis| for attenuation error accounting."""
    if len(element.node_ids) == 2:
        return np.full(2, element.length*.5)
    c = coefficients(len(element.node_ids) - 1)
    return element.length * np.sum(abs(c) / np.arange(1, len(c) + 1)[:, None], axis=0)


def plane_wave_moments(centers, edges, lengths, k, dirs, degree):
    # Gauss order resolves the phase as well as the polynomial; never silently
    # clip the order on explicit, electrically long elements.
    phase_span = abs(float(k)) * np.max(np.linalg.norm(edges, axis=1), initial=0.)
    order = max(12, degree + 3 + int(np.ceil(phase_span)))
    if order > 256:
        raise ValueError('Polynomial excitation needs a finer wavelength mesh.')
    q, w = np.polynomial.legendre.leggauss(order)
    t = (q + 1) / 2
    phi = values(t, degree)
    base = np.exp(1j * float(k) * (centers @ dirs.T)) * lengths[:, None]
    z = float(k) * (edges @ dirs.T)
    result = np.zeros((len(lengths), degree + 1, len(dirs)), complex)
    for i in range(order):
        term = base * np.exp(1j * (t[i] - .5) * z) * (w[i] / 2)
        result += term[:, None, :] * phi[i][None, :, None]
    return result
