"""Local weak mass terms without an N-by-N dense mass allocation."""
import numpy as np
from scipy.sparse import coo_matrix


def sparse_mass(mesh, coefficients=None):
    from ghost_backend.twod.operators import _linear_mass_block
    count = len(mesh.elements)
    coefficients = np.ones(count, complex) if coefficients is None else np.asarray(coefficients, complex).reshape(-1)
    if coefficients.size != count or not np.all(np.isfinite(coefficients)):
        raise ValueError('Mass coefficients must be finite and match the elements.')
    ids = np.asarray([e.node_ids for e in mesh.elements], dtype=int).reshape(-1, 2)
    values = np.asarray([c * _linear_mass_block(e) for e, c in zip(mesh.elements, coefficients)], complex).reshape(-1)
    rows = np.repeat(ids, 2, axis=1).reshape(-1)
    cols = np.tile(ids, (1, 2)).reshape(-1)
    return coo_matrix((values, (rows, cols)), shape=(len(mesh.nodes), len(mesh.nodes))).tocsr()


def add_mass(matrix, mesh, coefficient=1.0, element_coefficients=None):
    """Add the assembled weak mass term to an owned dense destination in place."""
    mass = sparse_mass(mesh, element_coefficients).tocoo()
    matrix[mass.row, mass.col] += coefficient * mass.data
    return matrix
