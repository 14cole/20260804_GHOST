"""Checked HODLR inverse with exact-matrix residual refinement."""
from ghost_backend.execution.options import environment_value
import os
import warnings
import numpy as np
import scipy.linalg as la
from ghost_backend.linalg.workspace import matrix_inf_norm


class HierarchicalRejected(RuntimeError):
    pass


def factor_mode():
    value = environment_value('GHOST_CPU_FACTORIZATION', 'dense').strip().lower()
    if value not in ('dense', 'hierarchical', 'auto', 'compressed'):
        raise ValueError('GHOST_CPU_FACTORIZATION must be dense, hierarchical, auto, or compressed.')
    return value


def factor_storage_budget(matrix_bytes):


    return max(16*512**2, int(matrix_bytes*.65))


def spatial_order(coordinates, ids, leaf=128):
    """Spatial permutation without a recursive closure retaining coordinates."""
    if len(ids) <= leaf:
        return ids
    axis = int(np.argmax(np.ptp(coordinates[ids], axis=0)))
    ids = ids[np.argsort(coordinates[ids, axis], kind='mergesort')]
    mid = len(ids)//2
    return np.r_[spatial_order(coordinates, ids[:mid], leaf),
                 spatial_order(coordinates, ids[mid:], leaf)]


class Block:
    def __init__(self, a, rows, cols, checkpoint):
        self.a, self.rows, self.cols, self.checkpoint = a, rows, cols, checkpoint
        self.shape = len(rows), len(cols)

    def row(self, i):
        return self.a[self.rows[i], self.cols].copy()

    def col(self, j):
        return self.a[self.rows, self.cols[j]].copy()

    def error(self, u, v):
        total = error = largest = 0.
        pivot = 0
        for start in range(0, len(self.rows), 32):
            self.checkpoint()
            original = self.a[np.ix_(self.rows[start:start+32], self.cols)]
            total += float(np.vdot(original, original).real)
            original -= u[start:start+32] @ v
            norms = np.sum(np.abs(original)**2, axis=1)
            error += float(np.sum(norms))
            if len(norms) and norms.max() > largest:
                largest = float(norms.max())
                pivot = start+int(np.argmax(norms))
        return np.sqrt(error/max(total, 1e-300)), pivot


def compress(block, tolerance=2e-10, maximum_rank=256):
    m, n = block.shape
    limit = min(maximum_rank, min(m, n)//2)
    columns = np.empty((m, limit), complex, order='F')
    rows = np.empty((limit, n), complex)
    u, v = columns[:, :0], rows[:0]
    used = np.zeros(m, bool)
    pivot = 0
    error = np.inf
    accumulated = 0.
    def balanced(error):
        if not u.shape[1]:
            return u.copy(), v.copy(), error
        q, r = la.qr(u, mode='economic', check_finite=False)
        return q, r @ v, error
    for rank in range(limit):
        block.checkpoint()
        row = block.row(pivot) - u[pivot] @ v
        column = int(np.argmax(np.abs(row)))
        scale = row[column]
        if abs(scale) < 1e-290:
            error, pivot = block.error(u, v)
            if error <= tolerance:
                return balanced(error)
            row = block.row(pivot) - u[pivot] @ v
            column = int(np.argmax(np.abs(row)))
            scale = row[column]
            if abs(scale) < 1e-290:
                raise HierarchicalRejected('Unable to resolve an off-diagonal pivot.')
        col = block.col(column) - u @ v[:, column]
        used[pivot] = True
        columns[:, rank], rows[rank] = col/scale, row
        u, v = columns[:, :rank+1], rows[:rank+1]
        contribution = np.linalg.norm(columns[:, rank])*np.linalg.norm(row)
        accumulated += contribution
        candidate = np.abs(col)
        candidate[used] = -1
        pivot = int(np.argmax(candidate))


        if contribution <= tolerance*accumulated or rank+1 == limit:
            error, worst = block.error(u, v)
            if error <= tolerance:

                return balanced(error)
            pivot = worst
    raise HierarchicalRejected('Off-diagonal rank exceeds {} (relative error {:.3g}).'.format(limit, error))


class Node:
    def solve(self, b, trans=0):
        if self.leaf:
            return la.lu_solve(self.lu, b, trans=trans, check_finite=False)
        n = self.left.n
        if self.lu is None:
            return np.vstack((self.left.solve(b[:n], trans), self.right.solve(b[n:], trans)))
        if trans == 0:
            z1, z2 = self.left.solve(b[:n]), self.right.solve(b[n:])
            small = np.vstack((self.v12 @ z2, self.v21 @ z1))
            correction = la.lu_solve(self.lu, small, check_finite=False)
            r = self.e1.shape[1]
            return np.vstack((z1-self.e1 @ correction[:r], z2-self.e2 @ correction[r:]))
        def adj(a):
            return a.T if trans == 1 else a.conj().T
        small = np.vstack((adj(self.e1) @ b[:n], adj(self.e2) @ b[n:]))
        correction = la.lu_solve(self.lu, small, trans=trans, check_finite=False)
        r = self.e1.shape[1]
        return np.vstack((self.left.solve(b[:n]-adj(self.v21) @ correction[r:], trans),
                          self.right.solve(b[n:]-adj(self.v12) @ correction[:r], trans)))


class HierarchicalFactor:
    def __init__(self, matrix, coordinates=None, checkpoint=None, matrix_norm=None):
        self.a = matrix
        self.checkpoint = checkpoint or (lambda: None)
        self.n = len(matrix)
        self.budget = factor_storage_budget(matrix.nbytes)
        self.root = None
        self.evidence = dict(backend='hodlr', builds=0, tighter_rebuilds=0)
        coordinates = np.arange(self.n)[:, None] if coordinates is None else np.asarray(coordinates)
        self.permutation = spatial_order(coordinates, np.arange(self.n))
        self.norms = (matrix_inf_norm(matrix) if matrix_norm is None else matrix_norm,
                      matrix_inf_norm(matrix.T))
        failed = False
        try:
            self._rebuild(1e-6)
        except (HierarchicalRejected, np.linalg.LinAlgError, RuntimeWarning) as exc:
            self.evidence['coarse_rejection'] = str(exc)
            failed = True
        if failed:


            self._rebuild(2e-10)

    def _rebuild(self, tolerance):


        self.root = None
        self.bytes = 0
        self.tolerance = tolerance
        self.refinement_tolerance = 3e-15 if tolerance > 2e-10 else 3e-14
        self.evidence['builds'] += 1
        self.evidence['tighter_rebuilds'] += int(tolerance == 2e-10)
        self.evidence.update(leaves=0, low_rank_blocks=0, max_rank=0,
            max_block_error=0., tolerance=tolerance, factor_bytes=0, refinements=0,
            refinement_tolerance=self.refinement_tolerance)
        self.root = self._build(self.permutation)
        self.evidence['factor_bytes'] = self.bytes

    def _reserve(self, count):
        self.bytes += int(count)
        if self.bytes > self.budget:
            raise HierarchicalRejected('Hierarchical factor exceeded its storage budget.')

    def _lu(self, matrix):
        self._reserve(matrix.nbytes + 8*len(matrix))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            return la.lu_factor(np.asarray(matrix, order='F'), overwrite_a=True, check_finite=False)

    def _build(self, ids):
        self.checkpoint()
        node = Node()
        node.n, node.leaf = len(ids), len(ids) <= 128
        if node.leaf:
            node.lu = self._lu(self.a[np.ix_(ids, ids)])
            self.evidence['leaves'] += 1
            return node
        mid = len(ids)//2
        left, right = ids[:mid], ids[mid:]


        try:
            u12, node.v12, err1 = compress(Block(self.a, left, right, self.checkpoint), self.tolerance)
            u21, node.v21, err2 = compress(Block(self.a, right, left, self.checkpoint), self.tolerance)
        except HierarchicalRejected:
            if node.n > 512:
                raise
            node.leaf = True
        if node.leaf:


            u12 = None
            node.v12 = node.v21 = None
            node.lu = self._lu(self.a[np.ix_(ids, ids)])
            self.evidence['leaves'] += 1
            return node
        r, s = u12.shape[1], u21.shape[1]
        self._reserve(u12.nbytes+u21.nbytes+node.v12.nbytes+node.v21.nbytes)
        self.evidence['low_rank_blocks'] += 2
        self.evidence['max_rank'] = max(self.evidence['max_rank'], r, s)
        self.evidence['max_block_error'] = max(self.evidence['max_block_error'], err1, err2)
        node.left, node.right = self._build(left), self._build(right)
        node.e1, node.e2 = node.left.solve(u12), node.right.solve(u21)
        small = np.eye(r+s, dtype=complex)
        small[:r, r:] = node.v12 @ node.e2
        small[r:, :r] = node.v21 @ node.e1
        node.lu = self._lu(small) if r+s else None
        return node

    def _apply(self, b, trans):
        ordered = self.root.solve(b[self.permutation], trans)
        result = np.empty_like(ordered)
        result[self.permutation] = ordered
        return result

    def solve(self, rhs, trans=0, return_residual=False):
        if trans not in (0, 1, 2):
            raise ValueError('Invalid transpose mode.')
        b = np.asarray(rhs, complex)
        if b.ndim not in (1, 2) or b.shape[0] != self.n or (b.ndim == 2 and not b.shape[1]):
            raise ValueError('RHS must match the nonempty system.')
        if not np.all(np.isfinite(b)):
            raise ValueError('Nonfinite RHS.')
        try:
            return self._solve_refined(b, trans, return_residual)
        except (HierarchicalRejected, np.linalg.LinAlgError, RuntimeWarning) as exc:
            if self.tolerance <= 2e-10:
                raise
            self.evidence['coarse_rejection'] = str(exc)

        self._rebuild(2e-10)
        return self._solve_refined(b, trans, return_residual)

    def _solve_refined(self, b, trans, return_residual):
        vector = b.ndim == 1
        if vector:
            b = b[:, None]
        x = self._apply(b, trans)

        def product(value):
            if trans == 2:
                return (self.a.T @ value.conj()).conj()
            return (self.a if trans == 0 else self.a.T) @ value
        for step in range(8):
            self.checkpoint()
            residual = b-product(x)
            denominator = self.norms[int(trans != 0)]*np.max(abs(x),axis=0)+np.max(abs(b),axis=0)
            error = float(np.max(np.max(abs(residual),axis=0)/np.maximum(denominator,1e-300)))
            if np.isfinite(error) and error <= self.refinement_tolerance:
                self.evidence['refinements'] = max(self.evidence['refinements'], step)
                solution = x[:, 0] if vector else x
                if return_residual:
                    return solution, residual[:, 0] if vector else residual
                return solution
            if step == 7:
                break
            x += self._apply(residual, trans)
        raise HierarchicalRejected('Exact-matrix residual refinement did not converge.')
