"""Bounded coefficient queries for BOR's existing modal equation recipes.

Expressions preserve block signs, material weights and sparse pole/junction
constraints while deferring coefficient evaluation until a tile is requested.
No full nodal far block or assembled modal matrix is created on this path.
"""
import math
import numpy as np
from scipy.sparse import csr_matrix, issparse


def ids(index, size):
    if isinstance(index, slice):
        return np.arange(size)[index]
    value = np.asarray(index).reshape(-1)
    return np.flatnonzero(value) if value.dtype == bool else value.astype(np.intp)


class TileExpression:
    __array_priority__ = 1000

    def __init__(self, shape, query, row_coordinates=None, column_coordinates=None):
        self.shape = tuple(shape)
        self._query = query
        self.n = self.shape[0]
        self.calls = self.entries = self.dropped_routes = 0
        self.row_coordinates = row_coordinates
        self.column_coordinates = column_coordinates

    def get(self, rows, cols):
        return self._query(np.asarray(rows, dtype=np.intp), np.asarray(cols, dtype=np.intp))

    def get_with_error(self, rows, cols):
        self.calls += 1
        self.entries += len(rows)*len(cols)
        value = self.get(rows, cols)
        return value, np.zeros(value.shape, float)

    def __getitem__(self, key):
        rows, cols = ids(key[0], self.shape[0]), ids(key[1], self.shape[1])
        query = self._query
        return TileExpression((len(rows), len(cols)), lambda r, c: query(rows[r], cols[c]),
            None if self.row_coordinates is None else self.row_coordinates[rows],
            None if self.column_coordinates is None else self.column_coordinates[cols])

    def __setitem__(self, key, value):
        rows, cols = ids(key[0], self.shape[0]), ids(key[1], self.shape[1])
        value = expression(value, (len(rows), len(cols)))
        if value.shape != (len(rows), len(cols)):
            raise ValueError('Modal block assignment has incompatible dimensions.')
        old, new = self._query, value._query
        for name, selected, count in (('row_coordinates', rows, self.shape[0]),
                                       ('column_coordinates', cols, self.shape[1])):
            coordinates = getattr(value, name)
            if coordinates is not None:
                target = getattr(self, name)
                if target is None:
                    target = np.zeros((count, coordinates.shape[1]))
                    setattr(self, name, target)
                target[selected] = coordinates
        row_lookup = {int(value): i for i, value in enumerate(rows)}
        col_lookup = {int(value): i for i, value in enumerate(cols)}
        # Immutable snapshots make block += safe, including repeated regional terms.
        def query(r, c):
            out = old(r, c)
            ri = [i for i, value in enumerate(r) if int(value) in row_lookup]
            ci = [i for i, value in enumerate(c) if int(value) in col_lookup]
            rr = np.array([row_lookup[int(r[i])] for i in ri], dtype=np.intp)
            cc = np.array([col_lookup[int(c[i])] for i in ci], dtype=np.intp)
            if len(ri) and len(ci):
                out[np.ix_(ri, ci)] = new(rr, cc)
            return out
        self._query = query

    def __add__(self, other):
        other = expression(other, self.shape)
        if other.shape != self.shape:
            raise ValueError('Incompatible modal operator dimensions.')
        left, right = self._query, other._query
        return TileExpression(self.shape, lambda r, c: left(r, c) + right(r, c),
            self.row_coordinates if self.row_coordinates is not None else other.row_coordinates,
            self.column_coordinates if self.column_coordinates is not None else other.column_coordinates)

    __radd__ = __add__

    def __neg__(self):
        return self * -1.

    def __sub__(self, other):
        return self + -expression(other, self.shape)

    def __mul__(self, scale):
        if not np.isscalar(scale):
            return NotImplemented
        query = self._query
        return TileExpression(self.shape, lambda r, c: scale * query(r, c),
                              self.row_coordinates, self.column_coordinates)

    __rmul__ = __mul__

    def __matmul__(self, right):
        if not issparse(right):
            raise TypeError('Modal coefficient expressions only multiply sparse constraint maps.')
        right = csr_matrix(right)
        query = self._query
        def product(r, c):
            local = right[:, c].tocsr()
            used = np.flatnonzero(np.diff(local.indptr))
            return (local[used].T @ query(r, used).T).T
        return TileExpression((self.shape[0], right.shape[1]), product,
                              self.row_coordinates, projected_coordinates(right, self.column_coordinates))

    def reduce(self, transform):
        q = csr_matrix(transform)
        query = self._query
        def reduced(r, c):
            left, right = q[:, r].tocsr(), q[:, c].tocsr()
            rr = np.flatnonzero(np.diff(left.indptr))
            cc = np.flatnonzero(np.diff(right.indptr))
            return left[rr].conj().T @ (right[cc].T @ query(rr, cc).T).T
        return TileExpression((q.shape[1], q.shape[1]), reduced,
                              projected_coordinates(q, self.row_coordinates),
                              projected_coordinates(q, self.column_coordinates))


def projected_coordinates(transform, coordinates):
    if coordinates is None:
        return None
    weights = abs(transform)
    sums = np.asarray(weights.sum(axis=0)).ravel()
    return (weights.T @ coordinates)/np.maximum(sums[:, None], 1.e-300)


def expression(value, shape=None):
    if isinstance(value, TileExpression):
        return value
    if np.isscalar(value):
        return TileExpression(shape, lambda r, c: np.full((len(r), len(c)), value, complex))
    if issparse(value):
        value = csr_matrix(value)
        return TileExpression(value.shape, lambda r, c: value[r][:, c].toarray())
    value = np.asarray(value)
    return TileExpression(value.shape, lambda r, c: value[np.ix_(r, c)].copy())


def modal_matrix(shape, compressed=False):
    return expression(0., shape) if compressed else np.zeros(shape, complex)


def modal_block(blocks):
    if not any(isinstance(b, TileExpression) for row in blocks for b in row):
        return np.block(blocks)
    heights = [row[0].shape[0] for row in blocks]
    widths = [b.shape[1] for b in blocks[0]]
    out = modal_matrix((sum(heights), sum(widths)), True)
    r = 0
    for row, height in zip(blocks, heights):
        c = 0
        for value, width in zip(row, widths):
            out[r:r+height, c:c+width] = value
            c += width
        r += height
    return out


def mass_expression(solver, weight=None):
    g = solver.g
    weights = g.w*g.rho*(1. if weight is None else weight)*2*np.pi
    def query(rows, cols):
        points, t, _ = support(solver, np.union1d(rows, cols))
        nodes = np.union1d(rows, cols)
        return (t[np.searchsorted(nodes, rows)] * weights[points]) @ t[np.searchsorted(nodes, cols)].T
    return TileExpression((solver.Nn, solver.Nn), query, solver.gen.nodes, solver.gen.nodes)


def support(solver, nodes):
    """Local triangle bases without a global N-by-Gauss matrix."""
    g = solver.g
    elems = np.unique(np.concatenate((nodes-1, nodes)))
    elems = elems[(elems >= 0) & (elems < solver.gen.n_elems)]
    points = (elems[:, None]*solver.gauss_order + np.arange(solver.gauss_order)).ravel()
    e = g.elem[points]
    t = (nodes[:, None] == e)*g.T0[points] + (nodes[:, None] == e+1)*g.T1[points]
    d = (nodes[:, None] == e)*g.dRT0[points] + (nodes[:, None] == e+1)*g.dRT1[points]
    return points, t, d


class Primitive:
    def __init__(self, owner, kind, m, m_max, weight=None, element_weight=None):
        self.owner, self.kind, self.m, self.mm = owner, kind, m, m_max
        self.cross = hasattr(owner, 'sp')
        self.sp = owner.sp if self.cross else owner
        self.sq = owner.sq if self.cross else owner
        self.weight, self.element_weight = weight, element_weight
        self.checkpoint = getattr(self.sp, '_checkpoint', None) or (lambda: None)
        self.cache_key = self.cache = None
        from ghost_backend.bor.cache import current_cache
        self.tile_cache = current_cache()
        self.cache_token = object()
        self.shape = (2*self.sp.Nn, 2*self.sq.Nn)

    def query(self, rows, cols):
        # Chunk the nodal contraction as well as the angular FFT workspace.
        out = np.zeros((len(rows), len(cols)), complex)
        if self.tile_cache is not None and self.tile_cache.budget:
            # Canonical nodal tiles are reused across t/phi field families,
            # matrix tiles and sparse constraint routes within this mode.
            width = max(1, min(4, int(math.sqrt(8e6 / (64.*(2*self.mm+3))) /
                                      (2*max(self.sp.gauss_order, self.sq.gauss_order)))))
            rg, cg = (rows % self.sp.Nn)//width, (cols % self.sq.Nn)//width
            for rb in np.unique(rg):
                ri = np.flatnonzero(rg == rb)
                rn = np.arange(rb*width, min((rb+1)*width, self.sp.Nn))
                for cb in np.unique(cg):
                    self.checkpoint()
                    ci = np.flatnonzero(cg == cb)
                    cn = np.arange(cb*width, min((cb+1)*width, self.sq.Nn))
                    key = (self.cache_token, int(rb), int(cb))
                    blocks = self.tile_cache.get(key)
                    if blocks is None:
                        blocks = self.blocks(rn, cn)
                        self.tile_cache.put(key, blocks)
                    r, c = rows[ri], cols[ci]
                    field = 2*(r//self.sp.Nn)[:, None] + (c//self.sq.Nn)[None, :]
                    out[np.ix_(ri, ci)] = blocks[field, (r % self.sp.Nn-rn[0])[:, None],
                                                 (c % self.sq.Nn-cn[0])[None, :]]
            return out
        width = max(1, min(8, int(math.sqrt(8e6 / (64.*(2*self.mm+3))) /
                                  (2*max(self.sp.gauss_order, self.sq.gauss_order)))))
        for i in range(0, len(rows), width):
            for j in range(0, len(cols), width):
                self.checkpoint()
                r, c = rows[i:i+width], cols[j:j+width]
                rn, cn = np.unique(r % self.sp.Nn), np.unique(c % self.sq.Nn)
                blocks = self.blocks(rn, cn)
                a, b = r//self.sp.Nn, c//self.sq.Nn
                ri, ci = np.searchsorted(rn, r % self.sp.Nn), np.searchsorted(cn, c % self.sq.Nn)
                out[i:i+len(r), j:j+len(c)] = blocks[(2*a[:, None]+b[None, :]), ri[:, None], ci[None, :]]
        return out

    def blocks(self, rows, cols):
        from ghost_backend.bor.solver import _pair_blocks
        from ghost_backend.bor.kernels import (modal_kernels_fft, kernels_for_mode,
            mfie_kernels_fft, ibc_kernels_fft, n_xi_for_pairs)
        key = (rows.tobytes(), cols.tobytes())
        if key == self.cache_key:
            return self.cache
        sp, sq, owner = self.sp, self.sq, self.owner
        ip, tp, dp = support(sp, rows)
        iq, tq, dq = support(sq, cols)
        gp, gq = sp.g, sq.g
        ep, eq = gp.elem[ip], gq.elem[iq]
        near = np.zeros((len(ip), len(iq)), bool)
        pairs = owner.near_set if self.cross else None
        for e in np.unique(ep):
            fs = [f for f in np.unique(eq) if ((int(e), int(f)) in pairs if self.cross
                    else int(f) in owner._near_sources_by_element[int(e)])]
            near[np.ix_(ep == e, np.isin(eq, fs))] = True
        gap = owner._far_gap if self.cross else owner._far_gap()
        rho = max(float(np.max(gp.rho)), float(np.max(gq.rho)))
        bracket = self.kind != 'T'
        nx = n_xi_for_pairs(owner.k, rho, self.mm, gap, bracket=bracket)
        args = (gp.rho[ip, None], gp.z[ip, None], gp.trho[ip, None], gp.tz[ip, None],
                gq.rho[None, iq], gq.z[None, iq], gq.trho[None, iq], gq.tz[None, iq])
        # Use the exact production angular rule, retaining only this mode.
        if not bracket:
            table = modal_kernels_fft(args[0], args[1], args[4], args[5], owner.k, abs(self.m), n_xi=nx)
            table[near] = 0.
            gs = kernels_for_mode(table, self.m)
            blocks = list(_pair_blocks(self.m, owner.k,
                gp.rho[ip], gp.trho[ip], gp.tz[ip], tp, dp, gp.w[ip],
                gq.rho[iq], gq.trho[iq], gq.tz[iq], tq, dq, gq.w[iq], *gs))
        else:
            fn = mfie_kernels_fft if self.kind == 'K' else ibc_kernels_fft
            inputs = args if self.kind == 'K' else tuple(a.ravel() for a in args)
            tables = fn(*inputs, owner.k, abs(self.m), n_xi=nx)
            wp, wq = gp.w[ip]*gp.rho[ip], gq.w[iq]*gq.rho[iq]
            if self.weight is not None:
                wq = wq*self.weight[iq]
            blocks = []
            for table in tables:
                value = table[..., self.m+abs(self.m)].copy()
                value[near] = 0.
                blocks.append(2*np.pi*(tp*wp) @ value @ (tq*wq).T)
        near_kind = 'efie' if self.kind == 'T' else ('mfie' if self.kind == 'K' else 'ibc')
        rmap, cmap = {int(r): i for i, r in enumerate(rows)}, {int(c): i for i, c in enumerate(cols)}
        if self.cross:
            for e, f in owner.near_pairs:
                if not (e in rmap or e+1 in rmap) or not (f in cmap or f+1 in cmap):
                    continue
                data = owner._near_data(e, f, self.mm)[near_kind][:, self.m+self.mm]
                for a in range(2):
                    for b in range(2):
                        if e+a in rmap and f+b in cmap:
                            for uv in range(4):
                                blocks[uv][rmap[e+a], cmap[f+b]] += data[uv, a, b]
        else:
            data = owner._prepared_near(near_kind, self.mm)
            select = np.isin(data['rows'], rows) & np.isin(data['cols'], cols)
            rr = np.array([rmap[int(r)] for r in data['rows'][select]], dtype=np.intp)
            cc = np.array([cmap[int(c)] for c in data['cols'][select]], dtype=np.intp)
            weight = 1. if self.element_weight is None else self.element_weight[data['source_elems'][select]]
            for uv in range(4):
                np.add.at(blocks[uv], (rr, cc), data['values'][uv, self.m+self.mm, select]*weight)
        if self.kind == 'T':
            blocks = [b*(1j*owner.k*owner.eta*2*np.pi) for b in blocks]
        elif self.kind == 'K':
            mass = mass_expression(owner).get(rows, cols)
            blocks = [0.5*mass-blocks[0], -blocks[1], -blocks[2], 0.5*mass-blocks[3]]
        elif self.kind == 'P':
            blocks = [-blocks[1], blocks[0], -blocks[3], blocks[2]]
        elif self.kind == 'IBC':
            mass = mass_expression(owner, self.weight).get(rows, cols)
            blocks[0] += 0.5*mass
            blocks[3] += 0.5*mass
        self.cache_key, self.cache = key, np.asarray(blocks)
        return self.cache


def primitive(owner, kind, m, m_max, weight=None, element_weight=None):
    query = Primitive(owner, kind, m, m_max, weight, element_weight)
    return TileExpression(query.shape, query.query,
                          np.tile(query.sp.gen.nodes, (2, 1)), np.tile(query.sq.gen.nodes, (2, 1)))
