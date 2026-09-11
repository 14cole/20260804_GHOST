"""Rectangular operator storage with explicit global-to-local node maps."""
import numpy as np


class CompactOperator:
    def __init__(self, node_count, rows, columns, enabled=True):
        self.row_ids = self._ids(rows, node_count)
        self.column_ids = self._ids(columns, node_count)
        self.row_map = np.full(node_count, -1, dtype=np.int64)
        self.column_map = np.full(node_count, -1, dtype=np.int64)
        self.row_map[self.row_ids] = np.arange(len(self.row_ids))
        self.column_map[self.column_ids] = np.arange(len(self.column_ids))
        shape = (len(self.row_ids), len(self.column_ids))
        self.values = (np.zeros(shape, dtype=np.complex128) if enabled else
                       np.broadcast_to(np.zeros((), dtype=np.complex128), shape))

    @staticmethod
    def _ids(ids, count):
        values = np.asarray(ids)
        if values.size == 0:
            return np.empty(0, dtype=np.int64)
        if values.ndim != 1 or values.dtype.kind not in 'iu':
            raise ValueError('Compact operator node IDs must be a one-dimensional integer array.')
        values = values.astype(np.int64, copy=True)
        if np.any(values < 0) or np.any(values >= count) or len(np.unique(values)) != len(values):
            raise ValueError('Compact operator node IDs must be unique and within the mesh.')
        return values

    @property
    def nbytes(self):
        return (self.values.nbytes if any(self.values.strides) else 16) + sum(
            a.nbytes for a in (self.row_ids, self.column_ids, self.row_map, self.column_map))

    @property
    def strides(self):
        return self.values.strides

    @property
    def flags(self):
        return self.values.flags

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, key):
        rows, columns = key
        local_rows, local_columns = self.row_map[rows], self.column_map[columns]
        if np.any(local_rows < 0) or np.any(local_columns < 0):
            raise IndexError('Requested nodes are outside the retained operator block.')
        return self.values[local_rows, local_columns]

    def block(self, rows, columns):
        """A bounded equation block, with zeros for deliberately omitted rows."""
        rr, cc = self.row_map[rows], self.column_map[columns]
        if np.any(cc < 0):
            raise IndexError('Requested source nodes are outside the retained operator block.')
        result = np.zeros((len(rr), len(cc)), dtype=np.complex128)
        keep = rr >= 0
        if np.any(keep):
            result[keep] = self.values[np.ix_(rr[keep], cc)]
        return result


def scatter_operator_add(matrix, rows, columns, values):
    """Accumulate one bounded tile, retaining only requested rows/columns."""
    if hasattr(matrix, 'scatter_add'):
        matrix.scatter_add(rows, columns, values)
    elif isinstance(matrix, CompactOperator):
        rows, columns = np.broadcast_arrays(matrix.row_map[rows], matrix.column_map[columns])
        keep = (rows >= 0) & (columns >= 0)
        np.add.at(matrix.values, (rows[keep], columns[keep]), np.broadcast_to(values, keep.shape)[keep])
    else:
        np.add.at(matrix, (rows, columns), values)
