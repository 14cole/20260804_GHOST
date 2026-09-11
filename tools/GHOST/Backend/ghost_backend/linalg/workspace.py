"""Linear-system checks with bounded temporary arrays (Python 3.6+)."""
import numpy as np


def first_nonfinite(value, block_bytes=1024*1024):
    array = np.asarray(value)
    if array.ndim == 0:
        return None if np.isfinite(array) else ()
    row_entries = int(np.prod(array.shape[1:])) if array.ndim > 1 else 1
    rows = max(1, int(block_bytes) // max(1, row_entries))
    for start in range(0, len(array), rows):
        finite = np.isfinite(array[start:start+rows])
        if not np.all(finite):
            index = np.unravel_index(int(np.argmin(finite)), finite.shape)
            return (start + int(index[0]),) + tuple(int(i) for i in index[1:])
    return None


def matrix_inf_norm(matrix, block_bytes=1024*1024):
    a = np.asarray(matrix)
    if not a.size:
        return 0.
    if a.flags.f_contiguous and not a.flags.c_contiguous:
        sums = np.zeros(a.shape[0])
        width = max(1, int(block_bytes)//max(1, a.shape[0]*8))
        for start in range(0, a.shape[1], width):
            sums += np.sum(np.abs(a[:, start:start+width]), axis=1)
        return float(np.max(sums))
    rows = max(1, int(block_bytes) // max(1, a.shape[1]*8))
    largest = 0.0
    for start in range(0, len(a), rows):
        largest = max(largest, float(np.max(np.sum(np.abs(a[start:start+rows]), axis=1))))
    return largest
