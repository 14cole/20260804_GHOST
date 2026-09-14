"""Lazy native acceleration for already-qualified kernel interpolation tables."""
import ctypes as ct
from functools import lru_cache
import os
from pathlib import Path
import numpy as np


@lru_cache(None)
def library():
    path = Path(__file__).with_name('ghost_table.dll' if os.name == 'nt' else 'libghost_table.so')
    try:
        dll = ct.CDLL(str(path))
        fun = dll.ghost_table_eval
        fun.argtypes = [ct.c_int64, ct.c_void_p, ct.c_int, ct.c_void_p,
                        ct.c_int, ct.c_void_p, ct.c_int, ct.c_void_p]
        fun.restype = ct.c_int
        return fun
    except (OSError, AttributeError):
        return None


def evaluate(bounds, coefficients, distances, channel=-1):
    fun = library()
    if fun is None:
        return None
    if (not isinstance(bounds,np.ndarray) or bounds.dtype!=np.float64 or bounds.ndim!=1 or
        not bounds.flags.c_contiguous or not 2<=len(bounds)<2**31 or
        not isinstance(coefficients,np.ndarray) or coefficients.dtype!=np.complex128 or
        coefficients.ndim!=3 or not coefficients.flags.c_contiguous or
        coefficients.shape[0]!=len(bounds)-1 or coefficients.shape[2]!=2 or
        not 1<=coefficients.shape[1]<=33 or channel not in (-1,0,1)):
        raise ValueError('Invalid native kernel-table layout.')
    shape=np.shape(distances)
    x = np.ascontiguousarray(distances, dtype=np.float64)
    output = np.empty(shape + ((2,) if channel == -1 else ()), dtype=np.complex128)
    status = fun(x.size, x.ctypes.data, len(bounds)-1, bounds.ctypes.data,
                 coefficients.shape[1]-1, coefficients.ctypes.data, channel, output.ctypes.data)
    if status:
        raise RuntimeError('Native kernel-table evaluation rejected its inputs.')
    return output
