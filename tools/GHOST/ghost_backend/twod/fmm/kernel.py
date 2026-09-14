"""Checked ctypes interface to the pinned FMM2D Fortran kernels.

GHOST uses exp(+i omega t), so G2(k) = -conj(G1(conj(k))). No Python ABI
dependency or import-time compilation is involved. Self point terms are omitted.
"""
import ctypes as ct
from functools import lru_cache
from pathlib import Path
import platform
import threading
import weakref
from ghost_backend.execution.errors import BackendNumericalError
import numpy as np

_LOCK = threading.RLock()


@lru_cache(None)
def library():
    name = ('ghost_fmm.windows-amd64.dll' if platform.system() == 'Windows'
            else 'libghost_fmm.dylib' if platform.system() == 'Darwin' else 'libghost_fmm.so')
    path = Path(__file__).with_name('native') / name
    if not path.is_file():
        raise RuntimeError('FMM native library is unavailable. Build twod/fmm/native/build.py first.')
    dll = ct.CDLL(str(path))
    dll.hfmm2d_.argtypes = [ct.c_void_p]*22
    dll.hfmm2d_.restype = None
    dll.omp_set_num_threads_.argtypes = [ct.POINTER(ct.c_int)]
    if hasattr(dll, 'ghost_fmm_create'):
        for name, count in (('ghost_fmm_create',7), ('ghost_fmm_apply',12), ('ghost_fmm_destroy',1)):
            function = getattr(dll,name)
            function.argtypes = [ct.c_void_p]*count
            function.restype = None
    return dll


def _destroy_plan(dll, handle):
    with _LOCK:
        dll.ghost_fmm_destroy(ct.byref(ct.c_int64(handle)))


class NativePlan:
    """One immutable geometry/wavenumber plan, released with its owning kernel.

    Older installed libraries retain the checked stateless evaluation route.
    No density, polarization, or frequency results are reused implicitly.
    """
    def __init__(self, points, k, eps=1e-10):
        array=np.asarray(points,dtype=float)
        self._points=np.frombuffer(array.tobytes(),dtype=float).reshape(array.shape)
        self._k, self._eps = complex(k), float(eps)
        self.handle = ct.c_int64(0)
        self.bytes = ct.c_int64(0)
        self.builds = 0
        self.peak_bytes = 0
        self._finalizer = None

    @property
    def points(self):return self._points

    @property
    def k(self):return self._k

    @property
    def eps(self):return self._eps

    def close(self):
        if self._finalizer is not None:
            self._finalizer()
        self.handle.value = 0
        self.bytes.value = 0

    def prepare(self, dll, xy, z, tolerance):
        if not self.handle.value:
            ier = ct.c_int(0)
            dll.ghost_fmm_create(ct.byref(ct.c_int(len(self.points))), ct.c_void_p(xy.ctypes.data),
                ct.c_void_p(z.ctypes.data), ct.byref(tolerance), ct.byref(self.handle),
                ct.byref(self.bytes), ct.byref(ier))
            if ier.value or not self.handle.value:
                raise BackendNumericalError('FMM plan creation failed (ier={}).'.format(ier.value))
            self._finalizer = weakref.finalize(self, _destroy_plan, dll, self.handle.value)
            self.builds += 1


def evaluate(points, k, charges=None, dipoles=None, normals=None, gradient=False,
             eps=1e-10, threads=1, plan=None):
    """Return potentials (N,nd) and optional gradients (N,2,nd)."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError('FMM points must be finite (N,2) coordinates.')
    if not np.isfinite(k) or complex(k).real <= 0 or complex(k).imag > 0:
        raise ValueError('FMM requires a positive-real, passive Helmholtz wavenumber.')
    if not 1e-14 <= eps <= 1e-3 or type(threads) is not int or threads < 1:
        raise ValueError('Invalid FMM tolerance or thread count.')
    if plan is not None and (complex(k) != plan.k or eps != plan.eps or
                            points is not plan.points and not np.array_equal(points,plan.points)):
        raise ValueError('FMM plan must match the points, wavenumber and tolerance.')
    data = charges if charges is not None else dipoles
    if data is None:
        raise ValueError('FMM needs charges or dipoles.')
    data = np.asarray(data, complex)
    if data.ndim == 1: data = data[:, None]
    n, nd = data.shape
    if n != len(points) or not n or not nd or n>=2**31 or nd>=2**31:
        raise ValueError('FMM strengths must match the nonempty point set.')
    if nd > 8:
        # Bound persistent native density workspaces during wide-angle residual
        # checks. Output storage remains linear in requested density count.
        cs=None if charges is None else np.asarray(charges,complex).reshape(n,nd)
        ds=None if dipoles is None else np.asarray(dipoles,complex).reshape(n,nd)
        potential=np.empty((n,nd),complex)
        gradients=np.empty((n,2,nd),complex) if gradient else None
        for first in range(0,nd,8):
            last=min(nd,first+8)
            p,g=evaluate(points,k,None if cs is None else cs[:,first:last],
                None if ds is None else ds[:,first:last],normals,gradient,eps,threads,plan)
            potential[:,first:last]=p
            if gradient:gradients[:,:,first:last]=g
        return potential,gradients
    def strengths(value):
        if value is None: return np.zeros((1,1),complex)
        value = np.asarray(value,complex).reshape(n,nd)
        if not np.all(np.isfinite(value)): raise ValueError('Nonfinite FMM strengths.')
        return np.asfortranarray(value.conj().T)
    c, d = strengths(charges), strengths(dipoles)
    if dipoles is not None:
        normals = np.asarray(normals,float)
        if normals.shape != (n,2) or not np.all(np.isfinite(normals)):
            raise ValueError('FMM dipoles need finite (N,2) directions.')
        v = np.asfortranarray(np.broadcast_to(normals.T[None],(nd,2,n)))
    else: v = np.zeros((1,2,1))
    xy = np.asfortranarray(points.T)
    z = np.array(complex(k).conjugate(),dtype=complex)
    p = np.empty((nd,n),complex,order='F')
    g = np.empty((nd,2,n) if gradient else (1,2,1),complex,order='F')
    dummy = np.zeros(3,complex)
    integers = [ct.c_int(i) for i in (nd,n,int(charges is not None),int(dipoles is not None),0,2 if gradient else 1,0,0,0)]
    ndi,ni,ic,idp,iper,pg,nt,pgt,ier = integers
    tolerance = ct.c_double(eps)
    ptr = lambda a: a.ctypes.data_as(ct.c_void_p)
    args = [ct.byref(ndi),ct.byref(tolerance),ptr(z),ct.byref(ni),ptr(xy),ct.byref(ic),ptr(c),
            ct.byref(idp),ptr(d),ptr(v),ct.byref(iper),ct.byref(pg),ptr(p),ptr(g),ptr(dummy),
            ct.byref(nt),ptr(dummy),ct.byref(pgt),ptr(dummy),ptr(dummy),ptr(dummy),ct.byref(ier)]
    dll=library()
    # The static OpenMP runtime is private to this library. Serialize its setting
    # and calls so independent Python solve contexts cannot change it mid-call.
    with _LOCK:
        dll.omp_set_num_threads_(ct.byref(ct.c_int(threads)))
        if plan is not None and hasattr(dll,'ghost_fmm_create'):
            plan.prepare(dll,xy,z,tolerance)
            dll.ghost_fmm_apply(ct.byref(plan.handle),ct.byref(ndi),ct.byref(ic),ptr(c),
                ct.byref(idp),ptr(d),ptr(v),ct.byref(pg),ptr(p),ptr(g),
                ct.byref(plan.bytes),ct.byref(ier))
            plan.peak_bytes=max(plan.peak_bytes,plan.bytes.value)
        else:
            dll.hfmm2d_(*args)
    if ier.value or not np.all(np.isfinite(p)) or gradient and not np.all(np.isfinite(g)):
        raise BackendNumericalError('FMM kernel failed (ier={}).'.format(ier.value))
    return -p.conj().T, -g.conj().transpose(2,1,0) if gradient else None
