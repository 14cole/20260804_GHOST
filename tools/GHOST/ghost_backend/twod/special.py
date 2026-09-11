"""Trusted Bessel/Hankel backends and scalar fallbacks for 2-D kernels."""

import ctypes
import ctypes.util
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from ghost_backend.twod.constants import EULER_GAMMA


try:
    from scipy import special as _SCIPY_SPECIAL
except Exception:
    _SCIPY_SPECIAL = None


try:
    import mpmath as _MPMATH
except Exception:
    _MPMATH = None


class _BesselBackend:
    """
    Real-argument Bessel backend.

    Backend preference:
    1) libc/libm j0/y0/j1/y1
    2) scipy.special j0/y0/j1/y1
    3) local series/asymptotic approximations
    """

    def __init__(self):
        self._lib = None
        self._j0 = None
        self._y0 = None
        self._j1 = None
        self._y1 = None
        self._backend_name = "series-fallback"

        libname = ctypes.util.find_library("m")
        if libname:
            try:
                lib = ctypes.CDLL(libname)
                self._j0 = lib.j0
                self._j0.argtypes = [ctypes.c_double]
                self._j0.restype = ctypes.c_double
                self._y0 = lib.y0
                self._y0.argtypes = [ctypes.c_double]
                self._y0.restype = ctypes.c_double
                self._j1 = lib.j1
                self._j1.argtypes = [ctypes.c_double]
                self._j1.restype = ctypes.c_double
                self._y1 = lib.y1
                self._y1.argtypes = [ctypes.c_double]
                self._y1.restype = ctypes.c_double
                self._lib = lib
                self._backend_name = "libm"
                return
            except Exception:
                self._lib = None
                self._j0 = None
                self._y0 = None
                self._j1 = None
                self._y1 = None

        if _SCIPY_SPECIAL is not None:
            try:

                float(_SCIPY_SPECIAL.j0(0.0))
                float(_SCIPY_SPECIAL.y0(1.0))
                float(_SCIPY_SPECIAL.j1(0.0))
                float(_SCIPY_SPECIAL.y1(1.0))
                self._backend_name = "scipy-special"
            except Exception:
                self._backend_name = "series-fallback"

    @property
    def available(self) -> 'bool':
        return self._backend_name != "series-fallback"

    @property
    def backend_name(self) -> 'str':
        return self._backend_name

    def j0(self, x: 'float') -> 'float':
        if self._j0 is not None:
            return float(self._j0(float(x)))
        if self._backend_name == "scipy-special" and _SCIPY_SPECIAL is not None:
            return float(_SCIPY_SPECIAL.j0(float(x)))
        return _j0_fallback(x)

    def y0(self, x: 'float') -> 'float':
        if self._y0 is not None:
            return float(self._y0(float(x)))
        if self._backend_name == "scipy-special" and _SCIPY_SPECIAL is not None:
            return float(_SCIPY_SPECIAL.y0(float(x)))
        return _y0_fallback(x)

    def j1(self, x: 'float') -> 'float':
        if self._j1 is not None:
            return float(self._j1(float(x)))
        if self._backend_name == "scipy-special" and _SCIPY_SPECIAL is not None:
            return float(_SCIPY_SPECIAL.j1(float(x)))
        return _j1_fallback(x)

    def y1(self, x: 'float') -> 'float':
        if self._y1 is not None:
            return float(self._y1(float(x)))
        if self._backend_name == "scipy-special" and _SCIPY_SPECIAL is not None:
            return float(_SCIPY_SPECIAL.y1(float(x)))
        return _y1_fallback(x)

_BESSEL = _BesselBackend()


def _j0_fallback(x: 'float') -> 'float':
    ax = abs(float(x))
    if ax < 12.0:
        xsq = 0.25 * ax * ax
        term = 1.0
        acc = 1.0
        for m in range(1, 80):
            term *= -xsq / (m * m)
            acc += term
            if abs(term) < 1e-16:
                break
        return acc

    phase = ax - math.pi / 4.0
    amp = math.sqrt(2.0 / (math.pi * ax))
    return amp * math.cos(phase)

def _y0_fallback(x: 'float') -> 'float':
    ax = max(abs(float(x)), 1e-12)
    if ax < 12.0:
        j0 = _j0_fallback(ax)
        xsq = 0.25 * ax * ax
        term = 1.0
        harmonic = 0.0
        acc = 0.0
        for m in range(1, 80):
            harmonic += 1.0 / m
            term *= -xsq / (m * m)
            acc -= harmonic * term
            if abs(term * harmonic) < 1e-16:
                break
        return (2.0 / math.pi) * ((math.log(ax / 2.0) + EULER_GAMMA) * j0 + acc)

    phase = ax - math.pi / 4.0
    amp = math.sqrt(2.0 / (math.pi * ax))
    return amp * math.sin(phase)

def _j1_fallback(x: 'float') -> 'float':
    ax = abs(float(x))
    sign = -1.0 if x < 0.0 else 1.0
    if ax < 12.0:
        xhalf = 0.5 * ax
        term = xhalf
        acc = term
        for m in range(1, 80):
            term *= -(xhalf * xhalf) / (m * (m + 1.0))
            acc += term
            if abs(term) < 1e-16:
                break
        return sign * acc

    phase = ax - 3.0 * math.pi / 4.0
    amp = math.sqrt(2.0 / (math.pi * ax))
    return sign * (amp * math.cos(phase))

def _y1_fallback(x: 'float') -> 'float':
    ax = max(abs(float(x)), 1e-12)
    sign = -1.0 if x < 0.0 else 1.0
    if ax < 12.0:


        j1 = _j1_fallback(ax)
        xhalf = 0.5 * ax
        xhalf2 = xhalf * xhalf
        term = xhalf
        h_k = 0.0
        h_k1 = 1.0
        acc = (h_k + h_k1) * term
        for k in range(1, 80):
            term *= -xhalf2 / (k * (k + 1.0))
            h_k += 1.0 / k
            h_k1 = h_k + 1.0 / (k + 1.0)
            contrib = (h_k + h_k1) * term
            acc += contrib
            if abs(contrib) < 1e-16 * max(1.0, abs(acc)):
                break
        return sign * (
            (2.0 / math.pi) * (math.log(ax / 2.0) + EULER_GAMMA) * j1
            - (2.0 / (math.pi * ax))
            - (1.0 / math.pi) * acc
        )

    phase = ax - 3.0 * math.pi / 4.0
    amp = math.sqrt(2.0 / (math.pi * ax))
    return sign * (amp * math.sin(phase))

def _complex_hankel_backend_name() -> 'str':
    """Report which complex Hankel implementation is active."""

    if _SCIPY_SPECIAL is not None:
        return "scipy-special"
    if _MPMATH is not None:
        return "mpmath"
    return "unavailable"

def _raise_if_untrusted_math_backends() -> 'None':
    """Abort production solves when only approximation fallback math backends are available."""

    if _BESSEL.backend_name == "series-fallback":
        raise RuntimeError(
            "Aborting solve: real-argument Bessel evaluation is using the native series/asymptotic "
            "fallback backend. Install SciPy or provide libm j0/y0/j1/y1 before running production solves."
        )

def _hankel2_0(x: 'Union[complex, float]') -> 'complex':
    """Hankel H_0^(2), with real fast path and no approximation fallback in production."""

    z = complex(x)
    if abs(z.imag) <= 1e-14 and z.real >= 0.0:
        xx = max(float(z.real), 1e-12)
        return complex(_BESSEL.j0(xx), -_BESSEL.y0(xx))
    if _SCIPY_SPECIAL is not None:
        try:
            return complex(_SCIPY_SPECIAL.hankel2(0, z))
        except Exception:
            pass
    if _MPMATH is not None:
        try:
            return complex(_MPMATH.hankel2(0, z))
        except Exception:
            pass
    raise RuntimeError(
        "Aborting solve: complex Hankel H_0^(2) evaluation requires SciPy or mpmath. "
        "Native complex series/asymptotic fallback is disabled for production runs."
    )

def _hankel2_1(x: 'Union[complex, float]') -> 'complex':
    """Hankel H_1^(2), with real fast path and no approximation fallback in production."""

    z = complex(x)
    if abs(z.imag) <= 1e-14 and z.real >= 0.0:
        xx = max(float(z.real), 1e-12)
        return complex(_BESSEL.j1(xx), -_BESSEL.y1(xx))
    if _SCIPY_SPECIAL is not None:
        try:
            return complex(_SCIPY_SPECIAL.hankel2(1, z))
        except Exception:
            pass
    if _MPMATH is not None:
        try:
            return complex(_MPMATH.hankel2(1, z))
        except Exception:
            pass
    raise RuntimeError(
        "Aborting solve: complex Hankel H_1^(2) evaluation requires SciPy or mpmath. "
        "Native complex series/asymptotic fallback is disabled for production runs."
    )
