
"""
2D boundary-integral / MoM RCS solver.

High-level workflow:
1) Parse geometry and material definitions into boundary primitives.
2) Build boundary-integral operators (single-layer plus normal-derivative terms).
3) Select the supported Robin, dielectric, multi-region, or sheet
   formulation and solve it with continuous linear Galerkin basis/testing.
4) Post-process the solved boundary unknowns into monostatic far-field RCS.

Notes:
- Uses e^{+j omega t} engineering convention: outgoing Green's function is
  G = (j/4) H_0^(2)(kR), lossy media have eps = eps' - j*eps'' (negative
  imaginary part), and incident plane waves use exp(+j k . r).
- Supports lossy media via complex wavenumber in the coupled formulation.
- Discretization uses continuous two-node linear boundary elements.
"""

import cmath
import csv
import ctypes
import ctypes.util
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from geometry_io import (
    is_legacy_tabulated_row,
    material_filename_from_row,
)
try:
    from scipy import special as _SCIPY_SPECIAL
except Exception:
    _SCIPY_SPECIAL = None
try:
    from scipy import linalg as _SCIPY_LINALG
except Exception:
    _SCIPY_LINALG = None
try:
    from scipy.sparse import linalg as _SCIPY_SPARSE_LINALG
except Exception:
    _SCIPY_SPARSE_LINALG = None
try:
    from scipy.linalg import blas as _SCIPY_BLAS
except Exception:
    _SCIPY_BLAS = None

_GMRES_KWARGS = None


def _gmres_compat(*args, rtol, atol, **kwargs):
    """scipy.sparse.linalg.gmres across scipy versions (HPC clusters ship
    very old builds).  Signature history:

      scipy >= 1.12 : rtol + atol   (tol removed in 1.14)
      1.1 .. 1.11   : tol  + atol
      < 1.1         : tol only      (no atol; legacy stop at ||r||/||b|| < tol)

    The tolerance intent is preserved in every tier: our call sites always
    pass atol == rtol, and the legacy relative-only criterion matches that
    for any nonzero RHS.  The signature is inspected once and cached.
    """

    global _GMRES_KWARGS
    if _GMRES_KWARGS is None:
        import inspect
        try:
            params = inspect.signature(_SCIPY_SPARSE_LINALG.gmres).parameters
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                             for p in params.values())
            _GMRES_KWARGS = ("rtol" if ("rtol" in params or has_kwargs) else "tol",
                             ("atol" in params) or has_kwargs)
        except (TypeError, ValueError):
            _GMRES_KWARGS = ("rtol", True)
    tol_name, has_atol = _GMRES_KWARGS
    kw = dict(kwargs)
    kw[tol_name] = rtol
    if has_atol:
        kw["atol"] = atol
    return _SCIPY_SPARSE_LINALG.gmres(*args, **kw)

try:
    import mpmath as _MPMATH
except Exception:
    _MPMATH = None

C0 = 299_792_458.0
ETA0 = 376.730313668
EPS = 1e-12
MATERIAL_SINGULAR_TOL = EPS
RCS_DB_FLOOR_LINEAR = EPS
VIRTUAL_SHEET_REGION_START = 900_000
EULER_GAMMA = 0.5772156649015329
CFIE_ALPHA_DEFAULT = 0.0
MAX_PANELS_DEFAULT = 20_000
DEFAULT_PANELS_PER_WAVELENGTH = 20
# Explicit positive N remains a user-controlled mesh, but it may not request
# a discretization so coarse that the boundary phase is plainly unresolved.
# This is only a gross safety floor; production accuracy still requires the
# separate base/fine complex-field mesh-convergence certification.
MIN_EXPLICIT_PANELS_PER_WAVELENGTH = 4
GMRES_NODE_THRESHOLD = 3000
GMRES_RESTART = 50
GMRES_MAXITER = 200
GMRES_TOL = 1e-8
# Monostatic 2D RCS normalization controls.
#
# For the physical asymptotic convention,
#   G(r) = (j/4) H_0^(2)(k r),
# and for a far-field amplitude A defined such that
#   u_s(r,phi) ~ sqrt(1 / (8*pi*k*r)) * exp(-j(kr-pi/4)) * A(phi),
# the 2D scattering width per unit length is
#   sigma_2d(phi) = |A(phi)|^2 / (4 k).
#
# Historical solver/projector outputs store the bare layer-potential integral
# B, for which A = +j*B.  That global unit-modulus factor leaves scattering
# width unchanged, but it matters to consumers of complex phase.  Keep B for
# feature-delta compatibility and label the convention explicitly everywhere.
#
# Use physical 2D scattering-width normalization by default.
#
RCS_NORM_NUMERATOR = 0.25
RCS_NORM_MODE_DEFAULT = "physical"
RCS_NORM_MODE_PHYSICAL = "physical"
RCS_AMPLITUDE_CONVENTION = "A_physical_asymptotic = +j * B_stored"

@dataclass
class Panel:
    """Single discretized boundary element used by the solver mesh builder."""

    name: 'str'
    seg_type: 'int'
    ibc_flag: 'int'
    pos_mat: 'int'
    neg_mat: 'int'
    p0: 'np.ndarray'
    p1: 'np.ndarray'
    center: 'np.ndarray'
    tangent: 'np.ndarray'
    normal: 'np.ndarray'
    length: 'float'
    # Arc-length position of the panel center within its parent segment,
    # normalized so that 0.0 = segment start (as drawn) and 1.0 = segment end.
    # Used to sample spatially tapered surface impedance.  Default 0.5 makes
    # existing constant-impedance behaviour unchanged for any non-tapered flag.
    arc_s_center: 'float' = 0.5

@dataclass
class LinearNode:
    """Unique mesh node for a continuous piecewise-linear boundary discretization."""

    xy: 'np.ndarray'
    key: 'Tuple[int, int]'

@dataclass
class LinearElement:
    """Two-node straight boundary element used by the Galerkin discretization."""

    name: 'str'
    seg_type: 'int'
    ibc_flag: 'int'
    pos_mat: 'int'
    neg_mat: 'int'
    node_ids: 'Tuple[int, int]'
    p0: 'np.ndarray'
    p1: 'np.ndarray'
    center: 'np.ndarray'
    tangent: 'np.ndarray'
    normal: 'np.ndarray'
    length: 'float'
    panel_index: 'int'
    arc_s_center: 'float' = 0.5

@dataclass
class LinearMesh:
    """Continuous linear boundary mesh assembled from boundary primitives."""

    nodes: 'List[LinearNode]'
    elements: 'List[LinearElement]'

@dataclass
class PanelCoupledInfo:
    """
    Per-element material and interface bookkeeping for the coupled formulation.

    The unknown vector is [u_trace, q_minus]. This record maps each element's
    plus-side and minus-side constitutive data into the assembled system.
    """

    seg_type: 'int'
    plus_region: 'int'
    minus_region: 'int'
    plus_has_incident: 'bool'
    minus_has_incident: 'bool'
    eps_plus: 'complex'
    mu_plus: 'complex'
    eps_minus: 'complex'
    mu_minus: 'complex'
    k_plus: 'complex'
    k_minus: 'complex'
    q_plus_beta: 'complex'
    q_plus_gamma: 'complex'
    bc_kind: 'str'
    # Surface impedance Z_s for Leontovich IBC, stored exactly as entered
    # (no sign conversion happens anywhere: `MaterialLibrary.from_entries`
    # and `_load_impedance_table` pass values through verbatim).  Z_s uses
    # the standard physics convention E_t = Z_s * (n_out x H) with n_out
    # pointing away from the conductor; `_surface_robin_alpha` converts it
    # to the Robin coefficient under the solver's stored-normal convention
    # (validated against the impedance-cylinder Mie series in both
    # polarizations).  Re(Z_s) >= 0 for passive lossy sheets.
    robin_impedance: 'complex'

@dataclass
class ComplexTable:
    """Frequency-dependent complex scalar table with linear interpolation."""

    freqs_ghz: 'np.ndarray'
    values: 'np.ndarray'

    def sample(self, freq_ghz: 'float') -> 'complex':
        freq_ghz = float(freq_ghz)
        if not math.isfinite(freq_ghz):
            raise ValueError("Material-table sample frequency must be finite.")
        fmin = float(self.freqs_ghz[0])
        fmax = float(self.freqs_ghz[-1])
        if freq_ghz < fmin or freq_ghz > fmax:
            raise ValueError(
                f"Material-table sample frequency {freq_ghz:g} GHz is outside "
                f"the characterized range [{fmin:g}, {fmax:g}] GHz."
            )
        if len(self.freqs_ghz) == 1:
            return complex(self.values[0])
        real = np.interp(freq_ghz, self.freqs_ghz, self.values.real)
        imag = np.interp(freq_ghz, self.freqs_ghz, self.values.imag)
        return complex(real, imag)

@dataclass
class ImpedanceTaper:
    """Spatially tapered surface impedance along a segment.

    The segment is parametrized by arc_s in [0, 1], running from the segment's
    start endpoint (as drawn by the user) to the end endpoint.  At arc_s = 0
    the impedance equals z_start, at arc_s = 1 it equals z_end, with the
    interpolation weighting determined by ``kind``:

    - ``"linear"``   : w = s                               (straight ramp)
    - ``"cosine"``   : w = 0.5 * (1 - cos(pi * s))          (Hann/raised-cosine:
                                                             C^1 at both ends,
                                                             good for edge taper)
    - ``"exp"``      : log-space interpolation              (octave-per-length
                                                             ramp between two
                                                             nonzero impedances)

    For ``"linear"`` and ``"cosine"`` the endpoints may be zero (PEC) or
    arbitrary complex.  For ``"exp"`` both endpoints must be nonzero; zero
    endpoints are coerced to a tiny nonzero floor.

    This model is independent of frequency.  Combining a taper with a
    frequency-dependent material table would require a 2-D (s, f) model and is
    out of scope for the initial implementation.
    """

    kind: 'str'
    z_start: 'complex'
    z_end: 'complex'

    _ALLOWED_KINDS = ("constant", "linear", "cosine", "exp")

    def __post_init__(self) -> 'None':
        if self.kind not in self._ALLOWED_KINDS:
            raise ValueError(
                f"Unknown impedance taper kind '{self.kind}'. "
                f"Expected one of {self._ALLOWED_KINDS}."
            )
        self.z_start = _validate_passive_surface_impedance(
            self.z_start, "taper z_start"
        )
        if self.kind == "constant":
            # End values are placeholders for constant; force end = start so
            # any downstream consumer that samples z_end gets the right value.
            self.z_end = self.z_start
        else:
            self.z_end = _validate_passive_surface_impedance(
                self.z_end, "taper z_end"
            )

    def evaluate(self, arc_s: 'float') -> 'complex':
        s = float(max(0.0, min(1.0, arc_s)))
        if self.kind == "constant":
            return self.z_start
        if self.kind == "linear":
            w = s
            return (1.0 - w) * self.z_start + w * self.z_end
        if self.kind == "cosine":
            w = 0.5 * (1.0 - math.cos(math.pi * s))
            return (1.0 - w) * self.z_start + w * self.z_end
        # "exp": log-space interpolation.  Floor zero endpoints so log is defined.
        z1 = self.z_start if abs(self.z_start) > EPS else complex(EPS, 0.0)
        z2 = self.z_end if abs(self.z_end) > EPS else complex(EPS, 0.0)
        return cmath.exp((1.0 - s) * cmath.log(z1) + s * cmath.log(z2))

@dataclass
class MediumTable:
    """Frequency-dependent (eps, mu) table with linear interpolation."""

    freqs_ghz: 'np.ndarray'
    eps_values: 'np.ndarray'
    mu_values: 'np.ndarray'

    def sample(self, freq_ghz: 'float') -> 'Tuple[complex, complex]':
        freq_ghz = float(freq_ghz)
        if not math.isfinite(freq_ghz):
            raise ValueError("Material-table sample frequency must be finite.")
        fmin = float(self.freqs_ghz[0])
        fmax = float(self.freqs_ghz[-1])
        if freq_ghz < fmin or freq_ghz > fmax:
            raise ValueError(
                f"Material-table sample frequency {freq_ghz:g} GHz is outside "
                f"the characterized range [{fmin:g}, {fmax:g}] GHz."
            )
        if len(self.freqs_ghz) == 1:
            return complex(self.eps_values[0]), complex(self.mu_values[0])
        eps_r = np.interp(freq_ghz, self.freqs_ghz, self.eps_values.real)
        eps_i = np.interp(freq_ghz, self.freqs_ghz, self.eps_values.imag)
        mu_r = np.interp(freq_ghz, self.freqs_ghz, self.mu_values.real)
        mu_i = np.interp(freq_ghz, self.freqs_ghz, self.mu_values.imag)
        return complex(eps_r, eps_i), complex(mu_r, mu_i)

class MaterialLibrary:
    """Material lookup facade for inline values and frequency tables."""

    def __init__(
        self,
        impedance_models: 'Dict[int, Union[complex, ComplexTable]]',
        dielectric_models: 'Dict[int, Union[Tuple[complex, complex], MediumTable]]',
    ):
        self.impedance_models = impedance_models
        self.dielectric_models = dielectric_models
        self.warnings: 'List[str]' = []
        self._warning_seen: 'Set[str]' = set()

    @classmethod
    def from_entries(
        cls,
        ibcs_entries: 'List[List[str]]',
        dielectric_entries: 'List[List[str]]',
        base_dir: 'str',
    ) -> "MaterialLibrary":
        impedance_models: 'Dict[int, Union[complex, ComplexTable]]' = {}
        dielectric_models: 'Dict[int, Union[Tuple[complex, complex], MediumTable]]' = {}
        seen_impedance_flags: 'Set[int]' = set()
        seen_dielectric_flags: 'Set[int]' = set()

        for row in ibcs_entries:
            if not row:
                continue
            flag = _parse_material_definition_flag(
                row[0], "IBC material definition flag"
            )
            if flag in seen_impedance_flags:
                raise ValueError(f"Duplicate IBC material flag {flag}.")
            seen_impedance_flags.add(flag)
            if len(row) == 2:
                filename = material_filename_from_row(row)
                if filename is None:
                    raise ValueError(
                        f"IBC flag {flag}: file-backed material definitions "
                        "must use 'flag filename.csv'."
                    )
                path = _resolve_material_file(base_dir, filename)
                impedance_models[flag] = _load_impedance_csv(path)
                continue
            if is_legacy_tabulated_row(row):
                path = _resolve_mat_file(base_dir, flag)
                impedance_models[flag] = _load_impedance_table(path)
                continue

            # Inline format (6 tokens including flag):
            #   flag  <kind>  R_start  X_start  R_end  X_end
            # where kind is one of constant/linear/cosine/exp. For "constant",
            # only R_start/X_start matter; R_end/X_end are ignored.
            tokens = [str(t).strip() for t in row[1:] if str(t).strip() != ""]
            if len(tokens) != 5:
                raise ValueError(
                    f"IBC flag {flag}: inline impedance requires "
                    "'<kind> R_start X_start R_end X_end' "
                    f"(got {len(tokens)} data tokens after the flag)."
                )
            kind = tokens[0].strip().lower()
            r_start = _parse_material_float(tokens[1], f"IBC flag {flag} R_start")
            x_start = _parse_material_float(tokens[2], f"IBC flag {flag} X_start")
            r_end = _parse_material_float(tokens[3], f"IBC flag {flag} R_end")
            x_end = _parse_material_float(tokens[4], f"IBC flag {flag} X_end")
            impedance_models[flag] = ImpedanceTaper(
                kind=kind,
                z_start=complex(r_start, x_start),
                z_end=complex(r_end, x_end),
            )

        for row in dielectric_entries:
            if not row:
                continue
            flag = _parse_material_definition_flag(
                row[0], "Dielectric material definition flag"
            )
            if flag in seen_dielectric_flags:
                raise ValueError(f"Duplicate dielectric material flag {flag}.")
            seen_dielectric_flags.add(flag)
            if len(row) == 2:
                filename = material_filename_from_row(row)
                if filename is None:
                    raise ValueError(
                        f"Dielectric flag {flag}: inline material requires "
                        "exactly 'flag eps_real eps_imag mu_real mu_imag'; "
                        "file-backed definitions require "
                        "'flag filename.csv'."
                    )
                path = _resolve_material_file(base_dir, filename)
                dielectric_models[flag] = _load_dielectric_csv(path)
                continue
            if is_legacy_tabulated_row(row):
                path = _resolve_mat_file(base_dir, flag)
                dielectric_models[flag] = _load_dielectric_table(path)
                continue
            if len(row) != 5 or any(
                    str(token).strip() == "" for token in row):
                raise ValueError(
                    f"Dielectric flag {flag}: inline material requires exactly "
                    "'flag eps_real eps_imag mu_real mu_imag' with no blank "
                    f"fields (got {len(row)} fields).")
            eps_real = _parse_material_float(
                row[1], f"Dielectric flag {flag} epsilon real part")
            eps_imag = _parse_material_float(
                row[2], f"Dielectric flag {flag} epsilon imaginary part")
            mu_real = _parse_material_float(
                row[3], f"Dielectric flag {flag} mu real part")
            mu_imag = _parse_material_float(
                row[4], f"Dielectric flag {flag} mu imaginary part")
            # Imaginary parts are stored exactly as entered (exp(+j*omega*t)
            # convention): lossy media use NEGATIVE eps''/mu'' in the input.
            eps_raw = _ensure_finite_complex(
                complex(eps_real, eps_imag),
                f"Dielectric flag {flag} epsilon",
            )
            mu_raw = _ensure_finite_complex(
                complex(mu_real, mu_imag),
                f"Dielectric flag {flag} mu",
            )
            eps, mu = _validate_passive_medium(
                eps_raw, mu_raw, f"Dielectric flag {flag}"
            )
            dielectric_models[flag] = (eps, mu)

        return cls(impedance_models=impedance_models, dielectric_models=dielectric_models)

    def get_impedance(self, flag: 'int', freq_ghz: 'float', arc_s: 'Optional[float]' = None) -> 'complex':
        if flag <= 0:
            return 0.0 + 0.0j
        model = self.impedance_models.get(flag)
        if model is None:
            raise ValueError(f"Undefined IBC flag {flag}.")
        if isinstance(model, ComplexTable):
            return _validate_passive_surface_impedance(
                model.sample(freq_ghz),
                f"IBC flag {flag} impedance sampled at {freq_ghz:g} GHz",
            )
        if isinstance(model, ImpedanceTaper):
            s = 0.5 if arc_s is None else float(arc_s)
            return _validate_passive_surface_impedance(
                model.evaluate(s),
                f"IBC flag {flag} tapered impedance at s={s:g}",
            )
        return _validate_passive_surface_impedance(
            model, f"IBC flag {flag} impedance"
        )

    def is_tapered_impedance(self, flag: 'int') -> 'bool':
        """True if the IBC flag is spatially tapered along the segment."""
        model = self.impedance_models.get(flag)
        return (
            isinstance(model, ImpedanceTaper)
            and model.kind != "constant"
        )

    def get_medium(self, flag: 'int', freq_ghz: 'float') -> 'Tuple[complex, complex]':
        if flag <= 0:
            return 1.0 + 0.0j, 1.0 + 0.0j
        model = self.dielectric_models.get(flag)
        if model is None:
            raise ValueError(f"Undefined dielectric flag {flag}.")
        if isinstance(model, MediumTable):
            eps, mu = model.sample(freq_ghz)
            return _validate_passive_medium(
                eps,
                mu,
                f"Dielectric flag {flag} sampled at {freq_ghz:g} GHz",
            )
        eps, mu = model
        return _validate_passive_medium(eps, mu, f"Dielectric flag {flag}")

    def _warn_once(self, message: 'str') -> 'None':
        if message in self._warning_seen:
            return
        self._warning_seen.add(message)
        self.warnings.append(message)

    def warn_once(self, message: 'str') -> 'None':
        self._warn_once(message)

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
                # Ensure required real-order functions are present/callable.
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

# --- Special-function helpers -------------------------------------------------
# Real-argument helpers are used heavily for lossless/real-k paths.
# Complex-argument Hankel is needed for lossy media (complex-k kernels).
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
        # Full series using harmonic numbers (Abramowitz & Stegun 9.1.56):
        # Y1(x) = (2/pi)[J1(x)(ln(x/2)+gamma) - 1/x]
        #        - (1/pi) Sum_{k=0}^inf (-1)^k (H_k+H_{k+1}) (x/2)^{2k+1} / (k!(k+1)!)
        # where H_0=0, H_k = 1 + 1/2 + ... + 1/k.
        j1 = _j1_fallback(ax)
        xhalf = 0.5 * ax
        xhalf2 = xhalf * xhalf
        term = xhalf  # k=0: (x/2)^1 / (0! * 1!)
        h_k = 0.0     # H_0 = 0
        h_k1 = 1.0    # H_1 = 1
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

def _parse_flag(token: 'Any') -> 'int':
    text = str(token).strip().lower()
    if not text:
        return 0
    if text.startswith("mat."):
        text = text.split("mat.", 1)[1]
    try:
        return int(float(text))
    except ValueError:
        return 0

def _parse_float(token: 'Any', default: 'float' = 0.0) -> 'float':
    try:
        return float(token)
    except (TypeError, ValueError):
        return default

def _parse_int(token: 'Any', default: 'int' = 0) -> 'int':
    try:
        return int(round(float(token)))
    except (TypeError, ValueError):
        return default

def _parse_geometry_float(token: 'Any', context: 'str') -> 'float':
    """Strict numeric parser for solver-facing geometry snapshots."""

    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} must be a finite numeric value; got {token!r}."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite; got {token!r}.")
    return value

def _parse_geometry_integer(
    token: 'Any',
    context: 'str',
    *,
    allow_mat_prefix: 'bool' = False,
) -> 'int':
    """Strict integral parser for TYPE/N/material flag fields."""

    text = str(token).strip()
    if allow_mat_prefix and text.lower().startswith("mat."):
        text = text[4:]
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer; got {token!r}.") from exc
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"{context} must be an integer; got {token!r}.")
    return int(value)

def _parse_material_definition_flag(token: 'Any', context: 'str') -> 'int':
    """Strict positive ID parser for user-supplied material-library rows."""

    value = _parse_geometry_integer(
        token, context, allow_mat_prefix=True
    )
    if value <= 0:
        raise ValueError(
            f"{context} must be a positive integer; got {token!r}."
        )
    return value

def _ensure_finite_complex(value: 'complex', context: 'str') -> 'complex':
    z = complex(value)
    if not np.isfinite(z.real) or not np.isfinite(z.imag):
        raise ValueError(f"{context} contains non-finite value {z!r}.")
    return z

def _parse_material_float(token: 'Any', context: 'str') -> 'float':
    """Parse one explicitly supplied material field without silent defaults."""

    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a finite numeric value; got {token!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite; got {token!r}.")
    return value

def _passivity_tolerance(value: 'complex') -> 'float':
    return 64.0 * np.finfo(float).eps * max(1.0, abs(complex(value)))

def _validate_passive_surface_impedance(value: 'complex', context: 'str') -> 'complex':
    """Validate a passive Leontovich impedance under the solver convention."""

    z = _ensure_finite_complex(value, context)
    if z.real < -_passivity_tolerance(z):
        raise ValueError(
            f"{context} has negative resistance Re(Zs)={z.real:g} ohm. "
            "Active/gain surface impedances are not supported."
        )
    return z

def _validate_passive_medium(
    eps: 'complex',
    mu: 'complex',
    context: 'str',
) -> 'Tuple[complex, complex]':
    """
    Validate constitutive values supported by the passive 2D formulations.

    The solver uses e^(+j*omega*t), so passive loss has Im(eps), Im(mu) <= 0.
    Exactly/near-singular ENZ or MNZ media require a dedicated limiting
    formulation; rejecting them is safer than the historical substitution
    of 1+0j (free space).
    """

    eps_eval = _ensure_finite_complex(eps, f"{context} epsilon")
    mu_eval = _ensure_finite_complex(mu, f"{context} mu")
    if abs(eps_eval) <= MATERIAL_SINGULAR_TOL:
        raise ValueError(
            f"{context} has unsupported singular/near-ENZ epsilon {eps_eval!r} "
            f"(|epsilon| <= {MATERIAL_SINGULAR_TOL:g}); it will not be replaced with free space."
        )
    if abs(mu_eval) <= MATERIAL_SINGULAR_TOL:
        raise ValueError(
            f"{context} has unsupported singular/near-MNZ mu {mu_eval!r} "
            f"(|mu| <= {MATERIAL_SINGULAR_TOL:g}); it will not be replaced with free space."
        )
    if eps_eval.imag > _passivity_tolerance(eps_eval):
        raise ValueError(
            f"{context} epsilon has gain-sign Im(epsilon)={eps_eval.imag:g}. "
            "For e^(+j*omega*t), passive media require Im(epsilon) <= 0; "
            "active/gain media are not supported."
        )
    if mu_eval.imag > _passivity_tolerance(mu_eval):
        raise ValueError(
            f"{context} mu has gain-sign Im(mu)={mu_eval.imag:g}. "
            "For e^(+j*omega*t), passive media require Im(mu) <= 0; "
            "active/gain media are not supported."
        )
    return eps_eval, mu_eval


def _resolve_mat_file(base_dir: 'str', flag: 'int') -> 'str':
    """Resolve a legacy mat.<flag> relative to the geometry directory."""

    name = f"mat.{flag}"
    return _resolve_material_file(base_dir, name)


def _resolve_material_file(base_dir: 'str', filename: 'str') -> 'str':
    """Resolve a validated material sidecar in the geometry directory only."""

    name = str(filename)
    folder = os.path.abspath(str(base_dir))
    path = os.path.join(folder, name)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(
        f"Could not locate material file {name} in declared material directory "
        f"{folder}. Material tables are never searched in the process working "
        "directory.")


def _material_base_dir_for_snapshot(
    geometry_snapshot: 'Dict[str, Any]',
    material_base_dir: 'Optional[str]',
) -> 'str':
    """Return the single declared directory used for material sidecars.

    An explicit directory has priority.  Otherwise a file-backed snapshot
    inherits the directory containing ``source_path``.  Only a genuinely
    pathless, programmatic snapshot uses the process working directory as its
    documented default.  Thus changing cwd cannot change a loaded geometry's
    material model.
    """

    if material_base_dir is not None and str(material_base_dir).strip():
        return os.path.abspath(os.path.expanduser(str(material_base_dir)))
    source_path = str(geometry_snapshot.get("source_path", "") or "").strip()
    if source_path:
        return os.path.dirname(
            os.path.abspath(os.path.expanduser(source_path))
        )
    return os.path.abspath(os.getcwd())


def _read_numeric_rows(path: 'str', expected_columns: 'int') -> 'List[List[float]]':
    """Read a strict material table and return rows sorted by frequency.

    Every non-comment row must have exactly ``expected_columns`` finite numeric
    fields.  Frequencies must be positive and unique.  A malformed row must
    never disappear silently: doing so can change an intended dispersion model
    into a different, apparently valid one.
    """

    rows: 'List[List[float]]' = []
    with open(path, "r") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) != expected_columns:
                raise ValueError(
                    f"Material file '{path}' line {lineno} must contain exactly "
                    f"{expected_columns} numeric columns; found {len(tokens)}."
                )
            try:
                parsed = [float(token) for token in tokens]
            except ValueError as exc:
                raise ValueError(
                    f"Material file '{path}' line {lineno} contains a "
                    f"non-numeric value."
                ) from exc
            if not all(math.isfinite(v) for v in parsed):
                raise ValueError(
                    f"Material file '{path}' line {lineno} contains non-finite "
                    f"numeric value(s): {tokens}."
                )
            if parsed[0] <= 0.0:
                raise ValueError(
                    f"Material file '{path}' line {lineno} has non-positive "
                    f"frequency {parsed[0]:g} GHz."
                )
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No numeric material rows found in {path}")
    rows.sort(key=lambda row: row[0])
    for previous, current in zip(rows, rows[1:]):
        if current[0] == previous[0]:
            raise ValueError(
                f"Material file '{path}' contains duplicate frequency "
                f"{current[0]:g} GHz."
            )
    return rows

def _load_impedance_table(path: 'str') -> 'ComplexTable':
    """Load frequency -> complex impedance table: f(GHz) z_real z_imag."""

    rows = _read_numeric_rows(path, 3)
    freqs = np.asarray([r[0] for r in rows], dtype=float)
    vals = np.asarray([complex(r[1], r[2]) for r in rows], dtype=np.complex128)
    for row_index, value in enumerate(vals, start=1):
        _validate_passive_surface_impedance(
            value, f"Impedance table '{path}' data row {row_index}"
        )
    return ComplexTable(freqs_ghz=freqs, values=vals)

def _load_dielectric_table(path: 'str') -> 'MediumTable':
    """Load frequency -> (eps, mu) table: f eps_r eps_i mu_r mu_i.

    Imaginary parts are used as entered (exp(+j*omega*t) convention):
    lossy media use NEGATIVE eps''/mu'' columns in the mat.<N> file.
    """

    rows = _read_numeric_rows(path, 5)
    freqs = np.asarray([r[0] for r in rows], dtype=float)
    eps_vals = np.asarray([complex(r[1], r[2]) for r in rows], dtype=np.complex128)
    mu_vals = np.asarray([complex(r[3], r[4]) for r in rows], dtype=np.complex128)
    for row_index, (eps, mu) in enumerate(zip(eps_vals, mu_vals), start=1):
        _validate_passive_medium(
            eps, mu, f"Dielectric table '{path}' data row {row_index}"
        )
    return MediumTable(freqs_ghz=freqs, eps_values=eps_vals, mu_values=mu_vals)


def _read_csv_numeric_rows(
    path: 'str',
    expected_header: 'List[str]',
) -> 'List[List[float]]':
    """Read a strict CSV material table whose first column is frequency in Hz."""

    rows: 'List[List[float]]' = []
    with open(path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        try:
            raw_header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV material file '{path}' is empty.")
        header = [str(value).strip().lower() for value in raw_header]
        if header != expected_header:
            raise ValueError(
                f"CSV material file '{path}' must have header "
                f"{','.join(expected_header)}; found {','.join(header)}."
            )
        for raw_row in reader:
            lineno = reader.line_num
            if not raw_row or all(not str(value).strip() for value in raw_row):
                continue
            if len(raw_row) != len(expected_header):
                raise ValueError(
                    f"CSV material file '{path}' line {lineno} must contain "
                    f"exactly {len(expected_header)} columns; found "
                    f"{len(raw_row)}."
                )
            try:
                parsed = [float(str(value).strip()) for value in raw_row]
            except ValueError as exc:
                raise ValueError(
                    f"CSV material file '{path}' line {lineno} contains a "
                    "non-numeric value."
                ) from exc
            if not all(math.isfinite(value) for value in parsed):
                raise ValueError(
                    f"CSV material file '{path}' line {lineno} contains "
                    "non-finite numeric value(s)."
                )
            if parsed[0] <= 0.0:
                raise ValueError(
                    f"CSV material file '{path}' line {lineno} has "
                    f"non-positive frequency {parsed[0]:g} Hz."
                )
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No numeric material rows found in {path}")
    rows.sort(key=lambda row: row[0])
    for previous, current in zip(rows, rows[1:]):
        if current[0] == previous[0]:
            raise ValueError(
                f"CSV material file '{path}' contains duplicate frequency "
                f"{current[0]:g} Hz."
            )
    return rows


def _load_impedance_csv(path: 'str') -> 'ComplexTable':
    """Load frequency_hz,resistance_ohm,reactance_ohm."""

    rows = _read_csv_numeric_rows(
        path,
        ["frequency_hz", "resistance_ohm", "reactance_ohm"],
    )
    freqs = np.asarray([row[0] * 1.0e-9 for row in rows], dtype=float)
    values = np.asarray(
        [complex(row[1], row[2]) for row in rows],
        dtype=np.complex128,
    )
    for row_index, value in enumerate(values, start=1):
        _validate_passive_surface_impedance(
            value, f"Impedance CSV '{path}' data row {row_index}"
        )
    return ComplexTable(freqs_ghz=freqs, values=values)


def _load_dielectric_csv(path: 'str') -> 'MediumTable':
    """Load frequency_hz,eps_real,eps_imag,mu_real,mu_imag."""

    rows = _read_csv_numeric_rows(
        path,
        ["frequency_hz", "eps_real", "eps_imag", "mu_real", "mu_imag"],
    )
    freqs = np.asarray([row[0] * 1.0e-9 for row in rows], dtype=float)
    eps_values = np.asarray(
        [complex(row[1], row[2]) for row in rows],
        dtype=np.complex128,
    )
    mu_values = np.asarray(
        [complex(row[3], row[4]) for row in rows],
        dtype=np.complex128,
    )
    for row_index, (eps, mu) in enumerate(
        zip(eps_values, mu_values), start=1
    ):
        _validate_passive_medium(
            eps, mu, f"Dielectric CSV '{path}' data row {row_index}"
        )
    return MediumTable(
        freqs_ghz=freqs,
        eps_values=eps_values,
        mu_values=mu_values,
    )

def _canonical_user_polarization_label(label: 'Optional[str]') -> 'str':
    text = str(label or '').strip().upper()
    if text in {'TM', 'HH', 'H', 'HORIZONTAL'}:
        return 'TM'
    if text in {'TE', 'VV', 'V', 'VERTICAL'}:
        return 'TE'
    raise ValueError(f"Unsupported polarization '{label}'. Use TM/TE or VV/HH.")

def _primary_alias_for_user_polarization(label: 'str') -> 'str':
    # Elevation-cut convention: z is horizontal, so E_z (TM) == HH.
    return 'HH' if _canonical_user_polarization_label(label) == 'TM' else 'VV'

def _normalize_polarization(polarization: 'str') -> 'str':
    """
    Normalize user-facing polarization labels without swapping TM and TE.

    Radar-alias convention in this project (2D geometries are elevation cuts,
    out-of-plane z axis is HORIZONTAL):
    - TM, HH, H, HORIZONTAL -> TM  (E along z = horizontal = HH)
    - TE, VV, V, VERTICAL   -> TE  (H along z; E in-plane has vertical component = VV)
    """

    pol = (polarization or "").strip().upper()
    if pol in {"TM", "HH", "H", "HORIZONTAL"}:
        return "TM"
    if pol in {"TE", "VV", "V", "VERTICAL"}:
        return "TE"
    raise ValueError(f"Unsupported polarization '{polarization}'. Use TM/TE or VV/HH.")

def _unit_scale_to_meters(units: 'str') -> 'float':
    value = (units or "").strip().lower()
    if value in {"inch", "inches", "in"}:
        return 0.0254
    if value in {"meter", "meters", "m"}:
        return 1.0
    raise ValueError(f"Unsupported geometry units '{units}'. Use inches or meters.")

def _discretize_primitive(p0: 'np.ndarray', p1: 'np.ndarray', count: 'int') -> 'List[np.ndarray]':
    """Generate panel endpoints for a straight-line primitive."""

    count = max(1, int(count))
    return [p0 + (p1 - p0) * (i / count) for i in range(count + 1)]

def _primitive_length(p0: 'np.ndarray', p1: 'np.ndarray') -> 'float':
    return float(np.linalg.norm(p1 - p0))

def _panel_count_from_n(n_prop: 'int', primitive_len: 'float', min_wavelength: 'float') -> 'int':
    """
    Convert geometry n property to panel count.

    n > 0: explicit panel count.
    n < 0: panels-per-wavelength style control.
    """

    if primitive_len <= EPS:
        return 1
    if n_prop > 0:
        return max(1, n_prop)
    if n_prop < 0:
        n_wave = max(1, abs(n_prop))
        target = max(min_wavelength / n_wave, primitive_len / 2000.0)
        return max(1, int(math.ceil(primitive_len / target)))
    # n_prop == 0: apply default panels-per-wavelength density.
    if min_wavelength > EPS:
        target = min_wavelength / float(DEFAULT_PANELS_PER_WAVELENGTH)
        return max(1, int(math.ceil(primitive_len / target)))
    return max(1, int(math.ceil(primitive_len / (primitive_len / 10.0 + EPS))))


def _reverse_point_pairs(point_pairs: 'List[Dict[str, Any]]') -> 'List[Dict[str, Any]]':
    """Reverse a primitive chain: last primitive first, endpoints swapped."""

    reversed_pairs: 'List[Dict[str, Any]]' = []
    for pair in reversed(point_pairs):
        reversed_pairs.append({
            'x1': pair.get('x2', 0.0),
            'y1': pair.get('y2', 0.0),
            'x2': pair.get('x1', 0.0),
            'y2': pair.get('y1', 0.0),
        })
    return reversed_pairs


def _check_segment_orientation_or_raise(segments: 'List[Dict[str, Any]]') -> 'None':
    """
    Run the shared winding / air-side consistency checks (geometry_io) and
    raise on any ERROR finding.

    The TM formulations are winding-insensitive, but the TE MFIE/Robin rows
    carry a +-1/2 mass jump tied to the normal direction, so a wrong winding
    or inconsistent air side silently corrupts TE results (residuals stay
    tiny).  The solver deliberately refuses to run rather than silently
    reorienting the user's geometry.
    """

    from geometry_io import chains_from_snapshot_segments, check_orientation_consistency

    findings = check_orientation_consistency(chains_from_snapshot_segments(segments))
    errors = [msg for severity, _idx, msg in findings if severity == "ERROR"]
    if errors:
        raise ValueError(
            "Geometry orientation check failed:\n  - " + "\n  - ".join(errors)
        )


def _normalize_segment_orientation(
    seg_type: 'int',
    point_pairs: 'List[Dict[str, Any]]',
    meters_scale: 'float',
) -> 'List[Dict[str, Any]]':
    """
    Pass-through: the user's endpoint order is the source of truth.

    This routine does not reorient contours.  The user is responsible for
    drawing each segment so that the normal (computed from endpoint order)
    points in the physically intended direction.

    The per-panel-type convention mapping from user-facing geometry to
    solver-internal plus/minus assignments is handled separately in
    `_apply_user_convention_flip` (called from `_build_panels`).
    """

    return point_pairs


def _apply_user_convention_flip(
    seg_type: 'int',
    point_pairs: 'List[Dict[str, Any]]',
) -> 'List[Dict[str, Any]]':
    """
    Translate the user's drawing convention to the solver's internal convention.

    User-facing convention (this is what the user is asked to do when drawing
    geometry in the GUI or writing a .geo file):

        TYPE 2 (PEC / IBC body in air):
            Draw the boundary so the normal points INTO AIR, i.e., away
            from the conductor.  Example: on the top of a PEC body drawn
            left-to-right, the normal points UP.

        TYPE 3 (air / dielectric interface):
            Draw the boundary so the normal points INTO AIR, away from
            the dielectric region.  pos_mat names the dielectric material
            ON THE OPPOSITE SIDE OF THE NORMAL.  Example: on the top of a
            dielectric body drawn left-to-right, the normal points UP
            (into air), and pos_mat is the dielectric below.

        TYPE 4 (dielectric / PEC interface):
            No air is involved.  Draw the boundary so the normal points
            FROM THE PEC INTO THE DIELECTRIC (i.e., into the pos_mat region).
            Example: on the top of a PEC-backed dielectric coating drawn
            left-to-right, the normal points UP into the dielectric
            coating that sits above.

        TYPE 5 (dielectric / dielectric interface):
            No air is involved.  The normal points FROM neg_mat INTO pos_mat,
            i.e., pos_mat is on the normal side.  User chooses which
            dielectric to label pos_mat and which to label neg_mat based on
            the endpoint order they drew.

        TYPE 1 (free-floating resistive / reactive card):
            Both sides of a free card are air; the sheet impedance BC is
            symmetric.  Normal direction is physically irrelevant; the
            user's endpoint order is accepted as-is.

    Solver-internal convention (unchanged):
        - TYPE 1 sheet:  plus = virtual sheet region,  minus = air
        - TYPE 2 PEC:    plus = interior (-1),         minus = air
        - TYPE 3 diel:   plus = pos_mat dielectric,       minus = air
        - TYPE 4 coat:   plus = pos_mat dielectric,       minus = PEC interior
        - TYPE 5 d/d:    plus = pos_mat,                  minus = neg_mat

    The solver's "plus" side is always the side the stored panel normal points
    toward.  For TYPE 2 and TYPE 3 the user draws the normal pointing away
    from the plus side, so we reverse endpoint order to align conventions.
    For TYPE 4 and TYPE 5 the user already draws with the normal pointing
    toward the plus / pos_mat side, so no flip is needed.  TYPE 1 is symmetric.
    """

    if seg_type not in (2, 3):
        return point_pairs
    return _reverse_point_pairs(point_pairs)

def _snapshot_segments(geometry_snapshot: 'Dict[str, Any]') -> 'List[Dict[str, Any]]':
    return list(geometry_snapshot.get('segments', []) or [])

def _solver_point_key(x: 'float', y: 'float', tol: 'float') -> 'Tuple[int, int]':
    inv = 1.0 / max(tol, 1e-12)
    return int(round(float(x) * inv)), int(round(float(y) * inv))

def _points_close(a: 'Tuple[float, float]', b: 'Tuple[float, float]', tol: 'float') -> 'bool':
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) <= (tol * tol)

def _segment_intersects_strict(
    a1: 'Tuple[float, float]',
    a2: 'Tuple[float, float]',
    b1: 'Tuple[float, float]',
    b2: 'Tuple[float, float]',
    tol: 'float',
) -> 'bool':
    if _points_close(a1, b1, tol) or _points_close(a1, b2, tol) or _points_close(a2, b1, tol) or _points_close(a2, b2, tol):
        return False

    def orient(p, q, r):
        return (float(q[0]) - float(p[0])) * (float(r[1]) - float(p[1])) - (float(q[1]) - float(p[1])) * (float(r[0]) - float(p[0]))

    def on_seg(p, q, r):
        return (
            min(float(p[0]), float(r[0])) - tol <= float(q[0]) <= max(float(p[0]), float(r[0])) + tol
            and min(float(p[1]), float(r[1])) - tol <= float(q[1]) <= max(float(p[1]), float(r[1])) + tol
        )

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)

    if ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)):
        return True
    if abs(o1) <= tol and on_seg(a1, b1, a2):
        return True
    if abs(o2) <= tol and on_seg(a1, b2, a2):
        return True
    if abs(o3) <= tol and on_seg(b1, a1, b2):
        return True
    if abs(o4) <= tol and on_seg(b1, a2, b2):
        return True
    return False

def validate_geometry_snapshot_for_solver(
    geometry_snapshot: 'Dict[str, Any]',
    base_dir: 'str',
    meters_scale: 'float' = 1.0,
    material_library: 'Optional[MaterialLibrary]' = None,
) -> 'Dict[str, Any]':
    """
    Strict solver-side preflight for geometry/material consistency.

    This complements the GUI validator and protects headless solves / exports.
    Fatal problems raise before assembly begins.

    ``meters_scale`` converts snapshot coordinates to meters (pass the same
    unit scale the solver uses).  It is needed to detect "cracks": endpoint
    gaps small enough to look connected at drawing precision but larger than
    the mesh node-snap tolerance (1e-9 m absolute), which would silently
    mesh a closed body as an open contour.
    """

    segments = _snapshot_segments(geometry_snapshot)
    if not segments:
        raise ValueError('Geometry snapshot contains no segments.')

    ibc_rows = [list(row) for row in (geometry_snapshot.get('ibcs', []) or []) if list(row)]
    diel_rows = [list(row) for row in (geometry_snapshot.get('dielectrics', []) or []) if list(row)]
    ibc_flags = {
        _parse_material_definition_flag(
            row[0], f"IBC definition row {idx + 1} flag"
        )
        for idx, row in enumerate(ibc_rows)
    }
    diel_flags = {
        _parse_material_definition_flag(
            row[0], f"Dielectric definition row {idx + 1} flag"
        )
        for idx, row in enumerate(diel_rows)
    }
    # Parse and validate every material definition before geometry assembly.
    # Both 2D and BoR call this preflight, so missing/malformed explicit CSV
    # sidecars fail identically in both solver paths.
    # Batched HPC planning already holds the immutable material library for
    # this geometry. Reusing it avoids reopening and reparsing the same CSV
    # sidecars hundreds of times. Ordinary solver callers omit the argument
    # and retain the original fail-closed construction path.
    if material_library is None:
        MaterialLibrary.from_entries(ibc_rows, diel_rows, base_dir)

    warnings: 'List[str]' = []
    primitives: 'List[Tuple[int, int, str, Tuple[float, float], Tuple[float, float]]]' = []
    all_points: 'List[Tuple[float, float]]' = []
    chain_discontinuities: 'List[Tuple[str, int]]' = []

    for seg_idx, seg in enumerate(segments):
        props = list(seg.get('properties', []) or [])
        # New 5-field layout: type n ibc pos_mat neg_mat. Pad direct,
        # programmatic snapshots with '' so blank tail fields use the explicit
        # zero defaults below. Parsed .geo files require all five fields.
        if len(props) < 5:
            props.extend([''] * (5 - len(props)))
        seg_name = str(seg.get('name', f'segment_{seg_idx + 1}'))
        header_type_token = seg.get('seg_type')
        property_type_token = props[0]
        has_header_type = (
            header_type_token is not None
            and bool(str(header_type_token).strip())
        )
        has_property_type = bool(str(property_type_token).strip())
        header_type = (
            _parse_geometry_integer(
                header_type_token, f"Segment '{seg_name}' header TYPE"
            )
            if has_header_type else None
        )
        property_type = (
            _parse_geometry_integer(
                property_type_token, f"Segment '{seg_name}' properties TYPE"
            )
            if has_property_type else None
        )
        if (
            header_type is not None
            and property_type is not None
            and header_type != property_type
        ):
            raise ValueError(
                f"Segment '{seg_name}' declares TYPE {header_type} in its "
                f"header but TYPE {property_type} in properties[0]. "
                "The two TYPE declarations must match."
            )
        seg_type = (
            property_type
            if property_type is not None
            else header_type if header_type is not None else 0
        )
        if str(props[1]).strip():
            _parse_geometry_integer(props[1], f"Segment '{seg_name}' N")
        ibc_flag = (
            _parse_geometry_integer(
                props[2], f"Segment '{seg_name}' IBC flag",
                allow_mat_prefix=True,
            )
            if str(props[2]).strip() else 0
        )
        pos_mat = (
            _parse_geometry_integer(
                props[3], f"Segment '{seg_name}' pos_mat flag",
                allow_mat_prefix=True,
            )
            if str(props[3]).strip() else 0
        )
        neg_mat = (
            _parse_geometry_integer(
                props[4], f"Segment '{seg_name}' neg_mat flag",
                allow_mat_prefix=True,
            )
            if str(props[4]).strip() else 0
        )
        point_pairs = list(seg.get('point_pairs', []) or [])

        if seg_type < 1 or seg_type > 5:
            raise ValueError(f"Segment '{seg_name}' has invalid TYPE '{props[0]}'; expected 1..5.")
        for field_name, flag in (
            ("IBC", ibc_flag),
            ("pos_mat", pos_mat),
            ("neg_mat", neg_mat),
        ):
            if flag < 0:
                raise ValueError(
                    f"Segment '{seg_name}' {field_name} flag must be "
                    f"non-negative; got {flag}."
                )
        if not point_pairs:
            raise ValueError(f"Segment '{seg_name}' has no primitives/point_pairs.")

        if ibc_flag > 0 and seg_type in (3, 5):
            raise ValueError(
                f"TYPE {seg_type} segment '{seg_name}' assigns IBC flag {ibc_flag} to a "
                "dielectric transmission interface. Surface impedance on TYPE 3/5 "
                "interfaces is not implemented by the 2D transmission formulations; "
                "remove the IBC flag or model the impedance on a supported TYPE 1, 2, "
                "or 4 boundary."
            )

        prev_end = None
        for prim_idx, pair in enumerate(point_pairs):
            context = f"Segment '{seg_name}' primitive {prim_idx + 1}"
            missing = [key for key in ('x1', 'y1', 'x2', 'y2') if key not in pair]
            if missing:
                raise ValueError(
                    f"{context} is missing coordinate field(s): "
                    + ", ".join(missing)
                )
            x1 = _parse_geometry_float(pair['x1'], f"{context} x1")
            y1 = _parse_geometry_float(pair['y1'], f"{context} y1")
            x2 = _parse_geometry_float(pair['x2'], f"{context} x2")
            y2 = _parse_geometry_float(pair['y2'], f"{context} y2")
            vals = [x1, y1, x2, y2]
            if not all(math.isfinite(v) for v in vals):
                raise ValueError(f"Segment '{seg_name}' primitive {prim_idx + 1} contains non-finite coordinates.")
            if ((x2 - x1) ** 2 + (y2 - y1) ** 2) <= EPS * EPS:
                raise ValueError(f"Segment '{seg_name}' primitive {prim_idx + 1} has near-zero length.")
            p1 = (x1, y1)
            p2 = (x2, y2)
            primitives.append((seg_idx, prim_idx, seg_name, p1, p2))
            all_points.extend([p1, p2])
            if prev_end is not None and not _points_close(prev_end, p1, 1e-9):
                # Defer until after the pairwise checks so a duplicate or
                # overlapping primitive receives its more specific error.
                chain_discontinuities.append((seg_name, prim_idx))
            prev_end = p2

        if ibc_flag > 0:
            if ibc_flag not in ibc_flags:
                raise ValueError(f"Segment '{seg_name}' references undefined IBC flag {ibc_flag}.")

        if seg_type == 3:
            if pos_mat <= 0:
                raise ValueError(f"TYPE 3 segment '{seg_name}' requires pos_mat > 0.")
            if pos_mat not in diel_flags:
                raise ValueError(f"TYPE 3 segment '{seg_name}' references undefined dielectric flag {pos_mat}.")
        elif seg_type == 4:
            if pos_mat <= 0:
                raise ValueError(f"TYPE 4 segment '{seg_name}' requires pos_mat > 0.")
            if pos_mat not in diel_flags:
                raise ValueError(f"TYPE 4 segment '{seg_name}' references undefined dielectric flag {pos_mat}.")
        elif seg_type == 5:
            if pos_mat <= 0 or neg_mat <= 0:
                raise ValueError(f"TYPE 5 segment '{seg_name}' requires pos_mat > 0 and neg_mat > 0.")
            if pos_mat == neg_mat:
                raise ValueError(
                    f"TYPE 5 segment '{seg_name}' assigns the same dielectric "
                    f"flag {pos_mat} to both sides. A same-medium interface is "
                    "physically redundant and must be removed."
                )
            for flag in (pos_mat, neg_mat):
                if flag not in diel_flags:
                    raise ValueError(f"TYPE 5 segment '{seg_name}' references undefined dielectric flag {flag}.")

    xs = [p[0] for p in all_points] if all_points else [0.0]
    ys = [p[1] for p in all_points] if all_points else [0.0]
    diag = max(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1.0)
    tol = max(1e-8, 1e-6 * diag)

    node_degree: 'Dict[Tuple[int, int], int]' = {}
    for _, _, _, p1, p2 in primitives:
        key1 = _solver_point_key(p1[0], p1[1], tol)
        key2 = _solver_point_key(p2[0], p2[1], tol)
        node_degree[key1] = node_degree.get(key1, 0) + 1
        node_degree[key2] = node_degree.get(key2, 0) + 1

    dangling_nodes = sum(1 for v in node_degree.values() if v == 1)
    high_degree_nodes = sum(1 for v in node_degree.values() if v > 2)
    if dangling_nodes > 0:
        warnings.append(f'Geometry contains {dangling_nodes} dangling endpoint node(s).')
    if high_degree_nodes > 0:
        warnings.append(f'Geometry contains {high_degree_nodes} high-degree node(s) (>2 connected primitives).')

    # -- Crack detection ---------------------------------------------------
    # Endpoints that nearly coincide (within the validator's drawing
    # tolerance) but sit farther apart than the mesh node-snap tolerance
    # (1e-9 m absolute) will NOT be merged during meshing: a visually closed
    # body silently meshes as an open contour, with wrong physics and tiny
    # residuals.  That gap window is a fatal error, not a warning.
    snap_tol_raw = 1.0e-9 / max(float(meters_scale), EPS)   # mesh snap tol in snapshot units
    crack_floor_raw = 1.0e-12 / max(float(meters_scale), EPS)  # below this: exact-coincidence float noise
    endpoint_list = sorted(set(all_points))
    for i in range(len(endpoint_list)):
        px, py = endpoint_list[i]
        j = i + 1
        while j < len(endpoint_list) and endpoint_list[j][0] - px <= tol:
            qx, qy = endpoint_list[j]
            j += 1
            gap = math.hypot(qx - px, qy - py)
            if gap > tol or gap <= crack_floor_raw:
                continue
            if gap > snap_tol_raw:
                raise ValueError(
                    f"Geometry crack: endpoints ({px:.9g}, {py:.9g}) and ({qx:.9g}, {qy:.9g}) "
                    f"are {gap * meters_scale:.3g} m apart -- close enough to look connected, but "
                    "beyond the 1e-9 m mesh node-snap tolerance, so they would mesh as an OPEN "
                    "gap. Make the endpoints exactly coincident (or separate them intentionally)."
                )

    # Broad-phase sweep before the exact pair checks below.  The previous
    # unconditional all-pairs loop made preflight O(P^2), which became a
    # noticeable part of every solve for already-discretized .geo files with
    # hundreds or thousands of primitives.  Any duplicate, overlap, or proper
    # intersection must have tolerance-expanded bounding boxes that overlap,
    # so the sweep only removes pairs that cannot possibly trigger a check.
    # Choose the geometry's longer axis to avoid the vertical/horizontal-chain
    # worst cases, then sort candidate pairs back into the old deterministic
    # (i, j) order before applying the exact predicates.
    primitive_bounds = []
    for _, _, _, p1, p2 in primitives:
        primitive_bounds.append((
            min(float(p1[0]), float(p2[0])),
            max(float(p1[0]), float(p2[0])),
            min(float(p1[1]), float(p2[1])),
            max(float(p1[1]), float(p2[1])),
        ))
    use_x_axis = (max(xs) - min(xs)) >= (max(ys) - min(ys))
    primary_min = 0 if use_x_axis else 2
    primary_max = 1 if use_x_axis else 3
    secondary_min = 2 if use_x_axis else 0
    secondary_max = 3 if use_x_axis else 1
    sweep_order = sorted(
        range(len(primitives)),
        key=lambda idx: (primitive_bounds[idx][primary_min], idx),
    )
    candidate_pairs: 'List[Tuple[int, int]]' = []
    for sweep_pos, raw_i in enumerate(sweep_order):
        bounds_i = primitive_bounds[raw_i]
        for later_pos in range(sweep_pos + 1, len(sweep_order)):
            raw_j = sweep_order[later_pos]
            bounds_j = primitive_bounds[raw_j]
            if bounds_j[primary_min] > bounds_i[primary_max] + tol:
                break
            if (
                bounds_j[secondary_min] > bounds_i[secondary_max] + tol
                or bounds_i[secondary_min] > bounds_j[secondary_max] + tol
            ):
                continue
            candidate_pairs.append(
                (raw_i, raw_j) if raw_i < raw_j else (raw_j, raw_i)
            )

    for i, j in sorted(candidate_pairs):
        seg_i, prim_i, name_i, a1, a2 = primitives[i]
        seg_j, prim_j, name_j, b1, b2 = primitives[j]

        # Duplicate primitive (same endpoints, either order): the boundary
        # would be discretized twice, doubling the equivalent currents.
        # Checked before the adjacency skip -- a duplicate is never legal.
        same_fwd = _points_close(a1, b1, tol) and _points_close(a2, b2, tol)
        same_rev = _points_close(a1, b2, tol) and _points_close(a2, b1, tol)
        if same_fwd or same_rev:
            raise ValueError(
                f"Duplicate primitive: '{name_i}' primitive {prim_i + 1} and "
                f"'{name_j}' primitive {prim_j + 1} have identical endpoints. "
                "Remove one -- a doubled boundary doubles the surface currents."
            )

        # Collinear overlap through a shared endpoint (e.g. (0,0)-(1,0)
        # and (0,0)-(2,0)): _segment_intersects_strict deliberately
        # ignores shared-endpoint pairs, so test overlap explicitly.
        shared = None
        for pa in (a1, a2):
            for pb in (b1, b2):
                if _points_close(pa, pb, tol):
                    shared = (pa, pb)
                    break
            if shared:
                break
        if shared is not None:
            oa = a2 if shared[0] is a1 else a1   # other end of primitive i
            ob = b2 if shared[1] is b1 else b1   # other end of primitive j
            ux, uy = oa[0] - shared[0][0], oa[1] - shared[0][1]
            vx, vy = ob[0] - shared[1][0], ob[1] - shared[1][1]
            lu = math.hypot(ux, uy)
            lv = math.hypot(vx, vy)
            if lu > EPS and lv > EPS:
                cross = abs(ux * vy - uy * vx) / (lu * lv)
                dot = (ux * vx + uy * vy) / (lu * lv)
                if cross < 1.0e-7 and dot > 0.0 and min(lu, lv) > tol:
                    raise ValueError(
                        f"Collinear overlapping primitives: '{name_i}' primitive {prim_i + 1} and "
                        f"'{name_j}' primitive {prim_j + 1} run along the same line from a shared "
                        f"endpoint, overlapping for {min(lu, lv) * meters_scale:.3g} m. "
                        "Split or remove the overlapping span."
                    )
            continue

        if seg_i == seg_j and abs(prim_i - prim_j) <= 1:
            continue
        if _segment_intersects_strict(a1, a2, b1, b2, tol):
            raise ValueError(
                f"Geometry contains an unsupported segment intersection between '{name_i}' primitive {prim_i + 1} and '{name_j}' primitive {prim_j + 1}."
            )

    if chain_discontinuities:
        seg_name, prim_idx = chain_discontinuities[0]
        raise ValueError(
            f"Segment '{seg_name}' has a disconnected primitive chain "
            f"between elements {prim_idx} and {prim_idx + 1}. "
            "Primitives within one segment must chain head-to-tail; "
            "put disconnected geometry in separate segments."
        )

    # Winding / air-side consistency: wrong orientation silently corrupts TE
    # results, so it is a fatal preflight error (never auto-corrected).
    _check_segment_orientation_or_raise(segments)

    return {
        'segment_count': int(len(segments)),
        'primitive_count': int(len(primitives)),
        'dangling_nodes': int(dangling_nodes),
        'high_degree_nodes': int(high_degree_nodes),
        'warning_count': int(len(warnings)),
        'warnings': warnings,
    }

def _build_panels(
    geometry_snapshot: 'Dict[str, Any]',
    meters_scale: 'float',
    min_wavelength: 'float',
    max_panels: 'int' = MAX_PANELS_DEFAULT,
) -> 'List[Panel]':
    """
    Discretize all geometry primitives into oriented boundary elements.

    Normal direction follows endpoint ordering of each primitive.  Wrong
    winding is a hard preflight error (see
    `_check_segment_orientation_or_raise`), never silently corrected here.
    """

    panels: 'List[Panel]' = []
    segments = geometry_snapshot.get("segments", []) or []
    refinement_factor = float(
        geometry_snapshot.get("_2d_certification_refinement_factor", 1.0)
        or 1.0
    )
    base_segment_n = list(
        geometry_snapshot.get("_2d_certification_base_segment_n", []) or []
    )
    if not math.isfinite(refinement_factor) or refinement_factor < 1.0:
        raise ValueError(
            "Internal 2-D certification refinement factor must be finite and >= 1."
        )
    if refinement_factor > 1.0 and len(base_segment_n) != len(segments):
        raise ValueError(
            "Internal 2-D certification refinement metadata does not match "
            "the geometry segment count."
        )

    for seg_idx, seg in enumerate(segments):
        props = list(seg.get("properties", []) or [])
        # TYPE resolution must mirror validate_geometry_snapshot_for_solver:
        # props[0] when non-blank, else the segment's seg_type field.  (The
        # old `props[0] else 2` default let a snapshot validated as one TYPE
        # be built as another when the properties list was empty.)
        seg_type = _parse_flag(
            props[0] if len(props) > 0 and str(props[0]).strip() else seg.get("seg_type", 2)
        )
        # Missing/blank N means "auto density" (n=0, lambda/N_default meshing),
        # NOT an explicit 1-panel-per-primitive request: a blank N on a long
        # primitive used to silently mesh a 10-wavelength line into 1 panel.
        n_prop = _parse_int(props[1] if len(props) > 1 else 0, 0)
        ibc_flag = _parse_flag(props[2] if len(props) > 2 else 0)
        pos_mat = _parse_flag(props[3] if len(props) > 3 else 0)
        neg_mat = _parse_flag(props[4] if len(props) > 4 else 0)
        name = str(seg.get("name", "segment"))

        point_pairs = list(seg.get("point_pairs", []) or [])
        # The user's endpoint order is the source of truth for winding
        # (wrong winding raises in preflight).  Translate the user's drawing
        # convention (normal points into air) to the solver's internal
        # convention (normal points into plus = pos_mat) for the boundary
        # types where air is semantically the minus side.
        point_pairs = _normalize_segment_orientation(seg_type, point_pairs, meters_scale)
        point_pairs = _apply_user_convention_flip(seg_type, point_pairs)

        # Remember which panels this segment contributes; arc-length positions
        # along the segment are normalized after the segment is fully discretized.
        seg_start_idx = len(panels)

        for pair in point_pairs:
            p0 = np.asarray([
                _parse_float(pair.get("x1", 0.0), 0.0) * meters_scale,
                _parse_float(pair.get("y1", 0.0), 0.0) * meters_scale,
            ], dtype=float)
            p1 = np.asarray([
                _parse_float(pair.get("x2", 0.0), 0.0) * meters_scale,
                _parse_float(pair.get("y2", 0.0), 0.0) * meters_scale,
            ], dtype=float)

            prim_len = _primitive_length(p0, p1)
            if refinement_factor > 1.0:
                # Certification must refine the mesh that was ACTUALLY used,
                # not merely tighten a wavelength threshold.  Pre-discretized
                # geometry can already have primitives shorter than both the
                # base and scaled thresholds, previously yielding identical
                # base/fine meshes and a vacuous convergence certificate.
                base_n_prop = _parse_int(base_segment_n[seg_idx], 0)
                base_count = _panel_count_from_n(
                    base_n_prop, prim_len, min_wavelength
                )
                count = max(
                    base_count + 1,
                    int(math.ceil(base_count * refinement_factor)),
                )
            else:
                count = _panel_count_from_n(
                    n_prop, prim_len, min_wavelength
                )
            if n_prop > 0 and min_wavelength > EPS:
                minimum_count = max(
                    1,
                    int(math.ceil(
                        prim_len
                        * float(MIN_EXPLICIT_PANELS_PER_WAVELENGTH)
                        / min_wavelength
                    )),
                )
                if count < minimum_count:
                    raise ValueError(
                        f"Segment '{name}' explicit N={n_prop} under-resolves "
                        f"a {prim_len:.6g} m primitive at the controlling "
                        f"wavelength {min_wavelength:.6g} m: at least "
                        f"N={minimum_count} is required for the gross "
                        f"{MIN_EXPLICIT_PANELS_PER_WAVELENGTH}-panels-per-"
                        "wavelength safety floor. Increase N or use N=0 for "
                        "automatic material-wavelength meshing. Production "
                        "results still require base/fine mesh convergence."
                    )
            pts = _discretize_primitive(p0, p1, count)

            for i in range(count):
                q0 = pts[i]
                q1 = pts[i + 1]
                vec = q1 - q0
                length = float(np.linalg.norm(vec))
                if length <= EPS:
                    continue
                tangent = vec / length
                # Project convention: a segment drawn left->right has an upward normal.
                # This makes pos_mat the medium on the GUI-indicated normal side.
                normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
                center = 0.5 * (q0 + q1)
                panels.append(
                    Panel(
                        name=name,
                        seg_type=seg_type,
                        ibc_flag=ibc_flag,
                        pos_mat=pos_mat,
                        neg_mat=neg_mat,
                        p0=q0,
                        p1=q1,
                        center=center,
                        tangent=tangent,
                        normal=normal,
                        length=length,
                        arc_s_center=0.5,  # placeholder, normalized below
                    )
                )

        # Assign normalized arc-length positions to panels from this segment.
        seg_panels = panels[seg_start_idx:]
        if seg_panels:
            total_len = sum(p.length for p in seg_panels)
            if total_len > EPS:
                cum = 0.0
                for p in seg_panels:
                    p.arc_s_center = (cum + 0.5 * p.length) / total_len
                    cum += p.length
            else:
                for p in seg_panels:
                    p.arc_s_center = 0.5

            # `_apply_user_convention_flip` reverses panel order for TYPE 2/3.
            # Invert arc_s so that arc_s=0 always corresponds to the start of
            # the segment as DRAWN by the user, regardless of the internal flip.
            if seg_type in (2, 3):
                for p in seg_panels:
                    p.arc_s_center = 1.0 - p.arc_s_center

    if not panels:
        raise ValueError("Geometry does not contain any valid discretized panels.")
    max_allowed = max(1, int(max_panels))
    if len(panels) > max_allowed:
        raise ValueError(
            f"Discretization produced {len(panels)} panels; limit is {max_allowed}. "
            "Reduce n/frequency range or increase max_panels."
        )
    return panels

def _linear_node_snap_key(xy: 'np.ndarray', tol: 'float' = 1.0e-9) -> 'Tuple[int, int]':
    scale = 1.0 / max(float(tol), EPS)
    return (int(round(float(xy[0]) * scale)), int(round(float(xy[1]) * scale)))

def _linear_shape_values(xi: 'float') -> 'np.ndarray':
    x = float(xi)
    return np.asarray([1.0 - x, x], dtype=float)

def _build_linear_mesh(
    panels: 'List[Panel]',
    node_snap_tol: 'float' = 1.0e-9,
) -> 'LinearMesh':
    """
    Convert boundary elements into a continuous two-node linear boundary mesh.

    This is the stage-1 data-structure upgrade for the future linear Galerkin path.
    Each panel becomes one linear element, while shared endpoints are merged into
    unique global nodes by snapped coordinates.
    """

    node_index: 'Dict[Tuple[int, int], int]' = {}
    nodes: 'List[LinearNode]' = []
    elements: 'List[LinearElement]' = []

    def get_node_id(xy: 'np.ndarray') -> 'int':
        key = _linear_node_snap_key(xy, tol=node_snap_tol)
        idx = node_index.get(key)
        if idx is not None:
            return idx
        idx = len(nodes)
        node_index[key] = idx
        nodes.append(LinearNode(xy=np.asarray(xy, dtype=float).copy(), key=key))
        return idx

    for pidx, panel in enumerate(panels):
        n0 = get_node_id(panel.p0)
        n1 = get_node_id(panel.p1)
        elements.append(
            LinearElement(
                name=panel.name,
                seg_type=panel.seg_type,
                ibc_flag=panel.ibc_flag,
                pos_mat=panel.pos_mat,
                neg_mat=panel.neg_mat,
                node_ids=(n0, n1),
                p0=np.asarray(panel.p0, dtype=float).copy(),
                p1=np.asarray(panel.p1, dtype=float).copy(),
                center=np.asarray(panel.center, dtype=float).copy(),
                tangent=np.asarray(panel.tangent, dtype=float).copy(),
                normal=np.asarray(panel.normal, dtype=float).copy(),
                length=float(panel.length),
                panel_index=int(pidx),
                arc_s_center=float(panel.arc_s_center),
            )
        )

    if not elements:
        raise ValueError("Linear mesh construction requires at least one element.")
    return LinearMesh(nodes=nodes, elements=elements)

def _linear_panel_signature_from_info(
    panel: 'Panel',
    info: 'PanelCoupledInfo',
) -> 'Tuple[Any, ...]':
    """Topology signature used to decide when linear nodes may be shared safely."""

    return (
        int(panel.seg_type),
        int(panel.ibc_flag),
        int(panel.pos_mat),
        int(panel.neg_mat),
        int(info.minus_region),
        int(info.plus_region),
        str(info.bc_kind),
    )

def _build_linear_mesh_interface_aware(
    panels: 'List[Panel]',
    infos: 'List[PanelCoupledInfo]',
    node_snap_tol: 'float' = 1.0e-9,
) -> 'Tuple[LinearMesh, Dict[str, int]]':
    """
    Build a linear boundary mesh that only shares nodes across the *same* interface signature.

    This hardens the linear/Galerkin path for ordinary corners where distinct interface types
    touch at the same geometric coordinate. Those cases should not be forced to share a single
    nodal DOF, because that incorrectly imposes trace continuity across different interfaces.

    True branching nodes where more than two elements of the same interface signature meet are
    still reported separately by `_linear_coupled_node_report` for diagnostics.
    solver in production runs.
    """

    if len(panels) != len(infos):
        raise ValueError("Interface-aware linear mesh requires matching panels and panel infos.")

    node_index: 'Dict[Tuple[Tuple[int, int], Tuple[Any, ...]], int]' = {}
    nodes: 'List[LinearNode]' = []
    elements: 'List[LinearElement]' = []
    geometric_keys: 'Set[Tuple[int, int]]' = set()

    def get_node_id(xy: 'np.ndarray', signature: 'Tuple[Any, ...]') -> 'int':
        geom_key = _linear_node_snap_key(xy, tol=node_snap_tol)
        geometric_keys.add(geom_key)
        full_key = (geom_key, signature)
        idx = node_index.get(full_key)
        if idx is not None:
            return idx
        idx = len(nodes)
        node_index[full_key] = idx
        nodes.append(LinearNode(xy=np.asarray(xy, dtype=float).copy(), key=geom_key))
        return idx

    for pidx, (panel, info) in enumerate(zip(panels, infos)):
        sig = _linear_panel_signature_from_info(panel, info)
        n0 = get_node_id(panel.p0, sig)
        n1 = get_node_id(panel.p1, sig)
        elements.append(
            LinearElement(
                name=panel.name,
                seg_type=panel.seg_type,
                ibc_flag=panel.ibc_flag,
                pos_mat=panel.pos_mat,
                neg_mat=panel.neg_mat,
                node_ids=(n0, n1),
                p0=np.asarray(panel.p0, dtype=float).copy(),
                p1=np.asarray(panel.p1, dtype=float).copy(),
                center=np.asarray(panel.center, dtype=float).copy(),
                tangent=np.asarray(panel.tangent, dtype=float).copy(),
                normal=np.asarray(panel.normal, dtype=float).copy(),
                length=float(panel.length),
                panel_index=int(pidx),
                arc_s_center=float(panel.arc_s_center),
            )
        )

    if not elements:
        raise ValueError("Interface-aware linear mesh construction requires at least one element.")

    mesh = LinearMesh(nodes=nodes, elements=elements)
    geometric_count = int(len(geometric_keys))
    total_nodes = int(len(nodes))
    split_nodes = max(0, total_nodes - geometric_count)

    # Count geometric locations where multiple interface signatures created separate nodes.
    geo_key_counts: 'Dict[Tuple[int, int], int]' = {}
    for (gk, _sig), _nid in node_index.items():
        geo_key_counts[gk] = geo_key_counts.get(gk, 0) + 1
    multi_sig = sum(1 for c in geo_key_counts.values() if c > 1)

    stats = {
        "linear_geometric_node_count": geometric_count,
        "linear_interface_split_nodes": split_nodes,
        "shared_node_count": geometric_count,
        "split_node_count": split_nodes,
        "split_boundary_primitive_count": int(len(elements)),
        "multi_signature_node_count": multi_sig,
    }
    return mesh, stats

def _linear_param_to_point(elem: 'LinearElement', xi: 'float') -> 'np.ndarray':
    return elem.p0 + float(xi) * (elem.p1 - elem.p0)

def _linear_interval_point(elem: 'LinearElement', interval: 'Tuple[float, float]', use_start: 'bool') -> 'np.ndarray':
    a, b = float(interval[0]), float(interval[1])
    return _linear_param_to_point(elem, a if use_start else b)

def _linear_interval_length(elem: 'LinearElement', interval: 'Tuple[float, float]') -> 'float':
    a, b = float(interval[0]), float(interval[1])
    return max(abs(b - a) * float(elem.length), 0.0)

def _linear_interval_midpoint(elem: 'LinearElement', interval: 'Tuple[float, float]') -> 'np.ndarray':
    a, b = float(interval[0]), float(interval[1])
    return _linear_param_to_point(elem, 0.5 * (a + b))

def _linear_map_local_to_parent(interval: 'Tuple[float, float]', local_xi: 'float', start_is_shared: 'bool') -> 'float':
    a, b = float(interval[0]), float(interval[1])
    h = b - a
    x = float(local_xi)
    return (a + h * x) if start_is_shared else (b - h * x)

def _linear_shared_interval_endpoint_info(
    obs_elem: 'LinearElement',
    obs_interval: 'Tuple[float, float]',
    src_elem: 'LinearElement',
    src_interval: 'Tuple[float, float]',
    tol: 'float' = 1.0e-12,
) -> 'Optional[Tuple[bool, bool]]':
    obs_pts = [
        _linear_interval_point(obs_elem, obs_interval, True),
        _linear_interval_point(obs_elem, obs_interval, False),
    ]
    src_pts = [
        _linear_interval_point(src_elem, src_interval, True),
        _linear_interval_point(src_elem, src_interval, False),
    ]
    for obs_is_start, op in enumerate(obs_pts):
        for src_is_start, sp in enumerate(src_pts):
            if float(np.linalg.norm(op - sp)) <= float(tol):
                return bool(obs_is_start == 0), bool(src_is_start == 0)
    return None

def _integrate_linear_pair_box(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    kernel_eval: 'Callable[[np.ndarray, np.ndarray], complex]',
    obs_interval: 'Tuple[float, float]',
    src_interval: 'Tuple[float, float]',
    obs_order: 'int',
    src_order: 'int',
) -> 'np.ndarray':
    qt_obs, qw_obs = _get_quadrature(max(2, int(obs_order)))
    qt_src, qw_src = _get_quadrature(max(2, int(src_order)))
    obs_scale = max(float(obs_interval[1]) - float(obs_interval[0]), 0.0)
    src_scale = max(float(src_interval[1]) - float(src_interval[0]), 0.0)
    obs_len = float(obs_elem.length) * obs_scale
    src_len = float(src_elem.length) * src_scale
    block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return block

    for tobs, wobs in zip(qt_obs, qw_obs):
        xi_obs = float(obs_interval[0]) + obs_scale * float(tobs)
        phi_obs = _linear_shape_values(xi_obs)
        robs = _linear_param_to_point(obs_elem, xi_obs)
        for tsrc, wsrc in zip(qt_src, qw_src):
            xi_src = float(src_interval[0]) + src_scale * float(tsrc)
            phi_src = _linear_shape_values(xi_src)
            rsrc = _linear_param_to_point(src_elem, xi_src)
            kval = complex(kernel_eval(robs, rsrc))
            block += (float(wobs) * float(wsrc) * kval) * np.outer(phi_obs, phi_src)

    return block * obs_len * src_len

def _integrate_linear_pair_box_sk_vectorized(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    obs_interval: 'Tuple[float, float]',
    src_interval: 'Tuple[float, float]',
    obs_order: 'int',
    src_order: 'int',
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """
    Vectorized tensor-Gauss 2x2 S and K block assembly for one element pair.

    Evaluates all quadrature point pairs at once using array Hankel functions,
    avoiding per-point Python-loop overhead.  Returns (S_block, K_block).
    """

    qt_obs, qw_obs = _get_quadrature(max(2, int(obs_order)))
    qt_src, qw_src = _get_quadrature(max(2, int(src_order)))
    oa, ob = float(obs_interval[0]), float(obs_interval[1])
    sa, sb = float(src_interval[0]), float(src_interval[1])
    obs_scale = max(ob - oa, 0.0)
    src_scale = max(sb - sa, 0.0)
    obs_len = float(obs_elem.length) * obs_scale
    src_len = float(src_elem.length) * src_scale
    s_block = np.zeros((2, 2), dtype=np.complex128)
    k_block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return s_block, k_block

    nobs = len(qt_obs)
    nsrc = len(qt_src)

    # Precompute all parametric coordinates and physical points.
    xi_obs_all = oa + obs_scale * np.asarray(qt_obs, dtype=float)          # (nobs,)
    xi_src_all = sa + src_scale * np.asarray(qt_src, dtype=float)          # (nsrc,)
    phi_obs_all = np.column_stack([1.0 - xi_obs_all, xi_obs_all])          # (nobs, 2)
    phi_src_all = np.column_stack([1.0 - xi_src_all, xi_src_all])          # (nsrc, 2)

    obs_seg = obs_elem.p1 - obs_elem.p0
    src_seg = src_elem.p1 - src_elem.p0
    robs_all = obs_elem.p0[None, :] + xi_obs_all[:, None] * obs_seg[None, :]  # (nobs, 2)
    rsrc_all = src_elem.p0[None, :] + xi_src_all[:, None] * src_seg[None, :]  # (nsrc, 2)

    # All pairwise differences: (nobs, nsrc, 2)
    diff = robs_all[:, None, :] - rsrc_all[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))   # (nobs, nsrc)
    dist_safe = np.maximum(dist, EPS)

    if not bool(compute_single_layer) and not bool(compute_double_layer):
        raise ValueError("At least one element-pair operator must be requested.")

    kr = np.asarray(complex(k0) * dist_safe, dtype=np.complex128)
    kr[np.abs(kr) <= 1e-12] = 1e-12 + 0.0j
    if compute_single_layer:
        # Green's function: G = j/4 * H_0^(2)(k*r)
        h0 = _hankel2_0_array(kr.ravel()).reshape(nobs, nsrc)
        g_vals = 0.25j * h0   # (nobs, nsrc)

    if compute_double_layer:
        h1 = _hankel2_1_array(kr.ravel()).reshape(nobs, nsrc)
        if obs_normal_deriv:
            # dG/dn_obs
            proj = np.sum(diff * obs_elem.normal[None, None, :], axis=2) / dist_safe
            dk_vals = (-0.25j * complex(k0)) * h1 * proj
        else:
            # dG/dn_src (diff = robs-rsrc; source normal flips sign)
            proj = np.sum(src_elem.normal[None, None, :] * diff, axis=2) / dist_safe
            dk_vals = (0.25j * complex(k0)) * h1 * proj
        dk_vals[dist <= EPS] = 0.0

    # Weight tensor: (nobs, nsrc)
    w_outer = np.outer(np.asarray(qw_obs, dtype=float), np.asarray(qw_src, dtype=float))

    # Accumulate 2x2 blocks using einsum.
    # weighted_g = w * G, shape (nobs, nsrc)
    if compute_single_layer:
        weighted_g = w_outer * g_vals
        # s_block[a,b] = sum weighted_g[i,j]*phi_obs[i,a]*phi_src[j,b]
        s_block = np.einsum(
            'ij,ia,jb->ab', weighted_g, phi_obs_all, phi_src_all
        )
    if compute_double_layer:
        weighted_k = w_outer * dk_vals
        k_block = np.einsum(
            'ij,ia,jb->ab', weighted_k, phi_obs_all, phi_src_all
        )

    scale = obs_len * src_len
    return s_block * scale, k_block * scale

def _single_layer_self_block_exact(
    elem: 'LinearElement',
    k0: 'Union[complex, float]',
    interval: 'Tuple[float, float]' = (0.0, 1.0),
) -> 'Optional[np.ndarray]':
    """
    Closed-form linear-Galerkin single-layer self block for a straight element.

    On a straight element the kernel depends only on u = |t - s|, so

        B_ij = l^2 * (j/4) * Int_0^1 H0^(2)(k*l*u) * C_ij(u) du

    with shape-pair weights (phi0 = 1-t, phi1 = t):

        C_diag(u) = (2 - 3u + u^3)/3      C_off(u) = (1 - u^3)/3

    (their sum reproduces the constant-basis weight 2(1-u) used by the
    exact `_single_layer_self_term`).  Substituting the small-argument
    series H0^(2)(x) = J0(x)[1 - j(2/pi)(ln(x/2)+gamma)] - j*R(x) turns the
    u-integral into exact moments:

        Int u^p du = 1/(p+1)        Int u^p ln(u) du = -1/(p+1)^2

    so the whole block is a rapidly convergent series -- machine precision,
    unlike the (u, uv) "Duffy" map, whose unresolved log singularity along
    the diagonal capped the self block at ~0.1-1% error.

    Returns None when |k*l| is too large for the series to be well
    conditioned (caller falls back to numeric quadrature).
    """

    a, b = float(interval[0]), float(interval[1])
    h = b - a
    ell = float(elem.length) * h
    if ell <= 0.0:
        return np.zeros((2, 2), dtype=np.complex128)
    z = complex(k0) * ell
    if abs(z) > 8.0:
        return None  # series cancellation grows beyond ~kl=8; use quadrature
    if abs(z) <= 1e-30:
        return None

    c_diag = (2.0 / 3.0, -1.0, 0.0, 1.0 / 3.0)   # coefficients of u^0..u^3
    c_off = (1.0 / 3.0, 0.0, 0.0, -1.0 / 3.0)

    def moment(p: 'int', coeffs) -> 'float':
        return sum(c / (p + q + 1) for q, c in enumerate(coeffs))

    def log_moment(p: 'int', coeffs) -> 'float':
        return -sum(c / (p + q + 1) ** 2 for q, c in enumerate(coeffs))

    two_over_pi = 2.0 / math.pi
    log_term = cmath.log(z / 2.0) + EULER_GAMMA
    z_quarter_sq = (z / 2.0) ** 2

    b_diag = 0.0 + 0.0j
    b_off = 0.0 + 0.0j
    alpha = 1.0 + 0.0j        # (-1)^m (z/2)^(2m) / (m!)^2
    harmonic = 0.0            # H_m
    m = 0
    while True:
        # H0^(2)(z*u) = sum_m alpha_m [1 - j(2/pi)(log_term - H_m)] u^(2m)
        #             - j(2/pi) ln(u) * sum_m alpha_m u^(2m)
        a_m = alpha * (1.0 - 1j * two_over_pi * (log_term - harmonic))
        p = 2 * m
        b_diag += a_m * moment(p, c_diag) - 1j * two_over_pi * alpha * log_moment(p, c_diag)
        b_off += a_m * moment(p, c_off) - 1j * two_over_pi * alpha * log_moment(p, c_off)
        m += 1
        alpha *= -z_quarter_sq / (m * m)
        harmonic += 1.0 / m
        if m > 60:
            return None
        if abs(alpha) < 1e-18 * max(1.0, abs(b_diag)):
            break

    block_local = (0.25j * ell * ell) * np.array(
        [[b_diag, b_off], [b_off, b_diag]], dtype=np.complex128,
    )
    if a == 0.0 and b == 1.0:
        return block_local
    # Sub-interval: parent shape functions restricted to [a, b] are linear
    # combinations of the local interval basis: phi_parent = T @ psi_local.
    t_mat = np.array([[1.0 - a, 1.0 - b], [a, b]], dtype=np.complex128)
    return t_mat @ block_local @ t_mat.T


def _integrate_linear_self_duffy(
    elem: 'LinearElement',
    kernel_eval: 'Callable[[np.ndarray, np.ndarray], complex]',
    interval: 'Tuple[float, float]',
    order: 'int' = 20,
) -> 'np.ndarray':
    qt, qw = _get_quadrature(max(4, int(order)))
    a, b = float(interval[0]), float(interval[1])
    h = max(b - a, 0.0)
    elem_len = float(elem.length) * h
    block = np.zeros((2, 2), dtype=np.complex128)
    if elem_len <= 0.0:
        return block

    for u, wu in zip(qt, qw):
        uu = float(u)
        jac_outer = float(wu) * uu
        t_major = a + h * uu
        s_major = t_major
        robs_major = _linear_param_to_point(elem, t_major)
        rsrc_major = _linear_param_to_point(elem, s_major)
        phi_t_major = _linear_shape_values(t_major)
        phi_s_major = _linear_shape_values(s_major)
        for v, wv in zip(qt, qw):
            vv = float(v)
            weight = jac_outer * float(wv)
            # Triangle: s <= t
            xi_t = a + h * uu
            xi_s = a + h * (uu * vv)
            phi_t = _linear_shape_values(xi_t)
            phi_s = _linear_shape_values(xi_s)
            robs = _linear_param_to_point(elem, xi_t)
            rsrc = _linear_param_to_point(elem, xi_s)
            block += weight * complex(kernel_eval(robs, rsrc)) * np.outer(phi_t, phi_s)
            # Triangle: t <= s
            xi_t2 = a + h * (uu * vv)
            xi_s2 = a + h * uu
            phi_t2 = _linear_shape_values(xi_t2)
            phi_s2 = _linear_shape_values(xi_s2)
            robs2 = _linear_param_to_point(elem, xi_t2)
            rsrc2 = _linear_param_to_point(elem, xi_s2)
            block += weight * complex(kernel_eval(robs2, rsrc2)) * np.outer(phi_t2, phi_s2)

    return block * (elem_len * elem_len)

def _integrate_linear_touching_duffy(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    kernel_eval: 'Callable[[np.ndarray, np.ndarray], complex]',
    obs_interval: 'Tuple[float, float]',
    src_interval: 'Tuple[float, float]',
    obs_start_is_shared: 'bool',
    src_start_is_shared: 'bool',
    order: 'int' = 20,
) -> 'np.ndarray':
    qt, qw = _get_quadrature(max(4, int(order)))
    obs_len = _linear_interval_length(obs_elem, obs_interval)
    src_len = _linear_interval_length(src_elem, src_interval)
    block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return block

    for u, wu in zip(qt, qw):
        uu = float(u)
        jac_outer = float(wu) * uu
        for v, wv in zip(qt, qw):
            vv = float(v)
            weight = jac_outer * float(wv)
            # Triangle 1: source local distance <= observation local distance
            xi_obs = _linear_map_local_to_parent(obs_interval, uu, obs_start_is_shared)
            xi_src = _linear_map_local_to_parent(src_interval, uu * vv, src_start_is_shared)
            phi_obs = _linear_shape_values(xi_obs)
            phi_src = _linear_shape_values(xi_src)
            robs = _linear_param_to_point(obs_elem, xi_obs)
            rsrc = _linear_param_to_point(src_elem, xi_src)
            block += weight * complex(kernel_eval(robs, rsrc)) * np.outer(phi_obs, phi_src)
            # Triangle 2: observation local distance <= source local distance
            xi_obs2 = _linear_map_local_to_parent(obs_interval, uu * vv, obs_start_is_shared)
            xi_src2 = _linear_map_local_to_parent(src_interval, uu, src_start_is_shared)
            phi_obs2 = _linear_shape_values(xi_obs2)
            phi_src2 = _linear_shape_values(xi_src2)
            robs2 = _linear_param_to_point(obs_elem, xi_obs2)
            rsrc2 = _linear_param_to_point(src_elem, xi_src2)
            block += weight * complex(kernel_eval(robs2, rsrc2)) * np.outer(phi_obs2, phi_src2)

    return block * (obs_len * src_len)


def _integrate_linear_touching_duffy_sk_vectorized(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    obs_interval: 'Tuple[float, float]',
    src_interval: 'Tuple[float, float]',
    obs_start_is_shared: 'bool',
    src_start_is_shared: 'bool',
    order: 'int' = 20,
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """Vectorized two-triangle Duffy rule for endpoint-touching panels."""

    if not bool(compute_single_layer) and not bool(compute_double_layer):
        raise ValueError("At least one touching-pair operator must be requested.")

    qt, qw = _get_quadrature(max(4, int(order)))
    obs_len = _linear_interval_length(obs_elem, obs_interval)
    src_len = _linear_interval_length(src_elem, src_interval)
    s_block = np.zeros((2, 2), dtype=np.complex128)
    k_block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return s_block, k_block

    u = np.asarray(qt, dtype=float)[:, None]
    v = np.asarray(qt, dtype=float)[None, :]
    weights = (
        np.asarray(qw, dtype=float)[:, None]
        * u
        * np.asarray(qw, dtype=float)[None, :]
    ).reshape(-1)
    uv = u * v

    def map_many(interval, local, start_is_shared):
        a, b = float(interval[0]), float(interval[1])
        h = b - a
        return (
            a + h * local
            if start_is_shared
            else b - h * local
        )

    # Match the scalar routine's summation domain exactly: triangle 1 uses
    # (obs=u, src=u*v), triangle 2 uses (obs=u*v, src=u).
    xi_obs = np.concatenate((
        np.broadcast_to(
            map_many(obs_interval, u, obs_start_is_shared), uv.shape
        ).reshape(-1),
        map_many(obs_interval, uv, obs_start_is_shared).reshape(-1),
    ))
    xi_src = np.concatenate((
        map_many(src_interval, uv, src_start_is_shared).reshape(-1),
        np.broadcast_to(
            map_many(src_interval, u, src_start_is_shared), uv.shape
        ).reshape(-1),
    ))
    weights = np.concatenate((weights, weights))

    phi_obs = np.column_stack((1.0 - xi_obs, xi_obs))
    phi_src = np.column_stack((1.0 - xi_src, xi_src))
    obs_seg = obs_elem.p1 - obs_elem.p0
    src_seg = src_elem.p1 - src_elem.p0
    robs = obs_elem.p0[None, :] + xi_obs[:, None] * obs_seg[None, :]
    rsrc = src_elem.p0[None, :] + xi_src[:, None] * src_seg[None, :]
    diff = robs - rsrc

    if compute_single_layer:
        g_vals = _green_2d_array(k0, np.linalg.norm(diff, axis=1))
        s_block = np.einsum(
            'q,q,qa,qb->ab', weights, g_vals, phi_obs, phi_src
        )

    if compute_double_layer:
        if obs_normal_deriv:
            dk_vals = _dgreen_dn_obs_array(k0, diff, obs_elem.normal)
        else:
            src_normals = np.broadcast_to(src_elem.normal, diff.shape)
            dk_vals = _dgreen_dn_src_array(k0, diff, src_normals)
        k_block = np.einsum(
            'q,q,qa,qb->ab', weights, dk_vals, phi_obs, phi_src
        )

    scale = obs_len * src_len
    return s_block * scale, k_block * scale


def _integrate_linear_pair_recursive(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    kernel_eval: 'Callable[[np.ndarray, np.ndarray], complex]',
    obs_interval: 'Tuple[float, float]',
    src_interval: 'Tuple[float, float]',
    obs_order: 'int',
    src_order: 'int',
    depth: 'int' = 0,
    max_depth: 'int' = 3,
) -> 'np.ndarray':
    obs_len = _linear_interval_length(obs_elem, obs_interval)
    src_len = _linear_interval_length(src_elem, src_interval)
    block = np.zeros((2, 2), dtype=np.complex128)
    if obs_len <= 0.0 or src_len <= 0.0:
        return block

    same_elem_same_interval = (
        obs_elem.panel_index == src_elem.panel_index
        and abs(float(obs_interval[0]) - float(src_interval[0])) <= 1.0e-15
        and abs(float(obs_interval[1]) - float(src_interval[1])) <= 1.0e-15
    )
    if same_elem_same_interval:
        order = max(6, int(max(obs_order, src_order)) + 1)
        return _integrate_linear_self_duffy(
            obs_elem,
            kernel_eval,
            interval=obs_interval,
            order=order,
        )

    shared = _linear_shared_interval_endpoint_info(obs_elem, obs_interval, src_elem, src_interval)
    if shared is not None:
        order = max(6, int(max(obs_order, src_order)) + 1)
        return _integrate_linear_touching_duffy(
            obs_elem,
            src_elem,
            kernel_eval,
            obs_interval=obs_interval,
            src_interval=src_interval,
            obs_start_is_shared=bool(shared[0]),
            src_start_is_shared=bool(shared[1]),
            order=order,
        )

    obs_mid = _linear_interval_midpoint(obs_elem, obs_interval)
    src_mid = _linear_interval_midpoint(src_elem, src_interval)
    distance = float(np.linalg.norm(obs_mid - src_mid))
    scale = max(obs_len, src_len, EPS)
    ratio = distance / scale

    # Refine near-singular element pairs adaptively before falling back to tensor Gauss.
    if depth < max_depth and ratio < 0.95:
        oa, ob = float(obs_interval[0]), float(obs_interval[1])
        sa, sb = float(src_interval[0]), float(src_interval[1])
        if ratio < 0.16:
            om = 0.5 * (oa + ob)
            sm = 0.5 * (sa + sb)
            sub_obs = [(oa, om), (om, ob)]
            sub_src = [(sa, sm), (sm, sb)]
            for oi in sub_obs:
                for si in sub_src:
                    block += _integrate_linear_pair_recursive(
                        obs_elem,
                        src_elem,
                        kernel_eval,
                        oi,
                        si,
                        obs_order=obs_order,
                        src_order=src_order,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
            return block
        if obs_len >= src_len:
            om = 0.5 * (oa + ob)
            return (
                _integrate_linear_pair_recursive(
                    obs_elem, src_elem, kernel_eval, (oa, om), src_interval,
                    obs_order=obs_order, src_order=src_order, depth=depth + 1, max_depth=max_depth,
                )
                + _integrate_linear_pair_recursive(
                    obs_elem, src_elem, kernel_eval, (om, ob), src_interval,
                    obs_order=obs_order, src_order=src_order, depth=depth + 1, max_depth=max_depth,
                )
            )
        sm = 0.5 * (sa + sb)
        return (
            _integrate_linear_pair_recursive(
                obs_elem, src_elem, kernel_eval, obs_interval, (sa, sm),
                obs_order=obs_order, src_order=src_order, depth=depth + 1, max_depth=max_depth,
            )
            + _integrate_linear_pair_recursive(
                obs_elem, src_elem, kernel_eval, obs_interval, (sm, sb),
                obs_order=obs_order, src_order=src_order, depth=depth + 1, max_depth=max_depth,
            )
        )

    adapt_order, _ = _near_singular_scheme(distance, scale)
    tensor_order = max(int(max(obs_order, src_order)), min(16, int(max(5, adapt_order))))
    return _integrate_linear_pair_box(
        obs_elem,
        src_elem,
        kernel_eval,
        obs_interval=obs_interval,
        src_interval=src_interval,
        obs_order=tensor_order,
        src_order=tensor_order,
    )

def _integrate_linear_pair_generic(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    kernel_eval: 'Callable[[np.ndarray, np.ndarray], complex]',
    obs_order: 'int' = 6,
    src_order: 'int' = 6,
) -> 'np.ndarray':
    """
    Assemble a 2x2 Galerkin block for one observation/source element pair.

    This upgraded implementation keeps the straight-element tensor-Gauss backbone but
    adds two accuracy-critical improvements for the experimental linear/Galerkin path:
    - Duffy-type quadrature for same-element and endpoint-touching singular pairs
    - adaptive recursive interval subdivision for near-singular pairs
    """

    return _integrate_linear_pair_recursive(
        obs_elem,
        src_elem,
        kernel_eval,
        obs_interval=(0.0, 1.0),
        src_interval=(0.0, 1.0),
        obs_order=obs_order,
        src_order=src_order,
        depth=0,
        max_depth=6,
    )

def _stable_hankel2_array(order: 'int', x: 'np.ndarray') -> 'np.ndarray':
    """Robust array Hankel evaluator for real and complex arguments.

    Uses scaled SciPy Hankel for complex arguments when available, then repairs
    any remaining non-finite entries with the existing scalar helpers.
    """

    z = np.asarray(x, dtype=np.complex128)
    out: 'Optional[np.ndarray]' = None
    if _SCIPY_SPECIAL is not None:
        try:
            # Real fast path when possible.
            if np.all(np.abs(z.imag) <= 1e-14) and np.all(z.real >= 0.0):
                xr = np.maximum(z.real.astype(float, copy=False), 1e-12)
                if order == 0:
                    out = np.asarray(_SCIPY_SPECIAL.j0(xr) - 1j * _SCIPY_SPECIAL.y0(xr), dtype=np.complex128)
                else:
                    out = np.asarray(_SCIPY_SPECIAL.j1(xr) - 1j * _SCIPY_SPECIAL.y1(xr), dtype=np.complex128)
            elif hasattr(_SCIPY_SPECIAL, 'hankel2e'):
                scaled = np.asarray(_SCIPY_SPECIAL.hankel2e(order, z), dtype=np.complex128)
                out = scaled * np.exp(-1j * z)
            else:
                out = np.asarray(_SCIPY_SPECIAL.hankel2(order, z), dtype=np.complex128)
        except Exception:
            out = None
    if out is None:
        vec = np.vectorize(_hankel2_0 if order == 0 else _hankel2_1, otypes=[np.complex128])
        return np.asarray(vec(z), dtype=np.complex128)

    finite = np.isfinite(out.real) & np.isfinite(out.imag)
    if not np.all(finite):
        vec = np.vectorize(_hankel2_0 if order == 0 else _hankel2_1, otypes=[np.complex128])
        repaired = np.asarray(vec(z[~finite]), dtype=np.complex128)
        out = np.asarray(out, dtype=np.complex128)
        out[~finite] = repaired
    return np.asarray(out, dtype=np.complex128)

def _hankel2_0_array(x: 'np.ndarray') -> 'np.ndarray':
    return _stable_hankel2_array(0, x)

def _hankel2_1_array(x: 'np.ndarray') -> 'np.ndarray':
    return _stable_hankel2_array(1, x)

def _green_2d_array(k0: 'Union[complex, float]', r: 'np.ndarray') -> 'np.ndarray':
    rr = np.maximum(np.asarray(r, dtype=float), EPS)
    x = np.asarray(complex(k0) * rr, dtype=np.complex128)
    x[np.abs(x) <= 1e-12] = 1e-12 + 0.0j
    return 0.25j * _hankel2_0_array(x)

def _dgreen_dn_obs_array(k0: 'Union[complex, float]', r_vec: 'np.ndarray', n_obs: 'np.ndarray') -> 'np.ndarray':
    rr = np.linalg.norm(r_vec, axis=1)
    out = np.zeros(rr.shape[0], dtype=np.complex128)
    mask = rr > EPS
    if not np.any(mask):
        return out
    rrm = rr[mask]
    x = np.asarray(complex(k0) * rrm, dtype=np.complex128)
    x[np.abs(x) <= 1e-12] = 1e-12 + 0.0j
    h1 = _hankel2_1_array(x)
    projection = (r_vec[mask] @ np.asarray(n_obs, dtype=float)) / rrm
    out[mask] = (-0.25j * complex(k0)) * h1 * projection
    return out

def _dgreen_dn_src_array(k0: 'Union[complex, float]', r_vec: 'np.ndarray', n_src: 'np.ndarray') -> 'np.ndarray':
    rr = np.linalg.norm(r_vec, axis=1)
    out = np.zeros(rr.shape[0], dtype=np.complex128)
    mask = rr > EPS
    if not np.any(mask):
        return out
    rrm = rr[mask]
    x = np.asarray(complex(k0) * rrm, dtype=np.complex128)
    x[np.abs(x) <= 1e-12] = 1e-12 + 0.0j
    h1 = _hankel2_1_array(x)
    projection = np.sum(np.asarray(n_src, dtype=float)[mask] * r_vec[mask], axis=1) / rrm
    out[mask] = (0.25j * complex(k0)) * h1 * projection
    return out



def _single_layer_block_linear(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    k0: 'Union[complex, float]',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
) -> 'np.ndarray':
    if obs_elem.panel_index == src_elem.panel_index:
        exact = _single_layer_self_block_exact(obs_elem, k0)
        if exact is not None:
            return exact
    return _integrate_linear_pair_generic(
        obs_elem,
        src_elem,
        lambda robs, rsrc: _green_2d(k0, max(float(np.linalg.norm(robs - rsrc)), EPS)),
        obs_order=obs_order,
        src_order=src_order,
    )


def _sk_blocks_near_linear(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """
    Compute S and K 2x2 blocks for a near element pair.

    Uses Duffy transforms for self and touching pairs (via the existing recursive
    path), and the vectorized tensor-Gauss path for separated-near pairs.
    """

    same_elem = obs_elem.panel_index == src_elem.panel_index
    # Singularity classification is geometric, not algebraic.  The
    # interface-aware mesh deliberately gives coincident endpoints distinct
    # node IDs when their boundary/material signatures differ.  Such panels
    # still share a physical endpoint and require touching-pair Duffy
    # quadrature; treating them as separated makes tensor subdivision chase
    # the endpoint singularity until max_depth without converging.
    shared = (
        None
        if same_elem
        else _linear_shared_interval_endpoint_info(
            obs_elem, (0.0, 1.0), src_elem, (0.0, 1.0), tol=1.0e-9
        )
    )

    zero = np.zeros((2, 2), dtype=np.complex128)
    if same_elem:
        # A straight panel's normal derivative of G against itself is exactly
        # zero: every source-observation displacement is tangential.  The jump
        # term is represented separately by the mass matrix, so no principal-
        # value work belongs in K/K' here.
        s_blk = (
            _single_layer_block_linear(
                obs_elem, src_elem, k0, obs_order, src_order
            ) if compute_single_layer else zero
        )
        return s_blk, zero

    if shared is not None:
        order = max(6, int(max(obs_order, src_order)) + 1)
        return _integrate_linear_touching_duffy_sk_vectorized(
            obs_elem=obs_elem,
            src_elem=src_elem,
            k0=k0,
            obs_normal_deriv=obs_normal_deriv,
            obs_interval=(0.0, 1.0),
            src_interval=(0.0, 1.0),
            obs_start_is_shared=bool(shared[0]),
            src_start_is_shared=bool(shared[1]),
            order=order,
            compute_single_layer=compute_single_layer,
            compute_double_layer=compute_double_layer,
        )

    # Separated-near pairs: a fixed tensor rule is not sufficient when two
    # panels are almost parallel and their gap is small compared with their
    # length.  In that limit the logarithmic S kernel (and the sharply peaked
    # normal-derivative kernel) is concentrated in a narrow band around the
    # projected diagonal.  Increasing the rule from 8 to 16 points without
    # subdividing still leaves percent-to-order-one block errors for
    # gap/length below a few percent.
    obs_mid = obs_elem.center
    src_mid = src_elem.center
    distance = float(np.linalg.norm(obs_mid - src_mid))
    scale = max(obs_elem.length, src_elem.length, EPS)
    adapt_order, _ = _near_singular_scheme(distance, scale)
    tensor_order = max(int(max(obs_order, src_order)), min(16, int(max(5, adapt_order))))

    if distance / scale < 0.75:
        return _integrate_linear_pair_adaptive_sk(
            obs_elem=obs_elem,
            src_elem=src_elem,
            k0=k0,
            obs_normal_deriv=obs_normal_deriv,
            obs_order=tensor_order,
            src_order=tensor_order,
            compute_single_layer=compute_single_layer,
            compute_double_layer=compute_double_layer,
        )

    return _integrate_linear_pair_box_sk_vectorized(
        obs_elem, src_elem, k0, obs_normal_deriv,
        obs_interval=(0.0, 1.0), src_interval=(0.0, 1.0),
        obs_order=tensor_order, src_order=tensor_order,
        compute_single_layer=compute_single_layer,
        compute_double_layer=compute_double_layer,
    )


NEAR_PAIR_QUADRATURE_RTOL = 1.0e-9
NEAR_PAIR_QUADRATURE_MAX_DEPTH = 12


def _integrate_linear_pair_adaptive_sk(
    obs_elem: 'LinearElement',
    src_elem: 'LinearElement',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    obs_order: 'int',
    src_order: 'int',
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
    rtol: 'float' = NEAR_PAIR_QUADRATURE_RTOL,
    max_depth: 'int' = NEAR_PAIR_QUADRATURE_MAX_DEPTH,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """Converged S/K quadrature for separated, nearly singular panel pairs.

    Each box is compared with its four-way bisection.  Only children whose
    parent comparison has not converged are refined further, so a narrow
    diagonal interaction costs O(2**depth), rather than uniformly applying a
    very high tensor rule to the whole pair.  A child block already computed
    for its parent's error estimate is reused as its own coarse estimate.

    The error denominator is the sum of child-block norms, rather than the
    norm of their possibly cancelling sum.  This prevents a physical null in
    one 2x2 block from making the convergence test spuriously permissive.
    Failure at ``max_depth`` is explicit: silently accepting an unresolved
    close-gap interaction can produce a small linear-system residual for the
    wrong discrete operator.
    """

    zero = np.zeros((2, 2), dtype=np.complex128)

    def evaluate(
        obs_interval: 'Tuple[float, float]',
        src_interval: 'Tuple[float, float]',
    ) -> 'Tuple[np.ndarray, np.ndarray]':
        return _integrate_linear_pair_box_sk_vectorized(
            obs_elem=obs_elem,
            src_elem=src_elem,
            k0=k0,
            obs_normal_deriv=obs_normal_deriv,
            obs_interval=obs_interval,
            src_interval=src_interval,
            obs_order=obs_order,
            src_order=src_order,
            compute_single_layer=compute_single_layer,
            compute_double_layer=compute_double_layer,
        )

    def relative_error(
        coarse: 'np.ndarray',
        children: 'List[np.ndarray]',
    ) -> 'float':
        # Do not pass ``start`` by keyword to the built-in sum.  Some Python
        # versions deployed on HPC systems expose the second argument only as
        # positional and raise ``TypeError: sum() takes no keyword arguments``
        # when a close panel pair reaches this adaptive path.
        fine = sum(children, zero.copy())
        scale_norm = sum(float(np.linalg.norm(block)) for block in children)
        floor = np.finfo(float).eps * max(
            1.0,
            float(obs_elem.length) * float(src_elem.length),
        )
        return float(np.linalg.norm(fine - coarse)) / max(scale_norm, floor)

    def recurse(
        obs_interval: 'Tuple[float, float]',
        src_interval: 'Tuple[float, float]',
        depth: 'int',
        coarse: 'Optional[Tuple[np.ndarray, np.ndarray]]' = None,
    ) -> 'Tuple[np.ndarray, np.ndarray]':
        coarse_s, coarse_k = coarse if coarse is not None else evaluate(
            obs_interval, src_interval
        )
        oa, ob = map(float, obs_interval)
        sa, sb = map(float, src_interval)
        om = 0.5 * (oa + ob)
        sm = 0.5 * (sa + sb)
        child_intervals = [
            ((oa, om), (sa, sm)),
            ((oa, om), (sm, sb)),
            ((om, ob), (sa, sm)),
            ((om, ob), (sm, sb)),
        ]
        child_blocks = [evaluate(oi, si) for oi, si in child_intervals]
        s_children = [block[0] for block in child_blocks]
        k_children = [block[1] for block in child_blocks]
        err_s = (
            relative_error(coarse_s, s_children)
            if compute_single_layer else 0.0
        )
        err_k = (
            relative_error(coarse_k, k_children)
            if compute_double_layer else 0.0
        )
        error = max(err_s, err_k)
        if error <= float(rtol):
            return (
                sum(s_children, zero.copy()),
                sum(k_children, zero.copy()),
            )
        if depth >= int(max_depth):
            gap_ratio = float(np.linalg.norm(obs_elem.center - src_elem.center)) / max(
                float(obs_elem.length), float(src_elem.length), EPS
            )
            raise FloatingPointError(
                "Separated-near Galerkin quadrature did not converge: "
                f"panel pair ({obs_elem.panel_index}, {src_elem.panel_index}), "
                f"center-gap/length={gap_ratio:.6g}, estimated relative block "
                f"error={error:.3e} after depth {depth}. Refine the boundary "
                "mesh or increase NEAR_PAIR_QUADRATURE_MAX_DEPTH."
            )

        s_total = zero.copy()
        k_total = zero.copy()
        for (oi, si), child in zip(child_intervals, child_blocks):
            child_s, child_k = recurse(
                oi, si, depth + 1, coarse=child
            )
            s_total += child_s
            k_total += child_k
        return s_total, k_total

    return recurse((0.0, 1.0), (0.0, 1.0), depth=0)

_TANGENT_OUTER = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.complex128)

def _hypersingular_block_from_s_block(
    s_block: 'np.ndarray',
    k0: 'Union[complex, float]',
    n_obs: 'np.ndarray',
    n_src: 'np.ndarray',
    obs_length: 'float',
    src_length: 'float',
) -> 'np.ndarray':
    """
    Compute the 2x2 hypersingular D block from the single-layer S block via Maue identity.

    The Maue regularisation recasts the hypersingular kernel integral as:
        D_ij = -k^2 (n_obs . n_src) S_ij
             + (1/(L_obs*L_src)) * tangent_outer_ij * sum(S_block)

    where tangent_outer = [[1,-1],[-1,1]] encodes the linear shape-function
    tangential derivatives.  This avoids all hypersingular quadrature.
    """

    k2 = complex(k0) ** 2
    n_dot_n = float(np.dot(n_obs, n_src))
    raw_integral = complex(np.sum(s_block))
    denom = max(float(obs_length) * float(src_length), EPS * EPS)
    return -k2 * n_dot_n * s_block + _TANGENT_OUTER * (raw_integral / denom)

# --- Batched far-pair quadrature engine -------------------------------------
#
# Assembling the dense boundary-integral operators is the whole cost of a large
# 2-D solve: the linear solve is O(N^3) with threaded BLAS behind it, while the
# assembly is O(N^2 Q^2) evaluations of a Hankel function.  The engine below
# keeps the mathematics of the original element-pair quadrature intact -- same
# quadrature rule, same far/near split, same near-pair Duffy treatment -- and
# only changes how the work is staged:
#
#   * Element pairs are processed in cache-sized tiles instead of as whole
#     N_elem x N_elem arrays.  The previous formulation held eight complex
#     N_elem^2 accumulators plus roughly six N_elem^2 temporaries live at once
#     (3.2 GB + 1.2 GB at 5 000 elements).  That capped how many solves fit on
#     a node far more tightly than the matrices themselves do, and made every
#     quadrature point a DRAM round trip.
#   * The kernel is evaluated once per unordered element pair.  G is symmetric
#     and the Galerkin block of the transposed pair is the transpose of the
#     block already computed, so the strictly-upper tiles cover everything.
#     The double-layer kernels are not symmetric, but they share the same
#     H_1^(2)(kr), which is what the time actually goes into -- so the
#     transposed projection is formed from the same Bessel evaluation.
#     This halves the dominant cost outright.
#   * The Hankel evaluation takes a real-argument fast path.  G = (j/4)H_0^(2)
#     and its normal derivative have closed real/imaginary parts in terms of
#     J_n/Y_n, so a real wavenumber needs no complex argument array and none of
#     the all-finite / all-real guard scans `_stable_hankel2_array` must run to
#     stay safe for arbitrary complex input.
#   * The far mask is applied once to the finished tile accumulators instead of
#     to every kernel evaluation.  Masked-out entries stay finite (distances
#     are floored at EPS), so accumulating and then discarding them is exact.
#
# Tile size and thread count are tunable because the right values depend on how
# the node is packed: one solve per core wants small tiles because the L3 slice
# is shared, while a handful of large solves wants big tiles and threads.

_ASSEMBLY_TILE_TARGET_BYTES = 24 * 1024 * 1024


def _env_positive_int(name: 'str', default: 'int') -> 'int':
    try:
        value = int(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


_ASSEMBLY_THREADS = _env_positive_int("GHOST_ASSEMBLY_THREADS", 1)
_ASSEMBLY_TILE = _env_positive_int("GHOST_ASSEMBLY_TILE", 0)

# Opt-in override for the FAR single-/double-layer quadrature order only.
#
# Roughly half of a large assembly is Hankel evaluations, and their count is
# exactly N_elem^2 * Q^2.  The default Q = 8 is inherited from the near-field
# rule, where it is needed; across a far pair -- separated by at least
# `far_ratio` element lengths, so with a smooth non-singular integrand -- it is
# heavily over-resolved, and a lower order costs (Q_new/8)^2 of the time.
#
# This is left OFF by default because it changes computed values: it is a
# different quadrature rule, not a faster evaluation of the same one.  Turning
# it on is a legitimate engineering trade, but it has to be validated for the
# geometries in question -- run tests/measure_far_quadrature.py, which reports
# the RCS shift against the default rule, and keep the mesh-convergence gate
# in the loop.  A solve that used the override says so in its warnings, so a
# published field always carries the fact.
_FAR_QUAD_ORDER = _env_positive_int("GHOST_FAR_QUAD_ORDER", 0)

# Compact the source axis when the active masks select less than this fraction
# of the mesh.  Compaction and the transposed-pair shortcut are mutually
# exclusive (one needs the axes to differ, the other needs them to match), and
# they break even at one half.  Settable to 0 to force the full-width path,
# which must produce identical matrices -- tests/test_assembly_equivalence.py
# checks exactly that.
_ASSEMBLY_COMPACT_BELOW = 0.5


def set_assembly_compaction(fraction: 'float') -> 'None':
    """Set the active-fraction threshold below which the source axis compacts."""

    global _ASSEMBLY_COMPACT_BELOW
    _ASSEMBLY_COMPACT_BELOW = float(fraction)


def set_far_quadrature_order(order: 'int') -> 'None':
    """Override the far-pair quadrature order (0 restores the default rule)."""

    global _FAR_QUAD_ORDER
    _FAR_QUAD_ORDER = max(0, int(order))


def get_far_quadrature_order() -> 'int':
    """Active far-pair quadrature override, or 0 when the default rule is used."""

    return int(_FAR_QUAD_ORDER)


def set_assembly_threads(count: 'int') -> 'None':
    """
    Set how many threads tiled operator assembly may use (1 = serial).

    Assembly tiles are independent and the heavy numpy/SciPy ufuncs inside them
    release the GIL, so this scales usefully when a node has more cores than
    concurrent solves.  When a run has at least one unit per core, leave it at
    1 and let the process pool own the parallelism -- threads and processes
    competing for the same cores is strictly worse than either alone.
    """

    global _ASSEMBLY_THREADS
    _ASSEMBLY_THREADS = max(1, int(count))


def get_assembly_threads() -> 'int':
    """Current tiled-assembly thread count."""

    return int(_ASSEMBLY_THREADS)


def _assembly_tile_size(nelems: 'int', bytes_per_entry: 'int') -> 'int':
    """Pick an element-tile edge so one tile's working set stays cache-sized.

    When assembly threads are enabled the tile is also capped so there are
    several observation blocks per thread: blocks are the unit of parallelism,
    and the symmetric traversal makes the first block the most expensive (it
    pairs with every later one), so a handful of coarse blocks would both
    starve threads and hand them wildly unequal work.
    """

    if _ASSEMBLY_TILE > 0:
        return max(1, min(int(nelems), _ASSEMBLY_TILE))
    if nelems <= 192:
        return int(nelems)
    entries = float(_ASSEMBLY_TILE_TARGET_BYTES) / float(max(1, int(bytes_per_entry)))
    tile = int(math.sqrt(max(1.0, entries)))
    tile = max(128, min(1024, min(int(nelems), tile)))
    if _ASSEMBLY_THREADS > 1:
        per_thread_blocks = 4
        tile = min(
            tile,
            max(64, int(math.ceil(nelems / (per_thread_blocks * _ASSEMBLY_THREADS)))),
        )
    return max(1, min(int(nelems), tile))


# Distance- and wavelength-graded far quadrature.
#
# The far pass used a fixed order (8) for every well-separated pair.  Measuring
# the order actually needed to hold a Galerkin block to 1e-12 relative --
# against an order-24 reference, worst case over collinear, parallel,
# perpendicular, and oblique pair orientations -- gives:
#
#     kL \ r      3    4    5    6    8   10   14   20   30   50
#     0.10        6    6    5    5    5    5    4    4    4    4
#     0.31        6    6    5    5    5    5    5    5    5    5
#     1.00        7    6    6    6    6    6    6    6    6    6
#     2.00        7    7    7    7    7    7    7    7    7    7
#     4.00        9    9    9    9    9    9    9    9    9    9
#
# where r is the centre separation in element lengths and kL is the element
# length in radians.  Two things fall out.  Separation barely matters once a
# pair is far at all -- what sets the order is how many wavelengths the element
# spans, because that is what the integrand oscillates over.  And at kL >= 4
# the fixed order 8 is *under*-resolved: a mesh that coarse needs 9.
#
# The table below carries a +1 margin over those measurements, and the result
# is clamped to the caller's configured order, so grading only ever reduces
# work.  Raising the order past what the caller asked for would change
# published values on coarse meshes -- worth doing, but not silently.
#
# The order is chosen per tile from the tile's *worst* pair, so it costs two
# reductions per tile rather than a branch per element pair.

# Each row is calibrated at its band's UPPER kL edge -- the worst case inside
# the band -- over five pair orientations, so the band itself is the margin.
# The r < 5 column carries one extra order as a hedge: centre separation can
# understate how close two elements actually come when a mesh is irregular.
_FAR_ORDER_TABLE = (
    # (kL upper bound, order for r < 5, for r < 10, for r >= 10)
    (0.15, 6, 5, 5),
    (0.50, 7, 6, 5),
    (1.50, 7, 6, 6),
    (3.00, 8, 8, 8),
    (float("inf"), 10, 10, 10),
)

_FAR_GRADED = _env_positive_int("GHOST_FAR_GRADED", 1) != 0


def set_far_quadrature_grading(enabled: 'bool') -> 'None':
    """Enable/disable per-tile far-quadrature grading (default on).

    Grading only reduces the order below what the caller configured, and only
    where a calibrated table says the reduction costs under 1e-12 relative on
    the element-pair block, so turning it off should change nothing that
    matters.  The switch exists to make that testable.
    """

    global _FAR_GRADED
    _FAR_GRADED = bool(enabled)


def _graded_far_order(kl_max: 'float', ratio_min: 'float', cap: 'int') -> 'int':
    """Quadrature order for a tile whose worst far pair has these parameters."""

    if not _FAR_GRADED:
        return int(cap)
    for bound, near, mid, far in _FAR_ORDER_TABLE:
        if kl_max <= bound:
            if ratio_min < 5.0:
                order = near
            elif ratio_min < 10.0:
                order = mid
            else:
                order = far
            return max(2, min(int(cap), int(order)))
    return int(cap)


def _wavenumber_is_real(k0: 'Union[complex, float]') -> 'bool':
    value = complex(k0)
    return value.imag == 0.0 and value.real > 0.0


def _axpy_into(acc: 'np.ndarray', src: 'np.ndarray', coeff: 'float',
               scratch: 'np.ndarray') -> 'None':
    """acc += coeff * src, in place, without allocating a temporary.

    Deliberately not scipy's BLAS axpy: its f2py wrapper carries a fixed
    per-call cost of several milliseconds, which swamps a cache-sized tile and
    only pays for itself on arrays of millions of elements.
    """

    np.multiply(src, coeff, out=scratch)
    np.add(acc, scratch, out=acc)


def _far_kernel_argument(k0: 'Union[complex, float]', dist: 'np.ndarray',
                         out: 'np.ndarray') -> 'None':
    """kr = k0 * dist on the real fast path, floored where the scalar
    evaluators floor it so the two agree entry for entry."""

    np.multiply(dist, complex(k0).real, out=out)
    np.maximum(out, 1e-12, out=out)


def _far_green_into(
    k0: 'Union[complex, float]',
    real_k: 'bool',
    dist: 'np.ndarray',
    kr: 'np.ndarray',
    scratch: 'np.ndarray',
    out: 'np.ndarray',
) -> 'None':
    """out <- (j/4) H_0^(2)(k0 r) over a whole tile.

    For real k0 this is (1/4)(Y_0(kr) + j J_0(kr)), so the two real Bessel
    evaluations write straight into the halves of the complex output buffer.
    """

    if real_k and _SCIPY_SPECIAL is not None:
        _SCIPY_SPECIAL.y0(kr, out=scratch)
        np.multiply(scratch, 0.25, out=out.real)
        _SCIPY_SPECIAL.j0(kr, out=scratch)
        np.multiply(scratch, 0.25, out=out.imag)
        return
    arg = np.asarray(complex(k0) * dist, dtype=np.complex128)
    np.multiply(_hankel2_0_array(arg), 0.25j, out=out)


def _far_hankel1_into(
    k0: 'Union[complex, float]',
    real_k: 'bool',
    dist: 'np.ndarray',
    kr: 'np.ndarray',
    scratch: 'np.ndarray',
    out: 'np.ndarray',
) -> 'None':
    """out <- (j/4) k0 H_1^(2)(k0 r) over a whole tile.

    Kept separate from the projection so both orientations of an element pair
    can reuse one Bessel evaluation -- the normal-derivative kernels differ
    only by which normal the displacement is projected onto.
    """

    if real_k and _SCIPY_SPECIAL is not None:
        coeff = 0.25 * complex(k0).real
        _SCIPY_SPECIAL.y1(kr, out=scratch)
        np.multiply(scratch, coeff, out=out.real)
        _SCIPY_SPECIAL.j1(kr, out=scratch)
        np.multiply(scratch, coeff, out=out.imag)
        return
    arg = np.asarray(complex(k0) * dist, dtype=np.complex128)
    np.multiply(_hankel2_1_array(arg), 0.25j * complex(k0), out=out)


def _run_tiled_obs_blocks(
    nelems: 'int',
    tile: 'int',
    body: 'Callable[[int, int], None]',
) -> 'None':
    """Run ``body(i0, i1)`` over every observation tile, threaded when asked."""

    starts = list(range(0, nelems, tile))
    workers = min(int(_ASSEMBLY_THREADS), len(starts))
    if workers <= 1:
        for i0 in starts:
            body(i0, min(i0 + tile, nelems))
        return
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda i0: body(i0, min(i0 + tile, nelems)), starts))


def _expand_near_chunks(
    chunks: 'List[Tuple[int, np.ndarray, int, np.ndarray, bool]]',
) -> 'Tuple[np.ndarray, np.ndarray]':
    """Flatten recorded near-pair tiles into ascending (obs, src) index arrays.

    Each chunk carries the tile's global source-element ids, because a masked
    assembly compacts the source axis and the recorded column is an index into
    that compacted axis rather than into the mesh.

    Sorting here is what keeps the near-field accumulation order independent of
    how tiles were scheduled, so a threaded assembly reproduces a serial one
    exactly rather than only to within floating-point reassociation.
    """

    if not chunks:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    obs_parts: 'List[np.ndarray]' = []
    src_parts: 'List[np.ndarray]' = []
    for row_base, src_global, ncols, flat, transposed in chunks:
        rows = flat // ncols
        cols = src_global[flat % ncols]
        if transposed:
            obs_parts.append(cols)
            src_parts.append(row_base + rows)
        else:
            obs_parts.append(row_base + rows)
            src_parts.append(cols)
    obs_idx = np.concatenate(obs_parts)
    src_idx = np.concatenate(src_parts)
    order = np.lexsort((src_idx, obs_idx))
    return obs_idx[order], src_idx[order]


def _warn_far_quadrature_override(materials: 'MaterialLibrary') -> 'None':
    """Record a far-quadrature override in the solve's own warnings.

    The override changes computed values, so a field produced under it must
    say so wherever it travels; the warning list is copied into the .grim
    audit, which is the record that survives the run directory.
    """

    if _FAR_QUAD_ORDER <= 0:
        return
    materials.warn_once(
        f"Far-pair quadrature order overridden to {_FAR_QUAD_ORDER} "
        "(default 8) via GHOST_FAR_QUAD_ORDER / set_far_quadrature_order. "
        "Well-separated element pairs are integrated with a coarser rule "
        "than the shipped default, so these values are NOT bit-comparable "
        "with default-rule results. The mesh-convergence certificate still "
        "applies to the discretization, not to this quadrature choice."
    )


def _assemble_linear_operator_matrices_multi(
    mesh: 'LinearMesh',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    source_element_masks: 'Sequence[Optional[np.ndarray]]',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    far_ratio: 'float' = 3.0,
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
    single_layer_observation_coefficients: 'Optional[np.ndarray]' = None,
) -> 'List[Tuple[np.ndarray, np.ndarray]]':
    """
    Assemble S and K/K' for several source-element masks in ONE traversal.

    The masks select which source elements contribute, but they are applied to
    the finished tile accumulators -- the quadrature itself does not depend on
    them.  Assembling each mask separately therefore repeats every Hankel
    evaluation, which is most of the cost of a solve.  The multi-region
    formulation asks for exactly this: one operator per (region, interface
    side), where both sides of a region share a wavenumber and differ only in
    which elements are active.

    Returns one (S, K) pair per mask, in the order given.  A mask selecting no
    element gets zero matrices without costing anything.  When
    ``single_layer_observation_coefficients`` is supplied, the returned S
    matrix is the Galerkin operator with the piecewise-constant element
    coefficient inside the observation integral,

        S_c[i,j] = sum_e integral_e phi_i(x) c_e (S phi_j)(x) ds,

    while K/K' remains unweighted.  This is required for spatially varying
    Robin and sheet coefficients: multiplying completed rows by a nodal
    average is not the same weak form.

    Two passes, as before:
    1. Far interactions: batched numpy quadrature over cache-sized element
       tiles, one kernel evaluation per unordered pair.
    2. Near interactions: per-element-pair recursive/Duffy quadrature.  Masks
       are disjoint in practice, so this pass does not duplicate work either.
    """

    if not bool(compute_single_layer) and not bool(compute_double_layer):
        raise ValueError("At least one linear operator must be requested.")

    nnodes = len(mesh.nodes)
    elements = list(mesh.elements)
    nelems = len(elements)
    n_masks = len(source_element_masks)
    if n_masks == 0:
        raise ValueError("At least one source-element mask must be requested.")
    # Preserve the long-standing shaped-zero return contract for a skipped
    # operator without allocating and zero-filling another dense N-by-N
    # array.  The zero-stride views are read-only, which is intentional: a
    # disabled output is a result placeholder, never an assembly target.
    zero_view = np.broadcast_to(
        np.zeros((), dtype=np.complex128), (nnodes, nnodes)
    )
    s_mats = [
        np.zeros((nnodes, nnodes), dtype=np.complex128)
        if compute_single_layer else zero_view
        for _ in range(n_masks)
    ]
    k_mats = [
        np.zeros((nnodes, nnodes), dtype=np.complex128)
        if compute_double_layer else zero_view
        for _ in range(n_masks)
    ]
    if not elements:
        return list(zip(s_mats, k_mats))

    src_masks = []
    for mask in source_element_masks:
        if mask is None:
            src_masks.append(np.ones(nelems, dtype=bool))
            continue
        resolved = np.asarray(mask, dtype=bool).reshape(-1)
        if resolved.size != nelems:
            raise ValueError("source_element_mask length must match mesh element count.")
        src_masks.append(resolved)

    if single_layer_observation_coefficients is None:
        slp_obs_coeff = None
    else:
        slp_obs_coeff = np.asarray(
            single_layer_observation_coefficients, dtype=np.complex128
        ).reshape(-1)
        if slp_obs_coeff.size != nelems:
            raise ValueError(
                "single_layer_observation_coefficients length must match "
                "mesh element count."
            )
        if not np.all(
            np.isfinite(slp_obs_coeff.real) & np.isfinite(slp_obs_coeff.imag)
        ):
            raise ValueError(
                "single-layer observation coefficients must all be finite."
            )
    # An empty mask contributes nothing; drop it from the traversal entirely
    # rather than paying a tile sweep to produce zeros.
    active = [index for index, mask in enumerate(src_masks) if bool(np.any(mask))]
    if not active:
        return list(zip(s_mats, k_mats))

    centers = np.stack([e.center for e in elements], axis=0)
    lengths = np.asarray([e.length for e in elements], dtype=float)
    node_ids = np.asarray([e.node_ids for e in elements], dtype=int)  # (nelems, 2)
    p0_arr = np.stack([e.p0 for e in elements], axis=0)               # (nelems, 2)
    seg_arr = np.stack([e.p1 - e.p0 for e in elements], axis=0)       # (nelems, 2)
    normals_arr = np.stack([e.normal for e in elements], axis=0)      # (nelems, 2)

    want_s = bool(compute_single_layer)
    want_k = bool(compute_double_layer)

    # The near pass keeps the requested orders; only the far quadrature honours
    # the override, because that is the only place the integrand is smooth
    # enough for a lower order to be defensible.
    far_obs_order = _FAR_QUAD_ORDER or int(obs_order)
    far_src_order = _FAR_QUAD_ORDER or int(src_order)
    # Grading picks a per-tile order, so the rule is cached rather than
    # precomputed once; tile quadrature points are built from p0/segment inside
    # the tile, which is cheap and avoids holding an (nelems, Q, 2) array per
    # order that might be used.
    _rule_cache: 'Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]' = {}

    def _rule(order: 'int') -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]':
        cached = _rule_cache.get(order)
        if cached is None:
            nodes, weights = _get_quadrature(max(2, int(order)))
            phi = np.array([_linear_shape_values(float(t)) for t in nodes])
            cached = (np.asarray(nodes, dtype=float),
                      np.asarray(weights, dtype=float), phi)
            _rule_cache[order] = cached
        return cached

    # Source-axis compaction.  A mask that selects a minority of the elements
    # -- normal for a coating or a small dielectric region -- otherwise pays a
    # full-width sweep whose result is then masked away.  Restricting the
    # source axis to the union of the active masks makes the work proportional
    # to what is actually wanted.
    #
    # It costs the transposed-pair shortcut, which needs both axes to index the
    # same set, so it is only worth it when the active set is under half the
    # mesh: compacted work is nelems * n_active, symmetric work nelems^2 / 2.
    union_mask = src_masks[active[0]].copy()
    for index in active[1:]:
        union_mask |= src_masks[index]
    n_active = int(np.count_nonzero(union_mask))
    compact = n_active < _ASSEMBLY_COMPACT_BELOW * nelems
    src_sel = (
        np.flatnonzero(union_mask) if compact
        else np.arange(nelems, dtype=np.int64)
    )
    n_src = int(src_sel.size)
    src_centers = centers[src_sel]
    src_lengths = lengths[src_sel]
    src_node_ids = node_ids[src_sel]
    src_normals = normals_arr[src_sel]

    # The transposed-pair shortcut needs both directions to share one
    # quadrature rule, and both axes to index the same elements.
    symmetric = (
        int(max(2, far_obs_order)) == int(max(2, far_src_order))
        and not compact
    )
    abs_k = abs(complex(k0))

    n_acc = 4 * (int(want_s) + (2 if want_k else 0))
    n_kernel = int(want_s) + (2 if want_k else 0)
    tile = _assembly_tile_size(nelems, 16 * (n_acc + n_kernel + 1) + 8 * 7)

    real_k = _wavenumber_is_real(k0)
    # `_dgreen_dn_obs` carries the minus sign; `_dgreen_dn_src` does not.
    dgreen_sign = -1.0 if obs_normal_deriv else 1.0
    # One near-pair list per mask: the masks are disjoint in the formulations
    # that use them, so the near pass costs the same in total as one assembly
    # over their union.
    near_chunks = [[] for _ in range(n_masks)]  # type: List[List[Tuple[int, int, int, np.ndarray, bool]]]
    write_lock = threading.Lock()

    def _far_pass(i0: 'int', i1: 'int') -> 'None':
        mb = i1 - i0
        obs_slice = slice(i0, i1)
        obs_nid = node_ids[obs_slice]
        obs_norm = normals_arr[obs_slice]
        obs_len = lengths[obs_slice]
        obs_p0 = p0_arr[obs_slice]
        obs_seg = seg_arr[obs_slice]
        obs_ctr = centers[obs_slice]
        obs_pts_cache: 'Dict[int, np.ndarray]' = {}
        local_near = [[] for _ in range(n_masks)]

        for j0 in range(i0 if symmetric else 0, n_src, tile):
            j1 = min(j0 + tile, n_src)
            nb = j1 - j0
            mirrored = symmetric and j0 > i0
            src_slice = slice(j0, j1)
            src_global = src_sel[src_slice]
            src_nid = src_node_ids[src_slice]
            src_len = src_lengths[src_slice]
            src_norm = src_normals[src_slice]
            # Orientation-independent part of the far predicate.
            mdx = obs_ctr[:, 0][:, None] - src_centers[src_slice, 0][None, :]
            mdy = obs_ctr[:, 1][:, None] - src_centers[src_slice, 1][None, :]
            centre_dist = np.sqrt(mdx * mdx + mdy * mdy)
            scale = np.maximum(np.maximum(obs_len[:, None], src_len[None, :]), EPS)
            far_sym = (centre_dist / scale) >= float(far_ratio)
            far_sym &= ~(
                (obs_nid[:, 0][:, None] == src_nid[None, :, 0])
                | (obs_nid[:, 0][:, None] == src_nid[None, :, 1])
                | (obs_nid[:, 1][:, None] == src_nid[None, :, 0])
                | (obs_nid[:, 1][:, None] == src_nid[None, :, 1])
            )
            # Self pairs are never far. With a compacted source axis the two
            # axes no longer share an index, so match on the global element id.
            np.logical_and(
                far_sym,
                np.arange(i0, i1)[:, None] != src_global[None, :],
                out=far_sym,
            )

            # Which elements are active is the only mask-dependent term, and
            # the only orientation-dependent one: everything above is shared.
            far_ij = {}
            far_ji = {}
            any_ij = False
            any_ji = False
            for mi in active:
                mask = src_masks[mi]
                src_msk = mask[src_global]
                fij = far_sym & src_msk[None, :]
                far_ij[mi] = fij
                any_ij = any_ij or bool(fij.any())
                near = ~fij
                near &= src_msk[None, :]
                flat = np.flatnonzero(near.ravel())
                if flat.size:
                    local_near[mi].append((i0, src_global, nb, flat, False))
                if mirrored:
                    obs_msk = mask[obs_slice]
                    fji = far_sym & obs_msk[:, None]
                    far_ji[mi] = fji
                    any_ji = any_ji or bool(fji.any())
                    near_t = ~fji
                    near_t &= obs_msk[:, None]
                    flat_t = np.flatnonzero(near_t.ravel())
                    if flat_t.size:
                        local_near[mi].append((i0, src_global, nb, flat_t, True))

            if not (any_ij or any_ji):
                continue

            # Worst far pair in this tile decides the order for all of it.
            any_far = far_sym
            ratio_min = float(np.min(
                np.where(any_far, centre_dist / scale, np.inf)
            )) if any_far.any() else float("inf")
            kl_max = abs_k * float(max(obs_len.max(), src_len.max()))
            tile_order = _graded_far_order(kl_max, ratio_min, far_obs_order)
            t_obs_f, qw_obs, phi_obs_arr = _rule(tile_order)
            t_src_f, qw_src, phi_src_arr = _rule(tile_order)

            obs_pts = obs_pts_cache.get(tile_order)
            if obs_pts is None:
                obs_pts = (obs_p0[:, None, :]
                           + t_obs_f[None, :, None] * obs_seg[:, None, :])
                obs_pts_cache[tile_order] = obs_pts
            src_p0 = p0_arr[src_global]
            src_seg = seg_arr[src_global]
            src_pts = src_p0[:, None, :] + t_src_f[None, :, None] * src_seg[:, None, :]
            acc_s = (
                [np.zeros((mb, nb), dtype=np.complex128) for _ in range(4)]
                if want_s else None
            )
            acc_k = (
                [np.zeros((mb, nb), dtype=np.complex128) for _ in range(4)]
                if want_k else None
            )
            acc_kt = (
                [np.zeros((mb, nb), dtype=np.complex128) for _ in range(4)]
                if (want_k and mirrored) else None
            )
            g_buf = np.empty((mb, nb), dtype=np.complex128) if want_s else None
            h1_buf = np.empty((mb, nb), dtype=np.complex128) if want_k else None
            dk_buf = np.empty((mb, nb), dtype=np.complex128) if want_k else None
            cscratch = np.empty((mb, nb), dtype=np.complex128)
            dx = np.empty((mb, nb), dtype=float)
            dy = np.empty((mb, nb), dtype=float)
            dist = np.empty((mb, nb), dtype=float)
            krbuf = np.empty((mb, nb), dtype=float)
            work = np.empty((mb, nb), dtype=float)
            proj = np.empty((mb, nb), dtype=float) if want_k else None

            # Which normals each orientation projects the displacement onto.
            if obs_normal_deriv:
                n_ij = (obs_norm[:, 0][:, None], obs_norm[:, 1][:, None])
                n_ji = (src_norm[None, :, 0], src_norm[None, :, 1])
            else:
                n_ij = (src_norm[None, :, 0], src_norm[None, :, 1])
                n_ji = (obs_norm[:, 0][:, None], obs_norm[:, 1][:, None])

            for qi in range(t_obs_f.size):
                r_obs = obs_pts[:, qi, :]
                w_obs_qi = float(qw_obs[qi])
                phi_o = phi_obs_arr[qi]

                for qj in range(t_src_f.size):
                    r_src = src_pts[:, qj, :]
                    w_src_qj = float(qw_src[qj])
                    phi_s = phi_src_arr[qj]

                    np.subtract(r_obs[:, 0][:, None], r_src[None, :, 0], out=dx)
                    np.subtract(r_obs[:, 1][:, None], r_src[None, :, 1], out=dy)
                    np.multiply(dx, dx, out=dist)
                    np.multiply(dy, dy, out=work)
                    np.add(dist, work, out=dist)
                    np.sqrt(dist, out=dist)
                    np.maximum(dist, EPS, out=dist)
                    if real_k:
                        _far_kernel_argument(k0, dist, krbuf)

                    if want_s:
                        _far_green_into(k0, real_k, dist, krbuf, work, g_buf)
                    if want_k:
                        _far_hankel1_into(k0, real_k, dist, krbuf, work, h1_buf)

                    w = w_obs_qi * w_src_qj
                    if want_k:
                        np.multiply(dx, n_ij[0], out=proj)
                        np.multiply(dy, n_ij[1], out=work)
                        np.add(proj, work, out=proj)
                        np.divide(proj, dist, out=proj)
                        np.multiply(h1_buf, proj, out=dk_buf)
                        if dgreen_sign < 0.0:
                            np.negative(dk_buf, out=dk_buf)
                    for a in range(2):
                        coeff_a = w * float(phi_o[a])
                        for b in range(2):
                            coeff = coeff_a * float(phi_s[b])
                            if acc_s is not None:
                                _axpy_into(acc_s[2 * a + b], g_buf, coeff, cscratch)
                            if acc_k is not None:
                                _axpy_into(acc_k[2 * a + b], dk_buf, coeff, cscratch)

                    if acc_kt is not None:
                        # Same point pair, roles swapped: the displacement
                        # reverses and the other element's normal is used, so
                        # only the projection is recomputed.
                        np.multiply(dx, n_ji[0], out=proj)
                        np.multiply(dy, n_ji[1], out=work)
                        np.add(proj, work, out=proj)
                        np.divide(proj, dist, out=proj)
                        np.multiply(h1_buf, proj, out=dk_buf)
                        if dgreen_sign > 0.0:
                            np.negative(dk_buf, out=dk_buf)
                        for a in range(2):
                            coeff_a = w * float(phi_s[a])
                            for b in range(2):
                                _axpy_into(
                                    acc_kt[2 * a + b], dk_buf,
                                    coeff_a * float(phi_o[b]), cscratch,
                                )

            len_prod = obs_len[:, None] * src_len[None, :]
            with write_lock:
                for mi in active:
                    fij = far_ij[mi]
                    if fij.any():
                        scale_ij = len_prod * fij
                        if slp_obs_coeff is not None:
                            scale_s_ij = (
                                scale_ij
                                * slp_obs_coeff[obs_slice][:, None]
                            )
                        else:
                            scale_s_ij = scale_ij
                        for a in range(2):
                            rows = obs_nid[:, a][:, None]
                            for b in range(2):
                                cols = src_nid[None, :, b]
                                if acc_s is not None:
                                    np.add.at(s_mats[mi], (rows, cols),
                                              acc_s[2 * a + b] * scale_s_ij)
                                if acc_k is not None:
                                    np.add.at(k_mats[mi], (rows, cols),
                                              acc_k[2 * a + b] * scale_ij)
                    fji = far_ji.get(mi)
                    if fji is not None and fji.any():
                        scale_ji = len_prod * fji
                        if slp_obs_coeff is not None:
                            scale_s_ji = (
                                scale_ji
                                * slp_obs_coeff[src_global][None, :]
                            )
                        else:
                            scale_s_ji = scale_ji
                        for a in range(2):
                            rows = src_nid[:, a][:, None]
                            for b in range(2):
                                cols = obs_nid[None, :, b]
                                if acc_s is not None:
                                    # S is symmetric: block (j,i)[a][b] is the
                                    # transpose of block (i,j)[b][a].
                                    np.add.at(s_mats[mi], (rows, cols),
                                              (acc_s[2 * b + a] * scale_s_ji).T)
                                if acc_kt is not None:
                                    np.add.at(k_mats[mi], (rows, cols),
                                              (acc_kt[2 * a + b] * scale_ji).T)

        if any(local_near):
            with write_lock:
                for mi in range(n_masks):
                    near_chunks[mi].extend(local_near[mi])

    _run_tiled_obs_blocks(nelems, tile, _far_pass)

    # --- Pass 2: Near interactions (self, touching, close pairs) ---
    for mi in range(n_masks):
        obs_idx, src_idx = _expand_near_chunks(near_chunks[mi])
        last_obs = -1
        obs_elem = None
        obs_ids = None
        for pos in range(obs_idx.size):
            obs_index = int(obs_idx[pos])
            if obs_index != last_obs:
                last_obs = obs_index
                obs_elem = elements[obs_index]
                obs_ids = np.asarray(obs_elem.node_ids, dtype=int)
            src_elem = elements[int(src_idx[pos])]
            src_ids = src_elem.node_ids
            s_blk, k_blk = _sk_blocks_near_linear(
                obs_elem=obs_elem,
                src_elem=src_elem,
                k0=k0,
                obs_normal_deriv=obs_normal_deriv,
                obs_order=obs_order,
                src_order=src_order,
                compute_single_layer=compute_single_layer,
                compute_double_layer=compute_double_layer,
            )
            if compute_single_layer:
                coeff = (
                    complex(slp_obs_coeff[obs_index])
                    if slp_obs_coeff is not None else 1.0 + 0.0j
                )
                s_mats[mi][np.ix_(obs_ids, src_ids)] += coeff * s_blk
            if compute_double_layer:
                k_mats[mi][np.ix_(obs_ids, src_ids)] += k_blk
    return list(zip(s_mats, k_mats))


def _assemble_linear_operator_matrices(
    mesh: 'LinearMesh',
    k0: 'Union[complex, float]',
    obs_normal_deriv: 'bool',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    far_ratio: 'float' = 3.0,
    source_element_mask: 'Optional[np.ndarray]' = None,
    compute_single_layer: 'bool' = True,
    compute_double_layer: 'bool' = True,
    single_layer_observation_coefficients: 'Optional[np.ndarray]' = None,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """
    Assemble dense linear-Galerkin S and K/K' matrices on global nodal DOFs.

    ``compute_single_layer`` and ``compute_double_layer`` let formulation
    callers skip an operator they do not consume.  A zero matrix is returned
    for a skipped operator so the long-standing two-array return contract is
    preserved.

    Single-mask front end for `_assemble_linear_operator_matrices_multi`; a
    caller wanting several masks at one wavenumber should use that directly
    so the quadrature is shared.
    """

    return _assemble_linear_operator_matrices_multi(
        mesh=mesh,
        k0=k0,
        obs_normal_deriv=obs_normal_deriv,
        source_element_masks=[source_element_mask],
        obs_order=obs_order,
        src_order=src_order,
        far_ratio=far_ratio,
        compute_single_layer=compute_single_layer,
        compute_double_layer=compute_double_layer,
        single_layer_observation_coefficients=(
            single_layer_observation_coefficients
        ),
    )[0]

def _assemble_linear_hypersingular_matrix(
    mesh: 'LinearMesh',
    k0: 'Union[complex, float]',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    far_ratio: 'float' = 3.0,
    source_element_mask: 'Optional[np.ndarray]' = None,
) -> 'np.ndarray':
    """
    Assemble the hypersingular D operator via the Maue identity.

    D is computed element-by-element from single-layer S blocks:
        D_block = -k^2 (n_obs . n_src) S_block
                + tangent_outer / (L_obs * L_src) * sum(S_block)

    This avoids all hypersingular quadrature; the log singularity in S is handled
    by the existing Duffy transforms.

    The S blocks come from `_integrate_linear_pair_generic`, whose recursion
    bottoms out in a plain tensor-Gauss box for every pair that is neither
    singular, endpoint-touching, nor near-singular (centre separation under
    0.95 element lengths) -- all but O(N) of the N^2 pairs.  Evaluating those
    one Python call at a time made this the only genuinely O(N^2)-interpreted
    routine in the solver; they are batched here over the same tiles the
    single-layer assembly uses, at the same quadrature order the recursion
    would have chosen, leaving the per-pair path only the singular work.

    Both the S blocks and the Maue combination are symmetric under swapping the
    element pair, so only the upper tiles are evaluated.
    """

    nnodes = len(mesh.nodes)
    d_mat = np.zeros((nnodes, nnodes), dtype=np.complex128)
    elements = list(mesh.elements)
    nelems = len(elements)
    if not elements:
        return d_mat

    if source_element_mask is None:
        src_mask = np.ones(nelems, dtype=bool)
    else:
        src_mask = np.asarray(source_element_mask, dtype=bool).reshape(-1)
        if src_mask.size != nelems:
            raise ValueError("source_element_mask length must match mesh element count.")
    if not np.any(src_mask):
        return d_mat

    lengths = np.asarray([e.length for e in elements], dtype=float)
    node_ids = np.asarray([e.node_ids for e in elements], dtype=int)
    panel_index = np.asarray([e.panel_index for e in elements], dtype=int)
    p0_arr = np.stack([e.p0 for e in elements], axis=0)
    p1_arr = np.stack([e.p1 for e in elements], axis=0)
    seg_arr = p1_arr - p0_arr
    mid_arr = 0.5 * (p0_arr + p1_arr)
    normals_arr = np.stack([e.normal for e in elements], axis=0)

    # `_integrate_linear_pair_recursive` stops subdividing at ratio >= 0.95 and
    # then picks tensor_order = max(obs_order, src_order, min(16, adaptive)),
    # which saturates at 16 everywhere in that regime.
    box_order = max(int(obs_order), int(src_order), 16)
    qt, qw = _get_quadrature(max(2, box_order))
    phi_arr = np.array([_linear_shape_values(float(t)) for t in qt])
    t_f = np.asarray(qt, dtype=float)
    quad_pts = p0_arr[:, None, :] + t_f[None, :, None] * seg_arr[:, None, :]

    tile = _assembly_tile_size(nelems, 16 * 7 + 8 * 5)
    real_k = _wavenumber_is_real(k0)
    k2 = complex(k0) ** 2
    touch_tol_sq = 1.0e-12 ** 2
    scalar_chunks: 'List[Tuple[int, int, int, np.ndarray, bool]]' = []
    write_lock = threading.Lock()

    def _batch_pass(i0: 'int', i1: 'int') -> 'None':
        mb = i1 - i0
        obs_slice = slice(i0, i1)
        obs_nid = node_ids[obs_slice]
        obs_len = lengths[obs_slice]
        obs_norm = normals_arr[obs_slice]
        obs_msk = src_mask[obs_slice]
        obs_pts = quad_pts[obs_slice]
        local_scalar: 'List[Tuple[int, int, int, np.ndarray, bool]]' = []

        for j0 in range(i0, nelems, tile):
            j1 = min(j0 + tile, nelems)
            nb = j1 - j0
            mirrored = j0 > i0
            src_slice = slice(j0, j1)
            src_nid = node_ids[src_slice]
            src_len = lengths[src_slice]
            src_norm = normals_arr[src_slice]
            src_msk = src_mask[src_slice]

            mdx = mid_arr[obs_slice, 0][:, None] - mid_arr[src_slice, 0][None, :]
            mdy = mid_arr[obs_slice, 1][:, None] - mid_arr[src_slice, 1][None, :]
            centre_dist = np.sqrt(mdx * mdx + mdy * mdy)
            scale = np.maximum(np.maximum(obs_len[:, None], src_len[None, :]), EPS)
            batch_sym = (centre_dist / scale) >= 0.95
            batch_sym &= (
                panel_index[obs_slice][:, None] != panel_index[src_slice][None, :]
            )
            if batch_sym.any():
                # Endpoint coincidence routes a pair to the touching-Duffy
                # branch, and node snapping (1e-9) is coarser than that 1e-12
                # test, so the geometric predicate is evaluated directly.
                touching = np.zeros((mb, nb), dtype=bool)
                for ends_o in (p0_arr, p1_arr):
                    for ends_s in (p0_arr, p1_arr):
                        edx = ends_o[obs_slice, 0][:, None] - ends_s[src_slice, 0][None, :]
                        edy = ends_o[obs_slice, 1][:, None] - ends_s[src_slice, 1][None, :]
                        touching |= (edx * edx + edy * edy) <= touch_tol_sq
                batch_sym &= ~touching

            batch_ij = batch_sym & src_msk[None, :]
            scalar = ~batch_ij
            scalar &= src_msk[None, :]
            flat = np.flatnonzero(scalar.ravel())
            if flat.size:
                local_scalar.append((i0, np.arange(j0, j1), nb, flat, False))
            batch_ji = None
            if mirrored:
                batch_ji = batch_sym & obs_msk[:, None]
                scalar_t = ~batch_ji
                scalar_t &= obs_msk[:, None]
                flat_t = np.flatnonzero(scalar_t.ravel())
                if flat_t.size:
                    local_scalar.append((i0, np.arange(j0, j1), nb, flat_t, True))

            any_ij = bool(batch_ij.any())
            any_ji = bool(batch_ji.any()) if mirrored else False
            if not (any_ij or any_ji):
                continue

            src_pts = quad_pts[src_slice]
            acc = [np.zeros((mb, nb), dtype=np.complex128) for _ in range(4)]
            g_buf = np.empty((mb, nb), dtype=np.complex128)
            cscratch = np.empty((mb, nb), dtype=np.complex128)
            dx = np.empty((mb, nb), dtype=float)
            dy = np.empty((mb, nb), dtype=float)
            dist = np.empty((mb, nb), dtype=float)
            krbuf = np.empty((mb, nb), dtype=float)
            work = np.empty((mb, nb), dtype=float)

            for qi in range(t_f.size):
                r_obs = obs_pts[:, qi, :]
                w_obs_qi = float(qw[qi])
                phi_o = phi_arr[qi]
                for qj in range(t_f.size):
                    r_src = src_pts[:, qj, :]
                    np.subtract(r_obs[:, 0][:, None], r_src[None, :, 0], out=dx)
                    np.subtract(r_obs[:, 1][:, None], r_src[None, :, 1], out=dy)
                    np.multiply(dx, dx, out=dist)
                    np.multiply(dy, dy, out=work)
                    np.add(dist, work, out=dist)
                    np.sqrt(dist, out=dist)
                    np.maximum(dist, EPS, out=dist)
                    if real_k:
                        _far_kernel_argument(k0, dist, krbuf)
                    _far_green_into(k0, real_k, dist, krbuf, work, g_buf)

                    w = w_obs_qi * float(qw[qj])
                    phi_s = phi_arr[qj]
                    for a in range(2):
                        coeff_a = w * float(phi_o[a])
                        for b in range(2):
                            _axpy_into(
                                acc[2 * a + b], g_buf,
                                coeff_a * float(phi_s[b]), cscratch,
                            )

            # Maue identity, applied to the whole tile at once.  n . n, the
            # length product, and the block sum are all symmetric under
            # swapping the pair, so the mirrored block reuses them with only
            # the (a, b) indices exchanged.
            len_prod = obs_len[:, None] * src_len[None, :]
            n_dot_n = (
                obs_norm[:, 0][:, None] * src_norm[None, :, 0]
                + obs_norm[:, 1][:, None] * src_norm[None, :, 1]
            )
            factor = -k2 * n_dot_n
            denom = np.maximum(len_prod, EPS * EPS)

            def _emit(mask: 'np.ndarray', swap: 'bool', transpose: 'bool') -> 'None':
                scale_mat = len_prod * mask
                scaled = [entry * scale_mat for entry in acc]
                block_sum = scaled[0] + scaled[1] + scaled[2] + scaled[3]
                block_sum /= denom
                rows_src = src_nid if transpose else obs_nid
                cols_src = obs_nid if transpose else src_nid
                for a in range(2):
                    rows = rows_src[:, a][:, None]
                    for b in range(2):
                        cols = cols_src[None, :, b]
                        index = (2 * b + a) if swap else (2 * a + b)
                        contrib = factor * scaled[index]
                        contrib += _TANGENT_OUTER[a, b] * block_sum
                        np.add.at(
                            d_mat, (rows, cols),
                            contrib.T if transpose else contrib,
                        )

            with write_lock:
                if any_ij:
                    _emit(batch_ij, swap=False, transpose=False)
                if any_ji:
                    _emit(batch_ji, swap=True, transpose=True)

        if local_scalar:
            with write_lock:
                scalar_chunks.extend(local_scalar)

    _run_tiled_obs_blocks(nelems, tile, _batch_pass)

    obs_idx, src_idx = _expand_near_chunks(scalar_chunks)
    last_obs = -1
    obs_elem = None
    obs_ids = None
    for pos in range(obs_idx.size):
        obs_index = int(obs_idx[pos])
        if obs_index != last_obs:
            last_obs = obs_index
            obs_elem = elements[obs_index]
            obs_ids = np.asarray(obs_elem.node_ids, dtype=int)
        src_elem = elements[int(src_idx[pos])]
        src_ids = np.asarray(src_elem.node_ids, dtype=int)
        s_blk = _single_layer_block_linear(
            obs_elem=obs_elem,
            src_elem=src_elem,
            k0=k0,
            obs_order=obs_order,
            src_order=src_order,
        )
        d_blk = _hypersingular_block_from_s_block(
            s_blk, k0, obs_elem.normal, src_elem.normal,
            obs_elem.length, src_elem.length,
        )
        d_mat[np.ix_(obs_ids, src_ids)] += d_blk
    return d_mat

def _build_linear_coupled_infos(
    mesh: 'LinearMesh',
    materials: 'MaterialLibrary',
    freq_ghz: 'float',
    pol: 'str',
    k0: 'float',
) -> 'List[PanelCoupledInfo]':
    pseudo_panels = [
        Panel(
            name=e.name,
            seg_type=e.seg_type,
            ibc_flag=e.ibc_flag,
            pos_mat=e.pos_mat,
            neg_mat=e.neg_mat,
            p0=e.p0,
            p1=e.p1,
            center=e.center,
            tangent=e.tangent,
            normal=e.normal,
            length=e.length,
            arc_s_center=float(e.arc_s_center),
        )
        for e in mesh.elements
    ]
    return _build_coupled_panel_info(pseudo_panels, materials, freq_ghz, pol, k0)

def _linear_element_incident_load_many(
    elem: 'LinearElement',
    k_air: 'float',
    elevations_deg: 'np.ndarray',
    order: 'int' = 8,
) -> 'np.ndarray':
    qt, qw = _get_quadrature(max(2, int(order)))
    seg = elem.p1 - elem.p0
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    phi = np.deg2rad(elev)
    dirs = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    out = np.zeros((2, elev.size), dtype=np.complex128)
    for t, w in zip(qt, qw):
        shape = _linear_shape_values(float(t))[:, None]
        rp = elem.p0 + float(t) * seg
        phase = np.exp((1j * k_air) * (dirs @ rp))
        out += float(w) * shape * phase[None, :]
    return out * float(elem.length)

def _linear_element_incident_dn_load_many(
    elem: 'LinearElement',
    k_air: 'float',
    elevations_deg: 'np.ndarray',
    order: 'int' = 8,
) -> 'np.ndarray':
    """
    Galerkin-tested normal derivative of the incident plane wave on one element.

    du_inc/dn = j*k*(d_inc . n) * exp(j*k*d_inc . r)

    Used by TE sheet and impedance/flux right-hand sides.
    """

    qt, qw = _get_quadrature(max(2, int(order)))
    seg = elem.p1 - elem.p0
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    phi = np.deg2rad(elev)
    dirs = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    # d_inc . n for each elevation angle
    d_dot_n = dirs @ np.asarray(elem.normal, dtype=float)  # shape (nelevations,)
    out = np.zeros((2, elev.size), dtype=np.complex128)
    for t, w in zip(qt, qw):
        shape = _linear_shape_values(float(t))[:, None]
        rp = elem.p0 + float(t) * seg
        phase = np.exp((1j * k_air) * (dirs @ rp))
        out += float(w) * shape * (1j * k_air * d_dot_n * phase)[None, :]
    return out * float(elem.length)




def _farfield_linear_density_many(
    mesh: 'LinearMesh',
    density: 'np.ndarray',
    k_air: 'float',
    observation_angles_deg: 'np.ndarray',
    potential: 'str',
    order: 'int' = 8,
    element_mask: 'Optional[np.ndarray]' = None,
) -> 'np.ndarray':
    """Vectorized SLP/DLP far field for one or matched-many densities.

    ``density`` may contain one column (one incidence projected at every
    observation angle) or one column per observation angle (the monostatic
    batched-solve case). Element tiling bounds the temporary phase matrix.
    """

    obs = np.asarray(observation_angles_deg, dtype=float).reshape(-1)
    rho = np.asarray(density, dtype=np.complex128)
    if rho.ndim == 1:
        rho = rho[:, None]
    if rho.shape[0] != len(mesh.nodes):
        raise ValueError("Far-field density height must match mesh node count.")
    if rho.shape[1] not in (1, obs.size):
        raise ValueError(
            "Far-field density must have one column or one per observation angle."
        )
    kind = str(potential).strip().upper()
    if kind not in {"SLP", "DLP"}:
        raise ValueError("Far-field potential must be 'SLP' or 'DLP'.")

    if element_mask is None:
        elements = list(mesh.elements)
    else:
        mask = np.asarray(element_mask, dtype=bool).reshape(-1)
        if mask.size != len(mesh.elements):
            raise ValueError("Far-field element mask must match mesh element count.")
        elements = [elem for elem, keep in zip(mesh.elements, mask) if keep]
    if not elements:
        return np.zeros(obs.size, dtype=np.complex128)
    qt, qw = _get_quadrature(max(2, int(order)))
    q = np.asarray(qt, dtype=float)
    wq = np.asarray(qw, dtype=float)
    phi_q = np.column_stack((1.0 - q, q))
    dirs = np.column_stack((
        np.cos(np.deg2rad(obs)), np.sin(np.deg2rad(obs))
    ))
    node_ids = np.asarray([elem.node_ids for elem in elements], dtype=int)
    p0 = np.asarray([elem.p0 for elem in elements], dtype=float)
    seg = np.asarray([elem.p1 - elem.p0 for elem in elements], dtype=float)
    lengths = np.asarray([elem.length for elem in elements], dtype=float)
    normals = np.asarray([elem.normal for elem in elements], dtype=float)

    amp = np.zeros(obs.size, dtype=np.complex128)
    phase_entries = 2_000_000  # about 32 MB of complex128 phase data
    tile = max(1, min(
        len(elements), phase_entries // max(1, obs.size * q.size)
    ))
    for start in range(0, len(elements), tile):
        stop = min(start + tile, len(elements))
        pts = (
            p0[start:stop, None, :]
            + q[None, :, None] * seg[start:stop, None, :]
        )
        phase = np.exp(
            1j * float(k_air) * np.einsum('ad,eqd->aeq', dirs, pts)
        )
        local = rho[node_ids[start:stop], :]
        rho_q = np.einsum('qi,eic->eqc', phi_q, local)
        weights = lengths[start:stop, None] * wq[None, :]
        if kind == "DLP":
            dot_n = dirs @ normals[start:stop].T
            phase *= (1j * float(k_air)) * dot_n[:, :, None]
        if rho.shape[1] == 1:
            amp += np.einsum(
                'aeq,eq,eq->a', phase, rho_q[:, :, 0], weights
            )
        else:
            amp += np.einsum(
                'aeq,eqa,eq->a', phase, rho_q, weights
            )
    return amp

def _linear_mass_block(elem: 'LinearElement') -> 'np.ndarray':
    """Consistent 2-node boundary mass matrix on one straight element."""

    l = float(elem.length)
    return l * np.asarray([[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]], dtype=np.complex128)

def _linear_coupled_interface_signature(elem: 'LinearElement', info: 'PanelCoupledInfo') -> 'Tuple[Any, ...]':
    return (
        int(elem.seg_type),
        int(elem.ibc_flag),
        int(elem.pos_mat),
        int(elem.neg_mat),
        int(info.minus_region),
        int(info.plus_region),
        str(info.bc_kind),
    )

def _linear_coupled_node_report(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
) -> 'Dict[str, int]':
    """
    Summarize node configurations for the nodal coupled solve.

    The linear/Galerkin path handles shared geometric junctions by
    augmenting the nodal system with trace-continuity and region-wise flux-balance rows.
    Branching and mixed-interface node counts are reported for diagnostics; they are
    not automatic blockers by themselves.
    """

    incident: 'Dict[int, List[int]]' = {}
    for eidx, elem in enumerate(mesh.elements):
        for nid in elem.node_ids:
            incident.setdefault(int(nid), []).append(int(eidx))

    branching_nodes = 0
    mixed_interface_nodes = 0
    for nid, elem_ids in incident.items():
        unique = sorted(set(int(v) for v in elem_ids))
        if len(unique) <= 1:
            continue
        sigs = {
            _linear_coupled_interface_signature(mesh.elements[eidx], infos[eidx])
            for eidx in unique
        }
        if len(unique) > 2:
            branching_nodes += 1
        if len(sigs) > 1:
            mixed_interface_nodes += 1

    return {
        "linear_node_count": int(len(mesh.nodes)),
        "linear_element_count": int(len(mesh.elements)),
        "linear_branching_nodes": int(branching_nodes),
        "linear_mixed_interface_nodes": int(mixed_interface_nodes),
        "linear_unsupported_nodes": 0,
    }

def _build_linear_junction_constraints(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
) -> 'Tuple[np.ndarray, Dict[str, int]]':
    """
    Build nodal junction constraints for the linear/Galerkin coupled solve.

    The linear trace unknown is continuous only across explicitly shared nodes. When the
    interface-aware mesh intentionally splits nodes at the same geometric coordinate, we
    restore pointwise continuity at true shared geometric junctions with explicit trace
    constraints. We also add region-wise flux-balance constraints using the endpoint sign
    convention.
    """

    nnodes = len(mesh.nodes)
    grouped: 'Dict[Tuple[int, int], List[Tuple[int, int, int]]]' = {}
    for eidx, elem in enumerate(mesh.elements):
        n0, n1 = (int(v) for v in elem.node_ids)
        grouped.setdefault(mesh.nodes[n0].key, []).append((int(eidx), 0, n0))
        grouped.setdefault(mesh.nodes[n1].key, []).append((int(eidx), 1, n1))

    rows: 'List[np.ndarray]' = []
    trace_count = 0
    flux_count = 0
    junction_nodes = 0
    orientation_conflict_nodes = 0
    constrained_nodes: 'Set[int]' = set()
    constrained_elems: 'Set[int]' = set()

    for entries in grouped.values():
        unique_elems = sorted({int(eidx) for eidx, _, _ in entries})
        unique_nodes = sorted({int(nid) for _, _, nid in entries})
        if len(unique_elems) < 2 and len(unique_nodes) < 2:
            continue

        by_elem_sign: 'Dict[int, int]' = {}
        seg_names: 'Set[str]' = set()
        region_set: 'Set[int]' = set()
        for eidx, local_end, nid in entries:
            endpoint_sign = +1 if int(local_end) == 0 else -1
            by_elem_sign[int(eidx)] = by_elem_sign.get(int(eidx), 0) + endpoint_sign
            seg_names.add(mesh.elements[int(eidx)].name)
            info = infos[int(eidx)]
            if info.minus_region >= 0:
                region_set.add(int(info.minus_region))
            if info.plus_region >= 0:
                region_set.add(int(info.plus_region))

        if len(seg_names) >= 2:
            signs = [int(np.sign(by_elem_sign.get(eidx, 0))) for eidx in unique_elems]
            has_pos = any(s > 0 for s in signs)
            has_neg = any(s < 0 for s in signs)
            if not (has_pos and has_neg):
                orientation_conflict_nodes += 1

        if len(unique_nodes) > 1:
            ref_nid = unique_nodes[0]
            for other_nid in unique_nodes[1:]:
                row = np.zeros(2 * nnodes, dtype=np.complex128)
                row[ref_nid] = 1.0 + 0.0j
                row[other_nid] = -1.0 + 0.0j
                rows.append(row)
                trace_count += 1
                constrained_nodes.add(ref_nid)
                constrained_nodes.add(other_nid)

        for region in sorted(region_set):
            row = np.zeros(2 * nnodes, dtype=np.complex128)
            terms = 0
            for eidx, local_end, nid in entries:
                endpoint_sign = +1 if int(local_end) == 0 else -1
                info = infos[int(eidx)]
                coeff_u = 0.0 + 0.0j
                coeff_q = 0.0 + 0.0j
                participates = False
                if info.minus_region == region:
                    coeff_q += 1.0 + 0.0j
                    participates = True
                if info.plus_region == region:
                    coeff_u += complex(info.q_plus_gamma)
                    coeff_q += complex(info.q_plus_beta)
                    participates = True
                if not participates:
                    continue

                w = complex(float(endpoint_sign), 0.0)
                nid_i = int(nid)
                row[nid_i] += w * coeff_u
                row[nnodes + nid_i] += w * coeff_q
                terms += 1
                constrained_nodes.add(nid_i)
                constrained_elems.add(int(eidx))

            if terms >= 2 and np.linalg.norm(row) > 0.0:
                rows.append(row)
                flux_count += 1

        junction_nodes += 1

    if not rows:
        return np.zeros((0, 2 * nnodes), dtype=np.complex128), {
            "junction_nodes": 0,
            "junction_constraints": 0,
            "junction_panels": 0,
            "junction_trace_constraints": 0,
            "junction_flux_constraints": 0,
            "junction_orientation_conflict_nodes": int(orientation_conflict_nodes),
        }

    c_mat = np.vstack(rows)
    return c_mat, {
        "junction_nodes": int(junction_nodes),
        "junction_constraints": int(c_mat.shape[0]),
        "junction_panels": int(len(constrained_elems)),
        "junction_trace_constraints": int(trace_count),
        "junction_flux_constraints": int(flux_count),
        "junction_orientation_conflict_nodes": int(orientation_conflict_nodes),
    }

def _ensure_finite_linear_system(a_mat: 'np.ndarray', rhs: 'Optional[np.ndarray]' = None, label: 'str' = "linear system") -> 'None':
    """Raise a clear error before calling LAPACK if the assembled system contains NaN/Inf."""

    a_eval = np.asarray(a_mat)
    if not np.all(np.isfinite(a_eval)):
        bad = np.argwhere(~np.isfinite(a_eval))
        first = tuple(int(v) for v in bad[0]) if bad.size else None
        raise ValueError(f"{label}: system matrix contains NaN/Inf at index {first}.")
    if rhs is None:
        return
    b_eval = np.asarray(rhs)
    if not np.all(np.isfinite(b_eval)):
        bad = np.argwhere(~np.isfinite(b_eval))
        first = tuple(int(v) for v in bad[0]) if bad.size else None
        raise ValueError(f"{label}: RHS contains NaN/Inf at index {first}.")

def _assemble_linear_mass_matrix(mesh: 'LinearMesh') -> 'np.ndarray':
    """Assemble the global consistent mass matrix for the linear boundary mesh."""

    nnodes = len(mesh.nodes)
    m_mat = np.zeros((nnodes, nnodes), dtype=np.complex128)
    for elem in mesh.elements:
        ids = np.asarray(elem.node_ids, dtype=int)
        m_mat[np.ix_(ids, ids)] += _linear_mass_block(elem)
    return m_mat


def _assemble_linear_weighted_mass_matrix(
    mesh: 'LinearMesh',
    element_coefficients: 'np.ndarray',
) -> 'np.ndarray':
    """Assemble ``integral phi_i c_h phi_j ds`` for elementwise-constant c.

    Material tables and impedance tapers are sampled at element centers, so
    the discrete coefficient represented by ``PanelCoupledInfo`` is naturally
    piecewise constant. Keeping it inside each element weak integral is exact
    for that discrete material model and avoids unweighted node averaging on
    nonuniform meshes and at taper endpoints.
    """

    coeff = np.asarray(element_coefficients, dtype=np.complex128).reshape(-1)
    if coeff.size != len(mesh.elements):
        raise ValueError(
            "Weighted mass coefficient count must match mesh element count."
        )
    if not np.all(np.isfinite(coeff.real) & np.isfinite(coeff.imag)):
        raise ValueError("Weighted mass coefficients must all be finite.")
    nnodes = len(mesh.nodes)
    weighted = np.zeros((nnodes, nnodes), dtype=np.complex128)
    for eidx, elem in enumerate(mesh.elements):
        ids = np.asarray(elem.node_ids, dtype=int)
        weighted[np.ix_(ids, ids)] += (
            complex(coeff[eidx]) * _linear_mass_block(elem)
        )
    return weighted


def _robin_alpha_elements(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
) -> 'Tuple[np.ndarray, np.ndarray]':
    """Return per-element Robin alpha and the PEC-element mask."""

    if len(mesh.elements) != len(infos):
        raise ValueError(
            "Robin coefficient construction requires matching elements and infos."
        )
    alpha = np.zeros(len(mesh.elements), dtype=np.complex128)
    pec = np.zeros(len(mesh.elements), dtype=bool)
    for eidx, info in enumerate(infos):
        z_surf = complex(info.robin_impedance)
        if abs(z_surf) <= EPS:
            pec[eidx] = True
            continue
        eps_m = info.eps_minus if info.minus_region >= 0 else info.eps_plus
        mu_m = info.mu_minus if info.minus_region >= 0 else info.mu_plus
        k_m = info.k_minus if info.minus_region >= 0 else info.k_plus
        alpha[eidx] = _surface_robin_alpha(
            pol, eps_m, mu_m, k_m, z_surf
        )
    return alpha, pec

def prepare_linear_galerkin_system(
    geometry_snapshot: 'Dict[str, Any]',
    frequency_ghz: 'float',
    polarization: 'str',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    node_snap_tol: 'float' = 1.0e-9,
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
) -> 'Dict[str, Any]':
    """
    Build the reusable linear-Galerkin coupled system for one frequency.

    The helper validates the geometry, builds boundary primitives, promotes them to a
    continuous two-node linear mesh, derives per-element coupled material data, and
    assembles dense nodal S/K region operators.

    It returns reusable nodal operators and metadata for external scripts.
    """

    freq_ghz = float(frequency_ghz)
    if (not math.isfinite(freq_ghz)) or freq_ghz <= 0.0:
        raise ValueError("frequency_ghz must be a positive finite value.")
    pol = _normalize_polarization(polarization)
    unit_scale = _unit_scale_to_meters(geometry_units)
    base_dir = _material_base_dir_for_snapshot(
        geometry_snapshot, material_base_dir
    )
    mesh_freq_ghz = float(mesh_reference_ghz) if mesh_reference_ghz is not None else freq_ghz
    if (not math.isfinite(mesh_freq_ghz)) or mesh_freq_ghz <= 0.0:
        raise ValueError("mesh_reference_ghz must be a positive finite GHz value when provided.")
    preflight = validate_geometry_snapshot_for_solver(geometry_snapshot, base_dir=base_dir, meters_scale=unit_scale)
    materials = MaterialLibrary.from_entries(
        geometry_snapshot.get("ibcs", []) or [],
        geometry_snapshot.get("dielectrics", []) or [],
        base_dir=base_dir,
    )
    for _msg in list(preflight.get('warnings', []) or []):
        materials.warn_once(str(_msg))
    _warn_far_quadrature_override(materials)
    lambda_min, mesh_max_index, mesh_material_flags = (
        _conservative_mesh_wavelength_for_frequencies(
            geometry_snapshot, materials, {freq_ghz, mesh_freq_ghz}
        )
    )
    panels = _build_panels(
        geometry_snapshot=geometry_snapshot,
        meters_scale=unit_scale,
        min_wavelength=lambda_min,
        max_panels=max_panels,
    )

    mesh = _build_linear_mesh(panels, node_snap_tol=node_snap_tol)
    k0 = 2.0 * math.pi * (freq_ghz * 1e9) / C0
    infos = _build_linear_coupled_infos(mesh, materials, freq_ghz=freq_ghz, pol=pol, k0=k0)

    region_to_k: 'Dict[int, complex]' = {}
    for info in infos:
        if info.minus_region >= 0:
            region_to_k[info.minus_region] = complex(info.k_minus)
        if info.plus_region >= 0:
            region_to_k[info.plus_region] = complex(info.k_plus)

    region_ops: 'Dict[int, Tuple[np.ndarray, np.ndarray]]' = {}
    cache: 'Dict[Tuple[float, float, bool], Tuple[np.ndarray, np.ndarray]]' = {}
    for region, k_region in region_to_k.items():
        key = (round(float(np.real(k_region)), 12), round(float(np.imag(k_region)), 12), False)
        if key not in cache:
            cache[key] = _assemble_linear_operator_matrices(
                mesh=mesh,
                k0=k_region if abs(k_region) > EPS else (EPS + 0.0j),
                obs_normal_deriv=False,
                obs_order=obs_order,
                src_order=src_order,
            )
        region_ops[region] = cache[key]

    return {
        "panels": panels,
        "mesh": mesh,
        "materials": materials,
        "infos": infos,
        "region_ops": region_ops,
        "metadata": {
            "frequency_ghz": float(freq_ghz),
            "mesh_reference_ghz": float(mesh_freq_ghz),
            "mesh_wavelength_m": float(lambda_min),
            "mesh_max_refractive_index": float(mesh_max_index),
            "mesh_material_flags": list(mesh_material_flags),
            "polarization_internal": pol,
            "panel_count": len(panels),
            "linear_element_count": len(mesh.elements),
            "linear_node_count": len(mesh.nodes),
            "node_snap_tol_m": float(node_snap_tol),
            "obs_order": int(obs_order),
            "src_order": int(src_order),
            "warnings": list(materials.warnings),
            "preflight": dict(preflight),
            "status": "stage1-system",
        },
    }


def _medium_eta(eps: 'complex', mu: 'complex') -> 'complex':
    eps, mu = _validate_passive_medium(eps, mu, "Medium")
    # Derive eta from the SAME causal index branch used for k.  This preserves
    # k*eta = omega*mu and gives positive real wave impedance for a passive
    # lossless double-negative medium.
    n = _causal_medium_index(eps, mu)
    return ETA0 * mu / n

def _medium_n(eps: 'complex', mu: 'complex') -> 'complex':
    eps, mu = _validate_passive_medium(eps, mu, "Medium")
    return cmath.sqrt(eps * mu)

def _safe_complex_div(num: 'complex', den: 'complex', fallback: 'complex') -> 'complex':
    if abs(den) <= EPS:
        return fallback
    return num / den




def _region_medium(materials: 'MaterialLibrary', region_flag: 'int', freq_ghz: 'float') -> 'Tuple[complex, complex]':
    # TYPE 1 sheets use distinct virtual region IDs solely to keep their two
    # traces topologically separate.  Those virtual regions are air by
    # construction and are not missing user dielectric definitions.
    if region_flag <= 0 or region_flag >= VIRTUAL_SHEET_REGION_START:
        return 1.0 + 0.0j, 1.0 + 0.0j
    return materials.get_medium(region_flag, freq_ghz)

def _causal_medium_index(eps: 'complex', mu: 'complex') -> 'complex':
    """
    Choose refractive-index branch consistent with passive media in e^{+jwt}.

    Passive attenuation requires Im(n) <= 0.  On the exactly lossless branch,
    select the sign that gives non-negative real wave impedance mu/n.  This is
    the limiting-absorption continuation and, in particular, makes a passive
    double-negative medium use Re(n) < 0 instead of discontinuously jumping to
    the positive-index root.
    """

    n = _medium_n(eps, mu)
    branch_tol = 64.0 * np.finfo(float).eps * max(1.0, abs(n))
    if n.imag > branch_tol:
        n = -n
    elif abs(n.imag) <= branch_tol:
        eta_try = complex(mu) / n
        if eta_try.real < 0.0:
            n = -n
    return n

def _mesh_wavelength_for_snapshot(
    geometry_snapshot: 'Dict[str, Any]',
    materials: 'MaterialLibrary',
    frequency_ghz: 'float',
) -> 'Tuple[float, float, List[int]]':
    """
    Return the shortest wavelength that the boundary mesh must resolve.

    Air is always present as the exterior reference, so the controlling
    refractive-index magnitude is at least one. Only dielectric flags actually
    referenced by TYPE 3/4/5 boundaries participate; unused library rows do not
    over-refine a model. For lossy media ``|n|`` is a conservative spatial scale
    covering both phase variation and attenuation.
    """

    freq = float(frequency_ghz)
    if not math.isfinite(freq) or freq <= 0.0:
        raise ValueError(
            f"Mesh wavelength requires a positive finite frequency; got {frequency_ghz!r}."
        )

    material_flags: 'Set[int]' = set()
    for seg in _snapshot_segments(geometry_snapshot):
        props = list(seg.get("properties", []) or [])
        if len(props) < 5:
            props.extend([""] * (5 - len(props)))
        seg_type = _parse_flag(
            props[0] if str(props[0]).strip() else seg.get("seg_type", 0)
        )
        pos_mat = _parse_flag(props[3])
        neg_mat = _parse_flag(props[4])
        if seg_type in (3, 4) and pos_mat > 0:
            material_flags.add(pos_mat)
        elif seg_type == 5:
            if pos_mat > 0:
                material_flags.add(pos_mat)
            if neg_mat > 0:
                material_flags.add(neg_mat)

    max_index = 1.0
    for flag in sorted(material_flags):
        eps, mu = materials.get_medium(flag, freq)
        index_mag = float(abs(_causal_medium_index(eps, mu)))
        if not math.isfinite(index_mag) or index_mag <= 0.0:
            raise ValueError(
                f"Dielectric flag {flag} produced invalid refractive-index magnitude "
                f"{index_mag!r} at {freq:g} GHz."
            )
        max_index = max(max_index, index_mag)

    free_space_wavelength = C0 / (freq * 1.0e9)
    return (
        float(free_space_wavelength / max_index),
        float(max_index),
        sorted(material_flags),
    )

def _conservative_mesh_wavelength_for_frequencies(
    geometry_snapshot: 'Dict[str, Any]',
    materials: 'MaterialLibrary',
    frequencies_ghz,
) -> 'Tuple[float, float, List[int]]':
    """Shortest referenced-material wavelength over every supplied frequency."""

    values = [
        _mesh_wavelength_for_snapshot(geometry_snapshot, materials, float(freq))
        for freq in frequencies_ghz
    ]
    if not values:
        raise ValueError("At least one mesh-control frequency is required.")
    return (
        min(value[0] for value in values),
        max(value[1] for value in values),
        sorted({flag for value in values for flag in value[2]}),
    )

def _medium_wavenumber(
    k0: 'float',
    eps: 'complex',
    mu: 'complex',
) -> 'complex':
    """Complex medium wavenumber used directly inside integral kernels."""

    return complex(k0) * _causal_medium_index(eps, mu)

def _impedance_to_admittance(z_value: 'complex') -> 'complex':
    z_eval = _ensure_finite_complex(z_value, "Surface impedance")
    if abs(z_eval) <= EPS:
        return 0.0 + 0.0j
    return 1.0 / z_eval

def _surface_robin_alpha(
    pol: 'str',
    eps_medium: 'complex',
    mu_medium: 'complex',
    k_medium: 'complex',
    z_surface: 'complex',
) -> 'complex':
    """
    Return the scalar Robin coefficient alpha for q + alpha*u = 0.

    Physical SIBC boundary conditions for 2D scalar wave equation:

    TM (E_z, Dirichlet-like for PEC):
      E_z + Z_s * H_phi = 0
      -> du/dn + j*k*eta/Z_s * u = 0
      -> alpha = j * k * eta / Z_s
      Limits: Z_s->0 -> alpha->inf (u=0, PEC TM)
              Z_s->inf -> alpha->0 (q=0, PMC TM)

    TE (H_z, Neumann-like for PEC):
      -> du/dn + j*k*Z_s/eta * u = 0
      -> alpha = +j * k * Z_s / eta
      Limits: Z_s->0 -> alpha->0 (q=0, PEC TE)
              Z_s->inf -> alpha->inf (u=0, PMC TE)
      Sign pinned by the flat-interface reflection coefficient
      R_H = (eta - Z_s) / (eta + Z_s) (matched absorber Z_s = eta must
      absorb, not amplify) and validated against the impedance-cylinder
      Mie series; both alphas flip together with the normal, so
      alpha_TM * alpha_TE = -k^2 under any single normal convention.
    """

    if abs(z_surface) <= EPS:
        return 0.0 + 0.0j
    eta_medium = _medium_eta(eps_medium, mu_medium)
    if pol == "TM":
        return 1j * complex(k_medium) * _safe_complex_div(eta_medium, z_surface, 0.0 + 0.0j)
    return 1j * complex(k_medium) * _safe_complex_div(z_surface, eta_medium, 0.0 + 0.0j)

def _q_plus_beta(
    pol: 'str',
    eps_minus: 'complex',
    mu_minus: 'complex',
    eps_plus: 'complex',
    mu_plus: 'complex',
) -> 'complex':
    """
    Scaling between minus-side and plus-side raw normal derivatives across
    a transmission interface:  q_plus = beta * q_minus.

    For the 2D scalar Helmholtz reduction of Maxwell's equations under the
    e^{+jwt} convention:
      - TM (u = E_z axial):  the continuous flux quantity is (1/mu) du/dn,
        so beta = mu_plus / mu_minus.
      - TE (u = H_z axial):  the continuous flux quantity is (1/eps) du/dn,
        so beta = eps_plus / eps_minus.

    This matches the flux-scaling factor used in `_solve_dielectric_indirect`
    (see "factor = mu_ext/mu_int" for TM there) and the Mie reference in
    `mie_reference.py`.
    """

    if pol == "TE":
        return _safe_complex_div(eps_plus, eps_minus, 1.0 + 0.0j)
    return _safe_complex_div(mu_plus, mu_minus, 1.0 + 0.0j)


def _build_coupled_panel_info(
    panels: 'List[Panel]',
    materials: 'MaterialLibrary',
    freq_ghz: 'float',
    pol: 'str',
    k0: 'float',
) -> 'List[PanelCoupledInfo]':
    """
    Translate geometry TYPE/IBC/IPN flags into coupled interface algebra per panel.

    Project convention:
    - the drawn panel normal points toward the pos_mat side,
    - TYPE 3: plus/pos_mat = dielectric, minus = air,
    - TYPE 5: plus/pos_mat, minus/neg_mat,
    - TYPE 4: plus/pos_mat = dielectric, minus = PEC/IBC side.

    The coupled assembly is allowed to use whichever side is the valid non-PEC side,
    so TYPE 4 remains solvable even though the PEC side is the minus side.
    """

    infos: 'List[PanelCoupledInfo]' = []
    sheet_region_by_name: 'Dict[str, int]' = {}
    next_sheet_region = VIRTUAL_SHEET_REGION_START

    # Medium properties are constant for a (region, frequency) pair. Large
    # meshes used to resample and revalidate the same epsilon/mu twice per
    # panel, which made even submit-time resource previews scale expensively
    # with panel count. Cache only within this call/frequency; dispersive
    # tables are still sampled independently at every requested frequency.
    medium_state = {}  # type: Dict[int, Tuple[complex, complex, complex]]

    def region_state(region):
        # type: (int) -> Tuple[complex, complex, complex]
        key = int(region)
        cached = medium_state.get(key)
        if cached is not None:
            return cached
        eps_value, mu_value = _region_medium(materials, key, freq_ghz)
        k_value = _medium_wavenumber(k0, eps_value, mu_value)
        if (
            abs(k_value.imag) > 1e-10
            and _complex_hankel_backend_name() == "unavailable"
        ):
            raise RuntimeError(
                "Lossy dielectric media require SciPy or mpmath for trustworthy "
                "complex-Hankel evaluation. Install one of those backends before "
                "running production dielectric solves."
            )
        cached = (eps_value, mu_value, k_value)
        medium_state[key] = cached
        return cached

    # Non-tapered impedance models are also constant over every panel carrying
    # the same flag at this frequency. Spatial tapers deliberately bypass this
    # cache because arc_s is part of their physical definition.
    impedance_cache = {}  # type: Dict[int, complex]

    def panel_impedance(panel):
        # type: (Panel) -> complex
        flag = int(panel.ibc_flag)
        if flag <= 0:
            return 0.0 + 0.0j
        if materials.is_tapered_impedance(flag):
            return materials.get_impedance(
                flag, freq_ghz, arc_s=float(panel.arc_s_center)
            )
        if flag not in impedance_cache:
            impedance_cache[flag] = materials.get_impedance(
                flag, freq_ghz, arc_s=float(panel.arc_s_center)
            )
        return impedance_cache[flag]

    for panel in panels:
        seg_type = panel.seg_type
        if seg_type == 3:
            if panel.pos_mat <= 0:
                raise ValueError(f"TYPE 3 panel '{panel.name}' requires pos_mat > 0.")
            plus_region = panel.pos_mat
            minus_region = 0
            bc_kind = "transmission"
            plus_has_incident = False
            minus_has_incident = True
        elif seg_type == 5:
            if panel.pos_mat <= 0 or panel.neg_mat <= 0:
                raise ValueError(f"TYPE 5 panel '{panel.name}' requires pos_mat > 0 and neg_mat > 0.")
            plus_region = panel.pos_mat
            minus_region = panel.neg_mat
            bc_kind = "transmission"
            plus_has_incident = False
            minus_has_incident = False
        elif seg_type == 4:
            if panel.pos_mat <= 0:
                raise ValueError(f"TYPE 4 panel '{panel.name}' requires pos_mat > 0.")
            plus_region = panel.pos_mat
            minus_region = -1
            bc_kind = "robin"
            plus_has_incident = False
            minus_has_incident = False
        elif seg_type == 2:
            minus_region = 0
            plus_region = -1
            bc_kind = "robin"
            minus_has_incident = True
            plus_has_incident = False
        elif seg_type == 1:
            if panel.ibc_flag <= 0:
                raise ValueError(
                    f"TYPE 1 panel '{panel.name}' requires IBC > 0 in coupled dielectric mode."
                )
            sheet_name = panel.name.strip() or "__type1_sheet__"
            sheet_region = sheet_region_by_name.get(sheet_name)
            if sheet_region is None:
                sheet_region = next_sheet_region
                sheet_region_by_name[sheet_name] = sheet_region
                next_sheet_region += 1
            minus_region = 0
            plus_region = sheet_region
            bc_kind = "transmission"
            minus_has_incident = True
            plus_has_incident = False
        else:
            minus_region = 0
            plus_region = -1
            bc_kind = "robin"
            minus_has_incident = True
            plus_has_incident = False

        eps_minus, mu_minus, k_minus = region_state(minus_region)
        eps_plus, mu_plus, k_plus = region_state(plus_region)
        z_card = panel_impedance(panel)
        if bc_kind == "transmission":
            if seg_type == 1:
                if abs(z_card) <= EPS:
                    raise ValueError(
                        f"TYPE 1 panel '{panel.name}' has zero impedance; provide non-zero IBC for sheet mode."
                    )
                q_plus_beta = -1.0 + 0.0j
                q_plus_gamma = _impedance_to_admittance(z_card)
            else:
                q_plus_beta = _q_plus_beta(pol, eps_minus, mu_minus, eps_plus, mu_plus)
                q_plus_gamma = _impedance_to_admittance(z_card)
        else:
            q_plus_beta = _q_plus_beta(pol, eps_minus, mu_minus, eps_plus, mu_plus)
            q_plus_gamma = 0.0 + 0.0j

        infos.append(
            PanelCoupledInfo(
                seg_type=seg_type,
                plus_region=plus_region,
                minus_region=minus_region,
                plus_has_incident=plus_has_incident,
                minus_has_incident=minus_has_incident,
                eps_plus=eps_plus,
                mu_plus=mu_plus,
                eps_minus=eps_minus,
                mu_minus=mu_minus,
                k_plus=k_plus,
                k_minus=k_minus,
                q_plus_beta=q_plus_beta,
                q_plus_gamma=q_plus_gamma,
                bc_kind=bc_kind,
                # robin_impedance carries the surface impedance Z_s and is
                # populated for any element whose BC involves one: both Robin
                # BC elements (PEC-backed IBC) and TYPE 1 free-floating sheets
                # (where Z_s also lives in q_plus_gamma = 1/Z_s).
                robin_impedance=(
                    z_card if bc_kind == "robin"
                    else (z_card if seg_type == 1 else 0.0 + 0.0j)
                ),
            )
        )

    return infos

def _green_2d(k0: 'Union[complex, float]', r: 'float') -> 'complex':
    """2D scalar Green's function G = j/4 * H0^(2)(k r)."""

    x = complex(k0) * max(r, EPS)
    if abs(x) <= 1e-12:
        x = 1e-12 + 0.0j
    return 0.25j * _hankel2_0(x)



def _quadrature_nodes(order: 'int' = 10) -> 'Tuple[np.ndarray, np.ndarray]':
    qx, qw = np.polynomial.legendre.leggauss(order)
    t = 0.5 * (qx + 1.0)
    w = 0.5 * qw
    return t, w

_QUAD_CACHE: 'Dict[int, Tuple[np.ndarray, np.ndarray]]' = {}
_QUAD_LOCK = threading.Lock()

def _get_quadrature(order: 'int') -> 'Tuple[np.ndarray, np.ndarray]':
    o = int(order)
    result = _QUAD_CACHE.get(o)
    if result is not None:
        return result
    with _QUAD_LOCK:
        # Double-check after acquiring lock.
        if o not in _QUAD_CACHE:
            _QUAD_CACHE[o] = _quadrature_nodes(o)
        return _QUAD_CACHE[o]

def _near_singular_scheme(distance: 'float', panel_length: 'float') -> 'Tuple[int, int]':
    """
    Choose quadrature order and source-panel subdivision count.

    This improves near-singular accuracy when observation points approach a panel.
    """

    ratio = float(distance) / max(float(panel_length), EPS)
    if ratio < 0.25:
        return 64, 16
    if ratio < 0.60:
        return 56, 10
    if ratio < 1.50:
        return 40, 6
    if ratio < 3.00:
        return 28, 3
    return 16, 1



def _residual_norm(a_mat: 'np.ndarray', x: 'np.ndarray', b: 'np.ndarray') -> 'float':
    denom = float(np.linalg.norm(b))
    if denom <= EPS:
        denom = 1.0
    return float(np.linalg.norm(a_mat @ x - b) / denom)

def _residual_norm_many(a_mat: 'np.ndarray', x_mat: 'np.ndarray', b_mat: 'np.ndarray') -> 'np.ndarray':
    """Vectorized residual norms for matrix right-hand-sides."""

    x_eval = np.asarray(x_mat)
    b_eval = np.asarray(b_mat)
    if x_eval.ndim == 1:
        return np.asarray([_residual_norm(a_mat, x_eval, b_eval)], dtype=float)

    residual = a_mat @ x_eval - b_eval
    num = np.linalg.norm(residual, axis=0)
    den = np.linalg.norm(b_eval, axis=0)
    den = np.where(den <= EPS, 1.0, den)
    return np.asarray(num / den, dtype=float)

def _summarize_residuals(values: 'List[float]') -> 'Tuple[float, float, int]':
    """Return finite max/mean and the number of non-finite residuals."""

    residuals = np.asarray(values, dtype=float).reshape(-1)
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        max_value = float("nan")
        mean_value = float("nan")
    else:
        max_value = float(np.max(finite))
        mean_value = float(np.mean(finite))
    return max_value, mean_value, int(residuals.size - finite.size)

def _equilibrated_condition_from_lu(
    a_mat: 'np.ndarray',
    lu: 'np.ndarray',
    piv: 'np.ndarray',
) -> 'float':
    """Estimate cond_1 of a row/column-equilibrated matrix from one LU.

    The raw block systems mix trace and flux equations with different scales,
    so their unscaled condition numbers are not comparable.  The inverse
    1-norm estimator needs only a handful of LU solves and avoids the second
    cubic SVD that certification previously performed.
    """

    if _SCIPY_LINALG is None or _SCIPY_SPARSE_LINALG is None:
        raise RuntimeError("equilibrated condition estimation requires SciPy")
    a_eval = np.asarray(a_mat, dtype=np.complex128)
    magnitude = np.abs(a_eval)
    row_scale = np.max(magnitude, axis=1)
    row_scale = np.where(row_scale > 0.0, row_scale, 1.0)
    row_equilibrated = magnitude / row_scale[:, None]
    col_scale = np.max(row_equilibrated, axis=0)
    col_scale = np.where(col_scale > 0.0, col_scale, 1.0)
    norm_a = float(np.max(np.sum(
        row_equilibrated / col_scale[None, :], axis=0
    )))
    n = int(a_eval.shape[0])

    def _inverse_matvec(vector):
        rhs = row_scale * np.asarray(
            vector, dtype=np.complex128
        ).reshape(-1)
        solved = _SCIPY_LINALG.lu_solve((lu, piv), rhs)
        return col_scale * solved

    def _inverse_rmatvec(vector):
        rhs = col_scale * np.asarray(
            vector, dtype=np.complex128
        ).reshape(-1)
        solved = _SCIPY_LINALG.lu_solve((lu, piv), rhs, trans=2)
        return row_scale * solved

    inverse = _SCIPY_SPARSE_LINALG.LinearOperator(
        (n, n), matvec=_inverse_matvec, rmatvec=_inverse_rmatvec,
        dtype=np.complex128,
    )
    inverse_norm = float(_SCIPY_SPARSE_LINALG.onenormest(inverse))
    estimate = norm_a * inverse_norm
    return estimate if math.isfinite(estimate) else float("inf")


def _solve_dense_system(
    a_mat: 'np.ndarray',
    rhs: 'np.ndarray',
    condition_diagnostics: 'Optional[Dict[str, Any]]' = None,
    label: 'str' = "dense system",
) -> 'np.ndarray':
    """Factor once, solve all RHS columns, and optionally estimate condition."""

    a_eval = np.asarray(a_mat, dtype=np.complex128)
    rhs_eval = np.asarray(rhs, dtype=np.complex128)
    if _SCIPY_LINALG is None:
        solution = np.linalg.solve(a_eval, rhs_eval)
        if condition_diagnostics is not None:
            # Production environments require SciPy, but keep a fail-closed
            # diagnostic fallback for minimal installations.
            row = np.max(np.abs(a_eval), axis=1)
            row = np.where(row > 0.0, row, 1.0)
            row_eq = a_eval / row[:, None]
            col = np.max(np.abs(row_eq), axis=0)
            col = np.where(col > 0.0, col, 1.0)
            try:
                estimate = float(np.linalg.cond(row_eq / col[None, :], p=1))
            except np.linalg.LinAlgError:
                estimate = float("inf")
            condition_diagnostics["condition_est"] = estimate
            condition_diagnostics["condition_method"] = (
                "equilibrated_1norm_numpy_fallback"
            )
    else:
        lu, piv = _SCIPY_LINALG.lu_factor(a_eval)
        solution = _SCIPY_LINALG.lu_solve((lu, piv), rhs_eval)
        if condition_diagnostics is not None:
            condition_diagnostics["condition_est"] = (
                _equilibrated_condition_from_lu(a_eval, lu, piv)
            )
            condition_diagnostics["condition_method"] = (
                "equilibrated_1norm_lu_onenormest"
            )
    if condition_diagnostics is not None:
        condition_diagnostics["condition_label"] = str(label)
    return np.asarray(solution, dtype=np.complex128)


def _consume_condition_estimate(
    values: 'List[float]',
    diagnostics: 'Optional[Dict[str, Any]]',
    label: 'str',
) -> 'None':
    """Append a requested estimate, refusing a silently unimplemented path."""

    if diagnostics is None:
        return
    if "condition_est" not in diagnostics:
        raise RuntimeError(
            f"{label} did not produce the requested condition-number "
            "diagnostic; no field is returned."
        )
    values.append(float(diagnostics["condition_est"]))

def _normalize_rcs_normalization_mode(mode: 'Optional[str]') -> 'str':
    """Accept only physical sigma_2d normalization aliases."""

    text = str(mode or "").strip().lower().replace("-", "_")
    if text in {"", "physical", "divide_by_k", "with_k", "k", "derived", "width", "sigma_2d"}:
        return RCS_NORM_MODE_PHYSICAL
    raise ValueError(
        f"Unsupported rcs_normalization_mode '{mode}'. This solver now supports only physical normalization "
        "sigma_2d = |A|^2 / (4k)."
    )

def _normalize_public_2d_solver_method(method: 'Any') -> 'str':
    """Validate the algorithms actually implemented by the public RCS paths."""

    normalized = str(method).strip().lower()
    if normalized == "gmres":
        raise ValueError(
            "solver_method='gmres' is not implemented by the active public "
            "2-D RCS formulations. Use 'auto'/'direct' for dense LU, or "
            "'fmm' on a supported TE-Robin or multi-region monostatic case."
        )
    if normalized not in {"auto", "direct", "fmm"}:
        raise ValueError(
            f"Unsupported 2-D solver_method {method!r}; expected 'auto', "
            "'direct', or 'fmm'."
        )
    return normalized

def _rcs_sigma_from_amp(
    amp_vec: 'np.ndarray',
    k_value: 'float',
) -> 'np.ndarray':
    """
    Apply physical 2D scattering-width normalization to the far-field amplitude.

    Linear scattering width is not presentation data: exact zeros and finite
    deep nulls are retained.  Only conversion to dB applies a display floor.
    """

    amp_eval = np.asarray(amp_vec, dtype=np.complex128)
    if not np.all(np.isfinite(amp_eval.real) & np.isfinite(amp_eval.imag)):
        raise FloatingPointError("Far-field amplitude contains non-finite value(s).")
    k_eval = float(k_value)
    if not math.isfinite(k_eval) or k_eval <= 0.0:
        raise ValueError(f"RCS normalization requires positive finite k; got {k_value!r}.")
    scale = float(RCS_NORM_NUMERATOR) / k_eval
    sigma_lin = scale * (np.abs(amp_eval) ** 2)
    if not np.all(np.isfinite(sigma_lin)):
        raise FloatingPointError("Computed linear scattering width contains non-finite value(s).")
    return np.asarray(sigma_lin, dtype=float)

def _rcs_db_from_sigma(
    sigma_linear: 'Union[float, np.ndarray]',
    floor_linear: 'float' = RCS_DB_FLOOR_LINEAR,
) -> 'np.ndarray':
    """Convert non-negative linear RCS to display dB with a display-only floor."""

    sigma = np.asarray(sigma_linear, dtype=float)
    if not np.all(np.isfinite(sigma)) or np.any(sigma < 0.0):
        raise ValueError("Linear RCS must contain finite non-negative values.")
    floor_eval = float(floor_linear)
    if not math.isfinite(floor_eval) or floor_eval <= 0.0:
        raise ValueError("RCS dB display floor must be positive and finite.")
    return np.asarray(10.0 * np.log10(np.maximum(sigma, floor_eval)), dtype=float)


def evaluate_quality_gate(
    metadata: 'Dict[str, Any]',
    thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
) -> 'Dict[str, Any]':
    """
    Evaluate a lightweight numeric quality gate from solver metadata.

    This does not prove correctness; it catches obvious numerical-risk runs.
    """

    defaults: 'Dict[str, Union[float, int]]' = {
        "residual_norm_max": 1.0e-6,
        "constraint_residual_norm_max": 1.0e-8,
        "condition_est_max": 1.0e6,
        "warnings_max": 10,
    }
    merged = dict(defaults)
    if thresholds:
        supplied = dict(thresholds)
        unknown = sorted(set(supplied) - set(defaults))
        if unknown:
            raise ValueError(
                "Unknown 2-D quality threshold field(s): "
                + ", ".join(str(key) for key in unknown)
            )
        merged.update(supplied)

    residual_limit = float(merged.get("residual_norm_max", defaults["residual_norm_max"]))
    constraint_limit = float(merged.get("constraint_residual_norm_max", defaults["constraint_residual_norm_max"]))
    condition_limit = float(merged.get("condition_est_max", defaults["condition_est_max"]))
    warnings_raw = merged.get("warnings_max", defaults["warnings_max"])
    try:
        warnings_float = float(warnings_raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("warnings_max must be a finite non-negative integer.") from exc
    if (
        not math.isfinite(residual_limit)
        or residual_limit < 0.0
        or not math.isfinite(constraint_limit)
        or constraint_limit < 0.0
        or not math.isfinite(condition_limit)
        or condition_limit < 0.0
    ):
        raise ValueError(
            "2-D residual and condition quality thresholds must be finite "
            "non-negative values."
        )
    if (
        not math.isfinite(warnings_float)
        or warnings_float < 0.0
        or not warnings_float.is_integer()
    ):
        raise ValueError("warnings_max must be a finite non-negative integer.")
    warnings_limit = int(warnings_float)

    residual_raw = metadata.get("residual_norm_max")
    try:
        residual_value = (
            float(residual_raw) if residual_raw is not None else float("nan")
        )
    except (TypeError, ValueError):
        residual_value = float("nan")
    residual_nonfinite_raw = metadata.get("residual_nonfinite_count", 0)
    try:
        residual_nonfinite_count = int(residual_nonfinite_raw)
    except (TypeError, ValueError, OverflowError):
        residual_nonfinite_count = -1
    constraint_value = float(metadata.get("constraint_residual_norm_max", 0.0) or 0.0)
    condition_raw = metadata.get("condition_est_max")
    try:
        condition_value = float(condition_raw) if condition_raw is not None else float("nan")
    except (TypeError, ValueError):
        condition_value = float("nan")
    if "condition_est_computed" in metadata:
        condition_computed = bool(metadata.get("condition_est_computed"))
    else:
        # Backward-compatible inference for older result dictionaries: a
        # finite estimate is usable, while a missing/NaN placeholder means
        # the condition number was not computed and must not fail the gate.
        condition_computed = math.isfinite(condition_value)
    warnings_count = len(list(metadata.get("warnings", []) or []))

    violations: 'List[str]' = []
    if not math.isfinite(residual_value) or residual_value > residual_limit:
        violations.append(
            f"residual_norm_max={residual_value:.6g} exceeds limit {residual_limit:.6g}"
        )
    if residual_nonfinite_count != 0:
        if residual_nonfinite_count > 0:
            violations.append(
                f"residual_nonfinite_count={residual_nonfinite_count} must be zero"
            )
        else:
            violations.append(
                "residual_nonfinite_count is missing a valid non-negative integer value"
            )
    if int(metadata.get("junction_constraints", 0) or 0) > 0 and (
        (not math.isfinite(constraint_value)) or constraint_value > constraint_limit
    ):
        violations.append(
            f"constraint_residual_norm_max={constraint_value:.6g} exceeds limit {constraint_limit:.6g}"
        )
    if condition_computed and (not math.isfinite(condition_value) or condition_value > condition_limit):
        violations.append(
            f"condition_est_max={condition_value:.6g} exceeds limit {condition_limit:.6g}"
        )
    if warnings_count > warnings_limit:
        violations.append(
            f"warnings_count={warnings_count} exceeds limit {warnings_limit}"
        )

    return {
        "passed": len(violations) == 0,
        "thresholds": {
            "residual_norm_max": residual_limit,
            "constraint_residual_norm_max": constraint_limit,
            "condition_est_max": condition_limit,
            "warnings_max": warnings_limit,
        },
        "values": {
            "residual_norm_max": residual_value,
            "residual_nonfinite_count": residual_nonfinite_count,
            "constraint_residual_norm_max": constraint_value,
            "condition_est_max": condition_value,
            "condition_est_computed": condition_computed,
            "warnings_count": warnings_count,
        },
        "violations": violations,
        "certification_scope": (
            "discrete_linear_system_residual_and_condition"
            if condition_computed
            else "discrete_linear_system_residual_only"
        ),
        "mesh_convergence_certified": bool(
            metadata.get("mesh_convergence_certified", False)
        ),
        "reason": (
            "; ".join(violations)
            if violations
            else (
                "discrete linear-system quality thresholds satisfied; "
                + (
                    "condition number was not requested; "
                    if not condition_computed else ""
                )
                + "mesh convergence is separately certified by the production workflow"
            )
        ),
    }


def _is_all_robin(infos: 'List[PanelCoupledInfo]') -> 'bool':
    """Return True if every element uses a Robin BC (PEC or IBC, no dielectric)."""
    return all(info.bc_kind == 'robin' for info in infos)

def _assert_supported_te_type2_contours(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
) -> 'None':
    """
    Reject open TYPE 2 contours before applying a closed-obstacle TE MFIE.

    Geometric endpoint keys are used instead of linear node IDs because the
    interface-aware mesh deliberately splits a shared node when two stitched
    TYPE 2 segments use different IBC flags. Such stitched contours are still
    physically closed and must remain supported.
    """

    if pol != "TE" or not _is_all_robin(infos):
        return

    type2_degree: 'Dict[Tuple[int, int], int]' = {}
    for elem, info in zip(mesh.elements, infos):
        if int(info.seg_type) != 2:
            continue
        for nid in elem.node_ids:
            key = mesh.nodes[int(nid)].key
            type2_degree[key] = type2_degree.get(key, 0) + 1

    open_endpoint_count = sum(1 for degree in type2_degree.values() if degree == 1)
    if open_endpoint_count > 0:
        raise ValueError(
            "Open TYPE 2 PEC/IBC contours are not supported for TE polarization: "
            "the available TE Robin MFIE is a closed-obstacle formulation and "
            f"the geometry has {open_endpoint_count} open TYPE 2 endpoint(s). "
            "Close/stitch the obstacle contour, or use a physically appropriate "
            "TYPE 1 sheet model for an open impedance card."
        )

def _detect_available_gb() -> 'float':
    """Memory this process may actually use, in GB.

    Checked in the order that reflects reality on a cluster: a SLURM
    allocation binds before a cgroup limit, and both bind before what
    /proc/meminfo says the machine has.
    """

    raw = os.environ.get("SLURM_MEM_PER_NODE", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return float(int(raw)) / 1024.0
    raw = os.environ.get("SLURM_MEM_PER_CPU", "").strip()
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if raw.isdigit() and int(raw) > 0 and cpus.isdigit() and int(cpus) > 0:
        return float(int(raw)) * float(int(cpus)) / 1024.0
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            with open(path) as stream:
                text = stream.read().strip()
        except OSError:
            continue
        if text.isdigit():
            limit = float(text) / (1024.0 ** 3)
            if 0.5 < limit < 1.0e6:
                return limit
    try:
        with open("/proc/meminfo") as stream:
            for line in stream:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / (1024.0 ** 2)
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


# Hard ceiling on one solve's estimated dense footprint.
#
# This used to be a flat 32 GB, which is the wrong constant in both directions:
# it refuses a perfectly feasible 40 GB solve on a 750 GB node, and it permits
# a 32 GB one on a 16 GB laptop. It is now derived from what the process can
# actually use, floored at the old value so nothing that ran before stops
# running, and overridable outright with GHOST_MAX_SOLVE_GB.
_MEMORY_LIMIT_FLOOR_GB = 32.0
_MEMORY_LIMIT_FRACTION = 0.9


def _solve_memory_limit_gb() -> 'float':
    override = os.environ.get("GHOST_MAX_SOLVE_GB", "").strip()
    if override:
        try:
            value = float(override)
        except ValueError:
            value = 0.0
        if value > 0.0:
            return value
    detected = _detect_available_gb()
    return max(_MEMORY_LIMIT_FLOOR_GB, _MEMORY_LIMIT_FRACTION * detected)


def _estimate_memory_gb(
    nnodes: 'int',
    use_cfie: 'bool',
    n_regions: 'int' = 1,
    system_dofs: 'Optional[int]' = None,
    operator_matrices: 'Optional[int]' = None,
    n_rhs: 'int' = 1000,
) -> 'float':
    """
    Estimate peak memory for the dense BIE/MoM solve in GB.

    Accounts for: system matrix, region operators, RHS, solution, factorization.
    """

    bytes_per_complex = 16  # complex128
    # System matrix + factorization copy.  The historical default is a 2N
    # coupled system; formulation-aware planning supplies the active system
    # dimension (N for sheet/Robin, 2N for a single dielectric, and the exact
    # interface-side DOF count for multi-region geometries).
    sys_size = (
        2 * nnodes if system_dofs is None else max(1, int(system_dofs))
    )
    sys_bytes = 2 * sys_size * sys_size * bytes_per_complex
    # Dense global operators retained while the system is formed.  Callers
    # that know the formulation provide the actual conservative count.
    if operator_matrices is None:
        ops_per_region = 4 if not use_cfie else 8
        operator_matrices = max(1, int(n_regions)) * ops_per_region
    region_bytes = (
        max(0, int(operator_matrices))
        * nnodes * nnodes * bytes_per_complex
    )
    # RHS + solution
    misc_bytes = 4 * sys_size * bytes_per_complex * max(1, int(n_rhs))
    total = sys_bytes + region_bytes + misc_bytes
    return total / (1024 ** 3)

def _solve_fmm_gmres_columns(
    operator: 'Any',
    rhs: 'np.ndarray',
    angles_deg: 'np.ndarray',
    *,
    formulation: 'str',
    restart: 'int',
    maxiter: 'int',
    rtol: 'float',
    preconditioner: 'Optional[Any]' = None,
) -> 'np.ndarray':
    """
    Solve FMM-backed right-hand sides and fail closed on nonconvergence.

    A nonzero SciPy GMRES ``info`` means the returned iterate is not a
    certified solution. Projecting it into a far field can look numerically
    plausible while being physically arbitrary, so FMM callers must not
    continue after such a status.
    """

    if _SCIPY_SPARSE_LINALG is None:
        raise ImportError("FMM solver requires scipy.sparse.linalg for GMRES.")

    rhs_eval = np.asarray(rhs, dtype=np.complex128)
    if rhs_eval.ndim == 1:
        rhs_eval = rhs_eval[:, None]
    angles = np.asarray(angles_deg, dtype=float).reshape(-1)
    if rhs_eval.shape[1] != angles.size:
        raise ValueError(
            f"{formulation} FMM solve received {rhs_eval.shape[1]} RHS column(s) "
            f"but {angles.size} angle label(s)."
        )

    solution = np.zeros_like(rhs_eval)
    for col, angle in enumerate(angles):
        candidate, info = _gmres_compat(
            operator,
            rhs_eval[:, col],
            rtol=float(rtol),
            atol=float(rtol),
            restart=int(restart),
            maxiter=int(maxiter),
            M=preconditioner,
        )
        candidate = np.asarray(candidate, dtype=np.complex128)
        if info != 0:
            try:
                residual = np.asarray(rhs_eval[:, col] - operator @ candidate)
                denom = max(float(np.linalg.norm(rhs_eval[:, col])), EPS)
                rel_residual = float(np.linalg.norm(residual) / denom)
            except Exception:
                rel_residual = float("nan")
            status = (
                f"iteration limit/status {int(info)}"
                if int(info) > 0
                else f"solver breakdown/illegal-input status {int(info)}"
            )
            residual_text = (
                f"{rel_residual:.3e}" if math.isfinite(rel_residual) else "unavailable"
            )
            raise RuntimeError(
                f"Aborting {formulation} FMM solve at elevation {float(angle):g} deg: "
                f"GMRES did not converge ({status}, relative residual {residual_text}, "
                f"requested rtol {float(rtol):.1e}, maxiter {int(maxiter)}). "
                "No unconverged field or RCS was returned. Check geometry/material "
                "conditioning and mesh resolution; if the dense memory estimate is "
                "acceptable, retry with solver_method='auto' for a direct solve."
            )
        if not np.all(np.isfinite(candidate.real) & np.isfinite(candidate.imag)):
            raise RuntimeError(
                f"Aborting {formulation} FMM solve at elevation {float(angle):g} deg: "
                "GMRES reported convergence but returned non-finite field values."
            )
        solution[:, col] = candidate
    return solution

def _solve_te_robin_mfie(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
    k0: 'float',
    elevations_deg: 'np.ndarray',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    solver_method: 'str' = "auto",
    condition_diagnostics: 'Optional[Dict[str, Any]]' = None,
) -> 'Tuple[np.ndarray, np.ndarray, float]':
    """
    Solve TE Robin (PEC or IBC) problems via a generalized MFIE.

    Uses the single-layer potential representation u_scat = SLP(sigma).
    The exterior-limit Robin BC gives:

        (-1/2 M + K' + alpha.S) sigma = -(du_inc/dn + alpha.u_inc)

    where alpha is retained as a piecewise-constant element coefficient inside
    the Galerkin observation integral (0 for PEC, nonzero for IBC).
    K' is the adjoint double-layer operator (obs_normal_deriv=True).

    When solver_method="fmm", uses FMM-accelerated GMRES instead of dense LU.

    Returns (rcs_linear, amplitude, residual_norm) arrays over elevations.
    """

    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)

    # The impedance model is sampled at element centers, so alpha is a
    # piecewise-constant coefficient in the discrete weak form.  Keep it
    # inside the observation and RHS element integrals; nodal row scaling is
    # not a Galerkin treatment when alpha varies spatially.
    alpha_elements, _ = _robin_alpha_elements(mesh, infos, pol)
    has_ibc = bool(np.any(np.abs(alpha_elements) > EPS))

    # RHS: -(du_inc/dn + alpha * u_inc)
    rhs_mfie = np.zeros((nnodes, elev.size), dtype=np.complex128)
    for eidx, elem in enumerate(mesh.elements):
        ids = np.asarray(elem.node_ids, dtype=int)
        load_dn = _linear_element_incident_dn_load_many(elem, k_air=k0, elevations_deg=elev)
        rhs_mfie[ids, :] -= load_dn
        if has_ibc:
            load_u = _linear_element_incident_load_many(elem, k_air=k0, elevations_deg=elev)
            rhs_mfie[ids, :] -= complex(alpha_elements[eidx]) * load_u

    use_fmm = (solver_method.strip().lower() == "fmm")
    if use_fmm and condition_diagnostics is not None:
        raise ValueError(
            "compute_condition_number=True is unavailable for the matrix-free "
            "2-D FMM path. Use solver_method='auto' for a dense diagnostic "
            "solve, or explicitly disable the condition-number request."
        )
    if not use_fmm:
        # Dense path (original).
        s_alpha_mat, kp_mat = _assemble_linear_operator_matrices(
            mesh, k0, obs_normal_deriv=True,
            obs_order=obs_order, src_order=src_order,
            compute_single_layer=bool(has_ibc),
            single_layer_observation_coefficients=(
                alpha_elements if has_ibc else None
            ),
        )
        mass_mat = _assemble_linear_mass_matrix(mesh)
        a_mfie = -0.5 * mass_mat + kp_mat
        if has_ibc:
            a_mfie += s_alpha_mat
        _ensure_finite_linear_system(a_mfie, rhs_mfie, label="TE Robin MFIE system")
        sigma_mat = _solve_dense_system(
            a_mfie, rhs_mfie, condition_diagnostics,
            "TE Robin MFIE system",
        )
        residual = np.linalg.norm(a_mfie @ sigma_mat - rhs_mfie, axis=0)
    else:
        # FMM path.
        alpha_unique = np.unique(np.round(alpha_elements, decimals=14))
        if alpha_unique.size > 1:
            raise ValueError(
                "Matrix-free TE Robin FMM currently supports only a spatially "
                "constant Robin coefficient (including pure PEC). Tapered or "
                "mixed PEC/IBC coefficients require the dense element-weighted "
                "Galerkin path; no nodal row-scaling approximation was used."
            )
        alpha_constant = (
            complex(alpha_elements[0]) if alpha_elements.size else 0.0 + 0.0j
        )
        try:
            from fmm_helmholtz_2d import FMMOperator
        except ImportError:
            raise ImportError("FMM solver requires fmm_helmholtz_2d.py in the Python path.")
        mass_mat = _assemble_linear_mass_matrix(mesh)
        fmm_kp = FMMOperator(mesh, k0, obs_normal_deriv=True, n_digits=6)
        fmm_s = FMMOperator(mesh, k0, obs_normal_deriv=False, n_digits=6) if has_ibc else None

        def mfie_matvec(x):
            y = -0.5 * (mass_mat @ x) + fmm_kp.matvec(x)
            if has_ibc and fmm_s is not None:
                y += alpha_constant * fmm_s.matvec(x)
            return y

        if _SCIPY_SPARSE_LINALG is None:
            raise ImportError("FMM solver requires scipy.sparse.linalg for GMRES.")
        A_op = _SCIPY_SPARSE_LINALG.LinearOperator(
            (nnodes, nnodes), matvec=mfie_matvec, dtype=np.complex128)
        sigma_mat = _solve_fmm_gmres_columns(
            A_op,
            rhs_mfie,
            elev,
            formulation="TE Robin MFIE",
            restart=50,
            maxiter=300,
            rtol=1.0e-10,
        )
        residual = np.zeros(elev.size)
        for col in range(elev.size):
            r = rhs_mfie[:, col] - mfie_matvec(sigma_mat[:, col])
            residual[col] = np.linalg.norm(r)

    rhs_norm = np.linalg.norm(rhs_mfie, axis=0)
    rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
    residual_vec = residual / rhs_norm

    amp = _farfield_linear_density_many(
        mesh, sigma_mat, k0, elev, "SLP", order=obs_order
    )

    rcs_lin = _rcs_sigma_from_amp(amp, k0)
    return rcs_lin, amp, float(np.max(residual_vec))

def _has_sheet(infos: 'List[PanelCoupledInfo]') -> 'bool':
    """True if any element is a TYPE 1 free-floating resistive/reactive sheet.

    Sheets carry their impedance via q_plus_gamma = 1/Z_s, and correctly
    modelling them requires a formulation that uses that term.  The
    dielectric-indirect and multi-region-indirect solvers do not -- and
    neither does the current coupled trace formulation, which also has
    pre-existing sign/normalization issues in the sheet case that produce
    unphysical results.

    The public RCS dispatch routes all-sheet and sheet + pure-PEC geometries
    to dedicated sheet solvers. It rejects TYPE 1 mixed with an IBC body,
    dielectric body, or layered coating rather than silently sending that
    combination through an operator that omits the sheet admittance. For a
    tapered resistance treatment on a conducting body, use TYPE 2 with a
    tapered IBC instead--that path is validated.
    """
    return any(int(info.seg_type) == 1 for info in infos)


def _assert_air_exterior(infos: 'List[PanelCoupledInfo]') -> 'None':
    """
    Reject geometries with no air-facing boundary.

    Every formulation in this solver poses the scattering problem in a free
    space background: the incident plane wave, the exterior Green's function,
    and the far-field projection all use the air wavenumber k0.  A geometry
    whose boundaries never touch region 0 (e.g. a TYPE 5-only contour with
    dielectric on BOTH sides) describes a non-air background, which the
    dispatch predicates would otherwise mis-capture: `_solve_dielectric_
    indirect` would silently treat the outer dielectric as air and solve a
    different problem.
    """

    for info in infos:
        if info.minus_region == 0 or info.plus_region == 0:
            return
    raise ValueError(
        "Geometry has no air-facing boundary: every interface separates "
        "non-air media (e.g. a TYPE 5 dielectric/dielectric contour with no "
        "enclosing TYPE 2/3 boundary). This solver poses scattering in a "
        "free-space background, so the unbounded exterior region must be "
        "air -- add the body's outer air boundary (TYPE 2/3), or model the "
        "background medium explicitly as an enclosing region."
    )


def _is_single_dielectric_body(infos: 'List[PanelCoupledInfo]') -> 'bool':
    """Return True if every element is a transmission interface (TYPE 3 dielectric).

    Excludes TYPE 1 sheets: those have bc_kind == 'transmission' but their
    impedance semantics live in q_plus_gamma, which the dielectric-indirect
    solver does not evaluate.  In the RCS dispatch, supported sheet
    geometries are handled by the dedicated sheet solvers (and unsupported
    mixes rejected by ``_assert_no_type1_sheet_for_mixed``) before this
    predicate, so it should never see a sheet; the ``_has_sheet`` guard
    below is belt-and-braces.
    """
    if _has_sheet(infos):
        return False
    return all(info.bc_kind == 'transmission' for info in infos)


def _assert_no_type1_sheet_for_mixed(infos: 'List[PanelCoupledInfo]') -> 'None':
    """Reject mixed TYPE 1 + non-sheet geometries that aren't supported.

    Supported geometries containing TYPE 1 sheets:
      - All-sheet (solved by _solve_tm_sheet / _solve_te_sheet)
      - Sheet + pure-PEC TYPE 2 body (solved by _solve_mixed_sheet_pec)

    Everything else -- sheet + IBC-coated body, sheet + dielectric body,
    sheet + layered coating -- still needs bespoke coupling work and is
    rejected.  The error message points at the workaround (TYPE 2 with
    tapered IBC for edge treatments on bodies).
    """
    if _has_sheet(infos) and not _is_all_sheet(infos) and not _is_sheet_plus_pec(infos):
        raise ValueError(
            "Mixed TYPE 1 sheet + (IBC-coated body / dielectric body / "
            "layered coating) geometries are not currently supported. "
            "Supported options: all-sheet (any number of TYPE 1 sheets, "
            "each with its own Z_s), or sheet + pure-PEC TYPE 2 body (mixed "
            "sheet+PEC is handled by _solve_mixed_sheet_pec).  For a "
            "tapered resistive treatment on a coated body, use TYPE 2 with "
            "a tapered IBC row -- that's the physically correct model for "
            "'resistance transitioning from air to a conducting body'."
        )


def _assert_no_type1_sheet(infos: 'List[PanelCoupledInfo]') -> 'None':
    """Raise if any element is a TYPE 1 sheet (boundary-density export path).

    TYPE 1 free-floating sheets ARE supported for RCS, via the dedicated
    sheet BIEs (``_solve_tm_sheet`` / ``_solve_te_sheet`` /
    ``_solve_mixed_sheet_pec``) reached through ``solve_monostatic_rcs_2d``
    and ``solve_bistatic_rcs_2d``.  This guard only protects
    ``compute_boundary_densities``, whose dispatch covers just the robin /
    multi-region / coupled-trace solvers -- none of which build the sheet
    representation, so they would ignore or mishandle the sheet admittance
    q_plus_gamma.  Rather than return wrong densities, fail fast here and
    point the user at the RCS entry points.
    """
    if _has_sheet(infos):
        raise ValueError(
            "TYPE 1 free-floating sheets are not supported by "
            "compute_boundary_densities (the boundary-density export path).  "
            "They ARE supported for RCS: use solve_monostatic_rcs_2d or "
            "solve_bistatic_rcs_2d, which route sheet geometries to the "
            "dedicated sheet BIEs (_solve_tm_sheet / _solve_te_sheet / "
            "_solve_mixed_sheet_pec).  See TAPERED_IBC.md."
        )

def _solve_dielectric_indirect(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
    k0: 'float',
    elevations_deg: 'np.ndarray',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    condition_diagnostics: 'Optional[Dict[str, Any]]' = None,
) -> 'Tuple[np.ndarray, np.ndarray, float]':
    """
    Solve dielectric scattering via the indirect two-density formulation.

    The coupled trace formulation degenerates for dielectrics because the exterior
    BIE alone determines the far-field amplitude regardless of the flux continuity
    parameter beta.  This indirect formulation uses separate densities:

        u_scat(r) = DL_0(mu)   (exterior, double-layer at k0)
        u_int(r)  = SL_1(sigma) (interior, single-layer at k1)

    Trace continuity:  u_inc + (mu/2 + K0*mu) = S1*sigma
    Flux continuity:   D0*mu + factor*(sigma/2 + K'1*sigma) = -du_inc/dn

    where factor = mu_ext/mu_int for E_z, eps_ext/eps_int for H_z.

    Far-field: A = integral jk0*(d.n)*mu * exp(jk0 d.r') ds'
    """

    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)

    # Determine interior wavenumber from coupled infos.
    k1_vals = {complex(info.k_plus) for info in infos if info.plus_region > 0}
    if not k1_vals:
        k1_vals = {complex(info.k_minus) for info in infos if info.minus_region > 0}
    if not k1_vals:
        raise ValueError("Dielectric indirect solver requires at least one dielectric region.")
    k1 = k1_vals.pop()

    # Determine flux scaling factor.
    info0 = infos[0]
    if pol == 'TM':
        # E_z: flux uses 1/mu -> factor = mu_ext/mu_int
        factor = complex(info0.mu_minus / info0.mu_plus) if abs(info0.mu_plus) > EPS else 1.0
    else:
        # H_z: flux uses 1/eps -> factor = eps_ext/eps_int
        factor = complex(info0.eps_minus / info0.eps_plus) if abs(info0.eps_plus) > EPS else 1.0

    # Assemble operators.
    _, K0 = _assemble_linear_operator_matrices(
        mesh, k0, obs_normal_deriv=False,
        obs_order=obs_order, src_order=src_order,
        compute_single_layer=False,
    )
    _, Kp1 = _assemble_linear_operator_matrices(
        mesh, k1, obs_normal_deriv=True,
        obs_order=obs_order, src_order=src_order,
        compute_single_layer=False,
    )
    S1, _ = _assemble_linear_operator_matrices(mesh, k1, obs_normal_deriv=False,
        obs_order=obs_order, src_order=src_order,
        compute_double_layer=False)
    D0 = _assemble_linear_hypersingular_matrix(
        mesh, k0, obs_order=obs_order, src_order=src_order,
    )
    M = _assemble_linear_mass_matrix(mesh)

    # Build system: 2N x 2N.
    a_sys = np.zeros((2 * nnodes, 2 * nnodes), dtype=np.complex128)
    # Row 1 (trace): 0.5*M*mu + K0*mu - S1*sigma = bu
    a_sys[:nnodes, :nnodes] = 0.5 * M + K0
    a_sys[:nnodes, nnodes:] = -S1
    # Row 2 (flux): D0*mu + factor*(0.5*M + K'1)*sigma = -bdn
    a_sys[nnodes:, :nnodes] = D0
    a_sys[nnodes:, nnodes:] = factor * (0.5 * M + Kp1)

    # Build RHS for all elevations.
    rhs_sys = np.zeros((2 * nnodes, elev.size), dtype=np.complex128)
    for elem in mesh.elements:
        ids = np.asarray(elem.node_ids, dtype=int)
        load_u = _linear_element_incident_load_many(elem, k_air=k0, elevations_deg=elev)
        load_dn = _linear_element_incident_dn_load_many(elem, k_air=k0, elevations_deg=elev)
        rhs_sys[ids, :] += load_u
        rhs_sys[nnodes + ids, :] -= load_dn

    _ensure_finite_linear_system(a_sys, rhs_sys, label="dielectric indirect system")
    sol = _solve_dense_system(
        a_sys, rhs_sys, condition_diagnostics,
        "dielectric indirect system",
    )
    if sol.ndim == 1:
        sol = sol.reshape(-1, 1)

    mu_mat = sol[:nnodes, :]  # DL density

    # Residual.
    residual = np.linalg.norm(a_sys @ sol - rhs_sys, axis=0)
    rhs_norm = np.linalg.norm(rhs_sys, axis=0)
    rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
    residual_vec = residual / rhs_norm

    amp = _farfield_linear_density_many(
        mesh, mu_mat, k0, elev, "DLP", order=obs_order
    )

    rcs_lin = _rcs_sigma_from_amp(amp, k0)
    return rcs_lin, amp, float(np.max(residual_vec))

def _geometric_sheet_endpoint_nodes(
    mesh: 'LinearMesh',
    infos: 'Optional[List[PanelCoupledInfo]]' = None,
) -> 'np.ndarray':
    """Return node IDs that are geometric open-strip endpoints (Meixner pin targets).

    A node is a strip endpoint iff only one sheet-element endpoint lands on
    its geometric key (across the whole mesh, regardless of signature).

    The signature-based mesh builder creates distinct node IDs for
    geometrically-coincident panels that have different material signatures
    (e.g., adjacent stair-step tapered-IBC segments with different flags);
    per-node incidence counting would wrongly flag every such node as an
    endpoint and pin mu=0 everywhere.  Counting by geometric key avoids this.

    When ``infos`` is provided, only elements with ``info.seg_type == 1``
    contribute -- so sheet endpoints that touch a PEC body are still flagged
    as endpoints (correct for Meixner), while PEC-body-internal nodes don't
    qualify.  When ``infos`` is None, all elements contribute (appropriate
    for all-sheet geometries).
    """
    geom_count: 'Dict[Tuple[int, int], int]' = {}
    for eidx, elem in enumerate(mesh.elements):
        if infos is not None and int(infos[eidx].seg_type) != 1:
            continue
        for nid in elem.node_ids:
            gk = tuple(mesh.nodes[int(nid)].key)
            geom_count[gk] = geom_count.get(gk, 0) + 1
    endpoint_ids: 'List[int]' = []
    for nid in range(len(mesh.nodes)):
        gk = tuple(mesh.nodes[int(nid)].key)
        if geom_count.get(gk, 0) == 1:
            endpoint_ids.append(int(nid))
    return np.asarray(endpoint_ids, dtype=np.int64)


def _is_all_sheet(infos: 'List[PanelCoupledInfo]') -> 'bool':
    """True if every element is a TYPE 1 free-floating sheet."""
    if not infos:
        return False
    return all(int(info.seg_type) == 1 for info in infos)


def _solve_tm_sheet(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    k0: 'float',
    elevations_deg: 'np.ndarray',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    condition_diagnostics: 'Optional[Dict[str, Any]]' = None,
) -> 'Tuple[np.ndarray, np.ndarray, float]':
    """
    Solve TM (E_z axial) scattering from a thin resistive/reactive sheet.

    Derivation under e^{+jwt}:
        - E_z is continuous across the sheet; J_z = E_z / Z_s on the sheet.
        - Scattered field from an axial current:  u_s = jketa . Int G . J_z dr'.
        - SIBC on the sheet:                      u_inc + u_s = Z_s . J_z.

    Introducing sigma = jketa . J_z so that u_s = SLP(sigma) matches the existing SLP
    far-field projector, the SIBC becomes
        (S - (Z_s / jketa) M) sigma = -RHS_uinc
    in nodal Galerkin form, where S is the single-layer operator at k0 and
    M is the consistent boundary mass matrix.  Per-node Z_s (which may
    vary on a tapered sheet) enters as a diagonal scaling on M.

    Limit cases:
        Z_s -> 0  (PEC sheet):        S sigma = -u_inc          (TM EFIE)
        Z_s -> inf (transparent):     sigma -> 0                (no scattering)

    Returns (rcs_lin, amp, residual_norm_max).  Far field uses the standard
    SLP projector already validated for TM PEC Mie cases.
    """

    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)

    z_elements = np.asarray(
        [complex(info.robin_impedance) for info in infos],
        dtype=np.complex128,
    )

    # Operators and mass.
    S_mat, _ = _assemble_linear_operator_matrices(
        mesh, k0, obs_normal_deriv=False,
        obs_order=obs_order, src_order=src_order,
        compute_double_layer=False)
    # SIBC coefficient remains inside the element weak integral.
    denom = 1j * float(k0) * ETA0
    sigma_factor_elements = z_elements / denom
    weighted_mass = _assemble_linear_weighted_mass_matrix(
        mesh, sigma_factor_elements
    )

    a_sys = S_mat - weighted_mass

    # RHS: -<phi, u_inc>.
    rhs_sys = np.zeros((nnodes, elev.size), dtype=np.complex128)
    for elem in mesh.elements:
        ids = np.asarray(elem.node_ids, dtype=int)
        load_u = _linear_element_incident_load_many(elem, k_air=float(k0), elevations_deg=elev)
        rhs_sys[ids, :] -= load_u

    _ensure_finite_linear_system(a_sys, rhs_sys, label="TM sheet system")
    sigma_mat = _solve_dense_system(
        a_sys, rhs_sys, condition_diagnostics, "TM sheet system"
    )
    if sigma_mat.ndim == 1:
        sigma_mat = sigma_mat.reshape(-1, 1)

    residual = np.linalg.norm(a_sys @ sigma_mat - rhs_sys, axis=0)
    rhs_norm = np.linalg.norm(rhs_sys, axis=0)
    rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
    residual_vec = residual / rhs_norm

    amp = _farfield_linear_density_many(
        mesh, sigma_mat, float(k0), elev, "SLP", order=obs_order
    )

    rcs_lin = _rcs_sigma_from_amp(amp, float(k0))
    return rcs_lin, amp, float(np.max(residual_vec))


def _solve_te_sheet(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    k0: 'float',
    elevations_deg: 'np.ndarray',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    condition_diagnostics: 'Optional[Dict[str, Any]]' = None,
) -> 'Tuple[np.ndarray, np.ndarray, float]':
    """
    Solve TE (H_z axial) scattering from a thin resistive/reactive sheet.

    Derivation under e^{+jwt}:
        - E_tangent is continuous across the sheet.
        - H_z jumps across the sheet by the induced tangential surface
          current K:  [H_z]+ - [H_z]- = K.
        - Ohm's law: K = E_tangent / Z_s.
        - E_tangent related to H_z via E_x = -(1/jomegaeps) dH_z/dy, which is
          continuous across the sheet (same on both sides).

    Represent u_s by a double-layer potential with density mu = K:
        u_s(r) = Int (dG(r,r')/dn_src) . mu(r') dr'.
    Then u_s jumps across the sheet by exactly mu (as required), and the
    normal derivative of u_s is continuous and equals (N mu)(r), where N
    is the hypersingular operator obtainable from S via the Maue identity.

    At the sheet:  q = q_inc + N mu,  E_tangent relates to q through the
    local frame, and K = Y . E_tangent gives:
        (N - jomegaeps . Z_s . I) mu = -q_inc    ->   (N - jk/eta . Z_s . M) mu = -RHS_qinc

    (Using omegaeps = k/eta with eta = free-space impedance.  The sign of the Z_s
    term mirrors the validated TM sheet system S - (Z_s/jketa).M: this
    code's Green's function is G = +(j/4)H0^(2) -- the negative of the
    textbook e^{+jomegat} fundamental solution -- which flips the sign of the
    operator terms relative to the mass term.  Validated against the
    analytic resistive-sheet jump-BC series; the naive '+' sign makes a
    passive sheet scatter above the PEC level.)

    Limit cases:
        Z_s -> 0  (PEC sheet):        N mu = -q_inc          (TE PEC Neumann)
        Z_s -> inf (transparent):     mu -> 0                (no scattering)

    Returns (rcs_lin, amp, residual_norm_max).  Far field uses the standard
    DLP projector.
    """

    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)

    z_elements = np.asarray(
        [complex(info.robin_impedance) for info in infos],
        dtype=np.complex128,
    )

    # Operators: hypersingular via Maue, plus mass matrix.
    N_mat = _assemble_linear_hypersingular_matrix(
        mesh, k0, obs_order=obs_order, src_order=src_order)
    # Coefficient: jomegaeps . Z_s = (jk/eta) . Z_s, retained inside
    # each element weak integral.
    coeff_elements = (1j * float(k0) / ETA0) * z_elements
    weighted_mass = _assemble_linear_weighted_mass_matrix(
        mesh, coeff_elements
    )

    a_sys = N_mat - weighted_mass

    # RHS: -<phi, du_inc/dn>.
    rhs_sys = np.zeros((nnodes, elev.size), dtype=np.complex128)
    for elem in mesh.elements:
        ids = np.asarray(elem.node_ids, dtype=int)
        load_dn = _linear_element_incident_dn_load_many(elem, k_air=float(k0), elevations_deg=elev)
        rhs_sys[ids, :] -= load_dn

    # Meixner edge condition: at open-strip endpoints mu -> 0 (H_z is
    # continuous across the strip edge, so the jump density vanishes).
    # See _geometric_sheet_endpoint_nodes for the subtle point about
    # signature-split nodes in stair-stepped tapers.
    endpoint_nodes = _geometric_sheet_endpoint_nodes(mesh)
    if endpoint_nodes.size > 0:
        a_sys[endpoint_nodes, :] = 0.0
        a_sys[endpoint_nodes, endpoint_nodes] = 1.0
        rhs_sys[endpoint_nodes, :] = 0.0

    _ensure_finite_linear_system(a_sys, rhs_sys, label="TE sheet system")
    mu_mat = _solve_dense_system(
        a_sys, rhs_sys, condition_diagnostics, "TE sheet system"
    )
    if mu_mat.ndim == 1:
        mu_mat = mu_mat.reshape(-1, 1)

    residual = np.linalg.norm(a_sys @ mu_mat - rhs_sys, axis=0)
    rhs_norm = np.linalg.norm(rhs_sys, axis=0)
    rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
    residual_vec = residual / rhs_norm

    amp = _farfield_linear_density_many(
        mesh, mu_mat, float(k0), elev, "DLP", order=obs_order
    )

    rcs_lin = _rcs_sigma_from_amp(amp, float(k0))
    return rcs_lin, amp, float(np.max(residual_vec))


def _is_sheet_plus_pec(infos: 'List[PanelCoupledInfo]') -> 'bool':
    """True if every element is either a TYPE 1 sheet or a pure-PEC TYPE 2.

    "Pure-PEC TYPE 2" means bc_kind == 'robin' with zero impedance, i.e., the
    Leontovich coefficient reduces to the Dirichlet (TM) / Neumann (TE)
    limit.  Such elements have no IBC layer -- they're hard PEC surfaces.
    """
    if not infos:
        return False
    has_sheet = False
    has_pec = False
    for info in infos:
        if int(info.seg_type) == 1:
            has_sheet = True
        elif info.bc_kind == 'robin' and abs(complex(info.robin_impedance)) <= EPS:
            has_pec = True
        else:
            # Anything else (dielectric, coated IBC, etc.) disqualifies.
            return False
    return has_sheet and has_pec


def _solve_mixed_sheet_pec(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
    k0: 'float',
    elevations_deg: 'np.ndarray',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    condition_diagnostics: 'Optional[Dict[str, Any]]' = None,
) -> 'Tuple[np.ndarray, np.ndarray, float]':
    """
    Solve mixed TYPE 1 sheet + TYPE 2 PEC body geometries in a single block.

    The key insight is that both the sheet BIE and the PEC body BIE can
    share a single boundary unknown and representation, differing only in a
    coefficient-weighted element mass term:

        TM (single-layer representation, u_s = S sigma):
            PEC   nodes:  row = S                          RHS = -<phi, u_inc>
            sheet nodes:  row = S - (Z_s / jketa) M          RHS = -<phi, u_inc>

            Unified: (S - diag(alpha_TM) M) sigma = -RHS_u
            where alpha_TM[i] = 0 on PEC, Z_s[i]/(jketa) on sheet.

        TE (double-layer representation, u_s = D mu):
            PEC   nodes:  row = N                          RHS = -<phi, dn_u_inc>
            sheet nodes:  row = N - (jk/eta) Z_s M           RHS = -<phi, dn_u_inc>

            Unified: (N - diag(alpha_TE) M) mu = -RHS_dn_u
            where alpha_TE[i] = 0 on PEC, (jk/eta) Z_s[i] on sheet.

    Because both the sheet and the PEC body share the same representation,
    the cross-coupling between sources on one and observations on the other
    is automatic -- the S (resp. N) matrix is assembled over ALL elements
    (sheet + PEC), and the BC is applied row-by-row.

    Caveat: the TM PEC body is solved via plain SLP-EFIE here, which can
    suffer interior-resonance issues for electrically large closed bodies.
    Condition diagnostics and mesh convergence must therefore be retained
    for production use.  Solving the sheet and body separately and adding
    their far fields is NOT a valid workaround in general because it omits
    their mutual multiple-scattering interaction.
    """
    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)

    z_elements = np.asarray([
        complex(info.robin_impedance) if int(info.seg_type) == 1 else 0.0 + 0.0j
        for info in infos
    ], dtype=np.complex128)

    if pol == "TM":
        S_mat, _ = _assemble_linear_operator_matrices(
            mesh, k0, obs_normal_deriv=False,
            obs_order=obs_order, src_order=src_order,
            compute_double_layer=False,
        )
        weighted_mass = _assemble_linear_weighted_mass_matrix(
            mesh, z_elements / (1j * float(k0) * ETA0)
        )
        a_sys = S_mat - weighted_mass

        # RHS: -<phi, u_inc>
        rhs_sys = np.zeros((nnodes, elev.size), dtype=np.complex128)
        for elem in mesh.elements:
            ids = np.asarray(elem.node_ids, dtype=int)
            load_u = _linear_element_incident_load_many(
                elem, k_air=float(k0), elevations_deg=elev,
            )
            rhs_sys[ids, :] -= load_u

        solve_label = "mixed sheet+PEC TM system"
        _ensure_finite_linear_system(a_sys, rhs_sys, label=solve_label)

    else:   # TE
        N_mat = _assemble_linear_hypersingular_matrix(
            mesh, k0, obs_order=obs_order, src_order=src_order,
        )
        # Sign matches _solve_te_sheet: N - (jk/eta)Z_s.M (see derivation there).
        weighted_mass = _assemble_linear_weighted_mass_matrix(
            mesh, (1j * float(k0) / ETA0) * z_elements
        )
        a_sys = N_mat - weighted_mass

        # RHS: -<phi, du_inc/dn>
        rhs_sys = np.zeros((nnodes, elev.size), dtype=np.complex128)
        for elem in mesh.elements:
            ids = np.asarray(elem.node_ids, dtype=int)
            load_dn = _linear_element_incident_dn_load_many(
                elem, k_air=float(k0), elevations_deg=elev,
            )
            rhs_sys[ids, :] -= load_dn

        # Meixner edge condition on open sheet endpoints: mu=0 (H_z continuous
        # at the strip edge).  Applied only to nodes that are geometric
        # endpoints of sheet elements -- not to closed-PEC-body nodes, and not
        # to stair-step-segment-boundary nodes that are geometrically interior
        # but have distinct signatures.  See _geometric_sheet_endpoint_nodes.
        endpoint_nodes = _geometric_sheet_endpoint_nodes(mesh, infos)
        if endpoint_nodes.size > 0:
            a_sys[endpoint_nodes, :] = 0.0
            a_sys[endpoint_nodes, endpoint_nodes] = 1.0
            rhs_sys[endpoint_nodes, :] = 0.0

        solve_label = "mixed sheet+PEC TE system"
        _ensure_finite_linear_system(a_sys, rhs_sys, label=solve_label)

    sol_mat = _solve_dense_system(
        a_sys, rhs_sys, condition_diagnostics, solve_label
    )

    if sol_mat.ndim == 1:
        sol_mat = sol_mat.reshape(-1, 1)

    residual = np.linalg.norm(a_sys @ sol_mat - rhs_sys, axis=0)
    rhs_norm = np.linalg.norm(rhs_sys, axis=0)
    rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
    residual_vec = residual / rhs_norm

    amp = _farfield_linear_density_many(
        mesh, sol_mat, float(k0), elev,
        "SLP" if pol == "TM" else "DLP", order=obs_order,
    )

    rcs_lin = _rcs_sigma_from_amp(amp, float(k0))
    return rcs_lin, amp, float(np.max(residual_vec))



def _assemble_robin_bie_system(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
    k0: 'float',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray]':
    """
    Assemble the all-Robin SLP BIE system shared by the monostatic and
    bistatic solvers (see `_solve_robin_bie` for the derivation).

    Returns (a_sys, alpha_elements, pec_node) where a_sys already contains the
    per-row TM-PEC EFIE override, alpha_elements is the piecewise-constant
    Robin coefficient used inside the weak observation integral, and pec_node
    marks nodes incident on a PEC (Z_s = 0) element.

    INTERIOR RESONANCES (why there is deliberately NO CFIE here): the
    system's conditioning spikes at the cavity's interior Dirichlet
    eigenfrequencies (both TM-EFIE S and TE-MFIE K'-1/2 share that
    resonance set), but with the INDIRECT SLP ansatz the exterior far field
    is immune: a resonant null density sigma_0 has S sigma_0 = 0 on the
    contour, so by exterior uniqueness S sigma_0 vanishes IDENTICALLY
    outside -- the null space radiates nothing, and a direct solve stays
    accurate (validated at the discrete resonance of a PEC circle:
    <= 0.01 dB vs the Mie series with cond spiked 40x;
    tests/validate_2d_resonance.py).  A Robin-style "CFIE" combination of
    trace and normal-derivative rows was tried and REJECTED: with the SLP
    ansatz it imposes a second, physically false boundary condition on PEC
    (u = 0 for TE, du/dn = 0 for TM) and shifts the RCS by ~9.5 dB
    everywhere.  A genuine resonance-free indirect scheme needs the
    Brakhage-Werner combined-SOURCE ansatz (double-layer + hypersingular
    operators) -- only worth building if iterative (GMRES/FMM) solves near
    dense resonance spectra ever stall; direct solves do not need it.
    The public 2-D entry points reject nonzero ``cfie_alpha`` so this setting
    cannot silently leave the physical operator unchanged.
    """

    nnodes = len(mesh.nodes)

    alpha_elements, pec_elements = _robin_alpha_elements(mesh, infos, pol)
    pec_node = np.zeros(nnodes, dtype=bool)
    for eidx, elem in enumerate(mesh.elements):
        if bool(pec_elements[eidx]):
            for nid in elem.node_ids:
                pec_node[int(nid)] = True

    has_ibc = bool(np.any(np.abs(alpha_elements) > EPS))
    all_tm_pec = bool(pol == 'TM' and np.all(pec_elements))

    # Assemble the single-layer and adjoint double-layer operators together.
    # S is independent of the normal-derivative selection, so the former two
    # full passes over every element pair were redundant.  Pure TM-PEC uses
    # only S because every row is replaced by the EFIE limit.
    need_kp = not all_tm_pec
    S_alpha, Kp_mat = _assemble_linear_operator_matrices(
        mesh, k0, obs_normal_deriv=True,
        obs_order=obs_order, src_order=src_order,
        compute_single_layer=(has_ibc or all_tm_pec),
        compute_double_layer=need_kp,
        single_layer_observation_coefficients=(
            alpha_elements if has_ibc and not all_tm_pec else None
        ),
    )
    M_mat = _assemble_linear_mass_matrix(mesh)

    # Default Robin row: (-1/2 M + K' + S_alpha) sigma,
    # where alpha remains inside each observation-element integral.
    a_sys = -0.5 * M_mat + Kp_mat + (S_alpha if has_ibc else 0.0)

    # TM PEC override: replace those rows with the EFIE  S sigma = -u_inc.
    # This is the alpha -> infinity limit of the Robin BIE, divided by alpha
    # to recover a well-conditioned operator.  For TE PEC, alpha = 0 already
    # gives the correct MFIE, so no override is needed there.
    tm_pec_rows = np.flatnonzero(pec_node) if pol == 'TM' else np.zeros(0, dtype=np.int64)
    if tm_pec_rows.size > 0:
        if all_tm_pec:
            S_plain = S_alpha
        else:
            S_plain, _ = _assemble_linear_operator_matrices(
                mesh, k0, obs_normal_deriv=False,
                obs_order=obs_order, src_order=src_order,
                compute_double_layer=False,
            )
        a_sys[tm_pec_rows, :] = S_plain[tm_pec_rows, :]

    return a_sys, alpha_elements, pec_node


def _robin_bie_rhs_many(
    mesh: 'LinearMesh',
    alpha_elements: 'np.ndarray',
    pec_node: 'np.ndarray',
    pol: 'str',
    k0: 'float',
    elevations_deg: 'np.ndarray',
) -> 'np.ndarray':
    """RHS columns for `_assemble_robin_bie_system`, one per elevation angle."""

    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)

    rhs_sys = np.zeros((nnodes, elev.size), dtype=np.complex128)
    tm_pec_rows = (
        np.flatnonzero(pec_node)
        if pol == 'TM' else np.zeros(0, dtype=np.int64)
    )
    all_tm_pec = bool(pol == 'TM' and np.all(pec_node))
    pec_rhs = (
        np.zeros_like(rhs_sys) if tm_pec_rows.size > 0 else None
    )
    alpha_eval = np.asarray(alpha_elements, dtype=np.complex128).reshape(-1)
    if alpha_eval.size != len(mesh.elements):
        raise ValueError("Robin RHS coefficient count must match mesh elements.")
    for eidx, elem in enumerate(mesh.elements):
        ids = np.asarray(elem.node_ids, dtype=int)
        load_u = _linear_element_incident_load_many(elem, k_air=k0, elevations_deg=elev)
        if not all_tm_pec:
            load_dn = _linear_element_incident_dn_load_many(
                elem, k_air=k0, elevations_deg=elev
            )
            rhs_sys[ids, :] -= load_dn
            rhs_sys[ids, :] -= complex(alpha_eval[eidx]) * load_u
        if pec_rhs is not None:
            for local_index, node_id in enumerate(ids):
                if pec_node[int(node_id)]:
                    pec_rhs[int(node_id), :] -= load_u[local_index, :]

    # TM PEC override on RHS: the EFIE row uses -u_inc, not -(du_inc/dn + alpha*u_inc).
    if tm_pec_rows.size > 0:
        rhs_sys[tm_pec_rows, :] = pec_rhs[tm_pec_rows, :]

    return rhs_sys


def _solve_robin_bie(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
    k0: 'float',
    elevations_deg: 'np.ndarray',
    obs_order: 'int' = 8,
    src_order: 'int' = 8,
    condition_diagnostics: 'Optional[Dict[str, Any]]' = None,
) -> 'Tuple[np.ndarray, np.ndarray, float]':
    """
    Solve all-Robin (PEC, IBC, or mixed PEC+IBC) scattering with SLP representation.

    Uses u_scat = SLP(sigma) and the exterior-limit Robin BC
    du/dn + alpha*u = 0 in its element-weighted Galerkin form:

        (-1/2 M + K' + S_alpha) sigma = -(du_inc/dn + alpha*u_inc)

    The Robin coefficient alpha is computed per element (function of pol, the
    medium adjacent to the surface, and Z_s) and retained inside that
    observation element's weak integral. This is the consistent Galerkin form
    for the element-center-sampled impedance model, including tapered IBCs.

    Polarisation/PEC handling (per row, so mixed PEC + IBC at TYPE 2 / TYPE 4
    interfaces are supported correctly):

      - TM PEC node (Z_s = 0):  alpha is formally infinite.  Divide the BC by
        alpha to obtain the well-conditioned EFIE row  S sigma = -u_inc.
      - TM IBC node (Z_s != 0): finite alpha_TM = j k_med eta_med / Z_s.
        Standard Robin BIE row above.
      - TE PEC node (Z_s = 0):  alpha = 0 already gives the correct MFIE row
        (-1/2 M + K') sigma = -du_inc/dn.
      - TE IBC node (Z_s != 0): finite alpha_TE = +j k_med Z_s / eta_med.
        Standard Robin BIE row above.

    Without the TM-PEC row override the alpha->infty limit cannot be taken
    numerically (the original implementation silently degenerated TM PEC to
    the TE MFIE, giving the wrong boundary condition and ~0.3-1.0 dB Mie
    error depending on ka).

    Far-field: SLP projector  A = integral sigma * exp(jk d.r') ds'.
    """

    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    a_sys, alpha_elements, pec_node = _assemble_robin_bie_system(
        mesh, infos, pol, k0, obs_order=obs_order, src_order=src_order)
    rhs_sys = _robin_bie_rhs_many(
        mesh, alpha_elements, pec_node, pol, k0, elev
    )

    _ensure_finite_linear_system(a_sys, rhs_sys, label="Robin-BIE IBC system")
    sigma_mat = _solve_dense_system(
        a_sys, rhs_sys, condition_diagnostics, "Robin-BIE IBC system"
    )
    if sigma_mat.ndim == 1:
        sigma_mat = sigma_mat.reshape(-1, 1)

    # Residual.
    residual = np.linalg.norm(a_sys @ sigma_mat - rhs_sys, axis=0)
    rhs_norm = np.linalg.norm(rhs_sys, axis=0)
    rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
    residual_vec = residual / rhs_norm

    amp = _farfield_linear_density_many(
        mesh, sigma_mat, k0, elev, "SLP", order=obs_order
    )

    rcs_lin = _rcs_sigma_from_amp(amp, k0)
    return rcs_lin, amp, float(np.max(residual_vec))


def _make_elem_mask(elem_ids, n_total):
    mask = np.zeros(n_total, dtype=bool)
    for eidx in elem_ids: mask[eidx] = True
    return mask

def _count_distinct_regions(infos):
    regions = set()
    for info in infos:
        if info.minus_region >= 0: regions.add(info.minus_region)
        if info.plus_region >= 0: regions.add(info.plus_region)
    return len(regions)

def _is_multi_region(infos):
    """True if geometry needs multi-region solver (layered, coated, or mixed PEC+diel).

    Excludes any geometry containing a TYPE 1 sheet -- the multi-region
    indirect solver treats its transmission interfaces as pure-medium
    boundaries and would ignore the sheet's impedance.  In the RCS dispatch,
    supported sheet geometries are routed to the dedicated sheet solvers
    (and unsupported mixes rejected by ``_assert_no_type1_sheet_for_mixed``)
    before this predicate; the ``_has_sheet`` check below is belt-and-braces.
    """
    if _has_sheet(infos):
        return False
    n_regions = _count_distinct_regions(infos)
    if n_regions > 2:
        return True
    # Mixed PEC+dielectric also needs multi-region because interior PEC
    # boundaries require interior wavenumber, not k_air.
    has_transmission = any(info.bc_kind == 'transmission' for info in infos)
    has_robin = any(info.bc_kind == 'robin' for info in infos)
    return has_transmission and has_robin


def _dense_formulation_resources(
    mesh: 'LinearMesh',
    infos: 'List[PanelCoupledInfo]',
    pol: 'str',
) -> 'Dict[str, Any]':
    """Return the dense system size and retained-operator budget.

    This is the single formulation classifier used by both the solver's
    pre-allocation memory gate and the sweep scheduler.  Keeping it beside the
    active dispatch predicates prevents a scheduler estimate from silently
    drifting back to the old generic 2N coupled-system assumption.
    """

    nnodes = int(len(mesh.nodes))
    regions = {
        int(region)
        for info in infos
        for region in (info.minus_region, info.plus_region)
        if int(region) >= 0
    }

    if _is_all_sheet(infos):
        formulation = "sheet"
        system_dofs = nnodes
        operator_matrices = 3
    elif _is_sheet_plus_pec(infos):
        formulation = "mixed_sheet_pec"
        system_dofs = nnodes
        operator_matrices = 3
    elif pol == "TE" and _is_all_robin(infos):
        formulation = "te_robin"
        system_dofs = nnodes
        operator_matrices = 3
    elif _is_multi_region(infos):
        interface_nodes = {}  # type: Dict[Tuple[int, int], Set[int]]
        for elem, info in zip(mesh.elements, infos):
            key = (int(info.minus_region), int(info.plus_region))
            interface_nodes.setdefault(key, set()).update(
                int(node) for node in elem.node_ids
            )
        formulation = "multi_region"
        system_dofs = sum(
            len(nodes) * int(r_minus >= 0)
            + len(nodes) * int(r_plus >= 0)
            for (r_minus, r_plus), nodes in interface_nodes.items()
        )
        operator_matrices = 4 * max(1, len(regions))
    elif _is_single_dielectric_body(infos):
        formulation = "single_dielectric"
        system_dofs = 2 * nnodes
        operator_matrices = 5
    elif _is_all_robin(infos):
        formulation = "robin"
        system_dofs = nnodes
        operator_matrices = 3
    else:
        raise ValueError(
            "Geometry does not match a supported dense 2-D formulation."
        )

    return {
        "nodes": nnodes,
        "n_regions": int(len(regions)),
        "formulation": formulation,
        "system_dofs": int(system_dofs),
        "operator_matrices": int(operator_matrices),
    }

def _solve_multi_region_indirect(
    mesh,
    infos,
    pol,
    k0,
    elevations_deg,
    obs_order=8,
    src_order=8,
    solver_method="auto",
    condition_diagnostics: 'Optional[Dict[str, Any]]' = None,
):
    r"""Multi-region indirect SLP formulation for layered dielectric coatings.

    BIE sign convention (validated against single-region solvers):
    - Element normal n points from minus_region toward plus_region.
    - Density on minus_region side: flux = (-1/2 M + K') * sigma
    - Density on plus_region side:  flux = (+1/2 M + K') * tau
    - Cross-interface operators (source != observer): no +/-1/2 jump.

    When solver_method="fmm", uses FMM-accelerated GMRES instead of dense LU.
    """
    nnodes = len(mesh.nodes)
    elev = np.asarray(elevations_deg, dtype=float).reshape(-1)
    elements = list(mesh.elements)
    nelems = len(elements)

    # 1. Discover regions.
    region_props = {}
    for info in infos:
        for rid, k, eps, mu, has_inc in [
            (info.minus_region, info.k_minus, info.eps_minus, info.mu_minus, info.minus_has_incident),
            (info.plus_region, info.k_plus, info.eps_plus, info.mu_plus, info.plus_has_incident),
        ]:
            if rid >= 0 and rid not in region_props:
                region_props[rid] = {'k': complex(k), 'eps': complex(eps), 'mu': complex(mu), 'has_incident': bool(has_inc)}

    # 2. Discover interfaces.
    iface_elems = {}
    for eidx, info in enumerate(infos):
        iface_elems.setdefault((info.minus_region, info.plus_region), []).append(eidx)

    ifaces = []
    for (r_m, r_p), eids in sorted(iface_elems.items()):
        nodes = sorted({nid for ei in eids for nid in elements[ei].node_ids})
        pec_minus = (r_m < 0)
        pec_plus = (r_p < 0)
        robin_alpha = np.zeros(len(nodes), dtype=np.complex128)
        robin_alpha_elements = np.zeros(nelems, dtype=np.complex128)
        if pec_minus or pec_plus:
            diel_rid = r_p if pec_minus else r_m
            if diel_rid >= 0 and diel_rid in region_props:
                rp = region_props[diel_rid]
                # The element normal points minus_region -> plus_region.  The
                # Robin rows below are written as q + alpha*u = 0, which is the
                # Leontovich BC when the normal points INTO the impedance
                # backing (pec_plus, TYPE 2 orientation).  For pec_minus
                # (TYPE 4) the normal points into the dielectric field region
                # instead, so the flux term flips sign: q - alpha*u = 0.  Bake
                # the side sign into the stored alpha so the matrix rows, RHS,
                # FMM matvec, and preconditioner all stay consistent.
                side_sign = -1.0 if pec_minus else 1.0
                for ei in eids:
                    z_s = complex(infos[ei].robin_impedance)
                    if abs(z_s) > EPS:
                        robin_alpha_elements[ei] = (
                            side_sign
                            * _surface_robin_alpha(
                                pol, rp['eps'], rp['mu'], rp['k'], z_s
                            )
                        )
                # Retained only for the constant-coefficient FMM matvec.  The
                # dense path uses robin_alpha_elements inside the weak
                # observation integral and never row-scales by this array.
                for ni, nid in enumerate(nodes):
                    incident = [
                        robin_alpha_elements[ei]
                        for ei in eids if nid in elements[ei].node_ids
                    ]
                    if incident:
                        robin_alpha[ni] = sum(incident) / len(incident)
        ifaces.append({'r_m': r_m, 'r_p': r_p, 'eids': eids, 'nodes': nodes, 'n': len(nodes),
                       'pec_minus': pec_minus, 'pec_plus': pec_plus,
                       'robin_alpha': robin_alpha,
                       'robin_alpha_elements': robin_alpha_elements,
                       'mask': _make_elem_mask(eids, nelems)})

    region_ifaces = {}
    for mi, ifc in enumerate(ifaces):
        for rid in [ifc['r_m'], ifc['r_p']]:
            if rid >= 0: region_ifaces.setdefault(rid, []).append(mi)

    # 3. DOF layout: one density per dielectric side per interface.
    dof_map = {}
    n_dof = 0
    for mi, ifc in enumerate(ifaces):
        if ifc['r_m'] >= 0:
            dof_map[(mi, 'minus')] = (n_dof, ifc['n']); n_dof += ifc['n']
        if ifc['r_p'] >= 0:
            dof_map[(mi, 'plus')] = (n_dof, ifc['n']); n_dof += ifc['n']

    # 4. Operator cache -- dense or FMM depending on solver_method.
    use_fmm = (isinstance(solver_method, str) and solver_method.strip().lower() == "fmm")
    if use_fmm and condition_diagnostics is not None:
        raise ValueError(
            "compute_condition_number=True is unavailable for the matrix-free "
            "2-D FMM path. Use solver_method='auto' for a dense diagnostic "
            "solve, or explicitly disable the condition-number request."
        )
    if use_fmm:
        for ifc in ifaces:
            if not (ifc['pec_minus'] or ifc['pec_plus']):
                continue
            vals = np.asarray(
                [ifc['robin_alpha_elements'][ei] for ei in ifc['eids']],
                dtype=np.complex128,
            )
            if np.unique(np.round(vals, decimals=14)).size > 1:
                raise ValueError(
                    "Multi-region FMM currently supports only a spatially "
                    "constant Robin coefficient on each PEC-backed interface. "
                    "Tapered or mixed coefficients require the dense "
                    "element-weighted Galerkin path; no nodal row-scaling "
                    "approximation was used."
                )
    M_global = _assemble_linear_mass_matrix(mesh)

    if use_fmm:
        try:
            from fmm_helmholtz_2d import FMMOperator, QuadTree, _build_lists
        except ImportError as exc:
            raise ImportError(
                "Multi-region FMM was explicitly requested, but "
                "fmm_helmholtz_2d.py is unavailable. Refusing an implicit dense "
                "fallback because the caller's FMM memory gate does not certify "
                "that dense allocation is safe; install the FMM backend or retry "
                "with solver_method='auto' so the dense memory gate is applied."
            ) from exc

    if not use_fmm:
        op_cache = {}
        def get_ops(k_val, src_mask):
            key = (complex(k_val), tuple(src_mask.tolist()))
            if key not in op_cache:
                S, Kp = _assemble_linear_operator_matrices(
                    mesh, k_val, True, obs_order, src_order,
                    source_element_mask=src_mask,
                )
                op_cache[key] = (S, Kp)
            return op_cache[key]

        weighted_s_cache = {}
        def get_weighted_s(k_val, src_mask, obs_coeff):
            coeff_eval = np.asarray(obs_coeff, dtype=np.complex128)
            key = (
                complex(k_val), tuple(src_mask.tolist()), coeff_eval.tobytes()
            )
            if key not in weighted_s_cache:
                S_alpha, _ = _assemble_linear_operator_matrices(
                    mesh, k_val, True, obs_order, src_order,
                    source_element_mask=src_mask,
                    compute_double_layer=False,
                    single_layer_observation_coefficients=coeff_eval,
                )
                weighted_s_cache[key] = S_alpha
            return weighted_s_cache[key]

        # Prefetch, grouped by wavenumber.  Every (k, mask) the matrix build
        # will ask for is already determined by the region/interface structure:
        # each region contributes its own wavenumber paired with the mask of
        # every interface touching it.  The masks only select which source
        # elements survive -- they are applied after the quadrature -- so
        # assembling one interface at a time repeats the element-pair sweep,
        # which is ~90% of the cost of a solve on a geometry with materials.
        # Grouping them means that sweep runs once per distinct wavenumber.
        _by_k = {}
        for _rid, _mis in region_ifaces.items():
            _k = complex(region_props[_rid]['k'])
            _slot = _by_k.setdefault(_k, [])
            for _mi in _mis:
                _mask = ifaces[_mi]['mask']
                _key = tuple(_mask.tolist())
                if _key not in {tuple(m.tolist()) for m in _slot}:
                    _slot.append(_mask)
        for _k, _masks in _by_k.items():
            if len(_masks) < 2:
                continue          # nothing to share; let get_ops do it lazily
            for _mask, _ops in zip(_masks, _assemble_linear_operator_matrices_multi(
                mesh=mesh, k0=_k, obs_normal_deriv=True,
                source_element_masks=_masks,
                obs_order=obs_order, src_order=src_order,
            )):
                op_cache[(_k, tuple(_mask.tolist()))] = _ops
        def sub(mat, obs_n, src_n):
            return mat[np.ix_(obs_n, src_n)]
    else:
        # Build tree and lists ONCE, share across all FMM operators.
        _elems = list(mesh.elements)
        _shared_geom = {
            'elements': _elems,
            'centers': np.array([e.center for e in _elems]),
            'lengths': np.array([e.length for e in _elems]),
            'normals': np.array([e.normal for e in _elems]),
            'p0s': np.array([e.p0 for e in _elems]),
            'segs': np.array([e.p1 - e.p0 for e in _elems]),
            'node_ids': np.array([e.node_ids for e in _elems], dtype=int),
        }
        # Fixed leaf size keeps the FMM near-field O(N). The previous
        # nnodes//15 heuristic grew the leaf with the mesh, pinning the quadtree
        # at ~52 leaves for any N, which left the near-field O(N^2) (defeating
        # the FMM). A constant leaf lets the tree refine so near-field work
        # stays proportional to N.
        _max_leaf = 40
        _shared_tree = QuadTree(_shared_geom['centers'], _max_leaf)
        _shared_lists = _build_lists(_shared_tree)

        fmm_cache = {}
        def get_fmm_ops(k_val, src_mask):
            key = (complex(k_val), tuple(src_mask.tolist()))
            if key not in fmm_cache:
                fmm_S = FMMOperator(mesh, k_val, obs_normal_deriv=False,
                                     source_element_mask=src_mask, n_digits=6,
                                     _shared_tree=_shared_tree, _shared_lists=_shared_lists,
                                     _shared_geom=_shared_geom)
                fmm_Kp = FMMOperator(mesh, k_val, obs_normal_deriv=True,
                                      source_element_mask=src_mask, n_digits=6,
                                      _shared_tree=_shared_tree, _shared_lists=_shared_lists,
                                      _shared_geom=_shared_geom)
                fmm_cache[key] = (fmm_S, fmm_Kp)
            return fmm_cache[key]

    # Incident field.
    bu = np.zeros((nnodes, elev.size), dtype=np.complex128)
    bdn = np.zeros((nnodes, elev.size), dtype=np.complex128)
    for elem in elements:
        ids = np.asarray(elem.node_ids, dtype=int)
        bu[ids] += _linear_element_incident_load_many(elem, k_air=k0, elevations_deg=elev)
        bdn[ids] += _linear_element_incident_dn_load_many(elem, k_air=k0, elevations_deg=elev)

    # 5. Assemble RHS (shared between dense and FMM paths).
    Brhs = np.zeros((n_dof, elev.size), dtype=np.complex128)

    for mi, ifc in enumerate(ifaces):
        obs_n = ifc['nodes']; nm = ifc['n']
        r_m, r_p = ifc['r_m'], ifc['r_p']
        alpha = ifc['robin_alpha']

        if ifc['pec_minus'] or ifc['pec_plus']:
            dof_side = 'plus' if ifc['pec_minus'] else 'minus'
            dm = dof_map[(mi, dof_side)]
            rid = r_p if ifc['pec_minus'] else r_m
            # Per-row TM PEC mask: TM nodes with alpha = 0 use EFIE RHS
            # (-u_inc); all other rows use Robin BIE RHS -(q_inc + alpha*u_inc).
            tm_pec_mask = (np.abs(alpha) <= EPS) if pol == 'TM' else np.zeros(nm, dtype=bool)
            if region_props[rid].get('has_incident'):
                # Default: Robin BIE RHS.
                if use_fmm:
                    # FMM reached here only for a constant alpha on this
                    # interface, for which row scaling is the exact weak form.
                    alpha_bu = alpha[:, None] * bu[obs_n]
                else:
                    alpha_bu_global = np.zeros_like(bu)
                    alpha_e = ifc['robin_alpha_elements']
                    for ei in ifc['eids']:
                        elem = elements[ei]
                        ids = np.asarray(elem.node_ids, dtype=int)
                        alpha_bu_global[ids] += (
                            complex(alpha_e[ei])
                            * _linear_element_incident_load_many(
                                elem, k_air=k0, elevations_deg=elev
                            )
                        )
                    alpha_bu = alpha_bu_global[obs_n]
                rhs_block = bdn[obs_n] + alpha_bu
                # TM PEC override (per row).
                if np.any(tm_pec_mask):
                    rhs_block[tm_pec_mask] = bu[obs_n][tm_pec_mask]
                Brhs[dm[0]:dm[0]+nm] -= rhs_block
        else:
            d_sigma = dof_map[(mi, 'minus')]; d_tau = dof_map[(mi, 'plus')]
            if pol == 'TM':
                beta = complex(region_props[r_p]['mu'] / region_props[r_m]['mu']) if abs(region_props[r_m]['mu']) > EPS else 1.0+0j
            else:
                beta = complex(region_props[r_p]['eps'] / region_props[r_m]['eps']) if abs(region_props[r_m]['eps']) > EPS else 1.0+0j
            if abs(beta) <= EPS: beta = 1.0+0j
            inv_beta = 1.0 / beta
            if region_props[r_m].get('has_incident'):
                Brhs[d_sigma[0]:d_sigma[0]+nm] -= bdn[obs_n]
                Brhs[d_tau[0]:d_tau[0]+nm]     -= bu[obs_n]
            if region_props[r_p].get('has_incident'):
                Brhs[d_sigma[0]:d_sigma[0]+nm] += inv_beta * bdn[obs_n]
                Brhs[d_tau[0]:d_tau[0]+nm]     += bu[obs_n]

    if not use_fmm:
        # -- Dense assembly path ------------------------------------------
        Asys = np.zeros((n_dof, n_dof), dtype=np.complex128)
        def sub(mat, obs_n, src_n):
            return mat[np.ix_(obs_n, src_n)]

        def _add_robin_block_dense(mi, ifc, dof_side, region_id, jump_sign):
            dm = dof_map[(mi, dof_side)]
            obs_n = ifc['nodes']; nm = ifc['n']
            k_d = region_props[region_id]['k']
            S_self, Kp_self = get_ops(k_d, ifc['mask'])
            S_alpha_self = get_weighted_s(
                k_d, ifc['mask'], ifc['robin_alpha_elements']
            )
            M_s = sub(M_global, obs_n, obs_n)
            alpha = ifc['robin_alpha']
            # Per-node TM PEC mask: nodes where alpha = 0 and pol = TM use
            # the EFIE row  S sigma = -u_inc (the Z_s -> 0, alpha -> infty
            # limit of the Robin BIE divided through by alpha).  Without
            # this override, alpha = 0 silently collapses to the TE MFIE
            # row, which is the wrong boundary condition for TM PEC and
            # produces ~0.3-1.0 dB Mie error.  For TE the alpha = 0 case
            # already gives the correct MFIE, so no override is needed.
            tm_pec_mask = (np.abs(alpha) <= EPS) if pol == 'TM' else np.zeros(nm, dtype=bool)

            S_sub = sub(S_self, obs_n, obs_n)
            Kp_sub = sub(Kp_self, obs_n, obs_n)
            block = (
                jump_sign * 0.5 * M_s
                + Kp_sub
                + sub(S_alpha_self, obs_n, obs_n)
            )
            if np.any(tm_pec_mask):
                block[tm_pec_mask, :] = S_sub[tm_pec_mask, :]
            Asys[dm[0]:dm[0]+nm, dm[0]:dm[0]+nm] += block

            for mj in region_ifaces.get(region_id, []):
                if mj == mi: continue
                ifj = ifaces[mj]
                side_j = 'minus' if ifj['r_m'] == region_id else 'plus'
                dj = dof_map.get((mj, side_j))
                if dj is None: continue
                S_x, Kp_x = get_ops(k_d, ifj['mask']); src_n = ifj['nodes']
                S_alpha_x = get_weighted_s(
                    k_d, ifj['mask'], ifc['robin_alpha_elements']
                )
                S_x_sub = sub(S_x, obs_n, src_n)
                Kp_x_sub = sub(Kp_x, obs_n, src_n)
                cross_block = (
                    Kp_x_sub + sub(S_alpha_x, obs_n, src_n)
                )
                if np.any(tm_pec_mask):
                    cross_block[tm_pec_mask, :] = S_x_sub[tm_pec_mask, :]
                Asys[dm[0]:dm[0]+nm, dj[0]:dj[0]+dj[1]] += cross_block

        for mi, ifc in enumerate(ifaces):
            r_m, r_p = ifc['r_m'], ifc['r_p']
            if ifc['pec_minus']:
                _add_robin_block_dense(mi, ifc, 'plus', r_p, +1.0)
            elif ifc['pec_plus']:
                _add_robin_block_dense(mi, ifc, 'minus', r_m, -1.0)
            else:
                obs_n = ifc['nodes']; nm = ifc['n']
                d_sigma = dof_map[(mi, 'minus')]; d_tau = dof_map[(mi, 'plus')]
                k_m_val = region_props[r_m]['k']; k_p_val = region_props[r_p]['k']
                if pol == 'TM':
                    beta = complex(region_props[r_p]['mu'] / region_props[r_m]['mu']) if abs(region_props[r_m]['mu']) > EPS else 1.0+0j
                else:
                    beta = complex(region_props[r_p]['eps'] / region_props[r_m]['eps']) if abs(region_props[r_m]['eps']) > EPS else 1.0+0j
                if abs(beta) <= EPS: beta = 1.0+0j
                inv_beta = 1.0 / beta
                S_m, Kp_m = get_ops(k_m_val, ifc['mask'])
                S_p, Kp_p = get_ops(k_p_val, ifc['mask'])
                M_s = sub(M_global, obs_n, obs_n)
                Asys[d_sigma[0]:d_sigma[0]+nm, d_sigma[0]:d_sigma[0]+nm] += -0.5*M_s + sub(Kp_m, obs_n, obs_n)
                Asys[d_sigma[0]:d_sigma[0]+nm, d_tau[0]:d_tau[0]+nm]     -= inv_beta*(0.5*M_s + sub(Kp_p, obs_n, obs_n))
                Asys[d_tau[0]:d_tau[0]+nm, d_sigma[0]:d_sigma[0]+nm]     += sub(S_m, obs_n, obs_n)
                Asys[d_tau[0]:d_tau[0]+nm, d_tau[0]:d_tau[0]+nm]         -= sub(S_p, obs_n, obs_n)
                for mj in region_ifaces.get(r_m, []):
                    if mj == mi: continue
                    ifj = ifaces[mj]; side_j = 'minus' if ifj['r_m'] == r_m else 'plus'
                    dj = dof_map.get((mj, side_j))
                    if dj is None: continue
                    S_x, Kp_x = get_ops(k_m_val, ifj['mask']); src_n = ifj['nodes']
                    Asys[d_sigma[0]:d_sigma[0]+nm, dj[0]:dj[0]+dj[1]] += sub(Kp_x, obs_n, src_n)
                    Asys[d_tau[0]:d_tau[0]+nm, dj[0]:dj[0]+dj[1]]     += sub(S_x, obs_n, src_n)
                for mj in region_ifaces.get(r_p, []):
                    if mj == mi: continue
                    ifj = ifaces[mj]; side_j = 'minus' if ifj['r_m'] == r_p else 'plus'
                    dj = dof_map.get((mj, side_j))
                    if dj is None: continue
                    S_x, Kp_x = get_ops(k_p_val, ifj['mask']); src_n = ifj['nodes']
                    Asys[d_sigma[0]:d_sigma[0]+nm, dj[0]:dj[0]+dj[1]] -= inv_beta * sub(Kp_x, obs_n, src_n)
                    Asys[d_tau[0]:d_tau[0]+nm, dj[0]:dj[0]+dj[1]]     -= sub(S_x, obs_n, src_n)

        # 6. Solve (dense).
        _ensure_finite_linear_system(Asys, Brhs, label="multi-region indirect system")
        sol = _solve_dense_system(
            Asys, Brhs, condition_diagnostics,
            "multi-region indirect system",
        )
        if sol.ndim == 1: sol = sol.reshape(-1, 1)
        residual = np.linalg.norm(Asys @ sol - Brhs, axis=0)
        rhs_norm = np.linalg.norm(Brhs, axis=0)
        rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
        max_res = float(np.max(residual / rhs_norm))
    else:
        # -- FMM matvec path ----------------------------------------------
        def _fmm_apply(fmm_op, src_nodes, obs_nodes, x_block):
            """Embed block density into global, apply FMM, extract obs nodes."""
            x_global = np.zeros(nnodes, dtype=np.complex128)
            x_global[src_nodes] = x_block
            y_global = fmm_op.matvec(x_global)
            return y_global[obs_nodes]

        # Precompute all FMM operators needed.
        for mi, ifc in enumerate(ifaces):
            r_m, r_p = ifc['r_m'], ifc['r_p']
            if ifc['pec_minus'] and r_p >= 0:
                get_fmm_ops(region_props[r_p]['k'], ifc['mask'])
            elif ifc['pec_plus'] and r_m >= 0:
                get_fmm_ops(region_props[r_m]['k'], ifc['mask'])
            else:
                if r_m >= 0: get_fmm_ops(region_props[r_m]['k'], ifc['mask'])
                if r_p >= 0: get_fmm_ops(region_props[r_p]['k'], ifc['mask'])
            # Cross-interface operators.
            for rid in [r_m, r_p]:
                if rid < 0 or rid not in region_props: continue
                for mj in region_ifaces.get(rid, []):
                    if mj == mi: continue
                    get_fmm_ops(region_props[rid]['k'], ifaces[mj]['mask'])

        def block_matvec(x_vec):
            """Compute Asys @ x using FMM operators."""
            y = np.zeros(n_dof, dtype=np.complex128)
            for mi, ifc in enumerate(ifaces):
                obs_n = np.array(ifc['nodes'], dtype=int); nm = ifc['n']
                r_m, r_p = ifc['r_m'], ifc['r_p']
                alpha = ifc['robin_alpha']

                if ifc['pec_minus'] or ifc['pec_plus']:
                    dof_side = 'plus' if ifc['pec_minus'] else 'minus'
                    region_id = r_p if ifc['pec_minus'] else r_m
                    jump_sign = +1.0 if ifc['pec_minus'] else -1.0
                    dm = dof_map[(mi, dof_side)]
                    k_d = region_props[region_id]['k']
                    fmm_S, fmm_Kp = get_fmm_ops(k_d, ifc['mask'])
                    M_s = M_global[np.ix_(obs_n, obs_n)]
                    # Per-node TM PEC mask (see dense path for derivation).
                    tm_pec_mask = (np.abs(alpha) <= EPS) if pol == 'TM' else np.zeros(nm, dtype=bool)
                    x_blk = x_vec[dm[0]:dm[0]+nm]
                    S_y = _fmm_apply(fmm_S, obs_n, obs_n, x_blk)
                    Kp_y = _fmm_apply(fmm_Kp, obs_n, obs_n, x_blk)
                    # Default Robin BIE rows: jump*1/2M + K' + alpha*S.
                    y_block = jump_sign * 0.5 * (M_s @ x_blk) + Kp_y + alpha * S_y
                    # TM PEC override: those rows use S sigma instead.
                    if np.any(tm_pec_mask):
                        y_block[tm_pec_mask] = S_y[tm_pec_mask]
                    y[dm[0]:dm[0]+nm] += y_block
                    for mj in region_ifaces.get(region_id, []):
                        if mj == mi: continue
                        ifj = ifaces[mj]
                        side_j = 'minus' if ifj['r_m'] == region_id else 'plus'
                        dj = dof_map.get((mj, side_j))
                        if dj is None: continue
                        src_n = np.array(ifj['nodes'], dtype=int)
                        fmm_Sx, fmm_Kpx = get_fmm_ops(k_d, ifj['mask'])
                        x_cross = x_vec[dj[0]:dj[0]+dj[1]]
                        S_cross = _fmm_apply(fmm_Sx, src_n, obs_n, x_cross)
                        Kp_cross = _fmm_apply(fmm_Kpx, src_n, obs_n, x_cross)
                        cross_y = Kp_cross + alpha * S_cross
                        if np.any(tm_pec_mask):
                            cross_y[tm_pec_mask] = S_cross[tm_pec_mask]
                        y[dm[0]:dm[0]+nm] += cross_y
                else:
                    # Transmission.
                    d_sigma = dof_map[(mi, 'minus')]; d_tau = dof_map[(mi, 'plus')]
                    k_m_val = region_props[r_m]['k']; k_p_val = region_props[r_p]['k']
                    if pol == 'TM':
                        beta = complex(region_props[r_p]['mu'] / region_props[r_m]['mu']) if abs(region_props[r_m]['mu']) > EPS else 1.0+0j
                    else:
                        beta = complex(region_props[r_p]['eps'] / region_props[r_m]['eps']) if abs(region_props[r_m]['eps']) > EPS else 1.0+0j
                    if abs(beta) <= EPS: beta = 1.0+0j
                    inv_beta = 1.0 / beta
                    fmm_Sm, fmm_Kpm = get_fmm_ops(k_m_val, ifc['mask'])
                    fmm_Sp, fmm_Kpp = get_fmm_ops(k_p_val, ifc['mask'])
                    M_s = M_global[np.ix_(obs_n, obs_n)]
                    x_sig = x_vec[d_sigma[0]:d_sigma[0]+nm]; x_tau = x_vec[d_tau[0]:d_tau[0]+nm]
                    # Flux row.
                    y[d_sigma[0]:d_sigma[0]+nm] += (
                        -0.5*(M_s @ x_sig) + _fmm_apply(fmm_Kpm, obs_n, obs_n, x_sig)
                        - inv_beta*(0.5*(M_s @ x_tau) + _fmm_apply(fmm_Kpp, obs_n, obs_n, x_tau)))
                    # Trace row.
                    y[d_tau[0]:d_tau[0]+nm] += (
                        _fmm_apply(fmm_Sm, obs_n, obs_n, x_sig)
                        - _fmm_apply(fmm_Sp, obs_n, obs_n, x_tau))
                    # Cross from r_minus.
                    for mj in region_ifaces.get(r_m, []):
                        if mj == mi: continue
                        ifj = ifaces[mj]; side_j = 'minus' if ifj['r_m'] == r_m else 'plus'
                        dj = dof_map.get((mj, side_j))
                        if dj is None: continue
                        src_n = np.array(ifj['nodes'], dtype=int)
                        fmm_Sx, fmm_Kpx = get_fmm_ops(k_m_val, ifj['mask'])
                        x_cross = x_vec[dj[0]:dj[0]+dj[1]]
                        y[d_sigma[0]:d_sigma[0]+nm] += _fmm_apply(fmm_Kpx, src_n, obs_n, x_cross)
                        y[d_tau[0]:d_tau[0]+nm]      += _fmm_apply(fmm_Sx, src_n, obs_n, x_cross)
                    # Cross from r_plus.
                    for mj in region_ifaces.get(r_p, []):
                        if mj == mi: continue
                        ifj = ifaces[mj]; side_j = 'minus' if ifj['r_m'] == r_p else 'plus'
                        dj = dof_map.get((mj, side_j))
                        if dj is None: continue
                        src_n = np.array(ifj['nodes'], dtype=int)
                        fmm_Sx, fmm_Kpx = get_fmm_ops(k_p_val, ifj['mask'])
                        x_cross = x_vec[dj[0]:dj[0]+dj[1]]
                        y[d_sigma[0]:d_sigma[0]+nm] -= inv_beta * _fmm_apply(fmm_Kpx, src_n, obs_n, x_cross)
                        y[d_tau[0]:d_tau[0]+nm]      -= _fmm_apply(fmm_Sx, src_n, obs_n, x_cross)
            return y

        # 6. Build block-diagonal preconditioner from near-field self-interaction.
        # One dense block per interface (its dof ranges are contiguous by
        # construction of dof_map), factored independently -- storing the full
        # (n_dof x n_dof) matrix here would reintroduce the dense-memory
        # footprint the FMM path exists to avoid.
        precond_blocks: 'List[Tuple[int, np.ndarray]]' = []  # (dof_start, block)
        for mi, ifc in enumerate(ifaces):
            obs_n = np.array(ifc['nodes'], dtype=int); nm = ifc['n']
            r_m, r_p = ifc['r_m'], ifc['r_p']
            alpha = ifc['robin_alpha']
            if ifc['pec_minus'] or ifc['pec_plus']:
                dof_side = 'plus' if ifc['pec_minus'] else 'minus'
                region_id = r_p if ifc['pec_minus'] else r_m
                jump_sign = +1.0 if ifc['pec_minus'] else -1.0
                dm = dof_map[(mi, dof_side)]
                k_d = region_props[region_id]['k']
                fmm_S, fmm_Kp = get_fmm_ops(k_d, ifc['mask'])
                M_s = M_global[np.ix_(obs_n, obs_n)]
                # Per-node TM PEC mask (see dense path for derivation).
                tm_pec_mask = (np.abs(alpha) <= EPS) if pol == 'TM' else np.zeros(nm, dtype=bool)
                # Extract dense submatrices from the sparse CSR near-field operators.
                # csr_matrix[np.ix_(...)] returns a CSR slice; assigning that into a
                # dense Pdiag block raises "must be real number, not csr_matrix".
                S_sub = fmm_S._near_mat[np.ix_(obs_n, obs_n)].toarray()
                Kp_sub = fmm_Kp._near_mat[np.ix_(obs_n, obs_n)].toarray()
                block = jump_sign * 0.5 * M_s + Kp_sub + alpha[:, None] * S_sub
                if np.any(tm_pec_mask):
                    block[tm_pec_mask, :] = S_sub[tm_pec_mask, :]
                precond_blocks.append((dm[0], block))
            else:
                d_sigma = dof_map[(mi, 'minus')]; d_tau = dof_map[(mi, 'plus')]
                k_m_val = region_props[r_m]['k']; k_p_val = region_props[r_p]['k']
                if pol == 'TM':
                    beta = complex(region_props[r_p]['mu'] / region_props[r_m]['mu']) if abs(region_props[r_m]['mu']) > EPS else 1.0+0j
                else:
                    beta = complex(region_props[r_p]['eps'] / region_props[r_m]['eps']) if abs(region_props[r_m]['eps']) > EPS else 1.0+0j
                if abs(beta) <= EPS: beta = 1.0+0j
                inv_beta = 1.0 / beta
                fmm_Sm, fmm_Kpm = get_fmm_ops(k_m_val, ifc['mask'])
                fmm_Sp, fmm_Kpp = get_fmm_ops(k_p_val, ifc['mask'])
                M_s = M_global[np.ix_(obs_n, obs_n)]
                # Dense submatrix extracts -- see comment above for why .toarray() is required.
                Sm_sub = fmm_Sm._near_mat[np.ix_(obs_n, obs_n)].toarray()
                Sp_sub = fmm_Sp._near_mat[np.ix_(obs_n, obs_n)].toarray()
                Kpm_sub = fmm_Kpm._near_mat[np.ix_(obs_n, obs_n)].toarray()
                Kpp_sub = fmm_Kpp._near_mat[np.ix_(obs_n, obs_n)].toarray()
                # sigma/tau dof ranges of one interface are contiguous
                # (d_tau starts right after d_sigma), so the coupled 2x2
                # block occupies one contiguous diagonal block.
                blk = np.zeros((2 * nm, 2 * nm), dtype=np.complex128)
                blk[:nm, :nm] = -0.5 * M_s + Kpm_sub
                blk[:nm, nm:] = -inv_beta * (0.5 * M_s + Kpp_sub)
                blk[nm:, :nm] = Sm_sub
                blk[nm:, nm:] = -Sp_sub
                precond_blocks.append((d_sigma[0], blk))

        # LU-factor each diagonal block independently.
        try:
            from scipy.linalg import lu_factor, lu_solve
            factored_blocks = [
                (start, block.shape[0], lu_factor(block))
                for start, block in precond_blocks
            ]
            def precond_matvec(x):
                y = np.array(x, dtype=np.complex128, copy=True)
                for start, size, lu in factored_blocks:
                    y[start:start + size] = lu_solve(lu, x[start:start + size])
                return y
            M_precond = _SCIPY_SPARSE_LINALG.LinearOperator(
                (n_dof, n_dof), matvec=precond_matvec, dtype=np.complex128)
        except Exception:
            M_precond = None

        # 7. Solve with preconditioned GMRES.
        if _SCIPY_SPARSE_LINALG is None:
            raise ImportError("FMM solver requires scipy.sparse.linalg for GMRES.")
        A_op = _SCIPY_SPARSE_LINALG.LinearOperator(
            (n_dof, n_dof), matvec=block_matvec, dtype=np.complex128)
        sol = _solve_fmm_gmres_columns(
            A_op,
            Brhs,
            elev,
            formulation="multi-region indirect",
            restart=80,
            maxiter=500,
            rtol=1.0e-10,
            preconditioner=M_precond,
        )
        residual = np.zeros(elev.size)
        for col in range(elev.size):
            residual[col] = np.linalg.norm(Brhs[:, col] - block_matvec(sol[:, col]))
        rhs_norm = np.linalg.norm(Brhs, axis=0)
        rhs_norm = np.where(rhs_norm <= EPS, 1.0, rhs_norm)
        max_res = float(np.max(residual / rhs_norm))

    # 7. Extract exterior SLP density mapped to global node IDs.
    ext_rid = next((rid for rid, rp in region_props.items() if rp.get('has_incident')), 0)
    ext_density_global = np.zeros((nnodes, elev.size), dtype=np.complex128)
    ext_elem_mask = np.zeros(nelems, dtype=bool)

    for mi, ifc in enumerate(ifaces):
        if ifc['r_m'] == ext_rid:
            side = 'minus'
        elif ifc['r_p'] == ext_rid:
            side = 'plus'
        else:
            continue
        dm = dof_map.get((mi, side))
        if dm is None:
            continue
        density = sol[dm[0]:dm[0]+dm[1], :]
        for li, nid in enumerate(ifc['nodes']):
            ext_density_global[nid, :] += density[li, :]
        for eidx in ifc['eids']:
            ext_elem_mask[eidx] = True

    # 8. Far-field from exterior SLP density.
    amp = _farfield_linear_density_many(
        mesh,
        ext_density_global,
        k0,
        elev,
        "SLP",
        order=obs_order,
        element_mask=ext_elem_mask,
    )

    rcs_lin = _rcs_sigma_from_amp(amp, k0)
    return rcs_lin, amp, max_res, ext_density_global

def solve_monostatic_rcs_2d(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    elevations_deg: 'List[float]',
    polarization: 'str',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    quality_thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
    strict_quality_gate: 'bool' = True,
    compute_condition_number: 'bool' = False,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    rcs_normalization_mode: 'str' = RCS_NORM_MODE_DEFAULT,
    cfie_alpha: 'float' = CFIE_ALPHA_DEFAULT,
    abort_event: 'Optional[threading.Event]' = None,
    solver_method: 'str' = "auto",
) -> 'Dict[str, Any]':
    """
    Low-level monostatic 2-D RCS solve using the formulation selected for the
    supplied boundary types (Robin/MFIE, dielectric indirect, multi-region,
    or sheet BIE).

    Per frequency:
    - build the boundary discretization,
    - assemble the selected linear/Galerkin boundary-integral system,
    - solve all requested elevations,
    - compute monostatic backscatter RCS.

    The returned quality gate certifies the assembled discrete linear system,
    not mesh convergence.  Production callers must perform the base/fine
    complex-field mesh comparison implemented by ``step1_monostatic``.

    Angle convention (coming-from):
    - 0 deg: from right to left
    - +90 deg: from top to bottom
    - -90 deg: from bottom to top
    """

    if not frequencies_ghz:
        raise ValueError("At least one frequency is required.")
    if not elevations_deg:
        raise ValueError("At least one elevation angle is required.")

    frequencies = [float(f) for f in frequencies_ghz]
    elevations = [float(e) for e in elevations_deg]
    if any((not math.isfinite(f)) or f <= 0.0 for f in frequencies):
        raise ValueError("Frequencies must be positive finite GHz values.")
    if any(not math.isfinite(e) for e in elevations):
        raise ValueError("Elevation angles must all be finite.")

    mesh_ref_ghz: 'Optional[float]' = None
    if mesh_reference_ghz is not None:
        mesh_ref_ghz = float(mesh_reference_ghz)
        if (not math.isfinite(mesh_ref_ghz)) or mesh_ref_ghz <= 0.0:
            raise ValueError("mesh_reference_ghz must be a positive finite GHz value.")

    rcs_norm_mode = _normalize_rcs_normalization_mode(rcs_normalization_mode)
    solver_method = _normalize_public_2d_solver_method(solver_method)
    _raise_if_untrusted_math_backends()

    pol = _normalize_polarization(polarization)
    unit_scale = _unit_scale_to_meters(geometry_units)

    base_dir = _material_base_dir_for_snapshot(
        geometry_snapshot, material_base_dir
    )
    preflight_report = validate_geometry_snapshot_for_solver(geometry_snapshot, base_dir=base_dir, meters_scale=unit_scale)
    materials = MaterialLibrary.from_entries(
        geometry_snapshot.get("ibcs", []) or [],
        geometry_snapshot.get("dielectrics", []) or [],
        base_dir=base_dir,
    )
    for _msg in list(preflight_report.get('warnings', []) or []):
        materials.warn_once(str(_msg))
    _warn_far_quadrature_override(materials)

    samples: 'List[Dict[str, Any]]' = []
    total_steps = len(frequencies) * (len(elevations) + 1)
    done_steps = 0

    residual_values: 'List[float]' = []
    constraint_residual_values: 'List[float]' = []
    cond_values: 'List[float]' = []
    mesh_reference_values: 'List[float]' = []
    mesh_wavelength_values: 'List[float]' = []
    mesh_max_index_values: 'List[float]' = []
    mesh_material_flags_used: 'Set[int]' = set()
    panel_count_values: 'List[int]' = []
    panel_length_min_values: 'List[float]' = []
    panel_length_max_values: 'List[float]' = []
    elevations_arr = np.asarray(elevations, dtype=float)
    reused_matrix_solve_count = 0
    max_parallel_workers_used = 1
    formulation_label = "2D BIE/MoM coupled dielectric trace formulation (linear Galerkin)"
    junction_stats = {
        "junction_nodes": 0,
        "junction_constraints": 0,
        "junction_panels": 0,
        "junction_trace_constraints": 0,
        "junction_flux_constraints": 0,
        "junction_orientation_conflict_nodes": 0,
    }

    def emit_progress(message: 'str') -> 'None':
        if progress_callback is None:
            return
        try:
            progress_callback(done_steps, total_steps, message)
        except Exception:
            pass

    def check_abort() -> 'None':
        if abort_event is not None and abort_event.is_set():
            raise InterruptedError("Solve cancelled by user.")

    check_abort()
    emit_progress("Initializing solver")

    if abs(float(cfie_alpha)) > EPS:
        raise ValueError(
            "cfie_alpha is not implemented by any active 2-D formulation; "
            "use cfie_alpha=0. No unchanged field was returned under a "
            "different solver setting."
        )

    # --- Mesh caching: when mesh_reference_ghz is set, the mesh topology is
    # frequency-independent and can be built once before the frequency loop. ---
    cached_panels: 'Optional[List[Any]]' = None
    cached_mesh: 'Any' = None
    cached_mesh_stats: 'Optional[Dict[str, Any]]' = None
    cached_junction_constraints: 'Optional[np.ndarray]' = None
    cached_junction_stats: 'Optional[Dict[str, Any]]' = None
    cached_mesh_wavelength: 'Optional[float]' = None
    cached_mesh_max_index: 'Optional[float]' = None
    cached_mesh_material_flags: 'List[int]' = []

    if mesh_ref_ghz is not None and len(frequencies) > 1:
        (
            ref_lambda,
            cached_mesh_max_index,
            cached_mesh_material_flags,
        ) = _conservative_mesh_wavelength_for_frequencies(
            geometry_snapshot,
            materials,
            set(frequencies) | {mesh_ref_ghz},
        )
        cached_mesh_wavelength = ref_lambda
        ref_k0 = 2.0 * math.pi * mesh_ref_ghz * 1e9 / C0
        cached_panels = _build_panels(
            geometry_snapshot, unit_scale, ref_lambda, max_panels=max_panels,
        )
        # Build preview infos at reference frequency for interface-aware splitting.
        ref_infos = _build_coupled_panel_info(cached_panels, materials, mesh_ref_ghz, pol, ref_k0)
        cached_mesh, cached_mesh_stats = _build_linear_mesh_interface_aware(
            cached_panels, ref_infos,
        )
        cached_mesh_stats = dict(cached_mesh_stats)
        cached_mesh_stats.update(_linear_coupled_node_report(
            cached_mesh,
            _build_linear_coupled_infos(cached_mesh, materials, mesh_ref_ghz, pol, ref_k0),
        ))
        # Junction constraints depend on coupled_infos which may be freq-dependent.
        # Build once at reference freq; topology-based constraints are stable.
        ref_coupled = _build_linear_coupled_infos(cached_mesh, materials, mesh_ref_ghz, pol, ref_k0)
        cached_junction_constraints, cached_junction_stats = _build_linear_junction_constraints(
            cached_mesh, ref_coupled,
        )
        materials.warn_once(
            f"Mesh topology cached using the shortest referenced-material "
            f"wavelength across the requested frequencies and the "
            f"{mesh_ref_ghz:g} GHz reference "
            f"({len(cached_panels)} panels, {len(cached_mesh.nodes)} nodes). "
            f"Reusing for {len(frequencies)} frequencies."
        )

    for freq_ghz in frequencies:
        check_abort()
        freq_hz = freq_ghz * 1e9
        k0 = 2.0 * math.pi * freq_hz / C0
        mesh_freq_ghz = mesh_ref_ghz if mesh_ref_ghz is not None else float(freq_ghz)

        if cached_panels is not None and cached_mesh is not None:
            panels = cached_panels
            mesh = cached_mesh
            linear_mesh_stats_local = dict(cached_mesh_stats or {})
            lambda_min = float(cached_mesh_wavelength)
            mesh_max_index = float(cached_mesh_max_index)
            mesh_material_flags = list(cached_mesh_material_flags)
        else:
            (
                lambda_min,
                mesh_max_index,
                mesh_material_flags,
            ) = _mesh_wavelength_for_snapshot(
                geometry_snapshot, materials, mesh_freq_ghz
            )
            panels = _build_panels(
                geometry_snapshot, unit_scale, lambda_min, max_panels=max_panels,
            )
            preview_infos = _build_coupled_panel_info(panels, materials, freq_ghz, pol, k0)
            mesh, linear_mesh_stats_local = _build_linear_mesh_interface_aware(panels, preview_infos)
            linear_mesh_stats_local = dict(linear_mesh_stats_local)

        panel_lengths = np.asarray([p.length for p in panels], dtype=float)
        mesh_reference_values.append(float(mesh_freq_ghz))
        mesh_wavelength_values.append(float(lambda_min))
        mesh_max_index_values.append(float(mesh_max_index))
        mesh_material_flags_used.update(int(flag) for flag in mesh_material_flags)
        panel_count_values.append(int(len(panels)))
        panel_length_min_values.append(float(np.min(panel_lengths)) if len(panel_lengths) else 0.0)
        panel_length_max_values.append(float(np.max(panel_lengths)) if len(panel_lengths) else 0.0)

        coupled_infos = _build_linear_coupled_infos(mesh, materials, freq_ghz, pol, k0)
        _assert_no_type1_sheet_for_mixed(coupled_infos)
        _assert_air_exterior(coupled_infos)
        _assert_supported_te_type2_contours(mesh, coupled_infos, pol)
        if solver_method == "fmm" and not (
            (pol == "TE" and _is_all_robin(coupled_infos))
            or _is_multi_region(coupled_infos)
        ):
            raise ValueError(
                "solver_method='fmm' is implemented only for monostatic "
                "TE-Robin (PEC/IBC) and multi-region 2-D formulations. "
                "This geometry/polarization requires solver_method='auto' "
                "or 'direct'; no dense fallback was performed."
            )

        # Refuse before any formulation allocates its dense operators.  The
        # classifier is shared with the scheduler, including N-DOF sheet and
        # Robin systems and exact interface-side DOFs for multi-region solves.
        resources = _dense_formulation_resources(mesh, coupled_infos, pol)
        est_gb = _estimate_memory_gb(
            resources["nodes"],
            use_cfie=False,
            n_regions=max(1, resources["n_regions"]),
            system_dofs=resources["system_dofs"],
            operator_matrices=resources["operator_matrices"],
            n_rhs=max(1, len(elevations)),
        )
        fmm_requested = solver_method == "fmm"
        fmm_capable = (
            (pol == 'TE' and _is_all_robin(coupled_infos))
            or _is_multi_region(coupled_infos)
        )
        dense_gate_active = not (fmm_requested and fmm_capable)
        memory_limit_gb = _solve_memory_limit_gb()
        if est_gb > memory_limit_gb and dense_gate_active:
            raise MemoryError(
                f"Estimated peak memory {est_gb:.1f} GB exceeds the "
                f"{memory_limit_gb:.1f} GB limit for this process "
                f"({resources['system_dofs']} system DOFs, "
                f"{resources['n_regions']} region(s), "
                f"{resources['formulation']}; "
                f"{_detect_available_gb():.1f} GB detected). "
                f"Reduce panel count, frequency, use mesh_reference_ghz, "
                f"raise GHOST_MAX_SOLVE_GB if the memory really is there, or "
                f"solver_method='fmm' for TE all-Robin / multi-region problems."
            )
        if est_gb > 8.0 and dense_gate_active:
            materials.warn_once(
                f"Estimated peak memory {est_gb:.1f} GB for "
                f"{resources['system_dofs']} {resources['formulation']} "
                "system DOFs. Large problems may cause slowdowns or "
                "out-of-memory errors."
            )

        # --- TYPE 1 sheet dispatch ---
        # Pure-sheet geometries use a dedicated sheet BIE derived directly
        # from Maxwell's equations (see _solve_tm_sheet / _solve_te_sheet).
        # Mixed sheet+body is rejected above by the _for_mixed guard.
        if _is_all_sheet(coupled_infos):
            formulation_label = (
                "2D sheet BIE (TM: single-layer representation)"
                if pol == "TM"
                else "2D sheet BIE (TE: double-layer / hypersingular representation)"
            )
            sheet_solver = _solve_tm_sheet if pol == "TM" else _solve_te_sheet
            condition_diagnostics = {} if compute_condition_number else None
            rcs_lin_vec, amp_vec, sheet_residual = sheet_solver(
                mesh=mesh,
                infos=coupled_infos,
                k0=k0,
                elevations_deg=elevations_arr,
                condition_diagnostics=condition_diagnostics,
            )
            rcs_db_vec = _rcs_db_from_sigma(rcs_lin_vec)
            residual_vec = np.full(len(elevations), sheet_residual, dtype=float)
            constraint_residual_vec = np.zeros(len(elevations), dtype=float)
            _consume_condition_estimate(
                cond_values, condition_diagnostics, formulation_label
            )
            reused_matrix_solve_count += len(elevations)

            for idx, elev_deg in enumerate(elevations):
                amp_val = complex(amp_vec[idx])
                residual_local = float(residual_vec[idx])
                samples.append({
                    "frequency_ghz": float(freq_ghz),
                    "theta_inc_deg": float(elev_deg),
                    "theta_scat_deg": float(elev_deg),
                    "rcs_linear": float(rcs_lin_vec[idx]),
                    "rcs_db": float(rcs_db_vec[idx]),
                    "rcs_amp_real": float(np.real(amp_val)),
                    "rcs_amp_imag": float(np.imag(amp_val)),
                    "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                    "linear_residual": residual_local,
                })
                residual_values.append(residual_local)
                constraint_residual_values.append(0.0)
                done_steps += 1
                emit_progress(f"Sheet BIE solved {freq_ghz:g} GHz at {elev_deg:g} deg")
            continue
        # --- Mixed TYPE 1 sheet + TYPE 2 PEC dispatch ---
        # Handled with a unified SLP (TM) or DLP (TE) representation over
        # both surface types with an element-weighted impedance term.
        if _is_sheet_plus_pec(coupled_infos):
            formulation_label = (
                "2D mixed sheet+PEC BIE (TM: unified SLP representation)"
                if pol == "TM"
                else "2D mixed sheet+PEC BIE (TE: unified DLP / hypersingular representation)"
            )
            condition_diagnostics = {} if compute_condition_number else None
            rcs_lin_vec, amp_vec, mixed_residual = _solve_mixed_sheet_pec(
                mesh=mesh, infos=coupled_infos, pol=pol,
                k0=k0, elevations_deg=elevations_arr,
                condition_diagnostics=condition_diagnostics,
            )
            rcs_db_vec = _rcs_db_from_sigma(rcs_lin_vec)
            residual_vec = np.full(len(elevations), mixed_residual, dtype=float)
            constraint_residual_vec = np.zeros(len(elevations), dtype=float)
            _consume_condition_estimate(
                cond_values, condition_diagnostics, formulation_label
            )
            reused_matrix_solve_count += len(elevations)

            for idx, elev_deg in enumerate(elevations):
                amp_val = complex(amp_vec[idx])
                residual_local = float(residual_vec[idx])
                samples.append({
                    "frequency_ghz": float(freq_ghz),
                    "theta_inc_deg": float(elev_deg),
                    "theta_scat_deg": float(elev_deg),
                    "rcs_linear": float(rcs_lin_vec[idx]),
                    "rcs_db": float(rcs_db_vec[idx]),
                    "rcs_amp_real": float(np.real(amp_val)),
                    "rcs_amp_imag": float(np.imag(amp_val)),
                    "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                    "linear_residual": residual_local,
                })
                residual_values.append(residual_local)
                constraint_residual_values.append(0.0)
                done_steps += 1
                emit_progress(f"Mixed sheet+PEC BIE solved {freq_ghz:g} GHz at {elev_deg:g} deg")
            continue

        if cached_panels is None:
            linear_mesh_stats_local.update(_linear_coupled_node_report(mesh, coupled_infos))
        done_steps += 1
        emit_progress(f"Assembled linear/Galerkin coupled operators at {freq_ghz:g} GHz")

        if cached_junction_constraints is not None and cached_junction_stats is not None:
            linear_junction_constraints = cached_junction_constraints
            linear_junction_stats = dict(cached_junction_stats)
        else:
            linear_junction_constraints, linear_junction_stats = _build_linear_junction_constraints(
                mesh, coupled_infos,
            )
        junction_stats.update(linear_mesh_stats_local)
        junction_stats.update(linear_junction_stats)
        orientation_conflicts = int(linear_junction_stats.get("junction_orientation_conflict_nodes", 0))
        if orientation_conflicts > 0:
            raise ValueError(
                f"Detected {orientation_conflicts} cross-segment junction node(s) with "
                "inconsistent segment orientation. Refusing to solve because "
                "the material-side trace assignment is physically ambiguous; "
                "fix the geometry so shared junctions have a consistent "
                "plus/minus side assignment."
            )
        if linear_junction_constraints.size > 0:
            formulation_label = "2D BIE/MoM coupled dielectric trace formulation (linear Galerkin + junction constraints)"
            materials.warn_once(
                (
                    "Applied "
                    f"{int(linear_junction_stats.get('junction_constraints', 0))} linear/Galerkin junction constraint(s) "
                    f"(trace={int(linear_junction_stats.get('junction_trace_constraints', 0))}, "
                    f"flux={int(linear_junction_stats.get('junction_flux_constraints', 0))}) "
                    f"across {int(linear_junction_stats.get('junction_nodes', 0))} node(s)."
                )
            )

        check_abort()

        # --- TE Robin path: MFIE for PEC and IBC surfaces ---
        use_te_robin_mfie = (pol == 'TE' and _is_all_robin(coupled_infos))

        if use_te_robin_mfie:
            formulation_label = "2D MFIE TE Robin (SLP representation)"
            condition_diagnostics = {} if compute_condition_number else None
            rcs_lin_vec, amp_vec, mfie_residual = _solve_te_robin_mfie(
                mesh=mesh,
                infos=coupled_infos,
                pol=pol,
                k0=k0,
                elevations_deg=elevations_arr,
                solver_method=solver_method,
                condition_diagnostics=condition_diagnostics,
            )
            rcs_db_vec = _rcs_db_from_sigma(rcs_lin_vec)
            residual_vec = np.full(len(elevations), mfie_residual, dtype=float)
            constraint_residual_vec = np.zeros(len(elevations), dtype=float)
            _consume_condition_estimate(
                cond_values, condition_diagnostics, formulation_label
            )
            reused_matrix_solve_count += len(elevations)

            for idx, elev_deg in enumerate(elevations):
                amp_val = complex(amp_vec[idx])
                residual_local = float(residual_vec[idx])
                samples.append(
                    {
                        "frequency_ghz": float(freq_ghz),
                        "theta_inc_deg": float(elev_deg),
                        "theta_scat_deg": float(elev_deg),
                        "rcs_linear": float(rcs_lin_vec[idx]),
                        "rcs_db": float(rcs_db_vec[idx]),
                        "rcs_amp_real": float(np.real(amp_val)),
                        "rcs_amp_imag": float(np.imag(amp_val)),
                        "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                        "linear_residual": residual_local,
                    }
                )
                residual_values.append(residual_local)
                constraint_residual_values.append(0.0)
                done_steps += 1
                emit_progress(f"MFIE solved {freq_ghz:g} GHz at {elev_deg:g} deg")
            continue

        # --- Multi-region indirect formulation (layered coatings) ---
        use_multi_region = _is_multi_region(coupled_infos)

        if use_multi_region:
            formulation_label = "2D multi-region indirect SLP formulation (layered coating)"
            condition_diagnostics = {} if compute_condition_number else None
            rcs_lin_vec, amp_vec, multi_residual, _ = _solve_multi_region_indirect(
                mesh=mesh,
                infos=coupled_infos,
                pol=pol,
                k0=k0,
                elevations_deg=elevations_arr,
                solver_method=solver_method,
                condition_diagnostics=condition_diagnostics,
            )
            rcs_db_vec = _rcs_db_from_sigma(rcs_lin_vec)
            residual_vec = np.full(len(elevations), multi_residual, dtype=float)
            constraint_residual_vec = np.zeros(len(elevations), dtype=float)
            _consume_condition_estimate(
                cond_values, condition_diagnostics, formulation_label
            )
            reused_matrix_solve_count += len(elevations)

            for idx, elev_deg in enumerate(elevations):
                amp_val = complex(amp_vec[idx])
                residual_local = float(residual_vec[idx])
                samples.append(
                    {
                        "frequency_ghz": float(freq_ghz),
                        "theta_inc_deg": float(elev_deg),
                        "theta_scat_deg": float(elev_deg),
                        "rcs_linear": float(rcs_lin_vec[idx]),
                        "rcs_db": float(rcs_db_vec[idx]),
                        "rcs_amp_real": float(np.real(amp_val)),
                        "rcs_amp_imag": float(np.imag(amp_val)),
                        "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                        "linear_residual": residual_local,
                    }
                )
                residual_values.append(residual_local)
                constraint_residual_values.append(0.0)
                done_steps += 1
                emit_progress(f"Multi-region solved {freq_ghz:g} GHz at {elev_deg:g} deg")
            continue

        # --- Dielectric indirect formulation ---
        use_dielectric_indirect = _is_single_dielectric_body(coupled_infos)

        if use_dielectric_indirect:
            formulation_label = "2D indirect two-density dielectric formulation"
            condition_diagnostics = {} if compute_condition_number else None
            rcs_lin_vec, amp_vec, diel_residual = _solve_dielectric_indirect(
                mesh=mesh,
                infos=coupled_infos,
                pol=pol,
                k0=k0,
                elevations_deg=elevations_arr,
                condition_diagnostics=condition_diagnostics,
            )
            rcs_db_vec = _rcs_db_from_sigma(rcs_lin_vec)
            residual_vec = np.full(len(elevations), diel_residual, dtype=float)
            constraint_residual_vec = np.zeros(len(elevations), dtype=float)
            _consume_condition_estimate(
                cond_values, condition_diagnostics, formulation_label
            )
            reused_matrix_solve_count += len(elevations)

            for idx, elev_deg in enumerate(elevations):
                amp_val = complex(amp_vec[idx])
                residual_local = float(residual_vec[idx])
                samples.append(
                    {
                        "frequency_ghz": float(freq_ghz),
                        "theta_inc_deg": float(elev_deg),
                        "theta_scat_deg": float(elev_deg),
                        "rcs_linear": float(rcs_lin_vec[idx]),
                        "rcs_db": float(rcs_db_vec[idx]),
                        "rcs_amp_real": float(np.real(amp_val)),
                        "rcs_amp_imag": float(np.imag(amp_val)),
                        "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                        "linear_residual": residual_local,
                    }
                )
                residual_values.append(residual_local)
                constraint_residual_values.append(0.0)
                done_steps += 1
                emit_progress(f"Dielectric solved {freq_ghz:g} GHz at {elev_deg:g} deg")
            continue

        # --- All-Robin (PEC, IBC, or mixed PEC+IBC) SLP formulation ---
        # _solve_robin_bie handles every all-robin case correctly: TE PEC via
        # the alpha=0 MFIE limit, TM IBC via the standard Robin BIE, and TM
        # PEC via the per-row EFIE override (the alpha->infty limit).  TE
        # all-robin already short-circuits to the dedicated MFIE solver
        # above, so we only need to dispatch the remaining all-robin cases
        # here (primarily TM PEC bodies, which the coupled-trace path
        # does not handle correctly).
        use_robin_bie = _is_all_robin(coupled_infos)

        if use_robin_bie:
            formulation_label = (
                "2D Robin-BIE (SLP representation; element-weighted IBC, TM-PEC EFIE override)"
                if pol == 'TM'
                else "2D Robin-BIE IBC formulation (SLP representation)"
            )
            condition_diagnostics = {} if compute_condition_number else None
            rcs_lin_vec, amp_vec, robin_residual = _solve_robin_bie(
                mesh=mesh,
                infos=coupled_infos,
                pol=pol,
                k0=k0,
                elevations_deg=elevations_arr,
                condition_diagnostics=condition_diagnostics,
            )
            rcs_db_vec = _rcs_db_from_sigma(rcs_lin_vec)
            residual_vec = np.full(len(elevations), robin_residual, dtype=float)
            constraint_residual_vec = np.zeros(len(elevations), dtype=float)
            _consume_condition_estimate(
                cond_values, condition_diagnostics, formulation_label
            )
            reused_matrix_solve_count += len(elevations)

            for idx, elev_deg in enumerate(elevations):
                amp_val = complex(amp_vec[idx])
                residual_local = float(residual_vec[idx])
                samples.append(
                    {
                        "frequency_ghz": float(freq_ghz),
                        "theta_inc_deg": float(elev_deg),
                        "theta_scat_deg": float(elev_deg),
                        "rcs_linear": float(rcs_lin_vec[idx]),
                        "rcs_db": float(rcs_db_vec[idx]),
                        "rcs_amp_real": float(np.real(amp_val)),
                        "rcs_amp_imag": float(np.imag(amp_val)),
                        "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                        "linear_residual": residual_local,
                    }
                )
                residual_values.append(residual_local)
                constraint_residual_values.append(0.0)
                done_steps += 1
                emit_progress(f"Robin-BIE solved {freq_ghz:g} GHz at {elev_deg:g} deg")
            continue

        # --- No formulation matched ---
        # The dispatch above is exhaustive for physical inputs: sheets, TE
        # all-Robin (MFIE), multi-region (layered / mixed robin+transmission /
        # multiple bodies), single dielectric body, and TM/mixed all-Robin.
        # Only degenerate configurations (e.g. TYPE 5 interfaces with no
        # exterior boundary) can reach this point.  The old coupled-trace
        # fallback that lived here had inconsistent Green-identity signs and
        # a BC row that solved the wrong problem for TM PEC, so failing
        # loudly is strictly better than running it.
        raise ValueError(
            "Geometry did not match any supported monostatic formulation "
            "(sheet, all-Robin PEC/IBC, dielectric body, or multi-region). "
            "Check that the geometry encloses regions with a boundary to air "
            "(TYPE 5-only configurations without an exterior interface are "
            "not solvable)."
        )

    residual_norm_max, residual_norm_mean, residual_nonfinite_count = (
        _summarize_residuals(residual_values)
    )
    condition_est_computed = bool(compute_condition_number)

    metadata: 'Dict[str, Any]' = {
        "source_path": str(geometry_snapshot.get("source_path", "") or ""),
        "segment_count": int(len(geometry_snapshot.get("segments", []) or [])),
        "panel_count": int(np.max(panel_count_values)) if panel_count_values else 0,
        "panel_count_min": int(np.min(panel_count_values)) if panel_count_values else 0,
        "panel_count_max": int(np.max(panel_count_values)) if panel_count_values else 0,
        "panel_length_min_m": float(np.min(panel_length_min_values)) if panel_length_min_values else 0.0,
        "panel_length_max_m": float(np.max(panel_length_max_values)) if panel_length_max_values else 0.0,
        "mesh_reference_ghz": float(mesh_reference_values[0]) if len(set(round(v, 12) for v in mesh_reference_values)) == 1 and mesh_reference_values else None,
        "mesh_reference_ghz_min": float(np.min(mesh_reference_values)) if mesh_reference_values else 0.0,
        "mesh_reference_ghz_max": float(np.max(mesh_reference_values)) if mesh_reference_values else 0.0,
        "mesh_wavelength_m": float(mesh_wavelength_values[0]) if len(set(round(v, 15) for v in mesh_wavelength_values)) == 1 and mesh_wavelength_values else None,
        "mesh_wavelength_min_m": float(np.min(mesh_wavelength_values)) if mesh_wavelength_values else 0.0,
        "mesh_wavelength_max_m": float(np.max(mesh_wavelength_values)) if mesh_wavelength_values else 0.0,
        "mesh_max_refractive_index": float(np.max(mesh_max_index_values)) if mesh_max_index_values else 1.0,
        "mesh_material_flags": sorted(mesh_material_flags_used),
        "polarization_internal": pol,
        "polarization_user": _canonical_user_polarization_label(polarization),
        "polarization_aliases": [_canonical_user_polarization_label(polarization)],
        "polarization_export": _canonical_user_polarization_label(polarization),
        "polarization_export_alias": _primary_alias_for_user_polarization(polarization),
        "rcs_normalization_mode": rcs_norm_mode,
        "formulation": formulation_label,
        "solver_method": (
            "fmm_gmres" if solver_method == "fmm" else "dense_lu"
        ),
        "solver_method_requested": str(solver_method),
        "residual_norm_max": residual_norm_max,
        "residual_norm_mean": residual_norm_mean,
        "residual_nonfinite_count": residual_nonfinite_count,
        "constraint_residual_norm_max": float(np.max(constraint_residual_values)) if constraint_residual_values else 0.0,
        "constraint_residual_norm_mean": float(np.mean(constraint_residual_values)) if constraint_residual_values else 0.0,
        "condition_est_max": float(np.max(cond_values)) if cond_values else float("nan"),
        "condition_est_mean": float(np.mean(cond_values)) if cond_values else float("nan"),
        "condition_est_computed": bool(condition_est_computed),
        "condition_estimator": (
            "equilibrated_1norm_lu_onenormest"
            if condition_est_computed else "not_requested"
        ),
        "warnings": list(materials.warnings),
        "warning_count": int(len(materials.warnings)),
        "math_backend_real_bessel": _BESSEL.backend_name,
        "math_backend_complex_hankel": _complex_hankel_backend_name(),
        "reused_matrix_solve_count": int(reused_matrix_solve_count),
        "parallel_elevation_solve_count": 0,
        "max_parallel_workers_used": int(max_parallel_workers_used),
        "mesh_reference_frequency_used": bool(mesh_ref_ghz is not None),
        "cfie_alpha": float(cfie_alpha),
        "junction_nodes": int(junction_stats.get("junction_nodes", 0)),
        "junction_constraints": int(junction_stats.get("junction_constraints", 0)),
        "junction_panels": int(junction_stats.get("junction_panels", 0)),
        "junction_trace_constraints": int(junction_stats.get("junction_trace_constraints", 0)),
        "junction_flux_constraints": int(junction_stats.get("junction_flux_constraints", 0)),
        "junction_orientation_conflict_nodes": int(junction_stats.get("junction_orientation_conflict_nodes", 0)),
        "linear_node_count": int(junction_stats.get("linear_node_count", 0)),
        "linear_element_count": int(junction_stats.get("linear_element_count", 0)),
        "shared_node_count": int(junction_stats.get("shared_node_count", 0)),
        "split_node_count": int(junction_stats.get("split_node_count", 0)),
        "split_boundary_primitive_count": int(junction_stats.get("split_boundary_primitive_count", 0)),
        "multi_signature_node_count": int(junction_stats.get("multi_signature_node_count", 0)),
        "preflight": dict(preflight_report),
    }

    quality_gate = evaluate_quality_gate(metadata, thresholds=quality_thresholds)
    metadata["quality_gate"] = quality_gate
    if strict_quality_gate and not bool(quality_gate.get("passed", False)):
        reason = str(quality_gate.get("reason", "quality gate failed"))
        raise ValueError(f"Quality gate failed: {reason}")

    return {
        "solver": "2d_bie_mom_rcs",
        "scattering_mode": "monostatic",
        "amplitude_convention": RCS_AMPLITUDE_CONVENTION,
        "polarization": _canonical_user_polarization_label(polarization),
        "polarization_export": _canonical_user_polarization_label(polarization),
        "samples": samples,
        "metadata": metadata,
    }


def _farfield_at_angles_slp(
    mesh: 'LinearMesh',
    density: 'np.ndarray',
    k_air: 'float',
    obs_angles_deg: 'np.ndarray',
    order: 'int' = 8,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """SLP far-field projector at arbitrary observation angles (for TE PEC MFIE)."""

    obs = np.asarray(obs_angles_deg, dtype=float).reshape(-1)
    amp = _farfield_linear_density_many(
        mesh, density, k_air, obs, "SLP", order=order
    )

    rcs_lin = _rcs_sigma_from_amp(amp, k_air)
    return rcs_lin, amp


def _farfield_at_angles_dlp(
    mesh: 'LinearMesh',
    density: 'np.ndarray',
    k_air: 'float',
    obs_angles_deg: 'np.ndarray',
    order: 'int' = 8,
) -> 'Tuple[np.ndarray, np.ndarray]':
    """DLP far-field projector at arbitrary observation angles (for dielectric indirect)."""

    obs = np.asarray(obs_angles_deg, dtype=float).reshape(-1)
    amp = _farfield_linear_density_many(
        mesh, density, k_air, obs, "DLP", order=order
    )

    rcs_lin = _rcs_sigma_from_amp(amp, k_air)
    return rcs_lin, amp


def solve_bistatic_rcs_2d(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    incidence_angles_deg: 'List[float]',
    observation_angles_deg: 'List[float]',
    polarization: 'str',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    quality_thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
    strict_quality_gate: 'bool' = True,
    compute_condition_number: 'bool' = False,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    cfie_alpha: 'float' = CFIE_ALPHA_DEFAULT,
    abort_event: 'Optional[threading.Event]' = None,
    solver_method: 'str' = "auto",
) -> 'Dict[str, Any]':
    """
    Bistatic 2D RCS solver.

    For each frequency and incidence angle, solves the boundary integral equation
    and evaluates the far-field RCS at all requested observation angles.

    Returns samples with ``theta_inc_deg != theta_scat_deg`` in general.
    Compatible with ``export_result_to_grim`` which splits by incidence angle.
    """

    if not frequencies_ghz:
        raise ValueError("At least one frequency is required.")
    if not incidence_angles_deg:
        raise ValueError("At least one incidence angle is required.")
    if not observation_angles_deg:
        raise ValueError("At least one observation angle is required.")

    frequencies = [float(f) for f in frequencies_ghz]
    inc_angles = [float(a) for a in incidence_angles_deg]
    obs_angles = [float(a) for a in observation_angles_deg]
    if any((not math.isfinite(f)) or f <= 0.0 for f in frequencies):
        raise ValueError("Frequencies must be positive finite GHz values.")
    if any(not math.isfinite(a) for a in inc_angles):
        raise ValueError("Incidence angles must all be finite.")
    if any(not math.isfinite(a) for a in obs_angles):
        raise ValueError("Observation angles must all be finite.")

    solver_method = _normalize_public_2d_solver_method(solver_method)
    if solver_method == "fmm":
        raise ValueError(
            "solver_method='fmm' is not implemented by the bistatic 2-D "
            "path. Use solver_method='auto' or 'direct'; no dense fallback "
            "was performed."
        )
    _raise_if_untrusted_math_backends()
    pol = _normalize_polarization(polarization)
    unit_scale = _unit_scale_to_meters(geometry_units)
    base_dir = _material_base_dir_for_snapshot(
        geometry_snapshot, material_base_dir
    )

    mesh_ref_ghz = float(mesh_reference_ghz) if mesh_reference_ghz is not None else None
    if mesh_ref_ghz is not None and (
        not math.isfinite(mesh_ref_ghz) or mesh_ref_ghz <= 0.0
    ):
        raise ValueError("mesh_reference_ghz must be a positive finite GHz value.")

    preflight_report = validate_geometry_snapshot_for_solver(geometry_snapshot, base_dir=base_dir, meters_scale=unit_scale)
    materials = MaterialLibrary.from_entries(
        geometry_snapshot.get("ibcs", []) or [],
        geometry_snapshot.get("dielectrics", []) or [],
        base_dir=base_dir,
    )
    for _msg in list(preflight_report.get("warnings", []) or []):
        materials.warn_once(str(_msg))
    _warn_far_quadrature_override(materials)

    if abs(float(cfie_alpha)) > EPS:
        raise ValueError(
            "cfie_alpha is not implemented by any active 2-D formulation; "
            "use cfie_alpha=0."
        )

    samples: 'List[Dict[str, Any]]' = []
    residual_values: 'List[float]' = []
    cond_values: 'List[float]' = []
    total_steps = len(frequencies) * len(inc_angles)
    done_steps = 0
    obs_arr = np.asarray(obs_angles, dtype=float)
    mesh_wavelength_values: 'List[float]' = []
    mesh_max_index_values: 'List[float]' = []
    mesh_material_flags_used: 'Set[int]' = set()
    conservative_mesh = None
    if mesh_ref_ghz is not None:
        conservative_mesh = _conservative_mesh_wavelength_for_frequencies(
            geometry_snapshot,
            materials,
            set(frequencies) | {mesh_ref_ghz},
        )

    def check_abort() -> 'None':
        if abort_event is not None and abort_event.is_set():
            raise InterruptedError("Solve cancelled by user.")

    def emit_progress(msg: 'str') -> 'None':
        if progress_callback is not None:
            try:
                progress_callback(done_steps, total_steps, msg)
            except Exception:
                pass

    for freq_ghz in frequencies:
        frequency_condition_recorded = False
        check_abort()
        freq_hz = freq_ghz * 1e9
        k0 = 2.0 * math.pi * freq_hz / C0
        mesh_freq_ghz = mesh_ref_ghz if mesh_ref_ghz is not None else float(freq_ghz)
        if conservative_mesh is None:
            (
                lambda_min,
                mesh_max_index,
                mesh_material_flags,
            ) = _mesh_wavelength_for_snapshot(
                geometry_snapshot, materials, mesh_freq_ghz
            )
        else:
            (
                lambda_min,
                mesh_max_index,
                mesh_material_flags,
            ) = conservative_mesh
        mesh_wavelength_values.append(float(lambda_min))
        mesh_max_index_values.append(float(mesh_max_index))
        mesh_material_flags_used.update(int(flag) for flag in mesh_material_flags)

        panels = _build_panels(geometry_snapshot, unit_scale, lambda_min, max_panels=max_panels)
        preview_infos = _build_coupled_panel_info(panels, materials, freq_ghz, pol, k0)
        mesh, _ = _build_linear_mesh_interface_aware(panels, preview_infos)
        coupled_infos = _build_linear_coupled_infos(mesh, materials, freq_ghz, pol, k0)
        _assert_no_type1_sheet_for_mixed(coupled_infos)
        _assert_air_exterior(coupled_infos)
        _assert_supported_te_type2_contours(mesh, coupled_infos, pol)
        nnodes = len(mesh.nodes)

        use_sheet = _is_all_sheet(coupled_infos)
        use_mixed_sheet = _is_sheet_plus_pec(coupled_infos)
        use_te_robin_mfie = (pol == 'TE' and _is_all_robin(coupled_infos))
        # TM all-Robin (PEC, IBC, or mixed) must use the Robin BIE, exactly as
        # in the monostatic dispatch: the coupled-trace fallback below applies
        # a mass row to q for TYPE 2 surfaces, which ignores the impedance
        # entirely (TM IBC used to return bit-for-bit the PEC answer here).
        use_tm_robin_bie = (pol == 'TM' and _is_all_robin(coupled_infos))
        use_diel_indirect = _is_single_dielectric_body(coupled_infos) and not _is_multi_region(coupled_infos)
        use_multi_region = _is_multi_region(coupled_infos)

        resources = _dense_formulation_resources(mesh, coupled_infos, pol)
        est_gb = _estimate_memory_gb(
            resources["nodes"],
            use_cfie=False,
            n_regions=max(1, resources["n_regions"]),
            system_dofs=resources["system_dofs"],
            operator_matrices=resources["operator_matrices"],
            n_rhs=1,
        )
        memory_limit_gb = _solve_memory_limit_gb()
        if est_gb > memory_limit_gb:
            raise MemoryError(
                f"Estimated peak memory {est_gb:.1f} GB exceeds the "
                f"{memory_limit_gb:.1f} GB limit for this bistatic process "
                f"({resources['system_dofs']} system DOFs, "
                f"{resources['n_regions']} region(s), "
                f"{resources['formulation']}; "
                f"{_detect_available_gb():.1f} GB detected)."
            )
        if est_gb > 8.0:
            materials.warn_once(
                f"Estimated peak memory {est_gb:.1f} GB for "
                f"{resources['system_dofs']} {resources['formulation']} "
                "system DOFs. Large problems may cause slowdowns or "
                "out-of-memory errors."
            )

        # Pre-assemble system matrices (reused across incidence angles).

        # --- TM Robin BIE pre-assembly (shared with _solve_robin_bie) ---
        robin_sys = None
        robin_alpha_elements = None
        robin_pec_node = None
        if use_tm_robin_bie:
            robin_sys, robin_alpha_elements, robin_pec_node = _assemble_robin_bie_system(
                mesh, coupled_infos, pol, k0)

        # --- TE Robin MFIE pre-assembly ---
        mfie_sys = None
        mfie_alpha_elements = None
        if use_te_robin_mfie:
            mfie_alpha_elements, _ = _robin_alpha_elements(
                mesh, coupled_infos, pol
            )
            has_mfie_ibc = bool(np.any(np.abs(mfie_alpha_elements) > EPS))
            s_alpha, Kp = _assemble_linear_operator_matrices(
                mesh, k0, obs_normal_deriv=True,
                compute_single_layer=has_mfie_ibc,
                single_layer_observation_coefficients=(
                    mfie_alpha_elements if has_mfie_ibc else None
                ),
            )
            M_mat = _assemble_linear_mass_matrix(mesh)
            mfie_sys = -0.5 * M_mat + Kp
            if has_mfie_ibc:
                mfie_sys = mfie_sys + s_alpha

        # Pre-assemble sheet system operators once (reused across inc angles).
        # Handles both pure-sheet (all TYPE 1) and mixed sheet+PEC geometries.
        # Mixed uses the unified SLP (TM) / DLP (TE) representation with
        # element coefficient Z/(jketa) on sheets and zero on PEC elements.
        sheet_a_sys = None
        sheet_endpoint_nodes = None
        if use_sheet or use_mixed_sheet:
            z_elements_sheet = np.asarray([
                complex(info.robin_impedance)
                if int(info.seg_type) == 1 else 0.0 + 0.0j
                for info in coupled_infos
            ], dtype=np.complex128)
            if pol == "TM":
                S_sheet, _ = _assemble_linear_operator_matrices(
                    mesh, k0, obs_normal_deriv=False,
                    compute_double_layer=False)
                weighted_mass = _assemble_linear_weighted_mass_matrix(
                    mesh, z_elements_sheet / (1j * float(k0) * ETA0)
                )
                sheet_a_sys = S_sheet - weighted_mass
            else:
                N_sheet = _assemble_linear_hypersingular_matrix(mesh, k0)
                # Sign matches _solve_te_sheet: N - (jk/eta)Z_s.M.
                weighted_mass = _assemble_linear_weighted_mass_matrix(
                    mesh, (1j * float(k0) / ETA0) * z_elements_sheet
                )
                sheet_a_sys = N_sheet - weighted_mass
                # Meixner pin: mu=0 at OPEN-STRIP endpoints only.  See
                # _geometric_sheet_endpoint_nodes for why we count by geometric
                # key (handles signature-split nodes in stair-stepped tapers).
                sheet_endpoint_nodes = _geometric_sheet_endpoint_nodes(mesh, coupled_infos)
                if sheet_endpoint_nodes.size > 0:
                    sheet_a_sys[sheet_endpoint_nodes, :] = 0.0
                    sheet_a_sys[sheet_endpoint_nodes, sheet_endpoint_nodes] = 1.0

        # Assemble every incidence-angle RHS and solve it as one dense
        # multi-RHS system.  np.linalg.solve then performs one factorization
        # per frequency/formulation instead of refactoring the same matrix for
        # every incidence angle.  The monostatic path already follows this
        # pattern; keeping bistatic consistent is both faster and numerically
        # equivalent.
        inc_all_arr = np.asarray(inc_angles, dtype=float)
        batch_solution = None
        batch_residuals = None
        batch_exterior_density = None

        if use_sheet or use_mixed_sheet:
            rhs_all = np.zeros(
                (nnodes, inc_all_arr.size), dtype=np.complex128
            )
            for elem in mesh.elements:
                ids = np.asarray(elem.node_ids, dtype=int)
                if pol == "TM":
                    rhs_all[ids, :] -= _linear_element_incident_load_many(
                        elem, k_air=float(k0), elevations_deg=inc_all_arr
                    )
                else:
                    rhs_all[ids, :] -= _linear_element_incident_dn_load_many(
                        elem, k_air=float(k0), elevations_deg=inc_all_arr
                    )
            if sheet_endpoint_nodes is not None and sheet_endpoint_nodes.size > 0:
                rhs_all[sheet_endpoint_nodes, :] = 0.0
            _ensure_finite_linear_system(
                sheet_a_sys, rhs_all, label="bistatic sheet system"
            )
            condition_diagnostics = {} if compute_condition_number else None
            batch_solution = _solve_dense_system(
                sheet_a_sys, rhs_all, condition_diagnostics,
                "bistatic sheet system",
            )
            if condition_diagnostics is not None:
                cond_values.append(float(condition_diagnostics["condition_est"]))
                frequency_condition_recorded = True
            batch_residuals = _residual_norm_many(
                sheet_a_sys, batch_solution, rhs_all
            )

        elif use_multi_region:
            condition_diagnostics = {} if compute_condition_number else None
            _, _, multi_residual, batch_exterior_density = (
                _solve_multi_region_indirect(
                    mesh,
                    coupled_infos,
                    pol,
                    k0,
                    inc_all_arr,
                    condition_diagnostics=condition_diagnostics,
                )
            )
            batch_residuals = np.full(
                inc_all_arr.size, float(multi_residual), dtype=float
            )
            if condition_diagnostics is not None:
                _consume_condition_estimate(
                    cond_values,
                    condition_diagnostics,
                    "bistatic multi-region formulation",
                )
                frequency_condition_recorded = True

        elif use_te_robin_mfie:
            rhs_all = np.zeros(
                (nnodes, inc_all_arr.size), dtype=np.complex128
            )
            for eidx, elem in enumerate(mesh.elements):
                ids = np.asarray(elem.node_ids, dtype=int)
                rhs_all[ids, :] -= _linear_element_incident_dn_load_many(
                    elem, k_air=k0, elevations_deg=inc_all_arr,
                )
                if (
                    mfie_alpha_elements is not None
                    and abs(complex(mfie_alpha_elements[eidx])) > EPS
                ):
                    rhs_all[ids, :] -= (
                        complex(mfie_alpha_elements[eidx])
                        * _linear_element_incident_load_many(
                            elem, k_air=k0, elevations_deg=inc_all_arr,
                        )
                    )
            _ensure_finite_linear_system(
                mfie_sys, rhs_all, label="bistatic TE Robin MFIE system"
            )
            condition_diagnostics = {} if compute_condition_number else None
            batch_solution = _solve_dense_system(
                mfie_sys, rhs_all, condition_diagnostics,
                "bistatic TE Robin MFIE system",
            )
            if condition_diagnostics is not None:
                cond_values.append(float(condition_diagnostics["condition_est"]))
                frequency_condition_recorded = True
            batch_residuals = _residual_norm_many(
                mfie_sys, batch_solution, rhs_all
            )

        elif use_tm_robin_bie:
            rhs_all = _robin_bie_rhs_many(
                mesh,
                robin_alpha_elements,
                robin_pec_node,
                pol,
                k0,
                inc_all_arr,
            )
            _ensure_finite_linear_system(
                robin_sys, rhs_all, label="bistatic TM Robin BIE system"
            )
            condition_diagnostics = {} if compute_condition_number else None
            batch_solution = _solve_dense_system(
                robin_sys, rhs_all, condition_diagnostics,
                "bistatic Robin system",
            )
            if condition_diagnostics is not None:
                cond_values.append(float(condition_diagnostics["condition_est"]))
                frequency_condition_recorded = True
            batch_residuals = _residual_norm_many(
                robin_sys, batch_solution, rhs_all
            )

        elif use_diel_indirect:
            info0 = coupled_infos[0]
            k1_vals = {
                complex(info.k_plus)
                for info in coupled_infos
                if info.plus_region > 0
            }
            k1 = k1_vals.pop() if k1_vals else k0
            factor = (
                complex(info0.mu_minus / info0.mu_plus)
                if pol == "TM"
                else complex(info0.eps_minus / info0.eps_plus)
            )

            _, K0 = _assemble_linear_operator_matrices(
                mesh, k0, obs_normal_deriv=False,
                compute_single_layer=False,
            )
            _, Kp1 = _assemble_linear_operator_matrices(
                mesh, k1, obs_normal_deriv=True,
                compute_single_layer=False,
            )
            S1, _ = _assemble_linear_operator_matrices(
                mesh, k1, obs_normal_deriv=False,
                compute_double_layer=False,
            )
            D0 = _assemble_linear_hypersingular_matrix(mesh, k0)
            M = _assemble_linear_mass_matrix(mesh)
            diel_sys = np.zeros(
                (2 * nnodes, 2 * nnodes), dtype=np.complex128
            )
            diel_sys[:nnodes, :nnodes] = 0.5 * M + K0
            diel_sys[:nnodes, nnodes:] = -S1
            diel_sys[nnodes:, :nnodes] = D0
            diel_sys[nnodes:, nnodes:] = factor * (0.5 * M + Kp1)

            rhs_all = np.zeros(
                (2 * nnodes, inc_all_arr.size), dtype=np.complex128
            )
            for elem in mesh.elements:
                ids = np.asarray(elem.node_ids, dtype=int)
                rhs_all[ids, :] += _linear_element_incident_load_many(
                    elem, k0, inc_all_arr
                )
                rhs_all[nnodes + ids, :] -= (
                    _linear_element_incident_dn_load_many(
                        elem, k0, inc_all_arr
                    )
                )
            _ensure_finite_linear_system(
                diel_sys, rhs_all, label="bistatic dielectric indirect system"
            )
            condition_diagnostics = {} if compute_condition_number else None
            batch_solution = _solve_dense_system(
                diel_sys, rhs_all, condition_diagnostics,
                "bistatic dielectric indirect system",
            )
            if condition_diagnostics is not None:
                cond_values.append(float(condition_diagnostics["condition_est"]))
                frequency_condition_recorded = True
            batch_residuals = _residual_norm_many(
                diel_sys, batch_solution, rhs_all
            )

        else:
            raise ValueError(
                "Geometry did not match any supported bistatic formulation "
                "(sheet, all-Robin PEC/IBC, dielectric body, or multi-region). "
                "No deprecated coupled-trace fallback was performed."
            )

        for inc_index, inc_deg in enumerate(inc_angles):
            check_abort()

            if use_sheet or use_mixed_sheet:
                density = batch_solution[:, inc_index]
                residual_local = float(batch_residuals[inc_index])
                if pol == "TM":
                    rcs_lin, amp = _farfield_at_angles_slp(mesh, density, k0, obs_arr)
                else:
                    rcs_lin, amp = _farfield_at_angles_dlp(mesh, density, k0, obs_arr)

            elif use_multi_region:
                residual_local = float(batch_residuals[inc_index])
                rcs_lin, amp = _farfield_at_angles_slp(
                    mesh,
                    batch_exterior_density[:, inc_index],
                    k0,
                    obs_arr,
                )

            elif use_te_robin_mfie:
                sigma = batch_solution[:, inc_index]
                residual_local = float(batch_residuals[inc_index])
                rcs_lin, amp = _farfield_at_angles_slp(mesh, sigma, k0, obs_arr)

            elif use_tm_robin_bie:
                sigma = batch_solution[:, inc_index]
                residual_local = float(batch_residuals[inc_index])
                rcs_lin, amp = _farfield_at_angles_slp(
                    mesh, sigma, k0, obs_arr
                )

            elif use_diel_indirect:
                residual_local = float(batch_residuals[inc_index])
                mu = batch_solution[:nnodes, inc_index]
                rcs_lin, amp = _farfield_at_angles_dlp(mesh, mu, k0, obs_arr)

            else:
                raise AssertionError(
                    "Unreachable bistatic formulation dispatch."
                )

            rcs_db = _rcs_db_from_sigma(rcs_lin)
            residual_values.append(float(residual_local))
            for idx, obs_deg in enumerate(obs_angles):
                amp_val = complex(amp[idx])
                samples.append({
                    "frequency_ghz": float(freq_ghz),
                    "theta_inc_deg": float(inc_deg),
                    "theta_scat_deg": float(obs_deg),
                    "rcs_linear": float(rcs_lin[idx]),
                    "rcs_db": float(rcs_db[idx]),
                    "rcs_amp_real": float(np.real(amp_val)),
                    "rcs_amp_imag": float(np.imag(amp_val)),
                    "rcs_amp_phase_deg": float(math.degrees(cmath.phase(amp_val))),
                    "linear_residual": float(residual_local),
                })

            done_steps += 1
            emit_progress(f"Bistatic {freq_ghz:g} GHz inc={inc_deg:g} deg")

        if compute_condition_number and not frequency_condition_recorded:
            raise RuntimeError(
                "Bistatic 2-D solve did not produce the requested "
                "condition-number diagnostic; no field is returned."
            )

    residual_norm_max, residual_norm_mean, residual_nonfinite_count = (
        _summarize_residuals(residual_values)
    )
    metadata: 'Dict[str, Any]' = {
        "formulation": "bistatic 2D BIE/MoM",
        "cfie_alpha": float(cfie_alpha),
        "solver_method": "dense_lu",
        "solver_method_requested": str(solver_method),
        "mesh_wavelength_m": float(mesh_wavelength_values[0]) if len(set(round(v, 15) for v in mesh_wavelength_values)) == 1 and mesh_wavelength_values else None,
        "mesh_wavelength_min_m": float(np.min(mesh_wavelength_values)) if mesh_wavelength_values else 0.0,
        "mesh_wavelength_max_m": float(np.max(mesh_wavelength_values)) if mesh_wavelength_values else 0.0,
        "mesh_max_refractive_index": float(np.max(mesh_max_index_values)) if mesh_max_index_values else 1.0,
        "mesh_material_flags": sorted(mesh_material_flags_used),
        "residual_norm_max": residual_norm_max,
        "residual_norm_mean": residual_norm_mean,
        "residual_nonfinite_count": residual_nonfinite_count,
        "constraint_residual_norm_max": 0.0,
        "constraint_residual_norm_mean": 0.0,
        "condition_est_max": float(np.max(cond_values)) if cond_values else float("nan"),
        "condition_est_mean": float(np.mean(cond_values)) if cond_values else float("nan"),
        "condition_est_computed": bool(compute_condition_number),
        "condition_estimator": (
            "equilibrated_1norm_lu_onenormest"
            if compute_condition_number else "not_requested"
        ),
        "warnings": list(materials.warnings),
        "warning_count": int(len(materials.warnings)),
        "preflight": dict(preflight_report),
    }
    quality_gate = evaluate_quality_gate(metadata, thresholds=quality_thresholds)
    metadata["quality_gate"] = quality_gate
    if strict_quality_gate and not bool(quality_gate.get("passed", False)):
        reason = str(quality_gate.get("reason", "quality gate failed"))
        raise ValueError(f"Quality gate failed: {reason}")

    return {
        "solver": "2d_bie_mom_rcs",
        "scattering_mode": "bistatic",
        "amplitude_convention": RCS_AMPLITUDE_CONVENTION,
        "polarization": _canonical_user_polarization_label(polarization),
        "polarization_export": _canonical_user_polarization_label(polarization),
        "samples": samples,
        "metadata": metadata,
    }


def _run_certified_2d_pair(
    low_level_solver: 'Callable[..., Dict[str, Any]]',
    geometry_snapshot: 'Dict[str, Any]',
    solver_kwargs: 'Dict[str, Any]',
    mesh_convergence_policy: 'Optional[Dict[str, Any]]',
    progress_callback: 'Optional[Callable[[int, int, str], None]]',
) -> 'Dict[str, Any]':
    """Run base/fine 2-D solves and publish only a certified fine result."""

    from solver_quality import (
        evaluate_mesh_convergence,
        scale_snapshot_panel_density,
        validate_mesh_convergence_policy,
    )

    policy = validate_mesh_convergence_policy(mesh_convergence_policy)

    def _phase_callback(
        phase: 'str',
    ) -> 'Optional[Callable[[int, int, str], None]]':
        if progress_callback is None:
            return None

        def _mapped(done: 'int', total: 'int', message: 'str') -> 'None':
            total_i = max(1, int(total))
            done_i = max(0, min(int(done), total_i))
            if phase == "base":
                mapped_done = done_i
            else:
                mapped_done = total_i + done_i
            try:
                progress_callback(
                    mapped_done,
                    2 * total_i,
                    f"{phase.capitalize()} mesh: {message}",
                )
            except Exception:
                pass

        return _mapped

    common = dict(solver_kwargs)
    # Certification is intentionally non-optional here.  Both discrete
    # systems must pass their algebraic gate and supply condition telemetry
    # before their complex fields are compared.
    if str(common.get("solver_method", "auto")).strip().lower() == "fmm":
        raise ValueError(
            "Certified 2-D solves require a condition-reporting dense method; "
            "matrix-free FMM is available only through the low-level "
            "diagnostic solver until a fail-closed condition certificate is "
            "implemented. Use solver_method='auto' or 'direct'."
        )
    common["strict_quality_gate"] = True
    common["compute_condition_number"] = True

    base_kwargs = dict(common)
    base_kwargs["geometry_snapshot"] = geometry_snapshot
    base_kwargs["progress_callback"] = _phase_callback("base")
    base_result = low_level_solver(**base_kwargs)

    fine_snapshot = scale_snapshot_panel_density(
        geometry_snapshot, policy["fine_factor"]
    )
    # Keep the shared property scaling for BoR and provenance compatibility,
    # while giving the 2-D panel builder the original per-segment controls so
    # it can refine the realized base count of every primitive exactly.
    base_segment_n = []
    for segment in list(geometry_snapshot.get("segments", []) or []):
        props = list(segment.get("properties", []) or [])
        base_segment_n.append(props[1] if len(props) > 1 else 0)
    fine_snapshot["_2d_certification_refinement_factor"] = float(
        policy["fine_factor"]
    )
    fine_snapshot["_2d_certification_base_segment_n"] = base_segment_n
    fine_kwargs = dict(common)
    fine_kwargs["geometry_snapshot"] = fine_snapshot
    fine_kwargs["progress_callback"] = _phase_callback("fine")
    fine_result = low_level_solver(**fine_kwargs)

    base_panel_count = int(
        base_result.get("metadata", {}).get("panel_count", 0) or 0
    )
    fine_panel_count = int(
        fine_result.get("metadata", {}).get("panel_count", 0) or 0
    )
    if base_panel_count > 0 and fine_panel_count <= base_panel_count:
        raise ValueError(
            "Certified 2-D mesh refinement failed: the fine solve used "
            f"{fine_panel_count} panels versus {base_panel_count} on the base "
            "mesh. A mesh-convergence certificate requires a genuinely "
            "refined discretization."
        )

    mesh_gate = evaluate_mesh_convergence(
        base_result=base_result,
        fine_result=fine_result,
        rms_limit_db=policy["rms_limit_db"],
        max_abs_limit_db=policy["max_abs_limit_db"],
        complex_rms_limit=policy["complex_rms_limit"],
        complex_max_limit=policy["complex_max_limit"],
        phase_rms_limit_deg=policy["phase_rms_limit_deg"],
        phase_max_limit_deg=policy["phase_max_limit_deg"],
        phase_floor_relative=policy["phase_floor_relative"],
    )
    mesh_gate["schema"] = "ghost.solver.mesh-convergence.v1"
    mesh_gate["fine_factor"] = policy["fine_factor"]
    mesh_gate["published_mesh"] = "fine"
    mesh_gate["base_quality_gate"] = dict(
        base_result.get("metadata", {}).get("quality_gate", {}) or {}
    )
    mesh_gate["fine_quality_gate"] = dict(
        fine_result.get("metadata", {}).get("quality_gate", {}) or {}
    )
    mesh_gate["base_panel_count"] = base_panel_count
    mesh_gate["fine_panel_count"] = fine_panel_count
    mesh_gate["panel_refinement_ratio"] = (
        float(fine_panel_count) / float(base_panel_count)
        if base_panel_count > 0 else float("nan")
    )

    if not bool(mesh_gate.get("passed", False)):
        raise ValueError(
            "Certified 2-D mesh convergence failed: "
            + str(mesh_gate.get("reason", "unknown convergence failure"))
        )

    result = fine_result
    metadata = result.setdefault("metadata", {})
    metadata["mesh_convergence"] = mesh_gate
    metadata["mesh_convergence_certified"] = True
    metadata["certified_entry_point"] = True
    metadata["published_mesh"] = "fine"
    quality_gate = metadata.get("quality_gate")
    if isinstance(quality_gate, dict):
        quality_gate["mesh_convergence_certified"] = True
        quality_gate["certification_scope"] = (
            "discrete_linear_system_and_mesh_convergence"
        )
        if bool(quality_gate.get("passed", False)):
            quality_gate["reason"] = (
                "discrete linear-system quality thresholds and production "
                "mesh-convergence certification satisfied"
            )
    return result


def solve_monostatic_rcs_2d_certified(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    elevations_deg: 'List[float]',
    polarization: 'str',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    quality_thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
    mesh_convergence_policy: 'Optional[Dict[str, Any]]' = None,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    rcs_normalization_mode: 'str' = RCS_NORM_MODE_DEFAULT,
    cfie_alpha: 'float' = CFIE_ALPHA_DEFAULT,
    abort_event: 'Optional[threading.Event]' = None,
    solver_method: 'str' = "auto",
) -> 'Dict[str, Any]':
    """Canonical production monostatic entry: algebraic plus mesh certification."""

    return _run_certified_2d_pair(
        solve_monostatic_rcs_2d,
        geometry_snapshot,
        {
            "frequencies_ghz": frequencies_ghz,
            "elevations_deg": elevations_deg,
            "polarization": polarization,
            "geometry_units": geometry_units,
            "material_base_dir": material_base_dir,
            "quality_thresholds": quality_thresholds,
            "max_panels": max_panels,
            "mesh_reference_ghz": mesh_reference_ghz,
            "rcs_normalization_mode": rcs_normalization_mode,
            "cfie_alpha": cfie_alpha,
            "abort_event": abort_event,
            "solver_method": solver_method,
        },
        mesh_convergence_policy,
        progress_callback,
    )


def solve_monostatic_rcs_2d_survey(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    elevations_deg: 'List[float]',
    polarization: 'str',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    quality_thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    rcs_normalization_mode: 'str' = RCS_NORM_MODE_DEFAULT,
    cfie_alpha: 'float' = CFIE_ALPHA_DEFAULT,
    abort_event: 'Optional[threading.Event]' = None,
    solver_method: 'str' = "auto",
) -> 'Dict[str, Any]':
    """Single-mesh monostatic solve: algebraically gated, NOT mesh-certified.

    `solve_monostatic_rcs_2d_certified` solves the geometry twice -- once on
    the requested mesh and once refined by the policy's ``fine_factor`` -- and
    publishes the fine result only if the two agree.  That second solve is
    where most of the wall clock and all of the peak memory go, because cost
    scales with the square of the node count.

    This entry runs the base mesh alone, for screening: trade studies, sanity
    checks, picking which configurations are worth a real run.  The discrete
    linear system is still certified (the algebraic quality gate is untouched,
    so a badly conditioned or non-converged solve still fails closed); what is
    missing is any evidence that the *discretization* is fine enough, which is
    the error that silently biases an RCS number rather than announcing
    itself.

    The result is deliberately made unusable as production input.  It carries
    no ``metadata["mesh_convergence"]`` block, which is exactly what
    `feature_sum` requires before a field may enter a body or a delta, so the
    downstream pipeline rejects it on its own rather than trusting a label.
    Metadata and warnings say so explicitly as well, so an artifact found
    later identifies itself without needing this docstring.
    """

    result = solve_monostatic_rcs_2d(
        geometry_snapshot=geometry_snapshot,
        frequencies_ghz=frequencies_ghz,
        elevations_deg=elevations_deg,
        polarization=polarization,
        geometry_units=geometry_units,
        material_base_dir=material_base_dir,
        progress_callback=progress_callback,
        quality_thresholds=quality_thresholds,
        strict_quality_gate=True,
        compute_condition_number=True,
        max_panels=max_panels,
        mesh_reference_ghz=mesh_reference_ghz,
        rcs_normalization_mode=rcs_normalization_mode,
        cfie_alpha=cfie_alpha,
        abort_event=abort_event,
        solver_method=solver_method,
    )
    metadata = result.setdefault("metadata", {})
    metadata["mesh_convergence_certified"] = False
    metadata["certified_entry_point"] = False
    metadata["published_mesh"] = "base"
    metadata["survey_mode"] = True
    warning = (
        "SURVEY MODE: solved on the base mesh only. No mesh-convergence "
        "certificate exists for this field -- the discretization was never "
        "compared against a refined one, so its error is unmeasured. Use "
        "solve_monostatic_rcs_2d_certified for anything published, compared, "
        "or fed into a body or delta."
    )
    warnings = metadata.setdefault("warnings", [])
    if warning not in warnings:
        warnings.append(warning)
    quality_gate = metadata.get("quality_gate")
    if isinstance(quality_gate, dict):
        quality_gate["mesh_convergence_certified"] = False
        quality_gate["certification_scope"] = "discrete_linear_system_only"
    return result


def solve_bistatic_rcs_2d_certified(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    incidence_angles_deg: 'List[float]',
    observation_angles_deg: 'List[float]',
    polarization: 'str',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    quality_thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
    mesh_convergence_policy: 'Optional[Dict[str, Any]]' = None,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    cfie_alpha: 'float' = CFIE_ALPHA_DEFAULT,
    abort_event: 'Optional[threading.Event]' = None,
    solver_method: 'str' = "auto",
) -> 'Dict[str, Any]':
    """Canonical production bistatic entry: algebraic plus mesh certification."""

    return _run_certified_2d_pair(
        solve_bistatic_rcs_2d,
        geometry_snapshot,
        {
            "frequencies_ghz": frequencies_ghz,
            "incidence_angles_deg": incidence_angles_deg,
            "observation_angles_deg": observation_angles_deg,
            "polarization": polarization,
            "geometry_units": geometry_units,
            "material_base_dir": material_base_dir,
            "quality_thresholds": quality_thresholds,
            "max_panels": max_panels,
            "mesh_reference_ghz": mesh_reference_ghz,
            "cfie_alpha": cfie_alpha,
            "abort_event": abort_event,
            "solver_method": solver_method,
        },
        mesh_convergence_policy,
        progress_callback,
    )


def solve_bistatic_rcs_2d_survey(
    geometry_snapshot: 'Dict[str, Any]',
    frequencies_ghz: 'List[float]',
    incidence_angles_deg: 'List[float]',
    observation_angles_deg: 'List[float]',
    polarization: 'str',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    quality_thresholds: 'Optional[Dict[str, Union[float, int]]]' = None,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    mesh_reference_ghz: 'Optional[float]' = None,
    cfie_alpha: 'float' = CFIE_ALPHA_DEFAULT,
    abort_event: 'Optional[threading.Event]' = None,
    solver_method: 'str' = "auto",
) -> 'Dict[str, Any]':
    """Single-mesh bistatic solve with explicit survey provenance."""

    result = solve_bistatic_rcs_2d(
        geometry_snapshot=geometry_snapshot,
        frequencies_ghz=frequencies_ghz,
        incidence_angles_deg=incidence_angles_deg,
        observation_angles_deg=observation_angles_deg,
        polarization=polarization,
        geometry_units=geometry_units,
        material_base_dir=material_base_dir,
        progress_callback=progress_callback,
        quality_thresholds=quality_thresholds,
        strict_quality_gate=True,
        compute_condition_number=True,
        max_panels=max_panels,
        mesh_reference_ghz=mesh_reference_ghz,
        cfie_alpha=cfie_alpha,
        abort_event=abort_event,
        solver_method=solver_method,
    )
    metadata = result.setdefault("metadata", {})
    metadata["mesh_convergence_certified"] = False
    metadata["certified_entry_point"] = False
    metadata["published_mesh"] = "base"
    metadata["survey_mode"] = True
    warning = (
        "SURVEY MODE: solved on the base mesh only. No mesh-convergence "
        "certificate exists for this bistatic field."
    )
    warnings = metadata.setdefault("warnings", [])
    if warning not in warnings:
        warnings.append(warning)
    quality_gate = metadata.get("quality_gate")
    if isinstance(quality_gate, dict):
        quality_gate["mesh_convergence_certified"] = False
        quality_gate["certification_scope"] = "discrete_linear_system_only"
    return result


def compute_boundary_densities(
    geometry_snapshot: 'Dict[str, Any]',
    frequency_ghz: 'float',
    elevation_deg: 'float',
    polarization: 'str',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    cfie_alpha: 'float' = CFIE_ALPHA_DEFAULT,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
) -> 'Dict[str, Any]':
    """
    Compute formulation-specific boundary-integral unknowns for visualization.

    These SLP/DLP layer densities are mathematical representation unknowns;
    they are not generally physical electric or magnetic surface-current
    densities. Returns element-center positions, layer density, panel normals,
    and the formulation used for a single-frequency, single-angle debug solve.
    """

    if abs(float(cfie_alpha)) > EPS:
        raise ValueError(
            "cfie_alpha is not implemented by boundary-density diagnostics; "
            "use cfie_alpha=0."
        )
    pol = _normalize_polarization(polarization)
    unit_scale = _unit_scale_to_meters(geometry_units)
    base_dir = _material_base_dir_for_snapshot(
        geometry_snapshot, material_base_dir
    )
    frequency_ghz = float(frequency_ghz)
    if not math.isfinite(frequency_ghz) or frequency_ghz <= 0.0:
        raise ValueError("frequency_ghz must be a positive finite value.")
    freq_hz = frequency_ghz * 1e9
    k0 = 2.0 * math.pi * freq_hz / C0

    preflight = validate_geometry_snapshot_for_solver(geometry_snapshot, base_dir=base_dir, meters_scale=unit_scale)
    materials = MaterialLibrary.from_entries(
        geometry_snapshot.get("ibcs", []) or [],
        geometry_snapshot.get("dielectrics", []) or [],
        base_dir=base_dir,
    )
    lambda_min, mesh_max_index, mesh_material_flags = _mesh_wavelength_for_snapshot(
        geometry_snapshot, materials, frequency_ghz
    )
    panels = _build_panels(geometry_snapshot, unit_scale, lambda_min, max_panels=max_panels)
    preview_infos = _build_coupled_panel_info(panels, materials, frequency_ghz, pol, k0)
    mesh, _ = _build_linear_mesh_interface_aware(panels, preview_infos)
    coupled_infos = _build_linear_coupled_infos(mesh, materials, frequency_ghz, pol, k0)
    _assert_no_type1_sheet(coupled_infos)
    _assert_air_exterior(coupled_infos)
    _assert_supported_te_type2_contours(mesh, coupled_infos, pol)
    nnodes = len(mesh.nodes)
    elev_arr = np.asarray([elevation_deg], dtype=float)

    centers = np.asarray([e.center for e in mesh.elements], dtype=float)
    normals = np.asarray([e.normal for e in mesh.elements], dtype=float)
    lengths = np.asarray([e.length for e in mesh.elements], dtype=float)

    use_multi = _is_multi_region(coupled_infos)
    use_diel = _is_single_dielectric_body(coupled_infos) and not use_multi
    # All-Robin covers PEC, IBC (constant or tapered), and mixed PEC+IBC.
    # It reuses the shared Robin-BIE assembly (element-weighted alpha,
    # adjacent-medium wavenumber, per-row TM-PEC EFIE override) so the
    # visualized layer densities come from exactly the formulation the RCS solvers
    # use.  The previous branches applied a single alpha from element 0 to
    # the whole body (ignoring tapers, using raw k0) and dropped mixed
    # PEC+IBC TM bodies into a pure-PEC EFIE that ignored the impedance.
    use_robin = _is_all_robin(coupled_infos)

    resources = _dense_formulation_resources(mesh, coupled_infos, pol)
    est_gb = _estimate_memory_gb(
        resources["nodes"],
        use_cfie=False,
        n_regions=max(1, resources["n_regions"]),
        system_dofs=resources["system_dofs"],
        operator_matrices=resources["operator_matrices"],
        n_rhs=1,
    )
    memory_limit_gb = _solve_memory_limit_gb()
    if est_gb > memory_limit_gb:
        raise MemoryError(
            f"Estimated peak memory {est_gb:.1f} GB exceeds the "
            f"{memory_limit_gb:.1f} GB limit for boundary-density diagnostics "
            f"({resources['system_dofs']} system DOFs, "
            f"{resources['formulation']})."
        )

    if use_multi:
        # Multi-region: extract exterior SLP density.
        _, _, _, ext_density = _solve_multi_region_indirect(
            mesh, coupled_infos, pol, k0, elev_arr)
        sigma_nodes = ext_density[:, 0]
        density = np.asarray([
            0.5 * (sigma_nodes[e.node_ids[0]] + sigma_nodes[e.node_ids[1]])
            for e in mesh.elements
        ], dtype=np.complex128)
        formulation = "Multi-region indirect (exterior SLP density)"

    elif use_diel:
        # Assemble once and retain the DLP density directly.  The previous
        # implementation first called the complete RCS solver, discarded its
        # field, then rebuilt every dense operator and solved the identical
        # system a second time solely to expose this density.
        info0 = coupled_infos[0]
        k1_vals = {complex(i.k_plus) for i in coupled_infos if i.plus_region > 0}
        k1 = k1_vals.pop() if k1_vals else k0
        factor = complex(info0.mu_minus / info0.mu_plus) if pol == 'TM' else complex(info0.eps_minus / info0.eps_plus)
        _, K0 = _assemble_linear_operator_matrices(
            mesh, k0, False, compute_single_layer=False
        )
        _, Kp1 = _assemble_linear_operator_matrices(
            mesh, k1, True, compute_single_layer=False
        )
        S1, _ = _assemble_linear_operator_matrices(
            mesh, k1, False, compute_double_layer=False
        )
        D0 = _assemble_linear_hypersingular_matrix(mesh, k0)
        M = _assemble_linear_mass_matrix(mesh)
        a = np.zeros((2*nnodes, 2*nnodes), dtype=np.complex128)
        a[:nnodes,:nnodes] = 0.5*M+K0; a[:nnodes,nnodes:] = -S1
        a[nnodes:,:nnodes] = D0; a[nnodes:,nnodes:] = factor*(0.5*M+Kp1)
        rhs = np.zeros(2*nnodes, dtype=np.complex128)
        for elem in mesh.elements:
            ids = np.asarray(elem.node_ids, dtype=int)
            rhs[ids] += _linear_element_incident_load_many(elem, k0, elev_arr)[:,0]
            rhs[nnodes+ids] -= _linear_element_incident_dn_load_many(elem, k0, elev_arr)[:,0]
        sol = np.linalg.solve(a, rhs)
        mu_nodes = sol[:nnodes]
        density = np.asarray([
            0.5*(mu_nodes[e.node_ids[0]]+mu_nodes[e.node_ids[1]])
            for e in mesh.elements
        ], dtype=np.complex128)
        formulation = "Indirect dielectric (DLP density)"

    elif use_robin:
        # Shared Robin-BIE assembly: element-weighted alpha (tapered IBC),
        # adjacent-medium wavenumber, per-row TM-PEC EFIE override -- the same
        # system _solve_robin_bie / the bistatic dispatch solve.  For pure
        # PEC this reduces to the EFIE (TM) / MFIE (TE) exactly.
        a_sys, alpha_elements, pec_node = _assemble_robin_bie_system(
            mesh, coupled_infos, pol, k0
        )
        rhs = _robin_bie_rhs_many(
            mesh, alpha_elements, pec_node, pol, k0, elev_arr
        )
        sigma_nodes = np.linalg.solve(a_sys, rhs)[:, 0]
        density = np.asarray([
            0.5*(sigma_nodes[e.node_ids[0]]+sigma_nodes[e.node_ids[1]])
            for e in mesh.elements
        ], dtype=np.complex128)
        formulation = (
            "Robin BIE (SLP density; element-weighted alpha, TM-PEC EFIE rows)"
            if pol == "TM" else "Robin BIE / MFIE (SLP density; element-weighted alpha)"
        )

    else:
        # Safety net -- the dispatch above is exhaustive (all-Robin,
        # single-dielectric, multi-region), so this should be unreachable.
        raise ValueError(
            "compute_boundary_densities: geometry did not match any supported "
            "formulation (all-Robin, single dielectric, or multi-region)."
        )

    return {
        "quantity": "boundary_integral_layer_density",
        "is_physical_surface_current": False,
        "interpretation": (
            "Formulation-specific SLP/DLP representation density; do not "
            "interpret as electric or magnetic surface current without a "
            "formulation- and polarization-specific trace conversion."
        ),
        "formulation": formulation,
        "frequency_ghz": float(frequency_ghz),
        "elevation_deg": float(elevation_deg),
        "polarization": pol,
        "mesh_wavelength_m": float(lambda_min),
        "mesh_max_refractive_index": float(mesh_max_index),
        "mesh_material_flags": list(mesh_material_flags),
        "element_count": int(len(mesh.elements)),
        "node_count": int(nnodes),
        "centers_x": centers[:, 0].tolist(),
        "centers_y": centers[:, 1].tolist(),
        "normals_x": normals[:, 0].tolist(),
        "normals_y": normals[:, 1].tolist(),
        "lengths": lengths.tolist(),
        "density_real": np.real(density).tolist(),
        "density_imag": np.imag(density).tolist(),
        "density_abs": np.abs(density).tolist(),
        "density_phase_deg": np.degrees(np.angle(density)).tolist(),
    }


def compute_surface_currents(
    geometry_snapshot: 'Dict[str, Any]',
    frequency_ghz: 'float',
    elevation_deg: 'float',
    polarization: 'str',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    cfie_alpha: 'float' = CFIE_ALPHA_DEFAULT,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
) -> 'Dict[str, Any]':
    """Backward-compatible alias for :func:`compute_boundary_densities`.

    The old name was physically misleading: the returned arrays are indirect
    layer densities, not generally electromagnetic surface currents.
    """
    return compute_boundary_densities(
        geometry_snapshot=geometry_snapshot,
        frequency_ghz=frequency_ghz,
        elevation_deg=elevation_deg,
        polarization=polarization,
        geometry_units=geometry_units,
        material_base_dir=material_base_dir,
        cfie_alpha=cfie_alpha,
        max_panels=max_panels,
    )


def solve_adaptive_frequency_sweep(
    geometry_snapshot: 'Dict[str, Any]',
    freq_start_ghz: 'float',
    freq_stop_ghz: 'float',
    elevations_deg: 'List[float]',
    polarization: 'str',
    geometry_units: 'str' = "inches",
    material_base_dir: 'Optional[str]' = None,
    progress_callback: 'Optional[Callable[[int, int, str], None]]' = None,
    max_panels: 'int' = MAX_PANELS_DEFAULT,
    cfie_alpha: 'float' = CFIE_ALPHA_DEFAULT,
    abort_event: 'Optional[threading.Event]' = None,
    solver_method: 'str' = "auto",
    initial_points: 'int' = 11,
    max_refinements: 'int' = 3,
    db_threshold: 'float' = 1.0,
    max_total_points: 'int' = 201,
) -> 'Dict[str, Any]':
    """
    Adaptive broadband frequency sweep with automatic refinement.

    Starts with ``initial_points`` uniformly spaced frequencies, then inserts
    midpoints in intervals where adjacent samples differ by more than
    ``db_threshold`` dB.  Repeats up to ``max_refinements`` times or until
    ``max_total_points`` is reached.

    Parameters
    ----------
    freq_start_ghz, freq_stop_ghz : float
        Frequency range in GHz.
    initial_points : int
        Number of uniformly spaced initial samples (default 11).
    max_refinements : int
        Maximum number of adaptive refinement passes (default 3).
    db_threshold : float
        Insert midpoints where adjacent samples differ by more than this (default 1.0 dB).
    max_total_points : int
        Hard cap on total frequency points (default 201).

    Returns
    -------
    dict
        Same format as solve_monostatic_rcs_2d with additional metadata about
        the adaptive process (refinement_count, final_point_count).
    """

    if freq_start_ghz <= 0 or freq_stop_ghz <= 0:
        raise ValueError("Frequencies must be positive.")
    if freq_start_ghz >= freq_stop_ghz:
        raise ValueError("freq_start_ghz must be less than freq_stop_ghz.")
    if initial_points < 3:
        initial_points = 3

    # Round to the same precision used for freq_to_samples keys so midpoint
    # membership tests below compare like with like (an unrounded mid used to
    # slip past the rounded-key check and re-solve/duplicate a frequency).
    freqs = sorted({round(float(f), 12) for f in np.linspace(freq_start_ghz, freq_stop_ghz, initial_points)})
    all_samples: 'List[Dict[str, Any]]' = []
    freq_to_samples: 'Dict[float, List[Dict[str, Any]]]' = {}
    mesh_certifications: 'List[Dict[str, Any]]' = []

    def run_freqs(freq_list: 'List[float]') -> 'None':
        if not freq_list:
            return
        result = solve_monostatic_rcs_2d_certified(
            geometry_snapshot=geometry_snapshot,
            frequencies_ghz=freq_list,
            elevations_deg=elevations_deg,
            polarization=polarization,
            geometry_units=geometry_units,
            material_base_dir=material_base_dir,
            progress_callback=progress_callback,
            max_panels=max_panels,
            cfie_alpha=cfie_alpha,
            abort_event=abort_event,
            solver_method=solver_method,
        )
        mesh_certifications.append(dict(
            result.get("metadata", {}).get("mesh_convergence", {}) or {}
        ))
        for s in result.get("samples", []):
            f = round(float(s["frequency_ghz"]), 12)
            freq_to_samples.setdefault(f, []).append(s)
            all_samples.append(s)

    run_freqs(freqs)
    refinement_count = 0

    for _ in range(max_refinements):
        if abort_event is not None and abort_event.is_set():
            break
        if len(freqs) >= max_total_points:
            break

        # For each elevation, find intervals needing refinement.
        new_freqs: 'set' = set()
        sorted_freqs = sorted(freqs)
        for elev in elevations_deg:
            db_at_freq = {}
            for f in sorted_freqs:
                for s in freq_to_samples.get(round(f, 12), []):
                    if abs(s["theta_inc_deg"] - elev) < 0.01:
                        db_at_freq[f] = s["rcs_db"]
                        break

            for i in range(len(sorted_freqs) - 1):
                f0, f1 = sorted_freqs[i], sorted_freqs[i + 1]
                db0 = db_at_freq.get(f0)
                db1 = db_at_freq.get(f1)
                if db0 is not None and db1 is not None:
                    if abs(db1 - db0) > db_threshold:
                        mid = round(0.5 * (f0 + f1), 12)
                        if mid not in freq_to_samples and mid != f0 and mid != f1:
                            new_freqs.add(mid)

        if not new_freqs:
            break

        remaining = max_total_points - len(freqs)
        if remaining <= 0:
            break
        new_list = sorted(new_freqs)[:remaining]
        run_freqs(new_list)
        freqs = sorted(set(freqs) | set(new_list))
        refinement_count += 1

    return {
        "solver": "2d_bie_mom_rcs",
        "scattering_mode": "monostatic_adaptive",
        "amplitude_convention": RCS_AMPLITUDE_CONVENTION,
        "polarization": _canonical_user_polarization_label(polarization),
        "samples": sorted(all_samples, key=lambda s: (s["frequency_ghz"], s["theta_inc_deg"])),
        "metadata": {
            "formulation": "adaptive frequency sweep",
            "initial_points": initial_points,
            "final_point_count": len(freqs),
            "refinement_count": refinement_count,
            "db_threshold": db_threshold,
            "freq_start_ghz": freq_start_ghz,
            "freq_stop_ghz": freq_stop_ghz,
            "mesh_convergence_certified": bool(
                mesh_certifications
                and all(
                    bool(record.get("passed", False))
                    for record in mesh_certifications
                )
            ),
            "mesh_convergence_batches": mesh_certifications,
            "certified_entry_point": True,
        },
    }
