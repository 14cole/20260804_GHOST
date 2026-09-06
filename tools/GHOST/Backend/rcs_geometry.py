"""2-D materials, geometry validation, and panel/linear-mesh construction."""

import cmath
import csv
import math
import os
from dataclasses import dataclass
import numpy as np
from thin_sheet import ThinLayerDefinition
from geometry_io import is_legacy_tabulated_row, material_filename_from_row
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from rcs_constants import (
    C0,
    DEFAULT_PANELS_PER_WAVELENGTH,
    EPS,
    ETA0,
    MATERIAL_SINGULAR_TOL,
    MAX_PANELS_DEFAULT,
    MIN_EXPLICIT_PANELS_PER_WAVELENGTH,
    VIRTUAL_SHEET_REGION_START,
)
from rcs_special import (
    _complex_hankel_backend_name,
)


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
            if len(row) > 1 and str(row[1]).lower() == "thin_dielectric":
                impedance_models[flag] = ThinLayerDefinition.from_row(row)
                continue
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

        for model in impedance_models.values():
            if isinstance(model, ThinLayerDefinition) and model.dielectric_flag not in dielectric_models:
                raise ValueError(f"Thin layer references undefined dielectric flag {model.dielectric_flag}.")
        return cls(impedance_models=impedance_models, dielectric_models=dielectric_models)

    def get_impedance(self, flag: 'int', freq_ghz: 'float', arc_s: 'Optional[float]' = None) -> 'complex':
        if flag <= 0:
            return 0.0 + 0.0j
        model = self.impedance_models.get(flag)
        if model is None:
            raise ValueError(f"Undefined IBC flag {flag}.")
        if isinstance(model, ThinLayerDefinition):
            raise ValueError("A thin dielectric layer is a transmitting sheet, not an opaque IBC. Assign it only to TYPE 1.")
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
    """Read a nominal material CSV in Hz, with or without FREDDY's header."""

    rows: 'List[List[float]]' = []
    with open(path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        first_row = True
        for raw_row in reader:
            lineno = reader.line_num
            if not raw_row or all(not str(value).strip() for value in raw_row):
                continue
            if first_row:
                first_row = False
                header = [str(value).strip().lower() for value in raw_row]
                if header == expected_header:
                    continue
                try:
                    [float(value) for value in raw_row]
                except ValueError as exc:
                    raise ValueError(
                        f"CSV material file '{path}' must have header "
                        f"{','.join(expected_header)} or contain headerless numeric rows in Hz; "
                        f"found {','.join(header)}."
                    ) from exc
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
        target = min_wavelength / n_wave
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("The controlling mesh wavelength must be positive and finite.")
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


def _check_segment_orientation_or_raise(
    segments: 'List[Dict[str, Any]]',
    meters_scale: 'float' = 1.0,
) -> 'None':
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

    meters_scale = float(meters_scale)
    if not math.isfinite(meters_scale) or meters_scale <= 0.0:
        raise ValueError("meters_scale must be positive and finite.")
    chains = chains_from_snapshot_segments(segments)
    coordinates = [point for chain in chains for point in chain.points]
    if coordinates:
        xs = [point[0] for point in coordinates]
        ys = [point[1] for point in coordinates]
        diagonal_m = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) * meters_scale
    else:
        diagonal_m = 0.0
    # geometry_io's default tolerance is expressed in the coordinate unit and
    # floors the span at one coordinate unit.  Derive that same policy in
    # meters, then convert it back for the raw snapshot so meter and inch
    # representations receive identical winding decisions.
    tolerance_m = max(1.0e-12, 1.0e-9 * max(diagonal_m, 1.0))
    findings = check_orientation_consistency(
        chains, tol=tolerance_m / meters_scale
    )
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

    meters_scale = float(meters_scale)
    if not math.isfinite(meters_scale) or meters_scale <= 0.0:
        raise ValueError("meters_scale must be positive and finite.")

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
            p1 = (x1 * meters_scale, y1 * meters_scale)
            p2 = (x2 * meters_scale, y2 * meters_scale)
            if _points_close(p1, p2, EPS):
                raise ValueError(f"Segment '{seg_name}' primitive {prim_idx + 1} has near-zero length.")
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
    # From this point onward every geometric predicate operates in meters.
    # Keeping the coordinates, span floor, and absolute tolerances in one unit
    # makes preflight decisions invariant to the input's declared unit system.
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
    snap_tol_m = 1.0e-9
    crack_floor_m = 1.0e-12
    endpoint_list = sorted(set(all_points))
    for i in range(len(endpoint_list)):
        px, py = endpoint_list[i]
        j = i + 1
        while j < len(endpoint_list) and endpoint_list[j][0] - px <= tol:
            qx, qy = endpoint_list[j]
            j += 1
            gap = math.hypot(qx - px, qy - py)
            if gap > tol or gap <= crack_floor_m:
                continue
            if gap > snap_tol_m:
                raise ValueError(
                    f"Geometry crack: endpoints ({px:.9g}, {py:.9g}) m and ({qx:.9g}, {qy:.9g}) m "
                    f"are {gap:.3g} m apart -- close enough to look connected, but "
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
                        f"endpoint, overlapping for {min(lu, lv):.3g} m. "
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
    _check_segment_orientation_or_raise(segments, meters_scale)

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
        sheet_model = materials.impedance_models.get(_parse_flag(props[2]))
        if seg_type == 1 and isinstance(sheet_model, ThinLayerDefinition):
            material_flags.add(sheet_model.dielectric_flag)
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
        if isinstance(materials.impedance_models.get(flag), ThinLayerDefinition):
            if int(panel.seg_type) != 1:
                raise ValueError("Thin dielectric layers can be assigned only to a free TYPE 1 sheet.")
            return 0.0j  # No surface-impedance substitution; uses a separate sheet operator.
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
        thin_layer = isinstance(materials.impedance_models.get(panel.ibc_flag), ThinLayerDefinition)
        if thin_layer:
            bc_kind = "thin_layer"
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

