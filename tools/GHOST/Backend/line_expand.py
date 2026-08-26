#!/usr/bin/env python3
"""
Line-expanded feature components on a clean platform body.

A feature that is locally two-dimensional (a panel gap, a seal, a lap joint,
a step, a coating edge) is characterised by a 2D cross-section solve and then
expanded along its known 3D perimeter on the vehicle skin. Reuse requires the
same local cross-section, clean host stack, frequency, conventions, and
compatible curvature. A differential seam coupon must be an integrated
perturbation of that host; detached TYPE-1 absorber cards are not a validated
coupon termination and failed the cross-formulation phase scope sweep. The
expanded component is added to the BoR body's amplitude to build a
reduced-order vehicle signature.

Two quantities, in this order:

1.  **Seam coefficient** ``dA(phi_cut, pol)`` -- the DIFFERENTIAL 2D far-field
    amplitude of the joint: the featured cross-section minus the identical
    smooth one.  Differencing is what keeps the smooth skin from being counted
    twice (it is already in the BoR body) and cancels the coupon's own edge
    diffraction to first order.  The cut must be drawn with the feature
    centred on the 2D origin (0, 0) so the coefficient carries no placement
    phase of its own -- the placement phase comes from the perimeter.

2.  **Perimeter expansion** -- with the coefficient in hand, the ordered
    segment tangent and interpolated endpoint outward normals define the local
    seam frame, and

        F(d) = (1 / 4pi) * Int_perimeter
               [w_TM e^{j psi_TM} dA_TM + w_TE e^{j psi_TE} dA_TE]
               * e^{+2jk d.r(s)} ds

    evaluated in the local seam frame (tangent, surface normal).  Both solvers
    reference far-field phase to their drawing origin with the same
    exp(+2jk d.r') two-way factor (e^{+jwt}), so if the BoR generatrix and the
    perimeter file share a coordinate frame, the modeled translation phase is
    consistent without a hand-applied range correction.

``psi_TM`` and ``psi_TE`` are legacy empirical local-coefficient phase
mappings between the 2D solver's bare-integral amplitudes and the BoR
far-field convention. They are applied before the polarization contributions
are added.  The original ring-calibration fixture is not shipped, but the
repository now carries a replacement all-GHOST circumferential PEC-groove
regression in ``tests/test_feature_reconstruction_physics.py``.  That fixture
checks this mapping for one controlled case; each feature family still needs
the independent BoR or full-3D validation described in
``FEATURE_VALIDATION_GUIDE.md``. Magnitudes are absolute: the |A|^2/(4k) and
4pi|F|^2 normalisations are not fitted. The same expansion can be placed on an
external non-BoR body response when its matching triangle surface supplies
skin validation, outward normals, and optional geometric shadowing; the body
field itself is added in the radar frame and is not recomputed here.

Conventions:
  - e^{+jwt}; look directions ``d`` are COMING-FROM unit vectors.
  - 2D cut frame: +x = b_hat (in-surface, across the seam), +y = n_hat
    (outward surface normal).  Normal incidence on the outward face is
    therefore the 2D solver's 90 deg.
  - 2D TM = E along the seam tangent (out of the cut plane); TE = E in it.
  - sigma_3d = 4 pi |F|^2, m^2.
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from occluder import PackedVisibility, visible_query_adapter

C0 = 299792458.0

# This taper is part of the reusable line-response model contract, not merely
# a plotting preference.  Feature-library manifests bind this exact value so a
# future numerical-default change cannot silently reuse an older validation.
GRAZING_TAPER_DEG = 10.0

# Legacy local-polarization inter-solver calibration constants.  The original
# ring-calibration fixture is unavailable; the checked-in replacement PEC
# ring-groove regression tests this exact software path but does not establish
# transferability to another feature family. See FEATURE_VALIDATION_GUIDE.md
# for the required independent validation before release. They multiply the
# 2-D TM and TE coefficients
# BEFORE their projections are added. A finite perimeter
# mixes both local coefficients into an HH/VV channel away from broadside, so a
# phase measured from an already-summed output channel cannot simply be applied
# back to one coefficient. Magnitudes are absolute--there is no fitted
# amplitude scale. Re-measure if either solver's far-field convention changes.
PSI_VV_DEG = -9.2       # theta-pol-dominant channel (2D TE coefficient)
PSI_HH_DEG = 166.9      # phi-pol-dominant channel   (2D TM coefficient)
# No runtime grazing-angle gate is claimed here. The replacement ring fixture
# covers one PEC groove, and a single angular constant is not transferable
# across arbitrary cross-sections, materials, corners, and junctions. Validate
# the chosen coupon and placement envelope as described in
# FEATURE_VALIDATION_GUIDE.md.


# -----------------------------------------------------------------------------
# 1. Seam coefficients from a paired 2D cross-section solve
# -----------------------------------------------------------------------------

@dataclass
class SeamCoefficients:
    """Differential 2D amplitude of a joint, per polarization, one frequency.

    ``phi_deg`` is the 2D coming-from angle in the cut frame (90 deg = normal
    incidence on the outward face).  ``dA_tm``/``dA_te`` are complex.
    """

    frequency_ghz: 'float'
    phi_deg: 'np.ndarray'
    dA_tm: 'np.ndarray'
    dA_te: 'np.ndarray'
    label: 'str' = ""

    def __post_init__(self) -> 'None':
        self.frequency_ghz = float(self.frequency_ghz)
        self.phi_deg = np.asarray(self.phi_deg, dtype=float)
        self.dA_tm = np.asarray(self.dA_tm, dtype=complex)
        self.dA_te = np.asarray(self.dA_te, dtype=complex)
        if (not math.isfinite(self.frequency_ghz)) or self.frequency_ghz <= 0.0:
            raise ValueError(
                "SeamCoefficients: frequency_ghz must be positive and finite."
            )
        if self.phi_deg.ndim != 1:
            raise ValueError("SeamCoefficients: coefficient arrays must be 1-D.")
        if not (self.phi_deg.shape == self.dA_tm.shape == self.dA_te.shape):
            raise ValueError("SeamCoefficients: phi/dA_tm/dA_te shape mismatch.")
        if self.phi_deg.size < 3:
            raise ValueError("SeamCoefficients: need at least 3 angles.")
        if not (
            np.all(np.isfinite(self.phi_deg))
            and np.all(np.isfinite(self.dA_tm.real))
            and np.all(np.isfinite(self.dA_tm.imag))
            and np.all(np.isfinite(self.dA_te.real))
            and np.all(np.isfinite(self.dA_te.imag))
        ):
            raise ValueError("SeamCoefficients: angles/amplitudes must be finite.")
        order = np.argsort(self.phi_deg)
        self.phi_deg = self.phi_deg[order]
        self.dA_tm = self.dA_tm[order]
        self.dA_te = self.dA_te[order]
        if np.any(np.diff(self.phi_deg) <= 0.0):
            raise ValueError(
                "SeamCoefficients: phi angles must be unique after sorting."
            )

    def sample(self, phi_deg: 'np.ndarray') -> 'Tuple[np.ndarray, np.ndarray]':
        """Interpolate on the COMPLEX coefficient (never on |A| and phase).

        Queries outside the tabulated support raise.  A missing characterized
        incidence direction is not a physical zero.
        """
        q = np.asarray(phi_deg, dtype=float)
        if not np.all(np.isfinite(q)):
            raise ValueError("SeamCoefficients sample angles must be finite.")
        lo, hi = float(self.phi_deg[0]), float(self.phi_deg[-1])
        tolerance = 1.0e-10
        if np.any(q < lo - tolerance) or np.any(q > hi + tolerance):
            bad = q[(q < lo - tolerance) | (q > hi + tolerance)]
            raise ValueError(
                f"Seam coefficient query {float(bad.flat[0]):g} deg is "
                f"outside characterized support [{lo:g}, {hi:g}] deg."
            )
        q = np.clip(q, lo, hi)
        out = []
        for src in (self.dA_tm, self.dA_te):
            out.append(
                np.interp(q, self.phi_deg, src.real)
                + 1j * np.interp(q, self.phi_deg, src.imag)
            )
        return out[0], out[1]


def coefficients_from_2d(snapshot: 'Dict',
                         frequency_ghz: 'float',
                         phi_deg: 'Sequence[float]',
                         geometry_units: 'str' = "meters",
                         material_base_dir: 'Optional[str]' = None,
                         label: 'str' = "",
                         solver_kwargs: 'Optional[Dict]' = None) -> "SeamCoefficients":
    """Full-object 2D amplitude coefficient from a SINGLE cross-section solve.

    Unlike ``seam_coefficients_from_2d`` (which differences a featured and a
    smooth coupon for a perturbation), this returns the raw 2D far-field
    amplitude of a stand-alone object -- the airfoil/strip cross-section of a
    wing or fin, line-expanded along its span.  The ``2L^2/lambda`` strip
    factor is produced automatically by expand_perimeter's 1/(4pi) prefactor:
    with sigma_2d = |A|^2/(4k),  sigma_3d = 4pi|F|^2 = (2 L^2 / lambda) sigma_2d
    at broadside, which is the physical-optics plate/strip relation.
    """
    from rcs_solver import solve_monostatic_rcs_2d_certified

    kw = dict(solver_kwargs or {})
    angles = [float(a) for a in phi_deg]
    result = solve_monostatic_rcs_2d_certified(
        geometry_snapshot=snapshot,
        frequencies_ghz=[float(frequency_ghz)],
        elevations_deg=angles,
        geometry_units=geometry_units,
        material_base_dir=material_base_dir,
        **kw,
    )
    channels = result.get("co_solved_samples", {}) or {}
    amp: 'Dict[str, np.ndarray]' = {}
    for pol, exported in (("TM", "HH"), ("TE", "VV")):
        a = _amp_by_angle({"samples": channels.get(exported, [])})
        amp[pol] = np.array(
            [a[round(x / 1e-6) * 1e-6] for x in angles], dtype=complex
        )
    return SeamCoefficients(float(frequency_ghz), np.array(angles, dtype=float),
                            amp["TM"], amp["TE"], label=label)


def _amp_by_angle(result: 'Dict', tol: 'float' = 1e-6) -> 'Dict[float, complex]':
    out: 'Dict[float, complex]' = {}
    for s in result["samples"]:
        ang = round(float(s["theta_inc_deg"]) / tol) * tol
        out[ang] = complex(float(s["rcs_amp_real"]), float(s["rcs_amp_imag"]))
    return out


def seam_coefficients_from_2d(featured_snapshot: 'Dict',
                              smooth_snapshot: 'Dict',
                              frequency_ghz: 'float',
                              phi_deg: 'Sequence[float]',
                              geometry_units: 'str' = "meters",
                              material_base_dir: 'Optional[str]' = None,
                              label: 'str' = "",
                              solver_kwargs: 'Optional[Dict]' = None) -> 'SeamCoefficients':
    """Solve the featured and smooth cross-sections and difference them.

    Both snapshots MUST share a mesh everywhere except at the feature (same
    primitive endpoints outside the disturbed region) or the difference is
    dominated by discretisation noise rather than by the joint.
    """
    from rcs_solver import solve_monostatic_rcs_2d_certified

    kw = dict(solver_kwargs or {})
    angles = [float(a) for a in phi_deg]
    shared = dict(
        frequencies_ghz=[float(frequency_ghz)],
        elevations_deg=angles,
        geometry_units=geometry_units,
        material_base_dir=material_base_dir,
        **kw,
    )
    featured = solve_monostatic_rcs_2d_certified(
        geometry_snapshot=featured_snapshot, **shared
    )
    smooth = solve_monostatic_rcs_2d_certified(
        geometry_snapshot=smooth_snapshot, **shared
    )
    featured_channels = featured.get("co_solved_samples", {}) or {}
    smooth_channels = smooth.get("co_solved_samples", {}) or {}
    dA: 'Dict[str, np.ndarray]' = {}
    for pol, exported in (("TM", "HH"), ("TE", "VV")):
        fa = _amp_by_angle({"samples": featured_channels.get(exported, [])})
        sa = _amp_by_angle({"samples": smooth_channels.get(exported, [])})
        dA[pol] = np.array([fa[round(a / 1e-6) * 1e-6] - sa[round(a / 1e-6) * 1e-6]
                            for a in angles], dtype=complex)
    return SeamCoefficients(float(frequency_ghz), np.array(angles, dtype=float),
                            dA["TM"], dA["TE"], label=label)


# -----------------------------------------------------------------------------
# 2. Perimeters and surface normals
# -----------------------------------------------------------------------------

def read_perimeter_txt(path: 'str', scale: 'float' = 1.0) -> 'np.ndarray':
    """Read a segmented perimeter file: one segment per line,

        x1 y1 z1 x2 y2 z2

    Returns an (n_seg, 2, 3) array.  Blank lines and '#' comments are skipped.
    Segments are checked for head-to-tail continuity; a chain whose last point
    returns to its first is a closed loop (a door outline).
    """
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Perimeter scale must be a positive finite value.")
    segs: 'List[List[float]]' = []
    with open(path, "r") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 6:
                raise ValueError(f"{path}:{lineno}: expected 6 numbers "
                                 f"(x1 y1 z1 x2 y2 z2), got {len(parts)}.")
            try:
                values = [float(p) * scale for p in parts]
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{lineno}: perimeter coordinates must be numeric."
                ) from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"{path}:{lineno}: perimeter coordinates must be finite."
                )
            segs.append(values)
    if not segs:
        raise ValueError(f"{path}: no segments found.")
    arr = np.asarray(segs, dtype=float).reshape(len(segs), 2, 3)
    span = float(np.max(np.abs(arr))) or 1.0
    for i in range(len(arr) - 1):
        gap = float(np.linalg.norm(arr[i, 1] - arr[i + 1, 0]))
        if gap > 1e-6 * span:
            raise ValueError(f"{path}: segments {i + 1} and {i + 2} are not "
                             f"head-to-tail (gap {gap:.3e}).  Perimeters must "
                             f"chain as drawn, like .geo primitives.")
    return arr


def is_closed_loop(segments: 'np.ndarray', tol_rel: 'float' = 1e-6) -> 'bool':
    span = float(np.max(np.abs(segments))) or 1.0
    return bool(np.linalg.norm(segments[-1, 1] - segments[0, 0]) <= tol_rel * span)


def surface_of_revolution_normal(generatrix: 'np.ndarray') -> 'Callable[[np.ndarray], np.ndarray]':
    """Outward-normal callable for the skin drawn by a BoR generatrix.

    ``generatrix`` is the (rho, z) polyline in BoR convention (drawn +z -> -z,
    so the left-of-travel normal faces outward).  The returned callable maps
    (n, 3) vehicle-frame points to (n, 3) unit outward normals by finding the
    nearest generatrix segment in the (rho, z) half-plane and rotating its
    profile normal to the point's azimuth.
    """
    gen = np.asarray(generatrix, dtype=float)
    if gen.ndim != 2 or gen.shape[1] != 2 or len(gen) < 2:
        raise ValueError("generatrix must be an (n, 2) array of (rho, z).")
    if not np.all(np.isfinite(gen)):
        raise ValueError("generatrix contains NaN or infinite coordinates.")
    if np.any(gen[:, 0] < 0.0):
        raise ValueError("generatrix rho coordinates must be non-negative.")
    p0, p1 = gen[:-1], gen[1:]
    seg = p1 - p0
    seg_len2 = np.sum(seg ** 2, axis=1)
    if np.any(seg_len2 <= 0.0):
        raise ValueError("generatrix has a zero-length segment.")
    # left of travel in (rho, z): (-dz, drho)
    prof_n = np.column_stack([-seg[:, 1], seg[:, 0]])
    prof_n /= np.linalg.norm(prof_n, axis=1)[:, None]

    def normals(points: 'np.ndarray') -> 'np.ndarray':
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if pts.ndim != 2 or pts.shape[1] != 3 or not np.all(np.isfinite(pts)):
            raise ValueError("normal query points must be finite (n, 3) coordinates.")
        rho = np.hypot(pts[:, 0], pts[:, 1])
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        q = np.column_stack([rho, pts[:, 2]])
        # nearest point on each segment
        t = np.clip(((q[:, None, :] - p0[None, :, :]) * seg[None, :, :]).sum(-1)
                    / seg_len2[None, :], 0.0, 1.0)
        foot = p0[None, :, :] + t[:, :, None] * seg[None, :, :]
        d2 = np.sum((q[:, None, :] - foot) ** 2, axis=-1)
        idx = np.argmin(d2, axis=1)
        n_rho, n_z = prof_n[idx, 0], prof_n[idx, 1]
        return np.column_stack([n_rho * np.cos(phi), n_rho * np.sin(phi), n_z])

    return normals


def perimeter_surface_deviation(segments: 'np.ndarray',
                                generatrix: 'np.ndarray',
                                samples_per_segment: 'int' = 33) -> 'float':
    """Max sampled distance from a perimeter chord to the revolved surface.

    A polygonised outline of a curved door cuts chords inside the skin; this
    is the preflight number to compare against lambda/50 before trusting the
    placement phase.  Sampling only endpoints misses the chord sag entirely,
    because those endpoints are normally the points deliberately placed on the
    skin.
    """
    gen = np.asarray(generatrix, dtype=float)
    p0, p1 = gen[:-1], gen[1:]
    seg = p1 - p0
    seg_len2 = np.sum(seg ** 2, axis=1)
    if np.any(seg_len2 <= 0.0):
        raise ValueError("generatrix has a zero-length segment.")
    per = np.asarray(segments, dtype=float)
    if per.ndim != 3 or per.shape[1:] != (2, 3):
        raise ValueError("segments must have shape (n, 2, 3).")
    ns = max(3, int(samples_per_segment))
    u = np.linspace(0.0, 1.0, ns)
    pts = (per[:, 0, None, :] +
           u[None, :, None] * (per[:, 1, None, :] - per[:, 0, None, :]))
    pts = pts.reshape(-1, 3)
    q = np.column_stack([np.hypot(pts[:, 0], pts[:, 1]), pts[:, 2]])
    t = np.clip(((q[:, None, :] - p0[None, :, :]) * seg[None, :, :]).sum(-1)
                / seg_len2[None, :], 0.0, 1.0)
    foot = p0[None, :, :] + t[:, :, None] * seg[None, :, :]
    return float(np.sqrt(np.min(np.sum((q[:, None, :] - foot) ** 2, axis=-1), axis=1)).max())


# -----------------------------------------------------------------------------
# 3. The expansion
# -----------------------------------------------------------------------------

def _pol_unit_vectors(d: 'np.ndarray') -> 'Tuple[np.ndarray, np.ndarray]':
    """Incident E unit vectors for VV (theta-pol) and HH (phi-pol).

    ``d`` is the coming-from direction; theta from +z, phi about it, matching
    BOR_CONVENTIONS.md.  Degenerate axial looks fall back to a fixed basis
    (where VV and HH are equivalent by symmetry anyway).
    """
    d = np.atleast_2d(np.asarray(d, dtype=float))
    st = np.hypot(d[:, 0], d[:, 1])
    e_hh = np.zeros_like(d)
    e_vv = np.zeros_like(d)
    axial = st < 1e-12
    ph = np.arctan2(d[:, 1], d[:, 0])
    ct = d[:, 2]
    e_hh[:, 0], e_hh[:, 1] = -np.sin(ph), np.cos(ph)
    e_vv[:, 0] = ct * np.cos(ph)
    e_vv[:, 1] = ct * np.sin(ph)
    e_vv[:, 2] = -st
    if np.any(axial):
        e_hh[axial] = np.array([0.0, 1.0, 0.0])
        e_vv[axial] = np.array([1.0, 0.0, 0.0])
    return e_vv, e_hh


def _transverse_seam_basis(t_hat: 'np.ndarray',
                           b_hat: 'np.ndarray',
                           d: 'np.ndarray',
                           tol: 'float' = 1e-12) -> 'Tuple[np.ndarray, np.ndarray]':
    """Return signed local TM/TE electric-field axes transverse to ``d``.

    The 2D TM coefficient is referenced to electric field along the invariant
    seam direction.  For conical incidence that direction is not itself a
    valid electric-field axis, because it has a component along the propagation
    direction.  Its physical polarization axis is therefore the normalized
    projection into the plane transverse to ``d``::

        e_TM = (t - (t.d)d) / |t - (t.d)d|
        e_TE = e_TM x d = (t x d) / |t x d|

    The cross product fixes the TE sign; using ``sqrt(1 - (t.E)**2)`` loses
    that sign and can turn a polarization-isotropic coefficient into spurious
    cross-pol.

    At exactly seam-parallel incidence ``t x d`` vanishes and the TM/TE labels
    have no unique limiting orientation.  Such a surface piece is at grazing
    and receives zero illumination in ``expand_perimeter``.  A deterministic
    orthonormal fallback based on the local +b cut direction is nevertheless
    returned so near-degenerate floating-point input remains finite.
    """
    t = np.atleast_2d(np.asarray(t_hat, dtype=float))
    b = np.atleast_2d(np.asarray(b_hat, dtype=float))
    look = np.asarray(d, dtype=float).reshape(3)
    if t.shape != b.shape or t.shape[1] != 3:
        raise ValueError("t_hat and b_hat must be matching (n, 3) arrays.")

    te_raw = np.cross(t, look[None, :])
    te_norm = np.linalg.norm(te_raw, axis=1)
    regular = te_norm > float(tol)
    e_te = np.zeros_like(t)
    e_te[regular] = te_raw[regular] / te_norm[regular, None]

    # At seam-parallel incidence b is transverse to d.  Project it anyway to
    # make the fallback robust to roundoff and slightly non-orthogonal input.
    for idx in np.flatnonzero(~regular):
        seed = b[idx] - float(np.dot(b[idx], look)) * look
        seed_norm = float(np.linalg.norm(seed))
        if seed_norm <= float(tol):
            # Defensive last resort for malformed frames: choose the Cartesian
            # axis least aligned with d and project it into the transverse plane.
            axis = np.zeros(3, dtype=float)
            axis[int(np.argmin(np.abs(look)))] = 1.0
            seed = axis - float(np.dot(axis, look)) * look
            seed_norm = float(np.linalg.norm(seed))
        e_te[idx] = seed / seed_norm

    # d x e_TE is the projected +t direction and preserves
    # e_TE = e_TM x d under the coming-from convention.
    e_tm = np.cross(look[None, :], e_te)
    e_tm_norm = np.linalg.norm(e_tm, axis=1)
    e_tm /= e_tm_norm[:, None]
    return e_tm, e_te


def _subdivide(segments: 'np.ndarray', max_len: 'float',
               segment_normals: 'Optional[np.ndarray]' = None
               ) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]':
    """Split each segment so no piece exceeds ``max_len``.

    Returns (start points, unit tangents, lengths, midpoint normals).  When
    ``segment_normals`` is supplied it has shape (n_segments, 2, 3) and its
    endpoint vectors are linearly interpolated to each subdivided midpoint.
    The phase integral over each piece is evaluated in closed form, so
    subdivision only has to resolve variation of the coefficient and frame,
    not of the phase.
    """
    segments = np.asarray(segments, dtype=float)
    max_len = float(max_len)
    if segments.ndim != 3 or segments.shape[1:] != (2, 3) or len(segments) == 0:
        raise ValueError("perimeter must have shape (n_segments, 2, 3).")
    if not np.all(np.isfinite(segments)):
        raise ValueError("perimeter contains NaN or infinite coordinates.")
    if not math.isfinite(max_len) or max_len <= 0.0:
        raise ValueError("maximum subdivision length must be positive and finite.")
    normals = None
    if segment_normals is not None:
        normals = np.asarray(segment_normals, dtype=float)
        if normals.shape != segments.shape or not np.all(np.isfinite(normals)):
            raise ValueError(
                "segment_normals must contain two finite 3-vectors per segment."
            )
        if np.any(np.linalg.norm(normals, axis=2) <= 1.0e-12):
            raise ValueError("segment_normals contains a zero-length normal.")
    starts: 'List[np.ndarray]' = []
    tangents: 'List[np.ndarray]' = []
    lengths: 'List[float]' = []
    midpoint_normals: 'List[np.ndarray]' = []
    for segment_index, (a, b) in enumerate(segments):
        v = b - a
        L = float(np.linalg.norm(v))
        if L <= 0.0:
            raise ValueError("perimeter contains a zero-length segment.")
        u = v / L
        n = max(1, int(math.ceil(L / max_len)))
        dL = L / n
        for i in range(n):
            starts.append(a + u * (i * dL))
            tangents.append(u)
            lengths.append(dL)
            if normals is not None:
                fraction = (i + 0.5) / n
                midpoint_normals.append(
                    (1.0 - fraction) * normals[segment_index, 0]
                    + fraction * normals[segment_index, 1]
                )
    if not starts:
        raise ValueError("perimeter has no non-degenerate segments.")
    return (
        np.asarray(starts), np.asarray(tangents), np.asarray(lengths),
        None if normals is None else np.asarray(midpoint_normals),
    )


def prepare_perimeter_frame(
    segments: 'np.ndarray',
    max_piece_length_m: 'float',
    *,
    normal_fn: 'Optional[Callable[[np.ndarray], np.ndarray]]' = None,
    segment_normals: 'Optional[np.ndarray]' = None,
) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]':
    """Return the exact subdivided geometry/frame used by line expansion.

    The shared helper prevents production applicability checks from drifting
    away from the numerical solver.  It returns piece starts, physical chord
    tangents, lengths, midpoints, normalized skin normals, and the chord
    tangent projected into the local skin plane (the solver's signed ``+t``).
    """

    if segment_normals is not None and normal_fn is not None:
        raise ValueError("supply segment_normals or normal_fn, not both.")
    r0, path_t_hat, dL, supplied_normals = _subdivide(
        np.asarray(segments, dtype=float),
        float(max_piece_length_m),
        segment_normals=segment_normals,
    )
    r_mid = r0 + path_t_hat * (dL[:, None] / 2.0)
    if supplied_normals is None:
        if normal_fn is None:
            raise ValueError(
                "normal_fn is required when segment_normals is absent."
            )
        n_hat = np.asarray(normal_fn(r_mid), dtype=float)
    else:
        n_hat = supplied_normals
    if n_hat.shape != r_mid.shape or not np.all(np.isfinite(n_hat)):
        raise ValueError(
            "normal_fn must return one finite 3-vector per perimeter point."
        )
    normal_norm = np.linalg.norm(n_hat, axis=1)
    if np.any(normal_norm <= 1.0e-12):
        raise ValueError("normal_fn returned a zero-length surface normal.")
    n_hat = n_hat / normal_norm[:, None]

    # Translation follows the physical chord.  The local 2-D response frame
    # uses its projection into the skin tangent plane.
    frame_t_hat = path_t_hat - (
        np.sum(path_t_hat * n_hat, axis=1)[:, None] * n_hat
    )
    tangent_norm = np.linalg.norm(frame_t_hat, axis=1)
    if np.any(tangent_norm < 1e-9):
        raise ValueError(
            "a perimeter segment is normal to the skin -- check that the "
            "perimeter and the generatrix share a frame."
        )
    frame_t_hat /= tangent_norm[:, None]
    return r0, path_t_hat, dL, r_mid, n_hat, frame_t_hat


def expand_perimeter(segments: 'np.ndarray',
                     coefficients: 'SeamCoefficients',
                     normal_fn: 'Optional[Callable[[np.ndarray], np.ndarray]]',
                     directions: 'np.ndarray',
                     frequency_ghz: 'Optional[float]' = None,
                     psi_tm_deg: 'float' = 0.0,
                     psi_te_deg: 'float' = 0.0,
                     grazing_taper_deg: 'float' = GRAZING_TAPER_DEG,
                     max_piece_wavelengths: 'float' = 0.05,
                     max_piece_length_m: 'Optional[float]' = None,
                     occluder=None,
                     shadow_points: 'Optional[np.ndarray]' = None,
                     segment_normals: 'Optional[np.ndarray]' = None,
                     cancel_check: 'Optional[Callable[[], bool]]' = None,
                     progress_callback: 'Optional[Callable[[int, int], None]]' = None,
                     _shadow_visibility=None,
                     ) -> 'Dict[str, np.ndarray]':
    """Expand a seam coefficient along a 3D perimeter.

    ``segments``    (n_seg, 2, 3) vehicle-frame meters, head-to-tail.
    ``segment_normals`` optional (n_seg, 2, 3) endpoint outward normals.
                    When supplied these define the local skin frame and
                    ``normal_fn`` must be None.  The endpoint normals are
                    interpolated along each segment before normalization.
    ``directions``  (n_dir, 3) unit COMING-FROM look directions.
    ``psi_tm_deg``  candidate inter-solver phase for the 2D TM coefficient
                    (== PSI_HH_DEG, since phi-pol/HH is driven by TM). Applied
                    to the COEFFICIENT so cross-pol is consistent too.
    ``psi_te_deg``  same for the 2D TE coefficient (== PSI_VV_DEG).

    Pass 0/0 when comparing against an independent reference to determine
    whether any phase mapping is required. Do not treat the legacy defaults as
    certified without that comparison.

    Returns ``{"F_vv", "F_hh", "F_vh"}`` complex arrays over directions in the
    physical 3-D amplitude normalization (sigma = 4 pi |F|^2).
    """
    freq = float(frequency_ghz if frequency_ghz is not None else coefficients.frequency_ghz)
    if not math.isfinite(freq) or freq <= 0.0:
        raise ValueError("frequency_ghz must be positive and finite.")
    if abs(freq - coefficients.frequency_ghz) > 1e-6:
        raise ValueError(f"coefficient table is for {coefficients.frequency_ghz} GHz, "
                         f"asked for {freq} GHz.")
    k = 2.0 * math.pi * freq * 1e9 / C0
    lam = C0 / (freq * 1e9)
    phase_tm = float(psi_tm_deg)
    phase_te = float(psi_te_deg)
    if not math.isfinite(phase_tm) or not math.isfinite(phase_te):
        raise ValueError("line phase mappings must be finite degrees.")
    cal_tm = np.exp(1j * math.radians(phase_tm))
    cal_te = np.exp(1j * math.radians(phase_te))

    max_piece_wavelengths = float(max_piece_wavelengths)
    if (not math.isfinite(max_piece_wavelengths)
            or max_piece_wavelengths <= 0.0):
        raise ValueError("max_piece_wavelengths must be positive and finite.")
    if max_piece_length_m is None:
        maximum_piece_length = max_piece_wavelengths * lam
    else:
        maximum_piece_length = float(max_piece_length_m)
        if not math.isfinite(maximum_piece_length) or maximum_piece_length <= 0.0:
            raise ValueError("max_piece_length_m must be positive and finite.")
    r0, path_t_hat, dL, r_mid, n_hat, frame_t_hat = (
        prepare_perimeter_frame(
            np.asarray(segments, dtype=float),
            maximum_piece_length,
            normal_fn=normal_fn,
            segment_normals=segment_normals,
        )
    )
    visibility_origins = r_mid
    if shadow_points is not None:
        visibility_origins = np.asarray(shadow_points, dtype=float)
        if visibility_origins.shape != r_mid.shape or not np.all(
            np.isfinite(visibility_origins)
        ):
            raise ValueError(
                "shadow_points must contain one finite registered skin point "
                "for every solver line piece."
            )
    b_hat = np.cross(frame_t_hat, n_hat)

    dirs = np.atleast_2d(np.asarray(directions, dtype=float))
    if dirs.ndim != 2 or dirs.shape[1] != 3 or len(dirs) == 0:
        raise ValueError("directions must have shape (n, 3).")
    if not np.all(np.isfinite(dirs)):
        raise ValueError("directions contain NaN or infinite values.")
    direction_norm = np.linalg.norm(dirs, axis=1)
    if np.any(direction_norm <= 1.0e-12):
        raise ValueError("directions contain a zero-length vector.")
    dirs = dirs / direction_norm[:, None]
    e_vv, e_hh = _pol_unit_vectors(dirs)

    packed_shadow = None
    dense_shadow = None
    if _shadow_visibility is not None:
        if isinstance(_shadow_visibility, PackedVisibility):
            packed_shadow = _shadow_visibility
            visibility_shape = packed_shadow.shape
        else:
            dense_shadow = np.asarray(_shadow_visibility, dtype=bool)
            visibility_shape = dense_shadow.shape
        expected_shape = (len(visibility_origins), len(dirs))
        if visibility_shape != expected_shape:
            raise ValueError(
                "precomputed line visibility must have shape "
                "(n_solver_pieces, n_directions); expected "
                f"{expected_shape}, got {visibility_shape}."
            )
    visible_query = (
        visible_query_adapter(occluder)
        if occluder is not None and _shadow_visibility is None
        else None
    )

    F = {"F_vv": np.zeros(len(dirs), dtype=complex),
         "F_hh": np.zeros(len(dirs), dtype=complex),
         "F_vh": np.zeros(len(dirs), dtype=complex)}
    pref = 1.0 / (4.0 * math.pi)
    taper = float(grazing_taper_deg)
    if not math.isfinite(taper) or taper <= 0.0:
        raise ValueError("grazing_taper_deg must be positive and finite.")

    for i, d in enumerate(dirs):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Feature assembly cancelled.")
        d_n = n_hat @ d
        d_b = b_hat @ d
        d_path = path_t_hat @ d
        lit = d_n > 0.0
        if not np.any(lit):
            if progress_callback is not None:
                progress_callback(i + 1, len(dirs))
            continue
        phi = np.degrees(np.arctan2(d_n, d_b))            # 90 deg = normal
        a_tm = np.zeros(len(phi), dtype=complex)
        a_te = np.zeros(len(phi), dtype=complex)
        a_tm[lit], a_te[lit] = coefficients.sample(phi[lit])
        a_tm = a_tm * cal_tm
        a_te = a_te * cal_te
        # illumination mask with a cosine taper through grazing
        w_lit = np.clip(np.degrees(np.arcsin(np.clip(d_n, -1.0, 1.0))) / taper, 0.0, 1.0)
        w_lit = np.where(lit, 0.5 - 0.5 * np.cos(math.pi * w_lit), 0.0)
        # geometric body shadowing (STL): zero any point the body blocks
        if packed_shadow is not None:
            w_lit = w_lit * packed_shadow.column(i).astype(float)
        elif dense_shadow is not None:
            w_lit = w_lit * dense_shadow[:, i].astype(float)
        elif visible_query is not None:
            w_lit = w_lit * visible_query(
                visibility_origins, d, cancel_check=cancel_check
            ).astype(float)
        # closed-form phase integral over each piece: Int e^{j beta u} du
        beta = 2.0 * k * d_path
        x = 0.5 * beta * dL
        phase = (np.exp(2j * k * (r0 @ d)) * dL
                 * np.exp(1j * x) * np.sinc(x / math.pi))
        e_tm_local, e_te_local = _transverse_seam_basis(
            frame_t_hat, b_hat, d
        )
        for key, e_t, e_r in (("F_vv", e_vv[i], e_vv[i]),
                              ("F_hh", e_hh[i], e_hh[i]),
                              ("F_vh", e_hh[i], e_vv[i])):
            # Project transmit and receive fields onto the same signed local
            # polarization axes.  The resulting dyad is reciprocal, and when
            # a_tm == a_te it reduces to the identity on the transverse plane
            # (so a mere seam-frame rotation cannot manufacture cross-pol).
            tx_tm, rx_tm = e_tm_local @ e_t, e_tm_local @ e_r
            tx_te, rx_te = e_te_local @ e_t, e_te_local @ e_r
            coeff = tx_tm * rx_tm * a_tm + tx_te * rx_te * a_te
            F[key][i] = pref * np.sum(w_lit * coeff * phase)
        if progress_callback is not None:
            progress_callback(i + 1, len(dirs))
    return F


# -----------------------------------------------------------------------------
# 4. Vehicle signature
# -----------------------------------------------------------------------------

def combine(body_amp: 'Dict[str, np.ndarray]',
            feature_amps: 'Sequence[Dict[str, np.ndarray]]',
            mode: 'str' = "coherent") -> 'Dict[str, np.ndarray]':
    """Body + features -> sigma_3d per channel.

    mode:
      ``"coherent"`` -- everything summed in amplitude.  The truthful product
        when the relative phase is trustworthy; lobe positions carry
        millimeter sensitivity (2k*delta) and depend on the calibrated psi.
      ``"envelope"`` -- powers added.  The angular average of the coherent
        product; immune to phase error, blind to interference.
      ``"hybrid"`` -- features summed coherently WITH EACH OTHER,
        then power-added to the body.  The feature-feature phase depends only
        on their separation and is psi-independent only for components sharing
        the same calibrated convention. Common phase origin, time sign,
        polarization, normalization, and embedding assumptions are still
        required; the body-feature phase is a separate uncertainty.

    The default is ``"coherent"`` so every returned sigma has a corresponding
    physical complex field and obeys sigma = 4*pi*|F|^2.  Hybrid and envelope
    are explicitly requested statistical/engineering estimates.
    """
    m = str(mode).strip().lower()
    if m not in ("coherent", "envelope", "hybrid"):
        raise ValueError(f"unknown combine mode {mode!r}.")
    keys = ("F_vv", "F_hh", "F_vh")
    out: 'Dict[str, np.ndarray]' = {}
    for key in keys:
        body = np.asarray(body_amp.get(key, 0.0), dtype=complex)
        feats = [np.asarray(f.get(key, 0.0), dtype=complex) for f in feature_amps]
        total = body + sum(feats) if feats else body    # coherent complex field
        coherent_power = np.abs(total) ** 2
        if m == "coherent":
            power = coherent_power
        elif m == "envelope":
            power = np.abs(body) ** 2 + sum(np.abs(f) ** 2 for f in feats)
        else:
            cluster = sum(feats) if feats else np.zeros_like(body)
            power = np.abs(body) ** 2 + np.abs(cluster) ** 2
        out[key.replace("F_", "sigma_")] = 4.0 * math.pi * power
        out[key.replace("F_", "coherent_sigma_")] = (
            4.0 * math.pi * coherent_power)
        # the coherent total complex field is always returned (phase preserved),
        # regardless of which mode set the reported sigma
        out[key.replace("F_", "amp_")] = np.atleast_1d(total)
    out["mode"] = m
    return out


# -200 dBsm (1e-20 m^2) is only a DISPLAY floor: a fully shadowed or exact-zero
# return reads a finite value instead of -inf in terminal/UI summaries.  Linear
# arrays and .grim files retain the physical zero so rcs_power remains exactly
# tied to its complex amplitude.
DBSM_FLOOR = 1e-20


def dbsm(sigma: 'np.ndarray', floor: 'float' = DBSM_FLOOR) -> 'np.ndarray':
    return 10.0 * np.log10(np.maximum(np.asarray(sigma, dtype=float), floor))
