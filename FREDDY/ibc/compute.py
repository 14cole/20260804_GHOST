from __future__ import annotations

import bisect
import cmath
import math
from dataclasses import dataclass, field
from itertools import product

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False


# Vacuum constants
MU0 = 4.0e-7 * math.pi
EPS0 = 8.854187817e-12
ETA0 = math.sqrt(MU0 / EPS0)
C0 = 1.0 / math.sqrt(MU0 * EPS0)
INCH_TO_M = 0.0254
GHZ_TO_HZ = 1.0e9

@dataclass
class MaterialTable:
    freq_ghz: list[float]
    eps_r: list[complex]
    mu_r: list[complex]


@dataclass
class LayerConfig:
    thickness_in: float
    anisotropic: bool
    file_0deg: str
    file_90deg: str
    polarization_deg: float
    is_sheet: bool = False
    sheet_resistance: float = 0.0  # ohms/square
    # Optional inverse-design search bounds/resolution; None means "not set".
    # Bulk layers: thickness is the search variable.
    inv_t_min_in: float | None = None
    inv_t_max_in: float | None = None
    inv_t_accuracy_in: float | None = None  # snap thickness to this increment (in)
    # Resistive sheets: sheet resistance is the search variable. min/max are
    # optional; if unset the sheet stays fixed at ``sheet_resistance``.
    inv_rs_min: float | None = None  # ohms/square
    inv_rs_max: float | None = None  # ohms/square
    inv_rs_accuracy: float | None = None  # snap resistance to this increment (ohms)


@dataclass
class LoadedLayer:
    thickness_m: float
    anisotropic: bool
    polarization_deg: float
    table_0deg: MaterialTable | None
    table_90deg: MaterialTable | None
    is_sheet: bool = False
    sheet_resistance: float = 0.0


@dataclass
class UncertaintyConfig:
    enabled: bool
    thickness_pct: float
    eps_pct: float
    mu_pct: float


@dataclass
class InverseCandidate:
    score_db: float
    nominal_mean_db: float
    worst_mean_db: float
    avg_mean_db: float
    best_mean_db: float
    thickness_in: list[float]
    material_files: list[str]
    # Per-layer resistive-sheet resistance used for this candidate (0.0 for
    # non-sheet layers). Carries the searched value when sheet-R is optimized.
    sheet_resistance_ohm: list[float] = field(default_factory=list)


# --- Material mixing -------------------------------------------------------
# Effective-medium blending of two or more materials by volume "parts".
# Linear (volume average) is exact for laminae parallel to the field and is the
# natural reading of "1 part A, 2 parts B"; Lichtenecker (log) is the standard
# empirical rule for random dielectric/powder composites. Maxwell-Garnett
# treats components 2..N as spherical inclusions dilute in component 1 (the
# host/binder) and is the right model for low-to-moderate filler loading;
# Bruggeman is the symmetric self-consistent effective medium and stays
# sensible at high loading where no component is clearly "the host". All rules
# are applied to eps and mu independently.
MIX_RULE_LINEAR = "linear"
MIX_RULE_HARMONIC = "harmonic"
MIX_RULE_LOG = "log"
MIX_RULE_LOOYENGA = "looyenga"
MIX_RULE_MG = "maxwell-garnett"
MIX_RULE_BRUGGEMAN = "bruggeman"
MIX_RULES = (
    MIX_RULE_BRUGGEMAN,
    MIX_RULE_LOOYENGA,
    MIX_RULE_LOG,
    MIX_RULE_MG,
    MIX_RULE_LINEAR,
    MIX_RULE_HARMONIC,
)
MIX_RULE_LABELS = {
    MIX_RULE_BRUGGEMAN: "Bruggeman — symmetric particles",
    MIX_RULE_LOOYENGA: "Looyenga — random heterogeneous mix",
    MIX_RULE_LOG: "Lichtenecker — logarithmic empirical",
    MIX_RULE_MG: "Maxwell-Garnett — particles in material 1 host",
    MIX_RULE_LINEAR: "Wiener parallel — arithmetic laminate",
    MIX_RULE_HARMONIC: "Wiener series — harmonic laminate",
}

MIX_RULE_DESCRIPTIONS = {
    MIX_RULE_BRUGGEMAN: (
        "Symmetric self-consistent spherical-particle model. Use when no "
        "component is clearly a continuous host. Assumes electrically small, "
        "statistically isotropic constituents and does not model percolation."
    ),
    MIX_RULE_LOOYENGA: (
        "Cubic-root empirical law often used for random powders and porous or "
        "granular dielectrics. Requires ordinary passive materials with "
        "positive real permittivity/permeability."
    ),
    MIX_RULE_LOG: (
        "Logarithmic Lichtenecker empirical law. Useful for disordered mixtures "
        "when supported by measurements; it is not a morphology-independent "
        "first-principles result."
    ),
    MIX_RULE_MG: (
        "Material 1 is the continuous host; every later material is a dilute "
        "population of electrically small spherical inclusions. Reliability "
        "falls as loading, contrast, interaction, or percolation increases."
    ),
    MIX_RULE_LINEAR: (
        "Arithmetic Wiener expression; exact for ideal laminae with the applied "
        "field parallel to the layers. It is the classical upper bound for "
        "positive-real quasistatic scalar properties, but complex lossy values "
        "do not have a simple upper/lower ordering."
    ),
    MIX_RULE_HARMONIC: (
        "Harmonic Wiener expression; exact for ideal laminae with the applied "
        "field normal to the layers. It is the classical lower bound for "
        "positive-real quasistatic scalar properties, but complex lossy values "
        "do not have a simple upper/lower ordering."
    ),
}


@dataclass
class MixComponent:
    table: MaterialTable
    parts: float


@dataclass
class MixCandidate:
    score_db: float
    nominal_mean_db: float
    worst_mean_db: float
    avg_mean_db: float
    best_mean_db: float
    fractions: list[float]
    thickness_in: float
    component_files: list[str]
    rule: str
    # Weight fractions and blended density, available only when every
    # component has a known density (> 0 g/cc).
    weight_fractions: list[float] | None = None
    density_gcc: float | None = None


def _finite_complex(value: complex) -> bool:
    return math.isfinite(value.real) and math.isfinite(value.imag)


def _complex_tolerance(value: complex) -> float:
    return 64.0 * math.ulp(1.0) * max(1.0, abs(value))


def _validate_effective_value(value: complex, label: str) -> None:
    if not _finite_complex(value):
        raise ValueError(f"{label} is non-finite.")
    if abs(value) <= 1e-12:
        raise ValueError(f"{label} is singular/near-zero and is not supported.")
    if value.imag > _complex_tolerance(value):
        raise ValueError(
            f"{label} has gain-sign imaginary part {value.imag:g}; passive "
            "materials require Im <= 0 under the e^(+jωt) convention."
        )


def normalize_mix_rule(rule: str) -> str:
    r = rule.strip().lower()
    if (
        r in {MIX_RULE_LINEAR, "lin"}
        or r.startswith("linear")
        or "wiener parallel" in r
        or "arithmetic" in r
    ):
        return MIX_RULE_LINEAR
    if r in {MIX_RULE_HARMONIC, "series"} or "harmonic" in r or "wiener series" in r:
        return MIX_RULE_HARMONIC
    if r in {MIX_RULE_LOG, "lichtenecker"} or r.startswith("log") or "lichten" in r:
        return MIX_RULE_LOG
    if r in {MIX_RULE_LOOYENGA, "lll"} or "looyenga" in r or "cubic-root" in r:
        return MIX_RULE_LOOYENGA
    if r in {MIX_RULE_MG, "mg"} or "maxwell" in r or "garnett" in r:
        return MIX_RULE_MG
    if r == MIX_RULE_BRUGGEMAN or "brugge" in r or "self-consistent" in r:
        return MIX_RULE_BRUGGEMAN
    raise ValueError(f"Unsupported mixing rule: {rule}")


def parts_to_fractions(parts: list[float]) -> list[float]:
    if not parts or any(not math.isfinite(p) for p in parts):
        raise ValueError("Mix amounts must be a non-empty list of finite values.")
    total = sum(parts)
    if total <= 0:
        raise ValueError("Mix parts must sum to a positive value.")
    if any(p < 0 for p in parts):
        raise ValueError("Mix parts must be >= 0.")
    return [p / total for p in parts]


def validate_fraction_bounds(
    lower: list[float], upper: list[float], *, tol: float = 1e-12
) -> None:
    """Validate bounded volume fractions for inverse recipe design."""
    if not lower or len(lower) != len(upper):
        raise ValueError("Volume-fraction bounds must be non-empty and matching.")
    if any(not math.isfinite(v) for v in (*lower, *upper)):
        raise ValueError("Volume-fraction bounds must be finite.")
    if any(lo < 0 or hi > 1 or hi < lo for lo, hi in zip(lower, upper)):
        raise ValueError("Each volume-fraction bound must satisfy 0 <= min <= max <= 1.")
    if sum(lower) > 1.0 + tol or sum(upper) < 1.0 - tol:
        raise ValueError(
            "Volume-fraction bounds are infeasible: the minima must sum to <= "
            "100% and the maxima must sum to >= 100%."
        )


def project_bounded_fractions(
    values: list[float], lower: list[float], upper: list[float]
) -> list[float]:
    """Euclidean projection onto ``sum(f)=1`` with per-component bounds."""
    if len(values) != len(lower):
        raise ValueError("Fraction values and bounds must have matching lengths.")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Fraction proposal must contain finite values.")
    validate_fraction_bounds(lower, upper)

    lam_lo = min(v - hi for v, hi in zip(values, upper)) - 1.0
    lam_hi = max(v - lo for v, lo in zip(values, lower)) + 1.0
    for _ in range(100):
        lam = 0.5 * (lam_lo + lam_hi)
        total = sum(min(hi, max(lo, v - lam)) for v, lo, hi in zip(values, lower, upper))
        if total > 1.0:
            lam_lo = lam
        else:
            lam_hi = lam
    result = [
        min(hi, max(lo, v - 0.5 * (lam_lo + lam_hi)))
        for v, lo, hi in zip(values, lower, upper)
    ]
    residual = 1.0 - sum(result)
    if abs(residual) > 1e-12:
        candidates = [
            i
            for i, (f, lo, hi) in enumerate(zip(result, lower, upper))
            if (residual > 0 and f < hi) or (residual < 0 and f > lo)
        ]
        for i in candidates:
            room = upper[i] - result[i] if residual > 0 else result[i] - lower[i]
            delta = math.copysign(min(abs(residual), room), residual)
            result[i] += delta
            residual -= delta
            if abs(residual) <= 1e-12:
                break
    if abs(sum(result) - 1.0) > 1e-9:
        raise ValueError("Could not construct a feasible volume-fraction recipe.")
    return result


def mix_model_advisories(rule: str, fractions: list[float]) -> list[str]:
    """Return morphology/applicability cautions for a computed recipe."""
    normalized = parts_to_fractions(fractions)
    model = normalize_mix_rule(rule)
    notes: list[str] = []
    if model == MIX_RULE_MG:
        inclusion_fraction = sum(normalized[1:])
        if normalized[0] <= 0:
            notes.append("Maxwell-Garnett requires non-zero material 1 host fraction.")
        if inclusion_fraction > 0.30:
            notes.append(
                f"Total inclusion loading is {100 * inclusion_fraction:.1f}%; "
                "this is outside the usual dilute-particle regime and should "
                "be validated against measurements or a full-wave cell model."
            )
    if model in {MIX_RULE_MG, MIX_RULE_BRUGGEMAN}:
        notes.append(
            "Particle models assume inclusions are electrically small, "
            "approximately spherical, isotropically distributed, and below "
            "strong interaction/percolation regimes."
        )
    elif model in {MIX_RULE_LINEAR, MIX_RULE_HARMONIC}:
        notes.append(
            "This is an exact laminate-orientation result, not a generic random-"
            "mixture prediction. Classical Wiener upper/lower ordering applies "
            "to positive-real quasistatic scalar properties; complex lossy "
            "properties should be compared component-by-component."
        )
    else:
        notes.append(
            "This empirical law requires calibration or comparison with "
            "measurements for the actual material system and process."
        )
    return notes


def mix_overlap_grid(components: list[MixComponent]) -> list[float]:
    """Common frequency grid for a blend: the union of every component's
    breakpoints, clipped to the overlapping frequency range."""
    if not components:
        raise ValueError("At least one material is required to mix.")
    lo = max(c.table.freq_ghz[0] for c in components)
    hi = min(c.table.freq_ghz[-1] for c in components)
    if hi < lo - 1e-12:
        raise ValueError(
            "Mix components have no overlapping frequency range "
            f"(overlap [{lo:g}, {hi:g}] GHz)."
        )
    pts = {lo, hi}
    for c in components:
        for f in c.table.freq_ghz:
            if lo - 1e-12 <= f <= hi + 1e-12:
                pts.add(f)
    return sorted(pts)


def interp_components_on_grid(
    components: list[MixComponent], grid_ghz: list[float]
) -> tuple[list[list[complex]], list[list[complex]]]:
    """Resample every component's eps/mu onto a shared grid. Returns
    (eps_columns, mu_columns), one column per component. Each component must
    cover ``grid_ghz``."""
    eps_cols: list[list[complex]] = []
    mu_cols: list[list[complex]] = []
    for i, c in enumerate(components, start=1):
        if grid_ghz[0] < c.table.freq_ghz[0] or grid_ghz[-1] > c.table.freq_ghz[-1]:
            raise ValueError(
                f"Mix component {i} data range "
                f"[{c.table.freq_ghz[0]:g}, {c.table.freq_ghz[-1]:g}] GHz does not "
                f"cover the requested grid [{grid_ghz[0]:g}, {grid_ghz[-1]:g}] GHz."
            )
        eps_cols.append(interp_complex_many(grid_ghz, c.table.freq_ghz, c.table.eps_r))
        mu_cols.append(interp_complex_many(grid_ghz, c.table.freq_ghz, c.table.mu_r))
    return eps_cols, mu_cols


def _combine_columns(
    cols: list[list[complex]], fractions: list[float], rule: str
) -> list[complex]:
    n = len(cols[0])
    if rule == MIX_RULE_LINEAR:
        if NUMPY_AVAILABLE:
            acc = np.zeros(n, dtype=complex)
            for f, col in zip(fractions, cols):
                acc = acc + f * np.asarray(col, dtype=complex)
            return acc.tolist()
        return [sum(f * col[i] for f, col in zip(fractions, cols)) for i in range(n)]

    if rule == MIX_RULE_HARMONIC:
        out: list[complex] = []
        for i in range(n):
            den = sum(f / col[i] for f, col in zip(fractions, cols))
            if abs(den) <= 1e-12:
                raise ValueError(
                    "Harmonic mixing encountered a singular series denominator."
                )
            out.append(1.0 / den)
        return out

    # Lichtenecker logarithmic mix: eff = exp(sum f_i * log(z_i)). Uses the
    # principal branch of the complex log, which is valid for passive materials
    # (eps', mu' > 0) in the e^{+jwt} convention used by the loaders.
    if rule == MIX_RULE_LOG:
        out: list[complex] = []
        for i in range(n):
            if any(col[i].real <= 0 for col in cols):
                raise ValueError(
                    "Lichtenecker mixing requires positive-real constituent "
                    "properties so the complex-log branch is physically continuous."
                )
            s = 0.0 + 0.0j
            for f, col in zip(fractions, cols):
                s += f * cmath.log(col[i])
            out.append(cmath.exp(s))
        return out

    if rule == MIX_RULE_LOOYENGA:
        out = []
        for i in range(n):
            if any(col[i].real <= 0 for col in cols):
                raise ValueError(
                    "Looyenga mixing requires positive-real constituent "
                    "properties so the complex cube-root branch is continuous."
                )
            root_sum = sum(
                f * cmath.exp(cmath.log(col[i]) / 3.0)
                for f, col in zip(fractions, cols)
            )
            out.append(root_sum**3)
        return out

    # Maxwell-Garnett with spherical inclusions: component 1 is the host, the
    # rest are dilute inclusions. eff = host * (1 + 2S) / (1 - S) with
    # S = sum_i f_i (z_i - h) / (z_i + 2h) over the inclusion components.
    # This is a dilute-inclusion approximation; the caller reports a caution
    # above 30 vol% rather than silently claiming a sharp universal cutoff.
    if rule == MIX_RULE_MG:
        out = []
        for i in range(n):
            h = cols[0][i]
            s = 0.0 + 0.0j
            for f, col in zip(fractions[1:], cols[1:]):
                inclusion_den = col[i] + 2.0 * h
                if abs(inclusion_den) <= 1e-12:
                    raise ValueError(
                        "Maxwell-Garnett mixing encountered an inclusion/host "
                        "polarizability singularity."
                    )
                s += f * (col[i] - h) / inclusion_den
            den = 1.0 - s
            if abs(den) <= 1e-9:
                raise ValueError(
                    "Maxwell-Garnett mixing is singular for this loading and "
                    "contrast; no effective property was fabricated numerically."
                )
            out.append(h * (1.0 + 2.0 * s) / den)
        return out

    # Bruggeman symmetric effective medium: solve
    # sum_i f_i (z_i - eff) / (z_i + 2 eff) = 0 with damped complex Newton
    # iterations from multiple seeds. Select a passive root with a small
    # residual and prefer continuity with the previous frequency point.
    if rule == MIX_RULE_BRUGGEMAN:
        out = []
        previous: complex | None = None
        for i in range(n):
            zs = [col[i] for col in cols]
            linear_seed = sum(f * z for f, z in zip(fractions, zs))
            seeds = ([previous] if previous is not None else []) + [linear_seed] + zs
            candidates: list[tuple[float, float, complex]] = []
            for seed in seeds:
                if seed is None or not _finite_complex(seed):
                    continue
                eff = seed
                for _ in range(100):
                    dens = [z + 2.0 * eff for z in zs]
                    if min(abs(d) for d in dens) <= 1e-14:
                        break
                    residual = sum(
                        f * (z - eff) / den
                        for f, z, den in zip(fractions, zs, dens)
                    )
                    if abs(residual) <= 1e-11:
                        break
                    deriv = sum(
                        -3.0 * f * z / (den * den)
                        for f, z, den in zip(fractions, zs, dens)
                    )
                    if abs(deriv) <= 1e-14:
                        break
                    step = residual / deriv
                    base_res = abs(residual)
                    accepted = False
                    alpha = 1.0
                    for _ in range(20):
                        trial = eff - alpha * step
                        trial_dens = [z + 2.0 * trial for z in zs]
                        if min(abs(d) for d in trial_dens) > 1e-14:
                            trial_res = abs(
                                sum(
                                    f * (z - trial) / den
                                    for f, z, den in zip(fractions, zs, trial_dens)
                                )
                            )
                            if math.isfinite(trial_res) and trial_res < base_res:
                                eff = trial
                                accepted = True
                                break
                        alpha *= 0.5
                    if not accepted:
                        break
                dens = [z + 2.0 * eff for z in zs]
                if min(abs(d) for d in dens) <= 1e-14 or not _finite_complex(eff):
                    continue
                residual = abs(
                    sum(
                        f * (z - eff) / den
                        for f, z, den in zip(fractions, zs, dens)
                    )
                )
                continuity_ref = previous if previous is not None else linear_seed
                distance = abs(eff - continuity_ref) / max(abs(continuity_ref), 1e-12)
                if eff.imag <= _complex_tolerance(eff):
                    candidates.append((residual, distance, eff))
            if not candidates:
                raise ValueError(
                    "Bruggeman solver failed to find a finite passive physical root."
                )
            candidates.sort(key=lambda item: (item[0], item[1]))
            residual_score, _distance, eff_pt = candidates[0]
            if residual_score > 1e-8 or eff_pt.imag > _complex_tolerance(eff_pt):
                raise ValueError(
                    "Bruggeman solver did not converge to a passive physical root."
                )
            out.append(eff_pt)
            previous = eff_pt
        return out

    raise ValueError(f"Unsupported mixing rule: {rule}")


def combine_mix(
    grid_ghz: list[float],
    eps_cols: list[list[complex]],
    mu_cols: list[list[complex]],
    fractions: list[float],
    rule: str,
) -> MaterialTable:
    """Combine pre-interpolated component columns into one material. Splitting
    this from interpolation lets a Monte-Carlo search interpolate once and
    re-combine cheaply for every sampled ratio."""
    if not grid_ghz or not eps_cols or len(eps_cols) != len(mu_cols):
        raise ValueError("Mixing requires a non-empty grid and matching components.")
    if len(fractions) != len(eps_cols):
        raise ValueError("Mix fractions must match the number of components.")
    fractions = parts_to_fractions(fractions)
    for label, cols in (("epsilon", eps_cols), ("mu", mu_cols)):
        if any(len(col) != len(grid_ghz) for col in cols):
            raise ValueError(f"All {label} component columns must match the frequency grid.")
        for col in cols:
            for value in col:
                _validate_effective_value(value, f"Constituent {label}")
    rule_norm = normalize_mix_rule(rule)
    if rule_norm == MIX_RULE_MG and fractions[0] <= 1e-12:
        raise ValueError(
            "Maxwell-Garnett requires material 1 to have a non-zero continuous host fraction."
        )
    eps_out = _combine_columns(eps_cols, fractions, rule_norm)
    mu_out = _combine_columns(mu_cols, fractions, rule_norm)
    for value in eps_out:
        _validate_effective_value(value, "Mixed epsilon")
    for value in mu_out:
        _validate_effective_value(value, "Mixed mu")
    return MaterialTable(freq_ghz=list(grid_ghz), eps_r=eps_out, mu_r=mu_out)


def mix_material_tables(
    components: list[MixComponent],
    rule: str,
    grid_ghz: list[float] | None = None,
) -> MaterialTable:
    """Blend materials by volume parts into a single MaterialTable.

    ``parts`` are normalized to volume fractions. With ``grid_ghz=None`` the
    blend is sampled on the overlapping union grid; pass an explicit grid (e.g.
    a target sweep) to control the output sample points."""
    if not components:
        raise ValueError("At least one material is required to mix.")
    fractions = parts_to_fractions([c.parts for c in components])
    grid = list(grid_ghz) if grid_ghz is not None else mix_overlap_grid(components)
    eps_cols, mu_cols = interp_components_on_grid(components, grid)
    return combine_mix(grid, eps_cols, mu_cols, fractions, rule)


def property_match_error_curve(
    eps: list[complex],
    mu: list[complex],
    target_eps: list[complex],
    target_mu: list[complex],
    eps_weight: float = 1.0,
    mu_weight: float = 1.0,
    eps_scale: float = 1.0,
    mu_scale: float = 1.0,
) -> list[float]:
    """Per-frequency weighted relative mismatch (%) between a blend's eps/mu
    and a target. Each point combines the eps and mu relative errors (complex
    difference magnitude over target magnitude) with the given weights. The
    scale factors model a component-property tolerance corner."""
    if not (len(eps) == len(mu) == len(target_eps) == len(target_mu)):
        raise ValueError("Property-match vectors must have matching lengths.")
    w_total = eps_weight + mu_weight
    if eps_weight < 0 or mu_weight < 0 or w_total <= 0:
        raise ValueError("Property-match weights must be >= 0 and sum > 0.")
    out: list[float] = []
    for e, m, te, tm in zip(eps, mu, target_eps, target_mu):
        err_e = abs(e * eps_scale - te) / max(abs(te), 1e-6)
        err_m = abs(m * mu_scale - tm) / max(abs(tm), 1e-6)
        combined = (eps_weight * err_e**2 + mu_weight * err_m**2) / w_total
        out.append(100.0 * math.sqrt(combined))
    return out


def property_match_error(
    eps: list[complex],
    mu: list[complex],
    target_eps: list[complex],
    target_mu: list[complex],
    eps_weight: float = 1.0,
    mu_weight: float = 1.0,
    eps_scale: float = 1.0,
    mu_scale: float = 1.0,
) -> float:
    """RMS over frequency of ``property_match_error_curve`` (%). 0 = perfect
    match; 10 means the blend is off by ~10% of the target magnitude on
    average."""
    curve = property_match_error_curve(
        eps, mu, target_eps, target_mu, eps_weight, mu_weight, eps_scale, mu_scale
    )
    if not curve:
        raise ValueError("Property-match error needs at least one frequency point.")
    return math.sqrt(sum(v * v for v in curve) / len(curve))


def weight_fractions_from_volume(
    vol_fractions: list[float], densities_gcc: list[float]
) -> list[float] | None:
    """Convert volume fractions to weight fractions using per-component
    densities. Returns None when any density is unknown (<= 0), since a
    partial conversion would be misleading."""
    if len(vol_fractions) != len(densities_gcc):
        raise ValueError("Fractions and densities must have matching lengths.")
    if not densities_gcc or any(d <= 0 for d in densities_gcc):
        return None
    masses = [f * d for f, d in zip(vol_fractions, densities_gcc)]
    total = sum(masses)
    if total <= 0:
        return None
    return [m / total for m in masses]


def blend_density_gcc(
    vol_fractions: list[float], densities_gcc: list[float]
) -> float | None:
    """Volume-weighted density of the blend (g/cc), or None when any
    component density is unknown."""
    if len(vol_fractions) != len(densities_gcc):
        raise ValueError("Fractions and densities must have matching lengths.")
    if not densities_gcc or any(d <= 0 for d in densities_gcc):
        return None
    return sum(f * d for f, d in zip(vol_fractions, densities_gcc))


def snap_to_increment(
    value: float, increment: float | None, lo: float, hi: float
) -> float:
    """Clip ``value`` to ``[lo, hi]`` and, when an increment is given, round it
    to the nearest multiple of ``increment`` (grid anchored at 0) that still
    lies in range. Models a manufacturing resolution limit: an increment of
    0.001 in means a thickness is never reported finer than a thousandth of an
    inch; 1 ohm means a sheet resistance lands on whole ohms. If the band is
    narrower than one increment, the clipped value is returned unchanged."""
    v = min(max(value, lo), hi)
    if increment is None or increment <= 0:
        return v
    snapped = round(v / increment) * increment
    if snapped < lo:
        snapped += increment
    elif snapped > hi:
        snapped -= increment
    if snapped < lo or snapped > hi:
        return v
    return snapped


def make_sweep(f_start: float, f_stop: float, f_step: float) -> list[float]:
    if not all(math.isfinite(value) for value in (f_start, f_stop, f_step)):
        raise ValueError("Sweep start, stop, and step must be finite.")
    if f_step <= 0:
        raise ValueError("Sweep step must be > 0.")
    if f_stop < f_start:
        raise ValueError("Sweep stop must be >= start.")
    count = int(math.floor((f_stop - f_start) / f_step + 1e-12)) + 1
    if count <= 0:
        raise ValueError("Sweep is empty.")
    return [f_start + i * f_step for i in range(count)]


def make_frequency_sweep(f_start: float, f_stop: float, f_step: float) -> list[float]:
    """Build a positive-frequency GHz sweep for harmonic EM calculations."""
    if not math.isfinite(f_start) or f_start <= 0:
        raise ValueError("Frequency start must be finite and > 0 GHz.")
    return make_sweep(f_start, f_stop, f_step)


def validate_incidence_angle(theta_deg: float) -> float:
    """Validate an incidence angle without inventing an exact-grazing clamp."""
    theta = float(theta_deg)
    if not math.isfinite(theta) or theta < 0.0 or theta >= 90.0:
        raise ValueError(
            "Incidence angle must be finite and satisfy 0 <= angle < 90 deg. "
            "Exactly 90 deg is a singular grazing-incidence normalization."
        )
    return theta


def _validate_frequency_ghz(f_ghz: float) -> float:
    value = float(f_ghz)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Frequency must be finite and > 0 GHz.")
    return value


def _validate_frequency_vector(f_ghz: list[float]) -> None:
    if not f_ghz:
        return
    previous: float | None = None
    for value in f_ghz:
        checked = _validate_frequency_ghz(value)
        if previous is not None and checked <= previous:
            raise ValueError(
                "Frequency samples must be strictly increasing with no duplicates."
            )
        previous = checked


def normalize_backing(backing: str) -> str:
    b = backing.strip().lower().replace("_", "-")
    if b == "pec":
        return "pec"
    if b in {"air", "free-space", "freespace"}:
        return "air"
    raise ValueError(f"Unsupported backing: {backing}")


def interp_complex(x: float, xp: list[float], fp: list[complex]) -> complex:
    if x < xp[0] or x > xp[-1]:
        raise ValueError("Interpolation query is out of bounds.")
    i = bisect.bisect_left(xp, x)
    if i == 0:
        return fp[0]
    if i == len(xp):
        return fp[-1]
    if xp[i] == x:
        return fp[i]
    x0 = xp[i - 1]
    x1 = xp[i]
    t = (x - x0) / (x1 - x0)
    return fp[i - 1] + t * (fp[i] - fp[i - 1])


def interp_complex_many(x: list[float], xp: list[float], fp: list[complex]) -> list[complex]:
    if x[0] < xp[0] or x[-1] > xp[-1]:
        raise ValueError("Interpolation query is out of bounds.")
    if NUMPY_AVAILABLE:
        x_arr = np.asarray(x, dtype=float)
        xp_arr = np.asarray(xp, dtype=float)
        fp_arr = np.asarray(fp, dtype=complex)
        re = np.interp(x_arr, xp_arr, fp_arr.real)
        im = np.interp(x_arr, xp_arr, fp_arr.imag)
        return (re + 1j * im).tolist()
    return [interp_complex(v, xp, fp) for v in x]


def validate_sweep_coverage(sweep: list[float], table: MaterialTable, label: str) -> None:
    if sweep[0] < table.freq_ghz[0] or sweep[-1] > table.freq_ghz[-1]:
        raise ValueError(
            f"Sweep [{sweep[0]}, {sweep[-1]}] GHz is outside {label} data range "
            f"[{table.freq_ghz[0]}, {table.freq_ghz[-1]}] GHz."
        )


def mix_anisotropic(
    eps_0: complex,
    mu_0: complex,
    eps_90: complex,
    mu_90: complex,
    polarization_deg: float,
) -> tuple[complex, complex]:
    """Select one measured principal-axis material table.

    A scalar transmission-line model cannot represent arbitrary tensor
    rotation, birefringence, or cross-polarized fields. Only fields aligned
    with a measured 0- or 90-degree principal axis are therefore accepted.
    """
    angle = polarization_deg % 180.0
    if min(abs(angle), abs(angle - 180.0)) <= 1e-9:
        return eps_0, mu_0
    if abs(angle - 90.0) <= 1e-9:
        return eps_90, mu_90
    raise ValueError(
        "Directional/anisotropic layers are physically supported only for "
        "principal-axis polarization angles of 0 or 90 deg. Arbitrary tensor "
        "rotation requires a full anisotropic field formulation."
    )


def mix_anisotropic_many(
    eps_0: list[complex],
    mu_0: list[complex],
    eps_90: list[complex],
    mu_90: list[complex],
    polarization_deg: float,
) -> tuple[list[complex], list[complex]]:
    if not (len(eps_0) == len(mu_0) == len(eps_90) == len(mu_90)):
        raise ValueError("Anisotropic property vectors must have matching lengths.")

    angle = polarization_deg % 180.0
    if min(abs(angle), abs(angle - 180.0)) <= 1e-9:
        return list(eps_0), list(mu_0)
    if abs(angle - 90.0) <= 1e-9:
        return list(eps_90), list(mu_90)
    raise ValueError(
        "Directional/anisotropic layers are physically supported only for "
        "principal-axis polarization angles of 0 or 90 deg. Arbitrary tensor "
        "rotation requires a full anisotropic field formulation."
    )


def build_uncertainty_scales(cfg: UncertaintyConfig) -> list[tuple[float, float, float]]:
    if not cfg.enabled:
        return [(1.0, 1.0, 1.0)]

    dt = cfg.thickness_pct / 100.0
    de = cfg.eps_pct / 100.0
    dm = cfg.mu_pct / 100.0
    if not all(math.isfinite(value) for value in (dt, de, dm)):
        raise ValueError("Uncertainty percentages must be finite.")
    if dt < 0 or de < 0 or dm < 0:
        raise ValueError("Uncertainty percentages must be >= 0.")
    if dt >= 1 or de >= 1 or dm >= 1:
        raise ValueError(
            "Uncertainty percentages must be < 100 so thickness and material "
            "scales remain positive and passive."
        )

    t_vals = [1.0 - dt, 1.0 + dt] if dt > 0 else [1.0]
    e_vals = [1.0 - de, 1.0 + de] if de > 0 else [1.0]
    m_vals = [1.0 - dm, 1.0 + dm] if dm > 0 else [1.0]

    scales = sorted(set(product(t_vals, e_vals, m_vals)))
    if (1.0, 1.0, 1.0) not in scales:
        scales.append((1.0, 1.0, 1.0))
    return scales


def is_nominal_scale(t_scale: float, e_scale: float, m_scale: float, tol: float = 1e-12) -> bool:
    return (
        abs(t_scale - 1.0) <= tol
        and abs(e_scale - 1.0) <= tol
        and abs(m_scale - 1.0) <= tol
    )


def layer_properties(layer: LoadedLayer, f_ghz: float) -> tuple[complex, complex]:
    eps_0 = interp_complex(f_ghz, layer.table_0deg.freq_ghz, layer.table_0deg.eps_r)
    mu_0 = interp_complex(f_ghz, layer.table_0deg.freq_ghz, layer.table_0deg.mu_r)
    if not layer.anisotropic:
        return eps_0, mu_0

    assert layer.table_90deg is not None
    eps_90 = interp_complex(f_ghz, layer.table_90deg.freq_ghz, layer.table_90deg.eps_r)
    mu_90 = interp_complex(f_ghz, layer.table_90deg.freq_ghz, layer.table_90deg.mu_r)
    return mix_anisotropic(eps_0, mu_0, eps_90, mu_90, layer.polarization_deg)


def layer_properties_many(layer: LoadedLayer, f_ghz: list[float]) -> tuple[list[complex], list[complex]]:
    eps_0 = interp_complex_many(f_ghz, layer.table_0deg.freq_ghz, layer.table_0deg.eps_r)
    mu_0 = interp_complex_many(f_ghz, layer.table_0deg.freq_ghz, layer.table_0deg.mu_r)
    if not layer.anisotropic:
        return eps_0, mu_0

    assert layer.table_90deg is not None
    eps_90 = interp_complex_many(f_ghz, layer.table_90deg.freq_ghz, layer.table_90deg.eps_r)
    mu_90 = interp_complex_many(f_ghz, layer.table_90deg.freq_ghz, layer.table_90deg.mu_r)
    return mix_anisotropic_many(eps_0, mu_0, eps_90, mu_90, layer.polarization_deg)


def compute_stack_impedance(
    f_ghz: float,
    layers: list[LoadedLayer],
    backing: str,
    thickness_scale: float = 1.0,
    eps_scale: float = 1.0,
    mu_scale: float = 1.0,
) -> complex:
    f_ghz = _validate_frequency_ghz(f_ghz)
    if backing == "pec":
        z_load = 0.0 + 0.0j
    elif backing == "air":
        z_load = ETA0 + 0.0j
    else:
        raise ValueError(f"Unsupported backing: {backing}")

    f_hz = f_ghz * GHZ_TO_HZ
    omega = 2.0 * math.pi * f_hz
    k0 = omega / C0

    # Cascade from bottom layer to top layer.
    z_next = z_load
    for layer in reversed(layers):
        if layer.is_sheet:
            rs = complex(layer.sheet_resistance, 0.0)
            z_next = (z_next * rs) / (z_next + rs)
            continue
        eps_r, mu_r = layer_properties(layer, f_ghz)
        eps_r *= eps_scale
        mu_r *= mu_scale
        zc = ETA0 * cmath.sqrt(mu_r / eps_r)
        gamma = 1j * k0 * cmath.sqrt(mu_r * eps_r)
        t = cmath.tanh(gamma * layer.thickness_m * thickness_scale)
        z_next = zc * (z_next + zc * t) / (zc + z_next * t)

    return z_next


def compute_stack_impedance_many(
    f_ghz: list[float],
    layers: list[LoadedLayer],
    backing: str,
    thickness_scale: float = 1.0,
    eps_scale: float = 1.0,
    mu_scale: float = 1.0,
) -> list[complex]:
    if backing == "pec":
        z_load = 0.0 + 0.0j
    elif backing == "air":
        z_load = ETA0 + 0.0j
    else:
        raise ValueError(f"Unsupported backing: {backing}")

    if not f_ghz:
        return []
    _validate_frequency_vector(f_ghz)

    if NUMPY_AVAILABLE:
        f_arr_ghz = np.asarray(f_ghz, dtype=float)
        f_hz = f_arr_ghz * GHZ_TO_HZ
        k0 = (2.0 * math.pi * f_hz) / C0
        z_next = np.full_like(f_hz, z_load, dtype=complex)

        for layer in reversed(layers):
            if layer.is_sheet:
                rs = complex(layer.sheet_resistance, 0.0)
                z_next = (z_next * rs) / (z_next + rs)
                continue
            eps_r, mu_r = layer_properties_many(layer, f_ghz)
            eps_arr = np.asarray(eps_r, dtype=complex) * eps_scale
            mu_arr = np.asarray(mu_r, dtype=complex) * mu_scale
            zc = ETA0 * np.sqrt(mu_arr / eps_arr)
            gamma = 1j * k0 * np.sqrt(mu_arr * eps_arr)
            t = np.tanh(gamma * layer.thickness_m * thickness_scale)
            z_next = zc * (z_next + zc * t) / (zc + z_next * t)
        return z_next.tolist()

    return [
        compute_stack_impedance(
            f,
            layers,
            backing,
            thickness_scale=thickness_scale,
            eps_scale=eps_scale,
            mu_scale=mu_scale,
        )
        for f in f_ghz
    ]


def normalize_wave_polarization(pol: str) -> str:
    """Return plane-wave TE/TM polarization.

    HH/VV are retained as legacy aliases using the conventional vertical
    plane-of-incidence mapping HH=TE and VV=TM. The 2D RCS elevation-cut GUI
    uses different HH/VV aliases, so exported workflows should use TE/TM.
    """
    p = pol.strip().lower()
    if p in {"hh", "te", "hh (te)"}:
        return "te"
    if p in {"vv", "tm", "vv (tm)"}:
        return "tm"
    raise ValueError(f"Unsupported wave polarization: {pol}")


def _stable_kz(z: complex) -> complex:
    # Choose branch with non-negative attenuation.
    if z.imag < 0:
        return -z
    if abs(z.imag) < 1e-12 and z.real < 0:
        return -z
    return z


def layer_wave_params(
    f_hz: float,
    theta_deg: float,
    eps_r: complex,
    mu_r: complex,
    wave_pol: str,
) -> tuple[complex, complex]:
    f_hz = float(f_hz)
    if not math.isfinite(f_hz) or f_hz <= 0:
        raise ValueError("Frequency must be finite and > 0 Hz.")
    theta_deg = validate_incidence_angle(theta_deg)
    theta = math.radians(theta_deg)
    k0 = 2.0 * math.pi * f_hz / C0
    kx = k0 * math.sin(theta)
    k_layer = k0 * cmath.sqrt(eps_r * mu_r)
    kz = _stable_kz(cmath.sqrt(k_layer * k_layer - kx * kx))
    omega = 2.0 * math.pi * f_hz
    if wave_pol == "te":
        zc = omega * MU0 * mu_r / kz
    elif wave_pol == "tm":
        zc = kz / (omega * EPS0 * eps_r)
    else:
        raise ValueError(f"Unsupported wave polarization: {wave_pol}")
    return zc, kz


def ambient_wave_impedance(theta_deg: float, wave_pol: str) -> complex:
    theta_deg = validate_incidence_angle(theta_deg)
    theta = math.radians(theta_deg)
    c = math.cos(theta)
    if wave_pol == "te":
        return ETA0 / c
    if wave_pol == "tm":
        return ETA0 * c
    raise ValueError(f"Unsupported wave polarization: {wave_pol}")


def cascade_input_impedance(
    f_ghz: float,
    theta_deg: float,
    layers: list[LoadedLayer],
    wave_pol: str,
    z_load: complex,
    thickness_scale: float = 1.0,
    eps_scale: float = 1.0,
    mu_scale: float = 1.0,
) -> complex:
    f_hz = f_ghz * GHZ_TO_HZ
    z_next = z_load
    for layer in reversed(layers):
        if layer.is_sheet:
            rs = complex(layer.sheet_resistance, 0.0)
            z_next = (z_next * rs) / (z_next + rs)
            continue
        eps_r, mu_r = layer_properties(layer, f_ghz)
        eps_r *= eps_scale
        mu_r *= mu_scale
        zc, kz = layer_wave_params(f_hz, theta_deg, eps_r, mu_r, wave_pol)
        t = cmath.tan(kz * layer.thickness_m * thickness_scale)
        z_next = zc * (z_next + 1j * zc * t) / (zc + 1j * z_next * t)
    return z_next


def cascade_abcd(
    f_ghz: float,
    theta_deg: float,
    layers: list[LoadedLayer],
    wave_pol: str,
    thickness_scale: float = 1.0,
    eps_scale: float = 1.0,
    mu_scale: float = 1.0,
) -> tuple[complex, complex, complex, complex]:
    f_hz = f_ghz * GHZ_TO_HZ
    a = 1.0 + 0.0j
    b = 0.0 + 0.0j
    c = 0.0 + 0.0j
    d = 1.0 + 0.0j

    for layer in layers:
        if layer.is_sheet:
            y_s = 1.0 / layer.sheet_resistance
            a, b, c, d = (a + b * y_s, b, c + d * y_s, d)
            continue
        eps_r, mu_r = layer_properties(layer, f_ghz)
        eps_r *= eps_scale
        mu_r *= mu_scale
        zc, kz = layer_wave_params(f_hz, theta_deg, eps_r, mu_r, wave_pol)
        p = kz * layer.thickness_m * thickness_scale
        ai = cmath.cos(p)
        bi = 1j * zc * cmath.sin(p)
        ci = 1j * cmath.sin(p) / zc
        di = ai
        a, b, c, d = (
            a * ai + b * ci,
            a * bi + b * di,
            c * ai + d * ci,
            c * bi + d * di,
        )

    return a, b, c, d


def _db_from_mag(x: complex) -> float:
    if not (math.isfinite(x.real) and math.isfinite(x.imag)):
        raise ValueError("Scattering coefficient is non-finite.")
    mag = max(abs(x), 1e-15)
    return 20.0 * math.log10(mag)


def _db_from_power(x: float) -> float:
    if not math.isfinite(x):
        raise ValueError("Power fraction is non-finite.")
    tolerance = 256.0 * math.ulp(1.0)
    if x < -tolerance or x > 1.0 + tolerance:
        raise ValueError(
            f"Computed absorption fraction {x:g} is outside [0, 1]. "
            "Check material passivity and numerical conditioning."
        )
    x = min(1.0, max(0.0, x))
    return 10.0 * math.log10(max(x, 1e-15))


def _db_from_power_many(x: "np.ndarray") -> "np.ndarray":
    values = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Power fraction contains non-finite values.")
    tolerance = 256.0 * np.finfo(float).eps
    if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
        bad = values[(values < -tolerance) | (values > 1.0 + tolerance)][0]
        raise ValueError(
            f"Computed absorption fraction {float(bad):g} is outside [0, 1]. "
            "Check material passivity and numerical conditioning."
        )
    values = np.clip(values, 0.0, 1.0)
    return 10.0 * np.log10(np.maximum(values, 1e-15))


def _stable_kz_many(z: "np.ndarray") -> "np.ndarray":
    out = np.asarray(z, dtype=complex)
    mask = (out.imag < 0.0) | ((np.abs(out.imag) < 1e-12) & (out.real < 0.0))
    return np.where(mask, -out, out)


def _db_from_mag_many(x: "np.ndarray") -> "np.ndarray":
    values = np.asarray(x, dtype=complex)
    if not np.all(np.isfinite(values)):
        raise ValueError("Scattering coefficient contains non-finite values.")
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-15))


def compute_angle_metrics_many(
    f_ghz: list[float],
    theta_deg: float,
    layers: list[LoadedLayer],
    wave_pol: str,
    thickness_scale: float = 1.0,
    eps_scale: float = 1.0,
    mu_scale: float = 1.0,
) -> dict[str, list[float]]:
    if not f_ghz:
        return {
            "metal_loss_db": [],
            "metal_phase_deg": [],
            "air_loss_db": [],
            "air_phase_deg": [],
            "insertion_loss_db": [],
            "insertion_phase_deg": [],
            "metal_absorption_db": [],
            "air_absorption_db": [],
        }
    _validate_frequency_vector(f_ghz)
    theta_deg = validate_incidence_angle(theta_deg)

    if not NUMPY_AVAILABLE:
        rows = [
            compute_angle_metrics(
                f,
                theta_deg,
                layers,
                wave_pol,
                thickness_scale=thickness_scale,
                eps_scale=eps_scale,
                mu_scale=mu_scale,
            )
            for f in f_ghz
        ]
        return {
            "metal_loss_db": [r["metal_loss_db"] for r in rows],
            "metal_phase_deg": [r["metal_phase_deg"] for r in rows],
            "air_loss_db": [r["air_loss_db"] for r in rows],
            "air_phase_deg": [r["air_phase_deg"] for r in rows],
            "insertion_loss_db": [r["insertion_loss_db"] for r in rows],
            "insertion_phase_deg": [r["insertion_phase_deg"] for r in rows],
            "metal_absorption_db": [r["metal_absorption_db"] for r in rows],
            "air_absorption_db": [r["air_absorption_db"] for r in rows],
        }

    theta = math.radians(theta_deg)
    sin_t = math.sin(theta)
    z0 = ambient_wave_impedance(theta_deg, wave_pol)
    f_arr_ghz = np.asarray(f_ghz, dtype=float)
    f_hz = f_arr_ghz * GHZ_TO_HZ
    omega = 2.0 * math.pi * f_hz
    k0 = omega / C0
    kx = k0 * sin_t

    if wave_pol not in {"te", "tm"}:
        raise ValueError(f"Unsupported wave polarization: {wave_pol}")

    # Cache layer wave terms once; they are reused by both reflection cascades and ABCD propagation.
    # For sheet layers, terms entry is None and sheet_rs stores the resistance.
    layer_terms: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None] = []
    layer_sheet_rs: list[float] = []
    for layer in layers:
        if layer.is_sheet:
            layer_terms.append(None)
            layer_sheet_rs.append(layer.sheet_resistance)
            continue
        layer_sheet_rs.append(0.0)
        eps_r, mu_r = layer_properties_many(layer, f_ghz)
        eps_arr = np.asarray(eps_r, dtype=complex) * eps_scale
        mu_arr = np.asarray(mu_r, dtype=complex) * mu_scale
        k_layer = k0 * np.sqrt(eps_arr * mu_arr)
        kz = _stable_kz_many(np.sqrt(k_layer * k_layer - kx * kx))
        if wave_pol == "te":
            zc = omega * MU0 * mu_arr / kz
        else:
            zc = kz / (omega * EPS0 * eps_arr)
        p = kz * layer.thickness_m * thickness_scale
        layer_terms.append((zc, np.tan(p), np.sin(p), np.cos(p)))

    def _cascade(z_load: complex) -> np.ndarray:
        z_next = np.full_like(f_hz, z_load, dtype=complex)
        for idx in reversed(range(len(layer_terms))):
            if layer_terms[idx] is None:
                rs = complex(layer_sheet_rs[idx], 0.0)
                z_next = (z_next * rs) / (z_next + rs)
            else:
                zc, t, _si, _co = layer_terms[idx]
                z_next = zc * (z_next + 1j * zc * t) / (zc + 1j * z_next * t)
        return z_next

    zin_metal = _cascade(0.0 + 0.0j)
    gamma_metal = (zin_metal - z0) / (zin_metal + z0)

    zin_air = _cascade(z0)
    gamma_air = (zin_air - z0) / (zin_air + z0)

    a = np.ones_like(f_hz, dtype=complex)
    b = np.zeros_like(f_hz, dtype=complex)
    c = np.zeros_like(f_hz, dtype=complex)
    d = np.ones_like(f_hz, dtype=complex)

    for idx in range(len(layer_terms)):
        if layer_terms[idx] is None:
            y_s = 1.0 / layer_sheet_rs[idx]
            a, b, c, d = (a + b * y_s, b, c + d * y_s, d)
        else:
            zc, _t, si, ai = layer_terms[idx]
            bi = 1j * zc * si
            ci = 1j * si / zc
            di = ai
            a, b, c, d = (
                a * ai + b * ci,
                a * bi + b * di,
                c * ai + d * ci,
                c * bi + d * di,
            )

    den = a + b / z0 + c * z0 + d
    s21 = 2.0 / den

    metal_abs_frac = 1.0 - np.abs(gamma_metal) ** 2
    air_abs_frac = 1.0 - np.abs(gamma_air) ** 2 - np.abs(s21) ** 2

    return {
        "metal_loss_db": _db_from_mag_many(gamma_metal).tolist(),
        "metal_phase_deg": np.degrees(np.angle(gamma_metal)).tolist(),
        "air_loss_db": _db_from_mag_many(gamma_air).tolist(),
        "air_phase_deg": np.degrees(np.angle(gamma_air)).tolist(),
        "insertion_loss_db": _db_from_mag_many(s21).tolist(),
        "insertion_phase_deg": np.degrees(np.angle(s21)).tolist(),
        "metal_absorption_db": _db_from_power_many(metal_abs_frac).tolist(),
        "air_absorption_db": _db_from_power_many(air_abs_frac).tolist(),
    }


def compute_angle_metrics(
    f_ghz: float,
    theta_deg: float,
    layers: list[LoadedLayer],
    wave_pol: str,
    thickness_scale: float = 1.0,
    eps_scale: float = 1.0,
    mu_scale: float = 1.0,
) -> dict[str, float]:
    f_ghz = _validate_frequency_ghz(f_ghz)
    theta_deg = validate_incidence_angle(theta_deg)
    z0 = ambient_wave_impedance(theta_deg, wave_pol)

    # Metal-backed reflection.
    zin_metal = cascade_input_impedance(
        f_ghz,
        theta_deg,
        layers,
        wave_pol,
        0.0 + 0.0j,
        thickness_scale=thickness_scale,
        eps_scale=eps_scale,
        mu_scale=mu_scale,
    )
    gamma_metal = (zin_metal - z0) / (zin_metal + z0)

    # Air-backed reflection.
    zin_air = cascade_input_impedance(
        f_ghz,
        theta_deg,
        layers,
        wave_pol,
        z0,
        thickness_scale=thickness_scale,
        eps_scale=eps_scale,
        mu_scale=mu_scale,
    )
    gamma_air = (zin_air - z0) / (zin_air + z0)

    # Through transmission in air (insertion).
    a, b, c, d = cascade_abcd(
        f_ghz,
        theta_deg,
        layers,
        wave_pol,
        thickness_scale=thickness_scale,
        eps_scale=eps_scale,
        mu_scale=mu_scale,
    )
    den = a + b / z0 + c * z0 + d
    s21 = 2.0 / den

    return {
        "metal_loss_db": _db_from_mag(gamma_metal),
        "metal_phase_deg": math.degrees(cmath.phase(gamma_metal)),
        "air_loss_db": _db_from_mag(gamma_air),
        "air_phase_deg": math.degrees(cmath.phase(gamma_air)),
        "insertion_loss_db": _db_from_mag(s21),
        "insertion_phase_deg": math.degrees(cmath.phase(s21)),
        "metal_absorption_db": _db_from_power(1.0 - abs(gamma_metal) ** 2),
        "air_absorption_db": _db_from_power(
            1.0 - abs(gamma_air) ** 2 - abs(s21) ** 2
        ),
    }
