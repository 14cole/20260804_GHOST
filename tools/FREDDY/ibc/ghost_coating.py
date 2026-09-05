"""Assess information lost by exporting a PEC-backed stack as scalar Z(f).

This compares planar reflection, not finite-body RCS or solver convergence.
The reference plane is the outer air/coating surface in every comparison.
"""
from __future__ import annotations
import cmath
import math
from .compute import (C0, compute_stack_impedance_many, compute_angle_metrics_many,
                      ambient_wave_impedance, validate_incidence_angle)


def assess_scalar_coating(frequencies, layers, angles=(0., 15., 30., 45., 60., 75., 85.)):
    frequencies = [float(f) for f in frequencies]
    angles = [validate_incidence_angle(float(a)) for a in angles]
    if not frequencies or not angles or not layers:
        raise ValueError("A coating check requires layers, frequencies and angles.")
    if any(not math.isfinite(f) or f <= 0 for f in frequencies) or any(
            b <= a for a, b in zip(frequencies, frequencies[1:])):
        raise ValueError("Frequencies must be finite, positive and strictly increasing.")
    if len(frequencies)*len(angles)*2 > 200000:
        raise ValueError("Coating check is limited to 200,000 frequency/angle/polarization samples.")
    if any(layer.anisotropic for layer in layers):
        raise ValueError("A scalar GHOST coating check requires isotropic layers; anisotropic stacks need a tensor boundary model.")
    zs = compute_stack_impedance_many(frequencies, layers, "pec")
    if not all(math.isfinite(z.real) and math.isfinite(z.imag) for z in zs):
        raise ValueError("Stack impedance contains a pole or nonfinite value; refine or change the sampled band.")
    rows = []
    for angle in angles:
        for pol in ("te", "tm"):
            full = compute_angle_metrics_many(frequencies, angle, layers, pol)
            z0 = ambient_wave_impedance(angle, pol)
            errors, phases, magnitudes = [], [], []
            for z, db, phase in zip(zs, full["metal_loss_db"], full["metal_phase_deg"]):
                reference = 10**(db/20)*cmath.exp(1j*math.radians(phase))
                scalar = (z-z0)/(z+z0)
                error = abs(scalar-reference)
                if not math.isfinite(error):
                    raise ValueError("Nonfinite reflection encountered in coating assessment.")
                errors.append(error)
                if min(abs(scalar), abs(reference)) > 1e-6:
                    phases.append(abs(math.degrees(cmath.phase(scalar*reference.conjugate()))))
                    magnitudes.append(abs(20*math.log10(abs(scalar/reference))))
            worst = max(range(len(errors)), key=errors.__getitem__)
            rows.append(dict(angle_deg=angle, polarization=pol.upper(),
                             max_absolute_complex_reflection_error=errors[worst],
                             worst_frequency_ghz=frequencies[worst],
                             max_phase_error_deg=max(phases) if phases else None,
                             max_magnitude_error_db=max(magnitudes) if magnitudes else None))
    # A midpoint check cannot exclude a narrower resonance inside an interval.
    mids = [(a+b)/2 for a,b in zip(frequencies, frequencies[1:])]
    midpoint_error = None
    if mids:
        exact = compute_stack_impedance_many(mids, layers, "pec")
        z0 = ambient_wave_impedance(0., "te")
        errors = [abs((z-z0)/(z+z0) - (zi-z0)/(zi+z0))
                  for z,zi in zip(exact, ((a+b)/2 for a,b in zip(zs, zs[1:])))]
        if not all(math.isfinite(v) for v in errors):
            raise ValueError("Nonfinite midpoint reflection; refine the frequency grid.")
        midpoint_error = max(errors)
    thickness = sum(layer.thickness_m for layer in layers if not layer.is_sheet)
    return dict(schema="freddy.ghost.scalar-coating-check.v1", backing="pec",
                frequency_range_ghz=[frequencies[0],frequencies[-1]], frequency_count=len(frequencies),
                thickness_m=thickness, reference_surface="outer air/coating interface",
                free_space_k0d_max=2*math.pi*frequencies[-1]*1e9*thickness/C0,
                angles=rows, midpoint_absolute_reflection_error_max=midpoint_error,
                finite_body_accuracy_certified=False,
                interpretation="Planar local scalar-IBC assessment only. Does not certify finite-body RCS, curvature, edges, creeping waves, or coupling. Midpoint checks do not exclude unresolved narrow resonances.")


def coating_report_text(report):
    lo, hi = report["frequency_range_ghz"]
    lines = [f"PEC-backed coating: {report['thickness_m']*1000:g} mm; {lo:g}-{hi:g} GHz.",
             "Apply Z(f) on the OUTER coating envelope as TYPE 2 in 2D or BoR.",
             "No separate bulk layer or coincident PEC boundary is needed.",
             "", "Scalar IBC versus the complete planar stack:",
             "Angle / pol: max |delta Gamma|; max phase delta; worst complex-error frequency"]
    for row in report["angles"]:
        phase = row["max_phase_error_deg"]
        phase_text = f"{phase:.3g} deg" if phase is not None else "undefined near reflection nulls"
        lines.append(f"{row['angle_deg']:g} / {row['polarization']}: "
                     f"{row['max_absolute_complex_reflection_error']:.4g}; "
                     f"{phase_text}; {row['worst_frequency_ghz']:g} GHz")
    midpoint = report["midpoint_absolute_reflection_error_max"]
    if midpoint is not None:
        lines.append(f"\nCSV midpoint interpolation check, max |delta Gamma| at normal incidence: {midpoint:.4g}.")
    lines.extend(["Phase and magnitude errors exclude samples below |Gamma|=1e-6.",
                  "|delta Gamma| is an absolute complex-field difference, not an RCS percent error.",
                  "", report["interpretation"]])
    return "\n".join(lines)
