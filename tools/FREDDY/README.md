# FREDDY

FREDDY evaluates one-dimensional, infinite-planar material stacks. It computes
TE/TM reflection, transmission, absorption, and front-face input impedance for
frequency-dependent dielectric/magnetic layers and zero-thickness resistive
sheets. Layers are ordered from the incident side to the backing.

FREDDY does **not** calculate finite-object radar cross section or dBsm. Its
metal-backed impedance export can be used as a planar equivalent boundary
condition by the companion GHOST RCS solvers when that physical approximation
is appropriate.

## Launch

FREDDY is available as the **FREDDY** tab in the main GRIM window. From the
repository root, install the integrated application and run it with:

```text
py -3 -m pip install -e .
grim
```

The embedded tab and standalone window use the same authoritative code in
this directory. FREDDY background jobs are not cancellable; GRIM prevents the
application from closing while one is running.

For a standalone window on Windows, double-click `Launch_FREDDY_GUI.bat` or
run the following from this directory:

```text
py -3 -m pip install -r requirements.txt
py -3 impedance_gui.py
```

On macOS, use `Launch_FREDDY_GUI.command`, or run the same two commands with
`python3` instead of `py -3`. `FREDDY_ROOT_PATH` is an optional GRIM
development override that can point the embedded tab at another FREDDY root.

FREDDY output is CSV, not `.grim`: save a three-column IBC impedance file or a
five-column material file beside the active GHOST `.geo`, then select it for
the applicable boundary or material in GHOST. GRIM deliberately does not load
these files into its RCS dataset table.

## Material CSV format

Material inputs and mixed-material exports use exactly five comma-separated
columns:

```text
frequency_hz,eps_real,eps_imag,mu_real,mu_imag
1000000000,3.2,-0.15,1.0,0.0
```

Frequency is in Hz. FREDDY and GHOST use the `e^(+j omega t)` convention, so a
passive lossy material has negative imaginary permittivity and permeability.
Positive imaginary values are rejected as active/gain media. Frequencies must
be positive and unique; interpolation is linear in the real and imaginary
property components and extrapolation is not performed.

## Impedance CSV format

Nominal impedance exports use the solver-compatible schema:

```text
frequency_hz,resistance_ohm,reactance_ohm
```

Uncertainty bounds are written to a separate `_uncertainty.csv` analysis file
so the nominal file remains directly readable by GHOST. Phase uncertainty
bounds are unwrapped about the nominal phase and can therefore lie outside
`[-180, 180]`; this avoids false 360-degree spans at the phase branch cut.

## Numerical scope

- Incidence angle is measured from the surface normal and must satisfy
  `0 <= angle < 90 degrees`.
- Use TE/TM labels. HH=TE and VV=TM are retained only as legacy aliases for a
  vertical plane of incidence.
- Directional materials are supported only on their measured 0- or 90-degree
  principal axes. Arbitrary tensor rotation and cross-polarization require a
  full anisotropic field solver.
- Air-backed impedance is a planar analysis result, not generally a valid
  one-sided boundary condition for a closed transmitting body.
- Effective-medium rules are morphology-dependent approximations. The GUI
  reports their assumptions and should not be treated as a substitute for
  measured mixture properties.

## Material Mix workflows

The Material Mix tab supports three related jobs:

- Predict effective frequency-dependent ε and μ from a known volume recipe.
- Search bounded volume fractions for a target ε/μ curve or constant.
- Search bounded volume fractions and a specified layer thickness for a
  reflection, absorption, or transmission requirement across a frequency and
  incidence-angle grid.

Performance targets can use PEC or air backing and TE or TM polarization. A
candidate's requirement gap is evaluated at every requested frequency/angle
point: for an upper limit the gap is `max(value) - target`, and for a lower
limit it is `target - min(value)`. A gap at or below zero passes. When material
or thickness uncertainty is enabled, the results separately report the worst
uncertainty-corner gap; this is the pass/fail value even if average-corner
scoring was selected for ranking.

The predicted or optimized effective properties still depend on the selected
mixing law and its morphology assumptions. Performance optimization does not
remove that limitation; validate a promising recipe with measured mixture
data before treating the result as a manufactured-material specification.

### Public-data validation pack

`materials/validation/nist_bam_pdms` contains unmodified, checksum-verified
NIST broadband BaM/PDMS source tables, deterministic conversion to FREDDY's Hz
and negative-loss-imaginary convention, and solver-level forward/inverse
Maxwell-Garnett regressions. Rebuild and validate it from the repository root:

```text
py -3 tools/FREDDY/tools/convert_nist_bam_pdms.py
py -3 tools/FREDDY/tools/validate_material_mix.py
```

The converted files can also be selected directly in the Material Mix tab as
normal FREDDY material inputs or targets. See
`materials/validation/nist_bam_pdms/README.md` for provenance, exclusions, and
the numerical acceptance limits.

## Tests

From the `tools/FREDDY` directory:

```text
py -3 -m unittest discover -s tests -v
```

The regression suite covers CSV sign/units, reference slab and sheet cases,
TE/TM power conservation, causal negative-index branches, layer ordering,
scalar/vector equivalence, stable thick-loss transmission, phase wrapping,
effective-medium formulas, deterministic public-data conversion, and measured
forward/inverse material-mix validation.
