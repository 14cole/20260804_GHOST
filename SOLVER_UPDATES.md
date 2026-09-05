# Using the solver and Assembly updates

## PEC-backed coating collapsed by FREDDY (2D and BoR)

For a dielectric stack on PEC, use FREDDY's **PEC-backed nominal IBC CSV** on
a GHOST **TYPE 2** boundary at the **outer coating envelope**. This existing
scalar-IBC route supports both solvers and avoids modeling the bulk layers.
It is separate from the freestanding TYPE 1 thin-sheet feature below.

FREDDY now includes **Check GHOST coating approximation**, comparing the scalar
IBC with the complete planar stack over TE/TM incidence angles and checking
CSV frequency interpolation. GHOST includes **Apply IBC to selected TYPE 2
segments**. See the [workflow and 30 mil examples](tools/GHOST/geometry_tests/pec_backed_ibc/README.md)
for the 1-18 GHz setup, reference-plane requirements and approximation limits.

## Thin dielectric layer in 2D

Load [thin_strip.geo](tools/GHOST/geometry_tests/thin_dielectric_sheet/thin_strip.geo)
in GHOST Geometry, set geometry units to **meters**, then run the 2D solver at
1 GHz. The example is a 100 mm long, 0.5 mm thick free dielectric strip with
relative epsilon 3 - j0.02 and mu 1. The drawn line represents its midsurface.

To create a layer yourself:

1. Add its relative epsilon and mu to the dielectric table. Passive loss has
   negative imaginary parts under this solver's exp(+j omega t) convention.
2. Select **+ Thin layer**, choose the dielectric and enter thickness in mm.
   The saved thickness is always in meters, independent of geometry units.
3. Use TYPE 1 on the midsurface and assign the new surface-material flag.
   Set both region-material flags to zero.
4. Start with the Standard accuracy target; inspect the accuracy/performance
   report and compare important cases with explicit finite-thickness geometry.

This is a first-order transmitting dielectric layer with normal-polarization
terms, not an opaque surface impedance. It uses one geometric surface instead
of meshing both faces. For nonmagnetic TM, it also eliminates an identically zero
field-jump density. Both co-polarized channels and mono/bistatic complex fields
are supported. An air-valued layer is transparent.

Current limits: a uniform isotropic passive layer in air; all segments must be
thin layers with the same thickness/material; no branch junctions. The solver
rejects k0*d*max(1,abs(sqrt(epsilon*mu))) > 0.15 or d/local-radius > 0.05.
These checks limit use of the approximation but do not bound its actual error.
Sharp corners and terminations need particular care. Mesh convergence measures
discretization error, not the physical error introduced by collapsing thickness.
BoR thin dielectric, mixed thin-layer/body scenes and IBC on dielectric surfaces
are not implemented in this update.

## BoR electric sheets and reactive IBC

TYPE 1 with a conventional impedance material is now a transmitting electric
sheet in BoR. The field is solved on both sides, with surface current producing
the magnetic-field jump. A single connected generating curve can join sheet
and PEC segments. Separate bodies, sheet plus opaque IBC, and sheet plus bulk
dielectric are not supported by this route.

A uniform reactive IBC on a closed axis-to-axis surface now uses CFIE to avoid
the EFIE interior-resonance problem. Lossy and spatially varying IBCs retain
their existing route; unsupported reactive layouts still fail explicitly.
No artificial resistance is inserted to suppress a resonance.

## Accuracy, refinement and runtime

The GHOST Solver options include **Accuracy target** (Standard or Tight) and
**Accuracy and performance report**. Tight requests at most 1% peak-normalized
complex change between meshes, plus the displayed RMS, phase and power limits.
Only certified solves carry mesh-convergence evidence; surveys do not become
certified by selecting a target. BoR retains modal and quadrature checks.

In Geometry, **Find corners/junctions** selects segments touching bends, material
junctions or open ends. **Refine selected 2x** increases only their density
settings. This is manual local refinement; it is not an automatic adaptive solver.

The report includes stage times, linear-solver residual/conditioning evidence,
mesh/mode evidence when available, and sampled process peak RSS. Stage timings
are inclusive and may overlap; the RSS sample includes other allocations in the
process and may miss brief peaks. The Windows native BoR build now links its
compiler runtimes statically and verifies loading in a fresh isolated process.

**2D LU precision** offers an experimental mixed mode on CPU: single-precision
factors plus double-precision residual corrections, with double-LU fallback
when refinement fails. Double precision remains the reference/default. This
reduces factor memory, not the quadratic matrix-assembly requirement, and may
not be faster for every matrix. An explicit GPU request cannot use mixed mode.

## Assembly interference inspector

Validate an Assembly, open **Interference inspector**, and press **Inspect
validated assembly**. Select an exact stored radar sample and inspect again to
change angle/frequency. VV, HH and VH complex body fields must be present.

The table reports each feature's complex amplitude, phase relative to all other
contributions, interference term, and change in total RCS if it is removed.
Positive removal effect means that feature increases the total RCS; a negative
value means its cancellation lowers the total. Feature power alone cannot show
this. The plot shows the body, summed features and total complex field.

Toggle Use or change gain/phase for an immediate cached sensitivity preview.
These edits do not alter the Assembly, re-solve illumination or add mutual
coupling. The contribution cache is capped at 16 MiB/eight samples. Each new
inspection verifies source hashes; Assembly edits invalidate displayed results.

**Check corner / termination / curvature / pair study** evaluates a study JSON.
The supplied [13-case template](tools/GHOST/geometry_tests/feature_family_studies/study.template.json)
is ready for reference datasets, but **none of these new cases is physically
validated yet**. Follow the [study instructions](tools/GHOST/geometry_tests/feature_family_studies/README.md).

See [implementation and validation results](SOLVER_IMPROVEMENT_PLAN.md) for
measured gains, tests and remaining work.
