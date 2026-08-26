# Non-BoR line-feature reconstruction

This fixture validates line placement on non-axisymmetric bodies without using
the BoR solver as either the clean body or the featured truth.  The executable
regression is:

```text
tests/test_line_feature_non_bor_physics.py
```

Run it from `tools/GHOST` with:

```text
python -m unittest -v tests.test_line_feature_non_bor_physics
```

## Why the reference is analytic

GHOST currently has a 2-D full-wave solver and an axisymmetric BoR full-wave
solver, but no general 3-D full-wave solver.  A checked-in non-BoR test cannot
honestly claim Maxwell-level clean-versus-featured truth using only GHOST.

The test therefore uses an independently implemented thin-sheet/scalar
physical-optics reference.  It integrates the two-way phase
`exp(+2 j k d.r)` over finite rectangles in closed form.  A clean body is a
finite panel or set of facets.  The explicit featured body adds finite-width
surface-contrast strips.  The production path receives only the strip
centerlines and the matching per-unit-length coefficient.  No output is
amplitude-scaled, phase-rotated, translated, or otherwise fitted.

This is a strong placement and bookkeeping test.  It is not a replacement for
an independent 3-D MoM, FEM, FDTD, or validated measurement comparison.

## Covered bodies and feature cases

### Tilted finite flat panel

A rectangular panel is arbitrarily rotated and translated.  An off-center
straight feature is evaluated in its cut plane, where the finite strip
integral separates exactly into across-gap and along-gap factors.  Both an
open gap (contrast `-1`) and a complex lossy seal are tested.

The production result and explicit finite-strip result agree to a maximum
complex-field difference of `7.97e-18` in the lossy-seal case.  This checks:

- the coefficient origin and two-way translation phase;
- line direction, outward normal, and local cut orientation;
- absolute `1/(4 pi)` line normalization;
- identical isotropic VV/HH response and zero manufactured cross-pol; and
- coherent clean-plus-feature reconstruction.

### Closed rectangular door outline

A four-segment door perimeter is placed on a tilted finite panel.  The direct
featured reference is the geometric union of four finite-width strips; the
line model is the centerline limit and has no separate corner patch.

At 2 GHz the measured unfitted errors are:

| Gap width | Isolated-delta NRMS | Complex coherence | Whole-field NRMS |
| --- | ---: | ---: | ---: |
| 1.50 mm | 0.1647% | 0.99999942 | 0.000516% |
| 0.75 mm | 0.0821% | 0.99999985 | 0.000128% |

Halving the width halves the leading isolated-delta error.  That is the
expected convergence of an `O(width^2)` corner-overlap area relative to an
`O(width)` line response.  The gate is intentionally looser than the measured
values (`0.3%` delta NRMS and `0.999995` coherence) to allow ordinary floating-
point variation while still catching a placement regression.

### Folded two-facet panel

One continuous line crosses a sharp panel transition.  Each side has its own
constant outward normal, and the two normals are deliberately discontinuous at
the shared endpoint.  The coefficient is angle-dependent, anisotropic, and
complex, so the case produces nonzero cross-polarization and is sensitive to
both endpoint normals.

The finite-strip comparison measures:

- combined VV/HH/VH NRMS: `0.00334%`;
- complex coherence: `0.99999999975`;
- clean-plus-feature whole-field NRMS: `0.0000172%`; and
- peak cross-pol amplitude: `4.19e-4 m` in the fixture normalization.

Replacing both facet normals with their average changes the placed field by
`8.92%` NRMS.  This sensitivity check ensures the regression would catch a
future implementation that silently smoothed a crease normal.

### Signed path orientation

The ordered line direction is part of placement, not just drawing order:
`b = tangent x outward_normal` defines the coupon's signed `+x` direction.
The regression reverses an asymmetric seal path and mirrors its 2-D
coefficient as `A_reversed(phi) = A(180 - phi)`; the two physical fields agree
to roundoff.  Reversing the path without mirroring the asymmetric coupon
changes the field by `12.9%` NRMS.  For a closed door outline, users should
therefore choose one consistent winding and solve/draw the coupon with the
matching inside-to-outside direction.

### External clean-body GRIM

The final test writes a clean, non-BoR monostatic GRIM, calls the public
`add_features_to_monostatic_grim` workflow, and compares the result with the
explicit featured-panel field.  It verifies that:

- an external body needs no embedded BoR model;
- the clean artifact is not modified;
- the line contribution is added coherently sample by sample;
- radar VV/HH/VH fields and `4 pi |F|^2` remain consistent; and
- feature provenance records one line placement.

The analytic coefficient is already in the reference 3-D convention, so this
case explicitly selects and verifies a zero-degree TM/TE phase mapping in the
output provenance.  The separate BoR groove regression exercises the current
legacy mapping; neither fixture independently certifies those numerical phase
constants for a new feature family.

### Combined closed box: line plus orthogonal-face fasteners

`tests/test_non_bor_combined_features.py` builds a closed rectangular box from
the analytic six-face thin-sheet oracle.  A finite-width complex door seal is
placed on the `+z` face, while round and anisotropic fastener deltas are placed
on the orthogonal `+x` face.  The fastener truth comes from the separate
Cartesian reciprocal-dyadic oracle in `test_point_scatter_physics.py`; it does
not call the production point interpolator, polarization rotation, or phase
placement code.

The public external-body `add_features_to_monostatic_grim` path receives only
the clean box GRIM, line centerline/coefficient, and two local point patterns.
It is compared with `clean + explicit finite strip + two direct dyadic point
fields`, with no fitted correction.  The measured results are:

- isolated combined-feature NRMS: `2.78e-15`;
- whole-field NRMS: `6.54e-16`;
- complex feature coherence: `1.0`;
- peak difference from an incoherent power sum: `0.0380 m^2`; and
- nonzero anisotropic cross-pol, with one line and two point placements
  recorded in provenance.

The regression gates are `2e-10` isolated NRMS, `2e-11` whole-field NRMS, and
coherence above `1 - 2e-12`.  It also requires the coherent result to differ
from an incoherent component-power sum by more than `1e-4 m^2`.  The exact
agreement is expected because the selected look plane makes the line strip
separable and every fastener query lands on a tabulated five-degree pattern
node.

This case deliberately omits body-feature mutual coupling and multiple
scattering, and the output provenance must say so.  It validates combined
assembly mechanics and interference on distinct face normals; it is not a
full-wave certification of a real closed box with installed hardware.

## What is and is not certified

These tests certify, within the stated model:

- rigid placement and exact two-way translation phase;
- local tangent/normal frames, including facet-normal discontinuities;
- TM/TE projection into the full reciprocal VV/HH/VH Jones response;
- ordered open and closed line integration;
- the narrow-feature limit for door outlines and seals; and
- coherent addition to a generic externally solved body artifact.

They do not certify:

- body-feature mutual coupling or multiple reflections;
- resonant gap, fastener, or cavity behavior absent from the 2-D coefficient;
- shadowing by a general 3-D body;
- transfer of one coupon across materially different curvature, host stack,
  or conical-incidence conditions; or
- the legacy TM/TE phase constants for a new feature family.

## Recommended independent 3-D validation ladder

The following bodies add useful physics one controlled step at a time.  Each
study should use matching clean and explicit-feature meshes everywhere outside
the changed local region, then compare both the whole field and the isolated
complex delta (`featured - clean`) against `clean + placed delta`.

1. **Finite PEC rectangular plate** with one straight slot or impedance seal.
   This is the best first full-wave comparison because the line frame is
   constant and edge clearance can be swept independently.
2. **Finite PEC box** with a rectangular door outline on one face.  It adds
   four corners, body edges, and cavity coupling while retaining simple exact
   placement coordinates.
3. **Wedge or two-panel dihedral** with a seam crossing the fold.  This directly
   checks discontinuous endpoint normals and interaction with a nearby edge.
4. **Triaxial ellipsoid or ogive-like faceted shell** with a non-circumferential
   panel outline.  This adds continuously changing normals and curvature but
   is not a BoR feature placement.
5. **Representative vehicle/fuselage subassembly** with door gap, conductive
   seal, and nearby fasteners.  Treat this as a final applicability test, not
   the first debugging fixture.

For every family, sweep gap width/depth, feature-to-edge distance, curvature,
frequency, azimuth/elevation, and polarization.  Publish unfitted complex-field
NRMS, complex coherence, active-point magnitude/phase errors, and mesh-
refinement evidence.  A small whole-body error alone is not sufficient: a
large body return can hide a completely wrong feature delta.
