# Body-of-Revolution Solver Conventions

The BoR solver computes true three-dimensional monostatic RCS for an
axisymmetric body. Its output is `sigma_3d` in square metres and is displayed
as dBsm using `10 log10(sigma_3d / 1 m^2)`.

## Geometry

A `.geo` drawing is interpreted as the `(rho, z)` half-plane:

- drawing `x` = cylindrical radius `rho`, which must be non-negative;
- drawing `y` = the rotation-axis coordinate `z`;
- a closed body is represented by an open generatrix whose two endpoints are
  on `rho = 0`;
- draw a closed generatrix from its `+z` axis endpoint to its `-z` axis
  endpoint. The solver uses the left-of-travel normal as the exterior normal;
- aspect 0 degrees is a look along `+z`, 90 degrees is broadside, and 180
  degrees is a look along `-z`.

The geometry length unit is selected by the caller and is not embedded in the
coordinate values themselves.

## Time, wave, and material signs

The code uses the `exp(+j omega t)` convention and outgoing waves proportional
to `exp(-j k r)`. Passive dielectric and magnetic loss therefore use

```text
epsilon_r = epsilon' - j epsilon''
mu_r      = mu'      - j mu''        with epsilon'', mu'' >= 0
```

In input files, lossy imaginary parts of relative permittivity and
permeability are negative. A passive Leontovich surface impedance is
`Zs = R + jX` with `R >= 0`. The boundary condition uses the outward normal
and `E_t = Zs J_s`.

## Polarization

BoR `VV` is theta/meridian-plane polarization and `HH` is phi polarization.
The compatibility aliases are `TE -> VV` and `TM -> HH`. Each modal matrix is
factored once and both channels are solved together.

At an exactly axial look the meridian basis is undefined, but rotational
symmetry requires the VV and HH complex amplitudes to agree. The radar-frame
azimuth/elevation conversion checks that condition.

## Mesh and result status

Mesh certification is enabled by default. A certified solve compares the
complex VV and HH fields on the requested mesh and an internally refined mesh,
then publishes the refined result only if both pass the fixed convergence
policy.

Certification can be disabled whenever the user chooses. Such output is marked
with `survey_mode=true`, `mesh_convergence_certified=false`, and
`published_mesh=base`. It remains valid input for GRIM and downstream tools,
but must not be described as having passed a base/fine mesh comparison.

Every solve still enforces finite fields, non-negative RCS, modal convergence,
linear residual, conditioning, and amplitude/power consistency gates.

## Validated scope and feature boundary

The production solver accepts an arbitrary non-self-intersecting
axisymmetric generatrix, including smooth, faceted, slender, and re-entrant
profiles. The release gates cover PEC CFIE, passive Leontovich IBC EFIE,
homogeneous lossy dielectric PMCHWT, dielectric-coated PEC, multilayer and
partially coated PEC, and coating-termination junctions. Analytical sphere
comparisons are evaluated over axial, oblique, and broadside looks; a slender
missile-class profile is additionally gated by complex-field scale symmetry,
fore/aft symmetry, and mesh refinement.

This remains a body-of-revolution solver, not a general 3-D solver. An
axisymmetric annular cavity, circumferential groove/panel gap, or rotationally
symmetric material treatment may be included in the generatrix when its
boundary topology is one of the supported formulations. A localized antenna,
finite cavity, rectangular panel, longitudinal seam, fin, or other
non-axisymmetric feature must not be inserted into the BoR mesh.

The base result publishes coherent complex `amp_vv` and `amp_hh` together
with `sigma = 4 pi |amp|^2`, which is the contract used by GRIM feature
placement. Non-axisymmetric features are added as calibrated complex
installed-feature-minus-clean-skin deltas. That preserves phase and removes
the replaced skin response, but it is only as accurate as the differential
feature reference; a direct full-wave featured-body comparison remains the
validation standard when body-feature coupling is important.

## Monostatic output and placed features

The local and SLURM launchers expose three sweep inputs: `FREQUENCIES_GHZ`,
`AZIMUTHS_DEG`, and `ELEVATIONS_DEG`. They derive and solve every exact BoR
aspect required by that radar grid; the complex body field is not interpolated
between coarse aspect samples. Do not include both 0 and 360 degrees because
they are the same physical azimuth.

Each completed frequency immediately publishes solver-meridian VV and HH files
under `results/by_frequency/`. These are physical body-aspect results and the
restart inputs for final assembly. When the geometry is complete it also
publishes one `results/<geometry>.grim`; its primary arrays are the requested
radar-frame monostatic VV, HH, and VH response, and it embeds the body-aspect
field and `(rho,z)` profile required for downstream placement.

Edit and run `Backend/place_features.py` to coherently add:

- a door, seam, or other perimeter from a 2-D `featured - clean` complex delta;
- a compact cavity or similar installed feature from a calibrated 3-D
  installed-feature-minus-clean-skin pattern.

Point features use one strict placement CSV for all datasets. Its fixed columns
are `placement_id,dataset_id,x,y,z,nx,ny,nz,roll_x,roll_y,roll_z`;
`dataset_id` selects a configured GRIM pattern, while the explicit normal and
roll vectors fully orient it. Point deltas use only the canonical OPN-FRD
(`featured - clean`) order. Placement does not infer CSV variants, assume a
missing cross-polarization channel is zero, or silently reverse FRD-OPN data.

Line-expanded features likewise use one strict CSV and a configured dataset
lookup. Its fixed columns are
`line_id,dataset_id,segment_index,x1,y1,z1,x2,y2,z2,n1x,n1y,n1z,n2x,n2y,n2z`.
Rows for each `line_id` are contiguous, one-based, and head-to-tail; separate
IDs are separate physical instances and may reuse the same dataset. Endpoint
outward normals are interpolated along each segment. Together with the segment
tangent they define the complete local 2-D frame, so a line needs no roll
column. Line deltas also accept only canonical OPN-FRD.

Coordinates use the CAD frame `+y = nose`, `+x = right`, `+z = up`. Every line
element and compact feature receives the monostatic two-way translation phase
`exp(+2 j k d dot r)` for this repository's `exp(+j omega t)` convention.
Compact-pattern polarization is rotated from its local aperture frame into the
body/radar frame before addition. Coordinates are checked against the body skin
using both distance and worst-case two-way phase tolerances.

A standalone cavity field is not a valid compact delta: adding it to the body
would retain the unbroken skin response and omit installation coupling. Solve
the installed feature and clean reference with the same surrounding skin, then
coherently subtract them. Mesh certification is optional for all of these
datasets. Selecting a base or feature in `place_features.py` supplies semantic
tags a GUI may have dropped; normalization, finite-field, coordinate, angular
support, and the explicit power-only role check remain enforced.

The experimental wing/fin full-object expansion code remains separate. Its
2-D-to-span expansion is usable in isolation, but a phase-unknown wing-root
corner estimate is not included in the primary coherent monostatic result.

The internal translation, segment-splitting, and accumulation regressions do
not replace a direct full-wave featured-body comparison. Follow
[FEATURE_VALIDATION_GUIDE.md](FEATURE_VALIDATION_GUIDE.md) to construct,
import, and quantitatively compare an independent 3-D reference.
