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

## Monostatic output and placed features

The local and SLURM launchers expose three sweep inputs: `FREQUENCIES_GHZ`,
`AZIMUTHS_DEG`, and `ELEVATIONS_DEG`. They derive and solve every exact BoR
aspect required by that radar grid; the complex body field is not interpolated
between coarse aspect samples. Do not include both 0 and 360 degrees because
they are the same physical azimuth.

Each geometry publishes one `results/<geometry>.grim`. Its primary arrays are
the requested radar-frame monostatic VV, HH, and VH response. The same file
also embeds the body-aspect field and `(rho,z)` profile required for downstream
placement. Files in `.solver_units/` are hidden checkpoint/provenance state and
are not separate physical answers.

Edit and run `Backend/add_bor_features.py` to coherently add:

- a door, seam, or other perimeter from a 2-D `featured - clean` complex delta;
- a compact cavity or similar installed feature from a calibrated 3-D
  installed-feature-minus-clean-skin pattern.

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
datasets; format, normalization, phase-reference, and grid checks are always
enforced.

The experimental wing/fin line-expansion code remains separate for now. Its
2-D-to-span expansion is usable in isolation, but a phase-unknown wing-root
corner estimate is not included in the primary coherent monostatic result.

The internal translation, segment-splitting, and accumulation regressions do
not replace a direct full-wave featured-body comparison. Follow
[FEATURE_VALIDATION_GUIDE.md](FEATURE_VALIDATION_GUIDE.md) to construct,
import, and quantitatively compare an independent 3-D reference.
