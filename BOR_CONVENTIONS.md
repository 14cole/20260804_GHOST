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
