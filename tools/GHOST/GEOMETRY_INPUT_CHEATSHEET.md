# 2-D geometry input cheat sheet

This reference describes the `.geo` format consumed by
`Backend/geometry_io.py` and `Backend/rcs_solver.py`.

Updated September 5, 2026 for the thin dielectric layer, FREDDY material
handoff, coating workflows, and current material readers. Existing TYPE 1-5
codes and the five `properties:` fields are unchanged. `thin_dielectric` is
an optional new surface-material row; existing geometries need no conversion.

## Start with a complete example

The [material example folder](geometry_tests/material_examples/README.md)
contains ready-to-load `.geo` files for every boundary type and each supported
material definition form. All examples in that folder use **meters** and can
start at **1 GHz** in the **2D** solver. The tabulated examples cover 0.8-1.2 GHz.
These are illustrative inputs, not measured materials or certified results.

| Boundary or material | Complete example | Key assignment |
|---|---|---|
| TYPE 1, free impedance sheet | [Impedance card](geometry_tests/material_examples/type1_impedance_sheet.geo) | `1 0 10 0 0`; constant sheet impedance |
| TYPE 1, thin dielectric layer | [Thin strip](geometry_tests/material_examples/type1_thin_dielectric.geo) | `1 80 10 0 0`; `thin_dielectric` row |
| TYPE 2, ideal conductor | [PEC square](geometry_tests/material_examples/type2_pec.geo) | `2 0 0 0 0` |
| TYPE 2, opaque impedance boundary | [IBC square](geometry_tests/material_examples/type2_ibc.geo) | `2 0 10 0 0` |
| TYPE 3, bulk dielectric in air | [Dielectric square](geometry_tests/material_examples/type3_bulk_dielectric.geo) | `3 0 0 1 0` |
| TYPE 3, isotropic magnetic dielectric | [Epsilon and mu example](geometry_tests/material_examples/type3_magnetic_dielectric.geo) | Same TYPE 3; non-unit complex permeability |
| TYPE 4, explicit dielectric coating on PEC | [PEC-backed coating](geometry_tests/material_examples/type4_pec_backed_coating.geo) | Outer TYPE 3 plus inner `4 0 0 1 0` |
| TYPE 4, explicit coating on impedance backing | [IBC-backed coating](geometry_tests/material_examples/type4_ibc_backed_coating.geo) | Outer TYPE 3 plus inner `4 0 30 1 0` |
| TYPE 5, two bulk dielectrics | [Dielectric core and shell](geometry_tests/material_examples/type5_two_dielectrics.geo) | Outer TYPE 3 plus inner `5 0 0 1 2` |
| Spatial impedance tapers | [Linear](geometry_tests/material_examples/type1_linear_taper.geo), [cosine](geometry_tests/material_examples/type1_cosine_taper.geo), [exponential](geometry_tests/material_examples/type1_exp_taper.geo) | TYPE 1; one taper per segment |
| Frequency-dependent impedance | [CSV IBC](geometry_tests/material_examples/type2_csv_ibc.geo) | TYPE 2 and `40 surface_impedance.csv` |
| Frequency-dependent dielectric | [CSV bulk material](geometry_tests/material_examples/type3_csv_dielectric.geo) | TYPE 3 and `50 radome_material.csv` |
| Frequency-dependent thin layer | [CSV thin layer](geometry_tests/material_examples/type1_csv_thin_dielectric.geo) | `10 thin_dielectric 0.0005 50` plus dielectric CSV |
| Legacy frequency tables | [Legacy IBC](geometry_tests/material_examples/type2_legacy_ibc.geo), [legacy dielectric](geometry_tests/material_examples/type3_legacy_dielectric.geo) | One-token flags resolving to `mat.61` / `mat.62` |
| FREDDY-collapsed PEC-backed coating | [2D outer-envelope example](geometry_tests/pec_backed_ibc/example/2d_outer_envelope.geo) | Existing TYPE 2 plus nominal IBC CSV; this separate example uses **inches** |

Load a geometry through GHOST Geometry, set its coordinate units, then select
2D, 1 GHz and a few observation angles (for example 12, 48 and 86 degrees).
Use the default double precision. Select an accuracy target and request mesh
convergence for production work; a successful single-mesh solve is a setup
check, not certification. Keep all referenced CSV or `mat.*` files beside the
geometry when copying an example.

## Sign conventions first

The solver uses the `exp(+j omega t)` time convention.

| Quantity | Input convention |
|---|---|
| Passive dielectric | `eps = eps_real + j*eps_imag`, with `eps_imag <= 0` |
| Passive magnetic loss | `mu = mu_real + j*mu_imag`, with `mu_imag <= 0` |
| Surface impedance | `Zs = R + jX` ohms, entered exactly as written |
| Passive surface | `R = Re(Zs) >= 0`; either sign of `X` is allowed |
| Inductive reactance | `X > 0` under `exp(+j omega t)` |
| Capacitive reactance | `X < 0` under `exp(+j omega t)` |
| Incident plane wave | `exp(+j k dot r)` |
| Outgoing cylindrical wave | Hankel function `H0^(2)` |

For a dielectric specified by a positive loss tangent,

```text
eps_r = eps_real * (1 - j*tan_delta)
eps_imag = -eps_real * tan_delta
```

Example: `eps_real = 4.0`, `tan_delta = 0.05` becomes
`eps_r = 4.0 - j0.2`, so the input columns are `4.0 -0.2`.
Positive imaginary dielectric or permeability values represent gain under this
time convention and are rejected.

The Leontovich surface convention is

```text
E_t = Zs * (n_out cross H)
```

where `n_out` points away from the conductor. Do not negate a tabulated
reactance when transferring it into the file merely because another program
uses a different phasor convention; first convert that program's values to
`exp(+j omega t)`.

## File skeleton

```text
Title: descriptive name

Segment: segment_name TYPE
properties: TYPE N IBC_FLAG POS_MAT NEG_MAT
x1 y1 x2 y2
x2 y2 x3 y3

IBCS_Resistances:
# IBC definitions go here

Dielectrics:
# dielectric definitions go here
```

Rules:

- `Segment:` names should not contain spaces. The `TYPE` in the header and in
  `properties:` must match.
- Every coordinate row is one straight primitive: `x1 y1 x2 y2`.
- Primitives within one segment must form a continuous head-to-tail chain.
- Geometry units are not stored in `.geo`. Set the driver/API
  `geometry_units` to `"meters"` or `"inches"`.
  A comment naming units is a reminder, not an instruction to the solver.
- Blank lines and lines beginning with `#` are ignored in `.geo` files.
- Both material section headers must be present when the file is serialized;
  either section may contain no definitions.

### The five `properties:` fields

```text
properties: TYPE N IBC_FLAG POS_MAT NEG_MAT
```

| TYPE | Physical boundary | IBC_FLAG | POS_MAT | NEG_MAT |
|---:|---|---:|---:|---:|
| 1 | Free impedance sheet or thin dielectric layer in air | Required surface-material flag | 0 | 0 |
| 2 | Air-to-conductor boundary | 0 = PEC; positive = IBC flag | 0 | 0 |
| 3 | Air-to-dielectric interface | Must be 0 | Dielectric flag | 0 |
| 4 | Dielectric-to-conductor boundary | 0 = PEC; positive = IBC flag | Dielectric flag | 0 |
| 5 | Dielectric-to-dielectric interface | Must be 0 | Dielectric on normal side | Dielectric opposite normal |

`N` controls discretization of every coordinate primitive in that segment:

- `N = 0`: automatic mesh, nominally 20 panels per shortest applicable
  material wavelength.
- `N > 0`: exactly `N` panels per primitive, subject to a gross
  under-resolution safety check.
- `N < 0`: `abs(N)` panels per wavelength.

`N = 0` is the usual choice. Production results should still pass the
base/fine complex-field mesh-convergence check.
There is no longer an implicit 2,000-element-per-primitive cap; explicit
global panel limits and resource checks still apply. **Find corners/junctions**
and **Refine selected 2x** help adjust local density in Geometry. Refinement is
manual; it does not change the material syntax.

## Thin dielectric layer material (2D only)

A TYPE 1 midsurface can refer to a typed row in `IBCS_Resistances`.
Complete example: a 100 mm long, 0.5 mm thick strip, with coordinates in meters:

```text
Title: thin dielectric strip

Segment: thin_strip 1
properties: 1 80 10 0 0
-0.05 0 0.05 0

IBCS_Resistances:
10 thin_dielectric 0.0005 2

Dielectrics:
2 3.0 -0.02 1.0 0.0
```

The four surface fields are flag, `thin_dielectric`, thickness **in meters**,
and dielectric flag. Use `properties: 1 N 10 0 0` on the midsurface. This
transmitting approximation includes normal polarization; it is not an opaque
IBC. Current scope is a uniform isotropic layer in air, with only thin-layer
segments of identical material/thickness in the scene. BoR thin dielectric and
mixed thin-layer/body scenes are unsupported. Branch junctions are unsupported.
Connected midsurfaces are joined and oriented internally. With the same physical
geometry and realized mesh, changing names, grouping connected primitives into
segments, or reversing their entry direction does not change a uniform layer's
response. Changing `N` or subdividing primitives can still change mesh density.

The solver checks `k0*d*max(1,abs(sqrt(epsilon_r*mu_r))) <= 0.15` and
`d/local_radius <= 0.05`. Here `d` is physical thickness in meters and `k0` is
the free-space wavenumber. Check the full requested frequency band. Passing
these limits or converging the mesh does not certify the physical thin-layer
approximation; compare important cases against explicit thickness. See
[solver updates](../../SOLVER_UPDATES.md) for the current scope.

In the GUI, add the dielectric first, then choose **+ Thin layer**, select its
dielectric flag and enter thickness in **mm**. The saved row converts thickness
to **meters**, independently of coordinate units. Assign its new surface flag
to TYPE 1 and keep both region flags zero.

The referenced dielectric can be frequency dependent:

```text
IBCS_Resistances:
10 thin_dielectric 0.0005 50

Dielectrics:
50 radome_material.csv
```

The thickness stays fixed; epsilon and mu come from the dielectric table.

## Geometry normal and winding

For a line drawn from `(x1,y1)` to `(x2,y2)`, the user-facing normal points to
the **left** of travel:

```text
t = normalize([x2-x1, y2-y1])
n = [-t_y, t_x]
```

A horizontal line drawn left-to-right therefore has an upward normal.

| TYPE | Required direction of the user-facing normal |
|---:|---|
| 1 | Irrelevant; both sheet sides are air |
| 2 | Into air, away from the conductor |
| 3 | Into air; `POS_MAT` is on the opposite side |
| 4 | From the conductor into the `POS_MAT` dielectric |
| 5 | From `NEG_MAT` into `POS_MAT` |

Consequences for closed contours:

- A top-level TYPE 2 or TYPE 3 body is normally drawn **clockwise**, placing
  its left-hand normal into exterior air.
- A TYPE 2 or TYPE 3 boundary around an air void inside another body is drawn
  **counterclockwise**.
- A normal TYPE 4 inner boundary around a PEC core is drawn **clockwise**, so
  its normal points outward from the core into the coating.
- TYPE 5 endpoint order directly chooses which material is `POS_MAT`.

These directions matter especially for TE/VV because the boundary jump term
depends on the normal. The preflight rejects common reversed-winding cases.

## TYPE 3: hard-coded bulk dielectric example

This is a lossy dielectric square in air. The outer TYPE 3 contour is drawn
clockwise, and material flag 1 is behind the air-pointing normal.

```text
Title: inline lossy dielectric square
Segment: dielectric_square 3
properties: 3 0 0 1 0
-0.05 -0.05  -0.05  0.05
-0.05  0.05   0.05  0.05
 0.05  0.05   0.05 -0.05
 0.05 -0.05  -0.05 -0.05

IBCS_Resistances:

Dielectrics:
# flag eps_real eps_imag mu_real mu_imag
1 4.0 -0.2 1.0 0.0
```

This defines

```text
eps_r = 4.0 - j0.2
mu_r  = 1.0 + j0.0
```

All five dielectric fields are required. Near-zero epsilon or permeability is
not silently replaced by free space; unsupported ENZ/MNZ values are rejected.
The example is 100 mm square when coordinate units are meters.

An isotropic magnetic dielectric uses the same format. For epsilon
`2.5-j0.04` and mu `1.3-j0.02`, use:

```text
Dielectrics:
1 2.5 -0.04 1.3 -0.02
```

The [complete magnetic example](geometry_tests/material_examples/type3_magnetic_dielectric.geo)
uses that row. Setting `mu_real=1` and `mu_imag=0` gives the usual nonmagnetic
material. These are scalar isotropic definitions; tensor anisotropy is not
represented by extra columns.

## TYPE 2: ideal PEC example

This is a complete 100 mm square ideal conductor, coordinates in meters.
PEC uses surface flag zero and needs no material row:

```text
Title: PEC square

Segment: pec_body 2
properties: 2 0 0 0 0
-0.05 -0.05 -0.05 0.05
-0.05 0.05 0.05 0.05
0.05 0.05 0.05 -0.05
0.05 -0.05 -0.05 -0.05

IBCS_Resistances:

Dielectrics:
```

## TYPE 1 and TYPE 2: hard-coded impedance examples

An IBC definition has six fields:

```text
flag kind R_start X_start R_end X_end
```

Resistance and reactance are in ohms. For `constant`, the end fields are
required placeholders and are ignored; write them as zero.

```text
IBCS_Resistances:
# 75 - j20 ohm passive capacitive surface
10 constant 75.0 -20.0 0.0 0.0

# 25 + j12 ohm passive inductive surface
11 constant 25.0  12.0 0.0 0.0
```

For a complete opaque TYPE 2 example, change the PEC square above to
`properties: 2 0 10 0 0` and add the flag 10 row under `IBCS_Resistances:`.
The [ready-to-load IBC square](geometry_tests/material_examples/type2_ibc.geo)
already makes those assignments. Its header remains `Segment: ibc_body 2`.

For a complete transmitting TYPE 1 sheet, coordinates in meters:

```text
Title: free impedance card

Segment: impedance_card 1
properties: 1 0 10 0 0
-0.05 0 0.05 0

IBCS_Resistances:
10 constant 75.0 -20.0 0.0 0.0

Dielectrics:
```

The same numerical impedance has different boundary meaning: TYPE 1 is a
free sheet with air on both sides; TYPE 2 excludes the conductor interior.
Conventional impedance rows are supported on TYPE 1, TYPE 2, and TYPE 4 in 2D.
They are rejected on TYPE 3 and TYPE 5 transmission interfaces. The special
`thin_dielectric` row is supported **only on TYPE 1**.

## Impedance tapers

Tapers are hard-coded, spatial, and frequency independent:

```text
IBCS_Resistances:
# flag kind   R_start X_start R_end X_end
20 linear       10.0     0.0  200.0  40.0
21 cosine        0.0     0.0  376.73   0.0
22 exp           5.0     1.0  160.0  32.0
```

The taper coordinate `s` follows cumulative arc length through the complete
segment as **drawn by the user**:

```text
s = 0  at the first primitive's start point
s = 1  at the last primitive's end point
Z(s) = (1-w) Z_start + w Z_end
```

Weights are:

```text
linear:  w = s
cosine:  w = 0.5 * (1 - cos(pi*s))
exp:     Z = exp((1-s)*log(Z_start) + s*log(Z_end))
```

The cosine taper has zero slope at both ends and is generally preferable for
a smooth edge taper. Use nonzero endpoints for `exp`, and preferably keep the
complex endpoint phases on a continuous branch. A taper resets for each new
`Segment:` even when multiple segments use the same flag.

Example open card, tapered left-to-right from `10+j0` to `200+j40` ohms:

```text
Title: tapered impedance card
Segment: tapered_card 1
properties: 1 0 20 0 0
-0.05 0.0  0.0 0.0
 0.0 0.0  0.05 0.0

IBCS_Resistances:
20 linear 10.0 0.0 200.0 40.0

Dielectrics:
```

The solver samples the taper at element centers and retains the resulting
piecewise-constant coefficient inside the Galerkin weak integral.

## TYPE 4: explicit coating on PEC or impedance backing

For a stack **already collapsed by FREDDY**, use a single TYPE 2 outer-envelope
boundary and its nominal PEC-backed IBC CSV instead of the explicit layer
example below. Both 2D and BoR support that scalar-IBC route. See
[the collapsed-coating workflow](geometry_tests/pec_backed_ibc/README.md).

The outer TYPE 3 boundary points into air. The inner TYPE 4 boundary points
from the conductor into dielectric flag 1. Both square contours are clockwise.
IBC flag 30 is applied at the dielectric/conductor boundary.

```text
Title: lossy coating over an impedance conductor

Segment: coating_outer 3
properties: 3 0 0 1 0
-0.05 -0.05  -0.05  0.05
-0.05  0.05   0.05  0.05
 0.05  0.05   0.05 -0.05
 0.05 -0.05  -0.05 -0.05

Segment: coating_inner_ibc 4
properties: 4 0 30 1 0
-0.04 -0.04  -0.04  0.04
-0.04  0.04   0.04  0.04
 0.04  0.04   0.04 -0.04
 0.04 -0.04  -0.04 -0.04

IBCS_Resistances:
30 constant 35.0 8.0 0.0 0.0

Dielectrics:
1 2.8 -0.06 1.0 0.0
```

Coordinates are meters: 100 mm outer width, 80 mm core width, and 10 mm coating
thickness on each side. Set the TYPE 4 IBC flag to zero for an ideal PEC core;
then the now-unused flag 30 definition can be omitted. Both variants are
provided as complete files in the example index. A nonzero TYPE 4 backing IBC
is supported by 2D, but not currently by BoR.

## TYPE 2: a PEC-backed stack collapsed by FREDDY

Use FREDDY's **nominal PEC-backed IBC CSV** on the **outer air/coating
envelope**, assigned to TYPE 2. This route uses the existing file format in
both 2D and BoR:

```text
# Material assignment on an existing TYPE 2 segment
properties: 2 0 10 0 0

IBCS_Resistances:
10 example_coating_30mil.csv

Dielectrics:
```

The [complete 2D example and its sidecar](geometry_tests/pec_backed_ibc/README.md)
use inches: a 2-inch PEC core plus 0.03-inch coating is represented at the
2.03-inch outer envelope. Do not retain explicit coating interfaces or a
coincident PEC contour after collapsing the stack. The assignment operation
sets material flags; it does not offset the geometry.

In FREDDY, use **Check GHOST coating approximation**, then export the nominal
IBC and attach it to the current saved GHOST geometry. In Geometry, select
the TYPE 2 segments and material row, use **Apply IBC to selected TYPE 2
segments**, and save. An uncertainty or analysis export is not a nominal
material table. The scalar IBC retains the planar normal-incidence response;
the coating check assesses planar angle sensitivity and interpolation, not
finite-body RCS accuracy. It does not implement an IBC on a bulk dielectric
interface or turn the TYPE 1 thin-layer model into a metal-backed stack.

## Explicit CSV material tables

CSV sidecars are the preferred frequency-dependent format. The `.csv` file
must be in the **same directory** as the `.geo` file, and the geometry row
uses only its filename. Directory components, spaces, and other whitespace
in the filename are not supported; use names such as `radome_material.csv`.

Geometry references:

```text
IBCS_Resistances:
40 surface_impedance.csv

Dielectrics:
50 radome_material.csv
```

`surface_impedance.csv`:

```csv
frequency_hz,resistance_ohm,reactance_ohm
800000000,12.0,-4.0
1000000000,14.0,-3.0
1200000000,17.0,-1.0
```

`radome_material.csv`:

```csv
frequency_hz,eps_real,eps_imag,mu_real,mu_imag
800000000,3.20,-0.040,1.0,0.0
1000000000,3.18,-0.045,1.0,0.0
1200000000,3.15,-0.052,1.0,0.0
```

CSV rules:

- Headers are optional. If present, their names and order must match the
  examples (capitalization and surrounding whitespace are normalized).
  Headerless files must contain the same numeric columns in the same order.
  Frequencies remain **Hz** with or without a header.
- Frequencies are positive, unique, and expressed in Hz.
- Every data field must be finite and numeric; extra columns are rejected.
- Keep CSV files free of comment rows. Blank data rows are allowed.
- Real and imaginary parts are interpolated linearly with frequency.
- Extrapolation is forbidden. Every solve frequency must lie inside the
  table's characterized frequency range.
- `eps_imag` and `mu_imag` must be nonpositive; zero represents no loss in
  that property.
- IBC resistance must remain nonnegative in every row.

For example, this headerless FREDDY nominal impedance CSV is also accepted:

```csv
800000000,12.0,-4.0
1000000000,14.0,-3.0
1200000000,17.0,-1.0
```

These examples span **0.8-1.2 GHz**, so 1 GHz is a valid starting frequency.

A table flag is used in segment properties exactly like an inline flag:

```text
# TYPE 2 frequency-dependent IBC
properties: 2 0 40 0 0

# TYPE 3 frequency-dependent dielectric
properties: 3 0 0 50 0
```

One flag cannot simultaneously combine a spatial taper and a frequency table.
That would require a two-dimensional `Z(s,f)` model, which is not implemented.

## Legacy `mat.<flag>` tables

Legacy tables remain readable but new inputs should use explicit CSV names.
A one-token material definition with a flag greater than 50 resolves to a
same-directory file named `mat.<flag>`:

```text
IBCS_Resistances:
61
Dielectrics:
62
```

`mat.61` is a whitespace table in GHz with no header:

```text
# frequency_GHz  resistance_ohm  reactance_ohm
0.8 12.0 -4.0
1.0 14.0 -3.0
1.2 17.0 -1.0
```

`mat.62` is a whitespace table in GHz with no header:

```text
# frequency_GHz  eps_real  eps_imag  mu_real  mu_imag
0.8 3.20 -0.040 1.0 0.0
1.0 3.18 -0.045 1.0 0.0
1.2 3.15 -0.052 1.0 0.0
```

Unlike CSV files, legacy whitespace tables allow `#` comments. Frequencies
must still be positive and unique, and interpolation/extrapolation rules are
the same as for CSV.

## TYPE 5: complete two-dielectric example

This complete model places a dielectric core (material 2) inside a dielectric
shell (material 1), surrounded by air. Coordinates are meters. Both square
contours run clockwise; the inner TYPE 5 normal points outward from the core
into shell material 1, so `POS_MAT=1` and `NEG_MAT=2`.

```text
Title: dielectric core inside dielectric shell

Segment: outer_air_interface 3
properties: 3 0 0 1 0
-0.05 -0.05 -0.05 0.05
-0.05 0.05 0.05 0.05
0.05 0.05 0.05 -0.05
0.05 -0.05 -0.05 -0.05

Segment: core_interface 5
properties: 5 0 0 1 2
-0.025 -0.025 -0.025 0.025
-0.025 0.025 0.025 0.025
0.025 0.025 0.025 -0.025
0.025 -0.025 -0.025 -0.025

IBCS_Resistances:

Dielectrics:
1 2.2 -0.01 1.0 0.0
2 4.4 -0.10 1.0 0.0
```

The outer TYPE 3 contour provides the air interface. A standalone TYPE 5 line
is only a boundary-definition fragment, not a complete scattering geometry.
Close each intended material region or join its boundaries consistently.

## 2D versus BoR support

The files in the material example folder describe **2D cross sections**. BoR
requires an appropriate axisymmetric generating curve; switching the solver
on one of these square cross sections does not make it a valid BoR model.

- Conventional TYPE 1 electric impedance sheets now transmit in BoR as well
  as 2D. The BoR route supports a single connected meridian, including sheet
  segments joined to PEC. Disconnected sheets, sheet plus opaque IBC, and
  sheet plus bulk dielectric are outside that BoR route.
- `thin_dielectric` remains **2D only**, with the uniform all-layer limits
  above. It is a new model option within TYPE 1, not a new TYPE number.
- TYPE 2 scalar conductor IBC and FREDDY's PEC-backed nominal coating IBC
  work in both solvers. Uniform reactive IBC on a closed axis-to-axis BoR
  surface now uses CFIE; this does not extend to every reactive layout.
- A nonzero TYPE 4 backing impedance remains a **2D-only** capability.
  TYPE 3/5 bulk interfaces cannot carry an IBC flag in either solver.

See [BoR conventions](BOR_CONVENTIONS.md) and
[solver update limits](../../SOLVER_UPDATES.md) before adapting an example.

## Common mistakes

- Entering `+0.2` for lossy dielectric `eps_imag`; it must be `-0.2` here.
- Drawing a top-level TYPE 2/3 closed contour counterclockwise.
- Treating `POS_MAT` as always being on the normal side: TYPE 3 is the
  exception because its user-facing normal points into air.
- Using an IBC flag on TYPE 3 or TYPE 5.
- Assigning a `thin_dielectric` flag to TYPE 2 or TYPE 4.
- Saving thin-layer thickness in inches or millimeters; the row always stores
  meters, while the **+ Thin layer** dialog accepts millimeters.
- Mixing thin-layer and bulk/conductor segments in the same scene.
- Treating a TYPE 1 freestanding layer as a PEC-backed coating.
- Referencing a material flag without defining it in the matching section.
- Reusing a flag twice in one material section.
- Putting a CSV in another directory or writing a path instead of a basename.
- Running outside a table's characterized frequency interval.
- Entering GHz in a CSV frequency column; CSV uses Hz, legacy `mat.*` uses GHz.
- Assuming geometry coordinates are meters without matching the driver's
  `GEOMETRY_UNITS`/API `geometry_units` setting.
- Using disconnected primitives inside one `Segment:` instead of starting a
  new segment.
- Trusting an explicit coarse `N`; use automatic meshing and base/fine
  convergence for production work.

Polarization aliases used by the 2-D solver are `VV = TE` and `HH = TM`.
