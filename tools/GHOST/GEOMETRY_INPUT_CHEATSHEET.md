# 2-D geometry input cheat sheet

This reference describes the `.geo` format consumed by
`Backend/geometry_io.py` and `Backend/rcs_solver.py`.

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
- Blank lines and lines beginning with `#` are ignored in `.geo` files.
- Both material section headers must be present when the file is serialized;
  either section may contain no definitions.

### The five `properties:` fields

```text
properties: TYPE N IBC_FLAG POS_MAT NEG_MAT
```

| TYPE | Physical boundary | IBC_FLAG | POS_MAT | NEG_MAT |
|---:|---|---:|---:|---:|
| 1 | Free resistive/reactive sheet in air | Required material flag | 0 | 0 |
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

## Hard-coded dielectric example

This is a lossy dielectric square in air. The outer TYPE 3 contour is drawn
clockwise, and material flag 1 is behind the air-pointing normal.

```text
Title: inline lossy dielectric square
Segment: dielectric_square 3
properties: 3 0 0 1 0
-0.5 -0.5  -0.5  0.5
-0.5  0.5   0.5  0.5
 0.5  0.5   0.5 -0.5
 0.5 -0.5  -0.5 -0.5

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

## Hard-coded IBC examples

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

Reference flag 10 from a conductor boundary with:

```text
Segment: ibc_body 2
properties: 2 0 10 0 0
```

Reference it from a free sheet with:

```text
Segment: impedance_card 1
properties: 1 0 10 0 0
```

Use IBC flag zero for an ideal PEC TYPE 2 boundary:

```text
properties: 2 0 0 0 0
```

IBC flags are supported on TYPE 1, TYPE 2, and TYPE 4. They are rejected on
TYPE 3 and TYPE 5 transmission interfaces.

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
-0.5 0.0  0.0 0.0
 0.0 0.0  0.5 0.0

IBCS_Resistances:
20 linear 10.0 0.0 200.0 40.0

Dielectrics:
```

The solver samples the taper at element centers and retains the resulting
piecewise-constant coefficient inside the Galerkin weak integral.

## PEC-backed dielectric coating with an IBC

The outer TYPE 3 boundary points into air. The inner TYPE 4 boundary points
from the conductor into dielectric flag 1. Both square contours are clockwise.
IBC flag 30 is applied at the dielectric/conductor boundary.

```text
Title: lossy coating over an impedance conductor

Segment: coating_outer 3
properties: 3 0 0 1 0
-1.0 -1.0  -1.0  1.0
-1.0  1.0   1.0  1.0
 1.0  1.0   1.0 -1.0
 1.0 -1.0  -1.0 -1.0

Segment: coating_inner_ibc 4
properties: 4 0 30 1 0
-0.8 -0.8  -0.8  0.8
-0.8  0.8   0.8  0.8
 0.8  0.8   0.8 -0.8
 0.8 -0.8  -0.8 -0.8

IBCS_Resistances:
30 constant 35.0 8.0 0.0 0.0

Dielectrics:
1 2.8 -0.06 1.0 0.0
```

Set the TYPE 4 IBC flag to zero for an ideal PEC core.

## Explicit CSV material tables

CSV sidecars are the preferred frequency-dependent format. The `.csv` file
must be in the **same directory** as the `.geo` file, and the geometry row
uses only its filename—no directory components.

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
8000000000,12.0,-4.0
10000000000,14.0,-3.0
12000000000,17.0,-1.0
```

`radome_material.csv`:

```csv
frequency_hz,eps_real,eps_imag,mu_real,mu_imag
8000000000,3.20,-0.040,1.0,0.0
10000000000,3.18,-0.045,1.0,0.0
12000000000,3.15,-0.052,1.0,0.0
```

CSV rules:

- The header and column order must match exactly as shown (capitalization and
  surrounding whitespace are normalized).
- Frequencies are positive, unique, and expressed in Hz.
- Every data field must be finite and numeric; extra columns are rejected.
- Keep CSV files free of comment rows. Blank data rows are allowed.
- Real and imaginary parts are interpolated linearly with frequency.
- Extrapolation is forbidden. Every solve frequency must lie inside the
  table's characterized frequency range.
- Dielectric loss remains negative in every `eps_imag`/`mu_imag` row.
- IBC resistance must remain nonnegative in every row.

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
8.0  12.0  -4.0
10.0 14.0  -3.0
12.0 17.0  -1.0
```

`mat.62` is a whitespace table in GHz with no header:

```text
# frequency_GHz  eps_real  eps_imag  mu_real  mu_imag
8.0  3.20 -0.040  1.0 0.0
10.0 3.18 -0.045  1.0 0.0
12.0 3.15 -0.052  1.0 0.0
```

Unlike CSV files, legacy whitespace tables allow `#` comments. Frequencies
must still be positive and unique, and interpolation/extrapolation rules are
the same as for CSV.

## Two-dielectric interface example

For a TYPE 5 line drawn left-to-right, the normal points upward. This example
therefore places dielectric 2 above the line and dielectric 1 below it:

```text
Segment: dielectric_interface 5
properties: 5 0 0 2 1
-0.5 0.0 0.5 0.0

IBCS_Resistances:

Dielectrics:
1 2.2 -0.01 1.0 0.0
2 4.4 -0.10 1.0 0.0
```

In a complete multi-region model, TYPE 5 interfaces must join the surrounding
TYPE 3/4 boundaries consistently; do not leave an unintended open dielectric
region.

## Common mistakes

- Entering `+0.2` for lossy dielectric `eps_imag`; it must be `-0.2` here.
- Drawing a top-level TYPE 2/3 closed contour counterclockwise.
- Treating `POS_MAT` as always being on the normal side: TYPE 3 is the
  exception because its user-facing normal points into air.
- Using an IBC flag on TYPE 3 or TYPE 5.
- Referencing a material flag without defining it in the matching section.
- Reusing a flag twice in one material section.
- Putting a CSV in another directory or writing a path instead of a basename.
- Running outside a table's characterized frequency interval.
- Assuming geometry coordinates are meters without matching the driver's
  `GEOMETRY_UNITS`/API `geometry_units` setting.
- Using disconnected primitives inside one `Segment:` instead of starting a
  new segment.
- Trusting an explicit coarse `N`; use automatic meshing and base/fine
  convergence for production work.

Polarization aliases used by the 2-D solver are `VV = TE` and `HH = TM`.
