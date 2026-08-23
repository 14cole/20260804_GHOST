# Validating Platform Door and Cavity Reconstruction

The definitive validation compares the GHOST reduced-order reconstruction with
an independently solved, directly featured three-dimensional body. Agreement
must be checked on complex far-field amplitude, not only RCS magnitude.

## Required comparison set

Create these four whole-body datasets on exactly the same frequencies,
azimuths, elevations, and polarizations:

1. GHOST clean BoR monostatic output.
2. External 3-D clean body.
3. GHOST clean BoR output with the door or cavity coherently placed.
4. External 3-D body with that feature modeled explicitly.

First compare 1 against 2. This baseline must pass before feature validation;
otherwise the featured comparison mixes a feature-model error with different
body geometry, meshing, coordinates, polarization, or normalization. Then
compare 3 against 4.

A localized door or cavity breaks rotational symmetry, so it cannot be a
direct BoR truth model. Use a full 3-D Maxwell solver for the final reference.
For an all-GHOST line-expansion check, use a circumferential seam or groove:
that feature remains axisymmetric and can be modeled both explicitly by BoR and
as a 2-D delta expanded around the same ring.

## External 3-D solver conventions

Use the following without post-solve fitting:

- Coordinate frame: `+y = nose`, `+x = right`, `+z = up`.
- Global phase origin: the same `(0,0,0)` used by the GHOST body and placement
  coordinates.
- Angles: radar azimuth/elevation from the exact GHOST output grid. Do not
  include both 0 and 360 degrees.
- Look vector: GHOST uses a unit vector pointing from the target toward the
  monostatic radar. The incident propagation vector is its negative.
- Time convention: `exp(+j omega t)` with outgoing `exp(-jkr)` waves.
- Translation: moving a scatterer by `r` must multiply its monostatic field by
  `exp(+2 j k d dot r)`.
- Polarization: radar-frame VV and HH plus reciprocal VH. Do not equate a
  solver's spherical theta/phi labels with V/H without transforming the basis.
- Field normalization: export physical far-field amplitude `F` satisfying
  `sigma = 4 pi |F|^2` in square metres. RCS is displayed as dBsm.
- Preserve signed complex real/imaginary fields. Magnitude-only RCS cannot
  validate or support coherent placement.

If the external solver uses `exp(-j omega t)`, prefer changing its convention.
If that is impossible, determine the conversion with a known translated
scatterer and the equation above. Complex conjugation is often involved, but
do not apply it merely from a software label: far-field and angle definitions
can introduce additional signs.

## Mesh and boundary convergence

For both clean and featured whole-body solves:

- use identical geometry, material dispersion, excitation, radiation boundary
  or PML, angular grid, and far-field origin;
- converge the 3-D mesh independently until another refinement changes active
  complex samples materially less than the intended reconstruction tolerance;
- as an initial target, keep reference-solver changes below about 0.5 dB and 5
  degrees away from nulls;
- extend the radiation boundary/PML and confirm it does not control the result;
- retain the same meshing strategy between clean and featured cases wherever
  the solver permits it.

The external truth uncertainty must be smaller than the error being attributed
to the reduced-order feature model.

## Building a cavity pattern

Use a local 3-D model containing the cavity and enough surrounding body skin to
capture installation currents.

1. Put the local model origin at the cavity aperture phase centre.
2. Define local `+z` as the outward aperture normal.
3. Define local azimuth zero with the same clocking vector later supplied as
   `roll_ref`.
4. Solve the clean-skin reference and installed-cavity model with identical
   domain, mesh strategy, ports, plane waves, and far-field sampling.
5. Export the complete reciprocal Jones response VV, HH, and VH/HV over a full
   360-degree azimuth seam and every lit elevation needed by the vehicle run.
6. Form the complex difference

   `Delta F = F(installed cavity and skin) - F(clean skin)`.

Do not subtract dBsm or linear RCS. Do not use the standalone cavity field.
The latter would leave the unbroken body skin counted in the BoR solution and
would omit cavity-to-skin installation coupling.

Place this delta at the same physical aperture phase centre used as the local
3-D origin. The direct full-body featured solve then measures the approximation
left out by placement, principally long-range body/feature mutual coupling and
multiple scattering.

## Placing on an external platform result

An imported clean-platform monostatic GRIM does not need to originate in the
BoR solver. In `Backend/place_features.py`, set `BASE_MONOSTATIC_GRIM` to the
attested external GRIM and `SURFACE_MESH` to the matching indexed ASCII
`.facet` or STL surface. The surface supplies skin checks and outward normals;
the GRIM supplies the exact frequency/azimuth/elevation grid on which the
feature field is evaluated and coherently added.

Selecting `BASE_MONOSTATIC_GRIM` attests that the file is the coherent physical
platform far field at the global vehicle origin in radar-frame VV/HH/VH. This
supplies role/convention tags or raw real/imaginary redundancy stripped by a
GRIM GUI save; the stored sigma and phase still undergo full normalization and
consistency checks. A base explicitly tagged `combine_role=power` remains
invalid; other descriptive metadata is superseded by the explicit base
selection. The combined output is written back with the complete canonical
coherent schema.

The platform GRIM, surface mesh, and placement CSV must use the same physical
origin and orientation. A mismatch creates a deterministic two-way phase error
even when the feature magnitude looks plausible. Use `SHADOW=True` for binary
mesh ray blockage and `SHADOW=False` to disable that nonlocal test. Neither
choice adds body-feature mutual coupling: that limitation must still be
measured against a directly featured full-wave platform solve.

For a point feature, form the installed-minus-clean complex delta in the 3-D
solver or an independently validated lossless export process. Preserve the raw
complex Jones response; do not subtract dBsm or linear RCS. A legacy compact
result made with GRIM's **Coherent subtraction** can still be loaded when its
sigma, phase, complete VV/HH/VH matrix, azimuth seam, frequency support, and
declared conventions pass every check, but that compatibility path is not the
production delta-building recommendation.

For a 2-D line feature, use only the repository's strict CEM entry point:

```bash
python 1c_build_deltas/subtract_datasets.py OPN FRD Deltas
```

It reads the solver's preserved float64 complex fields, joins the compatible
frequency files, validates their embedded VV/HH pair, and writes canonical
OPN-FRD (`featured - clean`). Do not pre-concatenate, use GRIM overlap, or
convert the solver files before subtraction. The normal unique-look form is
accepted downstream and closed internally at a periodic interpolation seam;
incomplete or irregular partial angular coverage is not.

Every point pattern must contain VV, HH, and reciprocal VH/HV. For a locally
diagonal or axisymmetric fastener, write the physically zero VH channel into
the dataset rather than asking placement to infer it. The same declaration
model applies to a 2-D GUI subtraction listed in `LINE_FEATURE_DATASETS`:
missing semantic tags are not a blocker. Its stored linear quantity must still
be sigma_2d, elevation must be the singleton zero cut, and both TE/VV and
TM/HH must be present.

Subtraction order changes the coherent answer. The canonical delta is
`featured - clean`, which is OPN-FRD in the supplied GHOST/CEM workflow. Point
and line placement accept only that standard; neither exposes a
subtraction-order switch. Convert an FRD-OPN dataset once, before placing it,
by negating the complete complex field. Reversing order without that correction
changes the delta by 180 degrees and changes its interference with the platform.

## One point-placement CSV

All fasteners, antennas, compact cavities, and other point features share one
CSV. Configure a dataset lookup in `Backend/place_features.py`:

```python
POINT_FEATURE_LOCATIONS_CSV = "point_features.csv"
POINT_FEATURE_DATASETS = {
    "fastener": "fastener_opn_minus_frd.grim",
    "antenna": "antenna_opn_minus_frd.grim",
}
```

The CSV header and order are fixed:

```csv
placement_id,dataset_id,x,y,z,nx,ny,nz,roll_x,roll_y,roll_z
fastener_001,fastener,1.2,8.4,0.5,0,0,1,1,0,0
fastener_002,fastener,-1.2,8.4,0.5,0,0,1,1,0,0
antenna_001,antenna,0,12.7,2.1,0,0,1,1,0,0
```

Values use `COORDINATE_UNITS` and the CAD frame (`+y` nose, `+x` right,
`+z` up). `dataset_id` must exactly match the configured lookup.
`placement_id` must be unique. The normal is the local pattern `+z` direction
and is checked against the outward body normal. The roll vector defines local
pattern `+x`/azimuth zero after projection into the tangent plane; it must not
be parallel to the normal. Even an axisymmetric fastener supplies a roll vector
so there is one schema and no optional-column interpretation.

Feature size, fastener type, antenna state, material, or other response-changing
dimensions belong in `dataset_id`: provide a separately solved dataset and map
another ID. Placement does not multiply a pattern by a guessed geometric scale,
because electromagnetic scaling generally also changes its frequency response
and phase.

## One line-placement CSV

All doors, seams, panel gaps, coating edges, and other locally two-dimensional
features share one CSV. Configure the dataset lookup in
`Backend/place_features.py`:

```python
LINE_FEATURE_LOCATIONS_CSV = "line_features.csv"
LINE_FEATURE_DATASETS = {
    "door_seam": "door_seam_opn_minus_frd.grim",
    "panel_gap": "panel_gap_opn_minus_frd.grim",
}
```

The CSV header and order are fixed:

```csv
line_id,dataset_id,segment_index,x1,y1,z1,x2,y2,z2,n1x,n1y,n1z,n2x,n2y,n2z
door_001,door_seam,1,1.0,8.0,0.0,1.0,9.0,0.0,0,0,1,0,0,1
door_001,door_seam,2,1.0,9.0,0.0,0.0,9.5,0.2,0,0,1,0,0.1,0.995
gap_001,panel_gap,1,-2.0,6.0,0.0,-2.0,10.0,0.0,0,0,1,0,0,1
```

Values use `COORDINATE_UNITS` and the CAD frame (`+y` nose, `+x` right,
`+z` up). `line_id` identifies one physical chain; repeated instances use
different IDs even when they share a dataset. Rows for an ID must be contiguous,
`segment_index` must start at 1 and increase without gaps, and each endpoint
must meet the next segment head-to-tail. An open chain and a closed loop are
both valid. Branches use separate line IDs because a single chain has no
ambiguous junction ordering.

The two endpoint normals are mandatory nonzero direction vectors (they are
normalized after validation) and must point outward. They are checked against the body
skin and interpolated along the segment. No roll vector is needed: the ordered
segment tangent and skin normal define the local seam frame. Dimensions or
feature variants that alter the electromagnetic response belong in
`dataset_id`, not in an amplitude or geometric scale column. The repository
includes `line_features_template.csv` with the exact header.

## Building a door or seam delta

For the 2-D cross-section pair:

1. Use the same clean host stack and feature-centred origin.
2. Run the production 2-D sweep on the required frequency and angular grid;
   every solve automatically stores both VV/TE and HH/TM.
3. Run `python 1c_build_deltas/subtract_datasets.py OPN FRD Deltas`. This is
   the production OPN-FRD path and joins compatible solver units internally.
4. Draw the 3-D perimeter head-to-tail in the fixed line-placement CSV and
   supply the outward normal at both endpoints of every segment.
5. Ensure the perimeter lies on the body skin within the configured distance
   and worst-case two-way phase tolerances.

The direct 3-D featured body should include the exact door gap, seal, recess,
and nearby geometry represented by the coupon. Differences reveal curvature,
corners, terminations, mutual coupling, and nonlocal surface-current effects
that a locally two-dimensional expansion cannot contain.

## Importing and comparing the 3-D truth

1. Export the external complex monostatic field in a format supported by GRIM
   (`.out`, `.ss`, `.pio`, or theta/phi CSV/TXT).
2. Edit `Backend/import_3d_reference.py`, including its polarization map.
3. Complete the convention checklist above, set
   `ATTEST_GLOBAL_ORIGIN_EXP_PLUS_JWT_RADAR_VH = True`, and run it. The script
   stamps metadata but intentionally performs no fitted correction.
4. Edit `REFERENCE_PAIRS` in
   `Backend/validate_feature_reconstruction.py` with a clean baseline pair and
   a featured pair.
5. Run:

   ```bash
   python Backend/validate_feature_reconstruction.py
   ```

The report includes normalized complex RMS error, 95th-percentile magnitude
error, phase RMS, complex coherence, and per-channel diagnostics. Samples near
truth-field nulls remain in complex RMS but are excluded from pointwise phase
and dB statistics because phase at a null is undefined.

The reported best-fit global phase is diagnostic only and is never applied. A
large nearly constant phase offset usually indicates a mismatched origin, time
sign, propagation direction, or far-field definition—not a correction that
should be fitted away.

The default 3.5 dB and 25-degree gates are uncalibrated initial engineering
limits; the repository does not contain a reference fixture that establishes
them. Replace or justify them using your independent comparison, and validate
representative frequency, aspect, curvature, size, depth, material, and
placement extremes before distributing a feature library.
