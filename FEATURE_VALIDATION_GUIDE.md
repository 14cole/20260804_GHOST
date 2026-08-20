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

The platform GRIM, surface mesh, and placement CSV must use the same physical
origin and orientation. A mismatch creates a deterministic two-way phase error
even when the feature magnitude looks plausible. Use `SHADOW=True` for binary
mesh ray blockage and `SHADOW=False` to disable that nonlocal test. Neither
choice adds body-feature mutual coupling: that limitation must still be
measured against a directly featured full-wave platform solve.

## Building a door or seam delta

For the 2-D cross-section pair:

1. Use the same clean host stack and feature-centred origin.
2. Solve both TM and TE on identical frequency and angular grids.
3. Preserve complex amplitudes and form `featured - clean` with
   `feature_sum.make_delta_grim`.
4. Draw the 3-D perimeter head-to-tail in the CAD frame.
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

The default 3.5 dB and 25-degree gates are initial limits based on the existing
line-model benchmark envelope, not universal proof for every feature. Validate
representative frequency, aspect, curvature, size, depth, material, and
placement extremes before distributing a feature library.
