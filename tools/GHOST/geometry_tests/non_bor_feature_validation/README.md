# Non-BoR line and point feature validation

This study family validates placement on finite, non-axisymmetric bodies. GHOST
does not contain a full three-dimensional surface-current solver, so the
explicit clean and featured bodies must be solved by an independent 3-D
Maxwell solver and imported as coherent GRIM fields. Internal manufactured
tests can prove coordinate, phase, polarization, interpolation, and summation
math; only these external pairs can measure omitted body-feature mutual
coupling.

## Recommended body ladder

Run the cases in increasing geometric difficulty. Do not begin with a complete
vehicle because a failure will not identify which placement rule is wrong.

1. **Finite rectangular PEC plate.** Put one straight gap, one rectangular
   door loop, and one off-center fastener on the illuminated face. Repeat each
   at two translated positions and at two in-plane rotations. This is the best
   phase-origin, line-direction, roll-frame, and polarization control case.
2. **Closed rectangular PEC box.** Put equal fasteners on the top and side and
   straight gaps on two faces. This exercises distinct outward normals,
   visibility, radar-frame polarization rotation, and optional clean-body
   shadowing without curved-surface ambiguity.
3. **PEC wedge or ramp.** Put a gap and fastener on the sloped face, then place
   the same feature on the horizontal face. This isolates arbitrary face
   orientation and line endpoint-normal interpolation.
4. **Triaxial ellipsoid or rounded rectangular shell.** Use a curved door loop
   and several fasteners. Unequal principal radii keep the body non-BoR while
   testing curvature, continuously changing normals, and perimeter
   discretization.
5. **Swept wing or faceted panel assembly.** Use long straight and curved gaps,
   including a gap that crosses a mesh-facet boundary. This is the practical
   line-expansion stress test for segmentation invariance and local-frame
   continuity.
6. **Representative vehicle section, then the complete vehicle.** Include
   nearby edges, recesses, and multiple feature types. These cases quantify the
   approximation error from coupling, diffraction, terminations, and repeated
   features after the placement machinery has passed the simpler controls.

A sphere or circular cylinder is useful analytically, but it is not part of
this ladder because it can hide azimuth and roll errors through symmetry.

## Prepared external-solver handoff

`external_case_plan.json` is the canonical, machine-readable handoff for the
next four body families, in execution order: rounded enclosure, wedge/ramp,
swept-wing panel, and representative vehicle door/body section. It currently
expands to 14 concrete
configurations. The plan fixes dimensions, paths, point locations, analytic or
facet normals, feature material/geometry, the 8/10/12 GHz look grid, global
origin, time and outgoing-wave signs, radar polarization frame, amplitude
normalization, parameter sweeps, and the four artifact filenames. The wedge
and wing entries point back to their checked-in canonical source specs rather
than restating a different geometry.

Prepare an empty run outside source control from the GHOST directory:

```text
python geometry_tests/non_bor_feature_validation/prepare_external_cases.py --prepare --output PATH_TO_RUN
```

This writes one `case_spec.json` per configuration and a
`feature_cases.json` that already uses
`ghost.validation.feature-cases.v1`. It deliberately writes no `.grim` data.
Each case directory reserves exactly these externally produced artifacts:

```text
clean_truth.grim
clean_prediction.grim
featured_truth.grim
featured_prediction.grim
```

After exporting or assembling results, reject missing, off-grid, non-finite,
or convention-incompatible artifacts before grading them:

```text
python geometry_tests/non_bor_feature_validation/prepare_external_cases.py --preflight --output PATH_TO_RUN
python Backend/validate_feature_reconstruction.py --manifest PATH_TO_RUN/feature_cases.json --report PATH_TO_RUN/validation_report.json
```

The generated case manifest carries the plan's stable case IDs. The validation
report hashes every artifact and extracts the reusable feature-response hashes
from each assembled prediction. Use that report with
`Backend/create_feature_manifest.py create --validation-report ...`; Production
will not accept a current `validated` response manifest based on a free-form
case name alone.

The checked-in numerical gates are **uncalibrated engineering targets**, not
evidence that any body or feature has passed. No external Maxwell result ships
with this repository. Establish final per-family gates only after mesh,
boundary, and local-reference convergence is smaller than the measured
reconstruction error. Run the small smoke grid in each generated case spec
before committing HPC time to the acceptance grid.

## Four required artifacts per case

Use the exact same frequency, azimuth, elevation, polarization, global origin,
time convention, materials, and far-field normalization for:

1. independent 3-D clean-body truth;
2. the clean body as actually loaded into Assembly;
3. independent 3-D body with the feature explicitly modeled; and
4. the clean Assembly body plus the placed point or line delta.

The validator grades three quantities independently:

- clean baseline: `clean_prediction` versus `clean_truth`;
- featured total: `featured_prediction` versus `featured_truth`;
- isolated feature: `(featured_prediction - clean_prediction)` versus
  `(featured_truth - clean_truth)`.

The isolated comparison is mandatory. A large clean-body return can make a
wrong or sign-reversed feature look excellent in the total-field metric.
Subtraction is always performed on signed complex amplitude, never on dBsm or
linear RCS, and the validator applies no fitted phase, amplitude, range, or
coordinate correction.

Copy `feature_cases.template.json`, replace its paths, then run from the GHOST
folder:

```text
python Backend/validate_feature_reconstruction.py --manifest geometry_tests/non_bor_feature_validation/feature_cases.json --report geometry_tests/non_bor_feature_validation/report.json
```

Paths are resolved relative to the manifest. A case passes only when the clean
baseline, featured total, and isolated feature delta all pass. Treat the
checked-in gates as starting engineering limits; establish feature-family and
frequency-specific limits from converged reference studies.

## Feature-reference construction

For a line feature, solve matching two-dimensional clean and featured coupons
and use the canonical complex `OPN - FRD` result. For seals, the featured coupon
must include the actual material/IBC stack and gap geometry. Validate straight
lines before loops, corners, or curved paths.

For a point feature, solve a local three-dimensional patch twice: installed
feature plus surrounding skin and the identical clean-skin patch. Export their
complex difference in the point feature's local frame, with the origin at the
placement phase center. A standalone screw or fastener response is not the
correct delta because it omits the removed skin and installation interaction.

For both types, converge the clean and featured reference meshes as a pair and
keep the external solver's numerical change below the reconstruction error
being measured.
