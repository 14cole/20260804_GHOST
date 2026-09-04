# Validating Platform Door and Cavity Reconstruction

The definitive validation compares the GHOST reduced-order reconstruction with
an independently solved, directly featured three-dimensional body. Agreement
must be checked on complex far-field amplitude, not only RCS magnitude.

For a staged non-axisymmetric validation ladder, a four-artifact manifest
template, and a validator that separately grades the clean baseline, featured
total, and isolated feature delta, see
`geometry_tests/non_bor_feature_validation/README.md`. The isolated delta gate
is required because a dominant clean-body return can conceal a badly phased or
sign-reversed feature in whole-field metrics.

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

## Assembly tab workflow

The normal interactive path is GRIM's **Assembly** tab in the unified
application. It presents Body, Point Features, Line Features, and Review beside
the 3-D placement and response-comparison views:

1. Open **Assembly → Body**. **Preview layers** opens the advanced response
   tree and display-layer controls.
2. Select the clean base monostatic GRIM. For an external 3-D base, select its
   matching STL or indexed ASCII `.facet`
   surface and its units. A self-contained GHOST BoR result may preview the
   revolved profile embedded in its GRIM without a separate surface. Shadowing
   defaults on for newly selected bodies. BoR shadow geometry can be generated
   from the profile with recorded sag and normal-rotation bounds. Enter the
   installed host material/stack to match the feature-library declarations.
3. In **Point Features** and **Line Features**, select a placement CSV or use
   **Create / edit…**. The table supports add/duplicate/delete, undo/redo, point
   rows and circles, and ordered polylines (repeat the first vertex to close a
   boundary). Choose units first. **Derive normals** preserves coordinates;
   **Snap + normals** explicitly moves selected vertices onto the body. Save,
   preview, and revalidate after either operation. The form shows the
   exact shared local/HPC header and example and can write a blank template.
   Put data immediately after the header; do not add a units or comments row.
   Coordinate units come from the form. The form reads
   the `dataset_id` values and creates the dataset mapping rows automatically;
   choose one canonical OPN-FRD response for every row.
4. Select **Preview geometry** at any time to see the STL/facet or embedded
   BoR body together with CSV point locations and line paths. This staged view
   needs neither an output path nor mapped response files and is not a physics
   validation. If every feature is unchecked, it becomes an explicit clean-body
   preview. Use the **3-D display only** controls to select meter, inch, or
   foot axis labels; solid, edged, or wireframe body rendering; body opacity;
   and Fast/Balanced/High sampled-facet detail. **Faster rotation** temporarily
   uses the Fast display proxy while dragging. These settings do not change
   coordinate input units or any electromagnetic calculation.
5. In **Spatial Feature Configuration → Use**, select the point placements and
   line paths for this variant. Parent checkboxes apply recursively. An
   unchecked instance is omitted from preview, physical validation, response
   loading, and assembly; a response used only by disabled instances does not
   need to be mapped. **Find feature** filters large hierarchies without
   changing membership, and **Copy full selection** captures the exact mask.
   Map every response ID that is still active, choose the
   output, then select **Validate placements**. Check the body, feature
   locations, line ordering,
   units, and CAD orientation (`+x` right, `+y` nose, `+z` up). Supplied
   outward point and line-endpoint normals are drawn in magenta. The point roll
   reference is projected perpendicular to its normal and drawn in lavender as
   the solver-effective local `+x`/azimuth-zero direction. All arrows are
   normalized and use one display-only length derived from the non-vector scene
   extent; arrow length does not represent input magnitude or affect validation.
   The geometry-only preview omits zero or parallel arrows; validation
   remains responsible for reporting those physical placement errors.
   Enable signed line-frame arrows when checking a seam: `+t` follows each
   head-to-tail segment and `+b = +t × +n` is the coupon's signed across-gap
   axis. Reversing the line reverses `+b` and is physically meaningful for an
   asymmetric response.
6. Select **Assemble & save** only after the placement report is
   correct. Review every row in the linked placement-QA table and every mesh or
   feature-library warning. Full response compatibility is enforced while
   assembling. The progress bar reports frequency/direction work; cancellation
   is cooperative and occurs before atomic publication, so an existing output
   is retained. The saved result is automatically available to GRIM as a
   dataset. **Response comparison** opens body, feature-only, and coherent total
   cuts. A feature-only RCS curve is `4π|ΔF|²`, not a dB subtraction. Add saved
   family-only variants to compare contributions. Use the toolbar to save a plot.

Review accepts exact stored frequency/azimuth/elevation subsets. Blank keeps
the complete axis; unstored values are rejected. An all-disabled or no-CSV
configuration can validate and build a body-only baseline with zero feature
field. Translation-phase sampling warnings are useful lower-bound checks,
not proof of convergence between samples.

Before numerical execution, Assembly conservatively estimates peak RAM and
same-volume scratch for the component and final atomic staging archives. A job
that exceeds detected capacity is rejected before full response loading or
temporary-file creation, with the grid size and practical remedies in the
error. `GHOST_MAX_SOLVE_GB` may declare a confirmed process allocation on a
workstation or scheduler; do not raise it beyond memory actually reserved for
the process. Free scratch is checked again under the destination lock before
staging begins.

Save the setup as a named `.assembly.json` recipe before running variants. The
recipe records relative paths where possible, exact enabled/disabled IDs,
mappings, host declarations, exact study samples, tolerances, and source identities. A later load reports moved,
missing, or changed inputs instead of silently reusing the old validation.

Per-node **Show** checkboxes and the global **Show All** checkbox under
**Preview Layers** are preview controls only. They never include or exclude a
feature from the calculation, change coherent/incoherent response membership,
or alter the saved field. Use **Spatial Feature Configuration → Use** for that
purpose. The same rule applies to display units, style, opacity, and facet
detail: the full source body remains authoritative for skin checks, normals,
shadowing, and assembly even when the Matplotlib preview shows a bounded
sampled proxy.

The coherent/incoherent branches of the mathematical Assembly tree accept
commensurate response datasets, not unexpanded feature coupons. In particular,
a 2-D `sigma_2d` solver result is not a positioned 3-D field and must not be
coherently added to a `sigma_3d` body. It can be loaded for inspection and
response-library organization, but its OPN-FRD delta contributes to a vehicle
only through the line-feature form and line-expansion calculation.

The embedded **GHOST** tab calls the same numerical backend as standalone
GHOST. Its solver exports load automatically into GRIM; embedding changes the
application workflow, not the electromagnetic implementation.

For unattended studies, `Backend/place_features.py` remains a settings wrapper
and `Backend/feature_workflow.py` exposes the Qt-free
`FeatureAssemblyRequest` service. Both call the same validation and placement
implementation used by the GUI; they are automation alternatives, not a
second physics path. The wrapper defaults to `VALIDATION_PROFILE =
"advisory"`. Missing, stale, or conflicting metadata and manifests are recorded
without blocking execution or changing complex samples. The optional
`"external"` profile enables strict library and base metadata checks;
`"production"` also requires a certified GHOST BoR body. For a strict audit,
set `HOST_MATERIAL`, and `HOST_STACK_ID` if the library declares one.
Set `HOST_MINIMUM_RADIUS_M` when the library specifies
`applicability.minimum_principal_radius_m`: this is a reviewed lower bound on
both principal radii over every footprint, not a triangle-based estimate.
Missing principal-curvature evidence remains visible as a warning. The wrapper
also exposes `STUDY_FREQUENCIES_GHZ`, `STUDY_AZIMUTHS_DEG`, and
`STUDY_ELEVATIONS_DEG`. Body-only studies are supported without a placement CSV.

Preparation prints advisories without requiring a metadata waiver. Large
workload reviews and warnings in an explicitly selected strict profile require
the printed plan digest in `ACKNOWLEDGED_PLAN_SHA256`. Input changes invalidate
that acknowledgement. Numerical data, geometry, and file-integrity checks apply
in every profile.

## Creating and checking an evidence-bound feature manifest

A production point or line library must place exactly one UTF-8 JSON sidecar
next to each mapped GRIM response. Use the supported command to create the
sidecar only after a responsible team member has reviewed the declared host,
response conventions, applicability envelope, and the machine-generated
full-wave report. A `validated` manifest cannot be created from typed case IDs
alone:

```bash
python Backend/create_feature_manifest.py create door_seam.grim \
  --dataset-id door_seam --feature-kind line \
  --host-material "PEC outer skin" \
  --frequency-min-ghz 1 --frequency-max-ghz 12 \
  --footprint-radius-m 0.02 \
  --minimum-along-line-normal-turn-radius-m 0.5 \
  --maximum-conical-incidence-deg 20 \
  --maximum-path-vertex-turn-deg 30 \
  --validation-status validated \
  --validation-report validation_report.json \
  --validation-case-id door-seam-flat-pec-v3 \
  --phase-calibration-case-id door-seam-flat-pec-v3 \
  --attest-reviewed-evidence

python Backend/create_feature_manifest.py check door_seam.grim \
  --dataset-id door_seam --feature-kind line
```

The create command writes `door_seam.grim.feature.json` by default and refuses
to overwrite it without `--force`. For `door_seam.grim`, Assembly supports
either
`door_seam.grim.feature.json` or `door_seam.feature.json`, never both. The same
JSON may instead be embedded under `feature_library_manifest_json`; embedded
and sidecar declarations must agree when both exist. The supported creator
does not rewrite a response archive that already embeds a declaration. It
always writes the current v3 response schema and v2 line-calibration schema.
Existing v1/v2 declarations are advisory by default. Strict certification
requires current evidence-bound declarations.
Use `--host-stack-id` for a distinct material/coating stack and
`--minimum-principal-radius-m` for an evidence-supported bound on both host
principal radii. Material names are matched after whitespace/case normalization;
stack IDs distinguish physically different constructions. These declarations
describe reviewed applicability; they do not generate missing full-wave evidence.

Current GHOST 2-D responses must carry `amplitude_version=2`. This survives
coherent GUI subtraction and line-delta export. Regenerate incompatible earlier
responses; a delta label does not repair an old phase convention. Point-library
pole samples use the documented fixed Cartesian transverse basis; interpolation
transports these pole values into each surrounding azimuth meridian.

This is a complete line-response example (replace the engineering values and
case IDs with evidence from that exact response library):

```json
{
  "schema": "ghost.feature-library-manifest.v3",
  "dataset_id": "door_seam",
  "feature_kind": "line",
  "subtraction_order": "featured_minus_clean",
  "phase_origin": "placement_line_on_host_outer_skin",
  "frame_convention": "line_local:+t=head_to_tail;+n=outward;+b=cross(t,n)",
  "time_convention": "exp(+jwt)",
  "response_content_sha256": "<64-character digest written by the creator>",
  "host": {
    "material": "PEC outer skin used by clean and featured coupon solves"
  },
  "applicability": {
    "frequency_ghz": {"min": 1.0, "max": 12.0},
    "footprint_radius_m": 0.02,
    "minimum_along_line_normal_turn_radius_m": 0.5,
    "maximum_conical_incidence_deg": 20.0,
    "maximum_path_vertex_turn_deg": 30.0
  },
  "line_phase_calibration": {
    "schema": "ghost.line-phase-calibration.v2",
    "tm_deg": 166.9,
    "te_deg": -9.2,
    "grazing_taper_deg": 10.0,
    "case_ids": ["door-seam-flat-pec-v3"]
  },
  "validation": {
    "status": "validated",
    "case_ids": ["door-seam-flat-pec-v3"],
    "evidence": [{
      "schema": "ghost.validation.feature-case-evidence.v1",
      "case_id": "door-seam-flat-pec-v3",
      "passed": true,
      "report_sha256": "<digest of validation_report.json>",
      "comparison_sha256": "<digest of this case result>",
      "feature_response_content_sha256": "<same response digest as above>",
      "gate_limits": {
        "active_floor_db": -40.0,
        "max_normalized_rms": 0.25,
        "max_magnitude_p95_db": 3.5,
        "max_phase_rms_deg": 25.0,
        "min_coherence": 0.95
      },
      "artifact_sha256": {
        "clean_truth": "<digest>",
        "clean_prediction": "<digest>",
        "featured_truth": "<digest>",
        "featured_prediction": "<digest>"
      }
    }]
  }
}
```

For a point library, set `feature_kind` to `point`, use
`phase_origin = placement_point_at_pattern_phase_center`, and use this exact
frame string:

```text
cavity spherical: +z=aperture outward; az=atan2(y,x); el=asin(z); VV=theta; HH=phi; VH=HV
```

Point manifests omit the three line-only applicability values and the complete
`line_phase_calibration` object. `footprint_radius_m` bounds the local
installation region represented by the delta. Overlapping footprints identify
feature clusters whose mutual coupling is omitted. In Production an overlap is
rejected. In the default profile footprint annotations are retained for review
without enforcing them. Strict line placements are additionally
gated against the declared frequency, normal-turn-radius,
maximum-conical-incidence, and path-vertex-turn envelopes. The 10-degree
grazing taper records the fixed complex-amplitude visibility ramp used by the
current line-expansion implementation; it is not a fitted manifest parameter.

Host material/stack and curvature annotations are optional in the default
profile. Strict profiles require a matching installed host and any declared
principal-radius bound. The material response is supplied by the datasets.

The Production profile rejects missing, provisional, uncertified, legacy, or
unbound evidence. The validator report records stable case IDs, all four
artifact hashes, and every reusable response-content hash found in the
assembled prediction's provenance. The manifest creator re-hashes those four
artifacts and refuses a case unless it passed all three comparisons and
actually exercised the exact response being certified.

A library-certification case may place the same response at several locations,
but it must not combine different reusable response files. Combined line/point
or mixed-library cases remain valuable system regressions; they cannot certify
one member because opposite errors could cancel in the aggregate field.
The default advisory profile accepts external libraries without these contracts;
the profile and assumptions are recorded in provenance.
A manifest still includes a **team attestation**: the software can prove what
files were compared and which gates passed, but it cannot prove that an
external file came from Maxwell's equations or that its mesh converged. The
reviewer remains responsible for solver independence, convergence, materials,
and representative envelope coverage.

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
  `sigma = 4 pi |F|^2` in square meters. RCS is displayed as dBsm.
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

1. Put the local model origin at the cavity aperture phase center.
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

Place this delta at the same physical aperture phase center used as the local
3-D origin. The direct full-body featured solve then measures the approximation
left out by placement, principally long-range body/feature mutual coupling and
multiple scattering.

## Placing on an external platform result

An imported clean-platform monostatic GRIM does not need to originate in the
BoR solver. In the Assembly form, select the external GRIM as the base and
select its matching indexed ASCII `.facet` or STL surface. The surface
supplies skin checks and outward normals; the GRIM supplies the exact
frequency/azimuth/elevation grid on which the feature field is evaluated and
coherently added. Unlike a self-contained BoR profile, an external base cannot
be validated for placement without that surface.

Production also requires one canonical `<surface>.assembly.json` binding. It
binds the exact external base and surface bytes to the selected surface units,
fixed CAD frame, a team-controlled geometry revision, and the reviewed
solve-to-surface registration case:

```bash
python Backend/create_feature_manifest.py create-surface-binding \
  clean_vehicle.grim vehicle.stl --surface-units inches \
  --geometry-id vehicle-mesh-r7 \
  --attestation-case-id solver-registration-042 \
  --attest-reviewed-registration

python Backend/create_feature_manifest.py check-surface-binding \
  clean_vehicle.grim vehicle.stl --surface-units inches \
  --geometry-id vehicle-mesh-r7
```

For `vehicle.stl`, the only supported path is
`vehicle.stl.assembly.json`. The checker re-hashes both selected files and
requires `frame_convention = CAD:+y=nose;+x=right;+z=up`; a changed GRIM,
changed mesh, or different unit selection fails. This eliminates invalid
cross-format hash comparisons, but remains a team attestation: exact byte
identity cannot prove that the external solver actually used that mesh, origin,
scale, or orientation. The cited registration evidence must establish those
facts independently.

Assembly reconstructs edge topology and rejects duplicate faces, non-manifold
edges, mixed winding, and globally inward closed components (unless one valid
global normal flip is selected). It does not yet prove that nonadjacent
triangles never self-intersect or that nested/overlapping closed shells are
intentional. The cited registration evidence must therefore include a CAD/mesh
integrity check for self-intersections and unintended duplicate shells; do not
waive a topology warning merely because the preview looks solid.

Strict profiles require the base GRIM itself to declare that it is the coherent
physical platform far field at the global vehicle origin in radar-frame
VV/HH/VH, along with the required angular/grid conventions. The default profile
treats base selection as the working declaration even when descriptive tags
are absent or conflict, and records assumptions without converting fields.
In every profile, stored sigma
and phase undergo full normalization and consistency checks, and a base
explicitly tagged `combine_role=power` remains invalid. The combined output is
written back with the complete canonical coherent schema.

The platform GRIM, surface mesh, and placement CSV must use the same physical
origin and orientation. A mismatch creates a deterministic two-way phase error
even when the feature magnitude looks plausible. Enabling **Shadowing** uses
the triangle mesh for binary geometric ray blockage; therefore shadowing is
unavailable without an STL/facet surface. Disabling it retains the local
outward-facing test. Neither choice adds diffraction, creeping waves, or
body-feature mutual coupling: those limitations must still be measured against
a directly featured full-wave platform solve.

For a point feature, form the installed-minus-clean complex delta in the 3-D
solver or an independently validated lossless export process. Preserve the raw
complex Jones response; do not subtract dBsm or linear RCS. A legacy compact
result made with GRIM's **Coherent subtraction** can be loaded when its
sigma, phase, complete VV/HH/VH matrix, azimuth seam, and frequency support
are usable. Convention annotations need no validation in the default profile.

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
the dataset rather than asking placement to infer it. The same explicit-role
model applies when a 2-D GUI subtraction is selected for a line `dataset_id`:
missing semantic tags are not a blocker. Its stored linear quantity must still
be sigma_2d, elevation must be the singleton zero cut, and both TE/VV and
TM/HH must be present.

At elevation `+90` or `-90` degrees, local azimuth is geometrically undefined.
Use the fixed local `x/y` transverse basis at those poles and store the same
complex Jones values in every azimuth row. GHOST rejects an azimuth-dependent
pole instead of allowing roundoff to choose an arbitrary orientation for an
anisotropic point feature.

Subtraction order changes the coherent answer. The canonical delta is
`featured - clean`, which is OPN-FRD in the supplied GHOST/CEM workflow. Point
and line placement accept only that standard; neither exposes a
subtraction-order switch. Convert an FRD-OPN dataset once, before placing it,
by negating the complete complex field. Reversing order without that correction
changes the delta by 180 degrees and changes its interference with the platform.

## One point-placement CSV

All fasteners, antennas, compact cavities, and other point features share one
CSV. Select it in the Assembly form or use **Save blank point CSV template**.
The GUI validates the header, discovers
`fastener`, `antenna`, and any other `dataset_id` values actually present, and
creates one response-file selector for each ID. The user supplies the correct
OPN-FRD point dataset; the software does not infer a feature type from a file
name or CSV shape.

The CSV header and order are fixed:

```csv
placement_id,dataset_id,x,y,z,nx,ny,nz,roll_x,roll_y,roll_z
fastener_001,fastener,1.2,8.4,0.5,0,0,1,1,0,0
fastener_002,fastener,-1.2,8.4,0.5,0,0,1,1,0,0
antenna_001,antenna,0,12.7,2.1,0,0,1,1,0,0
```

Values use the coordinate units selected in the form and the CAD frame
(`+x` right, `+y` nose, `+z` up). `dataset_id` must exactly match its generated
mapping row (`COORDINATE_UNITS` and the dataset dictionary are the automation
equivalents).
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
features share one CSV. Select it in the Assembly form or use **Save blank line
CSV template**. The GUI validates the
header, discovers `door_seam`, `panel_gap`, and any other `dataset_id` values
actually present, and creates one response-file selector for each ID. Select
the correct OPN-FRD TE/TM dataset for each mapping.

The CSV header and order are fixed:

```csv
line_id,dataset_id,segment_index,x1,y1,z1,x2,y2,z2,n1x,n1y,n1z,n2x,n2y,n2z
door_001,door_seam,1,1.0,8.0,0.0,1.0,9.0,0.0,0,0,1,0,0,1
door_001,door_seam,2,1.0,9.0,0.0,0.0,9.5,0.2,0,0,1,0,0.1,0.995
gap_001,panel_gap,1,-2.0,6.0,0.0,-2.0,10.0,0.0,0,0,1,0,0,1
```

Values use the coordinate units selected in the form and the CAD frame
(`+x` right, `+y` nose, `+z` up). `line_id` identifies one physical chain;
repeated instances use
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
`dataset_id`, not in an amplitude or geometric scale column. The form can save
a blank line template with the exact header.

At a mesh crease, adjacent segments may use different outward endpoint normals
at the same shared coordinate. Placement validates one-sided samples within
each segment and uses the supplied normal only to resolve a true equal-distance
choice between incident facets; triangle storage order does not own the edge.
This does not relax the skin, positive-outward-dot, or angular-error gates.

Line order is therefore physical, not cosmetic. Reversing a path reverses the
coupon's signed across-gap axis; an asymmetric coupon must also be mirrored as
`A_reversed(phi) = A(180 - phi)`. Use one consistent winding for closed door
loops and document which side of the 2-D coupon corresponds to the path's
positive across-gap direction.

An immediate return along the previous segment is rejected, including a
near-retrace whose applicability footprints still overlap beyond the shared
endpoint neighborhood. Ordinary corners remain legal, but their turn angle
must stay inside the response manifest's reviewed corner envelope. Closed-loop
corners and junctions still require independent evidence because the straight
2-D coefficient does not create a 3-D corner interaction.

Line expansion applies a 10-degree **raised-cosine grazing illumination
ramp** to complex field amplitude: the weight is zero at grazing, one-half at
5 degrees above the local tangent plane, and unity at and beyond 10 degrees;
the back side is zero. This is a look-angle visibility model, not an arclength
window. It does not taper the ends of an open path, alter a closed-loop seam,
or add special weights at CSV segment joins.

## Building a door or seam delta

For the 2-D cross-section pair:

1. Use the same clean host stack and feature-centered origin.
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
4. Copy
   `geometry_tests/non_bor_feature_validation/feature_cases.template.json`,
   enter the four paths for each case, and keep them on exactly the same grid.
5. Run the manifest validator so the clean baseline, featured total, and
   isolated complex feature delta are all graded:

   ```bash
   python Backend/validate_feature_reconstruction.py --manifest geometry_tests/non_bor_feature_validation/feature_cases.json --report geometry_tests/non_bor_feature_validation/report.json
   ```

The report includes normalized complex RMS error, 95th-percentile magnitude
error, phase RMS, complex coherence, and per-channel diagnostics. Samples near
truth-field nulls remain in complex RMS but are excluded from pointwise phase
and dB statistics because phase at a null is undefined. Production evidence
must use an active-field floor of -40 dB or lower; raising that floor toward the
pattern peak would discard too much angular evidence even if the remaining
samples passed.

It also records the SHA-256 of all four artifacts and extracts the exact feature
response-content hashes from `featured_prediction` Assembly provenance. Keep
the four artifacts unchanged until the response manifest is created; the
creator checks them again and binds their hashes into the v3 sidecar.

The reported best-fit global phase is diagnostic only and is never applied. A
large nearly constant phase offset usually indicates a mismatched origin, time
sign, propagation direction, or far-field definition—not a correction that
should be fitted away.

For the deterministic rounded-enclosure, wedge/ramp, swept-wing, and vehicle-
door handoff, in that execution order, use
`geometry_tests/non_bor_feature_validation/external_case_plan.json` with
`prepare_external_cases.py`. It generates 14 case specifications and one
existing-schema validator manifest without generating any solver result. Its
preflight checks all four files against the exact 8/10/12 GHz, angle,
polarization, phase-reference, and complex-amplitude contract. The proposed
gates are uncalibrated engineering targets until independent converged
full-wave fields exist.

The always-run all-GHOST circumferential PEC-groove fixture in
`tests/test_feature_reconstruction_physics.py` now checks the direct BoR delta
against the placed 2-D delta for one controlled geometry. It is useful evidence
for coordinate, polarization, normalization, and phase-placement regressions,
but it is not an independent full-3-D platform reference. The default 3.5 dB
and 25-degree guide gates therefore remain uncalibrated initial engineering
limits for other feature families. Replace or justify them using your
independent comparison, and validate representative frequency, aspect,
curvature, size, depth, material, and placement extremes before distributing a
feature library.
