# GRIM application

Backend module locations and Python entry points are listed in the
[backend guide](BACKEND.md).

GRIM is the host application for this distribution. Its desktop tabs are
**Plotting | ISAR | FREDDY | GHOST | Assembly | PPT | Python**. GHOST and FREDDY remain
self-contained tools under `tools/`; GRIM embeds their authoritative user
interfaces instead of copying their numerical implementations into the
plotting code.

Run GRIM from the repository root after an editable installation:

```powershell
py -m pip install -e .
grim
```

## Importing an unfamiliar text table

Drop CSV, TXT, DAT, ASC, ASCII, or TSV files into GRIM, or use **Load**.
Recognized formats use their existing readers. When a text table is not
recognized, GRIM opens a column-labeling dialog after the background reader
finishes. Review the source preview, delimiter, header setting, and any
preamble lines to skip. Each selected variable can use a source column or
one constant value for every row. Unused source columns are ignored.

For a whitespace table ordered **frequency, azimuth, phase, magnitude**:

1. Map those four source columns to their variables.
2. Select the frequency unit, such as Hz; GRIM converts it to GHz.
3. Select degrees or radians for the angles and phase.
4. Set elevation to constant `0` and choose the polarization label (default `VV`).
5. Declare whether magnitude is RCS amplitude, linear RCS power, dBsm,
   scattering width, or a power/amplitude ratio, then import.

The angular mapping uses GRIM's conic azimuth/elevation convention. A missing
phase column stays unavailable when **phase** is unchecked. Check it and enter
a constant only when a fixed phase is intended. Missing grid combinations
remain missing; conflicting duplicate samples are rejected.

Imported mapped tables appear as unsaved datasets. **Save** writes their GRIM
representation, including the column and unit mapping in the history and
metadata. Source text files are kept unchanged. Cancelling one file's labeling
continues the remaining import queue. Use **FREDDY → File → File Converter…**
to save a standalone converted CSV or whitespace table instead.

## Appearance

Choose **View → Application Palette** to switch the complete integrated GUI
between **Colorful**, **Light**, **Dark**, and **Raytheon** chrome. The Raytheon
application palette uses only the official primary and secondary colors:
white, black, Cool Gray 1/5/10, and Red 186. Its five tertiary colors are
reserved for PowerPoint charts with enough data series to require them; they
are not used for application chrome or embedded-tool plots.
The choice is saved for the next GRIM session and is applied immediately to
GRIM controls, both plot canvases, GHOST geometry/solver views, FREDDY controls
and plots, and the Assembly 3-D preview. Changing appearance never changes a
dataset, solver input, FREDDY project, or assembly geometry. PowerPoint's white
slide preview remains output-faithful rather than following application chrome.

The **Plot Colormap** control under Plot/ISAR Settings is independent and
affects only heatmap-style data rendering. Explicit Plot Colors background,
grid, or text overrides also remain in effect when the application palette
changes.

## Flat RCS CSV interchange

GRIM and CEM Tools share the versioned `grim.flat-rcs.v1` CSV/TXT contract.
It is the supported human-readable interchange format; use `.grim` when every
solver-specific ancillary array and provenance field must remain lossless.
Native `.grim` saves are compressed and atomically published by default, so
they are normally much smaller and faster to reload than a dense text table;
CSV is intended for inspected or cross-tool interchange, not bulk working-set
storage. Multi-file saves/exports stage the full batch and roll back a late
publication failure instead of presenting a partial result as complete.
Version 1 repeats metadata on every data row so filtering or concatenating
rows cannot detach the samples from their units. Its columns are:

```text
grim_csv_schema,azimuth,azimuth_unit,elevation,elevation_unit,frequency,
frequency_unit,polarization,rcs_linear_quantity,rcs_log_unit,
angular_coordinate_system,great_circle_coordinate_convention,
angular_roll_deg,angular_tilt_deg,polarization_basis,time_convention,
phase_reference,phase_wrap,[magnitude column(s)],[phase_deg]
```

The selected magnitude columns are `magnitude_power_linear`, `magnitude_dbsm`,
`magnitude_dbke`, and/or `magnitude_db`; their names are never overloaded.
`magnitude_power_linear` is always the stored nonnegative linear power: 3-D
sigma for `sigma_3d`, 2-D sigma for `sigma_2d`, or a dimensionless power ratio
for `power_ratio`. It is never field amplitude. The only valid quantity/log
pairs are `sigma_3d`/`dBsm`, `sigma_2d`/`dBke`, and `power_ratio`/`dB`.
`azimuth_unit` and `elevation_unit` are independently `deg` or `rad`;
frequency is explicitly `Hz`, `kHz`, `MHz`, or `GHz`. `phase_deg` may be blank
for a magnitude-only sample. A nonblank phase is useful for coherent work only
when `phase_reference`, the polarization basis, and the time convention are
also physically meaningful. `phase_wrap` is `-180_180` or `0_360` and declares
the interval used for `phase_deg`. The current versioned writer emits that
declaration on every row and normalizes exported phase into the declared
interval; the reader restores it to the dataset's units metadata. Older tables
without the column retain the signed `-180_180` default. This is representation
metadata only: phase values separated by 360 degrees describe the same complex
field.

The reader also deliberately supports two older unversioned layouts:

- GRIM tables with plain `azimuth,elevation,frequency` axes interpret
  `magnitude_linear` as linear power. Missing frequency units retain the old
  magnitude-based inference, but the loaded dataset history and metadata now
  flag that inference instead of silently presenting it as authoritative.
- CEM Tools tables with `azimuth_deg,elevation_deg,frequency_GHz` interpret
  their legacy `magnitude_linear` as field amplitude and square it. New CEM
  exports always write the versioned power column.

Versioned files reject the ambiguous `magnitude_linear` name. When a file
contains both linear and logarithmic magnitude columns, GRIM converts and
cross-checks them rather than arbitrarily trusting the first. Before creating
power and phase arrays, the loader evaluates the Cartesian axis product and
available RAM. It scans large tables twice instead of retaining a Python
dictionary for every row. `GRIM_MAX_CSV_GRID_GB` can declare an intentional
per-process limit after an unusually large grid has been reviewed.

## Plotting and dataset operations

Click a curve or scatter point in the Plotting canvas to highlight every series
from that dataset with a light-gray outline and highlight its visible legend
entries. Legend entries are clickable too; click empty plot space to clear the
highlight. Right-click a curve, point, or legend entry to change that dataset's
line width, line type, color, or switch between Line and Scatter. Scatter symbols
and symbol sizes are also adjustable. Styles apply immediately to the plot and
legend and remain in effect when replotting or using Hold during the session.
Each dataset has its own style, so lines and scatters can share a plot. Pan and
Zoom Box retain their mouse controls while enabled.

The current row is the active dataset: its parameter lists and axis units are
the display reference. Other selected datasets may use compatible Hz/kHz/MHz/
GHz or degree/radian storage; GRIM converts selections and labels without
changing their files. Conic and great-circle charts, different great-circle
frames, different physical quantities (`sigma_3d`, `sigma_2d`, or ratio), and
different logarithmic conventions cannot be overlaid as though they were the
same ordinate. To correct a file-format coordinate assumption, select the
datasets and use **Geometry & Units → Set Coordinates** (also on the dataset
right-click menu). Choose **Azimuth / Elevation (Conic)** or **Aspect / Pitch
(Great Circle)**. The action creates and selects copies for plotting, preserving
all angle values, sample order, power, and phase. For example, select matching
PIO and PTM data and choose Azimuth / Elevation to overlay both as elevation
cuts. The declaration supports nonzero cuts and every polarization; it performs
no geometric conversion. Great-circle declarations include convention and
roll/tilt fields. Save the corrected copies as `.grim` to retain the choice.
The Python recorder uses `grid.set_angular_coordinate_system(...)` for replay.

RF Compare uses one explicit selected azimuth, elevation, or
frequency sector per dataset and matches coordinates one-to-one. Its 0–100 RF
Agreement Index replaces Pearson correlation with agreement measures for strong
returns, the full pattern, normalized power error, and local-sector agreement.
The plot presents these as plain-language percentages rather than statistical
acronyms. It also reports the typical first-minus-second level difference,
average level error, a bound containing 95% of point errors, peak-coordinate
shift, and the active statistics range. Statistics
use every common finite sample even when the displayed lines are decimated.
Phase comparisons are labeled as overall phase match, average phase alignment,
difference consistency, average phase difference, typical phase error, and the
weakest sub-sector, with wrap-safe calculations at the ±180° seam. A phase
comparison with undeclared phase-center/time/basis metadata remains viewable
for legacy data, but the status message states the physical assumption.

For azimuth RF Compare, a compact row above the plot sets the Min and Max
azimuth statistics bounds. The bounds initialize to the actual minimum and
maximum selected in the Azimuth parameter list. **Show all azimuths** defaults
off, so the plot initially shows only that sector. Turning it on displays the
complete common azimuth sweep for both datasets and the residual, but every
reported statistic remains restricted to the entered Min/Max sector.
The yellow region marks that entered statistics range only when it is narrower
than the full common azimuth span; a full-span comparison has no highlight.

**Delta Map** beside RF Compare compares two selected datasets over any two
of frequency, azimuth/aspect, and elevation/pitch. Select the displayed ranges
and one polarization in the sidebar, then choose the horizontal/vertical axes
and the fixed third coordinate above the plot. The fixed-coordinate dropdown
contains the reference dataset's coordinates. Dataset A and B are named above
the plot; **Swap A / B** reverses the signed subtraction.

Each cell is the difference of the two logarithmic levels, labeled **A - B
(dB)**. Source values in the hover/click readout retain dBsm, dBke, or dB.
Blue is negative, red is positive, and the scale is symmetric around zero.
Turn off **Auto color limits** to lock a symmetric ±dB range across slices.
Clipped scales have colorbar extensions. **Cell values** labels maps up to
200 cells; larger maps retain the detailed hover readout. Click a cell to keep
its readout after leaving the axes. Export Plot and the Python recorder support
the map, including its axes, fixed slice, source order, and color limits.

Matching uses the existing physical coordinate tolerances (1 kHz and 1e-6
degrees), with no interpolation, fixed-axis broadcasting, or implicit angular
frame conversion. Mixed physical quantities or log conventions are rejected.
The selected reference grid is preserved: unmatched cells, nonfinite values,
and nonpositive powers are gray, never silently filled or floored. An absent
fixed coordinate, ambiguous match, or wholly unpaired slice stops the plot.
Only the selected 2-D slice is gathered, with a two-million-cell limit checked
before gathering. Delta Map always compares logarithmic levels; use RF Compare
for phase comparisons. Each colored box represents a sampled coordinate pair,
not an average over its drawn area.

Rendering caps the number of visible lines, points, image cells, waterfall
panels, and explicit ticks. Line and magnitude-image reduction retain bucket
extrema; oversized phase images stop with a request to narrow the axes rather
than applying a nonphysical scalar phase reduction. These display limits do not
change stored samples or reported full-resolution statistics. Overlap,
statistics, interpolation, medianization, Range Cal, Support Ref -, native `.grim` saves,
dataset loading, and CSV export run in the dataset worker so large jobs do not
freeze the GUI. High-memory dataset work and ISAR reconstruction are serialized
to protect process memory; files dropped during ISAR are queued and load when
the reconstruction reaches an idle boundary. Interpolation and medianization
accept physical degrees while retaining each dataset's native angle unit and
preflight output axes/RAM before allocation. Statistics operates on linear
power—not on already-logarithmic dB samples—and creates compact reduced grids by
default; repeating a statistic across the original grid is an explicit,
memory-preflighted opt-in.

The standalone **Percentile** button is an azimuth-reduction preset of the same
statistics workflow. It asks only for the percentile (90 by default), computes
it across all azimuths on linear power, and creates one compact result per
selected dataset. Elevation, frequency and polarization axes are preserved;
azimuth becomes a singleton aggregate label rather than a measured direction.
Coherent phase is undefined. For a repeated reference curve on the original
azimuth grid, use **Stats → percentile**, reduce **Azimuth** only, and enable
**Repeat the reduced value across the original grid**.

The divider between Datasets and Parameters can be dragged vertically: dragging
it upward enlarges the parameter lists and reduces the dataset table, while
dragging it downward does the reverse. The active row continues to drive the
parameter lists. `Ctrl+O` opens datasets, `Ctrl+Shift+O` performs Overlap, and
ordered subtraction/division use selection order. Delta-dB and coherent
division require exactly two operands. **Join / Merge…** (also `Ctrl+J`) opens
one dialog for unioning existing bins without interpolation. The default merges
equal or complementary finite overlaps and rejects conflicts; priority and
averaging policies must be selected explicitly.

## ISAR formation and numerical results

The ISAR tab forms a static-scene, far-field, monostatic image from calibrated
complex phase history referenced to one fixed origin. Frequency units must be
explicitly declared as Hz, kHz, MHz, or GHz. Explicit near-field, bistatic,
quasi-/pseudo-monostatic, drifting-reference, or
uncompensated-motion metadata fails closed. Genuinely missing legacy convention
metadata and unknown producer vocabulary do not block formation: GRIM treats
them as user-owned assumptions and records every undeclared contract field in
the completed result and artifact.
This is especially important for Pioneer PIO, whose interchange header cannot
carry the full acquisition contract. The same contract also binds the two-way
range law: an explicit
`S~exp(+j*2*k*R)` declaration is blocked under the default axes unless both
Flip X and Flip Y are deliberately enabled and checked against a known
asymmetric target.

Fast PFA is the interactive narrow-look path; Accurate Cartesian PFA removes
the remaining range-curvature approximation for supported apertures. Sparse L1
is labeled **experimental**: it is a fixed-lambda sparse image reconstruction,
not target/contaminant classification, BPDN noise removal, pylon removal, or
bird removal. Its status reports convergence, residual, objective/duality gap,
support, and debias diagnostics. A strong unwanted scatterer can remain while a
weak wanted scatterer is suppressed. Wide selections use a labeled nonlinear
max-look composite of narrow subapertures and are qualitative rather than a
single coherent 360-degree reconstruction.

**Plan image** evaluates the selected acquired frequencies and angles without
forming an image. In **ISAR Settings**, enter the occupied cross-range and range
half extents in metres, then choose **Recommended PFA** for a scene-dependent
Fast/Accurate choice. The planner distinguishes native phase increments from
Fast PFA range-curvature error. Passing a sampling check is not a guarantee of
interpolation accuracy; upsampling cannot recover missing measurements. Without
entered bounds, advice uses nominal periodic scene limits and labels that
assumption. Coherent bounds use the mean-look frame; composite bounds use the
fixed body frame. Elevation projects the horizontal image plane and cannot
independently resolve height from one angular cut.

**Image mode** explicitly selects a single coherent image, a qualitative max-look
composite, or the legacy automatic policy (composite above 20 degrees). Coherent
PFA requires an aperture narrower than 90 degrees. Every composite sublook now
respects a 10-degree angular bound, including acquisitions whose angular density
changes. Regular strided angular selections form a band; disconnected physical
sectors form separate images. An isolated sector produces an actionable error.
Composite grid size is configurable from 32 to 4096 pixels per side. Scene bounds
crop the retained result but do not reduce the full coherent FFT or add resolution.
Frequency-band controls display the dataset's own units and acquired bounds.

The persistent **Quality** panel shows physical ranges, sampling/curvature
warnings, coverage/gaps, nominal resolution, and windowed origin PSF cuts. Power
FWHM, peak sidelobe ratio and integrated sidelobe ratio describe **one-dimensional
cuts** through the origin response on the actual gridded support, not a full 2D
ISLR or an off-center focusing guarantee. Sparse and composite formation are
nonlinear and have no single fixed PSF; composite artifacts retain individual
look diagnostics. Sparse quality also includes a native polar-data residual,
distinct from its gridded optimization residual. The bounded diagnostic uses up
to 4096 source positions and 512 retained image points, records sampling and
omitted-energy fractions, and explains when it cannot be computed. It evaluates
the full formed image before optional scene cropping and display flips.
**Cancel** stops at a processing block, with worker-stage progress in the status
bar. Sparse-only controls and the ignored Sparse taper are disabled appropriately.

Nonuniform samples are interpolated only within acquired support. Missing
frequency or azimuth sectors are placed on the uniform working grid with zero
measurement weight; GRIM reports their count, size, unsupported fraction, and
resulting phase coverage. It never turns a large unmeasured sector into
fully-observed synthetic samples. An excessive expansion stops with guidance to
form contiguous bands separately.
Accurate PFA shares interpolation geometry between the complex numerator and
coverage. Fully observed cubic stencils retain cubic interpolation; stencils
touching missing data use the same positive linear weights for both arrays.
This preserves the coherent gain of an origin point under missing support.

**Export ISAR Result** saves the latest completed full-resolution image as a
transactional `.isar.npz` artifact. Coherent looks include the complex image and
distance axes, with magnitude derived losslessly on load instead of stored as a
redundant second image. Magnitude-only wide composites retain their magnitude
array. Storage adaptively skips slow ZIP compression for noise-like complex data
and uses it when a bounded sample predicts useful savings, including for flipped
or strided images. Every artifact includes a versioned JSON manifest with
selected-source content digests, source history/conventions, formation settings,
user-assumed undeclared convention fields, coverage, sampling, and sparse
diagnostics. Save and load
preflight array headers, normalized working bytes, band/cell counts, and
recursively bounded metadata before numerical extraction; complex axes,
post-cast overflow, malformed legacy magnitude, object payloads, duplicate/path
members, and oversized manifests fail closed. Wide max-look composites
explicitly record that no complex image exists.

Artifacts also save all six gap diagnostics with units, engine version,
realized spatial-frequency support, phase/frame conventions, accuracy plans,
PSF/native residuals, memory accounting, and exact composite sublook indices.
Complex pixels use a spatial-frequency-origin-demodulated phase convention;
`physical_coefficients(image, x_range, y_range, image_contract)` restores physical
point-coefficient phases. It does not make a general FFT image an exact sparse
point model. Image intensity remains generic dB, not calibrated per-pixel dBsm
or dBke. Pixel-center axes are displayed using their outer half-cell boundaries.

**Open result** loads a numerical artifact without its original acquisition.
**Compare result** compares the current completed image against a saved artifact,
or loads two files when no current image exists. It checks image/frame,
normalization, acquisition and declared phase/calibration compatibility. Older
artifacts without the required contracts can be viewed but must be re-formed
for quantitative comparison. Comparisons provide shared intensity scales,
linked physical axes, A-minus-B intensity differences, peak profiles and
statistics for the current zoomed ROI. Different grids require explicit
resampling of **linear intensity** onto their physical overlap; dB is never
interpolated. The GUI limits comparisons to one million overlap cells and
preflights additional artifact arrays against a 512 MiB viewer allocation
allowance with space reserved for plotting. Coverage and PSF differences still
need interpretation; compatible metadata alone does not certify calibration.

**Save recipe** freezes the accepted formation recipe as non-executable
`.isar.json`, including physical azimuth, frequency, elevation and polarization
selectors. **Load recipe** validates those samples against the active dataset
and restores controls; it does not automatically form an image. Equivalent
Hz/GHz axes match. Settings outside GUI precision or limits are explained before
any control is changed. Headless replay uses the same validated selectors:

```python
from GRIM_Backend.scripting.api import load_recipe, recipe_arguments, form_isar

options = recipe_arguments(dataset, load_recipe("image.isar.json"))
bands, elapsed = form_isar(dataset, retain_complex=True, **options)
```

`plan_isar(unwrapped_azimuth_degrees, frequency_hz, ...)` and the bounded native
`PolarPointOperator` forward/adjoint are available through the scripting API.
The adjoint is not an inverse; Sparse L1 still uses its existing gridded LASSO
objective. A production accelerated native reconstruction remains future work.

Export Plot and numerical-result export are disabled while a newer formation is
pending, so a previous canvas cannot be mistaken for current settings. Clearing
the canvas invalidates only the picture export, not a still-valid numerical
artifact. Plotting-tab renders use an independent freshness counter and cannot
invalidate a still-current ISAR result. The Python recorder captures the exact
accepted worker-start recipe and current display style for headless replay,
rather than rereading controls that changed while the worker ran. Headless ISAR
uses the GUI's peak-preserving display bound, -120 dB intensity floor, physical
unit/frame labels, color scale, and aspect settings. Long selector and
interpolation axes are emitted as compact hard-coded
`numpy.linspace(start, stop, count)` expressions only when that expression
reproduces every float64 value exactly.

`isar_bpde.py` provides a tested headless foundation for future physical
component separation: named implicit dictionaries, cross-component coherence
screening, and residual-constrained complex BPDN that returns every component
phase history and the residual. Its direct point-scatterer dictionary is a
bounded reference operator for reviewed small problems: phase blocks are
reused only inside explicit cell, payload-byte, and block-count budgets, and an
uncached oversized iterative solve is gated both per dictionary and in
aggregate unless the caller deliberately opts in. The BPDN solve is internally
amplitude-normalized and reports convergence only after both scaled feasibility
and primal/dual fixed-point checks pass. PDHG steps use a certified operator-norm
upper bound (the tighter safe dense bound when available), while the power-method
value remains a diagnostic only. Workload gates include normalization,
identifiability sampling, norm estimation, solver, diagnostic, and final
reconstruction passes.
Production-scale point dictionaries remain deferred until a validated NUFFT
operator is available. The current identifiability report samples atom-to-atom
coherence; it is not a proof that component spans are distinguishable and can
miss an omitted duplicate atom. Reviewed dictionaries and stronger
sparsity-/span-aware certification remain mandatory before physical removal
claims.
BPDE is intentionally not exposed as a generic GUI cleanup button. A
target/support/cavity name has no classification power by itself; dictionaries
must be physically justified, distinguishable, and validated against
target-only, contaminant-only, combined, and measured cases.
`isar_repeats.py` similarly defines explicit acquisition IDs/timestamps and a
non-destructive repeat-domain outlier screen for future transient studies. It
does not overload azimuth as slow time and does not delete or label candidates
as birds. Recognized two-way range-phase declarations must agree across sweeps.
Missing, placeholder, and producer-specific unrecognized declarations are
recorded as assumptions and warnings; a definite opposite sign is still rejected.
Repeat loading is preallocated, robust statistics use bounded scratch blocks,
and both stack creation and screening fail before allocation when their
estimated retained result exceeds `maximum_working_bytes` (or the
`GRIM_REPEAT_WORKING_SET_MB` workstation limit). `axis_tolerance` is an
absolute tolerance in the already-matched declared axis units; for example,
datasets declared in GHz receive a GHz tolerance, not an Hz tolerance.
The reusable ISAR preprocessing cache is byte-bounded and synchronized so
independent headless image formations may run concurrently.
Small interpolation geometry plans share an additional 8 MiB LRU cache; blocks
adapt to long axes. Selected-source hashes use bounded blocks while preserving
the exact previous digest, including in-place mutation detection. Memory
diagnostics separate source power/phase arrays, retained earlier band results,
caches, and the additional formation allowance. This is not a total process-RSS
cap; the application, raw source arrays and plotting also consume memory.

### Crop / Slice and Regrid

**Crop / Slice** creates an exact subset without interpolation. It can use the
values selected in the parameter lists or inclusive numeric ranges, optionally
retaining every Nth azimuth, elevation, or frequency sample after the range is
applied. A stride is source-sample selection: it performs no averaging,
low-pass filtering, or anti-aliasing. Polarization can also be sliced. The GUI
shows angular ranges in degrees and frequency in the active dataset's display
unit, then converts the request for each selected dataset.

**Regrid** linearly interpolates one of azimuth, elevation, or frequency onto a
strictly increasing target grid. The GUI's start/stop/step form resolves the
largest grid point that does not exceed stop. Every target coordinate must be
inside the source extent; GRIM does not extrapolate. Cells with usable phase
are interpolated as a complex field, while magnitude-only cells use linear
power interpolation and keep phase unknown. Regridding to a coarser spacing is
not an anti-alias filter. The GUI reports that fact in status and performs the
requested regrid without a second prompt.

### Join / Merge

**Join / Merge…** forms the union of all four axes using existing coordinates,
without interpolation. Its default **Join: reject conflicting overlaps** is
the choice for complementary shards: any conflicting finite overlap stops the
operation. The same dialog offers four explicit conflict-resolution policies.
Input order is significant for the priority policies:

- `priority-first` keeps the first finite power/phase sample as one atomic
  sample; later inputs still fill missing cells.
- `priority-last` keeps the last finite power/phase sample as one atomic sample.
- `power-mean` averages repeated finite samples in linear power. Phase remains
  available in single-source cells and becomes unknown in cells with multiple
  contributors.
- `coherent-mean` averages usable complex fields. Samples with finite power but
  missing phase are masked individually and counted. Missing or conflicting
  phase-reference, time-convention, and polarization-basis annotations are
  recorded as advisories without changing the supplied samples.

The native-axis tolerance defaults to `1e-6`; selected datasets must use the
same storage units. The chosen tolerance is retained in history and recorded
Python commands for strict joins as well as merges. For merge policies, the GUI
reports overlap, missing, contributor, and resolved-conflict counts; the result's
`grim.stitch-provenance.v1` record retains policy, tolerance, counts, metadata
assumptions and input sources. All policies create a new unsaved dataset.

### Extrusion estimates

**Extrusion…** replaces the separate dBke/dBsm conversion buttons. Choose
**3D RCS → 2D width (dBsm → dBke)** or the reverse, and enter extrusion length
in inches, feet or meters. The initial direction follows the first selected
dataset's quantity and can be changed. Incompatible source quantities are
reported as skipped. The calculation and Python recording use length in meters.

The existing model assumes broadside illumination of a uniform extruded body:
`sigma_3D = sigma_2D * 2 * L² / wavelength`. This is an extrusion estimate, not a
general conversion of arbitrary 2D/3D geometries. The dialog shows the formula
for the selected direction. Output history records direction, length and the
geometry assumption; dimensionless power ratios are rejected.

### Phase and azimuth wrapping

**Wrap** can place phase values, azimuth coordinates, or both into `0_360` or
`-180_180`. Phase wrapping is a modulo-360 representation change only: stored
linear power, missing-phase cells, and the physical complex field are
unchanged. The resulting dataset records the choice as `units["phase_wrap"]`;
native `.grim` and versioned flat CSV preserve it. Azimuth wrapping is a
coordinate operation instead: it reorders the grid and may merge physically
equivalent seam aliases, but rejects conflicting finite samples that would
collapse onto the same wrapped coordinate.

## Assembly

Assembly presents **Body**, **Point Features**, **Line Features**, and **Review**.
Choose the body dataset, then use either or both feature tabs to create/edit
placements or load a CSV and its response datasets. The coordinate-unit choice is shared and can be
changed from either feature tab. Body geometry opens when a matching mesh is
needed; mesh options, feature exclusions, and placement tolerances are kept in
collapsed sections. Review contains the live run checklist, output, and placement QA;
the checklist updates whenever an input, mapping, option, or validation result
changes. Whole-response arithmetic and display visibility remain available from
the **Preview layers** button as an advanced secondary window, so a point or
line coupon cannot be confused with a complete platform response. The 3-D
view remains visible beside the workflow and uses the vehicle CAD frame:
`+x` right, `+y` nose, and `+z` up.

The feature form uses the exact strict CSV contracts used by GHOST's local and
unattended/HPC feature workflow; the GUI does not translate another format.
The header is followed directly by data rows—do not add a units row or comment
row. Choose the shared coordinate units in the form. A single point CSV can
contain all point families and a single line CSV can contain every ordered line
chain. The form displays an example and can save either blank template:

```csv
placement_id,dataset_id,x,y,z,nx,ny,nz,roll_x,roll_y,roll_z
```

```csv
line_id,dataset_id,segment_index,x1,y1,z1,x2,y2,z2,n1x,n1y,n1z,n2x,n2y,n2z
```

Selecting **Preview geometry** parses those same CSVs and displays their
locations with the selected STL/facet surface or embedded BoR profile before
an output path or response mapping is required. It is visual QA only.
Under **Review → Feature selection**, **Spatial Feature Configuration → Use** presents the clean body,
point families/placements, and line families/paths as a hierarchy. Unchecking
a family or instance omits it from preview, physical validation, response
loading, and build; disabled-only response families do not need a mapping.
This live selection survives a rescan of the same CSV for IDs that still exist,
while new IDs default enabled and choosing a different CSV resets the choice.
Use **Find feature** to filter by instance ID, dataset ID, or mapped response;
filtering never changes membership. **Copy full selection** records the exact
enabled/disabled configuration even when the on-screen summary is shortened.
Use the named **Reusable assembly recipe** bar to save the body, CSVs, response
mappings, tolerances, validation profile, and exact
membership as one portable
`.assembly.json` trade-study variant. Paths are stored relative to the recipe
when possible. Loading a recipe warns when an input is missing or has changed;
it never silently treats changed bytes as the saved configuration. Loading a
different recipe or closing GRIM with recipe edits presents **Save / Discard /
Cancel**, so a trade-study configuration is not silently lost. Current recipes
use schema version 5. Only the current
recipe format is accepted. Recipes now retain the installed host material,
stack ID, declared minimum principal radius, and exact study samples.
If every feature is unchecked, **Preview geometry** deliberately shows the
clean body alone. Validation and build also accept this body-only baseline,
including a body with no placement CSVs. A body-only build needs no mounting
mesh or host declaration and publishes a zero feature-only sibling.
The selected clean-body `.grim` is preflighted before it receives a ready check.
An embedded BoR response may supply its own preview geometry; an external 3-D
body with enabled features requires its matching STL/facet mesh. A malformed ZIP or incomplete GRIM
key set is shown as unready rather than being counted as a body.
For an external 3-D body, strict validation also requires a reviewed solve-to-mesh
binding. **Bind / refresh...** records the team geometry revision and the
registration evidence ID against the exact clean-body GRIM, mesh bytes, CAD
frame, and selected mesh units. **Check binding** verifies those exact inputs.
The adjacent status clearly reports Missing, Unchecked, Stale, Invalid, or
Current; strict validation remains locked until the current selection has
been checked. Embedded BoR geometry is self-bound and needs no sidecar.
**Validate placements** then checks the body skin, supplied normals, and mapping
completeness. The placement-QA table gives every enabled point and line a
pass/fail-ready record and selects the matching spatial-tree row when clicked.
Warnings remain prominent rather than disappearing into the log. Mesh QA also
reports open edges, nonmanifold edges, duplicate facets, and inconsistent
winding; these are warnings because an intentionally open skin can still be a
valid placement surface, but they deserve review before enabling shadowing.
**Assemble & save** is locked until every required
checklist row is ready and the exact current configuration
has completed Validate placements. It never performs an unreviewed validation
on the way to publication. Metadata and model advisories are visible but need
no waiver in the default profile. Large workload reviews and warnings in an
explicitly selected strict profile still require acknowledgement of the current
plan. The action performs the full response
evaluation and writes the result. Its progress bar covers the
direction/frequency work, and **Cancel
assembly** cooperatively stops before publication. A cancelled or failed build
keeps any existing output and removes its temporary artifact. An unchanged
validated plan is reused at assembly time; any
path, option, or source-file change invalidates it. Prepared base, surface,
placement CSV, and active response bytes are hash-checked again before the
atomic output is published. Existing output replacement
requires confirmation, and output aliases of the clean body or mapped responses
are rejected. Before loading the large response cubes, Assembly estimates peak
RAM and the scratch space needed for its two atomic staging archives. An
oversized job fails early with the requested grid, estimated requirement, and
remedies; a confirmed per-process allocation can be declared with
`GHOST_MAX_SOLVE_GB`. The preview draws locations and
paths, with magenta arrows for supplied outward point/line-endpoint normals.
Lavender point arrows show the roll reference projected perpendicular to the
normal—the solver-effective local `+x`/azimuth-zero direction. Arrow lengths
are normalized and scaled from the non-vector scene extent for display only;
they do not encode vector magnitude or alter validation. Preview Geometry omits
zero or parallel arrows instead of treating the preview as a validation pass;
**Validate placements** reports those errors precisely.
For line paths, the preview can additionally draw signed frame arrows: `+t`
follows the CSV head-to-tail order and `+b = +t × +n` identifies the coupon's
signed across-gap axis. Reversing line order is therefore a physical change for
an asymmetric response, not merely a display change.

The viewer's collapsed **View options** controls can show axis ticks in meters,
inches, or feet without changing the meter-valued CAD data. The body can be
drawn as **Solid**, **Solid + edges**, or **Wireframe**, with adjustable
opacity. **Preview facet detail** limits Matplotlib to 4,000 (Fast), 12,000
(Balanced), or 30,000 (High) sampled display facets. With **Faster rotation**
enabled, a body temporarily uses the Fast proxy while it is dragged and then
returns to the selected detail. The status line reports displayed versus
source facet counts.

Use the always-visible **Preview layers** button to open the tree. Its **Show**
controls and global **Show All** affect preview artists
only. They do not include or exclude a feature from the electromagnetic
assembly; that membership is controlled only by **Spatial Feature
Configuration → Use**. A body mesh used for shadowing is likewise kept at full
physics resolution even if its display is sampled or decimated. Display units, body
style, opacity, facet detail, and faster rotation never reinterpret an input
CSV/STL or modify placement validation, shadowing, or the assembled RCS.

Point datasets require compatible 3-D delta channels (VV, HH, and reciprocal
cross-polarization where used). Line datasets require the TE and TM 2-D delta
responses consumed by line expansion. New assemblies use **General body —
metadata advisory (default)**. Coherent 3-D body responses can come from any
solver. Missing, stale, or conflicting convention annotations, solver versions,
certificates, and manifests do not block the operation. The selected body or
delta role supplies the working frame and assumptions; the program never
conjugates, rephases, or rescales a field merely because an annotation differs.
**Strict library metadata (optional)** checks host material/stack, curvature,
surface bindings, and response certificates. **Require certified GHOST BoR
body** also audits the solver's body-mesh certificate. Saved recipes retain
their explicit profile.

The `.grim` extension identifies a data container, not the physical role of
its contents. Assembly checks dimensional units, numerical power/phase
consistency, polarization channels, and radar axes. A 2-D `sigma_2d`
response cannot be used directly as a 3-D `sigma_3d` body; a suitable 2-D feature
delta belongs in Line Features for expansion along its placement CSV.
Optional manifests describe each
response ID to its installed-minus-clean sign, phase origin, local frame, host
material and optional stack identity, frequency range, footprint, curvature/conical limits,
and validation case IDs. These annotations are retained in provenance;
they become requirements only in an explicitly selected strict profile. Placement skin
distance is displayed in millimeters (stored as meters in recipes); the safe
controls cap phase error at 90 degrees and provide a one-click reset to the
1 mm / 15 degree / 15 degree defaults.

Shadowing is geometric blockage accelerated on the full source mesh; it
does not add diffraction, creeping waves, or body-feature multiple scattering.
Assembly is a coherent first-order reduced model. Production confidence still
requires representative clean/featured full-wave comparisons for each feature
family and the intended host/material/curvature/aspect envelope.

**Create / edit…** opens the point or line placement table. Add, duplicate,
delete, and undo/redo rows; generate point rows/circles, distribute points by
arclength along an open or closed path, or replace a line with
ordered polyline vertices. Repeat the first vertex to close a boundary. Select
rows for **Derive normals** (coordinates preserved) or **Snap + normals**
(coordinates explicitly moved). Inspect the changes, save the CSV, then use
**Preview geometry** and **Validate placements**. Double-click a visible point
or line in the 3-D view to select it; selecting an editor row focuses its saved
preview. Unsaved coordinates are refreshed in 3-D after saving and previewing.
The editor accepts at most 10,000 rows / 16 MiB and keeps bounded undo history.

New embedded-BoR selections enable shadowing and can generate a revolved
shadow surface without an external mesh. Its radial sag is bounded by one
quarter of the active skin tolerance; its azimuthal normal rotation is bounded
by half the normal tolerance. The validation report records the actual mesh
and topology. Explicit recipe shadow choices are retained.

Under Review, **Exact stored study samples** accepts comma-separated frequency,
azimuth, and elevation values; blank retains an entire axis. No interpolation
is performed. Sampling warnings estimate phase changes caused by translated
features; finer body/library samples are still required to resolve their
intrinsic angular or frequency structure.

After saving, **Response comparison** opens body, feature-only, and coherent
total RCS cuts. Select an exact frequency/elevation/channel, or add saved
family-only and configuration variants. Feature RCS is `4π|ΔF|²`; it is not
the dB difference between total and body. The optional right-hand axis shows
that total-minus-body dB comparison separately. Cuts are streamed on a worker with
bounded archive reads. Use the plot toolbar to save a figure. Advanced tree
builds likewise run on a cancellable worker with a capacity check. Their
default is Strict axes; choosing Intersect records discarded samples.

GHOST 2-D `amplitude_version` is provenance, not an eligibility requirement.
Subtraction uses the supplied complex samples even with mixed or absent
versions. It records assumptions and preserves agreed declarations without
claiming that a mixed-version result was produced by the current solver.
An existing Assembly output can also be used in a later study. Known duplicate
feature instances remain rejected; without usable history, prior membership
cannot be checked automatically.

FREDDY nominal IBC CSVs contain `frequency_hz,resistance_ohm,reactance_ohm`;
dielectric CSVs contain `frequency_hz,eps_real,eps_imag,mu_real,mu_imag`.
GHOST reads both with and without the header and converts Hz to its internal
GHz scale. Signed imaginary parts are preserved. Seven-column uncertainty or
other analysis tables are separate artifacts and are not nominal material inputs.

## PPT reports

The **PPT** tab turns loaded GRIM datasets into consistent widescreen
PowerPoint reports. Its dataset check list is independent of the Plotting tab,
so report overlays can be reordered or changed without changing an active plot
or dataset-operation selection. **Use main selection** provides an explicit
one-click handoff when that is desired.

Choose a common polarization and elevation, then one of these fixed layouts.
When both co-polar channels are common, **VV and HH** creates separate VV and
HH plots in the same report instead of requiring a second export:

- **Azimuth — rectangular** or **Azimuth — polar**: one plot for each checked
  frequency, placed left-to-right in a fixed 3-column × 2-row grid. The seventh
  frequency begins at the first position of the next slide; unused positions
  stay empty instead of recentering the plots. GRIM initially checks the first
  six common frequencies and limits one report to 60 frequencies (10 slides),
  so very dense solver sweeps do not make the interface appear frozen.
- **Frequency sweep**: one full-width plot per slide at one elevation and
  polarization. The trace can use one exact common azimuth cut or a selected
  percentile across an inclusive azimuth band. A reversed Min/Max pair crosses
  the periodic seam. Band statistics use the same common stored azimuth samples
  for every overlay, operate in displayed dB units, and do not interpolate.

Selected datasets are overlaid within each plot. GRIM uses exact common fixed
axes and performs no hidden interpolation or extrapolation. Report magnitude
is taken from stored linear RCS power and converted with the dataset's dBsm or
dBke convention. **Shared automatic** vertical scaling is the default and is
calculated once across the complete report. Either axis can instead use one
fixed minimum, maximum, and major-tick step across every plot. Horizontal
settings are retained separately for azimuth degrees and frequency GHz, and
tick settings change only the view—not the dataset samples. Dataset legends
can appear once across the slide header, inside every plot, or not at all; the
master header legend is the default and follows the dataset order above.
Dataset rows can be dragged only to insertion positions; reordering preserves
every row, check state, and stable dataset identity. Slide footer/page furniture
is left to the selected PowerPoint master rather than duplicated by GRIM.
Azimuth-band percentiles are sample-weighted across the finite common stored
angles; the plot title reports the common sample count. Periodic endpoint
aliases such as 0°/360° are counted once, and limits outside the dataset's
stored angular convention are rejected instead of silently reinterpreted.

The report header matches the team slide standard: the title box is 11.82 in ×
0.36 in at X=0.76 in, Y=0.42 in. Plot rows begin at X=0.47 in, Y=1.09 in. The
master legend begins at X=0.76 in, Y=1.05 in and is explicitly layered above
the slightly overlapping plot images. The same title and header alignment is
used for frequency-sweep slides.

**Build Preview** renders the real 16:9 slide geometry used by export. Review
pages with Previous/Next, choose either a fresh blank deck or a widescreen 16:9
`.pptx`/`.potx` template, and then select **Export PPTX**. GRIM includes
`templates/GRIM_Report_Template.pptx` as an editable starting point and selects
it automatically when that file is present. It provides the named custom
layouts **GRIM Azimuth 3x2** and **GRIM Frequency Sweep** under **GRIM Report
Master**. The two layout fields can instead name layouts in a team template;
use `Master :: Layout` when a bare layout name is duplicated across masters.
Leave an individual layout field blank to add that slide family with
PowerPoint's generic blank layout, or clear the template path to create a fresh
deck.

The bundled template's two example slides make alignment easy to inspect and
prototype on a PowerPoint-equipped machine. They are positioning guides: GRIM
removes those seed slides during export after the report slides have inherited
their named layouts. Only styling or graphics placed on the master/layout are
inherited. When a named custom layout supplies a title placeholder, GRIM fills
that placeholder without replacing its master typography or placement;
layouts without one use the documented GRIM title rectangle. Legend and plot
rectangles retain their fixed coordinates. Report any desired coordinate
changes after tuning the examples.

Export writes to a staging file and replaces the requested output only after
PowerPoint succeeds.
An existing output requires explicit replacement confirmation. The preview
shows GRIM content on a white page; a custom template's theme and master
graphics appear only in the exported PPTX, which should be reviewed before use.
GRIM will not close while an export is running. Export closes only the temporary
report presentation created by GRIM; presentations already open in PowerPoint
and PowerPoint's application-level visibility and alert settings are preserved.
GRIM never issues PowerPoint's application-wide **Quit** command.

The slide preview uses GRIM's normal NumPy/Matplotlib/PySide dependencies.
Final export currently requires Windows, desktop Microsoft PowerPoint, and the
optional bridge installed with:

```powershell
py -m pip install -e ".[powerpoint]"
```

## GHOST integration

The GHOST tab loads the authoritative backend from
`tools/GHOST/ghost_backend`. `GHOST_BACKEND_PATH` remains an optional development
override. Solver exports are routed into GRIM's existing dataset loader.

## FREDDY integration

The FREDDY tab loads the authoritative planar-material tool from
`tools/FREDDY`. `FREDDY_ROOT_PATH` is an optional development override. FREDDY
analyzes material stacks and exports GHOST-compatible IBC or material CSV
files; it does not calculate finite-object RCS and does not produce `.grim`
files. Its CSV outputs therefore are not routed into GRIM's RCS dataset table.

FREDDY Inverse Design supports stop/keep/resume, while Material Mix can stop
and discard incomplete results. Ordinary analysis/export jobs run to completion.
GRIM blocks application close while a job runs so the shared process cannot be
torn down partway through a calculation.

FREDDY's searchable **About & Guide** describes every plot family and gives
mode-specific workflows with expected results. **F1** opens workflow help;
**Explain view** in Results opens the plot catalog. Analysis Results include
a sortable reflection margin and **Export comparison CSV…**, preserving the
completed run's context, target, band, bound, and sample identity.

FREDDY's **Material Explorer** is a read-only comparison workspace for measured
permittivity/permeability CSVs. It is available in both embedded and standalone
FREDDY, uses native file frequency grids, and does not alter the solver stack,
project dirty state, or current GHOST attachment.

## HPC scripts

Use GHOST's `run_hpc_monostatic.py` for 2D sweeps and
`run_hpc_bor_monostatic.py` for BoR sweeps. Both live under
`tools/GHOST/ghost_backend/`. Their local counterparts are
`run_local_monostatic.py` and `run_local_bor.py`. Configure geometry inputs,
frequency/angle grids, numerical settings, and resources in the driver's
CONFIG block or validated JSON configuration. A matching saved GHOST
`.run.json` setup can be embedded in that configuration as `run_setup`.

The [HPC guide](../../tools/GHOST/HPC.md) covers submission, scheduling,
resuming interrupted runs, and the bundle CLI for packaging portable inputs
on Windows and staging them on Linux. Use cluster tools to monitor and manage
jobs. Transfer completed `.grim` files locally, then open or drop them into
GRIM to inspect the results.

## Python recorder

The Python tab shows a readable script for successful dataset manipulations,
dataset saves, and supported rectangular/polar azimuth, frequency,
elevation-sweep, and ISAR plot creation/export. PBP, Hold overlays, and other
plot modes are noted in a comment rather than emitted as falsely equivalent
code. Use **Copy** or **Save As…** to run the same work headlessly. The recorder
ignores selection gestures, tab changes, zoom/pan, and non-dataset tool
workflows.

Crop / Slice, Regrid, Stitch, and phase wrapping are replayed with explicit
`crop_dataset`, `regrid_axis`, `stitch_datasets`, and `wrap_phase`/`RcsGrid`
calls. Recorded crop ranges, explicit regrid coordinates, stitch operand order
and policy, metadata assumptions, and wrap interval are therefore visible in
the script. Audit is diagnostic rather than a dataset mutation, so it is not
added to the manipulation chain; use the headless `--audit` report when a
machine-readable replay check is required.

## Dataset files

Files can be dropped onto the main dataset table or the Assembly tree. The
shared loader accepts `.grim`, native flat `.csv`, SENTRi `.csv`/`.txt`, CST
`.csv`/`.cst_data`, theta/phi `.txt`, `.out`, Pioneer `.pio`/`.cmplx_di`,
legacy `.ptm`, and Xpatch `.ss` files. Folder and headless loads use the same
extension registry.

Xpatch `.ss` imports retain the documented GHz frequency values and interpret
each binary signal record as one angular look with frequency-varying
VV/VH/HV/HH complex samples. Saving the imported dataset as `.grim` maps those
records into GRIM's azimuth/elevation/frequency/polarization grid without
transposing the physical axes or applying a frequency-magnitude heuristic. The
available SS header/reference does not establish an absolute RCS normalization,
so these files are deliberately labeled relative `power_ratio`/dB data. They can
be plotted and round-tripped, but PTM/PIO export, range calibration, and coherent
Assembly publication remain blocked until a reviewed conversion establishes
physical `sigma_3d` or `sigma_2d` units.

Generic theta/phi TXT input requires a unit-bearing column header and either an
explicit `frequency_ghz=` argument or a unit-qualified filename such as
`f=10GHz`; headerless column order and unitless filename numbers are not guessed.
Pioneer PIO input likewise requires explicit X/Y axis units, and any explicit
Elevation value must carry ElevationUnits. Explicit XVals/YVals are authoritative;
redundant Start/Stop/Step summaries may be rounded to their written decimal
precision, while materially contradictory summaries are rejected. A closed
full-turn azimuth sweep is
stored half-open: GRIM keeps the opening measurement and removes the repeated
closing row by periodic equivalence, even when the seam is an arbitrary measured
angle such as -178.84 degrees rather than exactly 0/360 or -180/+180. PIO export
applies the same rule. Other descending axes are accepted only when strictly
monotonic and are reversed together with their sample matrix.

`RcsGrid.read_SENTRi()` (also exposed as `GRIM_Backend.scripting.api.read_SENTRi()`) is the
named CREATE-RF SENTRi entry point. It strictly recognizes the two schemas in
the team's `READ_SENTRi.m`: compact MHz `pp/tt/pt/tp` columns and descriptive
Hz `PhiScat/ThetaScat` columns. SENTRi is not treated as CST. Its mapping is
native `elevation=Theta`, and GRIM stores the reported coherent phase with its
original sign. Phi sweeps contained within 0°–180° retain the positive 180°
endpoint, so a 0°–180° import stays in that order in the GUI. Other sweeps use
the signed [-180°, 180°) interval. Closed 0°/360° and -180°/+180° sweeps are
deduplicated at their canonical seam azimuth, with the source 360° or +180° closing record taking
precedence regardless of row order. The four channels map to `VV=tt`, `HV=pt`,
`VH=tp`, and `HH=pp`. Generic unitless theta/phi tables are not guessed to be
SENTRi. The normal two-row export—parameter names followed by an explicit
`Hz`/`MHz`, `deg`, `dBsm`, `deg` units row—is validated and the units row is
excluded from the samples; older header-plus-data exports remain supported.
A recognizable vendor-family header commits dispatch to the SENTRi
reader, while all 11 required columns must be present for the file to load;
damaged/partial SENTRi files therefore fail instead of falling through to a
looser numeric-text reader.

SENTRi exports use `exp(+jwt)`, reference the global coordinate origin, and
remove the outgoing `exp(-jkr)/r` factor. Incoming propagation is opposite the
outward look direction, with unchanged theta/phi polarization vectors. The
importer records these field conventions for Assembly without conjugating
phase or changing polarization signs. Stored RCS remains `sigma_3d`; Assembly
recovers physical amplitude as `sqrt(sigma_3d / (4*pi)) * exp(j*phase)`.
Existing saved imports carrying the recognized SENTRi source, phase, and
polarization mappings can supply missing field declarations during Assembly
without rewriting the input file. Convention annotations are advisory in the
default profile; numerical coordinate conversion remains necessary.
Assembly retains all four SENTRi polarization channels, adding the reciprocal
feature cross-polar contribution to both VH and HV without discarding either
measured body channel.

Import does not silently change SENTRi geometry. Select the loaded dataset and
use **Geometry & Units → SENTRi El→GRIM** when a conventional signed elevation
axis is needed. The exact mapping is `GRIM elevation = 90° - SENTRi Theta`, so
waterline is 0°, top-down is +90°, and bottom-up is -90°. GRIM stable-sorts the
new elevation axis and applies the same permutation to power, phase, and aligned
sample metadata; it performs no interpolation and does not change phase.
The converted dataset is stamped with the Production Assembly radar-coordinate
contract so it can be used directly as a body for line and point feature
placement. Accepted endpoint roundoff is normalized before grid construction,
preventing near-zero/360° seam bins or elevations just outside ±90°.

`RcsGrid.read_CST()` (also exposed as `GRIM_Backend.scripting.api.read_CST()`) is the named
CST entry point. It recognizes both the wide theta/phi export and row-oriented
`Elevation(deg), Azimuth(deg), Frequency(GHz), Polarity, Magnitude(dBsm),
Phase(deg), IQ` data. `load_theta_phi_csv()` remains as a compatibility alias;
native GRIM flat CSV is intentionally a separate format. When IQ parses, its
complex value is authoritative and the magnitude/phase columns are checked as
rounded redundant values; explicit magnitude/phase are used only as fallback
for an opaque vendor IQ token. A full sweep may contain both -180 and +180;
GRIM merges those seam aliases when their complex samples agree and rejects a
conflicting pair. CST headers must state their angular, frequency, RCS, and
phase units explicitly; generic `Abs(field)` tables, radian columns, and
headerless/order-guessed data are rejected rather than interpreted as RCS.

PTM import/export preserves complex IQ and writes one file for each selected
elevation/polarization slice. Use **Export as → PTM (.ptm)…** from the dataset
context menu. The interpreted legacy framing requires a 3-D `sigma_3d` field,
phase, a uniform aspect axis, a positive strictly increasing uniform frequency
axis, at least 37 frequency samples,
and a documented `VV`, `HH`, `VH`, or `HV` polarization. PTM import assumes a
great-circle cut; **Set Coordinates** overrides this assumption when the user
knows the file contains azimuth/elevation data. A grid already tagged
great-circle may carry any of those four
polarizations. Direct export of conic data is limited to unrotated VV/HH at
zero elevation, where GRIM defines signed GC aspect equal to conic azimuth;
conic VH/HV remains rejected because an external PTM's H/V signs are not
specified. GRIM marks files created under this convention as `GRIM_GC_V1` in
the PTM configuration field. An unmarked legacy PTM remains tagged
`legacy_ptm_unspecified`: the **Conic ↔ GC (0°)** tool will not reinterpret
it unless the user explicitly confirms that its aspect sign/origin and V/H
basis match GRIM's convention.

General Conic↔Great-Circle conversion is blocked because a fixed-pitch GC cut
maps to a curved conic path and the full conversion requires interpolation of
complex IQ plus scattering-matrix polarization-basis rotation. The only
lossless conversion offered in both directions is the unrotated, zero-plane
VV/HH relabel under the declared `GRIM_GC_V1` convention. The angular
convention is a physical
compatibility field, so GRIM rejects arithmetic between great-circle PTM data
and ordinary conic data even when their numeric axes happen to match. Stored
PTM roll/tilt values are part of that compatibility check as well; GRIM does
not apply them as rotations because the legacy reference does not define their
Euler semantics. Native GRIM CSV
preserves these fields; Pioneer export is refused for a great-circle grid
because that header cannot represent the distinction. Because no formal PTM
specification or known-good sample accompanied the reference code, byte-level
interoperability with the originating program remains provisional until it is
checked against one real file in each direction.

## Range calibration

The Dataset Operations panel has a **Range Cal** button for complex
substitution calibration. Select one or more measured DUT rows, then choose a
loaded measured calibration target and a loaded trusted complex exact/reference
response in the dialog. For signed offset `ΔR`, positive when the measured
calibrator is farther from radar than the DUT reference plane, GRIM applies

```text
Aout = Adut * Aexact * exp(-j*4*pi*f*ΔR/c) / Ameasured_cal
```

where `|A|² = sigma_3d`. All inputs must contain complex `sigma_3d`/dBsm data
on the same frequency axis. Every DUT polarization must exist in both
references; extra reference channels are ignored. Angular axes must match exactly;
a singleton calibration look may be broadcast only when the user explicitly
enables it. GRIM performs no interpolation, averaging, phase unwrapping, or
automatic range estimation. Calibration nulls, missing complex samples, and
corrections above the selected gain limit are masked per bin and counted; the
operation stops only if no calibratable bins remain. Incompatible axes, units,
quantities, or an explicit opposite phase sign still fail. The calculation runs in GRIM's dataset
worker so large sweeps do not block the GUI, and GRIM will not close while it is
active. Selecting the measured and exact roles expresses the calibration intent.
Unavailable acquisition or phase-center metadata is recorded as assumed rather
than requiring a checkbox. Recalibration is allowed and retains the prior Range
Cal record in provenance.

The exact response is supplied as a dataset so a cylinder, sphere, dihedral,
or another appropriate standard can be used. GHOST's analytical cylinder
reference is an infinite 2-D `sigma_2d` solution and is intentionally not used
as a finite 3-D range-calibration standard. A symmetric cylinder's theoretical
cross-pol response is zero, so use/slice to VV and HH unless the selected
standard supplies a valid nonzero cross-pol reference. Range-calibrated outputs preserve
the complex result and provenance but drop stale solver/certification metadata.

## Support-referenced complex difference

Use **Dataset Operations → Calibration → Support Ref -** when two phase-
coherent acquisitions represent (1) the target on its support and (2) the
support by itself. Select both rows and assign those roles explicitly. That
selection expresses the subtraction intent; missing acquisition declarations
are recorded as assumptions. GRIM then performs exactly

```text
A_difference = A_target_plus_support - A_support_only
```

The operation requires identical axes, units, physical quantities, and coordinate
frames; it never interpolates or regrids. Explicit
metadata conflicts fail closed, including opposite two-way range-phase signs
declared through `range_phase_convention`/`phase_law` aliases. Missing
phase-reference/time/basis/acquisition declarations do not block. GRIM adds the
unsaved result directly, reports finite coverage and QA in status, and stores
before/reference/after energy, closure, and coherence diagnostics in provenance.
The subtraction writes fresh
power/phase arrays in bounded tiles and refuses an unsafe estimated working set
before reading numerical tiles. Direct callers may set `maximum_working_bytes`;
`GRIM_COHERENT_WORKING_SET_MB` sets a process-wide cap. The output is a new
unsaved row with content hashes for both inputs and the result, QA, role labels,
and assumptions in durable `.grim` provenance. The Python tab records the same
`support_referenced_difference(...)` call for headless replay.

This is intentionally called a **support-referenced difference**, not pylon
removal or a free-space target reconstruction. Two-file subtraction cannot
recover target/support coupling, support shadowing, multiple-bounce terms, or
acquisition drift. Those limitations remain even when the algebraic closure
residual is zero.

## Headless interface

```powershell
grim-headless a.grim b.grim --operation coherent-add -o sum.grim
grim-headless --folder results --pattern "*.grim" --operation join -o joined.grim
grim-headless first.grim second.grim --operation stitch --stitch-policy priority-first --tol 1e-6 -o stitched.grim
grim-headless repeated-1.grim repeated-2.grim --operation stitch --stitch-policy coherent-mean -o coherent-mean.grim
grim-headless a.grim b.grim --audit
grim-headless --folder results --pattern "*.grim" --audit -o audit.json
```

Coherent operations require compatible axes, units, polarizations, dimensional
RCS quantity, and at least one common usable complex sample. A 2-D `sigma_2d` field cannot be
coherently added directly to a 3-D `sigma_3d` body; it must first go through
the line-expansion placement workflow. When inputs do not declare a phase
center, time convention, or polarization basis, coherent work uses the
available complex samples, masks unusable cells, and records missing facts
without fabricating values. Conflicting convention annotations are also
advisory; the arithmetic applies no inferred field conversion.
`--attest-coherent-metadata` can record a stronger user statement but is not
required for normal operations.

Headless stitch accepts the same `priority-first`, `priority-last`,
`power-mean`, and `coherent-mean` policies as the GUI. `--tol` controls numeric
coordinate matching and `--max-gib` caps the estimated dense working
allocation. `--audit` takes precedence over combination: it loads and audits
each raw input independently, creates no derived dataset, and writes JSON to
stdout unless `--output` names a separate report file.

The recorder's helpers can also be used directly without Qt:

```python
from GRIM_Backend.scripting.api import load_dataset
from GRIM_Backend.scripting.workspace import crop_dataset, regrid_axis, stitch_datasets, wrap_phase

source = load_dataset("source.grim")
cropped = crop_dataset(source, frequency_range=(8.0, 12.0), frequency_stride=2)
regridded = regrid_axis(cropped, "frequency", start=8.0, stop=12.0, step=0.25)
wrapped = wrap_phase(regridded, mode="0_360")
stitched, report = stitch_datasets(
    wrapped,
    load_dataset("extension.grim"),
    policy="priority-first",
    return_report=True,
)
```

Helper crop/regrid coordinates are in each dataset's native axis units. These
helpers return derived grids in memory; call `.save(...)` explicitly when an
artifact should be published.

Edit-and-run examples for strict folder joins, Cartesian azimuth sweeps,
frequency sweeps with optional azimuth-band percentiles, and unit-aware
coordinate/index queries are in [`examples/`](../examples/README.md). Each script
has a hard-coded settings block near the top and takes no command-line options.
The examples use the same validated loaders and numerical paths as GRIM rather
than duplicating file or plot parsing logic.

## Tests

From the repository root:

```powershell
py -m unittest discover -s GRIM_Backend/tests -p "test*.py" -v
```
