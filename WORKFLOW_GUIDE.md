# Workflow shortcuts

Updated September 7, 2026. These additions use the existing tabs and file formats.
Restart GRIM to load the updated application code.

## FREDDY: analyze every allowed design combination

1. Open **FREDDY → Inverse Design → Setup → Fixed / variable layers…**.
   Choose **Fixed** or **Vary** for each layer. Bulk layers vary thickness in
   inches; sheets vary resistance in Ω/sq. Varying a range requires minimum,
   maximum, and step. Fixed layers retain their current value. Equal minimum
   and maximum define a single value and do not require a step.
2. Values start at Minimum and advance by Step without exceeding Maximum.
   For example, 0.10–0.13 in with a 0.01 in step gives four values:
   0.10, 0.11, 0.12, and 0.13. A maximum between steps is excluded: changing
   that maximum to 0.135 still gives the same four values. The setup summary
   shows the number of values and the actual first and last value for each layer.
3. Choose frequencies, angles, polarization, and tolerance/scoring settings.
   The combination count updates automatically. **Check setup** also checks
   material coverage. The workload is combinations × frequencies × angles ×
   nominal/tolerance cases; plotting adds work. Counts multiply across layers:
   11 choices for each of three layers means 1,331 combinations; for six layers
   it means 1,771,561. The count is not an elapsed-time estimate.
4. Click **Analyze all combinations**. Every configured combination is evaluated
   in a repeatable order, with no random sampling, seed, sample budget, local
   refinement, or short-search stage. **Keep best for comparison** only limits
   the candidates retained for plotting and applying; it does not limit analysis.
   Progress shows completed and total combinations.
5. **Stop and keep best** retains only fully evaluated combinations and clearly
   marks the analysis incomplete. **Resume remaining** finishes the same grid
   without repeating their scores. If all scoring finished but plotting was
   interrupted, Resume finishes the comparison plots. The button is disabled
   after completion. To recover after restarting the application, select an
   optional **Recovery file** before starting. Completed scores are saved at
   combination boundaries about every 30 seconds and when scoring stops.
   Open the matching saved project, **Load checkpoint...**, and **Resume
   remaining** to restore the work. Recovery files do not contain the project
   or material files; keep those alongside your normal study inputs.
   Changes to layers, ranges, targets, polarization, scoring, tolerances, or
   material-file contents require analysis of the new setup.
   Fresh Analyze automatically chooses a new recovery filename when the selected
   file already exists. The earlier checkpoint remains available. A loaded
   checkpoint can also be copied with **Choose / save...** before resuming.
6. Select a candidate and use **Save selected stack…** to create a separate
   FREDDY project with separate output destinations. **Apply Selected** changes
   the active stack. Candidates cannot be applied or saved against changed inputs.

Existing projects load without a format change. Old inverse seed, sample-budget,
and refinement settings are ignored. A formerly continuous variable range needs
an explicit step before running; FREDDY does not silently choose one.

The completed analysis finds the best selected mean-dB score among the configured
grid combinations. It does not cover parameter values between steps. Uncertainty
corners apply systematic scales, and the point percentile summarizes analyzed
angles/corners; these are not manufacturing-yield estimates. No new `.geo` syntax
is needed. This change applies to **Inverse Design**; Material Mix has its own
recipe-search controls.

After a run, **Results** opens automatically within Inverse Design. Setup and
the layer editor remain together on **Setup**; Results uses the full workspace
width. The candidate table starts hidden to maximize plot height on laptops.
Scroll Setup to reach its analysis actions when needed.

### Compare a narrow deep null with a broader, shallower null

1. In Results, set **Target**, initially **−10 dB**, to the reflection level you
   need. Passing means at or below the target. Changing this display target
   does not rerun the solver or change the search objective.
2. **Reflection curves** overlays the first five retained candidates and the
   selected candidate. **Show candidate table** exposes Plot checkboxes for
   choosing other overlays. The selected candidate is always plotted, and its
   passing frequency bands are shaded. Use the toolbar to zoom, pan, or save
   the current plot; hide the table again for a larger plot.
3. **Depth vs bandwidth** compares every retained candidate. Farther right
   means a wider continuous passing band; lower means a deeper null. Click a
   point or use **Selected** to inspect that candidate. A candidate can have
   a slightly shallower null but more useful bandwidth.
4. In the candidate table, sort **Widest band (GHz)** or **Coverage (%)** to
   find broader responses. Coverage is the percentage of the sampled band
   meeting the target; widest band is the largest single connected passing
   interval. Separate narrow bands are not added together as one wide band.
   Deepest reflection, worst point, original search score, and total thickness
   are also shown. Apply/save uses the selected candidate even after sorting.
5. **Analysis history** shows every completed combination's score and the best
   score so far, including completed work preserved by Resume. A partial run
   does not establish the best design on the entire grid. After completion,
   the best score covers the whole configured grid, with ties kept in stable
   grid order. **Run details…** retains the completion report without a popup.

The default **Worst analyzed case** curve uses the highest reflection at each
frequency across all analyzed angles and tolerance corners. **Point percentile**
offers the existing percentile view; lower percentiles are more optimistic.
The curve choice drives the displayed bandwidth and coverage metrics.

Band widths are estimates obtained by linear interpolation in dB between sweep
samples, with no extrapolation. Use a finer sweep to check narrow features.
Discrete frequency lists and single-frequency runs report the fraction of
tested points meeting the target and leave bandwidth unavailable.

These comparisons cover the **retained** candidates. Search ranking still uses
the selected mean-dB objective, which can favor a deep narrow null. Increase
**Keep best** before starting the analysis to inspect more alternatives;
the new charts do not turn the search into a bandwidth optimization.

Two additional inverse views inspect the selected candidate. **Selected candidate
angle map** shows PEC reflection versus frequency and angle using the worst
analyzed tolerance at each point. **Selected candidate tolerance** shows the
nominal response and tolerance envelope at an angle chosen from that run.
These views use saved run labels, not subsequently edited setup values.

## FREDDY: IBC Batch, Thickness, Impedance, and Off Angle results

Each mode now has **Setup / Results** tabs. A successful run opens Results with
a full-width chart. **View** changes the chart; **Show comparison table** exposes
sortable metrics and Plot checkboxes for selecting overlays. The Selected
picker always highlights and plots the chosen thickness or angle. Hide the
table again for more plot height. **Run details…** retains the settings, stack,
and output information; the toolbar saves the current figure with run context.

The run's data and labels stay together when setup is edited. Each mode retains
its own last successful result, so an Impedance run no longer clears Off Angle.
Results and display choices are held for the session; loading another project
clears them. Recompute after reopening a project to populate its Results.

### IBC Batch: compare and choose an exported thickness

1. Configure the batch on Setup and use **Export IBC batch**. The existing
   three-column nominal PEC-backed CSV format and thickness units are unchanged.
2. Compare **Frequency curves**, **Heatmap**, **Bandwidth**, **Band coverage**,
   **Resistance**, **Reactance**, or **Frequency of minimum**. Large batches
   initially overlay up to five spread-out thicknesses; choose others through
   the table. The table and picker tooltips identify each exact output file.
3. Select a row and press **Use selected IBC for GHOST**. This validates that
   the exported file still matches the result before making it the selected
   attachable artifact. Sorting the table does not change the file's identity.
4. Use **Export and attach to current GHOST geometry** to perform the existing
   validated handoff, then save the geometry. A multi-file batch never guesses
   which thickness to attach.

The batch plot cache is bounded by the existing one-million-frequency-row
batch limit. Frequency overlays longer than 5,000 points use extrema-preserving
display reduction; exports and bandwidth metrics retain all computed samples.

### Thickness: locate a broad and tolerant operating region

- **Heatmap** adds a reflection threshold contour. Click to select a thickness
  and frequency; **At selected frequency** displays the corresponding slice.
- **Frequency curves** compares multiple thicknesses. Passing frequency
  intervals for the selected curve are shaded inside the chosen comparison band.
- **Bandwidth** and **Band coverage** compare nominal and worst analyzed
  reflection when tolerance results exist. Bandwidth means the widest single
  connected passing interval; coverage adds all passing frequency intervals.
- **Tolerance envelope** shows the selected thickness's nominal, lower, and
  upper reflection responses. **Frequency of minimum** tracks the sampled null
  location over the full run frequency range; use bandwidth to judge usefulness.
- **Power balance** shows nominal reflected, absorbed, and transmitted incident
  power as percentages. PEC-backed transmission is zero. Select an air-backed
  metric for an air-backed Thickness or Off Angle energy view.

Set **Target** and **Band GHz** to the reflection requirement. The summary lists
passing sampled thickness/angle ranges; the table shows each choice's bandwidth,
coverage, worst reflection in that band, and sampled null location. Ranges group
consecutive passing samples and do not certify untested intermediate thicknesses
or angles. Frequency boundaries are interpolated, without extrapolation.

The Metric selector applies to heatmaps, frequency curves, selected-frequency
slices, and energy views. Bandwidth, coverage, and tolerance views use PEC or
air-backed **reflection**, according to the selected metric family. The
**Upper analyzed bound** is the higher numerical response over the analyzed
tolerance cases; it is the worse case for reflection, not a universal ranking
for phase or absorption. These envelopes are not confidence or yield intervals.
The existing lower-bound and span displays remain available as **Lower analyzed
bound** and **Tolerance span**. Span is upper minus lower; an absolute reflection
threshold is not drawn on that difference. The comparison table uses nominal
reflection while a span view is displayed.

### Off Angle: compare angular performance and polarization

Use the same maps, overlays, bandwidth, coverage, tolerance, and energy views
with incidence angle as the swept coordinate. **Compute TE and TM comparison**
on Setup adds a second polarization calculation and enables matched-scale
side-by-side maps. The checkbox is saved in the project. The selected primary
polarization continues to be written to the existing analysis CSV; the second
polarization is retained in Results and can be plotted or saved as an image.

The directional two-principal-axis model does not support oblique TM. An
otherwise valid TE run still completes; its result explains why the optional
TM comparison is unavailable. Requesting an unsupported primary TM run still
reports the existing physical-model error.

### Impedance and the GHOST coating check

Impedance Results plots the exported **Resistance** and **Reactance**, reflection
curves, tolerance envelopes, and nominal **Power balance** for the selected
backing. Low reflection in an air-backed stack can mean transmission as well
as absorption. Its IBC attachment restrictions remain unchanged.

**Check GHOST coating** now opens **Coating error vs angle** and **Coating error
map** views within Impedance Results. Both compare the scalar IBC with the full
planar stack for TE/TM using absolute complex-reflection difference. The maps
share an error scale. Run details keeps the approximation report, including its
midpoint interpolation check. This analysis neither exports an IBC nor certifies
finite-body RCS accuracy, and it does not overwrite an existing impedance run.

## GHOST: reuse a 2D setup

**Boundary Densities** computes VV/HH at the first frequency and incident angle
and opens their magnitude on the geometry in GHOST's existing plot area.
Use **Result view** to inspect phase or return to the last RCS plot. The table
shows each element's coordinates and complex density values; the plot toolbar
can save an image. This action does not require a JSON output file. These are
formulation-specific SLP/DLP representation densities, with magnitude units
that depend on the formulation. Geometry or material edits during calculation
discard the out-of-date result; cancellation keeps previous results intact.

1. In **GHOST → Solver**, choose the geometry and its units. The dimensions beside
   the units show the X/Y spans in meters and inches.
2. Set the mesh-convergence checkbox, accuracy target, and LU precision directly.
   Mesh convergence compares base/fine results; disabling it runs one base mesh
   without a convergence certificate. Frequency and angle samples stay as entered.
3. Use **Check geometry and run setup**. The background check summarizes
   dimensions, geometry/material errors and warnings, sample/channel counts,
   quality settings, and the output destination. It includes thin-layer
   applicability checks. The normal solve repeats preflight before computation;
   numerical quality and convergence are determined by the actual run.
4. **Save run setup…** writes a `.run.json` file. **Load run setup…** in GHOST
   restores frequencies, incident angles, units, mesh certification, accuracy,
   LU precision, scattering mode, quality thresholds, and execution resources.
5. Select geometry and output locations separately. For local/HPC batches,
   embed a saved setup as `run_setup` in the driver JSON and configure input
   paths, output destinations, and cluster resources in the driver settings.
   See [saved profiles](tools/GHOST/RUN_PROFILES.md) and the
   [HPC guide](tools/GHOST/HPC.md).

The 2D batch drivers require monostatic setups with their standard quality
thresholds. GHOST also saves and restores bistatic and custom-threshold setups.
Conflicting or incompatible driver settings are rejected before a run starts.

## GHOST: reuse a BOR setup

Select BOR, open Advanced Settings, and choose the BOR factorization, aspect
batch size, incident-basis reuse, compressed storage cap, tile size, and tile
cache. Save or load these with **Save run setup / Load run setup**. The versioned
BOR recipe is distinct from the existing 2D recipe and preserves CFIE alpha,
frequencies, units, mesh certification, and accuracy as well.

GHOST's angles are body aspects from +z, between 0 and 180 degrees. Local/HPC
BoR drivers use radar azimuth/elevation plus body attitude. A saved GHOST
recipe embedded as `run_setup` maps to azimuth 0, elevation `90 - aspect`,
a +z body axis, and zero roll. The drivers export radar coordinates.

Existing recipes containing a radar grid and body attitude remain loadable
in GHOST: the solver uses their derived body aspects and plots in body
coordinates. The load notice describes this conversion. Geometry and output
paths are configured separately. Conflicting explicit driver settings are
rejected.

In GHOST, **Check geometry and run setup** validates the BOR geometry and material
coverage over the requested frequencies and forecasts peak allocation. This
does not assemble operators or certify the future solution. Running still
checks linear-system, modal-tail, and selected mesh-convergence criteria.

## ISAR: plan, form, inspect, and compare

1. Select a stationary far-field monostatic complex dataset, one polarization,
   one elevation, the frequency band and an angular sector. Frequency controls
   show the dataset's units. Regular angular strides are supported.
2. Use existing **Dataset Operations → Range Cal** for complex reference
   calibration, and **Support Ref −** for matched support-referenced differences
   when appropriate. These operations retain provenance; subtraction does not
   undo coupling or shadowing. **Workflow** in ISAR connects these steps.
3. In **ISAR Settings**, enter the occupied scene half extents in metres and
   choose **Recommended PFA**. Set **Image mode** explicitly when a coherent
   image or qualitative composite is required; automatic retains the legacy
   switch above 20 degrees. Use **Plan image** to inspect native sampling,
   curvature and memory estimates before **Apply ISAR Settings**.
4. Review **Quality**. Coverage, nominal resolution, origin PSF cuts and sparse
   native-data residuals answer different questions. A converged sparse solver
   can still disagree with the original polar measurements. **Cancel** stops
   formation at a processing block. Extra pixels and scene crops add no physical
   resolution.
5. **Export ISAR Result** saves full numerical arrays and diagnostics. **Save
   recipe** saves the accepted physical selectors and formation settings. Load
   the recipe against another compatible acquisition, inspect the plan, and
   form again. Display changes preserve the numerical result; numerical changes
   require a fresh formation before export.
6. **Open result** works without the original source dataset. **Compare result**
   offers linked physical axes and shared intensity scales, an A-minus-B map,
   **Peak profiles**, and **Statistics in current view** for the zoomed difference
   ROI. Different grids require explicit intensity resampling; incompatible
   image frames or acquisition conventions are explained. These are image-level
   dB differences, without a per-pixel dBsm/dBke calibration claim.

## Assembly: map responses and make variants

1. Read the point/line placement CSVs to populate their dataset IDs.
2. In either mapping table, choose **Suggest files from library folder…**.
   Suggestions use exact, case-insensitive filename stems—for example,
   `fastener` matches `fastener.grim`—and search subfolders.
3. Review the proposed mappings. Unique matches are preselected; duplicate
   matches remain unmapped until you choose a file. Existing mappings are kept.
   Filename matching does not establish physical compatibility: use the normal
   validation before building.
4. In **Review**, expand the readiness checklist. Double-click a requirement or
   use **Go to next required step** to return to its controls.
5. Expand **Reusable recipe → Create variant…**. Enter a variant name and a new
   recipe filename. The current inputs, settings, and enabled/disabled features
   are copied, and a separate total-response destination is assigned. Existing
   recipe files are protected. Adjust the variant's enabled features, then
   validate and build it.
6. Completed builds continue to open the existing **Response comparison** with
   the body, feature-only delta when available, and coherent total.

Variants keep their source responses and placement files as references. Edit
copies of source files when their contents should differ between variants.

## PPT: save a report definition and reuse a plot setup

The **Reusable report setup** group is below the slide text/template controls.

1. Configure the datasets and order, plot type, cuts, polarization, axes, legend,
   styles, slide title, template, and named layouts. **Save recipe…** writes a
   named `.report.json` file. The presentation output filename remains a separate
   choice, so loading a recipe does not redirect the export to an old output.
2. **Load recipe…** restores the definition. Saved datasets are matched by
   current identity, source path, or an unambiguous name. To apply it to new
   data, select the new datasets in PPT and check **Use current PPT datasets
   when loading**. Individual styles follow selected-series order in that mode.
3. Missing or ambiguous datasets, unavailable cuts/channels, invalid settings,
   and incompatible physical units/angle conventions are reported. A failed
   load restores the preceding setup. Degree/radian and supported frequency-unit
   conversions preserve the physical cuts.
4. **Use current Plotting setup** copies the last successful supported Plotting
   view's datasets, cuts, visible axes, legend, and consistent dataset line
   widths/dashes/colors. The PPT title/template remain as configured. Reports
   support rectangular/polar azimuth, elevation, and frequency magnitude plots
   in dB. The tool explains incompatible selections instead of substituting a
   different view: fixed cuts must be singular, frequency reports use the full
   frequency axis, and polar reports use North as zero. Hold, PBP, phase, linear
   magnitude, image plots, and differing styles for one dataset require their
   own export or an adjusted Plotting selection.
5. **Check PowerPoint and template** checks widescreen format, availability, and
   actual master/layout names using a read-only, untitled template copy. It
   closes its own copy without closing the user's PowerPoint application.
6. **Build preview**, review the slides, then **Export PowerPoint**. After a
   successful export, **Open exported presentation** opens the saved file.

The report recipe is a reusable definition, not a screenshot of the canvas;
slides retain GRIM's existing uniform report layout. Export still requires
desktop PowerPoint and its COM integration. Tests exercise the export bridge;
no user presentation or cluster job was created during workflow verification.
