# GRIM application

GRIM is the host application for this distribution. Its desktop tabs are
**Plotting | ISAR | PPT | Assembly | GHOST | FREDDY | Runs | Python**. GHOST and FREDDY remain
self-contained tools under `tools/`; GRIM embeds their authoritative user
interfaces instead of copying their numerical implementations into the
plotting code.

Run GRIM from the repository root after an editable installation:

```powershell
py -m pip install -e .
grim
```

## Assembly

Assembly keeps spatial feature placement under **Place Features** and
whole-response arithmetic under **Datasets + Preview Layers**, so a point
or line coupon cannot be confused with a complete platform response. The 3-D
view remains visible beside both workflows and uses the vehicle CAD frame:
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
After parsing, **Spatial Feature Configuration → Use** presents the clean body,
point families/placements, and line families/paths as a hierarchy. Unchecking
a family or instance omits it from preview, physical validation, response
loading, and build; disabled-only response families do not need a mapping.
This live selection survives a rescan of the same CSV for IDs that still exist,
while new IDs default enabled and choosing a different CSV resets the choice.
Use **Find feature** to filter by instance ID, dataset ID, or mapped response;
filtering never changes membership. **Copy full selection** records the exact
enabled/disabled configuration even when the on-screen summary is shortened.
If every feature is unchecked, **Preview geometry** deliberately shows the
clean body alone; validation and build still require at least one enabled
feature.
**Validate placements** then checks the body skin, supplied normals, and mapping
completeness. **Assemble & save** performs the full response evaluation and
writes the result. An unchanged validated plan is reused at assembly time; any
path, option, or source-file change invalidates it. Prepared base, surface,
placement CSV, and active response bytes are hash-checked again before the
atomic output is published. Existing output replacement
requires confirmation, and output aliases of the clean body or mapped responses
are rejected. The preview draws locations and
paths, with magenta arrows for supplied outward point/line-endpoint normals.
Lavender point arrows show the roll reference projected perpendicular to the
normal—the solver-effective local `+x`/azimuth-zero direction. Arrow lengths
are normalized and scaled from the non-vector scene extent for display only;
they do not encode vector magnitude or alter validation. Preview Geometry omits
zero or parallel arrows instead of treating the preview as a validation pass;
**Validate placements** reports those errors precisely.

The viewer's **3-D display only** controls can show axis ticks in meters,
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
responses consumed by line expansion. Shadowing is geometric blockage; it
does not add diffraction, creeping waves, or body-feature multiple scattering.

## PPT reports

The **PPT** tab turns loaded GRIM datasets into consistent widescreen
PowerPoint reports. Its dataset check list is independent of the Plotting tab,
so report overlays can be reordered or changed without changing an active plot
or dataset-operation selection. **Use main selection** provides an explicit
one-click handoff when that is desired.

Choose a common polarization and elevation, then one of these fixed layouts:

- **Azimuth — rectangular** or **Azimuth — polar**: one plot for each checked
  frequency, placed left-to-right in a fixed 3-column × 2-row grid. The seventh
  frequency begins at the first position of the next slide; unused positions
  stay empty instead of recentering the plots. GRIM initially checks the first
  six common frequencies and limits one report to 60 frequencies (10 slides),
  so very dense solver sweeps do not make the interface appear frozen.
- **Frequency sweep**: one full-width plot per slide at one exact common
  azimuth/elevation/polarization cut.

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

The report header matches the team slide standard: the title box is 11.82 in ×
0.36 in at X=0.76 in, Y=0.42 in. Plot rows begin at X=0.47 in, Y=1.09 in, with
the master legend placed between the title and plots. The same title and header
alignment is used for frequency-sweep slides.

**Build Preview** renders the real 16:9 slide geometry used by export. Review
pages with Previous/Next, choose either a fresh blank deck or an optional blank
widescreen 16:9 `.pptx`/`.potx` template, and then select **Export PPTX**. Export writes to a
staging file and replaces the requested output only after PowerPoint succeeds.
An existing output requires explicit replacement confirmation. The preview
shows GRIM content on a white page; a custom template's theme and master
graphics appear only in the exported PPTX, which should be reviewed before use.
GRIM will not close while an export is running.

The slide preview uses GRIM's normal NumPy/Matplotlib/PySide dependencies.
Final export currently requires Windows, desktop Microsoft PowerPoint, and the
optional bridge installed with:

```powershell
py -m pip install -e ".[powerpoint]"
```

## GHOST integration

The GHOST tab loads the authoritative backend from
`tools/GHOST/Backend`. `GHOST_BACKEND_PATH` remains an optional development
override. Solver exports are routed into GRIM's existing dataset loader.

## FREDDY integration

The FREDDY tab loads the authoritative planar-material tool from
`tools/FREDDY`. `FREDDY_ROOT_PATH` is an optional development override. FREDDY
analyzes material stacks and exports GHOST-compatible IBC or material CSV
files; it does not calculate finite-object RCS and does not produce `.grim`
files. Its CSV outputs therefore are not routed into GRIM's RCS dataset table.

FREDDY background calculations are not cancellable. GRIM blocks application
close while one is running so the shared process cannot be torn down partway
through a calculation.

FREDDY's **Material Explorer** is a read-only comparison workspace for measured
permittivity/permeability CSVs. It is available in both embedded and standalone
FREDDY, uses native file frequency grids, and does not alter the solver stack,
project dirty state, or current GHOST attachment.

## HPC Runs

The **Runs** tab builds GHOST 2-D or BoR sweep requests from saved `.geo`
files, their FRD/OPN/BoR roles, frequency and angle grids, geometry units,
mesh-certification choice, and SLURM resources. **Export Bundle** writes a
portable, hash-verified folder with relative geometry/material inputs and a
one-command Linux README. It is a request—not a Windows-created solver run.

**Upload & Submit** builds a fresh temporary copy of the visible request,
uploads it through Windows OpenSSH or a saved PuTTY/Plink session, and invokes
the matching `tools/GHOST/Backend/hpc_bundle.py` on the Linux login node. Linux
then creates the configured driver, absolute paths, runtime/source provenance,
run manifest, schedule, and SLURM submission. The returned job IDs are tracked
in the tab; **Refresh**, **Cancel Job**, and **Download Results** reconnect only
for that operation. Submitted SLURM jobs continue after SSH disconnects and do
not require GRIM to remain open.

GRIM records the bundle ID and expected remote `stage_result.json` before the
upload begins. If SSH drops while `sbatch` is running, the run is marked
**SUBMISSION UNKNOWN** rather than submitted again; use **Refresh** to recover
the stage result and any job IDs. Refresh also shows recent submission and
SLURM task logs. Results become downloadable only after SLURM reports a
terminal state, and GRIM refuses to merge them into an existing local
`results` folder. The **Remote Python** field defaults to `python3`; point it at
the cluster virtual-environment interpreter when the default environment does
not contain GHOST's dependencies.

GRIM stores non-secret connection metadata such as host/profile, username,
port, remote paths, and identity-file path. It never stores a password,
passphrase, or private-key contents. Unknown host keys fail closed; verify the
fingerprint through an approved SSH/PuTTY connection first. Password-only,
interactive MFA, VPN, jump-host, or site-policy restrictions may require an
OpenSSH config alias, an approved agent/session, or the manual bundle workflow.
See `tools/GHOST/HPC.md` for Linux staging and scheduler details.

If PuTTY reports **cannot answer interactive prompts in batch mode**, load the
exact saved session named in Runs and first expose the unanswered prompt from
PowerShell:

```powershell
plink.exe -v -T -load "Exact Saved Session Name" "echo GRIM_HPC_OK; hostname; id -un"
```

Save the username under **Connection > Data > Auto-login username**, accept a
host key only after verifying its fingerprint, and use an approved key already
unlocked in Pageant. Then verify the same path GRIM uses:

```powershell
plink.exe -batch -T -load "Exact Saved Session Name" "echo GRIM_HPC_OK; hostname; id -un"
```

If the site requires password or MFA entry on every new connection, enable
**Connection > SSH > Share SSH connections if possible** in that saved session,
open and authenticate the PuTTY session, and leave it open while GRIM runs.
`plink.exe -shareexists -load "Exact Saved Session Name"` returns exit code zero
when that upstream is reusable. If policy forbids either keys or connection
sharing, use **Export Bundle** and submit through the approved interactive path;
do not place a password on a Plink command line.

## Python recorder

The Python tab shows a readable script for successful dataset manipulations,
dataset saves, and supported rectangular/polar azimuth, frequency, and
elevation-sweep plot creation/export. PBP, Hold overlays, and other plot modes
are noted in a comment rather than emitted as falsely equivalent code. Use **Copy** or
**Save As…** to run the same work headlessly. The recorder ignores selection
gestures, tab changes, zoom/pan, and non-dataset tool workflows.

## Dataset files

Files can be dropped onto the main dataset table or the Assembly tree. The
shared loader accepts `.grim`, native flat `.csv`, SENTRi `.csv`/`.txt`, CST
`.csv`/`.cst_data`, theta/phi `.txt`, `.out`, Pioneer `.pio`/`.cmplx_di`,
legacy `.ptm`, and Xpatch `.ss` files. Folder and headless loads use the same
extension registry.

`RcsGrid.read_SENTRi()` (also exposed as `grim_headless.read_SENTRi()`) is the
named CREATE-RF SENTRi entry point. It strictly recognizes the two schemas in
the team's `READ_SENTRi.m`: compact MHz `pp/tt/pt/tp` columns and descriptive
Hz `PhiScat/ThetaScat` columns. SENTRi is not treated as CST. Its mapping is
`elevation=Theta`, and GRIM stores the reported coherent phase with its
original sign. The four channels map to `VV=tt`, `HV=pt`,
`VH=tp`, and `HH=pp`. Generic unitless theta/phi tables are not guessed to be
SENTRi. The normal two-row export—parameter names followed by an explicit
`Hz`/`MHz`, `deg`, `dBsm`, `deg` units row—is validated and the units row is
excluded from the samples; older header-plus-data exports remain supported.
A recognizable vendor-family header commits dispatch to the SENTRi
reader, while all 11 required columns must be present for the file to load;
damaged/partial SENTRi files therefore fail instead of falling through to a
looser numeric-text reader.

`RcsGrid.read_CST()` (also exposed as `grim_headless.read_CST()`) is the named
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
and a documented `VV`, `HH`, `VH`, or `HV` polarization. PTM is a great-circle
cut format. A grid already tagged great-circle may carry any of those four
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
automatic range estimation. Calibration nulls, missing phase, and incompatible
quantities fail closed. A user-visible maximum correction-gain gate catches
near-null/noise-floor calibration bins. The calculation runs in GRIM's dataset
worker so large sweeps do not block the GUI, and GRIM will not close while it is
active. The operation assumes additive background/support
scattering was already removed and that the DUT and measured calibrator share
one acquisition chain. The user confirms those assumptions in the dialog.

The exact response is supplied as a dataset so a cylinder, sphere, dihedral,
or another appropriate standard can be used. GHOST's analytical cylinder
reference is an infinite 2-D `sigma_2d` solution and is intentionally not used
as a finite 3-D range-calibration standard. A symmetric cylinder's theoretical
cross-pol response is zero, so use/slice to VV and HH unless the selected
standard supplies a valid nonzero cross-pol reference. Range-calibrated outputs preserve
the complex result and provenance but drop stale solver/certification metadata.

## Headless interface

```powershell
grim-headless a.grim b.grim --operation coherent-add -o sum.grim
grim-headless --folder results --pattern "*.grim" --operation join -o joined.grim
```

Coherent operations require compatible axes, units, phase reference,
polarizations, and dimensional RCS quantity. A 2-D `sigma_2d` field cannot be
coherently added directly to a 3-D `sigma_3d` body; it must first go through
the line-expansion placement workflow.

## Tests

From the repository root:

```powershell
py -m unittest discover -s GRIM_Revised_2 -p "test*.py" -v
```
