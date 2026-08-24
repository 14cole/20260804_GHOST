# GRIM application

GRIM is the host application for this distribution. Its desktop tabs are
**Plotting | ISAR | Assembly | GHOST | FREDDY**. GHOST and FREDDY remain
self-contained tools under `tools/`; GRIM embeds their authoritative user
interfaces instead of copying their numerical implementations into the
plotting code.

Run GRIM from the repository root after an editable installation:

```powershell
py -m pip install -e .
grim
```

## Assembly

Assembly has one canonical tree and one 3-D preview. Point and line feature
CSV files are selected in the form; each discovered `dataset_id` receives one
explicit OPN-FRD `.grim` mapping. The preview uses the vehicle CAD frame:
`+x` right, `+y` nose, and `+z` up.

The tree's **Show** controls and global **Show All** affect preview artists
only. They do not include or exclude a feature from the electromagnetic
assembly. A body mesh used for shadowing is likewise kept at full physics
resolution even if its display is decimated.

Point datasets require compatible 3-D delta channels (VV, HH, and reciprocal
cross-polarization where used). Line datasets require the TE and TM 2-D delta
responses consumed by line expansion. Shadowing is geometric blockage; it
does not add diffraction, creeping waves, or body-feature multiple scattering.

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

## Dataset files

Files can be dropped onto the main dataset table or the Assembly tree. The
shared loader accepts `.grim`, native flat `.csv`, SENTRi `.csv`/`.txt`, CST
`.csv`/`.cst_data`, theta/phi `.txt`, `.out`, Pioneer `.pio`/`.cmplx_di`,
legacy `.ptm`, and Xpatch `.ss` files. Folder and headless loads use the same
extension registry.

`RcsGrid.read_SENTRi()` (also exposed as `grim_headless.read_SENTRi()`) is the
named CREATE-RF SENTRi entry point. It strictly recognizes the two schemas in
the team's `READ_SENTRi.m`: compact MHz `pp/tt/pt/tp` columns and descriptive
Hz `PhiScat/ThetaScat` columns. SENTRi is not treated as CST. Its vendor mapping
is `elevation=Theta-90`, and GRIM stores the coherent phase with the negative
of the reported E-field phase. The four channels map to `VV=tt`, `HV=pt`,
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
