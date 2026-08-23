# GRIM RCS visualization and dataset tools

GRIM reads, plots, compares, transforms, and combines gridded RCS datasets. The
GUI and headless interface use the same `RcsGrid` physical data model.

## Install and run

```bash
python -m pip install -e .
grim
```

The unified desktop window is organized as **Plotting | ISAR | Assembly |
GHOST**. Plotting and ISAR share the loaded-dataset table. Assembly owns the
one canonical Assembly tree and its 3-D preview; there are no separate hidden
trees in the plot tabs. GHOST embeds the existing 2-D Geometry and Solver tabs
in the same application. It still calls the same GHOST numerical backend as
the standalone solver, and a solver export is automatically loaded into
GRIM's dataset table.

## Assembly and feature placement

Use the **Assembly** tab for point scatterers and line-expanded features:

1. Choose the clean base monostatic `.grim`.
2. For an external 3-D body, choose its matching STL or indexed ASCII
   `.facet` surface and declare its units. A self-contained GHOST BoR result
   may instead preview its embedded revolved profile. A triangle surface is
   still required if geometric shadowing is enabled.
3. Choose the exact point-placement and/or line-placement CSV. The form reads
   each CSV's `dataset_id` values and creates the required dataset mappings;
   select the corresponding OPN-FRD `.grim` for every ID.
4. Click **Validate & Preview**. The 3-D view uses the vehicle CAD frame:
   `+x` right, `+y` nose, and `+z` up.
5. Inspect the body, point groups, and line groups, then click
   **Assemble & Save** to write and load the combined result.

The tree's per-node **Show** boxes and global **Show All** box control only the
3-D preview. They never include or exclude a response from the mathematical
assembly and never alter feature-placement physics. Likewise, a display mesh
or decimated preview is not substituted for the full surface used for skin,
normal, or shadowing checks.

Point patterns must be coherent OPN-FRD (`featured - clean`) 3-D deltas with
VV, HH, and reciprocal VH/HV response. Line patterns must be coherent OPN-FRD
2-D deltas containing both TE and TM. Shadowing is a geometric ray-blockage
mask; it does not add diffraction, creeping waves, or body-feature multiple
scattering.

Do not place a raw 2-D solver result under a coherent-sum Assembly branch as
though it were an already positioned 3-D field. Its `sigma_2d` response has to
be expanded along the line CSV before it can contribute to a `sigma_3d` body.
The file may be loaded into GRIM for inspection and organized as a response
dataset, while the feature form supplies its placement semantics. The ordinary
coherent/incoherent Assembly branches remain for commensurate response
datasets that already share the required physical quantity, axes, units, and
phase convention.

The Qt-free command-line tool can join or add explicit files:

```bash
grim-headless a.grim b.grim --operation coherent-add -o sum.grim
grim-headless --folder results --pattern '*.grim' --operation join \
  --overlap error --max-gib 32 -o joined.grim
```

The equivalent Python API is:

```python
from grim_headless import load_dataset, load_folder, combine_datasets

a = load_dataset("a.grim")
b = load_dataset("b.grim")
coherent_sum = combine_datasets([a, b], "coherent-add")
joined = load_folder("results", pattern="*.grim", operation="join")
```

Folder processing is deterministic by pathname. It defaults to one loader
worker because large `.grim` archives are memory intensive; parallel loading
must be requested explicitly.

## PowerPoint Image Imprinter

The Windows-only helper copies picture location, size, and crop formatting
from pictures on one slide to pictures selected on another slide. Install the
PowerPoint automation dependency and launch it alongside desktop PowerPoint:

```powershell
py -m pip install -e ".[powerpoint]"
ppt-image-imprinter
```

After installation, you can also double-click
`Launch_PowerPoint_Image_Imprinter.bat` in the repository's top-level folder.
The launcher reports a clear installation command if Python, PySide6, or
pywin32 is unavailable.

Use **Capture selected** after selecting the source picture or pictures in
PowerPoint. Move to the destination slide, select its pictures, choose whether
to apply Location, Size, and/or Crop, and click **Apply to selected**. One
captured profile is broadcast to every destination. Multiple profiles require
the same number of destinations and pair in PowerPoint's selected-shape order.
Grouped pictures are supported; other selected shape types are skipped and
reported. If PowerPoint rejects any update, the helper attempts to restore the
entire destination selection to its pre-apply formatting.

Crop margins are copied using PowerPoint's native point values, which are
relative to each picture's original dimensions. Applying a crop to a different
underlying image can therefore produce a different visible composition even
though the PowerPoint crop values match.

## ISAR imaging

GRIM offers three phase-aware reconstruction paths:

- **Fast PFA** applies frequency-dependent keystone correction followed by a
  2-D FFT. It is intended for rapid aperture scrubbing and narrow looks.
- **Accurate PFA** grids the measured polar phase history onto an inscribed
  Cartesian wavenumber grid before the FFT. It removes the residual range
  curvature that can defocus scatterers away from the phase centre.
- **Sparse L1** uses a masked FISTA solve for sidelobe suppression. Missing
  phase samples are excluded from the data-fidelity objective rather than
  being treated as measured zeros.

Taper coherent gain and gridding coverage are normalized, so a unit point
target remains at 0 dB when the window changes. At nonzero elevation, axes are
reported in the horizontal 2-D target plane and include the cosine projection.
The status bar reports phase coverage, nominal resolution, and unambiguous
scene extents. Apertures crossing 0°/360° are circularly unwrapped.

Post-gridding phase histories are cached with a bounded default of 512 MiB;
set `GRIM_ISAR_CACHE_MB=0` to disable it or choose another limit. SciPy FFTs
use all available workers by default; set `GRIM_FFT_WORKERS=1` (or another
positive count) when running multiple GRIM processes concurrently.

The same full-resolution formation path is available without Qt or display
decimation:

```python
from grim_headless import form_isar, load_dataset

grid = load_dataset("target.grim")
bands, elapsed = form_isar(grid, reconstruction="accurate", window="Hanning")
image = bands[0]["magnitude"]
x = bands[0]["x_range"]
y = bands[0]["y_range"]
```

## Solver interchange contract

Current GHOST solver outputs are directly compatible. GRIM preserves these
fields:

- axes: `azimuths`, `elevations`, `frequencies`, `polarizations`;
- physical data: `rcs_power`, `rcs_phase`;
- units: angle/frequency units, `rcs_log_unit`, and
  `rcs_linear_quantity` (`sigma_2d` or `sigma_3d`);
- authoritative solver field, when present: `rcs_amp_real` and
  `rcs_amp_imag` as float64;
- phase convention metadata such as `phase_reference`.

For 2-D outputs, `rcs_power = |B|²/(4k)` and the display unit is absolute
dBke. For 3-D outputs, `rcs_power = 4π|F|²` and the display unit is dBsm. GRIM
normalizes either raw field to `sqrt(rcs_power)·exp(j phase)` for coherent
operations, so matching solver files add correctly while incompatible 2-D/3-D
quantities are rejected.

Coherent operations require finite phase and matching axes, units, physical
quantity, polarization ordering, and phase reference. Incoherent operations
work on physical power. A dB difference or coherent division is stored as a
dimensionless `power_ratio` with logarithmic unit `dB`, never as dBsm/dBke.

`.grim` loading disables pickle by default. A trusted legacy object-array file
can be migrated explicitly with `RcsGrid.load(path, allow_legacy_pickle=True)`.

## Supported inputs

- `.grim` solver/viewer archives
- solver `.out`
- Xpatch `.ss`
- Pioneer `.pio` / `.cmplx_di`
- GRIM flat CSV/TSV
- supported theta/phi CSV and TXT exports

Run the regression suite with:

```bash
python -m unittest discover -s GRIM_Revised_2 -p 'test*.py' -v
python GRIM_Revised_2/test_ss.py
```
