# GRIM RCS visualization and dataset tools

GRIM reads, plots, compares, transforms, and combines gridded RCS datasets. The
GUI and headless interface use the same `RcsGrid` physical data model.

## Install and run

```bash
python -m pip install -e .
grim
```

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
