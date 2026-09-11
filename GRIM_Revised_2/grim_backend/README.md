# GRIM backend

Dataset processing and file I/O live in this package. GUI widgets and workspace
controllers live in `GRIM_Revised_2`; numerical GHOST and FREDDY code lives in
`tools/`.

## Find a task

| Task | Module | Entry points |
| --- | --- | --- |
| Load a file or folder | `io/loaders.py` | `load_dataset`, `load_folder`, `is_supported_path` |
| Read or write `.grim` | `io/native.py` | `RcsGrid.load`, `RcsGrid.save`, payload validation |
| Read or write flat CSV | `io/csv.py` | `load_flat_csv`, `write_flat_csv`, schema fields |
| Import CST | `io/cst.py` | `RcsGrid.read_CST`, theta/phi table parsing |
| Import SENTRi | `io/sentri.py` | `RcsGrid.read_SENTRi`, header detection |
| Import OUT | `io/out.py` | `RcsGrid.load_out` |
| Import/export Pioneer | `io/pioneer.py` | `RcsGrid.load_pio`, `RcsGrid.save_pio` |
| Import/export PTM | `io/ptm.py` | `read_ptm`, `write_ptm`, `RcsGrid.load_ptm`, `RcsGrid.save_ptm` |
| Import Xpatch SS | `io/xpatch.py` | `read_ss`, `RcsGrid.load_ss` |
| Save/export batches | `io/batch.py` | `save_dataset_batch`, staging, compression, rollback |
| Construct a grid or read samples | `datasets/grid.py` | `RcsGrid`, `get`, `get_by_value`, `rcs_slice` |
| Crop, sort, align, interpolate, wrap axes | `datasets/axes.py` | `axis_crop`, `align_to`, `interpolate_axis`, `wrap_azimuth` |
| Change coordinate systems | `datasets/coordinates.py` | Wedge/conic, great-circle, and SENTRi transforms |
| Add/subtract or reduce data | `datasets/arithmetic.py` | Coherent/incoherent arithmetic, `statistics_dataset` |
| Join, stitch, or intersect grids | `datasets/combine.py` | `join_many`, `stitch_many`, `overlap_many`, `combine_datasets` |
| Calibrate range data | `datasets/calibration.py` | `RcsGrid.range_calibrate` |
| Copy, regrid, decimate, or offset data | `datasets/transforms.py` | `duplicate_dataset`, `regrid_axis`, `decimate_axis`, `offset_db`, `coherent_divide` |
| Check units and metadata | `datasets/metadata.py` | Scalar inspection, convention checks, derived metadata |
| Audit a dataset | `datasets/audit.py` | `audit_dataset`, content hashes, sample diagnostics |
| Check allocation limits | `datasets/memory.py` | Import preflight, memory limits, bounded array selections |
| Find constants and unit names | `datasets/constants.py` | Physical constants, units, metadata families, block sizes |
| Record Python actions | `scripting/recorder.py` | `PythonScriptRecorder`, `DatasetReference` |
| Render plots from Python | `scripting/plotting.py` | `plot_datasets` |
| Run from a terminal | `cli.py` | `main`, `grim-headless` |

`RcsGrid` inherits its operations and format adapters from the listed modules.
Call methods on a grid instance or on `RcsGrid`; implement changes in the module
that owns the operation. Dataset methods return new grids where specified in
their docstrings. Importers validate arrays and metadata before returning a grid.

## Python usage

```python
from grim_backend.datasets.grid import RcsGrid
from grim_backend.io.loaders import load_dataset, load_folder
from grim_backend.datasets.combine import combine_datasets
from grim_backend.datasets.transforms import decimate_axis
from grim_backend.io.batch import save_dataset_batch

grid = load_dataset("measurements.grim")
cut = grid.axis_crop(azimuth_range=(0.0, 180.0))
reduced = decimate_axis(cut, axis="azimuth", factor=2, mode="power")
save_dataset_batch([(reduced, "reduced.grim")])
```

`grim_dataset`, `grim_headless`, and `grim_python` also export their public APIs
for saved scripts. Their implementation lives here. Both `grim-headless` and
`python -m grim_headless` accept the command-line arguments defined in `cli.py`.

## Dependencies and documentation

Backend imports are independent of Qt. Plot rendering creates Matplotlib Agg
figures when requested. Format readers use call-time imports for grid-dependent
validation, keeping standalone format parsing available independently of the
data model.

Docstrings describe behavior, inputs, outputs, units, shapes, validation, and
resource ownership. Keep source comments limited to those details.

Run the GRIM suite from the repository root:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
& .venv/Scripts/python.exe -m unittest discover -s GRIM_Revised_2 -p 'test*.py' -v
```
