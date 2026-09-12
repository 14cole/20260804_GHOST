# GRIM backend

`GRIM_Backend` contains the application and its services. Numerical GHOST and
FREDDY code lives under the repository's `tools/` directory.

## Folder map

Only runnable entry points live at the package root:

- `run_gui.py`: open GRIM.
- `run_diagnostics.py`: check installation and tool discovery.
- `run_headless.py`: run dataset commands from a terminal.
- `run_image_imprinter.py`: open the PowerPoint image tool.

| Folder | Contents |
| --- | --- |
| `datasets/` | Data model, numerical dataset operations, units, metadata, and audits |
| `io/` | Loaders, format readers/writers, batch exports, and SS inspection |
| `io/reference/` | C++ and MATLAB SS-format reference readers |
| `ui/` | Application window, dataset controls, dialogs, theme, widgets, and `assets/` |
| `plotting/` | Plot orchestration, dataset styling, and rendering modes |
| `isar/` | ISAR artifacts, backprojection support, and repeated-pattern processing |
| `assembly/` | Assembly models, recipes, editors, workflows, and workspaces |
| `integrations/` | GHOST and FREDDY discovery and embedding |
| `execution/` | Background dataset jobs and startup diagnostics |
| `reports/` | PowerPoint reports, image imprinting, report recipes, and `templates/` |
| `scripting/` | Headless API, command-line interface, action recording, and plot scripting |
| `examples/` | Editable dataset and plotting examples |
| `tests/` | GRIM automated tests |
| `docs/` | Application usage, module map, and coordinate conventions |

Run from the repository root, or pass the absolute path to a run script:

```powershell
.venv/Scripts/python.exe GRIM_Backend/run_gui.py
.venv/Scripts/python.exe GRIM_Backend/run_diagnostics.py
.venv/Scripts/python.exe GRIM_Backend/run_headless.py --help
```

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
| Run from a terminal | `scripting/cli.py` | `main`, `grim-headless` |

`RcsGrid` inherits its operations and format adapters from the listed modules.
Call methods on a grid instance or on `RcsGrid`; implement changes in the module
that owns the operation. Dataset methods return new grids where specified in
their docstrings. Importers validate arrays and metadata before returning a grid.

## Python usage

```python
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.io.loaders import load_dataset, load_folder
from GRIM_Backend.datasets.combine import combine_datasets
from GRIM_Backend.datasets.transforms import decimate_axis
from GRIM_Backend.io.batch import save_dataset_batch

grid = load_dataset("measurements.grim")
cut = grid.axis_crop(azimuth_range=(0.0, 180.0))
reduced = decimate_axis(cut, axis="azimuth", factor=2, mode="power")
save_dataset_batch([(reduced, "reduced.grim")])
```

The public dataset facade is `GRIM_Backend.datasets.api`; scripting helpers
are in `GRIM_Backend.scripting.api` and `GRIM_Backend.scripting.workspace`.
Use these package imports in Python scripts. Both `grim-headless` and
`python -m GRIM_Backend.scripting.cli` use the arguments in `scripting/cli.py`.

## Dependencies and documentation

The `datasets`, `io`, and `scripting` packages import independently of Qt. Plot rendering creates Matplotlib Agg
figures when requested. Format readers use call-time imports for grid-dependent
validation, keeping standalone format parsing available independently of the
data model.

Docstrings describe behavior, inputs, outputs, units, shapes, validation, and
resource ownership. Keep source comments limited to those details.

Run the GRIM suite from the repository root:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
& .venv/Scripts/python.exe -m unittest discover -s GRIM_Backend/tests -p 'test*.py' -v
```
