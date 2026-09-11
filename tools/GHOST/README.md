# GHOST solver and feature workflows

Open `Launch_GHOST_GUI.bat` to start GHOST. The top level contains the launcher,
Markdown guides, and one `ghost_backend` folder.

| Folder | Contents |
| --- | --- |
| `ghost_backend/twod/`, `bor/` | 2-D and body-of-revolution solvers. |
| `ghost_backend/linalg/`, `compressed/` | Factorization, sweep reuse, matrix compression, and RAM forecasts. |
| `ghost_backend/assembly/` | Feature placement, field combination, and assembly validation. |
| `ghost_backend/io/` | Dataset loading, export, and filename operations. |
| `ghost_backend/runs/`, `execution/`, `hpc/` | Run configuration, CPU resources, provenance, and batch scheduling. |
| `ghost_backend/ui/` | Desktop geometry and solver controls. |
| `ghost_backend/data_tools/` | Dataset subtraction, joining, renaming, conversion, and their GUI/CLI. |
| `ghost_backend/geometry/` | Geometry loading, materials, mesh checks, rotation, sample geometries, and placement CSV templates. |
| `ghost_backend/validation/` | Material and geometry validation studies with their fixtures. |
| `ghost_backend/tests/` | Solver tests and performance benchmarks. |

Only `run_gui.py` and the four local/HPC run scripts are directly inside
`ghost_backend`. Native source, libraries, and build tools are under
`ghost_backend/bor/native/`; runtime support is under `ghost_backend/execution/`.

Batch drivers save new runs under `ghost_backend/results/rcs_runs` or
`ghost_backend/results/rcs_runs_bor` by default. Explicit output paths and saved
configurations keep their selected destinations.

Use the [backend source guide](BACKEND.md) to find solvers, data I/O,
geometry operations, feature assembly, and run management. The
[file-removal audit](DEAD_FILES.md) identifies cleanup candidates.

New 2D monostatic GUI and local/HPC runs default to the
[CPU streaming (experimental) solver method](EXPERIMENTAL_CPU.md) for lower
angle-workspace RAM and CPU acceleration of PEC/IBC and dielectric solves.
The default [compressed CPU assembly path](COMPRESSED_CPU.md) avoids the
global dense matrix and LU for supported 2D formulations. The saved resource
preset uses 8192 MiB compressed storage, four assembly/two BLAS threads, automatic
basis reuse, and 256-angle batches. Dense LU remains selectable for smaller runs.

The visible **Geometry preset** selector offers **Small Geometry (No RAM
Optimization)**, **Large Geometry (RAM Optimization)**, and **Balanced**. Large
Geometry is the default. Detailed numerical and resource controls are under
**Advanced Settings**, which starts collapsed in GHOST and GRIM Runs.
[Saved execution profiles](RUN_PROFILES.md) document the preset combinations,
factorization, RAM admission, compressed storage, temporary disk, CPU threads,
and sweep batching. Profiles transfer through local/HPC manifests and exports.
The guide also describes stage/RAM progress and the repeatable benchmark.

Local and HPC batch drivers accept `--config path/to/settings.config.json`.
They also automatically load an adjacent file with the same stem and the
`.config.json` suffix. For example:

```json
{
  "schema": "ghost.driver-config",
  "version": 1,
  "driver": "2d",
  "settings": {
    "FRD_DIR": "ghost_backend/geometry/geometries/FRD",
    "OPN_DIR": "ghost_backend/geometry/geometries/OPN",
    "OUTPUT_DIR": "rcs_runs",
    "FREQUENCIES_GHZ": [2.0, 4.0],
    "AZIMUTHS_DEG": [0.0, 90.0],
    "MESH_CERTIFICATION": true
  }
}
```

Use `"driver": "bor"` and `GEOMETRY_DIRS` for BoR. Each driver declares its
accepted setting names in `_CONFIG_KEYS`; omitted settings keep its defaults.
Paths in settings retain the driver's existing working-directory semantics.
An optional `run_setup` object can embed an exported desktop 2-D run recipe.
Batch drivers accept its monostatic/default-quality subset and reject
unsupported options or conflicting explicit settings.

`ghost_backend.hpc.common.configure_driver` stages Python source and writes
validated JSON instead of rewriting assignments. Submission copies both into
the run directory. Configuration content joins source/runtime provenance, so
workers reject changes to the settings that produced an existing run.
New source/configuration versions should use a fresh staging directory.

The recommended desktop workflow is the top-level GRIM application. Its
**GHOST** tab embeds the same `ghost_backend/run_gui.py` workspace and the same
2-D/BoR numerical implementation found here; no solver is duplicated.

The 2-D diagnostic API accepts `auto` and `direct` for reference kernels and
`experimental_cpu` for CPU streaming. Explicit execution profiles select dense,
hierarchical, or compressed factorization within the supported combinations.
NumPy and SciPy are required for numerical methods and condition-number checks.
BoR supports an optional native streaming kernel.

The September 2026 [solver audit updates](SOLVER_AUDIT_UPDATES.md) correct
2-D complex phase, bound BoR near storage, and add quadrature convergence
checks. Read the phase-compatibility notes before combining legacy complex
exports with newly generated results.

Build the native BoR sampler on the worker machine with:

```powershell
py ghost_backend/bor/native/build_kernel.py
```

On Windows, install MSYS2 in its default `C:\msys64` location, open the
**MSYS2 UCRT64** terminal, and install the compiler with:

```bash
pacman -Syu
pacman -S --needed mingw-w64-ucrt-x86_64-gcc
```

If the first update asks you to close the terminal, reopen **MSYS2 UCRT64**
and run both commands again. The build script discovers the default UCRT64
compiler automatically; no global PATH change is required. Verify from this
folder with `py ghost_backend/bor/native/build_kernel.py` and restart Python workers.

The build enables OpenMP outer-loop parallelism when the compiler supports it
and automatically retries a portable serial build otherwise. Use
`--no-openmp` to request the serial build explicitly. Result metadata reports
`stream_sampling_backend=native_c` or `numpy`, so production runs do not hide
which path was active.

Bounded far-block streaming is available for PEC/IBC, homogeneous dielectric
PMCHWT, simple coated-PEC bodies, and partial, layered, or banded junction
systems. The budget is enforced across every simultaneously retained self and
rectangular cross-surface block. Peak planning separately includes cached
junction projections and direct near/junction operators, which remain resident
when the far field is streamed. Result metadata records the sampling backend
for each medium side/mapping (a lossy material side uses the complex-wavenumber
NumPy sampler).

## Optional 2-D GPU dense solves

The 2-D survey path can offload complex dense LU solves to an NVIDIA GPU via
CuPy. This workstation was validated with the isolated CUDA 12 component
wheels:

```powershell
..\..\.venv\Scripts\python.exe -m pip install "cupy-cuda12x[ctk]"
$env:GHOST_DENSE_BACKEND = "auto"
$env:GHOST_DENSE_GPU_MIN_N = "768"
```

Use `GHOST_DENSE_BACKEND=cpu` for the default CPU-only behavior, `auto` for a
GPU attempt at or above the configured matrix order with audited CPU fallback,
or `gpu` to fail rather than silently fall back when a GPU-eligible solve
cannot run. GHOST performs a timed child-process cuSOLVER health check before
the first GPU solve, checks available device memory, and applies the existing
CPU backward-error gate to the returned solution. Runs requesting the release
condition-number estimate remain on CPU because that gate currently reuses a
SciPy LU factorization. Metadata records the backend, solve counts, device,
and any fallback reason.

## Standalone GHOST

Run commands from this folder:

```powershell
py ghost_backend/run_gui.py
```

On Windows, `Launch_GHOST_GUI.bat` first changes to this folder and then opens
the same workspace.

## Local and HPC drivers

Edit the configuration block in the relevant driver, then run:

```powershell
py ghost_backend/run_local_monostatic.py
py ghost_backend/run_local_bor.py
py ghost_backend/run_hpc_monostatic.py
py ghost_backend/run_hpc_bor_monostatic.py
```

The 2-D production path co-solves VV/TE and HH/TM and writes them into one
GRIM artifact per geometry/frequency. The BoR path produces a combined
feature-ready body artifact after its per-frequency restart units complete.

See:

- [HPC.md](HPC.md) for local/cluster operation and resource controls.
- [GEOMETRY_INPUT_CHEATSHEET.md](GEOMETRY_INPUT_CHEATSHEET.md) for `.geo`
  boundaries, regions, materials, winding, and units.
- [BOR_CONVENTIONS.md](BOR_CONVENTIONS.md) for BoR geometry, polarization,
  phasor, loss, and RCS conventions.
- [FEATURE_VALIDATION_GUIDE.md](FEATURE_VALIDATION_GUIDE.md) for point and
  line-feature dataset and placement requirements.
- [ghost_backend/validation/non_bor_feature_validation/README.md](ghost_backend/validation/non_bor_feature_validation/README.md)
  for the independent four-artifact clean/featured validation ladder and
  manifest-driven complex-field gates.
- [ghost_backend/validation/non_bor_line_reconstruction/README.md](ghost_backend/validation/non_bor_line_reconstruction/README.md)
  for the checked-in finite-plate, door-outline, and folded-panel line tests.
- [ghost_backend/validation/non_bor_curved_feature_placement/README.md](ghost_backend/validation/non_bor_curved_feature_placement/README.md)
  for the triaxial-ellipsoid point/line regression and shared-facet normal-tie
  controls.

## Feature assembly service

The GRIM Assembly form and automation wrapper both call
`ghost_backend/assembly/workflow.py`. `ghost_backend/assembly/place_features.py` remains a thin
settings-based wrapper for unattended work. It defaults to advisory metadata:
source certificates, version tags, convention labels, and feature/surface
manifests do not gate ordinary use. Numerical units, fields, axes, and placement
geometry are still checked. Strict library metadata and certified-body profiles
are optional. Only an explicitly selected strict profile or a large workload
review requires a warning acknowledgement. The GUI uses the same placement,
phase, expansion, and shadowing implementation.

The integrated tab supports editable placements, point rows/circles/polyline
patterns, line paths, explicit surface projection and normal derivation,
body-only baselines, exact stored-grid subsets, and body/feature/total response
comparison. Newly selected BoR bodies can generate their own bounded-error
shadow mesh. See the [Assembly workflow guide](FEATURE_VALIDATION_GUIDE.md).
The 2-D `amplitude_version` is advisory during subtraction and line loading.
Operations record convention assumptions without inferring a field conversion.

FREDDY nominal IBC and dielectric CSVs store frequencies in Hz and are readable
with or without headers. GHOST converts Hz to its internal GHz scale and
preserves signed complex material values. Analysis/uncertainty CSVs are separate
from nominal material tables.

Create and check a reviewed feature-response sidecar with:

```powershell
py ghost_backend/assembly/create_feature_manifest.py create --help
py ghost_backend/assembly/create_feature_manifest.py check --help
py ghost_backend/assembly/create_feature_manifest.py create-surface-binding --help
py ghost_backend/assembly/create_feature_manifest.py check-surface-binding --help
```

For `validated` libraries this is now an evidence-binding and integrity tool:
it consumes the full-wave validator report, re-hashes all four case artifacts,
and proves that the assembled prediction used the exact response. Team review
is still required because software cannot establish external-solver
independence or mesh convergence. See
[FEATURE_VALIDATION_GUIDE.md](FEATURE_VALIDATION_GUIDE.md) for the exact
manifest fields, headless settings, reduced-order limitations, and required
independent full-wave evidence.

Use `python ghost_backend/data_tools/run_cli.py subtract OPN FRD Deltas` for
canonical OPN-FRD 2-D deltas. General joins and dataset conversion use the same
CLI. See [Data Tools](DATA_TOOLS.md).

## Tests

From this folder:

```powershell
py -m unittest discover -s ghost_backend/tests -p "test*.py" -v
```

## Material and IBC files

Use headered, comma-separated `.csv` files with frequency in Hz, following the
[shared GHOST/FREDDY file format](MATERIAL_CSV_FORMAT.md). FREDDY material
and nominal IBC exports can be used directly. The geometry editor validates
CSV selections before adding them. Space/tab-separated tables, headerless
CSVs, and implicit `mat.<flag>` references are not accepted.
