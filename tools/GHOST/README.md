# GHOST solver and feature workflows

GHOST is bundled inside the GRIM distribution. This folder is a complete
solver project so its backend, tests, geometry studies, CEM utilities, and
launchers retain their established relative paths.

The recommended desktop workflow is the top-level GRIM application. Its
**GHOST** tab embeds the same `Backend/ghost_gui.py` workspace and the same
2-D/BoR numerical implementation found here; no solver is duplicated.

The 2-D solver uses direct dense LU. The diagnostic API accepts `auto` and
`direct`, both of which use the same direct solver and memory checks. FMM and
its native near-field extension have been removed; older scripts requesting
`solver_method="fmm"` must select `auto` or `direct`. NumPy and SciPy remain
required for the direct numerical methods and condition-number checks. BoR's
optional native streaming kernel remains supported.

Build the native BoR sampler on the worker machine with:

```powershell
py Backend\build_bor_stream_kernel.py
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
folder with `py Backend\build_bor_stream_kernel.py` and restart Python workers.

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
py Backend\ghost_gui.py
```

On Windows, `Launch_GHOST_GUI.bat` first changes to this folder and then opens
the same workspace.

## Local and HPC drivers

Edit the configuration block in the relevant driver, then run:

```powershell
py Backend\run_local_monostatic.py
py Backend\run_local_bor.py
py Backend\run_hpc_monostatic.py
py Backend\run_hpc_bor_monostatic.py
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
- [geometry_tests/non_bor_feature_validation/README.md](geometry_tests/non_bor_feature_validation/README.md)
  for the independent four-artifact clean/featured validation ladder and
  manifest-driven complex-field gates.
- [geometry_tests/non_bor_line_reconstruction/README.md](geometry_tests/non_bor_line_reconstruction/README.md)
  for the checked-in finite-plate, door-outline, and folded-panel line tests.
- [geometry_tests/non_bor_curved_feature_placement/README.md](geometry_tests/non_bor_curved_feature_placement/README.md)
  for the triaxial-ellipsoid point/line regression and shared-facet normal-tie
  controls.

## Feature assembly service

The GRIM Assembly form and automation wrapper both call
`Backend/feature_workflow.py`. `Backend/place_features.py` remains a thin
settings-based wrapper for unattended work. It defaults to the Production
profile: strict clean-body metadata and validated content-bound feature manifests.
Assembly does not require or compare host-material declarations. Legacy compatibility must be selected
explicitly. A warning-bearing plan prints its sealed SHA-256 and requires that
exact digest as the acknowledgment before execution, so a changed plan cannot
reuse an old waiver. The GUI does not reimplement placement, phase, expansion,
or shadowing physics.

Create and check a reviewed feature-response sidecar with:

```powershell
py Backend\create_feature_manifest.py create --help
py Backend\create_feature_manifest.py check --help
py Backend\create_feature_manifest.py create-surface-binding --help
py Backend\create_feature_manifest.py check-surface-binding --help
```

For `validated` libraries this is now an evidence-binding and integrity tool:
it consumes the full-wave validator report, re-hashes all four case artifacts,
and proves that the assembled prediction used the exact response. Team review
is still required because software cannot establish external-solver
independence or mesh convergence. See
[FEATURE_VALIDATION_GUIDE.md](FEATURE_VALIDATION_GUIDE.md) for the exact
manifest fields, headless settings, reduced-order limitations, and required
independent full-wave evidence.

Use `1c_build_deltas/subtract_datasets.py` for canonical OPN-FRD 2-D deltas.
General CEM joins and coherent subtraction are under `CEM_Tools`.

## Tests

From this folder:

```powershell
py -m unittest discover -s tests -p "test*.py" -v
```
