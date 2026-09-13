# GHOST backend

Open `ghost_backend` for the categorized implementation. The paths in the task
table are relative to GHOST. Only the GUI launcher and four local/HPC run
scripts sit directly inside `ghost_backend`. Support code lives in categories.

## Find a task

| Task | Location |
| --- | --- |
| 2-D monostatic/bistatic solve and mesh certification | `ghost_backend/twod/solver.py` |
| 2-D meshes, materials, panels, linear basis | `ghost_backend/twod/geometry.py` |
| 2-D quadrature and boundary operators | `ghost_backend/twod/operators.py` |
| PEC/IBC, dielectric, mixed-region, sheet, thin-layer equations | `ghost_backend/twod/formulations/` |
| Geometry plans, assembly sessions, compact blocks, mass terms | `ghost_backend/twod/assembly/` |
| BoR solving, dispatch, modal kernels, streaming | `ghost_backend/bor/` |
| Dense LU, refinement, HODLR, shared sweep basis | `ghost_backend/linalg/` |
| Compressed matrices, coefficient queries, inverse, RAM forecasts | `ghost_backend/compressed/` |
| Geometry loading, materials, mesh QA, surfaces, shadowing | `ghost_backend/geometry/` |
| Body/feature fields, placement, preparation, contracts, inspection | `ghost_backend/assembly/` |
| GRIM data loading/export, viewer bridge, filename rules | `ghost_backend/io/` |
| CPU state, timing/RSS, source hashes, output attestations | `ghost_backend/execution/` |
| Validated per-run factorization and resource settings | `ghost_backend/execution/options.py` |
| Driver profile capture, worker scope, and progress logging | `ghost_backend/runs/execution.py` |
| Shared desktop resource controls | `ghost_backend/ui/execution.py` |
| Run configuration, inputs, setup, quality gates | `ghost_backend/runs/` |
| HPC request bundles, staging, scheduling, result collection | `ghost_backend/hpc/` |
| Cylinder and sphere analytic references | `ghost_backend/validation/` |
| Desktop application, geometry tab, solver tab | `ghost_backend/ui/` |
| ghost_backend-root lookup | `ghost_backend/execution/paths.py` |

Within feature assembly, `fields.py` computes and combines responses;
`workflow.py` validates and executes placement requests; `preparation.py`
captures input sources; `contracts.py` checks reusable feature libraries.

Within the compressed solver, `operator.py` stores matrix tiles, `coefficients.py`
and `regional_coefficients.py` query equation coefficients, `inverse.py` builds
the inverse, `factor.py` checks solves, `polarization_cache.py` stages the partner
polarization, and `memory.py` predicts RAM.

## Commands and imports

| Command | Purpose |
| --- | --- |
| `run_gui.py` | Start the desktop GUI; `--check` verifies imports |
| `run_local_monostatic.py` | Configure and run a local 2-D sweep |
| `run_local_bor.py` | Configure and run a local BoR sweep |
| `run_hpc_monostatic.py` | Configure and submit a 2-D SLURM sweep |
| `run_hpc_bor_monostatic.py` | Configure and submit a BoR SLURM sweep |
| `hpc/bundle.py` | Create, verify, stage, inspect, or submit a portable request |
| `assembly/place_features.py` | Configure and execute feature placement |
| `assembly/create_feature_manifest.py` | Create or check feature-library evidence |
| `io/import_3d_reference.py` | Configure an external 3-D reference import |
| `validation/reconstruction.py` | Compare reconstruction with reference results |
| `validation/feature_family.py` | Validate a feature-family parameter study |
| `bor/native/build_kernel.py` | Build and load-check the native sampler |
| `hpc/check_environment.py` | Check headless dependencies and complex LU |

The four `run_*` drivers retain their editable defaults and adjacent JSON
configuration. HPC jobs copy the driver source into their run directory.
Deploy the complete `ghost_backend` directory to the compute environment.

Import solver and workflow APIs from their categorized packages.

The [module relocation map](IMPORTS.md)
lists each former implementation name and its package import. Use it when
updating separately maintained scripts or rerunning saved experiments.

New Python code can import the package directly:

```python
from ghost_backend.twod import solver
from ghost_backend.geometry.io import parse_geometry
from ghost_backend.assembly import workflow
from ghost_backend.execution.provenance import backend_source_inventory
from ghost_backend.execution.paths import backend_root
```

Add the enclosing GHOST directory to `PYTHONPATH` for package imports. Command
scripts run directly by filename. The package uses Python namespace-package
discovery, so the root needs no `__init__.py` file.

The solver supports the tested Python 3.6 HPC stack, except the desktop modules
under `ui/`. Native BoR source, builds, and libraries live in `bor/native/`.
The dataclasses fallback and its license live in `execution/`; see the top-level
[third-party guide](THIRD_PARTY.md).
Bundled BLAS thread controls and their licenses live in
`execution/thread_control/` and require no separate package installation.

Source fingerprints include package files under their relative paths. Moving
or editing an implementation changes the recorded source identity; resume and
certification checks compare that identity before accepting stored results.

Docstrings describe behavior, inputs, outputs, shapes, units, validation, and
resource ownership. Keep comments focused on those details.
