# GHOST backend

Implementations live in `ghost_backend/`. This directory contains the editable
run drivers, command scripts, public import entry points, and native artifacts.

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
| Backend-root lookup | `ghost_backend/paths.py` |

Within feature assembly, `fields.py` computes and combines responses;
`workflow.py` validates and executes placement requests; `preparation.py`
captures input sources; `contracts.py` checks reusable feature libraries.

Within the compressed solver, `operator.py` stores matrix tiles, `coefficients.py`
and `regional_coefficients.py` query equation coefficients, `inverse.py` builds
the inverse, `factor.py` checks solves, `polarization_cache.py` stages the partner
polarization, and `memory.py` predicts RAM.

## Commands and public imports

| Command | Purpose |
| --- | --- |
| `ghost_gui.py` | Start the desktop GUI; `--check` verifies imports |
| `run_local_monostatic.py` | Configure and run a local 2-D sweep |
| `run_local_bor.py` | Configure and run a local BoR sweep |
| `run_hpc_monostatic.py` | Configure and submit a 2-D SLURM sweep |
| `run_hpc_bor_monostatic.py` | Configure and submit a BoR SLURM sweep |
| `hpc_bundle.py` | Create, verify, stage, inspect, or submit a portable request |
| `place_features.py` | Configure and execute feature placement |
| `create_feature_manifest.py` | Create or check feature-library evidence |
| `import_3d_reference.py` | Configure an external 3-D reference import |
| `validate_feature_reconstruction.py` | Compare reconstruction with reference results |
| `feature_family_validation.py` | Validate a feature-family parameter study |
| `build_bor_stream_kernel.py` | Build and load-check the native sampler |
| `check_hpc_environment.py` | Check headless dependencies and complex LU |

The four `run_*` drivers retain their editable defaults and adjacent JSON
configuration. HPC jobs copy the driver source into their run directory.
Deploy the complete `Backend` directory, including `ghost_backend`, to the
compute environment.

Public imports remain available through `rcs_solver`, `bor_solver`,
`bor_dispatch`, `geometry_io`, `feature_sum`, `feature_workflow`, `grim_io`,
`grim_compat`, `ghost_gui`, `hpc_bundle`, and `workflow_provenance`. These entry
modules alias the corresponding implementation module; edits belong in the
package.

The [module relocation map](IMPORTS.md)
lists each former implementation name and its package import. Use it when
updating separately maintained scripts or rerunning saved experiments.

New Python code can import the package directly:

```python
from ghost_backend.twod import solver
from ghost_backend.geometry.io import parse_geometry
from ghost_backend.assembly import workflow
from ghost_backend.execution.provenance import backend_source_inventory
from ghost_backend.paths import backend_root
```

Backend Python imports support the tested Python 3.6 HPC stack, except the
desktop modules under `ui/`. Native BoR binaries and `bor_stream_kernel.c` live
beside the build script. `_ghost_dataclasses.py`, `DATACLASSES_LICENSE.txt`, and
`THIRD_PARTY.md` provide the Python 3.6 dataclasses fallback and license.

Source fingerprints include package files under their relative paths. Moving
or editing an implementation changes the recorded source identity; resume and
certification checks compare that identity before accepting stored results.

Docstrings describe behavior, inputs, outputs, shapes, units, validation, and
resource ownership. Keep comments focused on those details.
