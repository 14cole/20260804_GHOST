# Automatic 2-D solver

New 2-D monostatic runs use Automatic in the embedded GRIM/GHOST GUI, the public
solver API, and both local/HPC drivers. Ordinary runs need geometry, materials,
frequencies, angles, and accuracy settings; users do not need to pick a backend.
Kernel and factorization overrides are under Advanced Settings. Existing saved
explicit profiles retain their recorded settings.

## What is selected

Automatic compares compatible dense LU, compressed, and FMM implementations of
the same Galerkin equations on the selected polynomial mesh. FMM accelerates the interaction
operator; it does not replace Galerkin. Geometry, material formulations, angle
count, certification meshes, available memory, and estimated computation cost
enter the decision. HPC makes its scheduling choice on the execution node and
also considers concurrent workers within that node's CPU/RAM allocation.

The shared cost model is conservative and favors dense solving for small and
medium systems. It is a prediction, not a guarantee of the fastest wall time.
FMM supplies an option when dense storage becomes prohibitive. Missing native
libraries and unsupported formulations exclude FMM from Automatic.

An automatically selected backend may retry another admitted backend after a
numerical or storage failure, with unchanged accuracy tolerances. Scheduled
retries must fit the original unit's RAM reservation. Invalid input,
cancellation, and failed physical mesh convergence do not trigger a backend
retry. Explicit backend overrides retain their explicit failure behavior.
Result metadata records selection, forecasts, attempts, and total execution
time including planning and retries.

## Implemented numerical and performance work

- Adaptive quadratic/cubic basis functions with graded and locally refined
  panels, strict full-field convergence checks and a linear reference fallback.
  See [adaptive mesh details and measurements](ADAPTIVE_MESH.md).
- Corrected close-interaction quadrature, including separated panels with a
  small gap; geometry-aware spatial filtering retains the supplied geometry.
- Native FMM Galerkin operators with near-interaction corrections and memory
  admission, supporting the existing qualified PEC, IBC, dielectric, coated,
  layered, mixed-region, and impedance-sheet routes.
- Persistent native geometry plans and work buffers reused at fixed frequency.
- Formulation-aware GMRES/LGMRES angle solves with bounded augmentation,
  incremental incident-basis reuse, original-equation residual verification,
  and stagnation clearing. Explicit recycling overrides remain available.
- Bounded physical angle batches and native density workspaces, automatic
  backend selection shared by GUI/API/HPC, and recorded admission/retry reasons.
- Headless qualification tests, native source/build provenance, and an optional
  installable GHOST package inside the integrated source tree.

Existing restrictions on sheet/material coupling remain. Nonzero thin
dielectric layer approximations and bistatic FMM are excluded. The separate
smooth-surface Nystrom prototype is a research API; Automatic never substitutes
it for arbitrary imported geometry. Closed pure-PEC TM FMM now selects the
qualified outgoing combined field by default; explicit saved overrides remain
in force. Other material equations keep their formulations. The optional Pulse
discretization and the FMM changes are described in
[Pulse and FMM efficiency updates](PULSE_FMM_UPDATES.md).

Mesh certification compares fields on base and refined meshes. It does not
certify geometric fidelity to a curved object, and solver residuals do not
establish a physical error bound. FMM approximation and quadrature tolerances
are recorded separately. Keep mesh certification enabled for accuracy studies.

## Previously measured savings

These are measurements from the standalone implementation before integration,
not new speed measurements of the complete GRIM application. They establish
case-specific improvements, not expected gains on every model.

| Comparison | Measured result | Scope |
| --- | --- | --- |
| Updated FMM versus its first implementation | 13.9-54.3% less elapsed time; up to 34.3% less sampled process RAM | Seven polygon/material cases, 192 panels, 3 GHz, nine angles, both polarizations |
| Automatic dense versus explicit FMM | 9.43 s versus 388.86 s | A 6 m PEC rectangle at 1 GHz with 2,048 panels; the FMM timing preceded the final stagnation guard |

The second case explains why Automatic does not assume FMM is faster at every
size. A 20 GHz mesh study showed 8-14% field changes on refining 384 to 768 panels
for difficult gap/material cases. Those meshes were not physically converged.
Future benchmarks should use representative user geometries and converged
meshes, with isolated wall-time/RAM measurements on the deployment workstation.
The integrated [follow-up report](SOLVER_FOLLOWUPS.md) adds supplied-airfoil
measurements, refinement evidence, material-kernel acceleration and updated
workstation/HPC resource planning.

## Runtime and native setup

Current GHOST requires Python 3.10+, NumPy 2.0+, and SciPy 1.14+. The integrated
desktop uses its existing Windows Python 3.12 dependency lock, which also
includes pytest for accelerated-solver verification. Current headless workers
use `requirements/hpc.txt` at the project root. The historical Python 3.6
profile applies only to the older solver revision.

The optional FMM library is loaded from
`ghost_backend/twod/fmm/native/`. The local Windows merge includes the exercised
x86-64 DLL as an ignored runtime asset. It is not a tracked source artifact or
part of the GHOST wheel. Rebuild on other worker platforms using a complete GNU
Fortran installation; from the integrated project root:

```bash
python tools/GHOST/ghost_backend/twod/fmm/native/build.py
python tools/GHOST/ghost_backend/hpc/check_environment.py
```

Set `FC` to the compiler's absolute path when needed. The build uses vendored
sources and performs no downloads. Automatic remains available through dense
and compressed backends when FMM is absent. Copy all backend sources and the
correct platform library together for cluster execution.

Run the accelerated qualification separately or as part of full verification:

```bash
python tools/GHOST/scripts/check_headless.py
python verify_project.py --mode full
```

For standalone headless package development, install
`python -m pip install -e './tools/GHOST[test]'` from the project root. GRIM
continues to load its authoritative embedded `tools/GHOST` checkout.

The vendored FMM root license and some individual source headers disagree.
Both are preserved; this integration does not resolve their licensing status.
See [native source notices](ghost_backend/twod/fmm/native/THIRD_PARTY.md) before
redistribution. The existing source-release builder does not build the optional
FMM library automatically.

## Remaining work

The follow-up update adds selective spatial coarse correction, exact-request
timing evidence, itemized FMM memory allowances, fewer native routing copies,
and normalization of NumPy angle-array inputs. Remaining work includes broader
multilevel preconditioning and hardware calibration, further translation reuse,
rigorous adaptive error estimators and variable local polynomial degrees, additional material
couplings, independent material-junction validation and real Linux/HPC runs.
The current solver should not be described as a fully qualified industrial
solver across the entire 1-20 GHz and 10-20 ft geometry range.
