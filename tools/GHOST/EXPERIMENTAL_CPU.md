# CPU streaming (experimental)

The CPU method batches the requested monostatic angles while retaining
double precision, the original boundary-integral equations, mesh and quadrature
settings, and the selected accuracy/certification controls. Every requested
angle is computed and checked against its original equation. No GPU is required.

## Select the option

New 2D monostatic runs use **CPU streaming (experimental)** with compressed
assembly in GHOST and local/HPC drivers. The
[efficient resource preset](RUN_PROFILES.md) is the default for new runs.
Selecting CPU streaming sets LU precision to Double; mixed
precision is a separate option and cannot be combined with this method.
The newer optional hierarchical factor and automatic sweep compression are
documented in [COMPUTATION_RAM_UPDATES.md](COMPUTATION_RAM_UPDATES.md).
Direct geometry-to-compressed assembly is an additional explicit option,
documented in [COMPRESSED_CPU.md](COMPRESSED_CPU.md).

Saved `.run.json` recipes preserve `solver_method`. Older recipes without this
field select the reference method. The method applies to 2D monostatic fields;
the GUI returns to the reference method when selecting bistatic or BoR.
Both CPU modes now share bounded solves and checked density diagnostics;
the experimental selector additionally enables screened kernel tables and
analytic far-field projection.

For either `ghost_backend/run_local_monostatic.py` or
`ghost_backend/run_hpc_monostatic.py`, add these settings to the existing driver
configuration's `settings` object:

```json
"SOLVER_METHOD": "experimental_cpu",
"LU_PRECISION": "double"
```

Alternatively set the corresponding constants in the driver. Existing worker,
assembly-thread and BLAS-thread settings still apply. For one complete job at
a time, use `WORKERS=1` locally or `MAX_WORKERS_PER_NODE=1` on HPC, with thread
counts chosen for the CPUs allocated to the process. CPU streaming itself does
not change the number of simultaneous geometry/frequency jobs.

The public monostatic APIs accept the same option:

```python
result = solve_monostatic_rcs_2d_certified(
    snapshot, frequencies_ghz=[1.0],
    elevations_deg=list(range(181)), geometry_units="meters",
    solver_method="experimental_cpu",
)
```

`solve_monostatic_rcs_2d_survey` also accepts the option and retains its normal
survey status. Selecting the method never turns certification off.

## Scope and numerical behavior

Streaming covers PEC/IBC Robin systems, a single dielectric body,
multi-region dielectric/coated/mixed systems, sheets, sheet+PEC and the
supported transmitting thin-layer model. These use one FP64 factor per
polarization and mesh, then solve and project at most 256 azimuths at a time.
The default factor is dense LU; the optional hierarchical inverse has separate
storage and exact-matrix residual checks. Validated zero-contrast thin layers
return zero scattering without a factorization.
The hierarchical option (`GHOST_CPU_FACTORIZATION=hierarchical`, or `auto` for
dense fallback) first builds an inverse with a `1e-6` block tolerance. This is
only a preconditioner tolerance: every solve, transpose/adjoint and condition
estimate still uses refinement against the original matrix. The coarse inverse
targets `3e-15` normwise backward error to add accuracy margin. A rejected
construction or stalled refinement releases
the smaller inverse and retries once at the previous `2e-10` block tolerance
and `3e-14` refinement limit.
Metadata records both builds and the reason for retry. The original matrix,
storage cap, physical-angle checks and mesh certification remain required.
The new-run `compressed` preset assembles bounded tiles directly from geometry;
see [COMPRESSED_CPU.md](COMPRESSED_CPU.md)
for its storage cap, strict rejection behavior and qualification scope.
Exact straight-element plane-wave integrals and vectorized touching-element
quadrature reduce CPU work. Adjacent TE/TM solves reuse or transform the owned
system where the equations permit it; source operators are released before LU.
Certification runs TE/TM on the base mesh, then TE/TM on the fine mesh.

Lossy-material far kernels can use screened degree-16 piecewise polynomials.
Each table is checked against independent reference Hankel evaluations at
`2e-13` relative tolerance. Rejected domains/tables and out-of-range evaluations
use the original kernels. Real-wavenumber operators retain their original
Bessel fast path. Finite validation establishes agreement on tested cases,
not a proof for all possible materials and meshes.

Solver quality failures are reported normally. The original condition estimate, backward-error refinement,
residual gate and base/fine certification remain applicable. Cancellation is
checked between angle batches; an ongoing dense factorization must finish
before cancellation can be observed.

With dense factorization, the matrix and LU require quadratic RAM with mesh size. Angle
workspaces are bounded by the batch, but final output grows with the requested
grid. Kernel tables retain at most 32 MiB. The legacy operator cache has a
64 MiB cap, but active formulation solves disable it in favor of system reuse.
Memory admission conservatively includes both reservations and full-angle output.

## Results, provenance and compatibility

The requested method and `experimental_cpu` diagnostics travel in the `.grim`
solver metadata, including factor/batch counts, kernel checks and fallbacks.
The method also enters local/HPC manifests and resume fingerprints, so results
from the two methods cannot be reused as the same run specification.

The implementation uses ordinary ghost_backend modules, including `cpu_execution.py`,
`cpu_kernels.py`, `dense_factor.py`, `boundary_fields.py`, `assembly_session.py`
and the formulation assemblers. It does not import the
`experiments` folder or replace module functions at runtime. Install/copy the
complete updated ghost_backend when using a separate headless installation.

Qualification covers desktop Python 3.12 and actual Python 3.6.8 environments
with NumPy 1.19.5/SciPy 1.5.4 and NumPy 1.14.3/SciPy 1.0.0. HPC worker execution,
portable requests, export and resume were exercised locally; this does not
establish performance on a remote cluster or across multiple nodes.

Regression entry points:

```text
ghost_backend/tests/test_experimental_cpu.py
ghost_backend/tests/test_experimental_gui.py
ghost_backend/tests/test_experimental_headless.py
```
