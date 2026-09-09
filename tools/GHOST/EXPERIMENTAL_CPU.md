# CPU streaming (experimental)

The optional CPU method batches the requested monostatic angles while retaining
double precision, the original boundary-integral equations, mesh and quadrature
settings, and the selected accuracy/certification controls. Every requested
angle is solved. No GPU is required.

## Select the option

In **GHOST > Solver > 2D solver method**, select **CPU streaming (experimental)**.
The same selection is available in the GRIM Runs workspace. Dense LU remains
the default. Selecting CPU streaming sets LU precision to Double; mixed
precision is a separate option and cannot be combined with this method.

Saved `.run.json` recipes preserve `solver_method`. Older recipes without this
field select the reference method. The method applies to 2D monostatic fields;
the GUI returns to the reference method when selecting bistatic or BoR.
Boundary-density diagnostics retain their existing implementation.

For either `Backend/run_local_monostatic.py` or
`Backend/run_hpc_monostatic.py`, add these settings to the existing driver
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

Streaming covers PEC/IBC Robin systems, a single dielectric body, and
multi-region dielectric/coated/mixed systems. These use one FP64 LU per
polarization and mesh, then solve and project at most 256 azimuths at a time.
Exact straight-element plane-wave integrals and vectorized touching-element
quadrature reduce CPU work. A bounded, immutable operator cache can reuse
assembly across the two polarizations.

Lossy-material far kernels can use screened degree-16 piecewise polynomials.
Each table is checked against independent reference Hankel evaluations at
`2e-13` relative tolerance. Rejected domains/tables and out-of-range evaluations
use the original kernels. Real-wavenumber operators retain their original
Bessel fast path. Finite validation establishes agreement on tested cases,
not a proof for all possible materials and meshes.

Sheet and transmitting thin-layer formulations use the reference CPU path;
their fallback reason is recorded in the output. Solver quality failures are
reported normally. The original condition estimate, backward-error refinement,
residual gate and base/fine certification remain applicable. Cancellation is
checked between angle batches; an ongoing dense factorization must finish
before cancellation can be observed.

The dense matrix and LU still require quadratic RAM with mesh size. Angle
workspaces are bounded by the batch, but final output grows with the requested
grid. Cache limits are 64 MiB of operator payload and 32 MiB of retained kernel
tables, not limits on the whole process. Memory admission includes those
budgets and the full-angle output; sheet fallbacks retain their full-RHS estimate.

## Results, provenance and compatibility

The requested method and `experimental_cpu` diagnostics travel in the `.grim`
solver metadata, including factor/batch counts, kernel checks and fallbacks.
The method also enters local/HPC manifests and resume fingerprints, so results
from the two methods cannot be reused as the same run specification.

The implementation is in four ordinary Backend modules (`cpu_execution.py`,
`cpu_kernels.py`, `cpu_checked_lu.py`, `cpu_streaming.py`). It does not import the
`experiments` folder or replace module functions at runtime. Install/copy the
complete updated Backend when using a separate headless installation.

Qualification covers desktop Python 3.12 and actual Python 3.6.8 environments
with NumPy 1.19.5/SciPy 1.5.4 and NumPy 1.14.3/SciPy 1.0.0. HPC worker execution,
portable requests, export and resume were exercised locally; this does not
establish performance on a remote cluster or across multiple nodes.

Regression entry points:

```text
tests/test_experimental_cpu.py
tests/test_experimental_gui.py
tests/test_experimental_headless.py
```
