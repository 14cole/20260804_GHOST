# Saved 2D execution settings

## HPC and local batch presets

Both 2D sweep scripts now start with one solve choice:

```python
SOLVE_PRESET = "auto"             # auto | small | balanced | large
ADVANCED_OVERRIDES = {}          # optional execution fields
MAX_SOLVE_GB = None              # optional per-solve RAM admission limit
```

`small`, `balanced`, and `large` match the desktop geometry presets below.
`auto` targets **predicted batch completion time**. On the execution node it
compares compatible dense, compressed, FMM, and mixed schedules using each
frequency's geometry/material forecast, angle count, allocated CPUs, worker
ceiling, and schedulable RAM. Missing native FMM libraries and unsupported
formulations exclude that candidate. Completed outputs are verified before reuse.
The saved factorization value is `adaptive`; the legacy factorization value
`auto` still means hierarchical factorization with dense fallback.

Schedules with at most 4,096 candidate combinations are compared exhaustively;
larger searches are bounded. The shared `geometry_work_v3_pulse` cost model is a
conservative heuristic, not a guarantee of the fastest wall time. It uses no
trial solves and does not model queue delays. The model favors dense solves for
small and medium systems, while FMM offers a path beyond dense storage capacity.
Logs and result metadata retain the choice, forecasts, and reason. An automatic
backend retry must fit the original worker RAM reservation and retain the
accuracy tolerances. Invalid input, cancellation, and failed physical mesh
convergence are not backend-retry conditions.

Submission builds exact mesh dimensions for each frequency and certification
mesh, sharing topology between polarizations. Compressed RAM forecasts use
the retained-payload ceiling plus inverse and workspace allowances, without
evaluating matrix tiles. These conservative forecasts can reserve more RAM
than a sampled estimate. Actual solves retain storage sampling, memory gates,
residual checks and mesh certification. Automatic batch choices are passed to
workers, avoiding a second backend-selection mesh forecast in each solve.

A local Windows/Python 3.12 profiling run on `airfoil.geo`, 1-10 GHz in 1 GHz
steps, 181 angles and 1.5x certification refinement measured 290.3 seconds
before this change, 15.4 seconds with the new compressed forecast, and 17.0
seconds with both automatic candidates. All 20 channel mesh/DOF/formulation
records matched; the new compressed RAM reservations bounded the previous
sampled estimates. These are submission-planning measurements, not full-solve
or cluster wall-time benchmarks.

The same settings work inside a driver JSON `settings` object:

```json
{
  "SOLVE_PRESET": "auto",
  "ADVANCED_OVERRIDES": {"assembly_threads": 4, "blas_threads": 2},
  "MAX_SOLVE_GB": 32
}
```

Use `ADVANCED_OVERRIDES` for factorization, mesh strategy, compressed storage,
thread counts, basis reuse, angle batching, and temporary-directory settings.
Use `MAX_SOLVE_GB` for per-solve RAM and the existing node/worker settings for
the allocation. Conflicting duplicate settings are rejected. Presets leave
frequencies, angles, geometry units, accuracy and certification unchanged.

Legacy JSON recipes with explicit kernel, precision or execution settings
continue to use their recorded configuration. `SOLVE_PRESET="custom"` allows
direct use of the older script aliases. Named presets resolve to a complete
execution profile that is carried by the manifest and fresh worker processes;
ambient environment variables do not replace those explicit settings.

## Desktop presets

The GHOST solver tab shows Geometry Source, solver, solver units, frequency and azimuth inputs, scattering mode, the geometry/setup check,
and output controls. Geometry Source selects a .geo file or the current Geometry
tab. **Advanced Settings** starts collapsed and contains geometry presets, accuracy, mesh
certification, quality thresholds, kernel evaluation, factorization,
resource limits, saved setups, and boundary-density/report tools. Frequency and
azimuth inputs display either a list or a sweep, according to the selected mode.
BoR uses aspect angles from +z in place of the 2D azimuth input.

The GHOST preset selector and advanced **2D execution and resources** controls
are captured by **Save run setup**. Restore a setup in GHOST or embed it as
`run_setup` in a local/HPC driver JSON configuration. Geometry paths, output
paths, and cluster allocation settings are configured separately.

| Geometry preset | Kernel evaluation | Factorization | Sweep basis reuse |
| --- | --- | --- | --- |
| Small Geometry (No RAM Optimization) | Reference | Dense LU | Off |
| Large Geometry (RAM Optimization) | CPU streaming | Compressed assembly | Automatic |
| Balanced | CPU streaming | Dense LU | Automatic |
| Automatic, default | Automatic | Dense/compressed/FMM selected from geometry, work, and RAM forecasts | Automatic |

All four use double precision, up to four assembly threads and two BLAS threads,
and 256 angles per batch. Large Geometry uses an 8192 MiB compressed payload
allowance. Automatic uses a 2048 MiB retained-storage allowance. Small Geometry disables optional compression; normal allocation
checks and bounded solver workspaces still apply. Balanced retains a dense
matrix and can be faster when it fits in RAM; it does not automatically switch
to compressed assembly based on geometry size.

Choosing a preset changes performance settings and retains frequency, angle,
accuracy, mesh-certification, RAM-budget, and temporary-directory selections.
Manual performance changes display **Custom (Advanced Settings)** when they no
longer match a preset. Saved setups store the actual values, so loading one does
not silently reapply a preset. Existing saved profiles remain valid. Presets are
available for **2D monostatic** solves; BoR and bistatic retain their supported
controls in Advanced Settings.

New 2D monostatic desktop and local/HPC runs default to Automatic, double
precision, a 2048 MiB retained-storage allowance, up to four assembly and two
BLAS threads, automatic sweep basis reuse, and at most 256 angles per batch.
FMM bounds its physical angle batches to at most 32 and its native density
groups to at most eight. Thread defaults are reduced on smaller hosts. RAM
admission uses current available memory and any explicit user budget. Mesh
certification stays enabled. **Use efficient defaults** reapplies Automatic.

Earlier compressed-only airfoil measurements remain useful for manual tuning:
the 2 GHz certificate used 77.8% less sampled RAM and took 36.7% longer than
dense LU. The new selector accounts for both predicted time and storage.
See [the latest measured savings](AUTOMATIC_SOLVER.md) for FMM comparisons and
their limits.

| Control | Behavior |
| --- | --- |
| Dense LU | Assembles the full dense operator. |
| Hierarchical factor (dense assembly) | Assembles a dense operator and compresses its factorization. |
| Compressed assembly (low RAM) | Builds compressed tiles from geometry; requires CPU streaming kernels and double precision. |
| Hierarchical with dense fallback | Can fall back to dense LU; unsuitable when a dense allocation cannot fit. |
| Automatic backend | Compares compatible dense, compressed, and FMM costs and RAM for both certification meshes. Saved value: `adaptive`. |
| FMM Galerkin | Uses a native fast multipole operator with corrected near interactions and recycled iterative solves. Requires the optional native library and a supported material formulation. |
| Mesh sizing | `global` by default; optional `local` material sizing protects nearby boundaries and geometric features. Certified local runs retry global sizing if mesh convergence fails. |
| RAM budget per solve | Admission threshold for estimated total RAM, bounded by 90% of currently available memory. Available memory uses that bound alone. This does not enforce an OS memory limit. |
| Compressed storage cap | Retained numeric operator/inverse payload, including partner-polarization reservations. Workspace and runtime RAM are additional. |
| Assembly threads | Requested assembly concurrency, capped by a batch worker's allocation. Auto uses the batch scheduler; desktop Auto uses one thread. |
| BLAS threads per solve | Applied to already-loaded native BLAS libraries during a configured solve. Batch scheduling reserves these CPUs too. |
| Temporary disk directory | Existing directory on the execution host for owned polarization spool files. Empty selects that host's system temporary directory. |
| Sweep basis reuse | Reuses an illumination basis subject to the solver's checks. |
| Angles per batch | Bounds simultaneous physical illumination workspaces; 1 through 256. |

Selecting compressed assembly sets the compatible kernel and precision choices.
Resource controls are disabled during a running job. These saved resource
profiles apply to 2D; BoR retains its own modal and worker settings.

## Choosing resource settings

Automatic is the default starting point. For a manual compressed configuration, use **8192 MiB compressed storage, 4 assembly
threads, 2 BLAS threads, global mesh sizing, automatic sweep basis reuse,
256 angles per batch, and frequency checkpoints enabled**. Thread counts are
reduced on small CPU hosts. Total RAM requirements must still fit the machine.

| Setting | Options and purpose | Time and RAM tradeoff |
| --- | --- | --- |
| Compressed storage cap | Numeric allowance in MiB for the retained compressed operator and inverse, including reserved partner-polarization data. 8192 MiB is 8 GiB. Active for compressed assembly, FMM near storage, and automatic selection. | The cap does not preallocate RAM, change mesh accuracy, or limit total process memory. Lowering it can cause a storage-cap failure; it does not force tighter compression. Raising it allows larger representations but does not inherently make a solve faster. Workspaces, mesh data, threads, and outputs need additional RAM. |
| RAM budget per solve | Available memory, or a specified GiB admission budget. The check also limits admission to 90% of currently available memory. | Use this for total estimated solve RAM. It rejects a run forecast to exceed the budget; it is not an operating-system allocation limit. |
| Assembly threads | Auto or a positive integer. These workers evaluate geometry interactions and build matrix tiles. Auto follows a batch worker's allocation; desktop Auto uses one thread. | Start at 4, or 1-2 on a smaller machine. More workers can speed assembly, but their workspaces use RAM and they may compete for memory bandwidth. |
| BLAS threads per solve | Positive integer controlling native matrix operations such as factorization and matrix products. GHOST applies the bundled controller during the solve and restores prior limits afterward. | Start at 2. One minimizes thread contention; additional threads can help large dense factorizations. The best value depends on the CPU and matrix size. Assembly and BLAS parallelism can overlap, so high values in both controls can be slower. |
| Mesh sizing | Global material wavelength or Local material sizing (experimental). Global uses the shortest relevant material wavelength throughout wavelength-sized segments. Local can use larger panels on less demanding segments while retaining global sizing at corners, ends, junctions, and nearby boundaries. Explicit panel-count segments retain their counts. | Global is the default. Local can reduce unknowns, assembly time, and matrix storage for separated mixed-material geometries. All-PEC geometries have little material-wavelength benefit. Keep mesh certification enabled; a failed local convergence check retries global sizing and can increase completion time. |
| Sweep basis reuse | Automatic, Disabled, or Enabled. The solver can solve a smaller independent set of incident fields and reconstruct the requested angular results. | Automatic avoids this overhead on small systems. Enabled attempts reuse more broadly but retains savings and numerical checks; unsuitable batches fall back to ordinary solves. Disabled solves every requested right-hand side. Reuse is most useful for many related azimuths, retains a bounded basis in RAM, and does not skip output angles. |
| Angles per batch | Integer from 1 to 256. Limits the number of physical incident-angle right-hand sides handled together. | Start at 256 for long sweeps. Try 64 or 128 to reduce angular workspace RAM if needed. Smaller batches can add overhead and reduce basis-reuse opportunities; the frequency's matrix/factor is reused across batches. This changes working memory, not angle spacing or the total requested angles. |
| Keep completed frequencies and resume matching runs | Enabled or disabled; enabled by default for desktop 2D monostatic runs. Each completed frequency is saved to the application cache on disk and verified before reuse. | Leave enabled for expensive sweeps. It avoids recomputing completed matching frequencies after cancellation or restart, with disk-space and read/write overhead. It saves results and metadata, not assembled matrices or factors. An interrupted frequency restarts from its beginning; a single-frequency run has no partial-frequency recovery. |

Frequency checkpoints require matching geometry, angle selections, material
file contents, execution settings, precision, certification settings, and
solver source. Changed inputs or corrupt checkpoints are recomputed. Survey
and certified results are kept separate. Updating the bundled thread-control
source also changes the solver identity, so checkpoints from before that
update will not be reused. Cache files do not replace exporting the final result.

For the supplied airfoil, keep Automatic and mesh certification enabled.
The backend selects an admitted dense, compressed, or FMM path for the execution
host. Resource limits can reflect either a workstation or an HPC allocation.
Compressed storage is a payload allowance, not a prediction of process RAM.

The status text reports assembly, factorization, angle solving, and mesh
certification work, with elapsed solve time and sampled process RAM. Base and
refined mesh phases are identified separately. The percentage can remain fixed
while a long stage runs. RAM is sampled every 50 ms and includes other work in
the same process; it is not an exact allocation peak. Stage timings in exported
metadata are inclusive and may overlap.

## Python and batch configuration

Public 2D solve entry points accept `execution_options`:

```python
result = solver.solve_monostatic_rcs_2d_certified(
    snapshot, [10.0], list(range(361)), geometry_units="inches",
    solver_method="auto", max_panels=100000,
    execution_options={
        "assembly_threads": 4,
        "blas_threads": 2,
        "temporary_directory": "",
    },
)
```

In a 2D driver's JSON `settings` object use:

```json
{
  "SOLVE_PRESET": "auto",
  "ADVANCED_OVERRIDES": {
    "ram_budget_gib": null,
    "assembly_threads": "auto",
    "blas_threads": 2,
    "temporary_directory": ""
  }
}
```

Missing profile fields receive explicit defaults during validation. If legacy
`ASSEMBLY_THREADS`, `BLAS_THREADS_PER_WORKER`, or `MAX_SOLVE_GB` keys are also
provided, their values must match the profile. The complete validated record is
saved in the HPC manifest and passed to fresh workers. Workers use it even if
their launch environment selects a different factorization. Exported solver
metadata records the requested profile and effective assembly/BLAS thread counts.

Shared run setups use schema `grim.2d-run-setup`, version 2. Version 1 setups
migrate to explicit dense/default resource settings, independent of the launch
environment. Unsupported combinations and malformed profiles fail before
changing loaded controls or starting a solve. Absolute custom temporary paths
must exist on the execution host; leaving the path empty is portable.

Saved profiles and their versioned missing-field defaults remain reproducible.
New normal GUI and driver runs use Automatic. For Python integrations, request
`solver_method="auto"` as shown above; legacy explicit presets remain supported.
Explicit reference
kernel or mixed-precision driver configurations continue to use compatible dense
settings unless another supported profile is supplied.

Callers without a profile retain environment-based selection. Drivers capture
those values when planning a run. An explicit profile takes precedence over
the corresponding environment variables. A nested solve cannot replace an
active profile with different settings.

BLAS settings are process-wide. Configured solve sections are serialized within
one Python process to prevent competing native thread limits. Batch workers are
separate processes and can still execute concurrently. Install the updated
NumPy/SciPy dependencies; threadpoolctl itself is bundled under
`ghost_backend/execution/thread_control/` with its licenses. Python 3.9 and
newer use version 3.6.0; Python 3.6-3.8 use version 2.2.0. No separate
threadpoolctl installation is needed. Copy the complete backend directory.
The HPC environment checker exercises native thread limiting and a complex
LU solve without importing Qt.

## Performance regression checks

From the repository root:

```powershell
.venv/Scripts/python.exe tools/GHOST/ghost_backend/tests/benchmark_execution.py --output baseline.json
.venv/Scripts/python.exe tools/GHOST/ghost_backend/tests/benchmark_execution.py --output candidate.json --baseline baseline.json
```

The suite uses PEC and mixed PEC/dielectric geometries, both polarizations,
361 azimuths, and dense/compressed modes. Each of three repeats runs in a fresh
process. It records geometry and source hashes, the exact profile, native
library/thread details, wall time, stage timings, sampled peak RSS, and complex
fields. Compressed results must agree with dense results to a maximum normalized
complex-field error of `1e-8`.

Comparisons require matching inputs, profiles, and runtime environments.
Defaults flag more than 25% additional median solve time or 15% additional
median peak RSS; `--time-tolerance` and `--ram-tolerance` adjust those limits.
Use an idle machine for baseline comparisons. `--panels` changes the number of
panels per boundary; `--certified` includes base/refined mesh certification.
`--backend` selects another backend checkout for before/after measurements.
This small regression suite does not replace high-frequency airfoil
qualification or material-specific physical validation.
