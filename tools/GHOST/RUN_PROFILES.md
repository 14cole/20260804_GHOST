# Saved 2D execution settings

The GHOST solver tab shows solver, solver units, a geometry preset, frequency
and azimuth inputs, scattering mode, the geometry/setup check, and output controls.
**Advanced Settings** starts collapsed and contains geometry-file overrides,
accuracy, mesh certification, quality thresholds, kernel evaluation, factorization,
resource limits, saved setups, and boundary-density/report tools. Frequency and
azimuth inputs display either a list or a sweep, according to the selected mode.
BoR uses aspect angles from +z in place of the 2D azimuth input.

The GHOST solver tab and GRIM Runs workspace share the preset selector and
advanced **2D execution and resources** controls. In Runs, Advanced Settings
also contains cluster allocation and BoR body orientation. Save a **2D run setup**
to carry the numerical settings between the workspaces.
Geometry paths, output paths, and cluster connection settings remain separate.

| Geometry preset | Kernel evaluation | Factorization | Sweep basis reuse |
| --- | --- | --- | --- |
| Small Geometry (No RAM Optimization) | Reference | Dense LU | Off |
| Large Geometry (RAM Optimization), default | CPU streaming | Compressed assembly | Automatic |
| Balanced | CPU streaming | Dense LU | Automatic |

All three use double precision, up to four assembly threads and two BLAS threads,
and 256 angles per batch. Large Geometry uses an 8192 MiB compressed payload
allowance. Small Geometry disables optional compression; normal allocation
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

New 2D monostatic runs in the desktops and local/HPC drivers default to the
large-sweep preset: compressed assembly, CPU streaming kernels, double precision,
8192 MiB compressed storage, four assembly threads, two BLAS threads, automatic
sweep basis reuse, and 256 angles per batch. Thread defaults are reduced on hosts
with fewer CPUs. RAM uses current available memory, temporary files use the
execution host's system temporary directory, and mesh certification stays enabled.
**Use efficient defaults** reapplies the resource preset to an existing setup.

This preset prioritizes RAM for large sweeps. The measured 2 GHz airfoil
certificate used 77.8% less sampled RAM with compression and took 36.7% longer
than dense LU. Dense can be faster when its matrices fit comfortably in RAM.
The 256-angle batch retains basis reuse: earlier airfoil tests at 256/128/64
angles took 41.95/42.42/43.11 seconds and used 1244/1195/1181 MiB.
Smaller batches trade time for a modest workspace reduction in that test.

| Control | Behavior |
| --- | --- |
| Dense LU | Assembles the full dense operator. |
| Hierarchical factor (dense assembly) | Assembles a dense operator and compresses its factorization. |
| Compressed assembly (low RAM) | Default for new 2D monostatic runs. Builds compressed tiles from geometry; requires CPU streaming kernels and double precision. |
| Hierarchical with dense fallback | Can fall back to dense LU; unsuitable when a dense allocation cannot fit. |
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

For the supplied airfoil's 10 GHz, 0-360 by 1 degree qualification configuration,
choose compressed assembly, 8192 MiB compressed storage, four assembly threads,
two BLAS threads, and mesh certification. Leave RAM at Available memory or
enter an admission budget appropriate to the execution host. The 8192 MiB
setting is a numeric payload allowance, not a prediction of process RAM.

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
    solver_method="experimental_cpu", max_panels=50000,
    execution_options={
        "factorization": "compressed",
        "compressed_storage_mib": 8192,
        "assembly_threads": 4,
        "blas_threads": 2,
        "temporary_directory": "",
    },
)
```

In a 2D driver's JSON `settings` object use:

```json
{
  "SOLVER_METHOD": "experimental_cpu",
  "LU_PRECISION": "double",
  "EXECUTION_OPTIONS": {
    "factorization": "compressed",
    "compressed_storage_mib": 8192,
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
Low-level Python entry points retain their existing defaults; use
`solver_method="experimental_cpu", execution_options=efficient_defaults()` to
select this preset from `ghost_backend.execution.options`. Explicit reference
kernel or mixed-precision driver configurations continue to use compatible dense
settings unless another supported profile is supplied.

Callers without a profile retain environment-based selection. Drivers capture
those values when planning a run. An explicit profile takes precedence over
the corresponding environment variables. A nested solve cannot replace an
active profile with different settings.

BLAS settings are process-wide. Configured solve sections are serialized within
one Python process to prevent competing native thread limits. Batch workers are
separate processes and can still execute concurrently. Install the updated
dependencies: desktop `threadpoolctl==3.6.0`; Python 3.6 HPC
`threadpoolctl==2.2.0`. The HPC environment checker exercises native thread
limiting and a complex LU solve without importing Qt.

## Performance regression checks

From the repository root:

```powershell
.venv/Scripts/python.exe tools/GHOST/tests/benchmark_execution.py --output baseline.json
.venv/Scripts/python.exe tools/GHOST/tests/benchmark_execution.py --output candidate.json --baseline baseline.json
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
