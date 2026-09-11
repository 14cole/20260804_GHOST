# Saved GHOST execution profiles and progress

Implemented shared 2D execution/resource controls in GHOST and GRIM Runs, portable settings for local/HPC workers, live stage/RAM status, and a repeatable performance regression suite.

## Implementation

- Saved profiles specify factorization, compressed payload allowance, RAM admission budget, temporary disk directory, assembly/BLAS threads, RHS reuse, and angle batch size. Advanced tile/quadrature settings survive round trips.
- Public 2D solver APIs accept `execution_options`. Active profiles are scoped per run and propagated to tiled assembly threads. An explicit profile overrides corresponding launch variables.
- Setup version 2 includes the profile. Version 1 migrates to explicit dense/default resources. Incompatible or malformed settings are rejected before altering loaded controls.
- Local/HPC drivers capture complete profiles into manifests, forward them to fresh workers, and include them in provenance and result metadata. Contradictory profile/legacy driver values are rejected.
- Native BLAS limits apply to already-loaded NumPy/SciPy libraries. Scheduler CPU reservations cover both assembly and BLAS; effective assembly threads cannot exceed the worker allocation. Auto uses one assembly thread on the desktop.
- GUI status and worker logs report elapsed solve time, current stage, and sampled process RSS; base/refined mesh phases are distinct. Exported metadata includes stage timings and requested/effective thread settings.
- Fixed atomic profile switching in the GUI, exact saved RAM-budget round trips, restoration after failures, and fresh-worker startup with conflicting environment variables. BoR ignores these 2D-only controls.

## Performance measurements

PEC and mixed PEC/dielectric fixtures, 256 panels per boundary, 0.6 GHz, 361 azimuths, both polarizations, survey mode, one assembly/BLAS thread, 64 MiB compressed cap. Three fresh processes per case. Solver time excludes imports. Baseline is the package snapshot taken before these integration changes.

| Case | Before time (s) | After time (s) | Time change | Before peak RSS (MiB) | After peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| pec/dense | 0.318 | 0.309 | -2.9% | 100.0 | 93.8 |
| pec/compressed | 0.409 | 0.399 | -2.3% | 104.7 | 104.5 |
| mixed/dense | 1.443 | 1.459 | +1.1% | 127.2 | 124.3 |
| mixed/compressed | 1.742 | 1.724 | -1.0% | 120.0 | 116.3 |

These changes do not introduce another numerical algorithm. The measured changes are small-run variation, not a claimed speedup. Every before/after complex field matched exactly. Compressed-versus-dense field errors also passed the benchmark threshold. RSS is sampled every 50 ms and may miss short peaks.

Machine and package details, source/workload hashes, all repetitions, stage timings, and complex fields are retained in the benchmark JSON files. Comparisons reject different workloads, profiles, or runtime environments. Default gates are +25% median wall time, +15% median RSS, and 1e-8 maximum normalized complex-field error.

## Verification

- `ghost-full-tests.log`: Ran 654 tests in 942.751s; OK (skipped=1)
- `grim-full-tests.log`: Ran 1053 tests in 76.535s; OK (skipped=1)
- `final-profile-tests.log`: Ran 23 tests in 39.577s; OK
- `python36-final-tests.log`: Ran 27 tests in 6.659s; OK
- `python36-worker.log`: Ran 1 test in 1.688s; OK
- `captured-environment-worker.log`: Ran 1 test in 2.236s; OK
- `final-gui-release-tests.log`: Ran 66 tests in 17.342s; OK
- `requirements-tests.log`: Ran 13 tests in 0.163s; OK
- `hpc-scheduling.log`: 0 failure(s)
- `local-drivers.log`: 30 passed, 0 failed

Additional checks: source/wheel import inventories, ASCII-transfer compatibility, desktop startup diagnostics, Python 3.6 native BLAS thread control and complex LU, and offscreen GUI layout review.

The benchmark CLI also completed a 64-panel-per-boundary certified smoke run for PEC/mixed and dense/compressed combinations. All four mesh certificates passed; see `certified-benchmark-smoke.json`.

Numerical profile tests cover PEC, IBC, lossy dielectric, magnetic, coated, layered, and mixed materials. Lifecycle tests cover nested/threaded scope isolation, cleanup after failures, compressed spooling, physical fields, and PEC mesh certification. Fresh worker tests block Qt imports and deliberately conflict with the saved profile through the launch environment.

## Use and limits

See [the run profile guide](../../tools/GHOST/RUN_PROFILES.md). The new controls are under **2D execution and resources** in the solver form and Runs workspace. Save a 2D setup for transfer.

A loadable `airfoil-10GHz.run.json` is included beside this report: 10 GHz, 0-360 by 1 degree, inches, compressed assembly, an 8192 MiB numeric payload allowance, four assembly threads, two BLAS threads, and mesh certification. Select the airfoil geometry separately when loading it.

The current desktop environment has threadpoolctl 3.6.0 installed. Both workspace Python 3.6 validation environments have 2.2.0. Separate desktops/compute installations need the updated dependency files and complete Backend package.

RAM admission is an estimate checked against the configured budget and available memory; it is not an OS-enforced cap. The compressed cap covers numeric payload, and temporary disk is separate. Native thread limits are process-wide, so configured solves in one process serialize their limited sections; separate batch worker processes remain concurrent.

No new full 10 GHz airfoil qualification or live SLURM submission was performed for this integration. The prior airfoil measurement is documented in COMPRESSED_CPU.md. This work exposes and preserves that configuration; it does not replace its physical validation.
