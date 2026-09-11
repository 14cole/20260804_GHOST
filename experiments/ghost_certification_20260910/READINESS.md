# Readiness follow-up

The memory forecasts quoted in this historical readiness check are superseded
by [MEMORY_ESTIMATION.md](MEMORY_ESTIMATION.md): the 361-angle airfoil reservation
is now approximately 7.1 GiB instead of 16.0 GiB.

The compressed RCS path is ready for user testing with the launch configuration
in [COMPRESSED_CPU.md](../../tools/GHOST/COMPRESSED_CPU.md). This follow-up found
and fixed two GUI integration issues after the main numerical qualification.

## Fixes

1. The GUI inherited a 20,000-panel limit. The supplied airfoil at 10 GHz has
   18,726 base panels but 28,138 certificate panels, so the fine solve would
   have stopped after completing the base solve. Compressed GUI runs now pass
   a 100,000-panel allowance, matching the qualification harness, to the 2D
   certified/survey and density entry points. Dense GUI defaults remain 20,000.
   The existing memory admission and numerical gates still apply. Local/HPC
   drivers already configure their own allowance, normally 50,000 panels.
2. PEC/IBC and single-dielectric boundary-density diagnostics passed a
   `StreamedOperator` to `DenseFactor`, raising `TypeError`. They now select
   the compressed factorizer. An owned assembly session propagates cancellation
   into bounded diagnostic assembly and cleans up its state. Density plots
   do not compute unused far fields.

The solver tooltip no longer describes obsolete material fallbacks. The launch
guide now starts with the airfoil's 8192 MiB payload allowance, includes the
benchmark thread settings, and launches GHOST from the shell that sets them.
Double-clicking a separate normal launcher does not inherit those shell changes.

## Verification

- 46 desktop tests pass: compressed operator/solve contracts, seven material
  families' density comparisons in TE and TM, cancellation, GUI selection,
  worker execution, density display and GUI entry points. Density fields agree
  with dense references within 1e-10 of each reference peak. Tests prevent any
  dense factor or unnecessary far-field computation in compressed diagnostics.
- The mesh-limit regression builds a real 28,000-panel refined mesh through
  the worker's solver arguments. Dense retains its 20,000-panel rejection;
  compressed admits the refinement.
- 18 compressed tests pass on each legacy runtime: Python 3.6 with
  NumPy 1.19.5/SciPy 1.5.4 and NumPy 1.14.3/SciPy 1.0.0. Total: 82 final test
  executions across the three runtimes.
- The final GUI smoke test loads the actual `airfoil.geo` through the GUI file
  loader, runs setup validation, and executes its Qt worker at 1 GHz with
  361 angles and both polarizations. It passes the base/fine certificate and
  quality gate, reports `compressed_experimental_cpu`, and reaches 100% with
  monotonic progress. Worker execution took 69.46 s; this is an integration
  smoke test, not a new performance comparison.
- Resource planning at 10 GHz uses the final GUI panel allowance and admits
  all 28,138 fine panels / 42,686 unknowns per polarization. Scheduler forecasts
  are 16.0075 GiB for one angle and 16.0093 GiB for 361 angles. Removing its
  default 0.6 GiB plus 35% padding gives about 11.4 GiB. The forecast includes
  the allowed retained payload and conservative construction workspaces; it
  is not a prediction that exactly that much RAM will be allocated.
- Changed Backend modules parse with Python 3.6 grammar; `git diff --check`
  passes. Only `rcs_solver.py` (density diagnostics) and `solver_tab.py` changed
  relative to the main qualification's Backend hashes. The final smoke result
  records the new source hashes.

Evidence: `readiness-final-tests.log`, `readiness-python36-tests.log`,
`readiness-oldest-tests.log`, `readiness-probe.log`, `readiness-probe.json` and
`readiness-diff-check.log`. `readiness_probe.py` reproduces the actual-file
worker and resource checks. The earlier smoke result, before correcting the
GUI panel allowance, is retained as `readiness-initial-probe.json`.

## Remaining practical limits

There is no known blocker for the requested compressed monostatic RCS test.
Launch it with the documented environment, select CPU streaming and Double,
and keep mesh certification enabled. The normal launcher defaults to dense
without that environment.

Progress remains coarse during long assembly and factorization phases. A
stationary percentage can therefore accompany active computation. The 10 GHz
qualification's temporary polarization file peaked at approximately 1.58 GiB;
allow space in the local temporary directory. Its measured 3.30 GiB process
peak is a measurement, not a hard RAM limit.

The full 10 GHz solve was not repeated after these GUI/diagnostic fixes; its
53 min 34 s result and numerical qualification retain the scope in
[REPORT.md](REPORT.md). Final-source 1 GHz worker execution and exact 10 GHz
mesh planning cover the integration changes. Manual desktop interaction is
still the user's acceptance test. Further H2/FMM and inverse-query optimizations
are future work, not prerequisites for this test.
