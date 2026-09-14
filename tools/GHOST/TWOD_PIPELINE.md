# 2D pipeline performance controls

The geometry editor and 2D solver share spatial filtering for intersection
candidates. Validation runs on a captured geometry in a cancellable background
worker; editing invalidates its pending findings. Endpoint lookup is built once,
and normal arrows/ticks use two plotting collections. On large models, material
labels are limited to the selected segment and up to 99 flagged rows.

## Automatic backend selection

New monostatic GUI/API/HPC runs select **Automatic** by default. Manual kernel,
factorization, and geometry presets remain in Advanced Settings. The saved
factorization value is `adaptive`; explicit saved choices retain their meaning,
including the older hierarchical-with-dense-fallback value `auto`.

The selector compares compatible dense, compressed, and FMM Galerkin backends
using geometry, materials, angle count, certification meshes, estimated memory,
and the shared computation-cost model. Available memory and any configured RAM
budget constrain admission; supported candidates with extra headroom are
preferred. Missing native libraries and unsupported FMM formulations exclude
FMM. Automatic remains a timing heuristic, not a fastest-runtime guarantee.

Numerical/storage failures may retry another admitted backend with unchanged
accuracy tolerances. Invalid geometry, cancellation, and failed physical mesh
convergence do not trigger these retries. Metadata records the choice,
forecasts, reasons, failures, and total execution time. Local/HPC scheduling
compares mixed backend batches within node CPU/RAM reservations; retry choices
must fit the original unit's reservation. See [batch presets](RUN_PROFILES.md)
and [qualification and limitations](AUTOMATIC_SOLVER.md).

## Local material sizing

**Mesh sizing -> Local material sizing (experimental)** sets
`mesh_strategy="local"` for 2D monostatic solves. The default remains `global`.
Local sizing uses each segment's adjacent materials, including thin-layer
material definitions. It limits relaxation to four times the global wavelength
and retains global sizing at corners, open ends, junctions, and boundaries
within one free-space wavelength of another segment. User-explicit positive N
keeps its original global gross-resolution safety floor.

The feature indicators are not a posteriori error estimates. Certified runs
still compare base/fine complex fields and enforce the existing quality gates.
If that mesh comparison fails, the run retries global sizing and records the
fallback. Survey runs remain uncertified. Local sizing does not change the
piecewise-linear input geometry or certify its geometric approximation error.

## Shared preparation and compressed tiles

Preflight and all frequencies in one desktop run share captured material tables
and cached topology checks. Independent runs reload their inputs. Frequency
operators, factors, and polarization sharing retain their existing bounded
lifetimes; no frequency-dependent matrix is reused at a different frequency.

Spatially separated coefficient tiles can propose a low-rank basis from 16
columns. Acceptance still checks the complete original tile against the same
error tolerance and carries entrywise error bounds into factor checks.
Unsuccessful proposals use the existing full QR path. QR now rejects a rank
that cannot save storage before constructing its Q columns. This reduces
compression overhead; it does not remove all-pairs coefficient evaluation.

## Full angle sweeps

The piecewise-linear Galerkin discretization is retained for full 0–180-degree
sweeps. One factorization per fixed system serves all requested incident angles.
RHS compression can now refresh its bounded illumination basis when the old
span cannot accept another useful batch. The factorization remains reusable;
the retained basis never exceeds the existing capacity of at most 256 columns.

For batches of at least 256 illuminations, 64 distributed columns may propose
a basis. Acceptance checks the complete batch at the existing `2e-15` QR
threshold, followed by the existing full reconstruction and original-system
backward-error checks. This does not interpolate or omit output angles. A
failed proposal uses full pivoted QR. Proposals stop when their attempts exceed
twice their acceptances for the factor, limiting repeated unsuccessful work.
An unprofitable decomposition falls
back to ordinary checked solves without first forming an unnecessary full Q.

This changes sweep execution only. It does not change the geometry, basis
functions, Galerkin testing, quadrature, matrix assembly, mesh certification,
or error limits.

## Desktop frequency checkpoints

Advanced Settings enables **Keep completed frequencies and resume matching
runs** by default for desktop 2D monostatic solves. Each completed frequency is
written atomically as a compressed NumPy archive of typed columns and JSON
metadata. Certified and survey checkpoints are separate. No pickle is loaded.

Reuse verifies geometry, material file contents, angles, numerical/quality
settings, precision, solver source, and the saved archive digest. Failed or
incomplete frequencies are recomputed. Canceling retains completed frequencies;
running the same inputs again resumes them. Uncheck the control to compute
without desktop checkpoints. The report gives the checkpoint directory beneath
the application's cache location and the reused frequency count. Cache files
can be removed when no longer needed; they are separate from exported GRIM
datasets and compressed-operator temporary spools.

During computation, completed frequency data is kept on disk. Final result
assembly loads one chunk at a time into the existing export/plot interface, so
the completed combined dataset still needs RAM. Current-run timing excludes
cached assembly/solve stages. Local/HPC drivers retain their existing verified
output reuse rather than using the desktop cache.

Regression coverage is maintained in `ghost_backend/tests/test_pipeline_performance.py`,
`test_pipeline_gui.py`, and `test_galerkin_sweeps.py`.
