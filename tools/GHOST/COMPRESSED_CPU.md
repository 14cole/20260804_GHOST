# Compressed CPU assembly and solves

The compressed CPU path assembles bounded tiles directly from geometry.
It keeps an accurate compressed operator and a smaller hierarchical inverse,
then refines and checks every requested physical illumination. It avoids the
global dense matrix and global dense LU. It is the default for new 2D monostatic
GUI and batch runs. Dense LU remains available for explicit profiles.

## Selection

In GHOST or GRIM Runs, open **2D execution and resources**. New runs select
**Compressed assembly (low RAM)**. **Use efficient defaults** restores its
8192 MiB storage/four assembly/two BLAS thread preset. The compatible CPU streaming kernels and
double precision are selected automatically. Save a 2D run setup to preserve
these choices for local or HPC execution. See [run profiles](RUN_PROFILES.md)
for all controls, Python examples, and driver JSON.

For the supplied 10 GHz airfoil, choose an 8192 MiB compressed storage cap,
four assembly threads, and two BLAS threads; keep mesh certification enabled.
The earlier 53 min 34 s measurement used these settings. This profile does not
require relaunching the application with environment variables.

API callers can pass `execution_options` to the certified solver entry point.
Environment-based selection remains available to callers without a profile.
Copy the complete updated Backend and install the updated requirements on
separate installations; HPC workers apply the profile stored in the manifest.

`GHOST_COMPRESSED_STORAGE_MIB` defaults to 2048 and must be an integer of at least
16. It caps retained numeric operator and inverse payload, including the paired
polarization's reserved payload. Geometry, Python objects, RHS batches, assembly
and factor-construction workspaces are additional. This is **not a process-RAM
limit**. Admission still checks estimated total memory against available RAM.
The 10 GHz airfoil qualification uses an 8192 MiB payload allowance.

For that supplied airfoil, use `GHOST_COMPRESSED_STORAGE_MIB=8192`. Its full
10 GHz, TE/TM, 0–360 by 1 degree mesh certificate completed in 53 min 34 s with
3.30 GiB sampled peak RSS. The default 2048 MiB retained-payload cap is too small
for that test's paired fine-mesh operator payload.

The compressed GUI permits up to 100,000 panels, matching the qualification
harness's allowance. The legacy dense GUI cap is 20,000, below this case's
28,138 fine-mesh panels. The memory and numerical quality gates still apply.
API callers must explicitly pass a sufficient `max_panels`; local/HPC drivers
use their configured `MAX_PANELS` (normally 50,000).

The second polarization's compressed tiles are stored in an owned temporary
file in the configured temporary directory (system temporary directory by default) while the first polarization is solved.
Allow local disk space for that payload. Files are checked for truncation and
checksum mismatches, then removed after loading or cleanup. Cancellation is
observed between bounded assembly, factor and solve operations.
The measured 10 GHz airfoil spool peaked at approximately 1.58 GiB. Live status reports the current stage, elapsed time, and sampled process RAM.
The percentage can remain fixed while assembly or factorization is running.

The revised forecast separates RAM, possible partner-disk payload, and the
enforced numeric storage limit. With the example settings, the airfoil's 10 GHz,
361-angle forecast is about **6.9 GiB in the solver / 7.1 GiB in the scheduler**,
down from 11.4 / 16.0 GiB. A single angle reserves about **5.7 GiB** in the
scheduler; factor construction still dominates that case. The measured 3.30 GiB
RSS remains a measurement, not a hard process limit.

For systems above 8192 unknowns, planning samples at most four production
operator tiles per spatial-separation band. It reports an operator-storage
forecast and sampling allowance. Smaller systems use a dense-storage ceiling
without planning-time coefficient assembly. The 10 GHz base/fine, VV/HH plan
takes roughly 30 seconds on the qualification machine. Within a solve, only
small numeric forecasts are cached; sampled operators are not retained.

RAM planning takes the maximum of assembly, factor construction, and angle-solve
phases. It includes inverse-tree limits, the actual angle-batch width, geometry,
tile/spool bookkeeping, and configured kernel-tile overrides. Disk data is not
added to resident RAM, and raising an unused payload cap does not raise a
sampled forecast. For compressed scheduling, `MEMORY_SAFETY` applies only to
uncertain sampled operator storage. The scheduler floor is a minimum total
reservation; it is not added again to process overhead. Dense planning keeps
its existing safety/floor behavior.

The forecast remains conservative about inverse ranks and repair workspaces.
Sampled storage can differ for unsampled tiles, so the forecast is not a hard
bound. The retained-payload checks and numerical acceptance criteria remain
enforced. See `experiments/ghost_certification_20260910/MEMORY_ESTIMATION.md`
for measurements and validation.

## Numerical behavior

- Native Robin PEC/IBC, single dielectric, regional coated/layered/mixed,
  sheet, sheet+PEC and supported transmitting thin-layer equations have direct
  geometry queries. Thin-layer mass inverses are sparse solves applied to bounded
  column strips. Existing unsupported material combinations remain unsupported.
- TE/TM share quadrature work where their geometry, wavenumber and primitive
  operator agree. Their constitutive coefficients, jumps and endpoint terms stay
  separate. Only one polarization's operator/inverse is resident for solving.
- Off-diagonal operator tiles use checked QR compression at `1e-14`. A tile stays
  dense if compression is not profitable or fails its full coefficient check.
  This local choice never permits a global dense fallback.
- Whole attenuated regional contributions may be omitted only with a computed
  absolute kernel envelope. Row and column sums of coefficient error accompany
  normal, transpose, adjoint, condition and physical-angle residual checks.
- The inverse starts with `1e-6` block tolerance. Accurate-operator refinement
  targets `3e-15` normwise backward error; stalled columns have bounded
  preconditioned GMRES repair. A rejected coarse inverse can be replaced once at
  `2e-10`. The final original-coefficient backward-error bound must be at most
  `1e-12`. Failure or budget exhaustion rejects the solve.
- Incident-basis reuse reduces the number of solved RHS columns. Every requested
  angle is reconstructed and checked. The normal batch limit remains 256;
  arbitrary longer angle grids stream across batches.
- The existing condition, residual and base/fine mesh-convergence gates remain
  enabled in the certified API. Condition equilibration takes one stored-tile
  pass using row maxima recorded during assembly.

## Qualification and interpretation

The qualification report and raw evidence are in
`experiments/ghost_certification_20260910/REPORT.md`. They identify actual tested
geometries, materials, frequencies, runtimes, source hashes and measurements.
Use that scope when interpreting certification. Numerical mesh convergence does
not certify the geometry approximation or the physical accuracy of a thin-layer
model. These limitations continue to appear in result metadata.

At 2 GHz, the supplied airfoil's full TE/TM, 361-angle base/fine certificate took
178.88 seconds and sampled 577 MiB peak RSS, compared with 130.86 seconds and
2603 MiB for the dense certificate. This saves about 78% of RAM but takes about
37% longer on the tested machine. It is useful when memory limits dominate;
compressed assembly is not an unconditional speed improvement.

Revert to the default with `GHOST_CPU_FACTORIZATION=dense`. The separate
`hierarchical` selection retains dense A, while `auto` may fall back to global
dense LU. The strict `compressed` selection does neither.

Regression entry points include `tests/test_compressed_path.py` and the
qualification folder's `test_native_queries.py`. The implementation imports only
Backend modules; it does not import the experiments folder.
