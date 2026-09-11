**GHOST 2D efficiency implementation — September 10, 2026**

Follow-up: [hardening, faster streamed assembly, and native-material limits](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/experiments/ghost_hardening_20260910/REPORT.md). The measurements below preserve this earlier implementation's baseline.

The measured assembly, sweep, compression-scan and residual improvements are now in the production ghost_backend. A separate working experiment also assembles the airfoil directly into compressed storage without first allocating global dense A. That experimental path has passed both polarizations at 1 and 2 GHz; it is not yet selected by the GUI or public solver.

**Production measurements**

The supplied airfoil was solved at 2 GHz, inches, 361 azimuths including 0 and 360, both polarizations, with 5,742 unknowns. These complete runs include condition estimation, physical-RHS residual checks and field projection. They use the experimental CPU kernel option, two BLAS threads and four assembly threads. The saved baseline contains the previous implemented improvements, before this update.

| Case | Saved baseline | Updated | Result |
|---|---:|---:|---|
| Airfoil, dense factorization, median of two runs | 47.59 s | 42.03 s | About 11.7% faster |
| Airfoil, dense factorization, median sampled RSS | 1,213 MiB | 1,240 MiB | About 2% more workspace for the retained sweep basis and joint kernels |
| Airfoil, updated hierarchical factorization, one run | — | 41.73 s; 811 MiB | About 33% less peak RAM than the saved dense baseline |
| IBC, 4,096 nodes, dense factorization, one run each | 16.41 s; 867 MiB | 16.04 s; 701 MiB | About 19% less peak RAM and 2% faster |

The airfoil's maximum complex-field difference from the saved baseline was 5.60e-13 relative to the corresponding reference channel's peak. The IBC difference was below 6.76e-15. Sampled RSS is whole-process memory at 50 ms intervals, not a guaranteed allocation maximum. These are base-mesh comparisons, not a new mesh-convergence certificate. Timing varies with machine load; the listed measurements are not 10 GHz projections.

**Changes now in the solver**

- S and K/K-prime builders accept writable final destinations, including strided blocks of A. Robin contributions scatter directly into their final rows. PEC TE and sheet systems adopt owned operator storage. Homogeneous dielectric writes all four blocks directly, including W. Destination validation rejects incompatible shape, dtype and read-only arrays; destination-backed operators bypass the immutable cache.
- General two-density thin layers now assemble S/K into the top blocks of A, copy the required K transpose before changing K, then apply sparse mass-inverse operations in bounded columns. W is assembled into its final block. The one-density layer adopts S. These changes preserve the existing material laws, endpoint equations and quadrature.
- Screened lossy kernel tables prefer degree 12 with smaller intervals and evaluate S/K jointly where both are needed. The same 2e-13 construction screen remains. A rejected degree-12 table may use checked degree 16; rejection of both uses exact kernels. Highly attenuating domains retain a partial screened table and evaluate only out-of-domain distances exactly. Joint coefficient storage is included in the table budget.
- Sweep compression uses an incrementally extended pivoted-QR basis shared across the bounded angle batches. Both its basis and solved basis are capped by the configured batch capacity, at most 256 columns. The 2 GHz airfoil solved 51 basis columns per polarization for 361 illuminations; later batches reused the first batch's work. Every recovered physical RHS is checked against the original A. Failed columns are solved directly, exact zero illuminations remain zero, and rank/QR failures retain the checked ordinary solve.
- Hierarchical ACA construction no longer scans complete off-diagonal blocks every 16 updates. Small updates, unresolved pivots and the rank limit trigger complete checks; every accepted block still passes full-coefficient validation. Rank/storage limits, original-A refinement and dense-fallback policy remain.
- Hierarchical and mixed refinement can return the residual belonging to the returned solution. DenseFactor consumes it immediately instead of multiplying A by the same solution again. No persistent residual/RHS cache is introduced. Transpose and adjoint calls preserve their existing default return contract.
- Matrix infinity-norm scans follow column blocks for Fortran storage. Hierarchical construction reuses the outer factor's already-computed norm. Memory planning removes the deleted formulation-specific operator slots while retaining solve, thread, table and bounded workspace allowances.

The ownership, norm and sweep changes apply to the shared CPU pipeline. Screened tables apply when `experimental_cpu` is selected. Hierarchical factorization remains an explicit option; ordinary dense LU remains the default. No new material approximation or attenuation cutoff is enabled in production.

**Using the updated code**

Restart GHOST so it imports the changed ghost_backend modules. For the measured kernel improvements, select the experimental CPU solver and Double LU precision. The lower-memory hierarchical factor is selected in the environment before starting the GUI or driver:

```powershell
$env:GHOST_DENSE_BACKEND = 'cpu'
$env:GHOST_CPU_FACTORIZATION = 'hierarchical'
$env:GHOST_CPU_RHS_COMPRESSION = 'auto'
```

An already-running GUI does not inherit later environment changes. `hierarchical` refuses a case that exceeds its rank/storage/accuracy limits; `auto` reserves capacity for dense fallback. Keep planning and worker environments consistent on HPC. Existing source fingerprints automatically invalidate results made with an earlier ghost_backend. Copy the full ghost_backend when updating another installation.

**Direct compressed assembly experiment**

The new [implementation experiments](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/experiments/ghost_implementation_20260910/streamed_probe.py) extend the previous coefficient oracle into an end-to-end geometry-to-compressed solve:

1. Prepare regional routes, weighted basis integrals and sparse jump terms once.
2. Partition equation coordinates spatially and assemble bounded coefficient tiles from the existing Galerkin kernels. Each global coefficient position is requested once during this assembly stage; no full dense A is allocated. Shared element support can still repeat some quadrature across tile boundaries.
3. Compress each completed tile when that reduces storage; retain difficult tiles densely. Track compression error by absolute row sums and enforce a retained tile-storage cap.
4. Build a reusable compressed operator and inverse from those inexpensive stored tiles. Repeated factor-construction queries no longer repeat electromagnetic assembly. Release the geometry oracle and tile store before solving.
5. Solve an incident basis and check all 361 recovered illuminations. Qualification uses both an error allowance for the two compression stages and an independent saved original-A residual/field comparison.

| Experimental airfoil construction | 1 GHz TE / TM | 2 GHz TE / TM |
|---|---:|---:|
| Geometry assembly | 18.46 / 18.67 s | 46.44 / 46.26 s |
| Compressed inverse construction | 1.84 / 1.58 s | 5.77 / 6.45 s |
| Final operator+inverse numerical payload | 72.25 / 68.94 MB | 164.64 / 157.09 MB |
| Sampled construction RSS | 204 / 203 MiB | 370 / 357 MiB |
| Largest assembly tile including its error array | 0.83 MB | 3.09 MB |

Payload MB are decimal; RSS MiB are binary. Construction RSS excludes the later independent reference validation and is not the peak of a production two-polarization solve. The 2 GHz original-A backward errors were below 2.86e-14, error allowances below 9.45e-14, and field differences below 1.08e-12. Geometry and intermediate tile storage were released before the solve. Independent small layered/coated/mixed tests also checked normal, transpose and adjoint operator actions, arbitrary requested block ordering, the storage cap and absence of a full-system allocation during tile assembly.

This is a material RAM advance over the old experiment, which started with an already assembled dense A. It is currently slower than the production solve: the two polarizations are built separately, and tile SVD/plan traversal still costs time. The measurements establish a working lower-memory construction path, not a production speed improvement or a full 10 GHz result.

**Controlled attenuation experiment**

The experimental oracle can omit an entire routed far contribution when all its basis supports exceed the requested attenuation distance. It accumulates an analytic envelope for every omitted contribution, including source/observer basis integrals, material row weights and weighted Robin terms. Lossless, growing and unsupported kernel domains are retained.

For `k = real(k) - i alpha`, the Hankel/modified-Bessel connection and integral representation give `|H_nu^(2)(k r)| <= 2 K_nu(alpha r)/pi` for the used real orders. Gaussian bounds on that integral provide conservative S and K envelopes for `alpha r > 1`; the experiment uses distance 32 and a floating-point margin. This derivation uses [NIST DLMF 10.27.8](https://dlmf.nist.gov/10.27#E8) and [10.32.9](https://dlmf.nist.gov/10.32#E9). Independent kernel samples test both strongly oscillating and strongly attenuating cases.

This does not make distance 32 a universal production tolerance. The bound is carried through compression and checked for the solved system. The experiment remains limited to regional S/K formulations; sheet/thin-layer W integration, production condition estimation, public dispatch, resource planning, certification/cancellation behavior and broader material/geometry qualification still need integration before replacing the production backend. No accuracy gate was bypassed to enable this path in the GUI.

**Validation and reproducibility**

Production full-sweep comparisons cover 14 fixture families: PEC, IBC, PEC+IBC, lossless/lossy/magnetic dielectric, coated/layered bodies, dielectric+PEC, sheet, sheet+PEC, thin and magnetic thin layers, and transparent thin layers. Every discrete quality gate passed; the maximum field difference from earlier independent fixture results was 1.86e-14 with dense LU. All 14 families also passed with hierarchical factorization selected; their maximum field difference was 2.69e-11. The original 1e-12 backward-error release limit remains unchanged. Fixture coverage does not imply support for combinations the solver explicitly rejects.

All **135 production regression tests and two experimental tests passed**. They exercise destination ownership/accumulation, exact residual evidence and lifetimes, incremental basis reuse/extension, per-column fallback, zero illuminations, partial-domain kernel fallback, storage limits, cancellation, transpose/adjoint solves and existing material/physical acceptance cases. The final test log and [validated benchmark results](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/experiments/ghost_implementation_20260910/validated-results.json) are in the experiment directory. The local runtime was Python 3.12.14, NumPy 2.5.2 and SciPy 1.18.1. Changed ghost_backend files also passed Python 3.6 syntax parsing. Legacy-compatible APIs were retained; an actual remote Python 3.6 HPC deployment was not exercised.

`baseline_backend/` is a local ignored copy taken before edits; `baseline-hashes.json` records it. Each production benchmark includes the ghost_backend source hashes it used. `benchmark.py` runs a fresh process against either saved or updated sources. The experimental compressed path is invoked with:

```powershell
# Run from experiments/ghost_implementation_20260910.
../../.venv/Scripts/python.exe benchmark.py --materials
../../.venv/Scripts/python.exe benchmark.py --baseline
../../.venv/Scripts/python.exe benchmark.py
../../.venv/Scripts/python.exe validate_results.py
../../.venv/Scripts/python.exe -m unittest test_streamed_operator -q
../../.venv/Scripts/python.exe streamed_probe.py --frequency 2 --tile 512 --cut 32
```

The compressed probe's independent comparison requires the earlier captured systems under `experiments/ghost_ram_time_20260910/systems/`. Its output explicitly records acceptance or rejection; it does not export production results or alter solver defaults. The full 10 GHz matrix and mesh-certification solve were not run in this implementation pass.
