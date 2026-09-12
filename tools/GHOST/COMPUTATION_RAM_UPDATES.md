# Further 2D solve time and RAM reductions

Current execution controls and bounded QR sweep behavior are documented in
[TWOD_PIPELINE.md](TWOD_PIPELINE.md) and [COMPRESSED_CPU.md](COMPRESSED_CPU.md).
The measurements and SVD description below describe an earlier implementation.

Implemented September 10, 2026. These changes follow MATRIX_PIPELINE_UPDATES.md.
They preserve the mesh, linear Galerkin basis, material equations, requested
angle grid, condition checks, and base/fine certification requirements.

## Measured airfoil result

The supplied airfoil, inches, 2 GHz, 361 azimuths (0 through 360), both
polarizations, 5,742 unknowns, experimental CPU kernels, double precision:

| Mode | Complete solve | Sampled peak process RAM | Assembly time |
|---|---:|---:|---:|
| Saved backend before this update | 56.90 s | 1.189 GiB | 49.00 s |
| Updated default factorization | 50.10 s | 1.182 GiB | 42.65 s |
| Updated hierarchical factorization | 54.65 s | 0.766 GiB | 42.75 s |

The default completed about 12% faster. The hierarchical option reduced peak
RAM about 36% relative to the saved backend, while taking about 9% longer than
the updated dense solve. These are individual separate-process runs on this
machine, with two BLAS threads and one assembly thread; they are not timing
guarantees for other frequencies or hardware. RAM was sampled every 50 ms.

Sweep compression solved 64 basis right-hand sides instead of 361 per
polarization, reconstructing and checking all 361 physical right-hand sides.
The hierarchical factors occupied approximately 82.5 and 77.2 MiB versus
503.1 MiB for each full complex128 LU. Both updated modes agreed with the saved
baseline's complex fields within 6.5e-13 relative to the baseline peak field.
The benchmark checks the discrete system; it does not certify mesh convergence.

## What changed

- **Direct regional assembly into A.** Galerkin contributions go straight to
  their final equation rows and density columns. Separate regional S, K-prime,
  and weighted-S arrays are no longer retained. Equal-wavenumber regions still
  receive their individual material factors. The TE-to-TM conversion uses the
  same scatter mechanism for its required new terms. The old compact assembler
  remains an independent regression oracle and is outside the production path.
- **Both element axes are packed.** The assembler compares the work in a
  rectangular active-source/active-observer product with a symmetric traversal
  of their union. It preserves near-pair classification, quadrature rules,
  source masks, weighted integrals, and global node IDs. Threaded writes remain
  serialized at the existing accumulation lock.
- **Algebraic sweep compression.** Within the bounded angle batch, a row-scaled
  SVD identifies redundant incident loads. A factor solves the resulting basis
  and reconstructs the requested solutions. Acceptance checks the reconstructed
  RHS and the residual against the original A and every original RHS. Poor
  compression, SVD failure, or failed reconstruction checks trigger the ordinary
  batch solve with the existing factor. No angular interpolation is used.
- **Optional hierarchical inverse.** Spatial ordering, checked off-diagonal
  low-rank blocks, small dense leaf factors, and recursive correction systems
  replace the full-size LU. Construction uses preallocated rank workspaces.
  Every accepted off-diagonal block is checked over all its coefficients in
  bounded tiles. Physical solves and transpose/adjoint solves refine against
  the original matrix. Condition estimation uses that checked inverse.
- **Zero-contrast thin layers.** A validated layer with epsilon=mu=1 returns
  the exact zero scattered field without assembling operators, mass matrices,
  excitation matrices, or an LU. Geometry, passivity, thickness, orientation,
  curvature, and finite-angle validation still run.
- **Memory accounting and provenance.** The planner recognizes direct scatter
  and the hierarchical storage cap. Strict mode cannot silently allocate a full
  LU after rejection. Automatic fallback releases the rejected hierarchy before
  dense allocation. Result metadata reports compression counts, factor bytes,
  block errors, refinement, and fallback reasons. Runtime fingerprints include
  both new mode selections, preventing reuse across different settings.

## Selection

The assembly improvements are automatic for both CPU methods. Dense LU remains
the default factorization. Sweep compression defaults to `auto`: batches with
at least 32 columns and systems with at least 512 unknowns are considered.

For the low-memory option, set these variables **before starting the GUI or
driver from that PowerShell session**, and select Double LU precision:

```powershell
$env:GHOST_DENSE_BACKEND = 'cpu'
$env:GHOST_CPU_FACTORIZATION = 'hierarchical'
$env:GHOST_CPU_RHS_COMPRESSION = 'auto'
```

An already-running GUI does not inherit later environment changes. For an HPC
run, use the same environment at planning/submission and in the worker jobs.
Copy the complete updated ghost_backend to separate installations. These settings
are environment options; they are not new GUI controls or driver JSON keys.

`GHOST_CPU_FACTORIZATION` accepts:

- `dense` (default): standard reusable LU.
- `compressed`: assemble compressed tiles directly from geometry and retain a
  checked compressed inverse, with no global dense fallback. See
  [COMPRESSED_CPU.md](COMPRESSED_CPU.md) for its separate payload cap and
  qualification results.
- `hierarchical`: enforce the bounded hierarchical factor; report a failure if
  the rank, storage, factorization, or exact residual checks cannot be satisfied.
- `auto`: try hierarchical construction for systems with at least 2,048
  unknowns; fall back to dense LU on construction, condition, or solve rejection.
  Its memory estimate reserves dense fallback capacity.

`GHOST_CPU_RHS_COMPRESSION` accepts `auto`, `on` (also try smaller systems),
and `off`. Batches under 32 columns use ordinary solves. Compression remains
subject to its rank and accuracy checks in every enabled mode. The existing
`GHOST_CPU_ANGLE_BATCH_SIZE` limit of 1 through 256 still applies.

The hierarchical factor is CPU/double only; explicit GPU or mixed precision
combinations are rejected. Sweep compression also works with ordinary CPU
mixed-precision factors, retaining their existing refinement and fallback.

## Full 10 GHz sizing

With the original material-driven mesh, the base system still has 28,406
unknowns. A alone occupies 12.02 GiB. The 9.94 GiB of compact regional operator
payload formerly used during assembly is eliminated. A and full LU continue
to dominate the default estimate.

| 10 GHz, experimental CPU, 361 angles | Estimated peak |
|---|---:|
| Base mesh, dense or automatic fallback | 25.51 GiB |
| Base mesh, strict hierarchical | 21.30 GiB |
| 1.5x certification fine mesh, dense | 56.41 GiB |
| 1.5x certification fine mesh, strict hierarchical | 46.91 GiB |

These are conservative allocation plans, including factor workspace and angle
batches, without extra scheduler safety multipliers. The fine mesh contains
42,686 unknowns. The calculations allow 100,000 panels; the configured panel
limit must also accommodate the requested mesh. **The complete 10 GHz solve
was not run.** Actual hierarchical rank, completion, and peak usage at that
frequency have not been established. The strict estimate is a budget for an
accepted factor, not a promise that every matrix compresses within that budget.

The hierarchical implementation retains dense A for accurate residual and
condition checks. It is not a matrix-free/FMM backend and does not make this
10 GHz base case fit in 16 GB of RAM. Fully eliminating quadratic storage
requires a separate compressed or matrix-free operator representation.
Hypersingular and S/K quadrature still have separate passes; merging those
passes requires reconciling their different near/far quadrature contracts.

## Regression coverage

The maintained [compression tests](ghost_backend/tests/test_solver_compression.py)
cover matrix reconstruction, solve accuracy, storage limits, fallback and
cancellation. Physical-reference tests remain in `ghost_backend/tests`.
Previously unsupported material combinations remain unsupported; these
execution options do not add missing physical coupling models.
