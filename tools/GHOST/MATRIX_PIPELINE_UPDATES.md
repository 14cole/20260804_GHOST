# 2-D matrix pipeline updates — September 10, 2026

Implemented the validated storage, assembly-reuse, and solve-pipeline updates
from the solver-wide audit. The supported physical formulations, mesh density,
complex-field convention, material laws, requested angles, and numerical
acceptance gates are preserved.

## Measured result

The supplied airfoil, at **1 GHz**, inches, 361 incident angles, both
polarizations, experimental CPU, and condition diagnostics:

| Measurement | Before this update | Updated |
|---|---:|---:|
| Elapsed time | 32.82 s | 20.27 s |
| Operator assembly time | 30.88 s | 18.49 s |
| Sampled process peak RSS | 478.6 MiB | 413.5 MiB |
| LU factorizations | 2 | 2 |
| RHS solve batches | 4 | 4 |
| Peak-scaled complex-field difference | — | 2.29 × 10⁻¹³ |

This is a 38% elapsed-time reduction in one controlled before/after run on
this machine. The mesh still has 1,938 panels and 2,964 system unknowns.
The instrumentation used two BLAS threads and one assembly thread. RSS is
sampled every 50 ms, so it is not an exact allocation high-water mark.

The full **10 GHz solve was not run**. Exact mesh/resource planning gives:

| Storage or planned peak | Before this update | Updated |
|---|---:|---:|
| Compact regional operator payload | 12.381 GiB | 9.937 GiB |
| Dense CPU, single angle | 24.77 GiB | 24.12 GiB |
| Dense CPU, 361 angles | 25.95 GiB | 25.41 GiB |
| Experimental CPU, 361 angles | 25.51 GiB | 25.51 GiB |
| Dense CPU, 361 angles, 1.5× certification fine mesh | 57.12 GiB | 56.32 GiB |

These follow the earlier compact-storage fix that reduced the original
approximately 87 GiB prediction. They are raw allocation estimates, before
the HPC scheduler's safety factor and process allowance. Planning allowed
100,000 panels; the existing default panel limit is unchanged.

At 10 GHz the unmodified geometry still produces 18,726 panels, 18,732 nodes,
and 28,406 unknowns. **The system and its double-precision LU alone use
24.048 GiB.** This explains why eliminating additional operators improves
assembly storage substantially while changing the final peak much less, and
why one azimuth still needs nearly as much memory as a sweep.

## Implemented changes

1. **One common checked CPU factorization and bounded solve path.**
   `dense_factor.py` owns the matrix norm, factorization, condition estimate,
   refinement, backward-error check, and relative residual. The same residual
   is consumed by the caller. Corrections reuse the factor. A failed mixed
   factor is released before double fallback, including traceback references.

2. **Bounded sweeps for every supported CPU formulation.**
   `boundary_fields.py` batches RHS construction, solves and projection for
   PEC/IBC, dielectric, multi-region, sheet, sheet+PEC and thin-layer systems.
   Dense CPU and mixed-precision CPU now use the same default 256-column cap
   as experimental CPU. Bistatic output is projected directly in batches;
   density export requests no unused far field. Final output still grows
   with the requested angular grid. Optional GPU solves retain their existing
   full-RHS interface; GPU/cluster performance was not measured.

3. **Sparse/local mass terms and earlier operator release.**
   Active 2-D formulations no longer allocate global dense mass or weighted
   mass matrices. Operators are copied into their final system blocks and
   released before factorization. Pure TM PEC uses S directly and never
   builds K′ or a mass matrix. The dielectric assembly avoids a full temporary
   just to negate its S block.

4. **Independent S/K′ row plans and assembly reuse.**
   Multi-region assembly omits conductor rows that the selected polarization
   never consumes. Mixed PEC/IBC Robin assembly combines the required weighted
   S, K′ and PEC S rows in one kernel traversal. Quadrature grading considers
   the required pair union.

   A scoped session retains one owned system between adjacent TE and TM
   solves. Single-dielectric systems update their constitutive block. In
   multi-region systems, transmission blocks are rescaled, existing trace
   blocks supply reciprocal cross-interface S coefficients, and only changed
   conductor terms are integrated. IBC systems update the weighted S change.
   Reuse requires matching geometry, material records and quadrature;
   multi-region reciprocity reuse requires equal source/observation orders.
   Source operator caches are disabled in active formulation solves.

5. **Certification ordered by mesh.**
   Canonical certified runs perform base TE/TM, then fine TE/TM. Each channel
   retains its own discrete-system and complex mesh-convergence checks.
   Frequency sweeps complete one frequency before advancing. No LU is shared
   between physically different systems.

6. **Less incident-load work.**
   Exact straight-panel moments construct value, derivative and weighted
   Robin loads together. Only requested traces are allocated. Multi-region
   excitation skips interfaces that do not border the incident exterior.

7. **Sparse thin-layer coefficient operations.**
   The general thin-layer path applies `C @ solve(M, S_or_K)` in bounded
   column blocks with sparse C and sparse mass LU. It no longer builds dense
   `C M^-1` or performs the two associated cubic dense products. The exact
   zero-field-jump elimination still selects an N system when applicable;
   the resource planner now recognizes that branch.

8. **Default touching-pair integration and telemetry.**
   The existing vectorized touching-pair S integration is used by the default
   path, including hypersingular assembly. Metadata distinguishes actual
   factorizations, RHS batches and columns, and system reuse. Certified runs
   include separate base/fine work counters. CPU cancellation is checked
   between batches and the scoped retained system is cleared on exit.

## Validation

- Saved the complete pre-update Backend and ran separate-process comparisons.
- Compared 14 material configurations at 361 angles, TE/TM, monostatic and
  bistatic. Maximum peak-scaled complex-field differences were 1.37×10⁻¹³
  and 9.21×10⁻¹⁴ respectively. Experimental CPU also passed the material
  comparison and existing screened-kernel tests.
- The main 118-test regression run passed: physical cylinder references,
  PEC interior resonance, layered and magnetic media, junctions, reciprocity,
  sheet endpoints, thin-layer bulk references, certification, export,
  compaction, memory gates, CPU precision and cancellation.
- Additional pipeline and real headless-worker tests passed, including
  per-batch/factor counters, default CPU cancellation, multi-frequency
  bistatic output, density-only projection omission, weak-reference operator
  lifetime checks, and fresh-versus-reused matrix comparisons.
- The standalone HPC scheduling suite passed with zero failures, including
  exact resource planning, local worker execution, output verification and
  resume. Both modern Python 3.12 and actual Python 3.6.8 with
  NumPy 1.14.3/SciPy 1.0.0 passed the pipeline/experimental checks.

Evidence: [comparison results](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/solver-wide-updates-2026-09-10/comparison-results.json),
[resource results](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/solver-wide-updates-2026-09-10/resource-results.json),
[comparison runner](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/solver-wide-updates-2026-09-10/verify.py),
[regression tests](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/tools/GHOST/tests/test_solver_matrix_pipeline.py).

## Subsequent implementation

The next update adds direct regional scatter, packing of both element axes,
checked RHS compression, and an optional hierarchical factorization. See
[implementation and measurements](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/tools/GHOST/COMPUTATION_RAM_UPDATES.md)
for selection, validation, and the remaining dense-matrix storage limit.

See [CPU method usage](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/tools/GHOST/EXPERIMENTAL_CPU.md)
and [memory/batch settings](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/tools/GHOST/COMPACT_2D_MEMORY.md).
