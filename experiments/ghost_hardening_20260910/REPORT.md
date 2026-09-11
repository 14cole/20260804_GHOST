# GHOST 2D follow-up: hardening and measured bottlenecks

September 10, 2026. This continues the previous efficiency implementation; it does not replace its earlier baseline.

## Implemented in the production Backend

- Sweep state is bound to its factor with a weak reference. Reusing a state with another factor resets it, and keeping a state cannot keep an old dense matrix/LU alive. Physical RHS shape and finite values are checked before QR. Overflow from a previous row scale falls back to the ordinary checked solve.
- An already represented incident span skips QR. Reconstruction validation reuses its projection remainder, avoiding two simultaneous full batch temporaries. Complex Frobenius norms use a contiguous complex dot product, avoiding possible work copies of strided real/imaginary parts. A one-off call does not copy basis/solution arrays into a cache that immediately dies. Original-matrix checks still cover every recovered azimuth.
- Hierarchical spatial ordering no longer leaves a recursive closure holding coordinates for garbage collection. A rejected local ACA block releases its traceback/workspace and earlier factors before fallback LU. A rejected degree-12 kernel table similarly releases its validation arrays before trying degree 16.
- Repeated S/K queries can use explicitly prepared geometry arrays and a prepared kernel domain. These are scoped to an immutable mesh owned by the caller, contain linear-sized data, and bypass the operator cache. The streamed experiment uses this path; ordinary assembly calls retain their existing behavior. Discard the prepared object when changing its mesh.

## Production measurements

Full 2 GHz airfoil solves: 5,742 unknowns, 361 azimuths (0–360 by 1 degree), both polarizations, condition estimation and field projection, experimental CPU kernels, two BLAS threads and four assembly threads. Baseline and updated runs were sequential fresh processes. There are two dense baseline runs and one final-code dense run. The hierarchical measurement precedes the final norm change; the final code also passed all hierarchical material checks.

| Case | Time | Sampled process peak |
|---|---:|---:|
| Previous production, dense | 42.05 s median | 1230–1230 MiB |
| Final production, dense | 41.48 s | 1246 MiB |
| Hierarchical, before final norm change | 41.89 s | 805 MiB |

This pass does not establish a further full-run dense peak-RAM improvement: the final sample is about 1.3% above the previous baseline. Earlier hardening samples were 1,254–1,257 MiB; the final norm change measured 1,246 MiB. Sweep ablations also varied with module loading and did not establish the cause of the whole-process difference. A single final timing is insufficient to claim a repeatable speedup. The production gains are specific allocation/lifetime fixes and stronger failure handling. The airfoil still solves 51 incident-basis columns per polarization for 361 physical illuminations. Hierarchical mode retains the original dense A; it is not the matrix-free prototype.

## Faster, capped streamed experiment

The geometry oracle now reuses prepared arrays. Tile QR replaces full SVD by default; a full coefficient comparison accepts or rejects the candidate and accumulates its actual error. Uncompressible tiles remain dense local tiles. Decomposition work has a local lifetime, and the inverse avoids revalidating/sorting already valid tree indices on every row query.

The tile representation and inverse share a 512 MiB retained numeric-storage allowance (including their index/error arrays). Construction temporaries, geometry, Python objects and RHS work are outside that allowance; it is not a process-RAM cap. Tiles are limited to 1,024 nodes and coefficient queries to 16 MiB. Cancellation reaches tree construction, block checks, multiplication and inverse application. Singular LU warnings reject construction. The CLI exits nonzero on rejection.

The checked solve includes the incoming pruning/tile error and subsequent inverse representation error, supports transpose/adjoint, and refines or rejects against a 1e-12 bound. It can explicitly adopt an owned initial solution, and its residual evidence is consumed immediately instead of recomputing the same product.

The following are per-polarization 2 GHz measurements. Core time includes assembly, inverse construction and the 361-column solve/bound check; it excludes mesh preparation, file loading, dense-reference qualification, final field projection and mesh certification.

| Channel | Previous assembly | Updated assembly | Previous core | Updated core | Updated construction peak | Updated peak through RHS/check |
|---|---:|---:|---:|---:|---:|---:|
| TE | 46.33 s | 37.46 s | 52.36 s | 44.52 s | 371 MiB | 415 MiB |
| TM | 46.59 s | 37.96 s | 53.30 s | 44.34 s | 367 MiB | 410 MiB |

All four 1/2 GHz TE/TM cases passed all 361 original RHS checks and independent dense-field comparisons. Largest independent backward error: 2.86e-14; largest complex-field difference relative to the channel peak: 2.95e-12; largest coefficient-error-inclusive solve bound: 8.83e-14. Construction inputs released immediately. Every final coefficient was requested once from geometry (D² entries); subsequent inverse queries read compressed tiles.

The prototype still takes longer for both polarizations than the production solver, which already reuses much of its TE assembly for TM. It remains outside the GUI/public solver. No new full 10 GHz solve or mesh-convergence certification was run. Process RSS is sampled every 50 ms and may miss short peaks.

## Native material investigation

The shared production pipeline passed 14 material families under dense and hierarchical factorization, including PEC, IBC, PEC/IBC, lossless/lossy/magnetic dielectric, coated/layered/mixed regions, sheets, mixed sheet/PEC, thin dielectric/magnetic layers and zero contrast.

Separately, the stricter experimental inverse was tested using saved **native** dense coefficient oracles. This isolates algebra/storage suitability; these are not geometry-to-compressed assembly measurements. Of 26 nontrivial polarization systems, 22 passed and four rejected at the rank/error limit. Two transparent-layer cases require no system. The rejected cases were mixed-sheet TE, thin-layer TE, and magnetic thin-layer TE/TM. Their off-diagonal residual estimates at the rank cap were 1.16e-13 to 3.32e-13, above the prototype's 1e-13 construction target. The target was not loosened to force acceptance.

Among accepted cases, final operator-plus-inverse numeric payload ranged from about 29% to 100% of dense A+LU. Sheet TE saved essentially nothing. Native W-containing systems therefore need a different treatment from the regional airfoil S/K path; a blanket compressed-backend switch is not justified.

## Further opportunities, in priority order

1. **Reuse TE assembly for TM in compressed storage.** Production already transforms transmission rows and reconstructs conductor terms using reciprocity. The prototype assembles both channels independently. Porting this with correct propagation of row/column error bounds is the strongest next opportunity to reduce two-channel completion time.
2. **Separate operator compression from preconditioning.** The prototype still creates two compressed representations and inspects many stored coefficients while building the inverse. Reusing a single representation or using a coarser inverse as a checked preconditioner could reduce construction time/storage. This needs physical-RHS convergence measurements across the entire sweep and material families.
3. **Add native W/sheet/thin-layer coefficient queries and a block strategy for difficult couplings.** The material experiment identifies four concrete rank-limit cases. Retain exact near/endpoint terms and use bounded dense local blocks where needed; keep the global memory cap and original-equation checks. The regional oracle must not silently substitute different native material equations.
4. **Reduce quadrature and scatter traffic.** The 1 GHz profile contained 462 operator invocations, 240 SVD calls and repeated geometry setup. Prepared geometry and QR address part of this. Remaining kernel evaluation, scatter, repeated nodal-support work and separate polarization traversals dominate assembly. W still uses its established 16-point far boxes; changing its order or fusing S/W requires separate accuracy evidence.
5. **Lower-priority bookkeeping.** Individual and joint kernel polynomials duplicate coefficients for fast evaluation, but the entire two-table cache in this 2 GHz run was only about 0.64 MiB. Eliminating that duplication cannot materially resolve the airfoil's dense-matrix RAM demand.

## Verification and reproducibility

- 140 production regression tests passed; four experimental test methods cover geometry/material tiles, numerical error bounds, adjoint solves, cancellation, ownership, storage caps, invalid inputs and singular rejection.
- Fourteen material families passed independent saved-field comparisons under both production factors. Numerical representation tests retain a 1e-12 backward-error gate and 1e-10 channel-peak-relative field comparison.
- Changed Backend files parse with Python 3.6 grammar; the available execution runtime was Python 3.12. No actual Python 3.6 run is claimed. Git whitespace checks passed.
- Existing user changes and prior experiment results were preserved. This turn's Backend baseline and separate diff distinguish these edits from previous work.

Evidence: [validated measurements](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/experiments/ghost_hardening_20260910/validated-results.json), [native material results](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/experiments/ghost_hardening_20260910/native-matrix-results.json), [Backend changes from this pass](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/experiments/ghost_hardening_20260910/hardening.diff), [source hashes](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/experiments/ghost_hardening_20260910/final-source-hashes.json).

Reproduce sequentially from the repository root: `.venv/Scripts/python.exe experiments/ghost_hardening_20260910/verify_run.py`, then `supplemental_run.py`, `final_norm_run.py`, `validate_results.py`, and `build_report.py` in the same folder. Historical samples include their Backend source hashes; rerunning a benchmark uses the current code. The baseline experiment uses the preserved local baseline copy; dense reference systems live in the earlier experiment folder. Restart GHOST to load the production Backend changes.
