**GHOST 2D: further RAM and completion-time recommendations — September 10, 2026**

I recommend two tracks: integrate the measured changes that avoid unnecessary allocations and repeated calculations, and develop a backend that avoids storing the global dense matrix. The latter is required for a substantial further RAM reduction on the 10 GHz airfoil. The new experiments below extend the [previous investigation](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/experiments/ghost_ram_time_20260910/REPORT.md).

**These are isolated experimental implementations. Production Backend source hashes are unchanged during this follow-up.** Full numerical airfoil solves were tested at 2 GHz; the full 10 GHz system was not assembled or solved. The geometry was treated as data. Existing user files and earlier production edits were preserved.

**Measured opportunities and implementation order**

| Priority | Change | Evidence | Scope and qualification |
|---|---|---|---|
| 1 | Screened degree-12 kernel tables with joint S/K evaluation; shared pivoted-QR incident basis | Earlier full 2 GHz airfoil: 41.62 → 35.92 s, about 14%. Actual 10 GHz TE incident sweep: 361 columns represented by 99 basis columns. | Keep kernel fallback and checks on requested illuminations. Kernel-table benefit primarily concerns lossy regions. QR applies across formulations, with per-column fallback. |
| 2 | Assemble remaining operators into their owned final destinations | New 4,096-node IBC full solve: sampled peak RSS **867.6 → 693.2 MiB**, about **20% lower**; 13.20 → 12.84 s, about 2.8% faster. | Benefits Robin/IBC, PEC TE, homogeneous dielectric and sheets. The airfoil's multi-region path already uses direct scatter. This is not another 20% airfoil RAM saving. |
| 3 | Defer unnecessary intermediate ACA validation scans | New 2 GHz airfoil: current-style hierarchical factor construction **24–41% faster**; experimental compressed A+inverse construction **32–40% faster** at tolerance 1e-13. | Preserve full-coefficient acceptance, refinement and rank/storage limits. Benefit concerns hierarchical construction, not ordinary dense LU. |
| 4 | Return and consume the original-A residual already computed by refinement | New 2 GHz incident-basis solve phase: about **25% faster** with hierarchical factors; **10–13% faster** with mixed precision; identical solutions. | Factor construction, compression, projection and physical-column checks are excluded from these timings. This is a small full-run saving. |
| 5 | Exploit material attenuation using a controlled error budget | New 2 GHz airfoil pruning trial: **41.85 → 39.32 s**, about **6% faster** for the complete two-polarization solve. | Experimental approximation. No RAM saving while A remains dense. Requires a defensible omitted-interaction bound before production use. |
| 6 | Build the operator directly in compressed/local storage, then reuse its inverse or preconditioner across an incident basis | Earlier 2 GHz retained A+inverse payload was about **73–74% smaller** than dense A+the current hierarchical factor. | Highest RAM potential. These are numerical payload counts, not peak-process RAM. Construction still reads dense A and is not yet a production replacement. |

The percentages refer to different stages and baselines. They must not be summed or applied directly to a 10 GHz run.

**1. Complete the operator ownership work**

The [ownership prototype](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/experiments/ghost_ram_time_followup_20260910/owned_assembly.py) removes remaining build-then-copy paths. It routes weighted Robin contributions into A, writes homogeneous dielectric S/K/W contributions into A's four blocks, and lets PEC/sheet paths adopt their owned operator. It retains quadrature, mass jumps, endpoint equations and TE/TM reuse.

All 14 material-family fixtures passed their discrete quality gates for the full 361-angle, two-channel calculation: PEC, IBC, PEC+IBC, lossless/lossy/magnetic dielectric, coated/layered bodies, dielectric+PEC, sheet, sheet+PEC, thin dielectric, magnetic thin dielectric and transparent thin layer. Maximum peak-normalized complex-field difference was **2.02e-15**. These fixtures sample supported formulations; they do not certify every geometry or material value.

The large IBC comparison used two fresh processes per variant. The approximately 174 MiB median peak reduction is sampled process RSS, not a count of deleted arrays. Sampling occurs every 50 ms and can miss short peaks. The two owned runs differed from the reference fields by at most 1.63e-15. Homogeneous dielectric may still peak during A+LU factorization, so removing its assembly temporaries need not reduce the final peak as much as in IBC.

Next production work should expose explicit output destinations, preserve accumulation for repeated node indices, and update formulation-specific memory plans. General two-density thin layers still need bounded column streaming through their mass-inverse operations; this prototype does not finish that redesign.

**2. Stop rescanning compression blocks before the result can be accepted**

[hierarchical_factor.py](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/tools/GHOST/Backend/hierarchical_factor.py:98) currently scans every coefficient of an off-diagonal block every 16 ACA updates. The prototype checks when an update becomes small, a pivot is unresolved, or the rank limit is reached. Acceptance still checks the complete block at the same tolerance. A small update never establishes acceptance by itself.

| 2 GHz airfoil construction phase | Existing scan schedule | Deferred schedule |
|---|---:|---:|
| Current-style inverse, TE | 4.313 s | 2.564 s |
| Current-style inverse, TM | 3.547 s | 2.690 s |
| Compressed operator+inverse, TE, 1e-13 | 5.133 s | 3.082 s |
| Compressed operator+inverse, TM, 1e-13 | 4.698 s | 3.212 s |

The current-style inverse retained original A for refinement. All **28 material/polarization systems**, including the two airfoil systems, passed independent original-A residual and complex-field comparisons. Maximum field difference was 2.49e-11, maximum backward error 2.17e-13, and maximum tested conjugate-transpose backward error 1.41e-15. This does not relax the original 1e-12 release limit. Production qualification should also exercise transpose mode, cancellation, hard rank cases and fallback memory limits.

For the tighter compressed operator, TE coefficient accesses fell from 370 million to 188 million; TM fell from 342 million to 197 million. Retained payload remained approximately 165/158 MB. The new count is still roughly **5.7–6.0 D² accesses**. Connecting this builder directly to expensive electromagnetic quadrature would still repeat substantial work. The scan change improves the present algorithm, but does not make it efficient matrix-free assembly.

**3. Remove a duplicate exact residual, without weakening validation**

[Hierarchical refinement](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/tools/GHOST/Backend/hierarchical_factor.py:219) and [mixed refinement](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/tools/GHOST/Backend/refined_lu.py:57) compute `b - A x` immediately before returning a solution. [DenseFactor](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/tools/GHOST/Backend/dense_factor.py:143) then computes the same product again for diagnostics.

The [residual prototype](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/experiments/ghost_ram_time_followup_20260910/residual_reuse.py) consumes the inner residual only when matrix, RHS and solution object identities match and the solve is untransposed; otherwise it computes the original expression. Hierarchical TE/TM basis-solving times fell from 0.378/0.371 to 0.282/0.278 seconds. Mixed times fell from 1.041/0.720 to 0.935/0.629 seconds. Four timed solves per case returned identical solutions with an independent original-A check.

A production implementation should return residual evidence with the solved batch, consume it immediately, and invalidate it on any matrix/solution mutation. Avoid a persistent cache holding obsolete RHS or density arrays. Keep the separate recovered-illumination check. Reusing an already computed residual is distinct from estimating it using a reconstruction identity; the latter needs an additional floating-point error allowance. Accurate residual evaluation remains central to iterative refinement; see [LAPACK's error-bound analysis](https://netlib.org/lapack/lawnspdf/lawn165.pdf).

**4. Use strong material loss to avoid work before assembly**

For GHOST's convention `k = real(k) - i alpha`, the outgoing Hankel kernel decays with distance approximately as `exp(-alpha r)` times an algebraic factor. The asymptotic form and complex-argument error discussion are in [NIST DLMF 10.17](https://dlmf.nist.gov/10.17). This suggests distance locality in strongly lossy material; the asymptotic exponential alone does not bound a weighted Galerkin matrix entry.

First, I tested deleting captured airfoil coefficients only when their entire nodal basis supports were separated by at least an optical distance `alpha*r = 32`. At 2 GHz, this removed **11.03 million nonzero entries**, about **40.8% of all nonzeros** and 55.1% of lossy-region nonzeros. All 361 illuminations in both channels passed checks against the unchanged original matrix. Maximum field difference was 6.70e-13; a residual bound including the exact dropped row sums was below 3e-14.

This cutoff is not universal. Optical distance 16 failed the original residual limit for both channels. Distance 24 passed the directly measured residual but failed the conservative omitted-coefficient bound for TM. Those failed candidates are retained in the evidence.

Second, the [assembly prototype](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/experiments/ghost_ram_time_followup_20260910/loss_traversal.py) used an element-support distance bound before far assembly. It preserved the original near-pair classification and real-wavenumber path. It skipped **4.585 million of 10.836 million eligible far pairs**, or 42.3%. Both-channel full-run times fell about 6%; field differences were at most **5.05e-13**. A separate audited run checked every recovered physical RHS against the original A, with backward errors below 3.40e-14 and omitted-coefficient residual bounds below 3.50e-14. Audit overhead is excluded from the speed comparison.

Only 30 of 210 tiles became completely inactive. Mixed tiles still evaluated quadrature for many subsequently discarded pairs. The next experiment should group surviving pairs or subdivide tiles so discarded pairs avoid kernel evaluation entirely. This is a concrete assembly opportunity, although no further speedup is claimed yet.

I also combined pruning with degree-12/joint kernel evaluation. A late comparison measured 47.21 s for the unchanged baseline, 42.18 s for the kernel change alone, and 39.90/39.69 s for the combination. This suggests an additional benefit, but the unchanged factorization stage had slowed from about 4.3 to 7.1 s during that series. The late baseline and kernel-only measurements are single runs; do not present a firm cumulative percentage from them. Combined fields differed by at most 4.84e-13, and an independent audit retained an original-A backward error below 3.40e-14 and an omitted-coefficient residual bound below 3.51e-14. Audit time is excluded from those comparisons.

For production, derive a tail bound covering S, K, W, basis weights, material scales and the requested tolerance. Store discarded interactions implicitly. Simply writing zeros into a dense A saves no RAM; converting it blindly to CSR followed by sparse LU risks substantial fill. Pure PEC and lossless regions receive no attenuation benefit.

**5. Build a backend suited to the airfoil's mixed interactions**

At 10 GHz the existing mesh has 28,406 equation unknowns. One complex128 A alone costs **12.024 GiB**, independent of the number of illuminations. Current dense base/fine allocation plans are approximately 25.51/56.41 GiB before scheduler margins. These are previously calculated plans, not new measured full-size solves. Removing temporary operators or changing the inverse alone cannot eliminate A's storage.

My preferred experimental design is an exact local/near representation, controlled truncation of sufficiently attenuated interactions, and compressed or FMM evaluation of the remaining long-range interactions. Preserve the present material equations and singular/near-singular quadrature. Share geometry plans, near blocks, a preconditioner or compressed inverse, and an incident basis across the sweep. Keep separate polarization factors where their equations differ.

Use prepared geometry plans with bounded query workspaces; do not replace A with a persistent dense table of all pair distances. Partition by geometry and region as well as index order, subdividing difficult/high-rank blocks. Current HODLR rank failures in sheets/thin layers show why a fixed rank cap and one global tolerance are insufficient. Release coarse-mesh resources before fine-mesh certification. A dense fallback must fit the requested memory ceiling.

A concrete FMM candidate is [Flatiron's FMM2D](https://github.com/flatironinstitute/fmm2d), which provides two-dimensional Helmholtz interactions. Its [mathematical interface](https://fmm2d.readthedocs.io/en/latest/math.html) supports complex wavenumbers, charges, dipoles, target gradients and multiple density vectors at fixed points. That is a useful starting point for testing a batched incident basis. It uses first-kind Hankel kernels; GHOST uses second-kind kernels, so sign, conjugation, normalization, material masks and adjoints require explicit verification. It also omits self interactions, so the existing self/near quadrature must supply the corresponding corrections. No FMM package was installed or benchmarked here.

Start with operator-action agreement against small captured dense systems for every formulation; then test preconditioned basis solves and independent field accuracy. Only then compare complete wall time and sampled RAM as frequency grows. Earlier unpreconditioned GMRES did not converge within 120 iterations, while a coarse hierarchy required about 9–11 for selected airfoil illuminations. Plain GMRES or 361 independent iterative solves would be a poor default replacement.

**Secondary work and limits**

W pair grouping and compatible S/W quadrature fusion remain useful for dielectric/sheet/thin-layer cases; the multi-region airfoil does not build W. A compiled, fused distance/kernel/scatter loop could reduce temporary-array traffic, but needs a benchmark before assigning it a speed benefit. Norm traversal, immutable norm reuse and projecting solved basis vectors can save smaller repeated passes.

Avoid splitting azimuths into independent processes, caching full operators indefinitely, globally reducing precision or quadrature, or changing the material model to improve a benchmark. Interface-local meshing reduced this airfoil's unknown count only about 3.4% in the earlier study. A pulse-basis comparison needs matched material equations, unknown counts and accuracy, rather than equal physical width alone.

**Reproduction and evidence**

All new scripts and JSON results reside beside this report. `source-manifest.json` records Backend hashes; `validated-summary.json` aggregates checked comparisons. Modern local runtime: Python 3.12.14, NumPy 2.5.2, SciPy 1.18.1; two BLAS threads and four assembly threads for complete-solve timings. Full-run timings use two sequential runs per variant, and residual microbenchmarks use four. Experiments were not run concurrently. These are base-mesh discrete comparisons, not new mesh-convergence certificates or qualification of the legacy HPC runtime. Field errors are complex amplitude differences normalized by each reference channel's peak, not relative errors at radiation nulls.

From this directory, use the repository environment:

```powershell
../../.venv/Scripts/python.exe ownership_probe.py
../../.venv/Scripts/python.exe ownership_probe.py --kind ibc --mode owned --count 4096 --threads 4
../../.venv/Scripts/python.exe inverse_schedule.py --all-materials
../../.venv/Scripts/python.exe inverse_schedule.py
../../.venv/Scripts/python.exe compression_schedule.py airfoil-n256-f2.0
../../.venv/Scripts/python.exe residual_reuse.py
../../.venv/Scripts/python.exe loss_locality.py
../../.venv/Scripts/python.exe loss_assembly_probe.py --cut 32 --check
../../.venv/Scripts/python.exe loss_assembly_probe.py --cut 32 --variant table12_paired --check
../../.venv/Scripts/python.exe validate_evidence.py
```

Captured systems from the earlier audit are required by several scripts. `validate_evidence.py` checks saved numerical results and source hashes; it does not rerun the expensive assembly or certify production behavior. The prototypes use scoped source transformations with source-drift checks and are never imported by the production solver.
