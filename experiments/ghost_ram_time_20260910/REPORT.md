**GHOST 2D RAM and completion-time investigation — September 10, 2026**

The largest remaining RAM reduction requires replacing the dense system matrix, not just its LU. The best measured near-term speed improvement was a screened kernel-table change: at 2 GHz it reduced the complete airfoil solve from **41.62 to 35.92 seconds**, about **14%**, with four assembly threads. Both polarizations agreed with the baseline within **4.25e-13** relative to their respective peak complex fields. Sweep handling, matrix traversal, and operator ownership have further opportunities. Several tempting shortcuts failed accuracy checks or produced little practical benefit.

All changes in this investigation are isolated experiments. **The production Backend files were not edited.** The geometry's contents were treated as input data, not instructions. Results below distinguish complete solves, isolated algebra timings, storage accounting, and projections. They should not be added together as though each measured improvement were independent.

**Scope and measurement conditions**

I profiled the current production code, captured actual matrices and RHS arrays, compared alternative algorithms with independent dense solutions, and assembled selected matrix blocks directly from geometry. Coverage includes both polarizations for PEC, IBC, PEC+IBC, lossless and lossy dielectric, magnetic dielectric, coated and layered bodies, dielectric+PEC, sheets, sheet+PEC, nonmagnetic and magnetic thin layers, and transparent thin layers. This covers supported formulations; it does not add support for combinations the solver currently rejects.

Full airfoil discrete solves were exercised at 1 and 2 GHz. At 10 GHz I built the mesh and incident RHS, but **did not assemble or solve the full system**. Modern experiment runtime: Python 3.12.14, NumPy 2.5.2, SciPy 1.18.1, two BLAS threads. Most assembly comparisons are medians of two sequential runs; isolated algebra timings normally use three runs. This machine has about 31.1 GiB of physical RAM. Timing variation, caches, and machine load remain relevant. Experiments have not been ported to the legacy HPC Python runtime.

Complete-solve benchmarks retain condition estimation, both polarizations, all 361 angles, and the existing discrete-system checks. They are base-mesh solves, not mesh-convergence certificates. Field errors refer to complex amplitude differences normalized by the reference peak, not relative dB errors at radiation nulls. A small backward error alone does not establish accurate geometry discretization or identical fields.

**Why a twelve-inch object still creates a large problem**

The input requests `N=-20`: twenty panels per controlling material wavelength. Its largest complex refractive-index magnitude is about 22.33. At 10 GHz, the approximately 30 mm free-space wavelength becomes a controlling material wavelength of about 1.34 mm; the requested panel length is roughly 0.067 mm. Several long exterior and interior boundaries must be meshed. Physical width alone is consequently a poor predictor of the number of unknowns.

| Airfoil frequency | Panels | Nodes | Equation unknowns D | One complex128 A |
|---|---:|---:|---:|---:|
| 1 GHz | 1,938 | 1,944 | 2,964 | 0.131 GiB |
| 2 GHz | 3,772 | 3,778 | 5,742 | 0.491 GiB |
| 10 GHz | 18,726 | 18,732 | 28,406 | 12.024 GiB |

Dense A costs `16 D²` bytes; a full complex128 LU costs approximately the same again. The already-updated allocation plans are about **25.51 GiB** for the 10 GHz base mesh and **56.41 GiB** for its certification fine mesh using experimental CPU kernels and dense LU. Strict hierarchical mode still reserves about 21.30 and 46.91 GiB, respectively. These are plans from the prior implemented updates, not new full-size measurements. Scheduler margins can increase the machine reservation.

The original approximately 87 GiB prediction was dominated by matrix/operator allocations. The earlier recorded estimates were 86.79 GiB for one angle and 87.40 GiB for 361 angles. At a fixed frequency and polarization, azimuth changes the RHS, not A. The current solver already assembles and factors once per polarization and processes bounded batches. Reducing the sweep to one angle cannot remove the dominant D² storage. Certification additionally requires a finer mesh, whose larger system determines peak RAM; base and fine matrices need not coexist.

A pulse basis is not, by itself, an explanation for a dramatically smaller matrix. On a connected closed curve, linear nodal unknowns and pulse element unknowns are of similar order for the same element count. Material equations, mesh density, interface unknowns, collocation versus Galerkin integration, accuracy requirements, and compression backend must also be compared. The other solver's actual unknown count and material formulation were not available, so an equal-accuracy performance comparison cannot be established here.

**Matrix construction and lifetime audit**

Here N means mesh nodes; D is the final equation count. S is the single-layer operator, K/K′ a normal derivative, W the hypersingular operator, and M the sparse mass matrix. These mathematical actions may be necessary even when storing their separate dense arrays is unnecessary.

| Formulation | What the current equations use | Remaining avoidable storage/work |
|---|---|---|
| PEC | TM: S. TE: K′ and a mass jump. D=N. | TE copies K′ into owned Fortran-order A. No W or full second operator is built for PEC. Direct assembly into owned A would remove this copy. |
| IBC and PEC+IBC | Weighted S and K′ only on required Robin rows; S on PEC TM rows. D=N. | Compact operator payloads still coexist with final A during construction. A full IBC case can retain A plus two N×N operator payloads before accumulation. Route contributions straight into A as multi-region already does. |
| Homogeneous dielectric | Exterior K and W, interior S and K, sparse mass jumps. D=2N. | A coexists with up to two N×N source operators; they are copied into its four blocks. W is evaluated in a separate pass. Direct block destinations and shared quadrature would reduce memory traffic and construction storage. |
| Multi-region, including the airfoil | Only each region's required source/observer interactions, with material and interface weights. | Regional payloads have already been eliminated by direct scatter. The remaining large allocation is final A. Work remains in quadrature, kernel evaluation, near-pair handling, and scatter traffic. |
| Sheet and sheet+PEC | TM uses S; TE uses W, plus mass/endpoint conditions. D=N. | An operator is built then copied into A. Give the builder an owned destination; preserve endpoint equations. |
| Thin transmitting layers | One-density cases use S; general two-density cases use S, K, W and sparse mass solves. | One-density S is copied. Two-density S/K coexist with A before bounded mass-inverse applications, then W is built separately. Column streaming or operator actions can avoid complete intermediates. |
| Transparent thin layer | Exact zero scattered field after validation. | The existing shortcut already avoids A, operator construction, and LU. Nothing quadratic remains to remove on this path. |

Relevant production sources: [regional scatter](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/tools/GHOST/Backend/system_scatter.py), [multi-region equations](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/tools/GHOST/Backend/multi_region.py), [Robin systems](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/tools/GHOST/Backend/robin_system.py), [dielectric systems](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/tools/GHOST/Backend/dielectric_system.py), [sheets](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/tools/GHOST/Backend/sheet_system.py), [thin layers](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/tools/GHOST/Backend/thin_sheet.py), and [operator assembly](C:/Users/14col/Documents/ChatGPT/GHOST_FREDDY_GRIM/grim-integrated/tools/GHOST/Backend/rcs_operators.py).

For ordinary dense dielectric solves, removing temporary operators may not lower the overall peak because A+LU is larger than the assembly peak. For IBC or low-memory factorization, construction storage can determine the peak. These distinctions should appear in the resource planner, rather than assigning every removed array an equal end-to-end RAM saving.

The shared TE/TM assembly session intentionally retains one owned A to transform/reuse. It does not retain both LUs. The old regional operator cache is disabled on this path. Sparse mass matrices, bounded RHS batches, and skipping unused density/field exports are already implemented. Reintroducing a large operator cache would undo those savings. Equal wavenumbers alone do not permit sharing material coefficients; TE and TM generally still need different factors.

**Assembly: measured improvements and rejected shortcuts**

The 1 GHz airfoil profile spent about 17.97 of 19.77 instrumented seconds in multi-region assembly. Polynomial evaluation accounted for about 8.13 seconds; complex Hankel evaluation, near-pair quadrature, and scatter also mattered. These are inclusive profiler timings, not additive independent stages.

| Complete airfoil solve | Assembly threads | Median seconds |
|---|---:|---:|
| 1 GHz, current baseline | 1 | 18.14 |
| 1 GHz, joint S/K table evaluation | 1 | 17.71 |
| 1 GHz, degree-12 screened tables + joint evaluation | 1 | 15.97 |
| 1 GHz, current baseline | 4 | 14.51 |
| 1 GHz, degree-12 + joint evaluation | 4 | 12.76 |
| 1 GHz, degree-12 + joint evaluation | 8 | 14.65 |
| 2 GHz, current baseline | 4 | 41.62 |
| 2 GHz, degree-12 + joint evaluation | 4 | 35.92 |

Degree 12 uses smaller interpolation intervals and retains the current 2e-13 construction screen. Independent holdout distances agreed with the exact kernels below 8.1e-14 in tested accepted domains. Degree 10 and 8 configurations failed the existing coefficient-tail screen; they were not silently accepted. Joint S/K evaluation alone saved only about 2.4% at 1 GHz. The larger useful change was reducing screened polynomial work. This directly benefits lossy material regions; pure real-wavenumber PEC uses another fast path and did not gain from these table changes.

Four threads helped; eight were slower in this test. Thread count should be tuned with tile count and memory budgets. The result is not a recommendation to give every concurrent frequency worker all cores. Whole-process sampled RAM for the 2 GHz runs remained around 1.2 GiB; this is primarily a speed improvement, not a large RAM reduction.

A unique-index scatter prototype produced effectively no airfoil wall-time improvement (18.12 versus 18.14 seconds). Repeated node indices still require accumulation semantics. Optimizing indexing alone is lower priority than kernel work.

W currently uses a 16×16 tensor rule for regular far pairs. A blanket 8×8 rule reduced the small lossy fixture from 1.895 to 1.409 seconds, but that fixture does not qualify the rule for arbitrary long panels or close geometry. A pair-group prototype retained 16 points below three panel lengths and used eight farther away: 1.814 seconds, about 4% faster, with both-channel differences below 1.9e-14 across the material fixtures. A simpler tile-wide grading rule saved essentially nothing because a single close pair forced the entire tile to the higher order. Useful production work should group pairs, retain near-singular treatment, and fuse compatible S/W work. The airfoil multi-region path does not build W, so this optimization is for other formulations.

There is also a high-loss fast-path cutoff: a table is rejected when `-Im(k) × domain_diameter > 300`. For the strongest airfoil material, this crosses near 5 GHz. A partial-domain prototype tabulates only a screened range and evaluates the remaining distances exactly; it never discards small interactions. On a 10 GHz kernel microbenchmark, this reduced 0.0800 to 0.0725 seconds, about 9%; at 5 GHz the saving was about 23%. That modest high-frequency gain is not a full airfoil result, and depends strongly on the distance distribution. A rejection should not necessarily force every distance back to the slow kernel.

**Sweeps: solve a shared incident basis, keep checks on the requested illuminations**

Current compression recomputes a full SVD within each 256-angle batch. It can spend more identifying redundant RHS columns than it saves in triangular solves. The following times include compression, basis solving, reconstruction, and the candidate's residual work; they exclude factor construction, RHS loading, and far-field projection.

| Airfoil, double LU | 1 GHz TE / TM | 2 GHz TE / TM |
|---|---:|---:|
| Current SVD batches | 0.304 / 0.301 s | 0.910 / 0.938 s |
| One global pivoted QR basis | 0.242 / 0.229 s | 0.734 / 0.752 s |
| Fourier incident basis | 0.188 / 0.179 s | 0.665 / 0.680 s |
| QR plus residual identity | 0.138 / 0.141 s | 0.383 / 0.394 s |

Global QR required 34 independent columns at 1 GHz and 44 at 2 GHz, rather than solving overlapping bases in two batches. At **10 GHz, the actual 28,406-row TE incident RHS for 361 angles had a QR basis of 99 columns**, with scaled reconstruction error 2.45e-15. Constructing that basis took 0.77 seconds in a single measurement. This is an RHS experiment, not a full 10 GHz solve. All 26 captured nontransparent material/polarization systems passed the original backward-error limit with QR; their maximum complex-field difference was 3.9e-14.

Use a global or incrementally extended basis with bounded loading/reconstruction, not unlimited storage for all requested angles. At 10 GHz a single full 361-column complex array is about 156.5 MiB; several such temporaries are noticeable even though they are much smaller than A. A smaller maximum batch reduces workspace but can repeat rank discovery and sacrifice BLAS efficiency.

For `X = X_basis C`, the algebraic identity `A X - B = (A X_basis - B_basis) C + (B_basis C - B)` can avoid a full D×D multiplication for every physical RHS. The prototype also performed an independent original-A check afterward. Production use needs a floating-point error allowance and compatible diagnostic accounting; algebraic equality alone does not bound rounding in reconstructed X. The hierarchical and mixed inner solves also compute residuals that the outer dense wrapper recomputes. Returning validated residual evidence with a solved batch could remove some repeated matrix products without dropping the checks.

Fourier compression is promising but not ready for a fixed-rule substitution. At 10 GHz the initial `kR + 28` sampling rule produced a scaled RHS error of 1.92e-13; increasing padding to 40 reduced it to 5.24e-15. For the straight TE sheet, Fourier cancellation produced a nonzero solution at a zero or nearly zero incident column and a backward error of 0.51 despite tiny absolute field differences. Preserving exact zero columns and directly solving two failed columns restored the gate. Adaptive sampling, original-RHS validation, and per-column fallback are necessary.

Field projection can also operate on the solved basis. At 2 GHz, reconstructing all densities then projecting took about 0.065–0.067 seconds per polarization. Projecting the basis first took about 0.055 seconds; a separately tested harmonic observation expansion took about 0.018 seconds. Differences were below 6e-15 for that test. The basis density occupied 4.04 MB versus 33.17 MB for all 361 recovered density columns. This is a useful secondary optimization; residual checks and density export still need their appropriate data, and observation interpolation needs its own qualification.

**Factorization and the route to substantially lower RAM**

The current hierarchical option compresses the inverse but retains dense A for refinement and condition checks. Simply making the LU smaller cannot remove the 12.02 GiB A at 10 GHz.

I built an experimental hierarchical representation of both A and its inverse. At 2 GHz, using tolerance 1e-13:

| Retained numerical payload, one polarization | TE | TM |
|---|---:|---:|
| Dense A + dense LU | about 1,055 MB | about 1,055 MB |
| Dense A + current hierarchical factor | about 614 MB | about 609 MB |
| Compressed operator + inverse prototype | about 165 MB | about 157 MB |

The latter is roughly **73–74% smaller than A plus the current hierarchy**, and about 84–85% smaller than A+LU, for these systems. Original-matrix backward errors were below 2.9e-14; peak-relative field differences were below 1.8e-12. These are numerical-array payloads, excluding Python objects and RHS workspaces, **not measured end-to-end peak process RAM**.

The constructor currently reads an already assembled matrix through a file-backed coefficient oracle. Consequently it does not remove the dense assembly peak. Building the tighter 2 GHz representations took about 5–11 seconds in individual runs, before counting electromagnetic assembly, and performed roughly 10–12 times D² coefficient accesses because ACA repeatedly scans blocks to validate residuals. A direct connection to an expensive element assembler would multiply kernel work severely.

The separate geometry-oracle prototype did assemble selected airfoil blocks without global A: 32×128 blocks and full individual rows matched the captured matrices within 3.1e-17 peak-relative error. Each query still took roughly 0.037–0.082 seconds because it rebuilds plans and incurs quadrature/scatter overhead. This proves a usable integration boundary, not efficient compressed assembly. It currently handles multi-region equations only.

Accuracy and rank selection cannot use one universal compression tolerance. At tolerance 1e-10, compressed operators failed original residual limits even though their internal approximate residuals were tiny. At 1e-12, four dielectric/magnetic fixture channels exceeded a 1e-10 field-comparison threshold; two other channels exceeded the conservative residual bound. Tightening to 1e-13 improved accuracy, but four sheet/thin-layer channels then exceeded the prototype's rank cap. Simple equation equilibration was also insufficient as a universal remedy. The required production design needs scale-aware error budgets, stronger geometric block subdivision, rank/storage limits, and a fallback that respects the requested RAM ceiling.

Loosening only the existing inverse while retaining original-A refinement is less invasive. At 2 GHz, tolerance 1e-6 reduced TE/TM factor payloads from 86.6/81.0 MB to 57.4/51.6 MB, but extra refinement meant total speed did not improve uniformly. Tolerance 1e-4 failed on the 1 GHz TE airfoil and needed six or seven refinements at 2 GHz. A magnetic fixture also showed that a passing normwise residual need not reproduce fields to 1e-10 under a very loose inverse. Tolerance 1e-8 passed the tighter material comparison in this investigation. An adaptive policy should account for conditioning and correction cost, rather than globally loosening the current default.

Mixed-precision LU has the same tradeoff. At 2 GHz it reduced factor/condition setup time in these trials but made RHS refinement slower. The TE total for setup plus global QR solve was about 3.41 seconds mixed versus 3.22 double; TM was about 2.99 versus 3.38. At 1 GHz mixed was slower. It remains a RAM option with checked fallback, not a universal time optimization. Experimental CPU currently requires double precision; these mixed trials used captured systems outside that execution mode.

**Iterative solves require a preconditioner**

A bounded GMRES study used dense A only as an exact matvec oracle. Unpreconditioned airfoil solves failed to converge within 120 iterations. Spatial local-block preconditioning required about 62–71 iterations for selected TE/TM illuminations; some TM cases still exceeded the solver's original backward-error gate even though GMRES reported convergence. A coarse hierarchical preconditioner reduced iteration counts to approximately 9–11. Thus replacing LU with plain GMRES is not a sufficient fix.

The coarse-preconditioned airfoil runs passed the original backward-error check and changed the tested fields by at most 2.31e-13 relative to the full sweep's reference peak. They tested four selected illuminations, not a complete iterative 361-angle solve. Mathematical density differences were larger than field differences, which reinforces the need to check both equation residuals and the physical outputs.

For this sweep, solve an incident basis using a reusable preconditioner or compressed inverse. Solving 361 independent iterative problems would discard much of the available reuse. The prototype is not an FMM timing or RAM benchmark; removing A from matvecs and qualifying original-system residual/condition evidence remain separate work.

**Other bounded opportunities**

The current infinity-norm routine reads row blocks of Fortran-order A. A column-block traversal reduced the 2 GHz scan from about 0.063 to 0.026 seconds, with differences around machine precision. Preserve bounded workspace and choose traversal by layout. Condition equilibration already uses column-block passes; it should not be rewritten into full-sized scaled copies. The hierarchy also recomputes a norm already obtained by the outer factor. Share immutable norm/scaling evidence where the matrix is unchanged, invalidating it when TE-to-TM transformations change A.

Interface-local material wavelengths are worth considering for other geometries, but are a small airfoil win. At 10 GHz they reduced D from 28,406 to 27,448, only 3.37%; one A decreased from 12.024 to 11.226 GiB, about 6.6%. Most long airfoil interfaces really do border the high-index material. Changing the user's requested panel density or replacing bulk material with an impedance/thin-layer model is a different physical approximation and needs mesh/model validation.

At 1 GHz, local meshing changed the 361-angle two-channel fields by 1.49e-5 relative to their joint peak and reduced a run from 18.02 to 17.84 seconds. A separate 37-angle local-mesh base/fine certification passed both channels, using 1,902 versus 2,988 panels; maximum dB change was 0.00622 dB and maximum phase change 0.0247 degrees. That certificate applies to those 37 angles at 1 GHz, not the full 10 GHz sweep.

Exact redundant interfaces between identical media could potentially be eliminated before meshing, and fully transparent bulk objects could have an analytic shortcut analogous to the existing thin-layer case. These were identified by code inspection, not implemented or benchmarked. Topology, region incidence, and all material laws across the frequency range must be checked. Arbitrary interior regions cannot be discarded just because their incident RHS is zero; coupling may excite them.

The current HPC scheduling unit already groups frequency and both channels, allowing matrix reuse. Preserve this grouping. Splitting an azimuth sweep into independent workers would duplicate assembly, A and factor storage. More workers improve throughput only when per-worker memory and total thread reservations permit them. Certification must retain a genuine base/fine comparison; skipping it is not an equivalent solver optimization.

**Recommended implementation order**

1. Integrate screened degree-12/joint kernel evaluation with exact fallback, plus layout-aware norm traversal. These have direct, bounded evidence and retain the equations.
2. Replace repeated batch SVD discovery with checked pivoted QR and a global or incrementally extended incident basis. Reuse factor residual evidence where valid; qualify the reconstruction residual identity before using it to remove physical-column matrix products.
3. Extend owned destination/scatter assembly to Robin, homogeneous dielectric, sheet, and thin-layer builders. Keep sparse mass solves and endpoint conditions; update formulation-specific construction peaks in the planner.
4. Improve W pair grouping and compatible S/W quadrature fusion for dielectric/sheet cases. Preserve close-pair integration and test long elements, near gaps, material weights, and reversed orientations before lowering orders.
5. Develop compressed assembly as a separate backend: prepared geometry/coefficient plans, reusable near blocks, strong geometric admissibility, and batched block compression with bounded validation. Use block subdivision rather than forcing every off-diagonal interaction into the present HODLR rank cap.
6. For higher-frequency or larger problems, evaluate a directional hierarchical/FMM matvec with a reusable preconditioner and the same incident-basis strategy. Require an error contract for the operator and adjoint, condition estimation, refinement, certification, cancellation, memory rejection, and HPC provenance before treating it as a replacement for the dense backend.

Hierarchical/FMM approaches are established ways to avoid dense boundary-operator storage; this code still needs formulation-specific integration and validation. The [BEM++ assembly documentation](https://bempp.com/handbook/core/assembling_operators.html) describes dense and FMM assembly choices. High-frequency rank growth motivates methods such as [hybrid directional matrix compression](https://arxiv.org/abs/1809.04384) and [frequency extraction](https://arxiv.org/abs/2012.14287); those papers are background, not performance evidence for this solver. The QR prototype uses [SciPy's raw pivoted QR interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html) to form only the required basis columns.

**Reproducible evidence**

The scripts and JSON results live beside this report in `experiments/ghost_ram_time_20260910`. `baseline-manifest.json` records versions, the airfoil hash, and production source hashes. `assembly-*.json` contains measured runs; `algebra-*.json`, `hierarchy-*.json`, `material-*.json`, `material-tightening.json`, `angular-scaling.json`, `coefficient-oracle-results.json`, `partial-table-results.json`, `projection-*.json`, and `iterative-*.json` contain numerical evidence, including failed candidates. Captured matrices under `systems/` are generated local data and are git-ignored.

Use the repository virtual environment from this directory. Examples:

```powershell
../../.venv/Scripts/python.exe assembly_probe.py airfoil --variant baseline --frequency 2 --threads 4 --repeats 2
../../.venv/Scripts/python.exe assembly_probe.py airfoil --variant table12_paired --frequency 2 --threads 4 --repeats 2
../../.venv/Scripts/python.exe capture.py airfoil --frequency 2
../../.venv/Scripts/python.exe algebra.py airfoil-n256-f2.0
../../.venv/Scripts/python.exe hierarchy_probe.py airfoil-n256-f2.0
../../.venv/Scripts/python.exe material_probe.py
../../.venv/Scripts/python.exe angular_scale.py
../../.venv/Scripts/python.exe coefficient_oracle.py
```

`capture.py` writes matrices and performs extra reference solves, so its elapsed time is deliberately excluded from performance claims. Run benchmarks sequentially to avoid competing compute jobs. Prototype patches are scoped to their experiment process; they are never imported by the production solver.

Final verification: nine evidence-consistency checks passed in `check_results.py`, all experiment Python files compiled, and every Backend source hash plus the airfoil hash matched the baseline manifest. These checks validate the recorded experimental claims, not production readiness of the prototypes. Final material algebra results were recomputed by `recheck_material_algebra.py` with correct zero-RHS diagnostic handling; its JSON output supersedes preliminary diagnostic warnings in `material-probe.log`.
