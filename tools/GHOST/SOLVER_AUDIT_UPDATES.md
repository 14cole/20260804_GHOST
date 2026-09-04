**Solver audit updates — September 2026**

These changes address the seven findings in the accompanying workspace audit.

1. **2-D absolute phase.** Correct the dielectric trace/flux excitation signs,
   TE sheet and mixed sheet/PEC excitation signs, and multi-region exterior
   density extraction. Monostatic, bistatic, and boundary-density exports use
   the same physical representation. The Maue matrix is W = −d_n D; the shared
   far-field projectors and the A_physical = j B_stored convention are unchanged.
2. **BoR memory.** Integrate near point tiles directly into compact 2×2 modal
   blocks. Prepare one kernel family at a time, with bounded angular and modal
   projection scratch. Cross-surface caches also retain compact blocks. Include
   these allocations in conductor/dielectric streaming gates and scheduler
   previews. Release FFT batches before constructing their successors and
   include additional FFT/expression workspace in the point-tile size.
3. **Thin coatings and close folds.** Grade both meridional coordinates around
   the gap, compare the actual EFIE and rotated-PV blocks across integration
   orders, and reject unconverged results. Shared endpoints retain the graded
   singular-cell rule. Direct operator assembly uses the checked preparation
   path too.
4. **2-D bistatic memory.** Estimate every incidence RHS and account for retained
   sample dictionaries and rectangular projection work before assembly. Large
   grids can now be rejected before an unbudgeted allocation.
5. **2-D sweep memory.** Complete both polarizations at one frequency before
   advancing, including both certification meshes. Preserve same-frequency IBC
   operator reuse; omit unused PEC K′ caching. Specialist shared caches retain
   only the current frequency's dense operators and frequency-dependent mesh.
6. **Mesh density.** Remove the implicit 2,000-elements-per-primitive limit.
   Wavelength-based counts now honor the requested density; explicit global
   panel/element limits remain enforced. Splitting a straight primitive no
   longer changes its wavelength-density limit.
7. **BoR angular integration.** Resolve oscillation in both the singular core
   and angular tail, verify a second rule, and reject requests that exceed the
   angular-order limit. Point and mode tiling bound the temporary arrays. Use
   cached SciPy Gauss rules, including at high orders.

Pole and junction constraint reductions now apply sparse row/column relations
instead of dense Qᴴ A Q products. Dense LU and the established physical
formulations remain in place.

**Phase compatibility**

New 2-D result dictionaries and boundary-density results carry
`amplitude_version: 2`. GRIM's solver-metadata JSON preserves this version.
Missing version information is treated as legacy version 1 when writing that
metadata envelope; an old result is not silently relabeled as corrected.

Regenerate legacy complex datasets produced by the affected dielectric,
multi-region/coated, TE sheet, and TE mixed-sheet branches before combining
them coherently with current results. Their RCS power alone cannot identify the
phase error. Pure PEC and the tested TM sheet branch already had the correct
phase, so a blanket sign change to old GRIM files is inappropriate. Existing
user datasets are left untouched.

**Quadrature controls and interpretation**

`BorCrossOperators` and `solve_bor_coated_pec` accept `near_order` (default 12),
`near_rtol` (default 2e-5), and `near_max_order` (default 192, maximum 384).
The published block uses at least `near_order`; smoother pairs can use a
cheaper preliminary rule to verify that order. Narrow-gap pairs use grading
and additional refinement as needed. The memory estimate includes the maximum
configured meridian-rule workspace.

Coated-solver output includes `near_quadrature`, with the maximum accepted
disjoint-pair order and relative block change. The production BoR dispatcher
preserves that record. This evidence is distinct from linear-system residuals,
modal truncation, and mesh convergence. Self/junction singular-cell accuracy
still relies on the established graded rules and mesh/reference tests.

Angular near integration has a 4096-point order limit and a 2e-8 relative
refinement criterion. The comparison is scaled to each point pair's largest
modal component. Requests outside the resolved range raise an accuracy error.
It does not promise relative accuracy for every near-zero individual harmonic.

**Validation and remaining limits**

All 193 distinct selected regression tests passed, including 13 new audit-fix
tests. The vector reference integration also passed a focused rerun that treats
integration warnings as errors. Python compilation and diff whitespace checks
passed.

The new `tests/test_solver_audit_fixes.py` checks absolute cylindrical complex
amplitudes, certified phase, the lossless optical theorem, incidence-count
planning, frequency-local cache behavior, mesh counts, scalar and vector
angular kernels against independent adaptive integration, axis pairs, FFT
tiling, compact near storage, sparse constraint algebra, and an analytical
0.1 mm coated-sphere case.

The corrected certified cylindrical cases have complex errors below 0.004 in
the regression meshes. The audited 24-element thin sphere previously had
1.34 dB error; the corrected result has 0.0263 dB maximum error, against a
0.08 dB regression limit across both channels and three looks. A fresh-process
24-element/mode-20 memory measurement retained
2,393,088 bytes of compact near values and no raw near cache; incremental peak
RSS was about 194 MB against a 289 MB operator estimate.

Adaptive integration adds work where the previous fixed rule was inadequate.
Very thin gaps and large modal bandwidth can still be expensive. The far-stream
budget controls retained far blocks, not total process memory; compact near
blocks, dense solves, construction scratch, and fixed process overhead are
additional. Scheduler previews use a conservative local-neighbor inventory;
the runtime gate uses the actual classified pair inventory. Highly folded
geometries may require a larger reservation than the preview.

Real CUDA performance, large cluster runs, electrically large non-spherical
reference cases, and weak-return accuracy floors remain separate validation
work. These changes do not introduce an iterative/FMM solver or automatically
migrate existing complex datasets.

The workspace `solver-audit` directory contains the original audit, independent
probes, before/after logs, and the final implementation validation record.
