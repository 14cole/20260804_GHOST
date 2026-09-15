# GHOST 2-D solver audit — Pulse (P0 collocation) and Galerkin

Date: 2026-09-15
Scope: `tools/GHOST/ghost_backend/twod/**` (both discretizations, all three
factorization backends), plus the compressed/FMM oracles that share those
operators. Read-only audit: no solver source was modified on this branch.

Environment used for the measurements: Linux, CPython 3.11.15, NumPy 2.4.6,
SciPy 1.17.1. `twod/fmm/native` and `twod/assembly/native` were built from the
vendored sources with GNU Fortran 13 so the FMM paths could be exercised.

Baseline health: `test_pulse.py`, `test_fmm.py` and `test_fmm_efficiency.py`
pass in full once the native library exists (58 passed, 1 skipped, 6 subtests).
A wider 2-D sweep — `test_2d_capability_acceptance`, `test_rcs_physics_regression`,
`test_solver_audit_fixes`, `test_galerkin_sweeps`, `test_compact_multi_region`,
`test_assembly_equivalence`, `test_near_separation`, `test_nystrom`,
`test_2d_co_polarized` — gave 92 passed / 2 failed / 28 subtests, and both
failures are F9 below (a missing *optional* dependency), not solver defects.

---

## 1. Summary

| # | Severity | Area | Finding |
|---|---|---|---|
| F1 | High | Pulse + FMM | An odd `fmm_quadrature_order` places a Gauss node exactly on the collocation point; FMM accuracy silently drops from 3e-12 to 1e-2 with no error raised |
| F2 | High | Pulse accuracy | Pulse is **first order** for every formulation containing K′ (TE PEC, IBC, dielectric) while Galerkin is second order; the docs describe this as a constant-factor difference |
| F3 | High | Pulse, TE PEC | Interior-resonance blow-up (124 % amplitude error measured) passes the numeric quality gate; only the *certified* entry point rejects it |
| F4 | Medium | Galerkin, TM PEC | The combined-field formulation exists only on the FMM backend; dense and compressed solve the bare EFIE for the same input, so the backend choice changes the integral equation |
| F5 | Low | Docs/API | `fmm_quadrature_order` accepts 6…64 although only even orders are safe for Pulse (see F1) |
| F6 | Low | Performance | `pulse/coefficients.blocks` evaluates every near pair twice and discards the first result |
| F7 | Low | Performance | `PulseKernel` asks the FMM to evaluate at ~`order+1` times more targets than it uses, in both directions |
| F8 | Cosmetic | Maintainability | Three "docstrings" in `operators.py` are dead string expressions placed after the first statement |
| F9 | Low | Packaging | `twod/nystrom.py` imports `psutil` unguarded although the requirements declare it optional at runtime |

Verified-correct items are listed in §3; they matter because they bound where
the problems can be.

---

## 2. Findings

### F1 — Odd FMM quadrature order collides Gauss nodes with collocation points (High)

`PulseKernel.prepare` builds one native point list holding the panel Gauss
points followed by the panel midpoints:

- `twod/pulse/kernel.py:39-41` — `quad = p0 + t*segment`, then
  `self.points = np.vstack((quad, g.centers))`.
- `twod/pulse/coefficients.py:15-18` — `gauss(order)` maps the Legendre nodes to
  `(t+1)/2`. For **odd** order the node `t = 0` maps to exactly `0.5`, i.e. the
  panel midpoint, which is byte-identical to `g.centers` for that panel.

Two distinct point indices then sit at the same coordinates. FMM2D only skips
`i == j` self interaction, so the midpoint target receives a contribution from
the coincident Gauss source, while the local near correction
(`point_pairs`, which explicitly zeroes `r == 0`) assumes that contribution is
zero. The two no longer cancel.

Measured on the `reentrant` 64-panel mesh at k = 3 (S + K′ system, matrix-free
matvec against the dense assembly of the *same* equation):

| requested `fmm_quadrature_order` | effective order | min &#124;quad − centre&#124; | matvec relative error |
|---|---|---|---|
| 6 | 6 | 8.7e-4 | 3.0e-12 |
| **7** | **7** | **0.0** | **1.1e-2** |
| 8 | 8 | 6.7e-4 | 7.3e-12 |
| **9** | **9** | **0.0** | **8.9e-3** |

No exception is raised and no metadata records the degradation — the result is
simply wrong by about one per cent.

Reachability:
- `execution/options.py:105-107` accepts any integer 6…64 for
  `fmm_quadrature_order`, and `PULSE_FMM_UPDATES.md` advertises the knob.
- `twod/fmm/quadrature.py:24` computes `order = max(base, ceil(|k|h/2)+4)`,
  which is odd for `|k|h` in (8, 10]. Through the snapshot API this is out of
  reach because `_build_panels` enforces a 4-panels-per-wavelength floor
  (`|k|h <= pi/2`, `twod/geometry.py:1311-1328`), but any direct caller of
  `PulseKernel` — including future coarser-mesh policies — can hit it.
- Galerkin FMM is **not** affected: `GalerkinKernel` puts only Gauss points in
  the plan, with no midpoints.

Suggested fixes, cheapest first:
1. Round the effective order up to even inside `PulseKernel.__init__`
   (`self.order += self.order % 2`), and say so in the docstring.
2. Better: stop packing the midpoints into the source list. `hfmm2d_` already
   takes a separate target list (`nt`, `targ`, `ifpghtarg`) that
   `twod/fmm/kernel.py:evaluate` always passes as empty. Using it removes the
   collision by construction and also fixes F7.
3. Independently, reject odd values in `validate_options` when
   `discretization == 'pulse'`.

### F2 — Pulse converges at first order wherever K′ appears (High)

This is the most consequential accuracy property of the Pulse basis, and the
current documentation understates it. `PULSE_FMM_UPDATES.md` says "Pulse and
Galerkin have different errors at equal panel count" and tabulates the first
mesh that passes a fixed tolerance. That reads as a constant factor. It is a
**convergence-rate** difference, so the gap grows without bound under
refinement.

**Measurement A — fixed geometry (no geometric error).** A 24-gon of radius
0.06 m at 1 GHz, each side subdivided into 1…16 panels, so every run solves the
*same exact polygon*. Reference: Galerkin at 1536 panels. Backscatter complex
amplitude, relative error:

| panels | Galerkin TM | Galerkin TE | Pulse TM | Pulse TE |
|---|---|---|---|---|
| 24  | 2.20e-4 | 3.58e-4 | 6.80e-3 | 1.41e-2 |
| 48  | 4.33e-5 | 9.54e-5 | 1.96e-3 | 7.16e-3 |
| 96  | 1.45e-5 | 2.29e-5 | 5.64e-4 | 3.67e-3 |
| 192 | 4.61e-6 | 5.97e-6 | 1.61e-4 | 1.83e-3 |
| 384 | 1.45e-6 | 1.54e-6 | 4.59e-5 | 8.88e-4 |

Same geometry, IBC (50 − 10j Ω) and bulk dielectric (ε_r = 3 − 0.1j):

| panels | Gal IBC TM | Gal IBC TE | Pulse IBC TM | Pulse IBC TE | Gal diel TM | Gal diel TE | Pulse diel TM | Pulse diel TE |
|---|---|---|---|---|---|---|---|---|
| 24  | 1.72e-4 | 4.47e-4 | 8.83e-3 | 1.28e-2 | 6.70e-4 | 1.11e-3 | 2.04e-1 | 1.63e-1 |
| 48  | 1.92e-5 | 1.11e-4 | 3.98e-3 | 8.49e-3 | 5.54e-4 | 6.44e-4 | 1.08e-1 | 7.70e-2 |
| 96  | 5.71e-7 | 2.68e-5 | 1.87e-3 | 4.72e-3 | 1.16e-4 | 1.20e-4 | 5.64e-2 | 3.86e-2 |
| 192 | 1.10e-6 | 6.97e-6 | 8.92e-4 | 2.42e-3 | 2.87e-5 | 2.65e-5 | 2.93e-2 | 1.98e-2 |
| 384 | 5.77e-7 | 1.80e-6 | 4.24e-4 | 1.19e-3 | 6.99e-6 | 6.19e-6 | 1.53e-2 | 1.03e-2 |

Pulse halves per mesh doubling (O(h)); Galerkin quarters (O(h²)). The single
exception is **PEC TM**, where the CFIE contains no K′ term and Pulse keeps a
near-second-order rate (ratio ≈ 3.5) — this is the case the Pulse benchmarks
were built around.

**Measurement B — against the analytic cylinder series.** Circle of radius
0.06 m at 1 GHz (ka = 1.258), n-gon refined from 64 to 1024 panels, compared
with `ghost_backend/validation/cylinder`:

| panels | PEC TM gal / pulse | PEC TE gal / pulse | dielectric TM gal / pulse (σ) |
|---|---|---|---|
| 64   | 2.10e-3 / 3.03e-3 | 2.36e-3 / 4.71e-3 | 6.72e-3 / 1.10e-1 |
| 256  | 1.33e-4 / 1.90e-4 | 1.47e-4 / 1.45e-3 | 4.27e-4 / 2.71e-2 |
| 1024 | 8.54e-6 / 1.19e-5 | 9.18e-6 / 3.88e-4 | 2.70e-5 / 6.76e-3 |

At 1024 panels the dielectric Pulse error is 250× the Galerkin error, and the
ratio doubles with each further refinement.

**Measurement C — genuine polygon with sharp corners.** A 0.6 × 0.25 m PEC
rectangle at 1 GHz, 30° incidence, reference Galerkin at 1024 panels. Here both
discretizations are corner-limited, and the rate gap closes, but Pulse is still
10–20× worse at equal panel count:

| panels | Galerkin TM | Galerkin TE | Pulse TM | Pulse TE |
|---|---|---|---|---|
| 48  | 1.43e-2 | 6.10e-2 | 1.33e-1 | 2.96e-1 |
| 96  | 5.45e-3 | 1.76e-2 | 4.98e-2 | 1.03e-1 |
| 192 | 1.99e-3 | 5.64e-3 | 1.99e-2 | 5.37e-2 |
| 384 | 6.40e-4 | 1.71e-3 | 8.04e-3 | 3.21e-2 |

Assessment. The behaviour is consistent with theory rather than with a coding
error: on a faceted boundary the curvature of the K′ kernel is concentrated at
the vertices, and midpoint collocation samples that distribution with O(h)
consistency, whereas Galerkin testing integrates it and recovers O(h²). Both
discretizations were independently confirmed to be solving the *same* discrete
equation (the repo's own `test_dense_compressed_fmm_share_the_same_pulse_equation`
and my dense/FMM cross-checks agree to 1e-11), so this is a property of the
discretization, not of the assembly.

What should change is the guidance and the gating, not the maths:
- State the rate explicitly in `PULSE_FMM_UPDATES.md`: "Pulse is second order
  for closed PEC TM and first order for every formulation containing K′ (TE
  PEC, IBC, dielectric/multi-region). Its advantage at equal panel count is
  therefore not preserved under refinement."
- The published timing table ("Pulse FMM 41.6 % faster, 88.8 % less RAM") is a
  flat 6 × 2.5 m rectangle — a case with exact geometry and only four corners,
  i.e. the most favourable one for Pulse. It is fair as stated, but it should
  not be read as representative of curved or dielectric bodies.
- Consider warning (or requiring an explicit acknowledgement) when Pulse is
  selected together with a dielectric/multi-region or IBC geometry.

### F3 — Pulse TE PEC interior resonance passes the numeric quality gate (High)

TE PEC uses the single-layer Neumann equation (−½I + K′)σ = −∂u_inc/∂n, which
has spurious interior resonances. The combined-field remedy is gated on TM
only (`twod/pulse/runtime.py:66`, `twod/fmm/runtime.py:60`), so TE has no
protection on either discretization. `twod/nystrom.py` already documents the
same limitation for its own TE path.

PEC circle, radius 0.06 m, 512 panels, backscatter complex amplitude relative
error at ka = 3.8317 (j₁,₁, an interior Dirichlet eigenvalue of the disk):

| discretization | dense | FMM |
|---|---|---|
| Galerkin TE | 3.46e-4 | 3.46e-4 |
| **Pulse TE** | **1.24** | **1.24** |

That is a 124 % error. Frequency scan (512 panels, TE):

| ka | 3.60 | 3.70 | 3.80 | **3.8317** | 3.85 | 3.90 | 4.00 |
|---|---|---|---|---|---|---|---|
| Galerkin | 9.4e-5 | 1.0e-4 | 1.1e-4 | 3.5e-4 | 1.1e-4 | 1.1e-4 | 1.1e-4 |
| Pulse | 7.5e-3 | 1.4e-2 | 5.5e-2 | **1.24** | 9.4e-2 | 2.4e-2 | 8.4e-3 |

The narrow, localized spike confirms the resonance interpretation. Galerkin
rides through it because its consistency error is ~3 000× smaller.

The dangerous part is the detection, not the spike:

| entry point | outcome |
|---|---|
| `solve_monostatic_rcs_2d` (uncertified), `compute_condition_number=True` | returns the 124 %-wrong amplitude; **condition estimate 1.7e3** (limit 1e6); `quality_gate.passed == True` |
| `solve_monostatic_rcs_2d_certified` | correctly raises: "RMS normalized complex-field delta 0.0542 exceeds limit 0.02" |

The condition estimate is useless here: Pulse's larger discretization error
*shifts* the discrete spurious eigenvalue away from the exact ka, so the matrix
is well conditioned (1.7e3, versus 7.3e4 for Galerkin at the same ka) while the
solution is wrong. Any monitoring that relies on `condition_est_max` to catch
resonance corruption will not fire.

Recommendations:
1. Treat the certified base/fine gate as the only resonance guard, and say so —
   documenting that survey/uncertified 2-D runs have no resonance protection.
2. Longer term, give TE a combined-field option. The Galerkin side already has
   the hypersingular operator (kind `'W'`) needed for a Burton–Miller coupling;
   Pulse would need it added.
3. At minimum, stop presenting `condition_est_max` as evidence against
   resonance error in the metadata/reporting.

### F4 — TM PEC combined field exists only on the FMM backend (Medium)

For a closed pure-PEC TM geometry:

- `twod/fmm/runtime.py:60-68` builds `0.5·M + K + (−jk)·S` — a Brakhage–Werner
  combined field.
- `twod/formulations/robin.py:27-30` (dense) returns the bare single-layer
  matrix `S` — the pure EFIE.
- `compressed/coefficients.py:113-119` (compressed) likewise assembles only `S`
  for PEC rows.
- Pulse applies the CFIE consistently across all three backends
  (`twod/pulse/runtime.py:63-71`, used by dense, compressed and FMM alike).

So for Galerkin the *integral equation* depends on which factorization the user
(or `Automatic`) picks. In the far field the difference turned out to be benign — the
EFIE's resonant null space radiates nothing outside — and the two backends
agreed closely at three interior Dirichlet eigenvalues on a 512-panel PEC
circle (TM backscatter amplitude error, dense vs FMM): ka = 2.4048,
6.27e-5 vs 6.21e-5; ka = 3.8317, 9.91e-5 vs 9.81e-5; ka = 5.5201,
1.42e-4 vs 1.41e-4. The surface densities are not protected the same way, and
`compute_boundary_densities` runs on the dense path.

Either make the choice uniform across backends, or record the representation in
metadata and document that dense/compressed Galerkin TM PEC is an EFIE with
interior-resonance sensitivity in the density (not the far field).

### F5 — `fmm_quadrature_order` validation permits unsafe odd orders (Low)

`execution/options.py:105-107` allows any integer in 6…64. Given F1, odd values
are unsafe for `discretization='pulse'`. `PULSE_FMM_UPDATES.md` only exemplifies
`fmm_quadrature_order=8`, so the exposure is small, but validation is the right
place to close it.

### F6 — `blocks()` computes every near pair twice (Low, performance)

`twod/pulse/coefficients.py:150-158`:

```python
orders[near] = order                       # force the full point rule
...
for q in np.unique(orders):
    take = orders == q
    part = point_pairs(g, k, r[take], c[take], kinds, int(q))   # near pairs included
...
if np.any(near):
    exact = accurate_pairs(g, k, r[near], c[near], kinds)
    for kind in kinds: chunk[kind][near] = exact[kind]          # overwritten
```

The full-order `point_pairs` evaluation of the near entries is always thrown
away. Masking them out (`take = (orders == q) & ~near`) removes one full-order
Gauss evaluation per near pair from every dense and compressed assembly. The
near set is O(n) out of O(n²) entries, so the saving is modest, but it is free.

### F7 — Pulse FMM evaluates roughly twice the targets it needs (Low, performance)

`PulseKernel._point_apply` hands the native kernel a single list that is both
the source and the target set. In the forward direction only the `n` midpoint
potentials are used and the `n·order` Gauss-point values are discarded; in the
adjoint direction the reverse. `twod/fmm/kernel.py:evaluate` always calls
`hfmm2d_` with `nt = 0`, even though the routine accepts a distinct target list
(`targ`, `ifpghtarg`). Splitting sources from targets would cut the evaluation
work substantially and, as noted in F1, would remove the coincident-point
hazard. The current design is deliberate (one plan reused for both directions);
the comment in `twod/pulse/kernel.py:1-6` explains it. Worth revisiting if Pulse
FMM throughput matters — the standalone benchmark already shows Pulse FMM
losing to Pulse dense at 2 048 panels / 3 angles (13.80 s vs 2.12 s).

### F8 — Three dead "docstrings" in `operators.py` (Cosmetic)

`_sk_blocks_near_linear` (line 816), `_linear_element_incident_dn_load_many`
(line 2297) and `_farfield_linear_density_many` (line 2338) each place their
triple-quoted description *after* an early-return `if`, so it is a no-op string
expression, not `__doc__`. `help()` and any doc tooling show nothing. Move each
block above the first statement.

### F9 — `nystrom.py` hard-requires an optional dependency (Low)

`requirements/constraints-windows-py312.txt` lists psutil under "Optional
internal-team capabilities. They remain optional at runtime", and every other
use in the package is guarded (`twod/solver.py:1034`, `:1097`,
`execution/metrics.py:48`, `hpc/check_environment.py:40` all wrap it in
`try/except`). `twod/nystrom.py:19` does not:

```python
if memory_gib is None:
    import psutil
    memory_gib = min(8., .5*psutil.virtual_memory().available/1024**3)
```

`memory_gib=None` is the default, so on a machine without psutil both
`test_nystrom` cases fail with `ModuleNotFoundError` — which is exactly what
happened in this audit environment. Fall back the same way `solver.py` does
(or reuse `_detect_available_gb`).

---

## 3. Verified correct

These were checked rather than assumed, and they matter because they rule out
whole classes of explanation for F2 and F3.

**Sign and phase conventions are self-consistent end to end.** GHOST's
`G = (i/4)H₀⁽²⁾(kr)` satisfies `(Δ + k²)G = +δ` — the negative of the textbook
fundamental solution — which is exactly why the exterior traces read `+σ/2 + K`
for the double layer and `−σ/2 + K′` for the single layer under the inward
normal convention, matching `twod/fmm/runtime.py:63-71` and the mass-matrix
jumps in `twod/formulations/regions.py`. The Brakhage–Werner coupling
`η = −jk` is the correct sign in that convention, and `twod/nystrom.py`'s
outward-normal variant (`η = +jk`, `−½I`) is the consistent mirror image.

**The FMM2D conjugation transform is right.** `twod/fmm/kernel.py:evaluate`
passes `z = conj(k)` with conjugated strengths and returns `-p.conj()`, which
maps the vendor's `(i/4)H₀⁽¹⁾` convention to GHOST's `(i/4)H₀⁽²⁾`; the vendor's
dipole term is `v · ∇_y G`, matching GHOST's `K` kernel with a positive sign.

**`_single_layer_self_block_exact` is exact.** The shape-pair convolution
weights `C_diag = (2 − 3u + u³)/3` and `C_off = (1 − u³)/3` are the correct
weights, and the moment/log-moment series reproduces the `H₀⁽²⁾` logarithmic
expansion. Checked against `quad_vec` at (L, k) = (0.01, 20.9), (0.2, 20.9) and
(0.05, 3 − 0.4j): max relative difference **9.0e-16**.

**`polynomial_quadrature.log_moments` is exact.** Compared against `dblquad` of
`φᵢ(x)φⱼ(y)ln|x−y|` for degree 1: max difference **4.5e-10** (quadrature-limited).

**The Maue/hypersingular forms agree.** `Dᵀ S D` with the linear
`derivative_matrix` equals `_TANGENT_OUTER · sum(S)` exactly, so the FMM
Galerkin `'W'` block (`twod/fmm/galerkin.py:_build`) and the dense
`_hypersingular_block_from_s_block` build the same operator.

**Pulse's analytic singularity subtraction is correct.** The static
antiderivatives in `near_pairs` (`x(½ln(x²+d²) − 1) + d·atan2(x, d)` for the
potential, and the tangential/perpendicular pair for the gradient) are the exact
primitives; the split-Gauss remainder is applied to the correct residual, and
the self single-layer term is replaced by an adaptive integral with a length
correction whose omitted term is O(ΔL²).

**Matrix-free adjoints are genuinely conjugate transposes.** `FMMSystem._apply`
deliberately *swaps* `mask` and `coefficient` when it calls `kernel.apply` in
the adjoint direction; combined with conjugating the whole product this yields
`diag(conj(mask))·Kᴴ·diag(conj(coefficient))`, which is the correct
`(D_c K D_m)ᴴ`. `PulseKernel.apply_combined`'s adjoint likewise recovers
`conj(η)` from the outer conjugation. Both are covered by the repo's own tests.

**The multi-region Pulse system matches the Galerkin one term for term.** The
`±½` and `−½·inverse_beta` collocation jumps in `twod/pulse/runtime.py:78-93`
correspond exactly to the mass-matrix jumps in
`regions._assemble_system_fresh`, and the operator routes reuse the same
`multi_outputs` weights.

**Far-field normalization.** `σ = |B|²/(4k)` is the correct scattering width for
the stored amplitude `B = ∫σ(y)e^{jk ô·y}ds` given `A = (j/4)√(2/πk)e^{jπ/4}B`,
and Pulse's exact `sinc` panel moment equals `∫exp(jk d·y)dl` over a straight
panel.

**Near-pair enumeration is complete.** `fmm/galerkin.near_pairs` bins panels by
`log2(length)` and queries each tree with radius `3·max(l_i, upper_bin)`, which
covers every pair that the `dist ≤ 3·max(l_i, l_j)` predicate can accept — so no
near pair is missed when panel lengths vary. The same 3× criterion is used by
the dense assembly (`far_ratio = 3.0`) and by `pulse/coefficients.blocks`.

---

## 4. Reproduction

Scripts are in this directory; each writes the tables quoted above.

```bash
# build the optional native libraries first (needs gfortran)
cd tools/GHOST
python -m ghost_backend.twod.fmm.native.build
python -m ghost_backend.twod.assembly.native.build

cd audit/twod_2026-09
python conv.py      # F2 measurement B  (circle vs analytic series)
python fixed.py     # F2 measurement A  (fixed 24-gon, no geometry error)
python rect.py      # F2 measurement C  (rectangle)
python reson.py     # F3 table          (interior resonances, 4 backends)
python reson2.py    # F3 gate behaviour (certified vs uncertified)
python scan.py      # F3 frequency scan
python oddorder.py  # F1
python checks.py    # F8-adjacent: log_moments verification
python checks2.py   # self-block series and Maue identity verification
```

`common.py` holds the shared harness (it reuses the repo's own
`test_2d_capability_acceptance._circle/_segment` fixtures and
`ghost_backend.validation.cylinder` analytic references). The console output
captured for this report is in `measurements/`, one file per script, plus
`measurements/test_suite.out` for the wider regression sweep.

---

## 5. Not covered

- BoR, 3-D, GRIM integration, GUI, HPC scheduling and the checkpoint/session
  machinery were out of scope.
- Bistatic Pulse, thin sheets and thin-layer approximations are rejected by
  `pulse/runtime.validate_geometry` and were not exercised.
- `twod/nystrom.py` is a standalone experimental spectral solver with its own
  tests; it is not reachable from the production dispatch. It is the only path
  in the package that retains curved geometry, and — given F2 — it is the most
  promising foundation for a future high-order curved-boundary option.
- Timing and memory claims in `PULSE_FMM_UPDATES.md` were not re-measured; this
  audit only re-examined the accuracy claims that accompany them.
