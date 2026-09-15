# GHOST 2-D solver audit — Pulse (P0 collocation) and Galerkin

Date: 2026-09-15
Scope: `tools/GHOST/ghost_backend/twod/**` (both discretizations, all three
factorization backends), plus the compressed/FMM oracles that share those
operators.

Status: the audit itself was read-only; a follow-up commit on this branch then
fixed F1, F6, F8, F9, F10, F11 and F12 (see §1.1). F2, F3 and F4 are left open
because each is a design decision, not a defect to patch.

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
After the F1/F6/F8/F9/F10/F11 fixes the same sweep plus `test_pulse`, `test_fmm`
and `test_fmm_efficiency` gave **170 passed / 1 skipped / 48 subtests** (the skip
is the PySide6 GUI case), and the fixed-polygon convergence tables in §2 F2
re-ran byte-identical.

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
| F7 | Low | Performance | `PulseKernel` asks the FMM to evaluate at every point rather than only the collocation targets; measured cost ~9 % of FMM time |
| F13 | Medium | Pulse performance | The Pulse dense assembly does not scale with `assembly_threads`: 1.02× at two threads and 0.70× at four, where Galerkin gets 1.57×/1.63× on the same box |
| F8 | Cosmetic | Maintainability | Three "docstrings" in `operators.py` are dead string expressions placed after the first statement |
| F9 | Low | Packaging | `twod/nystrom.py` imports `psutil` unguarded although the requirements declare it optional at runtime |
| F10 | High | Galerkin + Pulse planning | `fmm/galerkin.near_pairs` passes an array radius to `cKDTree.query_ball_point`, which needs SciPy 1.9; older SciPy raises `TypeError: only size 1 arrays can be converted to Python scalars` |
| F11 | High | Pulse | `pulse/coefficients` imports `scipy.integrate.quad_vec`, which needs SciPy 1.4, for a scalar integrand |
| F12 | Medium | FMM backend | `twod/fmm/factor.py` uses four keywords newer than the rest of the package tolerates: `LinearOperator(rmatmat=)` (SciPy 1.4), `gmres`/`lgmres` `rtol=` (1.12), `atol=`/`callback_type=` (1.1) and `np.argsort(kind='stable')` (NumPy 1.15) |

Verified-correct items are listed in §3; they matter because they bound where
the problems can be.

### 1.1 Fixes applied on this branch

| # | Change |
|---|---|
| F1 | `PulseKernel.__init__` rounds the effective quadrature order up to even (`twod/pulse/kernel.py`). Requesting order 7 now runs order 8 and 9 runs 10; the measured matvec error returns to ~5e-12 in every case. |
| F6 | `blocks()` skips near entries in the point-rule pass instead of computing and overwriting them (`twod/pulse/coefficients.py`). Results are bit-identical. |
| F8 | The three docstrings in `operators.py` moved above the first statement, so they are real `__doc__` again. |
| F9 | `nystrom.operators` uses the solver's guarded `_detect_available_gb()` instead of importing psutil directly, and keeps the previous 8 GiB cap when no probe reports anything. |
| F10 | `_query_balls()` falls back to one scalar-radius query per distinct radius when SciPy rejects an array `r`, detected once and cached. |
| F11 | `_self_integral` integrates the real and imaginary parts with `scipy.integrate.quad`; verified to 6.3e-15 against `quad_vec` for k from 1e-3 to 200 and panel lengths from 1e-4 to 3 m. |
| F12 | `_krylov_kwargs()` builds the tolerance/callback keywords from the solver's own signature (the idiom `compressed/factor.py:80` already used), `_equation_operator()` drops `rmatmat` when SciPy will not take it, and the stable sort asks for `'mergesort'`. |

F2 (Pulse convergence rate), F3 (TE interior resonances) and F4 (backend-dependent
TM PEC formulation) are **not** fixed here: each needs a decision about what the
product should do, not a patch.

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

**Fixed on this branch** by option 1 below; requesting order 7 now runs order 8
and 9 runs 10, and the matvec error returns to ~5e-12 in all four cases above.

Fixes considered, cheapest first:
1. Round the effective order up to even inside `PulseKernel.__init__`
   (`self.order += self.order % 2`).
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

### F5 — `fmm_quadrature_order` validation permits odd orders (Low)

`execution/options.py:105-107` allows any integer in 6…64, and before the F1 fix
odd values were unsafe for `discretization='pulse'`. With `PulseKernel` now
rounding the effective order up to even, an odd request is honoured as the next
even order rather than silently corrupting the product, so no validation change
is needed. Recording it here because the rounding is the only thing keeping that
input safe.

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

### F7 — Pulse FMM evaluates at every point, not just the targets (Low, performance)

`PulseKernel._point_apply` hands the native kernel a single list that is both
the source and the target set, so the forward direction computes potentials at
all `n*order` Gauss points and uses only the `n` midpoints (the adjoint does
the reverse). `twod/fmm/kernel.py:evaluate` always calls `hfmm2d_` with
`nt = 0`, although the routine accepts a distinct target list (`targ`,
`ifpghtarg`).

**Measured, not estimated.** Calling `hfmm2d_` directly with a real target list
(sources = Gauss points, targets = midpoints, `ifpgh=0`) against the current
combined call, one thread, eps 1e-10, order 8:

| panels | all points are targets | collocation targets only | saving |
|---|---:|---:|---:|
| 2 048 | 2.040 s | 1.859 s | 8.9 % |
| 8 192 | 8.548 s | 7.726 s | 9.6 % |

The two agree to machine precision (max relative difference 9.2e-16 and 0.0).
An earlier draft of this report guessed the saving was "roughly half"; it is
about a tenth, because at this tolerance the FMM's cost is dominated by the
translations rather than by evaluation at the extra points. Worth doing if Pulse
FMM throughput matters — it would also have removed F1 by construction — but it
is not the lever it looked like.

### F13 — the Pulse dense assembly does not use its threads (Medium, performance)

Same box (4 cores), same problem — a 2 048-panel PEC circle, 1 GHz, three
angles, both polarizations, wall time for the whole solve:

| `assembly_threads` | Galerkin dense | Pulse dense |
|---|---:|---:|
| 1 | 14.51 s | 6.60 s |
| 2 | 9.24 s (1.57x) | 6.50 s (1.02x) |
| 4 | 8.88 s (1.63x) | 9.44 s (**0.70x**) |

Pulse is 2.2x faster than Galerkin on one thread, which is the advantage the
release notes claim. But it gains nothing from a second thread and is *slower
than serial* on four, at which point Galerkin overtakes it.

The cause is the GIL, not memory bandwidth or core count: running **four
concurrent single-threaded processes** finishes all four solves in 6.92 s wall,
against 6.4 s for one alone — so the work parallelises across the four cores
essentially perfectly when the interpreter lock is not shared.

The mechanism is visible in the profile. `dense_matrix` tiles at 32 rows x 512
columns, and `blocks()` splits each tile into 4 096-pair chunks and then into up
to four graded-order groups plus the near set, each selected with a boolean
mask. On a 1 024-panel run that is 1 520 `point_pairs` and 528 `near_pairs`
calls across 64 tiles — roughly 32 short array calls per tile, each on a masked
subset, so a large share of the time is GIL-held dispatch and temporary
allocation rather than long ufunc stretches. `point_pairs` self time (0.319 s)
is nearly as large as all the Bessel evaluation inside `green` (0.405 s) on that
run.

Galerkin does not have this problem because its far pass was written for buffer
reuse: `_far_green_into`, `_far_hankel1_into` and `_axpy_into` in
`operators.py` evaluate one tile's kernel directly into preallocated `out=`
buffers (writing `y0`/`j0` straight into the halves of the complex result), so
each tile is a few long GIL-releasing calls. The Pulse coefficient kernels have
no equivalent.

Fixing it means giving `pulse/coefficients` the same treatment — larger tiles,
preallocated per-thread scratch, and `out=` Bessel evaluation — rather than any
change to the discretization. Until then, `assembly_threads` above 2 is
counter-productive for Pulse, and the release-note timings (four threads on an
8-core machine) understate what Pulse could do.

### F8 — Three dead "docstrings" in `operators.py` (Cosmetic)

`_sk_blocks_near_linear` (line 816), `_linear_element_incident_dn_load_many`
(line 2297) and `_farfield_linear_density_many` (line 2338) each place their
triple-quoted description *after* an early-return `if`, so it is a no-op string
expression, not `__doc__`. `help()` and any doc tooling show nothing.
**Fixed on this branch**: each block moved above the first statement.

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
happened in this audit environment.
**Fixed on this branch** by reusing the solver's guarded `_detect_available_gb()`,
keeping the previous 8 GiB cap when no probe reports anything.

### F10 — array-valued KD-tree radius needs SciPy 1.9 (High)

`twod/fmm/galerkin.py` called

```python
candidates = tree.query_ball_point(centers[start:stop], 3*np.maximum(lengths[start:stop], upper))
```

Per-point (array-valued) `r` was added to `cKDTree.query_ball_point` in SciPy
1.9. Earlier releases coerce `r` with `float()`, so the call raises
`TypeError: only size 1 arrays can be converted to Python scalars`.

This is not confined to FMM runs: `_dense_formulation_resources`
(`twod/solver.py:1714-1718`) calls `near_pairs(..., count_only=True)` for memory
planning on **every** 2-D solve, so a plain Galerkin `run_hpc_monostatic` run
fails there before any assembly happens. It is the only array-radius query in
the package; every other `query_ball_point` call passes one point and a scalar.

Fixed by `_query_balls()`, which tries the vectorized call once, caches whether
it worked, and otherwise issues one query per distinct radius. Verified to
return identical pair lists with the fallback forced, and an end-to-end
Galerkin and Pulse solve completes against a KD-tree subclass that rejects
array radii the way SciPy 1.0 does.

### F11 — `quad_vec` needs SciPy 1.4 and was not needed at all (High)

`twod/pulse/coefficients.py` imported `scipy.integrate.quad_vec` at module
scope, so on SciPy older than 1.4 every Pulse run died with
`ImportError: cannot import name 'quad_vec'` before reaching the solver. The
only use was `_self_integral`, whose integrand is a **scalar** complex value —
`quad_vec` bought nothing over `quad`.

Fixed by integrating the real and imaginary parts with `scipy.integrate.quad`
(available in every SciPy), keeping the same tolerances. QUADPACK's
epsilon-extrapolation handles the endpoint logarithm at t=0 as well as
`quad_vec`'s adaptive rule did: across k = 1e-3 … 200 (real and complex) and
panel lengths 1e-4 … 3 m the worst relative difference from the old result is
**6.3e-15**, with warnings promoted to errors so an `IntegrationWarning` would
have failed the check.

Note on the wider context: F10, F11 and F12 were reported from an environment
running SciPy 1.0.0. The documented headless floor is **SciPy >= 1.14, NumPy >=
2.0** (`requirements/hpc.txt`, `HPC.md` §"Python environment"), and
`ghost_backend/hpc/check_environment.py` already fails loudly below it. These
fixes remove the avoidable hard dependencies on newer APIs that a scan of
`twod/`, `linalg/`, `compressed/` and `execution/` turned up; they are not a
claim that the package is supported on SciPy 1.0, which was not available to
test against directly. What was tested is that both discretizations run to the
same answers on dense, compressed and FMM with every one of those APIs made
unavailable — see `legacy_sim.py`, which asserts that each fallback was
actually taken rather than merely present.

### F12 — the FMM backend uses four keywords newer than the rest of the package (Medium)

`twod/fmm/factor.py` was the one module left demanding APIs newer than F10/F11
did:

| call | keyword | needs |
|---|---|---|
| `LinearOperator((n,n), …, rmatmat=rmm)` | `rmatmat` | SciPy 1.4 |
| `gmres(…, rtol=…)`, `lgmres(…, rtol=…)` | `rtol` (was `tol`) | SciPy 1.12 |
| `gmres(…, atol=0., callback_type='legacy')` | `atol`, `callback_type` | SciPy 1.1 |
| `lgmres(…, prepend_outer_v=True)` | `prepend_outer_v` | not in the oldest releases |
| `np.argsort(…, kind='stable')` | `'stable'` | NumPy 1.15 |

`compressed/factor.py:80` already solved the same problem with
`inspect.signature`, so the FMM backend was simply inconsistent with its
sibling.

**Fixed on this branch.** `_krylov_kwargs()` builds the keyword set from the
solver's own signature; `_equation_operator()` tries `rmatmat` once and falls
back without it; the sort asks for `'mergesort'`, which every NumPy accepts,
gives the same stable ordering, and is already what the sibling clustering
routine in `linalg/hierarchical.py:33` uses.

`ghost_backend/tests/test_scipy_compatibility.py` (added to the
`scripts/check_headless.py` qualification list) guards all of this: it feeds
`_krylov_kwargs` stand-ins carrying the exact SciPy 1.0 signatures, refuses
`rmatmat` and checks the adjoint still matches, runs `near_pairs` against a
scalar-only `query_ball_point` and asserts the fallback was actually taken, and
imports `pulse/coefficients` with `quad_vec` removed.

None of this changes behaviour on a supported runtime. The legacy `tol` is
measured against `norm(b)`, which is exactly what `rtol` with `atol=0` means;
`callback_type='legacy'` is the old default, so omitting it matches; a missing
`rmatmat` costs one `rmatvec` per column instead of a batched adjoint, and a
missing `prepend_outer_v` only reorders augmentation vectors — and every
reconstructed solution is still checked against the original, unscaled
equation.

---

## 2.1 Pulse efficiency: what holds up

These were measured on the same 4-core box while investigating F13, and are
recorded because they are the answers to "is anything unnecessarily built or
held".

**Vectorization.** There is no Python loop over element pairs anywhere hot.
`blocks()` loops only over bounded chunks and over the (at most four) graded
quadrature-order groups; `near_pairs` loops over the two split subintervals.
The single per-panel Python loop is the self-term assignment in `near_pairs`,
and the `lru_cache` keyed on a 12-significant-digit length collapses it to one
adaptive integral per *distinct* panel length. Because GHOST meshes each
straight primitive uniformly, that is the primitive count, not the panel count:
a 2 048-panel reentrant mesh has **5** distinct lengths and spends 0.035 s on
all its self terms. The cache design is well matched to the geometry model.

**The self integral is cheap.** Switching it from `quad_vec` to `quad` on the
real and imaginary parts (F11) turned out to be 7.4x faster as well as more
portable — 252 integrand evaluations against 1 659, because QUADPACK's
extrapolation handles the endpoint logarithm that `quad_vec`'s bisection has to
chase. 0.66 s against 4.89 s for 400 distinct lengths, agreeing to 1.9e-15.

**Nothing large is held unnecessarily.** For a 2 048-panel PEC circle the dense
run peaks at 231 MiB against a 64 MiB matrix; the doubling is the matrix plus
its LU (`lu_factor` is called without `overwrite_a`), and the matrix is
deliberately retained for the unscaled residual check. Galerkin peaks higher on
the same problem (276 MiB). The FMM path keeps 1.3 MiB of near operators at
2 048 panels and 2.5 MiB at 4 096, against 64 and 256 MiB dense — the ~89 % RAM
reduction in the release notes holds. `PulseKernel` does keep both `near` and
`correction` per kind, doubling near storage, but that is 1.3 MiB and not worth
restructuring.

**FMM Python overhead is negligible.** 95.5 % of a 24.5 s, 2 048-panel FMM run
is inside the native `evaluate`; `_point_apply`'s own time is 0.008 s. Whatever
is slow about Pulse FMM at small panel counts (the release notes already say it
loses to dense at 2 048 panels / three angles) is the native kernel and the
iteration count, not the Python wrapper.

**Small waste, not worth changing.** `PulseOracle.get_with_error` allocates a
same-shape zero array for the error estimate that `dense_matrix` immediately
discards, and `blocks()` copies through a `chunk` dict before writing into the
result slice. Both are inside `get_with_error`'s 0.017 s of a 1.93 s run.
`apply(adjoint=True)` rebuilds `correction.conj().T` on every call, but the
adjoint is only used for condition estimation.

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
python checks.py    # log_moments verification
python checks2.py   # self-block series and Maue identity verification
python verify_fixes.py  # F10/F11 fixes: fallback equivalence and quad accuracy
python legacy_sim.py    # F10/F11/F12: both discretizations on dense, compressed
                        # and FMM with every post-SciPy-1.0 API removed
python prof.py 1024 dense   # §2.1 profile (also: prof.py 2048 fmm)
python selfint.py           # §2.1 self-integral quad vs quad_vec
python targets.py           # F7 split source/target measurement
python one.py dense 2048 4  # F13 one timed solve: mode, panels, threads
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
