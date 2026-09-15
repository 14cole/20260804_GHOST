# 2-D workflow, option surface and method upkeep

Date: 2026-09-15. Companion to `TWOD_SOLVER_AUDIT.md`, which covers correctness.
This one answers three questions: is the option surface too large, can either
discretization be made cheaper in time or RAM, and is every method inside them
earning its keep.

Measurements are on the same 4-core box as the audit (CPython 3.11, NumPy 2.4.6,
SciPy 1.17.1, native FMM built). Where a judgement depends on what the product
is *for* rather than on what the code does, it says so.

---

## 1. The option surface

| surface | count | notes |
|---|---:|---|
| `execution_options` keys | 23 | the full API/profile surface |
| exposed in the GUI's Advanced Settings | 10 | factorization, basis, mesh strategy, RAM budget, compressed storage, assembly threads, BLAS threads, temp dir, RHS compression, angle batch |
| API/env only | 13 | `basis_order`, `far_grading`, `far_quadrature_order`, `assembly_tile`, `pulse_pec_cfie`, the six `fmm_*` knobs, `version`, `temporary_directory` |
| driver module constants | 6 | `SOLVE_PRESET`, `ADVANCED_OVERRIDES`, `ASSEMBLY_THREADS`, `BLAS_THREADS_PER_WORKER`, `MAX_SOLVE_GB`, `TASKS_PER_CHILD` |

The split is already better than the raw count suggests. `basis_order` is **not**
a user choice at all — it is the internal lever that "Automatic adaptive
accuracy" pulls, and it is correctly absent from the GUI. The `fmm_*` knobs are
expert overrides that Automatic sets. That is the right instinct; the problem is
that it was applied unevenly.

**Of the 10 GUI controls, only three are decisions a user can actually make
well:**

1. **Basis** (Galerkin / Pulse) — a physics decision with real consequences
   (§3), and currently presented as a performance choice.
2. **Mesh strategy** — Automatic / Global / Local.
3. **RAM budget** — the one number only the user knows.

The other seven are resource knobs the machine knows better than the user:

- *Factorization* — Automatic already ranks dense/compressed/FMM on a cost and
  memory model. Leaving the manual choice visible invites a worse answer.
- *Compressed storage MiB* — only meaningful if the user overrode factorization.
- *Assembly threads / BLAS threads* — derivable from cores and the unit
  concurrency; see §4, where the current defaults are measurably wrong.
- *Angles per batch*, *RHS compression*, *temporary directory* — throughput
  details with safe automatic values.

Nothing here needs deleting. What it needs is **demotion**: one "Advanced
(expert overrides)" collapsed section containing the seven, with the three real
decisions above it. That is a layout change, not an API change, and it does not
strand any saved profile.

The tooltips are unusually good — they state what is *not* guaranteed, which is
rare and worth preserving verbatim through any reshuffle.

---

## 2. Method inventory

Monostatic 2-D dispatch, in the order `solve_monostatic_rcs_2d_single_polarization`
tests it:

| # | branch | trigger | backends |
|---|---|---|---|
| 1 | Pulse collocation | `discretization='pulse'` | dense, compressed, FMM |
| 2 | Sheet TM / TE | all TYPE 1 | dense, compressed, FMM(TM/TE) |
| 2b | Thin dielectric layer | all `thin_layer` | dense, compressed |
| 3 | Mixed sheet + PEC | TYPE 1 + pure PEC | dense, compressed |
| 4 | TE Robin MFIE | `pol=='TE' and all robin` | dense, compressed, FMM |
| 5 | Multi-region indirect SLP | ≥2 regions | dense, compressed, FMM |
| 6 | Two-density dielectric | all transmission | dense, compressed, FMM |
| 7 | Robin BIE (TM EFIE / IBC) | all robin | dense, compressed, FMM |

Eight physical formulations × three backends × two discretizations is the real
combinatorial surface, and most of it is legitimate — each formulation solves a
genuinely different boundary condition, and all have test coverage (sheets 28
test files, dielectric 42, IBC 37, multi-region 18, thin layer 4).

### What is mechanically redundant

These are not judgement calls; they are duplication the code can lose without
losing a capability.

**Branch 4 is branch 7.** `_solve_te_robin_mfie` (`twod/solver.py:1338`) is a
pass-through: it calls `_normalize_public_2d_solver_method(solver_method)`,
discards the result, and forwards every argument to `_solve_robin_bie`. The two
dispatch blocks (lines 2298-2320 and 2434-2458) are identical apart from the
`formulation_label` string. Roughly 45 lines of dispatch exist to give TE a
different label.

**A dead label branch.** Because branch 4 catches every TE all-Robin geometry
and `continue`s, branch 7 only ever sees TM. Its `else` label — "2D Robin-BIE
IBC formulation (SLP representation)", line 2440 — is unreachable.

**A vestigial default.** `formulation_label` is initialised at line 1880 to
"2D BIE/MoM coupled dielectric trace formulation (linear Galerkin)". Every
branch overwrites it and the fallthrough raises, so the value is never
observable — and the solver's own docstring (line 1355) says that formulation
"has pre-existing sign/normalization issues that produce unphysical results".
A stale default naming a known-bad formulation is a trap for anyone grepping.

**Seven copies of the same 25-line sample builder.** The
`for idx, elev_deg in enumerate(elevations): samples.append({...})` block
appears seven times, once per branch, with identical keys. It is the main reason
`solve_monostatic_rcs_2d_single_polarization` is **837 lines**. Collapsing the
branches to `(label, solver_callable)` and running one shared tail would remove
several hundred lines with no behavioural change.

### What I am not qualified to call

Whether **thin sheets**, **thin dielectric layers** and **mixed sheet+PEC** earn
their upkeep is a product question, not a code question. What I can report: they
are the least-exercised paths (thin layer appears in 4 test files against 42 for
dielectric), they are the ones Pulse explicitly refuses, and they carry the
formulation-specific edge cases — Meixner endpoint pinning, the
`_geometric_sheet_endpoint_nodes` geometric-key logic, the TYPE 1 mixing
restrictions in `_assert_no_type1_sheet_for_mixed`. If they are rarely used,
they are where the maintenance cost concentrates. If they are the reason
customers buy the tool, that cost is simply the price. I have no evidence either
way and did not assume.

---

## 3. Which basis to offer

The audit measured this (fixed 24-gon, so no geometry error in play; see
`TWOD_SOLVER_AUDIT.md` F2/F3). Restated here because it should drive the UI:

| physics | Pulse | verdict |
|---|---|---|
| Closed PEC, TM | 2nd order, and now ~3x faster than Galerkin at 4 threads | **use Pulse** |
| PEC TE, IBC (either pol), dielectric / multi-region | **1st order** where Galerkin is 2nd | **use Galerkin** |
| Thin sheets, thin layers, bistatic, density export | not implemented | Galerkin only |

At 384 panels the dielectric error is 1.5e-2 (Pulse) against 7.0e-6 (Galerkin),
and the gap widens with refinement because it is a rate difference, not a
constant. Pulse TE also reached 124 % error at an interior resonance that the
condition gate did not catch.

**So "Pulse vs Galerkin" should not be a free-form performance dropdown.** The
solver already knows the boundary conditions before it picks a formulation. The
honest UI is to let Automatic choose the basis too: Pulse for closed pure-PEC TM,
Galerkin otherwise, with a manual override in the expert section. That removes
the one control most likely to be set wrong for the wrong reason, and it is the
single highest-value workflow change on this list.

---

## 4. Efficiency: what is left, measured

### Galerkin dense — 1.57x on four cores

n=2048 PEC circle, three angles, both polarizations: 14.06 s at one thread,
8.95 s at four. Profile at one thread (14.06 s total):

| | time | share |
|---|---:|---:|
| `_far_pass` (cumulative) | 10.22 s | 73 % |
| — `_far_green_into` + `_far_hankel1_into` + `_axpy_into` | 6.65 s | 47 % |
| near classification + touching/self pairs | ~3.4 s | 24 % |
| `lu_factor` | 0.88 s | 6 % |

The far pass is already well engineered — preallocated `out=` buffers, one
Bessel evaluation per tile, `y0`/`j0` written straight into the halves of the
complex result. The cap is the **near path, which is a Python loop over every
near pair** (`operators.py:1832`). Per pair it calls
`_linear_shared_interval_endpoint_info`, `requires_adaptive` and a distance
computation — 32 768 calls to the first alone at n=2048, plus 131 072
`_linear_param_to_point` and 155 654 `np.linalg.norm` calls. All of it holds the
interpreter lock, which is why four threads return 1.57x rather than the ~3.3x
the far pass alone achieves.

Two fixes, both using machinery the repo already has:

1. **Vectorize the near classification.** The predicates are simple geometry,
   and `assembly/separation.py:close_pairs` already computes exactly this kind
   of mask for all pairs at once with broadcasting. Worth roughly 13 % of
   assembly and, more importantly, it is most of what is blocking the threads.
2. **Batch the touching/self pairs.**
   `_integrate_linear_touching_duffy_sk_vectorized` is called once per pair
   (8 192 calls, 1.59 s). It is vectorized internally but tiny per call. The
   batching path `_integrate_linear_pairs_box_sk_batched` already exists for the
   fixed-order class; extending it to the touching class is the same pattern.

Expected: Galerkin assembly from 1.57x to roughly 2.5-3x on four cores. This is
the same class of problem as the Pulse chunk-size fix, one level down.

### What is *not* an opportunity

I expected `np.add.at` in the scatter to be slow — it is 29 044 calls for 0.99 s.
Measured against a `bincount` reformulation on the same access pattern, `add.at`
is **5x faster** (0.057 s vs 0.282 s for an equivalent workload). Modern NumPy
has optimized it. Leave it alone.

### Pulse — done, with ~9 % left

The chunk-granularity fix took four-thread dense assembly from 1.03x to 3.49x
(see F13). What remains is the FMM source/target split, measured at 8.9-9.6 %
(F7) — real but not a lever.

### RAM

Peak for a dense run is the matrix plus its LU, because `lu_factor` is called
without `overwrite_a` and the matrix is deliberately retained for the unscaled
residual check and for the partner polarization. That 2x is a design choice, not
waste, and it is what the compressed and FMM backends exist to avoid: at 2 048
panels the FMM keeps 1.3 MiB of near operators against a 64 MiB dense matrix.
Pulse dense already peaks lower than Galerkin on the same problem (231 vs
276 MiB). I found no unnecessary retention in either path.

---

## 5. Workflow changes worth making

**GUI**, in priority order:

1. Let Automatic pick the **basis**, not just the backend (§3). Demote the
   manual choice to expert overrides.
2. Collapse the seven resource knobs into one "Expert overrides" section, leaving
   basis (or nothing, if 1 lands), mesh strategy and RAM budget visible.
3. Derive **assembly/BLAS threads** from detected cores rather than shipping
   4 and 2 — §4 of the driver discussion measured that on a 4-core box a single
   unit wants both at the core count (8.65 s) rather than the current 4/2
   (10.07 s), a 16 % loss on the default.
4. Surface the certification outcome, not just the number. The audit found a
   124 % error that the condition estimate passed and only the base/fine mesh
   gate caught — so "mesh convergence certified" is the field that matters, and
   it should be the visible one.

**Both drivers:**

5. `BLAS_THREADS_PER_WORKER` has no `"auto"` while `ASSEMBLY_THREADS` does.
   Adding it (resolving to the same per-unit core share) removes the one number
   a user currently has to compute by hand.
6. The header comment should say what the audit measured: units parallelise
   perfectly across processes (4 units on 4 cores: 7.5 s via processes against
   13.5 s via threads), so threads are the fallback for when units run out.
   Right now nothing tells the user that ordering.
7. Print the chosen backend, basis and per-unit thread split in the pre-flight
   summary. The scheduler already computes all three.

**Not recommended:** removing any physical formulation, or changing the default
`blas_threads`, without measurements on the 64-96 core nodes these actually run
on. Everything in §4 was measured on four cores; the scaling conclusions should
be re-checked there before acting on the numbers.
