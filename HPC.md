# Running GHOST sweeps on HPC

This describes how the 2-D RCS solver and the SLURM drivers are set up for
throughput on nodes with 64–96 cores and 375–750 GB, across 0–50 nodes, and
what to turn when a sweep is not filling the machine.

The short version: **units are costed and packed by cost, claimed atomically at
run time, and admitted against the node's memory allocation rather than its
core count.** Array tasks are interchangeable, so a task that never starts, is
cancelled, or is preempted cannot strand work.

---

## 1. Which driver to run

| Path | Entry point | Output |
|---|---|---|
| Coupon library, numbered pipeline | `1b_solve_2d_hpc/run_monostatic_hpc.py` | straight into `results/{FRD,OPN}/` |
| Manifest-tracked 2-D sweep | `Backend/run_hpc_monostatic.py` | `<OUTPUT_DIR>/run_*/results/` |
| Body of revolution | `Backend/run_hpc_bor_monostatic.py` | `<OUTPUT_DIR>/run_*/results/` |
| One machine, no SLURM | `1a_solve_2d_local/run_monostatic_local.py` | `results/{FRD,OPN}/` |

All three HPC drivers share `Backend/hpc_scheduler.py`, so the tuning knobs
below mean the same thing in each.

Edit the CONFIG block at the top of a driver and run it with no arguments to
submit. `hpc_common.configure_driver` still works the same way: it rewrites
those top-level constants in a copy of the driver, and submitting that copy is
what carries the settings to the compute nodes.

---

## 2. Sizing a run

### Nodes and array tasks

```python
N_NODES = 25        # array size per submission
N_JOBS  = 2         # separate submissions (e.g. two partitions)
ARRAY_THROTTLE = None   # or an int to cap concurrent tasks
```

`N_NODES x N_JOBS` is just how much parallelism you are asking for. Because
tasks pull work from a shared claim directory rather than owning a fixed
slice, you can:

- submit a second job array later and have it join the run in progress;
- cancel part of an array without stranding its units;
- let SLURM preempt and requeue tasks (the scripts set `--requeue`) and lose
  only the units that were mid-flight.

There is no benefit to matching `N_NODES` to the unit count. More tasks than
units simply means some exit immediately.

### Cores and memory per node

```python
CORES_PER_NODE = None     # None -> --exclusive, whole node
MEM_PER_NODE   = "0"      # "0" -> all node memory (SLURM idiom)
MAX_WORKERS_PER_NODE = None   # ceiling on concurrent solves; memory usually binds first
MEMORY_HEADROOM = 0.85    # fraction of the allocation the scheduler may reserve
MEMORY_SAFETY   = 1.35    # multiplier on the solver's own peak estimate
```

Leave `MEM_PER_NODE = "0"` with `--exclusive`. Omitting the directive lets the
cluster default (often `DefMemPerCPU` ≈ 3.5 GB × CPUs) apply, which on a 750 GB
node is a fraction of what is there and will OOM-kill workers.

**Do not set `MAX_WORKERS_PER_NODE` to throttle memory.** The scheduler already
sizes concurrency from each unit's predicted peak: it runs many cheap units at
once and narrows to a few when the expensive ones come up. A fixed cap is the
wrong control because unit footprints in a frequency sweep differ by two orders
of magnitude.

### Threads

```python
BLAS_THREADS_PER_WORKER = 1
ASSEMBLY_THREADS        = "auto"
```

With at least one unit per core — the normal case for a many-geometry sweep —
leave both alone. One process per unit with single-threaded BLAS is the right
shape, and `"auto"` resolves to 1.

`"auto"` gives each pool worker `cores // pool_size` assembly threads, so
threads x processes never exceeds the allocation. It resolves to 1 whenever the
run has at least one unit per core, and only opens up when the run itself is
small — a couple of geometries at one frequency. It is keyed on the pool size
rather than on a task's planned share on purpose: a task that runs dry falls
through to stealing and can end up with a full pool, so sizing threads for the
planned share would oversubscribe the node exactly when it got busiest.

Scaling is real but sub-linear — only the tiled far-field pass is threaded, and
the scatter into the global matrices is serialized behind a lock. Measured on a
contended 4-core test host (1299 elements, S + K):

| Threads | Speedup |
|---:|---:|
| 1 | 1.00× |
| 2 | 1.78× |
| 3 | 2.09× |

Beyond that the host was oversubscribed. Treat a few threads per solve as a way
to use cores that would otherwise idle, not as a substitute for running more
units at once — processes are always the better parallelism when you have the
units to fill them.

---

## 3. How the work is scheduled

### Cost-aware planning

Assembly cost grows as the square of the boundary-node count, and node count
grows linearly with frequency, so across a 2–18 GHz sweep the most expensive
unit is roughly **80×** the cheapest.

At submit time each unit is meshed with the solver's own rule (once per
`(geometry, frequency)` — polarization does not change the discretization),
costed, and dealt out longest-processing-time-first. The plan is written to
`schedule.json` beside the manifest. It is deliberately *not* part of the
manifest: it says where work should run, not what is solved, and the manifest
fingerprint that per-unit attestations bind to must cover only the latter.

Submission prints the predicted balance:

```
Plan balance  : 1.00x the best any schedule could do (1.00 = optimal)
```

That is the plan's makespan against `max(total / slots, dearest unit)` — not
against the mean load. The mean is the wrong yardstick when one unit costs more
than an even share, or when you ask for more nodes than you have units: a
perfect plan still shows a large max/mean ratio there, and reading that as
imbalance would send you tuning a scheduler that is already optimal. If more
slots than units were requested, the report says how many will exit immediately.

For comparison, index round-robin on a four-frequency sweep across four slots
lands at 2.13× optimal — roughly twice the node-hours for the same sweep.

### Work stealing

Each task works its own planned share first (dearest first, which is what keeps
the makespan down), then falls through to everyone else's as a steal pool.
Claims are `O_CREAT | O_EXCL` files in `<run_dir>/claims/`, which is atomic on
any filesystem a run directory lives on, so no coordinating process is needed.

A claim carries a heartbeat. If a task dies, its claims go quiet and become
stealable after `CLAIM_STALE_SECONDS` (default 3600 for 2-D, 7200 for BoR — set
it comfortably above your longest single unit). The result file remains the real
idempotency guard, so the worst case for a wrongly stolen unit is that it is
solved twice and written once.

To force a completely fresh sweep, delete `claims/`.

### Restarts

A unit whose result already exists is re-dispatched for its attestation check
rather than skipped on filename alone, but only by the task that owns it in the
plan — so across a restart every output is verified exactly once, not once per
node. A mismatch fails loudly instead of silently reusing a file produced from
different inputs or a different solver build.

---

## 4. Skipping mesh certification for quick results

`solve_monostatic_rcs_2d_certified` — what the drivers call — solves every unit
**twice**: once on the requested mesh and once refined by the policy's
`fine_factor` (default **1.5**), publishing the fine result only if the two
agree. Cost scales with the square of the node count, so that second solve is
most of the wall clock and all of the peak memory.

Set `MESH_CERTIFICATION = False` in either 2-D driver to solve the base mesh
only:

```python
MESH_CERTIFICATION = False   # survey mode: screening, not production
```

Measured on `body.geo`, solve time only:

| | Certified | Survey | |
|---|---:|---:|---|
| 6 GHz, 135 panels | 1.02 s | 0.34 s | 3.00× |
| 18 GHz, 328 panels | 3.86 s | 1.34 s | 2.88× |
| 30 GHz, 520 panels | 7.52 s | 2.54 s | 2.96× |

End to end through the driver, small units see less (1.8× on a two-unit run)
because per-unit overhead — import, preflight, provenance — stops being
negligible. Big units get close to the 3×.

Memory follows the same shape: the reservation is built on the **fine** mesh,
so survey mode roughly halves it and about twice as many units fit per node.
At N = 5000 that is 15.4 GB reserved versus 7.4 GB.

**Survey output is not production data.** The algebraic quality gate still runs
— a badly conditioned or non-converged solve still fails closed — but nothing
establishes that the discretization is fine enough, which is the error that
biases an RCS number quietly rather than announcing itself. On the shipped
geometry the base mesh happened to land within 0.04 dB of the certified one;
that is a property of that geometry at that frequency, not a guarantee.

Three things keep survey results out of the production path:

- the grim carries **no `mesh_convergence` block**, and `feature_sum` requires
  `mesh_convergence.passed is True` with `published_mesh == "fine"` before a
  field may enter a body or a delta — so the pipeline rejects it on structure,
  not on a label it has to trust;
- metadata says so explicitly (`survey_mode: true`,
  `mesh_convergence_certified: false`, `published_mesh: "base"`) and the
  solve's warnings carry the reason, so an artifact found later identifies
  itself;
- in the coupon pipeline the flag is part of each unit's fingerprint, so
  switching back to `MESH_CERTIFICATION = True` will **not** accept the survey
  files as completed units — they get re-solved properly.

There is no way to weaken certification without switching it off: the policy
enforces `fine_factor > 1.0`, so you cannot quietly shrink the refinement and
keep the certificate. If you want the low-level uncertified entry directly, it
is `rcs_solver.solve_monostatic_rcs_2d_survey` (or the raw
`solve_monostatic_rcs_2d`).

The BoR driver has its own certification inside `bor_dispatch` and no
equivalent switch.

---

## 5. "Solver source ... differ from the HPC run manifest"

Every unit re-checks that the solver source still matches what the run
recorded at submit time — a hash over every top-level `.py`/`.so`/`.c`/... in
`Backend/`, plus the driver being executed. If it fires, the run and the code
on disk have diverged.

The message now names the files:

```
Solver source/native artifacts differ from the HPC run manifest ...
(changed: Backend/solver_utils.py; added: Backend/my_patch.py;
 removed: Backend/occluder.py). Either restore the recorded source or submit
a new run with the code you actually want to execute.
```

For a run submitted before per-file inventories existed, or to check a compute
node directly:

```bash
python tests/diagnose_provenance.py <run_dir> [--backend /path/to/Backend]
```

Usual causes, in order of likelihood:

1. **Code edited after submitting.** The manifest froze the old source; the
   worker sees the new one. Resubmit — the existing run's finished results
   stay valid, because they were produced by the source that run recorded.
2. **A partially updated tree.** A file copied to the cluster and one missed,
   so the file *set* differs. `removed:` and `added:` entries point straight
   at it.
3. **Login node and compute node see different trees** — different mount,
   different `PYTHONPATH`, a stale copy under a different prefix. The
   diagnostic prints the Backend directory it actually checked, which is
   usually enough to spot this.

Note that `.so` artifacts count: rebuilding `fmm_near.so` on a different host
changes the fingerprint even with identical sources.

Restoring the old tree and resubmitting are both valid; the check exists so a
run cannot silently mix fields from two different solver builds.

---

## 6. Per-unit overhead

Provenance is verified before and after every unit, which is correct and stays.
What changed is the cost of doing it:

- **Source hashing** goes through `hpc_scheduler.FingerprintCache`, keyed on
  each file's identity, size, and mtime, and flushed every 300 s so full
  re-reads keep happening inside long-lived workers. Previously every unit
  re-read the whole backend tree three times, from every worker, over a shared
  filesystem.
- **The manifest is parsed once per node**, not once per unit. It carries every
  unit's record, so the old pattern made a node's JSON cost quadratic in the
  size of the sweep.
- **Geometries are parsed once per node** and inherited by forked pool workers.
- **The solver is imported in the parent** before the pool forks. It used to be
  imported inside the worker function with `maxtasksperchild=1`, so every unit
  paid a fresh import of numpy, SciPy, and an 8 000-line module. Worker
  recycling is kept (`TASKS_PER_CHILD`, default 4) — a respawn now costs a fork.

---

## 7. Solver performance

Everything below is bit-comparable with the previous solver to floating-point
reassociation; `tests/test_assembly_equivalence.py` and
`tests/test_solver_equivalence.py` check that against a pristine copy.

### Operator assembly

Element pairs are processed in cache-sized tiles rather than as whole
`N_elem x N_elem` arrays, the kernel is evaluated once per *unordered* pair
(G is symmetric, and the two double-layer orientations share one Hankel
evaluation), and the real-wavenumber path uses `J_n`/`Y_n` directly with no
complex temporaries.

Measured on `body.geo` (single-layer + adjoint double-layer, one core):

| Elements | Before | After | Speedup | Extra peak RAM before → after |
|---:|---:|---:|---:|---|
| 338 | 2.7 s | 2.0 s | 1.4× | 17 MB → ~0 |
| 670 | 9.7 s | 5.7 s | 1.7× | 62 MB → ~0 |
| 1299 | 37.1 s | 17.3 s | 2.1× | 232 MB → ~0 |

The speedup grows with mesh size, and the memory column matters as much as the
time one: the old formulation held eight complex `N^2` accumulators plus about
six `N^2` temporaries live at once — 3.2 GB + 1.2 GB at 5 000 elements — which
capped how many solves fit on a node far more tightly than the matrices
themselves do.

### Geometries with materials

A coated or multi-region body assembles one operator per (region, interface
side). Both sides of a region share its wavenumber and differ only in which
elements are active — and masks are applied *after* the quadrature, so
assembling them separately repeated the whole element-pair sweep. Two changes:

- **One traversal per wavenumber.** The multi-region solver now prefetches
  every (wavenumber, mask) the matrix build will ask for, groups them, and
  assembles each group once.
- **Compacted source axis.** A mask selecting a minority of the mesh — normal
  for a thin coating — no longer pays a full-width sweep that is then masked
  away. Below 50% active the source axis compacts, making the work
  proportional to what is actually wanted. Above that the transposed-pair
  shortcut wins instead, so the threshold is where the two break even.

Measured on a 24 in PEC trapezoid with a 0.1 in lossy dielectric layer, versus
the pre-optimization solver:

| Elements | Before | After | Speedup |
|---:|---:|---:|---:|
| 234 | 6.8 s | 2.7 s | 2.5× |
| 691 | 51.2 s | 14.8 s | 3.5× |
| 1147 | 153.6 s | 36.8 s | 4.2× |

Compaction is a pure optimization: `tests/test_assembly_equivalence.py` checks
the compacted result against the full-width sweep, and grouped masks against
one-at-a-time assembly, for both real and lossy wavenumbers.

### Hypersingular operator (TE sheet / dielectric paths)

This was an O(N²) interpreted loop calling a per-pair quadrature routine. The
pairs that the recursion resolves with a plain tensor-Gauss box — all but O(N)
of them — are now batched over the same tiles at the same order:

| Elements | Before (measured/extrapolated) | After | Speedup |
|---:|---:|---:|---:|
| 88 | 30 s | 0.7 s | 42× |
| 228 | 199 s | 2.8 s | 72× |
| 430 | 709 s | 7.3 s | 97× |
| 820 | 2579 s | 18.8 s | 137× |

### Optional: far-pair quadrature order

Roughly half of a large assembly is Hankel evaluations, and their count is
exactly `N_elem^2 x Q^2`. The default `Q = 8` is inherited from the near-field
rule; across a far pair the integrand is smooth and it is over-resolved.

```bash
export GHOST_FAR_QUAD_ORDER=5     # or rcs_solver.set_far_quadrature_order(5)
```

**This is off by default because it changes computed values** — a different
quadrature rule, not a faster evaluation of the same one. Measured on
`body.geo` at 40 GHz (670 elements) with `tests/measure_far_quadrature.py`:

| Order | Speedup | Max RCS shift |
|---:|---:|---:|
| 8 (default) | 1.00× | — |
| 6 | 1.35× | 2.7e-13 dB |
| 5 | 1.69× | 2.9e-11 dB |
| 4 | 1.94× | 4.1e-09 dB |
| 3 | 2.29× | 6.3e-07 dB |

Measured again on a coated multi-region body (24 in PEC trapezoid with a
0.1 in lossy dielectric layer, 3 GHz), where the payoff is larger because that
path assembles three operator sets and a complex wavenumber makes each Hankel
evaluation ~10x dearer than a real one:

| Configuration | Speedup | Max RCS shift |
|---|---:|---:|
| certified, order 8 (default) | 1.00× | — |
| certified, order 4 | 3.09× | 0.000 dB |
| survey, order 8 | 3.11× | 0.017 dB |
| survey, order 4 | 8.49× | 0.017 dB |

Those shifts are far below anything physically meaningful, but they were
measured on specific geometries at specific frequencies — run the script on
your own before relying on it.

Why the order matters so much: linear Galerkin with an 8x8 rule spends 64
kernel evaluations per element pair, and the assembly is ~88% of a solve
(every quality gate, the condition estimate, the preflight, and the
mesh-convergence comparison together come to under 1.5%). Order 4 is 16
evaluations per pair — a quarter of the dominant cost. Near-field and singular quadrature are untouched. A solve
that used the override records it in its warnings, so the fact travels with the
published `.grim`.

### Other environment knobs

| Variable | Default | Effect |
|---|---|---|
| `GHOST_ASSEMBLY_THREADS` | 1 | Threads for tiled assembly (drivers set this per-solve via `ASSEMBLY_THREADS`) |
| `GHOST_ASSEMBLY_TILE` | auto | Element-tile edge; auto targets a ~24 MB working set |
| `GHOST_FAR_QUAD_ORDER` | 0 (off) | Far-pair quadrature order override |

Shrink `GHOST_ASSEMBLY_TILE` if you are packing many solves per node and the
shared L3 is thrashing; grow it for a few large solves with threads.

---

## 8. Tests

```bash
python tests/test_hpc_scheduling.py                       # scheduler + a real 2-task sweep
python tests/test_solver_equivalence.py  <pristine rcs_solver.py>
python tests/test_assembly_equivalence.py <pristine rcs_solver.py>
python tests/benchmark_assembly.py      [pristine rcs_solver.py]
python tests/measure_far_quadrature.py  [geometry.geo ...]
python tests/diagnose_provenance.py     <run_dir>       # why a source check failed
```

The two equivalence tests need a copy of the solver from before these changes
to compare against; keep one outside the tree.

---

## 9. Rules of thumb

- **Many geometries or frequencies:** leave everything at defaults, set
  `N_NODES` to what you can get, and let the planner and stealing do the work.
- **A few large geometries:** set `ASSEMBLY_THREADS = "auto"` (the default) so
  idle cores go into the assembly instead of sitting unused.
- **Sweeps that OOM-killed before:** stop capping `MAX_WORKERS_PER_NODE` and
  make sure `MEM_PER_NODE` is `"0"` or an explicit large value. If a single unit
  genuinely does not fit, the solver's own memory gate will say so with a
  number rather than the node dying.
- **Screening a trade study:** `MESH_CERTIFICATION = False` for ~3x throughput,
  then re-run the configurations you care about with it back on.
- **A run that got interrupted:** resubmit the same driver copy against the same
  run directory. Finished units are verified and skipped; unclaimed ones are
  picked up.
