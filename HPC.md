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

| Solver | SLURM entry point | Same sweep, one machine | Output |
|---|---|---|---|
| 2-D arbitrary geometry | `Backend/run_hpc_monostatic.py` | `Backend/run_local_monostatic.py` | `<OUTPUT_DIR>/run_*/results/{FRD,OPN}/` |
| Body of revolution | `Backend/run_hpc_bor_monostatic.py` | `Backend/run_local_bor.py` | one monostatic `results/<geometry>.grim`; hidden restart units in `.solver_units/` |

Every driver in the table shares `Backend/hpc_scheduler.py`, so the tuning
knobs below mean the same thing in each.

Edit the CONFIG block at the top of a driver and run it with no arguments to
submit. `hpc_common.configure_driver` still works the same way: it rewrites
those top-level constants in a copy of the driver, and submitting that copy is
what carries the settings to the compute nodes.

### Running without SLURM

The `run_local` drivers in the right-hand column are the same code paths minus
the array-task machinery: no manifest is copied to a compute node, no claim
directory, no `sbatch`. What they keep is everything that decides how fast a
sweep finishes and what it leaves behind --

- units costed from the mesh the solver will actually build, run dearest-first;
- concurrent solves admitted against a memory budget (section 2), not against
  the core count -- which matters more on a workstation than on a compute node,
  since there is far less headroom to absorb a wrong guess;
- assembly threads sized from the concurrency memory will actually permit
  (section 2, *Threads*);
- 2-D unit GRIMs in `results/`; one complete BoR monostatic GRIM per geometry
  in `results/`, with internal restart units under `.solver_units/` (section 3);
- the shared angular grid stored once, not once per unit (section 3);
- resumption by verifying the binding carried inside each existing result.

Local defaults differ in two places, both because the machine is shared with
whatever else you are doing: `MEMORY_HEADROOM` is `0.75` rather than `0.85`,
and `WORKERS = None` means *all but one core* rather than *every core*.

Run one with no arguments:

```bash
python Backend/run_local_monostatic.py
```

### Naming polarizations

The 2-D drivers take either spelling of the same two physical channels:

| Radar | 2-D | Why |
|---|---|---|
| `HH` | `TM` | these geometries are elevation cuts, so the out-of-plane z axis is *horizontal*; E along z is HH |
| `VV` | `TE` | TE's in-plane E carries the vertical component |

`V`, `H`, `VERTICAL`, and `HORIZONTAL` are accepted too. Which spelling reaches
the output file names depends on the driver, and it is fixed per driver so that
changing what you type in CONFIG never forks a sweep into two sets of files:

- `run_hpc_monostatic.py` / `run_local_monostatic.py` name files with the label
  you wrote, and default to `["VV", "HH"]`.
- The BoR drivers always co-solve VV and HH and publish VV, HH, and the derived
  radar-frame VH channel in the one monostatic file. Polarization is therefore
  not a user sweep control for BoR.

Listing one channel under two names (`["VV", "TE"]`) is rejected rather than
silently solving the same physics twice and publishing it under two names,
which a downstream concatenation would then treat as independent channels.

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

### Memory-heavy geometries

Concurrency follows each unit's predicted peak, so a heavy geometry
automatically runs fewer at a time. Planning builds the same interface-aware
base and certification meshes as the solver, selects the formulation for each
polarization, and reserves from its actual dense system DOFs. A PEC/IBC or
sheet solve is therefore budgeted as an N-unknown system, a single dielectric
as 2N, and a coated/multi-region body by its exact interface-side unknowns.
The longest-processing-time assignment uses those same formulation DOFs and
operator counts, so coated bodies are also charged for their larger LU and
additional assemblies instead of being balanced as ordinary N-unknown PEC
jobs.
Measured against a 100 GB budget:

| Per-unit peak | Concurrent solves | Reserved |
|---:|---:|---:|
| 0.7 GB | 20 | 14 GB |
| 12 GB | 8 | 96 GB |
| 45 GB | 2 | 90 GB |

Threads follow the same number rather than the pool width, so those two 45 GB
solves get `96/2 = 48` assembly threads each instead of a twelfth of the node.
Submission prints the narrowing when it happens:

```
Memory-limited : at most 2 concurrent solve(s); heaviest planned unit 45.3 GB
```

A unit larger than the whole budget still runs, alone, rather than deadlocking
the node -- verified, along with the table above, in
`tests/test_hpc_scheduling.py`.

Beyond that the solver has its own last-resort gate and refuses to allocate
before it starts, so an impossible unit fails with a number instead of being
OOM-killed. That ceiling is derived from what the process can actually use --
a SLURM allocation, then a cgroup limit, then `/proc/meminfo` -- times 0.9, and
floored at 32 GB so nothing that ran before stops running. `GHOST_MAX_SOLVE_GB`
overrides it outright.

It used to be a flat 32 GB, which was wrong in both directions: it refused a
feasible 40 GB solve on a 750 GB node and permitted a 32 GB one on a 16 GB
laptop.

**Do not raise the 32 GB floor to run something large.** It is a *minimum* on
the ceiling, not a cap -- `limit = max(floor, 0.9 x detected)` -- so raising it
does nothing on a big node and removes the guard on a small one. What the
ceiling resolves to:

| Situation | Detected | Ceiling | 300 GB solve? |
|---|---:|---:|---|
| `--mem=750G` | 750 GB | 675 GB | yes |
| `--mem-per-cpu=4G`, 96 cpus | 384 GB | 346 GB | yes |
| `--mem=0` (driver default) | *cgroup or /proc/meminfo* | depends | **check it** |
| detection returns nothing | 0 | 32 GB | no |

`MEM_PER_NODE = "0"` means "all node memory", and SLURM then reports
`SLURM_MEM_PER_NODE=0` -- so the ceiling falls back to the cgroup, then to
/proc/meminfo. On an exclusive node that is normally the full node, but it
depends on how the cluster configures cgroups, and if neither is meaningful the
ceiling collapses to 32 GB and a large solve is refused on a large machine.

For anything big, be explicit -- either

```python
MAX_SOLVE_GB = 400        # exported to the job as GHOST_MAX_SOLVE_GB
```

or give `MEM_PER_NODE` a real size (`"750G"`) so detection has a number it can
trust. Submission prints what it resolved to:

```
Solve ceiling : 400.0 GB per solve (MAX_SOLVE_GB)
```

To check what a compute node reports before committing a long run:

```bash
srun --mem=0 --exclusive python -c \
  "import sys; sys.path.insert(0,'Backend'); import rcs_solver as r; \
   print(r._detect_available_gb(), '->', r._solve_memory_limit_gb())"
```

Worth sizing the ambition too. There is no longer one defensible bytes-per-node
constant: retained dense storage is roughly 80 N-squared bytes for the current
N-unknown sheet/Robin paths and roughly 208 N-squared bytes for the 2N-unknown
single-dielectric path, before angle RHS storage, allocator safety, and the
scheduler's fixed process allowance. Multi-region storage depends on the exact
interface-side DOF count. Use the submission plan's per-unit `peak_gb` rather
than converting panel count with a universal constant. Assembly still scales
quadratically and is only partly threaded, so the largest units need generous
walltime even when they fit in RAM.

**Do not set `MAX_WORKERS_PER_NODE` to throttle memory.** The scheduler already
sizes concurrency from each unit's predicted peak: it runs many cheap units at
once and narrows to a few when the expensive ones come up. If the next large
unit does not fit, a smaller unit may backfill the remaining memory; the large
unit keeps its position and is reconsidered as soon as running work completes.
A fixed cap is the wrong control because unit footprints in a frequency sweep
differ by two orders of magnitude.

### Threads

```python
BLAS_THREADS_PER_WORKER = 1
ASSEMBLY_THREADS        = "auto"
```

Leave both alone for normal sweeps. One process per admitted unit with
single-threaded BLAS is the right shape. `"auto"` chooses assembly threads for
each unit from that unit's predicted memory concurrency. On a 96-core node, a
70 GB unit that fits four at a time gets 24 assembly threads, while a 10 GB unit
that fits 31 at a time gets 3. The dispatcher reserves those CPU counts along
with memory, including for mixed heavy/light backfill, so dynamic admission
cannot oversubscribe the node.

This is deliberately per unit rather than a pool-wide startup setting. A
frequency sweep can move from four large solves to dozens of small solves; a
fixed 24-thread setting would otherwise multiply into hundreds of runnable
threads when the small solves began.

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

Each task works its own planned share to completion first (dearest first, which
is what keeps the makespan down), and only then falls through to everyone
else's as a steal pool. The two phases are separate on purpose: a task's
concurrency is sized from its **own share**, not from the whole sweep, so a
fast starter cannot reach past its share and claim work planned for its peers.

Getting this wrong is easy and looks like success. If concurrency is sized from
the total unit count, the dispatcher fills its pool from the head of the
candidate list -- which begins with the task's own units and continues into
everyone else's -- so on a 96-core node with a 40-unit sweep the first task to
start claims the entire run, and the other nine report

```
planned for this slot: 1 ... wrote=0 skipped=0 failed=0,
left to other tasks=40.  0.2 s elapsed
```

Every unit still gets solved, but on one node instead of ten.
`tests/test_hpc_scheduling.py::test_no_task_starvation` covers it.

Claims themselves are `O_CREAT | O_EXCL` files in `<run_dir>/claims/`, which is
atomic on any filesystem a run directory lives on, so no coordinating process
is needed.

A claim carries a heartbeat. If a task dies, its claims go quiet and become
stealable after `CLAIM_STALE_SECONDS` (default 3600 for 2-D, 7200 for BoR — set
it comfortably above your longest single unit). The result file remains the real
idempotency guard, so the worst case for a wrongly stolen unit is that it is
solved twice and written once.

To force a completely fresh sweep, delete `claims/`.

### What lands in results/

For 2-D runs, those unit files are placed under `results/FRD/` and
`results/OPN/`, preserving the geometry role expected by the concatenate and
subtract tools. `1c_build_deltas/concat_pols.py` automatically selects the
newest complete `rcs_runs/run_*/results/` folder when no input path is given.

For BoR runs, `results/<geometry>.grim` is the sole user-facing monostatic
dataset. It contains the requested azimuth/elevation/frequency VV/HH/VH arrays
and embeds the exact BoR aspect field plus outer profile needed for coherent
feature placement. Per-frequency co-solved units live in `.solver_units/` only
as verified restart state. Publication performs no additional field solve and
can be rerun explicitly with:

```bash
python Backend/run_hpc_bor_monostatic.py --publish /path/to/run_dir
```

Each result is bound to its run -- source build, runtime, geometry inputs,
solve spec, angular grid -- by fields carried **inside** the artifact, not by a
`<name>.provenance.json` beside it. A sweep of thousands of units would
otherwise put thousands of extra tiny files in the results directory.

Integrity does not depend on the dropped sidecar hash. A `.grim` is an npz, npz
is a zip, and numpy validates a CRC-32 per member on read, so a corrupted
result raises `BadZipFile` on open rather than verifying and returning wrong
numbers. What the attestation is for -- catching a result produced by a
different source build, runtime, or input -- is fully covered by the embedded
fields, and `hpc_common.require_hpc_output_attestations` still checks the exact
expected file set.

### What the manifest holds

The run header and one compact record per unit: geometry, polarization,
frequency, input hash. Anything identical across units -- the angular grid
above all -- is recorded once at manifest level.

That matters at scale, because the grid was the bulk of the file:

| Sweep | Old manifest | Now |
|---|---:|---:|
| 10 geom x 91 freq x 2 pol, 361 azimuths | 7.6 MB | 0.67 MB |
| 50 geom x 91 freq x 2 pol, 361 azimuths | ~52 MB | ~4 MB |

93% of the old file was the same azimuth list repeated once per unit.

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

Set `MESH_CERTIFICATION = False` in any driver to solve the base mesh
only:

```python
MESH_CERTIFICATION = False   # base mesh only
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
so survey mode usually cuts it substantially and more units fit per node. The
exact ratio is formulation-dependent because the planner now counts the true
system DOFs and retained operators rather than applying one 2N model to every
geometry.

The algebraic quality gate still runs
— a badly conditioned or non-converged solve still fails closed — but nothing
establishes that the discretization is fine enough, which is the error that
biases an RCS number quietly rather than announcing itself. On the shipped
geometry the base mesh happened to land within 0.04 dB of the certified one;
that is a property of that geometry at that frequency, not a guarantee.

Certification is a user-selected accuracy check, not a downstream permission
system. Both base-mesh and certified GRIMs may be viewed, joined, subtracted,
collected into a body, and used in feature summation. Raw solver metadata still
records which choice produced the field so the two are not ambiguous later.
The choice is also part of the unit fingerprint, so changing it starts the
appropriate solve rather than silently reusing a result from the other mode.

There is no way to weaken certification without switching it off: the policy
enforces `fine_factor > 1.0`, so you cannot quietly shrink the refinement and
keep the certificate. If you want the low-level uncertified entry directly, it
is `rcs_solver.solve_monostatic_rcs_2d_survey` (or the raw
`solve_monostatic_rcs_2d`).

The GUI and both BoR sweep drivers use the same default-on switch. Set
`MESH_CERTIFICATION = False` in `run_local_bor.py` or
`run_hpc_bor_monostatic.py` to request a base-mesh solve. BoR outputs carry
the same factual provenance as 2-D outputs and remain usable downstream.

BoR memory admission is also per solve, not the old fixed
`STREAM_BUDGET_GB` reservation. Before dispatch, the driver previews the
frequency-specific material wavelength, certified or survey mesh, modal cap,
table/streaming choice, precision, dense system workspace, and internal mode
workers. Small frequencies can therefore backfill around a large solve while
the large solve retains its full reservation.

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

### Graded far quadrature (on by default)

The far pass used a fixed order 8 for every well-separated pair. Measuring the
order actually needed to hold a Galerkin block to 1e-12 relative — against an
order-24 reference, worst case over five pair orientations — gives:

| kL \ r | 3 | 5 | 10 | 25 |
|---:|---:|---:|---:|---:|
| 0.15 | 6 | 5 | 5 | 4 |
| 0.50 | 7 | 6 | 5 | 5 |
| 1.50 | 7 | 6 | 6 | 6 |
| 3.00 | 8 | 8 | 8 | 8 |

where `r` is centre separation in element lengths and `kL` is element length in
radians. Two things fall out:

- **Separation barely matters** once a pair is far at all. What sets the order
  is how many wavelengths the element spans, because that is what the
  integrand oscillates over.
- **At kL ≥ 4 the fixed order 8 is *under*-resolved** — that mesh needs 9. A
  λ/20 mesh is kL ≈ 0.31, so this only bites on deliberately coarse meshes.

Each tile now picks its order from its own worst far pair. Measured 1.24–1.43×
depending on wavenumber and mask, with results matching the ungraded sweep to
~5e-16 — it is a free speedup, not an accuracy trade, because it only drops
orders the calibration says are unnecessary.

Grading never raises the order above what the caller configured. Raising it
where the table says it is needed would improve coarse-mesh accuracy but change
published values, which is not something to do silently. `set_far_quadrature_grading(False)`
forces the flat order; `tests/test_assembly_equivalence.py` checks the two agree.

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

The condition diagnostic no longer performs a second dense SVD after solving.
It equilibrates the assembled system and estimates its 1-norm inverse through
the LU factors already needed for the field solve, preserving the fail-closed
quality gate without another cubic factorization.

### Other environment knobs

| Variable | Default | Effect |
|---|---|---|
| `GHOST_ASSEMBLY_THREADS` | 1 | Threads for tiled assembly (drivers set this per-solve via `ASSEMBLY_THREADS`) |
| `GHOST_ASSEMBLY_TILE` | auto | Element-tile edge; auto targets a ~24 MB working set |
| `GHOST_FAR_QUAD_ORDER` | 0 (off) | Far-pair quadrature order override |

Shrink `GHOST_ASSEMBLY_TILE` if you are packing many solves per node and the
shared L3 is thrashing; grow it for a few large solves with threads.

---

## 8. Source encoding

Every Python file in this repo is **pure ASCII**, and
`tests/test_source_is_ascii.py` enforces it.

The reason is a failure mode that only shows up after a file has been copied
somewhere. UTF-8 source that passes through a Windows editor, an FTP client in
text mode, or anything else that re-encodes on save comes back in a local
codepage. An em dash in a comment becomes a lone `0x97` byte, and Python 3
refuses the file outright:

```
SyntaxError: (unicode error) 'utf-8' codec can't decode byte 0x97
in position 0: invalid start byte
```

The module is then unimportable on the machine it was copied to, and the error
points at a decorative character rather than anything meaningful. If you hit
this on a file from elsewhere, restore it with `git checkout` or re-copy in
binary mode (`scp`, `rsync`, `git`) rather than through an editor.

Data files are exempt -- `.geo` and `.csv` inputs are read with an explicit
encoding and are yours, not ours.

---

## 9. Tests

```bash
python tests/test_hpc_scheduling.py                       # scheduler + a real 2-task sweep
python tests/test_local_drivers.py                        # the run_local drivers, end to end
python tests/test_rcs_physics_regression.py               # 2-D analytic/reciprocity gates
python tests/test_bor_physics_regression.py               # BoR Mie/streaming/workflow gates
python tests/test_solver_equivalence.py  <pristine rcs_solver.py>
python tests/test_assembly_equivalence.py <pristine rcs_solver.py>
python tests/benchmark_assembly.py      [pristine rcs_solver.py]
python tests/measure_far_quadrature.py  [geometry.geo ...]
python tests/diagnose_provenance.py     <run_dir>       # why a source check failed
python tests/test_source_is_ascii.py                    # source stays copy-safe
```

The two equivalence tests need a copy of the solver from before these changes
to compare against; keep one outside the tree.

---

## 10. Rules of thumb

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
