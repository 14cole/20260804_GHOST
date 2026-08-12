#!/usr/bin/env python3
"""Solve every geometries/FRD and geometries/OPN coupon on this machine.

Outputs are written directly to results/FRD and results/OPN -- one .grim per
solve unit and nothing else. Each file carries its own input/solver fingerprint
inside the artifact, and an existing result is reused only when that
fingerprint matches the current request.

This is run_monostatic_hpc.py without SLURM, and it schedules the same way:
units are costed from the mesh the solver will actually build and run
dearest-first, and concurrent solves are admitted against a memory budget
rather than filling every core regardless of unit size. Either input folder may
be empty.
"""

from multiprocessing import Pool
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Backend"))

import hpc_scheduler  # noqa: E402
from step1_monostatic import (  # noqa: E402
    _pin_blas,
    discover_jobs,
    prepare_jobs,
    prewarm_snapshots,
    solve_job_catching,
    validate_config,
)

# -- USER SETTINGS ----------------------------------------------------------
FREQUENCIES_GHZ = [3.0, 6.0]
ANGLES_DEG = np.arange(0.0, 180.1, 5.0)
POLARIZATIONS = ["TM", "TE"]
GEOMETRY_UNITS = "meters"
SOLVER_METHOD = "auto"
MAX_PANELS = 50_000
FORCE = False

# Mesh-convergence certification. True solves every unit twice -- the requested
# mesh and one refined by the policy's fine_factor -- and publishes the fine
# result only if they agree. That second solve is where most of the wall clock
# and all of the peak memory go: turning it off is about 3x faster per unit and
# roughly halves the memory, so more units fit in RAM at once too.
#
# SURVEY OUTPUT IS NOT PRODUCTION DATA. The algebraic quality gate still runs,
# but nothing establishes that the discretization is fine enough. Survey grims
# carry no mesh-convergence block, so the body/delta pipeline rejects them, and
# the flag is part of each unit's fingerprint -- a survey file will not be
# accepted as a completed unit once you switch back. Screening only.
MESH_CERTIFICATION = True

WORKERS = None                # hard ceiling on concurrent solves;
                              # None = all but one CPU core
BLAS_THREADS_PER_WORKER = 1
MEMORY_HEADROOM = 0.75        # fraction of detected RAM the scheduler may
                              # reserve for concurrent solves. Lower than the
                              # cluster default of 0.85: a workstation has a
                              # desktop and a page cache to leave room for.
MEMORY_SAFETY = 1.35          # multiplier on the solver's own peak estimate
ASSEMBLY_THREADS = "auto"     # "auto", or an integer >= 1
TASKS_PER_CHILD = 4           # pool worker lifetime, in units
# ---------------------------------------------------------------------------


def _validate_settings() -> 'None':
    if not 0.0 < float(MEMORY_HEADROOM) <= 1.0:
        raise SystemExit("MEMORY_HEADROOM must be in (0, 1].")
    if float(MEMORY_SAFETY) < 1.0:
        raise SystemExit("MEMORY_SAFETY must be >= 1.")
    if ASSEMBLY_THREADS != "auto" and int(ASSEMBLY_THREADS) < 1:
        raise SystemExit("ASSEMBLY_THREADS must be 'auto' or an integer >= 1.")
    if int(TASKS_PER_CHILD) < 1:
        raise SystemExit("TASKS_PER_CHILD must be >= 1.")
    if WORKERS is not None and int(WORKERS) < 1:
        raise SystemExit("WORKERS must be a positive integer or None.")


def _plan(jobs, n_angles):
    """Cost and size every job from the mesh the solver will actually build.

    The mesh depends on the geometry and frequency but not the polarization,
    so a coupon library with both polarizations meshes each pair once.
    """

    from solver_quality import validate_mesh_convergence_policy

    # fine_factor <= 1 tells the cost model there is only one mesh to solve.
    fine_factor = (
        float(validate_mesh_convergence_policy()["fine_factor"])
        if MESH_CERTIFICATION else 1.0
    )
    nodes_cache = {}
    costs = {}
    peaks = {}
    for job in jobs:
        key = (str(job["geometry"]), float(job["frequency_ghz"]))
        if key not in nodes_cache:
            nodes_cache[key] = hpc_scheduler.predict_2d_nodes(
                key[0], key[1], str(job["polarization"]),
                GEOMETRY_UNITS, MAX_PANELS,
            )
        nodes = int(nodes_cache[key])
        name = str(job["output"])
        costs[name] = (
            hpc_scheduler.unit_cost(nodes, n_angles, fine_factor)
            if nodes > 0 else 1.0
        )
        peaks[name] = (
            hpc_scheduler.unit_peak_gb(
                nodes, fine_factor, safety=float(MEMORY_SAFETY)
            ) if nodes > 0 else 0.0
        )
    return costs, peaks


def _resolve_assembly_threads(cores: 'int', concurrency: 'int') -> 'int':
    """Threads per solve, sized so threads x processes never exceeds the cores."""

    if ASSEMBLY_THREADS != "auto":
        return max(1, int(ASSEMBLY_THREADS))
    return max(1, int(cores) // max(1, int(concurrency)))


def _pool_initializer(blas_threads: 'int', assembly_threads: 'int') -> 'None':
    _pin_blas(blas_threads)
    import rcs_solver

    rcs_solver.set_assembly_threads(assembly_threads)


def main() -> 'None':
    _validate_settings()
    try:
        frequencies, angles, polarizations = validate_config(
            FREQUENCIES_GHZ, ANGLES_DEG, POLARIZATIONS
        )
        jobs = discover_jobs(HERE, frequencies, polarizations)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not jobs:
        raise SystemExit(
            "No .geo files found. Put coupons in geometries/FRD or "
            "geometries/OPN; either folder may otherwise be empty."
        )
    jobs = prepare_jobs(
        jobs,
        angles_deg=angles,
        geometry_units=GEOMETRY_UNITS,
        solver_method=SOLVER_METHOD,
        max_panels=MAX_PANELS,
        runner_path=__file__,
        certify=bool(MESH_CERTIFICATION),
    )

    # Cost every unit from the real mesh, then run dearest-first: a frequency
    # sweep's cost spread is large (it grows like the square of the node
    # count), and starting with the cheap end leaves the expensive tail with
    # nothing to overlap against.
    costs, peaks = _plan(jobs, len(angles))
    ordered = sorted(
        jobs, key=lambda j: (-costs.get(str(j["output"]), 1.0), str(j["output"]))
    )

    cores = hpc_scheduler.detect_cores()
    memory_gb = hpc_scheduler.detect_memory_gb()
    budget_gb = max(1.0, memory_gb * float(MEMORY_HEADROOM))
    worker_cap = max(1, cores - 1) if WORKERS is None else int(WORKERS)
    pool_size = max(1, min(cores, worker_cap, len(ordered)))
    # Threads are sized from the concurrency memory will actually permit, not
    # from the pool width: a memory-heavy geometry narrows admission, and
    # sizing threads for a width that will never be reached would leave most of
    # the cores idle for the whole run.
    heaviest = max(
        (peaks.get(str(j["output"]), 0.0) for j in ordered), default=0.0
    )
    memory_concurrency = (
        pool_size if heaviest <= 0.0
        else max(1, min(pool_size, int(budget_gb // heaviest)))
    )
    assembly_threads = _resolve_assembly_threads(cores, memory_concurrency)

    kwargs = {
        "angles_deg": angles,
        "geometry_units": GEOMETRY_UNITS,
        "solver_method": SOLVER_METHOD,
        "max_panels": MAX_PANELS,
        "force": FORCE,
    }
    print(
        f"Step 1 local: {len(ordered)} solve unit(s), {pool_size} worker(s) x "
        f"{assembly_threads} assembly thread(s) of {cores} cpu(s), "
        f"{budget_gb:.1f}/{memory_gb:.1f} GB schedulable; "
        "outputs -> results/{FRD,OPN}"
    )
    if memory_concurrency < pool_size:
        print(
            f"  memory-limited: at most {memory_concurrency} concurrent "
            f"solve(s); heaviest unit {heaviest:.1f} GB"
        )
    if not MESH_CERTIFICATION:
        print(
            "  [warn] MESH_CERTIFICATION is off: results are survey-grade and "
            "the body/delta pipeline will reject them. Screening only."
        )
    _pin_blas(BLAS_THREADS_PER_WORKER)

    # Parse each distinct geometry, and import the solver, before the pool
    # forks: workers then inherit both instead of repeating the work per unit.
    prewarm_snapshots(ordered)
    import rcs_solver  # noqa: F401
    import grim_io     # noqa: F401

    counters = {"written": 0, "skipped": 0, "failed": 0}
    started = time.time()
    total = len(ordered)

    def _prepare(job):
        return (
            str(job["output"]), peaks.get(str(job["output"]), 0.0),
            (solve_job_catching, ((job, kwargs),)),
        )

    def _done():
        return counters["written"] + counters["skipped"] + counters["failed"]

    def _on_result(key, payload):
        kind, first, second, job = payload
        if kind == "ok":
            counters["skipped" if first == "skipped" else "written"] += 1
            print(f"[{_done():4d}/{total}] {first:7s} "
                  f"{job['role']}/{Path(second).name}", flush=True)
        else:
            counters["failed"] += 1
            print(f"[{_done():4d}/{total}] FAILED  "
                  f"{job['role']}/{Path(job['output']).name}\n{first}",
                  flush=True)

    def _on_error(key, exc):
        counters["failed"] += 1
        print(f"[{_done():4d}/{total}] FAILED (dispatch) {Path(key).name}: "
              f"{exc!r}", flush=True)

    with Pool(
        processes=pool_size,
        initializer=_pool_initializer,
        initargs=(int(BLAS_THREADS_PER_WORKER), int(assembly_threads)),
        maxtasksperchild=int(TASKS_PER_CHILD),
    ) as pool:
        dispatcher = hpc_scheduler.MemoryAwareDispatcher(
            pool, budget_gb=budget_gb, max_concurrent=pool_size
        )
        dispatcher.run(ordered, _prepare, _on_result, _on_error)

    print(f"Done: wrote={counters['written']}, skipped={counters['skipped']}, "
          f"failed={counters['failed']}. {time.time() - started:.1f} s elapsed.")
    if counters["failed"]:
        raise SystemExit(f"{counters['failed']} solve unit(s) failed.")


if __name__ == "__main__":
    main()
