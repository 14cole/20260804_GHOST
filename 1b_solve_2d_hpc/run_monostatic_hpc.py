#!/usr/bin/env python3
"""Submit the step-1 coupon library to SLURM.

There are no staged run directories and no collection step. Array workers
write completed files directly to results/FRD or results/OPN, while SLURM
stdout/stderr goes to hpc_logs/.

Scheduling: units are costed from the mesh the solver will actually build and
dealt out longest-processing-time-first, then claimed at run time through
atomic files in claims/. That means

- a 2-18 GHz sweep does not strand every expensive unit on one array task
  (unit cost grows like the square of the boundary-node count, so the spread
  across a frequency sweep is large, and the old index-modulo split was blind
  to it);
- an array task that finishes early steals whatever is still unclaimed;
- a preempted or cancelled task loses only its in-flight units, and a later
  submission joins the same run rather than redoing it;
- concurrent solves are admitted against the node's memory allocation instead
  of filling every core regardless of unit size.

Delete claims/ to force a completely fresh sweep (results are still reused
unless FORCE is set).
"""

import argparse
from multiprocessing import Pool
import os
from pathlib import Path
import shlex
import shutil
import subprocess
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

# ── USER SETTINGS ──────────────────────────────────────────────────────────
FREQUENCIES_GHZ = [3.0, 6.0]
ANGLES_DEG = np.arange(0.0, 180.1, 5.0)
POLARIZATIONS = ["TM", "TE"]
GEOMETRY_UNITS = "meters"
SOLVER_METHOD = "auto"
MAX_PANELS = 50_000
FORCE = False

ARRAY_TASKS = 1               # number of simultaneously scheduled nodes
MAX_WORKERS_PER_TASK = None   # None = allocated CPU count
BLAS_THREADS_PER_WORKER = 1
MEMORY_HEADROOM = 0.85        # fraction of the node's memory the scheduler
                              # may reserve for concurrent solves
MEMORY_SAFETY = 1.35          # multiplier on the solver's own peak estimate
ASSEMBLY_THREADS = "auto"     # "auto", or an integer >= 1
TASKS_PER_CHILD = 4           # pool worker lifetime, in units
CLAIM_STALE_SECONDS = 3600    # a quiet claim older than this is stealable
SLURM_PARTITION = "compute"
SLURM_ACCOUNT = None
SLURM_TIME = None
SLURM_MEMORY = "0"            # all node memory; use None for cluster default
SLURM_CPUS = None             # None requests an exclusive node
SUBMIT = True
# ───────────────────────────────────────────────────────────────────────────


def _configuration():
    try:
        frequencies, angles, polarizations = validate_config(
            FREQUENCIES_GHZ, ANGLES_DEG, POLARIZATIONS
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if int(ARRAY_TASKS) < 1:
        raise SystemExit("ARRAY_TASKS must be at least 1.")
    jobs = discover_jobs(HERE, frequencies, polarizations)
    if not jobs:
        raise SystemExit(
            "No .geo files found. Put coupons in geometries/FRD or "
            "geometries/OPN; either folder may otherwise be empty."
        )
    return angles, prepare_jobs(
        jobs,
        angles_deg=angles,
        geometry_units=GEOMETRY_UNITS,
        solver_method=SOLVER_METHOD,
        max_panels=MAX_PANELS,
        runner_path=__file__,
    )


def _validate_config() -> 'None':
    """Validate user settings without discovering or submitting work."""
    try:
        validate_config(FREQUENCIES_GHZ, ANGLES_DEG, POLARIZATIONS)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if int(ARRAY_TASKS) < 1:
        raise SystemExit("ARRAY_TASKS must be at least 1.")
    if not 0.0 < float(MEMORY_HEADROOM) <= 1.0:
        raise SystemExit("MEMORY_HEADROOM must be in (0, 1].")
    if float(MEMORY_SAFETY) < 1.0:
        raise SystemExit("MEMORY_SAFETY must be >= 1.")
    if int(TASKS_PER_CHILD) < 1:
        raise SystemExit("TASKS_PER_CHILD must be >= 1.")
    if ASSEMBLY_THREADS != "auto" and int(ASSEMBLY_THREADS) < 1:
        raise SystemExit("ASSEMBLY_THREADS must be 'auto' or an integer >= 1.")


def _plan(jobs, n_angles, n_slots):
    """Cost and size every job, then deal them out longest-first.

    The mesh depends on the geometry and frequency but not the polarization,
    so a coupon library with both polarizations meshes each pair once.
    """

    from solver_quality import validate_mesh_convergence_policy

    fine_factor = float(validate_mesh_convergence_policy()["fine_factor"])
    nodes_cache = {}
    records = []
    for job in jobs:
        key = (str(job["geometry"]), float(job["frequency_ghz"]))
        if key not in nodes_cache:
            nodes_cache[key] = hpc_scheduler.predict_2d_nodes(
                key[0], key[1], str(job["polarization"]),
                GEOMETRY_UNITS, MAX_PANELS,
            )
        nodes = int(nodes_cache[key])
        records.append({
            "unit": str(job["output"]),
            "nodes": nodes,
            "cost": (
                hpc_scheduler.unit_cost(nodes, n_angles, fine_factor)
                if nodes > 0 else 1.0
            ),
            "peak_gb": (
                hpc_scheduler.unit_peak_gb(
                    nodes, fine_factor, safety=float(MEMORY_SAFETY)
                ) if nodes > 0 else 0.0
            ),
        })
    assignment = hpc_scheduler.balance_units(records, n_slots)
    for record, slot in zip(records, assignment):
        record["slot"] = int(slot)
    return records, hpc_scheduler.slot_plan_summary(records, assignment, n_slots)


def _claim_key(job) -> 'str':
    """Claim name for one job: its output file name plus its role folder.

    FRD and OPN can hold the same coupon stem, so the role has to be part of
    the key or the two would claim each other's units.
    """

    return f"{job['role']}__{Path(job['output']).name}"


def _resolve_assembly_threads(cores: 'int', pool_size: 'int') -> 'int':
    """Threads per solve, sized so threads x processes never exceeds the cores.

    Keyed on the pool size rather than on this task's planned share: a task
    that runs dry falls through to stealing and can end up with a full pool,
    so sizing threads for the planned share would oversubscribe the node
    exactly when it got busiest.
    """

    if ASSEMBLY_THREADS != "auto":
        return max(1, int(ASSEMBLY_THREADS))
    return max(1, int(cores) // max(1, int(pool_size)))


def _pool_initializer(blas_threads: 'int', assembly_threads: 'int') -> 'None':
    _pin_blas(blas_threads)
    import rcs_solver

    rcs_solver.set_assembly_threads(assembly_threads)


def worker(task_index: 'int') -> 'None':
    _validate_config()
    angles, jobs = _configuration()
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", ARRAY_TASKS))
    if task_count < 1:
        raise SystemExit("SLURM array task count must be at least 1.")

    records, _summary = _plan(jobs, len(angles), task_count)
    costs = {r["unit"]: float(r["cost"]) for r in records}
    peaks = {r["unit"]: float(r["peak_gb"]) for r in records}
    slots = {r["unit"]: int(r["slot"]) for r in records}

    def _order_key(job):
        return (-costs.get(str(job["output"]), 1.0), str(job["output"]))

    mine = [j for j in jobs if slots.get(str(j["output"]), -1) == int(task_index)]
    others = [j for j in jobs if slots.get(str(j["output"]), -1) != int(task_index)]
    # Own share first (dearest first, which is what keeps the makespan down),
    # then everyone else's as a steal pool for whoever finishes early.
    candidates = sorted(mine, key=_order_key) + sorted(others, key=_order_key)

    cores = hpc_scheduler.detect_cores()
    memory_gb = hpc_scheduler.detect_memory_gb()
    budget_gb = max(1.0, memory_gb * float(MEMORY_HEADROOM))
    worker_limit = cores if MAX_WORKERS_PER_TASK is None else int(MAX_WORKERS_PER_TASK)
    pool_size = max(1, min(cores, worker_limit, len(candidates)))
    assembly_threads = _resolve_assembly_threads(cores, pool_size)

    kwargs = {
        "angles_deg": angles,
        "geometry_units": GEOMETRY_UNITS,
        "solver_method": SOLVER_METHOD,
        "max_panels": MAX_PANELS,
        "force": FORCE,
    }
    print(
        f"Task {task_index}/{task_count - 1}: {len(jobs)} unit(s) in the sweep, "
        f"{len(mine)} planned here, {pool_size} worker(s) x "
        f"{assembly_threads} assembly thread(s), "
        f"{budget_gb:.1f}/{memory_gb:.1f} GB schedulable. "
        "Direct output to results/{FRD,OPN}.",
        flush=True,
    )
    _pin_blas(BLAS_THREADS_PER_WORKER)

    # Parse each distinct geometry, and import the solver, before the pool
    # forks: workers then inherit both instead of repeating the work per unit.
    prewarm_snapshots(candidates)
    import rcs_solver  # noqa: F401
    import grim_io     # noqa: F401

    broker = hpc_scheduler.ClaimBroker(
        HERE / "claims", stale_seconds=float(CLAIM_STALE_SECONDS)
    )
    broker.start_heartbeat()
    counters = {"written": 0, "skipped": 0, "failed": 0, "passed": 0}
    started = time.time()
    total = len(candidates)

    mine_outputs = {str(j["output"]) for j in mine}

    def _prepare(job):
        key = _claim_key(job)
        dispatch = (
            key, peaks.get(str(job["output"]), 0.0),
            (solve_job_catching, ((job, kwargs),)),
        )
        if not FORCE and Path(job["output"]).exists():
            # Dispatched, not skipped outright: solve_job re-verifies the
            # stored fingerprint and raises if the file on disk came from
            # different inputs or solver settings, which is what makes reusing
            # an interrupted sweep safe. Only the task that owns the unit does
            # it, so each output is checked once per sweep and no claim is
            # needed for a result that is already final.
            if str(job["output"]) not in mine_outputs:
                counters["passed"] += 1
                return None
            return dispatch
        if not broker.try_claim(key):
            counters["passed"] += 1
            return None
        return dispatch

    def _done():
        return counters["written"] + counters["skipped"] + counters["failed"]

    def _on_result(key, payload):
        kind, first, second, job = payload
        if kind == "ok":
            counters["skipped" if first == "skipped" else "written"] += 1
            broker.release(key)
            print(f"[{_done():4d}/{total}] {first:7s} "
                  f"{job['role']}/{Path(second).name}", flush=True)
        else:
            counters["failed"] += 1
            broker.abandon(key)
            print(f"[{_done():4d}/{total}] FAILED "
                  f"{job['role']}/{Path(job['output']).name}\n{first}", flush=True)

    def _on_error(key, exc):
        counters["failed"] += 1
        broker.abandon(key)
        print(f"[{_done():4d}/{total}] FAILED (dispatch) {key}: {exc!r}", flush=True)

    with Pool(
        processes=pool_size,
        initializer=_pool_initializer,
        initargs=(int(BLAS_THREADS_PER_WORKER), int(assembly_threads)),
        maxtasksperchild=int(TASKS_PER_CHILD),
    ) as pool:
        dispatcher = hpc_scheduler.MemoryAwareDispatcher(
            pool, budget_gb=budget_gb, max_concurrent=pool_size
        )
        try:
            dispatcher.run(candidates, _prepare, _on_result, _on_error)
        finally:
            broker.stop_heartbeat()

    print(f"Task {task_index} complete: wrote={counters['written']}, "
          f"skipped={counters['skipped']}, failed={counters['failed']}, "
          f"left to other tasks={counters['passed']}. "
          f"{time.time() - started:.1f} s elapsed.", flush=True)
    if counters["failed"]:
        raise SystemExit(f"{counters['failed']} solve unit(s) failed.")


def submit() -> 'None':
    _validate_config()
    angles, jobs = _configuration()
    if shutil.which("sbatch") is None and SUBMIT:
        raise SystemExit("SUBMIT=True but sbatch is not available.")
    logs = HERE / "hpc_logs"
    logs.mkdir(parents=True, exist_ok=True)
    (HERE / "claims").mkdir(parents=True, exist_ok=True)

    records, summary = _plan(jobs, len(angles), int(ARRAY_TASKS))
    peaks = [r["peak_gb"] for r in records if r["peak_gb"] > 0]

    command = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))} "
        "--worker ${SLURM_ARRAY_TASK_ID}"
    )
    args = [
        "sbatch",
        "--job-name=step1_2d",
        f"--array=0-{int(ARRAY_TASKS) - 1}",
        "--nodes=1",
        "--ntasks=1",
        f"--partition={SLURM_PARTITION}",
        f"--chdir={HERE}",
        f"--output={logs}/%A_%a.out",
        f"--error={logs}/%A_%a.err",
        # A requeued task rejoins the sweep and picks up whatever is unclaimed,
        # so preemption costs only the units that were in flight.
        "--requeue",
        "--open-mode=append",
    ]
    if SLURM_ACCOUNT:
        args.append(f"--account={SLURM_ACCOUNT}")
    if SLURM_TIME:
        args.append(f"--time={SLURM_TIME}")
    if SLURM_MEMORY:
        args.append(f"--mem={SLURM_MEMORY}")
    if SLURM_CPUS:
        args.append(f"--cpus-per-task={int(SLURM_CPUS)}")
    else:
        args.append("--exclusive")
    args.extend(["--wrap", command])
    print(
        f"Step 1 HPC: {len(jobs)} solve unit(s), {ARRAY_TASKS} array task(s)\n"
        f"  Plan balance : {summary['imbalance']:.2f}x the best any schedule could "
        "do (1.00 = optimal; stealing absorbs the rest)"
    )
    idle = int(summary.get("idle_slots", 0))
    if idle:
        print(f"  {idle} of {ARRAY_TASKS} task(s) have no planned work -- fewer "
              "units than tasks, so those exit at once")
    if peaks:
        print(f"  Unit peak RAM: {min(peaks):.2f}-{max(peaks):.2f} GB estimated "
              f"(incl. {MEMORY_SAFETY:g}x safety)")
    unknown = sum(1 for r in records if int(r["nodes"]) <= 0)
    if unknown:
        print(f"  [warn] {unknown} unit(s) could not be pre-meshed; they carry "
              "unit cost and no memory reservation.")
    print("Results write directly to results/{FRD,OPN}; logs write to hpc_logs/.")
    if not SUBMIT:
        print("SUBMIT=False. Command:\n" + " ".join(shlex.quote(x) for x in args))
        return
    completed = subprocess.run(
        args,
        check=False,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise SystemExit(completed.stderr or completed.stdout)
    print(completed.stdout.strip())


def main() -> 'None':
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    arguments = parser.parse_args()
    if arguments.worker is None:
        submit()
    else:
        worker(arguments.worker)


if __name__ == "__main__":
    main()
