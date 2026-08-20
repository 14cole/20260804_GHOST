#!/usr/bin/env python3
"""
HPC monostatic RCS sweep driver (SLURM).

Edit the CONFIG block below and run:

    python run_hpc_monostatic.py

Workflow:
- Discover geometry files under FRD_DIR + OPN_DIR.
- Expand into a (geometry x frequency x polarization) unit list. All azimuths
  for a unit are solved in a single solver call (matrix factored once).
- Cost every unit from the mesh the solver will actually build, deal the units
  out longest-processing-time-first, and write that plan beside the manifest.
- Write N_JOBS sbatch job arrays and submit them. Every array task runs the
  same worker: each takes its planned share, then steals whatever is still
  unclaimed, so the tail of a run rebalances itself.
- As each unit finishes, its result is exported immediately to
  "<POL>_<FREQ:.3f>GHz_<geometry_stem>.grim" in
  <run_dir>/results/{FRD,OPN}/.  The role-preserving layout can be passed
  directly to the downstream concatenate/subtract tools.

Scheduling notes -- what makes a big sweep finish sooner:

- COST-AWARE PLANNING.  Assembly is O(N^2) in boundary nodes and N grows with
  frequency, so a 2-18 GHz sweep spans roughly 80x in unit cost.  Handing units
  out round-robin over their index (what this used to do) can put every
  high-frequency unit on one slot.  Units are costed at submit time from the
  real mesh and packed longest-first instead.
- WORK STEALING.  Array tasks are interchangeable and coordinate only through
  atomic claim files, so an early-finishing node picks up someone else's
  backlog, a preempted or requeued task loses only its in-flight units, and a
  second submission on another partition can join a sweep already in progress.
- MEMORY-AWARE ADMISSION.  Solves start while the sum of their estimated peaks
  fits the node's memory allocation, instead of filling every core regardless.
  On a 96-core / 750 GB node that runs many small units at once and narrows to
  a few when the expensive ones come up, rather than OOM-killing pool workers
  (a cgroup kill, unlike a Python MemoryError, can wedge the pool).
- CHEAP PER-UNIT PROVENANCE.  The before/after source and input checks stay,
  but repeat hashes come from a stat-keyed cache that expires on a timer, so a
  sweep of thousands of units does not spend its time re-reading the backend
  tree over a shared filesystem.

Restartable: a unit whose .grim already exists is skipped once its attestation
verifies, so cancelling and resubmitting is always safe.

Internal worker invocation (called by SLURM, not by the user):
    python run_hpc_monostatic.py --worker <run_dir> <submission_index> <task_index>
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hpc_scheduler
import workflow_provenance as _workflow_provenance
from geometry_io import material_sidecar_paths
from solver_quality import validate_mesh_convergence_policy
from workflow_provenance import (
    backend_source_fingerprint,
    backend_source_inventory,
    describe_source_mismatch,
    manifest_solve_spec_fingerprint,
    runtime_environment_fingerprint,
    stable_json_fingerprint,
    embed_output_attestation,
    unit_solve_spec_fingerprint,
    verify_embedded_attestation,
)

# ===============================================================================
# CONFIG -- the only section most users need to edit
# ===============================================================================

# Input geometry folders. Every *.geo file found under these paths
# (recursively) is added to the sweep. Source folder is NOT injected into
# output filenames -- the geometry filename is preserved verbatim.
FRD_DIR = "geometries/FRD"
OPN_DIR = "geometries/OPN"

# Requested sweep.
FREQUENCIES_GHZ = [2.0, 4.0, 6.0, 8.0, 10.0]
AZIMUTHS_DEG    = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
POLARIZATIONS   = ["VV", "HH"]          # any subset of: VV, HH, TM, TE

# Output root. A new run_YYYYMMDD_HHMMSS/ subfolder is created inside.
OUTPUT_DIR = "rcs_runs"

# --- How much of the cluster to use ----------------------------------------
# N_NODES is the size of each sbatch job array, N_JOBS the number of separate
# submissions (use more than one to spread across partitions or accounts).
# Total parallelism is N_NODES x N_JOBS nodes.
#
# Unlike the previous round-robin scheme these are pure throughput knobs. Array
# tasks are interchangeable and pull work from a shared claim directory, so
# nothing is stranded if a task never starts, is cancelled, or is preempted,
# and raising N_NODES on a later submission simply adds capacity to a run that
# is already going.
N_NODES = 1
N_JOBS  = 1

# Cap on array tasks running at once (SLURM's `--array=...%N`). None = no cap.
ARRAY_THROTTLE = None

# ===============================================================================
# ADVANCED -- fine tuning (SLURM resources, solver knobs, env setup)
# ===============================================================================

# --- SLURM resources (per array task = one node) ---------------------------
SLURM_PARTITION = "compute"
SLURM_ACCOUNT   = None            # e.g. "my_project"; None to omit
SLURM_QOS       = None
SLURM_TIME      = None            # None = no walltime limit; or "HH:MM:SS"
CORES_PER_NODE  = None            # None = request whole node via --exclusive
                                  # (pool size auto-detected from SLURM env).
                                  # Or set an integer, e.g. 32.
MEM_PER_NODE    = "0"             # "0" = ALL memory of the node (SLURM idiom;
                                  # recommended with --exclusive). None = omit
                                  # the directive -> cluster default applies
                                  # (often DefMemPerCPU ~3.5G x CPUs, which can
                                  # be far less than node RAM). Or e.g. "64G".
MAX_WORKERS_PER_NODE = None       # Hard ceiling on concurrent solves per node.
                                  # None = one per allocated core. The memory
                                  # budget below is usually the binding
                                  # constraint, so this rarely needs setting.
MEMORY_HEADROOM = 0.85            # Fraction of the node's memory allocation the
                                  # scheduler may reserve for solves. The rest
                                  # covers the parent process, page cache, and
                                  # the gap between estimate and reality.
MEMORY_SAFETY   = 1.35            # Multiplier on the solver's own dense-storage
                                  # estimate, covering allocator slack and the
                                  # transient copies a factorization makes.
MAX_SOLVE_GB    = None            # Hard ceiling on ONE solve's estimated
                                  # footprint, exported to the job as
                                  # GHOST_MAX_SOLVE_GB. None = derive it from
                                  # what the node reports (0.9 x detected,
                                  # floored at 32 GB).
                                  #
                                  # Set it when you intend to run something
                                  # very large. Detection needs a number it
                                  # can trust: with MEM_PER_NODE = "0" SLURM
                                  # reports 0, so the limit falls back to the
                                  # cgroup or /proc/meminfo, and if neither is
                                  # meaningful the ceiling collapses to 32 GB
                                  # and a big solve is refused on a big node.
                                  # Either set this, or give MEM_PER_NODE an
                                  # explicit size like "750G".
SLURM_MAIL_TYPE = None            # e.g. "END,FAIL"
SLURM_MAIL_USER = None
SLURM_EXTRA_SBATCH = []  # type: List[str]  # raw extra lines, e.g. "--constraint=intel"

JOB_PROLOGUE = []  # type: List[str]

# --- Solver knobs (mirror run_monostatic.py) -------------------------------
GEOMETRY_UNITS          = "inches"       # "inches" or "meters"
SOLVER_METHOD           = "auto"         # "auto" | "direct"
# The present 2-D SLP formulations do not implement a distinct CFIE operator.
# Keep this at zero; nonzero values are rejected instead of acting as a dead
# production control.
CFIE_ALPHA              = 0.0
MAX_PANELS              = 50_000

# Mesh-convergence certification. True solves every unit twice -- the requested
# mesh and one refined by the policy's fine_factor -- and publishes the fine
# result only if the two agree. That second solve is where most of the wall
# clock and all of the peak memory go, because cost scales with the square of
# the node count: turning it off is about 3x faster per unit and roughly halves
# the memory, so more units fit per node as well.
#
# With this off the algebraic quality gate still runs, but no base/fine mesh
# comparison is performed.  The choice is recorded for provenance and does not
# prevent the result from being viewed, combined, subtracted, or used by the
# downstream feature workflow.
MESH_CERTIFICATION      = True
BLAS_THREADS_PER_WORKER = 1              # keeps N workers x BLAS threads sane

# Threads each solve may use inside the boundary-operator assembly.
# "auto" gives every concurrent solve an equal share of the node's cores. That
# only matters when a node holds fewer units than it has cores -- a couple of
# geometries at one frequency, or the tail of a sweep; with at least one unit
# per core "auto" resolves to 1 and the process pool owns the parallelism.
ASSEMBLY_THREADS        = "auto"         # "auto", or an integer >= 1

# Pool worker lifetime. Each worker is replaced after this many units so
# allocator growth from a big solve cannot accumulate across a long sweep. The
# solver is imported in the parent, so a forked worker inherits it and a
# respawn costs a fork rather than a re-import of numpy, SciPy, and the solver.
TASKS_PER_CHILD = 4

# A claim whose heartbeat has been quiet this long is treated as abandoned and
# may be taken over. Must comfortably exceed the longest single unit.
CLAIM_STALE_SECONDS = 3600

# --- Geometry discovery & submission ---------------------------------------
GEOMETRY_EXTS = (".geo",)
PYTHON_EXE    = sys.executable           # interpreter used inside the job
SUBMIT        = True                     # False -> write .slurm files but don't sbatch

# ===============================================================================

_SBATCH = shutil.which("sbatch") or "sbatch"
MANIFEST_SCHEMA = "ghost.hpc.2d-run.v1"
SCHEDULE_SCHEMA = "ghost.hpc.2d-schedule.v1"

# Parsed geometry snapshots, filled in the parent before the pool forks so
# workers inherit them instead of unpickling one per unit.
_SNAPSHOT_CACHE = {}  # type: Dict[str, Tuple[Dict[str, Any], str]]


# --- shared helpers --------------------------------------------------------

def _solver_source_records():
    # type: () -> Tuple[str, Dict[str, str]]
    """(backend directory, extra logical records) the fingerprint is built from.

    The running driver is recorded under a fixed logical name so that submit
    (which runs the configured copy) and the worker (which runs the copy SLURM
    execs out of the run directory) hash the same bytes under the same name.
    """

    backend_dir = str(Path(_workflow_provenance.__file__).resolve().parent)
    return backend_dir, {"driver_configured.py": str(Path(__file__).resolve())}


def _solver_source_fingerprint():
    # type: () -> str
    backend_dir, extra = _solver_source_records()
    return backend_source_fingerprint(backend_dir, extra)


def _solver_source_inventory():
    # type: () -> Dict[str, str]
    backend_dir, extra = _solver_source_records()
    return backend_source_inventory(backend_dir, extra)


def _verify_run_provenance(context):
    # type: (Dict[str, Any]) -> None
    """Re-check that the solver source and numerical runtime still match the run.

    Called around every unit, exactly as before. What changed is the cost: the
    file hashes underneath come from `hpc_scheduler.install_fingerprint_cache`,
    so a repeat check is a stat per backend file rather than a full re-read,
    and the cache expires on a timer so full re-reads keep happening inside
    long-lived workers.
    """

    expected_source = str(context.get("solver_source_sha256", ""))
    expected_runtime = str(context.get("runtime_environment_sha256", ""))
    if not expected_source or not expected_runtime:
        raise RuntimeError(
            "HPC run manifest lacks exact solver-source/runtime provenance; "
            "legacy runs must be regenerated before reuse."
        )
    if _solver_source_fingerprint() != expected_source:
        # Name the files. "Something under Backend/ differs" is not actionable,
        # and the usual cause is a tree that was only partly updated.
        detail = describe_source_mismatch(
            context.get("solver_source_inventory") or {},
            _solver_source_inventory(),
        )
        if not context.get("solver_source_inventory"):
            detail = ("this run predates per-file inventories, so the "
                      "differing file cannot be named -- run "
                      "tests/diagnose_provenance.py for what is checked")
        raise RuntimeError(
            "Solver source/native artifacts differ from the HPC run manifest; "
            "no cached or new field will be used from this mixed source state. "
            f"({detail}). Either restore the recorded source or submit a new "
            "run with the code you actually want to execute."
        )
    if runtime_environment_fingerprint() != expected_runtime:
        raise RuntimeError(
            "Python/platform/NumPy/SciPy/BLAS runtime differs from the HPC run "
            "manifest; start a new run in this numerical environment."
        )


def _unit_attestation_fields(context, unit):
    # type: (Dict[str, Any], Dict[str, Any]) -> Dict[str, Any]
    return {
        "run_id": str(context["run_id"]),
        "solver_source_sha256": str(context["solver_source_sha256"]),
        "runtime_environment_sha256": str(context["runtime_environment_sha256"]),
        "geometry_input_sha256": str(unit["geometry_input_sha256"]),
        "run_solve_spec_sha256": str(context["run_solve_spec_sha256"]),
        "unit_solve_spec_sha256": unit_solve_spec_fingerprint(unit),
        "solver_config_sha256": str(context["solver_config_sha256"]),
        "angular_grid_kind": "azimuths_deg",
        "angular_grid_sha256": str(context["angular_grid_sha256"]),
        "polarization": str(unit["polarization"]),
        "frequency_ghz": float(unit["frequency_ghz"]),
    }


def _verify_unit_input(unit, context):
    # type: (Dict[str, Any], Dict[str, Any]) -> None
    from feature_sum import geometry_input_fingerprint
    current = geometry_input_fingerprint(
        str(unit["geometry"]), str(context["geometry_units"])
    )
    if current != unit.get("geometry_input_sha256"):
        raise RuntimeError(
            f"Frozen geometry/material input changed during the HPC unit: "
            f"{unit['geometry']}"
        )


def _discover_geometries():
    # type: () -> List[Path]
    """Return every geometry file under FRD_DIR/OPN_DIR (deduplicated)."""
    found = []   # type: List[Path]
    seen = set()  # type: set
    for d in (FRD_DIR, OPN_DIR):
        root = Path(d)
        if not root.is_dir():
            print(f"  [warn] dir not found: {root}", file=sys.stderr)
            continue
        for ext in GEOMETRY_EXTS:
            for p in sorted(root.rglob(f"*{ext}")):
                rp = p.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                found.append(p)
    return found


def _geometry_role(path):
    # type: (Path) -> str
    """Return the configured FRD/OPN role for a discovered geometry."""

    resolved = Path(path).resolve()
    for role, directory in (("FRD", FRD_DIR), ("OPN", OPN_DIR)):
        try:
            resolved.relative_to(Path(directory).resolve())
            return role
        except ValueError:
            continue
    raise ValueError(f"Geometry is outside configured input folders: {path}")


def _unit_name(unit):
    # type: (Dict[str, Any]) -> str
    return (f"{unit['polarization']}_{float(unit['frequency_ghz']):.3f}GHz_"
            f"{unit['geometry_stem']}.grim")


def _unit_output_path(run_dir, unit):
    # type: (Path, Dict[str, Any]) -> Path
    role = str(unit.get("role", "")).strip().upper()
    folder = run_dir / "results" / role if role else run_dir / "results"
    return folder / _unit_name(unit)


def _load_snapshot(geometry_path):
    # type: (str) -> Tuple[Dict[str, Any], str]
    """Parsed snapshot for one geometry, built at most once per process.

    The parent fills this before forking the pool, so on a fork start method
    every worker inherits the snapshots copy-on-write. The fallback parse keeps
    the worker correct under a spawn start method, at the cost of one parse.
    """

    cached = _SNAPSHOT_CACHE.get(geometry_path)
    if cached is not None:
        return cached
    from geometry_io import parse_geometry, build_geometry_snapshot

    path = Path(geometry_path)
    title, segments, ibcs, dielectrics = parse_geometry(path.read_text())
    snapshot = build_geometry_snapshot(title, segments, ibcs, dielectrics)
    snapshot["source_path"] = str(path)
    entry = (snapshot, str(path.parent))
    _SNAPSHOT_CACHE[geometry_path] = entry
    return entry


def _pool_initializer(blas_threads):
    # type: (int) -> None
    hpc_scheduler.pin_blas_threads(blas_threads)
    hpc_scheduler.install_fingerprint_cache()
    import rcs_solver

    rcs_solver.set_assembly_threads(1)


def _solve_and_export(unit, context, run_dir_str):
    # type: (Dict[str, Any], Dict[str, Any], str) -> Tuple[str, str]
    """Pool-worker entry point: solve one unit, export .grim. Idempotent."""

    run_dir = Path(run_dir_str)
    out_path = _unit_output_path(run_dir, unit)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _verify_run_provenance(context)
    _verify_unit_input(unit, context)
    attestation = _unit_attestation_fields(context, unit)
    if out_path.exists():
        verify_embedded_attestation(str(out_path), attestation)
        return ("skipped", str(out_path))

    snapshot, material_base = _load_snapshot(str(unit["geometry"]))

    solve_kwargs = dict(
        geometry_snapshot=snapshot,
        frequencies_ghz=[float(unit["frequency_ghz"])],
        elevations_deg=[float(a) for a in context["azimuths_deg"]],
        polarization=unit["polarization"],
        geometry_units=context["geometry_units"],
        material_base_dir=material_base,
        max_panels=context["max_panels"],
        cfie_alpha=context["cfie_alpha"],
        solver_method=context["solver_method"],
    )
    if context["mesh_certification"]:
        from rcs_solver import solve_monostatic_rcs_2d_certified
        result = solve_monostatic_rcs_2d_certified(
            mesh_convergence_policy=context["mesh_convergence_policy"],
            **solve_kwargs
        )
    else:
        from rcs_solver import solve_monostatic_rcs_2d_survey
        result = solve_monostatic_rcs_2d_survey(**solve_kwargs)
    _verify_run_provenance(context)
    _verify_unit_input(unit, context)

    # Bind the result to its run state inside the artifact, before export, so
    # results/ holds one file per unit instead of a .grim and a sidecar.
    embed_output_attestation(result, attestation)

    from grim_io import export_result_to_grim
    written = export_result_to_grim(
        result, str(out_path),
        source_path=str(snapshot.get("source_path", "") or ""),
        history=(f"run_hpc_monostatic.py pol={unit['polarization']} "
                 f"freq={unit['frequency_ghz']}GHz"),
    )
    actual_path = str(written[0]) if written else str(out_path)
    _verify_run_provenance(context)
    _verify_unit_input(unit, context)
    return ("written", actual_path)


def _solve_and_export_star(args):
    # type: (tuple) -> tuple
    """Pool entry point: unpack args and catch exceptions in-band.

    The full traceback string is returned (not just str(exc)) so the SLURM log
    shows where the failure happened, not just the message.
    """

    unit, context, run_dir_str, assembly_threads = args
    try:
        import rcs_solver
        rcs_solver.set_assembly_threads(assembly_threads)
        status, path = _solve_and_export(unit, context, run_dir_str)
        return ("ok", status, path)
    except Exception:
        return ("err", traceback.format_exc(), "")


# --- submit mode (user-invoked) --------------------------------------------

def _plan_schedule(units, n_slots, fine_factor, n_angles):
    # type: (List[Dict[str, Any]], int, float, int) -> Dict[str, Any]
    """Cost every unit, size its memory, and deal the units out to slots.

    Geometry validation/material loading is batched once per geometry, and
    polarization-independent frequency meshes are shared only when their
    complete interface signatures match. Every unit still receives an exact
    formulation-specific memory record.
    """

    resource_cache = {}  # type: Dict[Tuple[str, float, str], Dict[str, Any]]
    grouped = {}  # type: Dict[str, Dict[str, List[Any]]]
    for unit in units:
        geometry = str(unit["geometry"])
        group = grouped.setdefault(geometry, {"frequencies": [], "polarizations": []})
        frequency = float(unit["frequency_ghz"])
        polarization = str(unit["polarization"])
        if frequency not in group["frequencies"]:
            group["frequencies"].append(frequency)
        if polarization not in group["polarizations"]:
            group["polarizations"].append(polarization)

    total = sum(
        len(group["frequencies"]) * len(group["polarizations"])
        for group in grouped.values()
    )
    started = time.monotonic()
    completed = 0
    report_step = max(1, total // 20)
    next_report = report_step
    last_report = started
    print(
        f"  Planning exact resources for {total} unit(s) across "
        f"{len(grouped)} geometry file(s)...",
        flush=True,
    )

    def planning_progress(_frequency, _polarization):
        # type: (float, str) -> None
        nonlocal completed, next_report, last_report
        completed += 1
        now = time.monotonic()
        if completed < total and completed < next_report and now - last_report < 10.0:
            return
        elapsed = max(now - started, 1.0e-9)
        rate = completed / elapsed
        eta = (total - completed) / rate if rate > 0.0 else 0.0
        print(
            f"    planned {completed}/{total} ({100.0 * completed / total:.0f}%) "
            f"in {elapsed:.1f}s, ETA {eta:.1f}s",
            flush=True,
        )
        while next_report <= completed:
            next_report += report_step
        last_report = now

    for geometry, group in grouped.items():
        batch = hpc_scheduler.predict_2d_resources_many(
            geometry,
            group["frequencies"],
            group["polarizations"],
            GEOMETRY_UNITS,
            MAX_PANELS,
            fine_factor=fine_factor,
            n_angles=n_angles,
            safety=float(MEMORY_SAFETY),
            progress=planning_progress,
        )
        for (frequency, polarization), planned in batch.items():
            resource_cache[(geometry, frequency, polarization)] = planned

    records = []     # type: List[Dict[str, Any]]
    for unit in units:
        key = (
            str(unit["geometry"]), float(unit["frequency_ghz"]),
            str(unit["polarization"]),
        )
        planned = resource_cache[key]
        nodes = int(planned["nodes"])
        records.append({
            "unit": _unit_name(unit),
            "nodes": nodes,
            "fine_nodes": int(planned["fine_nodes"]),
            "base_system_dofs": int(planned["base_system_dofs"]),
            "system_dofs": int(planned["system_dofs"]),
            "n_regions": int(planned["n_regions"]),
            "formulation": str(planned["formulation"]),
            "cost": (
                hpc_scheduler.unit_cost(
                    nodes, n_angles, fine_factor,
                    fine_nodes=int(planned["fine_nodes"]),
                    system_dofs=int(planned["base_system_dofs"]),
                    fine_system_dofs=int(planned["fine_system_dofs"]),
                    operator_matrices=int(planned["base_operator_matrices"]),
                    fine_operator_matrices=int(planned["fine_operator_matrices"]),
                )
            ),
            "peak_gb": float(planned["peak_gb"]),
        })
    assignment = hpc_scheduler.balance_units(records, n_slots)
    for record, slot in zip(records, assignment):
        record["slot"] = int(slot)
    elapsed = time.monotonic() - started
    print(f"  Resource plan ready in {elapsed:.1f}s.", flush=True)
    return {
        "schema": SCHEDULE_SCHEMA,
        "n_slots": int(n_slots),
        "fine_factor": float(fine_factor),
        "planning": {
            "method": "batched_exact",
            "elapsed_seconds": float(elapsed),
            "geometry_preflights": int(len(grouped)),
            "frequency_mesh_groups": int(
                sum(len(group["frequencies"]) for group in grouped.values())
            ),
            "unit_records": int(len(records)),
        },
        "units": records,
        "summary": hpc_scheduler.slot_plan_summary(records, assignment, n_slots),
    }


def _validate_config():
    # type: () -> Tuple[List[str], List[float], List[float]]
    pols = [p.strip().upper() for p in POLARIZATIONS if p and p.strip()]
    if not pols:            sys.exit("ERROR: POLARIZATIONS is empty.")
    if not FREQUENCIES_GHZ: sys.exit("ERROR: FREQUENCIES_GHZ is empty.")
    if not AZIMUTHS_DEG:    sys.exit("ERROR: AZIMUTHS_DEG is empty.")
    try:
        # Rejects an unknown label and, just as importantly, two spellings of
        # the same channel: ["VV", "TE"] looks like two polarizations and is
        # one, so it would otherwise solve identical physics twice and publish
        # it twice under different names.
        pols = hpc_scheduler.distinct_polarization_channels(pols)
    except ValueError as exc:
        sys.exit(f"ERROR: POLARIZATIONS is invalid -- {exc}")
    frequencies = [float(value) for value in FREQUENCIES_GHZ]
    if (
        not all(math.isfinite(value) and value > 0.0 for value in frequencies)
        or len(set(frequencies)) != len(frequencies)
        or len({f"{value:.3f}" for value in frequencies}) != len(frequencies)
    ):
        sys.exit(
            "ERROR: frequencies must be finite, positive, unique, and "
            "distinct at the 0.001 GHz output-name precision."
        )
    azimuths = [float(value) for value in AZIMUTHS_DEG]
    if (
        not all(math.isfinite(value) for value in azimuths)
        or len(set(azimuths)) != len(azimuths)
    ):
        sys.exit("ERROR: AZIMUTHS_DEG must be finite and unique.")
    if str(SOLVER_METHOD).strip().lower() not in {"auto", "direct"}:
        sys.exit(
            "ERROR: SOLVER_METHOD must be 'auto' or 'direct'; certified "
            "2-D production solves require a condition-reporting dense method."
        )
    if not math.isfinite(float(CFIE_ALPHA)) or float(CFIE_ALPHA) != 0.0:
        sys.exit(
            "ERROR: CFIE_ALPHA must be 0 for the current 2-D formulations; "
            "a nonzero value would not select a different operator."
        )
    if int(MAX_PANELS) < 1 or int(BLAS_THREADS_PER_WORKER) < 1:
        sys.exit("ERROR: MAX_PANELS and BLAS_THREADS_PER_WORKER must be >= 1.")
    if int(N_NODES) < 1 or int(N_JOBS) < 1:
        sys.exit("ERROR: N_NODES and N_JOBS must be >= 1.")
    if ARRAY_THROTTLE is not None and int(ARRAY_THROTTLE) < 1:
        sys.exit("ERROR: ARRAY_THROTTLE must be None or >= 1.")
    if not 0.0 < float(MEMORY_HEADROOM) <= 1.0:
        sys.exit("ERROR: MEMORY_HEADROOM must be in (0, 1].")
    if float(MEMORY_SAFETY) < 1.0:
        sys.exit("ERROR: MEMORY_SAFETY must be >= 1.")
    if int(TASKS_PER_CHILD) < 1:
        sys.exit("ERROR: TASKS_PER_CHILD must be >= 1.")
    if ASSEMBLY_THREADS != "auto" and int(ASSEMBLY_THREADS) < 1:
        sys.exit("ERROR: ASSEMBLY_THREADS must be 'auto' or an integer >= 1.")
    if int(CLAIM_STALE_SECONDS) < 60:
        sys.exit("ERROR: CLAIM_STALE_SECONDS must be at least 60.")
    if MAX_SOLVE_GB is not None and float(MAX_SOLVE_GB) <= 0.0:
        sys.exit("ERROR: MAX_SOLVE_GB must be positive or None.")
    return pols, frequencies, azimuths


def submit():
    # type: () -> None
    geometries = _discover_geometries()
    if not geometries:
        sys.exit("ERROR: no geometry files (*.geo) found under FRD_DIR or OPN_DIR.")

    pols, _frequencies, _azimuths = _validate_config()

    stems = [g.stem for g in geometries]
    if len(stems) != len(set(stems)):
        sys.exit("ERROR: geometry stems must be unique; per-unit result names "
                 "would otherwise overwrite one another.")

    run_id  = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = Path(OUTPUT_DIR).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    (run_dir / "results").mkdir()
    (run_dir / "results" / "FRD").mkdir()
    (run_dir / "results" / "OPN").mkdir()
    (run_dir / "claims").mkdir()

    # Freeze every geometry and its referenced material sidecars before
    # any worker can start.  A later submission may replace its staging bundle;
    # queued/archived runs must remain immutable.
    frozen_geometries = []
    for index, geom in enumerate(geometries):
        inp = run_dir / "inputs" / f"{index:04d}_{geom.stem}"
        inp.mkdir(parents=True, exist_ok=False)
        frozen = inp / geom.name
        shutil.copy2(str(geom), str(frozen))
        for table_name in material_sidecar_paths(str(geom)):
            table = Path(table_name)
            shutil.copy2(str(table), str(inp / table.name))
        frozen_geometries.append((geom, frozen))

    units = []  # type: List[Dict[str, Any]]
    from feature_sum import geometry_input_fingerprint
    for original, geom in frozen_geometries:
        input_fingerprint = geometry_input_fingerprint(str(geom), GEOMETRY_UNITS)
        for pol in pols:
            for f in FREQUENCIES_GHZ:
                # The angular grid is identical for every unit and is
                # recorded once at manifest level. Repeating it per unit made
                # the file 13x larger than it needed to be -- 52 MB for a
                # 9 100-unit sweep, 93% of it the same list over and over.
                units.append({
                    "geometry":      str(geom.resolve()),
                    "geometry_stem": geom.stem,
                    "geometry_original": str(original.resolve()),
                    "geometry_input_sha256": input_fingerprint,
                    "role":          _geometry_role(original),
                    "polarization":  pol,
                    "frequency_ghz": float(f),
                })

    mesh_policy = validate_mesh_convergence_policy()
    source_driver = Path(__file__).resolve()
    manifest = {
        "schema":          MANIFEST_SCHEMA,
        "run_id":          run_id,
        "created":         datetime.now().isoformat(),
        "frd_dir":         str(Path(FRD_DIR).resolve()),
        "opn_dir":         str(Path(OPN_DIR).resolve()),
        "output_dir":      str(run_dir),
        "frequencies_ghz": list(FREQUENCIES_GHZ),
        "azimuths_deg":    list(AZIMUTHS_DEG),
        "polarizations":   pols,
        "n_nodes":         int(N_NODES),
        "n_jobs":          int(N_JOBS),
        "n_slots":         int(N_NODES) * int(N_JOBS),
        "n_units":         len(units),
        "solver_source_sha256": _solver_source_fingerprint(),
        # Per-file hashes behind that fingerprint, so a later mismatch can say
        # which file moved instead of only that one did.
        "solver_source_inventory": _solver_source_inventory(),
        "runtime_environment_sha256": runtime_environment_fingerprint(),
        "solver_config": {
            "geometry_units":          GEOMETRY_UNITS,
            "solver_method":           SOLVER_METHOD,
            "cfie_alpha":              CFIE_ALPHA,
            "max_panels":              MAX_PANELS,
            "blas_threads_per_worker": BLAS_THREADS_PER_WORKER,
            "cores_per_node":          CORES_PER_NODE,
            "mesh_convergence_policy": mesh_policy,
            "mesh_certification": bool(MESH_CERTIFICATION),
        },
        "units": units,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # The schedule lives beside the manifest rather than inside it: it says
    # where work should run, not what is solved, and the manifest fingerprint
    # that per-unit attestations bind to must cover only the latter.
    n_slots = int(N_NODES) * int(N_JOBS)
    # fine_factor <= 1 tells the cost model there is only one mesh to solve.
    schedule = _plan_schedule(
        units, n_slots,
        float(mesh_policy["fine_factor"]) if MESH_CERTIFICATION else 1.0,
        len(AZIMUTHS_DEG),
    )
    (run_dir / "schedule.json").write_text(json.dumps(schedule, indent=2))

    script_path = run_dir / "driver_configured.py"
    shutil.copy2(str(source_driver), str(script_path))
    slurm_paths = []  # type: List[Path]
    for j in range(int(N_JOBS)):
        sp = run_dir / f"submit_job{j}.slurm"
        sp.write_text(hpc_scheduler.build_sbatch_script(
            job_name=f"rcs_{run_dir.name}_j{j}",
            run_dir=run_dir,
            script_path=script_path,
            array_size=int(N_NODES),
            array_throttle=ARRAY_THROTTLE,
            partition=SLURM_PARTITION,
            cpus_per_node=CORES_PER_NODE,
            mem_per_node=MEM_PER_NODE,
            walltime=SLURM_TIME,
            account=SLURM_ACCOUNT,
            qos=SLURM_QOS,
            mail_type=SLURM_MAIL_TYPE,
            mail_user=SLURM_MAIL_USER,
            extra_sbatch=SLURM_EXTRA_SBATCH,
            prologue=JOB_PROLOGUE,
            python_exe=PYTHON_EXE,
            worker_args=f"--worker {run_dir} {j} ${{SLURM_ARRAY_TASK_ID}}",
            submission_index=j,
            blas_threads=int(BLAS_THREADS_PER_WORKER),
            extra_env=(
                {"GHOST_MAX_SOLVE_GB": f"{float(MAX_SOLVE_GB):g}"}
                if MAX_SOLVE_GB else {}
            ),
        ))
        sp.chmod(0o755)
        slurm_paths.append(sp)

    peaks = [float(r["peak_gb"]) for r in schedule["units"] if r["peak_gb"] > 0]
    summary = schedule["summary"]
    print("=" * 70)
    print("HPC monostatic RCS sweep")
    print("=" * 70)
    print(f"  Run dir       : {run_dir}")
    print(f"  Geometries    : {len(geometries)}")
    print(f"  Polarizations : {', '.join(pols)}")
    print(f"  Frequencies   : {len(FREQUENCIES_GHZ)}  "
          f"({min(FREQUENCIES_GHZ):g}-{max(FREQUENCIES_GHZ):g} GHz)")
    print(f"  Azimuths      : {len(AZIMUTHS_DEG)}")
    print(f"  Units total   : {len(units)}  (geom x freq x pol)")
    print(f"  Slots         : {N_JOBS} job(s) x {N_NODES} node(s) "
          f"= {n_slots} parallel nodes")
    cores_str = str(CORES_PER_NODE) if CORES_PER_NODE is not None else "auto (--exclusive)"
    mem_str   = str(MEM_PER_NODE) if MEM_PER_NODE else "unlimited"
    time_str  = str(SLURM_TIME) if SLURM_TIME else "unlimited"
    print(f"  Per node      : {cores_str} cores, {mem_str} RAM, "
          f"{time_str} walltime")
    if peaks:
        print(f"  Unit peak RAM : {min(peaks):.2f}-{max(peaks):.2f} GB "
              f"estimated (incl. {MEMORY_SAFETY:g}x safety)")
        import rcs_solver as _solver
        ceiling = (
            float(MAX_SOLVE_GB) if MAX_SOLVE_GB
            else _solver._solve_memory_limit_gb()
        )
        source = ("MAX_SOLVE_GB" if MAX_SOLVE_GB
                  else f"{_solver._detect_available_gb():.0f} GB detected here")
        print(f"  Solve ceiling : {ceiling:.1f} GB per solve ({source})")
        if max(peaks) > ceiling:
            print("  [warn] the heaviest planned unit exceeds that ceiling. It "
                  "is evaluated on the COMPUTE node, not here, so this may be "
                  "fine -- but if the node cannot report its own memory the "
                  "ceiling falls back to 32 GB. Set MAX_SOLVE_GB, or give "
                  "MEM_PER_NODE an explicit size, to be sure.")
    idle = int(summary.get("idle_slots", 0))
    print(f"  Plan balance  : {summary['imbalance']:.2f}x the best any schedule "
          f"could do (1.00 = optimal; stealing absorbs the rest)")
    if idle:
        print(f"                  {idle} of {n_slots} slot(s) have no planned "
              "work -- fewer units than nodes, so those tasks exit at once")
    if not MESH_CERTIFICATION:
        print("  Certification : OFF -- base mesh only, typically ~3x faster")
        print("                  Downstream use remains enabled; the user owns the")
        print("                  mesh-resolution decision.")
    print(f"  Slurm scripts : {len(slurm_paths)} files in {run_dir}")

    if not SUBMIT:
        print("\n  SUBMIT=False -- submit manually with:")
        for sp in slurm_paths:
            print(f"    sbatch {sp}")
        return

    if shutil.which("sbatch") is None:
        print("\n  [warn] sbatch not on PATH. Submit manually:")
        for sp in slurm_paths:
            print(f"    sbatch {sp}")
        return

    for sp in slurm_paths:
        print(f"\n  Submitting: sbatch {sp.name}")
        res = subprocess.run(
            [_SBATCH, str(sp)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if res.returncode != 0:
            sys.exit(f"sbatch failed (exit {res.returncode}):\n"
                     f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}")
        print(f"  {res.stdout.strip()}")

    print(f"\nMonitor with:  squeue -u $USER")
    print(f"Outputs in:    {run_dir}/results/")


# --- worker mode (invoked by SLURM) ----------------------------------------

def _read_schedule(run_dir):
    # type: (Path) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]
    """Return (cost, peak_gb, slot), failing closed without a valid plan."""

    path = run_dir / "schedule.json"
    if not path.is_file():
        raise RuntimeError(
            f"Missing {path}; refusing to run without per-unit memory "
            "reservations. Regenerate the submission."
        )
    try:
        schedule = json.loads(path.read_text())
        records = schedule["units"]
        if not isinstance(records, list) or not records:
            raise ValueError("unit list is empty")
        for record in records:
            if float(record["peak_gb"]) <= 0.0:
                raise ValueError(
                    f"unit {record.get('unit', '<unknown>')} has no positive "
                    "memory reservation"
                )
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Unreadable or incomplete {path}; refusing to run without "
            f"per-unit memory reservations: {exc}"
        ) from exc
    costs = {str(r["unit"]): float(r.get("cost", 1.0)) for r in records}
    peaks = {str(r["unit"]): float(r.get("peak_gb", 0.0)) for r in records}
    slots = {str(r["unit"]): int(r.get("slot", 0)) for r in records}
    return costs, peaks, slots


def _planned_names(units, slots, slot, n_slots):
    # type: (List[Dict[str, Any]], Dict[str, int], int, int) -> set
    """Names of the units this slot owns in the submit-time plan.

    The worker refuses a missing or incomplete schedule before reaching here.
    """

    return {
        _unit_name(u) for u in units if slots.get(_unit_name(u), -1) == slot
    }


def _ordered_candidates(units, costs, slots, slot, n_slots):
    # type: (List[Dict[str, Any]], Dict[str, float], Dict[str, int], int, int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
    """(this slot's planned units, everyone else's), each dearest first.

    Returned as two lists, not one, and that separation matters. They used to
    be concatenated, and the dispatcher fills its pool from the head of
    whatever it is given -- so a task whose pool was wider than its own share
    reached straight past it into other tasks' work on the very first fill.
    With a 96-core node and a 40-unit sweep that meant the first task to start
    claimed the entire run and the other nine exited having written nothing.

    Stealing is for when a task has finished its own share, not for the moment
    it starts.
    """

    def _key(unit):
        name = _unit_name(unit)
        return (-costs.get(name, 1.0), name)

    mine_names = _planned_names(units, slots, slot, n_slots)
    mine = [u for u in units if _unit_name(u) in mine_names]
    others = [u for u in units if _unit_name(u) not in mine_names]
    return sorted(mine, key=_key), sorted(others, key=_key)


def _unit_assembly_threads(cores, pool_size, budget_gb, peak_gb):
    # type: (int, int, float, float) -> int
    """Thread count/CPU reservation derived from this unit's own footprint."""

    return hpc_scheduler.assembly_threads_for_unit(
        cores, pool_size, budget_gb, peak_gb, configured=ASSEMBLY_THREADS
    )


def worker(run_dir_str, submission_index, task_index):
    # type: (str, int, int) -> None
    hpc_scheduler.pin_blas_threads(BLAS_THREADS_PER_WORKER)
    hpc_scheduler.install_fingerprint_cache()

    run_dir  = Path(run_dir_str).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    solver_config = manifest["solver_config"]
    # Everything a unit needs from the manifest, resolved once. Workers used to
    # re-read and re-parse the whole manifest -- which carries every unit's
    # record -- inside every unit, making a node's parsing cost quadratic in
    # the size of the sweep.
    context = {
        "run_id": manifest["run_id"],
        "solver_source_sha256": manifest["solver_source_sha256"],
        "runtime_environment_sha256": manifest["runtime_environment_sha256"],
        "solver_source_inventory": manifest.get("solver_source_inventory") or {},
        "run_solve_spec_sha256": manifest_solve_spec_fingerprint(manifest),
        "solver_config_sha256": stable_json_fingerprint(solver_config),
        "geometry_units": solver_config["geometry_units"],
        "solver_method": solver_config["solver_method"],
        "cfie_alpha": solver_config["cfie_alpha"],
        "max_panels": solver_config["max_panels"],
        "mesh_convergence_policy": solver_config["mesh_convergence_policy"],
        "azimuths_deg": list(manifest["azimuths_deg"]),
        "angular_grid_sha256": stable_json_fingerprint(
            [float(value) for value in manifest["azimuths_deg"]]
        ),
        # Older manifests predate the switch and were always certified.
        "mesh_certification": bool(solver_config.get("mesh_certification", True)),
    }
    _verify_run_provenance(context)

    units   = manifest["units"]
    n_nodes = int(manifest.get("n_nodes", 1))
    n_jobs  = int(manifest.get("n_jobs", 1))
    n_slots = max(1, n_nodes * n_jobs)
    slot    = int(submission_index) * n_nodes + int(task_index)

    costs, peaks, slots = _read_schedule(run_dir)
    planned_units, steal_units = _ordered_candidates(
        units, costs, slots, slot, n_slots
    )
    candidates = planned_units + steal_units
    mine = _planned_names(units, slots, slot, n_slots)
    planned = len(planned_units)

    cores = hpc_scheduler.detect_cores()
    memory_gb = hpc_scheduler.detect_memory_gb()
    budget_gb = max(1.0, memory_gb * float(MEMORY_HEADROOM))
    worker_cap = cores if MAX_WORKERS_PER_NODE is None else max(1, int(MAX_WORKERS_PER_NODE))
    # Concurrency is sized from this task's OWN share, not from the whole
    # sweep. Sizing it from the total let one task claim every unit before the
    # others had started; it also left each solve with a sliver of the node
    # when the run was smaller than the cluster.
    pool_size = max(1, min(cores, worker_cap, max(1, planned)))
    heaviest = max(
        (peaks.get(_unit_name(u), 0.0) for u in planned_units), default=0.0
    )
    heaviest_concurrency = (
        pool_size if heaviest <= 0.0
        else max(1, min(pool_size, int(budget_gb // heaviest)))
    )
    planned_thread_counts = [
        _unit_assembly_threads(
            cores, pool_size, budget_gb, peaks.get(_unit_name(unit), 0.0)
        )
        for unit in planned_units
    ] or [1]
    min_threads = min(planned_thread_counts)
    max_threads = max(planned_thread_counts)
    thread_label = (
        str(min_threads) if min_threads == max_threads
        else f"{min_threads}-{max_threads} dynamic"
    )

    print("=" * 70)
    print(f"  Slot {slot}/{n_slots - 1}  "
          f"(submission={submission_index}, task={task_index})")
    print(f"  Units in run   : {len(units)}   planned for this slot: {planned}"
          f"   (then {len(steal_units)} stealable)")
    print(f"  Cores detected : {cores}   pool size: {pool_size}   "
          f"(BLAS threads/worker: {BLAS_THREADS_PER_WORKER}, "
          f"assembly threads/solve: {thread_label})")
    if heaviest_concurrency < pool_size:
        print(f"  Heaviest units : {heaviest_concurrency} concurrent at "
              f"{heaviest:.1f} GB each; smaller units expand dynamically")
    print(f"  Memory         : {memory_gb:.1f} GB allocated, "
          f"{budget_gb:.1f} GB schedulable")
    print("=" * 70, flush=True)

    if not candidates:
        print("  Nothing to do.")
        return

    # Parse each distinct geometry once, before the pool forks, so workers
    # inherit the snapshots instead of unpickling one per unit.
    for unit in candidates:
        path = Path(unit["geometry"])
        if not path.is_file():
            sys.exit(f"Geometry missing on compute node: {path}")
        _load_snapshot(str(path))
    # Import the solver in the parent for the same reason: a forked worker
    # inherits it, so replacing a worker costs a fork rather than a re-import
    # of numpy, SciPy, and an 8 000-line module. Importing it only inside the
    # child (as this used to) meant paying that on every single unit.
    import rcs_solver  # noqa: F401
    import grim_io     # noqa: F401

    done = {p.name for p in (run_dir / "results").rglob("*.grim")}
    broker = hpc_scheduler.ClaimBroker(
        run_dir / "claims", stale_seconds=float(CLAIM_STALE_SECONDS)
    )
    broker.start_heartbeat()

    counters = {"written": 0, "skipped": 0, "failed": 0, "passed": 0}
    started = time.time()
    total = len(candidates)

    def _prepare(unit):
        name = _unit_name(unit)
        peak_gb = peaks.get(name, 0.0)
        assembly_threads = _unit_assembly_threads(
            cores, pool_size, budget_gb, peak_gb
        )
        dispatch = (
            name, peak_gb,
            (_solve_and_export_star,
             ((unit, context, str(run_dir), assembly_threads),)),
        )
        if name in done:
            # An already-written result is dispatched, not skipped outright, so
            # its attestation is verified before the run is called complete --
            # that check is what makes reusing an interrupted run safe. Only
            # the slot that owns the unit does it, so across the whole run each
            # output is verified exactly once and no claim is needed (the check
            # is read-only and the result is already final).
            if name not in mine:
                counters["passed"] += 1
                return None
            return dispatch
        if not broker.try_claim(name):
            counters["passed"] += 1
            return None
        return dispatch

    def _finished():
        return counters["written"] + counters["skipped"] + counters["failed"]

    def _on_result(name, payload):
        kind, first, _second = payload
        if kind == "ok":
            if first == "skipped":
                counters["skipped"] += 1
            else:
                counters["written"] += 1
            broker.release(name)
            print(f"  [{_finished():4d}/{total}] {first:7s}  {name}", flush=True)
        else:
            counters["failed"] += 1
            # Hand the unit back so another task (or a later resubmission) can
            # retry it; a claim left behind would look busy until it went stale.
            broker.abandon(name)
            print(f"  [{_finished():4d}/{total}] FAILED   {name}", flush=True)
            for line in str(first).rstrip().splitlines():
                print(f"      {line}", flush=True)

    def _on_error(name, exc):
        counters["failed"] += 1
        broker.abandon(name)
        print(f"  [{_finished():4d}/{total}] FAILED (dispatch) {name}: {exc!r}",
              flush=True)

    with Pool(
        processes=pool_size,
        initializer=_pool_initializer,
        initargs=(int(BLAS_THREADS_PER_WORKER),),
        maxtasksperchild=int(TASKS_PER_CHILD),
    ) as pool:
        dispatcher = hpc_scheduler.MemoryAwareDispatcher(
            pool, budget_gb=budget_gb, max_concurrent=pool_size,
            cpu_budget=cores,
        )

        def _resources(unit):
            peak_gb = peaks.get(_unit_name(unit), 0.0)
            return (
                peak_gb,
                _unit_assembly_threads(
                    cores, pool_size, budget_gb, peak_gb
                ),
            )
        try:
            # Own share first. Only when it is finished does this task reach
            # for anyone else's, so a fast starter cannot swallow the run.
            dispatcher.run(
                planned_units, _prepare, _on_result, _on_error, _resources
            )
            if steal_units:
                dispatcher.run(
                    steal_units, _prepare, _on_result, _on_error, _resources
                )
        finally:
            broker.stop_heartbeat()

    elapsed = time.time() - started
    print(f"\n  Slot complete. wrote={counters['written']}, "
          f"skipped={counters['skipped']}, failed={counters['failed']}, "
          f"left to other tasks={counters['passed']}.  {elapsed:.1f} s elapsed.")
    if counters["failed"]:
        raise SystemExit(1)


# --- entry point -----------------------------------------------------------

def main():
    # type: () -> None
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "--worker", nargs=3,
        metavar=("RUN_DIR", "SUBMISSION_INDEX", "TASK_INDEX"),
        help="Internal: join a run as one array task. Invoked by SLURM.",
    )
    args = ap.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]), int(args.worker[2]))
    else:
        submit()


if __name__ == "__main__":
    main()
