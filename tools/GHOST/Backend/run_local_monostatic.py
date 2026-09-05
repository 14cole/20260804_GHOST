#!/usr/bin/env python3
"""Local monostatic RCS sweep -- run_hpc_monostatic.py without SLURM.

One .grim per (geometry, frequency) unit is written to
<OUTPUT_DIR>/run_YYYYMMDD_HHMMSS/results/{FRD,OPN}/ as soon as that unit
finishes, named "<FREQ:.3f>GHz_<geometry_stem>.grim". Every file contains the
canonical VV and HH channels. The role folders can be passed directly to the
downstream subtraction tool.
each file carries its own source/runtime/input attestation inside the artifact,
so a resumed run verifies what it reuses without a sidecar per output.

Scheduling matches the HPC path. Units are costed from the mesh the solver will
actually build and run dearest-first, and concurrent solves are admitted
against a memory budget rather than filling every core regardless of unit size
-- one 40 GB geometry does not get eight copies of itself started on a 32 GB
laptop.

Edit the CONFIG block and run:

    python run_local_monostatic.py
"""

import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hpc_scheduler
import workflow_provenance as _workflow_provenance
from solver_quality import accuracy_target_policy
from workflow_provenance import (
    backend_source_fingerprint,
    backend_source_inventory,
    describe_source_mismatch,
    embed_output_attestation,
    manifest_solve_spec_fingerprint,
    runtime_environment_fingerprint,
    stable_json_fingerprint,
    unit_solve_spec_fingerprint,
    verify_embedded_attestation,
)

# ===============================================================================
# CONFIG
# ===============================================================================

# Geometry folders. Every *.geo file under these (recursively) is included.
FRD_DIR = "geometries/FRD"
OPN_DIR = "geometries/OPN"

# Sweep.
FREQUENCIES_GHZ = [2.0, 4.0, 6.0, 8.0, 10.0]
AZIMUTHS_DEG    = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]

# Output root. A new run_YYYYMMDD_HHMMSS/ subfolder is created inside.
OUTPUT_DIR = "rcs_runs"

# Hard ceiling on concurrent solves. None -> max(1, cpu_count() - 1). The
# memory budget below is usually the binding constraint, so this rarely needs
# setting.
WORKERS = None

# Solver knobs. Production 2-D runs always use condition-reporting dense LU
# and co-solve both physical channels (VV/TE and HH/TM).
GEOMETRY_UNITS          = "inches"       # "inches" or "meters"
MAX_PANELS              = 50_000
BLAS_THREADS_PER_WORKER = 1

# Mesh-convergence certification. True solves every unit twice -- the requested
# mesh and one refined by the policy's fine_factor -- and publishes the fine
# result only if the two agree. That second solve is where most of the wall
# clock and all of the peak memory go, because cost scales with the square of
# the node count: turning it off is about 3x faster per unit and roughly halves
# the memory, so more units fit in RAM at once as well.
#
# With this off the algebraic quality gate still runs, but no base/fine mesh
# comparison is performed.  The choice is recorded for provenance and does not
# prevent downstream viewing, combination, or subtraction.
MESH_CERTIFICATION      = True
ACCURACY_TARGET         = "standard"     # "standard" | "tight"; mesh comparison limits
LU_PRECISION            = "double"       # "double" | "mixed"; CPU LU with refinement

# --- Memory admission ------------------------------------------------------
# A local machine has far less RAM than a compute node and is usually running
# other things, so the same guard the cluster path uses matters more here, not
# less. Concurrent solves are admitted while their estimated peaks fit the
# budget; one unit is always admitted, so a solve larger than the whole budget
# still runs (and fails loudly from the solver's own gate) instead of hanging.
MEMORY_HEADROOM = 0.75            # fraction of detected RAM the scheduler may
                                  # reserve for solves. Lower than the cluster
                                  # default of 0.85: a workstation has a
                                  # desktop, a browser, and a page cache to
                                  # leave room for.
MEMORY_SAFETY   = 1.35            # multiplier on the solver's own dense-storage
                                  # estimate, covering allocator slack and the
                                  # transient copies a factorization makes.
MAX_SOLVE_GB    = None            # Hard ceiling on ONE solve's estimated
                                  # footprint (GHOST_MAX_SOLVE_GB). None =
                                  # derive it from detected RAM (0.9 x detected,
                                  # floored at 32 GB). Set it to run something
                                  # deliberately larger than this machine's RAM
                                  # against swap, or to refuse earlier.

# Threads each solve may use inside the boundary-operator assembly. "auto"
# gives every concurrent solve an equal share of the cores -- which is what a
# local run usually wants, since it typically has fewer units in flight than
# the machine has cores.
ASSEMBLY_THREADS        = "auto"         # "auto", or an integer >= 1

# Pool worker lifetime, in units, so allocator growth from a big solve cannot
# accumulate across a long sweep. The solver is imported in the parent, so a
# respawn costs a fork rather than a re-import of numpy, SciPy, and the solver.
TASKS_PER_CHILD = 4

GEOMETRY_EXTS = (".geo",)

# ===============================================================================

MANIFEST_SCHEMA = "ghost.local.2d-run.v3"
OUTPUT_POLARIZATIONS = ("VV", "HH")

# Parsed geometry snapshots, filled in the parent before the pool forks so
# workers inherit them instead of unpickling one per unit.
_SNAPSHOT_CACHE = {}  # type: Dict[str, Tuple[Dict[str, Any], str]]


def _solver_source_records() -> 'Tuple[str, Dict[str, str]]':
    backend_dir = str(Path(_workflow_provenance.__file__).resolve().parent)
    return backend_dir, {"driver_configured.py": str(Path(__file__).resolve())}


def _solver_source_fingerprint() -> 'str':
    backend_dir, extra = _solver_source_records()
    return backend_source_fingerprint(backend_dir, extra)


def _solver_source_inventory() -> 'Dict[str, str]':
    backend_dir, extra = _solver_source_records()
    return backend_source_inventory(backend_dir, extra)


def _write_json_atomic(path: 'Path', payload: 'Dict[str, Any]') -> 'None':
    temporary = Path(str(path) + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _verify_run_provenance(context: 'Dict[str, Any]') -> 'None':
    """Re-check that the solver source and numerical runtime still match the run.

    Called around every unit. The file hashes underneath come from
    `hpc_scheduler.install_fingerprint_cache`, so a repeat check is a stat per
    backend file rather than a full re-read, and the cache expires on a timer
    so full re-reads keep happening inside long-lived workers.
    """

    if _solver_source_fingerprint() != context.get("solver_source_sha256"):
        detail = describe_source_mismatch(
            context.get("solver_source_inventory") or {},
            _solver_source_inventory(),
        )
        raise RuntimeError(
            "Local-run solver source/native artifacts changed; no mixed-state "
            f"field will be written or reused. ({detail})"
        )
    if runtime_environment_fingerprint() != context.get(
        "runtime_environment_sha256"
    ):
        raise RuntimeError(
            "Local-run Python/platform/NumPy/SciPy/BLAS runtime changed."
        )


def _unit_attestation_fields(
    context: 'Dict[str, Any]',
    unit: 'Dict[str, Any]',
) -> 'Dict[str, Any]':
    return {
        "run_id": str(context["run_id"]),
        "solver_source_sha256": str(context["solver_source_sha256"]),
        "runtime_environment_sha256":
            str(context["runtime_environment_sha256"]),
        "geometry_input_sha256": str(unit["geometry_input_sha256"]),
        "run_solve_spec_sha256": str(context["run_solve_spec_sha256"]),
        "unit_solve_spec_sha256": unit_solve_spec_fingerprint(unit),
        "solver_config_sha256": str(context["solver_config_sha256"]),
        "angular_grid_kind": "azimuths_deg",
        # The grid is a run-level property, so it is bound by hash here and
        # stored once in the manifest instead of being repeated in every unit
        # record and every attestation.
        "angular_grid_sha256": str(context["angular_grid_sha256"]),
        "polarizations": list(OUTPUT_POLARIZATIONS),
        "frequency_ghz": float(unit["frequency_ghz"]),
    }


def _verify_unit_input(
    unit: 'Dict[str, Any]',
    context: 'Dict[str, Any]',
) -> 'None':
    from feature_sum import geometry_input_fingerprint
    current = geometry_input_fingerprint(
        str(unit["geometry"]), str(context["geometry_units"])
    )
    if current != unit.get("geometry_input_sha256"):
        raise RuntimeError(
            f"Geometry/material input changed during the local run: "
            f"{unit['geometry']}"
        )


def _discover_geometries() -> 'List[Path]':
    found: 'List[Path]' = []
    seen: 'set' = set()
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


def _geometry_role(path: 'Path') -> 'str':
    resolved = Path(path).resolve()
    for role, directory in (("FRD", FRD_DIR), ("OPN", OPN_DIR)):
        try:
            resolved.relative_to(Path(directory).resolve())
            return role
        except ValueError:
            continue
    raise ValueError(f"Geometry is outside configured input folders: {path}")


def _unit_name(unit: 'Dict[str, Any]') -> 'str':
    return (f"{float(unit['frequency_ghz']):.3f}GHz_"
            f"{unit['geometry_stem']}.grim")


def _unit_output_path(results_dir: 'Path', unit: 'Dict[str, Any]') -> 'Path':
    role = str(unit.get("role", "")).strip().upper()
    folder = results_dir / role if role else results_dir
    return folder / _unit_name(unit)


def _load_snapshot(geometry_path: 'str') -> 'Tuple[Dict[str, Any], str]':
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


def _pool_initializer(blas_threads: 'int') -> 'None':
    hpc_scheduler.pin_blas_threads(blas_threads)
    hpc_scheduler.install_fingerprint_cache()
    import rcs_solver

    rcs_solver.set_assembly_threads(1)


def _solve_and_export(
    unit: 'Dict[str, Any]',
    context: 'Dict[str, Any]',
    results_dir_str: 'str',
) -> 'Tuple[str, str]':
    """Pool-worker entry point: solve one unit, export .grim. Idempotent."""

    results_dir = Path(results_dir_str)
    out_path = _unit_output_path(results_dir, unit)
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
        geometry_units=context["geometry_units"],
        material_base_dir=material_base,
        max_panels=context["max_panels"],
    )
    # Select precision inside each worker; context variables are process-local.
    from refined_lu import linear_precision
    with linear_precision(context.get("lu_precision", "double")):
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
        history=(f"run_local_monostatic.py pols=VV,HH "
                 f"freq={unit['frequency_ghz']}GHz"),
    )
    actual_path = str(written[0]) if written else str(out_path)
    _verify_run_provenance(context)
    _verify_unit_input(unit, context)
    return ("written", actual_path)


def _solve_and_export_star(args: 'tuple') -> 'tuple':
    """Pool entry point: unpack args and catch exceptions in-band.

    The full traceback string is returned (not just str(exc)) so a failure
    names the line it happened on rather than only its message.
    """

    unit, context, results_dir_str, assembly_threads = args
    try:
        import rcs_solver
        rcs_solver.set_assembly_threads(assembly_threads)
        status, path = _solve_and_export(unit, context, results_dir_str)
        return ("ok", status, path)
    except Exception:
        return ("err", traceback.format_exc(), "")


def _plan(
    units: 'List[Dict[str, Any]]',
    fine_factor: 'float',
    n_angles: 'int',
) -> 'Tuple[Dict[str, float], Dict[str, float]]':
    """Cost both channels and reserve their sequential peak per output unit."""

    resource_cache: 'Dict[Tuple[str, float], List[Dict[str, Any]]]' = {}
    costs: 'Dict[str, float]' = {}
    peaks: 'Dict[str, float]' = {}
    for unit in units:
        key = (str(unit["geometry"]), float(unit["frequency_ghz"]))
        if key not in resource_cache:
            resource_cache[key] = [
                hpc_scheduler.predict_2d_resources(
                    key[0], key[1], polarization, GEOMETRY_UNITS,
                    MAX_PANELS, fine_factor=fine_factor,
                    n_angles=n_angles, safety=float(MEMORY_SAFETY),
                )
                for polarization in ("TM", "TE")
            ]
        plans = resource_cache[key]
        name = _unit_name(unit)
        costs[name] = sum(
            hpc_scheduler.unit_cost(
                int(planned["nodes"]), n_angles, fine_factor,
                fine_nodes=int(planned["fine_nodes"]),
                system_dofs=int(planned["base_system_dofs"]),
                fine_system_dofs=int(planned["fine_system_dofs"]),
                operator_matrices=int(planned["base_operator_matrices"]),
                fine_operator_matrices=int(planned["fine_operator_matrices"]),
            )
            for planned in plans
        )
        # The canonical API solves the two formulations sequentially while
        # retaining only their compact sample arrays, so peak dense storage is
        # the larger channel rather than their sum.
        peaks[name] = max(float(planned["peak_gb"]) for planned in plans)
    return costs, peaks


def _unit_assembly_threads(
    cores: 'int', pool_size: 'int', budget_gb: 'float', peak_gb: 'float'
) -> 'int':
    """Thread count/CPU reservation derived from this unit's own footprint."""

    return hpc_scheduler.assembly_threads_for_unit(
        cores, pool_size, budget_gb, peak_gb, configured=ASSEMBLY_THREADS
    )


def _validate_config() -> 'Tuple[List[float], List[float]]':
    if ACCURACY_TARGET not in ("standard", "tight"):
        sys.exit("ERROR: ACCURACY_TARGET must be 'standard' or 'tight'.")
    if LU_PRECISION not in ("double", "mixed"):
        sys.exit("ERROR: LU_PRECISION must be 'double' or 'mixed'.")
    if not FREQUENCIES_GHZ: sys.exit("ERROR: FREQUENCIES_GHZ is empty.")
    if not AZIMUTHS_DEG:    sys.exit("ERROR: AZIMUTHS_DEG is empty.")
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
    if str(GEOMETRY_UNITS).strip().lower() not in {"inches", "meters"}:
        sys.exit("ERROR: GEOMETRY_UNITS must be 'inches' or 'meters'.")
    if int(MAX_PANELS) < 1 or int(BLAS_THREADS_PER_WORKER) < 1:
        sys.exit("ERROR: MAX_PANELS and BLAS_THREADS_PER_WORKER must be >= 1.")
    if not 0.0 < float(MEMORY_HEADROOM) <= 1.0:
        sys.exit("ERROR: MEMORY_HEADROOM must be in (0, 1].")
    if float(MEMORY_SAFETY) < 1.0:
        sys.exit("ERROR: MEMORY_SAFETY must be >= 1.")
    if MAX_SOLVE_GB is not None and float(MAX_SOLVE_GB) <= 0.0:
        sys.exit("ERROR: MAX_SOLVE_GB must be positive or None.")
    if ASSEMBLY_THREADS != "auto" and int(ASSEMBLY_THREADS) < 1:
        sys.exit("ERROR: ASSEMBLY_THREADS must be 'auto' or an integer >= 1.")
    if int(TASKS_PER_CHILD) < 1:
        sys.exit("ERROR: TASKS_PER_CHILD must be >= 1.")
    if WORKERS is not None and int(WORKERS) < 1:
        sys.exit("ERROR: WORKERS must be a positive integer or None.")
    return frequencies, azimuths


def main() -> 'None':
    frequencies, azimuths = _validate_config()
    if MAX_SOLVE_GB:
        # Read by the solver's own memory gate, in this process and every
        # forked worker.
        os.environ["GHOST_MAX_SOLVE_GB"] = f"{float(MAX_SOLVE_GB):g}"
    hpc_scheduler.pin_blas_threads(int(BLAS_THREADS_PER_WORKER))
    hpc_scheduler.install_fingerprint_cache()

    geometries = _discover_geometries()
    if not geometries:
        sys.exit("ERROR: no geometry files (*.geo) found under FRD_DIR or OPN_DIR.")
    stems = [geometry.stem for geometry in geometries]
    if len(stems) != len(set(stems)):
        sys.exit(
            "ERROR: geometry stems must be unique; output names would "
            "otherwise collide."
        )

    units: 'List[Dict[str, Any]]' = []
    from feature_sum import geometry_input_fingerprint
    for geom in geometries:
        input_fingerprint = geometry_input_fingerprint(
            str(geom), GEOMETRY_UNITS
        )
        for f in frequencies:
            # The azimuth grid is deliberately NOT repeated per unit: it is
            # identical for every unit in the run and is bound by hash.
            units.append({
                "geometry":      str(geom.resolve()),
                "geometry_stem": geom.stem,
                "geometry_input_sha256": input_fingerprint,
                "role":          _geometry_role(geom),
                "polarizations": list(OUTPUT_POLARIZATIONS),
                "frequency_ghz": float(f),
            })

    mesh_policy = accuracy_target_policy(ACCURACY_TARGET)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir     = Path(OUTPUT_DIR).resolve() / run_id
    results_dir = run_dir / "results"
    run_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir()
    (results_dir / "FRD").mkdir()
    (results_dir / "OPN").mkdir()
    solver_config = {
        "geometry_units": GEOMETRY_UNITS,
        "linear_solver": "dense_lu",
        "polarizations": list(OUTPUT_POLARIZATIONS),
        "max_panels": int(MAX_PANELS),
        "blas_threads_per_worker": int(BLAS_THREADS_PER_WORKER),
        "mesh_convergence_policy": mesh_policy,
        "accuracy_target": ACCURACY_TARGET,
        "lu_precision": LU_PRECISION,
        "mesh_certification": bool(MESH_CERTIFICATION),
    }
    manifest: 'Dict[str, Any]' = {
        "schema": MANIFEST_SCHEMA,
        "status": "running",
        "run_id": run_id,
        "created": datetime.now().isoformat(),
        "frequencies_ghz": frequencies,
        "azimuths_deg": azimuths,
        "polarizations": list(OUTPUT_POLARIZATIONS),
        "n_units": len(units),
        "solver_source_sha256": _solver_source_fingerprint(),
        # Per-file hashes behind that fingerprint, so a later mismatch can say
        # which file moved instead of only that one did.
        "solver_source_inventory": _solver_source_inventory(),
        "runtime_environment_sha256": runtime_environment_fingerprint(),
        "solver_config": solver_config,
        "units": units,
    }
    manifest_path = run_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)

    # Everything a unit needs from the manifest, resolved once in the parent.
    # Workers used to re-read and re-parse the whole manifest -- which carries
    # every unit's record -- inside every unit, making the parsing cost
    # quadratic in the size of the sweep.
    context = {
        "run_id": run_id,
        "solver_source_sha256": manifest["solver_source_sha256"],
        "solver_source_inventory": manifest["solver_source_inventory"],
        "runtime_environment_sha256": manifest["runtime_environment_sha256"],
        "run_solve_spec_sha256": manifest_solve_spec_fingerprint(manifest),
        "solver_config_sha256": stable_json_fingerprint(solver_config),
        "geometry_units": GEOMETRY_UNITS,
        "max_panels": int(MAX_PANELS),
        "mesh_convergence_policy": mesh_policy,
        "lu_precision": LU_PRECISION,
        "mesh_certification": bool(MESH_CERTIFICATION),
        "azimuths_deg": azimuths,
        "angular_grid_sha256": stable_json_fingerprint(
            azimuths
        ),
    }

    # Cost every unit from the real mesh, then run dearest-first: a frequency
    # sweep's cost spread is large (it grows like the square of the node
    # count), and starting with the cheap end leaves the expensive tail with
    # nothing to overlap against.
    fine_factor = (
        float(mesh_policy["fine_factor"]) if MESH_CERTIFICATION else 1.0
    )
    costs, peaks = _plan(units, fine_factor, len(azimuths))
    # Sorted into a new list, never in place: `units` is the same object the
    # manifest holds, and reordering it after the run fingerprint was taken
    # would make the manifest on disk hash differently from what every
    # attestation recorded.
    ordered = sorted(
        units, key=lambda u: (-costs.get(_unit_name(u), 1.0), _unit_name(u))
    )

    cores = hpc_scheduler.detect_cores()
    memory_gb = hpc_scheduler.detect_memory_gb()
    budget_gb = max(1.0, memory_gb * float(MEMORY_HEADROOM))
    worker_cap = max(1, cores - 1) if WORKERS is None else int(WORKERS)
    pool_size = max(1, min(cores, worker_cap, len(ordered)))
    heaviest = max((peaks.get(_unit_name(u), 0.0) for u in ordered), default=0.0)
    heaviest_concurrency = (
        pool_size if heaviest <= 0.0
        else max(1, min(pool_size, int(budget_gb // heaviest)))
    )
    unit_thread_counts = [
        _unit_assembly_threads(
            cores, pool_size, budget_gb, peaks.get(_unit_name(unit), 0.0)
        )
        for unit in ordered
    ] or [1]
    min_threads = min(unit_thread_counts)
    max_threads = max(unit_thread_counts)
    thread_label = (
        str(min_threads) if min_threads == max_threads
        else f"{min_threads}-{max_threads} dynamic"
    )

    print("=" * 70)
    print("Local monostatic RCS sweep")
    print("=" * 70)
    print(f"  Run dir       : {run_dir}")
    print(f"  Geometries    : {len(geometries)}")
    print("  Polarizations : VV, HH (co-solved)")
    print(f"  Frequencies   : {len(FREQUENCIES_GHZ)}  "
          f"({min(FREQUENCIES_GHZ):g}-{max(FREQUENCIES_GHZ):g} GHz)")
    print(f"  Azimuths      : {len(AZIMUTHS_DEG)}")
    print(f"  Units total   : {len(ordered)}  (geometry x frequency)")
    print(f"  Mesh check    : {'base + fine comparison' if MESH_CERTIFICATION else 'base only (no mesh comparison)'}")
    print(f"  Workers       : {pool_size} of {cores} cpus  "
          f"(BLAS threads/worker: {BLAS_THREADS_PER_WORKER}, "
          f"assembly threads/solve: {thread_label})")
    if heaviest_concurrency < pool_size:
        print(f"  Heaviest units: {heaviest_concurrency} concurrent at "
              f"{heaviest:.1f} GB each; smaller units expand dynamically")
    print(f"  Memory        : {memory_gb:.1f} GB detected, "
          f"{budget_gb:.1f} GB schedulable")
    if not MESH_CERTIFICATION:
        print("  Mesh comparison is off: downstream use remains enabled; the")
        print("  user owns the mesh-resolution decision.")
    print("=" * 70, flush=True)

    # Parse each distinct geometry once, before the pool forks, so workers
    # inherit the snapshots instead of unpickling one per unit. Import the
    # solver here for the same reason: replacing a worker then costs a fork
    # rather than a re-import of numpy, SciPy, and the solver module.
    for unit in ordered:
        _load_snapshot(str(unit["geometry"]))
    import rcs_solver  # noqa: F401
    import grim_io     # noqa: F401

    counters = {"written": 0, "skipped": 0, "failed": 0}
    started = time.time()
    total = len(ordered)

    def _prepare(unit):
        name = _unit_name(unit)
        peak_gb = peaks.get(name, 0.0)
        assembly_threads = _unit_assembly_threads(
            cores, pool_size, budget_gb, peak_gb
        )
        return (
            name, peak_gb,
            (_solve_and_export_star,
             ((unit, context, str(results_dir), assembly_threads),)),
        )

    def _finished():
        return counters["written"] + counters["skipped"] + counters["failed"]

    def _on_result(name, payload):
        kind, first, _second = payload
        if kind == "ok":
            counters["skipped" if first == "skipped" else "written"] += 1
            print(f"  [{_finished():4d}/{total}] {first:7s}  {name}", flush=True)
        else:
            counters["failed"] += 1
            print(f"  [{_finished():4d}/{total}] FAILED   {name}", flush=True)
            for line in str(first).rstrip().splitlines():
                print(f"      {line}", flush=True)

    def _on_error(name, exc):
        counters["failed"] += 1
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

        dispatcher.run(
            ordered, _prepare, _on_result, _on_error, _resources
        )

    elapsed = time.time() - started
    print(f"\n  Done. wrote={counters['written']}, "
          f"skipped={counters['skipped']}, failed={counters['failed']}.  "
          f"{elapsed:.1f} s elapsed.")
    print(f"  Outputs: {results_dir}/")
    manifest["status"] = "failed" if counters["failed"] else "complete"
    _write_json_atomic(manifest_path, manifest)
    if counters["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
