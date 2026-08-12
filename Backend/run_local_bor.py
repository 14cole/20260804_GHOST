#!/usr/bin/env python3
"""Local BoR monostatic RCS aspect sweep -- run_hpc_bor_monostatic.py without SLURM.

Each unit is a true 3-D (dBsm) body-of-revolution solve over the ASPECT angles
(degrees from the +z rotation axis: 0 = nose-on, 90 = broadside, 180 = tail-on).
Geometries are .geo half-profiles: x = rho (>= 0), y = z (rotation axis), drawn
from the +z axis end to the -z axis end (see bor_dispatch.py).

One .grim per (geometry, frequency, polarization) unit is written to
<OUTPUT_DIR>/run_YYYYMMDD_HHMMSS/results/ as soon as that unit finishes, named
"<POL>_<FREQ:.3f>GHz_<geometry_stem>.grim". Nothing else lands in results/:
each file carries its own source/runtime/input attestation inside the artifact,
so a resumed run verifies what it reuses without a sidecar per output.

Scheduling matches the HPC path. A BoR unit's cost grows roughly as the fourth
power of frequency (elements^3 x modes), so a frequency sweep is badly
lopsided; units are costed and run dearest-first, and concurrent solves are
admitted against a memory budget instead of filling every core.

Edit the CONFIG block and run:

    python run_local_bor.py
"""

import json
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

GEOMETRY_DIRS = ["geometries/BOR"]      # every *.geo under these, recursively

FREQUENCIES_GHZ = [1.0, 2.0, 4.0]
ASPECTS_DEG     = [float(a) for a in range(0, 181, 5)]
POLARIZATIONS   = ["VV", "HH"]

OUTPUT_DIR = "rcs_runs_bor"

# Hard ceiling on units in flight. None -> cores // WORKERS_PER_UNIT, since
# each unit already runs WORKERS_PER_UNIT threads for its mode sweep and
# streaming tiles. The memory budget below is usually the binding constraint.
WORKERS = None

# --- Solver knobs (mirror bor_dispatch.solve_monostatic_rcs_bor) -----------
GEOMETRY_UNITS   = "inches"       # "inches" or "meters"
CFIE_ALPHA       = 0.5            # closed PEC bodies -> CFIE
N_MODES          = None           # None = auto (adaptive truncation)
MODE_TOL         = 1e-6
MAX_ELEMENTS     = 50_000
ASSEMBLY         = "auto"         # "auto" | "tables" | "streaming"
TABLE_PRECISION  = "auto"         # "auto" | "single" | "double"
STREAM_BUDGET_GB = 8.0            # streaming-block budget held per unit; also
                                  # what the scheduler reserves for one unit
EXPAND_TO_360    = False          # mirror samples about the axis to fill the
                                  # full polar cut (exact for a BoR)
WORKERS_PER_UNIT = 4              # threads inside one BoR solve
BLAS_THREADS_PER_WORKER = 1

# --- Memory admission ------------------------------------------------------
# A local machine has far less RAM than a compute node and is usually running
# other things, so the same guard the cluster path uses matters more here, not
# less. One unit is always admitted, so a solve larger than the whole budget
# still runs (and fails loudly) instead of hanging.
MEMORY_HEADROOM = 0.75            # fraction of detected RAM the scheduler may
                                  # reserve for concurrent solves. Lower than
                                  # the cluster default of 0.85: a workstation
                                  # has a desktop and a page cache to leave
                                  # room for.

# Pool worker lifetime, in units. Lower than the 2-D default because a BoR
# unit's streaming blocks are large and worth returning to the OS promptly.
TASKS_PER_CHILD = 2

# ===============================================================================

MANIFEST_SCHEMA = "ghost.local.bor-run.v2"

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

    The file hashes underneath come from `hpc_scheduler.install_fingerprint_cache`,
    so a repeat check is a stat per backend file rather than a full re-read, and
    the cache expires on a timer so full re-reads keep happening inside
    long-lived workers.
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
        "angular_grid_kind": "aspects_deg",
        # The grid is a run-level property, so it is bound by hash here and
        # stored once in the manifest instead of being repeated in every unit
        # record and every attestation.
        "angular_grid_sha256": str(context["angular_grid_sha256"]),
        "polarization": str(unit["polarization"]),
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
    for d in GEOMETRY_DIRS:
        root = Path(d)
        if not root.is_dir():
            print(f"  [warn] dir not found: {root}", file=sys.stderr)
            continue
        for p in sorted(root.rglob("*.geo")):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                found.append(p)
    return found


def _unit_name(unit: 'Dict[str, Any]') -> 'str':
    return (f"{unit['polarization']}_{float(unit['frequency_ghz']):.3f}GHz_"
            f"{unit['geometry_stem']}.grim")


def _unit_output_path(results_dir: 'Path', unit: 'Dict[str, Any]') -> 'Path':
    return results_dir / _unit_name(unit)


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
    import bor_dispatch  # noqa: F401


def _solve_and_export(
    unit: 'Dict[str, Any]',
    context: 'Dict[str, Any]',
    results_dir_str: 'str',
) -> 'Tuple[str, str]':
    """Pool-worker entry point: solve one unit, export .grim. Idempotent."""

    results_dir = Path(results_dir_str)
    out_path = _unit_output_path(results_dir, unit)
    _verify_run_provenance(context)
    _verify_unit_input(unit, context)
    attestation = _unit_attestation_fields(context, unit)
    if out_path.exists():
        verify_embedded_attestation(str(out_path), attestation)
        return ("skipped", str(out_path))

    snapshot, material_base = _load_snapshot(str(unit["geometry"]))

    from bor_dispatch import solve_monostatic_rcs_bor
    result = solve_monostatic_rcs_bor(
        geometry_snapshot=snapshot,
        frequencies_ghz=[float(unit["frequency_ghz"])],
        elevations_deg=[float(a) for a in context["aspects_deg"]],
        polarization=unit["polarization"],
        geometry_units=context["geometry_units"],
        material_base_dir=material_base,
        cfie_alpha=context["cfie_alpha"],
        n_modes=context["n_modes"],
        mode_tol=context["mode_tol"],
        max_elements=context["max_elements"],
        workers=context["workers_per_unit"],
        table_precision=context["table_precision"],
        assembly=context["assembly"],
        stream_budget_gb=context["stream_budget_gb"],
        expand_to_360=context["expand_to_360"],
    )
    for warning in result.get("metadata", {}).get("warnings", []) or []:
        print(f"      [warn] {unit['geometry_stem']}: {warning}", flush=True)
    _verify_run_provenance(context)
    _verify_unit_input(unit, context)

    # Bind the result to its run state inside the artifact, before export, so
    # results/ holds one file per unit instead of a .grim and a sidecar.
    embed_output_attestation(result, attestation)

    from grim_io import export_result_to_grim
    written = export_result_to_grim(
        result, str(out_path),
        source_path=str(snapshot.get("source_path", "") or ""),
        history=(f"run_local_bor.py pol={unit['polarization']} "
                 f"freq={unit['frequency_ghz']}GHz"),
    )
    actual_path = str(written[0]) if written else str(out_path)
    _verify_run_provenance(context)
    _verify_unit_input(unit, context)
    return ("written", actual_path)


def _solve_and_export_star(args: 'tuple') -> 'tuple':
    """Pool entry point: unpack args and catch exceptions in-band."""

    unit, context, results_dir_str = args
    try:
        status, path = _solve_and_export(unit, context, results_dir_str)
        return ("ok", status, path)
    except Exception:
        return ("err", traceback.format_exc(), "")


def _plan(units: 'List[Dict[str, Any]]') -> 'Dict[str, float]':
    """Relative cost of every unit, read off the .geo profile.

    Deliberately coarser than the 2-D model: there is no equally cheap way to
    predict a BoR discretization exactly, and an approximate order plus
    dearest-first dispatch beats an exact plan that costs a solve to compute.
    """

    extents: 'Dict[str, Tuple[float, float]]' = {}
    costs: 'Dict[str, float]' = {}
    for unit in units:
        path = str(unit["geometry"])
        if path not in extents:
            extents[path] = hpc_scheduler.predict_bor_extent(
                path, GEOMETRY_UNITS
            )
        arc, radius = extents[path]
        costs[_unit_name(unit)] = hpc_scheduler.bor_unit_cost(
            arc, radius, float(unit["frequency_ghz"]), len(ASPECTS_DEG)
        )
    return costs


def _validate_config() -> 'List[str]':
    pols = [p.strip().upper() for p in POLARIZATIONS if p and p.strip()]
    if not pols:            sys.exit("ERROR: POLARIZATIONS is empty.")
    if not FREQUENCIES_GHZ: sys.exit("ERROR: FREQUENCIES_GHZ is empty.")
    if not ASPECTS_DEG:     sys.exit("ERROR: ASPECTS_DEG is empty.")
    if str(ASSEMBLY).strip().lower() not in {"auto", "tables", "streaming"}:
        sys.exit("ERROR: ASSEMBLY must be auto, tables, or streaming.")
    if str(TABLE_PRECISION).strip().lower() not in {"auto", "single", "double"}:
        sys.exit("ERROR: TABLE_PRECISION must be auto, single, or double.")
    if float(STREAM_BUDGET_GB) <= 0.0:
        sys.exit("ERROR: STREAM_BUDGET_GB must be positive.")
    if not 0.0 < float(MEMORY_HEADROOM) <= 1.0:
        sys.exit("ERROR: MEMORY_HEADROOM must be in (0, 1].")
    if int(WORKERS_PER_UNIT) < 1:
        sys.exit("ERROR: WORKERS_PER_UNIT must be >= 1.")
    if int(TASKS_PER_CHILD) < 1:
        sys.exit("ERROR: TASKS_PER_CHILD must be >= 1.")
    if WORKERS is not None and int(WORKERS) < 1:
        sys.exit("ERROR: WORKERS must be a positive integer or None.")
    return pols


def main() -> 'None':
    pols = _validate_config()
    hpc_scheduler.pin_blas_threads(int(BLAS_THREADS_PER_WORKER))
    hpc_scheduler.install_fingerprint_cache()

    geometries = _discover_geometries()
    if not geometries:
        sys.exit("ERROR: no geometry files (*.geo) found under GEOMETRY_DIRS.")
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
        for pol in pols:
            for f in FREQUENCIES_GHZ:
                # The aspect grid is deliberately NOT repeated per unit: it is
                # identical for every unit in the run, and carrying it here made
                # the manifest grow with (units x aspects) instead of with the
                # sweep. It lives once at manifest level and is bound into each
                # attestation by hash.
                units.append({
                    "geometry":      str(geom.resolve()),
                    "geometry_stem": geom.stem,
                    "geometry_input_sha256": input_fingerprint,
                    "polarization":  pol,
                    "frequency_ghz": float(f),
                })

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = Path(OUTPUT_DIR).resolve() / run_id
    results_dir = run_dir / "results"
    run_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir()
    solver_config = {
        "geometry_units": GEOMETRY_UNITS,
        "cfie_alpha": float(CFIE_ALPHA),
        "n_modes": N_MODES,
        "mode_tol": float(MODE_TOL),
        "max_elements": int(MAX_ELEMENTS),
        "assembly": ASSEMBLY,
        "table_precision": TABLE_PRECISION,
        "stream_budget_gb": float(STREAM_BUDGET_GB),
        "expand_to_360": bool(EXPAND_TO_360),
        "workers_per_unit": int(WORKERS_PER_UNIT),
        "blas_threads_per_worker": int(BLAS_THREADS_PER_WORKER),
    }
    manifest: 'Dict[str, Any]' = {
        "schema": MANIFEST_SCHEMA,
        "status": "running",
        "run_id": run_id,
        "created": datetime.now().isoformat(),
        "solver": "bor_mom_rcs",
        "frequencies_ghz": [float(f) for f in FREQUENCIES_GHZ],
        "aspects_deg": [float(a) for a in ASPECTS_DEG],
        "polarizations": pols,
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

    # Everything a unit needs from the manifest, resolved once in the parent
    # instead of re-read and re-parsed inside every unit.
    context = {
        "run_id": run_id,
        "solver_source_sha256": manifest["solver_source_sha256"],
        "solver_source_inventory": manifest["solver_source_inventory"],
        "runtime_environment_sha256": manifest["runtime_environment_sha256"],
        "run_solve_spec_sha256": manifest_solve_spec_fingerprint(manifest),
        "solver_config_sha256": stable_json_fingerprint(solver_config),
        "geometry_units": GEOMETRY_UNITS,
        "cfie_alpha": float(CFIE_ALPHA),
        "n_modes": N_MODES,
        "mode_tol": float(MODE_TOL),
        "max_elements": int(MAX_ELEMENTS),
        "assembly": ASSEMBLY,
        "table_precision": TABLE_PRECISION,
        "stream_budget_gb": float(STREAM_BUDGET_GB),
        "expand_to_360": bool(EXPAND_TO_360),
        "workers_per_unit": int(WORKERS_PER_UNIT),
        "aspects_deg": [float(a) for a in ASPECTS_DEG],
        "angular_grid_sha256": stable_json_fingerprint(
            [float(a) for a in ASPECTS_DEG]
        ),
    }

    # Dearest-first. Sorted into a new list, never in place: `units` is the
    # same object the manifest holds, and reordering it after the run
    # fingerprint was taken would make the manifest on disk hash differently
    # from what every attestation recorded.
    costs = _plan(units)
    ordered = sorted(
        units, key=lambda u: (-costs.get(_unit_name(u), 1.0), _unit_name(u))
    )

    cores = hpc_scheduler.detect_cores()
    memory_gb = hpc_scheduler.detect_memory_gb()
    budget_gb = max(1.0, memory_gb * float(MEMORY_HEADROOM))
    by_threads = max(1, cores // max(1, int(WORKERS_PER_UNIT)))
    worker_cap = by_threads if WORKERS is None else int(WORKERS)
    pool_size = max(1, min(by_threads, worker_cap, len(ordered)))

    print("=" * 70)
    print("Local BoR monostatic RCS aspect sweep")
    print("=" * 70)
    print(f"  Run dir       : {run_dir}")
    print(f"  Geometries    : {len(geometries)}")
    print(f"  Polarizations : {', '.join(pols)}")
    print(f"  Frequencies   : {len(FREQUENCIES_GHZ)}  "
          f"({min(FREQUENCIES_GHZ):g}-{max(FREQUENCIES_GHZ):g} GHz)")
    print(f"  Aspects       : {len(ASPECTS_DEG)}  "
          f"({min(ASPECTS_DEG):g}-{max(ASPECTS_DEG):g} deg from +z axis)")
    print(f"  Units total   : {len(ordered)}  (geom x freq x pol)")
    print(f"  Workers       : {pool_size} procs x {WORKERS_PER_UNIT} threads "
          f"of {cores} cpus  (BLAS threads/worker: {BLAS_THREADS_PER_WORKER})")
    print(f"  Memory        : {memory_gb:.1f} GB detected, {budget_gb:.1f} GB "
          f"schedulable at {STREAM_BUDGET_GB:g} GB reserved per unit")
    print("=" * 70, flush=True)

    # Parse each distinct geometry once, and import the solver, before the pool
    # forks: workers then inherit both instead of repeating the work per unit.
    for unit in ordered:
        _load_snapshot(str(unit["geometry"]))
    import bor_dispatch  # noqa: F401
    import grim_io       # noqa: F401

    counters = {"written": 0, "skipped": 0, "failed": 0}
    started = time.time()
    total = len(ordered)

    def _prepare(unit):
        name = _unit_name(unit)
        return (
            name, float(STREAM_BUDGET_GB),
            (_solve_and_export_star, ((unit, context, str(results_dir)),)),
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
            pool, budget_gb=budget_gb, max_concurrent=pool_size
        )
        dispatcher.run(ordered, _prepare, _on_result, _on_error)

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
