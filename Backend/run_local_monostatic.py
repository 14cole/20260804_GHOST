#!/usr/bin/env python3
"""
Local monostatic RCS sweep — same naming/streaming as run_hpc_monostatic.py,
no SLURM. Defaults to (cpu_count - 1) workers and exports one .grim per
(geometry, frequency, polarization) unit as soon as it finishes.

Edit the CONFIG block and run:

    python run_local_monostatic.py

Outputs go to <OUTPUT_DIR>/run_YYYYMMDD_HHMMSS/results/ as files named
"<POL>_<FREQ:.3f>GHz_<geometry_stem>.grim". Every resumable output carries
an exact byte/source/runtime/input attestation; an unverified existing file is
rejected rather than reused.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import workflow_provenance as _workflow_provenance
from solver_quality import validate_mesh_convergence_policy
from workflow_provenance import (
    backend_source_fingerprint,
    manifest_solve_spec_fingerprint,
    runtime_environment_fingerprint,
    sha256_file,
    stable_json_fingerprint,
    unit_solve_spec_fingerprint,
    verify_output_attestation,
    write_output_attestation,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

# Geometry folders. Every *.geo file under these (recursively) is included.
FRD_DIR = "geometries/FRD"
OPN_DIR = "geometries/OPN"

# Sweep.
FREQUENCIES_GHZ = [2.0, 4.0, 6.0, 8.0, 10.0]
AZIMUTHS_DEG    = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
POLARIZATIONS   = ["VV", "HH"]          # any subset of: VV, HH, TM, TE

# Output root. A new run_YYYYMMDD_HHMMSS/ subfolder is created inside.
OUTPUT_DIR = "rcs_runs"

# Worker pool size. None → max(1, cpu_count() - 1).
WORKERS = None

# Solver knobs (mirror run_monostatic.py).
GEOMETRY_UNITS          = "inches"       # "inches" or "meters"
SOLVER_METHOD           = "auto"         # "auto" | "direct" | supported "fmm"
CFIE_ALPHA              = 0.0
MAX_PANELS              = 50_000
BLAS_THREADS_PER_WORKER = 1

GEOMETRY_EXTS = (".geo",)

# ═══════════════════════════════════════════════════════════════════════════════

def _solver_source_fingerprint() -> 'str':
    return backend_source_fingerprint(
        str(Path(_workflow_provenance.__file__).resolve().parent)
    )


def _write_json_atomic(path: 'Path', payload: 'Dict[str, Any]') -> 'None':
    temporary = Path(str(path) + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _verify_run_provenance(manifest: 'Dict[str, Any]') -> 'None':
    if _solver_source_fingerprint() != manifest.get(
        "solver_source_sha256"
    ):
        raise RuntimeError(
            "Local-run solver source/native artifacts changed; no mixed-state "
            "field will be written or reused."
        )
    if runtime_environment_fingerprint() != manifest.get(
        "runtime_environment_sha256"
    ):
        raise RuntimeError(
            "Local-run Python/platform/NumPy/SciPy/BLAS runtime changed."
        )


def _unit_attestation_fields(
    manifest: 'Dict[str, Any]',
    unit: 'Dict[str, Any]',
) -> 'Dict[str, Any]':
    return {
        "run_id": str(manifest["run_id"]),
        "solver_source_sha256": str(manifest["solver_source_sha256"]),
        "runtime_environment_sha256":
            str(manifest["runtime_environment_sha256"]),
        "geometry_input_sha256":
            str(unit["geometry_input_sha256"]),
        "run_solve_spec_sha256":
            manifest_solve_spec_fingerprint(manifest),
        "unit_solve_spec_sha256":
            unit_solve_spec_fingerprint(unit),
        "solver_config_sha256":
            stable_json_fingerprint(manifest.get("solver_config", {})),
        "angular_grid_kind": "azimuths_deg",
        "angular_grid_deg":
            [float(value) for value in unit["azimuths_deg"]],
        "polarization": str(unit["polarization"]),
        "frequency_ghz": float(unit["frequency_ghz"]),
    }


def _verify_unit_input(unit: 'Dict[str, Any]') -> 'None':
    from feature_sum import geometry_input_fingerprint
    current = geometry_input_fingerprint(
        unit["geometry"], GEOMETRY_UNITS
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


def _pin_blas_threads(n: 'int') -> 'None':
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)


def _unit_output_path(results_dir: 'Path', unit: 'Dict[str, Any]') -> 'Path':
    pol  = unit["polarization"]
    freq = float(unit["frequency_ghz"])
    stem = unit["geometry_stem"]
    return results_dir / f"{pol}_{freq:.3f}GHz_{stem}.grim"


def _solve_and_export(unit, snapshot, material_base, results_dir_str):
    """Pool-worker entry point: solve one unit, export .grim. Idempotent."""
    results_dir = Path(results_dir_str)
    run_dir = results_dir.parent
    manifest = json.loads((run_dir / "manifest.json").read_text())
    _verify_run_provenance(manifest)
    _verify_unit_input(unit)
    attestation = _unit_attestation_fields(manifest, unit)
    out_path = _unit_output_path(results_dir, unit)
    if out_path.exists():
        verify_output_attestation(str(out_path), attestation)
        return ("skipped", str(out_path))

    from rcs_solver import solve_monostatic_rcs_2d_certified
    result = solve_monostatic_rcs_2d_certified(
        geometry_snapshot=snapshot,
        frequencies_ghz=[float(unit["frequency_ghz"])],
        elevations_deg=[float(a) for a in unit["azimuths_deg"]],
        polarization=unit["polarization"],
        geometry_units=GEOMETRY_UNITS,
        material_base_dir=material_base,
        max_panels=MAX_PANELS,
        cfie_alpha=CFIE_ALPHA,
        solver_method=SOLVER_METHOD,
        mesh_convergence_policy=manifest["solver_config"][
            "mesh_convergence_policy"
        ],
    )
    _verify_run_provenance(manifest)
    _verify_unit_input(unit)

    from grim_io import export_result_to_grim
    written = export_result_to_grim(
        result, str(out_path),
        source_path=str(snapshot.get("source_path", "") or ""),
        history=(f"run_local_monostatic.py pol={unit['polarization']} "
                 f"freq={unit['frequency_ghz']}GHz"),
    )
    actual_path = str(written[0]) if written else str(out_path)
    write_output_attestation(actual_path, attestation)
    return ("written", actual_path)


def main() -> 'None':
    _pin_blas_threads(BLAS_THREADS_PER_WORKER)
    from geometry_io import parse_geometry, build_geometry_snapshot

    geometries = _discover_geometries()
    if not geometries:
        sys.exit("ERROR: no geometry files (*.geo) found under FRD_DIR or OPN_DIR.")

    pols = [p.strip().upper() for p in POLARIZATIONS if p and p.strip()]
    if not pols:            sys.exit("ERROR: POLARIZATIONS is empty.")
    if not FREQUENCIES_GHZ: sys.exit("ERROR: FREQUENCIES_GHZ is empty.")
    if not AZIMUTHS_DEG:    sys.exit("ERROR: AZIMUTHS_DEG is empty.")
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
                units.append({
                    "geometry":      str(geom.resolve()),
                    "geometry_stem": geom.stem,
                    "geometry_input_sha256": input_fingerprint,
                    "polarization":  pol,
                    "frequency_ghz": float(f),
                    "azimuths_deg":  [float(a) for a in AZIMUTHS_DEG],
                })

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir     = Path(OUTPUT_DIR).resolve() / run_id
    results_dir = run_dir / "results"
    run_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir()
    manifest: 'Dict[str, Any]' = {
        "schema": "ghost.local.2d-run.v1",
        "status": "running",
        "run_id": run_id,
        "solver_source_sha256": _solver_source_fingerprint(),
        "runtime_environment_sha256":
            runtime_environment_fingerprint(),
        "solver_config": {
            "geometry_units": GEOMETRY_UNITS,
            "solver_method": SOLVER_METHOD,
            "cfie_alpha": float(CFIE_ALPHA),
            "max_panels": int(MAX_PANELS),
            "blas_threads_per_worker":
                int(BLAS_THREADS_PER_WORKER),
            "mesh_convergence_policy":
                validate_mesh_convergence_policy(),
        },
        "units": units,
    }
    manifest_path = run_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)

    cpu = os.cpu_count() or 1
    n_workers = WORKERS if WORKERS else max(1, cpu - 1)
    n_workers = max(1, min(int(n_workers), len(units)))

    print("=" * 70)
    print("Local monostatic RCS sweep")
    print("=" * 70)
    print(f"  Run dir       : {run_dir}")
    print(f"  Geometries    : {len(geometries)}")
    print(f"  Polarizations : {', '.join(pols)}")
    print(f"  Frequencies   : {len(FREQUENCIES_GHZ)}  "
          f"({min(FREQUENCIES_GHZ):g}-{max(FREQUENCIES_GHZ):g} GHz)")
    print(f"  Azimuths      : {len(AZIMUTHS_DEG)}")
    print(f"  Units total   : {len(units)}  (geom × freq × pol)")
    print(f"  Workers       : {n_workers} of {cpu} cpus  "
          f"(BLAS threads/worker: {BLAS_THREADS_PER_WORKER})")
    print("=" * 70, flush=True)

    # Parse each geometry once so workers share the snapshot via pickle.
    snapshots: 'Dict[str, Tuple[Dict[str, Any], str]]' = {}
    for u in units:
        gpath = u["geometry"]
        if gpath in snapshots:
            continue
        p = Path(gpath)
        title, segments, ibcs, dielectrics = parse_geometry(p.read_text())
        snap = build_geometry_snapshot(title, segments, ibcs, dielectrics)
        snap["source_path"] = str(p)
        snapshots[gpath] = (snap, str(p.parent))

    t0 = time.time()
    n_done = n_skipped = n_failed = 0
    total = len(units)
    # Python 3.6 ProcessPoolExecutor has no initializer/initargs. The parent
    # environment was pinned above and is inherited by spawned workers.
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        fut_to_unit = {}
        for u in units:
            snap, mat_base = snapshots[u["geometry"]]
            fut = pool.submit(_solve_and_export, u, snap, mat_base, str(results_dir))
            fut_to_unit[fut] = u

        for fut in as_completed(fut_to_unit):
            u = fut_to_unit[fut]
            tag = (f"{u['polarization']} {u['frequency_ghz']:7.3f}GHz "
                   f"{u['geometry_stem']}")
            try:
                status, path = fut.result()
                if status == "skipped":
                    n_skipped += 1
                else:
                    n_done += 1
                idx = n_done + n_skipped + n_failed
                print(f"  [{idx:3d}/{total}] {status:7s}  {tag}  -> "
                      f"{Path(path).name}", flush=True)
            except Exception as exc:
                n_failed += 1
                idx = n_done + n_skipped + n_failed
                print(f"  [{idx:3d}/{total}] FAILED   {tag}: {exc}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Done. wrote={n_done}, skipped={n_skipped}, failed={n_failed}.  "
          f"{elapsed:.1f} s elapsed.")
    print(f"  Outputs: {results_dir}/")
    if n_failed:
        manifest["status"] = "failed"
        _write_json_atomic(manifest_path, manifest)
        raise SystemExit(1)
    expected_paths = [
        _unit_output_path(results_dir, unit) for unit in units
    ]
    manifest["status"] = "complete"
    manifest["output_sha256"] = {
        path.name: sha256_file(str(path)) for path in expected_paths
    }
    _write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
