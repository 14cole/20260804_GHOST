#!/usr/bin/env python3
"""
HPC monostatic BoR RCS sweep driver (SLURM) -- the body-of-revolution
counterpart of run_hpc_monostatic.py.

Edit the CONFIG block below and run:

    python run_hpc_bor_monostatic.py

Workflow (mirrors the 2D driver):
- Discover geometry files under FRD_DIR + OPN_DIR (BoR .geo files: x = rho,
  y = z, generatrices traversed +z -> -z; see bor_dispatch).
- The user requests frequencies, radar azimuths, and radar elevations. Exact
  BoR body-aspect RHS columns are derived automatically. VV and HH share each
  azimuthal-mode factorization.
- Distribute units round-robin across N_NODES x N_JOBS slots, write sbatch
  job-array scripts, submit. Restartable: units whose .grim exists are
  skipped.
- Every completed frequency immediately publishes its solver-frame VV and HH
  GRIMs under results/by_frequency/. Once complete, one self-contained
  <geometry>.grim is also published in results/. Its primary arrays are the
  requested monostatic az/el/frequency VV/HH/VH response and it embeds the
  exact body model/profile needed for coherent feature placement.

BoR-specific notes:
- A BoR unit parallelizes INTERNALLY (threads across azimuthal modes and
  streaming-assembly tiles), so the node is divided into a few workers with
  several threads each (WORKERS_PER_UNIT) instead of one process per core.
- The dispatch auto-selects table vs streaming assembly and single/double
  precision by memory estimate; override with ASSEMBLY / TABLE_PRECISION /
  STREAM_BUDGET_GB if needed.
Mesh certification is optional and never changes whether a result may be
published or used downstream.

Internal worker invocation (called by SLURM, not by the user):
    python run_hpc_bor_monostatic.py --worker <run_dir> <job_index> <node_index>
"""

import argparse
import json
import math
import shlex
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
from workflow_provenance import (
    backend_source_fingerprint,
    backend_source_inventory,
    describe_source_mismatch,
    manifest_solve_spec_fingerprint,
    runtime_environment_fingerprint,
    stable_json_fingerprint,
    unit_solve_spec_fingerprint,
    embed_output_attestation,
    verify_embedded_attestation,
)

# ===============================================================================
# CONFIG -- the only section most users need to edit
# ===============================================================================

FRD_DIR = "geometries/FRD"
OPN_DIR = "geometries/OPN"

# Requested monostatic grid.
FREQUENCIES_GHZ = [1.0, 2.0, 3.0]
AZIMUTHS_DEG    = [float(a) for a in range(0, 360, 5)]
ELEVATIONS_DEG  = [float(e) for e in range(-60, 61, 5)]

# Body-axis attitude in the radar/earth frame. Default: horizontal with the
# nose at azimuth 0. Roll affects later feature orientation, not the bare BoR.
BODY_AXIS_AZ_DEG = 0.0
BODY_AXIS_EL_DEG = 0.0
BODY_ROLL_DEG    = 0.0

# Output root. A new run_YYYYMMDD_HHMMSS/ subfolder is created inside.
OUTPUT_DIR = "rcs_runs_bor"

# --- Multi-node / multi-submission parallelism -----------------------------
N_NODES = 1
N_JOBS  = 1

# ===============================================================================
# ADVANCED -- fine tuning (SLURM resources, solver knobs, monostatic grid)
# ===============================================================================

SLURM_PARTITION = "compute"
SLURM_ACCOUNT   = None
SLURM_QOS       = None
SLURM_TIME      = None
CORES_PER_NODE  = None            # None = whole node via --exclusive
MEM_PER_NODE    = "0"             # "0" = all node memory (see 2D driver notes)
MAX_WORKERS_PER_NODE = None       # cap concurrent UNITS per node (memory-heavy
                                  # units: peak ~ streaming blocks + mode LU)
SLURM_MAIL_TYPE = None
SLURM_MAIL_USER = None
SLURM_EXTRA_SBATCH = []  # type: List[str]
JOB_PROLOGUE = []  # type: List[str]

# --- Solver knobs (mirror bor_dispatch.solve_monostatic_rcs_bor) -----------
GEOMETRY_UNITS          = "inches"       # "inches" or "meters"
CFIE_ALPHA              = 0.5            # closed PEC bodies -> CFIE
N_MODES                 = None           # None = auto (adaptive truncation)
MODE_TOL                = 1e-6
MAX_ELEMENTS            = 50_000
ASSEMBLY                = "auto"         # "auto" | "tables" | "streaming"
TABLE_PRECISION         = "auto"         # "auto" | "single" | "double"
STREAM_BUDGET_GB        = 8.0            # held streaming-block budget per unit
MESH_CERTIFICATION      = True           # recommended base/fine comparison;
                                         # False = base mesh only
WORKERS_PER_UNIT        = 4              # threads inside one BoR solve (modes
                                         # + streaming tiles); pool size =
                                         # cores // WORKERS_PER_UNIT
BLAS_THREADS_PER_WORKER = 1

# --- Scheduling ------------------------------------------------------------
# Units are costed at submit time and dealt out longest-processing-time-first,
# then claimed at run time through atomic files in <run_dir>/claims/.  A BoR
# unit's cost grows roughly as the fourth power of frequency (elements^3 x
# modes), so a frequency sweep is even more lopsided than the 2-D one and an
# index-modulo split -- what this used to do -- strands the expensive units on
# whichever slot happens to draw them.
MEMORY_HEADROOM     = 0.85   # fraction of node memory the scheduler may reserve
CLAIM_STALE_SECONDS = 7200   # a quiet claim older than this is stealable
TASKS_PER_CHILD     = 2      # pool worker lifetime, in units

GEOMETRY_EXTS = (".geo",)
PYTHON_EXE    = sys.executable
SUBMIT        = True

# ===============================================================================

_SBATCH = shutil.which("sbatch") or "sbatch"


# --- shared helpers --------------------------------------------------------

def _solver_source_records():
    # type: () -> Tuple[str, Dict[str, str]]
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


def _verify_run_provenance(manifest):
    # type: (Dict[str, Any]) -> None
    expected_source = str(manifest.get("solver_source_sha256", ""))
    expected_runtime = str(manifest.get("runtime_environment_sha256", ""))
    if not expected_source or not expected_runtime:
        raise RuntimeError(
            "HPC run manifest lacks exact solver-source/runtime provenance; "
            "legacy runs must be regenerated before reuse."
        )
    current_source = _solver_source_fingerprint()
    current_runtime = runtime_environment_fingerprint()
    if current_source != expected_source:
        # Name the files: "something under Backend/ differs" is not actionable,
        # and the usual cause is a tree that was only partly updated.
        recorded = manifest.get("solver_source_inventory") or {}
        detail = (
            describe_source_mismatch(recorded, _solver_source_inventory())
            if recorded else
            "this run predates per-file inventories, so the differing file "
            "cannot be named -- run tests/diagnose_provenance.py"
        )
        raise RuntimeError(
            "Solver source/native artifacts differ from the HPC run manifest; "
            "no cached or new field will be used from this mixed source state. "
            f"({detail}). Either restore the recorded source or submit a new "
            "run with the code you actually want to execute."
        )
    if current_runtime != expected_runtime:
        raise RuntimeError(
            "Python/platform/NumPy/SciPy/BLAS runtime differs from the HPC run "
            "manifest; start a new run in this numerical environment."
        )


def _unit_attestation_fields(manifest, unit):
    # type: (Dict[str, Any], Dict[str, Any]) -> Dict[str, Any]
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
        "angular_grid_kind": "aspects_deg",
        "angular_grid_deg":
            [float(value) for value in manifest["aspects_deg"]],
        "polarization": str(unit["polarization"]),
        "frequency_ghz": float(unit["frequency_ghz"]),
    }


def _verify_unit_input(unit, manifest):
    # type: (Dict[str, Any], Dict[str, Any]) -> None
    from feature_sum import geometry_input_fingerprint
    current = geometry_input_fingerprint(
        str(unit["geometry"]),
        str(manifest["solver_config"]["geometry_units"]),
    )
    if current != unit.get("geometry_input_sha256"):
        raise RuntimeError(
            f"Frozen geometry/material input changed during the HPC unit: "
            f"{unit['geometry']}"
        )


def _discover_geometries():
    # type: () -> List[Path]
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


def _pin_blas_threads(n):
    # type: (int) -> None
    hpc_scheduler.pin_blas_threads(n)


def _detect_cores():
    # type: () -> int
    return hpc_scheduler.detect_cores()


def _unit_output_path(run_dir, unit):
    # type: (Path, Dict[str, Any]) -> Path
    pol  = unit["polarization"]
    freq = float(unit["frequency_ghz"])
    stem = unit["geometry_stem"]
    return (run_dir / "results" / "by_frequency" /
            f"{pol}_{freq:.3f}GHz_{stem}.grim")


def publish_monostatic(run_dir_str, require_complete=True):
    # type: (str, bool) -> Tuple[int, int]
    """Publish one user-facing monostatic GRIM per completed geometry."""

    run_dir = Path(run_dir_str).resolve()
    manifest = _manifest_for(run_dir)
    units = list(manifest.get("units", []) or [])
    missing = [
        _unit_output_path(run_dir, unit)
        for unit in units
        if not _unit_output_path(run_dir, unit).is_file()
    ]
    if missing:
        if require_complete:
            raise ValueError(
                f"Cannot publish monostatic response: {len(missing)} solver "
                f"unit(s) remain; first missing: {missing[0]}."
            )

    from hpc_common import (
        bodies_from_units,
        read_unit_grims,
        require_hpc_output_attestations,
        require_hpc_run_provenance,
    )
    require_hpc_run_provenance(manifest, "ghost.hpc.bor-run.v1")
    if not missing:
        require_hpc_output_attestations(run_dir, manifest)
    else:
        # Partial publication is allowed only for a geometry whose own units
        # are all present. Verify every available unit against the immutable
        # manifest before using it; do not require unrelated geometries yet.
        for unit in units:
            path = _unit_output_path(run_dir, unit)
            if path.is_file():
                verify_embedded_attestation(
                    str(path), _unit_attestation_fields(manifest, unit)
                )
    records = read_unit_grims(run_dir / str(manifest["unit_output_dir"]))

    from feature_sum import (
        load_body_grim,
        load_body_requested_radar_grid,
        outer_generatrix,
        save_monostatic_grim,
    )
    from geometry_io import build_geometry_snapshot, parse_geometry

    grid = dict(manifest["radar_grid"])
    out_dir = run_dir / "results"
    out_dir.mkdir(exist_ok=True)
    written = skipped = 0
    for stem in sorted({str(unit["geometry_stem"]) for unit in units}):
        matching = [unit for unit in units if unit["geometry_stem"] == stem]
        if any(
            not _unit_output_path(run_dir, unit).is_file()
            for unit in matching
        ):
            continue
        geometry = Path(str(matching[0]["geometry"]))
        destination = out_dir / f"{stem}.grim"
        if destination.is_file():
            try:
                load_body_grim(str(destination))
                stored_grid = load_body_requested_radar_grid(str(destination))
                comparable = {
                    key: stored_grid[key]
                    for key in (
                        "azimuths_deg", "elevations_deg", "frequencies_ghz",
                        "axis_az_deg", "axis_el_deg", "roll_deg",
                    )
                }
                if comparable == {
                    key: grid[key]
                    for key in comparable
                }:
                    skipped += 1
                    continue
            except (OSError, TypeError, ValueError):
                pass
        snapshot = build_geometry_snapshot(
            *parse_geometry(geometry.read_text(encoding="utf-8"))
        )
        profile = outer_generatrix(
            snapshot, str(manifest["solver_config"]["geometry_units"])
        )
        save_monostatic_grim(
            bodies_from_units(records, stem=stem),
            profile,
            str(destination),
            azimuths_deg=grid["azimuths_deg"],
            elevations_deg=grid["elevations_deg"],
            axis_az_deg=float(grid["axis_az_deg"]),
            axis_el_deg=float(grid["axis_el_deg"]),
            roll_deg=float(grid.get("roll_deg", 0.0)),
            history="run_hpc_bor_monostatic.py monostatic response",
            source_path=str(matching[0].get("geometry_original", geometry)),
            artifact_metadata={
                "geometry_input_sha256": str(
                    matching[0].get("geometry_input_sha256", "")
                ),
                "solver_source_sha256": str(
                    manifest.get("solver_source_sha256", "")
                ),
                "runtime_environment_sha256": str(
                    manifest.get("runtime_environment_sha256", "")
                ),
                "run_solve_spec_sha256": manifest_solve_spec_fingerprint(manifest),
            },
        )
        written += 1
    return written, skipped


def _paired_solve_units(output_units):
    # type: (List[Dict[str, Any]]) -> List[Dict[str, Any]]
    """Keep manifest outputs separate while co-solving compatible channels."""
    pairs = {}  # type: Dict[Tuple[str, float], Dict[str, Any]]
    for unit in output_units:
        key = (str(unit["geometry"]), float(unit["frequency_ghz"]))
        pair = pairs.setdefault(key, {
            "geometry": unit["geometry"],
            "geometry_stem": unit["geometry_stem"],
            "geometry_input_sha256": unit["geometry_input_sha256"],
            "frequency_ghz": float(unit["frequency_ghz"]),
            "channel_units": [],
        })
        pair["channel_units"].append(unit)
        if "estimated_peak_gb" in unit:
            estimate = float(unit["estimated_peak_gb"])
            previous = pair.get("estimated_peak_gb")
            if previous is not None and float(previous) != estimate:
                raise ValueError(
                    "Paired BoR output units carry different memory estimates."
                )
            pair["estimated_peak_gb"] = estimate
    return list(pairs.values())


def _channel_result(result, polarization):
    # type: (Dict[str, Any], str) -> Dict[str, Any]
    samples = list(
        (result.get("co_solved_samples", {}) or {}).get(polarization, []) or []
    )
    if not samples:
        raise RuntimeError(
            f"BoR solve did not return its co-solved {polarization} channel."
        )
    channel = dict(result)
    channel["samples"] = samples
    channel["polarization"] = polarization
    channel["polarization_export"] = polarization
    channel["metadata"] = dict(result.get("metadata", {}) or {})
    return channel


# The manifest carries every unit in the run, so re-reading and re-parsing it
# inside each unit made a node's JSON cost quadratic in the size of the sweep.
# It is immutable once written, so a per-process memo keyed on the file's
# identity is exact.
_MANIFEST_CACHE = {}  # type: Dict[str, Tuple[Tuple[int, int, int], Dict[str, Any]]]


def _manifest_for(run_dir):
    # type: (Path) -> Dict[str, Any]
    path = Path(run_dir) / "manifest.json"
    stat = path.stat()
    key = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
    cached = _MANIFEST_CACHE.get(str(path))
    if cached is not None and cached[0] == key:
        return cached[1]
    manifest = json.loads(path.read_text())
    _MANIFEST_CACHE[str(path)] = (key, manifest)
    return manifest


def _solve_and_export(pair, snapshot, material_base, run_dir_str):
    # type: (Dict[str, Any], Dict[str, Any], str, str) -> Tuple[str, str]
    """Solve one geometry/frequency pair and export requested channels."""
    run_dir = Path(run_dir_str)
    channel_units = list(pair["channel_units"])
    manifest = _manifest_for(run_dir)
    _verify_run_provenance(manifest)
    for unit in channel_units:
        _verify_unit_input(unit, manifest)
    missing = []
    paths = []
    for unit in channel_units:
        out_path = _unit_output_path(run_dir, unit)
        paths.append(out_path)
        if out_path.exists():
            verify_embedded_attestation(
                str(out_path), _unit_attestation_fields(manifest, unit)
            )
        else:
            missing.append(unit)
    if not missing:
        return ("skipped", ", ".join(str(path) for path in paths))

    from bor_dispatch import (
        solve_monostatic_rcs_bor_certified,
        solve_monostatic_rcs_bor_survey,
    )
    certified = bool(
        manifest.get("solver_config", {}).get("mesh_certification", True)
    )
    solver_config = dict(manifest.get("solver_config", {}) or {})
    solve = (solve_monostatic_rcs_bor_certified if certified
             else solve_monostatic_rcs_bor_survey)
    result = solve(
        geometry_snapshot=snapshot,
        frequencies_ghz=[float(pair["frequency_ghz"])],
        elevations_deg=[float(a) for a in manifest["aspects_deg"]],
        polarization=missing[0]["polarization"],
        geometry_units=str(solver_config.get(
            "geometry_units", GEOMETRY_UNITS
        )),
        material_base_dir=material_base,
        cfie_alpha=float(solver_config.get("cfie_alpha", CFIE_ALPHA)),
        n_modes=solver_config.get("n_modes", N_MODES),
        mode_tol=float(solver_config.get("mode_tol", MODE_TOL)),
        max_elements=int(solver_config.get("max_elements", MAX_ELEMENTS)),
        workers=int(solver_config.get(
            "workers_per_unit", WORKERS_PER_UNIT
        )),
        table_precision=str(solver_config.get(
            "table_precision", TABLE_PRECISION
        )),
        assembly=str(solver_config.get("assembly", ASSEMBLY)),
        stream_budget_gb=float(solver_config.get(
            "stream_budget_gb", STREAM_BUDGET_GB
        )),
        expand_to_360=False,
    )
    for w in result["metadata"].get("warnings", []) or []:
        print(f"      [warn] {pair['geometry_stem']}: {w}", flush=True)
    _verify_run_provenance(manifest)
    for unit in channel_units:
        _verify_unit_input(unit, manifest)

    from grim_io import export_result_to_grim
    actual_paths = []
    for unit in missing:
        out_path = _unit_output_path(run_dir, unit)
        channel_result = _channel_result(result, str(unit["polarization"]))
        embed_output_attestation(
            channel_result, _unit_attestation_fields(manifest, unit)
        )
        written = export_result_to_grim(
            channel_result, str(out_path),
            source_path=str(snapshot.get("source_path", "") or ""),
            history=(f"run_hpc_bor_monostatic.py pol={unit['polarization']} "
                     f"freq={unit['frequency_ghz']}GHz"),
        )
        actual_paths.append(str(written[0]) if written else str(out_path))
    _verify_run_provenance(manifest)
    for unit in channel_units:
        _verify_unit_input(unit, manifest)

    return ("written", ", ".join(actual_paths))


def _solve_and_export_star(args):
    # type: (tuple) -> tuple
    u, snap, mat_base, run_dir_str = args
    try:
        status, path = _solve_and_export(u, snap, mat_base, run_dir_str)
        return ("ok", status, path, u)
    except Exception:
        return ("err", traceback.format_exc(), "", u)


# --- submit mode (user-invoked) --------------------------------------------

def _build_slurm(script_path, run_dir, job_index):
    # type: (Path, Path, int) -> str
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=bor_{run_dir.name}_j{job_index}",
        f"#SBATCH --array=0-{N_NODES - 1}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --partition={SLURM_PARTITION}",
        f"#SBATCH --output={run_dir}/logs/job{job_index}_task_%A_%a.out",
        f"#SBATCH --error={run_dir}/logs/job{job_index}_task_%A_%a.err",
        # A requeued task rejoins the run and takes whatever is unclaimed, so
        # preemption costs only the units that were in flight.
        "#SBATCH --requeue",
        "#SBATCH --open-mode=append",
    ]
    if CORES_PER_NODE is not None:
        lines.append(f"#SBATCH --cpus-per-task={CORES_PER_NODE}")
    else:
        lines.append("#SBATCH --exclusive")
    if MEM_PER_NODE:
        lines.append(f"#SBATCH --mem={MEM_PER_NODE}")
    if SLURM_TIME:
        lines.append(f"#SBATCH --time={SLURM_TIME}")
    if SLURM_ACCOUNT:   lines.append(f"#SBATCH --account={SLURM_ACCOUNT}")
    if SLURM_QOS:       lines.append(f"#SBATCH --qos={SLURM_QOS}")
    if SLURM_MAIL_TYPE: lines.append(f"#SBATCH --mail-type={SLURM_MAIL_TYPE}")
    if SLURM_MAIL_USER: lines.append(f"#SBATCH --mail-user={SLURM_MAIL_USER}")
    for extra in SLURM_EXTRA_SBATCH:
        e = extra.strip()
        if e:
            lines.append(e if e.startswith("#SBATCH") else f"#SBATCH {e}")

    lines += [
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(script_path.parent))}",
        *JOB_PROLOGUE,
        # The configured driver lives in the run directory, not beside its
        # solver modules.  Put the exact Backend tree used to create the
        # manifest first, even when the login environment inherited an older
        # GHOST checkout on PYTHONPATH.  Without this, a new driver can import
        # an old grim_io on compute nodes and fail only during derived export.
        ("export PYTHONPATH="
         f"{shlex.quote(str(Path(_workflow_provenance.__file__).resolve().parent))}"
         ":${PYTHONPATH:-}"),
        (f"exec {shlex.quote(PYTHON_EXE)} {shlex.quote(str(script_path))} "
         f"--worker {shlex.quote(str(run_dir))} {job_index} "
         f"${{SLURM_ARRAY_TASK_ID}}"),
        "",
    ]
    return "\n".join(lines)


def submit():
    # type: () -> None
    geometries = _discover_geometries()
    if not geometries:
        sys.exit("ERROR: no geometry files (*.geo) found under FRD_DIR or OPN_DIR.")

    pols = ["VV", "HH"]
    if not FREQUENCIES_GHZ: sys.exit("ERROR: FREQUENCIES_GHZ is empty.")
    try:
        from feature_sum import radar_grid_aspects, validate_radar_grid
        validate_radar_grid(AZIMUTHS_DEG, ELEVATIONS_DEG)
        aspects = [float(value) for value in radar_grid_aspects(
            AZIMUTHS_DEG,
            ELEVATIONS_DEG,
            BODY_AXIS_AZ_DEG,
            BODY_AXIS_EL_DEG,
        )]
        attitude = [float(value) for value in (
            BODY_AXIS_AZ_DEG, BODY_AXIS_EL_DEG, BODY_ROLL_DEG
        )]
        if not all(math.isfinite(value) for value in attitude):
            raise ValueError("body-axis attitude angles must be finite")
    except ValueError as exc:
        sys.exit(f"ERROR: radar grid is invalid -- {exc}")
    frequencies = [float(value) for value in FREQUENCIES_GHZ]
    if (
        not all(math.isfinite(value) and value > 0.0
                for value in frequencies)
        or len(set(frequencies)) != len(frequencies)
        or len({f"{value:.3f}" for value in frequencies})
        != len(frequencies)
    ):
        sys.exit(
            "ERROR: frequencies must be finite, positive, unique, and "
            "distinct at the 0.001 GHz output-name precision."
        )
    if (
        not math.isfinite(float(CFIE_ALPHA))
        or not 0.0 <= float(CFIE_ALPHA) <= 1.0
    ):
        sys.exit("ERROR: CFIE_ALPHA must be finite and lie in [0, 1].")
    if (
        N_MODES is not None and int(N_MODES) < 1
        or not math.isfinite(float(MODE_TOL))
        or float(MODE_TOL) <= 0.0
        or int(MAX_ELEMENTS) < 1
        or not math.isfinite(float(STREAM_BUDGET_GB))
        or float(STREAM_BUDGET_GB) <= 0.0
        or int(WORKERS_PER_UNIT) < 1
        or int(BLAS_THREADS_PER_WORKER) < 1
        or not math.isfinite(float(MEMORY_HEADROOM))
        or not 0.0 < float(MEMORY_HEADROOM) <= 1.0
        or int(TASKS_PER_CHILD) < 1
        or (
            MAX_WORKERS_PER_NODE is not None
            and int(MAX_WORKERS_PER_NODE) < 1
        )
    ):
        sys.exit("ERROR: BoR numerical resource/tolerance controls are invalid.")
    if str(ASSEMBLY).strip().lower() not in {
        "auto", "tables", "streaming"
    }:
        sys.exit("ERROR: ASSEMBLY must be auto, tables, or streaming.")
    if str(TABLE_PRECISION).strip().lower() not in {
        "auto", "single", "double"
    }:
        sys.exit(
            "ERROR: TABLE_PRECISION must be auto, single, or double."
        )
    if int(N_NODES) < 1 or int(N_JOBS) < 1:
        sys.exit("ERROR: N_NODES and N_JOBS must be >= 1.")

    stems = [g.stem for g in geometries]
    if len(stems) != len(set(stems)):
        sys.exit("ERROR: geometry stems must be unique; per-unit result names "
                 "would otherwise overwrite one another.")

    run_id  = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = Path(OUTPUT_DIR).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    (run_dir / "results").mkdir()
    (run_dir / "results" / "by_frequency").mkdir()
    (run_dir / "claims").mkdir()

    # Freeze the exact solver inputs inside the run.  Referencing the discovery
    # folder directly makes an active/archived run mutable: a later staging
    # operation can replace its .geo or material tables before a worker reads
    # them.  Each geometry gets its own folder so equally named material tables
    # cannot collide.
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

    from bor_dispatch import estimate_bor_resources
    from geometry_io import build_geometry_snapshot, parse_geometry

    resource_estimates = {}  # type: Dict[Tuple[str, float], Dict[str, Any]]
    for _original, geom in frozen_geometries:
        snapshot = build_geometry_snapshot(
            *parse_geometry(geom.read_text(encoding="utf-8"))
        )
        snapshot["source_path"] = str(geom.resolve())
        for frequency in frequencies:
            resource_estimates[(str(geom.resolve()), float(frequency))] = (
                estimate_bor_resources(
                    snapshot,
                    frequency,
                    aspects,
                    geometry_units=GEOMETRY_UNITS,
                    material_base_dir=str(geom.parent),
                    n_modes=N_MODES,
                    max_elements=MAX_ELEMENTS,
                    workers=WORKERS_PER_UNIT,
                    table_precision=TABLE_PRECISION,
                    assembly=ASSEMBLY,
                    stream_budget_gb=STREAM_BUDGET_GB,
                    mesh_certification=bool(MESH_CERTIFICATION),
                )
            )

    units = []  # type: List[Dict[str, Any]]
    from feature_sum import geometry_input_fingerprint
    for original, geom in frozen_geometries:
        input_fingerprint = geometry_input_fingerprint(
            str(geom), GEOMETRY_UNITS
        )
        for pol in pols:
            for f in FREQUENCIES_GHZ:
                units.append({
                    "geometry":      str(geom.resolve()),
                    "geometry_stem": geom.stem,
                    "geometry_original": str(original.resolve()),
                    "geometry_input_sha256": input_fingerprint,
                    "polarization":  pol,
                    "frequency_ghz": float(f),
                    "estimated_peak_gb": float(resource_estimates[
                        (str(geom.resolve()), float(f))
                    ]["estimated_peak_gb"]),
                })

    source_driver = Path(__file__).resolve()
    manifest = {
        "schema":          "ghost.hpc.bor-run.v1",
        "run_id":          run_id,
        "created":         datetime.now().isoformat(),
        "solver":          "bor_mom_rcs",
        "frd_dir":         str(Path(FRD_DIR).resolve()),
        "opn_dir":         str(Path(OPN_DIR).resolve()),
        "output_dir":      str(run_dir),
        "frequencies_ghz": list(FREQUENCIES_GHZ),
        "aspects_deg":     aspects,
        "polarizations":   pols,
        "expand_to_360":   False,
        "unit_output_dir": "results/by_frequency",
        "n_nodes":         int(N_NODES),
        "n_jobs":          int(N_JOBS),
        "n_slots":         int(N_NODES) * int(N_JOBS),
        "n_units":         len(units),
        "resource_planning_method": "bor-frequency-specific-v1",
        "solver_source_sha256": _solver_source_fingerprint(),
        "solver_source_inventory": _solver_source_inventory(),
        "runtime_environment_sha256":
            runtime_environment_fingerprint(),
        "solver_config": {
            "geometry_units":          GEOMETRY_UNITS,
            "cfie_alpha":              CFIE_ALPHA,
            "n_modes":                 N_MODES,
            "mode_tol":                MODE_TOL,
            "max_elements":            MAX_ELEMENTS,
            "assembly":                ASSEMBLY,
            "table_precision":         TABLE_PRECISION,
            "stream_budget_gb":        STREAM_BUDGET_GB,
            "mesh_certification":      bool(MESH_CERTIFICATION),
            "workers_per_unit":        WORKERS_PER_UNIT,
            "blas_threads_per_worker": BLAS_THREADS_PER_WORKER,
            "cores_per_node":          CORES_PER_NODE,
        },
        "radar_grid": {
            "azimuths_deg":   [float(value) for value in AZIMUTHS_DEG],
            "elevations_deg": [float(value) for value in ELEVATIONS_DEG],
            "frequencies_ghz": frequencies,
            "axis_az_deg":    float(BODY_AXIS_AZ_DEG),
            "axis_el_deg":    float(BODY_AXIS_EL_DEG),
            "roll_deg":       float(BODY_ROLL_DEG),
        },
        "units": units,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    script_path = run_dir / "driver_configured.py"
    shutil.copy2(str(source_driver), str(script_path))
    slurm_paths = []  # type: List[Path]
    for j in range(int(N_JOBS)):
        sp = run_dir / f"submit_job{j}.slurm"
        sp.write_text(_build_slurm(script_path, run_dir, j))
        sp.chmod(0o755)
        slurm_paths.append(sp)

    print("=" * 70)
    print("HPC monostatic BoR RCS sweep")
    print("=" * 70)
    print(f"  Run dir       : {run_dir}")
    print(f"  Geometries    : {len(geometries)}")
    print(f"  Polarizations : VV, HH (co-solved; VH also published)")
    print(f"  Frequencies   : {len(FREQUENCIES_GHZ)}  "
          f"({min(FREQUENCIES_GHZ):g}-{max(FREQUENCIES_GHZ):g} GHz)")
    print(f"  Radar grid    : {len(AZIMUTHS_DEG)} az x "
          f"{len(ELEVATIONS_DEG)} el")
    print(f"  Solver aspects: {len(aspects)} exact body-aspect RHS columns")
    print(f"  Deliverables  : {len(geometries)} monostatic GRIM(s)")
    print(f"  Physical solves: {len(_paired_solve_units(units))}  (VV/HH co-solved)")
    print(f"  Mesh check    : {'base + fine comparison' if MESH_CERTIFICATION else 'base only (no mesh comparison)'}")
    print(f"  Slots         : {N_JOBS} job(s) x {N_NODES} node(s)")
    print(f"  Per unit      : {WORKERS_PER_UNIT} threads (modes + streaming "
          f"tiles), assembly={ASSEMBLY}, precision={TABLE_PRECISION}")
    reservations = [
        float(record["estimated_peak_gb"])
        for record in resource_estimates.values()
    ]
    print(f"  Memory reserve: {min(reservations):.2f}-"
          f"{max(reservations):.2f} GB per physical solve")
    print(f"  Slurm scripts : {len(slurm_paths)} files in {run_dir}")

    if not SUBMIT or shutil.which("sbatch") is None:
        why = "SUBMIT=False" if not SUBMIT else "[warn] sbatch not on PATH"
        print(f"\n  {why} -- submit manually with:")
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
    print(f"Monostatic outputs in: {run_dir}/results/")


# --- worker mode (invoked by SLURM) ----------------------------------------

def _unit_claim_key(unit):
    # type: (Dict[str, Any]) -> str
    channels = "+".join(
        str(channel["polarization"]) for channel in unit["channel_units"]
    )
    return (f"{channels}_{float(unit['frequency_ghz']):.3f}GHz_"
            f"{unit['geometry_stem']}")


def _plan_units(units, n_slots, manifest_aspects, geometry_units):
    # type: (List[Dict[str, Any]], int, List[float], str) -> Tuple[Dict[str, float], Dict[str, int]]
    """Cost every unit and deal them out longest-processing-time-first.

    Extents are read straight off the .geo, so the whole plan costs one parse
    per geometry rather than a trial solve.
    """

    extents = {}  # type: Dict[str, Tuple[float, float]]
    records = []  # type: List[Dict[str, Any]]
    for unit in units:
        path = str(unit["geometry"])
        if path not in extents:
            extents[path] = hpc_scheduler.predict_bor_extent(
                path, geometry_units
            )
        arc, radius = extents[path]
        records.append({
            "unit": _unit_claim_key(unit),
            "cost": hpc_scheduler.bor_unit_cost(
                arc, radius, float(unit["frequency_ghz"]),
                len(manifest_aspects),
            ),
        })
    assignment = hpc_scheduler.balance_units(records, n_slots)
    costs = {r["unit"]: float(r["cost"]) for r in records}
    slots = {r["unit"]: int(slot) for r, slot in zip(records, assignment)}
    return costs, slots


def worker(run_dir_str, job_index, node_index):
    # type: (str, int, int) -> None
    hpc_scheduler.install_fingerprint_cache()
    from geometry_io import parse_geometry, build_geometry_snapshot

    run_dir  = Path(run_dir_str).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    _verify_run_provenance(manifest)
    solver_config = dict(manifest.get("solver_config", {}) or {})
    blas_threads = int(solver_config.get(
        "blas_threads_per_worker", BLAS_THREADS_PER_WORKER
    ))
    workers_per_unit = int(solver_config.get(
        "workers_per_unit", WORKERS_PER_UNIT
    ))
    geometry_units = str(solver_config.get(
        "geometry_units", GEOMETRY_UNITS
    ))
    _pin_blas_threads(blas_threads)
    output_units = manifest["units"]
    units = _paired_solve_units(output_units)
    n_nodes  = int(manifest.get("n_nodes", 1))
    n_jobs   = int(manifest.get("n_jobs", 1))
    n_slots  = max(1, n_nodes * n_jobs)
    slot_id  = job_index * n_nodes + node_index

    costs, slots = _plan_units(
        units, n_slots, manifest["aspects_deg"], geometry_units
    )
    mine_keys = {k for k, v in slots.items() if v == slot_id}

    def _order_key(unit):
        key = _unit_claim_key(unit)
        return (-costs.get(key, 1.0), key)

    mine = [u for u in units if _unit_claim_key(u) in mine_keys]
    others = [u for u in units if _unit_claim_key(u) not in mine_keys]
    # Planned share first (dearest first), then everyone else's as a steal
    # pool. Array tasks are interchangeable: nothing is stranded when a task is
    # cancelled, preempted, or never scheduled.
    candidates = sorted(mine, key=_order_key) + sorted(others, key=_order_key)

    cores    = _detect_cores()
    memory_gb = hpc_scheduler.detect_memory_gb()
    budget_gb = max(1.0, memory_gb * float(MEMORY_HEADROOM))
    # BoR units are internally threaded: divide the node into pool workers of
    # WORKERS_PER_UNIT threads each.
    by_threads = max(1, cores // max(1, workers_per_unit))
    worker_cap = by_threads if MAX_WORKERS_PER_NODE is None else \
        max(1, min(by_threads, int(MAX_WORKERS_PER_NODE)))
    pool_size = max(1, min(worker_cap, len(candidates))) if candidates else 1

    print("=" * 70)
    print(f"  Slot {slot_id}/{n_slots - 1}  "
          f"(job={job_index}, node={node_index})")
    print(f"  Physical solves: {len(units)}   planned here: {len(mine)}")
    print(f"  Output files: {len(output_units)}")
    print(f"  Cores: {cores}   pool: {pool_size} x {workers_per_unit} threads "
          f"(BLAS/worker: {blas_threads})")
    print(f"  Memory: {memory_gb:.1f} GB allocated, {budget_gb:.1f} GB "
          "schedulable with per-solve admission")
    print("=" * 70, flush=True)

    if not candidates:
        print("  Nothing to do.")
        return

    snapshots = {}  # type: Dict[str, Tuple[Dict[str, Any], str]]
    from feature_sum import geometry_input_fingerprint
    for u in candidates:
        gpath = u["geometry"]
        if gpath in snapshots:
            continue
        p = Path(gpath)
        if not p.is_file():
            sys.exit(f"Geometry missing on compute node: {p}")
        expected_input = str(u.get("geometry_input_sha256", ""))
        actual_input = geometry_input_fingerprint(
            str(p), str(manifest["solver_config"]["geometry_units"])
        )
        if not expected_input or actual_input != expected_input:
            sys.exit(
                f"Frozen geometry/material input fingerprint mismatch for "
                f"{p}; no field was solved."
            )
        title, segments, ibcs, dielectrics = parse_geometry(p.read_text())
        snap = build_geometry_snapshot(title, segments, ibcs, dielectrics)
        snap["source_path"] = str(p)
        snapshots[gpath] = (snap, str(p.parent))

    unplanned = [
        unit for unit in candidates if "estimated_peak_gb" not in unit
    ]
    if unplanned:
        from bor_dispatch import estimate_bor_resources
    for unit in unplanned:
        snapshot, material_base = snapshots[unit["geometry"]]
        resource_estimate = estimate_bor_resources(
            snapshot,
            float(unit["frequency_ghz"]),
            [float(value) for value in manifest["aspects_deg"]],
            geometry_units=str(solver_config.get(
                "geometry_units", GEOMETRY_UNITS
            )),
            material_base_dir=material_base,
            n_modes=solver_config.get("n_modes", N_MODES),
            max_elements=int(solver_config.get(
                "max_elements", MAX_ELEMENTS
            )),
            workers=int(solver_config.get(
                "workers_per_unit", WORKERS_PER_UNIT
            )),
            table_precision=str(solver_config.get(
                "table_precision", TABLE_PRECISION
            )),
            assembly=str(solver_config.get("assembly", ASSEMBLY)),
            stream_budget_gb=float(solver_config.get(
                "stream_budget_gb", STREAM_BUDGET_GB
            )),
            mesh_certification=bool(solver_config.get(
                "mesh_certification", True
            )),
        )
        unit["estimated_peak_gb"] = float(
            resource_estimate["estimated_peak_gb"]
        )
    reservations = [
        float(unit["estimated_peak_gb"])
        for unit in candidates
    ]
    print(
        f"  Per-solve memory reservations: {min(reservations):.2f}-"
        f"{max(reservations):.2f} GB",
        flush=True,
    )

    broker = hpc_scheduler.ClaimBroker(
        run_dir / "claims", stale_seconds=float(CLAIM_STALE_SECONDS)
    )
    broker.start_heartbeat()

    t0 = time.time()
    counters = {"written": 0, "skipped": 0, "failed": 0, "passed": 0}
    total = len(candidates)

    def _prepare(unit):
        key = _unit_claim_key(unit)
        dispatch = (
            key,
            float(unit["estimated_peak_gb"]),
            (_solve_and_export_star,
             ((unit, snapshots[unit["geometry"]][0],
               snapshots[unit["geometry"]][1], str(run_dir)),)),
        )
        all_outputs_exist = all(
            _unit_output_path(run_dir, channel).exists()
            for channel in unit["channel_units"]
        )
        if all_outputs_exist:
            # Verify the existing output's attestation rather than trusting the
            # filename, but only on the slot that owns the unit, so each result
            # is re-checked once per run instead of once per node.
            if key not in mine_keys:
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
        kind, a, b, u = payload
        channels = "+".join(
            str(channel["polarization"]) for channel in u["channel_units"]
        )
        tag = (f"{channels} {u['frequency_ghz']:7.3f}GHz "
               f"{u['geometry_stem']}")
        if kind == "ok":
            counters["skipped" if a == "skipped" else "written"] += 1
            broker.release(key)
            print(f"  [{_done():3d}/{total}] {a:7s}  {tag}  -> "
                  f"{b}", flush=True)
        else:
            counters["failed"] += 1
            broker.abandon(key)
            print(f"  [{_done():3d}/{total}] FAILED   {tag}", flush=True)
            for line in str(a).rstrip().splitlines():
                print(f"      {line}", flush=True)

    def _on_error(key, exc):
        counters["failed"] += 1
        broker.abandon(key)
        print(f"  [{_done():3d}/{total}] FAILED (dispatch) {key}: {exc!r}",
              flush=True)

    with Pool(processes=pool_size,
              initializer=_pool_initializer,
              initargs=(blas_threads,),
              maxtasksperchild=int(TASKS_PER_CHILD)) as pool:
        dispatcher = hpc_scheduler.MemoryAwareDispatcher(
            pool, budget_gb=budget_gb, max_concurrent=pool_size
        )
        try:
            dispatcher.run(candidates, _prepare, _on_result, _on_error)
        finally:
            broker.stop_heartbeat()

    elapsed = time.time() - t0
    print(f"\n  Slot complete. wrote={counters['written']}, "
          f"skipped={counters['skipped']}, failed={counters['failed']}, "
          f"left to other tasks={counters['passed']}.  {elapsed:.1f} s elapsed.")
    if counters["failed"]:
        raise SystemExit(1)
    try:
        n_written, n_skipped = publish_monostatic(
            str(run_dir), require_complete=False
        )
        if n_written or n_skipped:
            print(
                f"  Monostatic GRIMs: {n_written} written, {n_skipped} "
                f"already present in {run_dir / 'results'}/",
                flush=True,
            )
    except Exception:
        print("      [warn] monostatic publication failed:", flush=True)
        for line in traceback.format_exc().rstrip().splitlines():
            print(f"        {line}", flush=True)


def _pool_initializer(blas_threads):
    # type: (int) -> None
    hpc_scheduler.pin_blas_threads(blas_threads)
    hpc_scheduler.install_fingerprint_cache()


# --- entry point -----------------------------------------------------------

def main():
    # type: () -> None
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "--worker", nargs=3, metavar=("RUN_DIR", "JOB_INDEX", "NODE_INDEX"),
        help="Internal: execute one array-task slice. Invoked by SLURM.",
    )
    ap.add_argument(
        "--publish", metavar="RUN_DIR",
        help="Publish/rebuild the one monostatic GRIM per geometry after a "
             "completed solve.",
    )
    args = ap.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]), int(args.worker[2]))
    elif args.publish:
        written, skipped = publish_monostatic(args.publish)
        print(
            f"Monostatic GRIMs: {written} written, {skipped} already present."
        )
    else:
        submit()


if __name__ == "__main__":
    main()
