#!/usr/bin/env python3
"""
HPC monostatic BoR RCS sweep driver (SLURM) -- the body-of-revolution
counterpart of run_hpc_monostatic.py.

Edit the CONFIG block below and run:

    python run_hpc_bor_monostatic.py

Workflow (mirrors the 2D driver):
- Discover geometry files under FRD_DIR + OPN_DIR (BoR .geo files: x = rho,
  y = z, generatrices traversed +z -> -z; see bor_dispatch).
- Expand into a (geometry x frequency x polarization) unit list. All ASPECT
  angles for a unit are solved in one call (each azimuthal mode is factored
  once; extra aspects are RHS columns).
- Distribute units round-robin across N_NODES x N_JOBS slots, write sbatch
  job-array scripts, submit. Restartable: units whose .grim exists are
  skipped.
- Each unit exports "<POL>_<FREQ:.3f>GHz_<geometry_stem>.grim" (sigma_3d,
  dBsm) with the complex far-field amplitudes preserved.

BoR-specific notes:
- A BoR unit parallelizes INTERNALLY (threads across azimuthal modes and
  streaming-assembly tiles), so the node is divided into a few workers with
  several threads each (WORKERS_PER_UNIT) instead of one process per core.
- The dispatch auto-selects table vs streaming assembly and single/double
  precision by memory estimate; override with ASSEMBLY / TABLE_PRECISION /
  STREAM_BUDGET_GB if needed.
- EXPAND_TO_360 mirrors each aspect sweep about the axis (exact for a BoR).

Radar-frame (azimuth, elevation) polarimetric grids -- VV/HH/VH -- are built
AUTOMATICALLY during the sweep: as each (geometry, frequency) pair's second
polarization finishes, that worker writes the pair's az/el grids to
<run_dir>/azel/ (AZEL_ENABLE / AZEL_* in the config).  A manual backfill
CLI exists for re-runs:  python run_hpc_bor_monostatic.py --azel <run_dir>

Internal worker invocation (called by SLURM, not by the user):
    python run_hpc_bor_monostatic.py --worker <run_dir> <job_index> <node_index>
"""

import argparse
import json
import math
import os
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
    sha256_file,
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

# Requested sweep.  ASPECTS are angles from the +z rotation axis (0 =
# nose-on, 90 = broadside, 180 = tail-on).  [0, 180] fully characterizes the
# monostatic response; EXPAND_TO_360 fills the mirrored half in the export.
FREQUENCIES_GHZ = [1.0, 2.0, 3.0]
ASPECTS_DEG     = [float(a) for a in range(0, 181, 3)]
POLARIZATIONS   = ["VV", "HH"]          # keep both if you want --azel grids
EXPAND_TO_360   = False

# Output root. A new run_YYYYMMDD_HHMMSS/ subfolder is created inside.
OUTPUT_DIR = "rcs_runs_bor"

# --- Multi-node / multi-submission parallelism -----------------------------
N_NODES = 1
N_JOBS  = 1

# ===============================================================================
# ADVANCED -- fine tuning (SLURM resources, solver knobs, az/el product)
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
WORKERS_PER_UNIT        = 4              # threads inside one BoR solve (modes
                                         # + streaming tiles); pool size =
                                         # cores // WORKERS_PER_UNIT
BLAS_THREADS_PER_WORKER = 1

# --- Radar-frame (az, el) product ------------------------------------------
# Built AUTOMATICALLY inside the workers: whenever a unit finishes and its
# partner polarization's .grim already exists, that worker writes the
# VV/HH/VH az/el grids for the (geometry, frequency) pair into
# <run_dir>/azel/ (atomic renames, so two nodes racing on the same pair are
# safe; whichever finishes second does the work).  Requires both "VV" and
# "HH" in POLARIZATIONS.  The `--azel <run_dir>` CLI remains only as a
# manual backfill / re-run (e.g. after editing the grid below).
AZEL_ENABLE         = True
AZEL_AZIMUTHS_DEG   = [float(a) for a in range(0, 360, 5)]
AZEL_ELEVATIONS_DEG = [float(e) for e in range(-60, 61, 5)]
AZEL_AXIS_AZ_DEG    = 0.0     # target axis orientation in the earth frame
AZEL_AXIS_EL_DEG    = 0.0     # horizontal, nose toward azimuth 0.  NOTE the
                              # polarization label mapping for horizontal
                              # axes (bor_az_el_grid docstring).

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
    return run_dir / "results" / f"{pol}_{freq:.3f}GHz_{stem}.grim"


def _azel_out_paths(run_dir, stem, freq):
    # type: (Path, str, float) -> List[Path]
    return [run_dir / "azel" / f"azel_{freq:.3f}GHz_{stem}_{ch}.grim"
            for ch in ("VV", "HH", "VH")]


def _azel_attestation_fields(
    manifest, stem, freq, channel, azel_cfg, vv_path, hh_path
):
    # type: (Dict[str, Any], str, float, str, Dict[str, Any], Path, Path) -> Dict[str, Any]
    return {
        "run_id": str(manifest["run_id"]),
        "solver_source_sha256": str(manifest["solver_source_sha256"]),
        "runtime_environment_sha256":
            str(manifest["runtime_environment_sha256"]),
        "run_solve_spec_sha256":
            manifest_solve_spec_fingerprint(manifest),
        "derived_product": "bor_az_el_grid",
        "geometry_stem": str(stem),
        "frequency_ghz": float(freq),
        "polarization": str(channel),
        "azel_config_sha256": stable_json_fingerprint(azel_cfg),
        "source_vv_sha256": sha256_file(str(vv_path)),
        "source_hh_sha256": sha256_file(str(hh_path)),
    }


def _build_azel_pair(run_dir, stem, freq, azel_cfg):
    # type: (Path, str, float, Dict[str, Any]) -> bool
    """Build the radar-frame az/el grids for one (geometry, frequency) pair
    from its VV + HH .grim exports.  Idempotent; atomic via temp + rename so
    concurrent attempts from different nodes cannot interleave writes."""
    outs = _azel_out_paths(run_dir, stem, freq)
    vv = run_dir / "results" / f"VV_{freq:.3f}GHz_{stem}.grim"
    hh = run_dir / "results" / f"HH_{freq:.3f}GHz_{stem}.grim"
    if not (vv.exists() and hh.exists()):
        return False
    manifest = _manifest_for(run_dir)
    for polarization, path in (("VV", vv), ("HH", hh)):
        matching = [
            unit for unit in manifest.get("units", [])
            if unit.get("geometry_stem") == stem
            and unit.get("polarization") == polarization
            and float(unit.get("frequency_ghz")) == float(freq)
        ]
        if len(matching) != 1:
            raise ValueError(
                f"Cannot identify one {polarization} unit attestation for "
                f"{stem} at {freq:g} GHz."
            )
        verify_embedded_attestation(
            str(path), _unit_attestation_fields(manifest, matching[0])
        )

    expected_by_path = {
        path: _azel_attestation_fields(
            manifest, stem, freq, channel, azel_cfg, vv, hh
        )
        for path, channel in zip(outs, ("VV", "HH", "VH"))
    }
    if all(path.exists() for path in outs):
        try:
            for path in outs:
                verify_embedded_attestation(
                    str(path), expected_by_path[path]
                )
            return False
        except ValueError:
            # Changed grid/config/source or edited bytes: rebuild the complete
            # three-channel derived product below.
            pass

    from bor_dispatch import bor_az_el_grid
    from grim_io import save_bor_az_el_grim
    rv = _result_from_grim(vv, "VV")
    rh = _result_from_grim(hh, "HH")
    grid = bor_az_el_grid(
        rv, rh, azel_cfg["azimuths_deg"], azel_cfg["elevations_deg"],
        axis_az_deg=float(azel_cfg["axis_az_deg"]),
        axis_el_deg=float(azel_cfg["axis_el_deg"]))
    out_dir = run_dir / "azel"
    out_dir.mkdir(exist_ok=True)
    tmp_stem = out_dir / f".tmp_{os.getpid()}_{freq:.3f}GHz_{stem}.grim"
    # Each channel carries its own attestation inside the file it is written
    # into, so the derived products need no sidecars either.
    channel_metadata = {
        channel: {"output_attestation": dict(
            expected_by_path[path],
            schema="ghost.workflow.embedded-attestation.v1",
        )}
        for path, channel in zip(outs, ("VV", "HH", "VH"))
    }
    written = save_bor_az_el_grim(
        grid, str(tmp_stem), source_path=stem,
        history=f"run_hpc_bor_monostatic.py azel {freq}GHz",
        channel_metadata=channel_metadata)
    for tmp, final in zip(sorted(written), sorted(str(p) for p in outs)):
        os.replace(tmp, final)
    return True


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


def _solve_and_export(unit, snapshot, material_base, run_dir_str):
    # type: (Dict[str, Any], Dict[str, Any], str, str) -> Tuple[str, str]
    """Pool-worker entry point: solve one BoR unit, export .grim, and (when
    its partner polarization is already done) write the pair's az/el
    product. Idempotent."""
    run_dir = Path(run_dir_str)
    out_path = _unit_output_path(run_dir, unit)
    manifest = _manifest_for(run_dir)
    _verify_run_provenance(manifest)
    _verify_unit_input(unit, manifest)
    attestation = _unit_attestation_fields(manifest, unit)
    if out_path.exists():
        verify_embedded_attestation(str(out_path), attestation)
        return ("skipped", str(out_path))

    from bor_dispatch import solve_monostatic_rcs_bor
    result = solve_monostatic_rcs_bor(
        geometry_snapshot=snapshot,
        frequencies_ghz=[float(unit["frequency_ghz"])],
        elevations_deg=[float(a) for a in manifest["aspects_deg"]],
        polarization=unit["polarization"],
        geometry_units=GEOMETRY_UNITS,
        material_base_dir=material_base,
        cfie_alpha=CFIE_ALPHA,
        n_modes=N_MODES,
        mode_tol=MODE_TOL,
        max_elements=MAX_ELEMENTS,
        workers=WORKERS_PER_UNIT,
        table_precision=TABLE_PRECISION,
        assembly=ASSEMBLY,
        stream_budget_gb=STREAM_BUDGET_GB,
        expand_to_360=EXPAND_TO_360,
    )
    for w in result["metadata"].get("warnings", []) or []:
        print(f"      [warn] {unit['geometry_stem']}: {w}", flush=True)
    _verify_run_provenance(manifest)
    _verify_unit_input(unit, manifest)

    # Bind the result to its run state inside the artifact, before export, so
    # results/ holds one file per unit instead of a .grim and a sidecar.
    embed_output_attestation(result, attestation)

    from grim_io import export_result_to_grim
    written = export_result_to_grim(
        result, str(out_path),
        source_path=str(snapshot.get("source_path", "") or ""),
        history=(f"run_hpc_bor_monostatic.py pol={unit['polarization']} "
                 f"freq={unit['frequency_ghz']}GHz"),
    )
    actual_path = str(written[0]) if written else str(out_path)
    _verify_run_provenance(manifest)
    _verify_unit_input(unit, manifest)

    # az/el product: whichever polarization of the pair finishes second
    # builds it (config from the manifest, so all nodes agree).
    if AZEL_ENABLE:
        try:
            cfg = manifest.get("azel_config") or {}
            if cfg and _build_azel_pair(run_dir, unit["geometry_stem"],
                                        float(unit["frequency_ghz"]), cfg):
                print(f"      azel grids written for {unit['geometry_stem']} "
                      f"{unit['frequency_ghz']:.3f}GHz", flush=True)
        except Exception:
            print("      [warn] azel product failed:", flush=True)
            for line in traceback.format_exc().rstrip().splitlines():
                print(f"        {line}", flush=True)

    return ("written", actual_path)


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

    pols = [p.strip().upper() for p in POLARIZATIONS if p and p.strip()]
    if not pols:            sys.exit("ERROR: POLARIZATIONS is empty.")
    if not FREQUENCIES_GHZ: sys.exit("ERROR: FREQUENCIES_GHZ is empty.")
    if not ASPECTS_DEG:     sys.exit("ERROR: ASPECTS_DEG is empty.")
    if (
        len(set(pols)) != len(pols)
        or not set(pols).issubset({"VV", "HH"})
    ):
        sys.exit("ERROR: POLARIZATIONS must be a unique subset of VV/HH.")
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
    aspects = [float(value) for value in ASPECTS_DEG]
    if (
        not all(math.isfinite(value) for value in aspects)
        or len(set(aspects)) != len(aspects)
    ):
        sys.exit("ERROR: ASPECTS_DEG must be finite and unique.")
    if any(a < 0.0 or a > 180.0 for a in aspects):
        sys.exit("ERROR: BoR aspects must lie in [0, 180] deg from the +z "
                 "axis (use EXPAND_TO_360 for the mirrored half).")
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
        "aspects_deg":     list(ASPECTS_DEG),
        "polarizations":   pols,
        "expand_to_360":   bool(EXPAND_TO_360),
        "n_nodes":         int(N_NODES),
        "n_jobs":          int(N_JOBS),
        "n_slots":         int(N_NODES) * int(N_JOBS),
        "n_units":         len(units),
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
            "workers_per_unit":        WORKERS_PER_UNIT,
            "blas_threads_per_worker": BLAS_THREADS_PER_WORKER,
            "cores_per_node":          CORES_PER_NODE,
        },
        "azel_config": {
            "enabled":        bool(AZEL_ENABLE),
            "azimuths_deg":   list(AZEL_AZIMUTHS_DEG),
            "elevations_deg": list(AZEL_ELEVATIONS_DEG),
            "axis_az_deg":    AZEL_AXIS_AZ_DEG,
            "axis_el_deg":    AZEL_AXIS_EL_DEG,
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
    print(f"  Polarizations : {', '.join(pols)}")
    print(f"  Frequencies   : {len(FREQUENCIES_GHZ)}  "
          f"({min(FREQUENCIES_GHZ):g}-{max(FREQUENCIES_GHZ):g} GHz)")
    print(f"  Aspects       : {len(ASPECTS_DEG)}  (0-180 from the axis"
          f"{', mirrored to 360 on export' if EXPAND_TO_360 else ''})")
    print(f"  Units total   : {len(units)}  (geom x freq x pol)")
    print(f"  Slots         : {N_JOBS} job(s) x {N_NODES} node(s)")
    print(f"  Per unit      : {WORKERS_PER_UNIT} threads (modes + streaming "
          f"tiles), assembly={ASSEMBLY}, precision={TABLE_PRECISION}")
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
    print(f"Outputs in:    {run_dir}/results/"
          + (f"  (+ az/el grids in {run_dir}/azel/)" if AZEL_ENABLE else ""))


# --- worker mode (invoked by SLURM) ----------------------------------------

def _unit_claim_key(unit):
    # type: (Dict[str, Any]) -> str
    return (f"{unit['polarization']}_{float(unit['frequency_ghz']):.3f}GHz_"
            f"{unit['geometry_stem']}")


def _plan_units(units, n_slots, manifest_aspects):
    # type: (List[Dict[str, Any]], int, List[float]) -> Tuple[Dict[str, float], Dict[str, int]]
    """Cost every unit and deal them out longest-processing-time-first.

    Extents are read straight off the .geo, so the whole plan costs one parse
    per geometry rather than a trial solve.
    """

    extents = {}  # type: Dict[str, Tuple[float, float]]
    records = []  # type: List[Dict[str, Any]]
    for unit in units:
        path = str(unit["geometry"])
        if path not in extents:
            extents[path] = hpc_scheduler.predict_bor_extent(path, GEOMETRY_UNITS)
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
    _pin_blas_threads(BLAS_THREADS_PER_WORKER)
    hpc_scheduler.install_fingerprint_cache()
    from geometry_io import parse_geometry, build_geometry_snapshot

    run_dir  = Path(run_dir_str).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    _verify_run_provenance(manifest)
    units    = manifest["units"]
    n_nodes  = int(manifest.get("n_nodes", 1))
    n_jobs   = int(manifest.get("n_jobs", 1))
    n_slots  = max(1, n_nodes * n_jobs)
    slot_id  = job_index * n_nodes + node_index

    costs, slots = _plan_units(units, n_slots, manifest["aspects_deg"])
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
    by_threads = max(1, cores // max(1, WORKERS_PER_UNIT))
    worker_cap = by_threads if MAX_WORKERS_PER_NODE is None else \
        max(1, min(by_threads, int(MAX_WORKERS_PER_NODE)))
    pool_size = max(1, min(worker_cap, len(candidates))) if candidates else 1

    print("=" * 70)
    print(f"  Slot {slot_id}/{n_slots - 1}  "
          f"(job={job_index}, node={node_index})")
    print(f"  Units in run: {len(units)}   planned here: {len(mine)}")
    print(f"  Cores: {cores}   pool: {pool_size} x {WORKERS_PER_UNIT} threads "
          f"(BLAS/worker: {BLAS_THREADS_PER_WORKER})")
    print(f"  Memory: {memory_gb:.1f} GB allocated, {budget_gb:.1f} GB "
          f"schedulable at {STREAM_BUDGET_GB:g} GB reserved per unit")
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
            key, float(STREAM_BUDGET_GB),
            (_solve_and_export_star,
             ((unit, snapshots[unit["geometry"]][0],
               snapshots[unit["geometry"]][1], str(run_dir)),)),
        )
        if _unit_output_path(run_dir, unit).exists():
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
        tag = (f"{u['polarization']} {u['frequency_ghz']:7.3f}GHz "
               f"{u['geometry_stem']}")
        if kind == "ok":
            counters["skipped" if a == "skipped" else "written"] += 1
            broker.release(key)
            print(f"  [{_done():3d}/{total}] {a:7s}  {tag}  -> "
                  f"{Path(b).name}", flush=True)
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
              initargs=(BLAS_THREADS_PER_WORKER,),
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


def _pool_initializer(blas_threads):
    # type: (int) -> None
    hpc_scheduler.pin_blas_threads(blas_threads)
    hpc_scheduler.install_fingerprint_cache()


# --- az/el post-processing mode (login node, after the sweep) --------------

def _result_from_grim(path, pol):
    # type: (Path, str) -> Dict[str, Any]
    """Reconstruct the minimal result dict bor_az_el_grid needs from a .grim
    (the exports preserve the complex far-field amplitudes)."""
    import numpy as np
    d = np.load(str(path))
    # NpzFile.get() only exists on numpy >= 1.25; use .files membership so
    # this runs on older HPC numpy builds too.
    if ("raw_complex_amplitude_preserved" not in getattr(d, "files", [])
            or not bool(d["raw_complex_amplitude_preserved"])):
        raise ValueError(f"{path.name}: complex amplitudes were not preserved.")
    samples = []
    az = d["azimuths"]           # aspect angles for BoR exports
    freqs = d["frequencies"]
    ar, ai = d["rcs_amp_real"], d["rcs_amp_imag"]
    for i, th in enumerate(az):
        if float(th) > 180.0:
            continue             # skip the EXPAND_TO_360 mirror half
        for kf, f in enumerate(freqs):
            samples.append({
                "frequency_ghz": float(f),
                "theta_inc_deg": float(th),
                "theta_scat_deg": float(th),
                "rcs_amp_real": float(ar[i, 0, kf, 0]),
                "rcs_amp_imag": float(ai[i, 0, kf, 0]),
                "rcs_linear": float(d["rcs_power"][i, 0, kf, 0]),
                "rcs_db": 10.0 * math.log10(max(float(d["rcs_power"][i, 0, kf, 0]), 1e-30)),
                "rcs_amp_phase_deg": 0.0,
            })
    return {"polarization": pol, "samples": samples}


def azel(run_dir_str):
    # type: (str) -> None
    """Manual backfill: normally unnecessary -- the workers build each pair's
    az/el product automatically as the second polarization finishes.  Use
    this only to (re)build after editing the AZEL_* grid in the manifest, or
    if AZEL_ENABLE was off during the sweep."""

    run_dir  = Path(run_dir_str).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = dict(manifest.get("azel_config") or {})
    cfg.setdefault("azimuths_deg", AZEL_AZIMUTHS_DEG)
    cfg.setdefault("elevations_deg", AZEL_ELEVATIONS_DEG)
    cfg.setdefault("axis_az_deg", AZEL_AXIS_AZ_DEG)
    cfg.setdefault("axis_el_deg", AZEL_AXIS_EL_DEG)

    keys = sorted({(u["geometry_stem"], float(u["frequency_ghz"]))
                   for u in manifest["units"]})
    n_done = n_skip = 0
    for stem, freq in keys:
        if _build_azel_pair(run_dir, stem, freq, cfg):
            n_done += 1
            print(f"  {stem} {freq:.3f}GHz -> azel_{freq:.3f}GHz_{stem}_[VV|HH|VH].grim")
        else:
            n_skip += 1
    print(f"\n  az/el grids: {n_done} written, {n_skip} skipped (already "
          f"built or missing a polarization).  Outputs: {run_dir / 'azel'}/")


# --- entry point -----------------------------------------------------------

def main():
    # type: () -> None
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "--worker", nargs=3, metavar=("RUN_DIR", "JOB_INDEX", "NODE_INDEX"),
        help="Internal: execute one array-task slice. Invoked by SLURM.",
    )
    ap.add_argument(
        "--azel", metavar="RUN_DIR",
        help="Post-process a completed run into radar-frame (az, el) "
             "polarimetric grids (needs both VV and HH units).",
    )
    args = ap.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]), int(args.worker[2]))
    elif args.azel:
        azel(args.azel)
    else:
        submit()


if __name__ == "__main__":
    main()
