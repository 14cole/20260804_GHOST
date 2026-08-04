"""Shared implementation for the simple local and SLURM step-1 runners."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Iterable

import numpy as np

from feature_sum import geometry_input_fingerprint
from solver_quality import (
    evaluate_mesh_convergence,
    scale_snapshot_panel_density,
    validate_mesh_convergence_policy,
)
from workflow_provenance import (
    backend_source_fingerprint,
    runtime_environment_fingerprint,
    stable_json_fingerprint,
)


UNIT_SCHEMA = "ghost.workflow.2d-unit.v1"


def validate_config(
    frequencies_ghz: 'Iterable[float]',
    angles_deg: 'Iterable[float]',
    polarizations: 'Iterable[str]',
) -> 'tuple[list[float], list[float], list[str]]':
    frequencies = [float(value) for value in frequencies_ghz]
    angles = [float(value) for value in angles_deg]
    pols = [str(value).strip().upper() for value in polarizations]
    if (
        not frequencies
        or not all(math.isfinite(value) and value > 0.0 for value in frequencies)
        or len(set(frequencies)) != len(frequencies)
        or len({f"{value:.3f}" for value in frequencies}) != len(frequencies)
    ):
        raise ValueError(
            "FREQUENCIES_GHZ must be positive, finite, unique, and distinct "
            "at the 0.001 GHz output-name precision."
        )
    if (
        not angles
        or not all(math.isfinite(value) for value in angles)
        or len(set(angles)) != len(angles)
    ):
        raise ValueError("ANGLES_DEG must be non-empty, finite, and unique.")
    if len(pols) != 2 or set(pols) != {"TM", "TE"}:
        raise ValueError(
            "POLARIZATIONS must contain exactly TM and TE; a complete feature "
            "delta needs both physical channels."
        )
    return frequencies, angles, pols


def discover_jobs(
    step_dir: 'str | os.PathLike[str]',
    frequencies_ghz: 'Iterable[float]',
    polarizations: 'Iterable[str]',
) -> 'list[dict[str, Any]]':
    """Discover both role folders. Either role may be empty."""
    root = Path(step_dir).resolve()
    jobs: 'list[dict[str, Any]]' = []
    output_names: 'set[tuple[str, str]]' = set()
    for role in ("FRD", "OPN"):
        geometry_dir = root / "geometries" / role
        geometry_dir.mkdir(parents=True, exist_ok=True)
        (root / "results" / role).mkdir(parents=True, exist_ok=True)
        for geometry in sorted(geometry_dir.glob("*.geo")):
            base = geometry.stem
            if base.upper().endswith(("_FRD", "_OPN")):
                base = base[:-4]
            for polarization in polarizations:
                for frequency in frequencies_ghz:
                    name = (
                        f"{polarization}_{float(frequency):.3f}GHz_"
                        f"{base}_{role}.grim"
                    )
                    key = (role, name)
                    if key in output_names:
                        raise ValueError(
                            f"multiple inputs map to results/{role}/{name}"
                        )
                    output_names.add(key)
                    jobs.append(
                        {
                            "geometry": str(geometry.resolve()),
                            "role": role,
                            "base": base,
                            "polarization": str(polarization),
                            "frequency_ghz": float(frequency),
                            "output": str((root / "results" / role / name).resolve()),
                        }
                    )
    return jobs


def source_fingerprint(runner_path: 'str | os.PathLike[str]') -> 'str':
    backend = Path(__file__).resolve().parent
    return backend_source_fingerprint(
        str(backend), {"step1_runner.py": str(Path(runner_path).resolve())}
    )


def prepare_jobs(
    jobs: 'list[dict[str, Any]]',
    *,
    angles_deg: 'Iterable[float]',
    geometry_units: 'str',
    solver_method: 'str',
    max_panels: 'int',
    runner_path: 'str | os.PathLike[str]',
    mesh_convergence_policy: 'dict[str, Any]' = None,
) -> 'list[dict[str, Any]]':
    source_sha = source_fingerprint(runner_path)
    runtime_sha = runtime_environment_fingerprint()
    angles = [float(value) for value in angles_deg]
    mesh_policy = validate_mesh_convergence_policy(
        mesh_convergence_policy
    )
    prepared = []
    for original in jobs:
        job = dict(original)
        specification = {
            "schema": UNIT_SCHEMA,
            "geometry_input_sha256": geometry_input_fingerprint(
                job["geometry"], geometry_units
            ),
            "solver_source_sha256": source_sha,
            "runtime_environment_sha256": runtime_sha,
            "geometry_units": str(geometry_units).strip().lower(),
            "angles_deg": angles,
            "polarization": job["polarization"],
            "frequency_ghz": job["frequency_ghz"],
            "solver_method": str(solver_method).strip().lower(),
            "max_panels": int(max_panels),
            "mesh_convergence_policy": mesh_policy,
            "published_mesh": "fine",
        }
        specification["unit_sha256"] = stable_json_fingerprint(specification)
        job["specification"] = specification
        prepared.append(job)
    return prepared


def _stored_unit_sha(path: 'str | os.PathLike[str]') -> 'str':
    try:
        with np.load(path, allow_pickle=False) as payload:
            raw = np.asarray(payload["solver_metadata_json"]).reshape(()).item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        audit = json.loads(str(raw))
        return str(audit["metadata"]["workflow_unit"]["unit_sha256"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return ""


def _pin_blas(threads: 'int') -> 'None':
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = str(max(1, int(threads)))


# Parsed geometry snapshots, keyed by absolute path.  Filled in the parent
# before a pool forks, so workers inherit them copy-on-write instead of each
# re-reading and re-parsing the same .geo for every (polarization, frequency)
# unit built from it.
_SNAPSHOT_CACHE: 'dict[str, dict[str, Any]]' = {}


def load_snapshot(geometry_path: 'str | os.PathLike[str]') -> 'dict[str, Any]':
    """Geometry snapshot for one .geo, parsed at most once per process."""

    from geometry_io import build_geometry_snapshot, parse_geometry

    key = str(Path(geometry_path).resolve())
    cached = _SNAPSHOT_CACHE.get(key)
    if cached is not None:
        return cached
    title, segments, ibcs, dielectrics = parse_geometry(
        Path(key).read_text(encoding="utf-8")
    )
    snapshot = build_geometry_snapshot(title, segments, ibcs, dielectrics)
    snapshot["source_path"] = key
    _SNAPSHOT_CACHE[key] = snapshot
    return snapshot


def prewarm_snapshots(jobs: 'Iterable[dict[str, Any]]') -> 'None':
    """Parse every distinct geometry a set of jobs needs, in this process."""

    for job in jobs:
        load_snapshot(job["geometry"])


def certify_2d_mesh_convergence(
    base_result: 'dict[str, Any]',
    fine_result: 'dict[str, Any]',
    mesh_convergence_policy: 'dict[str, Any]' = None,
) -> 'dict[str, Any]':
    """Certify one production 2-D unit and describe the published mesh."""

    policy = validate_mesh_convergence_policy(mesh_convergence_policy)
    gate = evaluate_mesh_convergence(
        base_result=base_result,
        fine_result=fine_result,
        rms_limit_db=policy["rms_limit_db"],
        max_abs_limit_db=policy["max_abs_limit_db"],
        complex_rms_limit=policy["complex_rms_limit"],
        complex_max_limit=policy["complex_max_limit"],
        phase_rms_limit_deg=policy["phase_rms_limit_deg"],
        phase_max_limit_deg=policy["phase_max_limit_deg"],
        phase_floor_relative=policy["phase_floor_relative"],
    )
    gate["schema"] = "ghost.solver.mesh-convergence.v1"
    gate["fine_factor"] = policy["fine_factor"]
    gate["published_mesh"] = "fine"
    gate["base_quality_gate"] = dict(
        base_result.get("metadata", {}).get("quality_gate", {}) or {}
    )
    gate["fine_quality_gate"] = dict(
        fine_result.get("metadata", {}).get("quality_gate", {}) or {}
    )
    return gate


def solve_job(
    job: 'dict[str, Any]',
    *,
    angles_deg: 'Iterable[float]',
    geometry_units: 'str',
    solver_method: 'str',
    max_panels: 'int',
    force: 'bool',
    mesh_convergence_policy: 'dict[str, Any]' = None,
) -> 'tuple[str, str]':
    """Solve one unit and atomically publish one GRIM directly to results/."""
    output = Path(job["output"])
    expected_sha = str(job["specification"]["unit_sha256"])
    if output.exists() and not force:
        if _stored_unit_sha(output) == expected_sha:
            return "skipped", str(output)
        raise RuntimeError(
            f"{output} already exists but was produced from different inputs "
            "or solver settings. Move it aside or set FORCE=True."
        )

    from grim_io import export_result_to_grim
    from rcs_solver import solve_monostatic_rcs_2d_certified

    geometry = Path(job["geometry"])
    snapshot = load_snapshot(geometry)
    mesh_policy = validate_mesh_convergence_policy(
        mesh_convergence_policy
        if mesh_convergence_policy is not None
        else job["specification"].get("mesh_convergence_policy")
    )

    result = solve_monostatic_rcs_2d_certified(
        geometry_snapshot=snapshot,
        frequencies_ghz=[float(job["frequency_ghz"])],
        elevations_deg=[float(value) for value in angles_deg],
        polarization=str(job["polarization"]),
        geometry_units=geometry_units,
        material_base_dir=str(geometry.parent),
        solver_method=solver_method,
        max_panels=int(max_panels),
        mesh_convergence_policy=mesh_policy,
    )
    for warning in result.get("metadata", {}).get("warnings", []) or []:
        print(
            f"[warn] {geometry.name} {job['frequency_ghz']:g} GHz "
            f"{job['polarization']} certified fine mesh: {warning}",
            flush=True,
        )
    result.setdefault("metadata", {})["workflow_unit"] = dict(job["specification"])
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".grim", dir=output.parent
    )
    os.close(descriptor)
    try:
        written = export_result_to_grim(
            result,
            temporary_name,
            source_path=str(geometry),
            history=(
                f"step 1 {job['role']} {job['polarization']} "
                f"{job['frequency_ghz']:g}GHz"
            ),
        )
        os.replace(written[0], output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return "written", str(output)


def solve_job_catching(args: 'tuple[Any, ...]') -> 'tuple[str, str, str, dict[str, Any]]':
    job, kwargs = args
    try:
        status, path = solve_job(job, **kwargs)
        return "ok", status, path, job
    except Exception:
        return "error", traceback.format_exc(), "", job
