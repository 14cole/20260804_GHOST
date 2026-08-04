"""Shared implementation for simplified local and SLURM BoR body runners."""

import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Iterable

import numpy as np

from feature_sum import (
    geometry_input_fingerprint,
    outer_generatrix,
    radar_grid_aspects,
    save_body_grim,
    solve_vehicle_body,
    validate_radar_grid as _validate_radar_grid,
)
from geometry_io import build_geometry_snapshot, parse_geometry
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


def validate_config(frequencies: 'Iterable[float]', aspects: 'Iterable[float]') -> 'tuple[list[float], list[float]]':
    freqs = [float(value) for value in frequencies]
    angles = [float(value) for value in aspects]
    if (
        not freqs
        or not all(np.isfinite(value) and value > 0.0 for value in freqs)
        or len(set(freqs)) != len(freqs)
    ):
        raise ValueError("FREQUENCIES_GHZ must be positive, finite, and unique.")
    if (
        not angles
        or not all(np.isfinite(value) and 0.0 <= value <= 180.0 for value in angles)
        or len(set(angles)) != len(angles)
    ):
        raise ValueError("Body aspects must be unique finite values in [0, 180].")
    return freqs, angles


def validate_radar_grid(
    azimuths_deg: 'Iterable[float]',
    elevations_deg: 'Iterable[float]',
) -> 'tuple[list[float], list[float]]':
    """Validate the user-requested radar look grid used to derive BoR aspects."""
    return _validate_radar_grid(azimuths_deg, elevations_deg)


def _validated_requested_radar_grid(
    requested: 'dict[str, Any]',
    frequencies: 'list[float]',
    aspects: 'list[float]',
) -> 'dict[str, Any]':
    grid = dict(requested)
    required_keys = {
        "azimuths_deg",
        "elevations_deg",
        "frequencies_ghz",
        "axis_az_deg",
        "axis_el_deg",
    }
    if set(grid) != required_keys:
        raise ValueError(
            "requested_radar_grid must contain exactly "
            + ", ".join(sorted(required_keys))
            + "."
        )
    azimuths, elevations = validate_radar_grid(
        grid["azimuths_deg"], grid["elevations_deg"]
    )
    requested_frequencies = [
        float(value) for value in grid["frequencies_ghz"]
    ]
    if sorted(requested_frequencies) != sorted(frequencies):
        raise ValueError(
            "requested_radar_grid frequencies must match the body solve."
        )
    axis_az = float(grid["axis_az_deg"])
    axis_el = float(grid["axis_el_deg"])
    if (
        not np.isfinite(axis_az)
        or not np.isfinite(axis_el)
        or not -90.0 <= axis_el <= 90.0
    ):
        raise ValueError(
            "requested_radar_grid body-axis angles must be finite and "
            "axis_el_deg must be in [-90, 90]."
        )
    mapped = radar_grid_aspects(
        azimuths, elevations, axis_az, axis_el
    )
    supplied = np.asarray(aspects, dtype=float)
    if (
        len(mapped) != len(supplied)
        or any(
            not np.any(np.isclose(
                supplied, value, rtol=0.0, atol=1.0e-9
            ))
            for value in mapped
        )
    ):
        raise ValueError(
            "Body aspects do not exactly match the requested radar "
            "azimuth/elevation grid."
        )
    return {
        "azimuths_deg": azimuths,
        "elevations_deg": elevations,
        "frequencies_ghz": list(frequencies),
        "axis_az_deg": axis_az,
        "axis_el_deg": axis_el,
    }


def discover_jobs(step_dir: 'str | os.PathLike[str]') -> 'list[dict[str, Any]]':
    root = Path(step_dir).resolve()
    geometry_dir = root / "geometries"
    result_dir = root / "results"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for geometry in sorted(geometry_dir.glob("*.geo")):
        jobs.append(
            {
                "geometry": str(geometry.resolve()),
                "name": geometry.stem,
                "output": str((result_dir / f"{geometry.stem}.grim").resolve()),
            }
        )
    return jobs


def prepare_jobs(
    jobs: 'list[dict[str, Any]]',
    *,
    frequencies: 'list[float]',
    aspects: 'list[float]',
    geometry_units: 'str',
    runner_path: 'str',
    mesh_convergence_policy: 'dict[str, Any]' = None,
    requested_radar_grid: 'dict[str, Any]' = None,
) -> 'list[dict[str, Any]]':
    backend = Path(__file__).resolve().parent
    source_sha = backend_source_fingerprint(
        str(backend), {"step2_runner.py": str(Path(runner_path).resolve())}
    )
    runtime_sha = runtime_environment_fingerprint()
    mesh_policy = validate_mesh_convergence_policy(
        mesh_convergence_policy
    )
    normalized_radar_grid = (
        _validated_requested_radar_grid(
            requested_radar_grid, frequencies, aspects
        )
        if requested_radar_grid is not None
        else None
    )
    prepared = []
    for original in jobs:
        job = dict(original)
        spec = {
            "schema": "ghost.workflow.bor-body-unit.v1",
            "geometry_input_sha256": geometry_input_fingerprint(
                job["geometry"], geometry_units
            ),
            "solver_source_sha256": source_sha,
            "runtime_environment_sha256": runtime_sha,
            "geometry_units": str(geometry_units).strip().lower(),
            "frequencies_ghz": frequencies,
            "aspects_deg": aspects,
            "mesh_convergence_policy": mesh_policy,
            "published_mesh": "fine",
        }
        if normalized_radar_grid is not None:
            spec["requested_radar_grid"] = dict(normalized_radar_grid)
        spec["unit_sha256"] = stable_json_fingerprint(spec)
        job["specification"] = spec
        prepared.append(job)
    return prepared


def _stored_sha(path: 'Path') -> 'str':
    try:
        with np.load(path, allow_pickle=False) as payload:
            return str(np.asarray(payload["run_solve_spec_sha256"]).reshape(()).item())
    except (OSError, KeyError, TypeError, ValueError):
        return ""


def pin_blas_threads(threads: 'int' = 1) -> 'None':
    """Prevent nested BLAS threads inside the BoR mode-thread pool."""

    value = str(max(1, int(threads)))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value


def _body_channel_result(
    frequency_ghz: 'float',
    body: 'dict[str, Any]',
    polarization: 'str',
) -> 'dict[str, Any]':
    theta = list(body.get("theta_deg", []) or [])
    key = "amp_vv" if polarization == "VV" else "amp_hh"
    amplitudes = list(body.get(key, []) or [])
    if not theta or len(theta) != len(amplitudes):
        raise ValueError(
            f"BoR {polarization} body result has inconsistent aspect and "
            "amplitude arrays."
        )
    samples = []
    for angle, raw_amplitude in zip(theta, amplitudes):
        amplitude = complex(raw_amplitude)
        sigma = 4.0 * np.pi * (
            amplitude.real * amplitude.real
            + amplitude.imag * amplitude.imag
        )
        samples.append({
            "frequency_ghz": float(frequency_ghz),
            "theta_inc_deg": float(angle),
            "theta_scat_deg": float(angle),
            "rcs_linear": float(sigma),
            "rcs_db": float(10.0 * np.log10(max(sigma, 1.0e-300))),
            "rcs_amp_real": float(amplitude.real),
            "rcs_amp_imag": float(amplitude.imag),
        })
    return {"samples": samples}


def certify_body_mesh_convergence(
    base_bodies: 'dict[float, dict[str, Any]]',
    fine_bodies: 'dict[float, dict[str, Any]]',
    mesh_convergence_policy: 'dict[str, Any]' = None,
) -> 'dict[str, Any]':
    """Certify both co-solved BoR channels at every requested frequency."""

    policy = validate_mesh_convergence_policy(mesh_convergence_policy)
    base_frequencies = {float(value) for value in base_bodies}
    fine_frequencies = {float(value) for value in fine_bodies}
    if base_frequencies != fine_frequencies or not base_frequencies:
        raise ValueError(
            "BoR base and fine results must contain the same nonempty "
            "frequency grid."
        )

    per_frequency = {}
    violations = []
    for frequency in sorted(base_frequencies):
        per_pol = {}
        for polarization in ("VV", "HH"):
            gate = evaluate_mesh_convergence(
                _body_channel_result(
                    frequency, base_bodies[frequency], polarization
                ),
                _body_channel_result(
                    frequency, fine_bodies[frequency], polarization
                ),
                rms_limit_db=policy["rms_limit_db"],
                max_abs_limit_db=policy["max_abs_limit_db"],
                complex_rms_limit=policy["complex_rms_limit"],
                complex_max_limit=policy["complex_max_limit"],
                phase_rms_limit_deg=policy["phase_rms_limit_deg"],
                phase_max_limit_deg=policy["phase_max_limit_deg"],
                phase_floor_relative=policy["phase_floor_relative"],
            )
            per_pol[polarization] = gate
            if not bool(gate.get("passed", False)):
                violations.append(
                    f"{frequency:g} GHz {polarization}: "
                    + str(gate.get("reason", "mesh convergence failed"))
                )
        per_frequency[str(frequency)] = {
            "passed": all(
                bool(per_pol[pol]["passed"]) for pol in ("VV", "HH")
            ),
            "polarizations": per_pol,
        }

    return {
        "schema": "ghost.solver.mesh-convergence.v1",
        "passed": not violations,
        "fine_factor": policy["fine_factor"],
        "published_mesh": "fine",
        "co_solved_polarizations": ["VV", "HH"],
        "policy": policy,
        "per_frequency": per_frequency,
        "violations": violations,
        "reason": (
            "; ".join(violations)
            if violations
            else "BoR VV/HH complex-field mesh convergence passed"
        ),
    }


def solve_job(
    job: 'dict[str, Any]',
    *,
    frequencies: 'list[float]',
    aspects: 'list[float]',
    geometry_units: 'str',
    workers_per_body: 'int',
    force: 'bool',
    mesh_convergence_policy: 'dict[str, Any]' = None,
) -> 'tuple[str, str]':
    output = Path(job["output"])
    expected = str(job["specification"]["unit_sha256"])
    if output.exists() and not force:
        if _stored_sha(output) == expected:
            return "skipped", str(output)
        raise RuntimeError(
            f"{output} exists but does not match the current geometry or "
            "settings. Move it aside or set FORCE=True."
        )
    geometry = Path(job["geometry"])
    snapshot = build_geometry_snapshot(
        *parse_geometry(geometry.read_text(encoding="utf-8"))
    )
    snapshot["source_path"] = str(geometry)
    profile = outer_generatrix(snapshot, geometry_units)
    mesh_policy = validate_mesh_convergence_policy(
        mesh_convergence_policy
        if mesh_convergence_policy is not None
        else job["specification"].get("mesh_convergence_policy")
    )
    base_bodies, _profile, base_diagnostics = solve_vehicle_body(
        snapshot,
        frequencies,
        aspects,
        geometry_units=geometry_units,
        cfie_alpha=0.5,
        workers=int(workers_per_body),
        material_base_dir=str(geometry.parent),
        return_diagnostics=True,
    )
    fine_snapshot = scale_snapshot_panel_density(
        snapshot, mesh_policy["fine_factor"]
    )
    bodies, _fine_profile, solver_diagnostics = solve_vehicle_body(
        fine_snapshot,
        frequencies,
        aspects,
        geometry_units=geometry_units,
        cfie_alpha=0.5,
        workers=int(workers_per_body),
        material_base_dir=str(geometry.parent),
        return_diagnostics=True,
    )
    mesh_gate = certify_body_mesh_convergence(
        base_bodies, bodies, mesh_policy
    )
    if not bool(mesh_gate.get("passed", False)):
        raise RuntimeError(
            "Production BoR mesh convergence gate failed: "
            + str(mesh_gate.get("reason", "unknown convergence failure"))
        )
    for frequency in sorted(solver_diagnostics):
        fine_metadata = solver_diagnostics[frequency].setdefault(
            "metadata", {}
        )
        fine_metadata["mesh_convergence"] = (
            mesh_gate["per_frequency"][str(float(frequency))]
        )
        fine_metadata["mesh_convergence"]["fine_factor"] = (
            mesh_policy["fine_factor"]
        )
        fine_metadata["mesh_convergence"]["published_mesh"] = "fine"
        fine_metadata["mesh_convergence"]["base_quality_gate"] = dict(
            base_diagnostics.get(frequency, {}).get(
                "metadata", {}
            ).get("quality_gate", {}) or {}
        )
        fine_metadata["mesh_convergence"]["fine_quality_gate"] = dict(
            fine_metadata.get("quality_gate", {}) or {}
        )
    for frequency in sorted(solver_diagnostics):
        for label, diagnostics in (
            ("base", base_diagnostics),
            ("fine", solver_diagnostics),
        ):
            metadata = diagnostics[frequency].get("metadata", {}) or {}
            for warning in metadata.get("warnings", []) or []:
                print(
                    f"[warn] {geometry.name} {frequency:g} GHz "
                    f"{label} mesh: {warning}",
                    flush=True,
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".grim", dir=output.parent
    )
    os.close(descriptor)
    try:
        saved = save_body_grim(
            bodies,
            temporary,
            source_path=str(geometry),
            history=f"simplified step 2 body={geometry.name}",
            geometry_input_sha256=job["specification"]["geometry_input_sha256"],
            solver_source_sha256=job["specification"]["solver_source_sha256"],
            runtime_environment_sha256=job["specification"]["runtime_environment_sha256"],
            run_solve_spec_sha256=expected,
            body_profile=profile,
            solver_diagnostics=solver_diagnostics,
            requested_radar_grid=job["specification"].get(
                "requested_radar_grid"
            ),
        )
        os.replace(saved, output)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return "written", str(output)


def solve_job_catching(args):
    job, kwargs = args
    try:
        status, path = solve_job(job, **kwargs)
        return "ok", status, path, job
    except Exception:
        return "error", traceback.format_exc(), "", job
