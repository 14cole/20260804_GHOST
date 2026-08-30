#!/usr/bin/env python3
"""
Shared plumbing for the HPC steps -- three small jobs, all of them explicit.

1. CONFIGURE A DRIVER.  The canonical Backend drivers
   (run_hpc_bor_monostatic.py and run_hpc_monostatic.py) keep their settings
   in a CONFIG block of module-level constants, and the SLURM script they write
   execs THE SAME FILE with --worker.  So the compute nodes read the constants
   out of whichever copy submitted the job.  ``configure_driver`` therefore
   writes a COPY of the driver
   with those constants replaced -- and submitting that copy is what makes the
   settings reach the nodes.  Overriding them in your own process would not.

2. STAGE GEOMETRY.  The drivers discover every .geo under FRD_DIR / OPN_DIR, so
   each step stages exactly the files it wants into its own outputs/geometries/
   and points the configured copy at that (absolute) path.

3. COLLECT. Each BoR unit writes one frequency/channel GRIM. The publication
   path uses ``read_unit_grims`` and ``bodies_from_units`` to assemble the one
   self-contained monostatic body product. Coupon joins are handled by the
   shared CEM Tools operations rather than a second implementation here.
"""

import json
import os
import re
import shutil
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

BACKEND = Path(__file__).resolve().parent       # the drivers live beside this file
HPC_ROOT = BACKEND
REPO_ROOT = BACKEND
sys.path.insert(0, str(BACKEND))

BOR_DRIVER = BACKEND / "run_hpc_bor_monostatic.py"
TWOD_DRIVER = BACKEND / "run_hpc_monostatic.py"

_UNIT_RE = re.compile(r"^(?P<pol>[A-Z]{2})_(?P<freq>[0-9.]+)GHz_(?P<stem>.+)\.grim$")
_DUAL_UNIT_RE = re.compile(r"^(?P<freq>[0-9.]+)GHz_(?P<stem>.+)\.grim$")


def _unit_output_contract(unit: 'Dict[str, Any]', schema: 'str') -> 'tuple':
    """Return ``(name, polarizations)`` for a validated solver unit.

    The v2 2-D driver co-solves both channels into one file; legacy 2-D and
    BoR intermediate manifests retain one polarization per unit.
    """

    stem = unit.get("geometry_stem")
    frequency = unit.get("frequency_ghz")
    if (
        not isinstance(stem, str)
        or not stem
        or stem in (".", "..")
        or "/" in stem
        or "\\" in stem
        or isinstance(frequency, bool)
        or not isinstance(frequency, (int, float))
        or not math.isfinite(float(frequency))
        or float(frequency) <= 0.0
    ):
        raise ValueError("HPC unit cannot form a safe output filename.")
    if schema == "ghost.hpc.2d-run.v2":
        polarizations = unit.get("polarizations")
        if list(polarizations or []) != ["VV", "HH"]:
            raise ValueError(
                "2-D v2 HPC units must contain canonical polarizations "
                "['VV', 'HH']."
            )
        name = f"{float(frequency):.3f}GHz_{stem}.grim"
        return name, ["VV", "HH"]

    valid = {
        "ghost.hpc.2d-run.v1": {"TM", "TE"},
        "ghost.hpc.bor-run.v1": {"VV", "HH"},
    }.get(schema)
    polarization = unit.get("polarization")
    if (
        valid is None
        or not isinstance(polarization, str)
        or "/" in polarization
        or "\\" in polarization
        or polarization not in valid
    ):
        raise ValueError("HPC unit has an invalid polarization.")
    name = f"{polarization}_{float(frequency):.3f}GHz_{stem}.grim"
    return name, [polarization]


# -----------------------------------------------------------------------------
# 1. configure a copy of a driver
# -----------------------------------------------------------------------------

def configure_driver(driver: 'Path', out_path: 'Path', settings: 'Dict[str, Any]') -> 'Path':
    """Copy ``driver`` to ``out_path`` with each CONFIG constant in ``settings``
    replaced.  Every name must already exist as a top-level assignment in the
    driver, so a typo is an error here rather than a silently ignored setting.

    A PYTHONPATH line is added to JOB_PROLOGUE, because the SLURM script cd's to
    the configured copy's folder and the solver modules live in the repo root.
    """
    src = Path(driver).read_text().splitlines(keepends=True)
    want = dict(settings)
    want.setdefault("JOB_PROLOGUE",
                    [f"export PYTHONPATH={REPO_ROOT}:${{PYTHONPATH:-}}"])
    seen = set()
    out: 'List[str]' = []
    for line in src:
        m = re.match(r"^([A-Z_][A-Z0-9_]*)\s*=", line)
        if m and m.group(1) in want:
            key = m.group(1)
            if key in seen:                     # only the first (CONFIG) binding
                out.append(line)
                continue
            seen.add(key)
            out.append(f"{key} = {want[key]!r}\n")
        else:
            out.append(line)
    missing = sorted(set(want) - seen)
    if missing:
        raise ValueError(f"{driver.name}: no top-level assignment for {missing} "
                         f"-- check the spelling against the driver's CONFIG block.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("".join(out))
    Path(out_path).chmod(0o755)
    return Path(out_path)


def stage_geometry(files: 'Sequence[os.PathLike]', geom_root: 'Path') -> 'Path':
    """Copy the .geo files this step wants into ``geom_root``/FRD (and make an
    empty OPN beside it, so the driver's discovery does not warn)."""
    frd = Path(geom_root) / "FRD"
    opn = Path(geom_root) / "OPN"
    if frd.exists():
        shutil.rmtree(frd)
    frd.mkdir(parents=True)
    opn.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copyfile(f, frd / Path(f).name)
    return frd


def latest_run_dir(output_dir: 'os.PathLike') -> 'Path':
    """Newest run_YYYYMMDD_HHMMSS/ under a driver OUTPUT_DIR."""
    runs = sorted(Path(output_dir).glob("run_*"))
    if not runs:
        raise SystemExit(f"no run_* folder in {output_dir} -- submit first.")
    return runs[-1]


def _expected_unit_output_paths(
    run_dir: 'Path', manifest: 'Dict[str, Any]'
) -> 'tuple':
    """Return the unit-output root and exact manifest-derived result paths."""

    if not isinstance(manifest, dict):
        raise ValueError("HPC manifest must be a JSON object.")
    schema = str(manifest.get("schema", ""))
    if schema not in {
        "ghost.hpc.2d-run.v1",
        "ghost.hpc.2d-run.v2",
        "ghost.hpc.bor-run.v1",
    }:
        raise ValueError(f"HPC manifest has an unsupported schema {schema!r}.")
    units = manifest.get("units")
    n_units = manifest.get("n_units")
    if (
        not isinstance(units, list)
        or not units
        or isinstance(n_units, bool)
        or not isinstance(n_units, int)
        or n_units != len(units)
    ):
        raise ValueError("HPC manifest n_units does not match a nonempty unit inventory.")

    relative_root = str(manifest.get("unit_output_dir", "results"))
    root_path = Path(relative_root)
    if (
        not relative_root
        or root_path.is_absolute()
        or any(part in ("", ".", "..") for part in root_path.parts)
    ):
        raise ValueError("HPC manifest unit_output_dir is not a safe relative path.")
    results_dir = run_dir / root_path
    expected = []
    seen = set()
    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("Every HPC unit must be an object.")
        name, _polarizations = _unit_output_contract(unit, schema)
        role = str(unit.get("role", "")).strip().upper()
        if role and (
            role in {".", ".."}
            or Path(role).name != role
            or "/" in role
            or "\\" in role
        ):
            raise ValueError("HPC unit has an unsafe result role.")
        relative = Path(role) / name if role else Path(name)
        key = relative.as_posix()
        if key in seen:
            raise ValueError("HPC manifest contains colliding result paths.")
        seen.add(key)
        expected.append(results_dir / relative)
    return results_dir, tuple(expected)


def run_status(run_dir: 'os.PathLike') -> 'Dict[str, Any]':
    """Return fail-closed, manifest-exact completion for one HPC run.

    A filename is not proof that a solve finished.  Once the exact expected
    unit set is present, every unit's embedded attestation is verified against
    the immutable run manifest.  BoR runs additionally require one readable,
    run-bound published body product per geometry; the per-frequency files are
    restart records, not the user-facing completion contract.
    """

    run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results_dir, expected = _expected_unit_output_paths(run_dir, manifest)
    expected_set = set(expected)
    done = tuple(sorted(path for path in expected if path.is_file()))
    missing = tuple(sorted(expected_set - set(done)))
    actual = tuple(sorted(results_dir.rglob("*.grim"))) if results_dir.is_dir() else ()
    unexpected = tuple(sorted(set(actual) - expected_set))
    unit_exact = not missing and not unexpected
    attestation_verified = False
    attestation_error = ""
    if unit_exact:
        try:
            require_hpc_output_attestations(run_dir, manifest)
        except Exception as exc:
            # A corrupt NPZ may raise BadZipFile/EOFError rather than
            # ValueError.  Completion is a read-only health report, so retain
            # the diagnostic and fail closed instead of crashing a Runs-tab
            # refresh before it can show the scheduler state.
            attestation_error = f"{type(exc).__name__}: {exc}"
        else:
            attestation_verified = True
    unit_complete = unit_exact and attestation_verified

    derived_root = run_dir / "results"
    derived_expected = ()
    derived_done = ()
    derived_missing = ()
    derived_unexpected = ()
    publication_verified = True
    publication_error = ""
    if manifest.get("schema") == "ghost.hpc.bor-run.v1":
        stems = sorted({str(unit["geometry_stem"]) for unit in manifest["units"]})
        derived_expected = tuple(derived_root / f"{stem}.grim" for stem in stems)
        expected_publications = set(derived_expected)
        derived_done = tuple(
            sorted(path for path in derived_expected if path.is_file())
        )
        derived_missing = tuple(
            sorted(expected_publications - set(derived_done))
        )
        actual_publications = (
            tuple(sorted(derived_root.glob("*.grim")))
            if derived_root.is_dir()
            else ()
        )
        derived_unexpected = tuple(
            sorted(set(actual_publications) - expected_publications)
        )
        publication_verified = False
        if not derived_missing and not derived_unexpected:
            try:
                from feature_sum import _load_grim, load_body_grim
                from workflow_provenance import manifest_solve_spec_fingerprint

                run_spec = manifest_solve_spec_fingerprint(manifest)
                units_by_stem = {}
                for unit in manifest["units"]:
                    units_by_stem.setdefault(str(unit["geometry_stem"]), unit)
                for path in derived_expected:
                    payload = _load_grim(str(path))
                    # This checks the embedded compact body model in addition
                    # to the ordinary radar-grid GRIM fields loaded above.
                    load_body_grim(str(path), loaded_grim=payload)
                    source_unit = units_by_stem[path.stem]
                    expected_metadata = {
                        "geometry_input_sha256": str(
                            source_unit["geometry_input_sha256"]
                        ),
                        "solver_source_sha256": str(
                            manifest["solver_source_sha256"]
                        ),
                        "runtime_environment_sha256": str(
                            manifest["runtime_environment_sha256"]
                        ),
                        "run_solve_spec_sha256": run_spec,
                    }
                    for field, wanted in expected_metadata.items():
                        if field not in payload:
                            raise ValueError(
                                f"{path.name}: published BoR product is missing "
                                f"run-binding metadata {field!r}."
                            )
                        raw = np.asarray(payload[field])
                        if raw.size != 1:
                            raise ValueError(
                                f"{path.name}: published BoR metadata {field!r} "
                                "must be scalar."
                            )
                        actual_value = raw.reshape(-1)[0]
                        if isinstance(actual_value, bytes):
                            actual_value = actual_value.decode("utf-8")
                        if str(actual_value) != wanted:
                            raise ValueError(
                                f"{path.name}: published BoR metadata {field!r} "
                                "does not match this run."
                            )
            except Exception as exc:
                publication_error = f"{type(exc).__name__}: {exc}"
            else:
                publication_verified = True

    complete = unit_complete and publication_verified
    return {
        "run_dir": run_dir,
        "unit_output_dir": results_dir,
        "n_units": len(expected),
        "n_done": len(done),
        "manifest": manifest,
        "done": done,
        "missing": missing,
        "unexpected": unexpected,
        "pending": len(missing),
        "unit_exact": unit_exact,
        "attestation_verified": attestation_verified,
        "attestation_error": attestation_error,
        "unit_complete": unit_complete,
        "derived_root": derived_root,
        "derived_expected": derived_expected,
        "derived_done": derived_done,
        "derived_missing": derived_missing,
        "derived_unexpected": derived_unexpected,
        "publication_verified": publication_verified,
        "publication_error": publication_error,
        "complete": complete,
    }


def require_hpc_run_provenance(
    manifest: 'Dict[str, Any]',
    expected_schema: 'str',
) -> 'None':
    """Reject legacy/tampered HPC runs before their fields are collected."""

    if not isinstance(manifest, dict):
        raise ValueError("HPC manifest must be a JSON object.")
    if manifest.get("schema") != expected_schema:
        raise ValueError(
            f"HPC manifest schema is {manifest.get('schema')!r}, expected "
            f"{expected_schema!r}. Legacy runs must be regenerated."
        )
    for field in (
        "solver_source_sha256",
        "runtime_environment_sha256",
    ):
        value = str(manifest.get(field, ""))
        if len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise ValueError(
                f"HPC manifest has no valid {field}; refusing unprovenanced "
                "solver output."
            )

    raw_units = manifest.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise ValueError("HPC manifest units must be a non-empty array.")
    units = list(raw_units)
    n_units = manifest.get("n_units")
    if (
        isinstance(n_units, bool)
        or not isinstance(n_units, int)
        or n_units != len(units)
    ):
        raise ValueError(
            "HPC manifest n_units does not match its unit inventory."
        )
    solver_config = manifest.get("solver_config")
    if not isinstance(solver_config, dict):
        raise ValueError("HPC manifest solver_config must be an object.")
    geometry_units = str(solver_config.get("geometry_units", ""))
    if not geometry_units:
        raise ValueError(
            "HPC manifest has no units, solver configuration, or "
            "geometry-unit declaration."
        )
    from feature_sum import geometry_input_fingerprint
    checked = {}
    output_names = set()
    expected_angle_key = {
        "ghost.hpc.2d-run.v1": "azimuths_deg",
        "ghost.hpc.2d-run.v2": "azimuths_deg",
        "ghost.hpc.bor-run.v1": "aspects_deg",
    }.get(expected_schema)
    # The angular grid is declared once for the run rather than repeated in
    # every unit: identical per unit, and repeating it dominated the file.
    raw_angles = manifest.get(expected_angle_key) if expected_angle_key else None
    if (
        not isinstance(raw_angles, (list, tuple))
        or not raw_angles
        or not all(
            not isinstance(value, bool) and math.isfinite(float(value))
            for value in raw_angles
        )
    ):
        raise ValueError(
            f"HPC manifest has no valid {expected_angle_key!r} grid."
        )

    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("Every HPC unit must be an object.")
        path = str(unit.get("geometry", ""))
        expected = str(unit.get("geometry_input_sha256", ""))
        if (
            not path
            or not os.path.isabs(path)
            or len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)
        ):
            raise ValueError(
                "HPC unit lacks an exact frozen geometry/material "
                "fingerprint."
            )
        stem = unit.get("geometry_stem")
        if (
            not isinstance(stem, str)
            or not stem
            or stem in (".", "..")
            or "/" in stem
            or "\\" in stem
            or stem != Path(path).stem
        ):
            raise ValueError(
                "HPC unit has an unsafe or geometry-inconsistent output stem."
            )
        frequency = unit.get("frequency_ghz")
        if (
            isinstance(frequency, bool)
            or not isinstance(frequency, (int, float))
            or not math.isfinite(float(frequency))
            or float(frequency) <= 0.0
        ):
            raise ValueError(
                "HPC unit has a non-positive or non-finite frequency."
            )
        output_name, _polarizations = _unit_output_contract(
            unit, expected_schema
        )
        if output_name in output_names:
            raise ValueError(
                "HPC units collide after output filename formatting; "
                "frequencies and geometry stems must identify unique files."
            )
        output_names.add(output_name)
        if path not in checked:
            checked[path] = geometry_input_fingerprint(
                path, geometry_units
            )
        if checked[path] != expected:
            raise ValueError(
                f"Frozen HPC geometry/material input changed: {path}"
            )


def require_hpc_output_attestations(
    run_dir: 'os.PathLike',
    manifest: 'Dict[str, Any]',
) -> 'None':
    """Require the exact expected result set, each bound to this run.

    The binding travels inside each .grim rather than in a
    `<name>.provenance.json` beside it: a sweep of thousands of units would
    otherwise put thousands of extra tiny files in results/. Integrity is not
    lost with the sidecar -- a .grim is an npz, npz is a zip, and numpy
    validates a CRC-32 per member on read, so a corrupted result raises on
    open instead of verifying.
    """

    from workflow_provenance import (
        manifest_solve_spec_fingerprint,
        stable_json_fingerprint,
        unit_solve_spec_fingerprint,
        verify_embedded_attestation,
    )

    # Per-frequency BoR GRIMs are visible intermediate/restart records under
    # results/by_frequency/. The collected results/<geometry>.grim remains the
    # production body artifact. Older manifests and the 2-D driver may use
    # results/ directly, so keep the manifest-defined relative directory.
    unit_output_dir = str(manifest.get("unit_output_dir", "results"))
    if (
        not unit_output_dir
        or Path(unit_output_dir).is_absolute()
        or any(part in ("", ".", "..") for part in Path(unit_output_dir).parts)
    ):
        raise ValueError("HPC manifest unit_output_dir is not a safe relative path.")
    results_dir = Path(run_dir) / unit_output_dir
    if not isinstance(manifest, dict):
        raise ValueError("HPC manifest must be a JSON object.")
    schema = manifest.get("schema")
    expected_angle_key = {
        "ghost.hpc.2d-run.v1": "azimuths_deg",
        "ghost.hpc.2d-run.v2": "azimuths_deg",
        "ghost.hpc.bor-run.v1": "aspects_deg",
    }.get(schema)
    if expected_angle_key is None:
        raise ValueError(
            f"HPC manifest has an unsupported schema {schema!r}."
        )
    units = manifest.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("HPC manifest units must be a non-empty array.")
    n_units = manifest.get("n_units")
    if (
        isinstance(n_units, bool)
        or not isinstance(n_units, int)
        or n_units != len(units)
    ):
        raise ValueError(
            "HPC manifest n_units does not match its unit inventory."
        )
    for field in (
        "solver_source_sha256",
        "runtime_environment_sha256",
    ):
        value = manifest.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"HPC manifest has no valid {field}.")
    if (
        not isinstance(manifest.get("run_id"), str)
        or not manifest["run_id"]
    ):
        raise ValueError("HPC manifest has no valid run_id.")
    solver_config = manifest.get("solver_config")
    if not isinstance(solver_config, dict):
        raise ValueError("HPC manifest solver_config must be an object.")

    raw_angles = manifest.get(expected_angle_key)
    if (
        not isinstance(raw_angles, (list, tuple))
        or not raw_angles
        or not all(
            not isinstance(value, bool) and math.isfinite(float(value))
            for value in raw_angles
        )
    ):
        raise ValueError(
            f"HPC manifest has no valid {expected_angle_key!r} grid."
        )
    angular_grid_sha256 = stable_json_fingerprint(
        [float(value) for value in raw_angles]
    )
    run_spec = manifest_solve_spec_fingerprint(manifest)
    config_sha = stable_json_fingerprint(solver_config)

    expected_names = set()
    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("Every HPC unit must be an object.")
        stem = unit.get("geometry_stem")
        frequency = unit.get("frequency_ghz")
        if (
            not isinstance(stem, str)
            or not stem
            or stem in (".", "..")
            or "/" in stem
            or "\\" in stem
            or isinstance(frequency, bool)
            or not isinstance(frequency, (int, float))
            or not math.isfinite(float(frequency))
            or float(frequency) <= 0.0
        ):
            raise ValueError("HPC unit cannot form a safe output filename.")
        name, polarizations = _unit_output_contract(unit, schema)
        if name in expected_names:
            raise ValueError(
                "HPC manifest contains colliding result filenames."
            )
        expected_names.add(name)
        geometry_hash = unit.get("geometry_input_sha256")
        if (
            not isinstance(geometry_hash, str)
            or len(geometry_hash) != 64
            or any(char not in "0123456789abcdef" for char in geometry_hash)
        ):
            raise ValueError(f"HPC unit for {name} has no input fingerprint.")
        role = str(unit.get("role", "")).strip().upper()
        output_path = results_dir / role / name if role else results_dir / name
        expected_attestation = {
                "run_id": str(manifest["run_id"]),
                "solver_source_sha256":
                    str(manifest["solver_source_sha256"]),
                "runtime_environment_sha256":
                    str(manifest["runtime_environment_sha256"]),
                "geometry_input_sha256": geometry_hash,
                "run_solve_spec_sha256": run_spec,
                "unit_solve_spec_sha256": unit_solve_spec_fingerprint(unit),
                "solver_config_sha256": config_sha,
                "angular_grid_kind": expected_angle_key,
                "frequency_ghz": float(frequency),
        }
        if schema == "ghost.hpc.2d-run.v2":
            expected_attestation["polarizations"] = polarizations
        else:
            expected_attestation["polarization"] = polarizations[0]
        if schema == "ghost.hpc.bor-run.v1":
            expected_attestation["angular_grid_deg"] = [
                float(value) for value in raw_angles
            ]
        else:
            expected_attestation["angular_grid_sha256"] = angular_grid_sha256
        verify_embedded_attestation(str(output_path), expected_attestation)

    actual_names = {
        str(path.relative_to(results_dir))
        for path in results_dir.rglob("*.grim")
    }
    expected_paths = set()
    for unit in units:
        name, _polarizations = _unit_output_contract(unit, schema)
        role = str(unit.get("role", "")).strip().upper()
        expected_paths.add(str(Path(role) / name) if role else name)
    if actual_names != expected_paths:
        raise ValueError(
            "HPC results do not contain the exact manifest output set "
            f"(missing={sorted(expected_paths - actual_names)[:8]}, "
            f"unexpected={sorted(actual_names - expected_paths)[:8]})."
        )


def read_unit_grims(results_dir: 'os.PathLike') -> 'List[Dict[str, Any]]':
    """Read legacy single-channel and current dual-channel result units.

    Legacy ``<POL>_<FREQ>GHz_<stem>.grim`` records expose a one-item
    ``polarizations`` list and ``amp[angle, frequency]``. Current
    ``<FREQ>GHz_<stem>.grim`` records expose their stored polarization list and
    ``amp[angle, frequency, polarization]``. One record is returned per file,
    matching the scheduler's definition of a unit.

    ``angles_deg`` is the driver's sweep axis, stored in the grim's 'azimuths':
    for the BoR driver that is the ASPECT from the rotation axis (0 = nose-on,
    90 = broadside, 180 = tail-on -- a body of revolution has no second angle);
    for the 2-D driver it is the cut angle (90 = normal to the outer face).
    """
    out: 'List[Dict[str, Any]]' = []
    for p in sorted(Path(results_dir).rglob("*.grim")):
        legacy_match = _UNIT_RE.match(p.name)
        dual_match = _DUAL_UNIT_RE.match(p.name)
        if legacy_match is None and dual_match is None:
            continue
        with np.load(str(p)) as d:
            if not bool(d["raw_complex_amplitude_preserved"]):
                raise ValueError(f"{p.name}: complex amplitudes were not preserved.")
            amp = (np.asarray(d["rcs_amp_real"], float)
                   + 1j * np.asarray(d["rcs_amp_imag"], float))
            if legacy_match is not None:
                match = legacy_match
                polarizations = [match.group("pol")]
                stored_polarizations = [
                    str(value) for value in np.asarray(
                        d["polarizations"], dtype=str
                    ).reshape(-1)
                ]
                if stored_polarizations != polarizations:
                    raise ValueError(
                        f"{p.name}: filename channel {polarizations[0]} does "
                        f"not match stored polarization axis "
                        f"{stored_polarizations}."
                    )
                amplitude = amp[:, 0, :, 0]
                legacy_pol = polarizations[0]
            else:
                match = dual_match
                polarizations = [
                    str(value) for value in np.asarray(
                        d["polarizations"], dtype=str
                    ).reshape(-1)
                ]
                if polarizations != ["VV", "HH"]:
                    raise ValueError(
                        f"{p.name}: a polarization-free unit filename requires "
                        "the canonical [VV, HH] polarization axis."
                    )
                amplitude = amp[:, 0, :, :]
                legacy_pol = ""
            solver_audit = None
            if "solver_metadata_json" in d.files:
                raw_audit = np.asarray(
                    d["solver_metadata_json"]
                ).reshape(()).item()
                if isinstance(raw_audit, bytes):
                    raw_audit = raw_audit.decode("utf-8")
                try:
                    solver_audit = json.loads(str(raw_audit))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"{p.name}: solver diagnostics are malformed."
                    ) from exc
                if not isinstance(solver_audit, dict):
                    raise ValueError(
                        f"{p.name}: solver diagnostics must be a mapping."
                    )
            out.append({
                "pol": legacy_pol,
                "polarizations": polarizations,
                "freq_ghz": float(match.group("freq")),
                "stem": match.group("stem"),
                "path": p,
                "angles_deg": np.asarray(d["azimuths"], float),
                "frequencies_ghz": np.asarray(d["frequencies"], float),
                "amp": amplitude,
                "solver_audit": solver_audit,
            })
    return out


def bor_solver_diagnostics_from_units(
    units: 'Sequence[Dict[str, Any]]',
    stem: 'Optional[str]' = None,
) -> 'Optional[Dict[float, Dict[str, Any]]]':
    """Collect uniformly usable restart-unit audits into the body contract.

    Current BoR runs publish separate VV and HH restart records even though
    they are co-solved. Their physical solver metadata must agree after the
    channel-specific output attestation is removed.  Audit/certification
    metadata is optional evidence, not authorization to use otherwise valid
    fields: absent, partially present, or polarization-disagreeing
    certification evidence therefore returns ``None`` and the body publisher
    omits the derived solver-audit envelope.

    Evidence that *is* present is still checked against its own file.  A
    malformed solver contract, a selected-channel mismatch, or an attestation
    that names another field remains a hard provenance-integrity error.
    """

    selected = [
        unit for unit in units
        if stem is None or str(unit.get("stem")) == str(stem)
    ]
    if not selected:
        raise ValueError(f"No BoR solver units found for {stem!r}.")
    audits = [unit.get("solver_audit") for unit in selected]
    if all(audit is None for audit in audits):
        return None

    complete_evidence = not any(audit is None for audit in audits)

    grouped = {}  # type: Dict[float, List[Dict[str, Any]]]
    for unit in selected:
        grouped.setdefault(float(unit["freq_ghz"]), []).append(unit)

    from grim_io import SOLVER_METADATA_SCHEMA

    diagnostics = {}
    for frequency, records in sorted(grouped.items()):
        coverage = {
            polarization: sum(
                polarization in list(record.get("polarizations") or [])
                for record in records
            )
            for polarization in ("VV", "HH")
        }
        if coverage != {"VV": 1, "HH": 1}:
            raise ValueError(
                f"{frequency:g} GHz BoR diagnostics require exactly one "
                "VV and one HH source field."
            )

        common_metadata = None
        metadata_by_certification = {}
        certification_classes = set()
        source_attestations = {}
        for record in records:
            raw_audit = record.get("solver_audit")
            if raw_audit is None:
                continue
            if not isinstance(raw_audit, dict):
                raise ValueError(
                    f"{record['path'].name}: solver audit must be a mapping."
                )
            audit = dict(raw_audit)
            record_polarizations = list(record.get("polarizations") or [])
            if (
                audit.get("schema") != SOLVER_METADATA_SCHEMA
                or audit.get("solver") != "bor_mom_rcs"
                or audit.get("scattering_mode") != "monostatic"
                or audit.get("rcs_log_unit") != "dBsm"
                or audit.get("rcs_linear_quantity") != "sigma_3d"
            ):
                raise ValueError(
                    f"{record['path'].name}: not a monostatic BoR solver audit."
                )
            if len(record_polarizations) == 1:
                polarization = record_polarizations[0]
                if (
                    audit.get("polarization") != polarization
                    or audit.get("polarization_export") != polarization
                    or audit.get("polarizations") not in (None, [])
                    or audit.get("polarization_mapping") not in (None, {})
                ):
                    raise ValueError(
                        f"{record['path'].name}: selected-channel solver "
                        "audit does not match its filename."
                    )
            elif (
                record_polarizations != ["VV", "HH"]
                or audit.get("polarizations") != ["VV", "HH"]
                or audit.get("polarization_mapping")
                != {"VV": "VV", "HH": "HH"}
            ):
                raise ValueError(
                    f"{record['path'].name}: malformed dual-channel audit."
                )
            metadata = dict(audit.get("metadata", {}) or {})
            attestation = metadata.pop("output_attestation", None)
            if attestation is not None and not isinstance(attestation, dict):
                raise ValueError(
                    f"{record['path'].name}: embedded unit attestation is "
                    "malformed."
                )
            if isinstance(attestation, dict):
                attested_frequency = float(
                    attestation.get("frequency_ghz", math.nan)
                )
                attested_polarization = str(
                    attestation.get("polarization", "")
                )
                if (
                    len(record_polarizations) != 1
                    or attested_polarization != record_polarizations[0]
                    or not math.isclose(
                        attested_frequency,
                        frequency,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                ):
                    raise ValueError(
                        f"{record['path'].name}: embedded unit attestation "
                        "does not match its field."
                    )
            certification = {
                "mesh_convergence_certified": metadata.get(
                    "mesh_convergence_certified"
                ),
                "certified_entry_point": metadata.get(
                    "certified_entry_point"
                ),
                "published_mesh": metadata.get("published_mesh"),
                "survey_mode": metadata.get("survey_mode"),
                "mesh_convergence": metadata.get("mesh_convergence"),
                "quality_gate_certification": {
                    key: metadata.get("quality_gate", {}).get(key)
                    for key in (
                        "mesh_convergence_certified",
                        "certification_scope",
                    )
                    if isinstance(metadata.get("quality_gate"), dict)
                    and key in metadata["quality_gate"]
                },
            }
            certification_serialized = json.dumps(
                certification, sort_keys=True, separators=(",", ":")
            )
            certification_classes.add(certification_serialized)
            serialized = json.dumps(
                metadata, sort_keys=True, separators=(",", ":")
            )
            previous = metadata_by_certification.get(
                certification_serialized
            )
            if previous is None:
                metadata_by_certification[certification_serialized] = serialized
            elif serialized != previous:
                raise ValueError(
                    f"{frequency:g} GHz VV/HH solver diagnostics disagree."
                )
            if common_metadata is None:
                common_metadata = metadata
            if isinstance(attestation, dict):
                for polarization in record.get("polarizations") or []:
                    source_attestations[str(polarization)] = attestation
        if len(certification_classes) != 1:
            # Different certification classes cannot truthfully be collapsed
            # into one body-level solver-audit envelope.
            complete_evidence = False
        if common_metadata is None:
            complete_evidence = False
            continue
        if source_attestations:
            common_metadata = dict(common_metadata)
            common_metadata["source_output_attestations"] = (
                source_attestations
            )
        diagnostics[frequency] = {
            "solver": "bor_mom_rcs",
            "scattering_mode": "monostatic",
            "polarizations": ["VV", "HH"],
            "polarization_mapping": {"VV": "VV", "HH": "HH"},
            "certification_frequency_scope": "single_frequency_unit",
            "certification_frequency_scope_ghz": [frequency],
            "metadata": common_metadata,
        }
    return diagnostics if complete_evidence else None


def bodies_from_units(units: 'Sequence[Dict[str, Any]]', stem: 'Optional[str]' = None
                      ) -> 'Dict[float, Dict[str, np.ndarray]]':
    """Per-unit BoR grims -> the {frequency: body solve} dict the local pipeline
    uses (``theta_deg``, ``amp_vv``, ``amp_hh``), i.e. exactly what
    feature_sum.solve_vehicle_body returns and sum_features expects.

    Any noncanonical aspects above 180 are dropped: they are the same monostatic
    response reflected, and the body interpolation works on 0..180.
    """
    stems = sorted({u["stem"] for u in units})
    if stem is None:
        if len(stems) != 1:
            raise ValueError(f"results hold several geometries {stems}; pass stem=")
        stem = stems[0]
    bodies: 'Dict[float, Dict[str, np.ndarray]]' = {}
    for u in units:
        if u["stem"] != stem:
            continue
        keep = u["angles_deg"] <= 180.0 + 1e-9
        th = u["angles_deg"][keep]
        polarizations = list(u.get("polarizations") or [u.get("pol")])
        amplitude = np.asarray(u["amp"])
        if len(polarizations) == 1 and amplitude.ndim == 2:
            amplitude = amplitude[:, :, None]
        if amplitude.ndim != 3 or amplitude.shape[2] != len(polarizations):
            raise ValueError(
                f"{u['path'].name}: amplitude/polarization axes disagree."
            )
        for kf, f in enumerate(u["frequencies_ghz"]):
            b = bodies.setdefault(float(f), {"theta_deg": th})
            if not np.array_equal(b["theta_deg"], th):
                raise ValueError(f"{u['path'].name}: aspect sweep differs from "
                                 f"the other polarization at {f} GHz.")
            for kp, polarization in enumerate(polarizations):
                key = {"VV": "amp_vv", "HH": "amp_hh",
                       "TE": "amp_vv", "TM": "amp_hh"}.get(polarization)
                if key is None:
                    raise ValueError(
                        f"{u['path'].name}: unexpected polarization "
                        f"{polarization!r}."
                    )
                b[key] = amplitude[keep, kf, kp]
    for f, b in sorted(bodies.items()):
        miss = [k for k in ("amp_vv", "amp_hh") if k not in b]
        if miss:
            raise ValueError(
                f"{f} GHz is missing {miss}; the body artifact must contain "
                "both VV and HH."
            )
    return bodies
