#!/usr/bin/env python3
"""
Shared plumbing for the HPC steps -- three small jobs, all of them explicit.

1. CONFIGURE A DRIVER.  The root drivers (run_hpc_bor_monostatic.py, and
   run_hpc_monostatic.py for 2-D) keep their settings in a CONFIG block of
   module-level constants, and the SLURM script they write execs THE SAME FILE
   with --worker.  So the compute nodes read the constants out of whichever copy
   submitted the job.  ``configure_driver`` therefore writes a COPY of the driver
   with those constants replaced -- and submitting that copy is what makes the
   settings reach the nodes.  Overriding them in your own process would not.

2. STAGE GEOMETRY.  The drivers discover every .geo under FRD_DIR / OPN_DIR, so
   each step stages exactly the files it wants into its own outputs/geometries/
   and points the configured copy at that (absolute) path.

3. COLLECT.  Each unit writes results/<POL>_<FREQ:.3f>GHz_<stem>.grim -- ONE
   FILE PER FREQUENCY.  The local pipeline wants one artefact per body
   (``body.grim``, both pols, all frequencies) or one grim per polarization
   spanning ALL frequencies for a coupon, so ``read_unit_grims`` /
   ``merge_frequency_grims`` / ``bodies_from_units`` put the pieces back together.
"""

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


# ─────────────────────────────────────────────────────────────────────────────
# 1. configure a copy of a driver
# ─────────────────────────────────────────────────────────────────────────────

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


def run_status(run_dir: 'os.PathLike') -> 'Dict[str, Any]':
    """How far along a run is, from the manifest and what is on disk."""
    import json
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text())
    done = sorted((run_dir / "results").glob("*.grim"))
    return {"run_dir": run_dir, "n_units": int(man["n_units"]),
            "n_done": len(done), "manifest": man, "done": done,
            "pending": int(man["n_units"]) - len(done)}


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
    valid_polarizations = {
        "ghost.hpc.2d-run.v1": {"TM", "TE"},
        "ghost.hpc.bor-run.v1": {"VV", "HH"},
    }.get(expected_schema)
    expected_angle_key = {
        "ghost.hpc.2d-run.v1": "azimuths_deg",
        "ghost.hpc.bor-run.v1": "aspects_deg",
    }.get(expected_schema)
    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("Every HPC unit must be an object.")
        angle_keys = [
            key for key in ("azimuths_deg", "aspects_deg")
            if key in unit
        ]
        if len(angle_keys) != 1:
            raise ValueError(
                "Every HPC unit must declare exactly one angular grid."
            )
        if (
            expected_angle_key is not None
            and angle_keys[0] != expected_angle_key
        ):
            raise ValueError(
                f"HPC unit angular grid is {angle_keys[0]!r}, expected "
                f"{expected_angle_key!r} for {expected_schema!r}."
            )
        raw_angles = unit.get(angle_keys[0])
        if (
            not isinstance(raw_angles, (list, tuple))
            or not raw_angles
        ):
            raise ValueError(
                "HPC unit has an empty or malformed angular grid."
            )
        angles = list(raw_angles)
        if not angles or not all(
            not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in angles
        ):
            raise ValueError(
                "HPC unit has an empty or non-finite angular grid."
            )
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
        polarization = unit.get("polarization")
        if (
            not isinstance(polarization, str)
            or (
                valid_polarizations is not None
                and polarization not in valid_polarizations
            )
        ):
            raise ValueError("HPC unit has an invalid polarization.")
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
        output_name = (
            f"{polarization}_{float(frequency):.3f}GHz_{stem}.grim"
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
    """Require the exact expected result set and byte-level attestations."""

    from workflow_provenance import (
        manifest_solve_spec_fingerprint,
        stable_json_fingerprint,
        unit_solve_spec_fingerprint,
        verify_output_attestation,
    )

    results_dir = Path(run_dir) / "results"
    if not isinstance(manifest, dict):
        raise ValueError("HPC manifest must be a JSON object.")
    schema = manifest.get("schema")
    valid_polarizations = {
        "ghost.hpc.2d-run.v1": {"TM", "TE"},
        "ghost.hpc.bor-run.v1": {"VV", "HH"},
    }.get(schema)
    expected_angle_key = {
        "ghost.hpc.2d-run.v1": "azimuths_deg",
        "ghost.hpc.bor-run.v1": "aspects_deg",
    }.get(schema)
    if valid_polarizations is None or expected_angle_key is None:
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
    expected_paths = []
    records = []
    seen_names = set()
    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("Every HPC unit must be an object.")
        stem = unit.get("geometry_stem")
        polarization = unit.get("polarization")
        frequency = unit.get("frequency_ghz")
        if (
            not isinstance(stem, str)
            or not stem
            or stem in (".", "..")
            or "/" in stem
            or "\\" in stem
            or not isinstance(polarization, str)
            or "/" in polarization
            or "\\" in polarization
            or polarization not in valid_polarizations
            or isinstance(frequency, bool)
            or not isinstance(frequency, (int, float))
            or not math.isfinite(float(frequency))
            or float(frequency) <= 0.0
        ):
            raise ValueError("HPC unit cannot form a safe output filename.")
        name = (
            f"{polarization}_"
            f"{float(frequency):.3f}GHz_"
            f"{stem}.grim"
        )
        if name in seen_names:
            raise ValueError(
                "HPC manifest contains colliding result filenames."
            )
        seen_names.add(name)
        path = results_dir / name
        expected_paths.append(path)
        angle_keys = [
            key for key in ("azimuths_deg", "aspects_deg")
            if key in unit
        ]
        if len(angle_keys) != 1:
            raise ValueError(
                f"HPC unit for {name} must have exactly one angular grid."
            )
        angle_key = angle_keys[0]
        if angle_key != expected_angle_key:
            raise ValueError(
                f"HPC unit for {name} uses {angle_key!r}, expected "
                f"{expected_angle_key!r} for {schema!r}."
            )
        raw_angles = unit.get(angle_key)
        if (
            not isinstance(raw_angles, (list, tuple))
            or not raw_angles
            or not all(
                not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in raw_angles
            )
        ):
            raise ValueError(
                f"HPC unit for {name} has an invalid angular grid."
            )
        geometry_hash = unit.get("geometry_input_sha256")
        if (
            not isinstance(geometry_hash, str)
            or len(geometry_hash) != 64
            or any(
                char not in "0123456789abcdef"
                for char in geometry_hash
            )
        ):
            raise ValueError(
                f"HPC unit for {name} has no input fingerprint."
            )
        records.append(
            (
                unit,
                name,
                path,
                angle_key,
                raw_angles,
                geometry_hash,
                polarization,
                frequency,
            )
        )

    for (
        unit,
        name,
        path,
        angle_key,
        raw_angles,
        geometry_hash,
        polarization,
        frequency,
    ) in records:
        verify_output_attestation(
            str(path),
            {
                "run_id": str(manifest["run_id"]),
                "solver_source_sha256":
                    str(manifest["solver_source_sha256"]),
                "runtime_environment_sha256":
                    str(manifest["runtime_environment_sha256"]),
                "geometry_input_sha256":
                    geometry_hash,
                "run_solve_spec_sha256":
                    manifest_solve_spec_fingerprint(manifest),
                "unit_solve_spec_sha256":
                    unit_solve_spec_fingerprint(unit),
                "solver_config_sha256":
                    stable_json_fingerprint(
                        manifest.get("solver_config", {})
                    ),
                "angular_grid_kind": angle_key,
                "angular_grid_deg": [
                    float(value) for value in raw_angles
                ],
                "polarization": polarization,
                "frequency_ghz": float(frequency),
            },
        )
    expected_names = {path.name for path in expected_paths}
    actual_names = {path.name for path in results_dir.glob("*.grim")}
    if actual_names != expected_names:
        raise ValueError(
            "HPC results do not contain the exact manifest output set "
            f"(missing={sorted(expected_names - actual_names)[:8]}, "
            f"unexpected={sorted(actual_names - expected_names)[:8]})."
        )
    expected_sidecars = {
        f"{name}.provenance.json" for name in expected_names
    }
    actual_sidecars = {
        path.name
        for path in results_dir.glob("*.grim.provenance.json")
    }
    if actual_sidecars != expected_sidecars:
        raise ValueError(
            "HPC results do not contain the exact attestation set "
            f"(missing={sorted(expected_sidecars - actual_sidecars)[:8]}, "
            f"unexpected={sorted(actual_sidecars - expected_sidecars)[:8]})."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. collect: read the per-unit grims back
# ─────────────────────────────────────────────────────────────────────────────

def read_unit_grims(results_dir: 'os.PathLike') -> 'List[Dict[str, Any]]':
    """Parse results/<POL>_<FREQ>GHz_<stem>.grim into
    {pol, freq_ghz, stem, path, angles_deg, amp[angle, freq]} records.

    ``angles_deg`` is the driver's sweep axis, stored in the grim's 'azimuths':
    for the BoR driver that is the ASPECT from the rotation axis (0 = nose-on,
    90 = broadside, 180 = tail-on -- a body of revolution has no second angle);
    for the 2-D driver it is the cut angle (90 = normal to the outer face).
    """
    out: 'List[Dict[str, Any]]' = []
    for p in sorted(Path(results_dir).glob("*.grim")):
        m = _UNIT_RE.match(p.name)
        if not m:
            continue
        with np.load(str(p)) as d:
            if not bool(d["raw_complex_amplitude_preserved"]):
                raise ValueError(f"{p.name}: complex amplitudes were not preserved.")
            amp = (np.asarray(d["rcs_amp_real"], float)
                   + 1j * np.asarray(d["rcs_amp_imag"], float))
            out.append({"pol": m.group("pol"), "freq_ghz": float(m.group("freq")),
                        "stem": m.group("stem"), "path": p,
                        "angles_deg": np.asarray(d["azimuths"], float),
                        "frequencies_ghz": np.asarray(d["frequencies"], float),
                        "amp": amp[:, 0, :, 0]})
    return out


def merge_frequency_grims(paths: 'Sequence[os.PathLike]', out_path: 'os.PathLike'
                          ) -> 'Path':
    """Concatenate several single-frequency grims of ONE polarization into one
    multi-frequency grim, sorted by frequency.

    Needed because feature_sum._amp_tables (so make_delta_grim, and the seam /
    wing loaders) requires every grim it is handed to share ONE (angle,
    frequency) grid -- it merges polarizations, not frequencies.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("no grims to merge.")
    base = {k: v for k, v in np.load(str(paths[0])).items()}
    ang = np.asarray(base["azimuths"], float)
    rows = []
    for p in paths:
        with np.load(str(p)) as d:
            if not np.array_equal(np.asarray(d["azimuths"], float), ang):
                raise ValueError(f"{p.name}: different angle sweep from "
                                 f"{paths[0].name} -- cannot merge.")
            pols = [str(x) for x in np.asarray(d["polarizations"]).ravel()]
            if len(pols) != 1:
                raise ValueError(f"{p.name}: expected one polarization, got {pols}.")
            for kf, f in enumerate(np.asarray(d["frequencies"], float)):
                rows.append((float(f),
                             np.asarray(d["rcs_amp_real"], float)[:, :, kf, :],
                             np.asarray(d["rcs_amp_imag"], float)[:, :, kf, :],
                             np.asarray(d["rcs_power"], float)[:, :, kf, :],
                             np.asarray(d["rcs_phase"], float)[:, :, kf, :]))
    rows.sort(key=lambda r: r[0])
    freqs = np.asarray([r[0] for r in rows], float)
    if len(set(freqs.tolist())) != len(freqs):
        raise ValueError("duplicate frequencies among the grims to merge.")
    base["frequencies"] = freqs
    # A per-unit solver audit cannot truthfully describe the merged frequency
    # sweep. The frozen HPC manifest and unit files retain the individual
    # audits; do not stamp the first unit's metadata onto every frequency.
    base.pop("solver_metadata_json", None)
    for key, idx in (("rcs_amp_real", 1), ("rcs_amp_imag", 2),
                     ("rcs_power", 3), ("rcs_phase", 4)):
        dtype = np.float64 if key.startswith("rcs_amp_") else np.float32
        base[key] = np.stack([r[idx] for r in rows], axis=2).astype(dtype)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from grim_io import _save_grim_npz
    return Path(_save_grim_npz(base, str(out_path)))


def bodies_from_units(units: 'Sequence[Dict[str, Any]]', stem: 'Optional[str]' = None
                      ) -> 'Dict[float, Dict[str, np.ndarray]]':
    """Per-unit BoR grims -> the {frequency: body solve} dict the local pipeline
    uses (``theta_deg``, ``amp_vv``, ``amp_hh``), i.e. exactly what
    feature_sum.solve_vehicle_body returns and sum_features expects.

    Aspects above 180 (the EXPAND_TO_360 mirror) are dropped: they are the same
    monostatic response reflected, and the local interpolation works on 0..180.
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
        for kf, f in enumerate(u["frequencies_ghz"]):
            b = bodies.setdefault(float(f), {"theta_deg": th})
            if not np.array_equal(b["theta_deg"], th):
                raise ValueError(f"{u['path'].name}: aspect sweep differs from "
                                 f"the other polarization at {f} GHz.")
            key = {"VV": "amp_vv", "HH": "amp_hh",
                   "TE": "amp_vv", "TM": "amp_hh"}.get(u["pol"])
            if key is None:
                raise ValueError(f"{u['path'].name}: unexpected polarization "
                                 f"{u['pol']!r}.")
            b[key] = u["amp"][keep, kf]
    for f, b in sorted(bodies.items()):
        miss = [k for k in ("amp_vv", "amp_hh") if k not in b]
        if miss:
            raise ValueError(f"{f} GHz is missing {miss} -- solve BOTH "
                             f"polarizations (POLARIZATIONS = ['VV', 'HH']).")
    return bodies
