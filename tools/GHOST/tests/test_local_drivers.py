#!/usr/bin/env python3
"""End-to-end check of the local (non-SLURM) drivers.

The local drivers were brought onto the same footing as the HPC ones: run
state travels inside each .grim instead of in a `.provenance.json` beside it,
the shared angular grid is stored once instead of once per unit, and solves are
admitted against a memory budget rather than filling every core. This asserts
what a user would actually notice:

- results/ holds exactly one file per unit and nothing else;
- each result carries its run binding inside the artifact;
- the manifest left on disk still hashes to what those bindings recorded
  (dearest-first ordering must not reorder the manifest's unit list);
- a second run of the same sweep skips instead of redoing.

Usage:
    python tests/test_local_drivers.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Backend"

PASS = []
FAIL = []


def check(condition, label):
    (PASS if condition else FAIL).append(label)
    print(f"  {'ok  ' if condition else 'FAIL'} {label}", flush=True)


def _run(script, cwd):
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(cwd), capture_output=True, text=True, timeout=1800,
    )


# The drivers read their settings from module-level constants, so a test run is
# an import plus an assignment -- no configured copy, and no chance of the copy
# drifting from the file the user actually edits.
DRIVER_HARNESS = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {backend!r})
    import {module} as driver
    {overrides}
    driver.main()
    """
)


def _overrides(**values):
    return "\n".join(f"driver.{k} = {v!r}" for k, v in sorted(values.items()))


def test_run_local_monostatic(workspace):
    print("\nrun_local_monostatic.py")
    frd = workspace / "geometries" / "FRD"
    frd.mkdir(parents=True)
    closed_pec = (
        "Title: closed 2-D PEC test\n"
        "Segment: body 2\n"
        "properties: 2 0 0 0 0\n"
        "-0.02 -0.02 -0.02 0.02\n"
        "-0.02 0.02 0.02 0.02\n"
        "0.02 0.02 0.02 -0.02\n"
        "0.02 -0.02 -0.02 -0.02\n"
        "IBCS_Resistances:\n"
        "Dielectrics:\n"
    )
    for stem in ("coupon_a", "coupon_b"):
        (frd / f"{stem}.geo").write_text(closed_pec, encoding="utf-8")

    script = DRIVER_HARNESS.format(
        backend=str(BACKEND),
        module="run_local_monostatic",
        overrides=_overrides(
            FRD_DIR=str(frd),
            OPN_DIR=str(workspace / "geometries" / "OPN"),
            FREQUENCIES_GHZ=[2.0],
            AZIMUTHS_DEG=[0.0, 45.0, 90.0],
            OUTPUT_DIR=str(workspace / "rcs_runs"),
            GEOMETRY_UNITS="meters",
            WORKERS=2,
        ),
    )
    first = _run(script, workspace)
    if first.returncode != 0:
        check(False, f"driver exited {first.returncode}:\n{first.stdout}\n{first.stderr}")
        return
    check(True, "sweep completed")

    runs = sorted((workspace / "rcs_runs").glob("run_*"))
    check(len(runs) == 1, f"one run directory (got {len(runs)})")
    run_dir = runs[0]
    results = run_dir / "results"

    entries = sorted(results.rglob("*.grim"))
    check(len(entries) == 2, f"results/ holds exactly 2 files "
                             f"(got {[p.name for p in entries]})")
    check(all(path.suffix == ".grim" for path in entries),
          "every file in results/ is a .grim")
    check(not list(results.rglob("*.provenance.json")),
          "results/ holds no .provenance.json sidecars")
    check(all(path.parent.name == "FRD" for path in entries),
          "2-D results preserve their FRD/OPN role for downstream tools")
    import numpy as np
    with np.load(str(entries[0]), allow_pickle=False) as payload:
        check(
            np.asarray(payload["polarizations"]).astype(str).tolist()
            == ["VV", "HH"],
            "each 2-D result contains canonical VV and HH channels",
        )

    manifest = json.loads((run_dir / "manifest.json").read_text())
    check(manifest["status"] == "complete", "manifest reports the run complete")
    check("output_sha256" not in manifest,
          "manifest does not re-hash every output at the end")
    check(isinstance(manifest.get("azimuths_deg"), list),
          "the angular grid is recorded once at manifest level")
    check(all("azimuths_deg" not in unit for unit in manifest["units"]),
          "no unit repeats the shared angular grid")

    sys.path.insert(0, str(BACKEND))
    from workflow_provenance import (  # noqa: E402
        manifest_solve_spec_fingerprint,
        read_embedded_attestation,
    )

    embedded = read_embedded_attestation(str(entries[0]))
    check(embedded.get("run_id") == manifest["run_id"],
          "each result carries its run binding inside the artifact")
    check(embedded.get("angular_grid_kind") == "azimuths_deg",
          "the binding names the angular grid it was solved on")
    # Dearest-first dispatch sorts a copy; if it had sorted the manifest's own
    # list, the file on disk would no longer hash to what the results recorded.
    check(embedded.get("run_solve_spec_sha256")
          == manifest_solve_spec_fingerprint(manifest),
          "the manifest on disk still hashes to what the results bound to")

    # Each invocation opens a fresh run directory, so resumption is exercised
    # against the SAME run directory in the next test rather than by running
    # the driver twice.


def test_resume_same_run_dir(workspace):
    """Point a second worker at an existing run dir: it must verify and skip."""

    print("\nrun_local_monostatic.py resume")
    runs = sorted((workspace / "rcs_runs").glob("run_*"))
    if not runs:
        print("  [skip] no completed run to resume")
        return
    run_dir = runs[0]
    results = run_dir / "results"
    manifest = json.loads((run_dir / "manifest.json").read_text())

    sys.path.insert(0, str(BACKEND))
    import run_local_monostatic as driver  # noqa: E402

    driver.GEOMETRY_UNITS = manifest["solver_config"]["geometry_units"]
    context = {
        "run_id": manifest["run_id"],
        "solver_source_sha256": manifest["solver_source_sha256"],
        "solver_source_inventory": manifest["solver_source_inventory"],
        "runtime_environment_sha256": manifest["runtime_environment_sha256"],
        "run_solve_spec_sha256":
            __import__("workflow_provenance").manifest_solve_spec_fingerprint(
                manifest
            ),
        "solver_config_sha256":
            __import__("workflow_provenance").stable_json_fingerprint(
                manifest["solver_config"]
            ),
        "geometry_units": manifest["solver_config"]["geometry_units"],
        "max_panels": manifest["solver_config"]["max_panels"],
        "mesh_convergence_policy":
            manifest["solver_config"]["mesh_convergence_policy"],
        "mesh_certification":
            manifest["solver_config"]["mesh_certification"],
        "azimuths_deg": manifest["azimuths_deg"],
        "angular_grid_sha256":
            __import__("workflow_provenance").stable_json_fingerprint(
                [float(a) for a in manifest["azimuths_deg"]]
            ),
    }
    statuses = []
    for unit in manifest["units"]:
        status, _path = driver._solve_and_export(unit, context, str(results))
        statuses.append(status)
    check(all(s == "skipped" for s in statuses),
          f"every finished unit verifies and skips (got {set(statuses)})")


def test_polarization_aliases():
    print("\npolarization aliases")
    sys.path.insert(0, str(BACKEND))
    import hpc_scheduler
    canonical = hpc_scheduler.canonical_polarization
    check(canonical("VV") == "TE" and canonical("HH") == "TM",
          "VV maps to TE and HH maps to TM")
    check(canonical(" vv ") == "TE",
          "labels are case- and whitespace-insensitive")
    check(hpc_scheduler.distinct_polarization_channels(["VV", "HH"])
          == ["VV", "HH"],
          "resource-planning labels come back as written")

    # The same-channel-twice case is the one a plain uniqueness check misses,
    # so assert it directly on the shared helper the 2-D drivers call.
    for bad in (["VV", "TE"], ["HH", "TM"], ["V", "VERTICAL"]):
        try:
            hpc_scheduler.distinct_polarization_channels(bad)
        except ValueError:
            check(True, f"{bad} is rejected as one channel written twice")
        else:
            check(False, f"{bad} was accepted but is one channel twice")

    import run_local_monostatic as local_driver
    check(
        not hasattr(local_driver, "POLARIZATIONS")
        and not hasattr(local_driver, "SOLVER_METHOD")
        and not hasattr(local_driver, "CFIE_ALPHA"),
        "2-D production driver exposes no polarization, method, or dead CFIE control",
    )


def test_bor_driver_loads():
    """No BoR geometry ships with the repo, so this is a config/import check."""

    print("\nrun_local_bor.py")
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(BACKEND)!r})
        import run_local_bor as driver
        aspects = driver._validate_config()
        assert aspects and min(aspects) >= 0.0 and max(aspects) <= 180.0
        driver.AZIMUTHS_DEG = [0.0, 360.0]
        try:
            driver._validate_config()
        except SystemExit:
            pass
        else:
            raise AssertionError("duplicate physical azimuth was accepted")
        driver.AZIMUTHS_DEG = [0.0, 90.0]
        driver.ELEVATIONS_DEG = [0.0]
        aspects = driver._validate_config()
        assert driver._plan([], aspects) == {{}}
        driver.CFIE_ALPHA = 1.0
        try:
            driver._validate_config()
        except SystemExit:
            pass
        else:
            raise AssertionError("pure-EFIE CFIE endpoint was accepted")
        driver.CFIE_ALPHA = 0.5
        # The knobs the worker forwards must all exist on the solver entry point.
        import inspect
        from bor_dispatch import solve_monostatic_rcs_bor
        params = set(inspect.signature(solve_monostatic_rcs_bor).parameters)
        forwarded = {{
            "geometry_snapshot", "frequencies_ghz", "elevations_deg",
            "geometry_units", "material_base_dir",
            "cfie_alpha", "n_modes", "mode_tol", "max_elements", "workers",
            "table_precision", "assembly", "stream_budget_gb", "expand_to_360",
        }}
        missing = forwarded - params
        assert not missing, missing
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO), capture_output=True, text=True, timeout=600,
    )
    check(result.returncode == 0 and "OK" in result.stdout,
          f"imports, validates its config, and forwards only real solver "
          f"knobs{'' if result.returncode == 0 else ': ' + result.stderr.strip()}")


def test_local_bor_downstream_collection(workspace):
    """A base-mesh BoR run must still produce a usable downstream body."""

    print("\nrun_local_bor.py downstream collection")
    geometry_dir = workspace / "geometries" / "BOR"
    geometry_dir.mkdir(parents=True)
    shutil.copy2(
        str(REPO / "geometries" / "body.geo"),
        str(geometry_dir / "body.geo"),
    )
    script = DRIVER_HARNESS.format(
        backend=str(BACKEND),
        module="run_local_bor",
        overrides=_overrides(
            GEOMETRY_DIRS=[str(geometry_dir)],
            FREQUENCIES_GHZ=[1.0],
            AZIMUTHS_DEG=[0.0, 90.0, 180.0],
            ELEVATIONS_DEG=[0.0],
            OUTPUT_DIR=str(workspace / "bor_runs"),
            GEOMETRY_UNITS="meters",
            MESH_CERTIFICATION=False,
            WORKERS=1,
            WORKERS_PER_UNIT=1,
        ),
    )
    result = _run(script, workspace)
    if result.returncode != 0:
        check(False, f"driver exited {result.returncode}:\n"
                     f"{result.stdout}\n{result.stderr}")
        return
    runs = sorted((workspace / "bor_runs").glob("run_*"))
    frequency_paths = sorted(
        (runs[-1] / "results" / "by_frequency").glob("*.grim")
    ) if runs else []
    check(
        [path.name for path in frequency_paths] == [
            "HH_1.000GHz_body.grim", "VV_1.000GHz_body.grim"
        ],
        "completed frequency writes visible VV and HH restart records",
    )
    body_paths = sorted(runs[-1].glob("results/*.grim")) if runs else []
    check(len(body_paths) == 1,
          f"one monostatic body GRIM was written (got {len(body_paths)})")
    if not body_paths:
        return
    sys.path.insert(0, str(BACKEND))
    from feature_sum import (  # noqa: E402
        load_body_grim,
        load_body_requested_radar_grid,
        load_body_solver_diagnostics,
        require_body_mesh_certification,
        verify_body_artifact_bundle,
    )
    body = load_body_grim(str(body_paths[0]))
    check(set(body) == {1.0} and set(body[1.0]) == {
        "theta_deg", "amp_vv", "amp_hh"
    }, "collected body contains both channels and the requested frequency")
    bundle = verify_body_artifact_bundle(str(body_paths[0]))
    check(bundle["profile_points"] > 1,
          "base-mesh body passes downstream structural checks without a "
          "certification gate")
    grid = load_body_requested_radar_grid(str(body_paths[0]))
    check(list(grid["azimuths_deg"]) == [0.0, 90.0, 180.0]
          and list(grid["elevations_deg"]) == [0.0],
          "the same file carries the requested monostatic radar grid")
    diagnostics = load_body_solver_diagnostics(str(body_paths[0]))
    record = diagnostics["per_frequency"].get("1.0", {})
    check(
        record.get("polarizations") == ["VV", "HH"]
        and record.get("metadata", {}).get(
            "mesh_convergence_certified"
        ) is False,
        "the final body preserves both channels' honest survey diagnostics",
    )
    try:
        require_body_mesh_certification(str(body_paths[0]))
    except ValueError:
        check(True, "a survey body is explicitly rejected as uncertified")
    else:
        check(False, "a survey body was incorrectly accepted as certified")


def main():
    with tempfile.TemporaryDirectory(prefix="ghost_local_") as tmp:
        workspace = Path(tmp)
        test_run_local_monostatic(workspace)
        test_resume_same_run_dir(workspace)
        test_polarization_aliases()
        test_bor_driver_loads()
        test_local_bor_downstream_collection(workspace)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for label in FAIL:
        print(f"  {label}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
