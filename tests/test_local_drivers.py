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
import os
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
    geo_src = REPO / "geometries" / "body.geo"
    if not geo_src.is_file():
        print("  [skip] geometries/body.geo not present")
        return

    frd = workspace / "geometries" / "FRD"
    frd.mkdir(parents=True)
    for stem in ("coupon_a", "coupon_b"):
        shutil.copy2(str(geo_src), str(frd / f"{stem}.geo"))

    script = DRIVER_HARNESS.format(
        backend=str(BACKEND),
        module="run_local_monostatic",
        overrides=_overrides(
            FRD_DIR=str(frd),
            OPN_DIR=str(workspace / "geometries" / "OPN"),
            FREQUENCIES_GHZ=[2.0],
            AZIMUTHS_DEG=[0.0, 45.0, 90.0],
            POLARIZATIONS=["TM"],
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

    entries = sorted(p.name for p in results.iterdir())
    check(len(entries) == 2, f"results/ holds exactly 2 files (got {entries})")
    check(all(name.endswith(".grim") for name in entries),
          "every file in results/ is a .grim")
    check(not list(results.glob("*.provenance.json")),
          "results/ holds no .provenance.json sidecars")

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

    embedded = read_embedded_attestation(str(results / entries[0]))
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
        "solver_method": manifest["solver_config"]["solver_method"],
        "cfie_alpha": manifest["solver_config"]["cfie_alpha"],
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


def test_step1_local(workspace):
    print("\n1a_solve_2d_local/run_monostatic_local.py")
    source = REPO / "1a_solve_2d_local"
    # A shipped coupon, not geometries/body.geo: this driver requires both
    # physical channels, and body.geo is an open TYPE 2 contour that the TE
    # Robin MFIE (a closed-obstacle formulation) legitimately refuses.
    candidates = sorted((source / "geometries" / "FRD").glob("*.geo"))
    if not source.is_dir() or not candidates:
        print("  [skip] no coupon in 1a_solve_2d_local/geometries/FRD")
        return
    geo_src = candidates[0]

    # The driver resolves Backend/ as a sibling of its own folder, so the copy
    # needs that same shape.
    sandbox = workspace / "step1"
    (sandbox / "1a_solve_2d_local" / "geometries" / "FRD").mkdir(parents=True)
    (sandbox / "1a_solve_2d_local" / "geometries" / "OPN").mkdir(parents=True)
    os.symlink(str(BACKEND), str(sandbox / "Backend"))
    shutil.copy2(
        str(source / "run_monostatic_local.py"),
        str(sandbox / "1a_solve_2d_local" / "run_monostatic_local.py"),
    )
    shutil.copy2(
        str(geo_src),
        str(sandbox / "1a_solve_2d_local" / "geometries" / "FRD" / "coupon_a.geo"),
    )

    here = sandbox / "1a_solve_2d_local"
    script = DRIVER_HARNESS.format(
        backend=str(BACKEND),
        module="run_monostatic_local",
        overrides=_overrides(
            FREQUENCIES_GHZ=[2.0],
            # validate_config requires both physical channels: a feature delta
            # is not complete without them.
            POLARIZATIONS=["TM", "TE"],
            GEOMETRY_UNITS="meters",
            WORKERS=2,
        ) + "\nimport numpy; driver.ANGLES_DEG = numpy.array([0.0, 45.0, 90.0])",
    )
    # sys.path[0] must reach the copied driver, not the repo original.
    script = script.replace(
        f"sys.path.insert(0, {str(BACKEND)!r})",
        f"sys.path.insert(0, {str(BACKEND)!r})\nsys.path.insert(0, {str(here)!r})",
    )
    first = _run(script, here)
    if first.returncode != 0:
        check(False, f"driver exited {first.returncode}:\n{first.stdout}\n{first.stderr}")
        return
    check(True, "sweep completed")

    files = [p for p in (here / "results").rglob("*") if p.is_file()]
    check(len(files) == 2, f"results/ holds exactly one file per unit "
                           f"(got {[p.name for p in files]})")
    check(all(p.suffix == ".grim" for p in files),
          "every published file is a .grim")
    check(not list((here / "results").rglob("*.provenance.json")),
          "results/ holds no .provenance.json sidecars")

    second = _run(script, here)
    check(second.returncode == 0, "a second run completes")
    check("skipped=2" in second.stdout,
          f"finished units are skipped, not redone:\n{second.stdout.strip()}")

    # The radar spellings name the same two physical channels, so a sweep
    # written as VV/HH must land on the results the TM/TE sweep already
    # produced rather than forking a second set of files under other names.
    alias_script = script.replace(
        f"driver.POLARIZATIONS = {['TM', 'TE']!r}",
        f"driver.POLARIZATIONS = {['VV', 'HH']!r}",
    )
    if alias_script == script:
        check(False, "could not reconfigure the run to VV/HH")
        return
    aliased = _run(alias_script, here)
    check(aliased.returncode == 0,
          f"a VV/HH run is accepted:\n{aliased.stdout}\n{aliased.stderr}")
    check("skipped=2" in aliased.stdout,
          f"VV/HH reuses the TM/TE results instead of re-solving:\n"
          f"{aliased.stdout.strip()}")
    after = [p for p in (here / "results").rglob("*") if p.is_file()]
    check(len(after) == 2,
          f"VV/HH added no second set of files (got {[p.name for p in after]})")
    check(all(p.name.startswith(("TM_", "TE_")) for p in after),
          f"outputs stay canonically named (got {[p.name for p in after]})")


def test_polarization_aliases():
    print("\npolarization aliases")
    sys.path.insert(0, str(BACKEND))
    import hpc_scheduler
    from step1_monostatic import validate_config

    canonical = hpc_scheduler.canonical_polarization
    check(canonical("VV") == "TE" and canonical("HH") == "TM",
          "VV maps to TE and HH maps to TM")
    check(canonical(" vv ") == "TE",
          "labels are case- and whitespace-insensitive")
    # BoR channels are theta-pol/phi-pol, so there VV/HH is canonical and
    # TM/TE are the aliases -- the same grouping, spelled the other way.
    bor = hpc_scheduler.canonical_bor_polarization
    check(bor("TE") == "VV" and bor("TM") == "HH" and bor("VV") == "VV",
          "the BoR spelling maps TE to VV and TM to HH")

    check(hpc_scheduler.distinct_polarization_channels(["VV", "HH"])
          == ["VV", "HH"],
          "labels come back as written when the driver names files with them")
    check(hpc_scheduler.distinct_polarization_channels(["TM", "TE"], bor)
          == ["HH", "VV"],
          "and canonicalized when the driver's file names use a fixed spelling")

    _f, _a, pols = validate_config([3.0], [0.0], ["VV", "HH"])
    check(sorted(pols) == ["TE", "TM"],
          f"a VV/HH step-1 config canonicalizes to TM/TE (got {pols})")

    for bad, why in (
        (["VV", "V"], "two spellings of the same channel"),
        (["TM", "HH"], "two spellings of the same channel"),
        (["TM"], "only one channel"),
        (["TM", "TE", "VV"], "a repeated channel"),
        (["RHCP", "LHCP"], "an unsupported label"),
    ):
        try:
            validate_config([3.0], [0.0], bad)
        except ValueError:
            check(True, f"step-1 rejects {bad} ({why})")
        else:
            check(False, f"step-1 accepted {bad} but it is {why}")

    # The same-channel-twice case is the one a plain uniqueness check misses,
    # so assert it directly on the shared helper the 2-D drivers call.
    for bad in (["VV", "TE"], ["HH", "TM"], ["V", "VERTICAL"]):
        try:
            hpc_scheduler.distinct_polarization_channels(bad)
        except ValueError:
            check(True, f"{bad} is rejected as one channel written twice")
        else:
            check(False, f"{bad} was accepted but is one channel twice")


def test_bor_driver_loads():
    """No BoR geometry ships with the repo, so this is a config/import check."""

    print("\nrun_local_bor.py")
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(BACKEND)!r})
        import run_local_bor as driver
        assert driver._validate_config() == ["VV", "HH"]
        # TM/TE is accepted and lands on the VV/HH names the az/el pairing
        # and the grim channel labels expect.
        driver.POLARIZATIONS = ["TM", "TE"]
        assert driver._validate_config() == ["HH", "VV"], driver._validate_config()
        driver.POLARIZATIONS = ["VV", "TE"]
        try:
            driver._validate_config()
        except SystemExit:
            pass
        else:
            raise AssertionError("one channel written twice was accepted")
        driver.POLARIZATIONS = ["VV", "HH"]
        assert driver._plan([]) == {{}}
        # The knobs the worker forwards must all exist on the solver entry point.
        import inspect
        from bor_dispatch import solve_monostatic_rcs_bor
        params = set(inspect.signature(solve_monostatic_rcs_bor).parameters)
        forwarded = {{
            "geometry_snapshot", "frequencies_ghz", "elevations_deg",
            "polarization", "geometry_units", "material_base_dir",
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


def main():
    with tempfile.TemporaryDirectory(prefix="ghost_local_") as tmp:
        workspace = Path(tmp)
        test_run_local_monostatic(workspace)
        test_resume_same_run_dir(workspace)
        test_step1_local(workspace)
        test_polarization_aliases()
        test_bor_driver_loads()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for label in FAIL:
        print(f"  {label}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
