#!/usr/bin/env python3
"""
Explain a "solver source/native artifacts differ from the HPC run manifest" error.

That check hashes every top-level source and native artifact under Backend/,
plus the driver being executed, and compares the result to what the run
recorded at submit time. When it fires, the useful question is *which file* --
almost always a tree that was only partly updated, or code edited after the
run was submitted.

Run this on the compute node (or wherever the worker failed), against the run
directory it was pointed at:

    python tests/diagnose_provenance.py <run_dir> [--driver <driver_configured.py>]

Runs submitted before per-file inventories existed can only be diagnosed by
comparison, not by name; this says so rather than guessing.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Backend"
sys.path.insert(0, str(BACKEND))

import workflow_provenance as wp  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="run directory holding manifest.json")
    parser.add_argument(
        "--driver",
        help="driver actually executed (defaults to <run_dir>/driver_configured.py)",
    )
    parser.add_argument(
        "--backend",
        help="Backend directory to check (defaults to the one beside this script)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no manifest.json in {run_dir}")
    manifest = json.loads(manifest_path.read_text())

    backend_dir = Path(args.backend).resolve() if args.backend else BACKEND
    driver = Path(args.driver).resolve() if args.driver else (
        run_dir / "driver_configured.py"
    )
    extra = {"driver_configured.py": str(driver)} if driver.is_file() else {}
    if not extra:
        print(f"[warn] {driver} not found; checking Backend/ only, so the "
              "driver copy is excluded from this comparison.\n")

    recorded = str(manifest.get("solver_source_sha256", ""))
    current = wp.backend_source_fingerprint(str(backend_dir), extra)

    print(f"run directory : {run_dir}")
    print(f"backend       : {backend_dir}")
    print(f"driver        : {driver if extra else '(not checked)'}")
    print(f"recorded sha  : {recorded[:16]}...")
    print(f"current  sha  : {current[:16]}...")
    print(f"match         : {recorded == current}\n")

    inventory = manifest.get("solver_source_inventory")
    if not isinstance(inventory, dict) or not inventory:
        print("This run recorded no per-file inventory (submitted before that "
              "existed), so the differing file cannot be named from the "
              "manifest alone.")
        print("\nFiles the check covers right now:")
        for name in sorted(wp.backend_source_records(str(backend_dir), extra)):
            print(f"  {name}")
        print("\nCompare that list against the tree you submitted from — a "
              "file present in one and not the other is the usual cause.")
        return 0 if recorded == current else 1

    diff = wp.compare_source_inventories(inventory, wp.backend_source_inventory(
        str(backend_dir), extra
    ))
    if not any(diff.values()):
        print("Every recorded file matches. If the worker still fails, the "
              "runtime fingerprint is the other half of the check — compare "
              "Python, NumPy, SciPy, and BLAS versions against the submit host.")
        return 0

    labels = {
        "changed": "edited since the run was submitted",
        "added": "present now but not when the run was submitted",
        "removed": "recorded by the run but missing now",
    }
    for key, why in labels.items():
        if diff[key]:
            print(f"{len(diff[key])} file(s) {why}:")
            for name in diff[key]:
                print(f"    {name}")
            print()
    print("Fix either way round: restore the tree the run recorded, or submit "
          "a new run from the code you actually want to execute. Finished "
          "results in the old run stay valid — they were produced by the "
          "source that run recorded.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
