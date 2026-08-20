#!/usr/bin/env python3
"""Coherently subtract OPN - FRD using the same engine as CEM Tools.

The inputs may be raw solver folders or joined folders; the subtraction engine
joins all compatible frequencies and polarizations before pairing each OPN
case with its most-specific compatible FRD baseline. With no paths, the newest
complete canonical 2-D run containing both roles is selected automatically.
"""

import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
INPUT_ROOT = None  # None = results/ from the newest complete canonical 2-D run
OUTPUT_DIR = HERE / "Deltas"
OVERWRITE = False


def _imports():
    for root in (HERE, *HERE.parents):
        package = root / "CEM_Tools"
        backend = root / "Backend"
        if (package / "cem_tools").is_dir():
            sys.path.insert(0, str(package))
            if backend.is_dir():
                os.environ.setdefault("CEM_SOLVER_BACKEND_PATH", str(backend))
            break
    from cem_tools.errors import CemToolError
    from cem_tools.operations import subtract_datasets
    return CemToolError, subtract_datasets


def _default_input_root() -> 'Path':
    runs = sorted((HERE.parent / "rcs_runs").glob("run_*"))
    for run in reversed(runs):
        try:
            manifest = json.loads((run / "manifest.json").read_text())
            results = run / "results"
            if (
                len(list(results.rglob("*.grim"))) == int(manifest["n_units"])
                and any((results / "OPN").glob("*.grim"))
                and any((results / "FRD").glob("*.grim"))
            ):
                return results
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    raise SystemExit(
        "no complete 2-D run with both OPN and FRD results was found in "
        "rcs_runs; pass OPN and FRD folders explicitly"
    )


def main() -> 'None':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opn_dir", nargs="?", default=INPUT_ROOT)
    parser.add_argument("frd_dir", nargs="?", default=INPUT_ROOT)
    parser.add_argument("output_dir", nargs="?", default=str(OUTPUT_DIR))
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    args = parser.parse_args()
    if args.opn_dir is None and args.frd_dir is None:
        root = _default_input_root()
        opn_dir, frd_dir = root / "OPN", root / "FRD"
    elif args.opn_dir is None or args.frd_dir is None:
        parser.error("pass both opn_dir and frd_dir, or neither")
    else:
        opn_dir, frd_dir = Path(args.opn_dir), Path(args.frd_dir)
    CemToolError, operation = _imports()
    try:
        for folder in (opn_dir, frd_dir):
            paths = sorted(folder.resolve().glob("*.grim"))
            if not paths:
                raise CemToolError(f"no .grim files found in {folder}")
        result = operation(
            opn_dir,
            frd_dir,
            args.output_dir,
            overwrite=args.overwrite,
        )
    except CemToolError as exc:
        raise SystemExit(str(exc)) from exc
    print(result.summary())
    for warning in result.warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
