#!/usr/bin/env python3
"""Coherently subtract OPN - FRD using the same engine as CEM Tools.

The inputs may be raw solver folders or joined folders; the subtraction engine
joins all compatible frequencies and polarizations before pairing each OPN
case with its most-specific compatible FRD baseline.
"""

import argparse
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
INPUT_ROOT = HERE / "Joined_Freqs"
OPN_DIR = INPUT_ROOT / "OPN"
FRD_DIR = INPUT_ROOT / "FRD"
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
    from cem_tools.grim_native import (
        load_grim,
        require_production_mesh_certification,
    )
    from cem_tools.operations import subtract_datasets
    return (
        CemToolError,
        subtract_datasets,
        load_grim,
        require_production_mesh_certification,
    )


def main() -> 'None':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opn_dir", nargs="?", default=str(OPN_DIR))
    parser.add_argument("frd_dir", nargs="?", default=str(FRD_DIR))
    parser.add_argument("output_dir", nargs="?", default=str(OUTPUT_DIR))
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    args = parser.parse_args()
    CemToolError, operation, load_grim, require_certification = _imports()
    try:
        for folder in (Path(args.opn_dir), Path(args.frd_dir)):
            paths = sorted(folder.resolve().glob("*.grim"))
            if not paths:
                raise CemToolError(f"no .grim files found in {folder}")
            for path in paths:
                require_certification(load_grim(path), str(path))
        result = operation(
            args.opn_dir,
            args.frd_dir,
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
