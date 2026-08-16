#!/usr/bin/env python3
"""Join separate-frequency GRIM files.

Edit INPUT_DIR/OUTPUT_DIR or pass them as command-line arguments. If the input
contains FRD/ and OPN/, that layout is preserved in the output.
"""

import argparse
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "Joined_Pols"
OUTPUT_DIR = HERE / "Joined_Freqs"
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
    from cem_tools.operations import concatenate_frequencies
    return CemToolError, concatenate_frequencies


def _role_folders(path: 'Path'):
    roles = [(role, path / role) for role in ("FRD", "OPN")]
    return roles if any(folder.is_dir() for _role, folder in roles) else [("", path)]


def main() -> 'None':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", default=str(INPUT_DIR))
    parser.add_argument("output_dir", nargs="?", default=str(OUTPUT_DIR))
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    args = parser.parse_args()
    source, destination = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    CemToolError, operation = _imports()
    wrote = 0
    try:
        for role, folder in _role_folders(source):
            if not folder.is_dir() or not any(folder.glob("*.grim")):
                print(f"skip empty {role or folder.name}")
                continue
            result = operation(
                folder,
                destination / role if role else destination,
                overwrite=args.overwrite,
            )
            wrote += len(result.written)
            print(f"{role or 'files'}: {result.summary()}")
    except CemToolError as exc:
        raise SystemExit(str(exc)) from exc
    if not wrote:
        raise SystemExit(f"no .grim inputs found in {source}")


if __name__ == "__main__":
    main()
