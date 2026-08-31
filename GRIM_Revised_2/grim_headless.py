"""Qt-free GRIM loading, folder operations, and command-line entry point."""

from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from grim_csv_schema import (
    has_flat_csv_signature,
    load_flat_csv as _load_flat_csv_schema,
)
from grim_dataset import C0, RcsGrid, canonical_angular_coordinate_system
from plot_modes.isar_mode import form_isar


SUPPORTED_EXTENSIONS = (
    ".grim",
    ".csv",
    ".cst_data",
    ".txt",
    ".out",
    ".pio",
    ".cmplx_di",
    ".ptm",
    ".ss",
)


def is_supported_path(path: str) -> bool:
    return str(path).lower().endswith(SUPPORTED_EXTENSIONS)


def load_flat_csv(path: str) -> RcsGrid:
    """Load a versioned or deliberately supported legacy flat RCS table."""

    return _load_flat_csv_schema(
        path,
        grid_class=RcsGrid,
        canonical_angular_coordinate_system=canonical_angular_coordinate_system,
        c0=C0,
    )


def read_CST(path: str) -> RcsGrid:
    """Read a supported CST far-field table.

    This is the deliberately named public entry point.  ``RcsGrid.read_CST``
    recognizes both CST's wide theta/phi CSV export and the row-oriented
    ``.cst_data`` schema used by the team's MATLAB workflow.  Native GRIM flat
    CSV remains a separate format handled by :func:`load_flat_csv`.
    """
    return RcsGrid.read_CST(path)


def read_SENTRi(path: str) -> RcsGrid:
    """Read a CREATE-RF SENTRi RCS table with its vendor conventions."""

    return RcsGrid.read_SENTRi(path)


def load_dataset(path: str, *, allow_legacy_pickle=False) -> RcsGrid:
    """Load any GRIM-supported dataset without importing Qt."""
    path = str(path)
    lower = path.lower()
    if lower.endswith(".grim"):
        return RcsGrid.load(path, allow_legacy_pickle=allow_legacy_pickle)
    if lower.endswith(".out"):
        return RcsGrid.load_out(path)
    if lower.endswith(".ss"):
        return RcsGrid.load_ss(path)
    if lower.endswith(".ptm"):
        return RcsGrid.load_ptm(path)
    if lower.endswith((".pio", ".cmplx_di")):
        return RcsGrid.load_pio(path)
    if lower.endswith(".cst_data"):
        return read_CST(path)
    if lower.endswith((".csv", ".txt")) and RcsGrid.has_SENTRi_signature(path):
        # A recognized vendor header commits the file to the SENTRi parser.
        # Propagate corrupt-data errors instead of falling through to a loose
        # legacy numeric reader that could reinterpret the same row.
        return read_SENTRi(path)
    if lower.endswith((".csv", ".txt")) and has_flat_csv_signature(path):
        # A deliberate GRIM/CEM flat-table header also commits to its strict
        # parser.  Do not reinterpret a corrupt versioned table as CST or as a
        # loose theta/phi list after its schema has already identified it.
        return load_flat_csv(path)
    loaders = (
        (load_flat_csv, read_CST)
        if lower.endswith(".csv")
        else (
            RcsGrid.load_theta_phi_txt,
            load_flat_csv,
            read_CST,
        )
    )
    errors = []
    for loader in loaders:
        try:
            return loader(path)
        except Exception as exc:
            errors.append(f"{loader.__name__}: {exc}")
    raise ValueError("; ".join(errors))


def audit_dataset(dataset: RcsGrid, **kwargs):
    """Return the core read-only audit report for one loaded dataset."""

    return dataset.audit(**kwargs)


def _audit_json_default(value):
    """Serialize the numeric/container values used by audit reports."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value, key=str)
    for method_name in ("to_dict", "as_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            return method()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _matching_dataset_paths(folder: str, *, pattern="*", recursive=False):
    search = (
        os.path.join(str(folder), "**", pattern)
        if recursive
        else os.path.join(str(folder), pattern)
    )
    paths = [
        path
        for path in sorted(glob.glob(search, recursive=recursive))
        if os.path.isfile(path) and is_supported_path(path)
    ]
    if not paths:
        raise ValueError(f"no supported datasets matched {search!r}")
    return paths


def combine_datasets(
    grids,
    operation: str,
    *,
    overlap="error",
    max_output_bytes=None,
    coherent_metadata_attested=False,
    stitch_policy="priority-first",
    tol=1.0e-6,
):
    grids = list(grids)
    if not grids:
        raise ValueError("at least one dataset is required")
    if not isinstance(coherent_metadata_attested, (bool, np.bool_)):
        raise TypeError("coherent_metadata_attested must be True or False")
    operation = str(operation).strip().lower().replace("_", "-")
    if operation == "join":
        return RcsGrid.join_many(
            *grids,
            tol=float(tol),
            overlap=overlap,
            max_output_bytes=max_output_bytes,
        )
    if operation == "stitch":
        return RcsGrid.stitch_many(
            *grids,
            policy=stitch_policy,
            tol=float(tol),
            metadata_attested=coherent_metadata_attested,
            max_output_bytes=max_output_bytes,
            return_report=False,
        )
    result = grids[0]
    for grid in grids[1:]:
        if operation == "coherent-add":
            result = result.coherent_add(
                grid, metadata_attested=coherent_metadata_attested
            )
        elif operation == "incoherent-add":
            result = result.incoherent_add(grid)
        else:
            raise ValueError(
                "operation must be join, stitch, coherent-add, or incoherent-add"
            )
    return result


def load_folder(
    folder: str,
    *,
    pattern="*",
    recursive=False,
    operation="join",
    workers=1,
    overlap="error",
    max_output_bytes=None,
    coherent_metadata_attested=False,
    stitch_policy="priority-first",
    tol=1.0e-6,
):
    """Load matching files and combine them in deterministic pathname order."""
    paths = _matching_dataset_paths(
        folder, pattern=pattern, recursive=recursive
    )
    workers = max(1, int(workers))
    if workers == 1:
        grids = [load_dataset(path) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as pool:
            grids = list(pool.map(load_dataset, paths))
    return combine_datasets(
        grids,
        operation,
        overlap=overlap,
        max_output_bytes=max_output_bytes,
        coherent_metadata_attested=coherent_metadata_attested,
        stitch_policy=stitch_policy,
        tol=tol,
    )


def _parser():
    parser = argparse.ArgumentParser(description="Headless GRIM dataset operations")
    parser.add_argument("inputs", nargs="*", help="input files")
    parser.add_argument(
        "-o",
        "--output",
        help="output .grim path, or optional JSON report path with --audit",
    )
    parser.add_argument("--folder", help="load a folder instead of explicit inputs")
    parser.add_argument("--pattern", default="*", help="folder glob pattern")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--operation",
        choices=("join", "stitch", "coherent-add", "incoherent-add"),
        default="join",
        help=(
            "combination operation; join rejects conflicting finite overlaps, "
            "while stitch resolves them with --stitch-policy"
        ),
    )
    parser.add_argument("--overlap", choices=("error", "first", "last"), default="error")
    parser.add_argument(
        "--max-gib",
        type=float,
        default=None,
        help="maximum estimated dense output and working allocation",
    )
    parser.add_argument(
        "--stitch-policy",
        default="priority-first",
        help=(
            "stitch overlap policy: priority-first, priority-last, power-mean, "
            "or coherent-mean"
        ),
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-6,
        help="numeric coordinate matching tolerance for stitch",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help=(
            "audit each input independently and emit JSON instead of creating "
            "a derived dataset; --output is optional"
        ),
    )
    parser.add_argument(
        "--attest-coherent-metadata",
        action="store_true",
        help=(
            "legacy compatibility option: record an explicit user attestation "
            "for missing coherent metadata; operations no longer require it"
        ),
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    limit = None if args.max_gib is None else int(args.max_gib * 1024**3)
    if args.audit:
        if args.folder:
            paths = _matching_dataset_paths(
                args.folder, pattern=args.pattern, recursive=args.recursive
            )
        else:
            paths = list(args.inputs)
            if not paths:
                raise SystemExit("provide input files or --folder")
        reports = [
            {
                "input": os.path.abspath(path),
                "report": audit_dataset(load_dataset(path)),
            }
            for path in paths
        ]
        payload = {"operation": "audit", "datasets": reports}
        rendered = json.dumps(
            payload,
            default=_audit_json_default,
            indent=2,
            sort_keys=True,
        )
        if args.output:
            output = os.path.abspath(args.output)
            output_identity = os.path.normcase(os.path.realpath(output))
            input_paths = {
                os.path.normcase(os.path.realpath(os.path.abspath(path)))
                for path in paths
            }
            same_existing_file = False
            if os.path.exists(output):
                for path in paths:
                    try:
                        if os.path.samefile(output, path):
                            same_existing_file = True
                            break
                    except OSError:
                        continue
            if output_identity in input_paths or same_existing_file:
                raise SystemExit(
                    "audit --output must not overwrite an audited input dataset"
                )
            with open(output, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(rendered)
                stream.write("\n")
            print(output)
        else:
            print(rendered)
        return 0
    if args.folder:
        result = load_folder(
            args.folder, pattern=args.pattern, recursive=args.recursive,
            operation=args.operation, workers=args.workers, overlap=args.overlap,
            max_output_bytes=limit,
            coherent_metadata_attested=args.attest_coherent_metadata,
            stitch_policy=args.stitch_policy,
            tol=args.tol,
        )
    else:
        if not args.inputs:
            raise SystemExit("provide input files or --folder")
        result = combine_datasets(
            [load_dataset(path) for path in args.inputs], args.operation,
            overlap=args.overlap,
            max_output_bytes=limit,
            coherent_metadata_attested=args.attest_coherent_metadata,
            stitch_policy=args.stitch_policy,
            tol=args.tol,
        )
    if not args.output:
        raise SystemExit("--output is required")
    output = result.save(args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
