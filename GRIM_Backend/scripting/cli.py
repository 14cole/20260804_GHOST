"""Command-line dataset loading, operations, audit, and export."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from GRIM_Backend.datasets.combine import combine_datasets
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.io.loaders import _matching_dataset_paths, load_dataset, load_folder


def audit_dataset(dataset: RcsGrid, **kwargs):
    """Return a non-mutating, JSON-serializable dataset health report.

    The report contains status, errors, warnings, info, and metrics. Samples are
    checked in bounded blocks; malformed data is reported without repair.
    """

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
