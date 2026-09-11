"""Command-line interface for headless and HPC use."""

import argparse
import sys

from .errors import CemToolError
from .grim_bridge import OUTPUT_EXTENSIONS
from .operations import (
    concatenate_frequencies,
    concatenate_polarizations,
    convert_files,
    rename_files,
    subtract_datasets,
)


def _overwrite(parser: 'argparse.ArgumentParser') -> 'None':
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> 'argparse.ArgumentParser':
    parser = argparse.ArgumentParser(prog="cem-tools")
    commands = parser.add_subparsers(dest="command", required=True)
    subtract = commands.add_parser("subtract")
    subtract.add_argument("opn_dir")
    subtract.add_argument("frd_dir")
    subtract.add_argument("output_dir")
    _overwrite(subtract)
    pols = commands.add_parser("concat-pols")
    pols.add_argument("input_dir")
    pols.add_argument("output_dir")
    _overwrite(pols)
    frequencies = commands.add_parser("concat-freqs")
    frequencies.add_argument("input_dir")
    frequencies.add_argument("output_dir")
    _overwrite(frequencies)
    rename = commands.add_parser("rename")
    rename.add_argument("input_dir")
    rename.add_argument("keyword")
    rename.add_argument("replacement")
    rename.add_argument("--output-dir")
    rename.add_argument("--in-place", action="store_true")
    _overwrite(rename)
    convert = commands.add_parser("convert")
    convert.add_argument("input_dir")
    convert.add_argument("output_dir")
    convert.add_argument("extension", choices=OUTPUT_EXTENSIONS)
    _overwrite(convert)
    return parser


def main(argv: 'list[str] | None' = None) -> 'int':
    args = vars(build_parser().parse_args(argv))
    command = args.pop("command")
    functions = {
        "subtract": subtract_datasets,
        "concat-pols": concatenate_polarizations,
        "concat-freqs": concatenate_frequencies,
        "rename": rename_files,
        "convert": convert_files,
    }
    try:
        result = functions[command](**args)
    except CemToolError as exc:
        print(f"cem-tools: error: {exc}", file=sys.stderr)
        return 2
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
