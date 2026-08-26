#!/usr/bin/env python3
"""Build and load-check the optional native BoR streaming sampler."""

import argparse
import ctypes
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Destination directory (default: Backend beside the loader).",
    )
    parser.add_argument(
        "--compiler",
        default=os.environ.get("CC", "cc"),
        help="C compiler command (default: CC or cc).",
    )
    args = parser.parse_args()

    compiler = shutil.which(args.compiler)
    if compiler is None:
        raise SystemExit(
            f"C compiler {args.compiler!r} was not found. Install a C "
            "compiler or set CC."
        )
    source = Path(__file__).resolve().with_name("bor_stream_kernel.c")
    system_name = platform.system().lower()
    tag = f"{system_name}-{platform.machine().lower()}"
    output_extension = ".dll" if system_name == "windows" else ".so"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"bor_stream_kernel.{tag}{output_extension}"
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    command = [compiler, "-O3", "-std=c99", "-shared"]
    if system_name != "windows":
        command.append("-fPIC")
    command.extend(["-o", str(temporary), str(source), "-lm"])
    try:
        subprocess.run(command, check=True)
        library = ctypes.CDLL(str(temporary))
        for symbol in ("sample_g", "sample_mfie", "sample_ibc"):
            getattr(library, symbol)
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    print(f"Built and load-checked {output}")
    print("Re-run Python workers so they load the native kernel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
