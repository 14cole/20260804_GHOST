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


def _find_compiler(requested: 'str | None') -> 'str | None':
    """Resolve an explicit compiler or common native Windows toolchains."""

    if requested:
        return shutil.which(requested) or (
            str(Path(requested).resolve())
            if Path(requested).is_file() else None
        )
    for name in ("cc", "gcc", "clang"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    if platform.system().lower() == "windows":
        for path in (
            Path("C:/msys64/ucrt64/bin/gcc.exe"),
            Path("C:/msys64/mingw64/bin/gcc.exe"),
        ):
            if path.is_file():
                return str(path)
    return None


def _compiler_environment(compiler: str, system_name: str) -> dict[str, str]:
    """Return an environment in which the compiler's helper tools can run.

    A native Windows MSYS2 GCC keeps cc1.exe's runtime DLLs beside gcc.exe.
    Finding gcc by its standard absolute path is therefore not sufficient:
    that directory must also be on PATH while GCC launches its subprocesses.
    """

    environment = os.environ.copy()
    if system_name == "windows":
        compiler_dir = str(Path(compiler).resolve().parent)
        current_path = environment.get("PATH", "")
        path_entries = current_path.split(os.pathsep) if current_path else []
        if os.path.normcase(compiler_dir) not in {
            os.path.normcase(entry) for entry in path_entries
        }:
            environment["PATH"] = os.pathsep.join(
                [compiler_dir, *path_entries]
            )
    return environment


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
        default=os.environ.get("CC"),
        help=(
            "C compiler command. By default, use CC, PATH, or a standard "
            "MSYS2 UCRT64 installation."
        ),
    )
    parser.add_argument(
        "--no-openmp",
        action="store_true",
        help="Build the native sampler without OpenMP parallel loops.",
    )
    args = parser.parse_args()

    compiler = _find_compiler(args.compiler)
    if compiler is None:
        raise SystemExit(
            "No compatible C compiler was found. Install MSYS2 UCRT64 GCC, "
            "put gcc on PATH, set CC, or pass --compiler explicitly."
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
    commands = [command]
    if not args.no_openmp:
        commands.insert(0, command[:1] + ["-fopenmp"] + command[1:])
    compiler_environment = _compiler_environment(compiler, system_name)
    try:
        completed = None
        failures: list[tuple[list[str], subprocess.CompletedProcess[str]]] = []
        for candidate in commands:
            completed = subprocess.run(
                candidate,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=compiler_environment,
            )
            if completed.returncode == 0:
                command = candidate
                break
            failures.append((candidate, completed))
        else:
            diagnostics = []
            for failed_command, failed in failures:
                diagnostics.append(
                    f"Command: {subprocess.list2cmdline(failed_command)}\n"
                    f"Exit code: {failed.returncode}\n"
                    f"{failed.stdout or '(compiler produced no output)'}"
                )
            raise SystemExit(
                "Native BoR sampler compilation failed:\n"
                + "\n\n".join(diagnostics)
            )
        dll_handle = None
        if system_name == "windows" and hasattr(os, "add_dll_directory"):
            dll_handle = os.add_dll_directory(str(Path(compiler).parent))
        try:
            library = ctypes.CDLL(str(temporary))
        finally:
            if dll_handle is not None:
                dll_handle.close()
        for symbol in ("sample_g", "sample_mfie", "sample_ibc"):
            getattr(library, symbol)
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    print(f"Built and load-checked {output}")
    print(
        "OpenMP sampling: "
        + ("enabled" if "-fopenmp" in command else "unavailable/disabled")
    )
    print("Re-run Python workers so they load the native kernel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
