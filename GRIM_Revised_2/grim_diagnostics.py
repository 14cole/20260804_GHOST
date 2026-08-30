"""Read-only installation diagnostics for the combined GRIM desktop tool.

The diagnostic intentionally depends only on the Python standard library so
it can explain a broken GUI installation instead of failing with the same
missing import.  It verifies the source-tree layout selected by GRIM, probes
the dependencies imported during GUI startup, and reports optional PowerPoint
and GHOST acceleration capabilities without starting PowerPoint or a solver.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import importlib
from importlib import machinery, metadata
import os
from pathlib import Path
import platform
import re
import sys
from typing import Callable, Iterable, Mapping, Sequence, TextIO
import uuid


MINIMUM_PYTHON = (3, 10)

# Importing ``grim_cut_gui`` eagerly imports these repository modules before a
# window can be shown.  Keep this explicit, standard-library-only manifest in
# the diagnostics module so both the release builder and the diagnostic check
# validate the same startup contract without importing Qt, Matplotlib, NumPy,
# or SciPy.  Lazy format readers such as ``ptm_io`` and ``read_ss`` are still
# packaged, but they are not part of the launch-time gate.
GRIM_STARTUP_FILES = (
    "grim_cut_gui.py",
    "assembly_tree.py",
    "assembly_workspace.py",
    "feature_assembly_panel.py",
    "freddy_integration.py",
    "ghost_integration.py",
    "hpc_remote.py",
    "grim_dataset.py",
    "grim_diagnostics.py",
    "grim_headless.py",
    "grim_python.py",
    "isar_artifact.py",
    "grim_cut_dataset_mixin.py",
    "grim_cut_plot_mixin.py",
    "ppt_workspace.py",
    "ppt_plot_data.py",
    "ppt_report.py",
    "plot_models.py",
    "runs_workspace.py",
    Path("plot_modes") / "__init__.py",
    Path("plot_modes") / "az_vs_range_mode.py",
    Path("plot_modes") / "azimuth_polar_mode.py",
    Path("plot_modes") / "azimuth_rect_mode.py",
    Path("plot_modes") / "compare_mode.py",
    Path("plot_modes") / "elevation_sweep_mode.py",
    Path("plot_modes") / "frequency_mode.py",
    Path("plot_modes") / "isar_mode.py",
    Path("plot_modes") / "waterfall_mode.py",
)

# Backwards-compatible name retained for callers and tests that used the
# original, smaller sentinel tuple.
GRIM_SENTINELS = GRIM_STARTUP_FILES

# Keep this aligned with the reusable workspace contract in ghost_integration.
GHOST_SENTINELS = (
    "ghost_gui.py",
    "geometry_tab.py",
    "solver_tab.py",
    "geometry_io.py",
    "grim_io.py",
    "rcs_solver.py",
    "bor_dispatch.py",
    "bor_solver.py",
    "bor_kernels.py",
    "bor_streaming.py",
    "components.py",
    "solver_quality.py",
    "assembly_workload.py",
    "feature_sum.py",
    "feature_workflow.py",
    "hpc_bundle.py",
    "hpc_common.py",
    "hpc_scheduler.py",
    "run_hpc_bor_monostatic.py",
    "run_hpc_monostatic.py",
    "fmm_helmholtz_2d.py",
    "frame.py",
    "line_expand.py",
    "occluder.py",
    "surface_mesh.py",
    "workflow_provenance.py",
)

FREDDY_SENTINELS = (
    Path("ibc") / "__init__.py",
    Path("ibc") / "batch.py",
    Path("ibc") / "compute.py",
    Path("ibc") / "io.py",
    Path("ibc") / "material_explorer.py",
    Path("ibc") / "plot.py",
    Path("ibc") / "ui.py",
)


@dataclass(frozen=True)
class DiagnosticResult:
    """One user-facing diagnostic outcome."""

    key: str
    name: str
    status: str
    required: bool
    summary: str
    details: tuple[str, ...] = ()

    @property
    def blocks_startup(self) -> bool:
        return self.required and self.status == "FAIL"


@dataclass(frozen=True)
class DependencyProbe:
    available: bool
    version: str = ""
    detail: str = ""


DependencyProbeFunction = Callable[[str, str], DependencyProbe]
LibraryProbeFunction = Callable[[Sequence[Path], Sequence[str]], tuple[Path | None, str]]
PowerPointProbeFunction = Callable[[], tuple[bool, str]]


def default_repository_root() -> Path:
    """Return the complete source-tree root containing ``GRIM_Revised_2``."""

    return Path(__file__).resolve().parents[1]


def _resolved(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def _missing_files(base: Path, relative_paths: Iterable[str | Path]) -> list[str]:
    return [str(value) for value in relative_paths if not (base / value).is_file()]


def _ghost_backend_from(value: str | os.PathLike[str]) -> Path:
    candidate = _resolved(value)
    if candidate.name.lower() != "backend" and (candidate / "Backend").is_dir():
        candidate = (candidate / "Backend").resolve()
    return candidate


def _freddy_root_from(value: str | os.PathLike[str]) -> Path:
    candidate = _resolved(value)
    if candidate.name.lower() == "ibc" and candidate.is_dir():
        candidate = candidate.parent.resolve()
    return candidate


def _select_ghost_backend(
    repository_root: Path,
    environ: Mapping[str, str],
) -> tuple[Path, str, str]:
    configured = str(environ.get("GHOST_BACKEND_PATH", "")).strip()
    if configured:
        return (
            _ghost_backend_from(configured),
            "GHOST_BACKEND_PATH override",
            "An explicit GHOST override is authoritative; GRIM will not fall back "
            "to the bundled backend when that override is incomplete.",
        )
    return (
        (repository_root / "tools" / "GHOST" / "Backend").resolve(),
        "bundled tools/GHOST backend",
        "",
    )


def _select_freddy_root(
    repository_root: Path,
    environ: Mapping[str, str],
) -> tuple[Path, str, str]:
    bundled = (repository_root / "tools" / "FREDDY").resolve()
    configured = str(environ.get("FREDDY_ROOT_PATH", "")).strip()
    if configured:
        override = _freddy_root_from(configured)
        missing = _missing_files(override, FREDDY_SENTINELS)
        if not missing:
            return (
                override,
                "FREDDY_ROOT_PATH override",
                "The explicit FREDDY override is complete and is selected first.",
            )
        if not _missing_files(bundled, FREDDY_SENTINELS):
            return (
                bundled,
                "bundled tools/FREDDY fallback",
                "FREDDY_ROOT_PATH is incomplete and will be skipped: "
                f"{override} (missing {', '.join(missing)}).",
            )
        return (
            override,
            "incomplete FREDDY_ROOT_PATH override",
            "The override is incomplete and the bundled FREDDY package is also "
            "unavailable.",
        )
    return bundled, "bundled tools/FREDDY package", ""


def _module_spec_origin(module_name: str, search_path: Path) -> Path | None:
    """Resolve a module from exactly one authoritative directory, without import."""

    try:
        spec = machinery.PathFinder.find_spec(module_name, [str(search_path)])
    except (ImportError, OSError, ValueError):
        return None
    value = getattr(spec, "origin", None) if spec is not None else None
    if not value or value in {"built-in", "frozen"}:
        return None
    try:
        return Path(value).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _loaded_module_conflicts(module_names: Iterable[str], expected_dir: Path) -> list[str]:
    conflicts: list[str] = []
    for name in sorted(set(module_names)):
        module = sys.modules.get(name)
        if module is None:
            continue
        value = getattr(module, "__file__", None)
        if not value:
            continue
        try:
            origin = Path(value).resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            conflicts.append(f"{name} (unknown origin)")
            continue
        if origin.parent != expected_dir:
            conflicts.append(f"{name} ({origin})")
    return conflicts


def _backend_module_names(backend: Path) -> tuple[str, ...]:
    """Mirror the embedded GHOST bridge's flat-module origin boundary."""

    return tuple(
        sorted(
            path.stem
            for path in backend.glob("*.py")
            if path.stem != "__init__"
        )
    )


def _release_tuple(value: str) -> tuple[int, ...] | None:
    match = re.match(r"\s*(\d+(?:\.\d+)*)", str(value))
    if not match:
        return None
    return tuple(int(piece) for piece in match.group(1).split("."))


def _meets_minimum(value: str, minimum: str) -> bool | None:
    installed = _release_tuple(value)
    required = _release_tuple(minimum)
    if installed is None or required is None:
        return None
    width = max(len(installed), len(required))
    return installed + (0,) * (width - len(installed)) >= required + (0,) * (
        width - len(required)
    )


def _default_dependency_probe(module_name: str, distribution: str) -> DependencyProbe:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # imports can fail with DLL/load errors, not just ImportError
        return DependencyProbe(False, detail=f"{type(exc).__name__}: {exc}")
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        version = "unknown"
    return DependencyProbe(True, version=version)


def _dependency_result(
    *,
    key: str,
    name: str,
    module_name: str,
    distribution: str,
    minimum: str,
    required: bool,
    purpose: str,
    probe: DependencyProbeFunction,
) -> DiagnosticResult:
    try:
        outcome = probe(module_name, distribution)
    except Exception as exc:
        outcome = DependencyProbe(False, detail=f"probe failed: {type(exc).__name__}: {exc}")
    if not outcome.available:
        return DiagnosticResult(
            key,
            name,
            "FAIL" if required else "WARN",
            required,
            f"not importable; {purpose}",
            (outcome.detail,) if outcome.detail else (),
        )
    comparison = _meets_minimum(outcome.version, minimum)
    if comparison is False:
        return DiagnosticResult(
            key,
            name,
            "FAIL" if required else "WARN",
            required,
            f"version {outcome.version} is below the supported minimum {minimum}",
            (purpose,),
        )
    version_text = outcome.version or "unknown"
    suffix = "" if comparison is True else " (version metadata unavailable)"
    return DiagnosticResult(
        key,
        name,
        "PASS",
        required,
        f"{version_text}{suffix}; {purpose}",
    )


def _default_library_probe(
    candidates: Sequence[Path],
    required_symbols: Sequence[str],
) -> tuple[Path | None, str]:
    existing: list[Path] = []
    failures: list[str] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            continue
        existing.append(path)
        try:
            library = ctypes.CDLL(str(path))
        except OSError as exc:
            failures.append(f"{path.name}: {exc}")
            continue
        missing = [name for name in required_symbols if not hasattr(library, name)]
        if missing:
            failures.append(f"{path.name}: missing {', '.join(missing)}")
            continue
        return path, ""
    if failures:
        return None, "Found native file(s), but none loaded for this interpreter: " + "; ".join(
            failures
        )
    if existing:
        return None, "Native file(s) were found but were not usable."
    return None, "No matching native binary was found."


def _native_library_extensions(system_name: str) -> tuple[str, ...]:
    """Return only library formats the named host can safely load."""

    key = str(system_name).strip().lower()
    if key == "windows":
        return (".dll",)
    return (".so",)


def _native_results(
    backend: Path,
    *,
    system_name: str,
    machine_name: str,
    library_probe: LibraryProbeFunction,
) -> list[DiagnosticResult]:
    tag = f"{system_name.lower()}-{machine_name.lower()}"
    extensions = _native_library_extensions(system_name)
    results: list[DiagnosticResult] = []

    cython_origin = _module_spec_origin("fmm_near_cy", backend)
    if cython_origin is not None and cython_origin.parent == backend:
        results.append(
            DiagnosticResult(
                "native_fmm",
                "GHOST 2-D near-field acceleration",
                "PASS",
                False,
                f"current-Python Cython extension found: {cython_origin.name}",
            )
        )
    else:
        fmm_candidates = [
            backend / f"{base}{extension}"
            for base in (f"fmm_near.{tag}", "fmm_near")
            for extension in extensions
        ]
        loaded, detail = library_probe(fmm_candidates, ("compute_sk_blocks_batch_q",))
        if loaded is not None:
            results.append(
                DiagnosticResult(
                    "native_fmm",
                    "GHOST 2-D near-field acceleration",
                    "PASS",
                    False,
                    f"native library loaded: {loaded.name}",
                )
            )
        else:
            found = sorted(
                path.name
                for path in backend.glob("fmm_near*")
                if path.suffix.lower() in {".so", ".dll", ".pyd"}
            )
            extra = (f"Native files present: {', '.join(found)}.",) if found else ()
            results.append(
                DiagnosticResult(
                    "native_fmm",
                    "GHOST 2-D near-field acceleration",
                    "WARN",
                    False,
                    "unavailable; the solver remains usable with a much slower fallback",
                    (detail,) + extra,
                )
            )

    bor_candidates = tuple(
        backend / f"{base}{extension}"
        for base in (f"bor_stream_kernel.{tag}", "bor_stream_kernel")
        for extension in extensions
    )
    loaded, detail = library_probe(
        bor_candidates,
        ("sample_g", "sample_mfie", "sample_ibc"),
    )
    if loaded is not None:
        results.append(
            DiagnosticResult(
                "native_bor",
                "GHOST BoR streaming acceleration",
                "PASS",
                False,
                f"native library loaded: {loaded.name}",
            )
        )
    else:
        found = sorted(
            path.name
            for path in backend.glob("bor_stream_kernel.*")
            if path.suffix.lower() in {".so", ".dll"}
        )
        extra = (f"Native files present: {', '.join(found)}.",) if found else ()
        results.append(
            DiagnosticResult(
                "native_bor",
                "GHOST BoR streaming acceleration",
                "WARN",
                False,
                "unavailable; BoR streaming uses the equivalent NumPy fallback",
                (detail,) + extra,
            )
        )
    return results


def _registered_com_clsid(
    prog_id: str,
    registry_module=None,
) -> tuple[str, str]:
    """Return a registered local COM CLSID and server without activation.

    Recent pywin32 releases do not expose ``pythoncom.CLSIDFromProgID`` on
    every supported build.  The merged Windows classes registry is the
    authoritative, read-only place to check registration without starting the
    application.  Probe both registry views so a 32-bit Office installation is
    still diagnosed correctly from 64-bit Python.
    """

    if registry_module is None:
        import winreg as registry_module

    registry = registry_module
    views = []
    for value in (
        0,
        getattr(registry, "KEY_WOW64_64KEY", 0),
        getattr(registry, "KEY_WOW64_32KEY", 0),
    ):
        if value not in views:
            views.append(value)

    failures: list[str] = []
    for view in views:
        access = int(getattr(registry, "KEY_READ", 0)) | int(view)
        try:
            with registry.OpenKey(
                registry.HKEY_CLASSES_ROOT,
                rf"{prog_id}\CLSID",
                0,
                access,
            ) as key:
                raw_clsid, _value_type = registry.QueryValueEx(key, None)
            parsed = uuid.UUID(str(raw_clsid).strip().strip("{}"))
            clsid = "{" + str(parsed).upper() + "}"
            with registry.OpenKey(
                registry.HKEY_CLASSES_ROOT,
                rf"CLSID\{clsid}\LocalServer32",
                0,
                access,
            ) as key:
                raw_server, _value_type = registry.QueryValueEx(key, None)
            server = str(raw_server).strip()
            if not server:
                raise OSError("LocalServer32 is blank")
            return clsid, server
        except (OSError, TypeError, ValueError) as exc:
            failures.append(f"view {view:#x}: {type(exc).__name__}: {exc}")

    raise OSError(
        f"{prog_id} has no usable CLSID/LocalServer32 registration ("
        + "; ".join(failures)
        + ")"
    )


def _default_powerpoint_probe() -> tuple[bool, str]:
    """Check pywin32 and COM registration without launching PowerPoint."""

    try:
        pythoncom = importlib.import_module("pythoncom")
        importlib.import_module("win32com.client")
    except Exception as exc:
        return False, f"pywin32 is not importable: {type(exc).__name__}: {exc}"
    try:
        clsid, _server = _registered_com_clsid("PowerPoint.Application")
    except Exception as exc:
        return False, f"desktop PowerPoint is not registered for COM: {type(exc).__name__}: {exc}"
    try:
        version = metadata.version("pywin32")
    except metadata.PackageNotFoundError:
        version = "unknown"
    return (
        True,
        f"pywin32 {version}; PowerPoint.Application is registered as {clsid}. "
        "PowerPoint was not launched.",
    )


def _powerpoint_result(
    *,
    system_name: str,
    probe: PowerPointProbeFunction,
) -> DiagnosticResult:
    if system_name.lower() != "windows":
        return DiagnosticResult(
            "powerpoint",
            "PowerPoint export",
            "SKIP",
            False,
            "optional export requires Windows, pywin32, and desktop Microsoft PowerPoint",
        )
    try:
        ready, detail = probe()
    except Exception as exc:
        ready, detail = False, f"probe failed: {type(exc).__name__}: {exc}"
    return DiagnosticResult(
        "powerpoint",
        "PowerPoint export",
        "PASS" if ready else "WARN",
        False,
        detail,
        () if not ready else ("Registration was checked without rendering or opening a report.",),
    )


def collect_diagnostics(
    repository_root: str | os.PathLike[str] | None = None,
    *,
    module_directory: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    dependency_probe: DependencyProbeFunction | None = None,
    system_name: str | None = None,
    machine_name: str | None = None,
    library_probe: LibraryProbeFunction | None = None,
    powerpoint_probe: PowerPointProbeFunction | None = None,
) -> list[DiagnosticResult]:
    """Collect all checks without opening a window, solver, or Office app."""

    root = _resolved(repository_root or default_repository_root())
    grim_dir = _resolved(module_directory or Path(__file__).resolve().parent)
    environment = os.environ if environ is None else environ
    dep_probe = dependency_probe or _default_dependency_probe
    lib_probe = library_probe or _default_library_probe
    host_system = str(system_name or platform.system())
    host_machine = str(machine_name or platform.machine())
    ppt_probe = powerpoint_probe or _default_powerpoint_probe
    results: list[DiagnosticResult] = []

    py_ok = sys.version_info[:2] >= MINIMUM_PYTHON
    results.append(
        DiagnosticResult(
            "python",
            "Python runtime",
            "PASS" if py_ok else "FAIL",
            True,
            f"{platform.python_version()} ({sys.executable}); requires 3.10 or newer",
        )
    )

    grim_missing = _missing_files(grim_dir, GRIM_SENTINELS)
    expected_grim_dir = (root / "GRIM_Revised_2").resolve()
    grim_origin = _module_spec_origin("grim_cut_gui", grim_dir)
    grim_details: list[str] = [f"Repository root: {root}", f"GRIM modules: {grim_dir}"]
    if grim_dir != expected_grim_dir:
        grim_details.append(f"Expected modules below this tree: {expected_grim_dir}")
    if grim_missing:
        grim_details.append("Missing: " + ", ".join(grim_missing))
    if grim_origin is None:
        grim_details.append("grim_cut_gui could not be resolved from that directory.")
    grim_ok = not grim_missing and grim_dir == expected_grim_dir and grim_origin == (
        grim_dir / "grim_cut_gui.py"
    ).resolve()
    results.append(
        DiagnosticResult(
            "grim_source",
            "GRIM authoritative source",
            "PASS" if grim_ok else "FAIL",
            True,
            "complete source-checkout module path" if grim_ok else "source path is incomplete or not authoritative",
            tuple(grim_details),
        )
    )

    ghost_backend, ghost_source, ghost_note = _select_ghost_backend(root, environment)
    ghost_missing = _missing_files(ghost_backend, GHOST_SENTINELS)
    ghost_details = [f"Selected: {ghost_backend}", f"Source: {ghost_source}"]
    if ghost_note:
        ghost_details.append(ghost_note)
    if ghost_missing:
        ghost_details.append("Missing: " + ", ".join(ghost_missing))
    results.append(
        DiagnosticResult(
            "ghost_workspace",
            "GHOST workspace files",
            "PASS" if not ghost_missing else "FAIL",
            True,
            "all required backend sentinels are present" if not ghost_missing else "selected backend is incomplete",
            tuple(ghost_details),
        )
    )
    ghost_origin = _module_spec_origin("ghost_gui", ghost_backend)
    # Runtime GHOST discovery validates every flat Backend module, not only
    # the small sentinel subset. Diagnostics must catch the same stale
    # ``frame``/``components``/workflow imports before the GUI is launched.
    ghost_conflicts = _loaded_module_conflicts(
        _backend_module_names(ghost_backend), ghost_backend
    )
    expected_ghost_origin = (ghost_backend / "ghost_gui.py").resolve()
    ghost_path_ok = ghost_origin == expected_ghost_origin and not ghost_conflicts
    ghost_path_details = [
        f"ghost_gui resolves to: {ghost_origin or 'not found'}",
        f"Expected: {expected_ghost_origin}",
    ]
    if ghost_conflicts:
        ghost_path_details.append("Already loaded elsewhere: " + ", ".join(ghost_conflicts))
    results.append(
        DiagnosticResult(
            "ghost_origin",
            "GHOST authoritative module path",
            "PASS" if ghost_path_ok else "FAIL",
            True,
            "module resolution is confined to the selected backend" if ghost_path_ok else "module origin mismatch",
            tuple(ghost_path_details),
        )
    )

    freddy_root, freddy_source, freddy_note = _select_freddy_root(root, environment)
    freddy_missing = _missing_files(freddy_root, FREDDY_SENTINELS)
    freddy_status = "PASS" if not freddy_missing else "FAIL"
    # A stale FREDDY override is recoverable because the integration searches
    # the bundled package next, but it should be visible to the user.
    if not freddy_missing and freddy_note.startswith("FREDDY_ROOT_PATH is incomplete"):
        freddy_status = "WARN"
    freddy_details = [f"Selected: {freddy_root}", f"Source: {freddy_source}"]
    if freddy_note:
        freddy_details.append(freddy_note)
    if freddy_missing:
        freddy_details.append("Missing: " + ", ".join(freddy_missing))
    results.append(
        DiagnosticResult(
            "freddy_workspace",
            "FREDDY workspace files",
            freddy_status,
            True,
            "all required package sentinels are present" if not freddy_missing else "selected package is incomplete",
            tuple(freddy_details),
        )
    )
    freddy_origin = _module_spec_origin("ibc", freddy_root)
    expected_freddy_origin = (freddy_root / "ibc" / "__init__.py").resolve()
    freddy_path_ok = freddy_origin == expected_freddy_origin
    results.append(
        DiagnosticResult(
            "freddy_origin",
            "FREDDY authoritative module path",
            "PASS" if freddy_path_ok else "FAIL",
            True,
            "private package source resolves from the selected FREDDY root" if freddy_path_ok else "package origin mismatch",
            (
                f"ibc resolves to: {freddy_origin or 'not found'}",
                f"Expected: {expected_freddy_origin}",
                "GRIM loads this package under its private _grim_embedded_freddy_ibc namespace.",
            ),
        )
    )

    results.extend(
        (
            _dependency_result(
                key="numpy",
                name="NumPy",
                module_name="numpy",
                distribution="numpy",
                minimum="1.26",
                required=True,
                purpose="required by every GRIM dataset and solver path",
                probe=dep_probe,
            ),
            _dependency_result(
                key="pyside6",
                name="PySide6 Qt widgets",
                module_name="PySide6.QtWidgets",
                distribution="PySide6",
                minimum="6.6",
                required=True,
                purpose="imported during combined-GUI startup",
                probe=dep_probe,
            ),
            _dependency_result(
                key="matplotlib",
                name="Matplotlib Qt backend",
                module_name="matplotlib.backends.backend_qtagg",
                distribution="matplotlib",
                minimum="3.8",
                required=True,
                purpose="imported during combined-GUI startup and plot preview",
                probe=dep_probe,
            ),
            _dependency_result(
                key="scipy",
                name="SciPy solver support",
                module_name="scipy",
                distribution="scipy",
                minimum="1.11",
                required=True,
                purpose=(
                    "required by the bundled GHOST BoR/feature solvers and "
                    "FREDDY inverse-design paths"
                ),
                probe=dep_probe,
            ),
        )
    )

    results.append(_powerpoint_result(system_name=host_system, probe=ppt_probe))
    if ghost_missing:
        results.extend(
            (
                DiagnosticResult(
                    "native_fmm",
                    "GHOST 2-D near-field acceleration",
                    "SKIP",
                    False,
                    "not checked because the selected GHOST backend is incomplete",
                ),
                DiagnosticResult(
                    "native_bor",
                    "GHOST BoR streaming acceleration",
                    "SKIP",
                    False,
                    "not checked because the selected GHOST backend is incomplete",
                ),
            )
        )
    else:
        results.extend(
            _native_results(
                ghost_backend,
                system_name=host_system,
                machine_name=host_machine,
                library_probe=lib_probe,
            )
        )
    return results


def startup_exit_code(results: Iterable[DiagnosticResult]) -> int:
    """Return nonzero only when a required startup check failed."""

    return 1 if any(result.blocks_startup for result in results) else 0


def native_acceleration_status(
    results: Iterable[DiagnosticResult],
) -> tuple[bool, tuple[str, ...]]:
    """Report solver acceleration separately from functional readiness.

    The native libraries are optional because their NumPy/SciPy fallbacks are
    physically equivalent.  Calling a machine simply ``READY`` when both
    accelerators are absent is nevertheless misleading for vehicle-scale
    work, so the human-facing report exposes that performance limitation
    without turning it into a startup blocker.
    """

    expected = {
        "native_fmm": "GHOST 2-D near-field acceleration",
        "native_bor": "GHOST BoR streaming acceleration",
    }
    native = {
        result.key: result
        for result in results
        if result.key in expected
    }
    limited = tuple(
        native[key].name
        if key in native
        else expected[key] + " (diagnostic missing)"
        for key in expected
        if key not in native or native[key].status != "PASS"
    )
    return not limited, limited


def write_report(
    results: Sequence[DiagnosticResult],
    *,
    stream: TextIO = sys.stdout,
) -> None:
    print("GRIM integrated installation diagnostic", file=stream)
    print("No files are changed and no solver or PowerPoint instance is started.", file=stream)
    print(file=stream)
    for result in results:
        scope = "required" if result.required else "optional"
        print(f"[{result.status:<4}] [{scope}] {result.name}: {result.summary}", file=stream)
        for detail in result.details:
            if detail:
                print(f"       {detail}", file=stream)
    blockers = [result for result in results if result.blocks_startup]
    optional_notices = [
        result for result in results if not result.required and result.status in {"WARN", "SKIP"}
    ]
    native_ready, limited_native = native_acceleration_status(results)
    print(file=stream)
    if blockers:
        print(
            f"FUNCTIONAL READINESS: NOT READY - {len(blockers)} required "
            "startup blocker(s).",
            file=stream,
        )
        print(
            f"RESULT: NOT READY - {len(blockers)} required startup blocker(s).",
            file=stream,
        )
        print("Fix the required FAIL items, then run this diagnostic again.", file=stream)
    else:
        print("FUNCTIONAL READINESS: READY", file=stream)
        print("RESULT: READY - no required startup blockers were found.", file=stream)
        if optional_notices:
            print(
                f"Optional notices: {len(optional_notices)}. They do not prevent GRIM from starting.",
                file=stream,
            )
    if native_ready:
        print("SOLVER PERFORMANCE: ACCELERATED - native libraries are available.", file=stream)
    else:
        names = ", ".join(limited_native) or "native solver acceleration"
        print(
            "SOLVER PERFORMANCE: LIMITED - functional fallbacks are available, "
            f"but {names} is not accelerated.",
            file=stream,
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grim-diagnose",
        description=(
            "Check a combined GRIM/GHOST/FREDDY source installation without "
            "starting the GUI, a solver, or PowerPoint."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="combined checkout root (defaults to the tree containing this module)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    results = collect_diagnostics(args.root)
    write_report(results)
    return startup_exit_code(results)


if __name__ == "__main__":  # pragma: no cover - exercised through main tests
    raise SystemExit(main())
