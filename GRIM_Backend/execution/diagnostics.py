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

# Source modules required by the integrated window and its built-in workspaces.
# Most are eagerly imported; a few presentation adapters load on first use.
# Keep this standard-library-only manifest shared by diagnostics and the release
# builder so missing modules are detected without importing Qt, Matplotlib,
# NumPy, or SciPy. The development inventory check covers local imports,
# including deferred readers, without importing the application.
GRIM_STARTUP_FILES = (
    "assembly/__init__.py",
    "assembly/interference.py",
    "assembly/model.py",
    "assembly/panel.py",
    "assembly/placement_editor.py",
    "assembly/recipe.py",
    "assembly/response_comparison.py",
    "assembly/tree.py",
    "assembly/values.py",
    "assembly/workflow.py",
    "assembly/workspace.py",
    "datasets/__init__.py",
    "datasets/api.py",
    "datasets/arithmetic.py",
    "datasets/audit.py",
    "datasets/axes.py",
    "datasets/calibration.py",
    "datasets/combine.py",
    "datasets/constants.py",
    "datasets/coordinates.py",
    "datasets/grid.py",
    "datasets/memory.py",
    "datasets/metadata.py",
    "datasets/transforms.py",
    "execution/__init__.py",
    "execution/dataset_jobs.py",
    "execution/diagnostics.py",
    "integrations/__init__.py",
    "integrations/freddy.py",
    "integrations/ghost.py",
    "io/__init__.py",
    "io/batch.py",
    "io/cst.py",
    "io/csv.py",
    "io/inspect_ss.py",
    "io/loaders.py",
    "io/native.py",
    "io/out.py",
    "io/pioneer.py",
    "io/ptm.py",
    "io/samples.py",
    "io/sentri.py",
    "io/xpatch.py",
    "isar/__init__.py",
    "isar/artifact.py",
    "isar/bpde.py",
    "isar/repeats.py",
    "plotting/__init__.py",
    "plotting/actions.py",
    "plotting/dataset_style.py",
    "plotting/models.py",
    "plotting/modes/__init__.py",
    "plotting/modes/az_vs_range_mode.py",
    "plotting/modes/azimuth_polar_mode.py",
    "plotting/modes/azimuth_rect_mode.py",
    "plotting/modes/common.py",
    "plotting/modes/compare_mode.py",
    "plotting/modes/elevation_sweep_mode.py",
    "plotting/modes/frequency_mode.py",
    "plotting/modes/isar_mode.py",
    "plotting/modes/isar_render.py",
    "plotting/modes/waterfall_mode.py",
    "reports/__init__.py",
    "reports/image_imprinter.py",
    "reports/plot_data.py",
    "reports/report.py",
    "reports/workflow.py",
    "reports/workspace.py",
    "run_diagnostics.py",
    "run_gui.py",
    "run_headless.py",
    "run_image_imprinter.py",
    "runs/__init__.py",
    "runs/remote.py",
    "runs/workspace.py",
    "scripting/__init__.py",
    "scripting/api.py",
    "scripting/cli.py",
    "scripting/plotting.py",
    "scripting/recorder.py",
    "scripting/workspace.py",
    "ui/__init__.py",
    "ui/app.py",
    "ui/dataset_actions.py",
    "ui/dataset_dialogs.py",
    "ui/dataset_sidebar.py",
    "ui/palette.py",
    "ui/theme.py",
    "ui/widgets.py",
)

# Backwards-compatible name retained for callers and tests that used the
# original, smaller sentinel tuple.
GRIM_SENTINELS = GRIM_STARTUP_FILES

# Keep this aligned with the reusable workspace contract in ghost_integration.
GHOST_MODULE_LOCATIONS = {
    'assembly_geometry': 'twod/assembly/geometry_plan.py',
    'assembly_inspector': 'assembly/inspector.py',
    'assembly_session': 'twod/assembly/session.py',
    'assembly_workload': 'assembly/workload.py',
    'bor_dispatch': 'bor/dispatch.py',
    'bor_kernels': 'bor/kernels.py',
    'bor_solver': 'bor/solver.py',
    'bor_streaming': 'bor/streaming.py',
    'boundary_fields': 'twod/fields.py',
    'compact_operators': 'twod/assembly/compact.py',
    'components': 'assembly/components.py',
    'compressed_factor': 'compressed/factor.py',
    'compressed_inverse': 'compressed/inverse.py',
    'compressed_memory': 'compressed/memory.py',
    'compressed_native': 'compressed/coefficients.py',
    'compressed_operator': 'compressed/operator.py',
    'compressed_oracle': 'compressed/regional_coefficients.py',
    'compressed_pair': 'compressed/polarization_cache.py',
    'compressed_runtime': 'compressed/runtime.py',
    'cpu_execution': 'execution/cpu.py',
    'cpu_kernels': 'twod/assembly/kernels.py',
    'dense_factor': 'linalg/dense.py',
    'dense_workspace': 'linalg/workspace.py',
    'dielectric_system': 'twod/formulations/dielectric.py',
    'driver_config': 'runs/config.py',
    'driver_io': 'runs/inputs.py',
    'feature_library_contracts': 'assembly/contracts.py',
    'feature_preparation': 'assembly/preparation.py',
    'feature_sum': 'assembly/fields.py',
    'feature_workflow': 'assembly/workflow.py',
    'frame': 'geometry/frames.py',
    'geometry_io': 'geometry/io.py',
    'geometry_tab': 'ui/geometry.py',
    'ghost_gui': 'ui/app.py',
    'ghost_runtime': 'execution/runtime.py',
    'grim_compat': 'io/viewer_bridge.py',
    'grim_io': 'io/grim.py',
    'grim_naming': 'io/naming.py',
    'hierarchical_factor': 'linalg/hierarchical.py',
    'hpc_bundle': 'hpc/bundle.py',
    'hpc_common': 'hpc/common.py',
    'hpc_scheduler': 'hpc/scheduler.py',
    'line_expand': 'assembly/line_expansion.py',
    'local_mass': 'twod/assembly/mass.py',
    'material_models': 'geometry/materials.py',
    'mesh_guidance': 'geometry/guidance.py',
    'mesh_quality': 'geometry/quality.py',
    'mie_reference': 'validation/cylinder.py',
    'mie_sphere': 'validation/sphere.py',
    'multi_region': 'twod/formulations/regions.py',
    'occluder': 'geometry/occlusion.py',
    'rcs_constants': 'twod/constants.py',
    'rcs_geometry': 'twod/geometry.py',
    'rcs_operators': 'twod/operators.py',
    'rcs_solver': 'twod/solver.py',
    'rcs_special': 'twod/special.py',
    'refined_lu': 'linalg/refined_lu.py',
    'robin_system': 'twod/formulations/robin.py',
    'run_setup': 'runs/setup.py',
    'sheet_system': 'twod/formulations/sheet.py',
    'solver_metrics': 'execution/metrics.py',
    'solver_quality': 'runs/quality.py',
    'solver_tab': 'ui/solver.py',
    'surface_mesh': 'geometry/surface.py',
    'sweep_compression': 'linalg/sweep.py',
    'system_scatter': 'twod/assembly/scatter.py',
    'thin_sheet': 'twod/formulations/thin_layer.py',
    'workflow_provenance': 'execution/provenance.py',
}

GHOST_SENTINELS = (
    "bor/native/__init__.py",
    "bor/native/build_kernel.py",
    "assembly/create_feature_manifest.py",
    "assembly/place_features.py",
    "io/import_3d_reference.py",
    "execution/_dataclasses.py",
    "hpc/check_environment.py",
    "validation/feature_family.py",
    "assembly/__init__.py",
    "assembly/components.py",
    "assembly/contracts.py",
    "assembly/fields.py",
    "assembly/inspector.py",
    "assembly/line_expansion.py",
    "assembly/preparation.py",
    "assembly/workflow.py",
    "assembly/workload.py",
    "bor/__init__.py",
    "bor/dispatch.py",
    "bor/kernels.py",
    "bor/solver.py",
    "bor/streaming.py",
    "compressed/__init__.py",
    "compressed/coefficients.py",
    "compressed/factor.py",
    "compressed/inverse.py",
    "compressed/memory.py",
    "compressed/operator.py",
    "compressed/polarization_cache.py",
    "compressed/regional_coefficients.py",
    "compressed/runtime.py",
    "execution/__init__.py",
    "execution/cpu.py",
    "execution/metrics.py",
    "execution/provenance.py",
    "execution/runtime.py",
    "execution/options.py",
    "runs/execution.py",
    "ui/execution.py",
    "geometry/__init__.py",
    "geometry/frames.py",
    "geometry/guidance.py",
    "geometry/io.py",
    "geometry/materials.py",
    "geometry/occlusion.py",
    "geometry/quality.py",
    "geometry/surface.py",
    "hpc/__init__.py",
    "hpc/bundle.py",
    "hpc/common.py",
    "hpc/scheduler.py",
    "io/__init__.py",
    "io/grim.py",
    "io/naming.py",
    "io/viewer_bridge.py",
    "linalg/__init__.py",
    "linalg/dense.py",
    "linalg/hierarchical.py",
    "linalg/refined_lu.py",
    "linalg/sweep.py",
    "linalg/workspace.py",
    "execution/paths.py",
    "runs/__init__.py",
    "runs/config.py",
    "runs/inputs.py",
    "runs/quality.py",
    "runs/setup.py",
    "twod/__init__.py",
    "twod/assembly/__init__.py",
    "twod/assembly/compact.py",
    "twod/assembly/geometry_plan.py",
    "twod/assembly/kernels.py",
    "twod/assembly/mass.py",
    "twod/assembly/scatter.py",
    "twod/assembly/session.py",
    "twod/constants.py",
    "twod/fields.py",
    "twod/formulations/__init__.py",
    "twod/formulations/dielectric.py",
    "twod/formulations/regions.py",
    "twod/formulations/robin.py",
    "twod/formulations/sheet.py",
    "twod/formulations/thin_layer.py",
    "twod/geometry.py",
    "twod/operators.py",
    "twod/solver.py",
    "twod/special.py",
    "ui/__init__.py",
    "ui/app.py",
    "ui/geometry.py",
    "ui/solver.py",
    "validation/__init__.py",
    "validation/cylinder.py",
    "validation/sphere.py",
    "run_gui.py",
    "run_hpc_bor_monostatic.py",
    "run_hpc_monostatic.py",
    "validation/reconstruction.py",
)

FREDDY_SENTINELS = (
    "ibc/__init__.py",
    "ibc/analysis_data.py",
    "ibc/analysis_workflow.py",
    "ibc/batch.py",
    "ibc/compute.py",
    "ibc/design_search.py",
    "ibc/ghost_coating.py",
    "ibc/inverse_grid.py",
    "ibc/inverse_performance.py",
    "ibc/inverse_results.py",
    "ibc/inverse_workflow.py",
    "ibc/io.py",
    "ibc/material_explorer.py",
    "ibc/mix_analysis.py",
    "ibc/plot.py",
    "ibc/project_state.py",
    "ibc/search_checkpoint.py",
    "ibc/sweep_results.py",
    "ibc/ui.py",
    "ibc/ui_controls.py",
    "ibc/ui_dialogs.py",
    "ibc/ui_options.py",
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
    """Return the complete source-tree root containing ``GRIM_Backend``."""

    return Path(__file__).resolve().parents[2]


def _resolved(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def _missing_files(base: Path, relative_paths: Iterable[str | Path]) -> list[str]:
    return [str(value) for value in relative_paths if not (base / value).is_file()]


def _ghost_backend_from(value: str | os.PathLike[str]) -> Path:
    candidate = _resolved(value)
    if (candidate / "ghost_backend" / "run_gui.py").is_file():
        candidate = (candidate / "ghost_backend").resolve()
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
        (repository_root / "tools" / "GHOST" / "ghost_backend").resolve(),
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
        parts = module_name.split('.')
        if parts[0] == 'GRIM_Backend':
            parts = parts[1:]
        spec = machinery.PathFinder.find_spec(parts[-1], [str(search_path.joinpath(*parts[:-1]))])
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
    paths = ghost_module_paths(expected_dir)
    for name in sorted(set(module_names)):
        module = sys.modules.get(name)
        if module is None:
            continue
        if name == 'ghost_backend':
            roots = {Path(value).resolve() for value in getattr(module, '__path__', ())}
            if roots != {expected_dir.resolve()}:
                conflicts.append(f"{name} ({roots})")
            continue
        value = getattr(module, "__file__", None)
        if not value:
            continue
        try:
            origin = Path(value).resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            conflicts.append(f"{name} (unknown origin)")
            continue
        relative = paths.get(name)
        if relative is None or origin != (expected_dir / relative).resolve():
            conflicts.append(f"{name} ({origin})")
    return conflicts


def _backend_module_names(backend: Path) -> tuple[str, ...]:
    """Return public and package module names owned by this backend."""
    return tuple(sorted(ghost_module_paths(backend)))


def ghost_module_paths(backend: Path) -> dict[str, str]:
    """Map import names to their implementation paths in the selected backend."""
    result = dict(GHOST_MODULE_LOCATIONS)
    result['ghost_backend'] = ''
    result['run_gui'] = 'ui/app.py'
    for path in backend.glob('*.py'):
        if path.stem != '__init__':
            result.setdefault(path.stem, path.name)
    for path in backend.rglob('*.py'):
        relative = path.relative_to(backend).as_posix()
        if relative.split('/')[0] in ('tests', 'data_tools', 'results'):
            continue
        name = relative[:-3].replace('/', '.')
        if name.endswith('.__init__'):
            name = name[:-9]
        name = 'ghost_backend' if name == '__init__' else 'ghost_backend.' + name
        result[name] = result.get(path.stem, relative) if '/' not in relative else relative
    return result


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

    bor_candidates = tuple(
        backend / "bor" / "native" / f"{base}{extension}"
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
            for path in (backend / "bor" / "native").glob("bor_stream_kernel.*")
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
    grim_dir = _resolved(module_directory or Path(__file__).resolve().parents[1])
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
    expected_grim_dir = (root / "GRIM_Backend").resolve()
    grim_origin = _module_spec_origin("GRIM_Backend.ui.app", grim_dir)
    grim_details: list[str] = [f"Repository root: {root}", f"GRIM modules: {grim_dir}"]
    if grim_dir != expected_grim_dir:
        grim_details.append(f"Expected modules below this tree: {expected_grim_dir}")
    if grim_missing:
        grim_details.append("Missing: " + ", ".join(grim_missing))
    if grim_origin is None:
        grim_details.append("GRIM_Backend.ui.app could not be resolved from that directory.")
    grim_ok = not grim_missing and grim_dir == expected_grim_dir and grim_origin == (
        grim_dir / "ui/app.py"
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
    ghost_origin = _module_spec_origin("run_gui", ghost_backend)
    # Runtime GHOST discovery validates every flat Backend module, not only
    # the small sentinel subset. Diagnostics must catch the same stale
    # ``frame``/``components``/workflow imports before the GUI is launched.
    ghost_conflicts = _loaded_module_conflicts(
        _backend_module_names(ghost_backend), ghost_backend
    )
    expected_ghost_origin = (ghost_backend / "run_gui.py").resolve()
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
            _dependency_result(
                key="threadpoolctl",
                name="BLAS thread control",
                module_name="threadpoolctl",
                distribution="threadpoolctl",
                minimum="3.6",
                required=True,
                purpose="applies the saved GHOST CPU thread limits",
                probe=dep_probe,
            ),
        )
    )

    results.append(_powerpoint_result(system_name=host_system, probe=ppt_probe))
    if ghost_missing:
        results.extend(
            (
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

    The native BoR library is optional because its NumPy fallback is
    physically equivalent. Calling a machine simply ``READY`` when this
    accelerator is absent is nevertheless misleading for vehicle-scale
    work, so the human-facing report exposes that performance limitation
    without turning it into a startup blocker.
    """

    expected = {
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
