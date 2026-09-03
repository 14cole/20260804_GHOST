"""
Solver UI threshold guide.

Quality gate inputs:
- `residual <=`: max allowed linear solve residual norm (dimensionless).
- `cond <=`: max allowed condition-number estimate of the system matrix (dimensionless).
- `warns <=`: max allowed warning count emitted by the solver.

Mesh convergence is enabled by default for both solvers and may be disabled
explicitly for survey work. Survey results retain the algebraic quality gate
but are marked as uncertified base-mesh fields in their metadata.
"""

import copy
import hashlib
import json
import math
import os
import re
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from PySide6.QtCore import QObject, QStandardPaths, QThread, Qt, Signal, Slot
    from PySide6.QtWidgets import (
        QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
        QLineEdit, QMessageBox, QPushButton, QSplitter, QTableWidget,
        QTableWidgetItem, QVBoxLayout, QWidget, QComboBox, QCheckBox,
        QProgressBar, QSizePolicy, QToolButton,
    )
except ImportError:
    from PySide2.QtCore import (  # type: ignore
        QObject, QStandardPaths, QThread, Qt, Signal, Slot,
    )
    from PySide2.QtWidgets import (  # type: ignore
        QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
        QLineEdit, QMessageBox, QPushButton, QSplitter, QTableWidget,
        QTableWidgetItem, QVBoxLayout, QWidget, QComboBox, QCheckBox,
        QProgressBar, QSizePolicy, QToolButton,
    )

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from geometry_io import (
    build_geometry_snapshot,
    material_filename_from_row,
    parse_geometry,
)
from grim_io import (
    _ensure_grim_ext,
    _suffix_for_incidence,
    compute_dbke_from_linear,
    export_result_to_grim,
)
from rcs_solver import (
    solve_monostatic_rcs_2d_certified,
    solve_monostatic_rcs_2d_survey,
    solve_bistatic_rcs_2d_certified,
    solve_bistatic_rcs_2d_survey,
    compute_boundary_densities,
)
from bor_dispatch import (
    solve_monostatic_rcs_bor_certified,
    solve_monostatic_rcs_bor_survey,
)
from solver_quality import validate_mesh_convergence_policy


def _result_kind(result: 'Dict[str, Any]') -> 'str':
    """Return the physical result class used by labels and unit conversion."""

    if (
        str(result.get("solver", "")).strip().lower() == "bor_mom_rcs"
        or str(result.get("rcs_linear_quantity", "")).strip().lower()
        == "sigma_3d"
    ):
        return "bor"
    if str(result.get("scattering_mode", "")).strip().lower() == "bistatic":
        return "2d_bistatic"
    return "2d_monostatic"


def _planned_export_paths(
    result: 'Dict[str, Any]',
    output_path: 'str',
) -> 'List[str]':
    """Return every path a solver result will publish before writing it."""

    root = os.path.abspath(_ensure_grim_ext(output_path))
    mode = str(result.get("scattering_mode", "monostatic")).strip().lower()
    if mode != "bistatic":
        return [root]
    incidence_set = set()
    for row in _result_rows(result):
        try:
            value = float(row.get("theta_inc_deg", math.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            incidence_set.add(value)
    incidence_values = sorted(incidence_set)
    if not incidence_values:
        raise ValueError("The bistatic result has no finite incidence angles.")
    root_no_ext = root[:-5]
    return [
        f"{root_no_ext}_{_suffix_for_incidence(value)}.grim"
        for value in incidence_values
    ]


def _finite_unique_count(
    rows: 'List[Dict[str, Any]]',
    key: 'str',
) -> 'int':
    values = set()
    for row in rows:
        try:
            value = float(row.get(key, math.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.add(value)
    return len(values)


def _result_rows(result: 'Dict[str, Any]') -> 'List[Dict[str, Any]]':
    """Return every solved channel with an explicit VV/HH row label."""

    channels = result.get("co_solved_samples")
    if isinstance(channels, dict) and channels:
        rows: 'List[Dict[str, Any]]' = []
        for polarization in ("VV", "HH"):
            for source in channels.get(polarization, []) or []:
                row = dict(source)
                row["polarization"] = polarization
                rows.append(row)
        if rows:
            return rows
    polarization = str(
        result.get("polarization_export")
        or result.get("polarization")
        or ""
    )
    rows = []
    for source in result.get("samples", []) or []:
        row = dict(source)
        if polarization:
            row.setdefault("polarization", polarization)
        rows.append(row)
    return rows


def _result_sample_counts(result: 'Dict[str, Any]') -> 'Dict[str, int]':
    """Count the axes actually represented by the returned samples."""

    rows = _result_rows(result)
    metadata = result.get("metadata", {}) or {}

    def count_or_metadata(sample_key: 'str', metadata_key: 'str') -> 'int':
        count = _finite_unique_count(rows, sample_key)
        if count:
            return count
        try:
            fallback = int(metadata.get(metadata_key, 0))
        except (TypeError, ValueError):
            return 0
        return max(fallback, 0)

    return {
        "frequency_count": count_or_metadata(
            "frequency_ghz", "frequency_count"
        ),
        "incidence_count": count_or_metadata(
            "theta_inc_deg", "incidence_count"
        ),
        "observation_count": count_or_metadata(
            "theta_scat_deg", "observation_count"
        ),
        "aspect_count": count_or_metadata(
            "theta_inc_deg", "output_aspect_count"
        ),
    }


def _result_summary(result: 'Dict[str, Any]') -> 'str':
    kind = _result_kind(result)
    counts = _result_sample_counts(result)
    if kind == "bor":
        return (
            "Solved BoR monostatic RCS with "
            f"{counts['frequency_count']} frequency(ies) and "
            f"{counts['aspect_count']} aspect angle(s)."
        )
    if kind == "2d_bistatic":
        return (
            "Solved 2-D bistatic scattering width with "
            f"{counts['frequency_count']} frequency(ies), "
            f"{counts['incidence_count']} incidence angle(s), and "
            f"{counts['observation_count']} observation angle(s)."
        )
    return (
        "Solved 2-D monostatic scattering width with "
        f"{counts['frequency_count']} frequency(ies) and "
        f"{counts['observation_count']} cut angle(s)."
    )


def _quality_gate_suffix(metadata: 'Dict[str, Any]') -> 'str':
    """Render PASS only for an explicit, boolean pass attestation."""

    qg = metadata.get("quality_gate")
    if not isinstance(qg, dict) or qg.get("passed") is not True:
        if not isinstance(qg, dict) or "passed" not in qg:
            return " Quality gate: NOT REPORTED."
        violations = qg.get("violations", []) or []
        joined = "; ".join(str(v) for v in violations[:2])
        if len(violations) > 2:
            joined += f" (+{len(violations) - 2} more)"
        reason = joined or str(
            qg.get("reason", "") or "criteria not met"
        )
        return f" Quality gate: FAIL ({reason})."
    return " Quality gate: PASS."


def _result_history(
    result: 'Dict[str, Any]',
    *,
    units: 'str',
    manual_export: 'bool' = False,
) -> 'str':
    kind = _result_kind(result)
    counts = _result_sample_counts(result)
    fields = [
        f"solver={result.get('solver', 'unknown')}",
        f"result_kind={kind}",
        f"scattering_mode={result.get('scattering_mode', 'monostatic')}",
        f"freq_count={counts['frequency_count']}",
    ]
    if kind == "bor":
        fields.append(f"aspect_count={counts['aspect_count']}")
    elif kind == "2d_bistatic":
        fields.extend(
            [
                f"inc_count={counts['incidence_count']}",
                f"obs_count={counts['observation_count']}",
            ]
        )
    else:
        fields.append(f"cut_count={counts['observation_count']}")
    fields.extend([f"units={units}", "pols=VV,HH"])
    if manual_export:
        fields.extend(
            [
                "manual_export=1",
                "plot_source=linear_physical_quantity",
            ]
        )
    return " ".join(fields)


def _display_db_value(
    result: 'Dict[str, Any]',
    row: 'Dict[str, Any]',
) -> 'float':
    """Convert the stored physical linear quantity to its matching log unit."""

    lin = float(row.get("rcs_linear", 0.0))
    if _result_kind(result) == "bor":
        if not math.isfinite(lin) or lin <= 0.0:
            lin = 1.0e-12
        return 10.0 * math.log10(lin)
    return compute_dbke_from_linear(
        lin,
        float(row.get("frequency_ghz", math.nan)),
        frequency_unit="GHz",
    )


def _result_plot_groups(
    result: 'Dict[str, Any]',
) -> 'Dict[Tuple[float, Optional[float], str], List[Dict[str, Any]]]':
    """Keep separate bistatic incidence sweeps from becoming one curve."""

    bistatic = _result_kind(result) == "2d_bistatic"
    groups: 'Dict[\n    Tuple[float, Optional[float], str],\n    List[Dict[str, Any]],\n]' = {}
    for row in _result_rows(result):
        freq = float(row.get("frequency_ghz", 0.0))
        incidence = (
            float(row.get("theta_inc_deg", 0.0)) if bistatic else None
        )
        polarization = str(row.get("polarization", ""))
        groups.setdefault((freq, incidence, polarization), []).append(row)
    return groups


def _stable_sha256(path: 'str') -> 'str':
    """Hash one regular file and reject an ordinary concurrent rewrite."""

    source = Path(path)
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = source.stat()
    before_key = (
        int(before.st_size),
        int(before.st_mtime_ns),
        int(before.st_ctime_ns),
    )
    after_key = (
        int(after.st_size),
        int(after.st_mtime_ns),
        int(after.st_ctime_ns),
    )
    if before_key != after_key:
        raise RuntimeError(
            f"{source} changed while it was being read. Wait for the write to "
            "finish, then retry."
        )
    return digest.hexdigest()


def _snapshot_sha256(snapshot: 'Dict[str, Any]') -> 'str':
    """Return a deterministic identity for the immutable geometry payload."""

    payload = json.dumps(
        snapshot,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _material_input_sha256(
    snapshot: 'Dict[str, Any]', base_dir: 'str'
) -> 'Dict[str, str]':
    """Seal every material sidecar read by a background density solve."""

    identities: 'Dict[str, str]' = {}
    for table_name in ("ibcs", "dielectrics"):
        for row in snapshot.get(table_name, []) or []:
            filename = material_filename_from_row(list(row))
            if not filename:
                continue
            path = os.path.abspath(os.path.join(base_dir, filename))
            identities[path] = _stable_sha256(path)
    return identities


def _verify_input_sha256(identities: 'Dict[str, str]') -> 'None':
    """Fail closed when a geometry/material input changed during a job."""

    for path, expected in identities.items():
        try:
            observed = _stable_sha256(path)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"Input changed during the boundary-density calculation: {path}. "
                "No output was published."
            ) from exc
        if observed != expected:
            raise RuntimeError(
                f"Input changed during the boundary-density calculation: {path}. "
                "No output was published."
            )


def _stage_json_output(
    destination: 'str', payload: 'Dict[str, Any]'
) -> 'str':
    """Write and sync JSON beside its destination without publishing it."""

    output = os.path.abspath(destination)
    parent = os.path.dirname(output) or os.getcwd()
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(output)}.", suffix=".tmp", dir=parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return temporary


def _publish_staged_json(
    temporary: 'str',
    destination: 'str',
    *,
    expect_absent: 'bool',
    expected_sha256: 'Optional[str]',
) -> 'None':
    """Revalidate the reviewed destination, then atomically replace it."""

    output = os.path.abspath(destination)
    if expect_absent:
        if os.path.exists(output):
            raise RuntimeError(
                f"{output}: output was created while boundary densities were "
                "being computed; the newer file was not overwritten."
            )
    else:
        if not os.path.isfile(output):
            raise RuntimeError(
                f"{output}: reviewed output was removed or replaced while "
                "boundary densities were being computed."
            )
        if expected_sha256 is None or _stable_sha256(output) != expected_sha256:
            raise RuntimeError(
                f"{output}: output changed while boundary densities were being "
                "computed; the newer file was not overwritten."
            )
    os.replace(temporary, output)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class _SolveWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object, str)
    canceled = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        snapshot: 'Dict[str, Any]',
        source_path: 'str',
        base_dir: 'str',
        frequencies: 'List[float]',
        elevations: 'List[float]',
        units: 'str',
        quality_thresholds: 'Dict[str, float | int]',
        mesh_certification: 'bool' = True,
        cfie_alpha: 'float' = 0.0,
        abort_event: 'Optional[Any]' = None,
        scattering_mode: 'str' = "monostatic",
        observation_angles: 'Optional[List[float]]' = None,
        solver_kind: 'str' = "2d",
    ):
        super().__init__()
        self.solver_kind = str(solver_kind)
        self.snapshot = snapshot
        self.source_path = source_path
        self.base_dir = base_dir
        self.frequencies = frequencies
        self.elevations = elevations
        self.units = units
        self.quality_thresholds = dict(quality_thresholds)
        self.mesh_certification = bool(mesh_certification)
        self.cfie_alpha = float(cfie_alpha)
        self.abort_event = abort_event
        self.scattering_mode = str(scattering_mode)
        self.observation_angles = observation_angles

    def _on_progress(self, done_steps: 'int', total_steps: 'int', message: 'str') -> 'None':
        if total_steps <= 0:
            pct = 0
        else:
            pct = int(round(100.0 * float(done_steps) / float(total_steps)))
        pct = max(0, min(100, pct))
        self.progress.emit(pct, message)

    def _run_bor(self):
        """BoR (axisymmetric) route: elevations are ASPECT angles from the +z
        rotation axis. The displayed BoR CFIE value is passed unchanged."""

        if self.scattering_mode == "bistatic":
            raise ValueError("Bistatic sweeps are not supported by the BoR "
                             "solver yet; use Monostatic.")
        kwargs = dict(
            geometry_snapshot=self.snapshot,
            frequencies_ghz=self.frequencies,
            elevations_deg=self.elevations,
            geometry_units=self.units,
            material_base_dir=self.base_dir,
            cfie_alpha=self.cfie_alpha,
            abort_event=self.abort_event,
        )
        if self.mesh_certification:
            return solve_monostatic_rcs_bor_certified(
                progress_callback=self._on_progress,
                mesh_convergence_policy=validate_mesh_convergence_policy(),
                **kwargs
            )
        return solve_monostatic_rcs_bor_survey(
            progress_callback=self._on_progress,
            **kwargs
        )

    def _run_2d(self, snapshot, progress_callback):
        """Run the selected 2-D mode with optional mesh certification."""

        mesh_policy = validate_mesh_convergence_policy()
        if self.scattering_mode == "bistatic":
            if not self.observation_angles:
                raise ValueError(
                    "Bistatic mode requires at least one observation angle."
                )
            solve_bistatic = (
                solve_bistatic_rcs_2d_certified
                if self.mesh_certification
                else solve_bistatic_rcs_2d_survey
            )
            bistatic_kwargs = dict(
                geometry_snapshot=snapshot,
                frequencies_ghz=self.frequencies,
                incidence_angles_deg=self.elevations,
                observation_angles_deg=self.observation_angles,
                geometry_units=self.units,
                material_base_dir=self.base_dir,
                progress_callback=progress_callback,
                quality_thresholds=self.quality_thresholds,
                abort_event=self.abort_event,
            )
            if self.mesh_certification:
                bistatic_kwargs["mesh_convergence_policy"] = mesh_policy
            return solve_bistatic(**bistatic_kwargs)
        if self.scattering_mode != "monostatic":
            raise ValueError(
                f"Unsupported 2-D scattering mode {self.scattering_mode!r}."
            )
        solve_monostatic = (
            solve_monostatic_rcs_2d_certified
            if self.mesh_certification
            else solve_monostatic_rcs_2d_survey
        )
        monostatic_kwargs = dict(
            geometry_snapshot=snapshot,
            frequencies_ghz=self.frequencies,
            elevations_deg=self.elevations,
            geometry_units=self.units,
            material_base_dir=self.base_dir,
            progress_callback=progress_callback,
            quality_thresholds=self.quality_thresholds,
            abort_event=self.abort_event,
        )
        if self.mesh_certification:
            monostatic_kwargs["mesh_convergence_policy"] = mesh_policy
        return solve_monostatic(**monostatic_kwargs)

    @Slot()
    def run(self):
        try:
            if self.solver_kind == "bor":
                result = self._run_bor()
            else:
                result = self._run_2d(self.snapshot, self._on_progress)
            if self.abort_event is not None and self.abort_event.is_set():
                raise InterruptedError("Solve canceled by user.")
        except InterruptedError as exc:
            self.canceled.emit(str(exc) or "Solve canceled by user.")
            return
        except Exception as exc:
            if self.abort_event is not None and self.abort_event.is_set():
                self.canceled.emit("Solve canceled by user.")
                return
            self.error.emit(str(exc))
            return
        metadata = dict(result.get("metadata", {}) or {})
        metadata.setdefault("geometry_units_in", self.units)
        result["metadata"] = metadata
        self.finished.emit(result, self.source_path)


class _BoundaryDensityWorker(QObject):
    """Compute and stage one immutable VV/HH density diagnostic off-thread."""

    progress = Signal(int, str)
    finished = Signal(int, object, str)
    canceled = Signal(int, str)
    error = Signal(int, str)

    def __init__(
        self,
        *,
        run_id: 'int',
        snapshot: 'Dict[str, Any]',
        source_path: 'str',
        base_dir: 'str',
        frequency_ghz: 'float',
        elevation_deg: 'float',
        units: 'str',
        output_path: 'str',
        snapshot_sha256: 'str',
        input_sha256: 'Dict[str, str]',
        abort_event: 'threading.Event',
    ) -> 'None':
        super().__init__()
        self.run_id = int(run_id)
        self.snapshot = snapshot
        self.source_path = str(source_path)
        self.base_dir = str(base_dir)
        self.frequency_ghz = float(frequency_ghz)
        self.elevation_deg = float(elevation_deg)
        self.units = str(units)
        self.output_path = str(output_path)
        self.snapshot_sha256 = str(snapshot_sha256)
        self.input_sha256 = dict(input_sha256)
        self.abort_event = abort_event

    def _check_canceled(self) -> 'None':
        if self.abort_event.is_set():
            raise InterruptedError(
                "Boundary-density calculation canceled by user."
            )

    @Slot()
    def run(self) -> 'None':
        temporary = ""
        try:
            channels = {}
            for index, (label, polarization) in enumerate(
                (("VV", "TE"), ("HH", "TM"))
            ):
                self._check_canceled()
                self.progress.emit(
                    index * 45,
                    f"Computing {label} boundary-integral density...",
                )
                channels[label] = compute_boundary_densities(
                    geometry_snapshot=self.snapshot,
                    frequency_ghz=self.frequency_ghz,
                    elevation_deg=self.elevation_deg,
                    polarization=polarization,
                    geometry_units=self.units,
                    material_base_dir=self.base_dir,
                    cfie_alpha=0.0,
                    abort_event=self.abort_event,
                )
            self._check_canceled()
            result = {
                "polarizations": ["VV", "HH"],
                "frequency_ghz": self.frequency_ghz,
                "cut_angle_deg": self.elevation_deg,
                "geometry_units_in": self.units,
                "density_coordinate_units": "meters",
                "geometry_source_path": self.source_path,
                "geometry_snapshot_sha256": self.snapshot_sha256,
                "input_files_sha256": dict(self.input_sha256),
                "channels": channels,
            }
            self.progress.emit(92, "Staging boundary-density JSON...")
            temporary = _stage_json_output(self.output_path, result)
            self._check_canceled()
            summary = {
                "element_count": int(channels["VV"].get("element_count", 0)),
                "formulations": ", ".join(
                    f"{label}={channels[label].get('formulation', '?')}"
                    for label in ("VV", "HH")
                ),
            }
        except InterruptedError as exc:
            if temporary:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
            self.canceled.emit(
                self.run_id,
                str(exc) or "Boundary-density calculation canceled by user.",
            )
            return
        except Exception as exc:
            if temporary:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
            if self.abort_event.is_set():
                self.canceled.emit(
                    self.run_id,
                    "Boundary-density calculation canceled by user.",
                )
                return
            self.error.emit(self.run_id, str(exc))
            return
        self.finished.emit(self.run_id, summary, temporary)


class SolverTab(QWidget):
    # Unified hosts use this to load newly published solver files directly
    # into their dataset/project tree.  The second value is the stable result
    # family ("2d" or "bor"), not a display label.
    files_exported = Signal(list, str)

    def __init__(self, geometry_tab=None, parent=None):
        super().__init__(parent)
        self.geometry_tab = geometry_tab
        self.last_result: 'Optional[Dict[str, Any]]' = None
        self.last_source_path: 'str' = ""
        self.last_solve_context: 'Optional[Dict[str, Any]]' = None
        self._pending_solve_context: 'Optional[Dict[str, Any]]' = None
        self._solve_thread: 'Optional[QThread]' = None
        self._solve_worker: 'Optional[_SolveWorker]' = None
        self._solve_run_serial: 'int' = 0
        self._active_solve_run_id: 'Optional[int]' = None
        self._is_solving: 'bool' = False
        self._abort_event: 'Optional[threading.Event]' = None
        self._pending_density_context: 'Optional[Dict[str, Any]]' = None
        self._density_thread: 'Optional[QThread]' = None
        self._density_worker: 'Optional[_BoundaryDensityWorker]' = None
        self._density_abort_event: 'Optional[threading.Event]' = None
        self._density_run_serial: 'int' = 0
        self._active_density_run_id: 'Optional[int]' = None
        self._is_computing_density: 'bool' = False
        self._plot_theme: 'Optional[Dict[str, str]]' = None
        self._last_result_stale: 'bool' = False

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([400, 700])

        root = QHBoxLayout(self)
        root.addWidget(splitter)

        self._update_mode_enables()
        self._sync_export_state()
        dirty_signal = getattr(self.geometry_tab, "dirty_changed", None)
        if dirty_signal is not None:
            dirty_signal.connect(self._on_geometry_dirty_changed)
        geometry_signal = getattr(self.geometry_tab, "geometry_changed", None)
        if geometry_signal is not None:
            geometry_signal.connect(self._on_geometry_changed)

    @Slot()
    def _on_geometry_changed(self) -> 'None':
        self._mark_geometry_dependent_results_stale()

    @Slot(bool)
    def _on_geometry_dirty_changed(self, dirty: 'bool') -> 'None':
        if not dirty:
            return
        # Retain this connection for embedders that expose only the historical
        # dirty-state signal.  GeometryTab also emits ``geometry_changed`` for
        # every edit, including edits made while it is already dirty.
        self._mark_geometry_dependent_results_stale()

    def _mark_geometry_dependent_results_stale(self) -> 'None':
        pending = self._pending_solve_context
        if pending is not None and bool(pending.get("uses_geometry_tab", False)):
            pending["geometry_stale"] = True
        density = self._pending_density_context
        if density is not None and bool(density.get("uses_geometry_tab", False)):
            density["geometry_stale"] = True
        if (
            self.last_result is not None
            and self.last_solve_context is not None
            and bool(self.last_solve_context.get("uses_geometry_tab", False))
        ):
            self._last_result_stale = True
            self._sync_export_state()
            self.lbl_status.setText(
                "Geometry changed after the last solve. Run the solver again before export."
            )

    def _sync_export_state(self) -> 'None':
        stale = bool(self._last_result_stale)
        self.btn_export.setText(
            "Export Last Result (Stale)" if stale else "Export Last Result"
        )
        self.btn_export.setEnabled(
            not self._job_is_active()
            and self.last_result is not None
            and not stale
        )

    @staticmethod
    def _thread_is_running(thread: 'Optional[QThread]') -> 'bool':
        if thread is None:
            return False
        try:
            probe = getattr(thread, "isRunning", None)
            return bool(callable(probe) and probe())
        except RuntimeError:
            # A deferred-delete Qt wrapper can briefly outlive its C++ object.
            return False

    def _job_is_active(self) -> 'bool':
        """Include terminal QThread teardown before controls may re-enable."""

        return bool(
            self._is_solving
            or self._is_computing_density
            or self._thread_is_running(self._solve_thread)
            or self._thread_is_running(self._density_thread)
        )

    def job_is_running(self) -> 'bool':
        """Include worker shutdown windows so a host cannot destroy QThreads."""

        return self._job_is_active()

    def _result_is_current_for_export(
        self,
        result: 'Dict[str, Any]',
        solve_context: 'Optional[Dict[str, Any]]',
    ) -> 'bool':
        """Return whether a confirmed export still targets the active result."""

        return bool(
            not self._last_result_stale
            and self.last_result is result
            and self.last_solve_context is solve_context
        )

    def apply_plot_theme(
        self, *, background: 'str', text: 'str', grid: 'str'
    ) -> 'None':
        """Apply embedding-shell colors while preserving standalone defaults."""

        self._plot_theme = {
            "background": str(background),
            "text": str(text),
            "grid": str(grid),
        }
        self._apply_plot_theme_to_axes()
        self.canvas.draw_idle()

    def _apply_plot_theme_to_axes(self) -> 'None':
        if self._plot_theme is None:
            return
        background = self._plot_theme["background"]
        text = self._plot_theme["text"]
        grid = self._plot_theme["grid"]
        self.canvas.fig.patch.set_facecolor(background)
        ax = self.canvas.ax
        ax.set_facecolor(background)
        ax.title.set_color(text)
        ax.xaxis.label.set_color(text)
        ax.yaxis.label.set_color(text)
        ax.tick_params(axis="both", colors=text)
        for spine in ax.spines.values():
            spine.set_color(grid)
        ax.grid(True, color=grid, alpha=0.45)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(background)
            legend.get_frame().set_edgecolor(grid)
            for legend_text in legend.get_texts():
                legend_text.set_color(text)

    def _build_left_panel(self) -> 'QWidget':
        panel = QWidget()
        layout = QVBoxLayout(panel)

        geometry_group = QGroupBox("Geometry Source")
        geometry_grid = QGridLayout(geometry_group)
        self.edit_geo_path = QLineEdit()
        self.edit_geo_path.setPlaceholderText("Optional: use .geo file directly, or use current Geometry tab")
        self.btn_browse_geo = QPushButton("Browse...")
        self.btn_use_tab = QPushButton("Use Geometry Tab")
        self.lbl_geo = QLabel("No explicit file selected. Solver will use Geometry tab if available.")
        self.lbl_geo.setWordWrap(True)
        geometry_grid.addWidget(QLabel("Geometry File"), 0, 0)
        geometry_grid.addWidget(self.edit_geo_path, 0, 1, 1, 2)
        geometry_grid.addWidget(self.btn_browse_geo, 1, 1)
        geometry_grid.addWidget(self.btn_use_tab, 1, 2)
        geometry_grid.addWidget(self.lbl_geo, 2, 0, 1, 3)
        layout.addWidget(geometry_group)

        options_group = QGroupBox("Solve Options")
        options_form = QFormLayout(options_group)

        self.cmb_solver_kind = QComboBox()
        self.cmb_solver_kind.addItem("2D (cylindrical, dBke)", userData="2d")
        self.cmb_solver_kind.addItem("BoR (axisymmetric, dBsm)", userData="bor")
        self.cmb_solver_kind.setToolTip(
            "2D: infinite-cylinder elevation cut (2D width).\n"
            "BoR: true 3-D RCS of a body of revolution. The drawing is the\n"
            "(rho, z) half-profile: x = radius >= 0, y = rotation axis, drawn\n"
            "from the +z axis end to the -z axis end. Elevation angles become\n"
            "aspect angles from +z (0 = nose-on, 90 = broadside)."
        )
        self.cmb_solver_kind.currentIndexChanged.connect(self._on_solver_kind_changed)

        self.cmb_units = QComboBox()
        self.cmb_units.addItems(["inches", "meters"])
        self.cmb_units.setCurrentText("inches")

        self.lbl_solve_method = QLabel("Linear / Galerkin")
        self.chk_mesh_certification = QCheckBox(
            "Compare base/fine meshes (recommended)"
        )
        self.chk_mesh_certification.setToolTip(
            "When disabled, run one base mesh and mark the result as survey "
            "data with mesh_convergence_certified=false."
        )
        self.chk_mesh_certification.setChecked(True)
        self.edit_quality_residual_max = QLineEdit("1e-6")
        self.edit_quality_condition_max = QLineEdit("1e6")
        self.edit_quality_warnings_max = QLineEdit("10")
        quality_threshold_row = QWidget()
        quality_threshold_layout = QHBoxLayout(quality_threshold_row)
        quality_threshold_layout.setContentsMargins(0, 0, 0, 0)
        quality_threshold_layout.addWidget(QLabel("residual<="))
        quality_threshold_layout.addWidget(self.edit_quality_residual_max)
        quality_threshold_layout.addWidget(QLabel("cond<="))
        quality_threshold_layout.addWidget(self.edit_quality_condition_max)
        quality_threshold_layout.addWidget(QLabel("warns<="))
        quality_threshold_layout.addWidget(self.edit_quality_warnings_max)

        self.btn_advanced_settings = QToolButton()
        self.btn_advanced_settings.setText("Advanced Settings")
        self.btn_advanced_settings.setCheckable(True)
        self.btn_advanced_settings.setChecked(False)
        self.btn_advanced_settings.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_advanced_settings.setArrowType(Qt.RightArrow)

        self.advanced_settings_widget = QWidget()
        advanced_form = QFormLayout(self.advanced_settings_widget)
        advanced_form.setContentsMargins(0, 0, 0, 0)

        self.edit_cfie_alpha = QLineEdit("0.0")
        self.edit_cfie_alpha.setEnabled(False)
        self.edit_cfie_alpha.setToolTip(
            "BoR closed-PEC CFIE coupling, strictly between 0 and 1 "
            "(0.5 recommended). Use the explicit solver API when pure EFIE "
            "is intended. The 2-D solver has no CFIE control."
        )

        advanced_form.addRow("CFIE alpha", self.edit_cfie_alpha)
        advanced_form.addRow("Mesh Certification", self.chk_mesh_certification)
        advanced_form.addRow("Quality Thresholds", quality_threshold_row)
        self.advanced_settings_widget.setVisible(False)

        self.cmb_freq_mode = QComboBox()
        self.cmb_freq_mode.addItems(["Discrete List", "Start / Stop / Step"])
        self.edit_freq_list = QLineEdit("1.0, 3.0, 10.0")
        self.edit_freq_start = QLineEdit("1.0")
        self.edit_freq_stop = QLineEdit("10.0")
        self.edit_freq_step = QLineEdit("1.0")

        self.cmb_elev_mode = QComboBox()
        self.cmb_elev_mode.addItems(["Discrete List", "Start / Stop / Step"])
        self.edit_elev_list = QLineEdit("0, 30, 60, 90, 120, 150, 180")
        self.edit_elev_start = QLineEdit("0")
        self.edit_elev_stop = QLineEdit("180")
        self.edit_elev_step = QLineEdit("2")

        self.cmb_scatter_mode = QComboBox()
        self.cmb_scatter_mode.addItem("Monostatic", userData="monostatic")
        self.cmb_scatter_mode.addItem("Bistatic", userData="bistatic")
        self.cmb_scatter_mode.setToolTip(
            "Monostatic: backscatter (obs = inc). "
            "Bistatic: specify separate observation angles."
        )

        self.edit_obs_angles = QLineEdit("0, 30, 60, 90, 120, 150, 180")
        self.edit_obs_angles.setToolTip(
            "Observation angles (degrees) for bistatic mode. "
            "Backscatter convention: obs=inc is backscatter, obs=inc+180 is forward scatter."
        )
        self.lbl_obs_angles = QLabel("Observation Angles (deg)")
        self.edit_obs_angles.setVisible(False)
        self.lbl_obs_angles.setVisible(False)
        self.cmb_scatter_mode.currentIndexChanged.connect(self._on_scatter_mode_changed)

        freq_sweep_row = QWidget()
        freq_sweep_layout = QHBoxLayout(freq_sweep_row)
        freq_sweep_layout.setContentsMargins(0, 0, 0, 0)
        freq_sweep_layout.addWidget(QLabel("Start"))
        freq_sweep_layout.addWidget(self.edit_freq_start)
        freq_sweep_layout.addWidget(QLabel("Stop"))
        freq_sweep_layout.addWidget(self.edit_freq_stop)
        freq_sweep_layout.addWidget(QLabel("Step"))
        freq_sweep_layout.addWidget(self.edit_freq_step)

        elev_sweep_row = QWidget()
        elev_sweep_layout = QHBoxLayout(elev_sweep_row)
        elev_sweep_layout.setContentsMargins(0, 0, 0, 0)
        elev_sweep_layout.addWidget(QLabel("Start"))
        elev_sweep_layout.addWidget(self.edit_elev_start)
        elev_sweep_layout.addWidget(QLabel("Stop"))
        elev_sweep_layout.addWidget(self.edit_elev_stop)
        elev_sweep_layout.addWidget(QLabel("Step"))
        elev_sweep_layout.addWidget(self.edit_elev_step)

        options_form.addRow("Solver", self.cmb_solver_kind)
        options_form.addRow("Units In Geometry", self.cmb_units)
        options_form.addRow("Output Channels", QLabel("VV and HH (co-solved)"))
        options_form.addRow("Discretization", self.lbl_solve_method)
        options_form.addRow(self.btn_advanced_settings)
        options_form.addRow(self.advanced_settings_widget)
        options_form.addRow("Frequency Mode", self.cmb_freq_mode)
        options_form.addRow("Frequencies (GHz)", self.edit_freq_list)
        options_form.addRow("Frequency Sweep", freq_sweep_row)
        options_form.addRow("Elevation Mode", self.cmb_elev_mode)
        options_form.addRow("Elevations (deg)", self.edit_elev_list)
        options_form.addRow("Elevation Sweep", elev_sweep_row)
        options_form.addRow("Scattering Mode", self.cmb_scatter_mode)
        options_form.addRow(self.lbl_obs_angles, self.edit_obs_angles)
        layout.addWidget(options_group)

        output_group = QGroupBox("Output")
        output_grid = QGridLayout(output_group)
        self.edit_output = QLineEdit()
        self.edit_output.setPlaceholderText(
            "Automatic unique .grim beside the geometry (or in Documents/GRIM Outputs)"
        )
        self.btn_browse_output = QPushButton("Browse...")
        self.chk_export_after_solve = QCheckBox("Export .grim automatically after solve")
        self.chk_export_after_solve.setChecked(True)
        output_grid.addWidget(QLabel("GRIM Output"), 0, 0)
        output_grid.addWidget(self.edit_output, 0, 1)
        output_grid.addWidget(self.btn_browse_output, 0, 2)
        output_grid.addWidget(self.chk_export_after_solve, 1, 0, 1, 3)
        layout.addWidget(output_group)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Solver")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_export = QPushButton("Export Last Result")
        self.btn_currents = QPushButton("Boundary Densities")
        self.btn_currents.setToolTip(
            "Compute formulation-specific boundary-integral unknowns for the first "
            "frequency and elevation. These layer densities are not generally "
            "physical electric or magnetic surface currents."
        )
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_export)
        btn_row.addWidget(self.btn_currents)
        layout.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)
        layout.addStretch(1)

        self.btn_browse_geo.clicked.connect(self._browse_geo)
        self.btn_use_tab.clicked.connect(self._use_geometry_tab)
        self.btn_browse_output.clicked.connect(self._browse_output)
        self.btn_run.clicked.connect(self._run_solver)
        self.btn_cancel.clicked.connect(self._cancel_solver)
        self.btn_export.clicked.connect(self._export_last_result)
        self.btn_currents.clicked.connect(self._compute_currents)
        self.btn_advanced_settings.toggled.connect(self._toggle_advanced_settings)
        self.cmb_freq_mode.currentIndexChanged.connect(self._update_mode_enables)
        self.cmb_elev_mode.currentIndexChanged.connect(self._update_mode_enables)

        return panel

    def _build_right_panel(self) -> 'QWidget':
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.canvas = MplCanvas(panel)
        self.toolbar = NavigationToolbar(self.canvas, panel)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=2)

        self.table_results = QTableWidget()
        self.table_results.setColumnCount(4)
        self.table_results.setHorizontalHeaderLabels(
            ["Frequency (GHz)", "Elevation (deg)", "RCS (linear)", "RCS (dB)"]
        )
        layout.addWidget(self.table_results, stretch=1)
        return panel

    def _browse_geo(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select Geometry File", "", "Geometry Files (*.geo);;All Files (*)"
        )
        if not fname:
            return
        self.edit_geo_path.setText(fname)
        self.lbl_geo.setText(f"Using geometry file: {fname}")

    def _use_geometry_tab(self):
        self.edit_geo_path.clear()
        if self.geometry_tab is None:
            self.lbl_geo.setText("Geometry tab is not connected in this session.")
            return
        snapshot = self.geometry_tab.get_geometry_snapshot()
        segment_count = int(snapshot.get("segment_count", 0))
        title = snapshot.get("title", "Geometry")
        if segment_count <= 0:
            self.lbl_geo.setText("Geometry tab has no loaded segments yet.")
            return
        self.lbl_geo.setText(f"Using Geometry tab: {title} ({segment_count} segment(s)).")

    def _browse_output(self):
        suggested = self.edit_output.text().strip()
        if not suggested:
            suggested = self._resolve_output_path("", self.last_source_path)
        fname, _ = QFileDialog.getSaveFileName(
            self, "Select Output .grim", suggested, "GRIM Files (*.grim);;All Files (*)"
        )
        if not fname:
            return
        self.edit_output.setText(fname)

    @staticmethod
    def _documents_output_dir() -> 'Path':
        locations = getattr(QStandardPaths, "StandardLocation", QStandardPaths)
        documents = str(
            QStandardPaths.writableLocation(locations.DocumentsLocation) or ""
        ).strip()
        base = Path(documents).expanduser() if documents else Path.home() / "Documents"
        return base / "GRIM Outputs"

    def _resolve_output_path(self, requested: 'str', source_path: 'str') -> 'str':
        """Resolve an explicit path or create a collision-free desktop default."""

        source = Path(str(source_path or "")).expanduser()
        default_dir = (
            source.resolve().parent
            if str(source_path or "").strip()
            else self._documents_output_dir()
        )
        text = str(requested or "").strip()
        if text:
            candidate = Path(text).expanduser()
            if not candidate.is_absolute():
                candidate = default_dir / candidate
            return os.path.abspath(_ensure_grim_ext(str(candidate)))

        default_dir.mkdir(parents=True, exist_ok=True)
        stem = source.stem if str(source_path or "").strip() else "rcs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = default_dir / f"{stem}_rcs_{timestamp}.grim"
        suffix = 2
        while candidate.exists():
            candidate = default_dir / f"{stem}_rcs_{timestamp}_{suffix}.grim"
            suffix += 1
        return str(candidate.resolve())

    def _confirm_export_replacements(self, paths: 'List[str]') -> 'bool':
        existing = [path for path in paths if os.path.exists(path)]
        if not existing:
            return True
        shown = "\n".join(f"  {path}" for path in existing[:8])
        if len(existing) > 8:
            shown += f"\n  ... and {len(existing) - 8} more"
        buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
        answer = QMessageBox.question(
            self,
            "Replace Existing GRIM Output?",
            "The following output already exists:\n\n"
            + shown
            + "\n\nReplace it? The previous file cannot be recovered from GRIM.",
            buttons.Yes | buttons.No,
            buttons.No,
        )
        return answer == buttons.Yes

    def _toggle_advanced_settings(self, checked: 'bool'):
        self.advanced_settings_widget.setVisible(bool(checked))
        self.btn_advanced_settings.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def _update_mode_enables(self):
        busy = self._job_is_active()
        freq_discrete = self.cmb_freq_mode.currentIndex() == 0
        elev_discrete = self.cmb_elev_mode.currentIndex() == 0

        self.edit_freq_list.setEnabled(not busy and freq_discrete)
        self.edit_freq_start.setEnabled(not busy and not freq_discrete)
        self.edit_freq_stop.setEnabled(not busy and not freq_discrete)
        self.edit_freq_step.setEnabled(not busy and not freq_discrete)

        self.edit_elev_list.setEnabled(not busy and elev_discrete)
        self.edit_elev_start.setEnabled(not busy and not elev_discrete)
        self.edit_elev_stop.setEnabled(not busy and not elev_discrete)
        self.edit_elev_step.setEnabled(not busy and not elev_discrete)

    def _parse_list(self, text: 'str', field_name: 'str') -> 'List[float]':
        tokens = [tok for tok in re.split(r"[,\s]+", text.strip()) if tok]
        if not tokens:
            raise ValueError(f"{field_name}: no values were provided.")
        values: 'List[float]' = []
        for tok in tokens:
            try:
                v = float(tok)
            except ValueError:
                raise ValueError(f"{field_name}: invalid numeric token '{tok}'.")
            if not math.isfinite(v):
                raise ValueError(f"{field_name}: non-finite value '{tok}' (NaN/Inf not allowed).")
            values.append(v)
        return values

    def _parse_sweep(self, start_s: 'str', stop_s: 'str', step_s: 'str', field_name: 'str') -> 'List[float]':
        try:
            start = float(start_s)
            stop = float(stop_s)
            step = abs(float(step_s))
        except ValueError:
            raise ValueError(f"{field_name}: start, stop, and step must be numeric.")
        if not (math.isfinite(start) and math.isfinite(stop) and math.isfinite(step)):
            raise ValueError(f"{field_name}: start, stop, and step must be finite (NaN/Inf not allowed).")
        if step <= 0.0:
            raise ValueError(f"{field_name}: step must be > 0.")

        direction = 1.0 if stop >= start else -1.0
        signed_step = step * direction
        values: 'List[float]' = []
        current = start
        for _ in range(20_000):
            if direction > 0 and current > stop + 1e-9:
                break
            if direction < 0 and current < stop - 1e-9:
                break
            values.append(round(current, 12))
            current += signed_step
        if not values or abs(values[-1] - stop) > 1e-9:
            values.append(stop)
        if len(values) > 5000:
            raise ValueError(f"{field_name}: too many samples ({len(values)}).")
        return values

    def _collect_frequency_values(self) -> 'List[float]':
        if self.cmb_freq_mode.currentIndex() == 0:
            freqs = self._parse_list(self.edit_freq_list.text(), "Frequencies")
        else:
            freqs = self._parse_sweep(
                self.edit_freq_start.text(),
                self.edit_freq_stop.text(),
                self.edit_freq_step.text(),
                "Frequencies",
            )
        if any(f <= 0 for f in freqs):
            raise ValueError("Frequencies must be positive values in GHz.")
        return freqs

    def _collect_elevation_values(self) -> 'List[float]':
        if self.cmb_elev_mode.currentIndex() == 0:
            return self._parse_list(self.edit_elev_list.text(), "Elevations")
        return self._parse_sweep(
            self.edit_elev_start.text(),
            self.edit_elev_stop.text(),
            self.edit_elev_step.text(),
            "Elevations",
        )

    @staticmethod
    def _load_geometry_file_for_solver(
        path: 'str',
    ) -> 'Tuple[Dict[str, Any], str, str, str]':
        """Parse and identify the same stable bytes from an explicit .geo."""

        source_path = os.path.abspath(path)
        source = Path(source_path)
        before = source.stat()
        raw = source.read_bytes()
        after = source.stat()
        before_key = (
            int(before.st_size),
            int(before.st_mtime_ns),
            int(before.st_ctime_ns),
        )
        after_key = (
            int(after.st_size),
            int(after.st_mtime_ns),
            int(after.st_ctime_ns),
        )
        if before_key != after_key:
            raise RuntimeError(
                f"{source_path} changed while it was being read. Wait for the "
                "write to finish, then retry."
            )
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UnicodeError(
                f"Geometry file is not valid UTF-8: {source_path}"
            ) from exc
        title, segments, ibcs_entries, dielectric_entries = parse_geometry(text)
        snapshot = build_geometry_snapshot(
            title, segments, ibcs_entries, dielectric_entries
        )
        return (
            snapshot,
            source_path,
            os.path.dirname(source_path),
            hashlib.sha256(raw).hexdigest(),
        )

    def _load_geometry_for_solver(self) -> 'Tuple[Dict[str, Any], str, str]':
        path = self.edit_geo_path.text().strip()
        if path:
            snapshot, source_path, base_dir, _source_sha256 = (
                self._load_geometry_file_for_solver(path)
            )
            return snapshot, source_path, base_dir

        if self.geometry_tab is None:
            raise ValueError("No geometry file selected and Geometry tab is unavailable.")

        snapshot = self.geometry_tab.get_geometry_snapshot()
        segment_count = int(snapshot.get("segment_count", 0))
        if segment_count <= 0:
            raise ValueError("No geometry loaded. Load a .geo file or use the Geometry tab first.")

        source_path = str(snapshot.get("source_path", "") or "")
        if not source_path:
            source_path = str(getattr(self.geometry_tab, "loaded_path", "") or "")
        base_dir = os.path.dirname(source_path) if source_path else os.getcwd()
        return snapshot, source_path, base_dir

    def _apply_job_state(self) -> 'None':
        busy = self._job_is_active()
        is_bor = (self.cmb_solver_kind.currentData() == "bor")
        enable_2d_quality_thresholds = not busy and not is_bor
        self.btn_run.setEnabled(not busy)
        self._sync_export_state()
        self.btn_currents.setEnabled(not busy and not is_bor)
        self.btn_browse_geo.setEnabled(not busy)
        self.btn_use_tab.setEnabled(not busy)
        self.btn_browse_output.setEnabled(not busy)
        self.edit_geo_path.setEnabled(not busy)
        self.edit_output.setEnabled(not busy)
        self.cmb_solver_kind.setEnabled(not busy)
        self.cmb_units.setEnabled(not busy)
        self.cmb_scatter_mode.setEnabled(not busy and not is_bor)
        self.edit_obs_angles.setEnabled(not busy and not is_bor)
        self.edit_cfie_alpha.setEnabled(not busy and is_bor)
        self.cmb_freq_mode.setEnabled(not busy)
        self.cmb_elev_mode.setEnabled(not busy)
        self._update_mode_enables()
        self.chk_export_after_solve.setEnabled(not busy)
        self.chk_mesh_certification.setEnabled(not busy)
        self.btn_advanced_settings.setEnabled(not busy)
        self.edit_quality_residual_max.setEnabled(enable_2d_quality_thresholds)
        self.edit_quality_condition_max.setEnabled(enable_2d_quality_thresholds)
        self.edit_quality_warnings_max.setEnabled(enable_2d_quality_thresholds)
        self.btn_run.setText("Solving..." if self._is_solving else "Run Solver")
        self.btn_currents.setText(
            "Computing..."
            if self._is_computing_density
            else "Boundary Densities"
        )
        active_abort = None
        if self._is_solving:
            active_abort = self._abort_event
        elif self._is_computing_density:
            active_abort = self._density_abort_event
        self.btn_cancel.setEnabled(
            active_abort is not None and not active_abort.is_set()
        )

    def _set_solving_state(self, solving: 'bool') -> 'None':
        self._is_solving = bool(solving)
        self._apply_job_state()

    def _set_density_state(self, computing: 'bool') -> 'None':
        self._is_computing_density = bool(computing)
        self._apply_job_state()

    def _on_scatter_mode_changed(self, _index: 'int' = 0) -> 'None':
        is_bistatic = (self.cmb_scatter_mode.currentData() == "bistatic")
        self.edit_obs_angles.setVisible(is_bistatic)
        self.lbl_obs_angles.setVisible(is_bistatic)

    def _on_solver_kind_changed(self, _index: 'int' = 0) -> 'None':
        is_bor = (self.cmb_solver_kind.currentData() == "bor")
        if is_bor:
            # BoR is monostatic-only; the 2-D quality knobs do not apply.
            self.cmb_scatter_mode.setCurrentIndex(0)
            try:
                current_alpha = float(self.edit_cfie_alpha.text().strip())
            except ValueError:
                current_alpha = 0.0
            if current_alpha == 0.0:
                self.edit_cfie_alpha.setText("0.5")
        else:
            # CFIE is a real BoR-only control. Do not let its hidden value
            # poison a subsequent 2-D solve.
            self.edit_cfie_alpha.setText("0.0")
        self.lbl_solve_method.setText(
            "BoR-MoM (azimuthal modes)" if is_bor else "Linear / Galerkin")
        self._apply_job_state()

    def _compute_currents(self) -> 'None':
        """Compute boundary-integral densities at first freq/elev and save JSON."""
        if self._job_is_active():
            QMessageBox.information(
                self,
                "Boundary Densities",
                "A solver task is already running. Please wait or cancel it.",
            )
            return

        try:
            explicit_path = self.edit_geo_path.text().strip()
            if explicit_path:
                snapshot, source_path, base_dir, source_sha256 = (
                    self._load_geometry_file_for_solver(explicit_path)
                )
            else:
                snapshot, source_path, base_dir = self._load_geometry_for_solver()
                source_sha256 = None
            snapshot = copy.deepcopy(snapshot)
            frequencies = self._collect_frequency_values()
            elevations = self._collect_elevation_values()
            if not frequencies:
                raise ValueError("Need at least one frequency.")
            if not elevations:
                raise ValueError("Need at least one elevation.")
            units = str(self.cmb_units.currentText() or "inches")
            if str(self.cmb_solver_kind.currentData() or "2d") != "2d":
                raise ValueError(
                    "Boundary densities are currently available for the 2-D "
                    "solver only."
                )
            snapshot_digest = _snapshot_sha256(snapshot)
            input_sha256 = _material_input_sha256(snapshot, base_dir)
            uses_geometry_tab = not bool(explicit_path)
            if not uses_geometry_tab and source_path and source_sha256:
                input_sha256[os.path.abspath(source_path)] = source_sha256
        except Exception as exc:
            QMessageBox.critical(self, "Boundary Density Error", str(exc))
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Boundary Densities", "boundary_densities_vv_hh.json", "JSON Files (*.json)",
        )
        if not save_path:
            self.lbl_status.setText("Boundary-density calculation canceled before start.")
            return
        save_path = os.path.abspath(save_path)
        if not os.path.splitext(save_path)[1]:
            save_path += ".json"
        try:
            if not self._confirm_boundary_density_replacement(save_path):
                self.lbl_status.setText(
                    "Boundary-density calculation canceled before start."
                )
                return
            if os.path.exists(save_path) and not os.path.isfile(save_path):
                raise ValueError(f"Boundary-density output is not a file: {save_path}")
            expect_absent = not os.path.exists(save_path)
            expected_output_sha256 = (
                None if expect_absent else _stable_sha256(save_path)
            )
            output_parent = os.path.dirname(save_path) or os.getcwd()
            if not os.path.isdir(output_parent):
                raise FileNotFoundError(
                    f"Output folder does not exist: {output_parent}"
                )
        except Exception as exc:
            QMessageBox.critical(self, "Boundary Density Error", str(exc))
            return

        abort_event = threading.Event()
        self._density_abort_event = abort_event
        self._density_run_serial += 1
        run_id = self._density_run_serial
        self._active_density_run_id = run_id
        self._pending_density_context = {
            "uses_geometry_tab": uses_geometry_tab,
            "geometry_stale": False,
            "input_sha256": dict(input_sha256),
            "output_path": save_path,
            "expect_output_absent": expect_absent,
            "expected_output_sha256": expected_output_sha256,
        }

        thread = QThread(self)
        worker = _BoundaryDensityWorker(
            run_id=run_id,
            snapshot=snapshot,
            source_path=source_path,
            base_dir=base_dir,
            frequency_ghz=float(frequencies[0]),
            elevation_deg=float(elevations[0]),
            units=units,
            output_path=save_path,
            snapshot_sha256=snapshot_digest,
            input_sha256=input_sha256,
            abort_event=abort_event,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_density_progress)
        worker.finished.connect(self._on_density_finished)
        worker.canceled.connect(self._on_density_canceled)
        worker.error.connect(self._on_density_error)
        worker.finished.connect(thread.quit)
        worker.canceled.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.canceled.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda completed_run_id=run_id: self._on_density_thread_finished(
                completed_run_id
            )
        )

        self._density_thread = thread
        self._density_worker = worker
        self.progress.setValue(0)
        self.lbl_status.setText(
            "Starting boundary-density calculation in the background..."
        )
        self._set_density_state(True)
        thread.start()

    def _confirm_boundary_density_replacement(self, path: 'str') -> 'bool':
        if not os.path.exists(path):
            return True
        buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
        answer = QMessageBox.question(
            self,
            "Replace Existing Boundary Densities?",
            f"The output already exists:\n\n  {path}\n\nReplace it?",
            buttons.Yes | buttons.No,
            buttons.No,
        )
        return answer == buttons.Yes

    def _cancel_solver(self):
        """Signal the active solver-tab calculation to abort."""
        if self._is_computing_density and self._density_abort_event is not None:
            self._density_abort_event.set()
            self.lbl_status.setText(
                "Canceling boundary-density calculation... The current dense "
                "linear-algebra step must finish before cancellation completes."
            )
            self.btn_cancel.setEnabled(False)
        elif self._is_solving and self._abort_event is not None:
            self._abort_event.set()
            self.lbl_status.setText("Canceling solve...")
            self.btn_cancel.setEnabled(False)

    @Slot(int, str)
    def _on_density_progress(self, pct: 'int', message: 'str') -> 'None':
        self.progress.setValue(max(0, min(100, int(pct))))
        if message:
            self.lbl_status.setText(message)

    @Slot(int, object, str)
    def _on_density_finished(
        self, run_id: 'int', summary: 'Dict[str, Any]', temporary: 'str'
    ) -> 'None':
        try:
            if self._active_density_run_id != int(run_id):
                return
            context = self._pending_density_context
            if context is None:
                raise RuntimeError(
                    "Boundary-density job context was lost; no output was published."
                )
            if (
                self._density_abort_event is not None
                and self._density_abort_event.is_set()
            ):
                self.progress.setValue(0)
                self.lbl_status.setText(
                    "Boundary-density calculation canceled; no output was published."
                )
                return
            if bool(context.get("geometry_stale", False)):
                self.progress.setValue(0)
                self.lbl_status.setText(
                    "Geometry changed during the boundary-density calculation; "
                    "the stale output was not published."
                )
                QMessageBox.warning(
                    self,
                    "Boundary Densities Not Saved",
                    "Geometry changed while boundary densities were being "
                    "computed. Run the calculation again for the current geometry.",
                )
                return
            _verify_input_sha256(dict(context.get("input_sha256", {})))
            output_path = str(context["output_path"])
            _publish_staged_json(
                temporary,
                output_path,
                expect_absent=bool(context["expect_output_absent"]),
                expected_sha256=context.get("expected_output_sha256"),
            )
            temporary = ""
            self.progress.setValue(100)
            self.lbl_status.setText(
                f"Boundary densities saved: {output_path} "
                f"({int(summary.get('element_count', 0))} elements, "
                f"{summary.get('formulations', '')})"
            )
        except Exception as exc:
            self.progress.setValue(0)
            self.lbl_status.setText(f"Boundary-density output not saved: {exc}")
            QMessageBox.warning(self, "Boundary Density Save Error", str(exc))
        finally:
            if temporary:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
            if self._active_density_run_id == int(run_id):
                self._pending_density_context = None
                self._set_density_state(False)

    @Slot(int, str)
    def _on_density_canceled(self, run_id: 'int', _message: 'str') -> 'None':
        if self._active_density_run_id != int(run_id):
            return
        self._pending_density_context = None
        self.progress.setValue(0)
        self.lbl_status.setText(
            "Boundary-density calculation canceled; no output was published."
        )
        self._set_density_state(False)

    @Slot(int, str)
    def _on_density_error(self, run_id: 'int', message: 'str') -> 'None':
        if self._active_density_run_id != int(run_id):
            return
        if (
            self._density_abort_event is not None
            and self._density_abort_event.is_set()
        ):
            self._on_density_canceled(
                int(run_id), "Boundary-density calculation canceled by user."
            )
            return
        self._pending_density_context = None
        self.progress.setValue(0)
        self.lbl_status.setText(f"Boundary-density calculation failed: {message}")
        QMessageBox.critical(self, "Boundary Density Error", message)
        self._set_density_state(False)

    @Slot(int)
    def _on_density_thread_finished(self, run_id: 'int') -> 'None':
        if self._active_density_run_id != int(run_id):
            return
        self._density_worker = None
        self._density_thread = None
        self._density_abort_event = None
        self._active_density_run_id = None
        self._apply_job_state()

    @Slot(int, str)
    def _on_solver_progress(self, pct: 'int', message: 'str'):
        self.progress.setValue(max(0, min(100, int(pct))))
        if message:
            self.lbl_status.setText(f"Solving... {message}")

    @Slot(object, str)
    def _on_solver_finished(self, result: 'Dict[str, Any]', source_path: 'str'):
        # A completion can already be queued when the user clicks Cancel.
        # Honor that request before publishing/displaying the returned field.
        if self._abort_event is not None and self._abort_event.is_set():
            self._on_solver_canceled("Solve canceled by user.")
            return
        self.last_result = result
        self.last_source_path = source_path
        self.last_solve_context = self._pending_solve_context
        completed_context = self.last_solve_context
        self._last_result_stale = bool(
            (self.last_solve_context or {}).get("geometry_stale", False)
        )
        self._pending_solve_context = None
        self._sync_export_state()
        self._populate_results_table(result)
        self._plot_results(result)

        metadata = result.get("metadata", {}) or {}
        units = str(metadata.get("geometry_units_in", self.cmb_units.currentText()))
        write_summary = _result_summary(result)
        quality_suffix = _quality_gate_suffix(metadata)
        mesh_suffix = ""
        mesh = metadata.get("mesh_convergence", {}) or {}
        if bool(metadata.get("mesh_convergence_certified", False)):
            if bool(mesh.get("passed", False)):
                if "rms_db" in mesh and "max_abs_db" in mesh:
                    mesh_suffix = (
                        " Mesh convergence: PASS "
                        f"(rms={float(mesh['rms_db']):.3g} dB, "
                        f"max={float(mesh['max_abs_db']):.3g} dB)."
                    )
                else:
                    mesh_suffix = " Mesh convergence: PASS."
            else:
                reason = str(mesh.get("reason", "") or "criteria not met")
                mesh_suffix = f" Mesh convergence: FAIL ({reason})."
        elif bool(metadata.get("survey_mode", False)):
            mesh_suffix = " Survey mode: base mesh only (not mesh-certified)."

        self.progress.setValue(100)
        self.lbl_status.setText(write_summary + quality_suffix + mesh_suffix)
        if self._last_result_stale:
            self.lbl_status.setText(
                write_summary
                + quality_suffix
                + mesh_suffix
                + " Geometry changed during the solve; result is stale and was not exported."
            )

        if self.chk_export_after_solve.isChecked() and not self._last_result_stale:
            try:
                requested_output = self.edit_output.text()
                out_text = self._resolve_output_path(
                    requested_output,
                    source_path,
                )
                if requested_output.strip():
                    self.edit_output.setText(out_text)
                if not self._confirm_export_replacements(
                    _planned_export_paths(result, out_text)
                ):
                    self.lbl_status.setText(
                        write_summary
                        + quality_suffix
                        + mesh_suffix
                        + " Automatic export was canceled; the result remains available."
                    )
                    self._set_solving_state(False)
                    return
                # The replacement prompt runs a nested Qt event loop. A
                # queued geometry edit or result completion can therefore
                # invalidate/replace this result while the prompt is open.
                # Recheck immediately before the first file write.
                if not self._result_is_current_for_export(
                    result, completed_context
                ):
                    self.lbl_status.setText(
                        write_summary
                        + quality_suffix
                        + mesh_suffix
                        + " Automatic export skipped because geometry or the "
                        "active solver result changed during export confirmation."
                    )
                    self._set_solving_state(False)
                    return
                files = self._export_result_files(
                    result,
                    out_text,
                    source_path=source_path,
                    history=_result_history(
                        result,
                        units=units,
                    ),
                )
                self.lbl_status.setText(
                    write_summary
                    + quality_suffix
                    + mesh_suffix
                    + " Exported "
                    + ", ".join(os.path.basename(path) for path in files)
                )
                self.files_exported.emit(
                    [str(path) for path in files],
                    "bor" if _result_kind(result) == "bor" else "2d",
                )
            except Exception as exc:
                QMessageBox.warning(self, "Export Warning", f"Solve completed, but export failed:\n{exc}")

        self._set_solving_state(False)

    @Slot(str)
    def _on_solver_error(self, message: 'str'):
        if self._abort_event is not None and self._abort_event.is_set():
            self._on_solver_canceled("Solve canceled by user.")
            return
        self._pending_solve_context = None
        self.progress.setValue(0)
        self.lbl_status.setText(f"Solve failed: {message}")
        QMessageBox.critical(self, "Solver Error", message)
        self._set_solving_state(False)

    @Slot(str)
    def _on_solver_canceled(self, _message: 'str') -> 'None':
        """Treat an operator cancellation as a normal terminal state."""

        self._pending_solve_context = None
        self.progress.setValue(0)
        self.lbl_status.setText("Solve canceled; no result was published.")
        self._set_solving_state(False)

    @Slot(int)
    def _on_solver_thread_finished(self, run_id: 'int'):
        # A queued terminal signal can race QThread.finished. Only the matching
        # run may clear its worker/cancel handles; controls remain disabled while
        # the thread itself still reports running.
        if self._active_solve_run_id != int(run_id):
            return
        self._solve_worker = None
        self._solve_thread = None
        self._abort_event = None
        self._active_solve_run_id = None
        self._apply_job_state()

    def _run_solver(self):
        if self._job_is_active():
            return
        try:
            frequencies = self._collect_frequency_values()
            elevations = self._collect_elevation_values()
            snapshot, source_path, base_dir = self._load_geometry_for_solver()
            units = self.cmb_units.currentText()
            mesh_certification = bool(self.chk_mesh_certification.isChecked())
            solver_kind = str(self.cmb_solver_kind.currentData() or "2d")
            quality_thresholds: 'Dict[str, float | int]' = {}
            if solver_kind == "2d":
                quality_residual_max = float(
                    self.edit_quality_residual_max.text().strip()
                )
                quality_condition_max = float(
                    self.edit_quality_condition_max.text().strip()
                )
                quality_warnings_max = int(float(
                    self.edit_quality_warnings_max.text().strip()
                ))
                if quality_residual_max <= 0.0:
                    raise ValueError("Quality residual threshold must be > 0.")
                if quality_condition_max <= 0.0:
                    raise ValueError("Quality condition threshold must be > 0.")
                if quality_warnings_max < 0:
                    raise ValueError("Quality warning threshold must be >= 0.")
                quality_thresholds = {
                    "residual_norm_max": quality_residual_max,
                    "condition_est_max": quality_condition_max,
                    "warnings_max": quality_warnings_max,
                }

            # Advanced settings
            cfie_text = self.edit_cfie_alpha.text().strip()
            cfie_alpha = (
                float(cfie_text) if solver_kind == "bor" and cfie_text
                else 0.0
            )
            if solver_kind == "bor" and (
                not math.isfinite(cfie_alpha) or not (0.0 < cfie_alpha < 1.0)
            ):
                raise ValueError("BoR CFIE alpha must satisfy 0 < alpha < 1.")

            # Scattering mode and observation angles.
            scatter_mode = str(self.cmb_scatter_mode.currentData() or "monostatic")
            obs_angles_list: 'Optional[List[float]]' = None
            if scatter_mode == "bistatic":
                obs_angles_list = self._parse_list(self.edit_obs_angles.text(), "Observation angles")
                if not obs_angles_list:
                    raise ValueError("Bistatic mode requires at least one observation angle.")
        except Exception as exc:
            QMessageBox.critical(self, "Solver Error", str(exc))
            self.lbl_status.setText(f"Solve failed: {exc}")
            return

        self.progress.setValue(0)
        self.lbl_status.setText("Starting solver thread...")
        self._set_solving_state(True)

        abort_event = threading.Event()
        self._abort_event = abort_event
        self._solve_run_serial += 1
        run_id = self._solve_run_serial
        self._active_solve_run_id = run_id
        self._pending_solve_context = {
            "snapshot": snapshot,
            "solver_kind": solver_kind,
            "units": str(units),
            "aspects_deg": list(elevations),
            "uses_geometry_tab": not bool(self.edit_geo_path.text().strip()),
            "geometry_stale": False,
        }

        thread = QThread(self)
        worker = _SolveWorker(
            snapshot=snapshot,
            source_path=source_path,
            base_dir=base_dir,
            frequencies=frequencies,
            elevations=elevations,
            units=units,
            quality_thresholds=quality_thresholds,
            mesh_certification=mesh_certification,
            cfie_alpha=cfie_alpha,
            abort_event=abort_event,
            scattering_mode=scatter_mode,
            observation_angles=obs_angles_list,
            solver_kind=solver_kind,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_solver_progress)
        worker.finished.connect(self._on_solver_finished)
        worker.canceled.connect(self._on_solver_canceled)
        worker.error.connect(self._on_solver_error)
        worker.finished.connect(thread.quit)
        worker.canceled.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.canceled.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda completed_run_id=run_id: self._on_solver_thread_finished(
                completed_run_id
            )
        )

        self._solve_thread = thread
        self._solve_worker = worker
        thread.start()

    def _export_result_files(
        self,
        result: 'Dict[str, Any]',
        output_path: 'str',
        *,
        source_path: 'str',
        history: 'str',
    ) -> 'List[str]':
        """Publish a generic 2-D field or a feature-ready BoR body file."""

        if _result_kind(result) != "bor":
            return export_result_to_grim(
                result,
                output_path,
                source_path=source_path,
                history=history,
                preserve_raw_complex_amplitude=True,
            )
        context = self.last_solve_context
        if not isinstance(context, dict) or context.get("solver_kind") != "bor":
            raise ValueError(
                "The BoR solve context is unavailable; re-run the body before "
                "exporting it."
            )
        from feature_sum import (
            bodies_from_bor_solver_result,
            bor_solver_diagnostics_by_frequency,
            outer_generatrix,
            save_monostatic_grim,
        )

        bodies = bodies_from_bor_solver_result(result)
        profile = outer_generatrix(
            context["snapshot"], str(context["units"])
        )
        written = save_monostatic_grim(
            bodies,
            profile,
            output_path,
            # The GUI's BoR sweep control is the body aspect. With the default
            # body axis at radar (az=0, el=0), the azimuth cut at elevation 0
            # has exactly that 0..180 aspect axis.
            azimuths_deg=list(context["aspects_deg"]),
            elevations_deg=[0.0],
            source_path=source_path,
            history=history,
            solver_diagnostics=bor_solver_diagnostics_by_frequency(result),
        )
        return [written]

    def _export_last_result(self):
        if self._job_is_active():
            QMessageBox.information(
                self,
                "Export",
                "A solver task is currently running. Please wait for completion.",
            )
            return
        if not self.last_result:
            QMessageBox.information(self, "Export", "No solver result exists yet. Run the solver first.")
            return
        if self._last_result_stale:
            QMessageBox.warning(
                self,
                "Stale Solver Result",
                "Geometry changed after this result was computed. Run the solver "
                "again before exporting.",
            )
            return
        result = self.last_result
        solve_context = self.last_solve_context
        source_path = self.last_source_path
        out_text = self._resolve_output_path(
            self.edit_output.text(),
            source_path,
        )
        self.edit_output.setText(out_text)
        try:
            planned = _planned_export_paths(result, out_text)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
            return
        if not self._confirm_export_replacements(planned):
            self.lbl_status.setText(
                "Export canceled; the solver result remains available."
            )
            return
        # QFile replacement confirmation runs a nested event loop. Do not
        # write the result captured above if geometry changed or another
        # result became active while the question was open.
        if not self._result_is_current_for_export(result, solve_context):
            self._sync_export_state()
            self.lbl_status.setText(
                "Export skipped because geometry or the active solver result "
                "changed during export confirmation. No files were written."
            )
            QMessageBox.warning(
                self,
                "Solver Result Changed",
                "Geometry or the active solver result changed while export "
                "confirmation was open. No files were written. Review the "
                "current result and export again.",
            )
            return
        try:
            files = self._export_result_files(
                result,
                out_text,
                source_path=source_path,
                history=_result_history(
                    result,
                    units=str(
                        (result.get("metadata", {}) or {}).get(
                            "geometry_units_in",
                            self.cmb_units.currentText(),
                        )
                    ),
                    manual_export=True,
                ),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
            return
        QMessageBox.information(self, "Exported", "\n".join(files))
        self.lbl_status.setText("Exported: " + ", ".join(os.path.basename(path) for path in files))
        self.files_exported.emit(
            [str(path) for path in files],
            "bor" if _result_kind(result) == "bor" else "2d",
        )

    def _display_db_from_linear(
        self,
        result: 'Dict[str, Any]',
        row: 'Dict[str, Any]',
    ) -> 'float':
        return _display_db_value(result, row)

    def _populate_results_table(self, result: 'Dict[str, Any]'):
        kind = _result_kind(result)
        rows = sorted(
            _result_rows(result),
            key=lambda row: (
                float(row.get("frequency_ghz", 0.0)),
                str(row.get("polarization", "")),
                float(row.get("theta_inc_deg", 0.0)),
                float(row.get("theta_scat_deg", 0.0)),
            ),
        )
        self.table_results.clear()
        if kind == "2d_bistatic":
            self.table_results.setColumnCount(6)
            self.table_results.setHorizontalHeaderLabels(
                [
                    "Frequency (GHz)",
                    "Pol",
                    "Incidence (deg)",
                    "Observation (deg)",
                    "Width (m)",
                    "Width (dBke)",
                ]
            )
        elif kind == "bor":
            self.table_results.setColumnCount(5)
            self.table_results.setHorizontalHeaderLabels(
                [
                    "Frequency (GHz)",
                    "Pol",
                    "Aspect (deg)",
                    "RCS (m^2)",
                    "RCS (dBsm)",
                ]
            )
        else:
            self.table_results.setColumnCount(5)
            self.table_results.setHorizontalHeaderLabels(
                [
                    "Frequency (GHz)",
                    "Pol",
                    "Cut angle (deg)",
                    "Width (m)",
                    "Width (dBke)",
                ]
            )
        self.table_results.setRowCount(len(rows))
        for r, row in enumerate(rows):
            freq = float(row.get("frequency_ghz", 0.0))
            inc = float(row.get("theta_inc_deg", 0.0))
            obs = float(row.get("theta_scat_deg", 0.0))
            lin = float(row.get("rcs_linear", 0.0))
            db = self._display_db_from_linear(result, row)
            self.table_results.setItem(r, 0, QTableWidgetItem(f"{freq:.6g}"))
            self.table_results.setItem(
                r, 1, QTableWidgetItem(str(row.get("polarization", "")))
            )
            if kind == "2d_bistatic":
                self.table_results.setItem(
                    r, 2, QTableWidgetItem(f"{inc:.6g}")
                )
                self.table_results.setItem(
                    r, 3, QTableWidgetItem(f"{obs:.6g}")
                )
                self.table_results.setItem(
                    r, 4, QTableWidgetItem(f"{lin:.6e}")
                )
                self.table_results.setItem(
                    r, 5, QTableWidgetItem(f"{db:.3f}")
                )
            else:
                self.table_results.setItem(
                    r, 2, QTableWidgetItem(f"{obs:.6g}")
                )
                self.table_results.setItem(
                    r, 3, QTableWidgetItem(f"{lin:.6e}")
                )
                self.table_results.setItem(
                    r, 4, QTableWidgetItem(f"{db:.3f}")
                )

    def _plot_results(self, result: 'Dict[str, Any]'):
        kind = _result_kind(result)
        plot_groups = _result_plot_groups(result)
        ax = self.canvas.ax
        ax.clear()
        for freq, incidence, polarization in sorted(
            plot_groups.keys(),
            key=lambda key: (
                key[0],
                key[2],
                -math.inf if key[1] is None else key[1],
            ),
        ):
            rows = sorted(
                plot_groups[(freq, incidence, polarization)],
                key=lambda row: float(row.get("theta_scat_deg", 0.0)),
            )
            x = [float(row.get("theta_scat_deg", 0.0)) for row in rows]
            y = [
                self._display_db_from_linear(result, row) for row in rows
            ]
            label = f"{freq:g} GHz, {polarization}"
            if incidence is not None:
                label += f", incidence {incidence:g}deg"
            ax.plot(x, y, linewidth=1.8, label=label)

        if kind == "bor":
            ax.set_title("Monostatic RCS (BoR)")
            ax.set_xlabel("Aspect angle from +z axis (deg)")
            ax.set_ylabel("RCS (dBsm)")
        elif kind == "2d_bistatic":
            ax.set_title("Bistatic 2-D Scattering Width")
            ax.set_xlabel("Observation angle (deg)")
            ax.set_ylabel("2-D scattering width (dBke)")
        else:
            ax.set_title("Monostatic 2-D Scattering Width")
            ax.set_xlabel("Cut angle (deg)")
            ax.set_ylabel("2-D scattering width (dBke)")
        ax.grid(True, alpha=0.3)
        if len(plot_groups) <= 12:
            ax.legend(loc="best")
        self._apply_plot_theme_to_axes()
        self.canvas.draw_idle()
